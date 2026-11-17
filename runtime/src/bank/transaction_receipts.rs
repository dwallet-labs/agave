//! Transaction Receipt Tree (draft SIMD): one receipt per transaction of the
//! block, accumulated by the bank at commit, and `receipts_root`, slot 1 of
//! the commitment vector of the Block Commitment Root SIMD.
//!
//! # Which transactions have receipts
//!
//! Every transaction committed to the bank, that is every *processed*
//! transaction, gets exactly one receipt:
//!
//! - executed successfully: status 0, the fee, and the trace and events roots
//!   computed by the SVM ([`ReceiptRoots`]),
//! - executed with an error, or fee-only (failed to load but charged): status
//!   1, the fee, and the empty-tree sub-roots,
//! - no-op (SIMD-0290 fee payer failure, recorded in the block but charged
//!   nothing): status 1, fee 0, and the empty-tree sub-roots.
//!
//! Transactions that are not processed are not recorded in the block (the
//! leader drops them, and replay rejects a block that contains one), so they
//! have no receipt.
//!
//! # Block order
//!
//! Transactions of a block are committed in batches, possibly by several
//! threads at once, so commit order is not block order. Each committed batch
//! therefore carries the position of its transactions in the block
//! ([`ReceiptOrder::BlockPositions`]), and the receipts are sorted by position
//! when the root is computed:
//!
//! - replay: the position is the index of the transaction in the slot's
//!   entries, which `blockstore_processor` assigns before scheduling (it is
//!   the unified scheduler task id),
//! - block production: the position is assigned by the record channel when
//!   the batch is sent to PoH, under the lock that serializes sends; the
//!   channel is FIFO and every sent record is recorded, so the order of the
//!   positions is the order of the entries in the block.
//!
//! Positions only need to be ordered like the block; they need not start at
//! zero or be contiguous.

use {
    super::Bank,
    agave_feature_set as feature_set,
    log::*,
    solana_hash::Hash,
    solana_merkle_tree::{
        commitment_tree::{InclusionProof, make_inclusion_proof_from_leaf_hashes},
        receipt_tree::{TransactionReceipt, receipts_root_from_leaf_hashes},
    },
    solana_runtime_transaction::transaction_with_meta::TransactionWithMeta,
    solana_svm::{
        transaction_execution_result::ReceiptRoots,
        transaction_processing_result::{ProcessedTransaction, TransactionProcessingResult},
    },
    std::sync::Mutex,
};

/// How the transactions of a committed batch are placed in the block, for
/// the order of their receipts.
#[derive(Clone, Copy, Debug)]
pub enum ReceiptOrder<'a> {
    /// `positions[i]` is the position in the block of transaction `i` of
    /// the batch. Entries of transactions that are not committed are
    /// ignored. Used by replay and block production.
    BlockPositions(&'a [usize]),
    /// Receipts are appended in commit order. Only sound when a single caller
    /// commits all transactions of the bank sequentially and never mixes this
    /// with [`Self::BlockPositions`] (tests, tools that execute transactions
    /// directly against a bank).
    CommitOrder,
}

/// Receipts committed to a bank so far, as `(position, leaf hash)`.
#[derive(Debug, Default)]
pub(crate) struct ReceiptAccumulator {
    entries: Mutex<Vec<(u64, Hash)>>,
}

impl ReceiptAccumulator {
    fn push(&self, order: ReceiptOrder, receipts: impl Iterator<Item = (usize, Hash)>) {
        let mut entries = self.entries.lock().unwrap();
        match order {
            ReceiptOrder::BlockPositions(positions) => {
                for (index_in_batch, leaf_hash) in receipts {
                    let position = positions.get(index_in_batch).copied().unwrap_or_else(|| {
                        // The caller passed fewer positions than committed
                        // transactions. Sorting this receipt last keeps the
                        // root deterministic; it will not match the cluster.
                        error!("missing block position for committed transaction {index_in_batch}");
                        usize::MAX
                    });
                    entries.push((position as u64, leaf_hash));
                }
            }
            ReceiptOrder::CommitOrder => {
                for (_, leaf_hash) in receipts {
                    let position = entries.len() as u64;
                    entries.push((position, leaf_hash));
                }
            }
        }
    }

    /// Leaf hashes in block order.
    fn leaf_hashes_in_block_order(&self) -> Vec<Hash> {
        let mut entries = self.entries.lock().unwrap().clone();
        // The tie-break on the leaf hash only matters if two receipts claim
        // the same position, which is a bug; it keeps the result independent
        // of commit order even then.
        entries.sort_unstable();
        if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            error!("two transaction receipts have the same position in the block");
        }
        entries
            .into_iter()
            .map(|(_, leaf_hash)| leaf_hash)
            .collect()
    }

    fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }
}

/// The receipt of a processed transaction.
pub fn transaction_receipt(
    transaction: &impl TransactionWithMeta,
    processed_transaction: &ProcessedTransaction,
) -> TransactionReceipt {
    let signature = (*transaction.signature()).into();
    let message_hash = *transaction.message_hash();
    match processed_transaction {
        ProcessedTransaction::Executed(executed_tx) => {
            let fee = executed_tx.loaded_transaction.fee_details.total_fee();
            let details = &executed_tx.execution_details;
            match (&details.status, details.receipt_roots) {
                (
                    Ok(()),
                    Some(ReceiptRoots {
                        instruction_trace_root,
                        events_root,
                    }),
                ) => TransactionReceipt::success(
                    signature,
                    message_hash,
                    fee,
                    instruction_trace_root,
                    events_root,
                ),
                (Ok(()), None) => {
                    // The bank always asks the SVM for the roots while the
                    // feature is active, so this is unreachable. Report it
                    // and still produce a deterministic receipt.
                    error!(
                        "successful transaction {} has no receipt roots",
                        transaction.signature()
                    );
                    datapoint_error!(
                        "bank-transaction_receipt_missing_roots",
                        ("signature", transaction.signature().to_string(), String),
                    );
                    TransactionReceipt::failed(signature, message_hash, fee)
                }
                (Err(_), _) => TransactionReceipt::failed(signature, message_hash, fee),
            }
        }
        ProcessedTransaction::FeesOnly(fees_only_tx) => TransactionReceipt::failed(
            signature,
            message_hash,
            fees_only_tx.fee_details.total_fee(),
        ),
        // Recorded in the block, but charged nothing.
        ProcessedTransaction::NoOp(_) => TransactionReceipt::failed(signature, message_hash, 0),
    }
}

impl Bank {
    /// Whether this bank computes `receipts_root`: both the Transaction
    /// Receipt Tree feature and the Block Commitment Root feature, which
    /// carries the root, must be active.
    pub fn is_transaction_receipt_tree_active(&self) -> bool {
        self.feature_set
            .is_active(&feature_set::transaction_receipt_tree::id())
            && self.is_block_commitment_root_active()
    }

    /// Adds the receipts of the processed transactions of a batch. Called
    /// from `commit_transactions`, so every committed transaction gets its
    /// receipt.
    pub(super) fn accumulate_transaction_receipts(
        &self,
        transactions: &[impl TransactionWithMeta],
        processing_results: &[TransactionProcessingResult],
        order: ReceiptOrder,
    ) {
        if !self.is_transaction_receipt_tree_active() {
            return;
        }
        let receipts = transactions
            .iter()
            .zip(processing_results)
            .enumerate()
            .filter_map(|(index_in_batch, (transaction, processing_result))| {
                let processed_transaction = processing_result.as_ref().ok()?;
                let receipt = transaction_receipt(transaction, processed_transaction);
                Some((index_in_batch, receipt.leaf_hash()))
            });
        self.transaction_receipts.push(order, receipts);
    }

    /// Leaf hashes of the receipts committed so far, in block order.
    pub fn transaction_receipt_leaf_hashes(&self) -> Vec<Hash> {
        self.transaction_receipts.leaf_hashes_in_block_order()
    }

    /// `receipts_root` over the receipts committed so far, `None` while the
    /// feature is inactive. On a frozen bank this is the value in slot 1 of
    /// its commitment vector.
    pub fn receipts_root(&self) -> Option<Hash> {
        self.is_transaction_receipt_tree_active()
            .then(|| receipts_root_from_leaf_hashes(&self.transaction_receipt_leaf_hashes()))
    }

    /// Proof of the receipt at `index` (in block order) against
    /// [`Self::receipts_root`].
    pub fn transaction_receipt_inclusion_proof(&self, index: usize) -> Option<InclusionProof> {
        make_inclusion_proof_from_leaf_hashes(&self.transaction_receipt_leaf_hashes(), index)
    }

    /// `receipts_root` for the commitment vector, checking that every
    /// transaction committed to this bank has a receipt.
    pub(super) fn receipts_root_for_commitment(&self, parent: Option<&Bank>) -> Option<Hash> {
        if !self.is_transaction_receipt_tree_active() {
            return None;
        }
        if let Some(parent) = parent {
            let committed = self
                .transaction_count()
                .saturating_sub(parent.transaction_count());
            let receipts = self.transaction_receipts.len() as u64;
            if committed != receipts {
                error!(
                    "bank {} committed {committed} transactions but has {receipts} receipts",
                    self.slot()
                );
                datapoint_error!(
                    "bank-transaction_receipt_count_mismatch",
                    ("slot", self.slot(), i64),
                    ("committed", committed, i64),
                    ("receipts", receipts, i64),
                );
            }
        }
        self.receipts_root()
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        crate::{
            bank::{
                block_commitment::{
                    COMMITMENT_VECTOR_LEN, CommitmentSlot, CommitmentVector,
                    INACTIVE_COMMITMENT_SLOT,
                },
                test_utils::goto_end_of_slot,
            },
            genesis_utils::{
                GenesisConfigInfo, activate_feature, create_genesis_config_with_leader,
            },
            loader_utils::create_program_with_elf,
        },
        agave_votor_messages::consensus_message::BlockId,
        solana_fee_calculator::FeeRateGovernor,
        solana_instruction::{AccountMeta, Instruction},
        solana_keypair::Keypair,
        solana_merkle_tree::{
            commitment_tree::TreeId,
            receipt_tree::{
                EventKind, EventRecord, InstructionAccountMeta, InstructionRecord,
                empty_events_root, event_inclusion_proof, events_root,
                instruction_record_inclusion_proof, instruction_trace_root, log_data_payload,
                receipts_root, verify_event, verify_instruction_record, verify_receipt,
            },
        },
        solana_pubkey::Pubkey,
        solana_runtime_transaction::runtime_transaction::RuntimeTransaction,
        solana_sdk_ids::{bpf_loader_upgradeable, system_program},
        solana_signer::Signer,
        solana_svm::transaction_processor::ExecutionRecordingConfig,
        solana_svm_timings::ExecuteTimings,
        solana_system_transaction as system_transaction,
        solana_transaction::Transaction,
        std::sync::Arc,
    };

    /// See `runtime/tests/fixtures/receipt_test_program/receipt_test_program.c`.
    const RECEIPT_TEST_PROGRAM: &[u8] =
        include_bytes!("../../tests/fixtures/receipt_test_program/receipt_test_program.so");

    struct Setup {
        bank0: Arc<Bank>,
        mint_keypair: Keypair,
        program_id: Pubkey,
    }

    fn setup(receipt_tree: bool, block_commitment_root: bool) -> Setup {
        let GenesisConfigInfo {
            mut genesis_config,
            mint_keypair,
            ..
        } = create_genesis_config_with_leader(1_000_000_000_000, &Pubkey::new_unique(), 1_000_000);
        // Non-zero fees, so that the fee field is exercised.
        genesis_config.fee_rate_governor = FeeRateGovernor::new(5000, 0);
        if receipt_tree {
            activate_feature(
                &mut genesis_config,
                feature_set::transaction_receipt_tree::id(),
            );
        }
        if block_commitment_root {
            activate_feature(
                &mut genesis_config,
                feature_set::block_commitment_root::id(),
            );
        }
        let bank0 = Arc::new(Bank::new_for_tests(&genesis_config));
        let program_id = create_program_with_elf(
            &bank0,
            &bpf_loader_upgradeable::id(),
            &Pubkey::default(),
            RECEIPT_TEST_PROGRAM,
        );
        freeze_with_block_id(&bank0);
        Setup {
            bank0,
            mint_keypair,
            program_id,
        }
    }

    fn freeze_with_block_id(bank: &Arc<Bank>) {
        goto_end_of_slot(bank.clone());
        bank.freeze();
        bank.set_block_id(Some(BlockId::from(Hash::new_unique())));
    }

    fn new_child(parent: &Arc<Bank>) -> Arc<Bank> {
        // The genesis leader has a vote account to receive the fees.
        Arc::new(Bank::new_from_parent(
            parent.clone(),
            *parent.leader(),
            parent.slot() + 1,
        ))
    }

    fn program_transaction(
        setup: &Setup,
        bank: &Bank,
        recipient: &Pubkey,
        data: &[u8],
    ) -> Transaction {
        let payer = setup.mint_keypair.pubkey();
        let instruction = Instruction::new_with_bytes(
            setup.program_id,
            data,
            vec![
                AccountMeta::new(payer, true),
                AccountMeta::new(*recipient, false),
                AccountMeta::new_readonly(system_program::id(), false),
                AccountMeta::new_readonly(setup.program_id, false),
            ],
        );
        Transaction::new_signed_with_payer(
            &[instruction],
            Some(&payer),
            &[&setup.mint_keypair],
            bank.last_blockhash(),
        )
    }

    fn message_hash(transaction: &Transaction) -> Hash {
        *RuntimeTransaction::from_transaction_for_tests(transaction.clone()).message_hash()
    }

    fn meta(pubkey: &Pubkey, is_signer: bool, is_writable: bool) -> InstructionAccountMeta {
        InstructionAccountMeta {
            pubkey: pubkey.to_bytes(),
            is_signer,
            is_writable,
        }
    }

    /// Trace and events of `program_transaction(.., [0, extra..])`, written
    /// from the program source.
    fn expected_trace_and_events(
        setup: &Setup,
        recipient: &Pubkey,
        extra: &[u8],
    ) -> (Vec<InstructionRecord>, Vec<EventRecord>) {
        let payer = setup.mint_keypair.pubkey();
        let program = setup.program_id;
        let accounts = [
            meta(&payer, true, true),
            meta(recipient, false, true),
            meta(&system_program::id(), false, false),
            meta(&program, false, false),
        ];
        let mut top_level_data = vec![0];
        top_level_data.extend_from_slice(extra);
        let mut transfer = vec![2, 0, 0, 0];
        transfer.extend_from_slice(&1u64.to_le_bytes());
        let records = vec![
            InstructionRecord::new(1, program.to_bytes(), accounts, &top_level_data),
            InstructionRecord::new(2, program.to_bytes(), accounts, &[1]),
            InstructionRecord::new(
                3,
                system_program::id().to_bytes(),
                [meta(&payer, true, true), meta(recipient, false, true)],
                &transfer,
            ),
        ];
        let events = vec![
            EventRecord {
                program_id: program.to_bytes(),
                instruction_index: 0,
                kind: EventKind::Log,
                payload: b"outer".to_vec(),
            },
            EventRecord {
                program_id: program.to_bytes(),
                instruction_index: 1,
                kind: EventKind::LogData,
                payload: log_data_payload(&[b"inner".as_slice(), b""]),
            },
            // Emitted by the top-level instruction after its CPI returned.
            EventRecord {
                program_id: program.to_bytes(),
                instruction_index: 0,
                kind: EventKind::LogData,
                payload: log_data_payload(&[b"after".as_slice(), extra]),
            },
        ];
        (records, events)
    }

    fn signature_bytes(transaction: &Transaction) -> [u8; 64] {
        transaction.signatures[0].into()
    }

    #[test]
    fn test_receipts_success_with_cpi_and_failure() {
        let setup = setup(true, true);
        let bank = new_child(&setup.bank0);
        assert!(bank.is_transaction_receipt_tree_active());
        let payer = setup.mint_keypair.pubkey();
        let recipient = Pubkey::new_unique();
        bank.transfer(1_000_000_000, &setup.mint_keypair, &recipient)
            .unwrap();
        let transfer_tx = system_transaction::transfer(
            &setup.mint_keypair,
            &recipient,
            1_000_000_000,
            bank.last_blockhash(),
        );

        // Transaction 1: outer -> self CPI -> system transfer, with events
        // at stack heights 1 and 2. Log recording is off: events do not
        // depend on the log collector.
        let extra = [0xaa, 0xbb];
        let ok_tx = program_transaction(&setup, &bank, &recipient, &[0, 0xaa, 0xbb]);
        let balance = bank.get_balance(&payer);
        bank.process_transaction(&ok_tx).unwrap();
        let ok_fee = balance - bank.get_balance(&payer) - 1;

        // Transaction 2: the program logs and fails; the fee is charged.
        let fail_tx = program_transaction(&setup, &bank, &recipient, &[2]);
        let balance = bank.get_balance(&payer);
        assert!(bank.process_transaction(&fail_tx).is_err());
        let fail_fee = balance - bank.get_balance(&payer);
        assert_eq!(fail_fee, 5000);

        // Not processed: no receipt.
        assert!(bank.process_transaction(&ok_tx).is_err());

        let (records, events) = expected_trace_and_events(&setup, &recipient, &extra);
        let transfer_records = [InstructionRecord::new(
            1,
            system_program::id().to_bytes(),
            [meta(&payer, true, true), meta(&recipient, false, true)],
            &transfer_tx.message.instructions[0].data,
        )];
        // `bank.transfer` builds the same transaction as `transfer_tx`.
        let expected = vec![
            TransactionReceipt::success(
                signature_bytes(&transfer_tx),
                message_hash(&transfer_tx),
                5000,
                instruction_trace_root(&transfer_records),
                empty_events_root(),
            ),
            TransactionReceipt::success(
                signature_bytes(&ok_tx),
                message_hash(&ok_tx),
                ok_fee,
                instruction_trace_root(&records),
                events_root(&events),
            ),
            TransactionReceipt::failed(signature_bytes(&fail_tx), message_hash(&fail_tx), fail_fee),
        ];
        let expected_leaves: Vec<_> = expected.iter().map(TransactionReceipt::leaf_hash).collect();
        assert_eq!(bank.transaction_receipt_leaf_hashes(), expected_leaves);
        // One receipt per committed transaction.
        assert_eq!(bank.executed_transaction_count(), 3);
        let expected_root = receipts_root(&expected);
        assert_eq!(bank.receipts_root(), Some(expected_root));

        // The root lands in slot 1 of the commitment vector at freeze.
        freeze_with_block_id(&bank);
        let commitment = bank.block_commitment().unwrap();
        let mut vector = [INACTIVE_COMMITMENT_SLOT; COMMITMENT_VECTOR_LEN];
        vector[CommitmentSlot::BlockHistoryRoot.index()] = commitment.block_history_root();
        vector[CommitmentSlot::ReceiptsRoot.index()] = expected_root;
        let vector = CommitmentVector(vector);
        assert_eq!(bank.commitment_root(), Some(vector.root()));

        // commitment_root -> receipts_root -> receipt -> trace / event.
        let slot_proof = vector
            .inclusion_proof(CommitmentSlot::ReceiptsRoot.index())
            .unwrap();
        assert!(slot_proof.verify(
            TreeId::COMMITMENT_VECTOR,
            expected_root.as_ref(),
            &bank.commitment_root().unwrap()
        ));
        let proof = bank.transaction_receipt_inclusion_proof(1).unwrap();
        assert!(verify_receipt(&proof, &expected[1], &expected_root));
        assert!(!verify_receipt(&proof, &expected[2], &expected_root));
        let cpi_proof = instruction_record_inclusion_proof(&records, 2).unwrap();
        assert!(verify_instruction_record(
            &cpi_proof,
            &records[2],
            &expected[1].instruction_trace_root
        ));
        let event_proof = event_inclusion_proof(&events, 1).unwrap();
        assert!(verify_event(
            &event_proof,
            &events[1],
            &expected[1].events_root
        ));
        // The event names the instruction that emitted it.
        assert_eq!(
            records[events[1].instruction_index as usize].stack_height,
            2
        );
    }

    #[test]
    fn test_events_independent_of_log_collector() {
        // No log collector, a truncating one, and an unlimited one: the
        // receipt is the same function of the execution in all three cases.
        for log_messages_bytes_limit in [None, Some(100), Some(1_000_000)] {
            let setup = setup(true, true);
            let bank = new_child(&setup.bank0);
            let payer = setup.mint_keypair.pubkey();
            let recipient = Pubkey::new_unique();
            bank.transfer(1_000_000_000, &setup.mint_keypair, &recipient)
                .unwrap();
            // A large event, above the truncating limit.
            let extra = vec![0x5a; 900];
            let mut data = vec![0];
            data.extend_from_slice(&extra);
            let tx = program_transaction(&setup, &bank, &recipient, &data);
            let balance = bank.get_balance(&payer);
            let batch = bank.prepare_batch_for_tests(vec![tx.clone()]);
            let (commit_results, _) = bank.load_execute_and_commit_transactions(
                &batch,
                ExecutionRecordingConfig::new_single_setting(log_messages_bytes_limit.is_some()),
                &mut ExecuteTimings::default(),
                log_messages_bytes_limit,
            );
            let committed = commit_results[0].as_ref().unwrap();
            assert_eq!(committed.status, Ok(()));
            match log_messages_bytes_limit {
                None => assert!(committed.log_messages.is_none()),
                Some(100) => assert!(
                    committed
                        .log_messages
                        .as_ref()
                        .unwrap()
                        .iter()
                        .any(|line| line == "Log truncated")
                ),
                Some(_) => assert!(
                    !committed
                        .log_messages
                        .as_ref()
                        .unwrap()
                        .iter()
                        .any(|line| line == "Log truncated")
                ),
            }
            let fee = balance - bank.get_balance(&payer) - 1;
            let (records, events) = expected_trace_and_events(&setup, &recipient, &extra);
            let expected = TransactionReceipt::success(
                signature_bytes(&tx),
                message_hash(&tx),
                fee,
                instruction_trace_root(&records),
                events_root(&events),
            );
            assert_eq!(
                bank.transaction_receipt_leaf_hashes()[1],
                expected.leaf_hash(),
                "log_messages_bytes_limit: {log_messages_bytes_limit:?}"
            );
        }
    }

    #[test]
    fn test_empty_block() {
        let setup = setup(true, true);
        let bank = new_child(&setup.bank0);
        freeze_with_block_id(&bank);
        let empty_root = receipts_root(&[]);
        assert_eq!(bank.receipts_root(), Some(empty_root));
        let commitment = bank.block_commitment().unwrap();
        let mut vector = [INACTIVE_COMMITMENT_SLOT; COMMITMENT_VECTOR_LEN];
        vector[CommitmentSlot::BlockHistoryRoot.index()] = commitment.block_history_root();
        vector[CommitmentSlot::ReceiptsRoot.index()] = empty_root;
        assert_eq!(
            bank.commitment_root(),
            Some(CommitmentVector(vector).root())
        );
        assert!(bank.transaction_receipt_inclusion_proof(0).is_none());
    }

    #[test]
    fn test_inactive_features() {
        // Base feature only: slot 1 stays zero, nothing is accumulated.
        let base_only = setup(false, true);
        let bank = new_child(&base_only.bank0);
        assert!(!bank.is_transaction_receipt_tree_active());
        bank.transfer(1, &base_only.mint_keypair, &Pubkey::new_unique())
            .unwrap();
        assert_eq!(bank.receipts_root(), None);
        assert!(bank.transaction_receipt_leaf_hashes().is_empty());
        freeze_with_block_id(&bank);
        let commitment = bank.block_commitment().unwrap();
        let mut vector = [INACTIVE_COMMITMENT_SLOT; COMMITMENT_VECTOR_LEN];
        vector[CommitmentSlot::BlockHistoryRoot.index()] = commitment.block_history_root();
        assert_eq!(
            bank.commitment_root(),
            Some(CommitmentVector(vector).root())
        );

        // Receipt feature without the base feature: no commitment at all.
        let receipts_only = setup(true, false);
        let bank = new_child(&receipts_only.bank0);
        assert!(!bank.is_transaction_receipt_tree_active());
        bank.transfer(1, &receipts_only.mint_keypair, &Pubkey::new_unique())
            .unwrap();
        assert!(bank.transaction_receipt_leaf_hashes().is_empty());
        freeze_with_block_id(&bank);
        assert_eq!(bank.commitment_root(), None);
    }

    #[test]
    fn test_block_positions_order_receipts() {
        let setup = setup(true, true);
        let bank = new_child(&setup.bank0);
        let txs: Vec<_> = (0..4)
            .map(|_| {
                system_transaction::transfer(
                    &setup.mint_keypair,
                    &Pubkey::new_unique(),
                    1_000_000,
                    bank.last_blockhash(),
                )
            })
            .collect();
        // Commit batches out of block order, with block positions 10..14,
        // as a replaying or leader bank would.
        for (batch_txs, positions) in [
            (vec![txs[2].clone(), txs[3].clone()], vec![12, 13]),
            (vec![txs[0].clone()], vec![10]),
            (vec![txs[1].clone()], vec![11]),
        ] {
            let batch = bank.prepare_batch_for_tests(batch_txs);
            let (commit_results, _) = bank
                .load_execute_and_commit_transactions_with_pre_commit_callback(
                    &batch,
                    ExecutionRecordingConfig::new_single_setting(false),
                    &mut ExecuteTimings::default(),
                    None,
                    ReceiptOrder::BlockPositions(&positions),
                    |_| Ok(()),
                )
                .unwrap();
            assert!(commit_results.iter().all(|result| result.is_ok()));
        }
        let expected: Vec<_> = txs
            .iter()
            .map(|tx| {
                let record = InstructionRecord::new(
                    1,
                    system_program::id().to_bytes(),
                    [
                        meta(&tx.message.account_keys[0], true, true),
                        meta(&tx.message.account_keys[1], false, true),
                    ],
                    &tx.message.instructions[0].data,
                );
                TransactionReceipt::success(
                    signature_bytes(tx),
                    message_hash(tx),
                    5000,
                    instruction_trace_root(&[record]),
                    events_root(&[]),
                )
                .leaf_hash()
            })
            .collect();
        assert_eq!(bank.transaction_receipt_leaf_hashes(), expected);
    }
}
