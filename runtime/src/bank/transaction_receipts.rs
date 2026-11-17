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
    super::{Bank, block_commitment::CommitmentFailure},
    agave_feature_set as feature_set,
    solana_hash::Hash,
    solana_merkle_tree::{
        commitment_tree::{InclusionProof, make_inclusion_proof_from_leaf_hashes},
        receipt_tree::{TransactionReceipt, receipts_root_from_leaf_hashes},
    },
    solana_runtime_transaction::transaction_with_meta::TransactionWithMeta,
    solana_svm::{
        transaction_execution_result::{ReceiptLeaves, ReceiptRoots},
        transaction_processing_result::{ProcessedTransaction, TransactionProcessingResult},
    },
    std::sync::{Arc, Mutex},
};

/// A transaction receipt kept for light-client provers
/// (`RuntimeConfig::retain_light_client_proof_data`), with the leaves of its
/// instruction trace and events trees.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetainedReceipt {
    pub receipt: TransactionReceipt,
    /// The records and events under the receipt's roots: empty for a failed
    /// receipt. `None` only if the SVM did not return them for a successful
    /// transaction (it always does when they are requested).
    pub leaves: Option<Arc<ReceiptLeaves>>,
}

/// How the transactions of a committed batch are placed in the block, for
/// the order of their receipts.
#[derive(Clone, Copy, Debug)]
pub enum ReceiptOrder<'a> {
    /// `positions[i]` is the position in the block of transaction `i` of
    /// the batch. Entries of transactions that are not committed are
    /// ignored. Used by replay and block production. A committed transaction
    /// without an entry is a commitment failure.
    BlockPositions(&'a [usize]),
    /// Receipts are appended in commit order. Only sound when a single caller
    /// commits all transactions of the bank sequentially (tests, tools that
    /// execute transactions directly against a bank). Mixing it with
    /// [`Self::BlockPositions`] in one bank is a commitment failure, since
    /// the positions could collide.
    CommitOrder,
    /// The caller does not know where the batch's transactions are in the
    /// block: block production when the record channel does not give a
    /// transaction index (the validator always enables it). Committing a
    /// transaction this way is a commitment failure.
    Unknown,
}

/// Receipts committed to a bank so far.
#[derive(Debug, Default)]
pub(crate) struct ReceiptAccumulator {
    state: Mutex<ReceiptAccumulatorState>,
}

#[derive(Debug, Default)]
struct ReceiptAccumulatorState {
    /// `(position, leaf hash)` of every receipt.
    entries: Vec<(u64, Hash)>,
    /// `(position, receipt)` of every receipt, if the bank keeps them for
    /// light-client provers. Same positions as `entries`.
    retained: Vec<(u64, RetainedReceipt)>,
    /// Whether a batch was committed with [`ReceiptOrder::BlockPositions`].
    uses_block_positions: bool,
    /// Whether a batch was committed with [`ReceiptOrder::CommitOrder`].
    uses_commit_order: bool,
    /// The first reason the receipts cannot be put in block order.
    failure: Option<CommitmentFailure>,
}

impl ReceiptAccumulatorState {
    fn fail(&mut self, failure: CommitmentFailure) {
        self.failure.get_or_insert(failure);
    }
}

impl ReceiptAccumulator {
    /// Adds the receipts of a batch, `(index in the batch, leaf hash,
    /// receipt to keep)`, or the failure to compute one of them.
    fn push(
        &self,
        order: ReceiptOrder,
        receipts: impl Iterator<
            Item = Result<(usize, Hash, Option<RetainedReceipt>), CommitmentFailure>,
        >,
    ) {
        let mut state = self.state.lock().unwrap();
        let state = &mut *state;
        let mut num_receipts = 0usize;
        for receipt in receipts {
            let (index_in_batch, leaf_hash, retained) = match receipt {
                Ok(receipt) => receipt,
                Err(failure) => {
                    state.fail(failure);
                    continue;
                }
            };
            num_receipts = num_receipts.saturating_add(1);
            let position = match order {
                ReceiptOrder::BlockPositions(positions) => match positions.get(index_in_batch) {
                    Some(&position) => position as u64,
                    None => {
                        state.fail(CommitmentFailure::MissingBlockPosition { index_in_batch });
                        continue;
                    }
                },
                ReceiptOrder::CommitOrder => state.entries.len() as u64,
                ReceiptOrder::Unknown => continue,
            };
            state.entries.push((position, leaf_hash));
            if let Some(retained) = retained {
                state.retained.push((position, retained));
            }
        }
        if num_receipts == 0 {
            return;
        }
        match order {
            ReceiptOrder::BlockPositions(_) => state.uses_block_positions = true,
            ReceiptOrder::CommitOrder => state.uses_commit_order = true,
            ReceiptOrder::Unknown => {
                state.fail(CommitmentFailure::UnknownReceiptOrder { num_receipts });
            }
        }
        if state.uses_block_positions && state.uses_commit_order {
            state.fail(CommitmentFailure::MixedReceiptOrder);
        }
    }

    /// Leaf hashes in block order, or why they cannot be put in block order.
    fn leaf_hashes_in_block_order(&self) -> Result<Vec<Hash>, CommitmentFailure> {
        let mut entries = {
            let state = self.state.lock().unwrap();
            if let Some(failure) = &state.failure {
                return Err(failure.clone());
            }
            state.entries.clone()
        };
        entries.sort_unstable_by_key(|(position, _)| *position);
        if let Some(pair) = entries.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Err(CommitmentFailure::DuplicateBlockPosition {
                position: pair[0].0,
            });
        }
        Ok(entries
            .into_iter()
            .map(|(_, leaf_hash)| leaf_hash)
            .collect())
    }

    /// Kept receipts in block order, or why the receipts cannot be put in
    /// block order. Fails as well if not every receipt was kept.
    fn retained_in_block_order(&self) -> Result<Vec<RetainedReceipt>, CommitmentFailure> {
        let (mut retained, num_entries) = {
            let state = self.state.lock().unwrap();
            if let Some(failure) = &state.failure {
                return Err(failure.clone());
            }
            (state.retained.clone(), state.entries.len())
        };
        if retained.len() != num_entries {
            return Err(CommitmentFailure::ReceiptCountMismatch {
                committed: num_entries as u64,
                receipts: retained.len() as u64,
            });
        }
        retained.sort_unstable_by_key(|(position, _)| *position);
        if let Some(pair) = retained.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Err(CommitmentFailure::DuplicateBlockPosition {
                position: pair[0].0,
            });
        }
        Ok(retained.into_iter().map(|(_, receipt)| receipt).collect())
    }
}

/// The receipt of a processed transaction.
///
/// Fails if the transaction executed successfully but has no receipt roots,
/// which the SVM always computes while the feature is active.
pub fn transaction_receipt(
    transaction: &impl TransactionWithMeta,
    processed_transaction: &ProcessedTransaction,
) -> Result<TransactionReceipt, CommitmentFailure> {
    let signature = (*transaction.signature()).into();
    let message_hash = *transaction.message_hash();
    Ok(match processed_transaction {
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
                    return Err(CommitmentFailure::MissingReceiptRoots {
                        signature: *transaction.signature(),
                    });
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
    })
}

/// The leaves under the receipt of a processed transaction, as kept for
/// light-client provers.
fn retained_receipt_leaves(
    processed_transaction: &ProcessedTransaction,
) -> Option<Arc<ReceiptLeaves>> {
    match processed_transaction {
        ProcessedTransaction::Executed(executed_tx)
            if executed_tx.execution_details.status.is_ok() =>
        {
            executed_tx
                .execution_details
                .receipt_leaves
                .as_deref()
                .cloned()
                .map(Arc::new)
        }
        // A failed receipt commits to empty trees.
        _ => Some(Arc::default()),
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
    /// receipt. A receipt that cannot be computed or placed in the block is
    /// recorded as a commitment failure, reported when the bank is frozen.
    pub(super) fn accumulate_transaction_receipts(
        &self,
        transactions: &[impl TransactionWithMeta],
        processing_results: &[TransactionProcessingResult],
        order: ReceiptOrder,
    ) {
        if !self.is_transaction_receipt_tree_active() {
            return;
        }
        let retain = self.retain_light_client_proof_data;
        let receipts = transactions
            .iter()
            .zip(processing_results)
            .enumerate()
            .filter_map(|(index_in_batch, (transaction, processing_result))| {
                let processed_transaction = processing_result.as_ref().ok()?;
                Some(
                    transaction_receipt(transaction, processed_transaction).map(|receipt| {
                        let retained = retain.then(|| RetainedReceipt {
                            receipt,
                            leaves: retained_receipt_leaves(processed_transaction),
                        });
                        (index_in_batch, receipt.leaf_hash(), retained)
                    }),
                )
            });
        self.transaction_receipts.push(order, receipts);
    }

    /// The receipts committed so far with the leaves of their trees, in
    /// block order, if this bank keeps them for light-client provers
    /// (`RuntimeConfig::retain_light_client_proof_data`); `None` otherwise
    /// and while the feature is inactive. Its leaf hashes are
    /// [`Self::transaction_receipt_leaf_hashes`].
    pub fn retained_transaction_receipts(
        &self,
    ) -> Option<Result<Vec<RetainedReceipt>, CommitmentFailure>> {
        (self.retain_light_client_proof_data && self.is_transaction_receipt_tree_active())
            .then(|| self.transaction_receipts.retained_in_block_order())
    }

    /// Leaf hashes of the receipts committed so far, in block order, or the
    /// commitment failure that prevents putting them in block order.
    pub fn transaction_receipt_leaf_hashes(&self) -> Result<Vec<Hash>, CommitmentFailure> {
        self.transaction_receipts.leaf_hashes_in_block_order()
    }

    /// `receipts_root` over the receipts committed so far, `None` while the
    /// feature is inactive, and while the receipts cannot be put in block
    /// order (see [`Self::transaction_receipt_leaf_hashes`]). On a frozen
    /// bank with a commitment root this is the value in slot 1 of its
    /// commitment vector.
    pub fn receipts_root(&self) -> Option<Hash> {
        if !self.is_transaction_receipt_tree_active() {
            return None;
        }
        let leaf_hashes = self.transaction_receipt_leaf_hashes().ok()?;
        Some(receipts_root_from_leaf_hashes(&leaf_hashes))
    }

    /// Proof of the receipt at `index` (in block order) against
    /// [`Self::receipts_root`].
    pub fn transaction_receipt_inclusion_proof(&self, index: usize) -> Option<InclusionProof> {
        make_inclusion_proof_from_leaf_hashes(&self.transaction_receipt_leaf_hashes().ok()?, index)
    }

    /// `receipts_root` for the commitment vector (`None` while the feature is
    /// inactive), checking that every transaction committed to this bank has
    /// exactly one receipt and that the receipts are in block order.
    ///
    /// The number of committed transactions is the bank's own count
    /// ([`Bank::committed_transaction_count`]), so the check applies to every
    /// bank, with or without a parent: it starts at zero for a bank created
    /// from genesis (which commits no transactions), from a snapshot (whose
    /// block is not executed again, and whose receipts are not stored), or by
    /// test and tool constructors that take a cumulative `transaction_count`.
    pub(super) fn receipts_root_for_commitment(&self) -> Result<Option<Hash>, CommitmentFailure> {
        if !self.is_transaction_receipt_tree_active() {
            return Ok(None);
        }
        let leaf_hashes = self.transaction_receipt_leaf_hashes()?;
        let committed = self.committed_transaction_count();
        let receipts = leaf_hashes.len() as u64;
        if committed != receipts {
            return Err(CommitmentFailure::ReceiptCountMismatch {
                committed,
                receipts,
            });
        }
        Ok(Some(receipts_root_from_leaf_hashes(&leaf_hashes)))
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
                    INACTIVE_COMMITMENT_SLOT, tests::assert_commitment_failure,
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
                EventKind, EventRecord, InstructionAccountMeta, InstructionRecord, ReceiptStatus,
                empty_events_root, event_inclusion_proof, events_root,
                instruction_record_inclusion_proof, instruction_trace_root, log_data_payload,
                receipts_root, verify_event, verify_instruction_record, verify_receipt,
            },
        },
        solana_pubkey::Pubkey,
        solana_runtime_transaction::runtime_transaction::RuntimeTransaction,
        solana_sdk_ids::{bpf_loader_upgradeable, system_program},
        solana_signer::Signer,
        solana_svm::{
            account_loader::LoadedTransaction,
            transaction_execution_result::{ExecutedTransaction, TransactionExecutionDetails},
            transaction_processor::ExecutionRecordingConfig,
        },
        solana_svm_timings::ExecuteTimings,
        solana_system_transaction as system_transaction,
        solana_transaction::Transaction,
        std::{collections::HashMap, sync::Arc},
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
        assert_eq!(
            bank.transaction_receipt_leaf_hashes().unwrap(),
            expected_leaves
        );
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

    /// With `retain_light_client_proof_data`, the bank keeps every receipt
    /// with the records and events under its roots, in block order, and
    /// computes the same receipts as a bank that does not keep them.
    #[test]
    fn test_retained_receipts() {
        let setup = setup(true, true);
        let new_bank = |slot, retain| {
            let mut bank = Bank::new_from_parent(setup.bank0.clone(), *setup.bank0.leader(), slot);
            bank.set_retain_light_client_proof_data_for_tests(retain);
            Arc::new(bank)
        };
        let recipient = Pubkey::new_unique();
        let extra = [0xaa, 0xbb];
        let run = |bank: &Bank| {
            bank.transfer(1_000_000_000, &setup.mint_keypair, &recipient)
                .unwrap();
            let ok_tx = program_transaction(&setup, bank, &recipient, &[0, 0xaa, 0xbb]);
            bank.process_transaction(&ok_tx).unwrap();
            let fail_tx = program_transaction(&setup, bank, &recipient, &[2]);
            assert!(bank.process_transaction(&fail_tx).is_err());
            (ok_tx, fail_tx)
        };
        let kept = new_bank(1, true);
        let (ok_tx, fail_tx) = run(&kept);
        let not_kept = new_bank(2, false);
        run(&not_kept);
        assert_eq!(not_kept.retained_transaction_receipts(), None);
        // Keeping the receipts does not change them.
        assert_eq!(
            kept.transaction_receipt_leaf_hashes().unwrap(),
            not_kept.transaction_receipt_leaf_hashes().unwrap()
        );

        let retained = kept.retained_transaction_receipts().unwrap().unwrap();
        assert_eq!(retained.len(), 3);
        assert_eq!(
            retained
                .iter()
                .map(|retained| retained.receipt.leaf_hash())
                .collect::<Vec<_>>(),
            kept.transaction_receipt_leaf_hashes().unwrap()
        );
        let payer = setup.mint_keypair.pubkey();
        let transfer_records = vec![InstructionRecord::new(
            1,
            system_program::id().to_bytes(),
            [meta(&payer, true, true), meta(&recipient, false, true)],
            &system_transaction::transfer(
                &setup.mint_keypair,
                &recipient,
                1_000_000_000,
                kept.last_blockhash(),
            )
            .message
            .instructions[0]
                .data,
        )];
        let (records, events) = expected_trace_and_events(&setup, &recipient, &extra);
        let expected_leaves = [
            ReceiptLeaves {
                records: transfer_records,
                events: vec![],
            },
            ReceiptLeaves { records, events },
            ReceiptLeaves::default(),
        ];
        for (retained, expected) in retained.iter().zip(&expected_leaves) {
            let leaves = retained.leaves.as_deref().unwrap();
            assert_eq!(leaves, expected);
            let roots = leaves.roots();
            if retained.receipt.status == ReceiptStatus::Success {
                assert_eq!(
                    roots.instruction_trace_root,
                    retained.receipt.instruction_trace_root
                );
                assert_eq!(roots.events_root, retained.receipt.events_root);
            }
        }
        assert_eq!(retained[1].receipt.signature, signature_bytes(&ok_tx));
        assert_eq!(retained[2].receipt.signature, signature_bytes(&fail_tx));
        assert_eq!(retained[2].receipt.status, ReceiptStatus::Failed);

        // The kept receipts follow the block positions, not commit order.
        let bank = new_bank(3, true);
        let tx_a = system_transaction::transfer(
            &setup.mint_keypair,
            &Pubkey::new_unique(),
            1,
            bank.last_blockhash(),
        );
        let tx_b = system_transaction::transfer(
            &setup.mint_keypair,
            &Pubkey::new_unique(),
            2,
            bank.last_blockhash(),
        );
        for (tx, position) in [(&tx_a, 7), (&tx_b, 3)] {
            let batch = bank.prepare_batch_for_tests(vec![tx.clone()]);
            let (results, _) = bank
                .load_execute_and_commit_transactions_with_pre_commit_callback(
                    &batch,
                    ExecutionRecordingConfig::new_single_setting(false),
                    &mut ExecuteTimings::default(),
                    None,
                    ReceiptOrder::BlockPositions(&[position]),
                    |_| Ok(()),
                )
                .unwrap();
            assert!(results[0].is_ok());
        }
        let signatures: Vec<_> = bank
            .retained_transaction_receipts()
            .unwrap()
            .unwrap()
            .iter()
            .map(|retained| retained.receipt.signature)
            .collect();
        assert_eq!(signatures, [signature_bytes(&tx_b), signature_bytes(&tx_a)]);
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
                bank.transaction_receipt_leaf_hashes().unwrap()[1],
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
        assert!(bank.transaction_receipt_leaf_hashes().unwrap().is_empty());
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
        assert!(bank.transaction_receipt_leaf_hashes().unwrap().is_empty());
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
        assert_eq!(bank.transaction_receipt_leaf_hashes().unwrap(), expected);
    }
    fn transfers(setup: &Setup, bank: &Bank, count: usize) -> Vec<Transaction> {
        (0..count)
            .map(|_| {
                system_transaction::transfer(
                    &setup.mint_keypair,
                    &Pubkey::new_unique(),
                    1_000_000,
                    bank.last_blockhash(),
                )
            })
            .collect()
    }

    /// Commits `txs` as one batch whose receipts are placed with `order`.
    fn commit_batch(bank: &Bank, txs: Vec<Transaction>, order: ReceiptOrder) {
        let batch = bank.prepare_batch_for_tests(txs);
        let (commit_results, _) = bank
            .load_execute_and_commit_transactions_with_pre_commit_callback(
                &batch,
                ExecutionRecordingConfig::new_single_setting(false),
                &mut ExecuteTimings::default(),
                None,
                order,
                |_| Ok(()),
            )
            .unwrap();
        assert!(commit_results.iter().all(|result| result.is_ok()));
    }

    /// The receipts of `bank` cannot be put in block order because of
    /// `failure`: no receipts root, and once frozen, no commitment root and a
    /// failed verification with that reason.
    fn assert_receipt_failure(bank: &Arc<Bank>, failure: CommitmentFailure) {
        assert_eq!(bank.transaction_receipt_leaf_hashes(), Err(failure.clone()));
        assert_eq!(bank.receipts_root(), None);
        assert!(bank.transaction_receipt_inclusion_proof(0).is_none());
        freeze_with_block_id(bank);
        assert_commitment_failure(bank, failure);
    }

    #[test]
    fn test_missing_block_position_is_commitment_failure() {
        let setup = setup(true, true);
        let bank = new_child(&setup.bank0);
        // Two committed transactions, one position.
        commit_batch(
            &bank,
            transfers(&setup, &bank, 2),
            ReceiptOrder::BlockPositions(&[0]),
        );
        assert_receipt_failure(
            &bank,
            CommitmentFailure::MissingBlockPosition { index_in_batch: 1 },
        );
    }

    #[test]
    fn test_duplicate_block_position_is_commitment_failure() {
        let setup = setup(true, true);
        let bank = new_child(&setup.bank0);
        for tx in transfers(&setup, &bank, 2) {
            commit_batch(&bank, vec![tx], ReceiptOrder::BlockPositions(&[3]));
        }
        assert_receipt_failure(
            &bank,
            CommitmentFailure::DuplicateBlockPosition { position: 3 },
        );
    }

    #[test]
    fn test_mixed_receipt_order_is_commitment_failure() {
        let setup = setup(true, true);
        let bank = new_child(&setup.bank0);
        // Block position 0, then a transaction in commit order, which would
        // also get position 1 or less.
        let mut txs = transfers(&setup, &bank, 2);
        commit_batch(
            &bank,
            vec![txs.remove(0)],
            ReceiptOrder::BlockPositions(&[0]),
        );
        commit_batch(&bank, txs, ReceiptOrder::CommitOrder);
        assert_receipt_failure(&bank, CommitmentFailure::MixedReceiptOrder);

        // Either way round.
        let bank = new_child(&setup.bank0);
        bank.transfer(1, &setup.mint_keypair, &Pubkey::new_unique())
            .unwrap();
        commit_batch(
            &bank,
            transfers(&setup, &bank, 1),
            ReceiptOrder::BlockPositions(&[7]),
        );
        assert_receipt_failure(&bank, CommitmentFailure::MixedReceiptOrder);
    }

    #[test]
    fn test_unknown_receipt_order_is_commitment_failure() {
        let setup = setup(true, true);
        let bank = new_child(&setup.bank0);
        commit_batch(&bank, transfers(&setup, &bank, 2), ReceiptOrder::Unknown);
        assert_receipt_failure(
            &bank,
            CommitmentFailure::UnknownReceiptOrder { num_receipts: 2 },
        );

        // A batch without committed transactions has no receipt to place.
        let bank = new_child(&setup.bank0);
        let batch = bank.prepare_batch_for_tests(Vec::<Transaction>::new());
        bank.load_execute_and_commit_transactions_with_pre_commit_callback(
            &batch,
            ExecutionRecordingConfig::new_single_setting(false),
            &mut ExecuteTimings::default(),
            None,
            ReceiptOrder::Unknown,
            |_| Ok(()),
        )
        .unwrap();
        freeze_with_block_id(&bank);
        assert_eq!(bank.commitment_failure(), None);
        assert_eq!(bank.verify_commitment_root(), Ok(()));
    }

    #[test]
    fn test_receipt_count_mismatch_is_commitment_failure() {
        let setup = setup(true, true);
        let bank = new_child(&setup.bank0);
        commit_batch(
            &bank,
            transfers(&setup, &bank, 1),
            ReceiptOrder::BlockPositions(&[0]),
        );
        // A receipt for a transaction that was not committed.
        bank.transaction_receipts.push(
            ReceiptOrder::BlockPositions(&[1]),
            std::iter::once(Ok((0, Hash::new_unique(), None))),
        );
        assert_eq!(bank.transaction_receipt_leaf_hashes().unwrap().len(), 2);
        freeze_with_block_id(&bank);
        assert_commitment_failure(
            &bank,
            CommitmentFailure::ReceiptCountMismatch {
                committed: 1,
                receipts: 2,
            },
        );
    }

    /// A bank created from genesis, with both features active, that is not
    /// frozen yet and has no parent.
    fn new_genesis_bank() -> (Arc<Bank>, Keypair) {
        let GenesisConfigInfo {
            mut genesis_config,
            mint_keypair,
            ..
        } = create_genesis_config_with_leader(1_000_000_000_000, &Pubkey::new_unique(), 1_000_000);
        activate_feature(
            &mut genesis_config,
            feature_set::transaction_receipt_tree::id(),
        );
        activate_feature(
            &mut genesis_config,
            feature_set::block_commitment_root::id(),
        );
        let bank = Arc::new(Bank::new_for_tests(&genesis_config));
        assert!(bank.parent().is_none());
        (bank, mint_keypair)
    }

    fn genesis_transfers(bank: &Bank, mint_keypair: &Keypair, count: usize) -> Vec<Transaction> {
        (0..count)
            .map(|_| {
                system_transaction::transfer(
                    mint_keypair,
                    &Pubkey::new_unique(),
                    1_000_000,
                    bank.last_blockhash(),
                )
            })
            .collect()
    }

    #[test]
    fn test_receipt_count_checked_without_parent() {
        // Genesis commits no transactions, so a bank created from it counts
        // from zero.
        let (bank, mint_keypair) = new_genesis_bank();
        assert_eq!(bank.committed_transaction_count(), 0);
        commit_batch(
            &bank,
            genesis_transfers(&bank, &mint_keypair, 2),
            ReceiptOrder::BlockPositions(&[0, 1]),
        );
        assert_eq!(bank.committed_transaction_count(), 2);
        let leaf_hashes = bank.transaction_receipt_leaf_hashes().unwrap();
        assert_eq!(leaf_hashes.len(), 2);
        freeze_with_block_id(&bank);
        assert_eq!(bank.commitment_failure(), None);
        assert_eq!(bank.verify_commitment_root(), Ok(()));
        assert_eq!(
            bank.receipts_root(),
            Some(receipts_root_from_leaf_hashes(&leaf_hashes))
        );

        // Without a parent, a receipt for a transaction that was not
        // committed is caught as well.
        let (bank, mint_keypair) = new_genesis_bank();
        commit_batch(
            &bank,
            genesis_transfers(&bank, &mint_keypair, 1),
            ReceiptOrder::BlockPositions(&[0]),
        );
        bank.transaction_receipts.push(
            ReceiptOrder::BlockPositions(&[1]),
            std::iter::once(Ok((0, Hash::new_unique(), None))),
        );
        freeze_with_block_id(&bank);
        assert_commitment_failure(
            &bank,
            CommitmentFailure::ReceiptCountMismatch {
                committed: 1,
                receipts: 2,
            },
        );

        // A parentless bank whose cumulative transaction count does not start
        // at zero, as the test and tool constructors that take bank fields
        // create (`Bank::new_for_block_tests`), compares its own count.
        let (bank, mint_keypair) = new_genesis_bank();
        bank.transaction_count
            .store(1_000, std::sync::atomic::Ordering::Relaxed);
        commit_batch(
            &bank,
            genesis_transfers(&bank, &mint_keypair, 1),
            ReceiptOrder::BlockPositions(&[0]),
        );
        assert_eq!(bank.transaction_count(), 1_001);
        assert_eq!(bank.committed_transaction_count(), 1);
        freeze_with_block_id(&bank);
        assert_eq!(bank.commitment_failure(), None);
        assert_eq!(bank.verify_commitment_root(), Ok(()));
    }

    #[test]
    fn test_committed_transaction_count_is_per_bank() {
        let setup = setup(true, true);
        let bank1 = new_child(&setup.bank0);
        commit_batch(
            &bank1,
            transfers(&setup, &bank1, 2),
            ReceiptOrder::BlockPositions(&[0, 1]),
        );
        freeze_with_block_id(&bank1);
        let bank2 = new_child(&bank1);
        assert_eq!(bank2.committed_transaction_count(), 0);
        assert_eq!(bank2.transaction_count(), bank1.transaction_count());
        commit_batch(
            &bank2,
            transfers(&setup, &bank2, 1),
            ReceiptOrder::BlockPositions(&[0]),
        );
        assert_eq!(bank2.committed_transaction_count(), 1);
        assert_eq!(bank2.executed_transaction_count(), 1);
        freeze_with_block_id(&bank2);
        assert_eq!(bank2.verify_commitment_root(), Ok(()));
    }

    #[test]
    fn test_missing_receipt_roots_is_commitment_failure() {
        let setup = setup(true, true);
        let bank = new_child(&setup.bank0);
        let tx = transfers(&setup, &bank, 1).remove(0);
        let signature = tx.signatures[0];
        // Executed successfully, but without the roots that the SVM computes
        // while the feature is active.
        let processing_result = Ok(ProcessedTransaction::Executed(Box::new(
            ExecutedTransaction {
                loaded_transaction: LoadedTransaction::default(),
                execution_details: TransactionExecutionDetails {
                    status: Ok(()),
                    log_messages: None,
                    inner_instructions: None,
                    return_data: None,
                    executed_units: 0,
                    accounts_deltas: None,
                    receipt_roots: None,
                    receipt_leaves: None,
                },
                programs_modified_by_tx: HashMap::new(),
            },
        )));
        bank.accumulate_transaction_receipts(
            &[RuntimeTransaction::from_transaction_for_tests(tx)],
            &[processing_result],
            ReceiptOrder::BlockPositions(&[0]),
        );
        assert_receipt_failure(&bank, CommitmentFailure::MissingReceiptRoots { signature });
    }
}
