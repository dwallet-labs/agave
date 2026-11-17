//! End to end: a chain of banks with the six light-client commitment features
//! activating together at the first epoch boundary, proofs produced from the
//! banks, certificates signed with the validators' BLS keys, and a light
//! client that verifies blocks, an ancestor and its anchor, a transaction, an
//! instruction and an event, accounts (delta, unchanged, current state,
//! non-existence), both epoch boundaries, and the hand-off to the next epoch.
//!
//! Two inputs are made up because the banks are not produced from shreds:
//! the block id of every block is the root of a block id tree built with the
//! ledger's own functions from made-up FEC set roots and the real final leaf
//! (parent, FEC set count, and for blocks with the features active the bank
//! hash and commitment root). The banks do not keep the leaves of the block
//! history (only its frontier) nor receipts, traces and events (only receipt
//! leaf hashes), so the test keeps the history leaves and rebuilds the
//! receipt of the proven transaction from the program's source, as a prover
//! that stores blocks would.

#![cfg(feature = "agave-unstable-api")]
#![allow(clippy::arithmetic_side_effects)]

use {
    agave_bls_cert_verify::cert_verify::test_create_base2_unverified_certificate,
    agave_feature_set as feature_set,
    agave_votor_messages::{
        certificate::CertificateType,
        consensus_message::{Block, BlockId},
        unverified_vote_message::UnverifiedCertificate,
    },
    solana_account::{AccountSharedData, ReadableAccount, WritableAccount},
    solana_bls_signatures::keypair::Keypair as BlsKeypair,
    solana_entry::block_component::FooterBinding,
    solana_fee_calculator::FeeRateGovernor,
    solana_genesis_config::GenesisConfig,
    solana_hash::Hash,
    solana_instruction::{AccountMeta, Instruction},
    solana_keypair::Keypair,
    solana_ledger::shred::merkle_tree::{MerkleTree, double_merkle_final_leaf},
    solana_light_client::{
        AccountDeltaProof, AccountStateProof, AccountUnchangedProof, BlsSignature, CommitmentSlot,
        EpochBoundaryProof, EpochSchedule, Error, EventProof, FinalityProof, FooterProof,
        HistoryProof, InstructionProof, LightClient, SignerSet, SlotOpening, TransactionProof,
        account_hash, history_leaf,
        solana_merkle_tree::{
            commitment_tree::{TreeId, compute_root, make_inclusion_proof},
            receipt_tree::{
                EventKind, EventRecord, InstructionAccountMeta, InstructionRecord,
                TransactionReceipt, event_inclusion_proof, events_root,
                instruction_record_inclusion_proof, instruction_trace_root, log_data_payload,
            },
        },
    },
    solana_pubkey::Pubkey,
    solana_runtime::{
        bank::{Bank, test_utils::goto_end_of_slot},
        genesis_utils::{
            GenesisConfigInfo, ValidatorVoteKeypairs, create_genesis_config_with_vote_accounts,
            minimum_vote_account_balance_for_vat,
        },
        loader_utils::create_program_with_elf,
    },
    solana_runtime_transaction::runtime_transaction::RuntimeTransaction,
    solana_sdk_ids::{bpf_loader_upgradeable, system_program},
    solana_signer::Signer,
    solana_system_transaction as system_transaction,
    solana_transaction::Transaction,
    std::sync::Arc,
};

const SLOTS_PER_EPOCH: u64 = 64;
const SHRED_VERSION: u16 = 4242;
const LAMPORTS_PER_SIGNATURE: u64 = 5000;

/// See `runtime/tests/fixtures/receipt_test_program/receipt_test_program.c`:
/// with data `[0, extra..]` it logs "outer", invokes itself with `[1]` (which
/// logs data `["inner", ""]` and transfers one lamport from account 0 to
/// account 1 through the system program), then logs data `["after", extra]`.
const RECEIPT_TEST_PROGRAM: &[u8] =
    include_bytes!("../../runtime/tests/fixtures/receipt_test_program/receipt_test_program.so");

fn all_features() -> [Pubkey; 6] {
    [
        feature_set::block_commitment_root::id(),
        feature_set::transaction_receipt_tree::id(),
        feature_set::accounts_delta_root::id(),
        feature_set::epoch_validator_set_root::id(),
        feature_set::epoch_boundary_commitment::id(),
        feature_set::accounts_state_root::id(),
    ]
}

/// Three validators with BLS keys, 64-slot epochs with a one-epoch leader
/// schedule offset, non-zero fees, and the six features pending, so that
/// they activate at the first epoch boundary.
fn genesis() -> (GenesisConfig, Keypair, Vec<ValidatorVoteKeypairs>) {
    let validators: Vec<_> = (0..3).map(|_| ValidatorVoteKeypairs::new_rand()).collect();
    let stakes = vec![
        50 * minimum_vote_account_balance_for_vat(100),
        30 * minimum_vote_account_balance_for_vat(100),
        20 * minimum_vote_account_balance_for_vat(100),
    ];
    let GenesisConfigInfo {
        mut genesis_config,
        mint_keypair,
        ..
    } = create_genesis_config_with_vote_accounts(1_000_000_000_000, &validators, stakes);
    genesis_config.epoch_schedule = EpochSchedule::custom(SLOTS_PER_EPOCH, SLOTS_PER_EPOCH, false);
    genesis_config.fee_rate_governor = FeeRateGovernor::new(LAMPORTS_PER_SIGNATURE, 0);
    for feature_id in all_features() {
        let lamports = genesis_config
            .rent
            .minimum_balance(solana_feature_gate_interface::Feature::size_of())
            .max(1);
        genesis_config.accounts.insert(
            feature_id,
            solana_feature_gate_interface::create_account(
                &solana_feature_gate_interface::Feature { activated_at: None },
                lamports,
            )
            .into(),
        );
    }
    (genesis_config, mint_keypair, validators)
}

fn new_child(parent: &Arc<Bank>, slot: u64) -> Arc<Bank> {
    Arc::new(Bank::new_from_parent(
        parent.clone(),
        *parent.leader(),
        slot,
    ))
}

/// Freezes `bank` and gives it a block id: the root of a block id tree over
/// `fec_set_count` made-up FEC set roots and the final leaf built by the
/// ledger. The final leaf binds the footer's bank hash and commitment root
/// when the bank has a commitment root (its footer would be version 2).
/// Returns the footer proof of such a bank.
fn freeze_with_block_id(bank: &Arc<Bank>, fec_set_count: u32) -> Option<FooterProof> {
    goto_end_of_slot(bank.clone());
    bank.freeze();
    let Some(parent) = bank.parent() else {
        bank.set_block_id(Some(BlockId::from(Hash::new_unique())));
        return None;
    };
    let parent_block_id = parent.block_id().unwrap().to_hash();
    let binding = bank.commitment_root().map(|commitment_root| FooterBinding {
        bank_hash: bank.hash(),
        commitment_root,
    });
    let mut leaves: Vec<Hash> = (0..fec_set_count).map(|_| Hash::new_unique()).collect();
    leaves.push(double_merkle_final_leaf(
        parent.slot(),
        &parent_block_id,
        fec_set_count,
        binding.as_ref(),
    ));
    let tree = MerkleTree::try_new_with_len(leaves.iter().copied().map(Ok), leaves.len()).unwrap();
    bank.set_block_id(Some(BlockId::from(*tree.root())));
    let binding = binding?;
    Some(FooterProof {
        parent_slot: parent.slot(),
        parent_block_id,
        fec_set_count,
        bank_hash: binding.bank_hash,
        commitment_root: binding.commitment_root,
        proof: tree
            .make_merkle_proof(fec_set_count as usize, leaves.len())
            .map(|entry| *entry.unwrap())
            .collect(),
    })
}

fn block_id(bank: &Bank) -> Hash {
    bank.block_id().unwrap().to_hash()
}

fn leaf_of(bank: &Bank) -> [u8; 72] {
    history_leaf(
        bank.slot(),
        &block_id(bank),
        &bank.commitment_root().unwrap(),
    )
}

fn opening(bank: &Bank, slot: CommitmentSlot) -> SlotOpening {
    let proof = bank
        .commitment_vector()
        .unwrap()
        .inclusion_proof(slot.index() as usize)
        .unwrap();
    SlotOpening::from_inclusion_proof(slot, &proof).unwrap()
}

/// The validators' BLS keypairs in the rank order of `signer_set`.
fn keypairs_by_rank(
    validators: &[ValidatorVoteKeypairs],
    signer_set: &SignerSet,
) -> Vec<BlsKeypair> {
    signer_set
        .signers()
        .iter()
        .map(|signer| {
            validators
                .iter()
                .find(|v| {
                    v.bls_keypair.public.to_bytes_compressed() == signer.bls_pubkey_compressed
                })
                .unwrap()
                .bls_keypair
                .clone()
        })
        .collect()
}

/// A certificate signed by `ranks`, checked with the bank's own certificate
/// verification (the validator's rule) before it is handed to the client.
fn certificate(
    bank: &Bank,
    keypairs: &[BlsKeypair],
    cert_type: CertificateType,
    ranks: &[usize],
) -> (BlsSignature, Vec<u8>) {
    let cert = test_create_base2_unverified_certificate(keypairs, SHRED_VERSION, cert_type, ranks);
    let (signature, bitmap) = (cert.signature, cert.bitmap.clone());
    bank.verify_certificate(
        UnverifiedCertificate {
            cert_type,
            signature,
            bitmap: bitmap.clone(),
            shred_version: SHRED_VERSION,
        },
        None,
    )
    .expect("the validator accepts the certificate");
    (signature, bitmap)
}

fn slow_finality(bank: &Bank, keypairs: &[BlsKeypair], ranks: &[usize]) -> FinalityProof {
    let block = Block {
        slot: bank.slot(),
        block_id: bank.block_id().unwrap(),
    };
    FinalityProof::slow(
        certificate(bank, keypairs, CertificateType::Notarize(block), ranks),
        certificate(
            bank,
            keypairs,
            CertificateType::Finalize(bank.slot()),
            ranks,
        ),
    )
}

fn fast_finality(bank: &Bank, keypairs: &[BlsKeypair]) -> FinalityProof {
    let block = Block {
        slot: bank.slot(),
        block_id: bank.block_id().unwrap(),
    };
    let ranks: Vec<usize> = (0..keypairs.len()).collect();
    let (signature, bitmap) =
        certificate(bank, keypairs, CertificateType::FinalizeFast(block), &ranks);
    FinalityProof::fast(signature, bitmap)
}

fn meta(pubkey: &Pubkey, is_signer: bool, is_writable: bool) -> InstructionAccountMeta {
    InstructionAccountMeta {
        pubkey: pubkey.to_bytes(),
        is_signer,
        is_writable,
    }
}

fn message_hash(transaction: &Transaction) -> Hash {
    *RuntimeTransaction::from_transaction_for_tests(transaction.clone()).message_hash()
}

/// The program transaction of the test, its trace and events written from
/// the program's source.
struct ProgramTransaction {
    transaction: Transaction,
    records: Vec<InstructionRecord>,
    events: Vec<EventRecord>,
}

fn program_transaction(
    bank: &Bank,
    payer: &Keypair,
    program: &Pubkey,
    recipient: &Pubkey,
) -> ProgramTransaction {
    let extra = [0xaa, 0xbb];
    let data = [0, 0xaa, 0xbb];
    let instruction = Instruction::new_with_bytes(
        *program,
        &data,
        vec![
            AccountMeta::new(payer.pubkey(), true),
            AccountMeta::new(*recipient, false),
            AccountMeta::new_readonly(system_program::id(), false),
            AccountMeta::new_readonly(*program, false),
        ],
    );
    let transaction = Transaction::new_signed_with_payer(
        &[instruction],
        Some(&payer.pubkey()),
        &[payer],
        bank.last_blockhash(),
    );
    let accounts = [
        meta(&payer.pubkey(), true, true),
        meta(recipient, false, true),
        meta(&system_program::id(), false, false),
        meta(program, false, false),
    ];
    let mut transfer = vec![2, 0, 0, 0];
    transfer.extend_from_slice(&1u64.to_le_bytes());
    let records = vec![
        InstructionRecord::new(1, program.to_bytes(), accounts, &data),
        InstructionRecord::new(2, program.to_bytes(), accounts, &[1]),
        InstructionRecord::new(
            3,
            system_program::id().to_bytes(),
            [
                meta(&payer.pubkey(), true, true),
                meta(recipient, false, true),
            ],
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
        EventRecord {
            program_id: program.to_bytes(),
            instruction_index: 0,
            kind: EventKind::LogData,
            payload: log_data_payload(&[b"after".as_slice(), &extra]),
        },
    ];
    ProgramTransaction {
        transaction,
        records,
        events,
    }
}

fn hash_of(account: &AccountSharedData, address: &Pubkey) -> Hash {
    account_hash(
        account.lamports(),
        account.data(),
        account.executable(),
        account.owner(),
        address,
    )
}

fn transfer(bank: &Bank, from: &Keypair, to: &Pubkey, lamports: u64) {
    bank.process_transaction(&system_transaction::transfer(
        from,
        to,
        lamports,
        bank.last_blockhash(),
    ))
    .unwrap();
}

#[test]
fn test_light_client_end_to_end() {
    let (genesis_config, mint, validators) = genesis();
    let schedule = genesis_config.epoch_schedule.clone();

    // Epoch 0: the features are pending.
    let bank0 = Arc::new(Bank::new_for_tests(&genesis_config));
    let program = create_program_with_elf(
        &bank0,
        &bpf_loader_upgradeable::id(),
        &Pubkey::default(),
        RECEIPT_TEST_PROGRAM,
    );
    assert!(freeze_with_block_id(&bank0, 1).is_none());
    // An account changed only before the first base state.
    let quiet = Pubkey::new_unique();
    let bank1 = new_child(&bank0, 1);
    transfer(&bank1, &mint, &quiet, 1_000_000);
    assert!(freeze_with_block_id(&bank1, 2).is_none());
    // Slot 63 is skipped: bank 62 is the last block before activation.
    let bank62 = new_child(&bank1, 62);
    transfer(&bank62, &mint, &Pubkey::new_unique(), 1_000_000);
    assert!(freeze_with_block_id(&bank62, 3).is_none());
    assert_eq!(bank62.commitment_root(), None);

    // Bank 65 is the first block of epoch 1 and the first with the features:
    // its history is the anchor leaf of bank 62.
    let mut history = vec![history_leaf(62, &block_id(&bank62), &Hash::default())];
    let bank65 = new_child(&bank62, 65);
    assert!(bank65.is_accounts_state_root_active());
    let recipient = Pubkey::new_unique();
    transfer(&bank65, &mint, &recipient, 1_000_000_000);
    let program_tx = program_transaction(&bank65, &mint, &program, &recipient);
    let balance = bank65.get_balance(&mint.pubkey());
    bank65.process_transaction(&program_tx.transaction).unwrap();
    let program_fee = balance - bank65.get_balance(&mint.pubkey()) - 1;
    assert_eq!(program_fee, LAMPORTS_PER_SIGNATURE);
    let footer65 = freeze_with_block_id(&bank65, 4).unwrap();
    history.push(leaf_of(&bank65));

    let bank66 = new_child(&bank65, 66);
    let footer66 = freeze_with_block_id(&bank66, 5).unwrap();
    history.push(leaf_of(&bank66));

    // Bank 96 is at the switch slot of epoch 1: slot 5 is the root of
    // generation 1, whose base is the state of bank 62.
    let bank96 = new_child(&bank66, 96);
    let fresh = Pubkey::new_unique();
    transfer(&bank96, &mint, &fresh, 2_000_000);
    let footer96 = freeze_with_block_id(&bank96, 6).unwrap();
    assert_eq!(bank96.accounts_state_generation_in_use(), Some(1));
    history.push(leaf_of(&bank96));

    // Bank 100 is the last block of epoch 1.
    let bank100 = new_child(&bank96, 100);
    let footer100 = freeze_with_block_id(&bank100, 7).unwrap();
    let history100 = history.clone();
    history.push(leaf_of(&bank100));

    // Bank 130 is the first block of epoch 2, bank 160 at its switch slot.
    let bank130 = new_child(&bank100, 130);
    transfer(&bank130, &mint, &Pubkey::new_unique(), 1_000_000);
    freeze_with_block_id(&bank130, 8).unwrap();
    history.push(leaf_of(&bank130));
    let bank160 = new_child(&bank130, 160);
    transfer(&bank160, &mint, &fresh, 3_000_000);
    let footer160 = freeze_with_block_id(&bank160, 1).unwrap();
    assert_eq!(bank160.accounts_state_generation_in_use(), Some(2));

    // -----------------------------------------------------------------------
    // The light client starts from the signer set of epoch 1, the activation
    // epoch, from a trusted source.
    let signers1 = bank65.validator_set_signers(1).unwrap();
    assert_eq!(signers1.len(), 3);
    let set1 = SignerSet::new(1, signers1).unwrap();
    assert_eq!(
        Some(set1.validator_set_root()),
        bank65.validator_set_root(1)
    );
    let keys1 = keypairs_by_rank(&validators, &set1);
    let mut client = LightClient::new(schedule.clone(), SHRED_VERSION, set1.clone());

    // Block 100: slow finality by the two largest validators (80% of stake).
    let finality100 = slow_finality(&bank100, &keys1, &[0, 1]);
    let v100 = client
        .verify_block(100, &block_id(&bank100), &finality100, &footer100)
        .unwrap();
    assert_eq!(*v100.commitment_root(), bank100.commitment_root().unwrap());
    assert_eq!(*v100.bank_hash(), bank100.hash());
    assert_eq!(v100.parent().slot, 96);
    assert_eq!(v100.parent().block_id, block_id(&bank96));

    // Negative cases at the client level.
    // A notarization certificate alone.
    let notarize_only = FinalityProof {
        certificates: vec![finality100.certificates[0].clone()],
    };
    assert_eq!(
        client.verify_block(100, &block_id(&bank100), &notarize_only, &footer100),
        Err(Error::NotFinal)
    );
    // The largest validator alone (50%).
    let weak = slow_finality_unchecked(&bank100, &keys1, &[0]);
    assert!(matches!(
        client.verify_block(100, &block_id(&bank100), &weak, &footer100),
        Err(Error::InsufficientStake { .. })
    ));
    // The footer proof of another block.
    assert_eq!(
        client.verify_block(100, &block_id(&bank100), &finality100, &footer96),
        Err(Error::FooterProofMismatch)
    );
    // Another slot for the same block id.
    assert!(matches!(
        client.verify_block(101, &block_id(&bank100), &finality100, &footer100),
        Err(Error::InvalidSignature { .. })
    ));

    // Block 96: fast finality by every validator.
    let v96 = client
        .verify_block(
            96,
            &block_id(&bank96),
            &fast_finality(&bank96, &keys1),
            &footer96,
        )
        .unwrap();

    // -----------------------------------------------------------------------
    // An ancestor through the history of block 100: bank 65 at index 1 and
    // the anchor (bank 62) at index 0.
    let history_root = compute_root(TreeId::BLOCK_HISTORY, &history100);
    assert_eq!(
        history_root,
        bank100
            .commitment_vector()
            .unwrap()
            .get(solana_runtime::bank::block_commitment::CommitmentSlot::BlockHistoryRoot)
    );
    let history_proof = |index: usize, slot: u64, id: Hash, root: Hash| HistoryProof {
        block_history_root: history_root,
        opening: opening(&bank100, CommitmentSlot::BlockHistoryRoot),
        slot,
        block_id: id,
        commitment_root: root,
        proof: make_inclusion_proof(TreeId::BLOCK_HISTORY, &history100, index).unwrap(),
    };
    let a65 = client
        .verify_ancestor(
            &v100,
            &history_proof(1, 65, block_id(&bank65), bank65.commitment_root().unwrap()),
        )
        .unwrap();
    assert_eq!(*a65.commitment_root(), bank65.commitment_root().unwrap());
    let anchor = history_proof(0, 62, block_id(&bank62), Hash::default());
    assert_eq!(
        client.verify_ancestor(&v100, &anchor),
        Err(Error::AnchorLeaf)
    );
    let anchor = anchor.verify_anchor(&v100).unwrap();
    assert_eq!((anchor.slot, anchor.block_id), (62, block_id(&bank62)));
    // Bank 66 at index 2, claimed with bank 65's commitment root.
    assert_eq!(
        client.verify_ancestor(
            &v100,
            &history_proof(2, 66, block_id(&bank66), bank65.commitment_root().unwrap())
        ),
        Err(Error::HistoryProof)
    );
    // The parent of block 100 is in its history too.
    let a96 = client
        .verify_ancestor(
            &v100,
            &history_proof(
                3,
                96,
                v100.parent().block_id,
                bank96.commitment_root().unwrap(),
            ),
        )
        .unwrap();
    assert_eq!(a96.commitment_root(), v96.commitment_root());
    // The footers of blocks 65 and 66 lead to their block ids, and the first
    // block with the features names the anchor block as its parent.
    assert_eq!(footer65.verify(&block_id(&bank65)), Ok(()));
    assert_eq!(
        (footer65.parent_slot, footer65.parent_block_id),
        (anchor.slot, anchor.block_id)
    );
    assert_eq!(footer66.verify(&block_id(&bank66)), Ok(()));
    assert_eq!(
        footer66.verify(&block_id(&bank65)),
        Err(Error::FooterProofMismatch)
    );

    // -----------------------------------------------------------------------
    // The program transaction in bank 65, reached through the history.
    let signature: [u8; 64] = program_tx.transaction.signatures[0].into();
    let receipt = TransactionReceipt::success(
        signature,
        message_hash(&program_tx.transaction),
        program_fee,
        instruction_trace_root(&program_tx.records),
        events_root(&program_tx.events),
    );
    let transaction_proof = TransactionProof {
        receipts_root: bank65.receipts_root().unwrap(),
        opening: opening(&bank65, CommitmentSlot::ReceiptsRoot),
        receipt,
        // Second transaction of the block.
        proof: bank65.transaction_receipt_inclusion_proof(1).unwrap(),
    };
    let verified = transaction_proof.verify(&a65, &signature).unwrap();
    assert!(verified.is_success());
    assert_eq!(verified.fee(), LAMPORTS_PER_SIGNATURE);
    assert_eq!(verified.index, 1);
    // A failed status is not what the block committed to.
    let mut failed = transaction_proof.clone();
    failed.receipt = TransactionReceipt::failed(signature, receipt.message_hash, program_fee);
    assert_eq!(failed.verify(&a65, &signature), Err(Error::ReceiptProof));
    // Not in block 100.
    assert_eq!(
        transaction_proof.verify(&v100, &signature),
        Err(Error::SlotOpening(CommitmentSlot::ReceiptsRoot))
    );

    // The system transfer made through the nested CPI (trace record 2).
    let instruction_proof = |index: usize| InstructionProof {
        transaction: transaction_proof.clone(),
        record: program_tx.records[index],
        proof: instruction_record_inclusion_proof(&program_tx.records, index).unwrap(),
    };
    let cpi = instruction_proof(2).verify(&a65, &signature).unwrap();
    let mut transfer_data = vec![2, 0, 0, 0];
    transfer_data.extend_from_slice(&1u64.to_le_bytes());
    assert!(cpi.matches(
        3,
        &system_program::id(),
        &[
            meta(&mint.pubkey(), true, true),
            meta(&recipient, false, true)
        ],
        &transfer_data
    ));

    // The event logged by the inner invocation (event 1, instruction 1).
    let event_proof = EventProof {
        instruction: instruction_proof(1),
        event: program_tx.events[1].clone(),
        proof: event_inclusion_proof(&program_tx.events, 1).unwrap(),
    };
    let event = event_proof.verify(&a65, &signature).unwrap();
    assert_eq!(
        event.event.payload,
        log_data_payload(&[b"inner".as_slice(), b""])
    );
    assert_eq!(event.instruction.record.stack_height, 2);
    // Bound to the wrong instruction.
    let mut wrong = event_proof.clone();
    wrong.instruction = instruction_proof(0);
    assert_eq!(
        wrong.verify(&a65, &signature),
        Err(Error::EventInstructionMismatch)
    );

    // -----------------------------------------------------------------------
    // Accounts in bank 65: the recipient changed (delta), an untouched
    // address did not.
    let delta65 = bank65.accounts_delta().unwrap();
    let recipient_hash = hash_of(&bank65.get_account(&recipient).unwrap(), &recipient);
    let delta_proof = AccountDeltaProof {
        accounts_delta_root: delta65.root(),
        opening: opening(&bank65, CommitmentSlot::AccountsDeltaRoot),
        leaf: delta65.inclusion_proof(&recipient.to_bytes()).unwrap(),
    };
    assert_eq!(
        delta_proof.verify(&a65, &recipient, &recipient_hash),
        Ok(())
    );
    let mut richer = bank65.get_account(&recipient).unwrap();
    richer.set_lamports(richer.lamports() + 1);
    assert_eq!(
        delta_proof.verify(&a65, &recipient, &hash_of(&richer, &recipient)),
        Err(Error::AccountDeltaProof)
    );
    let untouched = Pubkey::new_unique();
    let unchanged = AccountUnchangedProof {
        accounts_delta_root: delta65.root(),
        opening: opening(&bank65, CommitmentSlot::AccountsDeltaRoot),
        proof: delta65.unchanged_proof(&untouched.to_bytes()).unwrap(),
    };
    assert_eq!(unchanged.verify(&a65, &untouched), Ok(()));
    assert_eq!(
        unchanged.verify(&a65, &recipient),
        Err(Error::AccountUnchangedProof)
    );

    // Current state at block 96 (generation 1): the mint changed since the
    // base (overlay), `quiet` did not (base leaf), and a new address has no
    // account.
    for (address, expect_generation) in [(mint.pubkey(), 1), (quiet, 1), (fresh, 1)] {
        let proof = AccountStateProof(bank96.accounts_state_proof(&address).unwrap());
        let hash = hash_of(&bank96.get_account(&address).unwrap(), &address);
        assert_eq!(proof.verify(&v96, &address, &hash), Ok(expect_generation));
        assert!(proof.verify_nonexistent(&v96, &address).is_err());
    }
    let nobody = Pubkey::new_unique();
    let proof = AccountStateProof(bank96.accounts_state_proof(&nobody).unwrap());
    assert_eq!(proof.verify_nonexistent(&v96, &nobody), Ok(1));
    assert!(proof.verify_nonexistent(&v100, &nobody).is_err());

    // The epoch boundary of epoch 1, committed by block 100: the last block
    // before it is bank 62, which predates the features (the anchor).
    let boundary1 = EpochBoundaryProof {
        boundary: bank100.epoch_boundary().unwrap(),
        opening: opening(&bank100, CommitmentSlot::EpochBoundary),
    }
    .verify(&v100, &schedule)
    .unwrap();
    assert_eq!(boundary1.boundary.first_slot, 65);
    assert_eq!(boundary1.boundary.parent_slot, 62);
    assert_eq!(boundary1.boundary.parent_block_id, anchor.block_id);
    assert_eq!(boundary1.last_block_before(), None);

    // -----------------------------------------------------------------------
    // Epoch hand-off. Block 160 cannot be verified with the signer set of
    // epoch 1.
    let set2_signers = bank100.validator_set_signers(2).unwrap();
    let set2 = SignerSet::new(2, set2_signers.clone()).unwrap();
    let keys2 = keypairs_by_rank(&validators, &set2);
    let finality160 = fast_finality(&bank160, &keys2);
    assert_eq!(
        client.verify_block(160, &block_id(&bank160), &finality160, &footer160),
        Err(Error::EpochMismatch {
            slot: 160,
            slot_epoch: 2,
            signer_set_epoch: 1,
        })
    );
    let opening3 = opening(&bank100, CommitmentSlot::ValidatorSetRoot);
    // The right signers with the wrong epoch number.
    assert_eq!(
        client.advance_epoch(
            &v100,
            &opening3,
            SignerSet::new(3, set2_signers.clone()).unwrap()
        ),
        Err(Error::NextEpoch {
            expected: 2,
            actual: 3
        })
    );
    // A signer set of epoch 2 with one stake changed.
    let mut tampered = set2_signers.clone();
    tampered[0].stake += 1;
    assert_eq!(
        client.advance_epoch(&v100, &opening3, SignerSet::new(2, tampered).unwrap()),
        Err(Error::SlotOpening(CommitmentSlot::ValidatorSetRoot))
    );
    // The wrong slot opened.
    assert_eq!(
        client.advance_epoch(
            &v100,
            &opening(&bank100, CommitmentSlot::EpochBoundary),
            set2.clone()
        ),
        Err(Error::SlotOpening(CommitmentSlot::ValidatorSetRoot))
    );
    assert_eq!(client.epoch(), 1);
    client
        .advance_epoch(&v100, &opening3, set2.clone())
        .unwrap();
    assert_eq!(client.epoch(), 2);
    assert_eq!(
        set2.validator_set_root(),
        bank160.validator_set_root(2).unwrap()
    );

    // Block 160 with the signer set of epoch 2.
    let v160 = client
        .verify_block(160, &block_id(&bank160), &finality160, &footer160)
        .unwrap();
    // The old block can no longer be verified by this client.
    assert!(matches!(
        client.verify_block(100, &block_id(&bank100), &finality100, &footer100),
        Err(Error::EpochMismatch { .. })
    ));

    // The epoch boundary of epoch 2: the last block before it is bank 100,
    // with the commitment root the client verified directly.
    let boundary2 = EpochBoundaryProof {
        boundary: bank160.epoch_boundary().unwrap(),
        opening: opening(&bank160, CommitmentSlot::EpochBoundary),
    }
    .verify(&v160, &schedule)
    .unwrap();
    assert_eq!(boundary2.boundary.first_slot, 130);
    assert_eq!(boundary2.boundary.history_len, 5);
    let last = boundary2.last_block_before().unwrap();
    assert_eq!(last.slot(), 100);
    assert_eq!(last.block_id(), v100.block_id());
    assert_eq!(last.commitment_root(), v100.commitment_root());
    // Epoch 1 had four blocks.
    assert_eq!(
        boundary1.boundary.blocks_until(&boundary2.boundary),
        Some(4)
    );

    // Current state at block 160 (generation 2, base: bank 100).
    let hash = hash_of(&bank160.get_account(&fresh).unwrap(), &fresh);
    let proof = AccountStateProof(bank160.accounts_state_proof(&fresh).unwrap());
    assert_eq!(proof.verify(&v160, &fresh, &hash), Ok(2));
    // The state at block 96 is not the current state.
    let old_hash = hash_of(&bank96.get_account(&fresh).unwrap(), &fresh);
    assert!(proof.verify(&v160, &fresh, &old_hash).is_err());
    let proof = AccountStateProof(bank160.accounts_state_proof(&nobody).unwrap());
    assert_eq!(proof.verify_nonexistent(&v160, &nobody), Ok(2));
}

/// A slow finality proof whose certificates the validator would reject (not
/// enough stake): built without the bank's check.
fn slow_finality_unchecked(bank: &Bank, keypairs: &[BlsKeypair], ranks: &[usize]) -> FinalityProof {
    let block = Block {
        slot: bank.slot(),
        block_id: bank.block_id().unwrap(),
    };
    let sign = |cert_type| {
        let cert =
            test_create_base2_unverified_certificate(keypairs, SHRED_VERSION, cert_type, ranks);
        assert!(
            bank.verify_certificate(
                UnverifiedCertificate {
                    cert_type,
                    signature: cert.signature,
                    bitmap: cert.bitmap.clone(),
                    shred_version: SHRED_VERSION,
                },
                None,
            )
            .is_err()
        );
        (cert.signature, cert.bitmap)
    };
    FinalityProof::slow(
        sign(CertificateType::Notarize(block)),
        sign(CertificateType::Finalize(bank.slot())),
    )
}
