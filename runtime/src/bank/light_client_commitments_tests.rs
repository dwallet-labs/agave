//! Bank-level test of the five light-client commitment features together:
//! Block Commitment Root (slot 0), Transaction Receipt Tree (slot 1),
//! Accounts Delta Root (slot 2), Epoch Validator Set Root (slot 3) and
//! Epoch Boundary Commitment (slot 4).
//!
//! Each feature has its own tests next to its module, run with only the base
//! feature and itself active. These check that the five coexist in one bank:
//! on a chain crossing an epoch boundary with real transactions, every slot
//! holds the value of its own module, the vector's root is the bank's
//! `commitment_root`, and every slot opens from that root.

use {
    super::{
        Bank,
        accounts_delta::account_hash,
        block_commitment::{
            COMMITMENT_VECTOR_LEN, CommitmentInputs, CommitmentSlot, CommitmentVector,
            INACTIVE_COMMITMENT_SLOT,
        },
        test_utils::goto_end_of_slot,
    },
    crate::genesis_utils::{
        GenesisConfigInfo, ValidatorVoteKeypairs, activate_feature,
        create_genesis_config_with_vote_accounts, minimum_vote_account_balance_for_vat,
    },
    agave_feature_set as feature_set,
    agave_votor_messages::consensus_message::BlockId,
    solana_clock::Slot,
    solana_epoch_schedule::EpochSchedule,
    solana_fee_calculator::FeeRateGovernor,
    solana_genesis_config::GenesisConfig,
    solana_hash::Hash,
    solana_keypair::Keypair,
    solana_merkle_tree::{
        commitment_tree::TreeId,
        epoch_boundary::EpochBoundary,
        receipt_tree::{
            InstructionAccountMeta, InstructionRecord, TransactionReceipt, empty_events_root,
            instruction_trace_root, receipts_root, verify_receipt,
        },
        validator_set::validator_set_root,
    },
    solana_pubkey::Pubkey,
    solana_runtime_transaction::runtime_transaction::RuntimeTransaction,
    solana_sdk_ids::system_program,
    solana_signer::Signer,
    solana_system_transaction as system_transaction,
    solana_transaction::Transaction,
    std::sync::Arc,
};

const SLOTS_PER_EPOCH: u64 = 32;
const LAMPORTS_PER_SIGNATURE: u64 = 5000;

fn all_features() -> [Pubkey; 5] {
    [
        feature_set::block_commitment_root::id(),
        feature_set::transaction_receipt_tree::id(),
        feature_set::accounts_delta_root::id(),
        feature_set::epoch_validator_set_root::id(),
        feature_set::epoch_boundary_commitment::id(),
    ]
}

/// A cluster of three validators with BLS keys (so the validator set is not
/// empty), 32-slot epochs with a one-epoch leader schedule offset, non-zero
/// fees, and the given features active from genesis.
fn genesis(features: &[Pubkey]) -> (GenesisConfig, Keypair) {
    let keypairs: Vec<_> = (0..3).map(|_| ValidatorVoteKeypairs::new_rand()).collect();
    let stakes = vec![
        50 * minimum_vote_account_balance_for_vat(100),
        30 * minimum_vote_account_balance_for_vat(100),
        20 * minimum_vote_account_balance_for_vat(100),
    ];
    let GenesisConfigInfo {
        mut genesis_config,
        mint_keypair,
        ..
    } = create_genesis_config_with_vote_accounts(1_000_000_000_000, &keypairs, stakes);
    genesis_config.epoch_schedule = EpochSchedule::custom(SLOTS_PER_EPOCH, SLOTS_PER_EPOCH, false);
    genesis_config.fee_rate_governor = FeeRateGovernor::new(LAMPORTS_PER_SIGNATURE, 0);
    for feature_id in features {
        activate_feature(&mut genesis_config, *feature_id);
    }
    (genesis_config, mint_keypair)
}

fn new_child(parent: &Arc<Bank>, slot: Slot) -> Arc<Bank> {
    // The genesis leader has a vote account to receive the fees.
    Arc::new(Bank::new_from_parent(
        parent.clone(),
        *parent.leader(),
        slot,
    ))
}

fn freeze_with_block_id(bank: &Arc<Bank>) {
    goto_end_of_slot(bank.clone());
    bank.freeze();
    bank.set_block_id(Some(BlockId::from(Hash::new_unique())));
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

/// Transactions of one block and their expected receipts, in block order,
/// built from the SIMD's receipt encoding rather than by the bank.
struct BlockTransactions {
    receipts: Vec<TransactionReceipt>,
    /// Accounts the block's transactions wrote.
    written: Vec<Pubkey>,
}

/// Commits two successful system transfers and one that fails (and is
/// charged its fee) to `bank`.
fn process_transactions(bank: &Bank, mint_keypair: &Keypair) -> BlockTransactions {
    let mut receipts = vec![];
    let mut written = vec![mint_keypair.pubkey()];
    let blockhash = bank.last_blockhash();

    let poor = Keypair::new();
    // Both new accounts are rent exempt.
    for (to, lamports) in [
        (Pubkey::new_unique(), 1_000_000),
        (poor.pubkey(), 1_000_000),
    ] {
        let tx = system_transaction::transfer(mint_keypair, &to, lamports, blockhash);
        bank.process_transaction(&tx).unwrap();
        let records = [InstructionRecord::new(
            1,
            system_program::id().to_bytes(),
            [
                meta(&mint_keypair.pubkey(), true, true),
                meta(&to, false, true),
            ],
            &tx.message.instructions[0].data,
        )];
        receipts.push(TransactionReceipt::success(
            tx.signatures[0].into(),
            message_hash(&tx),
            LAMPORTS_PER_SIGNATURE,
            instruction_trace_root(&records),
            empty_events_root(),
        ));
        written.push(to);
    }

    // `poor` pays the fee but cannot cover the transfer: executed with an
    // error, so it gets a failed receipt and its fee is still charged.
    let to = Pubkey::new_unique();
    let tx = system_transaction::transfer(&poor, &to, 10_000_000, blockhash);
    assert!(bank.process_transaction(&tx).is_err());
    assert_eq!(
        bank.get_balance(&poor.pubkey()),
        1_000_000 - LAMPORTS_PER_SIGNATURE
    );
    receipts.push(TransactionReceipt::failed(
        tx.signatures[0].into(),
        message_hash(&tx),
        LAMPORTS_PER_SIGNATURE,
    ));
    assert_eq!(bank.get_account(&to), None);

    BlockTransactions { receipts, written }
}

/// Checks the commitment vector of a frozen bank with all five features
/// active: every slot is the value its own module computes, the vector's root
/// is `commitment_root`, each slot opens from it, and returns the vector.
fn check_commitment(bank: &Bank, transactions: Option<&BlockTransactions>) -> CommitmentVector {
    assert!(bank.is_frozen());
    let commitment = bank.block_commitment().expect("bank has a commitment");
    let commitment_root = bank.commitment_root().unwrap();
    assert_eq!(commitment_root, commitment.commitment_root);

    // Slot 0: block history root, from the base feature's state.
    let slot0 = commitment.block_history_root();

    // Slot 1: receipts root, one receipt per committed transaction, in block
    // order, equal to the root of the independently encoded receipts.
    let slot1 = bank.receipts_root().unwrap();
    let leaf_hashes = bank.transaction_receipt_leaf_hashes();
    assert_eq!(bank.executed_transaction_count(), leaf_hashes.len() as u64);
    match transactions {
        Some(transactions) => {
            let expected: Vec<_> = transactions
                .receipts
                .iter()
                .map(TransactionReceipt::leaf_hash)
                .collect();
            assert_eq!(leaf_hashes, expected);
            assert_eq!(slot1, receipts_root(&transactions.receipts));
            for (index, receipt) in transactions.receipts.iter().enumerate() {
                let proof = bank.transaction_receipt_inclusion_proof(index).unwrap();
                assert!(verify_receipt(&proof, receipt, &slot1));
            }
        }
        None => {
            assert!(leaf_hashes.is_empty());
            assert_eq!(slot1, receipts_root(&[]));
        }
    }

    // Slot 2: accounts delta root over the accounts changed in the block,
    // each with the hash of its state after the block.
    let delta = bank.accounts_delta().expect("bank has an accounts delta");
    let slot2 = delta.root();
    assert_eq!(bank.accounts_delta_root(), Some(slot2));
    assert!(!delta.is_empty());
    for address in transactions.iter().flat_map(|t| t.written.iter()) {
        let account = bank.get_account(address).unwrap_or_default();
        let hash = account_hash(&account, address);
        let address = address.to_bytes();
        assert_eq!(delta.get(&address).unwrap().account_hash, hash);
        let proof = delta.inclusion_proof(&address).unwrap();
        assert!(proof.verify_account(&address, &hash, &slot2));
    }
    // An account the block did not touch is provably unchanged.
    let untouched = Pubkey::new_unique().to_bytes();
    assert!(delta.inclusion_proof(&untouched).is_none());
    assert!(
        delta
            .unchanged_proof(&untouched)
            .unwrap()
            .verify(&untouched, &slot2)
    );

    // Slot 3: validator set root of the next epoch, from the signer set of
    // this bank's epoch stakes for that epoch.
    let next_epoch = bank.epoch() + 1;
    let signers = bank.validator_set_signers(next_epoch).unwrap();
    assert_eq!(signers.len(), 3);
    let slot3 = bank.next_epoch_validator_set_root().unwrap();
    assert_eq!(slot3, validator_set_root(next_epoch, &signers).unwrap());
    assert_eq!(bank.validator_set_root(next_epoch), Some(slot3));

    // Slot 4: epoch boundary, zero in the genesis epoch.
    let slot4 = bank
        .epoch_boundary()
        .map(|epoch_boundary| epoch_boundary.hash())
        .unwrap_or(INACTIVE_COMMITMENT_SLOT);

    let mut vector = [INACTIVE_COMMITMENT_SLOT; COMMITMENT_VECTOR_LEN];
    vector[CommitmentSlot::BlockHistoryRoot.index()] = slot0;
    vector[CommitmentSlot::ReceiptsRoot.index()] = slot1;
    vector[CommitmentSlot::AccountsDeltaRoot.index()] = slot2;
    vector[CommitmentSlot::ValidatorSetRoot.index()] = slot3;
    vector[CommitmentSlot::EpochBoundary.index()] = slot4;
    let vector = CommitmentVector(vector);

    // Slots 0 to 3 are non-zero in every block; 5 to 7 are reserved.
    for slot in [
        CommitmentSlot::BlockHistoryRoot,
        CommitmentSlot::ReceiptsRoot,
        CommitmentSlot::AccountsDeltaRoot,
        CommitmentSlot::ValidatorSetRoot,
    ] {
        assert_ne!(vector.get(slot), INACTIVE_COMMITMENT_SLOT, "{slot:?}");
    }
    for index in 5..COMMITMENT_VECTOR_LEN {
        assert_eq!(vector.0[index], INACTIVE_COMMITMENT_SLOT);
    }

    // The bank's own slot methods give the same vector, and its root is the
    // bank's commitment root.
    let parent = bank.parent();
    assert_eq!(
        bank.compute_commitment_vector(&CommitmentInputs {
            block_history: &commitment.block_history,
            parent: parent.as_deref(),
        }),
        vector
    );
    assert_eq!(vector.root(), commitment_root);

    // Every slot opens from the commitment root.
    for index in 0..COMMITMENT_VECTOR_LEN {
        let proof = vector.inclusion_proof(index).unwrap();
        assert!(proof.verify(
            TreeId::COMMITMENT_VECTOR,
            vector.0[index].as_ref(),
            &commitment_root
        ));
        let mut wrong = vector.0[index].to_bytes();
        wrong[0] ^= 1;
        assert!(!proof.verify(TreeId::COMMITMENT_VECTOR, &wrong, &commitment_root));
    }

    // The value a V2 footer would carry verifies.
    bank.set_expected_commitment_root(commitment_root);
    assert_eq!(bank.verify_commitment_root(), Ok(()));
    vector
}

#[test]
fn test_all_light_client_commitments_across_epoch_boundary() {
    let (genesis_config, mint_keypair) = genesis(&all_features());
    let bank0 = Arc::new(Bank::new_for_tests(&genesis_config));
    assert!(bank0.is_block_commitment_root_active());
    assert!(bank0.is_transaction_receipt_tree_active());
    assert!(bank0.is_accounts_delta_root_active());
    assert!(bank0.is_epoch_validator_set_root_active());
    assert!(bank0.is_epoch_boundary_commitment_active());
    freeze_with_block_id(&bank0);
    let vector0 = check_commitment(&bank0, None);
    assert_eq!(
        bank0.block_commitment().unwrap().block_history.leaf_count(),
        0
    );
    // Genesis epoch: no epoch boundary.
    assert_eq!(bank0.epoch_boundary(), None);
    assert_eq!(
        vector0.get(CommitmentSlot::EpochBoundary),
        INACTIVE_COMMITMENT_SLOT
    );

    // Epoch 0, with transactions; slots 2 to 29 skipped.
    let bank1 = new_child(&bank0, 1);
    let transactions1 = process_transactions(&bank1, &mint_keypair);
    freeze_with_block_id(&bank1);
    let vector1 = check_commitment(&bank1, Some(&transactions1));
    assert_eq!(bank1.epoch_boundary(), None);

    let bank30 = new_child(&bank1, 30);
    let transactions30 = process_transactions(&bank30, &mint_keypair);
    freeze_with_block_id(&bank30);
    let vector30 = check_commitment(&bank30, Some(&transactions30));
    assert_eq!(bank30.epoch(), 0);
    assert_eq!(bank30.epoch_boundary(), None);
    // Same epoch: same validator set root.
    for vector in [&vector1, &vector30] {
        assert_eq!(
            vector.get(CommitmentSlot::ValidatorSetRoot),
            vector0.get(CommitmentSlot::ValidatorSetRoot)
        );
    }

    // Slots 31 and 32 skipped: bank 33 is `F` of epoch 1 and bank 30 is `P`.
    let bank33 = new_child(&bank30, 33);
    assert_eq!(bank33.epoch(), 1);
    let transactions33 = process_transactions(&bank33, &mint_keypair);
    freeze_with_block_id(&bank33);
    let vector33 = check_commitment(&bank33, Some(&transactions33));
    let history_len33 = bank33
        .block_commitment()
        .unwrap()
        .block_history
        .leaf_count();
    // Banks 0, 1 and 30.
    assert_eq!(history_len33, 3);
    let boundary = EpochBoundary {
        epoch: 1,
        first_slot: 33,
        parent_slot: 30,
        parent_block_id: bank30.block_id().unwrap().to_hash(),
        parent_commitment_root: vector30.root(),
        history_len: history_len33,
    };
    assert_eq!(bank33.epoch_boundary(), Some(boundary));
    assert_eq!(vector33.get(CommitmentSlot::EpochBoundary), boundary.hash());
    assert_ne!(boundary.hash(), INACTIVE_COMMITMENT_SLOT);
    // P's commitment root is the last leaf of F's block history.
    assert_eq!(boundary.parent_history_index(), Some(2));
    // New epoch: the validator set root moves to epoch 2.
    assert_ne!(
        vector33.get(CommitmentSlot::ValidatorSetRoot),
        vector30.get(CommitmentSlot::ValidatorSetRoot)
    );

    // A later block of epoch 1, without transactions, keeps the boundary
    // and the validator set root, and its receipts root is the empty one.
    let bank34 = new_child(&bank33, 34);
    freeze_with_block_id(&bank34);
    let vector34 = check_commitment(&bank34, None);
    assert_eq!(bank34.epoch_boundary(), Some(boundary));
    assert_eq!(
        vector34.get(CommitmentSlot::EpochBoundary),
        vector33.get(CommitmentSlot::EpochBoundary)
    );
    assert_eq!(
        vector34.get(CommitmentSlot::ValidatorSetRoot),
        vector33.get(CommitmentSlot::ValidatorSetRoot)
    );

    // Every block of the chain has a different commitment root.
    let roots = [&vector0, &vector1, &vector30, &vector33, &vector34].map(|v| v.root());
    for (i, a) in roots.iter().enumerate() {
        for b in &roots[i + 1..] {
            assert_ne!(a, b);
        }
    }
}

#[test]
fn test_all_light_client_commitments_inactive() {
    let (genesis_config, mint_keypair) = genesis(&[]);
    let bank0 = Arc::new(Bank::new_for_tests(&genesis_config));
    let mut chain = vec![bank0.clone()];
    freeze_with_block_id(&bank0);
    let mut parent = bank0;
    for slot in [1, 30, 33, 34] {
        let bank = new_child(&parent, slot);
        process_transactions(&bank, &mint_keypair);
        freeze_with_block_id(&bank);
        chain.push(bank.clone());
        parent = bank;
    }
    assert_eq!(parent.epoch(), 1);

    for bank in &chain {
        assert!(!bank.is_block_commitment_root_active());
        assert!(!bank.is_transaction_receipt_tree_active());
        assert!(!bank.is_accounts_delta_root_active());
        assert!(!bank.is_epoch_validator_set_root_active());
        assert!(!bank.is_epoch_boundary_commitment_active());
        // No commitment state of any of the five features: replay accepts
        // only V1 footers and has no commitment root to verify.
        assert_eq!(bank.block_commitment(), None);
        assert_eq!(bank.commitment_root(), None);
        assert_eq!(bank.receipts_root(), None);
        assert!(bank.transaction_receipt_leaf_hashes().is_empty());
        assert_eq!(bank.accounts_delta(), None);
        assert_eq!(bank.accounts_delta_root(), None);
        assert_eq!(bank.epoch_boundary(), None);
        assert_eq!(bank.expected_commitment_root(), None);
        assert_eq!(bank.verify_commitment_root(), Ok(()));
        // Only the genesis bank and the first bank of each epoch keep the
        // account hashes (a feature could activate there).
        assert_eq!(
            bank.collects_accounts_delta(),
            bank.parent()
                .is_none_or(|parent| parent.epoch() < bank.epoch())
        );
        // The snapshot fields of the features are empty, as in a snapshot
        // written before them.
        let fields = bank.get_fields_to_serialize();
        assert_eq!(fields.block_commitment, None);
        assert_eq!(fields.epoch_boundary, None);
    }
}
