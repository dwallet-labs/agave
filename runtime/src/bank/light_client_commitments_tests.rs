//! Bank-level test of the six light-client commitment features together:
//! Block Commitment Root (slot 0), Transaction Receipt Tree (slot 1),
//! Accounts Delta Root (slot 2), Epoch Validator Set Root (slot 3), Epoch
//! Boundary Commitment (slot 4) and Accounts State Root (slot 5).
//!
//! Each feature has its own tests next to its module, run with only the base
//! feature, its dependencies and itself active. These check that the six
//! coexist in one bank: on a chain crossing epoch boundaries with real
//! transactions, every slot holds the value of its own module (slots 0, 4 and
//! 5 recomputed independently from the chain), slots 6 and 7 are zero, the
//! vector's root is the bank's `commitment_root`, and every slot opens from
//! that root. One chain has the features active from genesis, one has them
//! activate together at an epoch boundary (the anchor leaf, an epoch boundary
//! whose `P` predates the features, and the first generations of the
//! accounts state root).

use {
    super::{
        Bank,
        accounts_delta::account_hash,
        accounts_state::{MIN_SLOTS_PER_EPOCH_FOR_GENERATION, epoch_has_generation},
        block_commitment::{
            COMMITMENT_VECTOR_LEN, CommitmentInputs, CommitmentSlot, CommitmentVector,
            HISTORY_LEAF_LEN, INACTIVE_COMMITMENT_SLOT, anchor_leaf, history_leaf,
        },
        test_utils::goto_end_of_slot,
    },
    crate::genesis_utils::{
        GenesisConfigInfo, ValidatorVoteKeypairs, activate_feature,
        create_genesis_config_with_vote_accounts, minimum_vote_account_balance_for_vat,
    },
    agave_feature_set as feature_set,
    agave_votor_messages::consensus_message::BlockId,
    solana_account::{AccountSharedData, ReadableAccount, WritableAccount},
    solana_clock::{Epoch, Slot},
    solana_epoch_schedule::EpochSchedule,
    solana_feature_gate_interface::{self as feature, Feature},
    solana_fee_calculator::FeeRateGovernor,
    solana_genesis_config::GenesisConfig,
    solana_hash::Hash,
    solana_keypair::Keypair,
    solana_merkle_tree::{
        accounts_state::AccountLookup,
        commitment_tree::{TreeId, compute_root, make_inclusion_proof},
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
    solana_sha256_hasher::hashv,
    solana_signer::Signer,
    solana_system_transaction as system_transaction,
    solana_transaction::Transaction,
    std::{
        collections::{BTreeMap, BTreeSet},
        sync::Arc,
    },
};

/// Epoch length of the chain with the features active from genesis: too short
/// for an accounts state generation, so slot 5 stays zero there.
const SLOTS_PER_EPOCH: u64 = 32;
/// Epoch length of the chain with the features activated at an epoch
/// boundary: long enough for accounts state generations.
const SLOTS_PER_LONG_EPOCH: u64 = 64;
const LAMPORTS_PER_SIGNATURE: u64 = 5000;

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

fn assert_all_features_active(bank: &Bank, active: bool) {
    assert_eq!(bank.is_block_commitment_root_active(), active);
    assert_eq!(bank.is_transaction_receipt_tree_active(), active);
    assert_eq!(bank.is_accounts_delta_root_active(), active);
    assert_eq!(bank.is_epoch_validator_set_root_active(), active);
    assert_eq!(bank.is_epoch_boundary_commitment_active(), active);
    assert_eq!(bank.is_accounts_state_root_active(), active);
}

/// A cluster of three validators with BLS keys (so the validator set is not
/// empty), 32-slot epochs with a one-epoch leader schedule offset, non-zero
/// fees, and the given features active from genesis.
fn genesis(features: &[Pubkey]) -> (GenesisConfig, Keypair) {
    genesis_with(SLOTS_PER_EPOCH, features, &[])
}

/// As [`genesis`], with `slots_per_epoch`, and with the `pending` features
/// activating at the first epoch boundary.
fn genesis_with(
    slots_per_epoch: u64,
    features: &[Pubkey],
    pending: &[Pubkey],
) -> (GenesisConfig, Keypair) {
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
    genesis_config.epoch_schedule = EpochSchedule::custom(slots_per_epoch, slots_per_epoch, false);
    genesis_config.fee_rate_governor = FeeRateGovernor::new(LAMPORTS_PER_SIGNATURE, 0);
    for feature_id in features {
        activate_feature(&mut genesis_config, *feature_id);
    }
    for feature_id in pending {
        let lamports = genesis_config
            .rent
            .minimum_balance(Feature::size_of())
            .max(1);
        genesis_config.accounts.insert(
            *feature_id,
            feature::create_account(&Feature { activated_at: None }, lamports).into(),
        );
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

/// Checks the commitment vector of a frozen bank with all six features
/// active: every slot is the value its own module computes, slot 0 is the
/// root over `history` (the history leaves of the bank's ancestors, built by
/// the test), slot 5 is `slot5` (computed by the test), slots 6 and 7 are
/// zero, the vector's root is `commitment_root`, each slot opens from it, and
/// returns the vector.
fn check_commitment(
    bank: &Bank,
    transactions: Option<&BlockTransactions>,
    history: &[[u8; HISTORY_LEAF_LEN]],
    slot5: Hash,
) -> CommitmentVector {
    assert!(bank.is_frozen());
    assert_eq!(bank.commitment_failure(), None, "slot {}", bank.slot());
    let commitment = bank.block_commitment().expect("bank has a commitment");
    let commitment_root = bank.commitment_root().unwrap();
    assert_eq!(commitment_root, commitment.commitment_root);

    // Slot 0: block history root over the history leaves of the chain.
    let slot0 = commitment.block_history_root();
    assert_eq!(commitment.block_history.leaf_count(), history.len() as u64);
    assert_eq!(slot0, compute_root(TreeId::BLOCK_HISTORY, history));

    // Slot 1: receipts root, one receipt per committed transaction, in block
    // order, equal to the root of the independently encoded receipts.
    let slot1 = bank.receipts_root().unwrap();
    let leaf_hashes = bank.transaction_receipt_leaf_hashes().unwrap();
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
    vector[CommitmentSlot::AccountsStateRoot.index()] = slot5;
    let vector = CommitmentVector(vector);

    // Slots 0 to 3 are non-zero in every block (the root of an empty history
    // is not zero); 6 and 7 are reserved.
    for slot in [
        CommitmentSlot::BlockHistoryRoot,
        CommitmentSlot::ReceiptsRoot,
        CommitmentSlot::AccountsDeltaRoot,
        CommitmentSlot::ValidatorSetRoot,
    ] {
        assert_ne!(vector.get(slot), INACTIVE_COMMITMENT_SLOT, "{slot:?}");
    }
    for index in (CommitmentSlot::AccountsStateRoot.index() + 1)..COMMITMENT_VECTOR_LEN {
        assert_eq!(vector.0[index], INACTIVE_COMMITMENT_SLOT, "slot {index}");
    }

    // The bank's own slot methods give the same vector, and its root is the
    // bank's commitment root.
    let parent = bank.parent();
    assert_eq!(
        bank.compute_commitment_vector(&CommitmentInputs {
            block_history: &commitment.block_history,
            parent: parent.as_deref(),
        }),
        Ok(vector)
    );
    assert_eq!(bank.commitment_vector(), Some(vector));
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

/// The history leaf of a frozen bank with a commitment root.
fn leaf_of(bank: &Bank) -> [u8; HISTORY_LEAF_LEN] {
    history_leaf(
        bank.slot(),
        &bank.block_id().unwrap().to_hash(),
        &bank.commitment_root().unwrap(),
    )
}

#[test]
fn test_all_light_client_commitments_across_epoch_boundary() {
    let (genesis_config, mint_keypair) = genesis(&all_features());
    let bank0 = Arc::new(Bank::new_for_tests(&genesis_config));
    assert_all_features_active(&bank0, true);
    // 32-slot epochs are too short for an accounts state generation: slot 5
    // is zero in every block of this chain, while the bank keeps an accounts
    // state without generations.
    const { assert!(SLOTS_PER_EPOCH < MIN_SLOTS_PER_EPOCH_FOR_GENERATION) };
    assert!(!epoch_has_generation(bank0.epoch_schedule(), 0));
    assert!(!epoch_has_generation(bank0.epoch_schedule(), 1));
    freeze_with_block_id(&bank0);
    // The genesis bank has no parent: its history is empty, without an
    // anchor leaf.
    let mut history = vec![];
    let vector0 = check_commitment(&bank0, None, &history, INACTIVE_COMMITMENT_SLOT);
    assert_eq!(
        bank0.block_commitment().unwrap().block_history.leaf_count(),
        0
    );
    assert!(bank0.accounts_state().unwrap().generations().is_empty());
    // Genesis epoch: no epoch boundary.
    assert_eq!(bank0.epoch_boundary(), None);
    assert_eq!(
        vector0.get(CommitmentSlot::EpochBoundary),
        INACTIVE_COMMITMENT_SLOT
    );
    history.push(leaf_of(&bank0));

    // Epoch 0, with transactions; slots 2 to 29 skipped.
    let bank1 = new_child(&bank0, 1);
    let transactions1 = process_transactions(&bank1, &mint_keypair);
    freeze_with_block_id(&bank1);
    let vector1 = check_commitment(
        &bank1,
        Some(&transactions1),
        &history,
        INACTIVE_COMMITMENT_SLOT,
    );
    assert_eq!(bank1.epoch_boundary(), None);
    history.push(leaf_of(&bank1));

    let bank30 = new_child(&bank1, 30);
    let transactions30 = process_transactions(&bank30, &mint_keypair);
    freeze_with_block_id(&bank30);
    let vector30 = check_commitment(
        &bank30,
        Some(&transactions30),
        &history,
        INACTIVE_COMMITMENT_SLOT,
    );
    assert_eq!(bank30.epoch(), 0);
    assert_eq!(bank30.epoch_boundary(), None);
    history.push(leaf_of(&bank30));
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
    let vector33 = check_commitment(
        &bank33,
        Some(&transactions33),
        &history,
        INACTIVE_COMMITMENT_SLOT,
    );
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
    assert_eq!(history[2], leaf_of(&bank30));
    // New epoch: the validator set root moves to epoch 2.
    assert_ne!(
        vector33.get(CommitmentSlot::ValidatorSetRoot),
        vector30.get(CommitmentSlot::ValidatorSetRoot)
    );
    history.push(leaf_of(&bank33));

    // A later block of epoch 1, without transactions, keeps the boundary
    // and the validator set root, and its receipts root is the empty one.
    let bank34 = new_child(&bank33, 34);
    freeze_with_block_id(&bank34);
    let vector34 = check_commitment(&bank34, None, &history, INACTIVE_COMMITMENT_SLOT);
    assert_eq!(bank34.epoch_boundary(), Some(boundary));
    assert_eq!(
        vector34.get(CommitmentSlot::EpochBoundary),
        vector33.get(CommitmentSlot::EpochBoundary)
    );
    assert_eq!(
        vector34.get(CommitmentSlot::ValidatorSetRoot),
        vector33.get(CommitmentSlot::ValidatorSetRoot)
    );
    history.push(leaf_of(&bank34));

    // Past the switch slot of epoch 1 (48) slot 5 would use generation 1,
    // which does not exist for a 32-slot epoch.
    let bank50 = new_child(&bank34, 50);
    freeze_with_block_id(&bank50);
    let vector50 = check_commitment(&bank50, None, &history, INACTIVE_COMMITMENT_SLOT);
    assert_eq!(bank50.accounts_state_generation_in_use(), Some(1));
    assert!(bank50.accounts_state().unwrap().generations().is_empty());
    assert_eq!(bank50.epoch_boundary(), Some(boundary));

    // Every block of the chain has a different commitment root.
    let roots = [
        &vector0, &vector1, &vector30, &vector33, &vector34, &vector50,
    ]
    .map(|v| v.root());
    for (i, a) in roots.iter().enumerate() {
        for b in &roots[i + 1..] {
            assert_ne!(a, b);
        }
    }
}

/// Independent `account_hash` (BLAKE3 of the SIMD-0215 input).
fn blake3_account_hash(address: &Pubkey, account: &AccountSharedData) -> Hash {
    if account.lamports() == 0 {
        return Hash::default();
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(&account.lamports().to_le_bytes());
    hasher.update(account.data());
    hasher.update(&[u8::from(account.executable())]);
    hasher.update(account.owner().as_ref());
    hasher.update(address.as_ref());
    Hash::new_from_array(*hasher.finalize().as_bytes())
}

/// The account hash of every account with lamports visible from `bank`,
/// read back from accounts-db.
fn scanned_state(bank: &Bank) -> BTreeMap<Pubkey, Hash> {
    let mut state = BTreeMap::new();
    bank.scan_all_accounts(|item| {
        if let Some((address, account, _slot)) = item
            && account.lamports() != 0
        {
            state.insert(*address, blake3_account_hash(address, &account));
        }
    })
    .unwrap();
    state
}

/// `KTH(0x08, S)` of the overlay, from the SIMD's recursive definition.
fn reference_overlay_kth(entries: &[([u8; 32], [u8; 64])]) -> Hash {
    let bit_of = |key: &[u8; 32], bit: usize| (key[bit / 8] >> (7 - bit % 8)) & 1;
    match entries {
        [] => Hash::default(),
        [(key, value)] => hashv(&[&[0x02, 0x08], key, value]),
        _ => {
            let bit = (0..256)
                .find(|bit| {
                    entries
                        .iter()
                        .any(|(key, _)| bit_of(key, *bit) != bit_of(&entries[0].0, *bit))
                })
                .unwrap();
            let (zeros, ones): (Vec<_>, Vec<_>) = entries
                .iter()
                .copied()
                .partition(|(key, _)| bit_of(key, bit) == 0);
            hashv(&[
                &[0x03],
                &(bit as u16).to_le_bytes(),
                reference_overlay_kth(&zeros).as_ref(),
                reference_overlay_kth(&ones).as_ref(),
            ])
        }
    }
}

/// `accounts_state_root` of generation `generation` at the last bank of
/// `states`, from scans alone: `states[0]` is the state of `P` (the base
/// state), the others the states of `F` to the bank, in chain order.
fn reference_accounts_state_root(generation: Epoch, states: &[&BTreeMap<Pubkey, Hash>]) -> Hash {
    let base = states[0];
    let now = states[states.len() - 1];
    // Every address changed by a block from `F` to the bank.
    let mut changed = BTreeSet::new();
    for pair in states.windows(2) {
        for address in pair[0].keys().chain(pair[1].keys()) {
            if pair[0].get(address) != pair[1].get(address) {
                changed.insert(*address);
            }
        }
    }
    let overlay: Vec<([u8; 32], [u8; 64])> = changed
        .iter()
        .map(|address| {
            let mut value = [0u8; 64];
            value[..32].copy_from_slice(now.get(address).copied().unwrap_or_default().as_ref());
            value[32..].copy_from_slice(base.get(address).copied().unwrap_or_default().as_ref());
            (address.to_bytes(), value)
        })
        .collect();
    let overlay_root = hashv(&[
        &[0x81, 0x08],
        &(overlay.len() as u64).to_le_bytes(),
        reference_overlay_kth(&overlay).as_ref(),
    ]);
    let base_leaves: Vec<Vec<u8>> = base
        .iter()
        .map(|(address, hash)| [address.as_ref(), hash.as_ref()].concat())
        .collect();
    let base_root = compute_root(TreeId::BASE_STATE, &base_leaves);
    hashv(&[
        &[0x84],
        &generation.to_le_bytes(),
        base_root.as_ref(),
        overlay_root.as_ref(),
    ])
}

/// Checks one current-state proof and one non-existence proof of slot 5 at
/// a frozen bank, from its commitment root. `current` must have an account,
/// `absent` must not; `expect_overlay` says whether `current` changed since
/// the base state (overlay membership) or not (overlay non-membership and a
/// base leaf).
fn check_accounts_state_proofs(
    bank: &Bank,
    state: &BTreeMap<Pubkey, Hash>,
    current: &Pubkey,
    expect_overlay: bool,
    absent: &Pubkey,
) {
    let commitment_root = bank.commitment_root().unwrap();
    let generation = bank.accounts_state_generation_in_use().unwrap();

    let hash = state[current];
    let proof = bank.accounts_state_proof(current).unwrap();
    assert_eq!(
        matches!(proof.lookup, AccountLookup::Overlay { .. }),
        expect_overlay
    );
    assert_eq!(
        proof.verify_current_state(&commitment_root, &current.to_bytes(), &hash),
        Ok(generation)
    );
    // Not another state, not non-existence, not another commitment root.
    let mut other = bank.get_account(current).unwrap();
    other.set_lamports(other.lamports() + 1);
    assert!(
        proof
            .verify_current_state(
                &commitment_root,
                &current.to_bytes(),
                &blake3_account_hash(current, &other)
            )
            .is_err()
    );
    assert!(
        proof
            .verify_nonexistence(&commitment_root, &current.to_bytes())
            .is_err()
    );
    assert!(
        proof
            .verify_current_state(&Hash::new_unique(), &current.to_bytes(), &hash)
            .is_err()
    );

    assert!(!state.contains_key(absent));
    let proof = bank.accounts_state_proof(absent).unwrap();
    assert_eq!(
        proof.verify_nonexistence(&commitment_root, &absent.to_bytes()),
        Ok(generation)
    );
    assert!(
        proof
            .verify_current_state(&commitment_root, &absent.to_bytes(), &hash)
            .is_err()
    );
}

/// All six features activate together at the first epoch boundary, on
/// 64-slot epochs. The first block of epoch 1 (`F`) has the anchor leaf of
/// the last block of epoch 0 (`P`) as its whole block history, an epoch
/// boundary with `P`'s zero commitment root and a history length of 1, and
/// zero in slot 5 until the switch slot of epoch 1. From the switch slot on,
/// slot 5 is the root of generation 1 (base: the state of `P`), and from the
/// switch slot of epoch 2 the root of generation 2 (base: the last block of
/// epoch 1), each equal to a value recomputed from account scans and opening
/// to current-state and non-existence proofs.
#[test]
fn test_all_light_client_commitments_activated_at_epoch_boundary() {
    let (genesis_config, mint_keypair) = genesis_with(SLOTS_PER_LONG_EPOCH, &[], &all_features());
    let bank0 = Arc::new(Bank::new_for_tests(&genesis_config));
    let epoch_schedule = bank0.epoch_schedule().clone();
    assert!(epoch_has_generation(&epoch_schedule, 1));
    let switch_slot = |epoch: Epoch| {
        epoch_schedule.get_first_slot_in_epoch(epoch) + epoch_schedule.get_slots_in_epoch(epoch) / 2
    };
    assert_eq!(switch_slot(1), 96);
    assert_eq!(switch_slot(2), 160);
    let mut states = BTreeMap::new();

    // Epoch 0: the features are pending.
    assert_all_features_active(&bank0, false);
    freeze_with_block_id(&bank0);
    assert_eq!(bank0.block_commitment(), None);
    let bank1 = new_child(&bank0, 1);
    let transactions1 = process_transactions(&bank1, &mint_keypair);
    freeze_with_block_id(&bank1);
    // Slot 63 is skipped: bank 62 is `P` of epoch 1.
    let bank62 = new_child(&bank1, 62);
    process_transactions(&bank62, &mint_keypair);
    freeze_with_block_id(&bank62);
    for bank in [&bank1, &bank62] {
        assert_all_features_active(bank, false);
        assert_eq!(bank.block_commitment(), None);
        assert_eq!(bank.commitment_failure(), None);
    }
    states.insert(62, scanned_state(&bank62));

    // Slot 64 is skipped: bank 65 is `F`, the first block with the features
    // active. Its history is the anchor leaf of bank 62 alone: bank 62's
    // history leaf with a zero commitment root.
    let bank65 = new_child(&bank62, 65);
    assert_eq!(bank65.epoch(), 1);
    assert_all_features_active(&bank65, true);
    let transactions65 = process_transactions(&bank65, &mint_keypair);
    freeze_with_block_id(&bank65);
    states.insert(65, scanned_state(&bank65));
    let anchor = anchor_leaf(62, &bank62.block_id().unwrap().to_hash());
    assert_eq!(&anchor[..8], &62u64.to_le_bytes());
    assert_eq!(
        &anchor[8..40],
        bank62.block_id().unwrap().to_hash().as_ref()
    );
    assert_eq!(&anchor[40..], &[0; 32]);
    let mut history = vec![anchor];
    // Slot 5 is zero before the switch slot of the activation epoch:
    // generation 0 does not exist.
    assert_eq!(bank65.accounts_state_generation_in_use(), Some(0));
    let vector65 = check_commitment(
        &bank65,
        Some(&transactions65),
        &history,
        INACTIVE_COMMITMENT_SLOT,
    );
    // The epoch boundary of epoch 1: `P` predates the features, so its
    // commitment root is zero and the history of `F` holds `P`'s anchor
    // leaf at index 0.
    let boundary1 = EpochBoundary {
        epoch: 1,
        first_slot: 65,
        parent_slot: 62,
        parent_block_id: bank62.block_id().unwrap().to_hash(),
        parent_commitment_root: Hash::default(),
        history_len: 1,
    };
    assert_eq!(bank65.epoch_boundary(), Some(boundary1));
    assert_eq!(
        vector65.get(CommitmentSlot::EpochBoundary),
        boundary1.hash()
    );
    assert_eq!(boundary1.parent_history_index(), Some(0));
    assert_eq!(
        history_leaf(
            boundary1.parent_slot,
            &boundary1.parent_block_id,
            &boundary1.parent_commitment_root
        ),
        history[0]
    );
    let history_proof = make_inclusion_proof(TreeId::BLOCK_HISTORY, &history, 0).unwrap();
    assert!(history_proof.verify(
        TreeId::BLOCK_HISTORY,
        &history[0],
        &vector65.get(CommitmentSlot::BlockHistoryRoot)
    ));
    history.push(leaf_of(&bank65));

    // A block of epoch 1 without transactions, still before the switch slot.
    let bank66 = new_child(&bank65, 66);
    freeze_with_block_id(&bank66);
    states.insert(66, scanned_state(&bank66));
    let vector66 = check_commitment(&bank66, None, &history, INACTIVE_COMMITMENT_SLOT);
    assert_eq!(bank66.epoch_boundary(), Some(boundary1));
    history.push(leaf_of(&bank66));

    // Bank 96 is at the switch slot of epoch 1: slot 5 is the root of
    // generation 1, whose base is the state of bank 62 and whose overlay has
    // the accounts changed by banks 65, 66 and 96.
    let bank96 = new_child(&bank66, 96);
    let transactions96 = process_transactions(&bank96, &mint_keypair);
    freeze_with_block_id(&bank96);
    states.insert(96, scanned_state(&bank96));
    assert_eq!(bank96.accounts_state_generation_in_use(), Some(1));
    let slot5_96 =
        reference_accounts_state_root(1, &[&states[&62], &states[&65], &states[&66], &states[&96]]);
    assert_ne!(slot5_96, INACTIVE_COMMITMENT_SLOT);
    let vector96 = check_commitment(&bank96, Some(&transactions96), &history, slot5_96);
    // The mint changed since the base state (overlay); the first recipient
    // of bank 1 did not (base leaf); a new address has no account.
    let untouched = transactions1.written[1];
    assert_eq!(states[&96].get(&untouched), states[&62].get(&untouched));
    check_accounts_state_proofs(
        &bank96,
        &states[&96],
        &mint_keypair.pubkey(),
        true,
        &Pubkey::new_unique(),
    );
    check_accounts_state_proofs(
        &bank96,
        &states[&96],
        &untouched,
        false,
        &Pubkey::new_unique(),
    );
    history.push(leaf_of(&bank96));

    let bank100 = new_child(&bank96, 100);
    freeze_with_block_id(&bank100);
    states.insert(100, scanned_state(&bank100));
    let slot5_100 = reference_accounts_state_root(
        1,
        &[
            &states[&62],
            &states[&65],
            &states[&66],
            &states[&96],
            &states[&100],
        ],
    );
    let vector100 = check_commitment(&bank100, None, &history, slot5_100);
    assert_ne!(slot5_100, slot5_96);
    // Same epoch: same validator set root and epoch boundary.
    for vector in [&vector66, &vector96, &vector100] {
        assert_eq!(
            vector.get(CommitmentSlot::ValidatorSetRoot),
            vector65.get(CommitmentSlot::ValidatorSetRoot)
        );
        assert_eq!(vector.get(CommitmentSlot::EpochBoundary), boundary1.hash());
    }
    history.push(leaf_of(&bank100));

    // Bank 130 is `F` of epoch 2 (`P` is bank 100). Before the switch slot of
    // epoch 2 it still uses generation 1, whose overlay now also has the
    // accounts changed by bank 130.
    let bank130 = new_child(&bank100, 130);
    assert_eq!(bank130.epoch(), 2);
    let transactions130 = process_transactions(&bank130, &mint_keypair);
    freeze_with_block_id(&bank130);
    states.insert(130, scanned_state(&bank130));
    assert_eq!(bank130.accounts_state_generation_in_use(), Some(1));
    let slot5_130 = reference_accounts_state_root(
        1,
        &[
            &states[&62],
            &states[&65],
            &states[&66],
            &states[&96],
            &states[&100],
            &states[&130],
        ],
    );
    let vector130 = check_commitment(&bank130, Some(&transactions130), &history, slot5_130);
    // The anchor leaf and banks 65, 66, 96 and 100.
    let boundary2 = EpochBoundary {
        epoch: 2,
        first_slot: 130,
        parent_slot: 100,
        parent_block_id: bank100.block_id().unwrap().to_hash(),
        parent_commitment_root: vector100.root(),
        history_len: 5,
    };
    assert_eq!(bank130.epoch_boundary(), Some(boundary2));
    assert_eq!(boundary2.parent_history_index(), Some(4));
    assert_eq!(history[4], leaf_of(&bank100));
    // Epoch 1 had four blocks.
    assert_eq!(boundary1.blocks_until(&boundary2), Some(4));
    assert_ne!(
        vector130.get(CommitmentSlot::ValidatorSetRoot),
        vector100.get(CommitmentSlot::ValidatorSetRoot)
    );
    history.push(leaf_of(&bank130));

    // Bank 160 is at the switch slot of epoch 2: generation 2, whose base is
    // the state of bank 100 and whose overlay has the accounts changed by
    // banks 130 and 160.
    let bank160 = new_child(&bank130, 160);
    let transactions160 = process_transactions(&bank160, &mint_keypair);
    freeze_with_block_id(&bank160);
    states.insert(160, scanned_state(&bank160));
    assert_eq!(bank160.accounts_state_generation_in_use(), Some(2));
    let slot5_160 =
        reference_accounts_state_root(2, &[&states[&100], &states[&130], &states[&160]]);
    let vector160 = check_commitment(&bank160, Some(&transactions160), &history, slot5_160);
    check_accounts_state_proofs(
        &bank160,
        &states[&160],
        &mint_keypair.pubkey(),
        true,
        &Pubkey::new_unique(),
    );
    check_accounts_state_proofs(
        &bank160,
        &states[&160],
        &untouched,
        false,
        &Pubkey::new_unique(),
    );

    // Every block of the chain has a different commitment root.
    let roots = [
        &vector65, &vector66, &vector96, &vector100, &vector130, &vector160,
    ]
    .map(|v| v.root());
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
        assert_all_features_active(bank, false);
        // No commitment state of any of the six features: replay accepts
        // only V1 footers and has no commitment root to verify.
        assert_eq!(bank.block_commitment(), None);
        assert_eq!(bank.commitment_root(), None);
        assert_eq!(bank.receipts_root(), None);
        assert!(bank.transaction_receipt_leaf_hashes().unwrap().is_empty());
        assert_eq!(bank.accounts_delta(), None);
        assert_eq!(bank.accounts_delta_root(), None);
        assert_eq!(bank.epoch_boundary(), None);
        assert!(bank.accounts_state().is_none());
        assert_eq!(bank.commitment_vector(), None);
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
        assert_eq!(fields.accounts_state, None);
    }
}

mod activation_order;
