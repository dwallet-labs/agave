//! Bank-level tests of the accounts state root (slot 5).
//!
//! Every frozen bank is checked against an independent reference computed
//! from full scans of the account state of the banks of its chain (read back
//! from accounts-db, hashed with the `blake3` crate), not from the overlays or
//! bases under test: the base state is the scanned state of `P`, the overlay
//! is the union of the addresses whose scanned state differs between a bank
//! and its parent, from `F` to the bank, and the keyed tree is recomputed by a
//! recursive implementation of the SIMD's definition.

use {
    super::*,
    crate::{
        bank::{
            block_commitment::{
                COMMITMENT_VECTOR_LEN, CommitmentSlot, CommitmentVector,
                tests::assert_commitment_failure,
            },
            test_utils::goto_end_of_slot,
        },
        bank_forks::BankForks,
        genesis_utils::{GenesisConfigInfo, activate_feature, create_genesis_config_with_leader},
        runtime_config::RuntimeConfig,
        snapshot_bank_utils,
        snapshot_utils::create_tmp_accounts_dir_for_tests,
    },
    agave_snapshots::snapshot_config::SnapshotConfig,
    agave_votor_messages::consensus_message::BlockId,
    solana_account::{AccountSharedData, WritableAccount},
    solana_accounts_db::accounts_db::ACCOUNTS_DB_CONFIG_FOR_TESTING,
    solana_feature_gate_interface::{self as feature, Feature},
    solana_genesis_config::GenesisConfig,
    solana_keypair::Keypair,
    solana_merkle_tree::{
        accounts_state::{AccountProofError, NO_ACCOUNT_HASH},
        commitment_tree::{TreeId, compute_root},
    },
    solana_native_token::LAMPORTS_PER_SOL,
    solana_sha256_hasher::hashv,
    solana_signer::Signer,
    solana_system_interface::instruction as system_instruction,
    solana_transaction::Transaction,
    std::{
        collections::{BTreeMap, BTreeSet},
        sync::RwLock,
    },
    tempfile::TempDir,
};

const SLOTS_PER_EPOCH: u64 = 64;

fn lookup_is_overlay(lookup: &solana_merkle_tree::accounts_state::AccountLookup) -> bool {
    matches!(
        lookup,
        solana_merkle_tree::accounts_state::AccountLookup::Overlay { .. }
    )
}

/// Independent `account_hash`.
fn blake3_account_hash(address: &Pubkey, account: &AccountSharedData) -> Hash {
    if account.lamports() == 0 {
        return NO_ACCOUNT_HASH;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(&account.lamports().to_le_bytes());
    hasher.update(account.data());
    hasher.update(&[u8::from(account.executable())]);
    hasher.update(account.owner().as_ref());
    hasher.update(address.as_ref());
    Hash::new_from_array(*hasher.finalize().as_bytes())
}

/// The account hashes of all accounts with lamports visible from `bank`.
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

/// Recursive `KTH` from the SIMD's definition.
fn reference_kth(entries: &[([u8; 32], [u8; 64])]) -> Hash {
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
                reference_kth(&zeros).as_ref(),
                reference_kth(&ones).as_ref(),
            ])
        }
    }
}

fn reference_keyed_root(entries: &BTreeMap<Pubkey, (Hash, Hash)>) -> Hash {
    let entries: Vec<([u8; 32], [u8; 64])> = entries
        .iter()
        .map(|(address, (now, base))| {
            let mut value = [0u8; 64];
            value[..32].copy_from_slice(now.as_ref());
            value[32..].copy_from_slice(base.as_ref());
            (address.to_bytes(), value)
        })
        .collect();
    hashv(&[
        &[0x81, 0x08],
        &(entries.len() as u64).to_le_bytes(),
        reference_kth(&entries).as_ref(),
    ])
}

fn reference_base_root(state: &BTreeMap<Pubkey, Hash>) -> Hash {
    let leaves: Vec<Vec<u8>> = state
        .iter()
        .map(|(address, hash)| [address.as_ref(), hash.as_ref()].concat())
        .collect();
    compute_root(TreeId::BASE_STATE, &leaves)
}

fn block_id_of(slot: Slot) -> BlockId {
    BlockId::from(hashv(&[b"block id", &slot.to_le_bytes()]))
}

/// The expected value of one generation of a bank.
#[derive(Debug, PartialEq, Eq)]
struct ReferenceGeneration {
    epoch: Epoch,
    base_root: Hash,
    overlay: BTreeMap<Pubkey, (Hash, Hash)>,
}

impl ReferenceGeneration {
    fn accounts_state_root(&self) -> Hash {
        accounts_state_root(
            self.epoch,
            &self.base_root,
            &reference_keyed_root(&self.overlay),
        )
    }
}

/// Banks of a test cluster, with the scanned state of every frozen bank.
struct Harness {
    genesis_config: GenesisConfig,
    mint_keypair: Keypair,
    bank_forks: Arc<RwLock<BankForks>>,
    states: HashMap<Slot, BTreeMap<Pubkey, Hash>>,
    parents: HashMap<Slot, Option<Slot>>,
    epoch_schedule: EpochSchedule,
    /// First epoch in which the feature is active.
    active_from: Epoch,
}

/// Feature states at genesis: `Some(true)` active, `Some(false)` pending
/// (activates at the first epoch boundary), `None` absent.
fn genesis(epoch_schedule: EpochSchedule, state_feature: Option<bool>) -> (GenesisConfig, Keypair) {
    let GenesisConfigInfo {
        mut genesis_config,
        mint_keypair,
        ..
    } = create_genesis_config_with_leader(
        1_000_000 * LAMPORTS_PER_SOL,
        &Pubkey::new_unique(),
        100 * LAMPORTS_PER_SOL,
    );
    genesis_config.epoch_schedule = epoch_schedule;
    activate_feature(
        &mut genesis_config,
        feature_set::block_commitment_root::id(),
    );
    activate_feature(&mut genesis_config, feature_set::accounts_delta_root::id());
    match state_feature {
        Some(true) => activate_feature(&mut genesis_config, feature_set::accounts_state_root::id()),
        Some(false) => {
            genesis_config.accounts.insert(
                feature_set::accounts_state_root::id(),
                feature::create_account(
                    &Feature { activated_at: None },
                    genesis_config
                        .rent
                        .minimum_balance(Feature::size_of())
                        .max(1),
                )
                .into(),
            );
        }
        None => {}
    }
    (genesis_config, mint_keypair)
}

impl Harness {
    fn new(epoch_schedule: EpochSchedule, state_feature: Option<bool>) -> (Self, Arc<Bank>) {
        let (genesis_config, mint_keypair) = genesis(epoch_schedule.clone(), state_feature);
        let (bank0, bank_forks) = Bank::new_with_bank_forks_for_tests(&genesis_config);
        let mut harness = Self {
            genesis_config,
            mint_keypair,
            bank_forks,
            states: HashMap::new(),
            parents: HashMap::new(),
            epoch_schedule,
            active_from: match state_feature {
                Some(true) => 0,
                Some(false) => 1,
                None => Epoch::MAX,
            },
        };
        harness.freeze(&bank0);
        (harness, bank0)
    }

    fn child(&self, parent: &Arc<Bank>, slot: Slot) -> Arc<Bank> {
        let leader = *parent.leader();
        Bank::new_from_parent_with_bank_forks(&self.bank_forks, parent.clone(), leader, slot)
    }

    /// Freezes `bank` (with ticks), gives it a block id, records its state,
    /// and checks its slot 5 and generations against the reference.
    fn freeze(&mut self, bank: &Arc<Bank>) {
        goto_end_of_slot(bank.clone());
        assert!(bank.is_frozen());
        bank.set_block_id(Some(block_id_of(bank.slot())));
        self.record(bank);
        self.check(bank);
    }

    fn record(&mut self, bank: &Bank) {
        self.states.insert(bank.slot(), scanned_state(bank));
        self.parents
            .insert(bank.slot(), bank.parent().map(|parent| parent.slot()));
    }

    fn epoch(&self, slot: Slot) -> Epoch {
        self.epoch_schedule.get_epoch(slot)
    }

    fn chain(&self, slot: Slot) -> Vec<Slot> {
        let mut chain = vec![slot];
        while let Some(Some(parent)) = self.parents.get(chain.last().unwrap()) {
            chain.push(*parent);
        }
        chain.reverse();
        chain
    }

    /// Reference generation `generation` of the bank at `slot`, from the
    /// recorded scans; `None` if it does not exist on the chain.
    fn reference_generation(&self, slot: Slot, generation: Epoch) -> Option<ReferenceGeneration> {
        if generation < self.active_from || self.epoch_schedule.get_slots_in_epoch(generation) < 64
        {
            return None;
        }
        let chain = self.chain(slot);
        let first = chain.iter().position(|s| self.epoch(*s) == generation)?;
        let empty = BTreeMap::new();
        let state_of = |index: Option<usize>| index.map_or(&empty, |i| &self.states[&chain[i]]);
        let base_state = state_of(first.checked_sub(1));
        let mut changed = BTreeSet::new();
        for index in first..chain.len() {
            let before = state_of(index.checked_sub(1));
            let after = state_of(Some(index));
            for address in before.keys().chain(after.keys()) {
                if before.get(address) != after.get(address) {
                    changed.insert(*address);
                }
            }
        }
        let now = &self.states[&slot];
        let overlay = changed
            .into_iter()
            .map(|address| {
                (
                    address,
                    (
                        now.get(&address).copied().unwrap_or_default(),
                        base_state.get(&address).copied().unwrap_or_default(),
                    ),
                )
            })
            .collect();
        Some(ReferenceGeneration {
            epoch: generation,
            base_root: reference_base_root(base_state),
            overlay,
        })
    }

    fn reference_slot5(&self, bank: &Bank) -> Hash {
        let epoch = self.epoch(bank.slot());
        let switch = self.epoch_schedule.get_first_slot_in_epoch(epoch)
            + self.epoch_schedule.get_slots_in_epoch(epoch) / 2;
        let generation = if bank.slot() < switch {
            epoch.checked_sub(1)
        } else {
            Some(epoch)
        };
        generation
            .and_then(|generation| self.reference_generation(bank.slot(), generation))
            .map_or(Hash::default(), |generation| {
                generation.accounts_state_root()
            })
    }

    fn check(&self, bank: &Bank) {
        assert_eq!(bank.commitment_failure(), None, "slot {}", bank.slot());
        let expected_slot5 = self.reference_slot5(bank);
        let vector = bank.commitment_vector().unwrap();
        assert_eq!(
            vector.get(CommitmentSlot::AccountsStateRoot),
            expected_slot5,
            "slot 5 of bank {}",
            bank.slot()
        );
        // The commitment root commits to it, with the other slots of this
        // branch.
        let mut expected = [Hash::default(); COMMITMENT_VECTOR_LEN];
        expected[CommitmentSlot::BlockHistoryRoot.index()] =
            bank.block_commitment().unwrap().block_history_root();
        expected[CommitmentSlot::AccountsDeltaRoot.index()] = bank.accounts_delta_root().unwrap();
        expected[CommitmentSlot::AccountsStateRoot.index()] = expected_slot5;
        assert_eq!(
            bank.commitment_root(),
            Some(CommitmentVector(expected).root())
        );
        if !bank.is_accounts_state_root_active() {
            assert!(bank.accounts_state().is_none());
            return;
        }
        // Every generation the bank holds matches the reference: the overlay
        // entry by entry, and the base root (waiting for it if needed).
        let state = bank.accounts_state().unwrap();
        for generation in state.generations() {
            let reference = self
                .reference_generation(bank.slot(), generation.epoch())
                .expect("the bank holds a generation that does not exist");
            let overlay: BTreeMap<_, _> = generation
                .overlay()
                .iter()
                .map(|(address, value)| {
                    let entry = OverlayEntry::from_value(*address, value);
                    (Pubkey::new_from_array(*address), (entry.now, entry.base))
                })
                .collect();
            assert_eq!(
                overlay,
                reference.overlay,
                "overlay of generation {} at bank {}",
                generation.epoch(),
                bank.slot()
            );
            assert_eq!(
                generation.base().wait_root(),
                Ok(reference.base_root),
                "base of generation {} at bank {}",
                generation.epoch(),
                bank.slot()
            );
        }
        // And the bank holds exactly the generations of the spec: the one in
        // use, and the one of its own epoch.
        let epoch = self.epoch(bank.slot());
        let held: Vec<_> = state.generations().iter().map(|g| g.epoch()).collect();
        let mut expected_held = Vec::new();
        if let Some(previous) = epoch.checked_sub(1)
            && bank.accounts_state_generation_in_use() == Some(previous)
            && self.reference_generation(bank.slot(), previous).is_some()
        {
            expected_held.push(previous);
        }
        if self.reference_generation(bank.slot(), epoch).is_some() {
            expected_held.push(epoch);
        }
        assert_eq!(held, expected_held, "generations of bank {}", bank.slot());
    }

    fn transfer(&self, bank: &Bank, to: &Pubkey, lamports: u64) {
        bank.register_unique_recent_blockhash_for_test();
        bank.transfer(lamports, &self.mint_keypair, to).unwrap();
    }

    /// A transfer of `lamports` from `from` to `to`, fees paid by the mint.
    fn transfer_from(&self, bank: &Bank, from: &Keypair, to: &Pubkey, lamports: u64) {
        bank.register_unique_recent_blockhash_for_test();
        let tx = Transaction::new_signed_with_payer(
            &[system_instruction::transfer(&from.pubkey(), to, lamports)],
            Some(&self.mint_keypair.pubkey()),
            &[&self.mint_keypair, from],
            bank.last_blockhash(),
        );
        bank.process_transaction(&tx).unwrap();
    }

    /// Checks the proofs of `addresses` at a frozen `bank` against its
    /// commitment root and the independently scanned state.
    fn check_proofs(&self, bank: &Bank, addresses: &[Pubkey]) -> usize {
        let commitment_root = bank.commitment_root().unwrap();
        let state = &self.states[&bank.slot()];
        let mut num_base = 0;
        for address in addresses {
            let proof = bank.accounts_state_proof(address).unwrap();
            if !lookup_is_overlay(&proof.lookup) {
                num_base += 1;
            }
            let generation = bank.accounts_state_generation_in_use().unwrap();
            match state.get(address) {
                Some(hash) => {
                    assert_eq!(
                        proof.verify_current_state(&commitment_root, &address.to_bytes(), hash),
                        Ok(generation),
                        "{address} at {}",
                        bank.slot()
                    );
                    assert!(
                        proof
                            .verify_nonexistence(&commitment_root, &address.to_bytes())
                            .is_err()
                    );
                    // Another state of the account does not verify.
                    let mut other = bank.get_account(address).unwrap();
                    other.set_lamports(other.lamports() + 1);
                    assert!(
                        proof
                            .verify_current_state(
                                &commitment_root,
                                &address.to_bytes(),
                                &blake3_account_hash(address, &other)
                            )
                            .is_err()
                    );
                }
                None => {
                    assert_eq!(
                        proof.verify_nonexistence(&commitment_root, &address.to_bytes()),
                        Ok(generation),
                        "{address} at {}",
                        bank.slot()
                    );
                    assert!(
                        proof
                            .verify_current_state(
                                &commitment_root,
                                &address.to_bytes(),
                                &Hash::new_from_array([7; 32])
                            )
                            .is_err()
                    );
                }
            }
            let hash = state.get(address).copied().unwrap_or_default();
            // Bound to the commitment root and to the generation.
            assert_eq!(
                proof.verify(&Hash::new_unique(), &address.to_bytes(), &hash),
                Err(AccountProofError::Opening)
            );
            let mut tampered = proof.clone();
            tampered.opening.generation += 1;
            assert!(
                tampered
                    .verify(&commitment_root, &address.to_bytes(), &hash)
                    .is_err()
            );
            let mut tampered = proof.clone();
            tampered.opening.base_root = Hash::new_unique();
            assert!(
                tampered
                    .verify(&commitment_root, &address.to_bytes(), &hash)
                    .is_err()
            );
        }
        num_base
    }
}

/// Activation at genesis, a chain across two epoch changes and three switch
/// slots with created, modified, modified-and-restored, closed,
/// created-and-closed and untouched accounts, and proofs at several banks.
#[test]
fn test_chain_across_epochs_and_switch_slots() {
    let (mut h, bank0) = Harness::new(
        EpochSchedule::custom(SLOTS_PER_EPOCH, SLOTS_PER_EPOCH, false),
        Some(true),
    );
    assert!(bank0.is_accounts_state_root_active());
    // Epoch 0 before its switch slot uses generation -1: zero.
    assert_eq!(
        bank0
            .commitment_vector()
            .unwrap()
            .get(CommitmentSlot::AccountsStateRoot),
        Hash::default()
    );

    let untouched = Keypair::new();
    let modified = Keypair::new();
    let restored = Keypair::new();
    let restored_peer = Keypair::new();
    let closed = Keypair::new();
    let created_and_closed = Keypair::new();
    let never = Pubkey::new_unique();

    let bank1 = h.child(&bank0, 1);
    for account in [&untouched, &modified, &restored, &restored_peer, &closed] {
        h.transfer(&bank1, &account.pubkey(), 10 * LAMPORTS_PER_SOL);
    }
    h.freeze(&bank1);

    // Epoch 0, before and at its switch slot (32).
    let mut parent = bank1;
    for slot in [2, 31, 32, 40, 63] {
        let bank = h.child(&parent, slot);
        h.transfer(&bank, &Pubkey::new_unique(), LAMPORTS_PER_SOL);
        h.freeze(&bank);
        parent = bank;
    }
    let bank63 = parent;
    // From slot 32 on, generation 0 (base: the empty state before genesis).
    assert_ne!(
        bank63
            .commitment_vector()
            .unwrap()
            .get(CommitmentSlot::AccountsStateRoot),
        Hash::default()
    );
    let addresses = [
        untouched.pubkey(),
        modified.pubkey(),
        closed.pubkey(),
        never,
        h.mint_keypair.pubkey(),
    ];
    h.check_proofs(&bank63, &addresses);

    // Epoch 1: first block (64), before the switch slot (96) it uses
    // generation 0 and also tracks generation 1.
    let bank64 = h.child(&bank63, 64);
    h.transfer_from(&bank64, &modified, &Pubkey::new_unique(), LAMPORTS_PER_SOL);
    h.freeze(&bank64);
    assert_eq!(bank64.accounts_state().unwrap().generations().len(), 2);
    // Modify and restore across blocks, close, create and close.
    let bank65 = h.child(&bank64, 65);
    h.transfer_from(&bank65, &restored, &restored_peer.pubkey(), 1);
    h.transfer(&bank65, &created_and_closed.pubkey(), LAMPORTS_PER_SOL);
    h.freeze(&bank65);
    let bank70 = h.child(&bank65, 70);
    h.transfer_from(&bank70, &restored_peer, &restored.pubkey(), 1);
    h.transfer_from(
        &bank70,
        &closed,
        &Pubkey::new_unique(),
        10 * LAMPORTS_PER_SOL,
    );
    h.transfer_from(
        &bank70,
        &created_and_closed,
        &Pubkey::new_unique(),
        LAMPORTS_PER_SOL,
    );
    h.freeze(&bank70);
    assert_eq!(bank70.get_balance(&closed.pubkey()), 0);
    // The restored account keeps its entry, with both halves equal.
    let entry = bank70
        .accounts_state_overlay_entry(1, &restored.pubkey())
        .unwrap();
    assert_eq!(entry.now, entry.base);
    assert_ne!(entry.now, NO_ACCOUNT_HASH);
    // The created-and-closed account has an entry with both halves zero.
    let entry = bank70
        .accounts_state_overlay_entry(1, &created_and_closed.pubkey())
        .unwrap();
    assert_eq!((entry.now, entry.base), (NO_ACCOUNT_HASH, NO_ACCOUNT_HASH));
    // The closed account has a zero first half.
    let entry = bank70
        .accounts_state_overlay_entry(1, &closed.pubkey())
        .unwrap();
    assert_eq!(entry.now, NO_ACCOUNT_HASH);
    assert_ne!(entry.base, NO_ACCOUNT_HASH);
    // The untouched account has none.
    assert!(
        bank70
            .accounts_state_overlay_entry(1, &untouched.pubkey())
            .is_none()
    );

    let mut parent = bank70;
    for slot in [95, 96, 97, 127] {
        let bank = h.child(&parent, slot);
        h.transfer(&bank, &modified.pubkey(), 1);
        h.freeze(&bank);
        parent = bank;
    }
    let bank127 = parent;
    // After the switch slot of epoch 1, one generation.
    assert_eq!(bank127.accounts_state().unwrap().generations().len(), 1);
    let addresses = [
        untouched.pubkey(),
        modified.pubkey(),
        restored.pubkey(),
        closed.pubkey(),
        created_and_closed.pubkey(),
        never,
        h.mint_keypair.pubkey(),
        sysvar::rent::id(),
    ];
    // The untouched account, the rent sysvar and the absent address are
    // proven through the base.
    assert!(h.check_proofs(&bank127, &addresses) >= 3);

    // Epoch 2 (base by merge) up to after its switch slot (160), and the
    // first block of epoch 3.
    let mut parent = bank127;
    for slot in [128, 129, 159, 160, 161, 192, 193] {
        let bank = h.child(&parent, slot);
        h.transfer(&bank, &Pubkey::new_unique(), LAMPORTS_PER_SOL);
        h.freeze(&bank);
        parent = bank;
    }
    h.check_proofs(&parent, &addresses);

    // The leaves of the bases are in files, of the two latest epochs only.
    let state = parent.accounts_state().unwrap();
    let base = state.generation(3).unwrap().base().wait_tree().unwrap();
    assert!(!base.is_empty());
    let dir = parent.accounts_state_bases.dir.clone();
    let mut epochs: Vec<String> = std::fs::read_dir(dir.path().unwrap())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .map(|name| {
            assert!(name.ends_with(".leaves"), "{name}");
            name.split('-').next().unwrap().to_string()
        })
        .collect();
    epochs.sort();
    assert_eq!(epochs, ["2", "3"]);
    let on_disk = dir.open(&state.generation(3).unwrap().base().id()).unwrap();
    assert_eq!(on_disk.bytes().len(), base.len() * 64);
}

/// After a restart, the leaves on disk are used only if they have the base
/// root the snapshot carries.
#[test]
fn test_base_file_reused_only_with_the_snapshot_root() {
    let accounts = TempDir::new().unwrap();
    let bases = AccountsStateBases::for_accounts_path(accounts.path());
    let id = BaseId {
        epoch: 7,
        parent: Some((447, Hash::new_unique())),
    };
    let mut leaves: Vec<_> = (0..3_000u64)
        .map(|i| {
            solana_merkle_tree::accounts_state::BaseLeaf::new(
                hashv(&[&i.to_le_bytes()]).to_bytes(),
                hashv(&[b"state", &i.to_le_bytes()]),
            )
        })
        .collect();
    leaves.sort_unstable();
    let mut writer = bases.dir.writer(&id);
    for leaf in &leaves {
        writer.push(*leaf).unwrap();
    }
    let tree = tree_from_storage(writer.finish().unwrap()).unwrap();
    assert_eq!(tree, BaseTree::new(leaves).unwrap());

    // A later process, on the same directory.
    let bases = AccountsStateBases::for_accounts_path(accounts.path());
    // A prover can open the file without a snapshot; it checks the root.
    assert_eq!(
        bases.open_from_disk(id).unwrap().wait_root(),
        Ok(tree.root())
    );
    let other = BaseId { epoch: 8, ..id };
    assert!(bases.open_from_disk(other).is_none());
    let first = BaseId {
        epoch: 0,
        parent: None,
    };
    assert_eq!(
        bases.open_from_disk(first).unwrap().wait_root(),
        Ok(BaseTree::empty().root())
    );
    let cell = bases.snapshot_cell(id, Some(tree.root()));
    assert_eq!(reuse_base_file(&bases.dir, &cell).unwrap(), tree);
    // The snapshot has no root for the base: the file cannot be checked. It
    // is left for a snapshot that has one.
    let cell = bases.snapshot_cell(id, None);
    assert!(reuse_base_file(&bases.dir, &cell).is_none());
    assert!(bases.dir.open(&id).is_some());
    // The snapshot has another root: the file is not its base.
    let cell = bases.snapshot_cell(id, Some(Hash::new_unique()));
    assert!(reuse_base_file(&bases.dir, &cell).is_none());
    assert!(bases.dir.open(&id).is_none());
}

/// The feature activates at the first epoch boundary: zero until the switch
/// slot of the activation epoch, the first base by scan, the next by merge.
#[test]
fn test_activation() {
    let (mut h, bank0) = Harness::new(
        EpochSchedule::custom(SLOTS_PER_EPOCH, SLOTS_PER_EPOCH, false),
        Some(false),
    );
    assert!(!bank0.is_accounts_state_root_active());
    assert!(bank0.accounts_state().is_none());
    let mut parent = bank0;
    for slot in [1, 40, 63, 64, 65, 95, 96, 97, 128, 159, 160] {
        let bank = h.child(&parent, slot);
        h.transfer(&bank, &Pubkey::new_unique(), LAMPORTS_PER_SOL);
        h.freeze(&bank);
        let slot5 = bank
            .commitment_vector()
            .unwrap()
            .get(CommitmentSlot::AccountsStateRoot);
        // Zero before activation and in the activation epoch before its
        // switch slot.
        assert_eq!(slot5 == Hash::default(), slot < 96, "slot {slot}");
        assert_eq!(bank.is_accounts_state_root_active(), slot >= 64);
        parent = bank;
    }
}

/// Epochs shorter than the minimum have no generation: with a warmup
/// schedule epoch 0 has 32 slots, so blocks of epoch 1 before its switch slot
/// have zero in slot 5 and the base of epoch 1 is built by scan.
#[test]
fn test_short_epoch() {
    let epoch_schedule = EpochSchedule::custom(256, 256, true);
    assert_eq!(epoch_schedule.get_slots_in_epoch(0), 32);
    assert_eq!(epoch_schedule.get_slots_in_epoch(1), 64);
    assert!(!epoch_has_generation(&epoch_schedule, 0));
    assert!(epoch_has_generation(&epoch_schedule, 1));
    let (mut h, bank0) = Harness::new(epoch_schedule.clone(), Some(true));
    assert!(bank0.accounts_state().unwrap().generations().is_empty());
    let mut parent = bank0;
    // Epoch 0: slots 0..32, switch 16; epoch 1: slots 32..96, switch 64;
    // epoch 2: slots 96..224, switch 160.
    for slot in [1, 16, 31, 32, 33, 63, 64, 65, 96, 159, 160] {
        let bank = h.child(&parent, slot);
        h.transfer(&bank, &Pubkey::new_unique(), LAMPORTS_PER_SOL);
        h.freeze(&bank);
        let slot5 = bank
            .commitment_vector()
            .unwrap()
            .get(CommitmentSlot::AccountsStateRoot);
        assert_eq!(slot5 == Hash::default(), slot < 64, "slot {slot}");
        parent = bank;
    }
}

/// Two forks with a different last block before epoch 2 get different bases;
/// forks with the same one share the base job.
#[test]
fn test_forks_with_different_base_blocks() {
    let (mut h, bank0) = Harness::new(
        EpochSchedule::custom(SLOTS_PER_EPOCH, SLOTS_PER_EPOCH, false),
        Some(true),
    );
    let mut parent = bank0;
    for slot in [1, 64, 100] {
        let bank = h.child(&parent, slot);
        h.transfer(&bank, &Pubkey::new_unique(), LAMPORTS_PER_SOL);
        h.freeze(&bank);
        parent = bank;
    }
    let bank100 = parent;
    // Fork A: 100 <- 120 (a change) <- 128.
    let bank120 = h.child(&bank100, 120);
    let only_on_a = Pubkey::new_unique();
    h.transfer(&bank120, &only_on_a, LAMPORTS_PER_SOL);
    h.freeze(&bank120);
    let a128 = h.child(&bank120, 128);
    h.freeze(&a128);
    // Fork B: 100 <- 129 (P = 100).
    let b129 = h.child(&bank100, 129);
    h.freeze(&b129);
    // Fork C: 120 <- 130, the same P as fork A.
    let c130 = h.child(&bank120, 130);
    h.freeze(&c130);

    let base_of = |bank: &Bank| {
        bank.accounts_state()
            .unwrap()
            .generation(2)
            .unwrap()
            .base()
            .clone()
    };
    assert_eq!(base_of(&a128).id().parent, Some((120, bank120.hash())));
    assert_eq!(base_of(&b129).id().parent, Some((100, bank100.hash())));
    assert!(Arc::ptr_eq(&base_of(&a128), &base_of(&c130)));
    assert_ne!(
        base_of(&a128).wait_root().unwrap(),
        base_of(&b129).wait_root().unwrap()
    );
    // Both forks cross the switch slot with their own base (checked against
    // the scans of their own `P` by the harness).
    let a160 = h.child(&a128, 160);
    h.freeze(&a160);
    let b161 = h.child(&b129, 161);
    h.freeze(&b161);
    assert_ne!(
        a160.commitment_vector()
            .unwrap()
            .get(CommitmentSlot::AccountsStateRoot),
        b161.commitment_vector()
            .unwrap()
            .get(CommitmentSlot::AccountsStateRoot)
    );
    // The account created only on fork A is proven on A and absent on B.
    assert_eq!(h.check_proofs(&a160, &[only_on_a]), 1);
    assert_eq!(h.check_proofs(&b161, &[only_on_a]), 1);
    assert!(!h.states[&b161.slot()].contains_key(&only_on_a));
}

/// A base job that failed is rebuilt synchronously by the first bank that
/// needs it, from the state of its parent.
#[test]
fn test_failed_base_job_is_rebuilt_synchronously() {
    let (mut h, bank0) = Harness::new(
        EpochSchedule::custom(SLOTS_PER_EPOCH, SLOTS_PER_EPOCH, false),
        Some(true),
    );
    let mut parent = bank0;
    for slot in [1, 63, 64, 70] {
        let bank = h.child(&parent, slot);
        h.transfer(&bank, &Pubkey::new_unique(), LAMPORTS_PER_SOL);
        h.freeze(&bank);
        parent = bank;
    }
    let cell = parent
        .accounts_state()
        .unwrap()
        .generation(1)
        .unwrap()
        .base()
        .clone();
    cell.fail_for_tests("injected");
    assert!(cell.wait_root().is_err());
    let bank96 = h.child(&parent, 96);
    h.freeze(&bank96);
    assert!(cell.wait_root().is_ok());
}

/// A base job that fails is run again: the cell stays pending and gets the
/// tree of the attempt that succeeds. After the last attempt the failure is
/// final. Nobody holding the cell ends the attempts.
#[test]
fn test_base_job_is_run_again_after_a_failure() {
    let id = |epoch| BaseId {
        epoch,
        parent: None,
    };
    let attempts = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let run = |cell: &Arc<BaseCell>, succeed_at: u32| {
        attempts.store(0, Ordering::Relaxed);
        let attempts = attempts.clone();
        spawn_base_job(vec![cell.clone()], move || {
            let attempt = attempts.fetch_add(1, Ordering::Relaxed) + 1;
            if attempt >= succeed_at {
                Ok(vec![BaseTree::empty()])
            } else {
                Err(format!("attempt {attempt} failed"))
            }
        });
    };

    let cell = Arc::new(BaseCell::new(id(1), None, false));
    run(&cell, 3);
    assert_eq!(cell.wait_root(), Ok(BaseTree::empty().root()));
    assert_eq!(attempts.load(Ordering::Relaxed), 3);
    assert!(cell.problem_to_report(true).is_none());

    let cell = Arc::new(BaseCell::new(id(2), None, false));
    // Still running a quarter of the epoch in: reported once.
    assert!(cell.problem_to_report(false).is_none());
    assert!(cell.problem_to_report(true).unwrap().contains("running"));
    assert!(cell.problem_to_report(true).is_none());
    run(&cell, u32::MAX);
    assert_eq!(
        cell.wait_root(),
        Err(format!("attempt {BASE_JOB_ATTEMPTS} failed"))
    );
    assert_eq!(attempts.load(Ordering::Relaxed), BASE_JOB_ATTEMPTS);

    // A failure for good is reported at any time, once.
    let cell = Arc::new(BaseCell::new(id(3), None, false));
    cell.fail_for_tests("injected");
    assert!(cell.problem_to_report(false).unwrap().contains("injected"));
    assert!(cell.problem_to_report(false).is_none());
    // A base whose root came with a snapshot is not late.
    let cell = Arc::new(BaseCell::new(id(4), Some(Hash::new_unique()), true));
    assert!(cell.problem_to_report(true).is_none());

    // No bank holds the cell: one attempt.
    let weak = {
        let cell = Arc::new(BaseCell::new(id(5), None, false));
        run(&cell, u32::MAX);
        Arc::downgrade(&cell)
    };
    while weak.strong_count() > 0 {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(attempts.load(Ordering::Relaxed) < BASE_JOB_ATTEMPTS);
}

/// Neither the state nor any base job exists while the feature is inactive,
/// and slot 5 is zero.
#[test]
fn test_inactive_feature() {
    let (genesis_config, mint_keypair) = genesis(
        EpochSchedule::custom(SLOTS_PER_EPOCH, SLOTS_PER_EPOCH, false),
        None,
    );
    let (bank0, bank_forks) = Bank::new_with_bank_forks_for_tests(&genesis_config);
    bank0.freeze();
    bank0.set_block_id(Some(block_id_of(0)));
    let mut parent = bank0;
    for slot in [1, 63, 64, 96, 128] {
        let leader = *parent.leader();
        let bank = Bank::new_from_parent_with_bank_forks(&bank_forks, parent.clone(), leader, slot);
        bank.register_unique_recent_blockhash_for_test();
        bank.transfer(1, &mint_keypair, &Pubkey::new_unique())
            .unwrap();
        goto_end_of_slot(bank.clone());
        bank.set_block_id(Some(block_id_of(slot)));
        assert!(!bank.is_accounts_state_root_active());
        assert!(bank.accounts_state().is_none());
        assert!(bank.accounts_state_for_snapshot().is_none());
        assert_eq!(
            bank.commitment_vector()
                .unwrap()
                .get(CommitmentSlot::AccountsStateRoot),
            Hash::default()
        );
        assert_eq!(
            bank.accounts_state_proof(&Pubkey::new_unique()),
            Err(AccountsStateProofError::Inactive)
        );
        parent = bank;
    }
    assert!(parent.accounts_state_bases.cells.lock().unwrap().is_empty());

    // State held while the feature is not in effect (which only a test can
    // produce) is not written to a snapshot.
    let mut genesis_config = genesis_config;
    let (state_genesis, _) = genesis(
        EpochSchedule::custom(SLOTS_PER_EPOCH, SLOTS_PER_EPOCH, false),
        Some(true),
    );
    let mut bank0 = Bank::new_for_tests(&state_genesis);
    bank0.freeze();
    bank0.set_block_id(Some(block_id_of(0)));
    assert!(bank0.accounts_state_for_snapshot().is_some());
    bank0.deactivate_feature(&feature_set::accounts_delta_root::id());
    assert!(bank0.accounts_state().is_some());
    assert!(bank0.accounts_state_for_snapshot().is_none());
    assert_eq!(bank0.get_fields_to_serialize().accounts_state, None);

    // The accounts state feature alone, without accounts_delta_root, does
    // nothing either.
    activate_feature(&mut genesis_config, feature_set::accounts_state_root::id());
    genesis_config
        .accounts
        .remove(&feature_set::accounts_delta_root::id());
    let bank0 = Bank::new_for_tests(&genesis_config);
    assert!(!bank0.is_accounts_state_root_active());
    bank0.freeze();
    assert!(bank0.accounts_state().is_none());
}

/// A child of a bank that should have an accounts state but does not (here
/// a parent whose state was dropped, as for a snapshot written without it)
/// has a commitment failure.
#[test]
fn test_missing_parent_state_is_commitment_failure() {
    let (mut h, bank0) = Harness::new(
        EpochSchedule::custom(SLOTS_PER_EPOCH, SLOTS_PER_EPOCH, false),
        Some(true),
    );
    let bank1 = h.child(&bank0, 1);
    h.freeze(&bank1);
    *bank1.accounts_state.write().unwrap() = None;
    let bank2 = h.child(&bank1, 2);
    goto_end_of_slot(bank2.clone());
    assert_commitment_failure(
        &bank2,
        CommitmentFailure::AccountsStateNotTracked { slot: 1 },
    );
}

/// Snapshot round trip: the restored bank's descendants have the same bank
/// hashes and commitment roots as the original chain's, across the next
/// switch slot and the next epoch (whose base is merged from the base leaves
/// rebuilt from the snapshot).
///
/// With `strip_base_roots`, the restored bank is given its state without the
/// base roots, as for a snapshot taken while the base jobs were running: the
/// roots then come from the leaves rebuilt by the scan of the restored bank.
fn snapshot_round_trip(snapshot_slot: Slot, descendants: &[Slot], strip_base_roots: bool) {
    let (mut h, bank0) = Harness::new(
        EpochSchedule::custom(SLOTS_PER_EPOCH, SLOTS_PER_EPOCH, false),
        Some(true),
    );
    let keep = Keypair::new();
    let mut parent = bank0;
    for slot in [1, 63, 64, 100, 128] {
        if slot >= snapshot_slot {
            break;
        }
        let bank = h.child(&parent, slot);
        h.transfer(&bank, &keep.pubkey(), LAMPORTS_PER_SOL);
        h.transfer(&bank, &Pubkey::new_unique(), LAMPORTS_PER_SOL);
        h.freeze(&bank);
        parent = bank;
    }
    let snapshot_bank = h.child(&parent, snapshot_slot);
    h.transfer(&snapshot_bank, &Pubkey::new_unique(), LAMPORTS_PER_SOL);
    h.freeze(&snapshot_bank);
    let snapshot_state = snapshot_bank.accounts_state_for_snapshot().unwrap();

    let (_tmp_dir, accounts_dir) = create_tmp_accounts_dir_for_tests();
    let bank_snapshots_dir = TempDir::new().unwrap();
    let full_snapshot_archives_dir = TempDir::new().unwrap();
    let incremental_snapshot_archives_dir = TempDir::new().unwrap();
    let snapshot_config = SnapshotConfig {
        full_snapshot_archives_dir: full_snapshot_archives_dir.path().to_path_buf(),
        incremental_snapshot_archives_dir: incremental_snapshot_archives_dir.path().to_path_buf(),
        bank_snapshots_dir: bank_snapshots_dir.path().to_path_buf(),
        ..SnapshotConfig::default()
    };
    let archive =
        snapshot_bank_utils::bank_to_full_snapshot_archive(&snapshot_config, &snapshot_bank)
            .unwrap();
    let restored = snapshot_bank_utils::bank_from_snapshot_archives(
        &[accounts_dir],
        &archive,
        None,
        &snapshot_config,
        &h.genesis_config,
        &RuntimeConfig::default(),
        None,
        Some(*snapshot_bank.leader()),
        None,
        false,
        ACCOUNTS_DB_CONFIG_FOR_TESTING,
        None,
        Arc::default(),
    )
    .unwrap();
    assert_eq!(restored.hash(), snapshot_bank.hash());
    assert_eq!(restored.commitment_root(), snapshot_bank.commitment_root());
    assert_eq!(
        restored.accounts_state_for_snapshot(),
        Some(snapshot_state.clone())
    );
    assert!(
        snapshot_state
            .generations
            .iter()
            .all(|generation| generation.base_root.is_some())
    );
    if strip_base_roots {
        let mut stripped = snapshot_state;
        for generation in &mut stripped.generations {
            generation.base_root = None;
        }
        *restored.accounts_state.write().unwrap() =
            Bank::accounts_state_from_snapshot(Some(stripped), &restored.accounts_state_bases);
    }
    let (restored, restored_forks) = restored.wrap_with_bank_forks_for_tests();

    let mut original_parent = snapshot_bank;
    let mut restored_parent = restored;
    for slot in descendants {
        let original = h.child(&original_parent, *slot);
        let leader = *restored_parent.leader();
        let copy = Bank::new_from_parent_with_bank_forks(
            &restored_forks,
            restored_parent.clone(),
            leader,
            *slot,
        );
        // The same changes on both chains (stores, which do not depend on
        // a recent blockhash).
        let address = Pubkey::new_from_array(hashv(&[&slot.to_le_bytes()]).to_bytes());
        let account = AccountSharedData::new_with_data(
            *slot * 1000 + 1,
            vec![*slot as u8; 8],
            &Pubkey::default(),
        );
        for bank in [&original, &copy] {
            bank.store_account_and_update_capitalization(&address, &account);
            bank.store_account_and_update_capitalization(&keep.pubkey(), &account);
        }
        h.freeze(&original);
        goto_end_of_slot(copy.clone());
        copy.set_block_id(Some(block_id_of(*slot)));
        assert_eq!(copy.commitment_failure(), None, "slot {slot}");
        assert_eq!(copy.hash(), original.hash(), "slot {slot}");
        assert_eq!(
            copy.commitment_root(),
            original.commitment_root(),
            "slot {slot}"
        );
        assert_eq!(copy.commitment_vector(), original.commitment_vector());
        // Proofs from the restored chain (through the rebuilt base leaves).
        if copy.accounts_state_generation_in_use().is_some()
            && copy
                .commitment_vector()
                .unwrap()
                .get(CommitmentSlot::AccountsStateRoot)
                != Hash::default()
        {
            let commitment_root = copy.commitment_root().unwrap();
            for address in [keep.pubkey(), sysvar::rent::id(), Pubkey::new_unique()] {
                let proof = copy.accounts_state_proof(&address).unwrap();
                let hash = h.states[slot].get(&address).copied().unwrap_or_default();
                assert!(
                    proof
                        .verify(&commitment_root, &address.to_bytes(), &hash)
                        .is_ok()
                );
            }
        }
        original_parent = original;
        restored_parent = copy;
    }
}

/// A snapshot whose base root is not the root of the base leaves rebuilt
/// from its accounts is invalid: once the rebuild has found the difference,
/// a bank that needs that base has a commitment failure. It uses neither
/// root, and the bad root is not written to a later snapshot.
#[test]
fn test_snapshot_base_root_mismatch_is_commitment_failure() {
    let (mut h, bank0) = Harness::new(
        EpochSchedule::custom(SLOTS_PER_EPOCH, SLOTS_PER_EPOCH, false),
        Some(true),
    );
    let mut parent = bank0;
    for slot in [1, 63, 64, 100, 128, 140] {
        let bank = h.child(&parent, slot);
        h.transfer(&bank, &Pubkey::new_unique(), LAMPORTS_PER_SOL);
        h.freeze(&bank);
        parent = bank;
    }
    let snapshot_bank = parent;
    // Slot 150 is before the switch slot of epoch 2 and uses generation 1.
    let mut snapshot_state = snapshot_bank.accounts_state_for_snapshot().unwrap();
    let generation = snapshot_state
        .generations
        .iter_mut()
        .find(|generation| generation.epoch == 1)
        .unwrap();
    let rebuilt = generation.base_root.unwrap();
    let snapshot = Hash::new_unique();
    generation.base_root = Some(snapshot);
    // As if the bank had been loaded from a snapshot with that root.
    *snapshot_bank.accounts_state.write().unwrap() = Bank::accounts_state_from_snapshot(
        Some(snapshot_state),
        &snapshot_bank.accounts_state_bases,
    );

    // The rebuild starts with the first child.
    let child = h.child(&snapshot_bank, 150);
    let state = snapshot_bank.accounts_state().unwrap();
    while !state.generations.iter().all(|g| g.base.is_finished()) {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let base = &state.generation(1).unwrap().base;
    assert_eq!(base.snapshot_mismatch(), Some((snapshot, rebuilt)));
    assert_eq!(base.root_if_known(), None);
    assert!(base.wait_root().is_err());
    assert!(base.wait_tree().is_err());

    goto_end_of_slot(child.clone());
    assert_commitment_failure(
        &child,
        CommitmentFailure::AccountsStateSnapshotBaseMismatch {
            generation: 1,
            snapshot,
            rebuilt,
        },
    );
}

/// Snapshot of a bank in epoch 2 before its switch slot (two generations).
#[test]
fn test_snapshot_before_switch_slot() {
    snapshot_round_trip(140, &[150, 160, 170, 192, 200, 224, 230], false);
}

/// Like [`test_snapshot_before_switch_slot`], without base roots in the
/// snapshot.
#[test]
fn test_snapshot_without_base_roots() {
    snapshot_round_trip(140, &[150, 160, 192, 224], true);
}

/// Snapshot of a bank in epoch 2 after its switch slot (one generation).
#[test]
fn test_snapshot_after_switch_slot() {
    snapshot_round_trip(170, &[180, 192, 200, 224, 230, 256, 288], false);
}

use solana_sdk_ids::sysvar;

/// Time to apply one block's changed accounts to an overlay that already
/// has many entries, on the freezing thread and on the overlay threads.
/// Run with `--ignored --nocapture`; `OVERLAY_BENCH_ENTRIES` and
/// `OVERLAY_BENCH_CHANGES` set the sizes.
#[test]
#[ignore]
fn bench_overlay_update() {
    let size = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    };
    let entries = size("OVERLAY_BENCH_ENTRIES", 1_000_000);
    let changes = size("OVERLAY_BENCH_CHANGES", 1_296);
    let address = |index: u64| solana_sha256_hasher::hash(&index.to_le_bytes()).to_bytes();
    let hash_of = |index: u64| solana_sha256_hasher::hash(&(index ^ u64::MAX).to_le_bytes());
    let mut overlay = new_overlay();
    for index in 0..entries as u64 {
        overlay.insert(
            address(index),
            solana_merkle_tree::accounts_state::overlay_value(&hash_of(index), &hash_of(index)),
        );
    }
    overlay.root();
    let cell = Arc::new(BaseCell::new(
        BaseId {
            epoch: 0,
            parent: None,
        },
        None,
        false,
    ));
    for generations in [1, 2] {
        for blocks_on_pool in [false, true] {
            let mut state: Vec<Generation> = (0..generations)
                .map(|_| Generation::from_parts(overlay.clone(), cell.clone()))
                .collect();
            let mut total = std::time::Duration::ZERO;
            const BLOCKS: u64 = 50;
            for block in 0..BLOCKS {
                // Half of the changed accounts are new to the overlay.
                let mut block_changes: Vec<(Address, (Hash, Hash))> = (0..changes as u64)
                    .map(|i| {
                        let index = if i % 2 == 0 {
                            (block * 7919 + i * 104_729) % entries as u64
                        } else {
                            entries as u64 + block * changes as u64 + i
                        };
                        (address(index), (hash_of(index), hash_of(index + block + 1)))
                    })
                    .collect();
                block_changes.sort();
                block_changes.dedup_by_key(|(address, _)| *address);
                let start = std::time::Instant::now();
                if blocks_on_pool {
                    update_overlays(&mut state, &block_changes);
                } else {
                    for generation in &mut state {
                        apply_block_changes_batch(
                            &mut generation.overlay,
                            &block_changes,
                            &Sequential,
                            usize::MAX,
                        );
                    }
                }
                for generation in &state {
                    std::hint::black_box(generation.overlay.root());
                }
                total += start.elapsed();
            }
            println!(
                "{entries} entries, {changes} changes, {generations} generation(s), {}: {:.2} ms \
                 per block",
                if blocks_on_pool {
                    "overlay threads"
                } else {
                    "one thread"
                },
                total.as_secs_f64() * 1000.0 / BLOCKS as f64,
            );
        }
    }
}
