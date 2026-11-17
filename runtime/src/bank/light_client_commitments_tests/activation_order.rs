//! Activation order of the six light-client commitment features.
//!
//! A feature activates at the first epoch boundary after it is requested, and
//! nothing ties the activation of one feature to another's, so every order
//! must be well defined. A companion feature (slots 1 to 5) takes effect only
//! in blocks where Block Commitment Root is in effect, and Accounts State Root
//! only where Accounts Delta Root is in effect too; until then its slot is 32
//! zero bytes. The epoch boundary of an epoch and the accounts state
//! generation of an epoch exist only if their feature was in effect in the
//! first block of that epoch on the chain.
//!
//! Every block of these chains is checked against a reference vector built
//! from the raw feature set of each bank and from account scans (see
//! [`Chain::expected_vector`]), not from the commitment state of the banks.

use {
    super::*,
    crate::{
        bank_forks::BankForks,
        runtime_config::RuntimeConfig,
        serde_snapshot::{self, CommitmentExtraFields},
        snapshot_bank_utils,
        snapshot_utils::create_tmp_accounts_dir_for_tests,
    },
    agave_snapshots::snapshot_config::SnapshotConfig,
    solana_accounts_db::accounts_db::ACCOUNTS_DB_CONFIG_FOR_TESTING,
    std::sync::RwLock,
    tempfile::TempDir,
};

/// The six features, in commitment slot order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Draft {
    BlockCommitmentRoot,
    TransactionReceiptTree,
    AccountsDeltaRoot,
    EpochValidatorSetRoot,
    EpochBoundaryCommitment,
    AccountsStateRoot,
}

use Draft::*;

const COMPANIONS: [Draft; 5] = [
    TransactionReceiptTree,
    AccountsDeltaRoot,
    EpochValidatorSetRoot,
    EpochBoundaryCommitment,
    AccountsStateRoot,
];

impl Draft {
    fn id(self) -> Pubkey {
        match self {
            BlockCommitmentRoot => feature_set::block_commitment_root::id(),
            TransactionReceiptTree => feature_set::transaction_receipt_tree::id(),
            AccountsDeltaRoot => feature_set::accounts_delta_root::id(),
            EpochValidatorSetRoot => feature_set::epoch_validator_set_root::id(),
            EpochBoundaryCommitment => feature_set::epoch_boundary_commitment::id(),
            AccountsStateRoot => feature_set::accounts_state_root::id(),
        }
    }
}

fn ids(drafts: &[Draft]) -> Vec<Pubkey> {
    drafts.iter().map(|draft| draft.id()).collect()
}

/// Whether `draft` is in effect for `bank`, from the raw feature set: its own
/// feature and every feature it depends on are active.
fn in_effect(bank: &Bank, draft: Draft) -> bool {
    let active = |draft: Draft| bank.feature_set.is_active(&draft.id());
    match draft {
        BlockCommitmentRoot => active(BlockCommitmentRoot),
        AccountsStateRoot => active(AccountsStateRoot) && in_effect(bank, AccountsDeltaRoot),
        _ => active(draft) && active(BlockCommitmentRoot),
    }
}

/// Stores a pending feature account for `draft`: the feature activates at the
/// next epoch boundary.
fn request_feature(bank: &Bank, draft: Draft) {
    assert!(!bank.feature_set.is_active(&draft.id()));
    let lamports = bank
        .get_minimum_balance_for_rent_exemption(Feature::size_of())
        .max(1);
    let account = feature::create_account(&Feature { activated_at: None }, lamports);
    bank.store_account_and_update_capitalization(&draft.id(), &account);
}

fn block_id_of(slot: Slot) -> BlockId {
    BlockId::from(hashv(&[b"block id", &slot.to_le_bytes()]))
}

/// Commits two system transfers from the mint to addresses derived from the
/// slot, the same on every chain, and returns their expected receipts.
fn commit_transfers(bank: &Bank, mint_keypair: &Keypair) -> Vec<TransactionReceipt> {
    let blockhash = bank.last_blockhash();
    (0..2u8)
        .map(|index| {
            let to = Pubkey::new_from_array(
                hashv(&[b"recipient", &bank.slot().to_le_bytes(), &[index]]).to_bytes(),
            );
            let tx = system_transaction::transfer(
                mint_keypair,
                &to,
                1_000_000 + u64::from(index),
                blockhash,
            );
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
            TransactionReceipt::success(
                tx.signatures[0].into(),
                message_hash(&tx),
                LAMPORTS_PER_SIGNATURE,
                instruction_trace_root(&records),
                empty_events_root(),
            )
        })
        .collect()
}

/// A frozen bank of a [`Chain`], with what the reference needs.
struct Block {
    bank: Arc<Bank>,
    /// Account hashes after the block, from a scan.
    state: BTreeMap<Pubkey, Hash>,
    /// Expected receipts of the block's transactions, in block order.
    receipts: Vec<TransactionReceipt>,
}

/// A linear chain of frozen banks, each checked against the reference when
/// it is added.
struct Chain {
    mint_keypair: Keypair,
    bank_forks: Option<Arc<RwLock<BankForks>>>,
    /// Features removed from the feature set of every bank right after it is
    /// created, leaving their accounts untouched: the banks behave as if the
    /// features were inactive. Such a chain is not checked.
    masked: Vec<Pubkey>,
    blocks: Vec<Block>,
}

impl Chain {
    fn new(genesis_config: &GenesisConfig, mint_keypair: &Keypair, with_forks: bool) -> Self {
        Self::build(genesis_config, mint_keypair, with_forks, vec![])
    }

    fn masked(genesis_config: &GenesisConfig, mint_keypair: &Keypair, masked: &[Draft]) -> Self {
        Self::build(genesis_config, mint_keypair, false, ids(masked))
    }

    fn build(
        genesis_config: &GenesisConfig,
        mint_keypair: &Keypair,
        with_forks: bool,
        masked: Vec<Pubkey>,
    ) -> Self {
        let mut bank0 = Bank::new_for_tests(genesis_config);
        for id in &masked {
            bank0.deactivate_feature(id);
        }
        let (bank0, bank_forks) = if with_forks {
            let (bank0, bank_forks) = bank0.wrap_with_bank_forks_for_tests();
            (bank0, Some(bank_forks))
        } else {
            (Arc::new(bank0), None)
        };
        let mut chain = Self {
            mint_keypair: mint_keypair.insecure_clone(),
            bank_forks,
            masked,
            blocks: vec![],
        };
        chain.finish(bank0, vec![]);
        chain
    }

    fn last(&self) -> &Arc<Bank> {
        &self.blocks.last().unwrap().bank
    }

    fn bank(&self, slot: Slot) -> &Arc<Bank> {
        &self
            .blocks
            .iter()
            .find(|block| block.bank.slot() == slot)
            .unwrap()
            .bank
    }

    /// Adds a child of the last block at `slot`, with two transfers.
    fn push(&mut self, slot: Slot) -> Arc<Bank> {
        self.push_with(slot, |_| {}, |_| {})
    }

    /// Like [`Self::push`]; `modify` is applied to the new bank before it is
    /// shared (to change its feature set), `before_freeze` before it is
    /// frozen (to request a feature).
    fn push_with(
        &mut self,
        slot: Slot,
        modify: impl FnOnce(&mut Bank),
        before_freeze: impl FnOnce(&Bank),
    ) -> Arc<Bank> {
        let parent = self.last().clone();
        let mut bank = Bank::new_from_parent(parent.clone(), *parent.leader(), slot);
        for id in &self.masked {
            bank.deactivate_feature(id);
        }
        modify(&mut bank);
        let bank = match &self.bank_forks {
            Some(bank_forks) => bank_forks
                .write()
                .unwrap()
                .insert(bank)
                .clone_without_scheduler(),
            None => Arc::new(bank),
        };
        let receipts = commit_transfers(&bank, &self.mint_keypair);
        before_freeze(&bank);
        self.finish(bank.clone(), receipts);
        bank
    }

    fn finish(&mut self, bank: Arc<Bank>, receipts: Vec<TransactionReceipt>) {
        goto_end_of_slot(bank.clone());
        bank.freeze();
        bank.set_block_id(Some(block_id_of(bank.slot())));
        self.blocks.push(Block {
            state: scanned_state(&bank),
            bank,
            receipts,
        });
        if self.masked.is_empty() {
            self.check(self.blocks.len() - 1);
        }
    }

    /// `H(B)` of block `index`: the anchor leaf of the parent of the first
    /// block with Block Commitment Root in effect (none for the genesis
    /// block), then the history leaves of the blocks from that one to the
    /// parent of `index`.
    fn history(&self, index: usize) -> Vec<[u8; HISTORY_LEAF_LEN]> {
        let first = (0..=index)
            .find(|&i| in_effect(&self.blocks[i].bank, BlockCommitmentRoot))
            .expect("Block Commitment Root is in effect");
        let mut history = vec![];
        if let Some(parent) = first.checked_sub(1) {
            let parent = &self.blocks[parent].bank;
            history.push(anchor_leaf(
                parent.slot(),
                &parent.block_id().unwrap().to_hash(),
            ));
        }
        history.extend(self.blocks[first..index].iter().map(|b| leaf_of(&b.bank)));
        history
    }

    /// The index of the first block of `epoch` on the chain up to `index`.
    fn first_of_epoch(&self, index: usize, epoch: Epoch) -> Option<usize> {
        (0..=index).find(|&i| self.blocks[i].bank.epoch() == epoch)
    }

    /// The commitment vector of block `index` from the drafts, or `None` if
    /// Block Commitment Root is not in effect for it.
    fn expected_vector(&self, index: usize) -> Option<CommitmentVector> {
        let block = &self.blocks[index];
        let bank = &block.bank;
        if !in_effect(bank, BlockCommitmentRoot) {
            return None;
        }
        let empty = BTreeMap::new();
        let state_before = |i: usize| i.checked_sub(1).map_or(&empty, |p| &self.blocks[p].state);
        let mut vector = [INACTIVE_COMMITMENT_SLOT; COMMITMENT_VECTOR_LEN];

        // Slot 0.
        vector[0] = compute_root(TreeId::BLOCK_HISTORY, &self.history(index));

        // Slot 1: the receipts of the block.
        if in_effect(bank, TransactionReceiptTree) {
            vector[1] = receipts_root(&block.receipts);
        }

        // Slot 2: the accounts whose state differs from the parent's.
        if in_effect(bank, AccountsDeltaRoot) {
            let before = state_before(index);
            let after = &block.state;
            let changed: BTreeSet<_> = before
                .keys()
                .chain(after.keys())
                .filter(|address| before.get(*address) != after.get(*address))
                .collect();
            let leaves: Vec<Vec<u8>> = changed
                .into_iter()
                .map(|address| {
                    let hash = after.get(address).copied().unwrap_or_default();
                    [address.as_ref(), hash.as_ref()].concat()
                })
                .collect();
            vector[2] = compute_root(TreeId::ACCOUNTS_DELTA, &leaves);
        }

        // Slot 3: the signer set of the next epoch.
        if in_effect(bank, EpochValidatorSetRoot) {
            let next_epoch = bank.epoch() + 1;
            if let Some(signers) = bank.validator_set_signers(next_epoch) {
                vector[3] = validator_set_root(next_epoch, &signers).unwrap();
            }
        }

        // Slot 4: zero unless the feature was in effect in `F`, the first
        // block of the epoch, which must have a parent `P`.
        if in_effect(bank, EpochBoundaryCommitment) {
            let first = self.first_of_epoch(index, bank.epoch()).unwrap();
            if first > 0 && in_effect(&self.blocks[first].bank, EpochBoundaryCommitment) {
                let f = &self.blocks[first].bank;
                let p = &self.blocks[first - 1].bank;
                let boundary = EpochBoundary {
                    epoch: f.epoch(),
                    first_slot: f.slot(),
                    parent_slot: p.slot(),
                    parent_block_id: p.block_id().unwrap().to_hash(),
                    parent_commitment_root: if in_effect(p, BlockCommitmentRoot) {
                        p.commitment_root().unwrap()
                    } else {
                        Hash::default()
                    },
                    history_len: self.history(first).len() as u64,
                };
                vector[4] = boundary.hash();
            }
        }

        // Slot 5: generation `G` (`E - 1` before the switch slot of `E`, `E`
        // from it), which exists if its epoch is long enough, has a block on
        // the chain, and the feature was in effect in its first block.
        if in_effect(bank, AccountsStateRoot) {
            let schedule = bank.epoch_schedule();
            let epoch = bank.epoch();
            let switch_slot =
                schedule.get_first_slot_in_epoch(epoch) + schedule.get_slots_in_epoch(epoch) / 2;
            let generation = if bank.slot() < switch_slot {
                epoch.checked_sub(1)
            } else {
                Some(epoch)
            };
            if let Some(generation) = generation
                && schedule.get_slots_in_epoch(generation) >= 64
                && let Some(first) = self.first_of_epoch(index, generation)
                && in_effect(&self.blocks[first].bank, AccountsStateRoot)
            {
                let mut states = vec![state_before(first)];
                states.extend(self.blocks[first..=index].iter().map(|b| &b.state));
                vector[5] = reference_accounts_state_root(generation, &states);
            }
        }
        Some(CommitmentVector(vector))
    }

    /// Checks block `index` against [`Self::expected_vector`].
    fn check(&self, index: usize) {
        let bank = &self.blocks[index].bank;
        let slot = bank.slot();
        assert_eq!(bank.commitment_failure(), None, "slot {slot}");
        // The bank's own predicates agree with the drafts.
        assert_eq!(
            bank.is_block_commitment_root_active(),
            in_effect(bank, BlockCommitmentRoot)
        );
        assert_eq!(
            bank.is_transaction_receipt_tree_active(),
            in_effect(bank, TransactionReceiptTree)
        );
        assert_eq!(
            bank.is_accounts_delta_root_active(),
            in_effect(bank, AccountsDeltaRoot)
        );
        assert_eq!(
            bank.is_epoch_validator_set_root_active(),
            in_effect(bank, EpochValidatorSetRoot)
        );
        assert_eq!(
            bank.is_epoch_boundary_commitment_active(),
            in_effect(bank, EpochBoundaryCommitment)
        );
        assert_eq!(
            bank.is_accounts_state_root_active(),
            in_effect(bank, AccountsStateRoot)
        );
        match self.expected_vector(index) {
            None => assert_no_commitment(bank),
            Some(expected) => {
                let vector = bank.commitment_vector().expect("bank has a vector");
                for i in 0..COMMITMENT_VECTOR_LEN {
                    assert_eq!(vector.0[i], expected.0[i], "slot {i} of bank {slot}");
                }
                let commitment = bank.block_commitment().unwrap();
                assert_eq!(commitment.commitment_root, expected.root());
                assert_eq!(
                    commitment.block_history.leaf_count(),
                    self.history(index).len() as u64
                );
                bank.set_expected_commitment_root(expected.root());
                assert_eq!(bank.verify_commitment_root(), Ok(()));
                // A companion that is not in effect has no state either.
                if !in_effect(bank, AccountsDeltaRoot) {
                    assert_eq!(bank.accounts_delta(), None);
                }
                if !in_effect(bank, TransactionReceiptTree) {
                    assert_eq!(bank.receipts_root(), None);
                }
                if !in_effect(bank, EpochBoundaryCommitment) {
                    assert_eq!(bank.epoch_boundary(), None);
                }
                if !in_effect(bank, AccountsStateRoot) {
                    assert!(bank.accounts_state().is_none());
                }
            }
        }
    }

    fn slot_value(&self, slot: Slot, commitment_slot: CommitmentSlot) -> Hash {
        self.bank(slot)
            .commitment_vector()
            .unwrap()
            .get(commitment_slot)
    }
}

/// A frozen bank with Block Commitment Root not in effect: no state of any of
/// the six features, a V1 footer (no commitment root to verify), and no
/// commitment field in its snapshot.
fn assert_no_commitment(bank: &Bank) {
    let slot = bank.slot();
    assert!(!bank.is_block_commitment_root_active(), "slot {slot}");
    assert_eq!(bank.block_commitment(), None, "slot {slot}");
    assert_eq!(bank.commitment_root(), None);
    assert_eq!(bank.commitment_vector(), None);
    assert_eq!(bank.verify_commitment_root(), Ok(()));
    assert_eq!(bank.receipts_root(), None);
    assert!(bank.transaction_receipt_leaf_hashes().unwrap().is_empty());
    assert_eq!(bank.accounts_delta(), None);
    assert_eq!(bank.epoch_boundary(), None);
    assert!(bank.accounts_state().is_none());
    let fields = bank.get_fields_to_serialize();
    assert_eq!(fields.block_commitment, None);
    assert_eq!(fields.epoch_boundary, None);
    assert_eq!(fields.accounts_state, None);
}

/// One generation: epoch, base block, overlay root and base root.
type GenerationSummary = (Epoch, Option<(Slot, Hash)>, Hash, Result<Hash, String>);

/// The generations of a bank's accounts state, waiting for their base roots.
/// The base root of a bank restored from a snapshot that does not carry it
/// is only known once the bank has a child.
fn accounts_state_summary(bank: &Bank) -> Option<Vec<GenerationSummary>> {
    bank.accounts_state().map(|state| {
        state
            .generations()
            .iter()
            .map(|generation| {
                (
                    generation.epoch(),
                    generation.base().id().parent,
                    generation.overlay().root(),
                    generation.base().wait_root(),
                )
            })
            .collect()
    })
}

fn snapshot_bytes(bank: &Bank) -> Vec<u8> {
    let mut bytes = vec![];
    serde_snapshot::bank_to_stream(
        &mut std::io::BufWriter::new(std::io::Cursor::new(&mut bytes)),
        bank,
    )
    .unwrap();
    bytes
}

/// Each companion, alone or with the others, active from genesis while Block
/// Commitment Root is not: every block is identical to a block with the
/// companions inactive (same bank hash and snapshot bank fields), has no
/// commitment root, and takes a V1 footer.
#[test]
fn test_companions_without_base_have_no_effect() {
    let mut cases: Vec<Vec<Draft>> = COMPANIONS.iter().map(|draft| vec![*draft]).collect();
    cases.push(vec![AccountsDeltaRoot, AccountsStateRoot]);
    cases.push(COMPANIONS.to_vec());
    for companions in cases {
        let (genesis_config, mint_keypair) =
            genesis_with(SLOTS_PER_LONG_EPOCH, &ids(&companions), &[]);
        let mut chain = Chain::new(&genesis_config, &mint_keypair, false);
        // The same accounts, with the companions masked out of the feature
        // set of every bank.
        let mut reference = Chain::masked(&genesis_config, &mint_keypair, &companions);
        let mut pairs = vec![(chain.last().clone(), reference.last().clone())];
        // Two epoch boundaries and the switch slots of epochs 1 and 2.
        for slot in [1, 63, 65, 96, 100, 128, 160, 170] {
            pairs.push((chain.push(slot), reference.push(slot)));
        }
        for (bank, reference_bank) in pairs {
            for draft in &companions {
                assert!(bank.feature_set.is_active(&draft.id()));
                assert!(!reference_bank.feature_set.is_active(&draft.id()));
            }
            assert_no_commitment(&bank);
            assert_no_commitment(&reference_bank);
            assert_eq!(
                bank.hash(),
                reference_bank.hash(),
                "{companions:?} slot {}",
                bank.slot()
            );
            assert_eq!(
                bank.collects_accounts_delta(),
                reference_bank.collects_accounts_delta()
            );
            // No commitment field is written: the snapshot has the format of
            // a bank with the six features inactive. (The blockhash queue is
            // a hash map, so the bytes of two banks only compare in size.)
            let fields = bank.get_fields_to_serialize();
            assert_eq!(
                CommitmentExtraFields::new(&fields),
                CommitmentExtraFields::default()
            );
            let bytes = snapshot_bytes(&bank);
            assert_eq!(bytes.len(), snapshot_bytes(&reference_bank).len());
            let (read, _) =
                serde_snapshot::fields_from_stream(&mut std::io::BufReader::new(&bytes[..]))
                    .unwrap();
            assert_eq!(read.block_commitment, None);
            assert_eq!(read.epoch_boundary, None);
            assert!(read.accounts_state.is_none());
        }
    }
}

/// Block Commitment Root active from genesis, a companion requested in
/// epoch 1 and active from epoch 2: its slot is zero before, the drafts'
/// value from the first block of epoch 2, and the block history continues
/// with one leaf per block.
#[test]
fn test_companion_activated_after_base() {
    for (at_genesis, later) in [
        (vec![], vec![TransactionReceiptTree]),
        (vec![], vec![AccountsDeltaRoot]),
        (vec![], vec![EpochValidatorSetRoot]),
        (vec![], vec![EpochBoundaryCommitment]),
        (vec![], vec![AccountsDeltaRoot, AccountsStateRoot]),
        (vec![AccountsDeltaRoot], vec![AccountsStateRoot]),
        (vec![], COMPANIONS.to_vec()),
    ] {
        let mut features = ids(&at_genesis);
        features.push(BlockCommitmentRoot.id());
        let (genesis_config, mint_keypair) = genesis_with(SLOTS_PER_LONG_EPOCH, &features, &[]);
        let mut chain = Chain::new(&genesis_config, &mint_keypair, false);
        for slot in [1, 40, 65] {
            chain.push(slot);
        }
        chain.push_with(
            70,
            |_| {},
            |bank| {
                for draft in &later {
                    request_feature(bank, *draft);
                }
            },
        );
        for slot in [96, 100, 128, 130, 160, 170, 192, 224, 230] {
            chain.push(slot);
        }
        for (index, block) in chain.blocks.iter().enumerate() {
            let bank = &block.bank;
            // One history leaf per earlier block: the history is not
            // restarted by the activation.
            assert_eq!(
                bank.block_commitment().unwrap().block_history.leaf_count(),
                index as u64
            );
            for draft in &later {
                assert_eq!(in_effect(bank, *draft), bank.slot() >= 128, "{draft:?}");
            }
        }
        // The value of each companion is zero before activation and set from
        // the activation block on (slot 5 from the switch slot of epoch 2).
        for draft in &later {
            let (commitment_slot, from) = match draft {
                TransactionReceiptTree => (CommitmentSlot::ReceiptsRoot, 128),
                AccountsDeltaRoot => (CommitmentSlot::AccountsDeltaRoot, 128),
                EpochValidatorSetRoot => (CommitmentSlot::ValidatorSetRoot, 128),
                EpochBoundaryCommitment => (CommitmentSlot::EpochBoundary, 128),
                AccountsStateRoot => (CommitmentSlot::AccountsStateRoot, 160),
                BlockCommitmentRoot => unreachable!(),
            };
            for block in &chain.blocks {
                let slot = block.bank.slot();
                assert_eq!(
                    chain.slot_value(slot, commitment_slot) == INACTIVE_COMMITMENT_SLOT,
                    slot < from,
                    "{draft:?} at slot {slot}"
                );
            }
        }
        // The epoch boundary of epoch 2 has `P` (bank 100) with its
        // commitment root.
        if later.contains(&EpochBoundaryCommitment) {
            let boundary = chain.bank(128).epoch_boundary().unwrap();
            assert_eq!(boundary.parent_slot, 100);
            assert_eq!(
                Some(boundary.parent_commitment_root),
                chain.bank(100).commitment_root()
            );
            assert_eq!(chain.bank(130).epoch_boundary(), Some(boundary));
        }
        // Generation 1 does not exist; generation 2 has bank 100 as `P`.
        if later.contains(&AccountsStateRoot) {
            let state = chain.bank(128).accounts_state().unwrap();
            let held: Vec<_> = state.generations().iter().map(|g| g.epoch()).collect();
            assert_eq!(held, vec![2]);
            assert_eq!(
                state.generation(2).unwrap().base().id().parent,
                Some((100, chain.bank(100).hash()))
            );
        }
    }
}

/// Companions active from genesis and Block Commitment Root requested in
/// epoch 1: no block has commitment state before epoch 2, and every
/// companion starts in the first block of epoch 2, as if all had activated
/// together there.
#[test]
fn test_companions_requested_before_base() {
    let mut cases: Vec<Vec<Draft>> = COMPANIONS.iter().map(|draft| vec![*draft]).collect();
    cases.push(vec![AccountsDeltaRoot, AccountsStateRoot]);
    cases.push(COMPANIONS.to_vec());
    for companions in cases {
        let (genesis_config, mint_keypair) =
            genesis_with(SLOTS_PER_LONG_EPOCH, &ids(&companions), &[]);
        let mut chain = Chain::new(&genesis_config, &mint_keypair, false);
        for slot in [1, 40, 65] {
            chain.push(slot);
        }
        chain.push_with(
            70,
            |_| {},
            |bank| request_feature(bank, BlockCommitmentRoot),
        );
        for slot in [96, 100, 128, 130, 160, 170, 192, 224] {
            chain.push(slot);
        }
        for block in &chain.blocks {
            let slot = block.bank.slot();
            assert_eq!(block.bank.block_commitment().is_some(), slot >= 128);
            for draft in &companions {
                // Accounts State Root also needs Accounts Delta Root.
                let effective =
                    *draft != AccountsStateRoot || companions.contains(&AccountsDeltaRoot);
                assert_eq!(
                    in_effect(&block.bank, *draft),
                    effective && slot >= 128,
                    "{draft:?}"
                );
            }
        }
        // The first block with the base active has the anchor leaf of `P`
        // as its whole history.
        let bank128 = chain.bank(128);
        assert_eq!(
            bank128
                .block_commitment()
                .unwrap()
                .block_history
                .leaf_count(),
            1
        );
        if companions.contains(&EpochBoundaryCommitment) {
            let boundary = bank128.epoch_boundary().unwrap();
            assert_eq!(boundary.parent_slot, 100);
            assert_eq!(boundary.parent_commitment_root, Hash::default());
            assert_eq!(boundary.history_len, 1);
            assert_ne!(
                chain.slot_value(128, CommitmentSlot::EpochBoundary),
                INACTIVE_COMMITMENT_SLOT
            );
        }
        if companions.contains(&AccountsStateRoot) && companions.contains(&AccountsDeltaRoot) {
            for slot in [128, 130] {
                assert_eq!(
                    chain.slot_value(slot, CommitmentSlot::AccountsStateRoot),
                    INACTIVE_COMMITMENT_SLOT
                );
            }
            for slot in [160, 170, 192, 224] {
                assert_ne!(
                    chain.slot_value(slot, CommitmentSlot::AccountsStateRoot),
                    INACTIVE_COMMITMENT_SLOT
                );
            }
        }
    }
}

/// Accounts State Root active from genesis (with Block Commitment Root) and
/// Accounts Delta Root requested in epoch 1: Accounts State Root takes effect
/// in epoch 2. Generation 1 does not exist (the feature was not in effect in
/// the first block of epoch 1), so slot 5 is zero until the switch slot of
/// epoch 2; generation 2 starts with the first block of epoch 2, its base by
/// scan of the last block of epoch 1. Also with the first half of epoch 2
/// skipped, so that its first block is past the switch slot and needs the
/// base of its own parent at once.
#[test]
fn test_accounts_state_before_accounts_delta() {
    for (epoch2, first) in [(vec![128, 130, 160, 170], 128), (vec![170, 180], 170)] {
        let (genesis_config, mint_keypair) = genesis_with(
            SLOTS_PER_LONG_EPOCH,
            &ids(&[BlockCommitmentRoot, AccountsStateRoot]),
            &[],
        );
        let mut chain = Chain::new(&genesis_config, &mint_keypair, false);
        for slot in [1, 40, 65] {
            chain.push(slot);
        }
        chain.push_with(70, |_| {}, |bank| request_feature(bank, AccountsDeltaRoot));
        for slot in [96, 100]
            .into_iter()
            .chain(epoch2)
            .chain([192, 200, 224, 230])
        {
            chain.push(slot);
        }
        for block in &chain.blocks {
            let bank = &block.bank;
            let slot = bank.slot();
            assert!(bank.feature_set.is_active(&AccountsStateRoot.id()));
            assert_eq!(bank.is_accounts_state_root_active(), slot >= first);
            assert_eq!(bank.accounts_state().is_some(), slot >= first);
            assert_eq!(
                chain.slot_value(slot, CommitmentSlot::AccountsStateRoot)
                    == INACTIVE_COMMITMENT_SLOT,
                slot < 160,
                "slot {slot}"
            );
        }
        let held = |slot| -> Vec<Epoch> {
            chain
                .bank(slot)
                .accounts_state()
                .unwrap()
                .generations()
                .iter()
                .map(|g| g.epoch())
                .collect()
        };
        assert_eq!(held(first), vec![2]);
        assert_eq!(held(170), vec![2]);
        // Epoch 3 before its switch slot uses generation 2, and holds 3.
        assert_eq!(held(192), vec![2, 3]);
        assert_eq!(held(230), vec![3]);
    }
}

/// Epoch Boundary Commitment taking effect after the first block `F` of an
/// epoch was processed (which feature activation at an epoch boundary does
/// not allow; only a test can do it): slot 4 is zero for the rest of that
/// epoch, and correct from the first block of the next one. Both for the
/// feature itself and for Block Commitment Root taking effect late.
#[test]
fn test_epoch_boundary_first_block_without_feature() {
    for (at_genesis, late) in [
        (BlockCommitmentRoot, EpochBoundaryCommitment),
        (EpochBoundaryCommitment, BlockCommitmentRoot),
    ] {
        let (genesis_config, mint_keypair) =
            genesis_with(SLOTS_PER_LONG_EPOCH, &[at_genesis.id()], &[]);
        let mut chain = Chain::new(&genesis_config, &mint_keypair, false);
        for slot in [1, 65] {
            chain.push(slot);
        }
        // Bank 65 is `F` of epoch 1, processed without the feature.
        assert!(!in_effect(chain.bank(65), EpochBoundaryCommitment));
        chain.push_with(70, |bank| bank.activate_feature(&late.id()), |_| {});
        for slot in [96, 100, 128, 130] {
            chain.push(slot);
        }
        for slot in [70, 96, 100] {
            let bank = chain.bank(slot);
            assert!(bank.is_epoch_boundary_commitment_active());
            assert_eq!(bank.epoch_boundary(), None);
            assert_eq!(
                chain.slot_value(slot, CommitmentSlot::EpochBoundary),
                INACTIVE_COMMITMENT_SLOT
            );
        }
        let boundary = chain.bank(128).epoch_boundary().unwrap();
        assert_eq!(boundary.epoch, 2);
        assert_eq!(boundary.parent_slot, 100);
        assert_eq!(
            Some(boundary.parent_commitment_root),
            chain.bank(100).commitment_root()
        );
        assert_eq!(chain.bank(130).epoch_boundary(), Some(boundary));
        assert_eq!(
            chain.slot_value(130, CommitmentSlot::EpochBoundary),
            boundary.hash()
        );
    }
}

/// Epoch Validator Set Root activated two epochs after Block Commitment Root:
/// slot 3 is zero before, and `validator_set_root(E + 1)` from the first
/// block of the activation epoch on.
#[test]
fn test_validator_set_root_activated_after_base() {
    let (genesis_config, mint_keypair) =
        genesis_with(SLOTS_PER_LONG_EPOCH, &[BlockCommitmentRoot.id()], &[]);
    let mut chain = Chain::new(&genesis_config, &mint_keypair, false);
    for slot in [1, 65] {
        chain.push(slot);
    }
    chain.push_with(
        70,
        |_| {},
        |bank| request_feature(bank, EpochValidatorSetRoot),
    );
    for slot in [100, 128, 130, 192] {
        chain.push(slot);
    }
    for slot in [0, 1, 65, 70, 100] {
        assert_eq!(
            chain.slot_value(slot, CommitmentSlot::ValidatorSetRoot),
            INACTIVE_COMMITMENT_SLOT
        );
    }
    for (slot, next_epoch) in [(128, 3), (130, 3), (192, 4)] {
        let bank = chain.bank(slot);
        let signers = bank.validator_set_signers(next_epoch).unwrap();
        assert_eq!(signers.len(), 3);
        assert_eq!(
            chain.slot_value(slot, CommitmentSlot::ValidatorSetRoot),
            validator_set_root(next_epoch, &signers).unwrap()
        );
    }
}

/// A snapshot taken between two activations: the bank restored from it and
/// its descendants are identical to the original chain (bank hashes,
/// commitment vectors and state) across the later activation, which the
/// original chain shows to be correct.
#[test]
fn test_snapshot_between_activations() {
    // (active at genesis, requested at slot 70 and active from epoch 2,
    // snapshot slot)
    for (at_genesis, requested, snapshot_slot) in [
        (vec![BlockCommitmentRoot], COMPANIONS.to_vec(), 100),
        (COMPANIONS.to_vec(), vec![BlockCommitmentRoot], 100),
        (
            vec![BlockCommitmentRoot, AccountsStateRoot],
            vec![AccountsDeltaRoot],
            100,
        ),
        // Snapshots of the first block of the activation epoch, and of a
        // later block before its switch slot: the accounts state has only
        // generation 2, without a generation 1.
        (
            vec![BlockCommitmentRoot, AccountsStateRoot],
            vec![AccountsDeltaRoot],
            128,
        ),
        (
            vec![BlockCommitmentRoot, AccountsStateRoot],
            vec![AccountsDeltaRoot],
            140,
        ),
        (vec![BlockCommitmentRoot], COMPANIONS.to_vec(), 128),
    ] {
        let (genesis_config, mint_keypair) =
            genesis_with(SLOTS_PER_LONG_EPOCH, &ids(&at_genesis), &[]);
        let mut chain = Chain::new(&genesis_config, &mint_keypair, true);
        for slot in [1, 65] {
            chain.push(slot);
        }
        chain.push_with(
            70,
            |_| {},
            |bank| {
                for draft in &requested {
                    request_feature(bank, *draft);
                }
            },
        );
        for slot in [96, 128] {
            if slot < snapshot_slot {
                chain.push(slot);
            }
        }
        let snapshot_bank = chain.push(snapshot_slot);
        // The snapshot carries the commitment fields of the features in
        // effect for its bank, and no other.
        let fields = snapshot_bank.get_fields_to_serialize();
        assert_eq!(
            fields.block_commitment.is_some(),
            in_effect(&snapshot_bank, BlockCommitmentRoot)
        );
        assert_eq!(
            fields.epoch_boundary.is_some(),
            in_effect(&snapshot_bank, EpochBoundaryCommitment)
        );
        assert_eq!(
            fields.accounts_state.is_some(),
            in_effect(&snapshot_bank, AccountsStateRoot)
        );

        let (_tmp_dir, accounts_dir) = create_tmp_accounts_dir_for_tests();
        let bank_snapshots_dir = TempDir::new().unwrap();
        let full_snapshot_archives_dir = TempDir::new().unwrap();
        let incremental_snapshot_archives_dir = TempDir::new().unwrap();
        let snapshot_config = SnapshotConfig {
            full_snapshot_archives_dir: full_snapshot_archives_dir.path().to_path_buf(),
            incremental_snapshot_archives_dir: incremental_snapshot_archives_dir
                .path()
                .to_path_buf(),
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
            &genesis_config,
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
        assert_eq!(
            restored.block_commitment(),
            snapshot_bank.block_commitment()
        );
        assert_eq!(restored.epoch_boundary(), snapshot_bank.epoch_boundary());
        let without_base_roots = |bank: &Bank| {
            accounts_state_summary(bank).map(|generations| {
                generations
                    .into_iter()
                    .map(|(epoch, base, overlay_root, _)| (epoch, base, overlay_root))
                    .collect::<Vec<_>>()
            })
        };
        assert_eq!(
            without_base_roots(&restored),
            without_base_roots(&snapshot_bank)
        );
        let (mut restored_parent, restored_forks) = restored.wrap_with_bank_forks_for_tests();

        for slot in [128, 130, 160, 170, 192, 200, 224, 230]
            .into_iter()
            .filter(|slot| *slot > snapshot_slot)
        {
            let original = chain.push(slot);
            let copy = Bank::new_from_parent_with_bank_forks(
                &restored_forks,
                restored_parent.clone(),
                *restored_parent.leader(),
                slot,
            );
            commit_transfers(&copy, &mint_keypair);
            goto_end_of_slot(copy.clone());
            copy.freeze();
            copy.set_block_id(Some(block_id_of(slot)));
            assert_eq!(copy.commitment_failure(), None, "slot {slot}");
            assert_eq!(copy.hash(), original.hash(), "slot {slot}");
            assert_eq!(copy.commitment_vector(), original.commitment_vector());
            assert_eq!(copy.block_commitment(), original.block_commitment());
            assert_eq!(copy.epoch_boundary(), original.epoch_boundary());
            assert_eq!(
                accounts_state_summary(&copy),
                accounts_state_summary(&original)
            );
            for draft in &requested {
                assert!(in_effect(&copy, *draft), "{draft:?} at {slot}");
            }
            restored_parent = copy;
        }
    }
}
