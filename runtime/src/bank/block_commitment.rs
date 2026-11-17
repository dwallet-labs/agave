//! Block Commitment Root (draft SIMD): the 8-slot commitment vector, its root
//! (`commitment_root`, carried in footer version 2) and the block history
//! accumulator (slot 0).
//!
//! The vector is computed when the bank is frozen, right after the accounts
//! lattice hash is finished and the bank hash is computed, and only when the
//! `block_commitment_root` feature is active for the bank.
//!
//! # Extension points for the companion proposals
//!
//! Each of the commitment slots 1 to 4 is produced by exactly one method on
//! [`Bank`], listed below. Today each returns 32 zero bytes, which the SIMD
//! defines as the value of a slot whose defining feature is not active. A
//! follow-up implements its slot by filling in its method (gated by its own
//! feature, returning [`INACTIVE_COMMITMENT_SLOT`] while that feature is
//! inactive) and nothing else in the vector changes:
//!
//! | Slot | Method                                         | Proposal                  |
//! |------|------------------------------------------------|---------------------------|
//! | 1    | [`Bank::commitment_slot_receipts_root`]        | Transaction Receipt Tree  |
//! | 2    | [`Bank::commitment_slot_accounts_delta_root`]  | Accounts Delta Root       |
//! | 3    | [`Bank::commitment_slot_validator_set_root`]   | Epoch Validator Set Root  |
//! | 4    | [`Bank::commitment_slot_epoch_boundary`]       | Epoch Boundary Commitment |
//!
//! Slots 5 to 7 are reserved and always zero. Every slot method receives the
//! same [`CommitmentInputs`], which carries this block's history state and
//! parent (the epoch boundary commitment needs both).

use {
    super::Bank,
    agave_feature_set as feature_set,
    agave_votor_messages::consensus_message::BlockId,
    log::*,
    solana_clock::Slot,
    solana_hash::{HASH_BYTES, Hash},
    solana_merkle_tree::commitment_tree::{
        AppendOnlyFrontier, InclusionProof, TreeId, compute_root, make_inclusion_proof,
    },
};

/// Number of slots in the commitment vector.
pub const COMMITMENT_VECTOR_LEN: usize = 8;

/// Value of a commitment slot whose defining feature is not active, and of
/// every reserved slot: 32 zero bytes.
pub const INACTIVE_COMMITMENT_SLOT: Hash = Hash::new_from_array([0; HASH_BYTES]);

/// Size of a block history leaf: `u64_le(slot) || block_id || commitment_root`.
pub const HISTORY_LEAF_LEN: usize = 8 + HASH_BYTES + HASH_BYTES;

/// Index of each assigned slot of the commitment vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum CommitmentSlot {
    /// `block_history_root` (this proposal).
    BlockHistoryRoot = 0,
    /// `receipts_root` (Transaction Receipt Tree).
    ReceiptsRoot = 1,
    /// `accounts_delta_root` (Accounts Delta Root).
    AccountsDeltaRoot = 2,
    /// `validator_set_root` (Epoch Validator Set Root).
    ValidatorSetRoot = 3,
    /// `epoch_boundary` (Epoch Boundary Commitment).
    EpochBoundary = 4,
}

impl CommitmentSlot {
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// The eight 32-byte values committed to by `commitment_root`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CommitmentVector(pub [Hash; COMMITMENT_VECTOR_LEN]);

impl CommitmentVector {
    pub fn get(&self, slot: CommitmentSlot) -> Hash {
        self.0[slot.index()]
    }

    /// `commitment_root = root(0x00, [slot_0, ..., slot_7])`
    pub fn root(&self) -> Hash {
        compute_root(TreeId::COMMITMENT_VECTOR, &self.0)
    }

    /// Proof opening slot `index` against [`Self::root`] (three siblings).
    pub fn inclusion_proof(&self, index: usize) -> Option<InclusionProof> {
        make_inclusion_proof(TreeId::COMMITMENT_VECTOR, &self.0, index)
    }
}

/// `history_leaf(A) = u64_le(A.slot) || A.block_id || A.commitment_root`
pub fn history_leaf(slot: Slot, block_id: &Hash, commitment_root: &Hash) -> [u8; HISTORY_LEAF_LEN] {
    let mut leaf = [0u8; HISTORY_LEAF_LEN];
    leaf[..8].copy_from_slice(&slot.to_le_bytes());
    leaf[8..8 + HASH_BYTES].copy_from_slice(block_id.as_ref());
    leaf[8 + HASH_BYTES..].copy_from_slice(commitment_root.as_ref());
    leaf
}

/// Block commitment state of a frozen bank `B` (or of a snapshot bank).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockCommitment {
    /// Frontier of `H(B)`: the history leaves of all ancestors of `B` for
    /// which the feature was active, oldest first. `B` itself is not in it.
    pub block_history: AppendOnlyFrontier,
    /// `commitment_root` of `B`.
    pub commitment_root: Hash,
}

impl BlockCommitment {
    /// `block_history_root(B) = root(0x01, H(B))`
    pub fn block_history_root(&self) -> Hash {
        self.block_history.root(TreeId::BLOCK_HISTORY)
    }

    /// `H(child)`: `H(B)` with `history_leaf(B)` appended, where `B` is the
    /// bank this state belongs to, at `slot` with `block_id`.
    pub fn child_block_history(&self, slot: Slot, block_id: &Hash) -> AppendOnlyFrontier {
        let mut history = self.block_history.clone();
        history
            .append(
                TreeId::BLOCK_HISTORY,
                &history_leaf(slot, block_id, &self.commitment_root),
            )
            .expect("block history cannot hold 2^64 blocks");
        history
    }
}

/// Inputs available to every commitment slot when the vector is computed.
pub struct CommitmentInputs<'a> {
    /// `H(B)` of the bank being frozen.
    pub block_history: &'a AppendOnlyFrontier,
    /// Parent of the bank being frozen, if it still has one.
    pub parent: Option<&'a Bank>,
}

impl Bank {
    /// Whether the Block Commitment Root feature is active for this bank.
    pub fn is_block_commitment_root_active(&self) -> bool {
        self.feature_set
            .is_active(&feature_set::block_commitment_root::id())
    }

    /// Block commitment state, set when the bank is frozen with the feature
    /// active, or loaded from a snapshot.
    pub fn block_commitment(&self) -> Option<BlockCommitment> {
        self.block_commitment.read().unwrap().clone()
    }

    /// `commitment_root` of this bank, set when the bank is frozen with the
    /// feature active, or loaded from a snapshot.
    pub fn commitment_root(&self) -> Option<Hash> {
        self.block_commitment
            .read()
            .unwrap()
            .as_ref()
            .map(|commitment| commitment.commitment_root)
    }

    /// Records the commitment root found in this block's footer. It is
    /// compared with the computed one by [`Self::verify_commitment_root`].
    pub fn set_expected_commitment_root(&self, commitment_root: Hash) {
        *self.expected_commitment_root.write().unwrap() = Some(commitment_root);
    }

    /// The commitment root found in this block's footer, if any.
    pub fn expected_commitment_root(&self) -> Option<Hash> {
        *self.expected_commitment_root.read().unwrap()
    }

    /// Compares the commitment root from the footer, if any, with the one
    /// computed when the bank was frozen. Must be called on a frozen bank.
    ///
    /// Returns `Err((expected, computed))` on mismatch; `computed` is the
    /// zero hash if the bank has no commitment root. Replay marks the block
    /// dead on error, as for a bank hash mismatch.
    pub fn verify_commitment_root(&self) -> std::result::Result<(), (Hash, Hash)> {
        debug_assert!(self.is_frozen());
        let Some(expected) = self.expected_commitment_root() else {
            return Ok(());
        };
        let computed = self.commitment_root().unwrap_or_default();
        if expected == computed {
            Ok(())
        } else {
            Err((expected, computed))
        }
    }

    /// Computes the block commitment state of this bank. Called once, from
    /// `freeze()`, after the accounts lattice hash is finished.
    pub(super) fn compute_block_commitment(&self) -> Option<BlockCommitment> {
        if !self.is_block_commitment_root_active() {
            return None;
        }
        let parent = self.parent();
        let block_history = self.block_history_for_commitment(parent.as_deref());
        let vector = self.compute_commitment_vector(&CommitmentInputs {
            block_history: &block_history,
            parent: parent.as_deref(),
        });
        let commitment_root = vector.root();
        info!(
            "bank {} commitment_root: {commitment_root} block_history_len: {}",
            self.slot(),
            block_history.leaf_count(),
        );
        Some(BlockCommitment {
            block_history,
            commitment_root,
        })
    }

    /// `H(B)` for this bank `B`: `H(parent)` with `history_leaf(parent)`
    /// appended if the feature was active for the parent, empty otherwise
    /// (first block with the feature active, or no parent).
    fn block_history_for_commitment(&self, parent: Option<&Bank>) -> AppendOnlyFrontier {
        let Some(parent) = parent else {
            return AppendOnlyFrontier::new();
        };
        let Some(parent_commitment) = parent.block_commitment() else {
            return AppendOnlyFrontier::new();
        };
        let parent_block_id = parent.block_id().unwrap_or_else(|| {
            // Every frozen bank that a child is built on has a block id in a
            // running validator: replayed banks get it from the blockstore
            // before freezing, leader banks from broadcast once the last
            // shred is made, snapshot banks from the snapshot. Without it the
            // history leaf cannot be computed; a zero block id makes this
            // bank's commitment root differ from the cluster's, so its own
            // block is rejected by replay instead of being silently accepted.
            error!(
                "bank {} parent {} has no block id; using the zero block id in the block \
                 history leaf",
                self.slot(),
                parent.slot()
            );
            datapoint_error!(
                "bank-block_commitment_missing_parent_block_id",
                ("slot", self.slot(), i64),
                ("parent_slot", parent.slot(), i64),
            );
            BlockId::default()
        });
        parent_commitment.child_block_history(parent.slot(), &parent_block_id.to_hash())
    }

    /// Computes the commitment vector of this bank from its inputs.
    pub fn compute_commitment_vector(&self, inputs: &CommitmentInputs) -> CommitmentVector {
        let mut vector = [INACTIVE_COMMITMENT_SLOT; COMMITMENT_VECTOR_LEN];
        vector[CommitmentSlot::BlockHistoryRoot.index()] =
            inputs.block_history.root(TreeId::BLOCK_HISTORY);
        vector[CommitmentSlot::ReceiptsRoot.index()] = self.commitment_slot_receipts_root(inputs);
        vector[CommitmentSlot::AccountsDeltaRoot.index()] =
            self.commitment_slot_accounts_delta_root(inputs);
        vector[CommitmentSlot::ValidatorSetRoot.index()] =
            self.commitment_slot_validator_set_root(inputs);
        vector[CommitmentSlot::EpochBoundary.index()] = self.commitment_slot_epoch_boundary(inputs);
        // Slots 5..8 are reserved and stay zero.
        CommitmentVector(vector)
    }

    /// Commitment slot 1: `receipts_root` (Transaction Receipt Tree SIMD),
    /// [`INACTIVE_COMMITMENT_SLOT`] unless the `transaction_receipt_tree`
    /// feature is active for this bank. See `bank::transaction_receipts`.
    pub fn commitment_slot_receipts_root(&self, inputs: &CommitmentInputs) -> Hash {
        self.receipts_root_for_commitment(inputs.parent)
            .unwrap_or(INACTIVE_COMMITMENT_SLOT)
    }

    /// Commitment slot 2: `accounts_delta_root` (Accounts Delta Root SIMD).
    ///
    /// [`INACTIVE_COMMITMENT_SLOT`] unless the `accounts_delta_root` feature
    /// is active for this bank. The root is built in `freeze()` from the
    /// account hashes collected by the lattice hash, after the last account
    /// update of the bank (see [`super::accounts_delta`]).
    pub fn commitment_slot_accounts_delta_root(&self, _inputs: &CommitmentInputs) -> Hash {
        if !self.is_accounts_delta_root_active() {
            return INACTIVE_COMMITMENT_SLOT;
        }
        // Unset only if the bank did not collect from its first account
        // update, which `finish_accounts_delta()` reports.
        self.accounts_delta_root()
            .unwrap_or(INACTIVE_COMMITMENT_SLOT)
    }

    /// Commitment slot 3: `validator_set_root` (Epoch Validator Set Root SIMD).
    ///
    /// Extension point: not implemented yet, so the slot is inactive and
    /// zero. The implementation must return [`INACTIVE_COMMITMENT_SLOT`]
    /// unless its own feature is active for this bank.
    pub fn commitment_slot_validator_set_root(&self, _inputs: &CommitmentInputs) -> Hash {
        INACTIVE_COMMITMENT_SLOT
    }

    /// Commitment slot 4: `epoch_boundary` (Epoch Boundary Commitment SIMD).
    ///
    /// Extension point: not implemented yet, so the slot is inactive and
    /// zero. The implementation must return [`INACTIVE_COMMITMENT_SLOT`]
    /// unless its own feature is active for this bank. `inputs` carries the
    /// parent bank (`P` when this bank is the first of its epoch) and
    /// `H(B)`, whose leaf count is `history_len`.
    pub fn commitment_slot_epoch_boundary(&self, _inputs: &CommitmentInputs) -> Hash {
        INACTIVE_COMMITMENT_SLOT
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        crate::{
            bank::{SlotLeader, test_utils::goto_end_of_slot},
            genesis_utils::{activate_feature, create_genesis_config},
        },
        solana_merkle_tree::commitment_tree::{hash_leaf, hash_node, hash_root},
        std::sync::Arc,
    };

    fn new_bank(feature_active: bool) -> Arc<Bank> {
        let mut genesis = create_genesis_config(1_000_000_000).genesis_config;
        if feature_active {
            activate_feature(&mut genesis, feature_set::block_commitment_root::id());
        }
        Bank::new_for_tests(&genesis).into()
    }

    fn new_child(parent: &Arc<Bank>, slot: Slot) -> Arc<Bank> {
        let child = Bank::new_from_parent(parent.clone(), SlotLeader::default(), slot);
        Arc::new(child)
    }

    fn freeze_with_block_id(bank: &Arc<Bank>) -> Hash {
        goto_end_of_slot(bank.clone());
        bank.freeze();
        let block_id = Hash::new_unique();
        bank.set_block_id(Some(BlockId::from(block_id)));
        block_id
    }

    #[test]
    fn test_commitment_vector_root() {
        let mut vector = CommitmentVector::default();
        assert_eq!(
            vector.root(),
            hash_root(
                TreeId::COMMITMENT_VECTOR,
                8,
                &hash_node(
                    &hash_node(
                        &hash_node(
                            &hash_leaf(TreeId::COMMITMENT_VECTOR, &[0; 32]),
                            &hash_leaf(TreeId::COMMITMENT_VECTOR, &[0; 32])
                        ),
                        &hash_node(
                            &hash_leaf(TreeId::COMMITMENT_VECTOR, &[0; 32]),
                            &hash_leaf(TreeId::COMMITMENT_VECTOR, &[0; 32])
                        )
                    ),
                    &hash_node(
                        &hash_node(
                            &hash_leaf(TreeId::COMMITMENT_VECTOR, &[0; 32]),
                            &hash_leaf(TreeId::COMMITMENT_VECTOR, &[0; 32])
                        ),
                        &hash_node(
                            &hash_leaf(TreeId::COMMITMENT_VECTOR, &[0; 32]),
                            &hash_leaf(TreeId::COMMITMENT_VECTOR, &[0; 32])
                        )
                    )
                )
            )
        );
        vector.0[CommitmentSlot::EpochBoundary.index()] = Hash::new_unique();
        let root = vector.root();
        for i in 0..COMMITMENT_VECTOR_LEN {
            let proof = vector.inclusion_proof(i).unwrap();
            assert_eq!(proof.siblings.len(), 3);
            assert!(proof.verify(TreeId::COMMITMENT_VECTOR, vector.0[i].as_ref(), &root));
        }
    }

    #[test]
    fn test_history_leaf_layout() {
        let block_id = Hash::new_unique();
        let commitment_root = Hash::new_unique();
        let leaf = history_leaf(0x0102_0304_0506_0708, &block_id, &commitment_root);
        assert_eq!(leaf.len(), 72);
        assert_eq!(&leaf[..8], &0x0102_0304_0506_0708u64.to_le_bytes());
        assert_eq!(&leaf[8..40], block_id.as_ref());
        assert_eq!(&leaf[40..], commitment_root.as_ref());
    }

    #[test]
    fn test_inactive_feature_has_no_commitment() {
        let bank0 = new_bank(false);
        assert!(!bank0.is_block_commitment_root_active());
        bank0.freeze();
        assert_eq!(bank0.block_commitment(), None);
        assert_eq!(bank0.commitment_root(), None);
        // Without a footer value there is nothing to verify.
        assert_eq!(bank0.verify_commitment_root(), Ok(()));
    }

    #[test]
    fn test_block_history_chain() {
        let bank0 = new_bank(true);
        assert!(bank0.is_block_commitment_root_active());
        let block_id0 = freeze_with_block_id(&bank0);
        let commitment0 = bank0.block_commitment().unwrap();
        // First block with the feature active: empty history.
        assert_eq!(commitment0.block_history.leaf_count(), 0);
        let expected_vector0 = CommitmentVector({
            let mut v = [INACTIVE_COMMITMENT_SLOT; COMMITMENT_VECTOR_LEN];
            v[0] = AppendOnlyFrontier::new().root(TreeId::BLOCK_HISTORY);
            v
        });
        assert_eq!(commitment0.commitment_root, expected_vector0.root());
        assert_eq!(bank0.commitment_root(), Some(expected_vector0.root()));

        // Slot 1 is skipped; slot 2 builds on slot 0.
        let bank2 = new_child(&bank0, 2);
        let block_id2 = freeze_with_block_id(&bank2);
        let commitment2 = bank2.block_commitment().unwrap();
        let leaves = [history_leaf(0, &block_id0, &commitment0.commitment_root)];
        assert_eq!(commitment2.block_history.leaf_count(), 1);
        assert_eq!(
            commitment2.block_history_root(),
            compute_root(TreeId::BLOCK_HISTORY, &leaves)
        );

        let bank3 = new_child(&bank2, 3);
        let _block_id3 = freeze_with_block_id(&bank3);
        let commitment3 = bank3.block_commitment().unwrap();
        let leaves = [
            history_leaf(0, &block_id0, &commitment0.commitment_root),
            history_leaf(2, &block_id2, &commitment2.commitment_root),
        ];
        assert_eq!(
            commitment3.block_history_root(),
            compute_root(TreeId::BLOCK_HISTORY, &leaves)
        );
        let mut vector3 = [INACTIVE_COMMITMENT_SLOT; COMMITMENT_VECTOR_LEN];
        vector3[0] = compute_root(TreeId::BLOCK_HISTORY, &leaves);
        assert_eq!(
            commitment3.commitment_root,
            CommitmentVector(vector3).root()
        );

        // An ancestor's commitment root is provable against the history root
        // in slot 0.
        let proof = make_inclusion_proof(TreeId::BLOCK_HISTORY, &leaves, 0).unwrap();
        assert!(proof.verify(
            TreeId::BLOCK_HISTORY,
            &leaves[0],
            &commitment3.block_history_root()
        ));

        // A fork from slot 0 at slot 4 does not see slot 2 or 3.
        let bank4 = new_child(&bank0, 4);
        freeze_with_block_id(&bank4);
        assert_eq!(
            bank4.block_commitment().unwrap().block_history,
            commitment2.block_history
        );
    }

    #[test]
    fn test_parent_without_feature_starts_empty_history() {
        let bank0 = new_bank(false);
        freeze_with_block_id(&bank0);
        let mut bank1 = Bank::new_from_parent(bank0.clone(), SlotLeader::default(), 1);
        bank1.activate_feature(&feature_set::block_commitment_root::id());
        let bank1 = Arc::new(bank1);
        freeze_with_block_id(&bank1);
        let commitment1 = bank1.block_commitment().unwrap();
        assert_eq!(commitment1.block_history.leaf_count(), 0);
    }

    #[test]
    fn test_verify_commitment_root() {
        let bank0 = new_bank(true);
        freeze_with_block_id(&bank0);
        let bank1 = new_child(&bank0, 1);
        let wrong = Hash::new_unique();
        bank1.set_expected_commitment_root(wrong);
        freeze_with_block_id(&bank1);
        let computed = bank1.commitment_root().unwrap();
        assert_eq!(bank1.verify_commitment_root(), Err((wrong, computed)));

        let bank2 = new_child(&bank1, 2);
        freeze_with_block_id(&bank2);
        // Recompute the expected value independently of the bank.
        let history = bank1
            .block_commitment()
            .unwrap()
            .child_block_history(1, &bank1.block_id().unwrap().to_hash());
        let mut vector = [INACTIVE_COMMITMENT_SLOT; COMMITMENT_VECTOR_LEN];
        vector[0] = history.root(TreeId::BLOCK_HISTORY);
        let expected = CommitmentVector(vector).root();
        assert_eq!(bank2.commitment_root(), Some(expected));
        bank2.set_expected_commitment_root(expected);
        assert_eq!(bank2.verify_commitment_root(), Ok(()));
    }

    #[test]
    fn test_rehash_keeps_commitment() {
        let bank0 = new_bank(true);
        freeze_with_block_id(&bank0);
        let commitment = bank0.block_commitment();
        bank0.rehash();
        assert_eq!(bank0.block_commitment(), commitment);
    }
}
