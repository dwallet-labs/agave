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
//! [`Bank`], listed below. A slot method returns [`INACTIVE_COMMITMENT_SLOT`]
//! (32 zero bytes, which the SIMD defines as the value of a slot whose
//! defining feature is not active) unless its proposal is implemented and its
//! own feature is active for the bank; nothing else in the vector depends on
//! which slots are implemented:
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
//!
//! # Commitment failures
//!
//! A slot method that cannot compute its value from the bank's own state
//! returns a [`CommitmentFailure`] instead of a substitute value. The bank then
//! has no commitment root: the failure is recorded on the bank when it is
//! frozen ([`Bank::commitment_failure`]), [`Bank::verify_commitment_root`]
//! fails with it, replay marks the block dead as for a commitment root
//! mismatch, and a leader does not publish the block's footer.

use {
    super::Bank,
    agave_feature_set as feature_set,
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

/// `anchor_leaf = u64_le(parent.slot) || parent.block_id || 32 zero bytes`:
/// the history leaf of the last block before activation, which has no
/// commitment root.
pub fn anchor_leaf(slot: Slot, block_id: &Hash) -> [u8; HISTORY_LEAF_LEN] {
    history_leaf(slot, block_id, &Hash::default())
}

/// `H(B)` of the first block `B` with the feature active: the anchor leaf of
/// its parent alone.
pub fn anchor_block_history(parent_slot: Slot, parent_block_id: &Hash) -> AppendOnlyFrontier {
    let mut history = AppendOnlyFrontier::new();
    history
        .append(
            TreeId::BLOCK_HISTORY,
            &anchor_leaf(parent_slot, parent_block_id),
        )
        .expect("an empty block history has room for one leaf");
    history
}

/// Block commitment state of a frozen bank `B` (or of a snapshot bank).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockCommitment {
    /// Frontier of `H(B)`: the anchor leaf of the last block before
    /// activation, followed by the history leaves of all ancestors of `B` for
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

/// Why a bank with the `block_commitment_root` feature active has no
/// commitment root.
///
/// Each variant is a condition under which a value of the commitment vector
/// cannot be computed from the bank's state. None of them happens in a
/// correctly running validator; each would otherwise have to be replaced by a
/// substitute value, giving a commitment root that the cluster does not
/// compute. The failure is recorded on the bank when it is frozen, and the
/// block is rejected instead (see [`Bank::verify_commitment_root`]).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CommitmentFailure {
    /// Slot 0: the parent has a commitment root but no block id, so its
    /// history leaf cannot be computed.
    #[error("parent {parent_slot} has no block id for its block history leaf")]
    MissingParentBlockId { parent_slot: Slot },
    /// Slot 0: the parent has the feature active but a commitment failure of
    /// its own, so `H(B)` is unknown.
    #[error("parent {parent_slot} has a commitment failure")]
    ParentCommitmentFailure { parent_slot: Slot },
    /// Slot 0: the parent has the feature active but neither block commitment
    /// state nor a recorded failure (for example a bank restored from a
    /// snapshot written before the field existed), so `H(B)` is unknown.
    #[error("parent {parent_slot} has the feature active but no block commitment state")]
    MissingParentCommitment { parent_slot: Slot },
}

impl CommitmentFailure {
    /// Name of the condition, for datapoints.
    pub fn name(&self) -> &'static str {
        match self {
            Self::MissingParentBlockId { .. } => "missing_parent_block_id",
            Self::ParentCommitmentFailure { .. } => "parent_commitment_failure",
            Self::MissingParentCommitment { .. } => "missing_parent_commitment",
        }
    }
}

/// Why [`Bank::verify_commitment_root`] rejects a frozen bank.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CommitmentVerificationError {
    /// The footer's commitment root differs from the computed one (the zero
    /// hash if the bank has none).
    #[error("expected commitment root {expected}, computed {computed}")]
    Mismatch { expected: Hash, computed: Hash },
    /// The bank could not compute its commitment root.
    #[error("{0}")]
    Failure(CommitmentFailure),
}

impl Bank {
    /// Whether the Block Commitment Root feature is active for this bank.
    pub fn is_block_commitment_root_active(&self) -> bool {
        self.feature_set
            .is_active(&feature_set::block_commitment_root::id())
    }

    /// Whether freezing this bank now would fail only because its parent has
    /// no block id yet.
    ///
    /// The history leaf of the parent needs the parent's block id. A block
    /// this validator produced gets its block id from broadcast, once its
    /// last shred is made, which can be after a child of it is complete. A
    /// caller that can wait freezes the bank once this is false, instead of
    /// freezing it into [`CommitmentFailure::MissingParentBlockId`].
    pub fn is_waiting_for_parent_block_id(&self) -> bool {
        !self.is_frozen()
            && self.is_block_commitment_root_active()
            && self.parent().is_some_and(|parent| {
                parent.block_id().is_none() && parent.commitment_failure().is_none()
            })
    }

    /// Block commitment state, set when the bank is frozen with the feature
    /// active, or loaded from a snapshot. `None` for a bank with a
    /// [`Self::commitment_failure`].
    pub fn block_commitment(&self) -> Option<BlockCommitment> {
        self.block_commitment.read().unwrap().clone()
    }

    /// Block commitment state written to a snapshot of this bank: `None`
    /// unless the feature is active for the bank, so that a snapshot taken
    /// before activation has the format it had before the field existed.
    pub(crate) fn block_commitment_for_snapshot(&self) -> Option<BlockCommitment> {
        self.block_commitment()
            .filter(|_| self.is_block_commitment_root_active())
    }

    /// `commitment_root` of this bank, set when the bank is frozen with the
    /// feature active, or loaded from a snapshot. `None` for a bank with a
    /// [`Self::commitment_failure`].
    pub fn commitment_root(&self) -> Option<Hash> {
        self.block_commitment
            .read()
            .unwrap()
            .as_ref()
            .map(|commitment| commitment.commitment_root)
    }

    /// Why this bank, frozen with the feature active, has no commitment root.
    /// Set when the bank is frozen; `None` for every bank that has one, and
    /// for every bank with the feature inactive.
    pub fn commitment_failure(&self) -> Option<CommitmentFailure> {
        self.commitment_failure.read().unwrap().clone()
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

    /// Verifies the commitment state of a frozen bank. Must be called on a
    /// frozen bank.
    ///
    /// Fails with [`CommitmentVerificationError::Failure`] if the bank has a
    /// [`Self::commitment_failure`], whether or not the footer carries a
    /// commitment root, and otherwise with
    /// [`CommitmentVerificationError::Mismatch`] if the commitment root from
    /// the footer, if any, differs from the computed one (`computed` is the
    /// zero hash if the bank has no commitment root). Replay marks the block
    /// dead on error, as for a bank hash mismatch.
    pub fn verify_commitment_root(&self) -> std::result::Result<(), CommitmentVerificationError> {
        debug_assert!(self.is_frozen());
        if let Some(failure) = self.commitment_failure() {
            return Err(CommitmentVerificationError::Failure(failure));
        }
        let Some(expected) = self.expected_commitment_root() else {
            return Ok(());
        };
        let computed = self.commitment_root().unwrap_or_default();
        if expected == computed {
            Ok(())
        } else {
            Err(CommitmentVerificationError::Mismatch { expected, computed })
        }
    }

    /// Computes the block commitment state of this bank and stores it, or the
    /// [`CommitmentFailure`] that prevents it. Called once, from `freeze()`,
    /// after the accounts lattice hash is finished, while the hash lock is
    /// held, so that a frozen bank has one or the other. Does nothing if the
    /// feature is inactive.
    pub(super) fn freeze_block_commitment(&self) {
        if !self.is_block_commitment_root_active() {
            return;
        }
        match self.compute_block_commitment() {
            Ok(commitment) => *self.block_commitment.write().unwrap() = Some(commitment),
            Err(failure) => {
                error!(
                    "bank {} has no commitment root: commitment failure {}: {failure}",
                    self.slot(),
                    failure.name(),
                );
                datapoint_error!(
                    "bank-commitment_failure",
                    ("slot", self.slot(), i64),
                    ("failure", failure.name(), String),
                    ("error", failure.to_string(), String),
                );
                *self.block_commitment.write().unwrap() = None;
                *self.commitment_failure.write().unwrap() = Some(failure);
            }
        }
    }

    /// Computes the block commitment state of this bank (the feature must be
    /// active).
    fn compute_block_commitment(&self) -> std::result::Result<BlockCommitment, CommitmentFailure> {
        let parent = self.parent();
        let block_history = self.block_history_for_commitment(parent.as_deref())?;
        let vector = self.compute_commitment_vector(&CommitmentInputs {
            block_history: &block_history,
            parent: parent.as_deref(),
        })?;
        let commitment_root = vector.root();
        info!(
            "bank {} commitment_root: {commitment_root} block_history_len: {}",
            self.slot(),
            block_history.leaf_count(),
        );
        Ok(BlockCommitment {
            block_history,
            commitment_root,
        })
    }

    /// `H(B)` for this bank `B`:
    ///
    /// - no parent (genesis): empty;
    /// - parent with the feature inactive (first block with the feature
    ///   active): the anchor leaf of the parent alone;
    /// - otherwise `H(parent)` with `history_leaf(parent)` appended.
    ///
    /// The parent's block id is required in both cases that have a parent. A
    /// parent with the feature active must have block commitment state; a
    /// parent with a commitment failure, or with neither state nor failure,
    /// leaves `H(B)` unknown.
    fn block_history_for_commitment(
        &self,
        parent: Option<&Bank>,
    ) -> std::result::Result<AppendOnlyFrontier, CommitmentFailure> {
        let Some(parent) = parent else {
            return Ok(AppendOnlyFrontier::new());
        };
        let parent_slot = parent.slot();
        let parent_commitment = if parent.is_block_commitment_root_active() {
            if parent.commitment_failure().is_some() {
                return Err(CommitmentFailure::ParentCommitmentFailure { parent_slot });
            }
            Some(
                parent
                    .block_commitment()
                    .ok_or(CommitmentFailure::MissingParentCommitment { parent_slot })?,
            )
        } else {
            None
        };
        // Every frozen bank that a child is built on gets a block id in a
        // running validator: replayed banks from the blockstore before
        // freezing, leader banks from broadcast once the last shred is made,
        // snapshot banks from the snapshot. A leader bank can get it after a
        // child is complete, so replay waits for it before freezing the child
        // (see `is_waiting_for_parent_block_id`). Without it the history leaf
        // (or the anchor leaf), and so this bank's commitment root, cannot be
        // computed.
        let parent_block_id = parent
            .block_id()
            .ok_or(CommitmentFailure::MissingParentBlockId { parent_slot })?
            .to_hash();
        Ok(match parent_commitment {
            Some(parent_commitment) => {
                parent_commitment.child_block_history(parent_slot, &parent_block_id)
            }
            None => anchor_block_history(parent_slot, &parent_block_id),
        })
    }

    /// Computes the commitment vector of this bank from its inputs.
    pub fn compute_commitment_vector(
        &self,
        inputs: &CommitmentInputs,
    ) -> std::result::Result<CommitmentVector, CommitmentFailure> {
        let mut vector = [INACTIVE_COMMITMENT_SLOT; COMMITMENT_VECTOR_LEN];
        vector[CommitmentSlot::BlockHistoryRoot.index()] =
            inputs.block_history.root(TreeId::BLOCK_HISTORY);
        vector[CommitmentSlot::ReceiptsRoot.index()] =
            self.commitment_slot_receipts_root(inputs)?;
        vector[CommitmentSlot::AccountsDeltaRoot.index()] =
            self.commitment_slot_accounts_delta_root(inputs)?;
        vector[CommitmentSlot::ValidatorSetRoot.index()] =
            self.commitment_slot_validator_set_root(inputs)?;
        vector[CommitmentSlot::EpochBoundary.index()] =
            self.commitment_slot_epoch_boundary(inputs)?;
        // Slots 5..8 are reserved and stay zero.
        Ok(CommitmentVector(vector))
    }

    /// Commitment slot 1: `receipts_root` (Transaction Receipt Tree SIMD).
    ///
    /// Extension point: not implemented yet, so the slot is inactive and
    /// zero. The implementation must return [`INACTIVE_COMMITMENT_SLOT`]
    /// unless its own feature is active for this bank.
    pub fn commitment_slot_receipts_root(
        &self,
        _inputs: &CommitmentInputs,
    ) -> std::result::Result<Hash, CommitmentFailure> {
        Ok(INACTIVE_COMMITMENT_SLOT)
    }

    /// Commitment slot 2: `accounts_delta_root` (Accounts Delta Root SIMD).
    ///
    /// Extension point: not implemented yet, so the slot is inactive and
    /// zero. The implementation must return [`INACTIVE_COMMITMENT_SLOT`]
    /// unless its own feature is active for this bank.
    pub fn commitment_slot_accounts_delta_root(
        &self,
        _inputs: &CommitmentInputs,
    ) -> std::result::Result<Hash, CommitmentFailure> {
        Ok(INACTIVE_COMMITMENT_SLOT)
    }

    /// Commitment slot 3: `validator_set_root` (Epoch Validator Set Root SIMD).
    ///
    /// Extension point: not implemented yet, so the slot is inactive and
    /// zero. The implementation must return [`INACTIVE_COMMITMENT_SLOT`]
    /// unless its own feature is active for this bank.
    pub fn commitment_slot_validator_set_root(
        &self,
        _inputs: &CommitmentInputs,
    ) -> std::result::Result<Hash, CommitmentFailure> {
        Ok(INACTIVE_COMMITMENT_SLOT)
    }

    /// Commitment slot 4: `epoch_boundary` (Epoch Boundary Commitment SIMD).
    ///
    /// Extension point: not implemented yet, so the slot is inactive and
    /// zero. The implementation must return [`INACTIVE_COMMITMENT_SLOT`]
    /// unless its own feature is active for this bank. `inputs` carries the
    /// parent bank (`P` when this bank is the first of its epoch) and
    /// `H(B)`, whose leaf count is `history_len`.
    pub fn commitment_slot_epoch_boundary(
        &self,
        _inputs: &CommitmentInputs,
    ) -> std::result::Result<Hash, CommitmentFailure> {
        Ok(INACTIVE_COMMITMENT_SLOT)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use {
        super::*,
        crate::{
            bank::{SlotLeader, test_utils::goto_end_of_slot},
            genesis_utils::{activate_feature, create_genesis_config},
        },
        agave_votor_messages::consensus_message::BlockId,
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
        assert_eq!(bank0.commitment_failure(), None);
        // Without a footer value there is nothing to verify.
        assert_eq!(bank0.verify_commitment_root(), Ok(()));
    }

    #[test]
    fn test_block_history_chain() {
        let bank0 = new_bank(true);
        assert!(bank0.is_block_commitment_root_active());
        let block_id0 = freeze_with_block_id(&bank0);
        let commitment0 = bank0.block_commitment().unwrap();
        // A block with no parent (genesis) has an empty history.
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
    fn test_first_active_block_has_anchor_leaf() {
        let bank0 = new_bank(false);
        let block_id0 = freeze_with_block_id(&bank0);
        assert_eq!(bank0.block_commitment(), None);
        let mut bank1 = Bank::new_from_parent(bank0.clone(), SlotLeader::default(), 1);
        bank1.activate_feature(&feature_set::block_commitment_root::id());
        let bank1 = Arc::new(bank1);
        let block_id1 = freeze_with_block_id(&bank1);
        let commitment1 = bank1.block_commitment().unwrap();

        // H(bank1) is exactly the anchor leaf of bank0: its history leaf
        // with a zero commitment root.
        let mut anchor = [0u8; HISTORY_LEAF_LEN];
        anchor[..8].copy_from_slice(&0u64.to_le_bytes());
        anchor[8..40].copy_from_slice(block_id0.as_ref());
        assert_eq!(anchor_leaf(0, &block_id0), anchor);
        assert_eq!(commitment1.block_history.leaf_count(), 1);
        assert_eq!(
            commitment1.block_history_root(),
            compute_root(TreeId::BLOCK_HISTORY, &[anchor])
        );
        let mut vector1 = [INACTIVE_COMMITMENT_SLOT; COMMITMENT_VECTOR_LEN];
        vector1[0] = compute_root(TreeId::BLOCK_HISTORY, &[anchor]);
        assert_eq!(
            bank1.commitment_root(),
            Some(CommitmentVector(vector1).root())
        );

        // The next block appends bank1's leaf after the anchor leaf, which
        // stays at index 0.
        let bank2 = new_child(&bank1, 2);
        freeze_with_block_id(&bank2);
        let leaves = [
            anchor,
            history_leaf(1, &block_id1, &commitment1.commitment_root),
        ];
        let commitment2 = bank2.block_commitment().unwrap();
        assert_eq!(commitment2.block_history.leaf_count(), 2);
        assert_eq!(
            commitment2.block_history_root(),
            compute_root(TreeId::BLOCK_HISTORY, &leaves)
        );
        let proof = make_inclusion_proof(TreeId::BLOCK_HISTORY, &leaves, 0).unwrap();
        assert!(proof.verify(
            TreeId::BLOCK_HISTORY,
            &anchor,
            &commitment2.block_history_root()
        ));
    }

    #[test]
    fn test_first_active_block_without_parent_block_id_is_commitment_failure() {
        let bank0 = new_bank(false);
        goto_end_of_slot(bank0.clone());
        bank0.freeze();
        // The parent predates the feature and has no block id, so the anchor
        // leaf cannot be computed.
        assert_eq!(bank0.block_id(), None);
        let mut bank1 = Bank::new_from_parent(bank0.clone(), SlotLeader::default(), 1);
        bank1.activate_feature(&feature_set::block_commitment_root::id());
        let bank1 = Arc::new(bank1);
        freeze_with_block_id(&bank1);
        assert_commitment_failure(
            &bank1,
            CommitmentFailure::MissingParentBlockId { parent_slot: 0 },
        );
    }

    #[test]
    fn test_active_parent_without_commitment_state_is_commitment_failure() {
        let bank0 = new_bank(true);
        freeze_with_block_id(&bank0);
        // A bank with the feature active but neither block commitment state
        // nor a recorded failure, as one restored from a snapshot written
        // before the field existed. Its child must not restart the history.
        *bank0.block_commitment.write().unwrap() = None;
        assert_eq!(bank0.commitment_failure(), None);
        let bank1 = new_child(&bank0, 1);
        freeze_with_block_id(&bank1);
        assert_commitment_failure(
            &bank1,
            CommitmentFailure::MissingParentCommitment { parent_slot: 0 },
        );
        // And the failure propagates.
        let bank2 = new_child(&bank1, 2);
        freeze_with_block_id(&bank2);
        assert_commitment_failure(
            &bank2,
            CommitmentFailure::ParentCommitmentFailure { parent_slot: 1 },
        );
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
        assert_eq!(
            bank1.verify_commitment_root(),
            Err(CommitmentVerificationError::Mismatch {
                expected: wrong,
                computed
            })
        );

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

    /// Asserts that `bank` is frozen with `failure` recorded, exposes no
    /// commitment root, and fails verification with that reason whether or
    /// not its footer carried a commitment root.
    pub(crate) fn assert_commitment_failure(bank: &Bank, failure: CommitmentFailure) {
        assert!(bank.is_frozen());
        assert_eq!(bank.commitment_failure(), Some(failure.clone()));
        assert_eq!(bank.block_commitment(), None);
        assert_eq!(bank.commitment_root(), None);
        let expected_err = Err(CommitmentVerificationError::Failure(failure));
        assert_eq!(bank.verify_commitment_root(), expected_err);
        // The footer's value, whatever it is, does not make it pass.
        bank.set_expected_commitment_root(Hash::default());
        assert_eq!(bank.verify_commitment_root(), expected_err);
        bank.set_expected_commitment_root(Hash::new_unique());
        assert_eq!(bank.verify_commitment_root(), expected_err);
    }

    #[test]
    fn test_missing_parent_block_id_is_commitment_failure() {
        let bank0 = new_bank(true);
        goto_end_of_slot(bank0.clone());
        bank0.freeze();
        // The parent has a commitment root but no block id.
        assert!(bank0.commitment_root().is_some());
        assert_eq!(bank0.block_id(), None);
        let bank1 = new_child(&bank0, 1);
        freeze_with_block_id(&bank1);
        assert_commitment_failure(
            &bank1,
            CommitmentFailure::MissingParentBlockId { parent_slot: 0 },
        );

        // A child of a bank with a commitment failure cannot compute its
        // block history either.
        let bank2 = new_child(&bank1, 2);
        freeze_with_block_id(&bank2);
        assert_commitment_failure(
            &bank2,
            CommitmentFailure::ParentCommitmentFailure { parent_slot: 1 },
        );

        // A sibling built on a parent with a block id is unaffected.
        bank0.set_block_id(Some(BlockId::from(Hash::new_unique())));
        let bank3 = new_child(&bank0, 3);
        freeze_with_block_id(&bank3);
        assert_eq!(bank3.commitment_failure(), None);
        assert!(bank3.commitment_root().is_some());
        assert_eq!(bank3.verify_commitment_root(), Ok(()));
    }

    #[test]
    fn test_block_commitment_for_snapshot() {
        let mut genesis = create_genesis_config(1_000_000_000).genesis_config;
        activate_feature(&mut genesis, feature_set::block_commitment_root::id());
        let mut bank0 = Bank::new_for_tests(&genesis);
        bank0.freeze();
        bank0.set_block_id(Some(BlockId::new_unique()));
        assert!(bank0.block_commitment().is_some());
        assert_eq!(
            bank0.block_commitment_for_snapshot(),
            bank0.block_commitment()
        );
        // State held without the feature active (which only a test can
        // produce) is not written to a snapshot.
        bank0.deactivate_feature(&feature_set::block_commitment_root::id());
        assert!(bank0.block_commitment().is_some());
        assert_eq!(bank0.block_commitment_for_snapshot(), None);
        assert_eq!(bank0.get_fields_to_serialize().block_commitment, None);
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
