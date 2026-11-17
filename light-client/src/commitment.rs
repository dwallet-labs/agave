//! Verified blocks, the commitment vector and the block history.
//!
//! ```text
//! commitment_root  = root(0x00, [slot_0, ..., slot_7])
//! slot_0           = block_history_root = root(0x01, H(B))
//! history_leaf(A)  = u64_le(A.slot) || A.block_id || A.commitment_root
//! anchor_leaf      = u64_le(P.slot) || P.block_id || 32 zero bytes
//! ```
//!
//! `H(B)` holds the history leaves of all ancestors of `B` since activation
//! of Block Commitment Root, oldest first, preceded by the anchor leaf of the
//! last block before activation (if the chain has one).

use {
    crate::error::Error,
    solana_hash::Hash,
    solana_merkle_tree::commitment_tree::{InclusionProof, TreeId},
};

/// Number of slots of the commitment vector.
pub const COMMITMENT_VECTOR_LEN: u64 = 8;

/// Number of sibling hashes that open one slot of the commitment vector.
pub const SLOT_OPENING_LEN: usize = 3;

/// Size of a block history leaf.
pub const HISTORY_LEAF_LEN: usize = 8 + 32 + 32;

/// The assigned slots of the commitment vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CommitmentSlot {
    /// `block_history_root` (Block Commitment Root).
    BlockHistoryRoot = 0,
    /// `receipts_root` (Transaction Receipt Tree).
    ReceiptsRoot = 1,
    /// `accounts_delta_root` (Accounts Delta Root).
    AccountsDeltaRoot = 2,
    /// `validator_set_root(E + 1)` (Epoch Validator Set Root).
    ValidatorSetRoot = 3,
    /// `epoch_boundary` (Epoch Boundary Commitment).
    EpochBoundary = 4,
    /// `accounts_state_root` (Accounts State Root).
    AccountsStateRoot = 5,
}

impl CommitmentSlot {
    /// Leaf index of the slot in the commitment vector.
    pub const fn index(self) -> u64 {
        self as u64
    }
}

/// The three sibling hashes that open one slot of the commitment vector
/// (leaf first). Which slot is opened is chosen by the verifier, never taken
/// from the proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotOpening {
    pub siblings: [Hash; SLOT_OPENING_LEN],
}

impl SlotOpening {
    /// The opening of `slot` from a commitment tree inclusion proof, as the
    /// runtime builds it (`CommitmentVector::inclusion_proof`). `None` if the
    /// proof is not one for `slot` of an eight-slot vector.
    pub fn from_inclusion_proof(slot: CommitmentSlot, proof: &InclusionProof) -> Option<Self> {
        if proof.leaf_index != slot.index() || proof.leaf_count != COMMITMENT_VECTOR_LEN {
            return None;
        }
        Some(Self {
            siblings: proof.siblings.as_slice().try_into().ok()?,
        })
    }

    /// Checks that `value` is in slot `slot` of the commitment vector with
    /// root `commitment_root`.
    pub fn verify(
        &self,
        commitment_root: &Hash,
        slot: CommitmentSlot,
        value: &Hash,
    ) -> Result<(), Error> {
        let proof = InclusionProof {
            leaf_index: slot.index(),
            leaf_count: COMMITMENT_VECTOR_LEN,
            siblings: self.siblings.to_vec(),
        };
        if proof.verify(TreeId::COMMITMENT_VECTOR, value.as_ref(), commitment_root) {
            Ok(())
        } else {
            Err(Error::SlotOpening(slot))
        }
    }
}

/// A block whose `commitment_root` is verified: a [`VerifiedBlock`] (finality
/// proof and footer proof) or a [`VerifiedAncestor`] (history proof from a
/// verified block). Every item proof starts from one.
///
/// Only this crate implements the trait for its verified types, so a value
/// can be trusted to come from a verifier.
pub trait CommittedBlock: sealed::Sealed {
    fn slot(&self) -> u64;
    fn block_id(&self) -> &Hash;
    fn commitment_root(&self) -> &Hash;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::VerifiedBlock {}
    impl Sealed for super::VerifiedAncestor {}
}

/// A block named by slot and block id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockRef {
    pub slot: u64,
    pub block_id: Hash,
}

/// A block verified with a finality proof and a footer proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedBlock {
    slot: u64,
    block_id: Hash,
    bank_hash: Hash,
    commitment_root: Hash,
    parent: BlockRef,
}

impl VerifiedBlock {
    pub(crate) fn new(
        slot: u64,
        block_id: Hash,
        bank_hash: Hash,
        commitment_root: Hash,
        parent: BlockRef,
    ) -> Self {
        Self {
            slot,
            block_id,
            bank_hash,
            commitment_root,
            parent,
        }
    }

    pub fn slot(&self) -> u64 {
        self.slot
    }

    pub fn block_id(&self) -> &Hash {
        &self.block_id
    }

    /// `bank_hash` of the block's footer.
    pub fn bank_hash(&self) -> &Hash {
        &self.bank_hash
    }

    /// `commitment_root` of the block's footer.
    pub fn commitment_root(&self) -> &Hash {
        &self.commitment_root
    }

    /// The parent named by the block id's final leaf. Its finality follows
    /// from the block's; its commitment root is reached through the history.
    pub fn parent(&self) -> &BlockRef {
        &self.parent
    }
}

impl CommittedBlock for VerifiedBlock {
    fn slot(&self) -> u64 {
        self.slot
    }

    fn block_id(&self) -> &Hash {
        &self.block_id
    }

    fn commitment_root(&self) -> &Hash {
        &self.commitment_root
    }
}

/// An ancestor of a verified block, with its commitment root, verified
/// through a history proof or an epoch boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedAncestor {
    slot: u64,
    block_id: Hash,
    commitment_root: Hash,
}

impl VerifiedAncestor {
    pub(crate) fn new(slot: u64, block_id: Hash, commitment_root: Hash) -> Self {
        Self {
            slot,
            block_id,
            commitment_root,
        }
    }

    pub fn slot(&self) -> u64 {
        self.slot
    }

    pub fn block_id(&self) -> &Hash {
        &self.block_id
    }

    pub fn commitment_root(&self) -> &Hash {
        &self.commitment_root
    }
}

impl CommittedBlock for VerifiedAncestor {
    fn slot(&self) -> u64 {
        self.slot
    }

    fn block_id(&self) -> &Hash {
        &self.block_id
    }

    fn commitment_root(&self) -> &Hash {
        &self.commitment_root
    }
}

/// `history_leaf(A) = u64_le(A.slot) || A.block_id || A.commitment_root`
pub fn history_leaf(slot: u64, block_id: &Hash, commitment_root: &Hash) -> [u8; HISTORY_LEAF_LEN] {
    let mut leaf = [0u8; HISTORY_LEAF_LEN];
    leaf[..8].copy_from_slice(&slot.to_le_bytes());
    leaf[8..40].copy_from_slice(block_id.as_ref());
    leaf[40..].copy_from_slice(commitment_root.as_ref());
    leaf
}

/// Proof that a block is an ancestor of a verified block, with its commitment
/// root: slot 0 of the verified block opened to `block_history_root`, and a
/// path from the ancestor's history leaf to it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryProof {
    pub block_history_root: Hash,
    pub opening: SlotOpening,
    /// The ancestor's history leaf: slot, block id and commitment root (zero
    /// for the anchor leaf).
    pub slot: u64,
    pub block_id: Hash,
    pub commitment_root: Hash,
    /// Path from the leaf to `block_history_root`.
    pub proof: InclusionProof,
}

impl HistoryProof {
    /// Verifies the ancestor and returns it with its commitment root.
    /// Rejects the anchor leaf, which has none; see [`Self::verify_anchor`].
    pub fn verify(&self, block: &impl CommittedBlock) -> Result<VerifiedAncestor, Error> {
        self.verify_leaf(block)?;
        if self.is_anchor() {
            return Err(Error::AnchorLeaf);
        }
        Ok(VerifiedAncestor::new(
            self.slot,
            self.block_id,
            self.commitment_root,
        ))
    }

    /// Verifies that the proven leaf is the anchor leaf: the last block
    /// before Block Commitment Root activated on the chain. It proves
    /// ancestry only; the anchor block has no commitment root.
    pub fn verify_anchor(&self, block: &impl CommittedBlock) -> Result<BlockRef, Error> {
        self.verify_leaf(block)?;
        if !self.is_anchor() {
            return Err(Error::NotAnchorLeaf);
        }
        Ok(BlockRef {
            slot: self.slot,
            block_id: self.block_id,
        })
    }

    fn is_anchor(&self) -> bool {
        self.commitment_root == Hash::default()
    }

    fn verify_leaf(&self, block: &impl CommittedBlock) -> Result<(), Error> {
        self.opening.verify(
            block.commitment_root(),
            CommitmentSlot::BlockHistoryRoot,
            &self.block_history_root,
        )?;
        // A zero commitment root appears only in the anchor leaf, which is
        // always leaf 0.
        if self.is_anchor() && self.proof.leaf_index != 0 {
            return Err(Error::HistoryProof);
        }
        let leaf = history_leaf(self.slot, &self.block_id, &self.commitment_root);
        if !self
            .proof
            .verify(TreeId::BLOCK_HISTORY, &leaf, &self.block_history_root)
        {
            return Err(Error::HistoryProof);
        }
        // Slots in a history are strictly increasing and below the block's.
        if self.slot >= block.slot() {
            return Err(Error::AncestorNotBefore {
                slot: block.slot(),
                ancestor_slot: self.slot,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use {
        super::*,
        solana_merkle_tree::commitment_tree::{compute_root, make_inclusion_proof},
    };

    /// A commitment vector with the given values in its slots, its root,
    /// and the opening of every slot.
    pub(crate) struct TestVector {
        pub(crate) values: [Hash; 8],
    }

    impl TestVector {
        pub(crate) fn new() -> Self {
            Self {
                values: std::array::from_fn(|i| Hash::new_from_array([0x40 + i as u8; 32])),
            }
        }

        pub(crate) fn set(&mut self, slot: CommitmentSlot, value: Hash) {
            self.values[slot.index() as usize] = value;
        }

        pub(crate) fn root(&self) -> Hash {
            compute_root(TreeId::COMMITMENT_VECTOR, &self.values)
        }

        pub(crate) fn opening(&self, slot: CommitmentSlot) -> SlotOpening {
            let proof = make_inclusion_proof(
                TreeId::COMMITMENT_VECTOR,
                &self.values,
                slot.index() as usize,
            )
            .unwrap();
            SlotOpening::from_inclusion_proof(slot, &proof).unwrap()
        }

        /// A verified block at `slot` committing to this vector.
        pub(crate) fn block(&self, slot: u64) -> VerifiedBlock {
            VerifiedBlock::new(
                slot,
                Hash::new_from_array([0xb1; 32]),
                Hash::new_from_array([0xb2; 32]),
                self.root(),
                BlockRef {
                    slot: slot.saturating_sub(1),
                    block_id: Hash::new_from_array([0xb3; 32]),
                },
            )
        }
    }

    const ALL_SLOTS: [CommitmentSlot; 6] = [
        CommitmentSlot::BlockHistoryRoot,
        CommitmentSlot::ReceiptsRoot,
        CommitmentSlot::AccountsDeltaRoot,
        CommitmentSlot::ValidatorSetRoot,
        CommitmentSlot::EpochBoundary,
        CommitmentSlot::AccountsStateRoot,
    ];

    #[test]
    fn test_slot_opening() {
        let vector = TestVector::new();
        let root = vector.root();
        for slot in ALL_SLOTS {
            let opening = vector.opening(slot);
            let value = vector.values[slot.index() as usize];
            assert_eq!(opening.verify(&root, slot, &value), Ok(()));
            // Wrong value.
            assert_eq!(
                opening.verify(&root, slot, &Hash::new_from_array([0xee; 32])),
                Err(Error::SlotOpening(slot))
            );
            // Wrong root.
            assert_eq!(
                opening.verify(&Hash::new_from_array([0xee; 32]), slot, &value),
                Err(Error::SlotOpening(slot))
            );
            // Tampered sibling.
            for i in 0..SLOT_OPENING_LEN {
                let mut tampered = opening;
                tampered.siblings[i] = Hash::new_from_array([0xee; 32]);
                assert_eq!(
                    tampered.verify(&root, slot, &value),
                    Err(Error::SlotOpening(slot))
                );
            }
            // The opening of a slot does not open another slot, even with
            // that slot's value.
            for other in ALL_SLOTS {
                if other != slot {
                    let other_value = vector.values[other.index() as usize];
                    assert_eq!(
                        opening.verify(&root, other, &other_value),
                        Err(Error::SlotOpening(other))
                    );
                    assert_eq!(
                        opening.verify(&root, other, &value),
                        Err(Error::SlotOpening(other))
                    );
                }
            }
        }
    }

    #[test]
    fn test_slot_opening_from_inclusion_proof() {
        let vector = TestVector::new();
        let proof = make_inclusion_proof(TreeId::COMMITMENT_VECTOR, &vector.values, 2).unwrap();
        assert!(
            SlotOpening::from_inclusion_proof(CommitmentSlot::AccountsDeltaRoot, &proof).is_some()
        );
        assert!(SlotOpening::from_inclusion_proof(CommitmentSlot::ReceiptsRoot, &proof).is_none());
        let mut wrong = proof.clone();
        wrong.leaf_count = 9;
        assert!(
            SlotOpening::from_inclusion_proof(CommitmentSlot::AccountsDeltaRoot, &wrong).is_none()
        );
        let mut wrong = proof.clone();
        wrong.siblings.pop();
        assert!(
            SlotOpening::from_inclusion_proof(CommitmentSlot::AccountsDeltaRoot, &wrong).is_none()
        );
    }

    struct TestHistory {
        leaves: Vec<[u8; HISTORY_LEAF_LEN]>,
        /// (slot, block id, commitment root) of each leaf.
        entries: Vec<(u64, Hash, Hash)>,
    }

    /// A history of `n` leaves: the anchor leaf of slot 10 if `anchor`, then
    /// blocks at slots 11, 13, 15, ...
    fn test_history(n: usize, anchor: bool) -> TestHistory {
        let entries: Vec<_> = (0..n)
            .map(|i| {
                let slot = 10 + 2 * i as u64 - u64::from(i > 0);
                let root = if anchor && i == 0 {
                    Hash::default()
                } else {
                    Hash::new_from_array([0x60 + i as u8; 32])
                };
                (slot, Hash::new_from_array([0x20 + i as u8; 32]), root)
            })
            .collect();
        let leaves = entries
            .iter()
            .map(|(slot, id, root)| history_leaf(*slot, id, root))
            .collect();
        TestHistory { leaves, entries }
    }

    fn history_proof(history: &TestHistory, vector: &TestVector, index: usize) -> HistoryProof {
        let (slot, block_id, commitment_root) = history.entries[index];
        HistoryProof {
            block_history_root: compute_root(TreeId::BLOCK_HISTORY, &history.leaves),
            opening: vector.opening(CommitmentSlot::BlockHistoryRoot),
            slot,
            block_id,
            commitment_root,
            proof: make_inclusion_proof(TreeId::BLOCK_HISTORY, &history.leaves, index).unwrap(),
        }
    }

    #[test]
    fn test_history_proof() {
        for anchor in [false, true] {
            for n in 1..=9 {
                let history = test_history(n, anchor);
                let mut vector = TestVector::new();
                vector.set(
                    CommitmentSlot::BlockHistoryRoot,
                    compute_root(TreeId::BLOCK_HISTORY, &history.leaves),
                );
                let block = vector.block(1_000);
                for index in 0..n {
                    let proof = history_proof(&history, &vector, index);
                    let (slot, block_id, commitment_root) = history.entries[index];
                    if anchor && index == 0 {
                        assert_eq!(proof.verify(&block), Err(Error::AnchorLeaf));
                        assert_eq!(proof.verify_anchor(&block), Ok(BlockRef { slot, block_id }));
                    } else {
                        let ancestor = proof.verify(&block).unwrap();
                        assert_eq!(ancestor.slot(), slot);
                        assert_eq!(*ancestor.block_id(), block_id);
                        assert_eq!(*ancestor.commitment_root(), commitment_root);
                        assert_eq!(proof.verify_anchor(&block), Err(Error::NotAnchorLeaf));
                    }

                    // Tampered leaf fields.
                    let mut tampered = proof.clone();
                    tampered.slot += 1;
                    assert_eq!(tampered.verify(&block), Err(Error::HistoryProof));
                    let mut tampered = proof.clone();
                    tampered.block_id = Hash::new_from_array([0xee; 32]);
                    assert_eq!(tampered.verify(&block), Err(Error::HistoryProof));
                    let mut tampered = proof.clone();
                    tampered.commitment_root = Hash::new_from_array([0xee; 32]);
                    assert_eq!(tampered.verify(&block), Err(Error::HistoryProof));
                    // Tampered sibling.
                    for i in 0..proof.proof.siblings.len() {
                        let mut tampered = proof.clone();
                        tampered.proof.siblings[i] = Hash::new_from_array([0xee; 32]);
                        assert_eq!(tampered.verify(&block), Err(Error::HistoryProof));
                    }
                    // Wrong index.
                    let mut tampered = proof.clone();
                    tampered.proof.leaf_index = (index as u64 + 1) % n as u64;
                    if n > 1 {
                        assert!(tampered.verify(&block).is_err());
                    }
                    // Tampered history root (the opening fails).
                    let mut tampered = proof.clone();
                    tampered.block_history_root = Hash::new_from_array([0xee; 32]);
                    assert_eq!(
                        tampered.verify(&block),
                        Err(Error::SlotOpening(CommitmentSlot::BlockHistoryRoot))
                    );
                    // Another block's commitment root.
                    let other = TestVector::new().block(1_000);
                    assert_eq!(
                        proof.verify(&other),
                        Err(Error::SlotOpening(CommitmentSlot::BlockHistoryRoot))
                    );
                    // An "ancestor" that is not older than the block.
                    let young = vector.block(slot);
                    assert_eq!(
                        proof.verify_leaf(&young),
                        Err(Error::AncestorNotBefore {
                            slot,
                            ancestor_slot: slot
                        })
                    );
                }
            }
        }
    }

    #[test]
    fn test_zero_commitment_root_only_at_index_zero() {
        // A history whose leaf 2 has a zero commitment root (which no valid
        // chain has): a proof for it is rejected as the anchor and as an
        // ancestor.
        let mut history = test_history(4, false);
        history.entries[2].2 = Hash::default();
        let (slot, id, root) = history.entries[2];
        history.leaves[2] = history_leaf(slot, &id, &root);
        let mut vector = TestVector::new();
        vector.set(
            CommitmentSlot::BlockHistoryRoot,
            compute_root(TreeId::BLOCK_HISTORY, &history.leaves),
        );
        let block = vector.block(1_000);
        let proof = history_proof(&history, &vector, 2);
        assert_eq!(proof.verify(&block), Err(Error::HistoryProof));
        assert_eq!(proof.verify_anchor(&block), Err(Error::HistoryProof));
    }
}
