//! The footer proof: `bank_hash` and `commitment_root` of a block, and its
//! parent, from its `block_id`.
//!
//! With Alpenglow, `block_id` is the root of the shred Merkle tree ("double
//! Merkle" tree) whose leaves are the Merkle roots of the block's FEC sets,
//! followed by one final leaf. For a block with a version 2 footer (Block
//! Commitment Root SIMD, "Binding the Footer to the Block Id") the final leaf
//! is
//!
//! ```text
//! final_leaf = sha256(u64_le(parent_slot) || parent_block_id
//!                     || u32_le(fec_set_count)
//!                     || bank_hash || commitment_root)
//! ```
//!
//! The tree is the shred Merkle tree, not the commitment tree: an inner node
//! is `sha256("\x01SOLANA_MERKLE_SHREDS_NODE" || left[..20] || right[..20])`,
//! a level with an odd number of nodes pairs its last node with itself, and
//! proof entries are 20-byte truncations. This module reimplements the
//! verifier of `solana_ledger::shred::merkle_tree` (`double_merkle_final_leaf`,
//! `verify_merkle_proof`) and the checks of the block id repair response
//! (`verify_parent_fec_set_count` in `solana_core::repair::serve_repair`),
//! so that the crate does not depend on the ledger.

use {crate::error::Error, solana_hash::Hash, solana_sha256_hasher::hashv};

/// Size of an entry of a shred Merkle proof: a node truncated to 20 bytes.
pub const MERKLE_PROOF_ENTRY_LEN: usize = 20;

/// One entry of a shred Merkle proof.
pub type MerkleProofEntry = [u8; MERKLE_PROOF_ENTRY_LEN];

/// Largest number of FEC sets in a block: `MAX_DATA_SHREDS_PER_SLOT` (32,768)
/// over `DATA_SHREDS_PER_FEC_BLOCK` (32), as `solana_ledger::shred`.
pub const MAX_FEC_SETS_PER_SLOT: u32 = 1024;

/// Prefix of an inner node of the shred Merkle tree.
const MERKLE_HASH_PREFIX_NODE: &[u8] = b"\x01SOLANA_MERKLE_SHREDS_NODE";

/// The final leaf of the block id tree of a block with a version 2 footer.
pub fn final_leaf(
    parent_slot: u64,
    parent_block_id: &Hash,
    fec_set_count: u32,
    bank_hash: &Hash,
    commitment_root: &Hash,
) -> Hash {
    hashv(&[
        &parent_slot.to_le_bytes(),
        parent_block_id.as_ref(),
        &fec_set_count.to_le_bytes(),
        bank_hash.as_ref(),
        commitment_root.as_ref(),
    ])
}

/// Number of proof entries of a leaf of a shred Merkle tree with
/// `num_leaves` leaves: `ceil(log2(num_leaves))`.
pub fn proof_len(num_leaves: u64) -> usize {
    let bits = u64::BITS.saturating_sub(num_leaves.leading_zeros());
    let len = if num_leaves.is_power_of_two() {
        bits.saturating_sub(1)
    } else {
        bits
    };
    len as usize
}

/// `sha256(prefix || left[..20] || right[..20])`
fn join_nodes(left: &[u8], right: &[u8]) -> Hash {
    let left = left.get(..MERKLE_PROOF_ENTRY_LEN).unwrap_or(left);
    let right = right.get(..MERKLE_PROOF_ENTRY_LEN).unwrap_or(right);
    hashv(&[MERKLE_HASH_PREFIX_NODE, left, right])
}

/// Root of a shred Merkle tree from the leaf at `index` and its proof, or
/// `None` if the proof is too short for `index`.
pub fn merkle_root(mut index: u64, leaf: &Hash, proof: &[MerkleProofEntry]) -> Option<Hash> {
    let mut node = *leaf;
    for sibling in proof {
        node = if index & 1 == 0 {
            join_nodes(node.as_ref(), sibling)
        } else {
            join_nodes(sibling, node.as_ref())
        };
        index >>= 1;
    }
    (index == 0).then_some(node)
}

/// Proof that a block with a version 2 footer has the given `bank_hash`,
/// `commitment_root` and parent: the contents of its final leaf and the path
/// from the final leaf to `block_id`.
///
/// The same fields form the validators' `ParentFecSetCountWithFooter` block id
/// repair response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FooterProof {
    pub parent_slot: u64,
    pub parent_block_id: Hash,
    /// Number of FEC sets of the block. The final leaf is leaf
    /// `fec_set_count` of a tree of `fec_set_count + 1` leaves.
    pub fec_set_count: u32,
    pub bank_hash: Hash,
    pub commitment_root: Hash,
    /// Path from the final leaf to `block_id`, leaf first.
    pub proof: Vec<MerkleProofEntry>,
}

impl FooterProof {
    /// Checks that the final leaf built from this proof's fields is the leaf
    /// at index `fec_set_count` of the tree with root `block_id`.
    ///
    /// The index check identifies the leaf as the final one: an FEC set root
    /// cannot be presented in its place. As validators do, the FEC set count
    /// is bounded and the path must have exactly the length of a tree of
    /// `fec_set_count + 1` leaves.
    pub fn verify(&self, block_id: &Hash) -> Result<(), Error> {
        if self.fec_set_count == 0 || self.fec_set_count > MAX_FEC_SETS_PER_SLOT {
            return Err(Error::InvalidFecSetCount(self.fec_set_count));
        }
        let index = u64::from(self.fec_set_count);
        let expected = proof_len(index.saturating_add(1));
        if self.proof.len() != expected {
            return Err(Error::FooterProofLength {
                expected,
                actual: self.proof.len(),
            });
        }
        let leaf = final_leaf(
            self.parent_slot,
            &self.parent_block_id,
            self.fec_set_count,
            &self.bank_hash,
            &self.commitment_root,
        );
        match merkle_root(index, &leaf, &self.proof) {
            Some(root) if root == *block_id => Ok(()),
            _ => Err(Error::FooterProofMismatch),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use {
        super::*,
        solana_entry::block_component::FooterBinding,
        solana_ledger::shred::{
            MAX_FEC_SETS_PER_SLOT as LEDGER_MAX_FEC_SETS_PER_SLOT,
            merkle_tree::{
                MerkleTree, double_merkle_final_leaf, get_proof_size, hash_as_merkle_proof_entry,
                verify_merkle_proof,
            },
        },
    };

    fn hash(i: u64, salt: u8) -> Hash {
        hashv(&[&i.to_le_bytes(), &[salt]])
    }

    /// A block id tree over `fec_set_count` made-up FEC set roots and the
    /// extended final leaf, built with the ledger's code, and the footer proof
    /// built from its `make_merkle_proof`.
    fn ledger_block(fec_set_count: u32) -> (Hash, FooterProof, MerkleTree, Vec<Hash>) {
        let parent_slot = 99;
        let parent_block_id = hash(1, 0xa0);
        let binding = FooterBinding {
            bank_hash: hash(fec_set_count as u64, 0xb0),
            commitment_root: hash(fec_set_count as u64, 0xc0),
        };
        let mut leaves: Vec<Hash> = (0..fec_set_count).map(|i| hash(i as u64, 0xd0)).collect();
        leaves.push(double_merkle_final_leaf(
            parent_slot,
            &parent_block_id,
            fec_set_count,
            Some(&binding),
        ));
        let tree =
            MerkleTree::try_new_with_len(leaves.iter().copied().map(Ok), leaves.len()).unwrap();
        let proof: Vec<MerkleProofEntry> = tree
            .make_merkle_proof(fec_set_count as usize, leaves.len())
            .map(|entry| *entry.unwrap())
            .collect();
        let footer = FooterProof {
            parent_slot,
            parent_block_id,
            fec_set_count,
            bank_hash: binding.bank_hash,
            commitment_root: binding.commitment_root,
            proof,
        };
        (*tree.root(), footer, tree, leaves)
    }

    #[test]
    fn test_constants_match_ledger() {
        assert_eq!(MAX_FEC_SETS_PER_SLOT, LEDGER_MAX_FEC_SETS_PER_SLOT);
        for n in 0..3000usize {
            assert_eq!(proof_len(n as u64), get_proof_size(n) as usize, "n = {n}");
        }
    }

    #[test]
    fn test_final_leaf_matches_ledger() {
        let binding = FooterBinding {
            bank_hash: hash(1, 1),
            commitment_root: hash(2, 2),
        };
        assert_eq!(
            final_leaf(
                12345,
                &hash(3, 3),
                17,
                &binding.bank_hash,
                &binding.commitment_root
            ),
            double_merkle_final_leaf(12345, &hash(3, 3), 17, Some(&binding))
        );
    }

    #[test]
    fn test_footer_proof_agrees_with_ledger() {
        let counts = (1..=40).chain([63, 64, 65, 127, 128, 129, 1000, MAX_FEC_SETS_PER_SLOT]);
        for fec_set_count in counts {
            let (block_id, footer, tree, leaves) = ledger_block(fec_set_count);
            assert_eq!(footer.verify(&block_id), Ok(()), "count {fec_set_count}");

            // Every leaf of the tree: our root computation agrees with the
            // ledger's verifier.
            for (index, leaf) in leaves.iter().enumerate() {
                let proof: Vec<MerkleProofEntry> = tree
                    .make_merkle_proof(index, leaves.len())
                    .map(|entry| *entry.unwrap())
                    .collect();
                assert_eq!(
                    merkle_root(index as u64, leaf, &proof),
                    Some(block_id),
                    "count {fec_set_count}, index {index}"
                );
                let flat: Vec<u8> = proof.iter().flatten().copied().collect();
                assert!(
                    verify_merkle_proof(hash_as_merkle_proof_entry(leaf), index, &flat, block_id)
                        .is_ok()
                );
            }

            // Tampered fields of the final leaf.
            let mut tampered = footer.clone();
            tampered.commitment_root = hash(0, 0xee);
            assert_eq!(tampered.verify(&block_id), Err(Error::FooterProofMismatch));
            let mut tampered = footer.clone();
            tampered.bank_hash = hash(0, 0xee);
            assert_eq!(tampered.verify(&block_id), Err(Error::FooterProofMismatch));
            let mut tampered = footer.clone();
            std::mem::swap(&mut tampered.bank_hash, &mut tampered.commitment_root);
            assert_eq!(tampered.verify(&block_id), Err(Error::FooterProofMismatch));
            let mut tampered = footer.clone();
            tampered.parent_slot += 1;
            assert_eq!(tampered.verify(&block_id), Err(Error::FooterProofMismatch));
            let mut tampered = footer.clone();
            tampered.parent_block_id = hash(0, 0xee);
            assert_eq!(tampered.verify(&block_id), Err(Error::FooterProofMismatch));

            // Tampered sibling.
            for i in 0..footer.proof.len() {
                let mut tampered = footer.clone();
                tampered.proof[i][0] ^= 1;
                assert_eq!(tampered.verify(&block_id), Err(Error::FooterProofMismatch));
            }

            // Wrong leaf index: the same leaf and path claimed for another
            // FEC set count (the count is also inside the leaf).
            for other in [fec_set_count - 1, fec_set_count + 1] {
                if other == 0 || other > MAX_FEC_SETS_PER_SLOT {
                    continue;
                }
                let mut tampered = footer.clone();
                tampered.fec_set_count = other;
                assert!(
                    tampered.verify(&block_id).is_err(),
                    "count {fec_set_count} as {other}"
                );
            }

            // Another block id.
            assert_eq!(
                footer.verify(&hash(0, 0xff)),
                Err(Error::FooterProofMismatch)
            );

            // Missing or extra path entries.
            let mut tampered = footer.clone();
            tampered.proof.pop();
            assert!(matches!(
                tampered.verify(&block_id),
                Err(Error::FooterProofLength { .. })
            ));
            let mut tampered = footer.clone();
            tampered.proof.push([0; 20]);
            assert!(matches!(
                tampered.verify(&block_id),
                Err(Error::FooterProofLength { .. })
            ));
        }
    }

    #[test]
    fn test_fec_set_root_cannot_pose_as_final_leaf() {
        // An FEC set root at an index other than `fec_set_count` has a valid
        // path to the block id, but no footer proof can claim it: the leaf is
        // recomputed from the claimed fields and the index is the count.
        let fec_set_count = 6;
        let (block_id, footer, tree, leaves) = ledger_block(fec_set_count);
        let index = 3;
        let proof: Vec<MerkleProofEntry> = tree
            .make_merkle_proof(index, leaves.len())
            .map(|entry| *entry.unwrap())
            .collect();
        assert_eq!(
            merkle_root(index as u64, &leaves[index], &proof),
            Some(block_id)
        );
        let mut tampered = footer.clone();
        tampered.proof = proof;
        assert_eq!(tampered.verify(&block_id), Err(Error::FooterProofMismatch));
        // Claiming the FEC set's index as the count.
        tampered.fec_set_count = index as u32;
        assert!(tampered.verify(&block_id).is_err());
    }

    #[test]
    fn test_invalid_fec_set_count() {
        let (block_id, footer, ..) = ledger_block(3);
        for fec_set_count in [0, MAX_FEC_SETS_PER_SLOT + 1, u32::MAX] {
            let mut tampered = footer.clone();
            tampered.fec_set_count = fec_set_count;
            assert_eq!(
                tampered.verify(&block_id),
                Err(Error::InvalidFecSetCount(fec_set_count))
            );
        }
    }

    #[test]
    fn test_merkle_root_rejects_short_proof() {
        assert_eq!(merkle_root(4, &hash(0, 0), &[[0; 20]; 2]), None);
        assert!(merkle_root(3, &hash(0, 0), &[[0; 20]; 2]).is_some());
    }
}
