//! The `epoch_boundary` commitment of the "Epoch Boundary Commitment" SIMD.
//!
//! For a block `B` in epoch `E`, let `F` be the first block of epoch `E` on
//! `B`'s chain (`F` may be `B`) and `P` the parent of `F`, the last block
//! before epoch `E`:
//!
//! ```text
//! epoch_boundary(B) = sha256( 0x83
//!                          || u64_le(E)
//!                          || u64_le(F.slot)
//!                          || u64_le(P.slot)
//!                          || P.block_id
//!                          || P.commitment_root
//!                          || u64_le(history_len(F)) )
//! ```
//!
//! `history_len(F)` is the number of leaves in the block history tree of
//! `F`; `P`'s history leaf is the last of them, at index
//! `history_len(F) - 1`. `P.commitment_root` is 32 zero bytes if `P` predates
//! Block Commitment Root, in which case `history_len(F)` is 1: the history of
//! `F` is the anchor leaf of `P` alone (`P`'s leaf with a zero commitment
//! root), at index 0.

use {
    crate::commitment_tree::EPOCH_BOUNDARY_PREFIX, solana_hash::Hash, solana_sha256_hasher::hashv,
};

/// The inputs of `epoch_boundary`: where the epoch before `epoch` ended on a
/// chain and where `epoch` began.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct EpochBoundary {
    /// `E`, the epoch of every block carrying this value.
    pub epoch: u64,
    /// `F.slot`: slot of the first block of `E` on the chain.
    pub first_slot: u64,
    /// `P.slot`: slot of the parent of `F`, the last block before `E`.
    pub parent_slot: u64,
    /// `P.block_id`.
    pub parent_block_id: Hash,
    /// `P.commitment_root`, or 32 zero bytes if `P` predates Block Commitment
    /// Root.
    pub parent_commitment_root: Hash,
    /// `history_len(F)`: number of leaves in the block history tree of `F`.
    pub history_len: u64,
}

impl EpochBoundary {
    /// `epoch_boundary`, the value of commitment slot 4.
    pub fn hash(&self) -> Hash {
        hashv(&[
            &[EPOCH_BOUNDARY_PREFIX],
            &self.epoch.to_le_bytes(),
            &self.first_slot.to_le_bytes(),
            &self.parent_slot.to_le_bytes(),
            self.parent_block_id.as_ref(),
            self.parent_commitment_root.as_ref(),
            &self.history_len.to_le_bytes(),
        ])
    }

    /// Index of `P`'s leaf in the block history tree of `F` (the anchor leaf,
    /// at index 0, if `P` predates Block Commitment Root). `None` only for a
    /// value with `history_len` 0, which no block with a parent has.
    pub fn parent_history_index(&self) -> Option<u64> {
        self.history_len.checked_sub(1)
    }

    /// Number of blocks of a chain from `F` of `self` (inclusive) to `F` of
    /// `next` (exclusive), i.e. the blocks of `self.epoch` and of any empty
    /// epochs up to `next.epoch`, when both are taken from the same chain
    /// with Block Commitment Root active at or before `self`'s `F` (if it
    /// activated at that boundary, the anchor leaf of `self`'s `P` counts in
    /// both). `None` if `next` is not later.
    pub fn blocks_until(&self, next: &EpochBoundary) -> Option<u64> {
        if next.epoch <= self.epoch {
            return None;
        }
        next.history_len.checked_sub(self.history_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash_from_hex(hex: &str) -> Hash {
        let bytes: [u8; 32] = hex::decode(hex).unwrap().try_into().unwrap();
        Hash::new_from_array(bytes)
    }

    #[test]
    fn test_vectors() {
        // Computed with an independent Python implementation (hashlib.sha256).
        let boundary = EpochBoundary {
            epoch: 7,
            first_slot: 230,
            parent_slot: 223,
            parent_block_id: Hash::new_from_array([0x11; 32]),
            parent_commitment_root: Hash::new_from_array([0x22; 32]),
            history_len: 1234,
        };
        assert_eq!(
            boundary.hash(),
            hash_from_hex("43a129530f31e4a85dd0eff3057303226e94db10b80d0725d472c4b0a787ce3c")
        );
        assert_eq!(boundary.parent_history_index(), Some(1233));

        // P predating Block Commitment Root: zero commitment root, and the
        // history of F is P's anchor leaf alone.
        let boundary = EpochBoundary {
            epoch: 1,
            first_slot: 32,
            parent_slot: 0,
            parent_block_id: Hash::new_from_array([0x33; 32]),
            parent_commitment_root: Hash::default(),
            history_len: 1,
        };
        assert_eq!(
            boundary.hash(),
            hash_from_hex("58cfe62e7b103fe269adbe705f86b151b424ccf68096cfa093a2a0413cdcdaf1")
        );
        assert_eq!(boundary.parent_history_index(), Some(0));

        // A value without history leaves (no block with a parent has one).
        let boundary = EpochBoundary {
            history_len: 0,
            ..boundary
        };
        assert_eq!(
            boundary.hash(),
            hash_from_hex("c5c498cdf30ff01e07ccc3f25fa1d61d6382f1acea8ba18f9c0b7ee3a8b87e3f")
        );
        assert_eq!(boundary.parent_history_index(), None);
    }

    #[test]
    fn test_every_field_is_bound() {
        let base = EpochBoundary {
            epoch: 3,
            first_slot: 100,
            parent_slot: 95,
            parent_block_id: Hash::new_from_array([1; 32]),
            parent_commitment_root: Hash::new_from_array([2; 32]),
            history_len: 50,
        };
        let variants = [
            EpochBoundary { epoch: 4, ..base },
            EpochBoundary {
                first_slot: 101,
                ..base
            },
            EpochBoundary {
                parent_slot: 96,
                ..base
            },
            EpochBoundary {
                parent_block_id: Hash::new_from_array([3; 32]),
                ..base
            },
            EpochBoundary {
                parent_commitment_root: Hash::new_from_array([4; 32]),
                ..base
            },
            EpochBoundary {
                history_len: 51,
                ..base
            },
        ];
        for variant in variants {
            assert_ne!(variant.hash(), base.hash(), "{variant:?}");
        }
    }

    #[test]
    fn test_blocks_until() {
        let first = EpochBoundary {
            epoch: 3,
            history_len: 50,
            ..EpochBoundary::default()
        };
        let next = EpochBoundary {
            epoch: 4,
            history_len: 80,
            ..EpochBoundary::default()
        };
        assert_eq!(first.blocks_until(&next), Some(30));
        assert_eq!(next.blocks_until(&first), None);
        assert_eq!(first.blocks_until(&first), None);
    }
}
