//! Epoch Boundary Commitment (draft SIMD): slot 4 of the commitment vector.
//!
//! Every block `B` of epoch `E` commits to `epoch_boundary(B)`, built from
//! `F`, the first block of `E` on `B`'s chain, and `P`, the parent of `F`
//! (see [`solana_merkle_tree::epoch_boundary`] for the encoding).
//!
//! # State
//!
//! Each bank holds the [`EpochBoundary`] inputs of its value. They are set
//! when the bank is frozen, in [`Bank::compute_epoch_boundary`]:
//!
//! - A bank whose parent is in an earlier epoch is `F` (the same test as the
//!   `parent_epoch < epoch` check that starts epoch processing when the bank
//!   is created). It records its own slot and its parent's slot, block id,
//!   commitment root and its own history length. Skipped slots at the start
//!   of an epoch make a later slot `F`; skipped slots at the end of the
//!   previous epoch make an earlier block `P`. Each fork has its own `F`.
//! - Any other bank copies its parent's state.
//!
//! The computation happens at freeze rather than at bank creation because
//! `P`'s block id may only be known after the child bank was created (leader
//! banks get their block id from broadcast); it uses the same parent and
//! history inputs as the block history leaf of the base SIMD. The state is
//! kept in snapshots (it cannot be recomputed from account state).
//!
//! # Edge cases
//!
//! - Feature inactive for the bank (either `block_commitment_root` or
//!   `epoch_boundary_commitment`): no state, slot 4 is zero.
//! - No parent: the genesis bank, whose epoch has no previous epoch. It has
//!   no state, so no block of the genesis epoch has one either; slot 4 is
//!   zero until the first block of the next epoch. (Snapshot banks have no
//!   parent either but are frozen when loaded and keep the snapshot value.)
//! - Feature activation: features activate in the first bank of an epoch,
//!   which is `F` and computes the value. A bank whose parent is in the same
//!   epoch but has no state (activation in the middle of an epoch, which the
//!   SIMD forbids and only tests can do) has none either.
//! - `P` without a commitment root (`block_commitment_root` activated at the
//!   same boundary): `P.commitment_root` is zero and `history_len(F)` is 1:
//!   the block history of `F` is the anchor leaf of `P` alone, which is `P`'s
//!   leaf (with a zero commitment root) at index 0, as in every other case.
//! - `P` with `block_commitment_root` active but without a commitment root
//!   (a commitment failure of its own, or no block commitment state at all,
//!   as after loading a snapshot written before the field existed): a
//!   commitment failure, never the zero commitment root, which stands only
//!   for a `P` that predates the feature.
//! - `P` without a block id: a commitment failure. The block history of `F`
//!   (its last leaf, or its anchor leaf) needs the same block id and fails
//!   first, with `MissingParentBlockId`.

use {
    super::{Bank, block_commitment::CommitmentFailure},
    agave_feature_set as feature_set,
    solana_hash::Hash,
    solana_merkle_tree::epoch_boundary::EpochBoundary,
};

impl Bank {
    /// Whether slot 4 of the commitment vector carries the epoch boundary:
    /// both `block_commitment_root` and `epoch_boundary_commitment` are
    /// active for this bank.
    pub fn is_epoch_boundary_commitment_active(&self) -> bool {
        self.is_block_commitment_root_active()
            && self
                .feature_set
                .is_active(&feature_set::epoch_boundary_commitment::id())
    }

    /// Epoch boundary state, set when the bank is frozen with the feature
    /// active (and the bank has a boundary), or loaded from a snapshot.
    pub fn epoch_boundary(&self) -> Option<EpochBoundary> {
        *self.epoch_boundary.read().unwrap()
    }

    /// Epoch boundary state written to a snapshot of this bank: `None` unless
    /// the feature is in effect for the bank, so that a snapshot taken before
    /// that has the format it had before the field existed.
    pub(crate) fn epoch_boundary_for_snapshot(&self) -> Option<EpochBoundary> {
        self.epoch_boundary()
            .filter(|_| self.is_epoch_boundary_commitment_active())
    }

    /// The epoch boundary of this bank, given its parent and the length of
    /// its block history `H(B)`. Used for commitment slot 4 and for the state
    /// stored at freeze, with the same inputs. Fails if this bank is the
    /// first of its epoch and its parent has no block id, or has
    /// `block_commitment_root` active but no commitment root.
    pub(super) fn compute_epoch_boundary(
        &self,
        parent: Option<&Bank>,
        history_len: u64,
    ) -> Result<Option<EpochBoundary>, CommitmentFailure> {
        if !self.is_epoch_boundary_commitment_active() {
            return Ok(None);
        }
        let Some(parent) = parent else {
            return Ok(None);
        };
        if parent.epoch() < self.epoch() {
            // This bank is `F`, the first block of its epoch on this fork, and
            // its parent is `P`. Every frozen bank that a child is built on
            // has a block id in a running validator (see
            // `block_history_for_commitment`); without it the boundary, and
            // so this bank's commitment root, cannot be computed.
            let parent_slot = parent.slot();
            let parent_block_id = parent
                .block_id()
                .ok_or(CommitmentFailure::EpochBoundaryMissingParentBlockId { parent_slot })?;
            // `P.commitment_root` is 32 zero bytes only for a `P` that
            // predates Block Commitment Root. A `P` with the feature active
            // has one, or a commitment failure; one with neither (no block
            // commitment state, e.g. loaded from an old snapshot) is a
            // failure as well, not a zero commitment root.
            let parent_commitment_root = if parent.is_block_commitment_root_active() {
                match parent.commitment_root() {
                    Some(commitment_root) => commitment_root,
                    None if parent.commitment_failure().is_some() => {
                        return Err(CommitmentFailure::ParentCommitmentFailure { parent_slot });
                    }
                    None => {
                        return Err(CommitmentFailure::MissingParentCommitment { parent_slot });
                    }
                }
            } else {
                Hash::default()
            };
            Ok(Some(EpochBoundary {
                epoch: self.epoch(),
                first_slot: self.slot(),
                parent_slot,
                parent_block_id: parent_block_id.to_hash(),
                parent_commitment_root,
                history_len,
            }))
        } else {
            let epoch_boundary = parent.epoch_boundary();
            debug_assert!(
                epoch_boundary.is_none_or(|epoch_boundary| epoch_boundary.epoch == self.epoch())
            );
            Ok(epoch_boundary)
        }
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        crate::{
            bank::{
                SlotLeader,
                block_commitment::{
                    COMMITMENT_VECTOR_LEN, CommitmentSlot, CommitmentVector,
                    INACTIVE_COMMITMENT_SLOT, history_leaf, tests::assert_commitment_failure,
                },
                test_utils::goto_end_of_slot,
            },
            genesis_utils::{activate_feature, create_genesis_config},
        },
        agave_votor_messages::consensus_message::BlockId,
        solana_clock::Slot,
        solana_epoch_schedule::EpochSchedule,
        solana_genesis_config::GenesisConfig,
        solana_merkle_tree::commitment_tree::{TreeId, compute_root, make_inclusion_proof},
        solana_pubkey::Pubkey,
        solana_sha256_hasher::hashv,
        std::sync::Arc,
    };

    fn genesis(features: &[Pubkey]) -> GenesisConfig {
        let mut genesis_config = create_genesis_config(1_000_000_000_000).genesis_config;
        genesis_config.epoch_schedule = EpochSchedule::custom(32, 32, false);
        for feature_id in features {
            activate_feature(&mut genesis_config, *feature_id);
        }
        genesis_config
    }

    fn both_features() -> [Pubkey; 2] {
        [
            feature_set::block_commitment_root::id(),
            feature_set::epoch_boundary_commitment::id(),
        ]
    }

    fn new_child(parent: &Arc<Bank>, slot: Slot) -> Arc<Bank> {
        Arc::new(Bank::new_from_parent(
            parent.clone(),
            SlotLeader::default(),
            slot,
        ))
    }

    fn freeze_with_block_id(bank: &Arc<Bank>) {
        goto_end_of_slot(bank.clone());
        bank.freeze();
        bank.set_block_id(Some(BlockId::from(Hash::new_unique())));
    }

    fn child(parent: &Arc<Bank>, slot: Slot) -> Arc<Bank> {
        let bank = new_child(parent, slot);
        freeze_with_block_id(&bank);
        bank
    }

    /// Independent `epoch_boundary` hash, from the SIMD text.
    fn expected_hash(epoch: u64, first_slot: Slot, parent: &Bank, history_len: u64) -> Hash {
        hashv(&[
            &[0x83],
            &epoch.to_le_bytes(),
            &first_slot.to_le_bytes(),
            &parent.slot().to_le_bytes(),
            parent.block_id().unwrap().to_hash().as_ref(),
            parent.commitment_root().unwrap_or_default().as_ref(),
            &history_len.to_le_bytes(),
        ])
    }

    /// Slot 4 of a frozen bank, checked against its commitment root through
    /// an independently rebuilt vector.
    fn slot4(bank: &Bank) -> Hash {
        let commitment = bank.block_commitment().expect("bank has a commitment");
        let mut vector = [INACTIVE_COMMITMENT_SLOT; COMMITMENT_VECTOR_LEN];
        vector[CommitmentSlot::BlockHistoryRoot.index()] = commitment.block_history_root();
        vector[CommitmentSlot::EpochBoundary.index()] = bank
            .epoch_boundary()
            .map(|epoch_boundary| epoch_boundary.hash())
            .unwrap_or(INACTIVE_COMMITMENT_SLOT);
        let vector = CommitmentVector(vector);
        assert_eq!(vector.root(), commitment.commitment_root);
        vector.get(CommitmentSlot::EpochBoundary)
    }

    #[test]
    fn test_inactive_features() {
        for features in [
            vec![],
            vec![feature_set::block_commitment_root::id()],
            vec![feature_set::epoch_boundary_commitment::id()],
        ] {
            let bank0 = Arc::new(Bank::new_for_tests(&genesis(&features)));
            freeze_with_block_id(&bank0);
            let bank32 = child(&bank0, 32);
            assert!(!bank32.is_epoch_boundary_commitment_active());
            assert_eq!(bank32.epoch_boundary(), None);
            if bank32.is_block_commitment_root_active() {
                assert_eq!(slot4(&bank32), INACTIVE_COMMITMENT_SLOT);
            } else {
                assert_eq!(bank32.block_commitment(), None);
            }
        }
    }

    #[test]
    fn test_epoch_boundary_for_snapshot() {
        let bank0 = Arc::new(Bank::new_for_tests(&genesis(&both_features())));
        freeze_with_block_id(&bank0);
        // The genesis epoch has no boundary, so nothing to write.
        assert_eq!(bank0.epoch_boundary_for_snapshot(), None);
        let bank32 = Bank::new_from_parent(bank0.clone(), SlotLeader::default(), 32);
        bank32.freeze();
        bank32.set_block_id(Some(BlockId::from(Hash::new_unique())));
        let boundary = bank32.epoch_boundary().unwrap();
        assert_eq!(bank32.epoch_boundary_for_snapshot(), Some(boundary));
        assert_eq!(
            bank32.get_fields_to_serialize().epoch_boundary,
            Some(boundary)
        );
        // State held while the feature is not in effect (which only a test
        // can produce) is not written to a snapshot, whichever feature is
        // missing.
        for feature_id in both_features() {
            let mut bank = Bank::new_from_parent(bank0.clone(), SlotLeader::default(), 32);
            bank.freeze();
            bank.set_block_id(Some(BlockId::from(Hash::new_unique())));
            assert_eq!(bank.epoch_boundary(), Some(boundary));
            bank.deactivate_feature(&feature_id);
            assert_eq!(bank.epoch_boundary(), Some(boundary));
            assert_eq!(bank.epoch_boundary_for_snapshot(), None);
            assert_eq!(bank.get_fields_to_serialize().epoch_boundary, None);
        }
    }

    #[test]
    fn test_genesis_epoch_has_no_boundary() {
        let bank0 = Arc::new(Bank::new_for_tests(&genesis(&both_features())));
        assert!(bank0.is_epoch_boundary_commitment_active());
        freeze_with_block_id(&bank0);
        assert_eq!(bank0.epoch_boundary(), None);
        assert_eq!(slot4(&bank0), INACTIVE_COMMITMENT_SLOT);
        let bank31 = child(&child(&bank0, 5), 31);
        assert_eq!(bank31.epoch_boundary(), None);
        assert_eq!(slot4(&bank31), INACTIVE_COMMITMENT_SLOT);
        // The first block of epoch 1 has one.
        let bank32 = child(&bank31, 32);
        assert_eq!(
            slot4(&bank32),
            expected_hash(
                1,
                32,
                &bank31,
                bank32
                    .block_commitment()
                    .unwrap()
                    .block_history
                    .leaf_count()
            )
        );
    }

    #[test]
    fn test_first_slots_skipped_and_constant_within_epoch() {
        let bank0 = Arc::new(Bank::new_for_tests(&genesis(&both_features())));
        freeze_with_block_id(&bank0);
        let bank1 = child(&bank0, 1);
        // Slot 31 (last of epoch 0) and slots 32 to 34 (first of epoch 1)
        // are skipped.
        let bank30 = child(&bank1, 30);
        let bank35 = child(&bank30, 35);
        let boundary = bank35.epoch_boundary().unwrap();
        assert_eq!(
            boundary,
            EpochBoundary {
                epoch: 1,
                first_slot: 35,
                parent_slot: 30,
                parent_block_id: bank30.block_id().unwrap().to_hash(),
                parent_commitment_root: bank30.commitment_root().unwrap(),
                history_len: 3,
            }
        );
        assert_eq!(slot4(&bank35), expected_hash(1, 35, &bank30, 3));

        // `P`'s history leaf is the last of `history_len(F)` leaves.
        let leaves: Vec<_> = [&bank0, &bank1, &bank30]
            .iter()
            .map(|bank| {
                history_leaf(
                    bank.slot(),
                    &bank.block_id().unwrap().to_hash(),
                    &bank.commitment_root().unwrap(),
                )
            })
            .collect();
        let history_root = bank35.block_commitment().unwrap().block_history_root();
        assert_eq!(compute_root(TreeId::BLOCK_HISTORY, &leaves), history_root);
        let index = boundary.parent_history_index().unwrap() as usize;
        let proof = make_inclusion_proof(TreeId::BLOCK_HISTORY, &leaves, index).unwrap();
        assert!(proof.verify(
            TreeId::BLOCK_HISTORY,
            &history_leaf(
                boundary.parent_slot,
                &boundary.parent_block_id,
                &boundary.parent_commitment_root
            ),
            &history_root
        ));

        // Every later block of epoch 1 copies it.
        let mut bank = bank35;
        for slot in [36, 40, 63] {
            bank = child(&bank, slot);
            assert_eq!(bank.epoch_boundary(), Some(boundary), "slot {slot}");
            assert_eq!(slot4(&bank), boundary.hash(), "slot {slot}");
        }
        // The next epoch has a new one.
        let bank64 = child(&bank, 64);
        let next = bank64.epoch_boundary().unwrap();
        assert_eq!(next.epoch, 2);
        assert_eq!(next.first_slot, 64);
        assert_eq!(next.parent_slot, 63);
        assert_ne!(next.hash(), boundary.hash());
    }

    #[test]
    fn test_forks_crossing_at_different_slots() {
        let bank0 = Arc::new(Bank::new_for_tests(&genesis(&both_features())));
        freeze_with_block_id(&bank0);
        let bank30 = child(&bank0, 30);
        // Same `P`, different `F`.
        let fork_a = child(&bank30, 32);
        let fork_b = child(&bank30, 33);
        assert_ne!(slot4(&fork_a), slot4(&fork_b));
        assert_eq!(fork_a.epoch_boundary().unwrap().first_slot, 32);
        assert_eq!(fork_b.epoch_boundary().unwrap().first_slot, 33);
        // Different `P`.
        let bank31 = child(&bank30, 31);
        let fork_c = child(&bank31, 34);
        assert_eq!(fork_c.epoch_boundary().unwrap().parent_slot, 31);
        assert_ne!(slot4(&fork_c), slot4(&fork_b));
        // Each fork keeps its own value.
        let a40 = child(&fork_a, 40);
        let b41 = child(&fork_b, 41);
        let c42 = child(&fork_c, 42);
        assert_eq!(slot4(&a40), slot4(&fork_a));
        assert_eq!(slot4(&b41), slot4(&fork_b));
        assert_eq!(slot4(&c42), slot4(&fork_c));
    }

    #[test]
    fn test_block_count_of_previous_epoch() {
        let bank0 = Arc::new(Bank::new_for_tests(&genesis(&both_features())));
        freeze_with_block_id(&bank0);
        let mut bank = child(&bank0, 3);
        bank = child(&bank, 31);
        // Epoch 1: blocks at 33, 34, 40, 50 and 63.
        let epoch1_slots = [33, 34, 40, 50, 63];
        let mut epoch1_boundary = None;
        for slot in epoch1_slots {
            bank = child(&bank, slot);
            epoch1_boundary.get_or_insert(bank.epoch_boundary().unwrap());
        }
        let epoch1_boundary = epoch1_boundary.unwrap();
        assert_eq!(epoch1_boundary.first_slot, 33);
        // Epoch 2: slots 64 and 65 skipped.
        let bank66 = child(&bank, 66);
        let epoch2_boundary = bank66.epoch_boundary().unwrap();
        assert_eq!(epoch2_boundary.parent_slot, 63);
        assert_eq!(
            epoch1_boundary.blocks_until(&epoch2_boundary),
            Some(epoch1_slots.len() as u64)
        );

        // Epoch 3 (96..128) has no block: epoch 4's boundary has its `P` in
        // epoch 2, and the count covers epochs 2 and 3.
        let bank70 = child(&bank66, 70);
        let bank130 = child(&bank70, 130);
        let epoch4_boundary = bank130.epoch_boundary().unwrap();
        assert_eq!(epoch4_boundary.epoch, 4);
        assert_eq!(epoch4_boundary.parent_slot, 70);
        assert_eq!(epoch2_boundary.blocks_until(&epoch4_boundary), Some(2));
    }

    #[test]
    fn test_proof_from_commitment_root_to_slot4() {
        let bank0 = Arc::new(Bank::new_for_tests(&genesis(&both_features())));
        freeze_with_block_id(&bank0);
        let bank33 = child(&child(&bank0, 20), 33);
        let bank40 = child(&bank33, 40);
        let commitment_root = bank40.commitment_root().unwrap();
        let boundary = bank40.epoch_boundary().unwrap();

        let mut vector = [INACTIVE_COMMITMENT_SLOT; COMMITMENT_VECTOR_LEN];
        vector[CommitmentSlot::BlockHistoryRoot.index()] =
            bank40.block_commitment().unwrap().block_history_root();
        vector[CommitmentSlot::EpochBoundary.index()] =
            expected_hash(1, 33, &bank33.parent().unwrap(), boundary.history_len);
        let vector = CommitmentVector(vector);
        assert_eq!(vector.root(), commitment_root);
        let proof = vector
            .inclusion_proof(CommitmentSlot::EpochBoundary.index())
            .unwrap();
        assert!(proof.verify(
            TreeId::COMMITMENT_VECTOR,
            boundary.hash().as_ref(),
            &commitment_root
        ));
        // A boundary for another epoch, or another first slot, does not open.
        for wrong in [
            EpochBoundary {
                epoch: 2,
                ..boundary
            },
            EpochBoundary {
                first_slot: 32,
                ..boundary
            },
        ] {
            assert!(!proof.verify(
                TreeId::COMMITMENT_VECTOR,
                wrong.hash().as_ref(),
                &commitment_root
            ));
        }
    }

    #[test]
    fn test_activation_at_epoch_boundary() {
        // Block Commitment Root active from genesis, the epoch boundary
        // feature from the first block of epoch 1.
        let bank0 = Arc::new(Bank::new_for_tests(&genesis(&[
            feature_set::block_commitment_root::id(),
        ])));
        freeze_with_block_id(&bank0);
        let bank10 = child(&bank0, 10);
        assert_eq!(slot4(&bank10), INACTIVE_COMMITMENT_SLOT);
        let mut bank32 = Bank::new_from_parent(bank10.clone(), SlotLeader::default(), 32);
        bank32.activate_feature(&feature_set::epoch_boundary_commitment::id());
        let bank32 = Arc::new(bank32);
        freeze_with_block_id(&bank32);
        let boundary = bank32.epoch_boundary().unwrap();
        assert_eq!(boundary.parent_slot, 10);
        assert_eq!(
            boundary.parent_commitment_root,
            bank10.commitment_root().unwrap()
        );
        assert_eq!(boundary.history_len, 2);
        assert_eq!(slot4(&bank32), expected_hash(1, 32, &bank10, 2));
        let bank33 = child(&bank32, 33);
        assert_eq!(slot4(&bank33), boundary.hash());
    }

    #[test]
    fn test_activation_mid_epoch_has_no_boundary() {
        // Not allowed by the SIMD (features activate at epoch boundaries);
        // the bank cannot know `F` and commits to nothing.
        let bank0 = Arc::new(Bank::new_for_tests(&genesis(&[
            feature_set::block_commitment_root::id(),
        ])));
        freeze_with_block_id(&bank0);
        let bank32 = child(&bank0, 32);
        let mut bank33 = Bank::new_from_parent(bank32.clone(), SlotLeader::default(), 33);
        bank33.activate_feature(&feature_set::epoch_boundary_commitment::id());
        let bank33 = Arc::new(bank33);
        freeze_with_block_id(&bank33);
        assert!(bank33.is_epoch_boundary_commitment_active());
        assert_eq!(bank33.epoch_boundary(), None);
        assert_eq!(slot4(&bank33), INACTIVE_COMMITMENT_SLOT);
    }

    #[test]
    fn test_parent_without_commitment_root() {
        // Both features activate in the first block of epoch 1: `P` has no
        // commitment root, and the history of `F` is the anchor leaf of `P`
        // alone, at index 0.
        let bank0 = Arc::new(Bank::new_for_tests(&genesis(&[])));
        freeze_with_block_id(&bank0);
        let bank5 = child(&bank0, 5);
        assert_eq!(bank5.commitment_root(), None);
        let mut bank32 = Bank::new_from_parent(bank5.clone(), SlotLeader::default(), 32);
        for feature_id in both_features() {
            bank32.activate_feature(&feature_id);
        }
        let bank32 = Arc::new(bank32);
        freeze_with_block_id(&bank32);
        let boundary = bank32.epoch_boundary().unwrap();
        assert_eq!(
            boundary,
            EpochBoundary {
                epoch: 1,
                first_slot: 32,
                parent_slot: 5,
                parent_block_id: bank5.block_id().unwrap().to_hash(),
                parent_commitment_root: Hash::default(),
                history_len: 1,
            }
        );
        assert_eq!(slot4(&bank32), expected_hash(1, 32, &bank5, 1));

        // `P`'s leaf, rebuilt from the boundary, is the anchor leaf at index
        // 0 of the history of `F`.
        assert_eq!(boundary.parent_history_index(), Some(0));
        let anchor = history_leaf(5, &bank5.block_id().unwrap().to_hash(), &Hash::default());
        let history_root = bank32.block_commitment().unwrap().block_history_root();
        assert_eq!(compute_root(TreeId::BLOCK_HISTORY, &[anchor]), history_root);
        let proof = make_inclusion_proof(TreeId::BLOCK_HISTORY, &[anchor], 0).unwrap();
        assert!(proof.verify(
            TreeId::BLOCK_HISTORY,
            &history_leaf(
                boundary.parent_slot,
                &boundary.parent_block_id,
                &boundary.parent_commitment_root
            ),
            &history_root
        ));

        // The count of blocks of epoch 1 includes `F`, from the anchor leaf on.
        let mut bank = bank32;
        for slot in [40, 50, 63] {
            bank = child(&bank, slot);
        }
        let bank64 = child(&bank, 64);
        assert_eq!(
            boundary.blocks_until(&bank64.epoch_boundary().unwrap()),
            Some(4)
        );
    }

    /// `F` whose parent `P` has no block id cannot compute its block history
    /// (the anchor leaf of `P`) or its epoch boundary: a commitment failure
    /// instead of the zero block id. The history fails first.
    #[test]
    fn test_parent_without_block_id_is_commitment_failure() {
        let bank0 = Arc::new(Bank::new_for_tests(&genesis(&[])));
        freeze_with_block_id(&bank0);
        let bank5 = new_child(&bank0, 5);
        goto_end_of_slot(bank5.clone());
        bank5.freeze();
        assert_eq!(bank5.block_id(), None);
        let mut bank32 = Bank::new_from_parent(bank5.clone(), SlotLeader::default(), 32);
        for feature_id in both_features() {
            bank32.activate_feature(&feature_id);
        }
        let bank32 = Arc::new(bank32);
        freeze_with_block_id(&bank32);
        assert_eq!(bank32.epoch_boundary(), None);
        assert_commitment_failure(
            &bank32,
            CommitmentFailure::MissingParentBlockId { parent_slot: 5 },
        );
        // The epoch boundary on its own fails as well.
        assert_eq!(
            bank32.compute_epoch_boundary(Some(&bank5), 1),
            Err(CommitmentFailure::EpochBoundaryMissingParentBlockId { parent_slot: 5 })
        );

        // Within the epoch, the boundary is copied from the parent, whose
        // block id is not needed; but the parent itself has no commitment.
        let bank33 = new_child(&bank32, 33);
        freeze_with_block_id(&bank33);
        assert_commitment_failure(
            &bank33,
            CommitmentFailure::ParentCommitmentFailure { parent_slot: 32 },
        );
    }

    /// `P` with Block Commitment Root active but no commitment root does not
    /// stand for a zero commitment root: the epoch boundary of `F` is a
    /// commitment failure, as its block history is.
    #[test]
    fn test_active_parent_without_commitment_root_is_commitment_failure() {
        let bank0 = Arc::new(Bank::new_for_tests(&genesis(&both_features())));
        freeze_with_block_id(&bank0);
        let bank5 = child(&bank0, 5);

        // No block commitment state and no failure, as for a bank loaded
        // from a snapshot written before the field existed.
        *bank5.block_commitment.write().unwrap() = None;
        assert!(bank5.is_block_commitment_root_active());
        assert_eq!(bank5.commitment_failure(), None);
        let bank32 = child(&bank5, 32);
        assert_eq!(bank32.epoch_boundary(), None);
        assert_commitment_failure(
            &bank32,
            CommitmentFailure::MissingParentCommitment { parent_slot: 5 },
        );
        assert_eq!(
            bank32.compute_epoch_boundary(Some(&bank5), 2),
            Err(CommitmentFailure::MissingParentCommitment { parent_slot: 5 })
        );

        // `P` with a commitment failure of its own.
        let bank31 = child(&bank5, 31);
        assert_eq!(
            bank31.commitment_failure(),
            Some(CommitmentFailure::MissingParentCommitment { parent_slot: 5 })
        );
        let bank33 = child(&bank31, 33);
        assert_commitment_failure(
            &bank33,
            CommitmentFailure::ParentCommitmentFailure { parent_slot: 31 },
        );
        assert_eq!(
            bank33.compute_epoch_boundary(Some(&bank31), 3),
            Err(CommitmentFailure::ParentCommitmentFailure { parent_slot: 31 })
        );
    }

    #[test]
    fn test_warp_across_epochs() {
        // A bank created by tooling that crosses epochs is `F`.
        let bank0 = Arc::new(Bank::new_for_tests(&genesis(&both_features())));
        freeze_with_block_id(&bank0);
        let warped = Bank::warp_from_parent(bank0.clone(), SlotLeader::default(), 100);
        let boundary = warped.epoch_boundary().unwrap();
        assert_eq!(boundary.epoch, 3);
        assert_eq!(boundary.first_slot, 100);
        assert_eq!(boundary.parent_slot, 0);
        assert_eq!(boundary.history_len, 1);
    }
}
