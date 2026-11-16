//! Epoch Validator Set Root (draft SIMD): slot 3 of the commitment vector.
//!
//! Every block of epoch `E` commits to `validator_set_root(E + 1)`, the root
//! of the signer set that verifies Alpenglow certificates for slots of epoch
//! `E + 1` (see [`solana_merkle_tree::validator_set`] for the encoding).
//!
//! # Data source
//!
//! The set is taken from this bank's `epoch_stakes(E + 1)`, through the same
//! [`BLSPubkeyToRankMap`](crate::epoch_stakes::BLSPubkeyToRankMap) that
//! certificate verification uses for slots of `E + 1`
//! ([`Bank::epoch_stakes_from_slot`], [`Bank::verify_certificate`]). That
//! entry is inserted once per fork, by the first bank whose leader schedule
//! epoch reaches `E + 1` (the first bank of `E` when the leader schedule
//! offset is one epoch, as on mainnet), is never replaced, and is inherited
//! by every descendant, including the banks of `E + 1` that verify
//! certificates with it.
//!
//! Leaves are in rank order: stake descending, then compressed BLS public
//! key ascending. Validators without a rank (zero stake, no or invalid BLS
//! key, duplicate BLS key or identity) have no leaf, and `total_stake` is the
//! rank map's total, the certificate threshold denominator.
//!
//! # Caching
//!
//! The signer tree (everything but the epoch number) is computed at most once
//! per epoch stakes entry and cached in it
//! ([`VersionedEpochStakes::signer_set_commitment`]): eagerly when the entry
//! is inserted at the epoch boundary, otherwise (e.g. after a snapshot load,
//! where the cache is not serialized) by the first bank that freezes with
//! it. Each block then costs one SHA-256 for the epoch binding.
//!
//! # Edge cases
//!
//! - Stakes for `E + 1` not yet available (leader schedule offset shorter
//!   than an epoch): slot 3 is zero, as the SIMD requires.
//! - No validator with a rank: the root of the empty set, with total stake 0.
//! - Genesis: the genesis bank holds epoch stakes up to its leader schedule
//!   epoch, so it commits to `validator_set_root(1)` whenever that epoch's
//!   stakes exist; warmup epochs fix `E + 1` at the start of `E` as well.

use {
    super::Bank,
    crate::epoch_stakes::VersionedEpochStakes,
    agave_feature_set as feature_set,
    solana_clock::Epoch,
    solana_hash::Hash,
    solana_merkle_tree::{
        commitment_tree::InclusionProof,
        validator_set::{Signer, signer_inclusion_proof},
    },
};

impl Bank {
    /// Whether slot 3 of the commitment vector carries the validator set
    /// root: both `block_commitment_root` and `epoch_validator_set_root` are
    /// active for this bank.
    pub fn is_epoch_validator_set_root_active(&self) -> bool {
        self.is_block_commitment_root_active()
            && self
                .feature_set
                .is_active(&feature_set::epoch_validator_set_root::id())
    }

    /// `validator_set_root(epoch)` computed from this bank's epoch stakes for
    /// `epoch`, or `None` if this bank has no stakes for that epoch.
    pub fn validator_set_root(&self, epoch: Epoch) -> Option<Hash> {
        self.epoch_stakes(epoch).map(|epoch_stakes| {
            epoch_stakes
                .signer_set_commitment()
                .validator_set_root(epoch)
        })
    }

    /// `validator_set_root(E + 1)` for this bank's epoch `E`: the value of
    /// commitment slot 3 when the feature is active.
    pub fn next_epoch_validator_set_root(&self) -> Option<Hash> {
        self.validator_set_root(self.epoch().checked_add(1)?)
    }

    /// The signer set of `epoch` in rank order (leaf order), from this bank's
    /// epoch stakes. Empty if no validator has a rank; `None` if this bank
    /// has no stakes for `epoch`.
    pub fn validator_set_signers(&self, epoch: Epoch) -> Option<Vec<Signer>> {
        self.epoch_stakes(epoch).map(signers_of)
    }

    /// Membership proof of the signer with rank `rank` in the signer set of
    /// `epoch`: the signer, its proof against `root(0x06, ...)`, and the
    /// total stake bound in `validator_set_root(epoch)`. `None` if this bank
    /// has no stakes for `epoch` or no signer has that rank.
    ///
    /// Verify with [`solana_merkle_tree::validator_set::verify_signer`].
    pub fn validator_set_signer_proof(
        &self,
        epoch: Epoch,
        rank: usize,
    ) -> Option<(Signer, InclusionProof, u64)> {
        let epoch_stakes = self.epoch_stakes(epoch)?;
        let signers = signers_of(epoch_stakes);
        let proof = signer_inclusion_proof(&signers, rank)?;
        let total_stake = epoch_stakes.signer_set_commitment().total_stake;
        Some((signers[rank], proof, total_stake))
    }
}

fn signers_of(epoch_stakes: &VersionedEpochStakes) -> Vec<Signer> {
    epoch_stakes
        .try_bls_pubkey_to_rank_map()
        .map(|rank_map| rank_map.signers())
        .unwrap_or_default()
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
                    INACTIVE_COMMITMENT_SLOT,
                },
                test_utils::goto_end_of_slot,
            },
            epoch_stakes::DeserializableVersionedEpochStakes,
            genesis_utils::{
                ValidatorVoteKeypairs, activate_feature, create_genesis_config,
                create_genesis_config_with_vote_accounts, create_validator,
                minimum_stake_lamports_for_vat, minimum_vote_account_balance_for_vat,
            },
        },
        agave_votor_messages::consensus_message::BlockId,
        solana_epoch_schedule::EpochSchedule,
        solana_genesis_config::GenesisConfig,
        solana_merkle_tree::{
            commitment_tree::{TreeId, compute_root},
            validator_set::{SignerSetCommitment, verify_signer},
        },
        solana_pubkey::Pubkey,
        solana_sha256_hasher::hashv,
        solana_signer::Signer as _,
        std::{collections::HashSet, sync::Arc},
    };

    const LAMPORTS: u64 = 1_000_000_000_000;

    struct TestCluster {
        genesis_config: GenesisConfig,
        /// Vote accounts that must have a rank.
        ranked_vote_pubkeys: Vec<Pubkey>,
        /// Vote accounts with stake but no BLS key.
        unranked_vote_pubkeys: Vec<Pubkey>,
    }

    /// A genesis with five validators with BLS keys, two of them with equal
    /// stake, and one staked validator without a BLS key.
    fn test_cluster(epoch_schedule: EpochSchedule, activate: bool) -> TestCluster {
        let keypairs: Vec<_> = (0..5).map(|_| ValidatorVoteKeypairs::new_rand()).collect();
        let stakes = vec![
            50 * minimum_vote_account_balance_for_vat(100),
            30 * minimum_vote_account_balance_for_vat(100),
            30 * minimum_vote_account_balance_for_vat(100),
            70 * minimum_vote_account_balance_for_vat(100),
            10 * minimum_vote_account_balance_for_vat(100),
        ];
        let mut genesis_config =
            create_genesis_config_with_vote_accounts(LAMPORTS, &keypairs, stakes).genesis_config;
        genesis_config.epoch_schedule = epoch_schedule;
        let no_bls_vote_pubkey = Pubkey::new_unique();
        let rent = genesis_config.rent.clone();
        for (pubkey, account) in create_validator(
            &rent,
            Pubkey::new_unique(),
            1_000_000,
            no_bls_vote_pubkey,
            minimum_vote_account_balance_for_vat(100),
            Pubkey::new_unique(),
            40 * minimum_vote_account_balance_for_vat(100),
            None,
        ) {
            genesis_config.accounts.insert(pubkey, account.into());
        }
        if activate {
            activate_feature(
                &mut genesis_config,
                feature_set::block_commitment_root::id(),
            );
            activate_feature(
                &mut genesis_config,
                feature_set::epoch_validator_set_root::id(),
            );
        }
        TestCluster {
            genesis_config,
            ranked_vote_pubkeys: keypairs.iter().map(|k| k.vote_keypair.pubkey()).collect(),
            unranked_vote_pubkeys: vec![no_bls_vote_pubkey],
        }
    }

    fn new_child(parent: &Arc<Bank>, slot: u64) -> Arc<Bank> {
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

    /// Slot 3 of the commitment vector of a frozen bank, checked against its
    /// commitment root through an independently rebuilt vector.
    fn slot3(bank: &Bank) -> Hash {
        let commitment = bank.block_commitment().expect("bank has a commitment");
        let mut vector = [INACTIVE_COMMITMENT_SLOT; COMMITMENT_VECTOR_LEN];
        vector[CommitmentSlot::BlockHistoryRoot.index()] = commitment.block_history_root();
        if bank.is_epoch_validator_set_root_active() {
            vector[CommitmentSlot::ValidatorSetRoot.index()] = bank
                .next_epoch_validator_set_root()
                .unwrap_or(INACTIVE_COMMITMENT_SLOT);
        }
        let vector = CommitmentVector(vector);
        assert_eq!(vector.root(), commitment.commitment_root);
        vector.get(CommitmentSlot::ValidatorSetRoot)
    }

    /// `validator_set_root(epoch)` computed from the raw epoch stakes of
    /// `bank`, without the rank map or the validator set module: every vote
    /// account in `ranked`, with its delegated stake and the BLS key bytes
    /// stored in its vote account, ordered by stake descending and then key
    /// ascending.
    fn independent_root(bank: &Bank, epoch: Epoch, ranked: &HashSet<Pubkey>) -> Hash {
        let vote_accounts = bank.epoch_stakes(epoch).unwrap().stakes().vote_accounts();
        let mut members: Vec<([u8; 48], u64, Pubkey)> = vote_accounts
            .as_ref()
            .iter()
            .filter(|(pubkey, _)| ranked.contains(pubkey))
            .map(|(pubkey, (stake, account))| {
                (
                    account.vote_state_view().bls_pubkey_compressed().unwrap(),
                    *stake,
                    *pubkey,
                )
            })
            .collect();
        assert_eq!(members.len(), ranked.len());
        members.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let leaves: Vec<Vec<u8>> = members
            .iter()
            .map(|(bls, stake, vote)| {
                let mut leaf = bls.to_vec();
                leaf.extend_from_slice(&stake.to_le_bytes());
                leaf.extend_from_slice(vote.as_ref());
                leaf
            })
            .collect();
        let total_stake: u64 = members.iter().map(|m| m.1).sum();
        hashv(&[
            &[0x82],
            &epoch.to_le_bytes(),
            &total_stake.to_le_bytes(),
            compute_root(TreeId::VALIDATOR_SET, &leaves).as_ref(),
        ])
    }

    fn ranked_set(bank: &Bank, epoch: Epoch) -> HashSet<Pubkey> {
        bank.epoch_stakes(epoch)
            .unwrap()
            .bls_pubkey_to_rank_map()
            .iter()
            .map(|(_, entry)| entry.vote_account_pubkey)
            .collect()
    }

    #[test]
    fn test_inactive_features() {
        // Neither feature: no commitment at all.
        let cluster = test_cluster(EpochSchedule::custom(32, 32, false), false);
        let bank0 = Arc::new(Bank::new_for_tests(&cluster.genesis_config));
        assert!(!bank0.is_epoch_validator_set_root_active());
        freeze_with_block_id(&bank0);
        assert_eq!(bank0.block_commitment(), None);

        // Only the base feature: slot 3 is zero.
        let mut genesis_config = cluster.genesis_config.clone();
        activate_feature(
            &mut genesis_config,
            feature_set::block_commitment_root::id(),
        );
        let bank0 = Arc::new(Bank::new_for_tests(&genesis_config));
        assert!(!bank0.is_epoch_validator_set_root_active());
        freeze_with_block_id(&bank0);
        let mut vector = [INACTIVE_COMMITMENT_SLOT; COMMITMENT_VECTOR_LEN];
        vector[0] = bank0.block_commitment().unwrap().block_history_root();
        assert_eq!(
            bank0.commitment_root(),
            Some(CommitmentVector(vector).root())
        );

        // Only the validator set feature: no commitment at all.
        let mut genesis_config = cluster.genesis_config;
        activate_feature(
            &mut genesis_config,
            feature_set::epoch_validator_set_root::id(),
        );
        let bank0 = Arc::new(Bank::new_for_tests(&genesis_config));
        assert!(!bank0.is_epoch_validator_set_root_active());
        freeze_with_block_id(&bank0);
        assert_eq!(bank0.block_commitment(), None);
    }

    #[test]
    fn test_root_matches_independent_computation() {
        let cluster = test_cluster(EpochSchedule::custom(32, 32, false), true);
        let bank0 = Arc::new(Bank::new_for_tests(&cluster.genesis_config));
        assert!(bank0.is_epoch_validator_set_root_active());
        freeze_with_block_id(&bank0);

        // The rank map holds exactly the staked validators with BLS keys.
        let ranked = ranked_set(&bank0, 1);
        assert_eq!(
            ranked,
            cluster.ranked_vote_pubkeys.iter().copied().collect()
        );
        for pubkey in &cluster.unranked_vote_pubkeys {
            assert!(bank0.epoch_stakes(1).unwrap().vote_account_stake(pubkey) > 0);
            assert!(!ranked.contains(pubkey));
        }

        // Genesis bank, epoch 0: commits to the set of epoch 1.
        let expected = independent_root(&bank0, 1, &ranked);
        assert_eq!(slot3(&bank0), expected);
        assert_eq!(bank0.next_epoch_validator_set_root(), Some(expected));
        assert_ne!(expected, INACTIVE_COMMITMENT_SLOT);

        // Leaf i is the validator with rank i, i.e. bit i of a certificate's
        // signer bitmap; the BLS bytes are the ones stored in the vote
        // account; the total stake is the threshold denominator.
        let epoch_stakes = bank0.epoch_stakes(1).unwrap();
        let rank_map = epoch_stakes.bls_pubkey_to_rank_map();
        let signers = bank0.validator_set_signers(1).unwrap();
        assert_eq!(signers.len(), rank_map.len());
        for (rank, entry) in rank_map.iter() {
            let signer = &signers[usize::from(rank)];
            assert_eq!(signer.vote_pubkey, entry.vote_account_pubkey.to_bytes());
            assert_eq!(signer.stake, entry.stake.get().get());
            let vote_account = epoch_stakes
                .stakes()
                .vote_accounts()
                .get(&entry.vote_account_pubkey)
                .unwrap();
            assert_eq!(
                Some(signer.bls_pubkey_compressed),
                vote_account.vote_state_view().bls_pubkey_compressed()
            );
        }
        assert_eq!(
            epoch_stakes.signer_set_commitment().total_stake,
            rank_map.total_stake().get().get()
        );
        // The two validators with equal stake are ordered by BLS key.
        let tied: Vec<_> = signers
            .windows(2)
            .filter(|pair| pair[0].stake == pair[1].stake)
            .collect();
        assert_eq!(tied.len(), 1);
        assert!(tied[0][0].bls_pubkey_compressed < tied[0][1].bls_pubkey_compressed);
        for pair in signers.windows(2) {
            assert!(pair[0].cmp_signer_order(&pair[1]).is_lt());
        }
    }

    #[test]
    fn test_constant_within_epoch_and_changes_with_stake() {
        let cluster = test_cluster(EpochSchedule::custom(32, 32, false), true);
        let bank0 = Arc::new(Bank::new_for_tests(&cluster.genesis_config));
        freeze_with_block_id(&bank0);
        let root_epoch0 = slot3(&bank0);

        // A new validator with the largest stake joins in epoch 0.
        let bank1 = new_child(&bank0, 1);
        let new_validator = ValidatorVoteKeypairs::new_rand();
        let new_vote_pubkey = new_validator.vote_keypair.pubkey();
        let rent = bank1.rent_collector().rent.clone();
        for (pubkey, account) in create_validator(
            &rent,
            new_validator.node_keypair.pubkey(),
            1_000_000,
            new_vote_pubkey,
            minimum_vote_account_balance_for_vat(100),
            new_validator.stake_keypair.pubkey(),
            90 * minimum_vote_account_balance_for_vat(100) + minimum_stake_lamports_for_vat(&rent),
            Some(new_validator.bls_keypair.public.to_bytes_compressed()),
        ) {
            bank1.store_account(&pubkey, &account);
        }
        freeze_with_block_id(&bank1);
        assert_eq!(slot3(&bank1), root_epoch0);

        let mut bank = bank1;
        for slot in [2, 17, 31] {
            bank = new_child(&bank, slot);
            freeze_with_block_id(&bank);
            assert_eq!(slot3(&bank), root_epoch0, "slot {slot}");
        }

        // Epoch 1 commits to the set of epoch 2, fixed at the boundary from
        // the stakes that now include the new validator, at rank 0.
        let bank32 = new_child(&bank, 32);
        freeze_with_block_id(&bank32);
        let root_epoch1 = slot3(&bank32);
        assert_ne!(root_epoch1, root_epoch0);
        let ranked2 = ranked_set(&bank32, 2);
        assert!(ranked2.contains(&new_vote_pubkey));
        assert_eq!(ranked2.len(), cluster.ranked_vote_pubkeys.len() + 1);
        assert_eq!(root_epoch1, independent_root(&bank32, 2, &ranked2));
        assert_eq!(
            bank32
                .epoch_stakes(2)
                .unwrap()
                .bls_pubkey_to_rank_map()
                .get_rank_for_vote_pubkey(&new_vote_pubkey),
            Some(&0)
        );
        // The signer tree itself changed, not only the epoch number.
        assert_ne!(
            bank32.epoch_stakes(1).unwrap().signer_set_commitment(),
            bank32.epoch_stakes(2).unwrap().signer_set_commitment(),
        );
        // The set of epoch 1 is unchanged in the banks of epoch 1.
        assert_eq!(bank32.validator_set_root(1), Some(root_epoch0));

        let mut bank = bank32;
        for slot in [33, 50, 63] {
            bank = new_child(&bank, slot);
            freeze_with_block_id(&bank);
            assert_eq!(slot3(&bank), root_epoch1, "slot {slot}");
        }
        let bank64 = new_child(&bank, 64);
        freeze_with_block_id(&bank64);
        let root_epoch2 = slot3(&bank64);
        assert_ne!(root_epoch2, root_epoch1);
        assert_eq!(
            root_epoch2,
            independent_root(&bank64, 3, &ranked_set(&bank64, 3))
        );
    }

    #[test]
    fn test_computed_once_per_epoch() {
        let cluster = test_cluster(EpochSchedule::custom(32, 32, false), true);
        let bank0 = Arc::new(Bank::new_for_tests(&cluster.genesis_config));
        freeze_with_block_id(&bank0);
        let bank1 = new_child(&bank0, 1);
        // Cached in the genesis bank's entry when it froze, inherited by the
        // child.
        assert!(
            bank1
                .epoch_stakes(1)
                .unwrap()
                .cached_signer_set_commitment()
                .is_some()
        );
        // Computed when the boundary bank inserted the stakes of epoch 2,
        // before it is frozen.
        let bank32 = new_child(&bank1, 32);
        assert!(
            bank32
                .epoch_stakes(2)
                .unwrap()
                .cached_signer_set_commitment()
                .is_some()
        );
    }

    #[test]
    fn test_proven_set_is_the_certificate_set_of_next_epoch() {
        let cluster = test_cluster(EpochSchedule::custom(32, 32, false), true);
        let bank0 = Arc::new(Bank::new_for_tests(&cluster.genesis_config));
        freeze_with_block_id(&bank0);
        let bank32 = new_child(&bank0, 32);
        freeze_with_block_id(&bank32);
        let root = slot3(&bank32);

        // The banks of epoch 2 verify certificates with
        // `epoch_stakes_from_slot(slot)`.
        let bank70 = new_child(&bank32, 70);
        freeze_with_block_id(&bank70);
        let rank_map = bank70.get_rank_map(bank70.slot()).unwrap();
        let signers = rank_map.signers();
        let total_stake = rank_map.total_stake().get().get();
        assert_eq!(
            SignerSetCommitment::new(&signers)
                .unwrap()
                .validator_set_root(2),
            root
        );
        for (rank, entry) in rank_map.iter() {
            let (signer, proof, proof_total_stake) = bank32
                .validator_set_signer_proof(2, usize::from(rank))
                .unwrap();
            assert_eq!(proof_total_stake, total_stake);
            assert_eq!(signer.vote_pubkey, entry.vote_account_pubkey.to_bytes());
            assert_eq!(
                signer.bls_pubkey_compressed,
                entry.bls_pubkey.to_bytes_compressed()
            );
            assert_eq!(signer.stake, entry.stake.get().get());
            assert!(verify_signer(&root, 2, total_stake, &signer, &proof));
            // Not valid as a member of the set of another epoch.
            assert!(!verify_signer(&root, 3, total_stake, &signer, &proof));
        }
        assert!(
            bank32
                .validator_set_signer_proof(2, rank_map.len())
                .is_none()
        );
    }

    #[test]
    fn test_proof_from_commitment_root_to_signer() {
        let cluster = test_cluster(EpochSchedule::custom(32, 32, false), true);
        let bank0 = Arc::new(Bank::new_for_tests(&cluster.genesis_config));
        freeze_with_block_id(&bank0);
        let bank5 = new_child(&bank0, 5);
        freeze_with_block_id(&bank5);
        let commitment_root = bank5.commitment_root().unwrap();

        // What a light client receives: the commitment vector opening of slot
        // 3, then one signer with its path.
        let mut vector = [INACTIVE_COMMITMENT_SLOT; COMMITMENT_VECTOR_LEN];
        vector[CommitmentSlot::BlockHistoryRoot.index()] =
            bank5.block_commitment().unwrap().block_history_root();
        vector[CommitmentSlot::ValidatorSetRoot.index()] =
            bank5.next_epoch_validator_set_root().unwrap();
        let vector = CommitmentVector(vector);
        let slot3_proof = vector
            .inclusion_proof(CommitmentSlot::ValidatorSetRoot.index())
            .unwrap();
        let validator_set_root = vector.get(CommitmentSlot::ValidatorSetRoot);
        assert!(slot3_proof.verify(
            TreeId::COMMITMENT_VECTOR,
            validator_set_root.as_ref(),
            &commitment_root
        ));
        let rank = 2;
        let (signer, signer_proof, total_stake) =
            bank5.validator_set_signer_proof(1, rank).unwrap();
        assert!(verify_signer(
            &validator_set_root,
            1,
            total_stake,
            &signer,
            &signer_proof
        ));
        assert_eq!(
            signer.vote_pubkey,
            bank5
                .get_rank_map(32)
                .unwrap()
                .get_pubkey_stake_entry(rank)
                .unwrap()
                .vote_account_pubkey
                .to_bytes()
        );
        // A wrong slot opening does not verify.
        assert!(!slot3_proof.verify(
            TreeId::COMMITMENT_VECTOR,
            Hash::new_unique().as_ref(),
            &commitment_root
        ));
    }

    #[test]
    fn test_forks_with_different_stakes() {
        let cluster = test_cluster(EpochSchedule::custom(32, 32, false), true);
        let bank0 = Arc::new(Bank::new_for_tests(&cluster.genesis_config));
        freeze_with_block_id(&bank0);

        // Fork A adds stake in epoch 0; fork B does not.
        let bank1 = new_child(&bank0, 1);
        let new_validator = ValidatorVoteKeypairs::new_rand();
        let rent = bank1.rent_collector().rent.clone();
        for (pubkey, account) in create_validator(
            &rent,
            new_validator.node_keypair.pubkey(),
            1_000_000,
            new_validator.vote_keypair.pubkey(),
            minimum_vote_account_balance_for_vat(100),
            new_validator.stake_keypair.pubkey(),
            20 * minimum_vote_account_balance_for_vat(100),
            Some(new_validator.bls_keypair.public.to_bytes_compressed()),
        ) {
            bank1.store_account(&pubkey, &account);
        }
        freeze_with_block_id(&bank1);
        let fork_a = new_child(&bank1, 32);
        freeze_with_block_id(&fork_a);
        let fork_b = new_child(&bank0, 33);
        freeze_with_block_id(&fork_b);
        assert_ne!(slot3(&fork_a), slot3(&fork_b));
        assert_eq!(
            slot3(&fork_b),
            independent_root(&fork_b, 2, &ranked_set(&fork_b, 2))
        );
        assert_eq!(
            slot3(&fork_a),
            independent_root(&fork_a, 2, &ranked_set(&fork_a, 2))
        );
    }

    #[test]
    fn test_short_leader_schedule_offset() {
        // Leader schedule offset of half an epoch: the stakes of `E + 1` are
        // fixed in the middle of `E`, and slot 3 is zero before that.
        let cluster = test_cluster(EpochSchedule::custom(64, 32, false), true);
        let bank0 = Arc::new(Bank::new_for_tests(&cluster.genesis_config));
        freeze_with_block_id(&bank0);
        assert!(bank0.epoch_stakes(1).is_none());
        assert_eq!(slot3(&bank0), INACTIVE_COMMITMENT_SLOT);
        let bank31 = new_child(&bank0, 31);
        freeze_with_block_id(&bank31);
        assert_eq!(slot3(&bank31), INACTIVE_COMMITMENT_SLOT);
        // Slot 32 is skipped; slot 33 is the first bank with the stakes of
        // epoch 1.
        let bank33 = new_child(&bank31, 33);
        freeze_with_block_id(&bank33);
        let root = slot3(&bank33);
        assert_ne!(root, INACTIVE_COMMITMENT_SLOT);
        assert_eq!(root, independent_root(&bank33, 1, &ranked_set(&bank33, 1)));
        let bank63 = new_child(&bank33, 63);
        freeze_with_block_id(&bank63);
        assert_eq!(slot3(&bank63), root);
        // Epoch 1, first half: the stakes of epoch 2 are not fixed yet.
        let bank64 = new_child(&bank63, 64);
        freeze_with_block_id(&bank64);
        assert!(bank64.epoch_stakes(2).is_none());
        assert_eq!(slot3(&bank64), INACTIVE_COMMITMENT_SLOT);
        let bank96 = new_child(&bank64, 96);
        freeze_with_block_id(&bank96);
        assert_eq!(
            slot3(&bank96),
            independent_root(&bank96, 2, &ranked_set(&bank96, 2))
        );
    }

    #[test]
    fn test_warmup_epochs() {
        let cluster = test_cluster(EpochSchedule::custom(256, 256, true), true);
        let epoch_schedule = cluster.genesis_config.epoch_schedule.clone();
        let mut bank = Arc::new(Bank::new_for_tests(&cluster.genesis_config));
        freeze_with_block_id(&bank);
        // Warmup epochs have 32, 64 and 128 slots.
        for epoch in 0..4 {
            let first_slot = epoch_schedule.get_first_slot_in_epoch(epoch);
            let last_slot = epoch_schedule.get_last_slot_in_epoch(epoch);
            for slot in [first_slot, last_slot] {
                if slot != bank.slot() {
                    bank = new_child(&bank, slot);
                    freeze_with_block_id(&bank);
                }
                assert_eq!(bank.epoch(), epoch);
                let expected = independent_root(&bank, epoch + 1, &ranked_set(&bank, epoch + 1));
                assert_eq!(slot3(&bank), expected, "epoch {epoch} slot {slot}");
            }
        }
    }

    #[test]
    fn test_empty_signer_set() {
        // The bootstrap validator has no stake: no validator has a rank, and
        // building the rank map would panic.
        let mut genesis_config = create_genesis_config(LAMPORTS).genesis_config;
        activate_feature(
            &mut genesis_config,
            feature_set::block_commitment_root::id(),
        );
        activate_feature(
            &mut genesis_config,
            feature_set::epoch_validator_set_root::id(),
        );
        let bank0 = Arc::new(Bank::new_for_tests(&genesis_config));
        freeze_with_block_id(&bank0);
        assert!(
            bank0
                .epoch_stakes(1)
                .unwrap()
                .try_bls_pubkey_to_rank_map()
                .is_none()
        );
        let expected = hashv(&[
            &[0x82],
            &1u64.to_le_bytes(),
            &0u64.to_le_bytes(),
            compute_root(TreeId::VALIDATOR_SET, &[] as &[&[u8]]).as_ref(),
        ]);
        assert_eq!(slot3(&bank0), expected);
        assert_eq!(SignerSetCommitment::empty().validator_set_root(1), expected);
        assert_eq!(bank0.validator_set_signers(1), Some(vec![]));
        assert!(bank0.validator_set_signer_proof(1, 0).is_none());
    }

    #[test]
    fn test_recomputed_after_snapshot_serialization() {
        let cluster = test_cluster(EpochSchedule::custom(32, 32, false), true);
        let bank0 = Arc::new(Bank::new_for_tests(&cluster.genesis_config));
        freeze_with_block_id(&bank0);
        let epoch_stakes = bank0.epoch_stakes(1).unwrap();
        assert!(epoch_stakes.cached_signer_set_commitment().is_some());
        // Epoch stakes are serialized in the snapshot without the cache.
        let bytes = wincode::serialize(epoch_stakes).unwrap();
        let loaded: VersionedEpochStakes =
            wincode::deserialize::<DeserializableVersionedEpochStakes>(&bytes)
                .unwrap()
                .into();
        assert!(loaded.cached_signer_set_commitment().is_none());
        assert_eq!(
            loaded.signer_set_commitment(),
            epoch_stakes.signer_set_commitment()
        );
    }

    #[test]
    fn test_activation_mid_chain() {
        // The feature activates at the start of epoch 1 (features activate
        // in the first bank of an epoch).
        let cluster = test_cluster(EpochSchedule::custom(32, 32, false), false);
        let mut genesis_config = cluster.genesis_config;
        activate_feature(
            &mut genesis_config,
            feature_set::block_commitment_root::id(),
        );
        let bank0 = Arc::new(Bank::new_for_tests(&genesis_config));
        freeze_with_block_id(&bank0);
        assert_eq!(slot3(&bank0), INACTIVE_COMMITMENT_SLOT);
        let mut bank32 = Bank::new_from_parent(bank0.clone(), SlotLeader::default(), 32);
        bank32.activate_feature(&feature_set::epoch_validator_set_root::id());
        let bank32 = Arc::new(bank32);
        freeze_with_block_id(&bank32);
        assert_eq!(
            slot3(&bank32),
            independent_root(&bank32, 2, &ranked_set(&bank32, 2))
        );
    }
}
