use {
    crate::bank::{
        accounts_state::{SnapshotAccountsState, SnapshotGeneration},
        block_commitment::BlockCommitment,
    },
    solana_account::AccountSharedData,
    solana_accounts_db::accounts_hash::AccountsLtHash,
    solana_clock::Epoch,
    solana_epoch_schedule::EpochSchedule,
    solana_hash::Hash,
    solana_lattice_hash::lt_hash::LtHash,
    solana_merkle_tree::{
        accounts_state::{OverlayEntry, overlay_value},
        commitment_tree::{AppendOnlyFrontier, FrontierError, TreeId},
        keyed_tree::KeyedTree,
    },
    solana_pubkey::Pubkey,
    solana_rent::Rent,
    solana_stake_history::StakeHistoryItem,
    solana_stake_interface::state::Delegation,
    wincode::{SchemaRead, SchemaWrite},
};
#[cfg(feature = "stable-abi")]
use {
    solana_frozen_abi::{
        rand::RngCore as FrozenAbiRng,
        stable_abi::{StableAbi as _, context::SequenceLenMax, sample_collection_sized},
    },
    solana_frozen_abi_macro::{StableAbi, StableAbiSample},
    solana_stake_interface::{
        stake_flags::StakeFlags,
        state::{Meta, Stake},
    },
};

/// Snapshot serde-safe AccountsLtHash
#[cfg_attr(feature = "stable-abi", derive(StableAbi, StableAbiSample))]
#[derive(Debug, Clone, PartialEq, Eq, SchemaRead, SchemaWrite)]
pub struct SerdeAccountsLtHash(pub [u16; LtHash::NUM_ELEMENTS]);

impl From<SerdeAccountsLtHash> for AccountsLtHash {
    fn from(accounts_lt_hash: SerdeAccountsLtHash) -> Self {
        Self(LtHash(accounts_lt_hash.0))
    }
}
impl From<AccountsLtHash> for SerdeAccountsLtHash {
    fn from(accounts_lt_hash: AccountsLtHash) -> Self {
        Self(accounts_lt_hash.0.0)
    }
}

/// Snapshot serde-safe block commitment state (Block Commitment Root SIMD):
/// the block history frontier of the snapshot bank and its commitment root.
#[cfg_attr(feature = "stable-abi", derive(StableAbi, StableAbiSample))]
#[derive(Debug, Clone, PartialEq, Eq, SchemaRead, SchemaWrite)]
pub struct SerdeBlockCommitment {
    /// Number of leaves in the block history of the snapshot bank.
    pub history_leaf_count: u64,
    /// Roots of the complete subtrees along the right edge of the block
    /// history tree, largest first. One per set bit of `history_leaf_count`
    /// (at most 64).
    pub history_subtree_roots: Vec<Hash>,
    /// `commitment_root` of the snapshot bank.
    pub commitment_root: Hash,
}

impl From<BlockCommitment> for SerdeBlockCommitment {
    fn from(block_commitment: BlockCommitment) -> Self {
        let BlockCommitment {
            block_history,
            commitment_root,
        } = block_commitment;
        let (history_leaf_count, history_subtree_roots) = block_history.into_parts();
        Self {
            history_leaf_count,
            history_subtree_roots,
            commitment_root,
        }
    }
}

impl TryFrom<SerdeBlockCommitment> for BlockCommitment {
    type Error = FrontierError;

    fn try_from(serde: SerdeBlockCommitment) -> Result<Self, Self::Error> {
        let SerdeBlockCommitment {
            history_leaf_count,
            history_subtree_roots,
            commitment_root,
        } = serde;
        Ok(Self {
            block_history: AppendOnlyFrontier::from_parts(
                history_leaf_count,
                history_subtree_roots,
            )?,
            commitment_root,
        })
    }
}

/// Snapshot serde-safe accounts state (Accounts State Root SIMD): the
/// generations held by the snapshot bank.
#[cfg_attr(feature = "stable-abi", derive(StableAbi, StableAbiSample))]
#[derive(Debug, Clone, PartialEq, Eq, SchemaRead, SchemaWrite)]
pub struct SerdeAccountsState {
    /// In ascending epoch order, at most two.
    pub generations: Vec<SerdeAccountsStateGeneration>,
}

/// One generation of [`SerdeAccountsState`].
#[cfg_attr(feature = "stable-abi", derive(StableAbi, StableAbiSample))]
#[derive(Debug, Clone, PartialEq, Eq, SchemaRead, SchemaWrite)]
pub struct SerdeAccountsStateGeneration {
    pub epoch: Epoch,
    /// The block whose state is the base state of the generation (`P`), None
    /// for a chain that starts without a parent (empty base state).
    pub base_parent: Option<SerdeBaseParent>,
    /// `base_root`, if it was known when the snapshot was taken.
    pub base_root: Option<Hash>,
    /// The overlay entries, sorted by address (96 bytes per entry).
    pub overlay: Vec<SerdeOverlayEntry>,
}

/// `(slot, bank_hash)` of the block whose state is a base state.
#[cfg_attr(feature = "stable-abi", derive(StableAbi, StableAbiSample))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, SchemaRead, SchemaWrite)]
pub struct SerdeBaseParent {
    pub slot: u64,
    pub bank_hash: Hash,
}

/// One overlay entry: address, `account_hash` now and in the base state.
#[cfg_attr(feature = "stable-abi", derive(StableAbi, StableAbiSample))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, SchemaRead, SchemaWrite)]
pub struct SerdeOverlayEntry {
    pub address: Pubkey,
    pub now: Hash,
    pub base: Hash,
}

impl From<SnapshotAccountsState> for SerdeAccountsState {
    fn from(state: SnapshotAccountsState) -> Self {
        Self {
            generations: state
                .generations
                .into_iter()
                .map(|generation| SerdeAccountsStateGeneration {
                    epoch: generation.epoch,
                    base_parent: generation
                        .base_parent
                        .map(|(slot, bank_hash)| SerdeBaseParent { slot, bank_hash }),
                    base_root: generation.base_root,
                    overlay: generation
                        .overlay
                        .iter()
                        .map(|(address, value)| {
                            let entry = OverlayEntry::from_value(*address, value);
                            SerdeOverlayEntry {
                                address: Pubkey::new_from_array(entry.address),
                                now: entry.now,
                                base: entry.base,
                            }
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

impl TryFrom<SerdeAccountsState> for SnapshotAccountsState {
    type Error = &'static str;

    /// Checks that generations are in strictly ascending epoch order, at most
    /// two, and that overlay addresses are strictly ascending.
    fn try_from(serde: SerdeAccountsState) -> Result<Self, Self::Error> {
        if serde.generations.len() > 2
            || serde
                .generations
                .windows(2)
                .any(|pair| pair[0].epoch >= pair[1].epoch)
        {
            return Err("accounts state generations are not two ascending epochs");
        }
        let generations = serde
            .generations
            .into_iter()
            .map(|generation| {
                if generation
                    .overlay
                    .windows(2)
                    .any(|pair| pair[0].address >= pair[1].address)
                {
                    return Err("accounts state overlay entries are not sorted and unique");
                }
                let overlay = KeyedTree::from_entries(
                    TreeId::STATE_OVERLAY,
                    generation.overlay.iter().map(|entry| {
                        (
                            entry.address.to_bytes(),
                            overlay_value(&entry.now, &entry.base),
                        )
                    }),
                )
                .map_err(|_| "accounts state overlay entries are not unique")?;
                Ok(SnapshotGeneration {
                    epoch: generation.epoch,
                    base_parent: generation
                        .base_parent
                        .map(|parent| (parent.slot, parent.bank_hash)),
                    base_root: generation.base_root,
                    overlay,
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { generations })
    }
}

/// Snapshot serde-safe RentCollector, which is now unused
#[cfg_attr(feature = "stable-abi", derive(StableAbi, StableAbiSample))]
#[derive(Debug, Clone, SchemaRead, SchemaWrite)]
pub struct UnusedRentCollector {
    epoch: Epoch,
    epoch_schedule: EpochSchedule,
    slots_per_year: f64,
    rent: Rent,
}

impl UnusedRentCollector {
    pub fn zeroed() -> Self {
        #[allow(deprecated)]
        Self {
            epoch: 0,
            epoch_schedule: EpochSchedule {
                slots_per_epoch: 0,
                leader_schedule_slot_offset: 0,
                warmup: false,
                first_normal_epoch: 0,
                first_normal_slot: 0,
            },
            slots_per_year: 0.0,
            rent: Rent {
                lamports_per_byte: 0,
                exemption_threshold: [0; 8],
                burn_percent: 0,
            },
        }
    }
}

/// Snapshot serde-safe Stakes<Delegation>, which is now unused
#[cfg_attr(feature = "stable-abi", derive(StableAbi, StableAbiSample))]
#[derive(Debug, Clone, Default, SchemaRead, SchemaWrite)]
pub struct UnusedStakes {
    #[cfg_attr(
        feature = "stable-abi",
        stable_abi_sample(with = "sample_vote_accounts(rng)")
    )]
    vote_accounts: Vec<(Pubkey, (/*stake:*/ u64, AccountSharedData))>,
    #[cfg_attr(
        feature = "stable-abi",
        stable_abi_sample(with = "sample_stake_delegations(rng)")
    )]
    stake_delegations: Vec<(Pubkey, Delegation)>,
    unused: u64,
    epoch: Epoch,
    stake_history: Vec<StakeHistoryItem>,
}

#[cfg(feature = "stable-abi")]
fn sample_vote_accounts(
    rng: &mut (impl FrozenAbiRng + ?Sized),
) -> Vec<(Pubkey, (u64, AccountSharedData))> {
    let vote_accounts = solana_vote::vote_account::VoteAccounts::random(rng);
    vote_accounts
        .as_ref()
        .iter()
        .map(|(pubkey, (stake, account))| (*pubkey, (*stake, AccountSharedData::from(account))))
        .collect()
}

#[cfg(feature = "stable-abi")]
fn sample_stake_delegations(rng: &mut (impl FrozenAbiRng + ?Sized)) -> Vec<(Pubkey, Delegation)> {
    let stakes: Vec<(Pubkey, (Meta, Stake, StakeFlags))> =
        sample_collection_sized(rng, SequenceLenMax(1));
    stakes
        .into_iter()
        .map(|(pubkey, (_, stake, _))| (pubkey, stake.delegation))
        .collect()
}
