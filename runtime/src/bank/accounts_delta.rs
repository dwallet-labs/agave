//! Accounts Delta Root (draft SIMD): commitment slot 2, `accounts_delta_root`.
//!
//! The root commits to the post-block state of every account whose state
//! `(lamports, data, executable, owner)` after the block differs from its
//! state before the block, with an account of zero lamports having the empty
//! state. The tree construction and the proofs live in
//! [`solana_merkle_tree::accounts_delta`]; this module computes the account
//! hashes and decides which accounts changed.
//!
//! # Collection
//!
//! Every account update of a bank, whatever path stores it (transaction
//! commits, fee distribution, rewards, sysvars, slot history, incinerator,
//! epoch boundary processing, feature activations and builtin migrations,
//! genesis, test-only stores), goes through one of the two enqueue functions
//! of the accounts lattice hash, `Bank::enqueue_on_chain_accounts_lt_hash_updates`
//! and `Bank::enqueue_off_chain_accounts_lt_hash_updates`, and every queued
//! update is hashed by `AccountsLtHashAsyncProgress::process_and_collect`.
//! That function is the only place where the lattice hash computes per-account
//! hashes, so it is where the account hashes of this tree are taken: the first
//! 32 bytes of the lattice hash output, which is the BLAKE3 digest of the same
//! input. No account is hashed a second time.
//!
//! An update pairs the previous version of an account (loaded when the update
//! is enqueued, before the store) with its new version, and the versions of
//! one account chain: each update's previous version is the new version of
//! the update before it, the first one is the state before the block and the
//! last one is the state after the block. The lattice hash relies on the same
//! chain, since it mixes out every previous version and mixes in every new
//! one. Updates are hashed on several threads in any order, possibly merged
//! (dedup) or split in two jobs (large accounts), so the order of the updates
//! is not known when they are hashed. Instead of keeping "the first previous
//! and the last new prefix" by order, each worker counts, per address, `+1`
//! for the prefix of every previous version and `-1` for the prefix of every
//! new version ([`AccountsDeltaPrefixes`]). Along a chain the intermediate
//! versions cancel, so once the bank is frozen each address is left with
//! either nothing (the state after the block equals the state before it, up to
//! the 32-byte prefix) or `+1` for the state before and `-1` for the state
//! after the block, which is the leaf value. This is the spec's "first
//! previous, last new" rule computed without depending on the order in which
//! the hashes were computed, so it is deterministic.
//!
//! # Gating and cost
//!
//! The prefixes are kept only by banks that may need the root: banks whose
//! parent has both `block_commitment_root` and `accounts_delta_root` active,
//! the first bank of every epoch (where features activate), and banks built
//! without a parent (genesis, tests). With the features inactive the cost is
//! one branch per hashing job, plus the collection in the first bank of each
//! epoch. When collecting, each hashing worker keeps a map entry per address
//! it saw (a pubkey and one or two 32-byte prefixes with their counts). At
//! freeze, the worker maps are merged, the changed accounts are sorted, and
//! the tree is built (two SHA-256 compressions for each 64-byte leaf and one
//! per inner node, about four per changed account). The sorted leaves (64
//! bytes per changed account) are kept by the frozen bank so that proofs can
//! be served.

use {
    super::{Bank, block_commitment::CommitmentFailure},
    agave_feature_set as feature_set,
    smallvec::SmallVec,
    solana_account::ReadableAccount,
    solana_accounts_db::accounts_db::AccountsDb,
    solana_clock::Epoch,
    solana_hash::Hash,
    solana_lattice_hash::lt_hash::LtHash,
    solana_merkle_tree::accounts_delta::{
        AccountsDelta, AccountsDeltaError, DELETED_ACCOUNT_HASH, DeltaEntry,
    },
    solana_pubkey::Pubkey,
    std::{collections::hash_map::Entry, sync::Arc},
};

/// `account_hash(account')` for `account` stored at `address`:
///
/// ```text
/// 32 zero bytes                                         if lamports == 0
/// blake3_256(lamports || data || executable || owner || pubkey)  otherwise
/// ```
///
/// The input is the SIMD-0215 single-account input. The value is the first 32
/// bytes of the account's lattice hash, which is how the validator obtains it;
/// this function computes it the same way, for proofs and tests.
pub fn account_hash(account: &impl ReadableAccount, address: &Pubkey) -> Hash {
    if account.lamports() == 0 {
        return DELETED_ACCOUNT_HASH;
    }
    lt_hash_prefix(&AccountsDb::lt_hash_account(account, address).0)
}

/// The first 32 bytes of the BLAKE3 extended output that forms `lt_hash`,
/// which equal the 32-byte BLAKE3 digest of the same input.
pub(super) fn lt_hash_prefix(lt_hash: &LtHash) -> Hash {
    // `LtHash::with()` fills the elements with the BLAKE3 output bytes in
    // order, so viewing the elements as bytes gives that output back,
    // independently of the platform's endianness.
    let bytes: &[u8] = bytemuck::must_cast_slice(&lt_hash.0);
    Hash::new_from_array(bytes[..32].try_into().expect("an lt hash has 2048 bytes"))
}

/// Net count of each prefix recorded for one address: `+1` for every
/// previous version, `-1` for every new version. Entries whose count reaches
/// zero are removed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct NetPrefixes(SmallVec<[(Hash, i64); 2]>);

impl NetPrefixes {
    fn add(&mut self, prefix: Hash, count: i64) {
        if let Some(index) = self.0.iter().position(|(p, _)| *p == prefix) {
            let net = &mut self.0[index].1;
            *net = net.saturating_add(count);
            if *net == 0 {
                self.0.swap_remove(index);
            }
        } else if count != 0 {
            self.0.push((prefix, count));
        }
    }

    /// `None` if the address is unchanged, the prefixes of the states before
    /// and after the block if it changed, or `Err` if the counts do not form
    /// a chain.
    fn resolve(&self) -> Result<Option<(Hash, Hash)>, ()> {
        match self.0.as_slice() {
            [] => Ok(None),
            [(before, 1), (after, -1)] | [(after, -1), (before, 1)] => Ok(Some((*before, *after))),
            _ => Err(()),
        }
    }
}

/// The 32-byte prefixes of the account versions hashed for one bank, as net
/// counts per address (see the module documentation).
#[derive(Debug, Default)]
pub(crate) struct AccountsDeltaPrefixes {
    by_address: ahash::HashMap<Pubkey, NetPrefixes>,
}

/// The recorded prefixes of an address do not form a single chain of
/// versions, which means an account update was missed or duplicated.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct InconsistentAccountUpdates {
    pub(crate) address: Pubkey,
    pub(crate) num_prefixes: usize,
}

impl AccountsDeltaPrefixes {
    /// Records the prefix of the previous version of `address` in an update.
    pub(crate) fn record_prev(&mut self, address: &Pubkey, prefix: Hash) {
        self.by_address.entry(*address).or_default().add(prefix, 1);
    }

    /// Records the prefix of the new version of `address` in an update.
    pub(crate) fn record_curr(&mut self, address: &Pubkey, prefix: Hash) {
        self.by_address.entry(*address).or_default().add(prefix, -1);
    }

    /// Number of addresses with an entry.
    pub(crate) fn num_addresses(&self) -> usize {
        self.by_address.len()
    }

    /// Adds the counts of `other` to `self`.
    pub(crate) fn merge(&mut self, other: Self) {
        for (address, net) in other.by_address {
            match self.by_address.entry(address) {
                Entry::Vacant(entry) => {
                    entry.insert(net);
                }
                Entry::Occupied(mut entry) => {
                    let held = entry.get_mut();
                    for (prefix, count) in net.0 {
                        held.add(prefix, count);
                    }
                }
            }
        }
    }

    /// The changed accounts: the addresses whose state after the block differs
    /// from their state before it, with the `account_hash` of the state after.
    #[cfg(test)]
    pub(crate) fn into_accounts_delta(self) -> Result<AccountsDelta, InconsistentAccountUpdates> {
        self.into_accounts_delta_with_prior()
            .map(|(delta, _prior)| delta)
    }

    /// Like [`Self::into_accounts_delta`], with the `account_hash` of each
    /// changed account before the block (the prefix of its first previous
    /// version), in the order of the delta's entries. The accounts state
    /// overlay takes it for addresses it inserts.
    pub(crate) fn into_accounts_delta_with_prior(
        self,
    ) -> Result<(AccountsDelta, Vec<Hash>), InconsistentAccountUpdates> {
        let mut changed = Vec::with_capacity(self.by_address.len());
        for (address, net) in &self.by_address {
            match net.resolve() {
                Ok(None) => {}
                Ok(Some((before, after))) => {
                    changed.push((DeltaEntry::new(address.to_bytes(), after), before))
                }
                Err(()) => {
                    return Err(InconsistentAccountUpdates {
                        address: *address,
                        num_prefixes: net.0.len(),
                    });
                }
            }
        }
        drop(self);
        // Addresses are unique (map keys), so an unstable sort is
        // deterministic.
        changed.sort_unstable_by_key(|(entry, _)| entry.address);
        let (entries, prior): (Vec<_>, Vec<_>) = changed.into_iter().unzip();
        let delta = AccountsDelta::new(entries).unwrap_or_else(
            |AccountsDeltaError::DuplicateAddress(_)| {
                unreachable!("addresses are the keys of a map")
            },
        );
        Ok((delta, prior))
    }
}

/// The accounts delta of a frozen bank, or the commitment failure that
/// prevents computing it.
pub(super) type AccountsDeltaState = Result<Arc<AccountsDelta>, CommitmentFailure>;

impl Bank {
    /// Whether this bank computes `accounts_delta_root`: both the Accounts
    /// Delta Root feature and the Block Commitment Root feature, which
    /// carries the root, must be active.
    pub fn is_accounts_delta_root_active(&self) -> bool {
        self.feature_set
            .is_active(&feature_set::accounts_delta_root::id())
            && self.is_block_commitment_root_active()
    }

    /// Whether a bank created from `parent` in `epoch` must collect the
    /// accounts delta prefixes. Decided before the bank's first account
    /// update: the features can only become active for a bank that starts a
    /// new epoch, so collecting in those banks and in every bank whose parent
    /// computes the root covers every bank that computes the root.
    pub(super) fn collects_accounts_delta_after(parent: &Bank, epoch: Epoch) -> bool {
        parent.is_accounts_delta_root_active() || parent.epoch() < epoch
    }

    /// Whether this bank keeps the account hashes for the accounts delta root
    /// (decided when the bank is created).
    pub fn collects_accounts_delta(&self) -> bool {
        self.accounts_lt_hash_async_progress
            .is_collecting_accounts_delta()
    }

    /// The changed accounts of this bank and their root, set when the bank is
    /// frozen with [`Self::is_accounts_delta_root_active`]. `None` otherwise,
    /// including for a bank loaded from a snapshot, and for a bank whose
    /// accounts delta could not be computed (a commitment failure).
    pub fn accounts_delta(&self) -> Option<Arc<AccountsDelta>> {
        self.accounts_delta
            .read()
            .unwrap()
            .as_ref()
            .and_then(|delta| delta.as_ref().ok())
            .cloned()
    }

    /// `accounts_delta_root` of this bank, if it has one.
    pub fn accounts_delta_root(&self) -> Option<Hash> {
        self.accounts_delta().map(|delta| delta.root())
    }

    /// `accounts_delta_root` for the commitment vector: `None` if the feature
    /// is inactive, and a commitment failure if the bank could not compute
    /// it. Called from `freeze()` after [`Self::finish_accounts_delta`].
    pub(super) fn accounts_delta_root_for_commitment(
        &self,
    ) -> Result<Option<Hash>, CommitmentFailure> {
        if !self.is_accounts_delta_root_active() {
            return Ok(None);
        }
        match self.accounts_delta.read().unwrap().as_ref() {
            Some(Ok(delta)) => Ok(Some(delta.root())),
            Some(Err(failure)) => Err(failure.clone()),
            // `finish_accounts_delta()` sets it for every bank that computes
            // the root, before the commitment vector is computed.
            None => Err(CommitmentFailure::AccountsDeltaNotCollected),
        }
    }

    /// Builds the accounts delta from the collected prefixes, or records why
    /// it cannot be built (a commitment failure, reported by `freeze()`).
    /// Called once, from `freeze()` (through
    /// `finish_accounts_lt_hash_updates`), after the last account update of
    /// the bank has been hashed. Returns the number of addresses
    /// updated in the bank (0 if this bank does not collect).
    ///
    /// Idempotent: once the delta (or its failure) is recorded, a later call
    /// changes nothing. Taking the prefixes again would otherwise find them
    /// empty and replace the delta with an empty one.
    pub(super) fn finish_accounts_delta(&self) -> usize {
        if self.accounts_delta.read().unwrap().is_some() {
            return 0;
        }
        let prefixes = self
            .accounts_lt_hash_async_progress
            .take_accounts_delta_prefixes();
        let num_addresses = prefixes
            .as_ref()
            .map_or(0, |prefixes| prefixes.num_addresses());
        if !self.is_accounts_delta_root_active() {
            return num_addresses;
        }
        let delta = match prefixes {
            // Unreachable through `new_from_parent` (see
            // `collects_accounts_delta_after`): only a bank whose feature set
            // changed after its first account update (a test that activates
            // the feature on an existing bank) gets here. Without the account
            // hashes of its first updates the root cannot be computed.
            None => Err(CommitmentFailure::AccountsDeltaNotCollected),
            // Each address's updates chain from its state before the block to
            // its state after it, which is what keeps the accounts lattice
            // hash correct. A broken chain means an update was lost or
            // duplicated, and then the lattice hash, and so the bank hash, is
            // wrong too.
            Some(prefixes) => prefixes
                .into_accounts_delta_with_prior()
                .map(|(delta, prior)| (Arc::new(delta), prior))
                .map_err(|err| CommitmentFailure::InconsistentAccountUpdates {
                    address: err.address,
                    num_prefixes: err.num_prefixes,
                }),
        };
        // The accounts state overlays (slot 5) are updated from the same
        // changed accounts, with their hashes before and after the block.
        self.finish_accounts_state(
            delta
                .as_ref()
                .map(|(delta, prior)| (delta.as_ref(), prior.as_slice()))
                .map_err(Clone::clone),
        );
        *self.accounts_delta.write().unwrap() = Some(delta.map(|(delta, _prior)| delta));
        num_addresses
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        solana_account::{AccountSharedData, WritableAccount},
    };

    fn prefix(byte: u8) -> Hash {
        Hash::new_from_array([byte; 32])
    }

    /// The account hash is the BLAKE3 digest of the SIMD-0215 input,
    /// computed here with the `blake3` crate independently of the lattice
    /// hash code, and equals the prefix of the lattice hash.
    #[test]
    fn test_account_hash_is_blake3_of_lattice_input() {
        let address = Pubkey::new_unique();
        let owner = Pubkey::new_unique();
        for (lamports, data, executable) in [
            (1u64, vec![], false),
            (42, vec![1, 2, 3], true),
            (u64::MAX, vec![0xab; 10_000], false),
        ] {
            let mut account = AccountSharedData::new_with_data(lamports, data.clone(), &owner);
            account.set_executable(executable);

            let mut hasher = blake3::Hasher::new();
            hasher.update(&lamports.to_le_bytes());
            hasher.update(&data);
            hasher.update(&[u8::from(executable)]);
            hasher.update(owner.as_ref());
            hasher.update(address.as_ref());
            let expected = Hash::new_from_array(*hasher.finalize().as_bytes());

            assert_eq!(account_hash(&account, &address), expected);
            assert_eq!(
                lt_hash_prefix(&AccountsDb::lt_hash_account(&account, &address).0),
                expected
            );
        }
    }

    /// Fixed vector, for other implementations: lamports 1_000_000, data
    /// `[0, 1, 2, 3]`, not executable, owner `[2; 32]`, address `[1; 32]`.
    /// Computed with the `blake3` crate over the SIMD-0215 input layout.
    #[test]
    fn test_account_hash_vector() {
        let address = Pubkey::new_from_array([1; 32]);
        let owner = Pubkey::new_from_array([2; 32]);
        let account = AccountSharedData::new_with_data(1_000_000, vec![0, 1, 2, 3], &owner);
        let mut input = Vec::new();
        input.extend_from_slice(&1_000_000u64.to_le_bytes());
        input.extend_from_slice(&[0, 1, 2, 3]);
        input.push(0);
        input.extend_from_slice(&[2; 32]);
        input.extend_from_slice(&[1; 32]);
        let expected = Hash::new_from_array(*blake3::hash(&input).as_bytes());
        assert_eq!(account_hash(&account, &address), expected);
        assert_eq!(
            expected.to_string(),
            "FHzVoswh9VG4k3TCDFGHHY2A1N4K1fDJ9i6buP7D33sH",
            "account_hash vector changed"
        );
    }

    #[test]
    fn test_zero_lamport_account_hash() {
        let address = Pubkey::new_unique();
        let account = AccountSharedData::new_with_data(0, vec![7; 100], &Pubkey::new_unique());
        assert_eq!(account_hash(&account, &address), DELETED_ACCOUNT_HASH);
        // The lattice hash of an empty account is the identity, whose prefix is
        // zero as well.
        assert_eq!(
            lt_hash_prefix(&AccountsDb::lt_hash_account(&account, &address).0),
            DELETED_ACCOUNT_HASH
        );
    }

    #[test]
    fn test_prefixes_chain() {
        let changed = Pubkey::new_unique();
        let restored = Pubkey::new_unique();
        let created_and_closed = Pubkey::new_unique();
        let closed = Pubkey::new_unique();
        let rewritten = Pubkey::new_unique();

        let mut a = AccountsDeltaPrefixes::default();
        let mut b = AccountsDeltaPrefixes::default();
        // changed: 1 -> 2 -> 3, the two updates hashed by different workers,
        // the later one first.
        b.record_prev(&changed, prefix(2));
        b.record_curr(&changed, prefix(3));
        a.record_prev(&changed, prefix(1));
        a.record_curr(&changed, prefix(2));
        // restored: 1 -> 2 -> 1
        a.record_prev(&restored, prefix(1));
        a.record_curr(&restored, prefix(2));
        b.record_prev(&restored, prefix(2));
        b.record_curr(&restored, prefix(1));
        // created and closed: empty -> 5 -> empty
        a.record_prev(&created_and_closed, DELETED_ACCOUNT_HASH);
        a.record_curr(&created_and_closed, prefix(5));
        a.record_prev(&created_and_closed, prefix(5));
        a.record_curr(&created_and_closed, DELETED_ACCOUNT_HASH);
        // closed: 6 -> empty, as a split update (two jobs).
        b.record_curr(&closed, DELETED_ACCOUNT_HASH);
        a.record_prev(&closed, prefix(6));
        // rewritten unchanged: 7 -> 7
        b.record_prev(&rewritten, prefix(7));
        b.record_curr(&rewritten, prefix(7));

        a.merge(b);
        assert_eq!(a.num_addresses(), 5);
        let delta = a.into_accounts_delta().unwrap();
        let mut expected = vec![
            DeltaEntry::new(changed.to_bytes(), prefix(3)),
            DeltaEntry::new(closed.to_bytes(), DELETED_ACCOUNT_HASH),
        ];
        expected.sort();
        assert_eq!(delta.entries(), expected.as_slice());
    }

    #[test]
    fn test_prefixes_inconsistent() {
        let address = Pubkey::new_unique();
        // Two updates from the same previous version: not a chain.
        let mut prefixes = AccountsDeltaPrefixes::default();
        prefixes.record_prev(&address, prefix(1));
        prefixes.record_curr(&address, prefix(2));
        prefixes.record_prev(&address, prefix(1));
        prefixes.record_curr(&address, prefix(3));
        assert_eq!(
            prefixes.into_accounts_delta(),
            Err(InconsistentAccountUpdates {
                address,
                num_prefixes: 3
            })
        );
        // A lone previous version.
        let mut prefixes = AccountsDeltaPrefixes::default();
        prefixes.record_prev(&address, prefix(1));
        assert!(prefixes.into_accounts_delta().is_err());
    }
}

/// Bank-level tests. Every bank is checked against an independent
/// computation of its changed accounts: the full account set of the parent
/// and of the bank, read back from accounts-db by scanning, compared field by
/// field, with the account hashes computed by the `blake3` crate.
#[cfg(test)]
mod bank_tests {
    use {
        super::*,
        crate::{
            bank::block_commitment::{
                COMMITMENT_VECTOR_LEN, CommitmentSlot, CommitmentVector, INACTIVE_COMMITMENT_SLOT,
                tests::assert_commitment_failure,
            },
            bank_forks::BankForks,
            genesis_utils::{
                GenesisConfigInfo, ValidatorVoteKeypairs, activate_feature,
                create_genesis_config_with_leader, create_genesis_config_with_vote_accounts,
            },
        },
        agave_votor_messages::consensus_message::BlockId,
        solana_account::{AccountSharedData, WritableAccount, state_traits::StateMutWincode as _},
        solana_epoch_schedule::EpochSchedule,
        solana_feature_gate_interface::{self as feature, Feature},
        solana_fee_structure::FeeStructure,
        solana_genesis_config::GenesisConfig,
        solana_keypair::Keypair,
        solana_merkle_tree::{
            accounts_delta::{UnchangedProof, compute_accounts_delta_root},
            commitment_tree::TreeId,
        },
        solana_native_token::LAMPORTS_PER_SOL,
        solana_sdk_ids::sysvar,
        solana_signer::Signer,
        solana_system_interface::instruction as system_instruction,
        solana_transaction::Transaction,
        solana_vote_interface::state::{VoteStateV4, VoteStateVersions},
        solana_vote_program::vote_state::handler::VoteStateHandler,
        std::{
            collections::{BTreeMap, HashMap, HashSet},
            sync::RwLock,
        },
    };

    const SLOTS_PER_EPOCH: u64 = 32;

    struct Setup {
        genesis_config: GenesisConfig,
        mint_keypair: Keypair,
    }

    /// Genesis with both features active (or as given), short epochs and
    /// non-zero fees.
    fn setup(base_active: bool, delta: Option<Option<u64>>) -> Setup {
        let GenesisConfigInfo {
            mut genesis_config,
            mint_keypair,
            ..
        } = create_genesis_config_with_leader(
            1_000_000 * LAMPORTS_PER_SOL,
            &Pubkey::new_unique(),
            100 * LAMPORTS_PER_SOL,
        );
        genesis_config.epoch_schedule =
            EpochSchedule::custom(SLOTS_PER_EPOCH, SLOTS_PER_EPOCH, false);
        if base_active {
            activate_feature(
                &mut genesis_config,
                feature_set::block_commitment_root::id(),
            );
        }
        // None: no feature account; Some(None): pending; Some(Some(0)): active.
        if let Some(activated_at) = delta {
            genesis_config.accounts.insert(
                feature_set::accounts_delta_root::id(),
                feature::create_account(
                    &Feature { activated_at },
                    genesis_config
                        .rent
                        .minimum_balance(Feature::size_of())
                        .max(1),
                )
                .into(),
            );
        }
        Setup {
            genesis_config,
            mint_keypair,
        }
    }

    fn setup_active() -> Setup {
        setup(true, Some(Some(0)))
    }

    /// Independent `account_hash`.
    fn blake3_account_hash(address: &Pubkey, account: &AccountSharedData) -> Hash {
        if account.lamports() == 0 {
            return DELETED_ACCOUNT_HASH;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(&account.lamports().to_le_bytes());
        hasher.update(account.data());
        hasher.update(&[u8::from(account.executable())]);
        hasher.update(account.owner().as_ref());
        hasher.update(address.as_ref());
        Hash::new_from_array(*hasher.finalize().as_bytes())
    }

    /// The accounts with lamports visible from `bank`.
    fn account_set(bank: &Bank) -> HashMap<Pubkey, AccountSharedData> {
        let mut accounts = HashMap::new();
        bank.scan_all_accounts(|item| {
            if let Some((address, account, _slot)) = item
                && account.lamports() != 0
            {
                accounts.insert(*address, account);
            }
        })
        .unwrap();
        accounts
    }

    /// `(lamports, data, executable, owner)`; the empty state for `None`.
    fn same_state(a: Option<&AccountSharedData>, b: Option<&AccountSharedData>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                a.lamports() == b.lamports()
                    && a.data() == b.data()
                    && a.executable() == b.executable()
                    && a.owner() == b.owner()
            }
            _ => false,
        }
    }

    /// Freezes `bank`, gives it a block id, and checks its accounts delta
    /// against `parent` (`None` for a bank without parent). Returns the delta.
    fn freeze_and_check(parent: Option<&Bank>, bank: &Bank) -> Arc<AccountsDelta> {
        bank.freeze();
        bank.set_block_id(Some(BlockId::from(Hash::new_unique())));
        let delta = bank.accounts_delta().expect("bank has an accounts delta");

        let before = parent.map(account_set).unwrap_or_default();
        let after = account_set(bank);

        // 1. The leaves are exactly the accounts whose state changed, with the
        //    account hash of their state after the block, sorted.
        let mut expected = BTreeMap::new();
        for address in before.keys().chain(after.keys()) {
            let (b, a) = (before.get(address), after.get(address));
            if !same_state(b, a) {
                let account_hash = a
                    .map(|a| blake3_account_hash(address, a))
                    .unwrap_or(DELETED_ACCOUNT_HASH);
                expected.insert(address.to_bytes(), account_hash);
            }
        }
        let expected: Vec<_> = expected
            .into_iter()
            .map(|(address, account_hash)| DeltaEntry::new(address, account_hash))
            .collect();
        assert_eq!(delta.entries(), expected.as_slice(), "slot {}", bank.slot());
        assert_eq!(delta.root(), compute_accounts_delta_root(&expected));

        // 2. Applying the leaves to the parent's account set gives the bank's.
        let mut applied: HashMap<_, _> = before
            .iter()
            .map(|(address, account)| (*address, blake3_account_hash(address, account)))
            .collect();
        for entry in delta.entries() {
            let address = Pubkey::new_from_array(entry.address);
            if entry.is_deletion() {
                assert!(applied.remove(&address).is_some());
            } else {
                applied.insert(address, entry.account_hash);
            }
        }
        let after_hashes: HashMap<_, _> = after
            .iter()
            .map(|(address, account)| (*address, blake3_account_hash(address, account)))
            .collect();
        assert_eq!(applied, after_hashes, "slot {}", bank.slot());

        // 3. The lattice hash moves from the parent's to the bank's by
        //    exactly the changed accounts.
        let mut lt_hash = parent
            .map(|parent| parent.accounts_lt_hash.lock().unwrap().0.clone())
            .unwrap_or_else(LtHash::identity);
        for entry in delta.entries() {
            let address = Pubkey::new_from_array(entry.address);
            if let Some(account) = before.get(&address) {
                lt_hash.mix_out(&AccountsDb::lt_hash_account(account, &address).0);
            }
            if let Some(account) = after.get(&address) {
                lt_hash.mix_in(&AccountsDb::lt_hash_account(account, &address).0);
            }
        }
        assert_eq!(lt_hash, bank.accounts_lt_hash.lock().unwrap().0);
        // The bank's lattice hash also matches a full recomputation.
        assert_eq!(
            *bank.accounts_lt_hash.lock().unwrap(),
            bank.calculate_accounts_lt_hash_for_tests()
        );

        // 4. Slot 2 of the commitment vector is the root.
        let commitment = bank.block_commitment().unwrap();
        let vector = commitment_vector(bank, &delta);
        assert_eq!(vector.root(), commitment.commitment_root);
        delta
    }

    /// The commitment vector of a frozen bank of this branch: slot 0 is the
    /// history root, slot 2 the accounts delta root, the others are zero.
    fn commitment_vector(bank: &Bank, delta: &AccountsDelta) -> CommitmentVector {
        let mut vector = [INACTIVE_COMMITMENT_SLOT; COMMITMENT_VECTOR_LEN];
        vector[CommitmentSlot::BlockHistoryRoot.index()] =
            bank.block_commitment().unwrap().block_history_root();
        vector[CommitmentSlot::AccountsDeltaRoot.index()] = delta.root();
        CommitmentVector(vector)
    }

    fn new_child(bank_forks: &RwLock<BankForks>, parent: &Arc<Bank>, slot: u64) -> Arc<Bank> {
        let leader = *parent.leader();
        Bank::new_from_parent_with_bank_forks(bank_forks, parent.clone(), leader, slot)
    }

    /// A transfer of `lamports` from `from` to `to`, with the fees paid by
    /// `payer`.
    fn transfer_paid_by(bank: &Bank, payer: &Keypair, from: &Keypair, to: &Pubkey, lamports: u64) {
        bank.register_unique_recent_blockhash_for_test();
        let tx = Transaction::new_signed_with_payer(
            &[system_instruction::transfer(&from.pubkey(), to, lamports)],
            Some(&payer.pubkey()),
            &[payer, from],
            bank.last_blockhash(),
        );
        bank.process_transaction(&tx).unwrap();
    }

    fn entry_of(delta: &AccountsDelta, address: &Pubkey) -> Option<Hash> {
        delta
            .get(&address.to_bytes())
            .map(|entry| entry.account_hash)
    }

    #[test]
    fn test_inactive_features() {
        // Base feature only.
        let Setup { genesis_config, .. } = setup(true, None);
        let (bank0, bank_forks) = Bank::new_with_bank_forks_for_tests(&genesis_config);
        assert!(!bank0.is_accounts_delta_root_active());
        bank0.freeze();
        bank0.set_block_id(Some(BlockId::from(Hash::new_unique())));
        assert!(bank0.accounts_delta().is_none());
        let bank1 = new_child(&bank_forks, &bank0, 1);
        // Neither feature is active in the parent and no epoch starts: the
        // child does not collect.
        assert!(!bank1.collects_accounts_delta());
        bank1.freeze();
        assert!(bank1.accounts_delta().is_none());
        let vector = bank1
            .compute_commitment_vector(&crate::bank::block_commitment::CommitmentInputs {
                block_history: &bank1.block_commitment().unwrap().block_history,
                parent: Some(&bank0),
            })
            .unwrap();
        assert_eq!(
            vector.get(CommitmentSlot::AccountsDeltaRoot),
            INACTIVE_COMMITMENT_SLOT
        );
        // The first bank of an epoch collects, in case a feature activates.
        let bank32 = new_child(&bank_forks, &bank1, SLOTS_PER_EPOCH);
        assert!(bank32.collects_accounts_delta());
        bank32.freeze();
        assert!(bank32.accounts_delta().is_none());

        // Delta feature only: no root without the base feature, and
        // `is_accounts_delta_root_active` requires both.
        let Setup { genesis_config, .. } = setup(false, Some(Some(0)));
        let (bank0, bank_forks) = Bank::new_with_bank_forks_for_tests(&genesis_config);
        assert!(
            bank0
                .feature_set
                .is_active(&feature_set::accounts_delta_root::id())
        );
        assert!(!bank0.is_accounts_delta_root_active());
        bank0.freeze();
        assert!(bank0.accounts_delta().is_none());
        assert!(bank0.block_commitment().is_none());
        let bank1 = new_child(&bank_forks, &bank0, 1);
        assert!(!bank1.collects_accounts_delta());
    }

    /// A bank that computes the root but did not keep the account hashes from
    /// its first update (its feature set changed after it was created) has a
    /// commitment failure instead of a zero slot 2.
    #[test]
    fn test_accounts_delta_not_collected_is_commitment_failure() {
        let Setup { genesis_config, .. } = setup(true, None);
        let (bank0, _bank_forks) = Bank::new_with_bank_forks_for_tests(&genesis_config);
        bank0.freeze();
        bank0.set_block_id(Some(BlockId::from(Hash::new_unique())));
        let mut bank1 = Bank::new_from_parent(bank0.clone(), *bank0.leader(), 1);
        assert!(!bank1.collects_accounts_delta());
        bank1.activate_feature(&feature_set::accounts_delta_root::id());
        assert!(bank1.is_accounts_delta_root_active());
        bank1.freeze();
        assert_eq!(bank1.accounts_delta(), None);
        assert_eq!(bank1.accounts_delta_root(), None);
        assert_commitment_failure(&bank1, CommitmentFailure::AccountsDeltaNotCollected);
    }

    /// Account updates that do not chain (here a previous version without
    /// its update) are a commitment failure, not a panic.
    #[test]
    fn test_inconsistent_account_updates_is_commitment_failure() {
        let Setup { genesis_config, .. } = setup_active();
        let (bank0, bank_forks) = Bank::new_with_bank_forks_for_tests(&genesis_config);
        freeze_and_check(None, &bank0);
        let bank1 = new_child(&bank_forks, &bank0, 1);
        let address = Pubkey::new_unique();
        bank1
            .accounts_lt_hash_async_progress
            .record_unmatched_prev_for_tests(&address, Hash::new_unique());
        bank1.freeze();
        assert_eq!(bank1.accounts_delta(), None);
        assert_eq!(bank1.accounts_delta_root(), None);
        assert_commitment_failure(
            &bank1,
            CommitmentFailure::InconsistentAccountUpdates {
                address,
                num_prefixes: 1,
            },
        );
    }

    #[test]
    fn test_genesis_bank() {
        let Setup { genesis_config, .. } = setup_active();
        let (bank0, _bank_forks) = Bank::new_with_bank_forks_for_tests(&genesis_config);
        // Every account of the genesis bank is new.
        let delta = freeze_and_check(None, &bank0);
        assert_eq!(delta.len(), account_set(&bank0).len());
    }

    #[test]
    fn test_transfers_and_runtime_changes() {
        let Setup {
            genesis_config,
            mint_keypair,
        } = setup_active();
        let mut bank0 = Bank::new_for_tests(&genesis_config);
        bank0.set_fee_structure(&FeeStructure {
            lamports_per_signature: 5000,
            ..FeeStructure::default()
        });
        let (bank0, bank_forks) = bank0.wrap_with_bank_forks_for_tests();
        freeze_and_check(None, &bank0);
        let bank1 = new_child(&bank_forks, &bank0, 1);
        assert!(bank1.collects_accounts_delta());

        let alice = Pubkey::new_unique();
        let bob = Pubkey::new_unique();
        bank1.register_unique_recent_blockhash_for_test();
        bank1
            .transfer(LAMPORTS_PER_SOL, &mint_keypair, &alice)
            .unwrap();
        bank1.register_unique_recent_blockhash_for_test();
        bank1
            .transfer(2 * LAMPORTS_PER_SOL, &mint_keypair, &bob)
            .unwrap();
        let leader = *bank1.leader();
        let leader_balance =
            |bank: &Bank| bank.get_balance(&leader.id) + bank.get_balance(&leader.vote_address);
        let leader_balance_before_freeze = leader_balance(&bank1);
        let delta = freeze_and_check(Some(&bank0), &bank1);

        for address in [mint_keypair.pubkey(), alice, bob] {
            let account = bank1.get_account(&address).unwrap();
            assert_eq!(
                entry_of(&delta, &address),
                Some(account_hash(&account, &address))
            );
        }
        // Fees are distributed to the leader at freeze; the leaf has the state
        // after the distribution.
        assert!(leader_balance(&bank1) > leader_balance_before_freeze);
        for address in [leader.id, leader.vote_address] {
            if bank1.get_balance(&address) != bank0.get_balance(&address) {
                let account = bank1.get_account(&address).unwrap();
                assert_eq!(
                    entry_of(&delta, &address),
                    Some(account_hash(&account, &address))
                );
            }
        }
        // Sysvars updated when the bank is created and when it is frozen.
        for sysvar in [
            sysvar::clock::id(),
            sysvar::slot_hashes::id(),
            sysvar::slot_history::id(),
        ] {
            let account = bank1.get_account(&sysvar).unwrap();
            assert_eq!(
                entry_of(&delta, &sysvar),
                Some(account_hash(&account, &sysvar)),
                "{sysvar}"
            );
        }
        // An account that was not touched is not a leaf.
        assert!(entry_of(&delta, &sysvar::rent::id()).is_none());
    }

    /// The accounts delta is built once, when the bank is frozen. Building
    /// it again (a second `finish_accounts_delta`, which a second
    /// `finish_accounts_lt_hash_updates` would do), freezing again or
    /// rehashing changes neither the delta nor the commitment root; without
    /// the guard a second build would take the already taken prefixes and
    /// replace the delta with an empty one.
    #[test]
    fn test_accounts_delta_is_built_once() {
        let Setup {
            genesis_config,
            mint_keypair,
        } = setup_active();
        let (bank0, bank_forks) = Bank::new_with_bank_forks_for_tests(&genesis_config);
        freeze_and_check(None, &bank0);
        let bank1 = new_child(&bank_forks, &bank0, 1);
        let alice = Pubkey::new_unique();
        bank1.register_unique_recent_blockhash_for_test();
        bank1
            .transfer(LAMPORTS_PER_SOL, &mint_keypair, &alice)
            .unwrap();
        let delta = freeze_and_check(Some(&bank0), &bank1);
        assert!(entry_of(&delta, &alice).is_some());
        let lt_hash = bank1.accounts_lt_hash.lock().unwrap().clone();
        let hash = bank1.hash();
        let commitment_root = bank1.commitment_root();

        // The prefixes were taken at freeze: what is left is empty.
        assert_eq!(
            bank1
                .accounts_lt_hash_async_progress
                .take_accounts_delta_prefixes()
                .map(|prefixes| prefixes.num_addresses()),
            Some(0)
        );
        assert_eq!(bank1.finish_accounts_delta(), 0);
        bank1.freeze();
        bank1.rehash();
        assert!(Arc::ptr_eq(&bank1.accounts_delta().unwrap(), &delta));
        assert_eq!(bank1.accounts_delta_root(), Some(delta.root()));
        assert_eq!(*bank1.accounts_lt_hash.lock().unwrap(), lt_hash);
        assert_eq!(bank1.hash(), hash);
        assert_eq!(bank1.commitment_root(), commitment_root);
        assert_eq!(bank1.verify_commitment_root(), Ok(()));
    }

    #[test]
    fn test_no_transactions() {
        let Setup { genesis_config, .. } = setup_active();
        let (bank0, bank_forks) = Bank::new_with_bank_forks_for_tests(&genesis_config);
        freeze_and_check(None, &bank0);
        let bank1 = new_child(&bank_forks, &bank0, 1);
        // A block without transactions still changes the sysvars, which are
        // its only leaves.
        let delta = freeze_and_check(Some(&bank0), &bank1);
        assert!(!delta.is_empty());
        for entry in delta.entries() {
            let address = Pubkey::new_from_array(entry.address);
            assert_eq!(
                bank1.get_account(&address).unwrap().owner(),
                &sysvar::id(),
                "{address}"
            );
        }
    }

    #[test]
    fn test_closed_created_and_restored_accounts() {
        let Setup {
            genesis_config,
            mint_keypair,
        } = setup_active();
        let (bank0, bank_forks) = Bank::new_with_bank_forks_for_tests(&genesis_config);
        freeze_and_check(None, &bank0);

        let closed = Keypair::new();
        let restored = Keypair::new();
        let restored_peer = Keypair::new();
        let rewritten = Keypair::new();
        let amount = LAMPORTS_PER_SOL;
        let bank1 = new_child(&bank_forks, &bank0, 1);
        for account in [&closed, &restored, &restored_peer, &rewritten] {
            bank1.register_unique_recent_blockhash_for_test();
            bank1
                .transfer(amount, &mint_keypair, &account.pubkey())
                .unwrap();
        }
        freeze_and_check(Some(&bank0), &bank1);

        let bank2 = new_child(&bank_forks, &bank1, 2);
        let sink = Pubkey::new_unique();
        // Closed: all lamports moved out, the mint pays the fee.
        transfer_paid_by(&bank2, &mint_keypair, &closed, &sink, amount);
        assert_eq!(bank2.get_balance(&closed.pubkey()), 0);
        // Created and closed in the same block.
        let ephemeral = Keypair::new();
        bank2.register_unique_recent_blockhash_for_test();
        bank2
            .transfer(amount, &mint_keypair, &ephemeral.pubkey())
            .unwrap();
        transfer_paid_by(&bank2, &mint_keypair, &ephemeral, &sink, amount);
        assert_eq!(bank2.get_balance(&ephemeral.pubkey()), 0);
        // Changed and restored: one lamport there and back.
        transfer_paid_by(&bank2, &mint_keypair, &restored, &restored_peer.pubkey(), 1);
        transfer_paid_by(&bank2, &mint_keypair, &restored_peer, &restored.pubkey(), 1);
        // Lamports sent to the incinerator are burned at freeze: the
        // incinerator is created and emptied in the block.
        bank2.register_unique_recent_blockhash_for_test();
        bank2
            .transfer(amount, &mint_keypair, &solana_sdk_ids::incinerator::id())
            .unwrap();
        // Written back unchanged, outside a transaction.
        let account = bank2.get_account(&rewritten.pubkey()).unwrap();
        bank2.store_account(&rewritten.pubkey(), &account);
        let delta = freeze_and_check(Some(&bank1), &bank2);

        assert_eq!(
            entry_of(&delta, &closed.pubkey()),
            Some(DELETED_ACCOUNT_HASH)
        );
        assert!(
            delta
                .get(&closed.pubkey().to_bytes())
                .unwrap()
                .is_deletion()
        );
        assert!(entry_of(&delta, &ephemeral.pubkey()).is_none());
        assert!(entry_of(&delta, &restored.pubkey()).is_none());
        assert!(entry_of(&delta, &restored_peer.pubkey()).is_none());
        assert!(entry_of(&delta, &rewritten.pubkey()).is_none());
        assert_eq!(bank2.get_balance(&solana_sdk_ids::incinerator::id()), 0);
        assert!(entry_of(&delta, &solana_sdk_ids::incinerator::id()).is_none());
        assert!(entry_of(&delta, &sink).is_some());

        // The deletion is provable, and so is "unchanged" for the others.
        let root = delta.root();
        let proof = delta.inclusion_proof(&closed.pubkey().to_bytes()).unwrap();
        assert!(proof.verify_account(&closed.pubkey().to_bytes(), &DELETED_ACCOUNT_HASH, &root));
        for address in [ephemeral.pubkey(), restored.pubkey(), rewritten.pubkey()] {
            let proof = delta.unchanged_proof(&address.to_bytes()).unwrap();
            assert!(proof.verify(&address.to_bytes(), &root));
        }
    }

    /// Accounts rewritten several times in one block, by transactions or by
    /// stores outside transactions, keep only their last state, and a large
    /// account (hashed as two jobs) is handled like any other.
    #[test]
    fn test_many_updates_and_large_account() {
        let Setup {
            genesis_config,
            mint_keypair,
        } = setup_active();
        let (bank0, bank_forks) = Bank::new_with_bank_forks_for_tests(&genesis_config);
        freeze_and_check(None, &bank0);
        let bank1 = new_child(&bank_forks, &bank0, 1);
        let large = Pubkey::new_unique();
        let owner = Pubkey::new_unique();
        let mut account =
            AccountSharedData::new_with_data(LAMPORTS_PER_SOL, vec![0; 100_000], &owner);
        bank1.store_account(&large, &account);
        freeze_and_check(Some(&bank0), &bank1);

        let bank2 = new_child(&bank_forks, &bank1, 2);
        let target = Pubkey::new_unique();
        // Stores outside transactions come before the first transaction (or
        // after the last), as `enqueue_off_chain_accounts_lt_hash_updates()`
        // requires.
        for i in 0..5u8 {
            account.data_as_mut_slice()[0] = i + 1;
            bank2.store_account(&large, &account);
        }
        // Each transaction is committed, and its updates queued, separately;
        // the hashing manager may merge the queued updates of `target`.
        for _ in 0..5 {
            bank2.register_unique_recent_blockhash_for_test();
            bank2
                .transfer(LAMPORTS_PER_SOL, &mint_keypair, &target)
                .unwrap();
        }
        let delta = freeze_and_check(Some(&bank1), &bank2);
        let large_account = bank2.get_account(&large).unwrap();
        assert_eq!(large_account.data()[0], 5);
        assert_eq!(
            entry_of(&delta, &large),
            Some(blake3_account_hash(&large, &large_account))
        );
        assert_eq!(bank2.get_balance(&target), 5 * LAMPORTS_PER_SOL);
    }

    /// The feature activates at an epoch boundary; the boundary bank, its
    /// rewards and the following banks are all covered.
    #[test]
    fn test_activation_and_epoch_boundary() {
        let Setup {
            genesis_config,
            mint_keypair,
        } = setup(true, Some(None));
        let (bank0, bank_forks) = Bank::new_with_bank_forks_for_tests(&genesis_config);
        assert!(!bank0.is_accounts_delta_root_active());
        bank0.freeze();
        bank0.set_block_id(Some(BlockId::from(Hash::new_unique())));
        let mut parent = bank0;
        for slot in [1, SLOTS_PER_EPOCH - 1] {
            let bank = new_child(&bank_forks, &parent, slot);
            assert!(!bank.collects_accounts_delta());
            bank.freeze();
            bank.set_block_id(Some(BlockId::from(Hash::new_unique())));
            assert!(bank.accounts_delta().is_none());
            parent = bank;
        }

        // The first bank of epoch 1 activates the feature (storing the feature
        // account) and computes rewards.
        let boundary = new_child(&bank_forks, &parent, SLOTS_PER_EPOCH);
        assert!(boundary.is_accounts_delta_root_active());
        assert!(boundary.collects_accounts_delta());
        let delta = freeze_and_check(Some(&parent), &boundary);
        let feature_account = boundary
            .get_account(&feature_set::accounts_delta_root::id())
            .unwrap();
        assert_eq!(
            feature::from_account(&feature_account)
                .unwrap()
                .activated_at,
            Some(SLOTS_PER_EPOCH)
        );
        assert_eq!(
            entry_of(&delta, &feature_set::accounts_delta_root::id()),
            Some(account_hash(
                &feature_account,
                &feature_set::accounts_delta_root::id()
            ))
        );

        // Following banks, through reward distribution and into epoch 2.
        let mut owners_changed = HashMap::<Pubkey, usize>::new();
        for entry in delta.entries() {
            let address = Pubkey::new_from_array(entry.address);
            if let Some(account) = boundary.get_account(&address) {
                *owners_changed.entry(*account.owner()).or_default() += 1;
            }
        }
        let mut parent = boundary;
        for slot in (SLOTS_PER_EPOCH + 1..SLOTS_PER_EPOCH + 4)
            .chain([2 * SLOTS_PER_EPOCH, 2 * SLOTS_PER_EPOCH + 1])
        {
            let bank = new_child(&bank_forks, &parent, slot);
            assert!(bank.collects_accounts_delta());
            bank.register_unique_recent_blockhash_for_test();
            bank.transfer(1_000_000, &mint_keypair, &Pubkey::new_unique())
                .unwrap();
            let delta = freeze_and_check(Some(&parent), &bank);
            for entry in delta.entries() {
                let address = Pubkey::new_from_array(entry.address);
                if let Some(account) = bank.get_account(&address) {
                    *owners_changed.entry(*account.owner()).or_default() += 1;
                }
            }
            parent = bank;
        }
        // Rewards were paid out to stake or vote accounts in these banks.
        assert!(
            owners_changed.contains_key(&solana_sdk_ids::stake::id())
                || owners_changed.contains_key(&solana_sdk_ids::vote::id()),
            "{owners_changed:?}"
        );
    }

    /// Partitioned epoch rewards: computed in the first bank of the epoch and
    /// stored into the stake accounts in the following banks (with the
    /// accounts loaded in parallel on the rewards thread pool).
    #[test]
    fn test_partitioned_epoch_rewards() {
        let validator_keypairs: Vec<_> =
            (0..4).map(|_| ValidatorVoteKeypairs::new_rand()).collect();
        let GenesisConfigInfo {
            mut genesis_config,
            mint_keypair,
            ..
        } = create_genesis_config_with_vote_accounts(
            1_000_000 * LAMPORTS_PER_SOL,
            &validator_keypairs,
            vec![2_000_000_000; validator_keypairs.len()],
        );
        genesis_config.epoch_schedule =
            EpochSchedule::custom(SLOTS_PER_EPOCH, SLOTS_PER_EPOCH, false);
        activate_feature(
            &mut genesis_config,
            feature_set::block_commitment_root::id(),
        );
        activate_feature(&mut genesis_config, feature_set::accounts_delta_root::id());
        let (bank0, bank_forks) = Bank::new_with_bank_forks_for_tests(&genesis_config);
        // Give the vote accounts credits in the epoch of `bank`, so that the
        // epoch pays stake rewards (as the partitioned rewards tests do).
        let give_credits = |bank: &Bank| {
            for keypairs in &validator_keypairs {
                let vote_address = keypairs.vote_keypair.pubkey();
                let mut vote_account = bank.get_account(&vote_address).unwrap();
                let mut vote_state = VoteStateHandler::new_v4(
                    VoteStateV4::deserialize(vote_account.data(), &vote_address).unwrap(),
                );
                vote_state.set_commission(0);
                vote_state.increment_credits(bank.epoch(), 42);
                let versioned = VoteStateVersions::V4(Box::new(vote_state.as_ref_v4().clone()));
                vote_account.set_state(&versioned).unwrap();
                bank.store_account_and_update_capitalization(&vote_address, &vote_account);
            }
        };
        give_credits(&bank0);
        freeze_and_check(None, &bank0);

        let stakers: Vec<_> = validator_keypairs
            .iter()
            .map(|keypairs| keypairs.stake_keypair.pubkey())
            .collect();
        let mut rewarded = HashSet::new();
        let mut parent = bank0;
        for slot in [1]
            .into_iter()
            .chain(SLOTS_PER_EPOCH..SLOTS_PER_EPOCH + 4)
            .chain(2 * SLOTS_PER_EPOCH..2 * SLOTS_PER_EPOCH + 4)
        {
            let bank = new_child(&bank_forks, &parent, slot);
            if slot == SLOTS_PER_EPOCH {
                give_credits(&bank);
            }
            bank.register_unique_recent_blockhash_for_test();
            bank.transfer(1_000_000, &mint_keypair, &Pubkey::new_unique())
                .unwrap();
            let delta = freeze_and_check(Some(&parent), &bank);
            for staker in &stakers {
                if bank.get_balance(staker) > parent.get_balance(staker) {
                    // Checked in full by `freeze_and_check()`; the reward is
                    // a change, so the stake account is a leaf.
                    assert!(delta.get(&staker.to_bytes()).is_some());
                    rewarded.insert(*staker);
                }
            }
            parent = bank;
        }
        assert_eq!(
            rewarded.len(),
            stakers.len(),
            "every stake account is rewarded"
        );
    }

    /// Warping (as ledger-tool does) creates and freezes the bank in one go.
    #[test]
    fn test_warp() {
        let Setup { genesis_config, .. } = setup_active();
        let (bank0, _bank_forks) = Bank::new_with_bank_forks_for_tests(&genesis_config);
        freeze_and_check(None, &bank0);
        let leader = *bank0.leader();
        let warped = Bank::warp_from_parent(bank0.clone(), leader, 3 * SLOTS_PER_EPOCH + 5);
        assert!(warped.is_frozen());
        // `freeze_and_check` freezes again, which is a no-op.
        freeze_and_check(Some(&bank0), &warped);
    }

    #[test]
    fn test_proof_from_commitment_root() {
        let Setup {
            genesis_config,
            mint_keypair,
        } = setup_active();
        let (bank0, bank_forks) = Bank::new_with_bank_forks_for_tests(&genesis_config);
        freeze_and_check(None, &bank0);
        let bank1 = new_child(&bank_forks, &bank0, 1);
        let alice = Pubkey::new_unique();
        bank1.register_unique_recent_blockhash_for_test();
        bank1
            .transfer(LAMPORTS_PER_SOL, &mint_keypair, &alice)
            .unwrap();
        let delta = freeze_and_check(Some(&bank0), &bank1);
        let commitment_root = bank1.commitment_root().unwrap();

        // What a light client holds: `commitment_root` (from the footer, bound
        // into the block id) and the account state claimed by a server.
        let claimed = bank1.get_account(&alice).unwrap();
        let claimed_hash = blake3_account_hash(&alice, &claimed);

        // Server side: open slot 2 of the commitment vector, then the leaf.
        let vector = commitment_vector(&bank1, &delta);
        let slot_proof = vector
            .inclusion_proof(CommitmentSlot::AccountsDeltaRoot.index())
            .unwrap();
        let leaf_proof = delta.inclusion_proof(&alice.to_bytes()).unwrap();
        let accounts_delta_root = delta.root();

        // Client side.
        assert!(slot_proof.verify(
            TreeId::COMMITMENT_VECTOR,
            accounts_delta_root.as_ref(),
            &commitment_root
        ));
        assert_eq!(slot_proof.leaf_index, 2);
        assert!(leaf_proof.verify_account(&alice.to_bytes(), &claimed_hash, &accounts_delta_root));
        // A different claimed state does not verify.
        let mut forged = claimed.clone();
        forged.set_lamports(claimed.lamports() + 1);
        assert!(!leaf_proof.verify_account(
            &alice.to_bytes(),
            &blake3_account_hash(&alice, &forged),
            &accounts_delta_root
        ));
        // An account not changed in this block: adjacency proof.
        let untouched = sysvar::rent::id();
        let unchanged = delta.unchanged_proof(&untouched.to_bytes()).unwrap();
        assert!(!matches!(unchanged, UnchangedProof::EmptyTree));
        assert!(unchanged.verify(&untouched.to_bytes(), &accounts_delta_root));
    }
}
