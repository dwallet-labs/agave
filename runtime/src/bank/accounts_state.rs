//! Accounts State Root (draft SIMD): commitment slot 5, `accounts_state_root`.
//!
//! ```text
//! accounts_state_root(B) = sha256(0x84 || u64_le(G) || base_root(G) || overlay_root(G, B))
//! ```
//!
//! The trees, the per-block overlay update, the derivation of base leaves and
//! the proofs live in [`solana_merkle_tree::accounts_state`]; this module keeps
//! the overlays of each bank, obtains the base trees, and computes the slot.
//!
//! # Generations
//!
//! A bank `B` in epoch `E` uses generation `G = E - 1` if
//! `B.slot < switch_slot(E)` and `G = E` otherwise, where
//! `switch_slot(E) = first_slot(E) + slots_in_epoch(E) / 2`. Generation `E`
//! exists on a chain if the features are active in epoch `E` and the epoch
//! has at least [`MIN_SLOTS_PER_EPOCH_FOR_GENERATION`] slots; it is created by
//! the first block of epoch `E` on the chain (`F`, whose parent is `P`), and
//! kept by every later block of the chain until `switch_slot(E + 1)`. A bank
//! therefore holds two generations from the first block of an epoch until its
//! switch slot, and one otherwise ([`AccountsState`]). If `G` does not exist
//! (activation epoch before its switch slot, short epoch, or no block of epoch
//! `G` on the chain), slot 5 is zero.
//!
//! # Overlays
//!
//! Each generation holds a persistent [`OverlayTree`] (structural sharing,
//! `O(1)` clone). When a bank is frozen, its generations are those of its
//! parent, filtered by the rules above, each updated with the changed accounts
//! of the bank as defined by Accounts Delta Root: the account hash after the
//! block, and for a first insertion the hash before the block. Both come from
//! the prefixes that the accounts delta collection keeps at the lattice hash
//! choke point (see [`super::accounts_delta`]); no account is hashed again.
//!
//! # Base trees
//!
//! `base_root(E)` is the root over the state at the end of `P`. It is held by
//! a [`BaseCell`] shared by every bank of the chain from `F` on, and computed
//! off the block path by a job started when `F` is frozen:
//!
//! - **By merge** (steady state): if `P` holds a generation `G'`, the base
//!   leaves are the base leaves of `G'` with the overlay of `G'` as of `P`
//!   applied. `P` is frozen, so that overlay is immutable; the job holds a
//!   clone of it. No account data is read. The job waits for the base of `G'`
//!   if that is still being computed.
//! - **By scan** (activation, or after an epoch without a generation): the
//!   state of `P` is read from accounts-db with `Bank::scan_all_accounts`,
//!   which pins the roots it reads against cleaning. The scanned state is
//!   checked against `P`'s accounts lattice hash, so a scan that did not see
//!   exactly `P`'s state (for example because a later root was set before the
//!   scan started) fails instead of producing a wrong base.
//! - **After a restart from a snapshot**: the snapshot carries the overlays of
//!   the snapshot bank `S` and the base roots already computed, but no base
//!   leaves. The leaves of every generation of `S` are rebuilt by one scan of
//!   `S`: an address without an overlay entry has its current state in the
//!   base, an address with one has its base hash in the entry. The scan starts
//!   when the first child of `S` is created; slot 5 uses the base roots from
//!   the snapshot without waiting for it. A rebuilt root that differs from
//!   the snapshot's makes the snapshot invalid: from then on every bank that
//!   needs that base has a
//!   [`CommitmentFailure::AccountsStateSnapshotBaseMismatch`], with no
//!   fallback to either root.
//!
//! Base cells are keyed by `(E, P.slot, P.bank_hash)`, so forks with different
//! `P` get different bases and forks with the same `P` share one job.
//!
//! A bank that must compute slot 5 with generation `E` (at or after
//! `switch_slot(E)`) waits for its base. If the job failed, the bank rebuilds
//! the base synchronously by a scan of its parent (verified the same way)
//! and, if that fails too, has a [`CommitmentFailure`].
//!
//! # Cost
//!
//! Per block, for each changed account and each generation in use, one path
//! update in the overlay (`O(depth)` node copies) and the hashes of the changed
//! nodes, each hashed once. A block with many changed accounts is divided
//! along the top branches of the overlay and spread over a few threads
//! ([`update_overlays`]). An overlay entry costs a leaf node and a branch node
//! (about 250 bytes) and is shared by every bank that holds the generation.
//!
//! The leaves of a base tree, 64 bytes per account, are in a file that is
//! mapped, not in memory (see [`base_files`]): the leaves of the two latest
//! epochs are kept on disk. A base is derived as a stream, by merge from the
//! file of the previous base, or by scan through a sorter that spills to
//! disk, and its leaves are hashed on a few threads. A node that restarts
//! from a snapshot uses the files it finds, if they have the base roots the
//! snapshot carries, and scans only otherwise.

use {
    super::{
        Bank,
        accounts_delta::lt_hash_prefix,
        block_commitment::{
            CommitmentFailure, CommitmentSlot, CommitmentVector, INACTIVE_COMMITMENT_SLOT,
        },
    },
    agave_feature_set as feature_set,
    log::*,
    rayon::prelude::*,
    solana_account::ReadableAccount,
    solana_accounts_db::accounts_db::AccountsDb,
    solana_clock::{Epoch, Slot},
    solana_epoch_schedule::EpochSchedule,
    solana_hash::Hash,
    solana_lattice_hash::lt_hash::LtHash,
    solana_merkle_tree::{
        accounts_delta::{AccountsDelta, Address},
        accounts_state::{
            AccountProof, AccountsStateOpening, BaseTree, DeriveError, OverlayEntry,
            OverlaySubstitution, OverlayTree, account_lookup, accounts_state_root,
            apply_block_changes_batch, apply_overlay_into, new_overlay,
        },
        keyed_tree::{Fork, Sequential},
    },
    solana_pubkey::Pubkey,
    std::{
        collections::HashMap,
        path::Path,
        sync::{
            Arc, Condvar, LazyLock, Mutex, Weak,
            atomic::{AtomicBool, Ordering},
        },
    },
};

mod base_files;

use base_files::{BaseDir, on_base_pool, tree_from_storage};

/// Minimum number of slots of an epoch for its generation to exist. Shorter
/// epochs have no generation, and the blocks that would use it have zero in
/// slot 5. Twice the minimum epoch length of the epoch schedule (32), so that
/// test clusters with 64-slot epochs exercise the feature, and the 32-slot
/// first epoch of a warmup schedule exercises the short-epoch rule.
pub const MIN_SLOTS_PER_EPOCH_FOR_GENERATION: u64 = 64;

/// `switch_slot(E) = first_slot(E) + slots_in_epoch(E) / 2`
pub fn switch_slot(epoch_schedule: &EpochSchedule, epoch: Epoch) -> Slot {
    epoch_schedule.get_first_slot_in_epoch(epoch) + epoch_schedule.get_slots_in_epoch(epoch) / 2
}

/// The generation used by a block at `slot` of `epoch`: `E - 1` before the
/// switch slot of `E`, `E` from it on. `None` for a block of epoch 0 before
/// its switch slot.
pub fn generation_in_use(
    epoch_schedule: &EpochSchedule,
    epoch: Epoch,
    slot: Slot,
) -> Option<Epoch> {
    if slot < switch_slot(epoch_schedule, epoch) {
        epoch.checked_sub(1)
    } else {
        Some(epoch)
    }
}

/// Whether epoch `epoch` is long enough to have a generation.
pub fn epoch_has_generation(epoch_schedule: &EpochSchedule, epoch: Epoch) -> bool {
    epoch_schedule.get_slots_in_epoch(epoch) >= MIN_SLOTS_PER_EPOCH_FOR_GENERATION
}

/// Identity of the base state of a generation: the epoch and the block `P`
/// whose state it is (`None` for a chain that starts at a block without a
/// parent, whose base state is empty).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BaseId {
    pub epoch: Epoch,
    /// `(slot, bank_hash)` of `P`.
    pub parent: Option<(Slot, Hash)>,
}

enum BaseState {
    Pending,
    Ready(Arc<BaseTree>),
    Failed(String),
    /// The leaves rebuilt from a snapshot do not have the base root the
    /// snapshot carries: the snapshot is invalid. Final.
    SnapshotMismatch {
        snapshot: Hash,
        rebuilt: Hash,
    },
}

fn snapshot_mismatch_reason(snapshot: &Hash, rebuilt: &Hash) -> String {
    format!("rebuilt base root {rebuilt} differs from the snapshot's base root {snapshot}")
}

/// The base tree of one generation on one chain, computed by a background
/// job (see the module documentation).
pub struct BaseCell {
    id: BaseId,
    /// Root known without the leaves (from a snapshot).
    known_root: Option<Hash>,
    state: Mutex<BaseState>,
    ready: Condvar,
    /// Cell loaded from a snapshot whose leaves are rebuilt by a scan of the
    /// snapshot bank, started with its first child.
    from_snapshot: bool,
    rebuild_started: AtomicBool,
}

impl std::fmt::Debug for BaseCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BaseCell")
            .field("id", &self.id)
            .field("known_root", &self.known_root)
            .finish_non_exhaustive()
    }
}

impl BaseCell {
    fn new(id: BaseId, known_root: Option<Hash>, from_snapshot: bool) -> Self {
        Self {
            id,
            known_root,
            state: Mutex::new(BaseState::Pending),
            ready: Condvar::new(),
            from_snapshot,
            rebuild_started: AtomicBool::new(false),
        }
    }

    pub fn id(&self) -> BaseId {
        self.id
    }

    /// Stores the result of the job. A cell that is ready keeps its tree. A
    /// tree whose root differs from the known root makes the snapshot that
    /// carried the root invalid, see [`Self::snapshot_mismatch`].
    fn finish(&self, result: Result<BaseTree, String>) {
        let mut state = self.state.lock().unwrap();
        if matches!(
            *state,
            BaseState::Ready(_) | BaseState::SnapshotMismatch { .. }
        ) {
            return;
        }
        *state = match result {
            Ok(tree) => match self.known_root {
                Some(snapshot) if snapshot != tree.root() => {
                    let rebuilt = tree.root();
                    error!(
                        "accounts state base {:?}: invalid snapshot: {}",
                        self.id,
                        snapshot_mismatch_reason(&snapshot, &rebuilt)
                    );
                    BaseState::SnapshotMismatch { snapshot, rebuilt }
                }
                _ => BaseState::Ready(Arc::new(tree)),
            },
            Err(reason) => {
                warn!("accounts state base {:?} failed: {reason}", self.id);
                BaseState::Failed(reason)
            }
        };
        self.ready.notify_all();
    }

    /// The base tree, waiting for the job if it is still running.
    pub fn wait_tree(&self) -> Result<Arc<BaseTree>, String> {
        let mut state = self.state.lock().unwrap();
        loop {
            match &*state {
                BaseState::Ready(tree) => return Ok(tree.clone()),
                BaseState::Failed(reason) => return Err(reason.clone()),
                BaseState::SnapshotMismatch { snapshot, rebuilt } => {
                    return Err(snapshot_mismatch_reason(snapshot, rebuilt));
                }
                BaseState::Pending => {
                    if self.from_snapshot && !self.rebuild_started.load(Ordering::Acquire) {
                        return Err(
                            "the base leaves have not been rebuilt from the snapshot".into()
                        );
                    }
                    state = self.ready.wait(state).unwrap();
                }
            }
        }
    }

    /// `(snapshot, rebuilt)` base roots, if the leaves rebuilt from a
    /// snapshot turned out not to have the root the snapshot carries.
    pub fn snapshot_mismatch(&self) -> Option<(Hash, Hash)> {
        match &*self.state.lock().unwrap() {
            BaseState::SnapshotMismatch { snapshot, rebuilt } => Some((*snapshot, *rebuilt)),
            _ => None,
        }
    }

    /// `base_root`, waiting for the job if the root is not known yet. A root
    /// from a snapshot is used until the rebuilt leaves contradict it.
    pub fn wait_root(&self) -> Result<Hash, String> {
        if let Some((snapshot, rebuilt)) = self.snapshot_mismatch() {
            return Err(snapshot_mismatch_reason(&snapshot, &rebuilt));
        }
        match self.known_root {
            Some(root) => Ok(root),
            None => self.wait_tree().map(|tree| tree.root()),
        }
    }

    /// `base_root` if it is known now, without waiting. `None` once a
    /// snapshot's root has been contradicted.
    pub fn root_if_known(&self) -> Option<Hash> {
        match &*self.state.lock().unwrap() {
            BaseState::Ready(tree) => Some(tree.root()),
            BaseState::SnapshotMismatch { .. } => None,
            BaseState::Pending | BaseState::Failed(_) => self.known_root,
        }
    }

    /// Whether the job has finished (successfully or not).
    pub fn is_finished(&self) -> bool {
        !matches!(*self.state.lock().unwrap(), BaseState::Pending)
    }

    #[cfg(test)]
    pub(crate) fn fail_for_tests(&self, reason: &str) {
        let mut state = self.state.lock().unwrap();
        *state = BaseState::Failed(reason.to_string());
        self.ready.notify_all();
    }
}

/// The base cells of all banks derived from one root bank, shared from parent
/// to child, so that forks with the same `P` share one job.
#[derive(Default)]
pub struct AccountsStateBases {
    cells: Mutex<HashMap<BaseId, Weak<BaseCell>>>,
    /// Where the leaves of the base trees are kept.
    dir: Arc<BaseDir>,
}

impl std::fmt::Debug for AccountsStateBases {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccountsStateBases").finish_non_exhaustive()
    }
}

impl AccountsStateBases {
    /// The bases of a node whose accounts are in `accounts_path`: the leaves
    /// of its base trees are kept in files near it (see [`base_files`]).
    /// The default keeps them in memory.
    pub fn for_accounts_path(accounts_path: &Path) -> Self {
        Self {
            cells: Mutex::default(),
            dir: Arc::new(BaseDir::for_accounts_path(accounts_path)),
        }
    }

    /// [`Self::for_accounts_path`] of the first path of an accounts database.
    pub(super) fn for_accounts(accounts: &solana_accounts_db::accounts::Accounts) -> Self {
        match accounts.accounts_db.paths.first() {
            Some(path) => Self::for_accounts_path(path),
            None => Self::default(),
        }
    }

    /// The cell of `id`, created with `job` running on a background thread
    /// if no bank holds it yet.
    fn request(
        &self,
        id: BaseId,
        job: impl FnOnce() -> Result<BaseTree, String> + Send + 'static,
    ) -> Arc<BaseCell> {
        let mut cells = self.cells.lock().unwrap();
        if let Some(cell) = cells.get(&id).and_then(Weak::upgrade) {
            return cell;
        }
        cells.retain(|_, cell| cell.strong_count() > 0);
        let cell = Arc::new(BaseCell::new(id, None, false));
        cells.insert(id, Arc::downgrade(&cell));
        drop(cells);
        spawn_base_job(vec![cell.clone()], move || job().map(|tree| vec![tree]));
        cell
    }

    /// The cell of `id`, already computed.
    fn ready(&self, id: BaseId, tree: BaseTree) -> Arc<BaseCell> {
        let mut cells = self.cells.lock().unwrap();
        if let Some(cell) = cells.get(&id).and_then(Weak::upgrade) {
            return cell;
        }
        let cell = Arc::new(BaseCell::new(id, None, false));
        cell.finish(Ok(tree));
        cells.insert(id, Arc::downgrade(&cell));
        cell
    }

    /// The base of `id` from the leaves this node has on disk, for a prover
    /// that serves blocks of a generation no bank holds any more: the cell
    /// some bank still holds, or a new one whose tree is read from the file
    /// on a background thread. `None` if there is neither.
    ///
    /// Nothing here says the file has the leaves of that base: the caller
    /// must check the root it gets against a commitment it trusts. The new
    /// cell is not given to any bank.
    pub fn open_from_disk(&self, id: BaseId) -> Option<Arc<BaseCell>> {
        if let Some(cell) = self.cells.lock().unwrap().get(&id).and_then(Weak::upgrade) {
            return Some(cell);
        }
        let storage = self.dir.open(&id)?;
        let cell = Arc::new(BaseCell::new(id, None, false));
        spawn_base_job(vec![cell.clone()], move || {
            tree_from_storage(storage)
                .map(|tree| vec![tree])
                .map_err(|err| format!("{err:?}"))
        });
        Some(cell)
    }

    /// A cell loaded from a snapshot, rebuilt later by a scan of the snapshot
    /// bank.
    fn snapshot_cell(&self, id: BaseId, known_root: Option<Hash>) -> Arc<BaseCell> {
        let cell = Arc::new(BaseCell::new(id, known_root, true));
        self.cells.lock().unwrap().insert(id, Arc::downgrade(&cell));
        cell
    }
}

/// Runs `job` on a new thread and stores its trees in `cells`, in order.
fn spawn_base_job(
    cells: Vec<Arc<BaseCell>>,
    job: impl FnOnce() -> Result<Vec<BaseTree>, String> + Send + 'static,
) {
    let thread_cells = cells.clone();
    let spawned = std::thread::Builder::new()
        .name("solAcctStBase".to_string())
        .spawn(move || {
            let start = std::time::Instant::now();
            let result = job();
            let us = start.elapsed().as_micros();
            match result {
                Ok(trees) => {
                    for (cell, tree) in thread_cells.iter().zip(trees) {
                        info!(
                            "accounts state base {:?}: {} leaves, root {}, computed in {us} us",
                            cell.id,
                            tree.len(),
                            tree.root()
                        );
                        cell.finish(Ok(tree));
                    }
                }
                Err(reason) => {
                    for cell in &thread_cells {
                        cell.finish(Err(reason.clone()));
                    }
                }
            }
        });
    if let Err(err) = spawned {
        for cell in &cells {
            cell.finish(Err(format!("could not start the base job: {err}")));
        }
    }
}

/// Number of accounts hashed together by a scan.
const SCAN_BATCH: usize = 4096;

/// Scans the account state of `bank`, checks it against the bank's accounts
/// lattice hash, and derives one base tree per `(base, overlay)`: the base
/// tree of the generation of each overlay, which must be the overlay of that
/// generation as of `bank` (an empty overlay gives the state of `bank`
/// itself).
///
/// The accounts are read and hashed on the base threads, each taking one
/// bin of the accounts index at a time. The leaves go through a sorter that
/// spills to disk, so memory does not grow with the number of accounts.
fn scan_base_trees(
    bank: &Bank,
    dir: &BaseDir,
    bases: &[(BaseId, &OverlayTree)],
) -> Result<Vec<BaseTree>, String> {
    // What one thread gathers from one bin of the accounts index.
    struct Part {
        leaves: Vec<(Address, Hash)>,
        lt_hash: LtHash,
    }
    let sorter = Mutex::new((dir.sorter(), None::<String>));
    let lt_hash = Mutex::new(LtHash::identity());
    let give_leaves = |leaves: &mut Vec<(Address, Hash)>| {
        let mut guard = sorter.lock().unwrap();
        let (sorter, failure) = &mut *guard;
        for (address, account_hash) in leaves.drain(..) {
            if failure.is_none()
                && let Err(err) = sorter.push(&address, &account_hash)
            {
                *failure = Some(format!("cannot sort the scanned accounts: {err}"));
            }
        }
    };
    on_base_pool(|| {
        bank.rc.accounts.scan_all_by_bin(
            &bank.ancestors,
            bank.bank_id(),
            || Part {
                leaves: Vec::new(),
                lt_hash: LtHash::identity(),
            },
            |part, address, account, _slot| {
                if account.lamports() == 0 {
                    return;
                }
                let account_lt_hash = AccountsDb::lt_hash_account(&account, address);
                part.lt_hash.mix_in(&account_lt_hash.0);
                part.leaves
                    .push((address.to_bytes(), lt_hash_prefix(&account_lt_hash.0)));
                if part.leaves.len() >= SCAN_BATCH {
                    give_leaves(&mut part.leaves);
                }
            },
            |mut part| {
                give_leaves(&mut part.leaves);
                lt_hash.lock().unwrap().mix_in(&part.lt_hash);
            },
        )
    })
    .map_err(|err| format!("scan of bank {} failed: {err}", bank.slot()))?;
    let (sorter, failure) = sorter.into_inner().unwrap();
    if let Some(failure) = failure {
        return Err(failure);
    }
    if lt_hash.into_inner().unwrap() != bank.accounts_lt_hash.lock().unwrap().0 {
        return Err(format!(
            "the scanned state of bank {} does not match its accounts lattice hash",
            bank.slot()
        ));
    }
    // One pass over the sorted state gives the leaves of every base.
    let mut outputs: Vec<_> = bases
        .iter()
        .map(|(id, overlay)| (OverlaySubstitution::new(overlay), dir.writer(id)))
        .collect();
    let derive_error = |err| match err {
        DeriveError::Mismatch(mismatch) => {
            format!("overlay does not match bank {}: {mismatch:?}", bank.slot())
        }
        DeriveError::Sink(err) => format!("cannot write the base leaves: {err}"),
    };
    sorter.into_sorted(|leaf| {
        outputs.iter_mut().try_for_each(|(substitution, writer)| {
            substitution
                .push(leaf, |derived| writer.push(derived))
                .map_err(derive_error)
        })
    })?;
    outputs
        .into_iter()
        .map(|(substitution, mut writer)| {
            substitution
                .finish(|derived| writer.push(derived))
                .map_err(derive_error)?;
            let storage = writer
                .finish()
                .map_err(|err| format!("cannot write the base leaves: {err}"))?;
            tree_from_storage(storage).map_err(|err| format!("{err:?}"))
        })
        .collect()
}

/// The base tree of a cell loaded from a snapshot, from the leaves this node
/// wrote before it restarted: used if they have the base root the snapshot
/// carries. A file with another root (damaged, or of another ledger) is
/// removed.
fn reuse_base_file(dir: &BaseDir, cell: &BaseCell) -> Option<BaseTree> {
    let known_root = cell.known_root?;
    let storage = dir.open(&cell.id)?;
    match tree_from_storage(storage) {
        Ok(tree) if tree.root() == known_root => {
            info!(
                "accounts state base {:?}: using the {} leaves on disk",
                cell.id,
                tree.len()
            );
            Some(tree)
        }
        result => {
            warn!(
                "accounts state base {:?}: the leaves on disk are not those of the snapshot's \
                 base root {known_root} ({:?}); scanning instead",
                cell.id,
                result.map(|tree| tree.root())
            );
            dir.remove(&cell.id);
            None
        }
    }
}

/// A block with fewer changed accounts than this updates its overlays on
/// the thread that freezes it.
const MIN_CHANGES_FOR_OVERLAY_POOL: usize = 128;

/// The smallest part of a block's changes that is given to another thread.
const MIN_CHANGES_PER_OVERLAY_TASK: usize = 32;

/// The threads that update the overlays of a block when it is frozen. Not
/// the threads of the base trees: those are busy for a minute or more after
/// each epoch starts, and a block must not wait for them.
static OVERLAY_POOL: LazyLock<rayon::ThreadPool> = LazyLock::new(|| {
    let threads = std::thread::available_parallelism().map_or(1, |cores| cores.get() / 4);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads.clamp(1, 8))
        .thread_name(|index| format!("solAcctStOvl{index:02}"))
        .build()
        .expect("a thread pool for overlays")
});

/// Divides a batch with `rayon::join`, on the pool it is called from.
struct RayonFork;

impl Fork for RayonFork {
    fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        rayon::join(a, b)
    }
}

/// Applies the changed accounts of a block (sorted by address) to the
/// overlay of each generation, and computes the hashes of the overlay nodes
/// that change. A block with many changes is spread over the overlay
/// threads, by parts of the address space and by generation.
fn update_overlays(generations: &mut [Generation], changes: &[(Address, (Hash, Hash))]) {
    if changes.len() < MIN_CHANGES_FOR_OVERLAY_POOL {
        for generation in generations {
            apply_block_changes_batch(&mut generation.overlay, changes, &Sequential, usize::MAX);
        }
        return;
    }
    OVERLAY_POOL.install(|| {
        generations.par_iter_mut().for_each(|generation| {
            apply_block_changes_batch(
                &mut generation.overlay,
                changes,
                &RayonFork,
                MIN_CHANGES_PER_OVERLAY_TASK,
            );
        });
    });
}

/// One generation held by a bank: its epoch, its overlay as of the bank, and
/// its base.
#[derive(Clone, Debug)]
pub struct Generation {
    epoch: Epoch,
    overlay: OverlayTree,
    base: Arc<BaseCell>,
}

impl Generation {
    /// A generation from its parts, for a prover that rebuilds the state of
    /// an earlier block from a later one's. Nothing a bank computes uses it.
    pub fn from_parts(overlay: OverlayTree, base: Arc<BaseCell>) -> Self {
        Self {
            epoch: base.id.epoch,
            overlay,
            base,
        }
    }

    pub fn epoch(&self) -> Epoch {
        self.epoch
    }

    /// `O(G, B)`.
    pub fn overlay(&self) -> &OverlayTree {
        &self.overlay
    }

    pub fn base(&self) -> &Arc<BaseCell> {
        &self.base
    }
}

/// The generations held by one bank, in ascending epoch order (at most two).
#[derive(Clone, Debug, Default)]
pub struct AccountsState {
    generations: Vec<Generation>,
}

impl AccountsState {
    /// A state from generations in ascending epoch order (see
    /// [`Generation::from_parts`]).
    pub fn from_generations(generations: Vec<Generation>) -> Self {
        Self { generations }
    }

    pub fn generations(&self) -> &[Generation] {
        &self.generations
    }

    /// A copy of this state whose base trees and roots are not computed
    /// yet, as while the background job of a new generation is running.
    #[cfg(feature = "dev-context-only-utils")]
    pub fn with_pending_bases_for_tests(&self) -> Self {
        Self {
            generations: self
                .generations
                .iter()
                .map(|generation| Generation {
                    epoch: generation.epoch,
                    overlay: generation.overlay.clone(),
                    base: Arc::new(BaseCell::new(generation.base.id, None, false)),
                })
                .collect(),
        }
    }

    pub fn generation(&self, epoch: Epoch) -> Option<&Generation> {
        self.generations.iter().find(|g| g.epoch == epoch)
    }

    /// Proof of the state of `address` in generation `generation_in_use` of
    /// this state, opened against `vector`, the commitment vector of the
    /// bank that holds this state (see [`Bank::accounts_state_proof`]).
    ///
    /// With [`BaseWait::NoWait`], fails with
    /// [`AccountsStateProofError::BaseNotReady`] instead of waiting for a base
    /// tree (or root) that the background job has not finished.
    pub fn account_proof(
        &self,
        generation_in_use: Option<Epoch>,
        vector: &CommitmentVector,
        address: &Pubkey,
        base_wait: BaseWait,
    ) -> Result<AccountProof, AccountsStateProofError> {
        let generation = generation_in_use
            .and_then(|epoch| self.generation(epoch))
            .ok_or(AccountsStateProofError::NoGeneration)?;
        if base_wait == BaseWait::NoWait
            && generation.base.root_if_known().is_none()
            && !generation.base.is_finished()
        {
            return Err(AccountsStateProofError::BaseNotReady);
        }
        let base_root = generation
            .base
            .wait_root()
            .map_err(AccountsStateProofError::BaseUnavailable)?;
        let opening = AccountsStateOpening {
            generation: generation.epoch,
            base_root,
            overlay_root: generation.overlay.root(),
            slot_proof: vector
                .inclusion_proof(CommitmentSlot::AccountsStateRoot.index())
                .expect("slot 5 is in range"),
        };
        if opening.accounts_state_root() != vector.get(CommitmentSlot::AccountsStateRoot) {
            return Err(AccountsStateProofError::BaseUnavailable(
                "the base root differs from the one committed in slot 5".into(),
            ));
        }
        let address = address.to_bytes();
        let lookup = match account_lookup(&generation.overlay, None, &address) {
            Some(lookup) => lookup,
            None => {
                if base_wait == BaseWait::NoWait && !generation.base.is_finished() {
                    return Err(AccountsStateProofError::BaseNotReady);
                }
                let base = generation
                    .base
                    .wait_tree()
                    .map_err(AccountsStateProofError::BaseUnavailable)?;
                if base.root() != base_root {
                    return Err(AccountsStateProofError::BaseUnavailable(
                        "the base leaves do not match the base root".into(),
                    ));
                }
                account_lookup(&generation.overlay, Some(&base), &address)
                    .expect("the base tree is given")
            }
        };
        Ok(AccountProof { opening, lookup })
    }

    /// The form persisted in snapshots: overlays, base identities and the base
    /// roots known now.
    pub fn to_snapshot(&self) -> SnapshotAccountsState {
        SnapshotAccountsState {
            generations: self
                .generations
                .iter()
                .map(|g| SnapshotGeneration {
                    epoch: g.epoch,
                    base_parent: g.base.id.parent,
                    base_root: g.base.root_if_known(),
                    overlay: g.overlay.clone(),
                })
                .collect(),
        }
    }

    /// Rebuilds the state of a snapshot bank. The base leaves are rebuilt
    /// later, see [`Bank::start_accounts_state_rebuild`].
    fn from_snapshot(snapshot: SnapshotAccountsState, bases: &AccountsStateBases) -> Self {
        Self {
            generations: snapshot
                .generations
                .into_iter()
                .map(|g| Generation {
                    epoch: g.epoch,
                    base: bases.snapshot_cell(
                        BaseId {
                            epoch: g.epoch,
                            parent: g.base_parent,
                        },
                        g.base_root,
                    ),
                    overlay: g.overlay,
                })
                .collect(),
        }
    }
}

/// One generation as persisted in a snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotGeneration {
    pub epoch: Epoch,
    /// `(slot, bank_hash)` of the block whose state is the base state.
    pub base_parent: Option<(Slot, Hash)>,
    /// `base_root`, if it was known when the snapshot was taken.
    pub base_root: Option<Hash>,
    pub overlay: OverlayTree,
}

/// The accounts state of a snapshot bank.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SnapshotAccountsState {
    pub generations: Vec<SnapshotGeneration>,
}

/// The accounts state of a frozen bank, or the commitment failure that
/// prevents computing it.
pub(super) type AccountsStateResult = Result<Arc<AccountsState>, CommitmentFailure>;

/// Why [`Bank::accounts_state_proof`] cannot build a proof.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AccountsStateProofError {
    #[error("the accounts state root is not active for the bank")]
    Inactive,
    #[error("the bank is not frozen")]
    NotFrozen,
    #[error("the bank has no commitment vector (commitment failure, or snapshot bank)")]
    NoCommitmentVector,
    #[error("the bank has no accounts state root (slot 5 is zero)")]
    NoGeneration,
    #[error("the base tree is not available: {0}")]
    BaseUnavailable(String),
    #[error("the base tree is still being computed")]
    BaseNotReady,
}

/// Whether [`AccountsState::account_proof`] waits for a base tree that is
/// still being computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseWait {
    Wait,
    NoWait,
}

impl Bank {
    /// Whether this bank computes `accounts_state_root`: the Accounts State
    /// Root feature and both features it depends on must be active.
    pub fn is_accounts_state_root_active(&self) -> bool {
        self.feature_set
            .is_active(&feature_set::accounts_state_root::id())
            && self.is_accounts_delta_root_active()
    }

    /// The accounts state of this bank (its generations and overlays), set
    /// when the bank is frozen with [`Self::is_accounts_state_root_active`],
    /// or loaded from a snapshot. `None` otherwise, and for a bank with a
    /// commitment failure in its accounts state.
    pub fn accounts_state(&self) -> Option<Arc<AccountsState>> {
        self.accounts_state
            .read()
            .unwrap()
            .as_ref()
            .and_then(|state| state.as_ref().ok())
            .cloned()
    }

    /// The generation this bank uses for slot 5, if it exists.
    pub fn accounts_state_generation_in_use(&self) -> Option<Epoch> {
        generation_in_use(&self.epoch_schedule, self.epoch(), self.slot())
    }

    /// Updates the overlays of this bank from its changed accounts (the
    /// accounts delta and the account hash of each before the block), or
    /// records why they cannot be updated. Called once, from `freeze()`, by
    /// `finish_accounts_delta()`.
    pub(super) fn finish_accounts_state(
        &self,
        changes: Result<(&AccountsDelta, &[Hash]), CommitmentFailure>,
    ) {
        if !self.is_accounts_state_root_active() {
            return;
        }
        let result = changes
            .and_then(|(delta, prior)| self.compute_accounts_state(delta, prior))
            .map(Arc::new);
        *self.accounts_state.write().unwrap() = Some(result);
    }

    fn compute_accounts_state(
        &self,
        delta: &AccountsDelta,
        prior: &[Hash],
    ) -> Result<AccountsState, CommitmentFailure> {
        let parent = self.parent();
        let epoch = self.epoch();
        let parent_generations = match parent.as_deref() {
            None => Vec::new(),
            Some(parent) if !parent.is_accounts_state_root_active() => Vec::new(),
            Some(parent) => match parent.accounts_state.read().unwrap().as_ref() {
                Some(Ok(state)) => state.generations.clone(),
                Some(Err(_)) => {
                    return Err(CommitmentFailure::ParentCommitmentFailure {
                        parent_slot: parent.slot(),
                    });
                }
                None => {
                    return Err(CommitmentFailure::AccountsStateNotTracked {
                        slot: parent.slot(),
                    });
                }
            },
        };
        let switch = switch_slot(&self.epoch_schedule, epoch);
        let mut generations: Vec<_> = parent_generations
            .iter()
            .filter(|g| g.epoch == epoch || (g.epoch + 1 == epoch && self.slot() < switch))
            .cloned()
            .collect();
        if !generations.iter().any(|g| g.epoch == epoch)
            && epoch_has_generation(&self.epoch_schedule, epoch)
        {
            match parent.as_ref() {
                // The generation is tracked from the first block of the epoch,
                // which did not track it (the feature was activated within the
                // epoch, which only tests do).
                Some(parent) if parent.epoch() == epoch => {
                    return Err(CommitmentFailure::AccountsStateNotTracked {
                        slot: parent.slot(),
                    });
                }
                // This bank is the first block of its epoch on its chain.
                parent => generations.push(Generation {
                    epoch,
                    overlay: new_overlay(),
                    base: self.request_base(epoch, parent, &parent_generations),
                }),
            }
        }
        debug_assert_eq!(delta.len(), prior.len());
        // Sorted by address, as the delta is.
        let changes: Vec<(Address, (Hash, Hash))> = delta
            .entries()
            .iter()
            .zip(prior)
            .map(|(entry, before)| (entry.address, (*before, entry.account_hash)))
            .collect();
        update_overlays(&mut generations, &changes);
        Ok(AccountsState { generations })
    }

    /// The base cell of generation `epoch` for a chain whose last block before
    /// the epoch is `parent`, with its job started if it is new.
    fn request_base(
        &self,
        epoch: Epoch,
        parent: Option<&Arc<Bank>>,
        parent_generations: &[Generation],
    ) -> Arc<BaseCell> {
        let Some(parent) = parent else {
            // A block without a parent: the state before it is empty.
            return self.accounts_state_bases.ready(
                BaseId {
                    epoch,
                    parent: None,
                },
                BaseTree::empty(),
            );
        };
        let id = BaseId {
            epoch,
            parent: Some((parent.slot(), parent.hash())),
        };
        let dir = self.accounts_state_bases.dir.clone();
        // By merge, from the latest generation of `P`, whose overlay as of `P`
        // is frozen.
        if let Some(source) = parent_generations.last() {
            let previous = source.base.clone();
            let overlay = source.overlay.clone();
            return self.accounts_state_bases.request(id, move || {
                let previous = previous
                    .wait_tree()
                    .map_err(|reason| format!("previous base unavailable: {reason}"))?;
                // Nothing older than the previous base is derived from again.
                dir.retain_epochs(|kept| kept == epoch || kept.checked_add(1) == Some(epoch));
                let mut writer = dir.writer(&id);
                apply_overlay_into(&previous, &overlay, |leaf| writer.push(leaf)).map_err(
                    |err| match err {
                        DeriveError::Mismatch(mismatch) => {
                            format!("overlay does not match the previous base: {mismatch:?}")
                        }
                        DeriveError::Sink(err) => format!("cannot write the base leaves: {err}"),
                    },
                )?;
                let storage = writer
                    .finish()
                    .map_err(|err| format!("cannot write the base leaves: {err}"))?;
                tree_from_storage(storage).map_err(|err| format!("{err:?}"))
            });
        }
        // By scan of `P` (activation, or no generation in the previous epoch).
        let parent = parent.clone();
        self.accounts_state_bases.request(id, move || {
            dir.retain_epochs(|kept| kept == epoch || kept.checked_add(1) == Some(epoch));
            scan_base_trees(&parent, &dir, &[(id, &new_overlay())]).map(|mut trees| trees.remove(0))
        })
    }

    /// Starts the rebuild of the base leaves of a bank loaded from a snapshot,
    /// by one scan of this bank. Called when its first child is created; does
    /// nothing for other banks or once started.
    pub(super) fn start_accounts_state_rebuild(self: &Arc<Self>) {
        let Some(state) = self.accounts_state() else {
            return;
        };
        let pending: Vec<_> = state
            .generations
            .iter()
            .filter(|g| {
                g.base.from_snapshot && !g.base.rebuild_started.swap(true, Ordering::AcqRel)
            })
            .cloned()
            .collect();
        if pending.is_empty() {
            return;
        }
        let bank = self.clone();
        let dir = self.accounts_state_bases.dir.clone();
        let cells = pending.iter().map(|g| g.base.clone()).collect();
        spawn_base_job(cells, move || {
            // The leaves this node wrote before it restarted, if they have
            // the root the snapshot carries.
            let mut trees: Vec<Option<BaseTree>> = pending
                .iter()
                .map(|g| reuse_base_file(&dir, &g.base))
                .collect();
            let missing: Vec<_> = pending
                .iter()
                .zip(&trees)
                .filter(|(_, tree)| tree.is_none())
                .map(|(g, _)| (g.base.id, &g.overlay))
                .collect();
            if !missing.is_empty() {
                let mut scanned = scan_base_trees(&bank, &dir, &missing)?.into_iter();
                for tree in trees.iter_mut().filter(|tree| tree.is_none()) {
                    *tree = scanned.next();
                }
            }
            trees
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| "the scan did not give every base".to_string())
        });
    }

    /// The bases of this node's banks (see [`AccountsStateBases::from_disk`]).
    pub fn accounts_state_bases(&self) -> &Arc<AccountsStateBases> {
        &self.accounts_state_bases
    }

    /// Commitment slot 5 value, see
    /// [`Self::commitment_slot_accounts_state_root`].
    pub(super) fn accounts_state_root_for_commitment(&self) -> Result<Hash, CommitmentFailure> {
        if !self.is_accounts_state_root_active() {
            return Ok(INACTIVE_COMMITMENT_SLOT);
        }
        let state = match self.accounts_state.read().unwrap().clone() {
            Some(Ok(state)) => state,
            Some(Err(failure)) => return Err(failure),
            // `finish_accounts_state()` sets it for every bank that computes
            // the root, before the commitment vector is computed.
            None => {
                return Err(CommitmentFailure::AccountsStateNotTracked { slot: self.slot() });
            }
        };
        let Some(generation) = self
            .accounts_state_generation_in_use()
            .and_then(|epoch| state.generation(epoch))
        else {
            return Ok(INACTIVE_COMMITMENT_SLOT);
        };
        let base_root = self.resolve_base_root(generation)?;
        Ok(accounts_state_root(
            generation.epoch,
            &base_root,
            &generation.overlay.root(),
        ))
    }

    /// `base_root` of `generation`, waiting for its job; if the job failed,
    /// rebuilt synchronously by a scan of the parent.
    fn resolve_base_root(&self, generation: &Generation) -> Result<Hash, CommitmentFailure> {
        let snapshot_mismatch = || {
            generation
                .base
                .snapshot_mismatch()
                .map(
                    |(snapshot, rebuilt)| CommitmentFailure::AccountsStateSnapshotBaseMismatch {
                        generation: generation.epoch,
                        snapshot,
                        rebuilt,
                    },
                )
        };
        let reason = match generation.base.wait_root() {
            Ok(root) => return Ok(root),
            Err(reason) => reason,
        };
        // An invalid snapshot is not repaired by another scan.
        if let Some(failure) = snapshot_mismatch() {
            return Err(failure);
        }
        warn!(
            "bank {} rebuilds the base of generation {} synchronously: {reason}",
            self.slot(),
            generation.epoch
        );
        // TODO(simd-accounts-state-root): this blocks the freeze for a full
        // scan; it only runs when the background job failed.
        self.rebuild_base_from_parent(generation)
            .map_err(|rebuild| {
                snapshot_mismatch().unwrap_or_else(|| {
                    CommitmentFailure::AccountsStateBaseUnavailable {
                        generation: generation.epoch,
                        reason: format!("{reason}; synchronous rebuild: {rebuild}"),
                    }
                })
            })
    }

    /// Rebuilds the base of `generation` from the state of the parent `X` of
    /// this bank: `X` is either in the generation's epoch, and then holds its
    /// overlay as of `X`, or it is `P` itself.
    fn rebuild_base_from_parent(&self, generation: &Generation) -> Result<Hash, String> {
        let parent = self.parent().ok_or("the bank has no parent")?;
        let overlay = if parent.epoch() == generation.epoch {
            parent
                .accounts_state()
                .and_then(|state| {
                    state
                        .generation(generation.epoch)
                        .map(|g| g.overlay.clone())
                })
                .ok_or("the parent does not hold the generation")?
        } else if generation.base.id.parent == Some((parent.slot(), parent.hash())) {
            new_overlay()
        } else {
            return Err("the parent is neither in the generation nor its base block".into());
        };
        let tree = scan_base_trees(
            &parent,
            &self.accounts_state_bases.dir,
            &[(generation.base.id, &overlay)],
        )?
        .remove(0);
        let root = tree.root();
        // Replace the failure, so that the other banks of the chain use the
        // result.
        let mut state = generation.base.state.lock().unwrap();
        let result = match generation.base.known_root {
            Some(snapshot) if snapshot != root => {
                *state = BaseState::SnapshotMismatch {
                    snapshot,
                    rebuilt: root,
                };
                Err(snapshot_mismatch_reason(&snapshot, &root))
            }
            _ => {
                *state = BaseState::Ready(Arc::new(tree));
                Ok(root)
            }
        };
        generation.base.ready.notify_all();
        result
    }

    /// Proof of the state of `address` at the end of this frozen bank, from
    /// its `commitment_root`: its current state (account hash), or that it has
    /// no account. Verify with [`AccountProof::verify_current_state`] or
    /// [`AccountProof::verify_nonexistence`].
    ///
    /// Waits for the base tree if the proof needs it. A base proof is built
    /// from the inner nodes the base tree keeps (a few thousand hashes).
    pub fn accounts_state_proof(
        &self,
        address: &Pubkey,
    ) -> Result<AccountProof, AccountsStateProofError> {
        if !self.is_accounts_state_root_active() {
            return Err(AccountsStateProofError::Inactive);
        }
        if !self.is_frozen() {
            return Err(AccountsStateProofError::NotFrozen);
        }
        let vector = self
            .commitment_vector
            .read()
            .unwrap()
            .ok_or(AccountsStateProofError::NoCommitmentVector)?;
        let state = self
            .accounts_state()
            .ok_or(AccountsStateProofError::NoCommitmentVector)?;
        state.account_proof(
            self.accounts_state_generation_in_use(),
            &vector,
            address,
            BaseWait::Wait,
        )
    }

    /// The overlay entry of `address` in generation `epoch` of this bank.
    pub fn accounts_state_overlay_entry(
        &self,
        epoch: Epoch,
        address: &Pubkey,
    ) -> Option<OverlayEntry> {
        let state = self.accounts_state()?;
        let address: Address = address.to_bytes();
        state
            .generation(epoch)?
            .overlay
            .get(&address)
            .map(|value| OverlayEntry::from_value(address, value))
    }

    /// The state of a snapshot bank's accounts state, for snapshots. `None`
    /// if the feature is inactive for the bank or it has no accounts state.
    pub(crate) fn accounts_state_for_snapshot(&self) -> Option<SnapshotAccountsState> {
        self.accounts_state()
            .filter(|_| self.is_accounts_state_root_active())
            .map(|state| state.to_snapshot())
    }

    /// Accounts state of a bank built from snapshot fields.
    pub(super) fn accounts_state_from_snapshot(
        snapshot: Option<SnapshotAccountsState>,
        bases: &AccountsStateBases,
    ) -> Option<AccountsStateResult> {
        snapshot.map(|snapshot| Ok(Arc::new(AccountsState::from_snapshot(snapshot, bases))))
    }
}

#[cfg(test)]
mod tests;
