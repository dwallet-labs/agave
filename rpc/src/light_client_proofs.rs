//! Proof store for the light-client RPC methods (`getFinalityProof` and
//! the others, see [`crate::rpc::rpc_light_client`]).
//!
//! # What is kept, and why
//!
//! A prover needs, for every finalized block it serves:
//!
//! - its block id, bank hash, commitment root and commitment vector (the
//!   bank keeps them while it is alive; banks older than the root are
//!   dropped);
//! - the leaves of its block history: the bank keeps only the frontier;
//! - its receipts with their instruction trace records and events: the
//!   bank keeps only the receipt leaf hashes unless
//!   `RuntimeConfig::retain_light_client_proof_data` is set, which the
//!   validator sets together with this store;
//! - its accounts delta and accounts state (both shared, `Arc`);
//! - the certificates that finalized it: the consensus pool prunes them at
//!   the root.
//!
//! The footer proof (final leaf contents and path to the block id) is read
//! from blockstore on request, where the double Merkle tree is kept for
//! repair; nothing is copied.
//!
//! The store is fed by the validator: [`LightClientProofStore::on_rooted_banks`]
//! from root handling (every newly rooted bank, before older banks are
//! dropped) and [`LightClientProofStore::on_certificate`] from the
//! consensus pool (every completed certificate). A node without the store
//! (the default) does none of this, and nothing the store does changes what
//! a bank computes.
//!
//! # Finality
//!
//! A block is final for this store when it was rooted: with Alpenglow a
//! block is rooted once it, or a descendant, is finalized. A rooted block
//! whose own certificates the node does not have (finalized through a
//! descendant, or certificates received before the node started) is served
//! with the certificates of the nearest later retained block that has them
//! and a history proof from that block.
//!
//! # Retention
//!
//! In memory, and on disk when [`LightClientProofConfig::persist_dir`] is
//! set (see [`persist`]):
//!
//! - the last [`LightClientProofConfig::retained_blocks`] rooted blocks with
//!   their data (receipts, accounts delta, accounts state, commitment
//!   vector);
//! - certificates of slots from the oldest retained block on;
//! - the last [`LightClientProofConfig::retained_history`] history leaves
//!   (32-byte hashes plus slot, block id and commitment root): history
//!   proofs reach that far back, never before the first block rooted after
//!   the node started (the leaves before it are folded into the frontier
//!   the node starts from);
//! - the signer sets of the last [`RETAINED_SIGNER_SET_EPOCHS`] epochs.
//!
//! # Restart
//!
//! With a directory, every retained block is written when it is rooted
//! (commitment vector, history frontier, receipts with their leaves,
//! accounts delta, epoch boundary), certificates are written at the next
//! root after they complete, and signer sets when first seen. A store
//! opened on the directory takes the blocks back in order, as if they were
//! rooted again, dropping any that blockstore does not have as a root. It
//! then serves the same proofs for them as before.
//!
//! The thread that roots writes nothing: it queues the files for a thread
//! of the store, which writes and syncs them in the same order
//! ([`persist::DiskWriter`]). What that thread had written and synced
//! survives a power loss; it is normally a root or two behind. If the disk
//! falls thousands of files behind, new block files are not queued until
//! it has caught up: those blocks are served from memory and are not there
//! after a restart.
//!
//! - **History.** The kept history leaves are written too, so history
//!   proofs reach as far back after a restart as before it. What is read
//!   back is checked against the history frontier of a kept block before
//!   it is used. If the node resumes after a gap (it starts from a snapshot
//!   later than its last kept block, or block files were not written), the
//!   leaves of the gap are taken from blockstore: each block's id and the
//!   commitment root in its footer, checked against the history of the
//!   first new block. If blockstore does not have the blocks of the gap
//!   (the node started from a snapshot made elsewhere), the history
//!   restarts at the first new block: older blocks are then served with
//!   their own certificates only.
//! - **Accounts state.** The trees are not written: they are shared
//!   between blocks and with the banks. Each block file has instead, for
//!   every account the block changed, what the overlay held for it before
//!   the block ([`StateRecord`]). After a restart the store takes the state
//!   of the first block it has a bank for and steps back from it, block by
//!   block, undoing each block's changes. Every state rebuilt this way is
//!   checked against slot 5 of its block. A block is not reached if the
//!   chain of kept blocks to it is broken, or if its generation ended
//!   before the first bank after the restart; it answers "not kept" as
//!   before. A generation ends once per epoch, at the switch slot, so this
//!   concerns a restart within [`LightClientProofConfig::retained_blocks`]
//!   blocks after a switch slot, and then only the kept blocks before it.
//!
//! A durable store would also cache the roots of complete history subtrees,
//! so that a history proof does not rehash every kept leaf.
//!
//! # Cost
//!
//! Ingesting a rooted block takes the store's write lock briefly: what is
//! read from the bank is read before, and the new history leaf is checked
//! against the block's history frontier in `O(log n)`. Nothing is written
//! to disk by the thread that roots. Certificates have
//! their own lock. Queries copy what they need under the read lock and
//! build proofs after releasing it: a history proof hashes every kept leaf
//! after the base (`O(retained_history)`), an account state proof from the
//! base tree hashes one chunk of base leaves and the inner nodes the tree
//! keeps (a few thousand hashes), as `Bank::accounts_state_proof` does.

use {
    agave_votor_messages::certificate::{
        Certificate as VotorCertificate, CertificateType as VotorCertificateType,
    },
    solana_clock::{Epoch, Slot},
    solana_hash::Hash,
    solana_ledger::blockstore::Blockstore,
    solana_light_client::{
        AccountDeltaProof, AccountStateProof, AccountUnchangedProof, Certificate, CertificateKind,
        CommitmentSlot, EpochBoundaryProof, EventProof, FinalityProof, FooterProof,
        HISTORY_LEAF_LEN, HistoryProof, InstructionProof, MERKLE_PROOF_ENTRY_LEN, SignerSet,
        SlotOpening, TransactionProof, history_leaf,
        rpc::{
            AccountDelta, AccountDeltaResponse, AccountStateResponse, BlockHistoryResponse,
            EpochBoundaryResponse, Finality, FinalityResponse, FinalizedDescendant, OpenedAt,
            TransactionResponse, ValidatorSetResponse,
        },
        solana_merkle_tree::{
            accounts_delta::AccountsDelta,
            accounts_state::{accounts_state_root, overlay_value, split_overlay_value},
            commitment_tree::{
                AppendOnlyFrontier, InclusionProof, TreeId, hash_leaf, hash_node,
                make_inclusion_proof_from_leaf_hashes,
            },
            epoch_boundary::EpochBoundary,
            receipt_tree::{event_inclusion_proof, instruction_record_inclusion_proof},
            validator_set::Signer,
        },
    },
    solana_pubkey::Pubkey,
    solana_rpc_client_api::{custom_error::LightClientProofError, light_client::RpcProofSelection},
    solana_runtime::bank::{
        Bank,
        accounts_state::{AccountsState, AccountsStateProofError, BaseId, BaseWait, Generation},
        block_commitment::CommitmentVector,
        transaction_receipts::RetainedReceipt,
    },
    solana_signature::Signature,
    std::{
        collections::{BTreeMap, BTreeSet, HashMap},
        path::PathBuf,
        sync::{
            Arc, Mutex, RwLock,
            atomic::{AtomicBool, AtomicU64, Ordering},
        },
    },
};

mod persist;

use persist::{BlockRecord, DiskWriter, Job, Persistence};

/// Default number of rooted blocks whose proof data is kept.
pub const DEFAULT_RETAINED_BLOCKS: usize = 1024;
/// Default number of history leaves kept.
pub const DEFAULT_RETAINED_HISTORY: usize = 1 << 18;
/// Number of epochs whose signer sets are kept.
pub const RETAINED_SIGNER_SET_EPOCHS: usize = 4;

type ProofResult<T> = Result<T, LightClientProofError>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LightClientProofConfig {
    pub retained_blocks: usize,
    pub retained_history: usize,
    /// Directory to keep the proof data in across restarts. `None`: in
    /// memory only.
    pub persist_dir: Option<PathBuf>,
}

impl Default for LightClientProofConfig {
    fn default() -> Self {
        Self {
            retained_blocks: DEFAULT_RETAINED_BLOCKS,
            retained_history: DEFAULT_RETAINED_HISTORY,
            persist_dir: None,
        }
    }
}

/// Which light-client features were active for a block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ActiveFeatures {
    block_commitment_root: bool,
    transaction_receipt_tree: bool,
    accounts_delta_root: bool,
    epoch_validator_set_root: bool,
    epoch_boundary_commitment: bool,
    accounts_state_root: bool,
}

impl ActiveFeatures {
    fn of(bank: &Bank) -> Self {
        Self {
            block_commitment_root: bank.is_block_commitment_root_active(),
            transaction_receipt_tree: bank.is_transaction_receipt_tree_active(),
            accounts_delta_root: bank.is_accounts_delta_root_active(),
            epoch_validator_set_root: bank.is_epoch_validator_set_root_active(),
            epoch_boundary_commitment: bank.is_epoch_boundary_commitment_active(),
            accounts_state_root: bank.is_accounts_state_root_active(),
        }
    }
}

/// What is written of the accounts state of a block: what stepping back
/// from the state after the block to the state after its parent needs.
struct StateRecord {
    parent_slot: Slot,
    generation_in_use: Option<Epoch>,
    /// The generations the block holds, in ascending epoch order.
    generations: Vec<GenerationRecord>,
}

struct GenerationRecord {
    base: BaseId,
    /// For each entry of the block's accounts delta, in its order: the
    /// account hash the overlay held for the address before the block
    /// (`None`: it had no entry). `None` for all of them: the state before
    /// the block was not at hand when the block was rooted.
    previous: Option<Vec<Option<Hash>>>,
}

impl StateRecord {
    fn new(
        parent_slot: Slot,
        state: &AccountsState,
        generation_in_use: Option<Epoch>,
        delta: Option<&AccountsDelta>,
        parent_state: Option<&AccountsState>,
    ) -> Self {
        let generations = state
            .generations()
            .iter()
            .map(|generation| {
                let base = generation.base().id();
                let previous = delta.zip(parent_state).map(|(delta, parent_state)| {
                    // A generation the parent does not hold starts with this
                    // block: its overlay was empty before.
                    let before = parent_state
                        .generations()
                        .iter()
                        .find(|parent| parent.base().id() == base);
                    delta
                        .entries()
                        .iter()
                        .map(|entry| {
                            before?
                                .overlay()
                                .get(&entry.address)
                                .map(|value| split_overlay_value(value).0)
                        })
                        .collect()
                });
                GenerationRecord { base, previous }
            })
            .collect();
        Self {
            parent_slot,
            generation_in_use,
            generations,
        }
    }
}

/// The accounts state after `block`, from the state after `next`, its child:
/// the overlay of each generation of `block` is the one of `next` with the
/// changes of `next` undone. A generation that `next` no longer holds is
/// left out. `None` if the records do not allow it, or if the result is not
/// the state `block` committed to in slot 5.
fn state_before(
    block: &RetainedBlock,
    next: &RetainedBlock,
    next_state: &AccountsState,
) -> Option<Arc<AccountsState>> {
    let record = block.state_record.as_deref()?;
    let next_record = next.state_record.as_deref()?;
    if next_record.parent_slot != block.slot {
        return None;
    }
    let mut generations = Vec::new();
    for generation in &record.generations {
        let later = next_state
            .generations()
            .iter()
            .find(|later| later.base().id() == generation.base);
        let previous = next_record
            .generations
            .iter()
            .find(|later| later.base == generation.base)
            .and_then(|later| later.previous.as_ref());
        let (Some(later), Some(previous)) = (later, previous) else {
            continue;
        };
        let entries = next.accounts_delta.as_ref().map(|delta| delta.entries())?;
        if entries.len() != previous.len() {
            return None;
        }
        let mut overlay = later.overlay().clone();
        for (entry, previous) in entries.iter().zip(previous) {
            match previous {
                Some(now) => {
                    let (_, base) = split_overlay_value(overlay.get(&entry.address)?);
                    overlay.insert(entry.address, overlay_value(now, &base));
                }
                None => {
                    overlay.remove(&entry.address)?;
                }
            }
        }
        generations.push(Generation::from_parts(overlay, later.base().clone()));
    }
    if generations.is_empty() && !record.generations.is_empty() {
        return None;
    }
    let state = AccountsState::from_generations(generations);
    if let Some(epoch) = record.generation_in_use {
        let generation = state.generation(epoch)?;
        let (_, vector) = block.commitment.as_ref().ok()?;
        // Without the base root (its job is still running) the check is
        // left to the proof, which opens slot 5 itself.
        if let Some(base_root) = generation.base().root_if_known() {
            let committed = vector.0[CommitmentSlot::AccountsStateRoot.index() as usize];
            if accounts_state_root(epoch, &base_root, &generation.overlay().root()) != committed {
                warn!(
                    "light-client proofs: the accounts state rebuilt for block {} is not the \
                     one in its commitment root; not serving it",
                    block.slot
                );
                return None;
            }
        }
    }
    Some(Arc::new(state))
}

/// What the store keeps of one rooted block.
#[derive(Clone)]
struct RetainedBlock {
    slot: Slot,
    epoch: Epoch,
    block_id: Option<Hash>,
    features: ActiveFeatures,
    /// Commitment root and vector, or why the block has none.
    commitment: Result<(Hash, CommitmentVector), String>,
    /// Number of leaves of `H(B)`; the block's own leaf has this index.
    history_len: Option<u64>,
    /// Receipts in block order, if kept.
    receipts: Option<Result<Arc<Vec<RetainedReceipt>>, String>>,
    accounts_delta: Option<Arc<AccountsDelta>>,
    accounts_state: Option<(Arc<AccountsState>, Option<Epoch>)>,
    /// The block was read back from disk, where the accounts state is not
    /// kept, and its state was not rebuilt (yet).
    accounts_state_not_kept: bool,
    state_record: Option<Arc<StateRecord>>,
    epoch_boundary: Option<EpochBoundary>,
}

impl RetainedBlock {
    /// `parent_state`: the accounts state of the parent block, if at hand.
    fn from_bank(bank: &Bank, parent_state: Option<&AccountsState>) -> Self {
        let features = ActiveFeatures::of(bank);
        let commitment = match (bank.block_commitment(), bank.commitment_vector()) {
            (Some(commitment), Some(vector)) => Ok((commitment.commitment_root, vector)),
            (Some(_), None) => Err("the commitment vector of this block is not kept (a bank \
                                    restored from a snapshot)"
                .to_string()),
            (None, _) => Err(match bank.commitment_failure() {
                Some(failure) => format!("the block has no commitment root: {failure}"),
                None if features.block_commitment_root => {
                    "the block has no commitment state".to_string()
                }
                None => "block_commitment_root is not active for the block".to_string(),
            }),
        };
        let receipts = bank
            .retained_transaction_receipts()
            .map(|receipts| receipts.map(Arc::new).map_err(|err| err.to_string()));
        let accounts_delta = bank.accounts_delta();
        let accounts_state = bank
            .accounts_state()
            .map(|state| (state, bank.accounts_state_generation_in_use()));
        let state_record = accounts_state.as_ref().map(|(state, generation_in_use)| {
            Arc::new(StateRecord::new(
                bank.parent_slot(),
                state,
                *generation_in_use,
                accounts_delta.as_deref(),
                parent_state,
            ))
        });
        Self {
            slot: bank.slot(),
            epoch: bank.epoch(),
            block_id: bank.block_id().map(|block_id| block_id.to_hash()),
            features,
            history_len: bank
                .block_commitment()
                .map(|commitment| commitment.block_history.leaf_count()),
            commitment,
            receipts,
            accounts_delta,
            accounts_state,
            accounts_state_not_kept: false,
            state_record,
            epoch_boundary: bank.epoch_boundary(),
        }
    }

    fn block_id(&self) -> ProofResult<Hash> {
        self.block_id
            .ok_or_else(|| data_not_available(self.slot, "the block has no block id"))
    }

    fn commitment(&self) -> ProofResult<(Hash, &CommitmentVector)> {
        match &self.commitment {
            Ok((commitment_root, vector)) => Ok((*commitment_root, vector)),
            Err(reason) => Err(data_not_available(self.slot, reason)),
        }
    }

    fn opening(&self, slot: CommitmentSlot) -> ProofResult<(Hash, SlotOpening)> {
        let (_, vector) = self.commitment()?;
        let proof = vector
            .inclusion_proof(slot.index() as usize)
            .expect("every commitment slot is in range");
        let opening =
            SlotOpening::from_inclusion_proof(slot, &proof).expect("an eight-slot vector");
        Ok((vector.0[slot.index() as usize], opening))
    }

    fn require(&self, active: bool, feature: &str) -> ProofResult<()> {
        if active {
            Ok(())
        } else {
            Err(LightClientProofError::FeatureInactive {
                slot: self.slot,
                feature: feature.to_string(),
            })
        }
    }
}

fn data_not_available(slot: Slot, reason: impl Into<String>) -> LightClientProofError {
    LightClientProofError::DataNotAvailable {
        slot,
        reason: reason.into(),
    }
}

/// The certificates the store holds for one slot.
#[derive(Clone, Default)]
struct SlotCertificates {
    notarize: Vec<(Hash, Certificate)>,
    finalize: Option<Certificate>,
    fast_finalize: Vec<(Hash, Certificate)>,
}

impl SlotCertificates {
    /// The finality proof of `(slot, block_id)`, fast if there is one.
    fn finality(&self, block_id: &Hash) -> Option<FinalityProof> {
        if let Some((_, fast)) = self.fast_finalize.iter().find(|(id, _)| id == block_id) {
            return Some(FinalityProof {
                certificates: vec![fast.clone()],
            });
        }
        let (_, notarize) = self.notarize.iter().find(|(id, _)| id == block_id)?;
        let finalize = self.finalize.as_ref()?;
        Some(FinalityProof {
            certificates: vec![notarize.clone(), finalize.clone()],
        })
    }
}

/// The history leaves of the rooted chain: a frontier over the leaves that
/// are no longer kept, then the kept leaves.
#[derive(Default)]
struct HistoryLog {
    /// Frontier of leaves `[0, base_len)`.
    base: AppendOnlyFrontier,
    /// Hashes of leaves `[base_len, base_len + leaf_hashes.len())`.
    leaf_hashes: Vec<Hash>,
    /// `(slot, block_id, commitment_root)` of each kept leaf, in the same
    /// order. Slots strictly increase along a chain.
    entries: Vec<(Slot, Hash, Hash)>,
    /// Frontier of all leaves, to check each new block against in
    /// `O(log n)`.
    all: AppendOnlyFrontier,
    /// Number of times the log was restarted: the files follow one log.
    restarts: u64,
}

impl HistoryLog {
    /// The log of the leaves after `base`.
    fn from_parts(base: AppendOnlyFrontier, entries: Vec<(Slot, Hash, Hash)>) -> Self {
        let mut log = Self {
            all: base.clone(),
            base,
            ..Self::default()
        };
        for (slot, block_id, commitment_root) in entries {
            log.push(slot, block_id, commitment_root);
        }
        log
    }

    /// The root of the first `leaf_count` leaves, if they are the base and
    /// kept leaves.
    fn root_of_first(&self, leaf_count: u64) -> Option<Hash> {
        let kept = usize::try_from(leaf_count.checked_sub(self.base_len())?).ok()?;
        let mut frontier = self.base.clone();
        for leaf_hash in self.leaf_hashes.get(..kept)? {
            frontier.append_leaf_hash(*leaf_hash).ok()?;
        }
        Some(frontier.root(TreeId::BLOCK_HISTORY))
    }

    fn base_len(&self) -> u64 {
        self.base.leaf_count()
    }

    fn len(&self) -> u64 {
        self.all.leaf_count()
    }

    fn frontier_root(&self) -> Hash {
        self.all.root(TreeId::BLOCK_HISTORY)
    }

    /// Restarts the log at `H(B)` of a block: its frontier, or, when it is
    /// the anchor leaf alone, that leaf (so that the anchor can be proven).
    fn reset(&mut self, frontier: AppendOnlyFrontier, anchor: Option<(Slot, Hash)>) {
        self.restarts = self.restarts.saturating_add(1);
        self.leaf_hashes.clear();
        self.entries.clear();
        self.all = frontier.clone();
        let anchor_leaf = anchor
            .filter(|_| frontier.leaf_count() == 1)
            .map(|(slot, block_id)| {
                (
                    slot,
                    block_id,
                    history_leaf(slot, &block_id, &Hash::default()),
                )
            })
            .filter(|(_, _, leaf)| {
                frontier.subtree_roots() == [hash_leaf(TreeId::BLOCK_HISTORY, leaf)]
            });
        match anchor_leaf {
            Some((slot, block_id, leaf)) => {
                self.base = AppendOnlyFrontier::new();
                self.leaf_hashes
                    .push(hash_leaf(TreeId::BLOCK_HISTORY, &leaf));
                self.entries.push((slot, block_id, Hash::default()));
            }
            None => self.base = frontier,
        }
    }

    fn push(&mut self, slot: Slot, block_id: Hash, commitment_root: Hash) {
        let leaf_hash = hash_leaf(
            TreeId::BLOCK_HISTORY,
            &history_leaf(slot, &block_id, &commitment_root),
        );
        self.all
            .append_leaf_hash(leaf_hash)
            .expect("a history never holds 2^64 leaves");
        self.leaf_hashes.push(leaf_hash);
        self.entries.push((slot, block_id, commitment_root));
    }

    /// Folds the oldest kept leaves into the base until at most `max` are
    /// kept.
    fn prune(&mut self, max: usize) {
        let excess = self.leaf_hashes.len().saturating_sub(max);
        for leaf_hash in self.leaf_hashes.drain(..excess) {
            self.base
                .append_leaf_hash(leaf_hash)
                .expect("a history never holds 2^64 leaves");
        }
        self.entries.drain(..excess);
    }

    /// Index and entry of the kept leaf of the block at `slot`.
    fn find(&self, slot: Slot) -> Option<(u64, (Slot, Hash, Hash))> {
        let position = self
            .entries
            .binary_search_by_key(&slot, |(slot, ..)| *slot)
            .ok()?;
        Some((self.base_len() + position as u64, self.entries[position]))
    }

    /// A copy of the first `leaf_count` leaves (the base frontier and the
    /// kept leaves), to build proofs from without holding the store's lock.
    fn snapshot(&self, leaf_count: u64) -> Option<HistorySnapshot> {
        let kept = leaf_count.checked_sub(self.base_len())?;
        Some(HistorySnapshot {
            base: self.base.clone(),
            leaf_hashes: self
                .leaf_hashes
                .get(..usize::try_from(kept).ok()?)?
                .to_vec(),
        })
    }

    #[cfg(test)]
    fn inclusion_proof(&self, index: u64, leaf_count: u64) -> Option<InclusionProof> {
        self.snapshot(leaf_count)?.inclusion_proof(index)
    }
}

/// The first leaves of a history: a frontier over leaves `[0, base_len)`,
/// then leaf hashes.
struct HistorySnapshot {
    base: AppendOnlyFrontier,
    leaf_hashes: Vec<Hash>,
}

impl HistorySnapshot {
    fn base_len(&self) -> u64 {
        self.base.leaf_count()
    }

    /// `MTH` of leaves `[start, end)`. Ranges entirely in the base must be
    /// subtrees of its frontier, which holds for every node a proof of a
    /// kept leaf needs (see [`Self::inclusion_proof`]).
    fn subtree_hash(&self, start: u64, end: u64) -> Option<Hash> {
        let base_len = self.base_len();
        if end <= base_len {
            let mut subtree_start = 0u64;
            for (bit, root) in (0..u64::BITS)
                .rev()
                .filter(|bit| base_len & (1 << bit) != 0)
                .zip(self.base.subtree_roots())
            {
                let subtree_end = subtree_start + (1 << bit);
                if (subtree_start, subtree_end) == (start, end) {
                    return Some(*root);
                }
                subtree_start = subtree_end;
            }
            return None;
        }
        if end - start == 1 {
            return self
                .leaf_hashes
                .get(usize::try_from(start.checked_sub(base_len)?).ok()?)
                .copied();
        }
        let split = split_point(end - start);
        Some(hash_node(
            &self.subtree_hash(start, start + split)?,
            &self.subtree_hash(start + split, end)?,
        ))
    }

    fn path(&self, index: u64, start: u64, end: u64, siblings: &mut Vec<Hash>) -> Option<()> {
        if end - start <= 1 {
            return Some(());
        }
        let split = start + split_point(end - start);
        if index < split {
            self.path(index, start, split, siblings)?;
            siblings.push(self.subtree_hash(split, end)?);
        } else {
            self.path(index, split, end, siblings)?;
            siblings.push(self.subtree_hash(start, split)?);
        }
        Some(())
    }

    /// Inclusion proof of leaf `index` in the history of all the leaves of
    /// this snapshot, for a leaf after the base. A sibling entirely in the
    /// base is then always one of the base's frontier subtrees: a node that
    /// covers base leaves only and whose parent also covers a later leaf is
    /// a left child, of size `2^k`, at an offset that is a multiple of
    /// `2^(k+1)`, ending at or before `base_len` but whose parent ends
    /// after it: that is bit `k` of `base_len`.
    ///
    /// Costs one hash per leaf after the base (`O(n)`).
    fn inclusion_proof(&self, index: u64) -> Option<InclusionProof> {
        let leaf_count = self.base_len() + self.leaf_hashes.len() as u64;
        if index < self.base_len() || index >= leaf_count {
            return None;
        }
        let mut siblings = Vec::new();
        self.path(index, 0, leaf_count, &mut siblings)?;
        Some(InclusionProof {
            leaf_index: index,
            leaf_count,
            siblings,
        })
    }
}

/// Where the finality of a block comes from, found under the store's lock.
enum FinalitySource {
    Own(FinalityProof),
    Descendant {
        descendant: Arc<RetainedBlock>,
        descendant_id: Hash,
        finality: FinalityProof,
        history: Box<HistoryInput>,
    },
}

/// What a history proof is built from, taken under the store's lock.
struct HistoryInput {
    at_slot: Slot,
    block_history_root: Hash,
    opening: SlotOpening,
    slot: Slot,
    block_id: Hash,
    commitment_root: Hash,
    index: u64,
    snapshot: HistorySnapshot,
}

impl HistoryInput {
    /// Builds the proof (`O(n)` hashes), checked against the block's history
    /// root.
    fn prove(self) -> ProofResult<HistoryProof> {
        let Self {
            at_slot,
            block_history_root,
            opening,
            slot,
            block_id,
            commitment_root,
            index,
            snapshot,
        } = self;
        let proof = snapshot
            .inclusion_proof(index)
            .ok_or_else(|| data_not_available(slot, "the history leaf is not kept"))?;
        let leaf: [u8; HISTORY_LEAF_LEN] = history_leaf(slot, &block_id, &commitment_root);
        if !proof.verify(TreeId::BLOCK_HISTORY, &leaf, &block_history_root) {
            return Err(data_not_available(
                at_slot,
                "the kept history does not match the block's history root",
            ));
        }
        Ok(HistoryProof {
            block_history_root,
            opening,
            slot,
            block_id,
            commitment_root,
            proof,
        })
    }
}

/// Largest power of two less than `n` (`n > 1`).
fn split_point(n: u64) -> u64 {
    debug_assert!(n > 1);
    1 << (u64::BITS - 1 - (n - 1).leading_zeros())
}

#[derive(Default)]
struct Inner {
    blocks: BTreeMap<Slot, Arc<RetainedBlock>>,
    /// Highest rooted slot ingested.
    highest_root: Option<Slot>,
    history: HistoryLog,
    /// First signature of every kept receipt: block and position.
    signatures: HashMap<[u8; 64], (Slot, u64)>,
    signer_sets: BTreeMap<Epoch, Arc<Vec<Signer>>>,
}

/// See the module documentation.
pub struct LightClientProofStore {
    config: LightClientProofConfig,
    blockstore: Arc<Blockstore>,
    /// Shared with the thread that writes the files.
    inner: Arc<RwLock<Inner>>,
    /// Certificates by slot, under their own lock: the consensus pool never
    /// waits for a query.
    certificates: Mutex<BTreeMap<Slot, SlotCertificates>>,
    /// Slots whose certificates changed since they were last written to
    /// disk. Written at the next root, not by the consensus pool.
    unwritten_certificates: Mutex<BTreeSet<Slot>>,
    /// Slot of the oldest kept block (0: none yet).
    oldest_retained: AtomicU64,
    /// The files, read at startup. `None`: the store keeps nothing on disk.
    persistence: Option<Arc<Persistence>>,
    /// The thread that writes the files. Nothing else writes them, so the
    /// thread that roots never waits for the disk.
    writer: Option<DiskWriter>,
    /// Blocks read back from disk may still get their accounts state from
    /// the first banks rooted after the restart.
    states_to_rebuild: AtomicBool,
}

impl LightClientProofStore {
    pub fn new(config: LightClientProofConfig, blockstore: Arc<Blockstore>) -> Self {
        let inner = Arc::<RwLock<Inner>>::default();
        let persistence = config
            .persist_dir
            .as_deref()
            .and_then(Persistence::open)
            .map(Arc::new);
        let writer = persistence
            .as_ref()
            .and_then(|persistence| DiskWriter::start(persistence.clone(), inner.clone()));
        let store = Self {
            config,
            blockstore,
            inner,
            certificates: Mutex::default(),
            unwritten_certificates: Mutex::default(),
            oldest_retained: AtomicU64::new(0),
            // Without the thread nothing is written, so nothing is read.
            persistence: persistence.filter(|_| writer.is_some()),
            writer,
            states_to_rebuild: AtomicBool::new(false),
        };
        store.load();
        store.flush_writes();
        store
    }

    /// Queues `job` for the thread that writes the files, if there is one.
    fn disk(&self, job: Job) {
        if let Some(writer) = &self.writer {
            writer.send(job);
        }
    }

    /// Waits until everything the store has to write is on disk and
    /// durable. Root handling never calls this.
    pub fn flush_writes(&self) {
        if let Some(writer) = &self.writer {
            writer.flush();
        }
    }

    /// Number of files not written because the disk was too far behind.
    pub fn dropped_writes(&self) -> u64 {
        self.writer.as_ref().map_or(0, DiskWriter::dropped)
    }

    /// Takes back what a store wrote to the directory before: the blocks in
    /// order, as if they were rooted again. A block that blockstore does
    /// not have as a root (another ledger) is dropped.
    fn load(&self) {
        let Some(persistence) = &self.persistence else {
            return;
        };
        let mut inner = self.inner.write().unwrap();
        inner.signer_sets.extend(
            persistence
                .load_signer_sets()
                .into_iter()
                .map(|(epoch, signers)| (epoch, Arc::new(signers))),
        );
        self.certificates
            .lock()
            .unwrap()
            .extend(persistence.load_certificates());
        let records: Vec<_> = persistence
            .load_blocks()
            .into_iter()
            .filter(|record| {
                let slot = record.block.slot;
                let is_root = self.blockstore.is_root(slot);
                if !is_root {
                    warn!(
                        "light-client proofs: kept block {slot} is not a root in blockstore, \
                         dropping it"
                    );
                    self.disk(Job::RemoveBlock(slot));
                }
                is_root
            })
            .collect();
        // The history leaves, checked against the latest kept block whose
        // history they cover. Leaves after it are checked when the next
        // block is rooted, like any other.
        let (files, history) = persistence.load_history();
        self.disk(Job::HistoryFiles(files));
        if let Some((base, entries)) = history {
            let log = HistoryLog::from_parts(base, entries);
            let mismatch = records
                .iter()
                .rev()
                .filter_map(|record| record.history.as_ref())
                .find_map(|frontier| {
                    let root = log.root_of_first(frontier.leaf_count())?;
                    Some(root != frontier.root(TreeId::BLOCK_HISTORY))
                })
                .unwrap_or(false);
            if mismatch {
                warn!(
                    "light-client proofs: the kept history leaves do not match the kept blocks, \
                     dropping them"
                );
                // The files follow restart 0: they are replaced at the next
                // write.
                inner.history.restarts = 1;
            } else {
                inner.history = log;
            }
        }
        let mut loaded = 0usize;
        for record in records {
            let BlockRecord {
                block,
                history,
                parent_slot,
            } = record;
            self.states_to_rebuild.fetch_or(
                block.accounts_state_not_kept && block.features.accounts_state_root,
                Ordering::Relaxed,
            );
            self.ingest(&mut inner, block, history, parent_slot);
            loaded += 1;
        }
        self.prune(&mut inner);
        if loaded > 0 {
            info!(
                "light-client proofs: took back {loaded} kept blocks, up to slot {:?}",
                inner.highest_root
            );
        }
    }

    pub fn config(&self) -> &LightClientProofConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Ingestion.

    /// Takes the banks rooted by one new root (in any order; banks already
    /// ingested are skipped). Called from root handling while the banks are
    /// still alive.
    pub fn on_rooted_banks(&self, banks: &[Arc<Bank>]) {
        let mut banks: Vec<_> = banks.iter().collect();
        banks.sort_unstable_by_key(|bank| bank.slot());
        // Read what is needed from the banks before taking the lock.
        let (highest_root, mut missing_epochs, mut states) = {
            let inner = self.inner.read().unwrap();
            let missing: BTreeSet<Epoch> = banks
                .iter()
                .flat_map(|bank| [bank.epoch(), bank.epoch() + 1])
                .filter(|epoch| !inner.signer_sets.contains_key(epoch))
                .collect();
            // The accounts state after each block, to record what a block
            // changes in it: of the latest kept block, and of the banks.
            let states: HashMap<Slot, Arc<AccountsState>> = inner
                .blocks
                .values()
                .next_back()
                .and_then(|block| Some((block.slot, block.accounts_state.as_ref()?.0.clone())))
                .into_iter()
                .collect();
            (inner.highest_root, missing, states)
        };
        states.extend(
            banks
                .iter()
                .filter_map(|bank| Some((bank.slot(), bank.accounts_state()?))),
        );
        let rebuild = self.states_to_rebuild.load(Ordering::Relaxed);
        let rooted_states: Vec<_> = banks
            .iter()
            .filter(|_| rebuild)
            .filter_map(|bank| {
                Some((
                    bank.slot(),
                    bank.block_id()?.to_hash(),
                    bank.accounts_state()?,
                    bank.accounts_state_generation_in_use(),
                ))
            })
            .collect();
        let new_blocks: Vec<_> = banks
            .into_iter()
            .filter(|bank| highest_root.is_none_or(|root| bank.slot() > root))
            .map(|bank| {
                let signer_sets: Vec<_> = [bank.epoch(), bank.epoch() + 1]
                    .into_iter()
                    .filter(|epoch| missing_epochs.remove(epoch))
                    .filter_map(|epoch| Some((epoch, Arc::new(bank.validator_set_signers(epoch)?))))
                    .collect();
                (
                    BlockRecord {
                        block: RetainedBlock::from_bank(
                            bank,
                            states.get(&bank.parent_slot()).map(Arc::as_ref),
                        ),
                        history: bank
                            .block_commitment()
                            .map(|commitment| commitment.block_history),
                        parent_slot: bank.parent_slot(),
                    },
                    signer_sets,
                )
            })
            .collect();
        // A gap between the kept history and the first new block (the node
        // restarted from a snapshot later than its last kept block) is
        // filled from blockstore, if the blocks are there.
        let backfill = new_blocks.first().and_then(|(record, _)| {
            self.history_gap_from_blockstore(record.block.slot, record.history.as_ref()?)
        });
        // The files are queued for the writer thread: neither this thread
        // nor a query waits for the disk.
        if self.writer.is_some() {
            for (record, signer_sets) in &new_blocks {
                self.disk(Job::Block(Box::new(record.clone())));
                for (epoch, signers) in signer_sets {
                    self.disk(Job::SignerSet(*epoch, signers.clone()));
                }
            }
            let unwritten = std::mem::take(&mut *self.unwritten_certificates.lock().unwrap());
            for slot in unwritten {
                let certificates = self.certificates.lock().unwrap().get(&slot).cloned();
                if let Some(certificates) = certificates {
                    self.disk(Job::Certificates(slot, certificates));
                }
            }
        }
        let ingested = !new_blocks.is_empty();
        if ingested {
            let mut inner = self.inner.write().unwrap();
            if let Some((kept, leaves)) = backfill
                && inner.history.len() == kept
            {
                info!(
                    "light-client proofs: took {} history leaves from blockstore, from slot {:?}",
                    leaves.len(),
                    leaves.first().map(|(slot, ..)| *slot)
                );
                for (slot, block_id, commitment_root) in leaves {
                    inner.history.push(slot, block_id, commitment_root);
                }
            }
            for (record, signer_sets) in new_blocks {
                let BlockRecord {
                    block,
                    history,
                    parent_slot,
                } = record;
                if inner.highest_root.is_some_and(|root| block.slot <= root) {
                    continue;
                }
                inner.signer_sets.extend(signer_sets);
                self.ingest(&mut inner, block, history, parent_slot);
            }
            self.prune(&mut inner);
        }
        if rebuild {
            self.rebuild_accounts_states(&rooted_states);
            // Once a new block is in, every block that can be reached from
            // a bank has been.
            if ingested {
                self.states_to_rebuild.store(false, Ordering::Relaxed);
            }
        }
        // The history leaves are written from the store itself, after the
        // files queued above.
        self.disk(Job::Commit);
    }

    /// The history leaves between the kept history and `frontier`, the
    /// history of the rooted block at `slot` before its own leaf, read from
    /// blockstore: the blocks from the parent of `slot` back to the last
    /// kept leaf, each with the commitment root its footer carries. With
    /// them the kept history has exactly the root of `frontier`, or `None`
    /// is returned: blockstore does not have every block of the gap (the
    /// node started from a snapshot it did not make), or there is no gap.
    ///
    /// Returns the number of leaves the kept history had, and the leaves.
    fn history_gap_from_blockstore(
        &self,
        slot: Slot,
        frontier: &AppendOnlyFrontier,
    ) -> Option<(u64, Vec<persist::HistoryEntry>)> {
        let (kept, (last_slot, last_block_id, _), mut all) = {
            let inner = self.inner.read().unwrap();
            let history = &inner.history;
            (
                history.len(),
                history.entries.last().copied()?,
                history.all.clone(),
            )
        };
        let missing = frontier
            .leaf_count()
            .checked_sub(kept)
            .filter(|missing| (1..=self.config.retained_history as u64).contains(missing))?;
        let parent_of = |meta: &solana_ledger::blockstore_meta::SlotMeta| {
            Some((meta.parent_slot?, meta.parent_block_id))
        };
        let mut at = parent_of(&self.blockstore.meta(slot).ok()??)?;
        let mut leaves = Vec::new();
        while at.0 > last_slot {
            if leaves.len() as u64 >= missing {
                return None;
            }
            let (meta, slot_meta) = self
                .blockstore
                .get_parent_repair_metadata(at.0, at.1)
                .ok()??;
            leaves.push((at.0, at.1, meta.footer_binding()?.commitment_root));
            at = parent_of(&slot_meta)?;
        }
        if at != (last_slot, last_block_id) || leaves.len() as u64 != missing {
            return None;
        }
        leaves.reverse();
        for (slot, block_id, commitment_root) in &leaves {
            all.append_leaf_hash(hash_leaf(
                TreeId::BLOCK_HISTORY,
                &history_leaf(*slot, block_id, commitment_root),
            ))
            .ok()?;
        }
        (all.root(TreeId::BLOCK_HISTORY) == frontier.root(TreeId::BLOCK_HISTORY))
            .then_some((kept, leaves))
    }

    /// Gives the blocks read back from disk their accounts state: from the
    /// rooted banks of the same blocks (`rooted`: slot, block id, state,
    /// generation in use), and for the blocks before a block with a state by
    /// undoing that block's changes (see [`state_before`]).
    fn rebuild_accounts_states(&self, rooted: &[(Slot, Hash, Arc<AccountsState>, Option<Epoch>)]) {
        let blocks: Vec<Arc<RetainedBlock>> = self
            .inner
            .read()
            .unwrap()
            .blocks
            .values()
            .cloned()
            .collect();
        let mut rebuilt: Vec<(Slot, Arc<AccountsState>, Option<Epoch>)> = Vec::new();
        // From the latest block down: `next` is the block after, with its
        // state, while the chain of states is unbroken.
        let mut next: Option<(&RetainedBlock, Arc<AccountsState>)> = None;
        for block in blocks.iter().rev() {
            if let Some((state, _)) = &block.accounts_state {
                next = Some((block, state.clone()));
                continue;
            }
            if !block.accounts_state_not_kept {
                next = None;
                continue;
            }
            let from_bank = rooted
                .iter()
                .find(|(slot, block_id, ..)| {
                    *slot == block.slot && Some(*block_id) == block.block_id
                })
                .map(|(_, _, state, generation_in_use)| (state.clone(), *generation_in_use));
            let state = from_bank.or_else(|| {
                let (after, after_state) = next.as_ref()?;
                let state = state_before(block, after, after_state)?;
                let generation_in_use = block.state_record.as_ref()?.generation_in_use;
                Some((state, generation_in_use))
            });
            next = state.map(|(state, generation_in_use)| {
                rebuilt.push((block.slot, state.clone(), generation_in_use));
                (block.as_ref(), state)
            });
        }
        if rebuilt.is_empty() {
            return;
        }
        info!(
            "light-client proofs: rebuilt the accounts state of {} kept blocks, back to slot {}",
            rebuilt.len(),
            rebuilt.last().map(|(slot, ..)| *slot).unwrap_or_default()
        );
        let mut inner = self.inner.write().unwrap();
        for (slot, state, generation_in_use) in rebuilt {
            if let Some(block) = inner.blocks.get_mut(&slot) {
                *block = Arc::new(RetainedBlock {
                    accounts_state: Some((state, generation_in_use)),
                    accounts_state_not_kept: false,
                    ..RetainedBlock::clone(block)
                });
            }
        }
    }

    fn ingest(
        &self,
        inner: &mut Inner,
        block: RetainedBlock,
        history: Option<AppendOnlyFrontier>,
        parent_slot: Slot,
    ) {
        let slot = block.slot;

        // History: the log must hold `H(B)` before `B`'s leaf is appended.
        if let (Some(frontier), Ok((commitment_root, _)), Some(block_id)) =
            (history, &block.commitment, block.block_id)
        {
            // A leaf read back from the history files is already there.
            let kept = frontier.leaf_count() < inner.history.len()
                && inner.history.find(slot)
                    == Some((frontier.leaf_count(), (slot, block_id, *commitment_root)));
            if !kept
                && (inner.history.len() != frontier.leaf_count()
                    || inner.history.frontier_root() != frontier.root(TreeId::BLOCK_HISTORY))
            {
                if inner.history.len() > 0 {
                    warn!(
                        "light-client proofs: the history of rooted block {slot} does not extend \
                         the kept history ({} leaves); restarting it",
                        inner.history.len()
                    );
                }
                let anchor = inner
                    .blocks
                    .get(&parent_slot)
                    .filter(|parent| !parent.features.block_commitment_root)
                    .and_then(|parent| Some((parent.slot, parent.block_id?)));
                inner.history.reset(frontier, anchor);
            }
            if !kept {
                inner.history.push(slot, block_id, *commitment_root);
            }
        }

        if let Some(Ok(receipts)) = &block.receipts {
            for (position, retained) in receipts.iter().enumerate() {
                inner
                    .signatures
                    .insert(retained.receipt.signature, (slot, position as u64));
            }
        }

        inner.blocks.insert(slot, Arc::new(block));
        inner.highest_root = Some(slot);
    }

    fn prune(&self, inner: &mut Inner) {
        while inner.blocks.len() > self.config.retained_blocks.max(1) {
            let (slot, block) = inner.blocks.pop_first().expect("not empty");
            if let Some(Ok(receipts)) = &block.receipts {
                for retained in receipts.iter() {
                    inner.signatures.remove(&retained.receipt.signature);
                }
            }
            self.disk(Job::RemoveBlock(slot));
        }
        if let Some(oldest) = inner.blocks.keys().next().copied() {
            self.oldest_retained.store(oldest, Ordering::Relaxed);
            let dropped = {
                let mut certificates = self.certificates.lock().unwrap();
                let kept = certificates.split_off(&oldest);
                !std::mem::replace(&mut *certificates, kept).is_empty()
            };
            if dropped {
                self.disk(Job::RemoveCertificatesBefore(oldest));
            }
        }
        inner.history.prune(self.config.retained_history.max(1));
        while inner.signer_sets.len() > RETAINED_SIGNER_SET_EPOCHS {
            if let Some((epoch, _)) = inner.signer_sets.pop_first() {
                self.disk(Job::RemoveSignerSet(epoch));
            }
        }
    }

    /// Takes a certificate completed by the consensus pool. Keeps the
    /// notarization, finalization and fast-finalization certificates of
    /// slots not older than the oldest kept block.
    pub fn on_certificate(&self, certificate: &VotorCertificate) {
        let (slot, block_id, kind) = match certificate.cert_type {
            VotorCertificateType::Notarize(block) => (
                block.slot,
                Some(block.block_id.to_hash()),
                CertificateKind::Notarize,
            ),
            VotorCertificateType::Finalize(slot) => (slot, None, CertificateKind::Finalize),
            VotorCertificateType::FinalizeFast(block) => (
                block.slot,
                Some(block.block_id.to_hash()),
                CertificateKind::FastFinalize,
            ),
            VotorCertificateType::NotarizeFallback(_)
            | VotorCertificateType::Skip(_)
            | VotorCertificateType::Genesis(_) => return,
        };
        let light_client_certificate = Certificate {
            kind,
            signature: certificate.signature,
            bitmap: certificate.bitmap.clone(),
        };
        // Blocks older than the oldest kept block are never served.
        if slot < self.oldest_retained.load(Ordering::Relaxed) {
            return;
        }
        if self.writer.is_some() {
            self.unwritten_certificates.lock().unwrap().insert(slot);
        }
        let mut certificates = self.certificates.lock().unwrap();
        let certificates = certificates.entry(slot).or_default();
        match (kind, block_id) {
            (CertificateKind::Notarize, Some(block_id)) => {
                if !certificates.notarize.iter().any(|(id, _)| *id == block_id) {
                    certificates
                        .notarize
                        .push((block_id, light_client_certificate));
                }
            }
            (CertificateKind::FastFinalize, Some(block_id)) => {
                if !certificates
                    .fast_finalize
                    .iter()
                    .any(|(id, _)| *id == block_id)
                {
                    certificates
                        .fast_finalize
                        .push((block_id, light_client_certificate));
                }
            }
            (CertificateKind::Finalize, _) => {
                certificates
                    .finalize
                    .get_or_insert(light_client_certificate);
            }
            _ => {}
        }
    }

    // -----------------------------------------------------------------------
    // Queries.

    /// The kept rooted block at `slot`, or why there is none.
    fn final_block(&self, inner: &Inner, slot: Slot) -> ProofResult<Arc<RetainedBlock>> {
        if let Some(block) = inner.blocks.get(&slot) {
            return Ok(block.clone());
        }
        let finalized_slot = inner.highest_root;
        if inner.highest_root.is_none_or(|root| slot > root) || !self.blockstore.is_root(slot) {
            return Err(LightClientProofError::BlockNotFinal {
                slot,
                finalized_slot,
            });
        }
        Err(LightClientProofError::BlockNotRetained {
            slot,
            first_retained: inner.blocks.keys().next().copied(),
            last_retained: inner.blocks.keys().next_back().copied(),
        })
    }

    /// Replaces the accounts state kept for the block at `slot`.
    #[cfg(test)]
    pub(crate) fn set_accounts_state_for_tests(&self, slot: Slot, state: Arc<AccountsState>) {
        let mut inner = self.inner.write().unwrap();
        let block = Arc::get_mut(inner.blocks.get_mut(&slot).unwrap()).unwrap();
        let (kept, _) = block.accounts_state.as_mut().unwrap();
        *kept = state;
    }

    /// The latest finalized slot this store has ingested.
    pub fn highest_root(&self) -> Option<Slot> {
        self.inner.read().unwrap().highest_root
    }

    /// The footer proof of `(slot, block_id)`, from blockstore, to answer a
    /// query about the block at `asked` (the block itself, or one that is
    /// final through it).
    ///
    /// The node that produced a block votes on it, and can root it, before
    /// its broadcast has written the last shreds of the block to blockstore.
    /// Until then the block cannot be proven final from this node: the
    /// answer is "block not final", which callers retry, as for a block
    /// that is not rooted yet.
    fn footer_proof(
        &self,
        block: &RetainedBlock,
        block_id: &Hash,
        asked: Slot,
    ) -> ProofResult<FooterProof> {
        let slot = block.slot;
        let Some((meta, slot_meta)) = self
            .blockstore
            .get_parent_repair_metadata(slot, *block_id)
            .map_err(|err| data_not_available(slot, format!("blockstore: {err}")))?
        else {
            return Err(match self.blockstore.meta(slot) {
                // Still being written. Every block before `asked` was
                // rooted earlier, so `asked - 1` is a lower bound of the
                // latest slot that is final here.
                Ok(Some(slot_meta)) if !slot_meta.is_full() => {
                    LightClientProofError::BlockNotFinal {
                        slot: asked,
                        finalized_slot: Some(asked.saturating_sub(1)),
                    }
                }
                _ => {
                    data_not_available(slot, "blockstore has no complete block with this block id")
                }
            });
        };
        let binding = meta.footer_binding().ok_or_else(|| {
            data_not_available(
                slot,
                "the block id does not bind the footer (no version 2 footer)",
            )
        })?;
        let parent_slot = slot_meta
            .parent_slot
            .ok_or_else(|| data_not_available(slot, "blockstore has no parent for the block"))?;
        let proof = meta
            .get_parent_info_proof()
            .ok_or_else(|| data_not_available(slot, "blockstore has no block id proof"))?;
        if proof.len() % MERKLE_PROOF_ENTRY_LEN != 0 {
            return Err(data_not_available(slot, "malformed block id proof"));
        }
        let footer = FooterProof {
            parent_slot,
            parent_block_id: slot_meta.parent_block_id,
            fec_set_count: meta.fec_set_count(),
            bank_hash: binding.bank_hash,
            commitment_root: binding.commitment_root,
            proof: proof.as_chunks::<MERKLE_PROOF_ENTRY_LEN>().0.to_vec(),
        };
        if let Ok((commitment_root, _)) = &block.commitment
            && *commitment_root != footer.commitment_root
        {
            return Err(data_not_available(
                slot,
                "the footer in blockstore does not match the bank's commitment root",
            ));
        }
        Ok(footer)
    }

    /// The finality proof of `(slot, block_id)` from the certificates kept.
    fn certificates_of(&self, slot: Slot, block_id: &Hash) -> Option<FinalityProof> {
        self.certificates
            .lock()
            .unwrap()
            .get(&slot)
            .and_then(|certificates| certificates.finality(block_id))
    }

    /// The block of `epoch` to open a commitment in when the caller names
    /// none: the latest kept block that passes `usable` and has certificates
    /// of its own.
    ///
    /// Certificates of a block are signed by the validators of the block's
    /// epoch. A block without its own is final through a later block, and
    /// the nearest one with certificates may be in the next epoch, signed by
    /// a set that a client still holding the signer set of `epoch` cannot
    /// check. Every block before one with certificates is final through it,
    /// so the latest block with certificates is the latest one such a client
    /// can verify. Without any, the latest usable block is returned: its
    /// proof still verifies for a client that holds the later set.
    fn latest_block_of_epoch(
        &self,
        inner: &Inner,
        epoch: Epoch,
        usable: impl Fn(&RetainedBlock) -> bool,
    ) -> Option<Arc<RetainedBlock>> {
        let mut candidates = inner
            .blocks
            .values()
            .rev()
            .skip_while(|block| block.epoch > epoch)
            .take_while(|block| block.epoch == epoch)
            .filter(|block| block.block_id.is_some() && usable(block));
        let latest = candidates.next()?;
        std::iter::once(latest)
            .chain(candidates)
            .find(|block| {
                block
                    .block_id
                    .is_some_and(|block_id| self.certificates_of(block.slot, &block_id).is_some())
            })
            .or(Some(latest))
            .cloned()
    }

    /// What the history proof of the kept leaf of the block at `slot`, from
    /// the block `at`, is built from.
    fn history_input(
        &self,
        inner: &Inner,
        slot: Slot,
        at: &RetainedBlock,
    ) -> ProofResult<HistoryInput> {
        at.require(at.features.block_commitment_root, "block_commitment_root")?;
        let (block_history_root, opening) = at.opening(CommitmentSlot::BlockHistoryRoot)?;
        let leaf_count = at
            .history_len
            .ok_or_else(|| data_not_available(at.slot, "the block has no block history"))?;
        let (index, (_, block_id, commitment_root)) =
            inner.history.find(slot).ok_or_else(|| {
                let first_kept = inner.history.entries.first().map(|(slot, ..)| *slot);
                match first_kept {
                    // The kept leaves are every rooted block from the first
                    // one on: a slot in that range without a leaf is not a
                    // finalized block (skipped, or on another fork), as for
                    // the other methods.
                    Some(first_kept) if slot >= first_kept && !self.blockstore.is_root(slot) => {
                        LightClientProofError::BlockNotFinal {
                            slot,
                            finalized_slot: inner.highest_root,
                        }
                    }
                    _ => data_not_available(
                        slot,
                        format!(
                            "no kept history leaf for this block (history proofs reach back to \
                             slot {first_kept:?})"
                        ),
                    ),
                }
            })?;
        if index >= leaf_count {
            return Err(LightClientProofError::NotFound {
                what: format!("block {slot} in the history of block {}", at.slot),
            });
        }
        let snapshot = inner
            .history
            .snapshot(leaf_count)
            .ok_or_else(|| data_not_available(slot, "the history leaf is not kept"))?;
        Ok(HistoryInput {
            at_slot: at.slot,
            block_history_root,
            opening,
            slot,
            block_id,
            commitment_root,
            index,
            snapshot,
        })
    }

    /// `getFinalityProof`.
    pub fn finality_proof(&self, slot: Slot) -> ProofResult<FinalityResponse> {
        // Under the lock: the block, and the descendant and history to use
        // if the block has no certificates here. Proofs are built after.
        let (block, block_id, source) = {
            let inner = self.inner.read().unwrap();
            let block = self.final_block(&inner, slot)?;
            block.require(
                block.features.block_commitment_root,
                "block_commitment_root",
            )?;
            let block_id = block.block_id()?;
            let source = match self.certificates_of(slot, &block_id) {
                Some(finality) => FinalitySource::Own(finality),
                // Finalized through a descendant, or its certificates are not
                // here: the nearest later kept block with certificates.
                None => {
                    let (descendant, descendant_id, finality) = inner
                        .blocks
                        .range(slot + 1..)
                        .find_map(|(descendant_slot, descendant)| {
                            let descendant_id = descendant.block_id?;
                            let finality =
                                self.certificates_of(*descendant_slot, &descendant_id)?;
                            Some((descendant.clone(), descendant_id, finality))
                        })
                        .ok_or(LightClientProofError::CertificatesNotAvailable { slot })?;
                    let history = Box::new(self.history_input(&inner, slot, &descendant)?);
                    FinalitySource::Descendant {
                        descendant,
                        descendant_id,
                        finality,
                        history,
                    }
                }
            };
            (block, block_id, source)
        };
        let footer = self.footer_proof(&block, &block_id, slot)?;
        let finality = match source {
            FinalitySource::Own(finality) => Finality::Certificates(finality),
            FinalitySource::Descendant {
                descendant,
                descendant_id,
                finality,
                history,
            } => Finality::Descendant(Box::new(FinalizedDescendant {
                slot: descendant.slot,
                block_id: descendant_id,
                finality,
                footer: self.footer_proof(&descendant, &descendant_id, slot)?,
                history: history.prove()?,
            })),
        };
        Ok(FinalityResponse {
            slot,
            block_id,
            footer,
            finality,
        })
    }

    /// `getBlockHistoryProof`.
    pub fn block_history_proof(
        &self,
        slot: Slot,
        at_slot: Slot,
    ) -> ProofResult<BlockHistoryResponse> {
        let (at_block_id, input) = {
            let inner = self.inner.read().unwrap();
            let at = self.final_block(&inner, at_slot)?;
            if slot >= at_slot {
                return Err(LightClientProofError::NotFound {
                    what: format!("block {slot} in the history of block {at_slot}"),
                });
            }
            (at.block_id()?, self.history_input(&inner, slot, &at)?)
        };
        Ok(BlockHistoryResponse {
            at_slot,
            at_block_id,
            history: input.prove()?,
        })
    }

    /// `getTransactionProof`.
    pub fn transaction_proof(
        &self,
        signature: &Signature,
        instructions: Option<&RpcProofSelection>,
        events: Option<&RpcProofSelection>,
    ) -> ProofResult<TransactionResponse> {
        let signature_bytes: [u8; 64] = (*signature).into();
        let found = {
            let inner = self.inner.read().unwrap();
            match inner.signatures.get(&signature_bytes).copied() {
                Some((slot, position)) => Some((position, self.final_block(&inner, slot)?)),
                None => None,
            }
        };
        let Some((position, block)) = found else {
            // Name the block if blockstore knows it (with transaction history).
            if let Ok(Some((slot, _))) = self.blockstore.get_rooted_transaction_status(*signature) {
                self.final_block(&self.inner.read().unwrap(), slot)?;
                return Err(data_not_available(
                    slot,
                    "the receipts of this block are not kept",
                ));
            }
            return Err(LightClientProofError::NotFound {
                what: format!("transaction {signature} in the kept blocks"),
            });
        };
        let slot = block.slot;
        block.require(
            block.features.transaction_receipt_tree,
            "transaction_receipt_tree",
        )?;
        let receipts = match &block.receipts {
            Some(Ok(receipts)) => receipts.clone(),
            Some(Err(reason)) => return Err(data_not_available(slot, reason)),
            None => return Err(data_not_available(slot, "receipts are not kept")),
        };
        let (receipts_root, opening) = block.opening(CommitmentSlot::ReceiptsRoot)?;
        let leaf_hashes: Vec<Hash> = receipts
            .iter()
            .map(|retained| retained.receipt.leaf_hash())
            .collect();
        let retained = &receipts[position as usize];
        let proof = make_inclusion_proof_from_leaf_hashes(&leaf_hashes, position as usize)
            .expect("the position is in range");
        if !proof.verify_leaf_hash(
            TreeId::TRANSACTION_RECEIPTS,
            leaf_hashes[position as usize],
            &receipts_root,
        ) {
            return Err(data_not_available(
                slot,
                "the kept receipts do not match the block's receipts root",
            ));
        }
        let transaction = TransactionProof {
            receipts_root,
            opening,
            receipt: retained.receipt,
            proof,
        };
        if instructions.is_none() && events.is_none() {
            return Ok(TransactionResponse {
                slot,
                block_id: block.block_id()?,
                transaction,
                instructions: vec![],
                events: vec![],
            });
        }
        let leaves = retained.leaves.as_deref().ok_or_else(|| {
            data_not_available(
                slot,
                "the trace and events of this transaction are not kept",
            )
        })?;
        let select = |selection: Option<&RpcProofSelection>,
                      len: usize,
                      what: &str|
         -> ProofResult<BTreeSet<usize>> {
            match selection {
                None => Ok(BTreeSet::new()),
                Some(RpcProofSelection::All) => Ok((0..len).collect()),
                Some(RpcProofSelection::Indices(indices)) => indices
                    .iter()
                    .map(|index| {
                        usize::try_from(*index)
                            .ok()
                            .filter(|index| *index < len)
                            .ok_or_else(|| LightClientProofError::NotFound {
                                what: format!("{what} {index} of transaction {signature}"),
                            })
                    })
                    .collect(),
            }
        };
        let event_indices = select(events, leaves.events.len(), "event")?;
        let mut instruction_indices = select(instructions, leaves.records.len(), "instruction")?;
        for index in &event_indices {
            instruction_indices.insert(leaves.events[*index].instruction_index as usize);
        }
        let instruction_proofs: BTreeMap<usize, InstructionProof> = instruction_indices
            .into_iter()
            .map(|index| {
                let proof = instruction_record_inclusion_proof(&leaves.records, index).ok_or_else(
                    || data_not_available(slot, "an event names an instruction out of range"),
                )?;
                Ok((
                    index,
                    InstructionProof {
                        transaction: transaction.clone(),
                        record: leaves.records[index],
                        proof,
                    },
                ))
            })
            .collect::<ProofResult<_>>()?;
        let event_proofs = event_indices
            .into_iter()
            .map(|index| {
                let event = &leaves.events[index];
                EventProof {
                    instruction: instruction_proofs[&(event.instruction_index as usize)].clone(),
                    event: event.clone(),
                    proof: event_inclusion_proof(&leaves.events, index)
                        .expect("the index is in range"),
                }
            })
            .collect();
        Ok(TransactionResponse {
            slot,
            block_id: block.block_id()?,
            transaction,
            instructions: instruction_proofs.into_values().collect(),
            events: event_proofs,
        })
    }

    /// `getAccountDeltaProof`.
    pub fn account_delta_proof(
        &self,
        pubkey: &Pubkey,
        slot: Slot,
    ) -> ProofResult<AccountDeltaResponse> {
        let block = self.final_block(&self.inner.read().unwrap(), slot)?;
        block.require(block.features.accounts_delta_root, "accounts_delta_root")?;
        let delta = block
            .accounts_delta
            .as_ref()
            .ok_or_else(|| data_not_available(slot, "the block has no accounts delta"))?;
        let (accounts_delta_root, opening) = block.opening(CommitmentSlot::AccountsDeltaRoot)?;
        if delta.root() != accounts_delta_root {
            return Err(data_not_available(
                slot,
                "the kept accounts delta does not match the block's accounts delta root",
            ));
        }
        let address = pubkey.to_bytes();
        let delta = match delta.inclusion_proof(&address) {
            Some(leaf) => AccountDelta::Changed(AccountDeltaProof {
                accounts_delta_root,
                opening,
                leaf,
            }),
            None => AccountDelta::Unchanged(AccountUnchangedProof {
                accounts_delta_root,
                opening,
                proof: delta.unchanged_proof(&address).ok_or_else(|| {
                    data_not_available(slot, "no proof that the account is unchanged")
                })?,
            }),
        };
        Ok(AccountDeltaResponse {
            slot,
            block_id: block.block_id()?,
            delta,
        })
    }

    /// `getAccountStateProof`: at `slot`, or at the latest kept rooted
    /// block.
    pub fn account_state_proof(
        &self,
        pubkey: &Pubkey,
        slot: Option<Slot>,
    ) -> ProofResult<AccountStateResponse> {
        let block =
            {
                let inner = self.inner.read().unwrap();
                let slot =
                    match slot {
                        Some(slot) => slot,
                        None => *inner.blocks.keys().next_back().ok_or(
                            LightClientProofError::NotFound {
                                what: "a finalized block on this node".to_string(),
                            },
                        )?,
                    };
                self.final_block(&inner, slot)?
            };
        // A proof from the base tree hashes a few thousand nodes: not under
        // the lock.
        let slot = block.slot;
        block.require(block.features.accounts_state_root, "accounts_state_root")?;
        let (state, generation_in_use) = block.accounts_state.as_ref().ok_or_else(|| {
            data_not_available(
                slot,
                if block.accounts_state_not_kept {
                    "the accounts state of a block rooted before this node restarted is not \
                         kept"
                } else {
                    "the block has no accounts state"
                },
            )
        })?;
        let (_, vector) = block.commitment()?;
        let proof = state
            .account_proof(*generation_in_use, vector, pubkey, BaseWait::NoWait)
            .map_err(|err| match err {
                AccountsStateProofError::BaseNotReady => {
                    LightClientProofError::BaseTreeNotReady { slot }
                }
                err => data_not_available(slot, err.to_string()),
            })?;
        Ok(AccountStateResponse {
            slot,
            block_id: block.block_id()?,
            proof: AccountStateProof(proof),
        })
    }

    /// `getValidatorSet`.
    pub fn validator_set(
        &self,
        epoch: Epoch,
        at_slot: Option<Slot>,
    ) -> ProofResult<ValidatorSetResponse> {
        let (signers, committing) = {
            let inner = self.inner.read().unwrap();
            let signers = inner
                .signer_sets
                .get(&epoch)
                .ok_or_else(|| LightClientProofError::NotFound {
                    what: format!("the signer set of epoch {epoch} on this node"),
                })?
                .clone();
            let committing = match at_slot {
                Some(at_slot) => {
                    let block = self.final_block(&inner, at_slot)?;
                    if Some(block.epoch) != epoch.checked_sub(1) {
                        return Err(LightClientProofError::NotFound {
                            what: format!(
                                "the signer set of epoch {epoch} in block {at_slot} (of epoch {})",
                                block.epoch
                            ),
                        });
                    }
                    block.require(
                        block.features.epoch_validator_set_root,
                        "epoch_validator_set_root",
                    )?;
                    Some(block)
                }
                None => epoch.checked_sub(1).and_then(|previous| {
                    self.latest_block_of_epoch(&inner, previous, |block| {
                        block.features.epoch_validator_set_root && block.commitment.is_ok()
                    })
                }),
            };
            (signers, committing)
        };
        let signer_set = SignerSet::new(epoch, signers.as_ref().clone()).map_err(|err| {
            LightClientProofError::NotFound {
                what: format!("a valid signer set of epoch {epoch} ({err})"),
            }
        })?;
        let committed_by = committing
            .map(|block| {
                let (value, opening) = block.opening(CommitmentSlot::ValidatorSetRoot)?;
                if value != signer_set.validator_set_root() {
                    return Err(data_not_available(
                        block.slot,
                        "the kept signer set does not match the block's validator set root",
                    ));
                }
                Ok(OpenedAt {
                    slot: block.slot,
                    block_id: block.block_id()?,
                    opening,
                })
            })
            .transpose()?;
        Ok(ValidatorSetResponse {
            signer_set,
            committed_by,
        })
    }

    /// `getEpochBoundary`.
    pub fn epoch_boundary(
        &self,
        epoch: Epoch,
        at_slot: Option<Slot>,
    ) -> ProofResult<EpochBoundaryResponse> {
        let inner = self.inner.read().unwrap();
        let block = match at_slot {
            Some(at_slot) => {
                let block = self.final_block(&inner, at_slot)?;
                if block.epoch != epoch {
                    return Err(LightClientProofError::NotFound {
                        what: format!(
                            "the boundary of epoch {epoch} in block {at_slot} (of epoch {})",
                            block.epoch
                        ),
                    });
                }
                block
            }
            None => self
                .latest_block_of_epoch(&inner, epoch, |_| true)
                .ok_or_else(|| LightClientProofError::NotFound {
                    what: format!("a kept finalized block of epoch {epoch}"),
                })?,
        };
        block.require(
            block.features.epoch_boundary_commitment,
            "epoch_boundary_commitment",
        )?;
        let boundary = block.epoch_boundary.ok_or_else(|| {
            data_not_available(
                block.slot,
                "the block commits to no epoch boundary (genesis epoch, or the feature took \
                 effect within the epoch)",
            )
        })?;
        let (value, opening) = block.opening(CommitmentSlot::EpochBoundary)?;
        if value != boundary.hash() {
            return Err(data_not_available(
                block.slot,
                "the kept epoch boundary does not match the block's commitment",
            ));
        }
        Ok(EpochBoundaryResponse {
            slot: block.slot,
            block_id: block.block_id()?,
            proof: EpochBoundaryProof { boundary, opening },
        })
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        solana_light_client::solana_merkle_tree::commitment_tree::{
            compute_root_from_leaf_hashes, make_inclusion_proof_from_leaf_hashes,
        },
    };

    fn leaf_hash(i: u64) -> Hash {
        hash_leaf(TreeId::BLOCK_HISTORY, &i.to_le_bytes())
    }

    /// The history files across segments: written in batches, read back,
    /// pruned from the front, restarted, and with a record cut short.
    #[test]
    fn test_history_files() {
        use persist::HISTORY_SEGMENT_LEN;
        let directory = solana_ledger::get_tmp_ledger_path!();
        let persistence = Persistence::open(&directory).unwrap();
        let segment_files = || {
            let mut names: Vec<u64> = std::fs::read_dir(directory.join("history"))
                .unwrap()
                .map(|entry| {
                    entry
                        .unwrap()
                        .file_name()
                        .to_str()
                        .unwrap()
                        .parse()
                        .unwrap()
                })
                .collect();
            names.sort_unstable();
            names
        };
        let entry = |i: u64| (i, leaf_hash(i), leaf_hash(i + 1_000_000));

        // A history that starts after five leaves that are not kept.
        let base = AppendOnlyFrontier::from_leaf_hashes(&(0..5).map(leaf_hash).collect::<Vec<_>>())
            .unwrap();
        let total = 2 * HISTORY_SEGMENT_LEN + 100;
        let entries: Vec<_> = (5..5 + total).map(entry).collect();
        let (mut files, loaded) = persistence.load_history();
        assert!(loaded.is_none());
        let mut written = 0usize;
        for (round, batch) in [1usize, 3000, 0, 1500, 4000, usize::MAX]
            .into_iter()
            .enumerate()
        {
            let end = written.saturating_add(batch).min(entries.len());
            let restart = (round == 0).then(|| (7, base.clone()));
            persistence.write_history(&mut files, restart, &entries[written..end], 5);
            written = end;
            assert_eq!(files.next, 5 + written as u64);
        }
        assert_eq!(files.restarts, 7);
        assert_eq!(
            segment_files(),
            [5, 5 + HISTORY_SEGMENT_LEN, 5 + 2 * HISTORY_SEGMENT_LEN]
        );
        let (reloaded, loaded) = persistence.load_history();
        assert_eq!(loaded, Some((base.clone(), entries.clone())));
        assert_eq!(reloaded.next, files.next);
        // The log rebuilt from them has the root of all the leaves.
        let log = HistoryLog::from_parts(base.clone(), entries.clone());
        assert_eq!(log.len(), 5 + total);
        assert_eq!(log.root_of_first(5), Some(base.root(TreeId::BLOCK_HISTORY)));
        assert_eq!(log.root_of_first(5 + total), Some(log.frontier_root()));
        assert_eq!(log.root_of_first(4), None);

        // A record cut short is cut off, and writing goes on after it.
        let last = directory
            .join("history")
            .join((5 + 2 * HISTORY_SEGMENT_LEN).to_string());
        let whole = std::fs::read(&last).unwrap();
        std::fs::write(&last, &whole[..whole.len() - 10]).unwrap();
        let (mut files, loaded) = persistence.load_history();
        assert_eq!(
            loaded,
            Some((base.clone(), entries[..entries.len() - 1].to_vec()))
        );
        persistence.write_history(&mut files, None, &entries[entries.len() - 1..], 5);
        assert_eq!(std::fs::read(&last).unwrap(), whole);

        // Leaves folded into the base: a segment goes once all of it is.
        persistence.write_history(&mut files, None, &[], 5 + HISTORY_SEGMENT_LEN - 1);
        assert_eq!(segment_files().len(), 3);
        persistence.write_history(&mut files, None, &[], 5 + HISTORY_SEGMENT_LEN);
        assert_eq!(
            segment_files(),
            [5 + HISTORY_SEGMENT_LEN, 5 + 2 * HISTORY_SEGMENT_LEN]
        );
        let (_, loaded) = persistence.load_history();
        let (loaded_base, loaded_entries) = loaded.unwrap();
        assert_eq!(loaded_base.leaf_count(), 5 + HISTORY_SEGMENT_LEN);
        assert_eq!(
            loaded_entries,
            entries[HISTORY_SEGMENT_LEN as usize..].to_vec()
        );

        // A segment that does not follow the one before is removed, with
        // those after it.
        std::fs::remove_file(
            directory
                .join("history")
                .join((5 + HISTORY_SEGMENT_LEN).to_string()),
        )
        .unwrap();
        std::fs::write(
            directory
                .join("history")
                .join((5 + HISTORY_SEGMENT_LEN).to_string()),
            &whole,
        )
        .unwrap();
        let (_, loaded) = persistence.load_history();
        assert!(loaded.is_none());
        assert!(segment_files().is_empty());

        // A restart replaces the files.
        let (mut files, _) = persistence.load_history();
        persistence.write_history(&mut files, Some((8, base.clone())), &entries[..10], 5);
        persistence.write_history(
            &mut files,
            Some((9, AppendOnlyFrontier::new())),
            &entries[..3],
            0,
        );
        assert_eq!(segment_files(), [0]);
        let (files, loaded) = persistence.load_history();
        assert_eq!(
            loaded,
            Some((AppendOnlyFrontier::new(), entries[..3].to_vec()))
        );
        assert_eq!(files.next, 3);
    }

    /// Proofs from a log whose first leaves are folded into the base agree
    /// with proofs built from all the leaves, for every kept leaf and every
    /// history length.
    #[test]
    fn test_history_log_proofs_from_frontier() {
        for total in 1..=40u64 {
            let all: Vec<Hash> = (0..total).map(leaf_hash).collect();
            for base_len in 0..total {
                let mut log = HistoryLog {
                    base: AppendOnlyFrontier::from_leaf_hashes(&all[..base_len as usize]).unwrap(),
                    all: AppendOnlyFrontier::from_leaf_hashes(&all).unwrap(),
                    ..HistoryLog::default()
                };
                for i in base_len..total {
                    log.leaf_hashes.push(all[i as usize]);
                    log.entries.push((i, Hash::default(), Hash::default()));
                }
                assert_eq!(
                    log.frontier_root(),
                    compute_root_from_leaf_hashes(TreeId::BLOCK_HISTORY, &all)
                );
                for leaf_count in 1..=total {
                    for index in 0..leaf_count {
                        let proof = log.inclusion_proof(index, leaf_count);
                        if index < base_len {
                            assert_eq!(proof, None);
                            continue;
                        }
                        let expected = make_inclusion_proof_from_leaf_hashes(
                            &all[..leaf_count as usize],
                            index as usize,
                        );
                        assert_eq!(proof, expected, "{total} {base_len} {leaf_count} {index}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_history_log_prune_and_reset() {
        let all: Vec<Hash> = (0..20).map(leaf_hash).collect();
        let mut log = HistoryLog {
            all: AppendOnlyFrontier::from_leaf_hashes(&all).unwrap(),
            ..HistoryLog::default()
        };
        for (i, leaf) in all.iter().enumerate() {
            log.leaf_hashes.push(*leaf);
            log.entries
                .push((i as u64 * 2, Hash::default(), Hash::default()));
        }
        log.prune(7);
        assert_eq!(log.base_len(), 13);
        assert_eq!(log.len(), 20);
        assert_eq!(
            log.frontier_root(),
            compute_root_from_leaf_hashes(TreeId::BLOCK_HISTORY, &all)
        );
        assert_eq!(log.find(24), None);
        assert_eq!(log.find(26).unwrap().0, 13);
        assert_eq!(
            log.inclusion_proof(15, 20),
            make_inclusion_proof_from_leaf_hashes(&all, 15)
        );

        // A reset to the anchor leaf alone keeps that leaf.
        let anchor_id = Hash::new_unique();
        let anchor_leaf = history_leaf(5, &anchor_id, &Hash::default());
        let frontier =
            AppendOnlyFrontier::from_leaf_hashes(&[hash_leaf(TreeId::BLOCK_HISTORY, &anchor_leaf)])
                .unwrap();
        log.reset(frontier.clone(), Some((5, anchor_id)));
        assert_eq!(log.base_len(), 0);
        assert_eq!(log.len(), 1);
        assert_eq!(log.frontier_root(), frontier.root(TreeId::BLOCK_HISTORY));
        assert_eq!(log.find(5).unwrap().0, 0);
        // The next leaf extends both.
        log.push(9, Hash::new_unique(), Hash::new_unique());
        assert_eq!(log.len(), 2);
        assert_eq!(
            log.frontier_root(),
            compute_root_from_leaf_hashes(TreeId::BLOCK_HISTORY, &log.leaf_hashes)
        );
        // Another anchor does not match the frontier: only the frontier.
        log.reset(frontier, Some((5, Hash::new_unique())));
        assert_eq!(log.base_len(), 1);
        assert_eq!(log.find(5), None);
    }
}
