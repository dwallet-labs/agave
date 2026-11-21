//! The trees and proofs of the "Accounts State Root" SIMD.
//!
//! `accounts_state_root` combines a **base tree** over all accounts as they
//! were at the start of an epoch and an **overlay** over the accounts changed
//! since then:
//!
//! ```text
//! base_leaf(a)        = a.pubkey || account_hash(a)                 // 64 bytes
//! base_root(E)        = root(0x07, [base_leaf(a) for a in A])       // sorted by pubkey
//! overlay_entry(a)    = (a.pubkey, account_hash now || account_hash in the base)
//! overlay_root(E, B)  = keyed_root(0x08, O(E, B))
//! accounts_state_root = sha256(0x84 || u64_le(G) || base_root(G) || overlay_root(G, B))
//! ```
//!
//! `root` is the commitment tree of [`crate::commitment_tree`] and
//! `keyed_root` the keyed tree of [`crate::keyed_tree`]. `account_hash` is
//! computed by the runtime (32 zero bytes for an account without lamports).
//!
//! This module provides the base tree ([`BaseTree`]), the overlay value
//! layout ([`OverlayEntry`]), the two ways to derive base leaves without
//! reading account data twice ([`apply_overlay`], [`substitute_overlay`]),
//! and the proofs of current state and of non-existence ([`AccountProof`]),
//! with a standalone verifier that starts from `commitment_root`.

use {
    crate::{
        accounts_delta::{ADDRESS_LEN, Address},
        commitment_tree::{
            ACCOUNTS_STATE_ROOT_PREFIX, InclusionProof, TreeId, hash_leaf, hash_root,
            make_inclusion_proof_from_leaf_hashes, tree_hash_from_leaf_hashes,
        },
        keyed_tree::{KeyedTree, MembershipProof, NonMembershipProof},
    },
    solana_hash::Hash,
    solana_sha256_hasher::hashv,
};

/// `account_hash` of an address without an account.
pub const NO_ACCOUNT_HASH: Hash = Hash::new_from_array([0; 32]);

/// Index of `accounts_state_root` in the commitment vector.
pub const ACCOUNTS_STATE_ROOT_SLOT_INDEX: u64 = 5;

/// Number of slots of the commitment vector.
pub const COMMITMENT_VECTOR_LEN: u64 = 8;

/// Length of a base leaf: `address || account_hash`.
pub const BASE_LEAF_LEN: usize = ADDRESS_LEN + 32;

/// Length of an overlay value: `account_hash now || account_hash in the base`.
pub const OVERLAY_VALUE_LEN: usize = 64;

/// An overlay value: `account_hash now || account_hash in the base`.
pub type OverlayValue = [u8; OVERLAY_VALUE_LEN];

/// The overlay of one generation: a keyed tree with tree id 0x08.
pub type OverlayTree = KeyedTree<OverlayValue>;

/// An empty overlay.
pub fn new_overlay() -> OverlayTree {
    KeyedTree::new(TreeId::STATE_OVERLAY)
}

/// `sha256(0x84 || u64_le(G) || base_root(G) || overlay_root(G, B))`
pub fn accounts_state_root(generation: u64, base_root: &Hash, overlay_root: &Hash) -> Hash {
    hashv(&[
        &[ACCOUNTS_STATE_ROOT_PREFIX],
        &generation.to_le_bytes(),
        base_root.as_ref(),
        overlay_root.as_ref(),
    ])
}

/// One overlay entry: an address changed since the base state, its
/// `account_hash` now and in the base state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OverlayEntry {
    pub address: Address,
    pub now: Hash,
    pub base: Hash,
}

impl OverlayEntry {
    pub fn new(address: Address, now: Hash, base: Hash) -> Self {
        Self { address, now, base }
    }

    /// `account_hash now || account_hash in the base`.
    pub fn value(&self) -> OverlayValue {
        overlay_value(&self.now, &self.base)
    }

    pub fn from_value(address: Address, value: &OverlayValue) -> Self {
        let (now, base) = split_overlay_value(value);
        Self { address, now, base }
    }
}

/// `now || base`
pub fn overlay_value(now: &Hash, base: &Hash) -> OverlayValue {
    let mut value = [0u8; OVERLAY_VALUE_LEN];
    value[..32].copy_from_slice(now.as_ref());
    value[32..].copy_from_slice(base.as_ref());
    value
}

/// `(now, base)` of an overlay value.
pub fn split_overlay_value(value: &OverlayValue) -> (Hash, Hash) {
    (
        Hash::new_from_array(value[..32].try_into().unwrap()),
        Hash::new_from_array(value[32..].try_into().unwrap()),
    )
}

/// Applies the changed accounts of one block to an overlay: an address with
/// an entry gets its first half replaced by `after`; an address without one
/// is inserted with `after || before` (its state before the block is its base
/// state, since it has not changed since). Entries are never removed.
///
/// `changes` are `(address, account_hash before the block, account_hash after
/// the block)` of the block's changed accounts.
pub fn apply_block_changes<'a>(
    overlay: &mut OverlayTree,
    changes: impl IntoIterator<Item = (&'a Address, &'a Hash, &'a Hash)>,
) {
    for (address, before, after) in changes {
        let base = match overlay.get(address) {
            Some(value) => split_overlay_value(value).1,
            None => *before,
        };
        overlay.insert(*address, overlay_value(after, &base));
    }
}

/// One leaf of the base tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BaseLeaf {
    pub address: Address,
    pub account_hash: Hash,
}

impl BaseLeaf {
    pub fn new(address: Address, account_hash: Hash) -> Self {
        Self {
            address,
            account_hash,
        }
    }

    /// `base_leaf(a) = a.pubkey || account_hash(a)`
    pub fn bytes(&self) -> [u8; BASE_LEAF_LEN] {
        let mut leaf = [0u8; BASE_LEAF_LEN];
        leaf[..ADDRESS_LEN].copy_from_slice(&self.address);
        leaf[ADDRESS_LEN..].copy_from_slice(self.account_hash.as_ref());
        leaf
    }

    /// `leaf(0x07, base_leaf(a))`
    pub fn leaf_hash(&self) -> Hash {
        hash_leaf(TreeId::BASE_STATE, &self.bytes())
    }
}

/// Why leaves do not form a base tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseTreeError {
    /// Leaf `index` is not strictly greater than the leaf before it.
    NotSortedOrDuplicate { index: usize },
    /// An address without lamports has no base leaf.
    NoAccount { address: Address },
    /// The stored bytes are not a whole number of leaves.
    Length { bytes: usize },
}

/// Where the leaves of a base tree are held: one `base_leaf` after the
/// other, [`BASE_LEAF_LEN`] bytes each, in address order. A `Vec<u8>` holds
/// them in memory; a node with many accounts maps a file instead.
pub trait BaseLeafStorage: Send + Sync {
    fn bytes(&self) -> &[u8];
}

impl BaseLeafStorage for Vec<u8> {
    fn bytes(&self) -> &[u8] {
        self
    }
}

/// Computes `one(0), .., one(count - 1)` in order, stopping at the first
/// error. The caller chooses how: one after the other
/// ([`sequential_chunks`]), or on a thread pool.
pub type ChunkMap<'a> = dyn Fn(
        usize,
        &(dyn Fn(usize) -> Result<Hash, BaseTreeError> + Sync),
    ) -> Result<Vec<Hash>, BaseTreeError>
    + 'a;

/// A [`ChunkMap`] that computes one chunk after the other.
pub fn sequential_chunks(
    count: usize,
    one: &(dyn Fn(usize) -> Result<Hash, BaseTreeError> + Sync),
) -> Result<Vec<Hash>, BaseTreeError> {
    (0..count).map(one).collect()
}

/// The base tree of one generation: the accounts with lamports of the base
/// state, sorted by address, and their root.
///
/// The leaves are in a [`BaseLeafStorage`] (64 bytes per account). The tree
/// itself keeps only the hashes of some inner nodes in memory
/// ([`ChunkLevels`]: about 32 bytes per [`CHUNK_LEN`] accounts), so that a
/// proof hashes a few thousand nodes whatever the size of the tree.
pub struct BaseTree {
    storage: Box<dyn BaseLeafStorage>,
    len: usize,
    chunks: ChunkLevels,
    root: Hash,
}

impl std::fmt::Debug for BaseTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BaseTree")
            .field("len", &self.len)
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

impl PartialEq for BaseTree {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root && self.storage.bytes() == other.storage.bytes()
    }
}

impl Eq for BaseTree {}

/// Number of leaves, or of nodes of a level, under one stored node.
pub const CHUNK_LEN: usize = 1 << CHUNK_BITS;
const CHUNK_BITS: u32 = 10;

/// Inner nodes of a commitment tree kept to build proofs without hashing
/// every leaf.
///
/// The tree splits `n > 1` leaves at the largest power of two below `n`.
/// So every run of [`CHUNK_LEN`] leaves that starts at a multiple of
/// `CHUNK_LEN` is the subtree of one node, and so is the shorter run at
/// the end; and the tree above those nodes is the same tree over them as
/// leaves. Level 0 holds those nodes, level 1 the same for level 0, and so
/// on until a level has at most `CHUNK_LEN` nodes.
///
/// A path from a leaf to the tree hash is then the path inside its chunk
/// of leaves, the path inside the chunk of each level, and the path over
/// the last level: at most `CHUNK_LEN` hashes per level.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ChunkLevels(Vec<Vec<Hash>>);

impl ChunkLevels {
    /// The levels above level 0, the nodes over each chunk of leaves (empty
    /// for a tree without leaves).
    fn new(level0: Vec<Hash>) -> Self {
        if level0.is_empty() {
            return Self::default();
        }
        let mut levels = vec![level0];
        while let Some(last) = levels.last().filter(|last| last.len() > CHUNK_LEN) {
            let next = last
                .chunks(CHUNK_LEN)
                .map(tree_hash_from_leaf_hashes)
                .collect();
            levels.push(next);
        }
        Self(levels)
    }

    /// `MTH` of the leaves.
    fn tree_hash(&self) -> Hash {
        tree_hash_from_leaf_hashes(self.0.last().map(Vec::as_slice).unwrap_or_default())
    }

    /// The siblings of leaf `index`, from the leaf towards the tree hash.
    /// `index` must be less than `leaf_count`, and `leaf_hashes(range)` the
    /// hashes of the leaves in `range`.
    fn siblings(
        &self,
        leaf_count: usize,
        index: usize,
        leaf_hashes: impl Fn(std::ops::Range<usize>) -> Vec<Hash>,
    ) -> Vec<Hash> {
        // The path of node `position` inside its chunk of a level with
        // `count` nodes.
        let path_in_chunk =
            |count: usize,
             position: usize,
             hashes: &dyn Fn(std::ops::Range<usize>) -> Vec<Hash>| {
                let start = position & !(CHUNK_LEN - 1);
                let end = start.saturating_add(CHUNK_LEN).min(count);
                make_inclusion_proof_from_leaf_hashes(&hashes(start..end), position - start)
                    .expect("the position is in its chunk")
                    .siblings
            };
        let mut siblings = path_in_chunk(leaf_count, index, &leaf_hashes);
        let mut position = index >> CHUNK_BITS;
        let Some((last, levels)) = self.0.split_last() else {
            return siblings;
        };
        for level in levels {
            siblings.extend(path_in_chunk(level.len(), position, &|range| {
                level[range].to_vec()
            }));
            position >>= CHUNK_BITS;
        }
        siblings.extend(
            make_inclusion_proof_from_leaf_hashes(last, position)
                .expect("the position is in the last level")
                .siblings,
        );
        siblings
    }
}

/// The bytes of the leaves in `range`.
fn leaf_bytes(bytes: &[u8], range: std::ops::Range<usize>) -> &[[u8; BASE_LEAF_LEN]] {
    let leaves = bytes.as_chunks::<BASE_LEAF_LEN>().0;
    &leaves[range]
}

fn leaf_from_bytes(bytes: &[u8; BASE_LEAF_LEN]) -> BaseLeaf {
    let (address, account_hash) = bytes.split_at(ADDRESS_LEN);
    BaseLeaf {
        address: address.try_into().expect("an address"),
        account_hash: Hash::new_from_array(account_hash.try_into().expect("a hash")),
    }
}

impl BaseTree {
    /// The base tree of an empty state.
    pub fn empty() -> Self {
        Self::new(Vec::new()).expect("no leaves")
    }

    /// The base tree of `leaves`, held in memory. Checks them as
    /// [`Self::from_storage`] does.
    pub fn new(leaves: Vec<BaseLeaf>) -> Result<Self, BaseTreeError> {
        let mut bytes = Vec::with_capacity(leaves.len().saturating_mul(BASE_LEAF_LEN));
        for leaf in &leaves {
            bytes.extend_from_slice(&leaf.bytes());
        }
        Self::from_storage(Box::new(bytes), &sequential_chunks)
    }

    /// The base tree of the leaves in `storage`. Checks that they are sorted
    /// by address, unique and have accounts, and computes the root: every
    /// leaf is read and hashed once, one chunk of [`CHUNK_LEN`] leaves at a
    /// time, as `map` schedules them.
    pub fn from_storage(
        storage: Box<dyn BaseLeafStorage>,
        map: &ChunkMap,
    ) -> Result<Self, BaseTreeError> {
        let bytes = storage.bytes();
        if !bytes.len().is_multiple_of(BASE_LEAF_LEN) {
            return Err(BaseTreeError::Length { bytes: bytes.len() });
        }
        let len = bytes.len() / BASE_LEAF_LEN;
        let chunk = |chunk: usize| {
            let start = chunk.saturating_mul(CHUNK_LEN);
            let end = start.saturating_add(CHUNK_LEN).min(len);
            // The leaf before the chunk, to check the order across chunks.
            let mut previous = start
                .checked_sub(1)
                .map(|before| &leaf_bytes(bytes, before..start)[0][..ADDRESS_LEN]);
            let mut hashes = Vec::with_capacity(end.saturating_sub(start));
            for (offset, leaf) in leaf_bytes(bytes, start..end).iter().enumerate() {
                let (address, account_hash) = leaf.split_at(ADDRESS_LEN);
                if previous.is_some_and(|previous| previous >= address) {
                    return Err(BaseTreeError::NotSortedOrDuplicate {
                        index: start.saturating_add(offset),
                    });
                }
                if account_hash == NO_ACCOUNT_HASH.as_ref() {
                    return Err(BaseTreeError::NoAccount {
                        address: address.try_into().expect("an address"),
                    });
                }
                hashes.push(hash_leaf(TreeId::BASE_STATE, leaf));
                previous = Some(address);
            }
            Ok(tree_hash_from_leaf_hashes(&hashes))
        };
        let chunks = ChunkLevels::new(map(len.div_ceil(CHUNK_LEN), &chunk)?);
        let root = hash_root(TreeId::BASE_STATE, len as u64, &chunks.tree_hash());
        Ok(Self {
            storage,
            len,
            chunks,
            root,
        })
    }

    /// `base_root`.
    pub fn root(&self) -> Hash {
        self.root
    }

    /// Leaf `index`, which must be less than [`Self::len`].
    pub fn leaf(&self, index: usize) -> BaseLeaf {
        leaf_from_bytes(&leaf_bytes(self.storage.bytes(), index..index.saturating_add(1))[0])
    }

    /// The leaves in address order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = BaseLeaf> + '_ {
        leaf_bytes(self.storage.bytes(), 0..self.len)
            .iter()
            .map(leaf_from_bytes)
    }

    /// A copy of all leaves: for small trees.
    pub fn to_vec(&self) -> Vec<BaseLeaf> {
        self.iter().collect()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn search(&self, address: &Address) -> Result<usize, usize> {
        leaf_bytes(self.storage.bytes(), 0..self.len)
            .binary_search_by(|leaf| leaf[..ADDRESS_LEN].cmp(address.as_slice()))
    }

    /// `account_hash` of `address` in the base state, if it had an account.
    pub fn get(&self, address: &Address) -> Option<Hash> {
        self.search(address)
            .ok()
            .map(|index| self.leaf(index).account_hash)
    }

    fn leaf_hashes(&self, range: std::ops::Range<usize>) -> Vec<Hash> {
        leaf_bytes(self.storage.bytes(), range)
            .iter()
            .map(|leaf| hash_leaf(TreeId::BASE_STATE, leaf))
            .collect()
    }

    /// Proof of leaf `index` (in range), from the stored inner nodes.
    fn leaf_proof(&self, index: usize) -> BaseLeafProof {
        BaseLeafProof {
            leaf: self.leaf(index),
            proof: InclusionProof {
                leaf_index: index as u64,
                leaf_count: self.len as u64,
                siblings: self
                    .chunks
                    .siblings(self.len, index, |range| self.leaf_hashes(range)),
            },
        }
    }

    /// Inclusion proof of the base leaf of `address`. `None` if it had no
    /// account in the base state.
    pub fn inclusion_proof(&self, address: &Address) -> Option<BaseLeafProof> {
        let index = self.search(address).ok()?;
        Some(self.leaf_proof(index))
    }

    /// Proof that `address` had no account in the base state: two adjacent
    /// leaves around it, the first or last leaf, or an empty tree. `None` if
    /// it had one.
    pub fn absence_proof(&self, address: &Address) -> Option<BaseAbsenceProof> {
        let index = self.search(address).err()?;
        if self.len == 0 {
            return Some(BaseAbsenceProof::EmptyTree);
        }
        Some(if index == 0 {
            BaseAbsenceProof::BeforeFirst {
                first: self.leaf_proof(0),
            }
        } else if index == self.len {
            BaseAbsenceProof::AfterLast {
                last: self.leaf_proof(index - 1),
            }
        } else {
            BaseAbsenceProof::Between {
                left: self.leaf_proof(index - 1),
                right: self.leaf_proof(index),
            }
        })
    }
}

/// Why base leaves could not be derived from an overlay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlayMismatch {
    /// The overlay's base half of `address` differs from its hash in the
    /// previous base state (merge).
    Base {
        address: Address,
        expected: Hash,
        found: Hash,
    },
    /// The overlay's first half of `address` differs from its hash in the
    /// scanned state (scan).
    Now {
        address: Address,
        expected: Hash,
        found: Hash,
    },
}

/// Why deriving base leaves stopped: the overlay does not match, or the
/// receiver of the leaves failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeriveError<E> {
    Mismatch(OverlayMismatch),
    Sink(E),
}

/// **By merge**: the leaves of the next base tree, in address order, from
/// the previous base tree `base` and the frozen overlay of its generation as
/// of the last block before the next epoch. Entries with a zero first half
/// are removed, the others replace or insert a leaf. Each leaf is given to
/// `sink` as it is derived; no more than one is held.
///
/// Checks that every entry's base half is its hash in `base` (zero if absent),
/// which holds for an overlay of `base`'s generation.
pub fn apply_overlay_into<E>(
    base: &BaseTree,
    overlay: &OverlayTree,
    mut sink: impl FnMut(BaseLeaf) -> Result<(), E>,
) -> Result<(), DeriveError<E>> {
    let mut base_leaves = base.iter().peekable();
    for (address, value) in overlay.iter() {
        let (now, base_half) = split_overlay_value(value);
        while let Some(leaf) = base_leaves.next_if(|leaf| leaf.address < *address) {
            sink(leaf).map_err(DeriveError::Sink)?;
        }
        let in_base = base_leaves
            .next_if(|leaf| leaf.address == *address)
            .map_or(NO_ACCOUNT_HASH, |leaf| leaf.account_hash);
        if in_base != base_half {
            return Err(DeriveError::Mismatch(OverlayMismatch::Base {
                address: *address,
                expected: in_base,
                found: base_half,
            }));
        }
        if now != NO_ACCOUNT_HASH {
            sink(BaseLeaf::new(*address, now)).map_err(DeriveError::Sink)?;
        }
    }
    for leaf in base_leaves {
        sink(leaf).map_err(DeriveError::Sink)?;
    }
    Ok(())
}

/// [`apply_overlay_into`] a vector: for small trees.
pub fn apply_overlay(
    base: &BaseTree,
    overlay: &OverlayTree,
) -> Result<Vec<BaseLeaf>, OverlayMismatch> {
    let mut leaves = Vec::new();
    let pushed = apply_overlay_into(base, overlay, |leaf| {
        leaves.push(leaf);
        Ok::<(), std::convert::Infallible>(())
    });
    match pushed {
        Ok(()) => Ok(leaves),
        Err(DeriveError::Mismatch(mismatch)) => Err(mismatch),
        Err(DeriveError::Sink(never)) => match never {},
    }
}

/// **By scan**: the leaves of the base tree of a generation, in address
/// order, from the complete account state of a block `X` of that generation
/// and the overlay of the generation as of `X`. An address without an
/// overlay entry has its current state in the base; an address with one has
/// its base hash in the entry.
///
/// The state is given leaf by leaf in address order (accounts with lamports
/// only) to [`Self::push`], then [`Self::finish`]; each derived leaf goes to
/// the sink at once. Checks that every entry's first half is its hash in the
/// state (zero if absent), which holds for an overlay as of the scanned
/// block.
pub struct OverlaySubstitution<'a> {
    entries: std::iter::Peekable<crate::keyed_tree::Iter<'a, OverlayValue>>,
}

impl<'a> OverlaySubstitution<'a> {
    pub fn new(overlay: &'a OverlayTree) -> Self {
        Self {
            entries: overlay.iter().peekable(),
        }
    }

    /// Handles the next overlay entry, whose address has `in_current` in the
    /// scanned state.
    fn entry<E>(
        &mut self,
        in_current: Hash,
        sink: &mut impl FnMut(BaseLeaf) -> Result<(), E>,
    ) -> Result<(), DeriveError<E>> {
        let Some((address, value)) = self.entries.next() else {
            return Ok(());
        };
        let (now, base_half) = split_overlay_value(value);
        if in_current != now {
            return Err(DeriveError::Mismatch(OverlayMismatch::Now {
                address: *address,
                expected: in_current,
                found: now,
            }));
        }
        if base_half != NO_ACCOUNT_HASH {
            sink(BaseLeaf::new(*address, base_half)).map_err(DeriveError::Sink)?;
        }
        Ok(())
    }

    /// The next leaf of the scanned state.
    pub fn push<E>(
        &mut self,
        leaf: BaseLeaf,
        mut sink: impl FnMut(BaseLeaf) -> Result<(), E>,
    ) -> Result<(), DeriveError<E>> {
        // Entries before the leaf have no account in the scanned state.
        while self
            .entries
            .peek()
            .is_some_and(|(address, _)| **address < leaf.address)
        {
            self.entry(NO_ACCOUNT_HASH, &mut sink)?;
        }
        if self
            .entries
            .peek()
            .is_some_and(|(address, _)| **address == leaf.address)
        {
            self.entry(leaf.account_hash, &mut sink)
        } else {
            sink(leaf).map_err(DeriveError::Sink)
        }
    }

    /// After the last leaf of the scanned state.
    pub fn finish<E>(
        mut self,
        mut sink: impl FnMut(BaseLeaf) -> Result<(), E>,
    ) -> Result<(), DeriveError<E>> {
        while self.entries.peek().is_some() {
            self.entry(NO_ACCOUNT_HASH, &mut sink)?;
        }
        Ok(())
    }
}

/// [`OverlaySubstitution`] over a slice, into a vector: for small trees.
pub fn substitute_overlay(
    current: &[BaseLeaf],
    overlay: &OverlayTree,
) -> Result<Vec<BaseLeaf>, OverlayMismatch> {
    let mut leaves = Vec::with_capacity(current.len());
    let mut sink = |leaf| {
        leaves.push(leaf);
        Ok::<(), std::convert::Infallible>(())
    };
    let mut substitution = OverlaySubstitution::new(overlay);
    let pushed = current
        .iter()
        .try_for_each(|leaf| substitution.push(*leaf, &mut sink))
        .and_then(|()| substitution.finish(&mut sink));
    match pushed {
        Ok(()) => Ok(leaves),
        Err(DeriveError::Mismatch(mismatch)) => Err(mismatch),
        Err(DeriveError::Sink(never)) => match never {},
    }
}

/// A base leaf and its inclusion proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseLeafProof {
    pub leaf: BaseLeaf,
    pub proof: InclusionProof,
}

impl BaseLeafProof {
    /// True if `self.leaf` is the leaf at `self.proof.leaf_index` of the base
    /// tree with root `base_root`.
    pub fn verify(&self, base_root: &Hash) -> bool {
        self.proof
            .verify_leaf_hash(TreeId::BASE_STATE, self.leaf.leaf_hash(), base_root)
    }
}

/// Proof that an address has no leaf in a base tree. Sound because leaves are
/// sorted and unique and the leaf count is bound in the root; adjacency is
/// checked from the leaf indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BaseAbsenceProof {
    EmptyTree,
    BeforeFirst {
        first: BaseLeafProof,
    },
    AfterLast {
        last: BaseLeafProof,
    },
    Between {
        left: BaseLeafProof,
        right: BaseLeafProof,
    },
}

impl BaseAbsenceProof {
    /// True if `address` has no leaf in the base tree with root `base_root`.
    pub fn verify(&self, address: &Address, base_root: &Hash) -> bool {
        match self {
            Self::EmptyTree => *base_root == BaseTree::empty().root(),
            Self::BeforeFirst { first } => {
                first.proof.leaf_index == 0
                    && *address < first.leaf.address
                    && first.verify(base_root)
            }
            Self::AfterLast { last } => {
                last.proof.leaf_index.checked_add(1) == Some(last.proof.leaf_count)
                    && last.leaf.address < *address
                    && last.verify(base_root)
            }
            Self::Between { left, right } => {
                left.proof.leaf_count == right.proof.leaf_count
                    && left.proof.leaf_index.checked_add(1) == Some(right.proof.leaf_index)
                    && left.leaf.address < *address
                    && *address < right.leaf.address
                    && left.verify(base_root)
                    && right.verify(base_root)
            }
        }
    }
}

/// Opening of `accounts_state_root` from `commitment_root`: its three
/// components and the inclusion proof of slot 5 of the commitment vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccountsStateOpening {
    pub generation: u64,
    pub base_root: Hash,
    pub overlay_root: Hash,
    pub slot_proof: InclusionProof,
}

impl AccountsStateOpening {
    /// `accounts_state_root` of the opened components.
    pub fn accounts_state_root(&self) -> Hash {
        accounts_state_root(self.generation, &self.base_root, &self.overlay_root)
    }

    /// True if the components hash to slot 5 of the commitment vector with
    /// root `commitment_root`.
    pub fn verify(&self, commitment_root: &Hash) -> bool {
        self.slot_proof.leaf_index == ACCOUNTS_STATE_ROOT_SLOT_INDEX
            && self.slot_proof.leaf_count == COMMITMENT_VECTOR_LEN
            && self.slot_proof.verify(
                TreeId::COMMITMENT_VECTOR,
                self.accounts_state_root().as_ref(),
                commitment_root,
            )
    }
}

/// Where the state of an address is found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AccountLookup {
    /// The address changed since the base state: its overlay entry. The
    /// first half is its `account_hash` now (zero: no account).
    Overlay {
        entry: OverlayEntry,
        proof: MembershipProof,
    },
    /// The address did not change since the base state: overlay
    /// non-membership, and its leaf in the base tree or the absence of one.
    Base {
        overlay_absence: Box<NonMembershipProof<OverlayValue>>,
        base: BaseLookup,
    },
}

/// State of an address in the base tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BaseLookup {
    Present(BaseLeafProof),
    Absent(BaseAbsenceProof),
}

/// Builds the lookup of `address` (prover side). `base` is needed only if
/// `address` has no overlay entry; `None` is returned in that case without
/// it.
pub fn account_lookup(
    overlay: &OverlayTree,
    base: Option<&BaseTree>,
    address: &Address,
) -> Option<AccountLookup> {
    if let Some((value, proof)) = overlay.membership_proof(address) {
        return Some(AccountLookup::Overlay {
            entry: OverlayEntry::from_value(*address, &value),
            proof,
        });
    }
    let overlay_absence = overlay.non_membership_proof(address)?;
    let base = base?;
    let base = match base.inclusion_proof(address) {
        Some(proof) => BaseLookup::Present(proof),
        None => BaseLookup::Absent(base.absence_proof(address)?),
    };
    Some(AccountLookup::Base {
        overlay_absence: Box::new(overlay_absence),
        base,
    })
}

/// A proof of the state of one address at the end of a block, from the
/// block's `commitment_root`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccountProof {
    pub opening: AccountsStateOpening,
    pub lookup: AccountLookup,
}

/// Why an [`AccountProof`] is rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccountProofError {
    /// The opening does not lead to `commitment_root`.
    Opening,
    /// A current-state proof was asked for the zero account hash; use
    /// [`AccountProof::verify_nonexistence`].
    ZeroAccountHash,
    /// The overlay entry is not for the address, does not carry the claimed
    /// hash, or its proof does not verify against `overlay_root`.
    OverlayEntry,
    /// The overlay non-membership proof does not verify for the address.
    OverlayAbsence,
    /// The base leaf is not for the address, does not carry the claimed
    /// hash, or its proof does not verify against `base_root`.
    BaseLeaf,
    /// The base absence proof does not verify for the address, or a base
    /// leaf was given where the account is claimed not to exist.
    BaseAbsence,
}

impl AccountProof {
    /// Verifies that `address` ends the block with the state whose
    /// `account_hash` is `account_hash` (zero: no account), and returns the
    /// generation the proof is bound to. The verifier computes `account_hash`
    /// from the claimed account state.
    pub fn verify(
        &self,
        commitment_root: &Hash,
        address: &Address,
        account_hash: &Hash,
    ) -> Result<u64, AccountProofError> {
        if !self.opening.verify(commitment_root) {
            return Err(AccountProofError::Opening);
        }
        match &self.lookup {
            AccountLookup::Overlay { entry, proof } => {
                if entry.address != *address
                    || entry.now != *account_hash
                    || !proof.verify(
                        TreeId::STATE_OVERLAY,
                        address,
                        &entry.value(),
                        &self.opening.overlay_root,
                    )
                {
                    return Err(AccountProofError::OverlayEntry);
                }
            }
            AccountLookup::Base {
                overlay_absence,
                base,
            } => {
                // Both halves are required: a base leaf alone proves only the
                // state at the start of the generation.
                if !overlay_absence.verify(
                    TreeId::STATE_OVERLAY,
                    address,
                    &self.opening.overlay_root,
                ) {
                    return Err(AccountProofError::OverlayAbsence);
                }
                match base {
                    BaseLookup::Present(leaf) => {
                        if *account_hash == NO_ACCOUNT_HASH
                            || leaf.leaf.address != *address
                            || leaf.leaf.account_hash != *account_hash
                            || !leaf.verify(&self.opening.base_root)
                        {
                            return Err(AccountProofError::BaseLeaf);
                        }
                    }
                    BaseLookup::Absent(absence) => {
                        if *account_hash != NO_ACCOUNT_HASH
                            || !absence.verify(address, &self.opening.base_root)
                        {
                            return Err(AccountProofError::BaseAbsence);
                        }
                    }
                }
            }
        }
        Ok(self.opening.generation)
    }

    /// **Current state**: `address` has exactly the state whose (non-zero)
    /// `account_hash` is given, at the end of the block.
    pub fn verify_current_state(
        &self,
        commitment_root: &Hash,
        address: &Address,
        account_hash: &Hash,
    ) -> Result<u64, AccountProofError> {
        if *account_hash == NO_ACCOUNT_HASH {
            return Err(AccountProofError::ZeroAccountHash);
        }
        self.verify(commitment_root, address, account_hash)
    }

    /// **Non-existence**: `address` has no account at the end of the block.
    pub fn verify_nonexistence(
        &self,
        commitment_root: &Hash,
        address: &Address,
    ) -> Result<u64, AccountProofError> {
        self.verify(commitment_root, address, &NO_ACCOUNT_HASH)
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        crate::commitment_tree::{compute_root, make_inclusion_proof},
        solana_sha256_hasher::hash,
    };

    fn hex_hash(s: &str) -> Hash {
        Hash::new_from_array(<[u8; 32]>::try_from(hex::decode(s).unwrap()).unwrap())
    }

    fn h(byte: u8) -> Hash {
        Hash::new_from_array([byte; 32])
    }

    fn addr(byte: u8) -> Address {
        [byte; 32]
    }

    fn leaf_hashes(leaves: &[BaseLeaf]) -> Vec<Hash> {
        leaves.iter().map(BaseLeaf::leaf_hash).collect()
    }

    fn base_of(entries: &[(u8, u8)]) -> BaseTree {
        BaseTree::new(
            entries
                .iter()
                .map(|(a, v)| BaseLeaf::new(addr(*a), h(*v)))
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn test_base_tree_root_matches_commitment_tree() {
        let tree = base_of(&[(1, 10), (3, 30), (5, 50)]);
        let leaves: Vec<_> = tree.iter().map(|leaf| leaf.bytes()).collect();
        assert_eq!(tree.root(), compute_root(TreeId::BASE_STATE, &leaves));
        assert_eq!(
            BaseTree::empty().root(),
            compute_root::<[u8; 64]>(TreeId::BASE_STATE, &[])
        );
        assert_eq!(tree.get(&addr(3)), Some(h(30)));
        assert_eq!(tree.get(&addr(4)), None);
    }

    /// The stored inner nodes give the root and the proofs that hashing
    /// every leaf gives, for trees of one chunk, of chunks that are not
    /// full, and of one, two and three levels.
    #[test]
    fn test_base_tree_chunks_match_full_tree() {
        use crate::commitment_tree::compute_root_from_leaf_hashes;
        let two_levels = CHUNK_LEN * CHUNK_LEN;
        for len in [
            0,
            1,
            2,
            3,
            CHUNK_LEN - 1,
            CHUNK_LEN,
            CHUNK_LEN + 1,
            2 * CHUNK_LEN - 1,
            2 * CHUNK_LEN,
            2 * CHUNK_LEN + 1,
            5 * CHUNK_LEN + 77,
            two_levels + 3 * CHUNK_LEN + 5,
        ] {
            let leaves: Vec<BaseLeaf> = (0..len as u64)
                .map(|i| {
                    let mut address = [0u8; ADDRESS_LEN];
                    address[..8].copy_from_slice(&i.to_be_bytes());
                    BaseLeaf::new(address, hash(&i.to_le_bytes()))
                })
                .collect();
            let tree = BaseTree::new(leaves.clone()).unwrap();
            let hashes = leaf_hashes(&leaves);
            assert_eq!(
                tree.root(),
                compute_root_from_leaf_hashes(TreeId::BASE_STATE, &hashes),
                "{len} leaves"
            );
            let last = len.saturating_sub(1);
            let mut indexes = vec![
                0,
                1,
                CHUNK_LEN - 1,
                CHUNK_LEN,
                len / 2,
                two_levels.saturating_sub(1),
                two_levels,
                last.saturating_sub(CHUNK_LEN),
                last.saturating_sub(1),
                last,
            ];
            indexes.retain(|index| *index < len);
            indexes.sort_unstable();
            indexes.dedup();
            for index in indexes {
                let proof = tree.inclusion_proof(&leaves[index].address).unwrap();
                // The full-tree proof hashes every leaf, so for the large
                // tree the proof is only verified against the root, which
                // was compared above.
                if len <= two_levels {
                    assert_eq!(
                        proof.proof,
                        make_inclusion_proof_from_leaf_hashes(&hashes, index).unwrap(),
                        "leaf {index} of {len}"
                    );
                }
                assert_eq!(proof.proof.leaf_index, index as u64);
                assert!(proof.verify(&tree.root()));
            }
        }
    }

    #[test]
    fn test_base_tree_rejects_bad_leaves() {
        assert_eq!(
            BaseTree::new(vec![
                BaseLeaf::new(addr(2), h(1)),
                BaseLeaf::new(addr(1), h(1))
            ]),
            Err(BaseTreeError::NotSortedOrDuplicate { index: 1 })
        );
        assert_eq!(
            BaseTree::new(vec![
                BaseLeaf::new(addr(1), h(1)),
                BaseLeaf::new(addr(1), h(2))
            ]),
            Err(BaseTreeError::NotSortedOrDuplicate { index: 1 })
        );
        assert_eq!(
            BaseTree::new(vec![BaseLeaf::new(addr(1), NO_ACCOUNT_HASH)]),
            Err(BaseTreeError::NoAccount { address: addr(1) })
        );
    }

    /// A tree over stored bytes: chunks computed in any order give the same
    /// tree, and the order of leaves is checked across chunks.
    #[test]
    fn test_base_tree_from_storage() {
        let leaves: Vec<BaseLeaf> = (0..3 * CHUNK_LEN as u64 + 9)
            .map(|i| {
                let mut address = [0u8; ADDRESS_LEN];
                address[..8].copy_from_slice(&i.to_be_bytes());
                BaseLeaf::new(address, hash(&i.to_le_bytes()))
            })
            .collect();
        let bytes: Vec<u8> = leaves.iter().flat_map(BaseLeaf::bytes).collect();
        let expected = BaseTree::new(leaves.clone()).unwrap();
        // Last chunk first, as a thread pool may.
        let backwards: &ChunkMap = &|count, one| {
            let mut hashes = (0..count).rev().map(one).collect::<Result<Vec<_>, _>>()?;
            hashes.reverse();
            Ok(hashes)
        };
        let tree = BaseTree::from_storage(Box::new(bytes.clone()), backwards).unwrap();
        assert_eq!(tree, expected);
        assert_eq!(tree.len(), leaves.len());
        assert_eq!(tree.to_vec(), leaves);
        assert_eq!(tree.leaf(CHUNK_LEN), leaves[CHUNK_LEN]);
        assert_eq!(
            tree.get(&leaves[2 * CHUNK_LEN].address),
            Some(leaves[2 * CHUNK_LEN].account_hash)
        );

        // The first leaf of a chunk is not after the last of the one before.
        let mut swapped = bytes.clone();
        let (before, after) = swapped.split_at_mut(CHUNK_LEN * BASE_LEAF_LEN);
        before[(CHUNK_LEN - 1) * BASE_LEAF_LEN..].swap_with_slice(&mut after[..BASE_LEAF_LEN]);
        assert_eq!(
            BaseTree::from_storage(Box::new(swapped), &sequential_chunks).unwrap_err(),
            BaseTreeError::NotSortedOrDuplicate { index: CHUNK_LEN }
        );
        // Not a whole number of leaves.
        let mut short = bytes;
        short.pop();
        assert_eq!(
            BaseTree::from_storage(Box::new(short.clone()), &sequential_chunks).unwrap_err(),
            BaseTreeError::Length { bytes: short.len() }
        );
    }

    #[test]
    fn test_base_proofs() {
        let tree = base_of(&[(2, 20), (4, 40), (6, 60), (8, 80), (10, 100)]);
        let root = tree.root();
        for leaf in &tree.to_vec() {
            let proof = tree.inclusion_proof(&leaf.address).unwrap();
            assert!(proof.verify(&root));
            assert!(tree.absence_proof(&leaf.address).is_none());
        }
        for a in [0u8, 1, 3, 5, 7, 9, 11, 0xff] {
            let proof = tree.absence_proof(&addr(a)).unwrap();
            assert!(proof.verify(&addr(a), &root), "{a}");
            // Not valid for a present address.
            assert!(!proof.verify(&addr(4), &root));
        }
        // Non-adjacent leaves do not prove absence.
        let left = tree.inclusion_proof(&addr(2)).unwrap();
        let right = tree.inclusion_proof(&addr(6)).unwrap();
        let bad = BaseAbsenceProof::Between { left, right };
        assert!(!bad.verify(&addr(3), &root));
        assert!(!bad.verify(&addr(5), &root));
        // Not first.
        let bad = BaseAbsenceProof::BeforeFirst {
            first: tree.inclusion_proof(&addr(4)).unwrap(),
        };
        assert!(!bad.verify(&addr(3), &root));
        // Not last.
        let bad = BaseAbsenceProof::AfterLast {
            last: tree.inclusion_proof(&addr(8)).unwrap(),
        };
        assert!(!bad.verify(&addr(9), &root));
        assert!(!BaseAbsenceProof::EmptyTree.verify(&addr(1), &root));
        let empty = BaseTree::empty();
        assert_eq!(
            empty.absence_proof(&addr(1)),
            Some(BaseAbsenceProof::EmptyTree)
        );
        assert!(BaseAbsenceProof::EmptyTree.verify(&addr(1), &empty.root()));
    }

    #[test]
    fn test_apply_block_changes() {
        let mut overlay = new_overlay();
        // Block 1: insert a (existed in base) and b (new).
        apply_block_changes(
            &mut overlay,
            [
                (&addr(1), &h(10), &h(11)),
                (&addr(2), &NO_ACCOUNT_HASH, &h(21)),
            ],
        );
        let after_block1 = overlay.clone();
        // Block 2: a restored to its base state, b deleted, c (in base)
        // deleted.
        apply_block_changes(
            &mut overlay,
            [
                (&addr(1), &h(11), &h(10)),
                (&addr(2), &h(21), &NO_ACCOUNT_HASH),
                (&addr(3), &h(30), &NO_ACCOUNT_HASH),
            ],
        );
        let entries: Vec<_> = overlay
            .iter()
            .map(|(a, v)| OverlayEntry::from_value(*a, v))
            .collect();
        assert_eq!(
            entries,
            vec![
                OverlayEntry::new(addr(1), h(10), h(10)),
                OverlayEntry::new(addr(2), NO_ACCOUNT_HASH, NO_ACCOUNT_HASH),
                OverlayEntry::new(addr(3), NO_ACCOUNT_HASH, h(30)),
            ]
        );
        // The overlay of block 1 is unchanged.
        assert_eq!(after_block1.len(), 2);
        assert_eq!(
            OverlayEntry::from_value(addr(2), after_block1.get(&addr(2)).unwrap()),
            OverlayEntry::new(addr(2), h(21), NO_ACCOUNT_HASH)
        );
    }

    #[test]
    fn test_merge_and_substitute() {
        // Base: 1, 3, 5, 7.
        let base = base_of(&[(1, 10), (3, 30), (5, 50), (7, 70)]);
        let mut overlay = new_overlay();
        apply_block_changes(
            &mut overlay,
            [
                (&addr(3), &h(30), &h(31)),           // modified
                (&addr(5), &h(50), &NO_ACCOUNT_HASH), // closed
                (&addr(0), &NO_ACCOUNT_HASH, &h(1)),  // created first
                (&addr(9), &NO_ACCOUNT_HASH, &h(91)), // created last
                (&addr(4), &NO_ACCOUNT_HASH, &h(41)), // created and closed
                (&addr(7), &h(70), &h(71)),           // restored below
            ],
        );
        apply_block_changes(
            &mut overlay,
            [
                (&addr(4), &h(41), &NO_ACCOUNT_HASH),
                (&addr(7), &h(71), &h(70)),
            ],
        );
        let next = apply_overlay(&base, &overlay).unwrap();
        let expected = base_of(&[(0, 1), (1, 10), (3, 31), (7, 70), (9, 91)]);
        assert_eq!(next, expected.to_vec());

        // By scan of the state after these changes, the previous base comes
        // back.
        let rebuilt = substitute_overlay(&expected.to_vec(), &overlay).unwrap();
        assert_eq!(rebuilt, base.to_vec());

        // A base half that does not match the base is detected.
        let mut bad = overlay.clone();
        bad.insert(addr(1), overlay_value(&h(12), &h(99)));
        assert_eq!(
            apply_overlay(&base, &bad),
            Err(OverlayMismatch::Base {
                address: addr(1),
                expected: h(10),
                found: h(99)
            })
        );
        // A first half that does not match the scanned state is detected.
        assert_eq!(
            substitute_overlay(&base.to_vec(), &overlay),
            Err(OverlayMismatch::Now {
                address: addr(0),
                expected: NO_ACCOUNT_HASH,
                found: h(1)
            })
        );
    }

    fn commitment_vector_with(slot5: Hash) -> Vec<[u8; 32]> {
        let mut vector = vec![[0u8; 32]; 8];
        vector[0] = [0xaa; 32];
        vector[5] = slot5.to_bytes();
        vector
    }

    /// Current state and non-existence, through the overlay and through the
    /// base, from `commitment_root`, and their rejection when tampered.
    #[test]
    fn test_account_proofs() {
        let base = base_of(&[(1, 10), (3, 30), (5, 50), (7, 70)]);
        let mut overlay = new_overlay();
        apply_block_changes(
            &mut overlay,
            [
                (&addr(3), &h(30), &h(31)),
                (&addr(5), &h(50), &NO_ACCOUNT_HASH),
                (&addr(9), &NO_ACCOUNT_HASH, &h(91)),
            ],
        );
        let generation = 4;
        let state_root = accounts_state_root(generation, &base.root(), &overlay.root());
        let vector = commitment_vector_with(state_root);
        let commitment_root = compute_root(TreeId::COMMITMENT_VECTOR, &vector);
        let opening = AccountsStateOpening {
            generation,
            base_root: base.root(),
            overlay_root: overlay.root(),
            slot_proof: make_inclusion_proof(TreeId::COMMITMENT_VECTOR, &vector, 5).unwrap(),
        };
        assert!(opening.verify(&commitment_root));
        let prove = |a: u8| AccountProof {
            opening: opening.clone(),
            lookup: account_lookup(&overlay, Some(&base), &addr(a)).unwrap(),
        };

        // (address, current account hash): 1 untouched, 3 modified, 5 closed,
        // 7 untouched, 9 created, 2 and 0xff never existed.
        let current = [
            (1u8, h(10)),
            (3, h(31)),
            (5, NO_ACCOUNT_HASH),
            (7, h(70)),
            (9, h(91)),
            (2, NO_ACCOUNT_HASH),
            (0xff, NO_ACCOUNT_HASH),
        ];
        for (a, hash) in current {
            let proof = prove(a);
            assert_eq!(
                proof.verify(&commitment_root, &addr(a), &hash),
                Ok(4),
                "{a}"
            );
            if hash == NO_ACCOUNT_HASH {
                assert_eq!(proof.verify_nonexistence(&commitment_root, &addr(a)), Ok(4));
                assert_eq!(
                    proof.verify_current_state(&commitment_root, &addr(a), &hash),
                    Err(AccountProofError::ZeroAccountHash)
                );
                assert!(proof.verify(&commitment_root, &addr(a), &h(1)).is_err());
            } else {
                assert_eq!(
                    proof.verify_current_state(&commitment_root, &addr(a), &hash),
                    Ok(4)
                );
                assert!(
                    proof
                        .verify_nonexistence(&commitment_root, &addr(a))
                        .is_err()
                );
                assert!(proof.verify(&commitment_root, &addr(a), &h(0x42)).is_err());
            }
            // Bound to the address (a proof of absence may also show the
            // absence of other addresses, which is correct).
            if hash != NO_ACCOUNT_HASH {
                assert!(
                    proof
                        .verify(&commitment_root, &addr(a ^ 0x40), &hash)
                        .is_err()
                );
            }
            // Bound to the commitment root.
            assert_eq!(
                proof.verify(&h(0x77), &addr(a), &hash),
                Err(AccountProofError::Opening)
            );
            // Bound to the generation.
            let mut bad = proof.clone();
            bad.opening.generation += 1;
            assert_eq!(
                bad.verify(&commitment_root, &addr(a), &hash),
                Err(AccountProofError::Opening)
            );
            // The opening must be slot 5.
            let mut bad = proof.clone();
            bad.opening.slot_proof =
                make_inclusion_proof(TreeId::COMMITMENT_VECTOR, &vector, 0).unwrap();
            assert!(bad.verify(&commitment_root, &addr(a), &hash).is_err());
        }

        // The untouched address 1 proven by its base leaf alone, with an
        // overlay non-membership proof of another address, fails.
        let AccountLookup::Base { base: base1, .. } = prove(1).lookup else {
            panic!("address 1 is not in the overlay");
        };
        // The modified address 3 with its stale base leaf: overlay
        // non-membership cannot be shown for it.
        let stale = AccountProof {
            opening: opening.clone(),
            lookup: AccountLookup::Base {
                overlay_absence: Box::new(overlay.non_membership_proof(&addr(2)).unwrap()),
                base: BaseLookup::Present(base.inclusion_proof(&addr(3)).unwrap()),
            },
        };
        assert_eq!(
            stale.verify(&commitment_root, &addr(3), &h(30)),
            Err(AccountProofError::OverlayAbsence)
        );
        // Closed address 5: its base leaf says it exists, the overlay says it
        // does not; the base path is rejected.
        let stale = AccountProof {
            opening: opening.clone(),
            lookup: AccountLookup::Base {
                overlay_absence: Box::new(overlay.non_membership_proof(&addr(4)).unwrap()),
                base: BaseLookup::Present(base.inclusion_proof(&addr(5)).unwrap()),
            },
        };
        assert!(stale.verify(&commitment_root, &addr(5), &h(50)).is_err());
        // A base leaf of another address.
        let bad = AccountProof {
            opening: opening.clone(),
            lookup: AccountLookup::Base {
                overlay_absence: Box::new(overlay.non_membership_proof(&addr(7)).unwrap()),
                base: base1,
            },
        };
        assert_eq!(
            bad.verify(&commitment_root, &addr(7), &h(10)),
            Err(AccountProofError::BaseLeaf)
        );
        // Tampered overlay entry.
        let AccountLookup::Overlay { entry, proof } = prove(3).lookup else {
            panic!("address 3 is in the overlay");
        };
        let bad = AccountProof {
            opening: opening.clone(),
            lookup: AccountLookup::Overlay {
                entry: OverlayEntry {
                    now: h(32),
                    ..entry
                },
                proof: proof.clone(),
            },
        };
        assert_eq!(
            bad.verify(&commitment_root, &addr(3), &h(32)),
            Err(AccountProofError::OverlayEntry)
        );
        let bad = AccountProof {
            opening: opening.clone(),
            lookup: AccountLookup::Overlay {
                entry: OverlayEntry {
                    base: h(33),
                    ..entry
                },
                proof,
            },
        };
        assert_eq!(
            bad.verify(&commitment_root, &addr(3), &h(31)),
            Err(AccountProofError::OverlayEntry)
        );
        // Without the base tree, only overlay lookups can be built.
        assert!(account_lookup(&overlay, None, &addr(3)).is_some());
        assert!(account_lookup(&overlay, None, &addr(1)).is_none());
    }

    // Vectors computed by the independent Python implementation in
    // `scratch-accounts-state-root/vectors.py`.
    #[test]
    fn test_vectors() {
        // Base tree over addresses sha256("a" || i), account hash
        // sha256("h" || i), sorted by address.
        let mut leaves: Vec<_> = (0..5u8)
            .map(|i| BaseLeaf::new(hash(&[b'a', i]).to_bytes(), hash(&[b'h', i])))
            .collect();
        leaves.sort();
        let base = BaseTree::new(leaves.clone()).unwrap();
        assert_eq!(
            base.root(),
            hex_hash("8eca8c6cfa461b69fca92fd6b4d2aa309e2bff89653ea7ab0ef291edc09f3852")
        );
        assert_eq!(
            BaseTree::empty().root(),
            hex_hash("c39074923925f824434a1ce94ea594b7bc74365b1a1370a67e17318c4601475e")
        );

        // Overlay across three blocks: insert, second change, restore, and
        // deletion, over that base.
        let a = |i: u8| hash(&[b'a', i]).to_bytes();
        let hh = |i: u8| hash(&[b'h', i]);
        let new_hash = |i: u8| hash(&[b'n', i]);
        let mut overlay = new_overlay();
        // Block 1: 0 modified, 5 created.
        apply_block_changes(
            &mut overlay,
            [
                (&a(0), &hh(0), &new_hash(0)),
                (&a(5), &NO_ACCOUNT_HASH, &new_hash(5)),
            ],
        );
        assert_eq!(
            overlay.root(),
            hex_hash("8850f916a02670ea03ea66008d4fe3819081d8d2fae05f7094b32102e4e50945")
        );
        // Block 2: 0 changed again, 1 modified.
        apply_block_changes(
            &mut overlay,
            [
                (&a(0), &new_hash(0), &new_hash(10)),
                (&a(1), &hh(1), &new_hash(1)),
            ],
        );
        assert_eq!(
            overlay.root(),
            hex_hash("ab595bbde48ba9420936878580270c37377f9841e751cf8c6fd40e5cc9f60632")
        );
        // Block 3: 1 restored, 2 deleted.
        apply_block_changes(
            &mut overlay,
            [
                (&a(1), &new_hash(1), &hh(1)),
                (&a(2), &hh(2), &NO_ACCOUNT_HASH),
            ],
        );
        assert_eq!(
            overlay.root(),
            hex_hash("49add68b4d1bd926f67177c6aad426cc51d637cdf78afb6f0b1025927e10c9ed")
        );
        assert_eq!(
            accounts_state_root(7, &base.root(), &overlay.root()),
            hex_hash("9f38edb5a980de7fd1865095e08b85592e694bd3913029114c09956c082b00a4")
        );
        // The next base tree by merge.
        let next = BaseTree::new(apply_overlay(&base, &overlay).unwrap()).unwrap();
        assert_eq!(
            next.root(),
            hex_hash("b54d35be8ff06c889ba9c8281aa685a13a686f5df189d9bb6178fb11fcf4dd42")
        );
    }
}
