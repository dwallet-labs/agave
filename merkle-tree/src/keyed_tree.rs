//! The keyed tree of the "Accounts State Root" SIMD: a binary radix
//! (PATRICIA) tree over 32-byte keys, used for the state overlay.
//!
//! Bits of a key are numbered from 0, the most significant bit of the first
//! byte, to 255. For a set `S` of `(key, value)` entries with distinct keys:
//!
//! ```text
//! kleaf(tree_id, key, value) = sha256(0x02 || tree_id || key || value)
//! kbranch(bit, left, right)  = sha256(0x03 || u16_le(bit) || left || right)
//!
//! KTH(tree_id, {})           = 32 zero bytes
//! KTH(tree_id, {(k, v)})     = kleaf(tree_id, k, v)
//! KTH(tree_id, S), |S| > 1   = kbranch(b, KTH(tree_id, S0), KTH(tree_id, S1))
//!     where b  = the first bit at which the keys of S do not all agree
//!           S0 = the entries of S whose key has bit b equal to 0
//!           S1 = the entries of S whose key has bit b equal to 1
//!
//! keyed_root(tree_id, S) = sha256(0x81 || tree_id || u64_le(|S|) || KTH(tree_id, S))
//! ```
//!
//! The tree is a function of the set alone.
//!
//! # Persistence
//!
//! [`KeyedTree`] is a persistent structure: nodes are reference counted and
//! never modified once shared. Cloning a tree is `O(1)`; an insertion copies
//! only the nodes on one path (`O(depth)`) and shares every other node with
//! the trees it was cloned from. A node that is not shared (refcount 1) is
//! updated in place. Node hashes are computed lazily, once per node, so a
//! batch of insertions followed by [`KeyedTree::root`] hashes every changed
//! node once.
//!
//! # Proofs
//!
//! A [`MembershipProof`] is the entry count and the `(bit, sibling)` pairs from
//! the leaf to the top. A [`NonMembershipProof`] for a key `k` is the
//! membership proof of the entry `(k', v')` that `k` reaches when descending by
//! its bits, with `k != k'`, or the entry count 0 for an empty tree.
//! The verifiers enforce the checks required by the SIMD's Security
//! Considerations: bit numbers strictly decrease from the leaf towards the
//! top, each sibling is placed on the side opposite to the bit of the proven
//! key, and for non-membership the two keys agree at every bit of the path.

use {
    crate::commitment_tree::{KEYED_BRANCH_PREFIX, KEYED_LEAF_PREFIX, KEYED_ROOT_PREFIX, TreeId},
    solana_hash::Hash,
    solana_sha256_hasher::hashv,
    std::sync::{Arc, OnceLock},
};

/// Length of a key, in bytes.
pub const KEY_LEN: usize = 32;

/// Number of bits of a key.
pub const KEY_BITS: u16 = (KEY_LEN * 8) as u16;

/// A 32-byte key, ordered lexicographically.
pub type Key = [u8; KEY_LEN];

/// `KTH` of an empty set: 32 zero bytes.
pub const EMPTY_KEYED_TREE_HASH: Hash = Hash::new_from_array([0; 32]);

/// Bit `bit` of `key` (bit 0 is the most significant bit of the first byte).
pub fn key_bit(key: &Key, bit: u16) -> u8 {
    debug_assert!(bit < KEY_BITS);
    (key[usize::from(bit / 8)] >> (7 - (bit % 8))) & 1
}

/// First bit at which `a` and `b` differ, `None` if they are equal.
pub fn first_differing_bit(a: &Key, b: &Key) -> Option<u16> {
    a.iter().zip(b).enumerate().find_map(|(i, (x, y))| {
        let diff = x ^ y;
        (diff != 0).then(|| (i * 8) as u16 + diff.leading_zeros() as u16)
    })
}

/// `kleaf(tree_id, key, value) = sha256(0x02 || tree_id || key || value)`
pub fn hash_keyed_leaf(tree_id: TreeId, key: &Key, value: &[u8]) -> Hash {
    hashv(&[&[KEYED_LEAF_PREFIX, tree_id.0], key, value])
}

/// `kbranch(bit, left, right) = sha256(0x03 || u16_le(bit) || left || right)`
pub fn hash_keyed_branch(bit: u16, left: &Hash, right: &Hash) -> Hash {
    hashv(&[
        &[KEYED_BRANCH_PREFIX],
        &bit.to_le_bytes(),
        left.as_ref(),
        right.as_ref(),
    ])
}

/// `keyed_root = sha256(0x81 || tree_id || u64_le(|S|) || KTH(tree_id, S))`
pub fn hash_keyed_root(tree_id: TreeId, entry_count: u64, tree_hash: &Hash) -> Hash {
    hashv(&[
        &[KEYED_ROOT_PREFIX, tree_id.0],
        &entry_count.to_le_bytes(),
        tree_hash.as_ref(),
    ])
}

/// `keyed_root` of the empty set.
pub fn empty_keyed_root(tree_id: TreeId) -> Hash {
    hash_keyed_root(tree_id, 0, &EMPTY_KEYED_TREE_HASH)
}

/// Value of a keyed tree entry: hashed as raw bytes.
pub trait KeyedValue: AsRef<[u8]> + Clone + Eq {}
impl<T: AsRef<[u8]> + Clone + Eq> KeyedValue for T {}

enum NodeKind<V> {
    Leaf {
        key: Key,
        value: V,
    },
    /// `children[0]` holds the keys with bit `bit` equal to 0. Every key below
    /// a branch agrees with every other at all bits before `bit`.
    Branch {
        bit: u16,
        children: [Arc<Node<V>>; 2],
    },
}

struct Node<V> {
    kind: NodeKind<V>,
    /// `KTH` of the subtree, computed on first use. Reset whenever the node is
    /// modified, which only happens while it is not shared.
    hash: OnceLock<Hash>,
}

impl<V: Clone> Clone for Node<V> {
    fn clone(&self) -> Self {
        let kind = match &self.kind {
            NodeKind::Leaf { key, value } => NodeKind::Leaf {
                key: *key,
                value: value.clone(),
            },
            NodeKind::Branch { bit, children } => NodeKind::Branch {
                bit: *bit,
                children: children.clone(),
            },
        };
        Self {
            kind,
            hash: self.hash.clone(),
        }
    }
}

impl<V: KeyedValue> Node<V> {
    fn leaf(key: Key, value: V) -> Arc<Self> {
        Arc::new(Self {
            kind: NodeKind::Leaf { key, value },
            hash: OnceLock::new(),
        })
    }

    fn hash(&self, tree_id: TreeId) -> Hash {
        *self.hash.get_or_init(|| match &self.kind {
            NodeKind::Leaf { key, value } => hash_keyed_leaf(tree_id, key, value.as_ref()),
            NodeKind::Branch { bit, children } => {
                hash_keyed_branch(*bit, &children[0].hash(tree_id), &children[1].hash(tree_id))
            }
        })
    }
}

/// `(bit, sibling)` pairs of a path.
type Path = Vec<(u16, Hash)>;

/// Persistent keyed tree (see the module documentation).
pub struct KeyedTree<V> {
    tree_id: TreeId,
    root: Option<Arc<Node<V>>>,
    len: u64,
}

impl<V> Clone for KeyedTree<V> {
    /// `O(1)`: the clone shares every node.
    fn clone(&self) -> Self {
        Self {
            tree_id: self.tree_id,
            root: self.root.clone(),
            len: self.len,
        }
    }
}

impl<V: KeyedValue + std::fmt::Debug> std::fmt::Debug for KeyedTree<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyedTree")
            .field("tree_id", &self.tree_id)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

/// Two trees are equal if they have the same tree id and the same entries.
impl<V: KeyedValue> PartialEq for KeyedTree<V> {
    fn eq(&self, other: &Self) -> bool {
        self.tree_id == other.tree_id
            && self.len == other.len
            && (match (&self.root, &other.root) {
                (Some(a), Some(b)) if Arc::ptr_eq(a, b) => true,
                _ => self.iter().eq(other.iter()),
            })
    }
}
impl<V: KeyedValue> Eq for KeyedTree<V> {}

impl<V: KeyedValue> KeyedTree<V> {
    /// An empty tree.
    pub fn new(tree_id: TreeId) -> Self {
        Self {
            tree_id,
            root: None,
            len: 0,
        }
    }

    /// Builds a tree from entries with distinct keys, in any order. Returns
    /// the duplicated key if a key appears twice.
    pub fn from_entries(
        tree_id: TreeId,
        entries: impl IntoIterator<Item = (Key, V)>,
    ) -> Result<Self, Key> {
        let mut tree = Self::new(tree_id);
        for (key, value) in entries {
            if tree.insert(key, value).is_some() {
                return Err(key);
            }
        }
        Ok(tree)
    }

    pub fn tree_id(&self) -> TreeId {
        self.tree_id
    }

    /// Number of entries.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The leaf reached by descending from the top by the bits of `key`.
    fn descend(&self, key: &Key) -> Option<(&Key, &V)> {
        let mut node = self.root.as_deref()?;
        loop {
            match &node.kind {
                NodeKind::Leaf { key, value } => return Some((key, value)),
                NodeKind::Branch { bit, children } => {
                    node = &children[usize::from(key_bit(key, *bit))];
                }
            }
        }
    }

    /// The value of `key`, if it has an entry.
    pub fn get(&self, key: &Key) -> Option<&V> {
        self.descend(key)
            .and_then(|(found, value)| (found == key).then_some(value))
    }

    pub fn contains_key(&self, key: &Key) -> bool {
        self.get(key).is_some()
    }

    /// Inserts or replaces the entry of `key`. Returns the previous value.
    /// Copies the nodes on the path of `key` that are shared with other
    /// trees; nodes owned by this tree alone are updated in place.
    pub fn insert(&mut self, key: Key, value: V) -> Option<V> {
        let Some((found, _)) = self.descend(&key) else {
            self.root = Some(Node::leaf(key, value));
            self.len = 1;
            return None;
        };
        let split_bit = first_differing_bit(found, &key);
        let root = self.root.as_mut().expect("tree is not empty");
        match split_bit {
            // `make_mut` copies the node only if it is shared.
            None => Some(Self::replace_value(Arc::make_mut(root), &key, value)),
            Some(split_bit) => {
                Self::insert_new(root, key, value, split_bit);
                self.len += 1;
                None
            }
        }
    }

    /// Replaces the value of the existing `key` below `node` (made unique by
    /// the caller).
    fn replace_value(node: &mut Node<V>, key: &Key, value: V) -> V {
        node.hash = OnceLock::new();
        match &mut node.kind {
            NodeKind::Leaf { value: old, .. } => std::mem::replace(old, value),
            NodeKind::Branch { bit, children } => {
                let child = &mut children[usize::from(key_bit(key, *bit))];
                Self::replace_value(Arc::make_mut(child), key, value)
            }
        }
    }

    /// Inserts a key that is not in the tree and that first differs at
    /// `split_bit` from the key its bits lead to. The new branch goes above
    /// the first node on that path whose bit is greater than `split_bit`.
    fn insert_new(node: &mut Arc<Node<V>>, key: Key, value: V, split_bit: u16) {
        if let NodeKind::Branch { bit, .. } = &node.kind
            && *bit < split_bit
        {
            let node = Arc::make_mut(node);
            node.hash = OnceLock::new();
            let NodeKind::Branch { bit, children } = &mut node.kind else {
                unreachable!("checked above")
            };
            let child = &mut children[usize::from(key_bit(&key, *bit))];
            Self::insert_new(child, key, value, split_bit);
            return;
        }
        let existing = node.clone();
        let leaf = Node::leaf(key, value);
        let children = if key_bit(&key, split_bit) == 0 {
            [leaf, existing]
        } else {
            [existing, leaf]
        };
        *node = Arc::new(Node {
            kind: NodeKind::Branch {
                bit: split_bit,
                children,
            },
            hash: OnceLock::new(),
        });
    }

    /// `KTH(tree_id, S)`.
    pub fn tree_hash(&self) -> Hash {
        self.root
            .as_ref()
            .map_or(EMPTY_KEYED_TREE_HASH, |root| root.hash(self.tree_id))
    }

    /// `keyed_root(tree_id, S)`.
    pub fn root(&self) -> Hash {
        hash_keyed_root(self.tree_id, self.len, &self.tree_hash())
    }

    /// The entries in ascending key order.
    pub fn iter(&self) -> Iter<'_, V> {
        Iter {
            stack: self.root.as_deref().into_iter().collect(),
        }
    }

    /// The `(bit, sibling)` pairs from the top to the leaf reached by `key`,
    /// and that leaf.
    fn path(&self, key: &Key) -> Option<(Path, &Key, &V)> {
        let mut node = self.root.as_deref()?;
        let mut path = Vec::new();
        loop {
            match &node.kind {
                NodeKind::Leaf { key, value } => return Some((path, key, value)),
                NodeKind::Branch { bit, children } => {
                    let side = usize::from(key_bit(key, *bit));
                    path.push((*bit, children[1 - side].hash(self.tree_id)));
                    node = &children[side];
                }
            }
        }
    }

    /// Membership proof of `key`, `None` if it has no entry.
    pub fn membership_proof(&self, key: &Key) -> Option<(V, MembershipProof)> {
        let (mut path, found, value) = self.path(key)?;
        if found != key {
            return None;
        }
        path.reverse();
        Some((
            value.clone(),
            MembershipProof {
                entry_count: self.len,
                path,
            },
        ))
    }

    /// Non-membership proof of `key`, `None` if it has an entry.
    pub fn non_membership_proof(&self, key: &Key) -> Option<NonMembershipProof<V>> {
        let Some((mut path, found, value)) = self.path(key) else {
            return Some(NonMembershipProof::EmptyTree);
        };
        if found == key {
            return None;
        }
        path.reverse();
        Some(NonMembershipProof::Neighbor {
            key: *found,
            value: value.clone(),
            proof: MembershipProof {
                entry_count: self.len,
                path,
            },
        })
    }
}

/// In-order iterator over the entries of a [`KeyedTree`] (ascending keys).
pub struct Iter<'a, V> {
    stack: Vec<&'a Node<V>>,
}

impl<'a, V> Iterator for Iter<'a, V> {
    type Item = (&'a Key, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let node = self.stack.pop()?;
            match &node.kind {
                NodeKind::Leaf { key, value } => return Some((key, value)),
                NodeKind::Branch { children, .. } => {
                    self.stack.push(&children[1]);
                    self.stack.push(&children[0]);
                }
            }
        }
    }
}

/// Membership proof of one entry: the entry count and the `(bit, sibling)`
/// pairs from the leaf to the top.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MembershipProof {
    pub entry_count: u64,
    pub path: Vec<(u16, Hash)>,
}

impl MembershipProof {
    /// Recomputes `keyed_root` from the entry `(key, value)`. Returns `None`
    /// if the proof is malformed: bit numbers out of range or not strictly
    /// decreasing from the leaf towards the top, an entry count of zero, or a
    /// path longer than a tree with that many entries can have.
    pub fn root_from_entry(&self, tree_id: TreeId, key: &Key, value: &[u8]) -> Option<Hash> {
        // A tree with `n` entries has `n - 1` branches.
        if self.entry_count == 0 || self.path.len() as u64 >= self.entry_count {
            return None;
        }
        let mut hash = hash_keyed_leaf(tree_id, key, value);
        let mut previous_bit = KEY_BITS;
        for (bit, sibling) in &self.path {
            if *bit >= previous_bit {
                return None;
            }
            previous_bit = *bit;
            hash = if key_bit(key, *bit) == 0 {
                hash_keyed_branch(*bit, &hash, sibling)
            } else {
                hash_keyed_branch(*bit, sibling, &hash)
            };
        }
        Some(hash_keyed_root(tree_id, self.entry_count, &hash))
    }

    /// True if `(key, value)` is an entry of the tree with root `root`.
    pub fn verify(&self, tree_id: TreeId, key: &Key, value: &[u8], root: &Hash) -> bool {
        self.root_from_entry(tree_id, key, value)
            .is_some_and(|computed| computed == *root)
    }
}

/// Proof that a key has no entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NonMembershipProof<V> {
    /// The tree is empty (entry count 0).
    EmptyTree,
    /// The entry that the key reaches when descending by its bits, which has
    /// a different key.
    Neighbor {
        key: Key,
        value: V,
        proof: MembershipProof,
    },
}

impl<V: AsRef<[u8]>> NonMembershipProof<V> {
    /// True if `key` has no entry in the tree with root `root`.
    pub fn verify(&self, tree_id: TreeId, key: &Key, root: &Hash) -> bool {
        match self {
            Self::EmptyTree => *root == empty_keyed_root(tree_id),
            Self::Neighbor {
                key: neighbor,
                value,
                proof,
            } => {
                neighbor != key
                    && proof.path.iter().all(|(bit, _)| {
                        *bit < KEY_BITS && key_bit(key, *bit) == key_bit(neighbor, *bit)
                    })
                    && proof.verify(tree_id, neighbor, value.as_ref(), root)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use {super::*, solana_sha256_hasher::hash};

    const TID: TreeId = TreeId(0x08);

    fn hex_hash(s: &str) -> Hash {
        Hash::new_from_array(<[u8; 32]>::try_from(hex::decode(s).unwrap()).unwrap())
    }

    /// Recursive `KTH` over a set, written from the definition, independently
    /// of the tree under test.
    fn reference_kth(tree_id: TreeId, entries: &[(Key, Vec<u8>)]) -> Hash {
        match entries {
            [] => Hash::new_from_array([0; 32]),
            [(k, v)] => hashv(&[&[0x02, tree_id.0], k, v]),
            _ => {
                let bit = (0..256u16)
                    .find(|b| {
                        let first = key_bit(&entries[0].0, *b);
                        entries.iter().any(|(k, _)| key_bit(k, *b) != first)
                    })
                    .unwrap();
                let (s0, s1): (Vec<_>, Vec<_>) = entries
                    .iter()
                    .cloned()
                    .partition(|(k, _)| key_bit(k, bit) == 0);
                hashv(&[
                    &[0x03],
                    &bit.to_le_bytes(),
                    reference_kth(tree_id, &s0).as_ref(),
                    reference_kth(tree_id, &s1).as_ref(),
                ])
            }
        }
    }

    fn reference_root(tree_id: TreeId, entries: &[(Key, Vec<u8>)]) -> Hash {
        hashv(&[
            &[0x81, tree_id.0],
            &(entries.len() as u64).to_le_bytes(),
            reference_kth(tree_id, entries).as_ref(),
        ])
    }

    /// Deterministic pseudo-random bytes (SHA-256 in counter mode).
    struct Prng(u64);
    impl Prng {
        fn key(&mut self) -> Key {
            self.0 += 1;
            hash(&self.0.to_le_bytes()).to_bytes()
        }
        fn below(&mut self, n: u64) -> u64 {
            u64::from_le_bytes(self.key()[..8].try_into().unwrap()) % n
        }
    }

    fn tree_of(entries: &[(Key, Vec<u8>)]) -> KeyedTree<Vec<u8>> {
        KeyedTree::from_entries(TID, entries.iter().cloned()).unwrap()
    }

    fn check_proofs(tree: &KeyedTree<Vec<u8>>, entries: &[(Key, Vec<u8>)], absent: &[Key]) {
        let root = tree.root();
        for (k, v) in entries {
            let (value, proof) = tree.membership_proof(k).unwrap();
            assert_eq!(&value, v);
            assert!(proof.verify(TID, k, v, &root));
            assert!(!proof.verify(TID, k, b"other", &root));
            assert!(!proof.verify(TreeId(0x07), k, v, &root));
            assert!(tree.non_membership_proof(k).is_none());
        }
        for k in absent {
            assert!(tree.membership_proof(k).is_none());
            let proof = tree.non_membership_proof(k).unwrap();
            assert!(proof.verify(TID, k, &root), "{k:?}");
            // The neighbor's own key is not proven absent by this proof.
            if let NonMembershipProof::Neighbor { key, .. } = &proof {
                assert!(!proof.verify(TID, key, &root));
            }
        }
    }

    #[test]
    fn test_key_bits() {
        let mut key = [0u8; 32];
        key[0] = 0b1000_0000;
        key[31] = 1;
        assert_eq!(key_bit(&key, 0), 1);
        assert_eq!(key_bit(&key, 1), 0);
        assert_eq!(key_bit(&key, 255), 1);
        assert_eq!(first_differing_bit(&key, &key), None);
        let mut other = key;
        other[31] = 0;
        assert_eq!(first_differing_bit(&key, &other), Some(255));
        other = [0; 32];
        assert_eq!(first_differing_bit(&key, &other), Some(0));
        other = key;
        other[2] ^= 0b0001_0000;
        assert_eq!(first_differing_bit(&key, &other), Some(19));
    }

    #[test]
    fn test_empty() {
        let tree = KeyedTree::<Vec<u8>>::new(TID);
        assert!(tree.is_empty());
        assert_eq!(tree.tree_hash(), EMPTY_KEYED_TREE_HASH);
        assert_eq!(tree.root(), reference_root(TID, &[]));
        assert_eq!(tree.root(), empty_keyed_root(TID));
        assert_ne!(tree.root(), empty_keyed_root(TreeId(0x07)));
        let key = [5; 32];
        assert!(tree.membership_proof(&key).is_none());
        let proof = tree.non_membership_proof(&key).unwrap();
        assert_eq!(proof, NonMembershipProof::EmptyTree);
        assert!(proof.verify(TID, &key, &tree.root()));
        // An "empty tree" proof does not verify against a non-empty tree.
        let one = tree_of(&[([1; 32], vec![1])]);
        assert!(!proof.verify(TID, &key, &one.root()));
        assert_eq!(tree.iter().count(), 0);
    }

    #[test]
    fn test_single() {
        let entries = [([7; 32], vec![1, 2, 3])];
        let tree = tree_of(&entries);
        assert_eq!(tree.len(), 1);
        assert_eq!(
            tree.tree_hash(),
            hashv(&[&[0x02, 0x08], &[7; 32], &[1, 2, 3]])
        );
        assert_eq!(tree.root(), reference_root(TID, &entries));
        let (_, proof) = tree.membership_proof(&[7; 32]).unwrap();
        assert!(proof.path.is_empty());
        assert_eq!(proof.entry_count, 1);
        check_proofs(&tree, &entries, &[[0; 32], [8; 32], [0xff; 32]]);
    }

    #[test]
    fn test_two_and_update() {
        let a = [0u8; 32];
        let mut b = [0u8; 32];
        b[0] = 0x80;
        let mut tree = tree_of(&[(a, vec![1]), (b, vec![2])]);
        let expected = hashv(&[
            &[0x03],
            &0u16.to_le_bytes(),
            hashv(&[&[0x02, 0x08], &a, &[1]]).as_ref(),
            hashv(&[&[0x02, 0x08], &b, &[2]]).as_ref(),
        ]);
        assert_eq!(tree.tree_hash(), expected);
        let before = tree.clone();
        assert_eq!(tree.insert(a, vec![3]), Some(vec![1]));
        assert_eq!(tree.len(), 2);
        assert_eq!(
            tree.root(),
            reference_root(TID, &[(a, vec![3]), (b, vec![2])])
        );
        // The clone is unaffected.
        assert_eq!(before.get(&a), Some(&vec![1]));
        assert_eq!(
            before.root(),
            reference_root(TID, &[(a, vec![1]), (b, vec![2])])
        );
    }

    /// Keys that share long prefixes (only the last bits differ), and keys
    /// that differ in the first bit, in one tree.
    #[test]
    fn test_long_shared_prefixes() {
        let mut entries = Vec::new();
        for last in [0u8, 1, 2, 3, 0x80, 0xff] {
            let mut key = [0xab; 32];
            key[31] = last;
            entries.push((key, vec![last]));
        }
        let mut key = [0xab; 32];
        key[0] = 0x2b;
        entries.push((key, vec![9]));
        let mut key = [0xab; 32];
        key[16] = 0;
        entries.push((key, vec![10]));
        let tree = tree_of(&entries);
        assert_eq!(tree.root(), reference_root(TID, &entries));
        let mut absent = Vec::new();
        for last in [4u8, 5, 0x81, 0xfe] {
            let mut key = [0xab; 32];
            key[31] = last;
            absent.push(key);
        }
        absent.push([0; 32]);
        absent.push([0xff; 32]);
        check_proofs(&tree, &entries, &absent);
        // The path of the keys that differ only in the last two bits has a
        // branch per distinct key, not per shared bit.
        let mut k = [0xab; 32];
        k[31] = 1;
        let (_, proof) = tree.membership_proof(&k).unwrap();
        assert!(proof.path.len() < entries.len());
        assert_eq!(proof.path[0].0, 255);
    }

    #[test]
    fn test_random_sets_against_reference() {
        let mut prng = Prng(0);
        for n in [0usize, 1, 2, 3, 4, 5, 7, 8, 16, 33, 100, 257] {
            let mut entries: Vec<(Key, Vec<u8>)> = (0..n)
                .map(|i| (prng.key(), vec![i as u8; 1 + i % 3]))
                .collect();
            let tree = tree_of(&entries);
            assert_eq!(tree.len(), n as u64);
            assert_eq!(tree.root(), reference_root(TID, &entries), "n = {n}");
            // Iteration is in ascending key order.
            entries.sort();
            let iterated: Vec<_> = tree.iter().map(|(k, v)| (*k, v.clone())).collect();
            assert_eq!(iterated, entries);
            let absent: Vec<_> = (0..20).map(|_| prng.key()).collect();
            check_proofs(&tree, &entries, &absent);
        }
    }

    /// Insertion order does not matter; updates and structural sharing keep
    /// every version correct.
    #[test]
    fn test_persistence_and_order_independence() {
        let mut prng = Prng(1000);
        let keys: Vec<Key> = (0..200).map(|_| prng.key()).collect();
        let mut tree = KeyedTree::new(TID);
        let mut versions = Vec::new();
        let mut model: Vec<(Key, Vec<u8>)> = Vec::new();
        for round in 0..10u8 {
            for _ in 0..30 {
                let key = keys[prng.below(keys.len() as u64) as usize];
                let value = vec![round, prng.below(256) as u8];
                tree.insert(key, value.clone());
                match model.iter_mut().find(|(k, _)| *k == key) {
                    Some(entry) => entry.1 = value,
                    None => model.push((key, value)),
                }
            }
            // Hash only every other round, so that some rounds update nodes
            // whose hash was never computed.
            if round % 2 == 0 {
                assert_eq!(tree.root(), reference_root(TID, &model));
            }
            versions.push((tree.clone(), model.clone()));
        }
        for (version, model) in &versions {
            assert_eq!(version.root(), reference_root(TID, model));
            assert_eq!(version.len(), model.len() as u64);
            let mut shuffled = model.clone();
            shuffled.reverse();
            assert_eq!(tree_of(&shuffled).root(), version.root());
            assert_eq!(tree_of(&shuffled), *version);
        }
    }

    #[test]
    fn test_from_entries_rejects_duplicates() {
        let err = KeyedTree::from_entries(TID, [([1; 32], vec![1]), ([1; 32], vec![2])]);
        assert_eq!(err.unwrap_err(), [1; 32]);
    }

    #[test]
    fn test_proof_tampering() {
        let mut prng = Prng(77);
        let entries: Vec<(Key, Vec<u8>)> = (0..50).map(|i| (prng.key(), vec![i])).collect();
        let tree = tree_of(&entries);
        let root = tree.root();
        let (key, value) = &entries[17];
        let (_, proof) = tree.membership_proof(key).unwrap();
        assert!(proof.path.len() >= 2);
        assert!(proof.verify(TID, key, value, &root));

        // Wrong entry count.
        for count in [0, proof.entry_count - 1, proof.entry_count + 1] {
            let mut bad = proof.clone();
            bad.entry_count = count;
            assert!(!bad.verify(TID, key, value, &root));
        }
        // Path longer than the count allows.
        let mut bad = proof.clone();
        bad.entry_count = bad.path.len() as u64;
        assert!(!bad.verify(TID, key, value, &root));
        // Tampered sibling or bit.
        let mut bad = proof.clone();
        bad.path[0].1 = Hash::new_from_array([1; 32]);
        assert!(!bad.verify(TID, key, value, &root));
        let mut bad = proof.clone();
        bad.path[0].0 += 1;
        assert!(!bad.verify(TID, key, value, &root));
        // Missing or extra pair.
        let mut bad = proof.clone();
        bad.path.pop();
        assert!(!bad.verify(TID, key, value, &root));
        let mut bad = proof.clone();
        bad.path.push((0, Hash::default()));
        assert!(!bad.verify(TID, key, value, &root));
        // Bits not strictly decreasing towards the top.
        let mut bad = proof.clone();
        bad.path.swap(0, 1);
        assert!(bad.root_from_entry(TID, key, value).is_none());
        let mut bad = proof.clone();
        bad.path[1].0 = bad.path[0].0;
        assert!(bad.root_from_entry(TID, key, value).is_none());
        let mut bad = proof.clone();
        bad.path[0].0 = KEY_BITS;
        assert!(bad.root_from_entry(TID, key, value).is_none());
        // The sibling placed on the wrong side: flipping the key's bit at the
        // lowest branch gives another key, which is not an entry.
        let mut flipped = *key;
        let bit = proof.path[0].0;
        flipped[usize::from(bit / 8)] ^= 0x80 >> (bit % 8);
        assert!(!proof.verify(TID, &flipped, value, &root));

        // Non-membership: the absent key must agree with the neighbor at every
        // bit of the path.
        let absent = prng.key();
        let NonMembershipProof::Neighbor {
            key: neighbor,
            value: neighbor_value,
            proof: neighbor_proof,
        } = tree.non_membership_proof(&absent).unwrap()
        else {
            panic!("tree is not empty");
        };
        let good = NonMembershipProof::Neighbor {
            key: neighbor,
            value: neighbor_value.clone(),
            proof: neighbor_proof.clone(),
        };
        assert!(good.verify(TID, &absent, &root));
        // A valid membership proof of some other entry, not on the absent
        // key's path, does not prove absence.
        let other = entries
            .iter()
            .find(|(k, _)| {
                k != &neighbor && {
                    let (_, p) = tree.membership_proof(k).unwrap();
                    p.path
                        .iter()
                        .any(|(b, _)| key_bit(k, *b) != key_bit(&absent, *b))
                }
            })
            .unwrap();
        let (_, other_proof) = tree.membership_proof(&other.0).unwrap();
        let wrong = NonMembershipProof::Neighbor {
            key: other.0,
            value: other.1.clone(),
            proof: other_proof,
        };
        assert!(!wrong.verify(TID, &absent, &root));
        // A present key is not proven absent by its own membership proof.
        let (_, own) = tree.membership_proof(key).unwrap();
        let own = NonMembershipProof::Neighbor {
            key: *key,
            value: value.clone(),
            proof: own,
        };
        assert!(!own.verify(TID, key, &root));
        // Tampered neighbor value.
        let bad = NonMembershipProof::Neighbor {
            key: neighbor,
            value: vec![0xee],
            proof: neighbor_proof,
        };
        assert!(!bad.verify(TID, &absent, &root));
        // An empty-tree claim against a non-empty tree.
        assert!(!NonMembershipProof::<Vec<u8>>::EmptyTree.verify(TID, &absent, &root));
    }

    // Vectors computed by the independent Python implementation in
    // `scratch-accounts-state-root/vectors.py` (hashlib.sha256, written from the
    // SIMD text), tree id 0x08, 64-byte values.
    fn vector_entries(n: usize) -> Vec<(Key, Vec<u8>)> {
        (0..n)
            .map(|i| {
                let key = hash(&[b'k', i as u8]).to_bytes();
                let value: Vec<u8> = (0..64).map(|j| (i * 64 + j) as u8).collect();
                (key, value)
            })
            .collect()
    }

    fn prefix_entries() -> Vec<(Key, Vec<u8>)> {
        [0x00u8, 0x01, 0x02, 0x03, 0x80, 0xfe]
            .iter()
            .map(|last| {
                let mut key = [0x5a; 32];
                key[31] = *last;
                (key, vec![*last; 64])
            })
            .collect()
    }

    #[test]
    fn test_vectors() {
        let expected = [
            (
                0,
                "bbc8d6bd7a636cb119fe36567bd69c36659b5a63f64d494515dda07077272b26",
            ),
            (
                1,
                "8dc7b8045256cda86665af942e7ea44d50ab497a52c5ba6cf1354dd07fdf3c48",
            ),
            (
                2,
                "3446944ec915be6b0ec392f7a79c78b6a07542378beaad34f398b802a642caba",
            ),
            (
                3,
                "b34147c775bd0957cc417b37aa9f379be2bc436d690d0c4b9802119b40da914c",
            ),
            (
                10,
                "3fd340bc95a2cc803d52533f6141125eace539e2007647b8373a7feb7fcf8716",
            ),
        ];
        for (n, root) in expected {
            let tree = tree_of(&vector_entries(n));
            assert_eq!(tree.root(), hex_hash(root), "n = {n}");
        }
        let tree = tree_of(&prefix_entries());
        assert_eq!(
            tree.root(),
            hex_hash("b62b5e127fa957cad917e51c1d02b67cfbaef7f66f0cdbe136d716a3fc39f4ae")
        );
        // Membership proof of entry 3 of the 10-entry tree.
        let entries = vector_entries(10);
        let tree = tree_of(&entries);
        let (_, proof) = tree.membership_proof(&entries[3].0).unwrap();
        let expected_path: Vec<(u16, Hash)> = PY_PROOF_PATH
            .iter()
            .map(|(bit, sibling)| (*bit, hex_hash(sibling)))
            .collect();
        assert_eq!(proof.path, expected_path);
    }

    const PY_PROOF_PATH: &[(u16, &str)] = &[
        (
            4,
            "94a62941b221638f1cb27d177e2cad103c926aff8be4c05cb9180be5b5bda065",
        ),
        (
            1,
            "c84753ca0456b5ca8395138d3015f745bef62c6638c0cb611ecda97f2f942288",
        ),
        (
            0,
            "6d1a6c389d57354ba83631a0048ef166c1706af4b9d0a0efa2aadffeb0b77689",
        ),
    ];
}
