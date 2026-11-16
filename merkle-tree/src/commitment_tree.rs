//! The commitment tree of the "Block Commitment Root" SIMD.
//!
//! All trees defined by the block commitment root proposal and its companions
//! (transaction receipts, accounts delta, epoch validator set, ...) use this
//! single construction over SHA-256. It follows RFC 6962 (no node
//! duplication, so the shape of a tree is a function of its leaf count alone)
//! and adds a one-byte [`TreeId`] to every leaf and to the root for domain
//! separation between trees:
//!
//! ```text
//! leaf(tree_id, d)        = sha256(0x00 || tree_id || d)
//! MTH(tree_id, [])        = 32 zero bytes
//! MTH(tree_id, [d_0])     = leaf(tree_id, d_0)
//! MTH(tree_id, D), n > 1  = sha256(0x01 || MTH(tree_id, D[0..k]) || MTH(tree_id, D[k..n]))
//!                           where k is the largest power of two less than n
//! root(tree_id, D)        = sha256(0x80 || tree_id || u64_le(n) || MTH(tree_id, D))
//! ```
//!
//! This module provides:
//! - the hash primitives ([`hash_leaf`], [`hash_node`], [`hash_root`]),
//! - root computation over a full list of leaves ([`compute_root`],
//!   [`compute_root_from_leaf_hashes`], [`tree_hash_from_leaf_hashes`]),
//! - inclusion proofs ([`make_inclusion_proof`], [`InclusionProof`]), as in
//!   RFC 6962 section 2.1.1 and verified as in RFC 9162 section 2.1.3.2,
//! - an append-only accumulator ([`AppendOnlyFrontier`]) that keeps only the
//!   leaf count and the roots of the complete subtrees along the right edge
//!   (at most 64 hashes), as used by the block history tree.

use {solana_hash::Hash, solana_sha256_hasher::hashv};

/// First byte of the hash input of a commitment tree leaf.
pub const LEAF_PREFIX: u8 = 0x00;
/// First byte of the hash input of a commitment tree inner node.
pub const NODE_PREFIX: u8 = 0x01;
/// First byte of the hash input of a commitment tree root.
pub const ROOT_PREFIX: u8 = 0x80;
/// First byte of the `validator_set_root` hash (Epoch Validator Set Root SIMD).
/// Reserved here so that all prefixes are defined in one place.
pub const VALIDATOR_SET_ROOT_PREFIX: u8 = 0x82;
/// First byte of the `epoch_boundary` hash (Epoch Boundary Commitment SIMD).
/// Reserved here so that all prefixes are defined in one place.
pub const EPOCH_BOUNDARY_PREFIX: u8 = 0x83;

/// `MTH` of a tree with no leaves: 32 zero bytes.
pub const EMPTY_TREE_HASH: Hash = Hash::new_from_array([0; 32]);

/// Maximum number of subtree roots held by an [`AppendOnlyFrontier`]: one per
/// bit of the `u64` leaf count.
pub const MAX_FRONTIER_LEN: usize = u64::BITS as usize;

/// One-byte domain separator of a commitment tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TreeId(pub u8);

impl TreeId {
    /// The 8-slot commitment vector (Block Commitment Root).
    pub const COMMITMENT_VECTOR: Self = Self(0x00);
    /// The block history tree (Block Commitment Root, slot 0).
    pub const BLOCK_HISTORY: Self = Self(0x01);
    /// Transaction receipts (Transaction Receipt Tree).
    pub const TRANSACTION_RECEIPTS: Self = Self(0x02);
    /// Instruction trace (Transaction Receipt Tree).
    pub const INSTRUCTION_TRACE: Self = Self(0x03);
    /// Events (Transaction Receipt Tree).
    pub const EVENTS: Self = Self(0x04);
    /// Accounts delta (Accounts Delta Root).
    pub const ACCOUNTS_DELTA: Self = Self(0x05);
    /// Validator set (Epoch Validator Set Root).
    pub const VALIDATOR_SET: Self = Self(0x06);
}

/// `leaf(tree_id, data) = sha256(0x00 || tree_id || data)`
pub fn hash_leaf(tree_id: TreeId, data: &[u8]) -> Hash {
    hashv(&[&[LEAF_PREFIX, tree_id.0], data])
}

/// `node(left, right) = sha256(0x01 || left || right)`
pub fn hash_node(left: &Hash, right: &Hash) -> Hash {
    hashv(&[&[NODE_PREFIX], left.as_ref(), right.as_ref()])
}

/// `root(tree_id, D) = sha256(0x80 || tree_id || u64_le(n) || MTH(tree_id, D))`
pub fn hash_root(tree_id: TreeId, leaf_count: u64, tree_hash: &Hash) -> Hash {
    hashv(&[
        &[ROOT_PREFIX, tree_id.0],
        &leaf_count.to_le_bytes(),
        tree_hash.as_ref(),
    ])
}

/// Largest power of two strictly less than `n`. Requires `n > 1`.
fn split_point(n: usize) -> usize {
    debug_assert!(n > 1);
    let bits = usize::BITS - (n - 1).leading_zeros();
    1usize << (bits - 1)
}

/// `MTH` over leaves that are already hashed with [`hash_leaf`].
pub fn tree_hash_from_leaf_hashes(leaf_hashes: &[Hash]) -> Hash {
    match leaf_hashes {
        [] => EMPTY_TREE_HASH,
        [leaf] => *leaf,
        _ => {
            let k = split_point(leaf_hashes.len());
            hash_node(
                &tree_hash_from_leaf_hashes(&leaf_hashes[..k]),
                &tree_hash_from_leaf_hashes(&leaf_hashes[k..]),
            )
        }
    }
}

/// Hashes every leaf with [`hash_leaf`].
pub fn hash_leaves<T: AsRef<[u8]>>(tree_id: TreeId, leaves: &[T]) -> Vec<Hash> {
    leaves
        .iter()
        .map(|leaf| hash_leaf(tree_id, leaf.as_ref()))
        .collect()
}

/// `root(tree_id, D)` over leaves that are already hashed with [`hash_leaf`].
pub fn compute_root_from_leaf_hashes(tree_id: TreeId, leaf_hashes: &[Hash]) -> Hash {
    hash_root(
        tree_id,
        leaf_hashes.len() as u64,
        &tree_hash_from_leaf_hashes(leaf_hashes),
    )
}

/// `root(tree_id, D)` over raw leaf data.
pub fn compute_root<T: AsRef<[u8]>>(tree_id: TreeId, leaves: &[T]) -> Hash {
    compute_root_from_leaf_hashes(tree_id, &hash_leaves(tree_id, leaves))
}

/// Inclusion proof of one leaf, as in RFC 6962 section 2.1.1.
///
/// `siblings` are ordered from the leaf towards the tree hash (`MTH`). The
/// leaf count is part of the proof because it is bound in the root and fixes
/// the shape of the tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InclusionProof {
    pub leaf_index: u64,
    pub leaf_count: u64,
    pub siblings: Vec<Hash>,
}

impl InclusionProof {
    /// Recomputes `MTH` from the hash of the proven leaf, following
    /// RFC 9162 section 2.1.3.2. Returns `None` if the proof is malformed for
    /// its `leaf_index` and `leaf_count`.
    pub fn tree_hash_from_leaf_hash(&self, leaf_hash: Hash) -> Option<Hash> {
        if self.leaf_index >= self.leaf_count {
            return None;
        }
        let mut f_n = self.leaf_index;
        let mut s_n = self.leaf_count - 1;
        let mut r = leaf_hash;
        for sibling in &self.siblings {
            if s_n == 0 {
                return None;
            }
            if f_n & 1 == 1 || f_n == s_n {
                r = hash_node(sibling, &r);
                if f_n & 1 == 0 {
                    while f_n & 1 == 0 && f_n != 0 {
                        f_n >>= 1;
                        s_n >>= 1;
                    }
                }
            } else {
                r = hash_node(&r, sibling);
            }
            f_n >>= 1;
            s_n >>= 1;
        }
        (s_n == 0).then_some(r)
    }

    /// Recomputes the root from the hash of the proven leaf.
    pub fn root_from_leaf_hash(&self, tree_id: TreeId, leaf_hash: Hash) -> Option<Hash> {
        self.tree_hash_from_leaf_hash(leaf_hash)
            .map(|tree_hash| hash_root(tree_id, self.leaf_count, &tree_hash))
    }

    /// Returns true if `leaf_data` is the leaf at `leaf_index` of the tree
    /// with root `root`.
    pub fn verify(&self, tree_id: TreeId, leaf_data: &[u8], root: &Hash) -> bool {
        self.verify_leaf_hash(tree_id, hash_leaf(tree_id, leaf_data), root)
    }

    /// Like [`Self::verify`] for a leaf that is already hashed.
    pub fn verify_leaf_hash(&self, tree_id: TreeId, leaf_hash: Hash, root: &Hash) -> bool {
        self.root_from_leaf_hash(tree_id, leaf_hash)
            .is_some_and(|computed| computed == *root)
    }
}

fn inclusion_path(leaf_index: usize, leaf_hashes: &[Hash], siblings: &mut Vec<Hash>) {
    if leaf_hashes.len() <= 1 {
        return;
    }
    let k = split_point(leaf_hashes.len());
    if leaf_index < k {
        inclusion_path(leaf_index, &leaf_hashes[..k], siblings);
        siblings.push(tree_hash_from_leaf_hashes(&leaf_hashes[k..]));
    } else {
        inclusion_path(leaf_index - k, &leaf_hashes[k..], siblings);
        siblings.push(tree_hash_from_leaf_hashes(&leaf_hashes[..k]));
    }
}

/// Builds the inclusion proof of leaf `leaf_index` over hashed leaves.
/// Returns `None` if `leaf_index` is out of range.
pub fn make_inclusion_proof_from_leaf_hashes(
    leaf_hashes: &[Hash],
    leaf_index: usize,
) -> Option<InclusionProof> {
    if leaf_index >= leaf_hashes.len() {
        return None;
    }
    let mut siblings = Vec::new();
    inclusion_path(leaf_index, leaf_hashes, &mut siblings);
    Some(InclusionProof {
        leaf_index: leaf_index as u64,
        leaf_count: leaf_hashes.len() as u64,
        siblings,
    })
}

/// Builds the inclusion proof of leaf `leaf_index` over raw leaf data.
/// Returns `None` if `leaf_index` is out of range.
pub fn make_inclusion_proof<T: AsRef<[u8]>>(
    tree_id: TreeId,
    leaves: &[T],
    leaf_index: usize,
) -> Option<InclusionProof> {
    make_inclusion_proof_from_leaf_hashes(&hash_leaves(tree_id, leaves), leaf_index)
}

/// Error returned when a frontier cannot take another leaf, or when its parts
/// are inconsistent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrontierError {
    /// The leaf count would overflow `u64`.
    Full,
    /// The number of subtree roots does not match the leaf count.
    InvalidSubtreeRoots { leaf_count: u64, num_roots: usize },
}

/// Append-only accumulator over a commitment tree.
///
/// Holds the leaf count and the roots of the complete (perfect) subtrees
/// along the right edge of the tree, largest (leftmost) first. There is one
/// subtree per set bit of the leaf count, so the frontier holds at most
/// [`MAX_FRONTIER_LEN`] hashes. Appending a leaf costs `O(log n)` hashes.
///
/// The frontier does not carry its [`TreeId`]: leaves are appended already
/// hashed, or through [`Self::append`], and the root is computed with
/// [`Self::root`]. Callers use the same `tree_id` for all of them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AppendOnlyFrontier {
    leaf_count: u64,
    subtree_roots: Vec<Hash>,
}

impl AppendOnlyFrontier {
    /// An empty tree.
    pub const fn new() -> Self {
        Self {
            leaf_count: 0,
            subtree_roots: Vec::new(),
        }
    }

    /// Rebuilds a frontier from its parts, e.g. after deserialization.
    pub fn from_parts(leaf_count: u64, subtree_roots: Vec<Hash>) -> Result<Self, FrontierError> {
        if subtree_roots.len() != leaf_count.count_ones() as usize {
            return Err(FrontierError::InvalidSubtreeRoots {
                leaf_count,
                num_roots: subtree_roots.len(),
            });
        }
        Ok(Self {
            leaf_count,
            subtree_roots,
        })
    }

    /// Builds a frontier holding all `leaf_hashes`.
    pub fn from_leaf_hashes(leaf_hashes: &[Hash]) -> Result<Self, FrontierError> {
        let mut frontier = Self::new();
        for leaf_hash in leaf_hashes {
            frontier.append_leaf_hash(*leaf_hash)?;
        }
        Ok(frontier)
    }

    /// Number of leaves appended so far.
    pub fn leaf_count(&self) -> u64 {
        self.leaf_count
    }

    /// Roots of the complete subtrees along the right edge, largest first.
    pub fn subtree_roots(&self) -> &[Hash] {
        &self.subtree_roots
    }

    /// Splits the frontier into its parts. Inverse of [`Self::from_parts`].
    pub fn into_parts(self) -> (u64, Vec<Hash>) {
        (self.leaf_count, self.subtree_roots)
    }

    /// Appends a leaf that is already hashed with [`hash_leaf`].
    pub fn append_leaf_hash(&mut self, leaf_hash: Hash) -> Result<(), FrontierError> {
        let new_leaf_count = self
            .leaf_count
            .checked_add(1)
            .ok_or(FrontierError::Full)?;
        // Each trailing one bit of the old count is a complete subtree of the
        // same size as the node being carried; merge them, smallest first.
        let mut node = leaf_hash;
        let mut count = self.leaf_count;
        while count & 1 == 1 {
            let left = self
                .subtree_roots
                .pop()
                .expect("frontier holds one subtree root per set bit of the leaf count");
            node = hash_node(&left, &node);
            count >>= 1;
        }
        self.subtree_roots.push(node);
        self.leaf_count = new_leaf_count;
        debug_assert_eq!(
            self.subtree_roots.len(),
            self.leaf_count.count_ones() as usize
        );
        Ok(())
    }

    /// Appends raw leaf data.
    pub fn append(&mut self, tree_id: TreeId, data: &[u8]) -> Result<(), FrontierError> {
        self.append_leaf_hash(hash_leaf(tree_id, data))
    }

    /// `MTH` of the leaves appended so far.
    pub fn tree_hash(&self) -> Hash {
        let mut roots = self.subtree_roots.iter().rev();
        let Some(smallest) = roots.next() else {
            return EMPTY_TREE_HASH;
        };
        roots.fold(*smallest, |acc, larger| hash_node(larger, &acc))
    }

    /// `root(tree_id, D)` of the leaves appended so far.
    pub fn root(&self, tree_id: TreeId) -> Hash {
        hash_root(tree_id, self.leaf_count, &self.tree_hash())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash_from_hex(hex: &str) -> Hash {
        let bytes: [u8; 32] = (0..32)
            .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        Hash::new_from_array(bytes)
    }

    fn test_leaves(n: usize) -> Vec<[u8; 32]> {
        (0..n).map(|i| [i as u8; 32]).collect()
    }

    /// Straightforward recursive `MTH` over raw data, written independently
    /// of the code under test.
    fn reference_mth(tree_id: TreeId, leaves: &[[u8; 32]]) -> Hash {
        match leaves.len() {
            0 => Hash::default(),
            1 => hashv(&[&[0x00, tree_id.0], &leaves[0]]),
            n => {
                let mut k = 1;
                while k * 2 < n {
                    k *= 2;
                }
                hashv(&[
                    &[0x01],
                    reference_mth(tree_id, &leaves[..k]).as_ref(),
                    reference_mth(tree_id, &leaves[k..]).as_ref(),
                ])
            }
        }
    }

    // Vectors computed with an independent Python implementation of the
    // construction in the SIMD (hashlib.sha256), leaves d_i = [i; 32],
    // tree_id 0x00.
    const VECTORS: &[(usize, &str)] = &[
        (
            0,
            "a2d0f2ddde325d9019ab3eb869e99ed6797a310d7902a9bf07dbe62afeb6d914",
        ),
        (
            1,
            "fa59330d4c36b3161df0b8ee57cc2f5c784f8698e96f6957dd045588b7219bfb",
        ),
        (
            2,
            "d812aeab056b715ae4b8d779bd5adb373504161c21c0fa87ac5d1cfa7fb52f71",
        ),
        (
            3,
            "781186e31b905a33e542cf6e32b0b1dc8edeec00f26dcc6b34f155e5d30bf621",
        ),
        (
            5,
            "44f5d04983fac85319ffddbe2ef0b845dd19ce2413172592df98d08ceef9be4d",
        ),
        (
            8,
            "56dd34a13539720b252c60538d854c5924cf99161d66546ed584bb212db61bb2",
        ),
    ];

    #[test]
    fn test_vectors() {
        for (n, expected) in VECTORS {
            let leaves = test_leaves(*n);
            let expected = hash_from_hex(expected);
            assert_eq!(
                compute_root(TreeId::COMMITMENT_VECTOR, &leaves),
                expected,
                "n = {n}"
            );
            let frontier = AppendOnlyFrontier::from_leaf_hashes(&hash_leaves(
                TreeId::COMMITMENT_VECTOR,
                &leaves,
            ))
            .unwrap();
            assert_eq!(frontier.root(TreeId::COMMITMENT_VECTOR), expected);
        }
    }

    #[test]
    fn test_vector_proofs_of_last_leaf() {
        let expected: &[(usize, &[&str])] = &[
            (1, &[]),
            (
                2,
                &["eb142b0cae0baa72a767ebc0823d1be94e14c5bfc52d8e417fc4302fceb6240c"],
            ),
            (
                3,
                &["c81c03cb8f6c4f8a1a3f91f427f1b1d4f76e55f6094fa33fbed13fc1124d1f41"],
            ),
            (
                5,
                &["2c2b39a396137cdcbd52c058b5b8361d17c09360496f95398ceb73401ffd1df6"],
            ),
            (
                8,
                &[
                    "0cb447e4a1d45b7bc31cf97458a2403d0b86afcc283e14576b3c342d0b97c98a",
                    "f6dd965f37e5ee4ac0c4252c16a8b5d95c03b53eda6ff106c40ac8179f26d8fe",
                    "2c2b39a396137cdcbd52c058b5b8361d17c09360496f95398ceb73401ffd1df6",
                ],
            ),
        ];
        for (n, siblings) in expected {
            let leaves = test_leaves(*n);
            let proof = make_inclusion_proof(TreeId::COMMITMENT_VECTOR, &leaves, n - 1).unwrap();
            let siblings: Vec<_> = siblings.iter().map(|s| hash_from_hex(s)).collect();
            assert_eq!(proof.siblings, siblings, "n = {n}");
            assert_eq!(proof.leaf_count, *n as u64);
            assert_eq!(proof.leaf_index, (*n - 1) as u64);
        }
    }

    #[test]
    fn test_block_history_vector() {
        let leaves: Vec<Vec<u8>> = [3u64, 4, 7]
            .iter()
            .map(|slot| {
                let mut leaf = slot.to_le_bytes().to_vec();
                leaf.extend_from_slice(&[*slot as u8; 32]);
                leaf.extend_from_slice(&[0xff - *slot as u8; 32]);
                leaf
            })
            .collect();
        assert_eq!(
            compute_root(TreeId::BLOCK_HISTORY, &leaves),
            hash_from_hex("f84e3c83bc83886ef664e900bb0337147a3b883a806bccea2581ccd4fd933426"),
        );
    }

    #[test]
    fn test_empty_tree() {
        let leaves: [&[u8]; 0] = [];
        assert_eq!(
            tree_hash_from_leaf_hashes(&[]),
            Hash::new_from_array([0; 32])
        );
        let root = compute_root(TreeId::BLOCK_HISTORY, &leaves);
        assert_eq!(
            root,
            hashv(&[&[0x80, 0x01], &0u64.to_le_bytes(), &[0; 32]])
        );
        assert_ne!(root, Hash::default());
        // Empty trees with different tree ids have different roots.
        assert_ne!(root, compute_root(TreeId::COMMITMENT_VECTOR, &leaves));
        assert_eq!(AppendOnlyFrontier::new().root(TreeId::BLOCK_HISTORY), root);
        assert_eq!(AppendOnlyFrontier::new().tree_hash(), EMPTY_TREE_HASH);
        assert!(make_inclusion_proof(TreeId::BLOCK_HISTORY, &leaves, 0).is_none());
    }

    #[test]
    fn test_single_leaf() {
        let leaf = b"single leaf";
        let leaf_hash = hashv(&[&[0x00, 0x05], leaf]);
        assert_eq!(hash_leaf(TreeId::ACCOUNTS_DELTA, leaf), leaf_hash);
        assert_eq!(tree_hash_from_leaf_hashes(&[leaf_hash]), leaf_hash);
        let root = compute_root(TreeId::ACCOUNTS_DELTA, &[leaf]);
        assert_eq!(
            root,
            hashv(&[&[0x80, 0x05], &1u64.to_le_bytes(), leaf_hash.as_ref()])
        );
        let proof = make_inclusion_proof(TreeId::ACCOUNTS_DELTA, &[leaf], 0).unwrap();
        assert!(proof.siblings.is_empty());
        assert!(proof.verify(TreeId::ACCOUNTS_DELTA, leaf, &root));
        assert!(!proof.verify(TreeId::ACCOUNTS_DELTA, b"other leaf", &root));
        assert!(!proof.verify(TreeId::EVENTS, leaf, &root));
    }

    #[test]
    fn test_matches_reference_for_many_sizes() {
        for n in 0..=70 {
            let leaves = test_leaves(n);
            let expected_mth = reference_mth(TreeId::EVENTS, &leaves);
            let leaf_hashes = hash_leaves(TreeId::EVENTS, &leaves);
            assert_eq!(tree_hash_from_leaf_hashes(&leaf_hashes), expected_mth);
            let frontier = AppendOnlyFrontier::from_leaf_hashes(&leaf_hashes).unwrap();
            assert_eq!(frontier.leaf_count(), n as u64);
            assert_eq!(frontier.subtree_roots().len(), n.count_ones() as usize);
            assert_eq!(frontier.tree_hash(), expected_mth);
            assert_eq!(
                frontier.root(TreeId::EVENTS),
                hash_root(TreeId::EVENTS, n as u64, &expected_mth)
            );
            assert_eq!(
                frontier.root(TreeId::EVENTS),
                compute_root(TreeId::EVENTS, &leaves)
            );
        }
    }

    #[test]
    fn test_proof_round_trips() {
        for n in 1..=33 {
            let leaves = test_leaves(n);
            let root = compute_root(TreeId::TRANSACTION_RECEIPTS, &leaves);
            for (i, leaf) in leaves.iter().enumerate() {
                let proof = make_inclusion_proof(TreeId::TRANSACTION_RECEIPTS, &leaves, i).unwrap();
                assert!(
                    proof.siblings.len() <= 64 - (n as u64).leading_zeros() as usize,
                    "n = {n}, i = {i}"
                );
                assert!(
                    proof.verify(TreeId::TRANSACTION_RECEIPTS, leaf, &root),
                    "n = {n}, i = {i}"
                );
                // Wrong tree id.
                assert!(!proof.verify(TreeId::INSTRUCTION_TRACE, leaf, &root));
                // Wrong leaf data.
                assert!(!proof.verify(TreeId::TRANSACTION_RECEIPTS, &[0xaa; 32], &root));
                // Wrong index.
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let mut wrong = proof.clone();
                    wrong.leaf_index = j as u64;
                    assert!(
                        !wrong.verify(TreeId::TRANSACTION_RECEIPTS, leaf, &root),
                        "n = {n}, i = {i}, j = {j}"
                    );
                }
                // Out of range index.
                let mut wrong = proof.clone();
                wrong.leaf_index = n as u64;
                assert!(!wrong.verify(TreeId::TRANSACTION_RECEIPTS, leaf, &root));
                // Wrong leaf count (the count is bound in the root).
                let mut wrong = proof.clone();
                wrong.leaf_count += 1;
                assert!(!wrong.verify(TreeId::TRANSACTION_RECEIPTS, leaf, &root));
                // Tampered, missing or extra sibling.
                if !proof.siblings.is_empty() {
                    let mut wrong = proof.clone();
                    wrong.siblings[0] = Hash::new_from_array([0xbb; 32]);
                    assert!(!wrong.verify(TreeId::TRANSACTION_RECEIPTS, leaf, &root));
                    let mut wrong = proof.clone();
                    wrong.siblings.pop();
                    assert!(!wrong.verify(TreeId::TRANSACTION_RECEIPTS, leaf, &root));
                }
                let mut wrong = proof.clone();
                wrong.siblings.push(Hash::new_from_array([0xcc; 32]));
                assert!(!wrong.verify(TreeId::TRANSACTION_RECEIPTS, leaf, &root));
            }
            assert!(make_inclusion_proof(TreeId::TRANSACTION_RECEIPTS, &leaves, n).is_none());
        }
    }

    #[test]
    fn test_commitment_vector_opening_costs_three_siblings() {
        let leaves = test_leaves(8);
        for i in 0..8 {
            let proof = make_inclusion_proof(TreeId::COMMITMENT_VECTOR, &leaves, i).unwrap();
            assert_eq!(proof.siblings.len(), 3);
        }
    }

    #[test]
    fn test_frontier_from_parts() {
        let leaves = test_leaves(13);
        let frontier = AppendOnlyFrontier::from_leaf_hashes(&hash_leaves(
            TreeId::BLOCK_HISTORY,
            &leaves,
        ))
        .unwrap();
        let (leaf_count, roots) = frontier.clone().into_parts();
        assert_eq!(leaf_count, 13);
        assert_eq!(roots.len(), 3);
        assert_eq!(
            AppendOnlyFrontier::from_parts(leaf_count, roots.clone()).unwrap(),
            frontier
        );
        assert_eq!(
            AppendOnlyFrontier::from_parts(12, roots.clone()),
            Err(FrontierError::InvalidSubtreeRoots {
                leaf_count: 12,
                num_roots: 3
            })
        );
        // The largest subtree root covers leaves 0..8.
        assert_eq!(
            roots[0],
            tree_hash_from_leaf_hashes(&hash_leaves(TreeId::BLOCK_HISTORY, &leaves[..8]))
        );
    }

    #[test]
    fn test_frontier_full() {
        let roots = vec![Hash::new_from_array([1; 32]); MAX_FRONTIER_LEN];
        let mut frontier = AppendOnlyFrontier::from_parts(u64::MAX, roots).unwrap();
        assert_eq!(frontier.subtree_roots().len(), MAX_FRONTIER_LEN);
        let before = frontier.clone();
        assert_eq!(
            frontier.append_leaf_hash(Hash::default()),
            Err(FrontierError::Full)
        );
        assert_eq!(frontier, before);

        // One below the maximum: 63 roots, appending merges all of them.
        let roots = vec![Hash::new_from_array([1; 32]); MAX_FRONTIER_LEN - 1];
        let mut frontier = AppendOnlyFrontier::from_parts(u64::MAX >> 1, roots).unwrap();
        frontier.append_leaf_hash(Hash::default()).unwrap();
        assert_eq!(frontier.leaf_count(), 1 << 63);
        assert_eq!(frontier.subtree_roots().len(), 1);
    }

    #[test]
    fn test_prefixes_are_distinct() {
        let prefixes = [
            LEAF_PREFIX,
            NODE_PREFIX,
            ROOT_PREFIX,
            VALIDATOR_SET_ROOT_PREFIX,
            EPOCH_BOUNDARY_PREFIX,
        ];
        for (i, a) in prefixes.iter().enumerate() {
            for b in &prefixes[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }
}
