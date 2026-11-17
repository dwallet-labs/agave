//! The accounts delta tree of the "Accounts Delta Root" SIMD.
//!
//! The tree commits to the post-block state of every account changed in a
//! block. Its leaves are sorted by address and unique:
//!
//! ```text
//! delta_leaf(a)       = a.pubkey || account_hash(a')        // 64 bytes
//! accounts_delta_root = root(0x05, [delta_leaf(c) for c in C])
//! ```
//!
//! where `C` is the list of changed accounts sorted by address in ascending
//! lexicographic byte order and `root` is the commitment tree construction of
//! [`crate::commitment_tree`] with [`TreeId::ACCOUNTS_DELTA`].
//!
//! `account_hash` is defined over account state (32 zero bytes for an account
//! with zero lamports, otherwise the BLAKE3 digest of the SIMD-0215 lattice
//! hash input) and is computed by the runtime. This module only handles the
//! 32-byte values: building the sorted tree, inclusion proofs ("state at a
//! block", "deletion") and adjacency proofs ("unchanged in a block").

use {
    crate::commitment_tree::{
        InclusionProof, TreeId, compute_root_from_leaf_hashes, hash_leaf,
        make_inclusion_proof_from_leaf_hashes,
    },
    solana_hash::Hash,
};

/// Length of an account address.
pub const ADDRESS_LEN: usize = 32;

/// Length of a delta leaf: `address || account_hash`.
pub const DELTA_LEAF_LEN: usize = ADDRESS_LEN + 32;

/// `account_hash` of an account with zero lamports (the empty state): the
/// leaf value that proves a deletion.
pub const DELETED_ACCOUNT_HASH: Hash = Hash::new_from_array([0; 32]);

/// An account address, as raw bytes. Ordered lexicographically.
pub type Address = [u8; ADDRESS_LEN];

/// `delta_leaf(a) = a.pubkey || account_hash(a')`
pub fn delta_leaf(address: &Address, account_hash: &Hash) -> [u8; DELTA_LEAF_LEN] {
    let mut leaf = [0u8; DELTA_LEAF_LEN];
    leaf[..ADDRESS_LEN].copy_from_slice(address);
    leaf[ADDRESS_LEN..].copy_from_slice(account_hash.as_ref());
    leaf
}

/// `leaf(0x05, delta_leaf(a))`
pub fn hash_delta_leaf(address: &Address, account_hash: &Hash) -> Hash {
    hash_leaf(TreeId::ACCOUNTS_DELTA, &delta_leaf(address, account_hash))
}

/// One changed account: its address and the `account_hash` of its state at
/// the end of the block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeltaEntry {
    pub address: Address,
    pub account_hash: Hash,
}

impl DeltaEntry {
    pub fn new(address: Address, account_hash: Hash) -> Self {
        Self {
            address,
            account_hash,
        }
    }

    /// True if the entry proves that the account was emptied.
    pub fn is_deletion(&self) -> bool {
        self.account_hash == DELETED_ACCOUNT_HASH
    }

    /// `delta_leaf` of this entry.
    pub fn leaf(&self) -> [u8; DELTA_LEAF_LEN] {
        delta_leaf(&self.address, &self.account_hash)
    }

    /// `leaf(0x05, delta_leaf)` of this entry.
    pub fn leaf_hash(&self) -> Hash {
        hash_delta_leaf(&self.address, &self.account_hash)
    }
}

/// Error building an [`AccountsDelta`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccountsDeltaError {
    /// The same address appears twice; addresses in the tree are unique.
    DuplicateAddress(Address),
}

/// `root(0x05, [delta_leaf(c) for c in entries])`.
///
/// `entries` must already be sorted by address and unique; use
/// [`AccountsDelta::new`] to sort and check them.
pub fn compute_accounts_delta_root(entries: &[DeltaEntry]) -> Hash {
    compute_root_from_leaf_hashes(TreeId::ACCOUNTS_DELTA, &leaf_hashes(entries))
}

fn leaf_hashes(entries: &[DeltaEntry]) -> Vec<Hash> {
    entries.iter().map(DeltaEntry::leaf_hash).collect()
}

/// The changed accounts of a block, sorted by address, and their root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccountsDelta {
    entries: Vec<DeltaEntry>,
    root: Hash,
}

impl Default for AccountsDelta {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            root: compute_accounts_delta_root(&[]),
        }
    }
}

impl AccountsDelta {
    /// Sorts `entries` by address and computes the root. Fails if an address
    /// appears more than once.
    pub fn new(mut entries: Vec<DeltaEntry>) -> Result<Self, AccountsDeltaError> {
        // Addresses are unique (checked below), so an unstable sort is
        // deterministic.
        entries.sort_unstable_by_key(|entry| entry.address);
        if let Some(pair) = entries
            .windows(2)
            .find(|pair| pair[0].address == pair[1].address)
        {
            return Err(AccountsDeltaError::DuplicateAddress(pair[0].address));
        }
        let root = compute_accounts_delta_root(&entries);
        Ok(Self { entries, root })
    }

    /// `accounts_delta_root`.
    pub fn root(&self) -> Hash {
        self.root
    }

    /// The changed accounts, sorted by address.
    pub fn entries(&self) -> &[DeltaEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Index of `address` in the tree (`Ok`), or the index where it would be
    /// inserted (`Err`).
    fn search(&self, address: &Address) -> Result<usize, usize> {
        self.entries
            .binary_search_by(|entry| entry.address.cmp(address))
    }

    /// The entry of `address`, if it was changed in the block.
    pub fn get(&self, address: &Address) -> Option<&DeltaEntry> {
        self.search(address).ok().map(|index| &self.entries[index])
    }

    fn leaf_proof(&self, leaf_hashes: &[Hash], index: usize) -> DeltaLeafProof {
        DeltaLeafProof {
            entry: self.entries[index],
            proof: make_inclusion_proof_from_leaf_hashes(leaf_hashes, index)
                .expect("index is in range"),
        }
    }

    /// Proof of the state of `address` at the end of the block ("state at a
    /// block", or "deletion" if the leaf carries [`DELETED_ACCOUNT_HASH`]).
    /// `None` if the account was not changed in the block.
    pub fn inclusion_proof(&self, address: &Address) -> Option<DeltaLeafProof> {
        let index = self.search(address).ok()?;
        Some(self.leaf_proof(&leaf_hashes(&self.entries), index))
    }

    /// Proof that `address` was not changed in the block. `None` if it was.
    pub fn unchanged_proof(&self, address: &Address) -> Option<UnchangedProof> {
        let index = self.search(address).err()?;
        if self.entries.is_empty() {
            return Some(UnchangedProof::EmptyTree);
        }
        let leaf_hashes = leaf_hashes(&self.entries);
        Some(if index == 0 {
            UnchangedProof::BeforeFirst {
                first: self.leaf_proof(&leaf_hashes, 0),
            }
        } else if index == self.entries.len() {
            UnchangedProof::AfterLast {
                last: self.leaf_proof(&leaf_hashes, index - 1),
            }
        } else {
            UnchangedProof::Between {
                left: self.leaf_proof(&leaf_hashes, index - 1),
                right: self.leaf_proof(&leaf_hashes, index),
            }
        })
    }
}

/// A delta leaf and its inclusion proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeltaLeafProof {
    pub entry: DeltaEntry,
    pub proof: InclusionProof,
}

impl DeltaLeafProof {
    /// True if `self.entry` is the leaf at `self.proof.leaf_index` of the tree
    /// with root `accounts_delta_root`.
    pub fn verify(&self, accounts_delta_root: &Hash) -> bool {
        self.proof.verify_leaf_hash(
            TreeId::ACCOUNTS_DELTA,
            self.entry.leaf_hash(),
            accounts_delta_root,
        )
    }

    /// True if the proof shows that `address` ended the block with the state
    /// whose `account_hash` is `account_hash`. A verifier computes
    /// `account_hash` from the claimed account state.
    pub fn verify_account(
        &self,
        address: &Address,
        account_hash: &Hash,
        accounts_delta_root: &Hash,
    ) -> bool {
        self.entry.address == *address
            && self.entry.account_hash == *account_hash
            && self.verify(accounts_delta_root)
    }
}

/// Proof that an address was not changed in a block: the tree is empty, or
/// the address falls before the first leaf, after the last leaf, or between
/// two adjacent leaves.
///
/// Sound because the leaves are sorted and unique (a consensus rule) and the
/// leaf count is bound in the root: each opened leaf proves its index and the
/// leaf count, and [`Self::verify`] checks adjacency from them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnchangedProof {
    /// No account was changed in the block.
    EmptyTree,
    /// The address is smaller than the first leaf.
    BeforeFirst { first: DeltaLeafProof },
    /// The address is larger than the last leaf.
    AfterLast { last: DeltaLeafProof },
    /// The address lies strictly between two adjacent leaves.
    Between {
        left: DeltaLeafProof,
        right: DeltaLeafProof,
    },
}

impl UnchangedProof {
    /// True if the proof shows that `address` is not a leaf of the tree with
    /// root `accounts_delta_root`.
    pub fn verify(&self, address: &Address, accounts_delta_root: &Hash) -> bool {
        match self {
            Self::EmptyTree => *accounts_delta_root == compute_accounts_delta_root(&[]),
            Self::BeforeFirst { first } => {
                first.proof.leaf_index == 0
                    && *address < first.entry.address
                    && first.verify(accounts_delta_root)
            }
            Self::AfterLast { last } => {
                last.proof.leaf_index.checked_add(1) == Some(last.proof.leaf_count)
                    && last.entry.address < *address
                    && last.verify(accounts_delta_root)
            }
            Self::Between { left, right } => {
                left.proof.leaf_count == right.proof.leaf_count
                    && left.proof.leaf_index.checked_add(1) == Some(right.proof.leaf_index)
                    && left.entry.address < *address
                    && *address < right.entry.address
                    && left.verify(accounts_delta_root)
                    && right.verify(accounts_delta_root)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use {super::*, crate::commitment_tree::compute_root, solana_sha256_hasher::hashv};

    fn hash_from_hex(hex: &str) -> Hash {
        Hash::new_from_array(<[u8; 32]>::try_from(hex::decode(hex).unwrap()).unwrap())
    }

    /// Entry `i` of the test vectors: address `[2i + 1; 32]`, account hash
    /// `[0x80 + i; 32]`, except entry 1, which is a deletion (zero hash).
    fn test_entry(i: u8) -> DeltaEntry {
        let account_hash = if i == 1 {
            DELETED_ACCOUNT_HASH
        } else {
            Hash::new_from_array([0x80 + i; 32])
        };
        DeltaEntry::new([2 * i + 1; 32], account_hash)
    }

    fn test_entries(n: u8) -> Vec<DeltaEntry> {
        (0..n).map(test_entry).collect()
    }

    // Vectors computed with an independent Python implementation of the
    // construction in the SIMD (hashlib.sha256), entries as in `test_entry`.
    const ROOT_VECTORS: &[(u8, &str)] = &[
        (
            0,
            "f5a5694a77ae16bd1ef2c02233335649ef3d6b1f33e5d8aad49852057957c4e4",
        ),
        (
            1,
            "7c0bf2e7fb123ca6bba9c05725695c4386ec7ed793278b306ac8302109bab57d",
        ),
        (
            2,
            "49e018c63f8744718516a0646b3b8565d1eef4a624cf0706d965fe0c3431ed06",
        ),
        (
            3,
            "db31198957c59540fbf8042ace9565c0941ad520992f835e2a52ab5cf9f29554",
        ),
        (
            5,
            "9f87432e000dad7dffb72f0c71bf8286039d1dfe1157ca4e5a714448595cf359",
        ),
    ];

    #[test]
    fn test_delta_leaf_layout() {
        let address = [7; 32];
        let account_hash = Hash::new_from_array([9; 32]);
        let leaf = delta_leaf(&address, &account_hash);
        assert_eq!(leaf.len(), 64);
        assert_eq!(&leaf[..32], &address);
        assert_eq!(&leaf[32..], account_hash.as_ref());
        assert_eq!(
            hash_delta_leaf(&address, &account_hash),
            hashv(&[&[0x00, 0x05], &leaf])
        );
    }

    #[test]
    fn test_root_vectors() {
        for (n, expected) in ROOT_VECTORS {
            let entries = test_entries(*n);
            let expected = hash_from_hex(expected);
            assert_eq!(compute_accounts_delta_root(&entries), expected, "n = {n}");
            let leaves: Vec<_> = entries.iter().map(DeltaEntry::leaf).collect();
            assert_eq!(compute_root(TreeId::ACCOUNTS_DELTA, &leaves), expected);
            // The constructor sorts.
            let mut reversed = entries.clone();
            reversed.reverse();
            assert_eq!(AccountsDelta::new(reversed).unwrap().root(), expected);
        }
    }

    #[test]
    fn test_empty_delta() {
        let delta = AccountsDelta::new(vec![]).unwrap();
        assert_eq!(delta, AccountsDelta::default());
        assert!(delta.is_empty());
        assert_eq!(
            delta.root(),
            hashv(&[&[0x80, 0x05], &0u64.to_le_bytes(), &[0; 32]])
        );
        let address = [1; 32];
        assert!(delta.inclusion_proof(&address).is_none());
        let proof = delta.unchanged_proof(&address).unwrap();
        assert_eq!(proof, UnchangedProof::EmptyTree);
        assert!(proof.verify(&address, &delta.root()));
        // An empty-tree proof does not verify against a non-empty tree.
        let other = AccountsDelta::new(test_entries(1)).unwrap();
        assert!(!proof.verify(&address, &other.root()));
    }

    #[test]
    fn test_sorting_and_duplicates() {
        let entries = vec![
            DeltaEntry::new([3; 32], Hash::new_from_array([1; 32])),
            DeltaEntry::new([1; 32], Hash::new_from_array([2; 32])),
            DeltaEntry::new([2; 32], Hash::new_from_array([3; 32])),
        ];
        let delta = AccountsDelta::new(entries).unwrap();
        let addresses: Vec<_> = delta.entries().iter().map(|e| e.address[0]).collect();
        assert_eq!(addresses, [1, 2, 3]);

        // Ordering is lexicographic over all 32 bytes, not by the first byte
        // only or as a little-endian integer.
        let mut low = [0; 32];
        low[31] = 0xff;
        let mut high = [0; 32];
        high[0] = 0x01;
        let delta = AccountsDelta::new(vec![
            DeltaEntry::new(high, Hash::default()),
            DeltaEntry::new(low, Hash::default()),
        ])
        .unwrap();
        assert_eq!(delta.entries()[0].address, low);

        let duplicate = vec![
            DeltaEntry::new([5; 32], Hash::new_from_array([1; 32])),
            DeltaEntry::new([4; 32], Hash::new_from_array([1; 32])),
            DeltaEntry::new([5; 32], Hash::new_from_array([2; 32])),
        ];
        assert_eq!(
            AccountsDelta::new(duplicate),
            Err(AccountsDeltaError::DuplicateAddress([5; 32]))
        );
    }

    #[test]
    fn test_inclusion_proofs() {
        for n in 1..=17 {
            let delta = AccountsDelta::new(test_entries(n)).unwrap();
            let root = delta.root();
            for entry in delta.entries() {
                let proof = delta.inclusion_proof(&entry.address).unwrap();
                assert_eq!(proof.entry, *entry);
                assert_eq!(proof.proof.leaf_count, n as u64);
                assert!(proof.verify(&root));
                assert!(proof.verify_account(&entry.address, &entry.account_hash, &root));
                // Wrong claimed state or address.
                assert!(!proof.verify_account(
                    &entry.address,
                    &Hash::new_from_array([0xdd; 32]),
                    &root
                ));
                assert!(!proof.verify_account(&[0xee; 32], &entry.account_hash, &root));
                // Tampered entry.
                let mut wrong = proof.clone();
                wrong.entry.account_hash = Hash::new_from_array([0xdd; 32]);
                assert!(!wrong.verify(&root));
                assert!(delta.unchanged_proof(&entry.address).is_none());
            }
            // The deletion leaf proves a deletion.
            if n > 1 {
                let deleted = test_entry(1);
                let proof = delta.inclusion_proof(&deleted.address).unwrap();
                assert!(proof.entry.is_deletion());
                assert!(proof.verify_account(&deleted.address, &DELETED_ACCOUNT_HASH, &root));
            }
        }
    }

    #[test]
    fn test_unchanged_proofs() {
        for n in 1..=17u8 {
            let delta = AccountsDelta::new(test_entries(n)).unwrap();
            let root = delta.root();
            // Leaves sit at odd bytes [1], [3], ...; probe every even byte and
            // every byte between, including both ends.
            for probe in 0..=(2 * n + 1) {
                let address = [probe; 32];
                if probe % 2 == 1 && probe < 2 * n {
                    assert!(delta.unchanged_proof(&address).is_none());
                    continue;
                }
                let proof = delta.unchanged_proof(&address).unwrap();
                match (&proof, probe) {
                    (UnchangedProof::BeforeFirst { .. }, 0) => {}
                    (UnchangedProof::AfterLast { .. }, p) if p >= 2 * n => {}
                    (UnchangedProof::Between { .. }, _) => {}
                    _ => panic!("unexpected proof {proof:?} for probe {probe}"),
                }
                assert!(proof.verify(&address, &root), "n = {n}, probe = {probe}");
                // The same proof does not cover a changed address.
                for entry in delta.entries() {
                    assert!(!proof.verify(&entry.address, &root));
                }
                assert!(!proof.verify(&address, &Hash::new_from_array([0xdd; 32])));
            }
        }
    }

    #[test]
    fn test_unchanged_proof_rejects_non_adjacent_leaves() {
        let delta = AccountsDelta::new(test_entries(5)).unwrap();
        let root = delta.root();
        // Address [4; 32] lies between leaves 1 ([3]) and 2 ([5]). Leaves 0
        // and 2 are valid leaves that also surround it, but are not adjacent.
        let address = [4; 32];
        let UnchangedProof::Between { left, right } = delta.unchanged_proof(&address).unwrap()
        else {
            panic!("expected a between proof");
        };
        let first = delta.inclusion_proof(&[1; 32]).unwrap();
        assert!(first.verify(&root));
        let skipping = UnchangedProof::Between {
            left: first.clone(),
            right: right.clone(),
        };
        assert!(!skipping.verify(&address, &root));
        // Swapped sides.
        let swapped = UnchangedProof::Between {
            left: right.clone(),
            right: left.clone(),
        };
        assert!(!swapped.verify(&address, &root));
        // "Before first" with a leaf that is not the first.
        let not_first = UnchangedProof::BeforeFirst {
            first: right.clone(),
        };
        assert!(!not_first.verify(&[0; 32], &root));
        // "After last" with a leaf that is not the last.
        let not_last = UnchangedProof::AfterLast { last: left };
        assert!(!not_last.verify(&[0xff; 32], &root));
        // An empty-tree claim against a non-empty tree.
        assert!(!UnchangedProof::EmptyTree.verify(&address, &root));
    }

    // Inclusion proof of the deletion leaf (index 1) in the 5-entry vector
    // tree, from the same Python implementation.
    const PROOF_VECTOR_5_INDEX_1: &[&str] = &[
        "d598cb9af4c5caeeabdb9c73ea8fa0d46c731b5aa45ed3d41502da19044e453f",
        "6933e1fe0ac3aa7c3829e1f1ee45fc24eefee47eddea6f0c896b4e474e6c08cf",
        "f01dab8ab54c23bfe29b7829f122e3865262fe5209e207b9f0cd4446a993b5a7",
    ];

    #[test]
    fn test_proof_vector() {
        let delta = AccountsDelta::new(test_entries(5)).unwrap();
        let proof = delta.inclusion_proof(&test_entry(1).address).unwrap();
        assert_eq!(proof.proof.leaf_index, 1);
        assert_eq!(proof.proof.leaf_count, 5);
        let expected: Vec<_> = PROOF_VECTOR_5_INDEX_1
            .iter()
            .map(|hex| hash_from_hex(hex))
            .collect();
        assert_eq!(proof.proof.siblings, expected);
    }
}
