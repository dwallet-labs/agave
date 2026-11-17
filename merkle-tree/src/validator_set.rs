//! The validator set tree of the "Epoch Validator Set Root" SIMD.
//!
//! Every block of epoch `E` commits, in slot 3 of its commitment vector, to
//! the signer set of epoch `E + 1`: the validators that have an Alpenglow
//! signer index (rank) in that epoch, in signer index order.
//!
//! ```text
//! signer_leaf(v)         = v.bls_pubkey_compressed      // 48 bytes
//!                       || u64_le(v.stake)              // lamports
//!                       || v.vote_pubkey                // 32 bytes
//!
//! total_stake            = sum of v.stake over S(N)
//! validator_set_root(N)  = sha256( 0x82
//!                               || u64_le(N)
//!                               || u64_le(total_stake)
//!                               || root(0x06, [signer_leaf(v) for v in S(N)]) )
//! ```
//!
//! The leaf at position `i` is the signer with index (rank) `i`, which is the
//! bit a certificate's signer bitmap uses for it. Signer indices order the
//! set by stake descending, then by compressed BLS public key ascending
//! (lexicographic byte order), see [`Signer::cmp_signer_order`]. The runtime
//! takes the order from its rank map; this module only hashes what it is
//! given, in the order given.
//!
//! `root` is the commitment tree construction of [`crate::commitment_tree`]
//! with [`TreeId::VALIDATOR_SET`]; it binds the number of signers.

use {
    crate::commitment_tree::{
        InclusionProof, TreeId, VALIDATOR_SET_ROOT_PREFIX, compute_root_from_leaf_hashes,
        hash_leaf, make_inclusion_proof_from_leaf_hashes,
    },
    solana_hash::Hash,
    solana_sha256_hasher::hashv,
    std::cmp::Ordering,
};

/// Length of a compressed BLS public key.
pub const BLS_PUBKEY_COMPRESSED_LEN: usize = 48;

/// Length of a vote account address.
pub const VOTE_PUBKEY_LEN: usize = 32;

/// Length of a signer leaf: `bls_pubkey_compressed || u64_le(stake) || vote_pubkey`.
pub const SIGNER_LEAF_LEN: usize = BLS_PUBKEY_COMPRESSED_LEN + 8 + VOTE_PUBKEY_LEN;

/// `signer_leaf(v) = v.bls_pubkey_compressed || u64_le(v.stake) || v.vote_pubkey`
pub fn signer_leaf(
    bls_pubkey_compressed: &[u8; BLS_PUBKEY_COMPRESSED_LEN],
    stake: u64,
    vote_pubkey: &[u8; VOTE_PUBKEY_LEN],
) -> [u8; SIGNER_LEAF_LEN] {
    let mut leaf = [0u8; SIGNER_LEAF_LEN];
    leaf[..BLS_PUBKEY_COMPRESSED_LEN].copy_from_slice(bls_pubkey_compressed);
    leaf[BLS_PUBKEY_COMPRESSED_LEN..BLS_PUBKEY_COMPRESSED_LEN + 8]
        .copy_from_slice(&stake.to_le_bytes());
    leaf[BLS_PUBKEY_COMPRESSED_LEN + 8..].copy_from_slice(vote_pubkey);
    leaf
}

/// One member of a signer set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Signer {
    /// BLS public key in the compressed encoding stored in the vote account.
    pub bls_pubkey_compressed: [u8; BLS_PUBKEY_COMPRESSED_LEN],
    /// Stake in lamports used for certificate thresholds.
    pub stake: u64,
    /// Address of the vote account.
    pub vote_pubkey: [u8; VOTE_PUBKEY_LEN],
}

impl Signer {
    /// `signer_leaf` of this signer.
    pub fn leaf(&self) -> [u8; SIGNER_LEAF_LEN] {
        signer_leaf(&self.bls_pubkey_compressed, self.stake, &self.vote_pubkey)
    }

    /// `leaf(0x06, signer_leaf(v))`
    pub fn leaf_hash(&self) -> Hash {
        hash_leaf(TreeId::VALIDATOR_SET, &self.leaf())
    }

    /// Signer index order (SIMD-0326 with the BLS signer index amendment):
    /// stake descending, then compressed BLS public key ascending. Two
    /// members of one set never compare equal, since the rank map admits a
    /// BLS key only once.
    pub fn cmp_signer_order(&self, other: &Self) -> Ordering {
        other
            .stake
            .cmp(&self.stake)
            .then_with(|| self.bls_pubkey_compressed.cmp(&other.bls_pubkey_compressed))
    }
}

/// `validator_set_root(N)` from the signer tree root `root(0x06, ...)`.
pub fn validator_set_root_from_signer_root(
    epoch: u64,
    total_stake: u64,
    signer_root: &Hash,
) -> Hash {
    hashv(&[
        &[VALIDATOR_SET_ROOT_PREFIX],
        &epoch.to_le_bytes(),
        &total_stake.to_le_bytes(),
        signer_root.as_ref(),
    ])
}

/// The part of `validator_set_root(N)` that does not depend on `N`: the
/// signer tree root and the total stake of one signer set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignerSetCommitment {
    /// Number of signers (leaves).
    pub signer_count: u64,
    /// Sum of the stake of all signers.
    pub total_stake: u64,
    /// `root(0x06, [signer_leaf(v) for v in S])`
    pub signer_root: Hash,
}

impl SignerSetCommitment {
    /// The commitment to the empty signer set.
    pub fn empty() -> Self {
        Self::from_leaf_hashes(0, &[])
    }

    /// Commitment to `signers`, in the order given (signer index order).
    /// Returns `None` if the total stake overflows `u64`.
    pub fn new(signers: &[Signer]) -> Option<Self> {
        let total_stake = signers
            .iter()
            .try_fold(0u64, |total, signer| total.checked_add(signer.stake))?;
        let leaf_hashes: Vec<Hash> = signers.iter().map(Signer::leaf_hash).collect();
        Some(Self::from_leaf_hashes(total_stake, &leaf_hashes))
    }

    fn from_leaf_hashes(total_stake: u64, leaf_hashes: &[Hash]) -> Self {
        Self {
            signer_count: leaf_hashes.len() as u64,
            total_stake,
            signer_root: compute_root_from_leaf_hashes(TreeId::VALIDATOR_SET, leaf_hashes),
        }
    }

    /// `validator_set_root(epoch)` of this signer set.
    pub fn validator_set_root(&self, epoch: u64) -> Hash {
        validator_set_root_from_signer_root(epoch, self.total_stake, &self.signer_root)
    }
}

/// `validator_set_root(epoch)` of `signers`, in the order given (signer index
/// order). Returns `None` if the total stake overflows `u64`.
pub fn validator_set_root(epoch: u64, signers: &[Signer]) -> Option<Hash> {
    SignerSetCommitment::new(signers).map(|commitment| commitment.validator_set_root(epoch))
}

/// Proof that `signers[index]` is the signer with index `index` of the set.
/// Returns `None` if `index` is out of range.
pub fn signer_inclusion_proof(signers: &[Signer], index: usize) -> Option<InclusionProof> {
    let leaf_hashes: Vec<Hash> = signers.iter().map(Signer::leaf_hash).collect();
    make_inclusion_proof_from_leaf_hashes(&leaf_hashes, index)
}

/// Returns true if `signer` is the signer with index `proof.leaf_index` of a
/// set of `proof.leaf_count` signers with total stake `total_stake`, whose
/// `validator_set_root(epoch)` is `validator_set_root`.
///
/// `total_stake` is the denominator of the certificate thresholds; a
/// verifier that opens only some signers gets it from here.
pub fn verify_signer(
    validator_set_root: &Hash,
    epoch: u64,
    total_stake: u64,
    signer: &Signer,
    proof: &InclusionProof,
) -> bool {
    proof
        .root_from_leaf_hash(TreeId::VALIDATOR_SET, signer.leaf_hash())
        .is_some_and(|signer_root| {
            validator_set_root_from_signer_root(epoch, total_stake, &signer_root)
                == *validator_set_root
        })
}

#[cfg(test)]
mod tests {
    use {super::*, crate::commitment_tree::compute_root};

    fn hash_from_hex(hex: &str) -> Hash {
        let bytes: [u8; 32] = hex::decode(hex).unwrap().try_into().unwrap();
        Hash::new_from_array(bytes)
    }

    fn signer(bls: u8, stake: u64, vote: u8) -> Signer {
        Signer {
            bls_pubkey_compressed: [bls; BLS_PUBKEY_COMPRESSED_LEN],
            stake,
            vote_pubkey: [vote; VOTE_PUBKEY_LEN],
        }
    }

    fn sorted(mut signers: Vec<Signer>) -> Vec<Signer> {
        signers.sort_by(Signer::cmp_signer_order);
        signers
    }

    /// Signer `i` of the 2,000 member vector set.
    fn large_set_signer(i: u16) -> Signer {
        let mut bls_pubkey_compressed = [0x5a; BLS_PUBKEY_COMPRESSED_LEN];
        bls_pubkey_compressed[..2].copy_from_slice(&i.to_be_bytes());
        let mut vote_pubkey = [0xc3; VOTE_PUBKEY_LEN];
        vote_pubkey[..2].copy_from_slice(&i.to_be_bytes());
        Signer {
            bls_pubkey_compressed,
            stake: 3_000_000 - u64::from(i) * 1_000,
            vote_pubkey,
        }
    }

    // Vectors computed with an independent Python implementation of the
    // construction in the SIMD (hashlib.sha256 over the byte strings), for
    // the sets built in `test_vectors`.
    const VECTOR_EMPTY: &str = "d1bd5e8d0a7a2a04723548dea90dc7211fcf772d4e389df630e18f558837827a";
    const VECTOR_ONE: &str = "515d899ba02a234d2880a39193b14fe9e1bda49247c9fe67fc92115ca36a7f2e";
    const VECTOR_TWO: &str = "4ac17f65f77c8c46d7a42d9b2a696201e04e6295d5dd2e3b2f494f92c96af298";
    const VECTOR_THREE_WITH_TIE: &str =
        "a7d6be01f0f5c9768d9c2505b508429a22f3ef8139465fcc383d2e6a51a3782f";
    const VECTOR_2000: &str = "40524879fde9935f6226b1b9f500034aa40b883dc5b4997891e09b02f366499e";

    #[test]
    fn test_signer_leaf_layout() {
        let bls = [0x11; BLS_PUBKEY_COMPRESSED_LEN];
        let vote = [0x22; VOTE_PUBKEY_LEN];
        let leaf = signer_leaf(&bls, 0x0102_0304_0506_0708, &vote);
        assert_eq!(leaf.len(), 88);
        assert_eq!(&leaf[..48], &bls);
        assert_eq!(&leaf[48..56], &0x0102_0304_0506_0708u64.to_le_bytes());
        assert_eq!(&leaf[56..], &vote);
        let signer = Signer {
            bls_pubkey_compressed: bls,
            stake: 0x0102_0304_0506_0708,
            vote_pubkey: vote,
        };
        assert_eq!(signer.leaf(), leaf);
        assert_eq!(signer.leaf_hash(), hashv(&[&[0x00, 0x06], leaf.as_slice()]));
    }

    #[test]
    fn test_vectors() {
        assert_eq!(
            validator_set_root(5, &[]).unwrap(),
            hash_from_hex(VECTOR_EMPTY)
        );
        assert_eq!(
            SignerSetCommitment::empty().validator_set_root(5),
            hash_from_hex(VECTOR_EMPTY)
        );
        assert_eq!(
            validator_set_root(7, &[signer(0x11, 1_000, 0x21)]).unwrap(),
            hash_from_hex(VECTOR_ONE)
        );
        let two = [signer(0x12, 500, 0x22), signer(0x13, 300, 0x23)];
        assert_eq!(
            validator_set_root(8, &two).unwrap(),
            hash_from_hex(VECTOR_TWO)
        );

        // Two signers with equal stake are ordered by compressed BLS key.
        let three = sorted(vec![
            signer(0x02, 300, 0x31),
            signer(0x09, 100, 0x32),
            signer(0x01, 300, 0x33),
        ]);
        assert_eq!(
            three
                .iter()
                .map(|s| s.bls_pubkey_compressed[0])
                .collect::<Vec<_>>(),
            vec![0x01, 0x02, 0x09]
        );
        assert_eq!(
            validator_set_root(9, &three).unwrap(),
            hash_from_hex(VECTOR_THREE_WITH_TIE)
        );

        let large: Vec<Signer> = (0..2000).map(large_set_signer).collect();
        assert_eq!(sorted(large.clone()), large);
        let commitment = SignerSetCommitment::new(&large).unwrap();
        assert_eq!(commitment.signer_count, 2000);
        assert_eq!(commitment.total_stake, 4_001_000_000);
        assert_eq!(
            commitment.validator_set_root(1234),
            hash_from_hex(VECTOR_2000)
        );
    }

    #[test]
    fn test_root_binds_epoch_total_stake_and_order() {
        let signers = sorted(vec![
            signer(1, 100, 1),
            signer(2, 200, 2),
            signer(3, 300, 3),
        ]);
        let commitment = SignerSetCommitment::new(&signers).unwrap();
        assert_eq!(commitment.total_stake, 600);
        assert_eq!(
            commitment.signer_root,
            compute_root(
                TreeId::VALIDATOR_SET,
                &signers.iter().map(Signer::leaf).collect::<Vec<_>>()
            )
        );
        let root = commitment.validator_set_root(10);
        assert_eq!(
            root,
            hashv(&[
                &[0x82],
                &10u64.to_le_bytes(),
                &600u64.to_le_bytes(),
                commitment.signer_root.as_ref()
            ])
        );
        // Another epoch number gives another root for the same set.
        assert_ne!(root, commitment.validator_set_root(11));
        // Another order gives another root.
        let mut reversed = signers.clone();
        reversed.reverse();
        assert_ne!(validator_set_root(10, &reversed).unwrap(), root);
        // Another stake gives another root.
        let mut changed = signers.clone();
        changed[2].stake += 1;
        assert_ne!(validator_set_root(10, &changed).unwrap(), root);
        // A wrong total stake does not match.
        assert_ne!(
            validator_set_root_from_signer_root(10, 601, &commitment.signer_root),
            root
        );
    }

    #[test]
    fn test_total_stake_overflow() {
        assert!(SignerSetCommitment::new(&[signer(1, u64::MAX, 1), signer(2, 1, 2)]).is_none());
        assert!(validator_set_root(0, &[signer(1, u64::MAX, 1)]).is_some());
    }

    #[test]
    fn test_signer_proofs() {
        for n in [1usize, 2, 3, 5, 8, 13, 100] {
            let signers: Vec<Signer> = (0..n as u16).map(large_set_signer).collect();
            let commitment = SignerSetCommitment::new(&signers).unwrap();
            let epoch = 42;
            let root = commitment.validator_set_root(epoch);
            for (i, s) in signers.iter().enumerate() {
                let proof = signer_inclusion_proof(&signers, i).unwrap();
                assert_eq!(proof.leaf_index, i as u64);
                assert_eq!(proof.leaf_count, n as u64);
                assert!(verify_signer(
                    &root,
                    epoch,
                    commitment.total_stake,
                    s,
                    &proof
                ));
                // Wrong epoch, total stake, signer data or index.
                assert!(!verify_signer(
                    &root,
                    epoch + 1,
                    commitment.total_stake,
                    s,
                    &proof
                ));
                assert!(!verify_signer(
                    &root,
                    epoch,
                    commitment.total_stake - 1,
                    s,
                    &proof
                ));
                let mut wrong_signer = *s;
                wrong_signer.stake += 1;
                assert!(!verify_signer(
                    &root,
                    epoch,
                    commitment.total_stake,
                    &wrong_signer,
                    &proof
                ));
                if n > 1 {
                    let mut wrong_index = proof.clone();
                    wrong_index.leaf_index = ((i + 1) % n) as u64;
                    assert!(!verify_signer(
                        &root,
                        epoch,
                        commitment.total_stake,
                        s,
                        &wrong_index
                    ));
                }
            }
            assert!(signer_inclusion_proof(&signers, n).is_none());
        }
    }
}
