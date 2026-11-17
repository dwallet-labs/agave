//! The signer set of one epoch: who signs Alpenglow certificates for the
//! slots of that epoch, and with how much stake.

use {
    crate::error::Error,
    solana_bls_signatures::pubkey::{PopVerified, PubkeyAffine, PubkeyCompressed},
    solana_hash::Hash,
    solana_merkle_tree::validator_set::{Signer, SignerSetCommitment},
    std::{cmp::Ordering, collections::HashSet},
};

/// The signer set of an epoch, in signer index (rank) order: signer `i` is
/// the validator that bit `i` of a certificate bitmap refers to.
///
/// This is the set validators hold in their rank map
/// (`BLSPubkeyToRankMap`) for the epoch, as committed by
/// `validator_set_root(epoch)` (Epoch Validator Set Root SIMD). The total
/// stake is the sum over the set, which is the denominator validators use for
/// certificate thresholds.
///
/// The BLS public keys are decompressed once, when the set is built. They are
/// used for aggregation without a proof-of-possession check, exactly as
/// validators use the keys of their rank map: possession is proven when a key
/// is registered in a vote account (SIMD-0387). A set obtained from
/// [`Self::from_root`] inherits that guarantee from the root it matches; a set
/// passed to [`Self::new`] directly is trusted as given.
#[derive(Clone, Debug)]
pub struct SignerSet {
    epoch: u64,
    signers: Vec<Signer>,
    pubkeys: Vec<PopVerified<PubkeyAffine>>,
    commitment: SignerSetCommitment,
}

impl SignerSet {
    /// Builds the signer set of `epoch` from its signers in rank order.
    ///
    /// Rejects what no validator's rank map contains: an empty set, a signer
    /// with zero stake or an invalid BLS public key, signers out of rank
    /// order (stake descending, then compressed key ascending), a BLS key or
    /// vote account used twice, and a total stake that overflows `u64`.
    pub fn new(epoch: u64, signers: Vec<Signer>) -> Result<Self, Error> {
        if signers.is_empty() {
            return Err(Error::EmptySignerSet);
        }
        let mut pubkeys = Vec::with_capacity(signers.len());
        let mut bls_keys = HashSet::with_capacity(signers.len());
        let mut vote_keys = HashSet::with_capacity(signers.len());
        for (index, signer) in signers.iter().enumerate() {
            if signer.stake == 0 {
                return Err(Error::ZeroStake { index });
            }
            if let Some(previous) = index.checked_sub(1).and_then(|i| signers.get(i))
                && previous.cmp_signer_order(signer) != Ordering::Less
            {
                return Err(Error::SignerOrder { index });
            }
            if !bls_keys.insert(signer.bls_pubkey_compressed)
                || !vote_keys.insert(signer.vote_pubkey)
            {
                return Err(Error::DuplicateSigner { index });
            }
            let pubkey = PubkeyAffine::try_from(PubkeyCompressed(signer.bls_pubkey_compressed))
                .map_err(|_| Error::InvalidBlsPubkey { index })?;
            // SAFETY: proof of possession of every key of a signer set is
            // checked when the key is registered in its vote account
            // (SIMD-0387). Validators wrap the keys of their rank map the same
            // way (`bls_pubkey_compressed_bytes_to_bls_pubkey` in the
            // runtime), and the set either matches a `validator_set_root`
            // finalized by the previous signer set or is trusted by the
            // caller (see the type documentation).
            pubkeys.push(unsafe { PopVerified::new_unchecked(pubkey) });
        }
        let commitment = SignerSetCommitment::new(&signers).ok_or(Error::StakeOverflow)?;
        Ok(Self {
            epoch,
            signers,
            pubkeys,
            commitment,
        })
    }

    /// Builds the signer set of `epoch` and checks it against
    /// `validator_set_root`, e.g. the value of commitment slot 3 of a
    /// verified block of epoch `epoch - 1`.
    pub fn from_root(
        epoch: u64,
        signers: Vec<Signer>,
        validator_set_root: &Hash,
    ) -> Result<Self, Error> {
        let signer_set = Self::new(epoch, signers)?;
        signer_set.verify_root(validator_set_root)?;
        Ok(signer_set)
    }

    /// The epoch whose certificates this set verifies.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The signers, in rank order.
    pub fn signers(&self) -> &[Signer] {
        &self.signers
    }

    /// Number of signers.
    pub fn len(&self) -> usize {
        self.signers.len()
    }

    /// Always false: an empty set is rejected by [`Self::new`].
    pub fn is_empty(&self) -> bool {
        self.signers.is_empty()
    }

    /// Sum of the stake of all signers: the certificate threshold
    /// denominator.
    pub fn total_stake(&self) -> u64 {
        self.commitment.total_stake
    }

    /// `validator_set_root(epoch)` of this set.
    pub fn validator_set_root(&self) -> Hash {
        self.commitment.validator_set_root(self.epoch)
    }

    /// Checks that this set is the one committed by `validator_set_root`.
    /// The root binds the epoch number, so a set with the right signers but
    /// the wrong epoch is rejected.
    pub fn verify_root(&self, validator_set_root: &Hash) -> Result<(), Error> {
        if self.validator_set_root() == *validator_set_root {
            Ok(())
        } else {
            Err(Error::ValidatorSetRootMismatch { epoch: self.epoch })
        }
    }

    /// Stake and public key of the signer with rank `rank`.
    pub(crate) fn rank(&self, rank: usize) -> Option<(u64, &PopVerified<PubkeyAffine>)> {
        let stake = self.signers.get(rank)?.stake;
        Some((stake, self.pubkeys.get(rank)?))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use {
        super::*, solana_bls_signatures::keypair::Keypair as BlsKeypair,
        solana_merkle_tree::validator_set::validator_set_root,
    };

    /// `n` BLS keypairs and their signers, in rank order, with the given
    /// stakes (which must be descending).
    pub(crate) fn test_signers(stakes: &[u64]) -> (Vec<BlsKeypair>, Vec<Signer>) {
        let mut keyed: Vec<(BlsKeypair, Signer)> = stakes
            .iter()
            .enumerate()
            .map(|(i, stake)| {
                let keypair = BlsKeypair::new();
                let signer = Signer {
                    bls_pubkey_compressed: keypair.public.to_bytes_compressed(),
                    stake: *stake,
                    vote_pubkey: [i as u8; 32],
                };
                (keypair, signer)
            })
            .collect();
        keyed.sort_by(|(_, a), (_, b)| a.cmp_signer_order(b));
        keyed.into_iter().unzip()
    }

    #[test]
    fn test_signer_set_root() {
        let (_, signers) = test_signers(&[50, 30, 20]);
        let set = SignerSet::new(7, signers.clone()).unwrap();
        assert_eq!(set.epoch(), 7);
        assert_eq!(set.len(), 3);
        assert_eq!(set.total_stake(), 100);
        let root = validator_set_root(7, &signers).unwrap();
        assert_eq!(set.validator_set_root(), root);
        assert_eq!(set.verify_root(&root), Ok(()));
        assert!(SignerSet::from_root(7, signers.clone(), &root).is_ok());
        // The root binds the epoch number.
        let other_epoch = validator_set_root(8, &signers).unwrap();
        assert_eq!(
            set.verify_root(&other_epoch),
            Err(Error::ValidatorSetRootMismatch { epoch: 7 })
        );
        assert_eq!(
            SignerSet::from_root(8, signers, &root).unwrap_err(),
            Error::ValidatorSetRootMismatch { epoch: 8 }
        );
    }

    #[test]
    fn test_signer_set_rejects_invalid_sets() {
        assert_eq!(
            SignerSet::new(0, vec![]).unwrap_err(),
            Error::EmptySignerSet
        );

        let (_, signers) = test_signers(&[50, 30, 20]);
        let mut zero = signers.clone();
        zero[2].stake = 0;
        assert_eq!(
            SignerSet::new(0, zero).unwrap_err(),
            Error::ZeroStake { index: 2 }
        );

        let mut reversed = signers.clone();
        reversed.reverse();
        assert_eq!(
            SignerSet::new(0, reversed).unwrap_err(),
            Error::SignerOrder { index: 1 }
        );

        let duplicate = vec![signers[0], signers[0]];
        assert_eq!(
            SignerSet::new(0, duplicate).unwrap_err(),
            Error::SignerOrder { index: 1 }
        );
        // The same BLS key or vote account with another stake.
        let mut same_key = signers.clone();
        same_key[2].bls_pubkey_compressed = same_key[0].bls_pubkey_compressed;
        assert_eq!(
            SignerSet::new(0, same_key).unwrap_err(),
            Error::DuplicateSigner { index: 2 }
        );
        let mut same_vote = signers.clone();
        same_vote[1].vote_pubkey = same_vote[0].vote_pubkey;
        assert_eq!(
            SignerSet::new(0, same_vote).unwrap_err(),
            Error::DuplicateSigner { index: 1 }
        );

        let mut invalid_key = signers.clone();
        invalid_key[1].bls_pubkey_compressed = [0xff; 48];
        assert_eq!(
            SignerSet::new(0, invalid_key).unwrap_err(),
            Error::InvalidBlsPubkey { index: 1 }
        );

        let (_, mut overflow) = test_signers(&[u64::MAX, 1]);
        overflow.sort_by(|a, b| a.cmp_signer_order(b));
        assert_eq!(
            SignerSet::new(0, overflow).unwrap_err(),
            Error::StakeOverflow
        );
    }
}
