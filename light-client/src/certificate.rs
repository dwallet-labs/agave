//! Alpenglow certificates and the finality proof of a block.
//!
//! A certificate is an aggregate BLS signature over one serialized vote and a
//! bitmap of the signer indices (ranks) that contributed to it. Verification
//! follows the validator's (`Bank::verify_certificate`, which calls
//! `agave_bls_cert_verify::verify_certificate`) for the three certificate
//! types that show finality:
//!
//! | Certificate     | Signed vote                      | Threshold |
//! |-----------------|----------------------------------|-----------|
//! | Notarize        | notarization vote `(slot, block_id)` | 60%   |
//! | Finalize        | finalization vote `(slot)`       | 60%       |
//! | Fast finalize   | notarization vote `(slot, block_id)` | 80%   |
//!
//! The signed vote includes the cluster's shred version. The bitmap must be
//! the base-2 encoding of `solana-signer-store`, decoded with the size of the
//! signer set as the maximum length. The stake of the signers is summed with
//! saturation and compared with the threshold as `signed / total >= pct /
//! 100`, without rounding.

use {
    crate::{error::Error, signer_set::SignerSet},
    solana_bls_signatures::{
        PubkeyProjective, Signature as BlsSignature, VerifySignature,
        pubkey::{PopVerified, PubkeyAffine},
    },
    solana_epoch_schedule::EpochSchedule,
    solana_hash::Hash,
    solana_signer_store::{Decoded, decode},
};

/// Size of a serialized notarization vote: tag, slot, block id, shred version.
pub const NOTARIZATION_VOTE_LEN: usize = 1 + 8 + 32 + 2;
/// Size of a serialized finalization vote: tag, slot, shred version.
pub const FINALIZATION_VOTE_LEN: usize = 1 + 8 + 2;

/// Wire tag of a notarization vote (`VotePayloadToSign::Notar`).
const NOTARIZATION_VOTE_TAG: u8 = 1;
/// Wire tag of a finalization vote (`VotePayloadToSign::Finalize`).
const FINALIZATION_VOTE_TAG: u8 = 2;

/// The bytes a validator signs for a notarization vote of `(slot,
/// block_id)`: `1 || u64_le(slot) || block_id || u16_le(shred_version)`, the
/// wincode encoding of `VotePayloadToSign::Notar`.
pub fn notarization_vote_payload(
    slot: u64,
    block_id: &Hash,
    shred_version: u16,
) -> [u8; NOTARIZATION_VOTE_LEN] {
    let mut payload = [0u8; NOTARIZATION_VOTE_LEN];
    payload[0] = NOTARIZATION_VOTE_TAG;
    payload[1..9].copy_from_slice(&slot.to_le_bytes());
    payload[9..41].copy_from_slice(block_id.as_ref());
    payload[41..].copy_from_slice(&shred_version.to_le_bytes());
    payload
}

/// The bytes a validator signs for a finalization vote of `slot`:
/// `2 || u64_le(slot) || u16_le(shred_version)`, the wincode encoding of
/// `VotePayloadToSign::Finalize`.
pub fn finalization_vote_payload(slot: u64, shred_version: u16) -> [u8; FINALIZATION_VOTE_LEN] {
    let mut payload = [0u8; FINALIZATION_VOTE_LEN];
    payload[0] = FINALIZATION_VOTE_TAG;
    payload[1..9].copy_from_slice(&slot.to_le_bytes());
    payload[9..].copy_from_slice(&shred_version.to_le_bytes());
    payload
}

/// The type of a certificate that can be part of a finality proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CertificateKind {
    /// Notarization of `(slot, block_id)`, 60% of stake.
    Notarize,
    /// Finalization of `slot`, 60% of stake.
    Finalize,
    /// Fast finalization of `(slot, block_id)`, 80% of stake.
    FastFinalize,
}

impl CertificateKind {
    /// Percentage of the signer set's stake the certificate needs
    /// (`CertificateType::threshold`).
    pub const fn threshold_percent(self) -> u64 {
        match self {
            Self::Notarize | Self::Finalize => 60,
            Self::FastFinalize => 80,
        }
    }
}

/// One certificate: an aggregate signature and its signer bitmap.
///
/// The slot and block id it is for are not part of the certificate; they are
/// given to the verifier, which rebuilds the signed vote from them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Certificate {
    pub kind: CertificateKind,
    /// Aggregate signature, uncompressed affine encoding (as in the
    /// validator's `Certificate::signature`).
    pub signature: BlsSignature,
    /// Signer bitmap in the `solana-signer-store` encoding.
    pub bitmap: Vec<u8>,
}

impl Certificate {
    /// Verifies this certificate for `(slot, block_id)` against `signer_set`:
    /// the aggregate signature over the vote the certificate type signs, and
    /// the stake threshold. Does not check that `slot` is in the signer set's
    /// epoch; [`FinalityProof::verify`] does.
    pub fn verify(
        &self,
        signer_set: &SignerSet,
        shred_version: u16,
        slot: u64,
        block_id: &Hash,
    ) -> Result<(), Error> {
        let kind = self.kind;
        let ranks = match decode(&self.bitmap, signer_set.len()) {
            Ok(Decoded::Base2(ranks)) => ranks,
            // These certificate types have no fallback vote: the validator
            // rejects a base-3 bitmap for them (`Error::WrongEncoding`).
            Ok(Decoded::Base3(..)) | Err(_) => return Err(Error::InvalidBitmap { kind }),
        };

        let mut signed_stake = 0u64;
        let mut pubkeys: Vec<&PopVerified<PubkeyAffine>> = Vec::with_capacity(ranks.count_ones());
        for rank in ranks.iter_ones() {
            let (stake, pubkey) = signer_set.rank(rank).ok_or(Error::InvalidBitmap { kind })?;
            signed_stake = signed_stake.saturating_add(stake);
            pubkeys.push(pubkey);
        }

        // An empty bitmap fails here (no key to aggregate), as for the
        // validator.
        let aggregate = PubkeyProjective::aggregate(pubkeys.into_iter())
            .map_err(|_| Error::InvalidSignature { kind })?;
        let verified = match kind {
            CertificateKind::Notarize | CertificateKind::FastFinalize => aggregate
                .verify_signature(
                    &self.signature,
                    &notarization_vote_payload(slot, block_id, shred_version),
                ),
            CertificateKind::Finalize => aggregate.verify_signature(
                &self.signature,
                &finalization_vote_payload(slot, shred_version),
            ),
        };
        verified.map_err(|_| Error::InvalidSignature { kind })?;

        let total_stake = signer_set.total_stake();
        let threshold_percent = kind.threshold_percent();
        // signed / total >= pct / 100, cross-multiplied in u128 as the
        // validator's `Fraction` does. Neither product can overflow.
        let signed = u128::from(signed_stake).saturating_mul(100);
        let required = u128::from(total_stake).saturating_mul(u128::from(threshold_percent));
        if signed >= required {
            Ok(())
        } else {
            Err(Error::InsufficientStake {
                kind,
                signed_stake,
                total_stake,
                threshold_percent,
            })
        }
    }
}

/// Certificates showing that a block is final: one fast-finalization
/// certificate, or one notarization certificate and one finalization
/// certificate for the same slot (in either order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinalityProof {
    pub certificates: Vec<Certificate>,
}

impl FinalityProof {
    /// A fast finalization.
    pub fn fast(signature: BlsSignature, bitmap: Vec<u8>) -> Self {
        Self {
            certificates: vec![Certificate {
                kind: CertificateKind::FastFinalize,
                signature,
                bitmap,
            }],
        }
    }

    /// A slow finalization: notarization and finalization certificates.
    pub fn slow(notarize: (BlsSignature, Vec<u8>), finalize: (BlsSignature, Vec<u8>)) -> Self {
        Self {
            certificates: vec![
                Certificate {
                    kind: CertificateKind::Notarize,
                    signature: notarize.0,
                    bitmap: notarize.1,
                },
                Certificate {
                    kind: CertificateKind::Finalize,
                    signature: finalize.0,
                    bitmap: finalize.1,
                },
            ],
        }
    }

    /// Verifies that the block `(slot, block_id)` is final.
    ///
    /// `signer_set` must be the set of the epoch that contains `slot`
    /// according to `epoch_schedule`, as validators select it.
    pub fn verify(
        &self,
        signer_set: &SignerSet,
        epoch_schedule: &EpochSchedule,
        shred_version: u16,
        slot: u64,
        block_id: &Hash,
    ) -> Result<(), Error> {
        let slot_epoch = epoch_schedule.get_epoch(slot);
        if slot_epoch != signer_set.epoch() {
            return Err(Error::EpochMismatch {
                slot,
                slot_epoch,
                signer_set_epoch: signer_set.epoch(),
            });
        }
        use CertificateKind::*;
        match self.certificates.as_slice() {
            [fast] if fast.kind == FastFinalize => {
                fast.verify(signer_set, shred_version, slot, block_id)
            }
            [a, b]
                if matches!(
                    (a.kind, b.kind),
                    (Notarize, Finalize) | (Finalize, Notarize)
                ) =>
            {
                a.verify(signer_set, shred_version, slot, block_id)?;
                b.verify(signer_set, shred_version, slot, block_id)
            }
            [single] if matches!(single.kind, Notarize | Finalize) => Err(Error::NotFinal),
            _ => Err(Error::UnexpectedCertificates),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use {
        super::*,
        crate::signer_set::tests::test_signers,
        agave_bls_cert_verify::cert_verify::verify_certificate as validator_verify_certificate,
        agave_votor_messages::{
            certificate::CertificateType,
            consensus_message::{Block, BlockId},
            stake::NonZeroStake,
            unverified_vote_message::UnverifiedCertificate,
            vote::Vote,
            wire::get_vote_payload_to_sign,
        },
        bitvec::vec::BitVec,
        solana_bls_signatures::{SignatureProjective, keypair::Keypair as BlsKeypair},
        solana_signer_store::{encode_base2, encode_base3},
    };

    const SHRED_VERSION: u16 = 0x1234;
    const SLOT: u64 = 1_000;

    fn epoch_schedule() -> EpochSchedule {
        EpochSchedule::custom(432, 432, false)
    }

    struct Setup {
        keypairs: Vec<BlsKeypair>,
        signer_set: SignerSet,
        block_id: Hash,
    }

    /// Ten signers with stakes 19, 17, ..., 1 (total 100), for the epoch of
    /// `SLOT`.
    fn setup() -> Setup {
        let stakes: Vec<u64> = (0..10).map(|i| 19 - 2 * i).collect();
        let (keypairs, signers) = test_signers(&stakes);
        let epoch = epoch_schedule().get_epoch(SLOT);
        Setup {
            keypairs,
            signer_set: SignerSet::new(epoch, signers).unwrap(),
            block_id: Hash::new_from_array([7; 32]),
        }
    }

    fn base2_bitmap(len: usize, ranks: &[usize]) -> Vec<u8> {
        let mut bits = BitVec::repeat(false, len);
        for rank in ranks {
            bits.set(*rank, true);
        }
        encode_base2(&bits).unwrap()
    }

    fn aggregate_sign(keypairs: &[BlsKeypair], ranks: &[usize], payload: &[u8]) -> BlsSignature {
        let mut signature = SignatureProjective::identity();
        for rank in ranks {
            signature
                .aggregate_with(std::iter::once(&keypairs[*rank].sign(payload)))
                .unwrap();
        }
        signature.into()
    }

    fn certificate(
        setup: &Setup,
        kind: CertificateKind,
        ranks: &[usize],
        shred_version: u16,
    ) -> Certificate {
        let payload = match kind {
            CertificateKind::Notarize | CertificateKind::FastFinalize => {
                notarization_vote_payload(SLOT, &setup.block_id, shred_version).to_vec()
            }
            CertificateKind::Finalize => finalization_vote_payload(SLOT, shred_version).to_vec(),
        };
        Certificate {
            kind,
            signature: aggregate_sign(&setup.keypairs, ranks, &payload),
            bitmap: base2_bitmap(setup.signer_set.len(), ranks),
        }
    }

    /// The validator's verdict on `certificate` for `(SLOT, block_id)`.
    fn validator_accepts(setup: &Setup, certificate: &Certificate, shred_version: u16) -> bool {
        let block = Block {
            slot: SLOT,
            block_id: BlockId::from(setup.block_id),
        };
        let cert_type = match certificate.kind {
            CertificateKind::Notarize => CertificateType::Notarize(block),
            CertificateKind::Finalize => CertificateType::Finalize(SLOT),
            CertificateKind::FastFinalize => CertificateType::FinalizeFast(block),
        };
        let signers = setup.signer_set.signers();
        let total_stake = NonZeroStake::new(setup.signer_set.total_stake()).unwrap();
        validator_verify_certificate(
            UnverifiedCertificate {
                cert_type,
                signature: certificate.signature,
                bitmap: certificate.bitmap.clone(),
                shred_version,
            },
            signers.len(),
            total_stake,
            |rank| {
                let signer = signers.get(rank)?;
                let pubkey = setup
                    .keypairs
                    .iter()
                    .find(|kp| kp.public.to_bytes_compressed() == signer.bls_pubkey_compressed)?
                    .public;
                Some((NonZeroStake::new(signer.stake)?, pubkey))
            },
            None,
        )
        .is_ok()
    }

    /// Checks our verdict on `certificate` and that the validator's agrees.
    fn check(setup: &Setup, certificate: &Certificate, expected: Result<(), Error>) {
        let ours = certificate.verify(&setup.signer_set, SHRED_VERSION, SLOT, &setup.block_id);
        assert_eq!(ours, expected, "{certificate:?}");
        assert_eq!(
            validator_accepts(setup, certificate, SHRED_VERSION),
            expected.is_ok(),
            "validator disagrees on {certificate:?}"
        );
    }

    #[test]
    fn test_vote_payloads_match_validator_encoding() {
        for (slot, block_id, shred_version) in [
            (0, Hash::default(), 0),
            (42, Hash::new_from_array([0xab; 32]), 0x1234),
            (u64::MAX, Hash::new_from_array([0xff; 32]), u16::MAX),
        ] {
            let block = Block {
                slot,
                block_id: BlockId::from(block_id),
            };
            assert_eq!(
                notarization_vote_payload(slot, &block_id, shred_version).as_slice(),
                get_vote_payload_to_sign(Vote::new_notarization_vote(block), shred_version)
            );
            assert_eq!(
                finalization_vote_payload(slot, shred_version).as_slice(),
                get_vote_payload_to_sign(Vote::new_finalization_vote(slot), shred_version)
            );
        }
    }

    #[test]
    fn test_certificate_thresholds() {
        let setup = setup();
        // Stakes 19, 17, 15, 13 = 64 >= 60, below 80.
        let ranks_64 = [0, 1, 2, 3];
        // 19 + 17 + 15 = 51 < 60.
        let ranks_51 = [0, 1, 2];
        // 19 + 17 + 15 + 13 + 11 + 5 = 80, exactly the fast threshold.
        let ranks_80 = [0, 1, 2, 3, 4, 7];
        // 79: one stake short.
        let ranks_79 = [0, 1, 2, 3, 4, 8, 9];

        for kind in [CertificateKind::Notarize, CertificateKind::Finalize] {
            check(
                &setup,
                &certificate(&setup, kind, &ranks_64, SHRED_VERSION),
                Ok(()),
            );
            check(
                &setup,
                &certificate(&setup, kind, &ranks_51, SHRED_VERSION),
                Err(Error::InsufficientStake {
                    kind,
                    signed_stake: 51,
                    total_stake: 100,
                    threshold_percent: 60,
                }),
            );
        }
        let kind = CertificateKind::FastFinalize;
        check(
            &setup,
            &certificate(&setup, kind, &ranks_80, SHRED_VERSION),
            Ok(()),
        );
        for (ranks, signed_stake) in [(&ranks_79[..], 79), (&ranks_64[..], 64)] {
            check(
                &setup,
                &certificate(&setup, kind, ranks, SHRED_VERSION),
                Err(Error::InsufficientStake {
                    kind,
                    signed_stake,
                    total_stake: 100,
                    threshold_percent: 80,
                }),
            );
        }
    }

    #[test]
    fn test_certificate_rejections() {
        let setup = setup();
        let all: Vec<usize> = (0..10).collect();
        let kind = CertificateKind::Notarize;
        let good = certificate(&setup, kind, &all, SHRED_VERSION);
        check(&setup, &good, Ok(()));

        // Signed with another shred version.
        check(
            &setup,
            &certificate(&setup, kind, &all, SHRED_VERSION + 1),
            Err(Error::InvalidSignature { kind }),
        );

        // Bitmap claims a signer that did not sign.
        let mut tampered = certificate(&setup, kind, &all[..9], SHRED_VERSION);
        tampered.bitmap = base2_bitmap(10, &all);
        check(&setup, &tampered, Err(Error::InvalidSignature { kind }));

        // Bitmap drops a signer that did sign.
        let mut tampered = good.clone();
        tampered.bitmap = base2_bitmap(10, &all[1..]);
        check(&setup, &tampered, Err(Error::InvalidSignature { kind }));

        // One bit flipped in the encoded bitmap.
        let mut tampered = good.clone();
        *tampered.bitmap.last_mut().unwrap() ^= 0x01;
        check(&setup, &tampered, Err(Error::InvalidSignature { kind }));

        // Bitmap longer than the signer set.
        let mut tampered = good.clone();
        tampered.bitmap = base2_bitmap(11, &all);
        check(&setup, &tampered, Err(Error::InvalidBitmap { kind }));

        // Truncated and malformed encodings.
        for bitmap in [
            vec![],
            vec![0],
            good.bitmap[..good.bitmap.len() - 1].to_vec(),
        ] {
            let mut tampered = good.clone();
            tampered.bitmap = bitmap;
            check(&setup, &tampered, Err(Error::InvalidBitmap { kind }));
        }
        let mut tampered = good.clone();
        tampered.bitmap[0] = 7;
        check(&setup, &tampered, Err(Error::InvalidBitmap { kind }));

        // Base-3 encoding is rejected for these certificate types.
        let mut bits = BitVec::repeat(false, 10);
        for rank in &all {
            bits.set(*rank, true);
        }
        let mut tampered = good.clone();
        tampered.bitmap = encode_base3(&bits, &BitVec::repeat(false, 10)).unwrap();
        check(&setup, &tampered, Err(Error::InvalidBitmap { kind }));

        // Empty bitmap.
        let mut tampered = good.clone();
        tampered.bitmap = base2_bitmap(10, &[]);
        check(&setup, &tampered, Err(Error::InvalidSignature { kind }));

        // A shorter bitmap is accepted, as by the validator.
        let short = Certificate {
            kind,
            signature: aggregate_sign(
                &setup.keypairs,
                &[0, 1, 2, 3],
                &notarization_vote_payload(SLOT, &setup.block_id, SHRED_VERSION),
            ),
            bitmap: base2_bitmap(4, &[0, 1, 2, 3]),
        };
        check(&setup, &short, Ok(()));

        // A notarization signature is not a finalization signature.
        let mut wrong_kind = good.clone();
        wrong_kind.kind = CertificateKind::Finalize;
        check(
            &setup,
            &wrong_kind,
            Err(Error::InvalidSignature {
                kind: CertificateKind::Finalize,
            }),
        );

        // Another block id or slot.
        assert_eq!(
            good.verify(
                &setup.signer_set,
                SHRED_VERSION,
                SLOT,
                &Hash::new_from_array([8; 32])
            ),
            Err(Error::InvalidSignature { kind })
        );
        assert_eq!(
            good.verify(&setup.signer_set, SHRED_VERSION, SLOT + 1, &setup.block_id),
            Err(Error::InvalidSignature { kind })
        );
    }

    #[test]
    fn test_finality_proof() {
        let setup = setup();
        let schedule = epoch_schedule();
        let all: Vec<usize> = (0..10).collect();
        let verify = |proof: &FinalityProof| {
            proof.verify(
                &setup.signer_set,
                &schedule,
                SHRED_VERSION,
                SLOT,
                &setup.block_id,
            )
        };
        let notarize = certificate(&setup, CertificateKind::Notarize, &all, SHRED_VERSION);
        let finalize = certificate(&setup, CertificateKind::Finalize, &all, SHRED_VERSION);
        let fast = certificate(&setup, CertificateKind::FastFinalize, &all, SHRED_VERSION);

        let slow = FinalityProof::slow(
            (notarize.signature, notarize.bitmap.clone()),
            (finalize.signature, finalize.bitmap.clone()),
        );
        assert_eq!(verify(&slow), Ok(()));
        let swapped = FinalityProof {
            certificates: vec![finalize.clone(), notarize.clone()],
        };
        assert_eq!(verify(&swapped), Ok(()));
        assert_eq!(
            verify(&FinalityProof::fast(fast.signature, fast.bitmap.clone())),
            Ok(())
        );

        // A notarization certificate alone is not final, whatever its stake;
        // nor is a finalization certificate alone.
        for single in [&notarize, &finalize] {
            assert_eq!(
                verify(&FinalityProof {
                    certificates: vec![single.clone()]
                }),
                Err(Error::NotFinal)
            );
        }
        for certificates in [
            vec![],
            vec![notarize.clone(), notarize.clone()],
            vec![fast.clone(), fast.clone()],
            vec![fast.clone(), notarize.clone()],
            vec![notarize.clone(), finalize.clone(), fast.clone()],
        ] {
            assert_eq!(
                verify(&FinalityProof { certificates }),
                Err(Error::UnexpectedCertificates)
            );
        }

        // Slow finalization needs both certificates to verify.
        let weak_finalize = certificate(&setup, CertificateKind::Finalize, &[0, 1], SHRED_VERSION);
        assert_eq!(
            verify(&FinalityProof {
                certificates: vec![notarize.clone(), weak_finalize]
            }),
            Err(Error::InsufficientStake {
                kind: CertificateKind::Finalize,
                signed_stake: 36,
                total_stake: 100,
                threshold_percent: 60,
            })
        );

        // A fast-finalization certificate with notarization stake only.
        let weak_fast = certificate(
            &setup,
            CertificateKind::FastFinalize,
            &[0, 1, 2, 3],
            SHRED_VERSION,
        );
        assert!(matches!(
            verify(&FinalityProof {
                certificates: vec![weak_fast]
            }),
            Err(Error::InsufficientStake { .. })
        ));

        // The signer set of another epoch.
        let other_epoch = SignerSet::new(
            setup.signer_set.epoch() + 1,
            setup.signer_set.signers().to_vec(),
        )
        .unwrap();
        assert_eq!(
            slow.verify(
                &other_epoch,
                &schedule,
                SHRED_VERSION,
                SLOT,
                &setup.block_id
            ),
            Err(Error::EpochMismatch {
                slot: SLOT,
                slot_epoch: setup.signer_set.epoch(),
                signer_set_epoch: setup.signer_set.epoch() + 1,
            })
        );
        // The epoch is computed with the epoch schedule.
        assert_eq!(
            slow.verify(
                &setup.signer_set,
                &EpochSchedule::custom(32, 32, false),
                SHRED_VERSION,
                SLOT,
                &setup.block_id
            ),
            Err(Error::EpochMismatch {
                slot: SLOT,
                slot_epoch: SLOT / 32,
                signer_set_epoch: setup.signer_set.epoch(),
            })
        );
    }
}
