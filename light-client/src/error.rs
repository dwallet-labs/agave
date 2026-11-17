//! The error type of every verifier in this crate.

use {
    crate::{certificate::CertificateKind, commitment::CommitmentSlot},
    solana_merkle_tree::accounts_state::AccountProofError,
};

/// Why a proof, a certificate or a signer set is rejected.
///
/// Verifiers never panic on proof data; every rejection is one of these.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    // Signer set.
    /// A signer set must have at least one signer with stake.
    #[error("the signer set is empty")]
    EmptySignerSet,
    /// Validators give no signer index to a vote account without stake.
    #[error("signer {index} has zero stake")]
    ZeroStake { index: usize },
    /// The total stake does not fit in a `u64`.
    #[error("the total stake of the signer set overflows u64")]
    StakeOverflow,
    /// The compressed BLS public key is not a valid G1 point; validators give
    /// no signer index to such a key.
    #[error("signer {index} has an invalid BLS public key")]
    InvalidBlsPubkey { index: usize },
    /// Signers must be in signer index order: stake descending, then
    /// compressed BLS public key ascending, without duplicates.
    #[error("signer {index} is not in signer index order")]
    SignerOrder { index: usize },
    /// A BLS public key or vote account appears twice; validators give it
    /// at most one signer index.
    #[error("signer {index} repeats a BLS public key or vote account")]
    DuplicateSigner { index: usize },
    /// The signer set does not hash to the expected `validator_set_root`.
    #[error("the signer set does not match the validator set root of epoch {epoch}")]
    ValidatorSetRootMismatch { epoch: u64 },

    // Finality proof.
    /// The slot of the certificate is not in the epoch of the signer set.
    #[error("slot {slot} is in epoch {slot_epoch}, the signer set is for epoch {signer_set_epoch}")]
    EpochMismatch {
        slot: u64,
        slot_epoch: u64,
        signer_set_epoch: u64,
    },
    /// A notarization certificate or a finalization certificate alone does
    /// not show that the block is final.
    #[error("the certificates do not show that the block is final")]
    NotFinal,
    /// The proof is neither one fast-finalization certificate nor one
    /// notarization and one finalization certificate.
    #[error("the finality proof does not have the certificates of a fast or a slow finalization")]
    UnexpectedCertificates,
    /// The signer bitmap does not decode as a base-2 bitmap over the signer
    /// set.
    #[error("{kind:?} certificate: invalid signer bitmap")]
    InvalidBitmap { kind: CertificateKind },
    /// The aggregate signature does not verify for the signers in the bitmap.
    #[error("{kind:?} certificate: invalid aggregate signature")]
    InvalidSignature { kind: CertificateKind },
    /// The signers in the bitmap do not hold the certificate's threshold.
    #[error(
        "{kind:?} certificate: signed stake {signed_stake} of {total_stake} is below \
         {threshold_percent}%"
    )]
    InsufficientStake {
        kind: CertificateKind,
        signed_stake: u64,
        total_stake: u64,
        threshold_percent: u64,
    },

    // Footer proof.
    /// `fec_set_count` must be between 1 and `MAX_FEC_SETS_PER_SLOT`.
    #[error("invalid FEC set count {0}")]
    InvalidFecSetCount(u32),
    /// The path does not have the length of a tree with `fec_set_count + 1`
    /// leaves.
    #[error("the footer proof has {actual} entries, expected {expected}")]
    FooterProofLength { expected: usize, actual: usize },
    /// The final leaf and its path do not lead to the block id.
    #[error("the footer proof does not lead to the block id")]
    FooterProofMismatch,
    /// The parent named by the footer proof is not older than the block.
    #[error("parent slot {parent_slot} is not before slot {slot}")]
    ParentNotBefore { slot: u64, parent_slot: u64 },

    // Commitment vector.
    /// The claimed value and siblings do not lead to `commitment_root`.
    #[error("the opening of commitment slot {0:?} does not lead to the commitment root")]
    SlotOpening(CommitmentSlot),

    // Block history.
    /// The history leaf and its path do not lead to `block_history_root`, or
    /// a zero commitment root is claimed at an index other than 0.
    #[error("the history path does not lead to the block history root")]
    HistoryProof,
    /// The proven leaf is the anchor leaf, which has no commitment root.
    #[error("the history leaf is the anchor leaf")]
    AnchorLeaf,
    /// The proven leaf is not the anchor leaf.
    #[error("the history leaf is not the anchor leaf")]
    NotAnchorLeaf,
    /// An ancestor must have a smaller slot than the block whose history
    /// holds it.
    #[error("ancestor slot {ancestor_slot} is not before slot {slot}")]
    AncestorNotBefore { slot: u64, ancestor_slot: u64 },

    // Transaction receipts.
    /// The receipt is not for the requested transaction signature.
    #[error("the receipt is for another signature")]
    SignatureMismatch,
    /// Only receipts of version 1 are defined.
    #[error("unsupported receipt version {0}")]
    UnsupportedReceiptVersion(u8),
    /// The receipt and its path do not lead to `receipts_root`.
    #[error("the receipt path does not lead to the receipts root")]
    ReceiptProof,
    /// A failed transaction commits no instruction trace and no events.
    #[error("the transaction failed; it has no instruction trace or events")]
    TransactionFailed,
    /// The record and its path do not lead to `instruction_trace_root`.
    #[error("the instruction record path does not lead to the instruction trace root")]
    InstructionProof,
    /// The event and its path do not lead to `events_root`.
    #[error("the event path does not lead to the events root")]
    EventProof,
    /// The event's `instruction_index` or `program_id` is not that of the
    /// proven instruction.
    #[error("the event was not emitted by the proven instruction")]
    EventInstructionMismatch,

    // Accounts.
    /// A state proof was asked for the zero account hash, which means "no
    /// account"; use the deletion or non-existence verifier.
    #[error("the zero account hash means no account")]
    ZeroAccountHash,
    /// The delta leaf is not for the address, does not carry the claimed
    /// hash, or its path does not lead to `accounts_delta_root`.
    #[error("the accounts delta proof does not show the claimed state")]
    AccountDeltaProof,
    /// The adjacency proof does not show that the address is not a leaf.
    #[error("the proof does not show that the account is unchanged in the block")]
    AccountUnchangedProof,
    /// The accounts state proof is rejected by the overlay or base tree
    /// verifier.
    #[error("the accounts state proof is rejected: {0:?}")]
    AccountStateProof(AccountProofError),

    // Epochs.
    /// The block is not in the epoch the proof requires.
    #[error("the block at slot {slot} is in epoch {block_epoch}, expected epoch {expected}")]
    BlockEpoch {
        slot: u64,
        block_epoch: u64,
        expected: u64,
    },
    /// The next signer set is not for the epoch after the block's epoch.
    #[error("the next signer set is for epoch {actual}, expected epoch {expected}")]
    NextEpoch { expected: u64, actual: u64 },
    /// The six inputs of `epoch_boundary` are not consistent with each other,
    /// with the epoch schedule or with the block.
    #[error("the epoch boundary inputs are inconsistent")]
    InvalidEpochBoundary,
}
