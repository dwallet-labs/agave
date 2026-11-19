#![cfg(feature = "agave-unstable-api")]
//! Client-side verification of the light-client commitments: the Block
//! Commitment Root SIMD and its companions (Transaction Receipt Tree,
//! Accounts Delta Root, Epoch Validator Set Root, Epoch Boundary Commitment,
//! Accounts State Root).
//!
//! The core of this crate only verifies: it does no networking and fetches
//! no proof. With the `fetch` feature, [`fetch::ProofFetcher`] fetches the
//! proofs it needs through a transport the caller provides and verifies
//! them; the `rpc-client` feature provides that transport over JSON-RPC.
//!
//! # From a certificate to an item
//!
//! 1. [`LightClient::verify_block`]: a [`FinalityProof`] (one fast
//!    finalization certificate, or a notarization certificate and a
//!    finalization certificate) for `(slot, block_id)`, checked with the
//!    [`SignerSet`] of the slot's epoch, and a [`FooterProof`] from
//!    `block_id` to the block's `bank_hash` and `commitment_root`. The result
//!    is a [`VerifiedBlock`].
//! 2. Optionally, a [`HistoryProof`] from the verified block to an ancestor
//!    ([`VerifiedAncestor`]), through slot 0 of the commitment vector.
//! 3. One proof per question, each opening one slot of the block's
//!    commitment vector ([`SlotOpening`]) and then one tree:
//!    - [`TransactionProof`]: a transaction's receipt (status, fee) by
//!      signature (slot 1);
//!    - [`InstructionProof`]: an instruction it executed (slot 1);
//!    - [`EventProof`]: an event an instruction emitted (slot 1);
//!    - [`AccountDeltaProof`]: the state of an account the block changed, or
//!      its deletion (slot 2);
//!    - [`AccountUnchangedProof`]: the block did not change an account
//!      (slot 2);
//!    - [`AccountStateProof`]: the current state of an account, or that it
//!      does not exist (slot 5);
//!    - [`verify_next_signer_set`]: the signer set of the next epoch (slot 3),
//!      used by [`LightClient::advance_epoch`];
//!    - [`EpochBoundaryProof`]: where the previous epoch ended (slot 4).
//!
//! Every verifier returns a typed [`Error`]; none panics on proof data.
//!
//! # Trust
//!
//! A verified block is as trustworthy as the Alpenglow assumption for its
//! epoch (less than 20% of stake Byzantine). The starting signer set must
//! come from a trusted source and must not be older than the weak
//! subjectivity period. Proofs show what a block committed to; they do not
//! show that no newer block exists, so callers apply their own freshness
//! rule.

mod account;
mod block_id;
mod certificate;
mod client;
mod commitment;
mod error;
#[cfg(feature = "fetch")]
pub mod fetch;
mod proofs;
#[cfg(feature = "rpc")]
pub mod rpc;
mod signer_set;

pub use {
    account::account_hash,
    block_id::{
        FooterProof, MAX_FEC_SETS_PER_SLOT, MERKLE_PROOF_ENTRY_LEN, MerkleProofEntry, final_leaf,
        merkle_root, proof_len,
    },
    certificate::{
        Certificate, CertificateKind, FINALIZATION_VOTE_LEN, FinalityProof, NOTARIZATION_VOTE_LEN,
        finalization_vote_payload, notarization_vote_payload,
    },
    client::LightClient,
    commitment::{
        BlockRef, COMMITMENT_VECTOR_LEN, CommitmentSlot, CommittedBlock, HISTORY_LEAF_LEN,
        HistoryProof, SLOT_OPENING_LEN, SlotOpening, VerifiedAncestor, VerifiedBlock, history_leaf,
    },
    error::Error,
    proofs::{
        AccountDeltaProof, AccountStateProof, AccountUnchangedProof, EpochBoundaryProof,
        EventProof, InstructionProof, TransactionProof, VerifiedEpochBoundary, VerifiedEvent,
        VerifiedInstruction, VerifiedTransaction, verify_next_signer_set,
    },
    signer_set::SignerSet,
    // The crates whose types appear in the API: the trees and leaves proofs
    // are made of, the BLS signature and the epoch schedule.
    solana_bls_signatures::Signature as BlsSignature,
    solana_epoch_schedule::EpochSchedule,
    solana_merkle_tree,
};
