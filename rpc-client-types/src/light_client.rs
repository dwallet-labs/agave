//! JSON forms of the light-client proofs served by `getFinalityProof`,
//! `getBlockHistoryProof`, `getTransactionProof`, `getAccountDeltaProof`,
//! `getAccountStateProof`, `getValidatorSet` and `getEpochBoundary`.
//!
//! Encodings, as elsewhere in the RPC: hashes, addresses, transaction
//! signatures, compressed BLS public keys and shred Merkle proof entries are
//! base-58 strings; BLS signatures, signer bitmaps and other byte strings are
//! base-64 strings.
//!
//! These types only carry data. `solana-light-client` (feature `rpc`)
//! converts them losslessly to and from the types its verifiers take.

use {
    serde::{Deserialize, Serialize},
    solana_clock::{Epoch, Slot},
};

// ---------------------------------------------------------------------------
// Request configurations.

/// Which instruction or event proofs `getTransactionProof` includes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RpcProofSelection {
    /// Every record (or event) of the transaction: `"all"`.
    All,
    /// The records (or events) at these positions: `{"indices": [..]}`.
    /// Instruction positions are in execution order, event positions in
    /// emission order.
    Indices(Vec<u64>),
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RpcTransactionProofConfig {
    /// Instruction proofs to include.
    pub instructions: Option<RpcProofSelection>,
    /// Event proofs to include. The proof of the instruction that emitted
    /// each selected event is included as well.
    pub events: Option<RpcProofSelection>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RpcAccountStateProofConfig {
    /// Block whose state is proven. Default: the latest finalized block.
    pub slot: Option<Slot>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RpcCommittingBlockConfig {
    /// Block whose commitment opens the requested value. Default: the
    /// latest finalized block of the right epoch that this node has.
    pub at_slot: Option<Slot>,
}

// ---------------------------------------------------------------------------
// Building blocks.

/// One Alpenglow certificate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcCertificate {
    /// `"notarize"`, `"finalize"` or `"fastFinalize"`.
    pub kind: RpcCertificateKind,
    /// Aggregate BLS signature, uncompressed (base-64).
    pub signature: String,
    /// Signer bitmap in the `solana-signer-store` encoding (base-64).
    pub bitmap: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RpcCertificateKind {
    Notarize,
    Finalize,
    FastFinalize,
}

/// Path from the final leaf of a block id tree to the block id, with the
/// final leaf's contents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcFooterProof {
    pub parent_slot: Slot,
    pub parent_block_id: String,
    pub fec_set_count: u32,
    pub bank_hash: String,
    pub commitment_root: String,
    /// 20-byte entries, leaf first.
    pub proof: Vec<String>,
}

/// An inclusion proof in a commitment tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcInclusionProof {
    pub leaf_index: u64,
    pub leaf_count: u64,
    /// Leaf first.
    pub siblings: Vec<String>,
}

/// History leaf of an ancestor and its path, from a block's commitment root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcHistoryProof {
    pub block_history_root: String,
    /// Opening of commitment slot 0: three sibling hashes, leaf first.
    pub opening: Vec<String>,
    pub slot: Slot,
    pub block_id: String,
    /// Zero (32 zero bytes, base-58) for the anchor leaf.
    pub commitment_root: String,
    pub proof: RpcInclusionProof,
}

// ---------------------------------------------------------------------------
// getFinalityProof

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcFinalityProof {
    pub slot: Slot,
    pub block_id: String,
    /// Footer proof of this block.
    pub footer: RpcFooterProof,
    /// The certificates that finalize this block, if this node has them.
    pub certificates: Option<Vec<RpcCertificate>>,
    /// Otherwise (the block was finalized through a descendant, or its
    /// certificates are gone): the nearest later finalized block for which
    /// this node has certificates, and the path from it to this block.
    pub finalized_descendant: Option<RpcFinalizedDescendant>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcFinalizedDescendant {
    pub slot: Slot,
    pub block_id: String,
    pub certificates: Vec<RpcCertificate>,
    pub footer: RpcFooterProof,
    /// History proof of the requested block from this block.
    pub history: RpcHistoryProof,
}

// ---------------------------------------------------------------------------
// getBlockHistoryProof

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcBlockHistoryProof {
    /// The block whose history is opened.
    pub at_slot: Slot,
    pub at_block_id: String,
    /// The ancestor.
    pub history: RpcHistoryProof,
}

// ---------------------------------------------------------------------------
// getTransactionProof

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcTransactionReceipt {
    pub version: u8,
    /// `"success"` or `"failed"`.
    pub status: RpcReceiptStatus,
    /// First signature of the transaction (base-58).
    pub signature: String,
    pub message_hash: String,
    pub fee: u64,
    pub instruction_trace_root: String,
    pub events_root: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RpcReceiptStatus {
    Success,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcInstructionRecord {
    pub stack_height: u8,
    pub program_id: String,
    pub accounts_hash: String,
    pub data_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcInstructionProof {
    pub record: RpcInstructionRecord,
    /// Path in the instruction trace; the leaf index is the record's
    /// position in execution order.
    pub proof: RpcInclusionProof,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcEventRecord {
    pub program_id: String,
    /// Position, in the instruction trace, of the emitting instruction.
    pub instruction_index: u32,
    /// `"logData"` or `"log"`.
    pub kind: RpcEventKind,
    /// Base-64.
    pub payload: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RpcEventKind {
    LogData,
    Log,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcEventProof {
    pub event: RpcEventRecord,
    /// Path in the events tree; the leaf index is the event's position in
    /// emission order.
    pub proof: RpcInclusionProof,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcTransactionProof {
    pub slot: Slot,
    pub block_id: String,
    pub receipts_root: String,
    /// Opening of commitment slot 1.
    pub opening: Vec<String>,
    pub receipt: RpcTransactionReceipt,
    /// Path in the receipts tree; the leaf index is the transaction's
    /// position in the block.
    pub proof: RpcInclusionProof,
    /// Requested instruction proofs, and those of the instructions that
    /// emitted the requested events, in execution order.
    pub instructions: Vec<RpcInstructionProof>,
    /// Requested event proofs, in emission order.
    pub events: Vec<RpcEventProof>,
}

// ---------------------------------------------------------------------------
// getAccountDeltaProof

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcDeltaLeafProof {
    pub address: String,
    /// Zero (base-58 of 32 zero bytes) if the block deleted the account.
    pub account_hash: String,
    pub proof: RpcInclusionProof,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RpcUnchangedProof {
    EmptyTree,
    BeforeFirst {
        first: RpcDeltaLeafProof,
    },
    AfterLast {
        last: RpcDeltaLeafProof,
    },
    Between {
        left: RpcDeltaLeafProof,
        right: RpcDeltaLeafProof,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RpcAccountDelta {
    /// The block changed the account: its leaf.
    Changed(RpcDeltaLeafProof),
    /// The block did not change the account.
    Unchanged(RpcUnchangedProof),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcAccountDeltaProof {
    pub slot: Slot,
    pub block_id: String,
    pub accounts_delta_root: String,
    /// Opening of commitment slot 2.
    pub opening: Vec<String>,
    pub delta: RpcAccountDelta,
}

// ---------------------------------------------------------------------------
// getAccountStateProof

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcMembershipProof {
    pub entry_count: u64,
    pub path: Vec<RpcKeyedPathEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcKeyedPathEntry {
    pub bit: u16,
    pub sibling: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RpcOverlayAbsence {
    EmptyTree,
    Neighbor {
        key: String,
        /// `now || base` account hashes of the neighbor (base-64, 64 bytes).
        value: String,
        proof: RpcMembershipProof,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcBaseLeafProof {
    pub address: String,
    pub account_hash: String,
    pub proof: RpcInclusionProof,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RpcBaseAbsence {
    EmptyTree,
    BeforeFirst {
        first: RpcBaseLeafProof,
    },
    AfterLast {
        last: RpcBaseLeafProof,
    },
    Between {
        left: RpcBaseLeafProof,
        right: RpcBaseLeafProof,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RpcBaseLookup {
    Present(RpcBaseLeafProof),
    Absent(RpcBaseAbsence),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RpcAccountLookup {
    /// The account changed since the base state.
    Overlay {
        address: String,
        /// Account hash now (zero: no account).
        now: String,
        /// Account hash in the base state (zero: no account).
        base: String,
        proof: RpcMembershipProof,
    },
    /// The account did not change since the base state.
    Base {
        overlay_absence: RpcOverlayAbsence,
        base: RpcBaseLookup,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcAccountStateProof {
    pub slot: Slot,
    pub block_id: String,
    /// Epoch of the base state.
    pub generation: Epoch,
    pub base_root: String,
    pub overlay_root: String,
    /// Path of `accounts_state_root` (slot 5) in the commitment vector.
    pub slot_proof: RpcInclusionProof,
    pub lookup: RpcAccountLookup,
}

// ---------------------------------------------------------------------------
// getValidatorSet

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcSigner {
    pub bls_pubkey_compressed: String,
    pub stake: u64,
    pub vote_pubkey: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcValidatorSet {
    pub epoch: Epoch,
    /// The signer set in signer index (rank) order.
    pub signers: Vec<RpcSigner>,
    pub validator_set_root: String,
    /// A finalized block of epoch `epoch - 1` that commits to this set, and
    /// the opening of its commitment slot 3. `None` for the set of an epoch
    /// whose previous epoch has no such block on this node (the client must
    /// then get the set from a trusted source).
    pub committed_by: Option<RpcSlotOpeningAt>,
}

/// One commitment slot of a block, opened.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcSlotOpeningAt {
    pub slot: Slot,
    pub block_id: String,
    pub opening: Vec<String>,
}

// ---------------------------------------------------------------------------
// getEpochBoundary

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcEpochBoundaryInputs {
    pub epoch: Epoch,
    pub first_slot: Slot,
    pub parent_slot: Slot,
    pub parent_block_id: String,
    /// Zero if the parent predates Block Commitment Root.
    pub parent_commitment_root: String,
    pub history_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcEpochBoundary {
    pub slot: Slot,
    pub block_id: String,
    pub boundary: RpcEpochBoundaryInputs,
    /// Opening of commitment slot 4.
    pub opening: Vec<String>,
}

#[cfg(test)]
mod tests {
    use {super::*, serde_json::json};

    #[test]
    fn test_proof_selection_json() {
        assert_eq!(
            serde_json::to_value(RpcProofSelection::All).unwrap(),
            json!("all")
        );
        assert_eq!(
            serde_json::to_value(RpcProofSelection::Indices(vec![0, 2])).unwrap(),
            json!({"indices": [0, 2]})
        );
        let config: RpcTransactionProofConfig =
            serde_json::from_value(json!({"instructions": "all", "events": {"indices": [1]}}))
                .unwrap();
        assert_eq!(
            config,
            RpcTransactionProofConfig {
                instructions: Some(RpcProofSelection::All),
                events: Some(RpcProofSelection::Indices(vec![1])),
            }
        );
        assert!(
            serde_json::from_value::<RpcTransactionProofConfig>(json!({"instruction": "all"}))
                .is_err()
        );
    }

    #[test]
    fn test_certificate_kind_json() {
        assert_eq!(
            serde_json::to_value(RpcCertificateKind::FastFinalize).unwrap(),
            json!("fastFinalize")
        );
        assert_eq!(
            serde_json::to_value(RpcAccountDelta::Unchanged(RpcUnchangedProof::EmptyTree)).unwrap(),
            json!({"unchanged": "emptyTree"})
        );
    }
}
