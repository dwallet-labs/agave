//! Conversions between the proofs of this crate and their JSON forms served
//! over RPC ([`solana_rpc_client_types::light_client`]), in both directions:
//! the server encodes with `From`, a client decodes with `TryFrom` and
//! verifies.
//!
//! ```text
//! let response: RpcFinalityProof = /* getFinalityProof(slot) */;
//! let finality = FinalityResponse::try_from(&response)?;
//! let block = finality.verify(&client, slot)?;
//! let response: RpcTransactionProof = /* getTransactionProof(signature) */;
//! let transaction = TransactionResponse::try_from(&response)?;
//! let verified = transaction.transaction.verify(&block, &signature)?;
//! ```
//!
//! The responses that name a block (`slot`, `block_id`) are verified
//! against a [`CommittedBlock`] the client has verified itself, from
//! [`FinalityResponse::verify`] for that block or a history proof. Decoding
//! checks encodings and lengths only; every claim is checked by the
//! verifiers.

use {
    crate::{
        block_id::{FooterProof, MERKLE_PROOF_ENTRY_LEN, MerkleProofEntry},
        certificate::{Certificate, CertificateKind, FinalityProof},
        client::LightClient,
        commitment::{
            CommitmentSlot, CommittedBlock, HistoryProof, SLOT_OPENING_LEN, SlotOpening,
            VerifiedAncestor, VerifiedBlock,
        },
        error::Error,
        proofs::{
            AccountDeltaProof, AccountStateProof, AccountUnchangedProof, EpochBoundaryProof,
            EventProof, InstructionProof, TransactionProof, verify_next_signer_set,
        },
        signer_set::SignerSet,
    },
    base64::{Engine, prelude::BASE64_STANDARD},
    solana_bls_signatures::Signature as BlsSignature,
    solana_epoch_schedule::EpochSchedule,
    solana_hash::Hash,
    solana_merkle_tree::{
        accounts_delta::{DeltaEntry, DeltaLeafProof, UnchangedProof},
        accounts_state::{
            AccountLookup, AccountProof, AccountsStateOpening, BaseAbsenceProof, BaseLeaf,
            BaseLeafProof, BaseLookup, OVERLAY_VALUE_LEN, OverlayEntry, OverlayValue,
        },
        commitment_tree::InclusionProof,
        epoch_boundary::EpochBoundary,
        keyed_tree::{MembershipProof, NonMembershipProof},
        receipt_tree::{
            EventKind, EventRecord, InstructionRecord, ReceiptStatus, TransactionReceipt,
        },
        validator_set::Signer,
    },
    solana_rpc_client_types::light_client::*,
};

/// A JSON proof that does not decode: a field with a bad encoding or length,
/// or parts that do not fit together.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid {field}: {reason}")]
pub struct WireError {
    pub field: &'static str,
    pub reason: String,
}

impl WireError {
    fn new(field: &'static str, reason: impl Into<String>) -> Self {
        Self {
            field,
            reason: reason.into(),
        }
    }
}

type WireResult<T> = Result<T, WireError>;

// ---------------------------------------------------------------------------
// Encodings.

fn b58(bytes: impl AsRef<[u8]>) -> String {
    bs58::encode(bytes).into_string()
}

fn b64(bytes: impl AsRef<[u8]>) -> String {
    BASE64_STANDARD.encode(bytes)
}

fn from_b58<const N: usize>(field: &'static str, value: &str) -> WireResult<[u8; N]> {
    let bytes = bs58::decode(value)
        .into_vec()
        .map_err(|err| WireError::new(field, format!("not base-58: {err}")))?;
    let len = bytes.len();
    bytes
        .try_into()
        .map_err(|_| WireError::new(field, format!("{len} bytes, expected {N}")))
}

fn from_b64(field: &'static str, value: &str) -> WireResult<Vec<u8>> {
    BASE64_STANDARD
        .decode(value)
        .map_err(|err| WireError::new(field, format!("not base-64: {err}")))
}

fn hash(field: &'static str, value: &str) -> WireResult<Hash> {
    from_b58::<32>(field, value).map(Hash::new_from_array)
}

fn hashes(field: &'static str, values: &[String]) -> WireResult<Vec<Hash>> {
    values.iter().map(|value| hash(field, value)).collect()
}

fn encode_hashes(values: &[Hash]) -> Vec<String> {
    values.iter().map(b58).collect()
}

fn inclusion_proof(field: &'static str, proof: &RpcInclusionProof) -> WireResult<InclusionProof> {
    Ok(InclusionProof {
        leaf_index: proof.leaf_index,
        leaf_count: proof.leaf_count,
        siblings: hashes(field, &proof.siblings)?,
    })
}

fn encode_inclusion_proof(proof: &InclusionProof) -> RpcInclusionProof {
    RpcInclusionProof {
        leaf_index: proof.leaf_index,
        leaf_count: proof.leaf_count,
        siblings: encode_hashes(&proof.siblings),
    }
}

fn slot_opening(field: &'static str, siblings: &[String]) -> WireResult<SlotOpening> {
    let siblings: [Hash; SLOT_OPENING_LEN] =
        hashes(field, siblings)?
            .try_into()
            .map_err(|siblings: Vec<_>| {
                WireError::new(
                    field,
                    format!("{} siblings, expected {SLOT_OPENING_LEN}", siblings.len()),
                )
            })?;
    Ok(SlotOpening { siblings })
}

fn encode_slot_opening(opening: &SlotOpening) -> Vec<String> {
    encode_hashes(&opening.siblings)
}

// ---------------------------------------------------------------------------
// Certificates and footer proofs.

impl From<&Certificate> for RpcCertificate {
    fn from(certificate: &Certificate) -> Self {
        Self {
            kind: match certificate.kind {
                CertificateKind::Notarize => RpcCertificateKind::Notarize,
                CertificateKind::Finalize => RpcCertificateKind::Finalize,
                CertificateKind::FastFinalize => RpcCertificateKind::FastFinalize,
            },
            signature: b64(certificate.signature.0),
            bitmap: b64(&certificate.bitmap),
        }
    }
}

impl TryFrom<&RpcCertificate> for Certificate {
    type Error = WireError;

    fn try_from(certificate: &RpcCertificate) -> WireResult<Self> {
        let signature = from_b64("certificate signature", &certificate.signature)?;
        let len = signature.len();
        let signature = BlsSignature(signature.try_into().map_err(|_| {
            WireError::new(
                "certificate signature",
                format!("{len} bytes, expected an uncompressed signature"),
            )
        })?);
        Ok(Self {
            kind: match certificate.kind {
                RpcCertificateKind::Notarize => CertificateKind::Notarize,
                RpcCertificateKind::Finalize => CertificateKind::Finalize,
                RpcCertificateKind::FastFinalize => CertificateKind::FastFinalize,
            },
            signature,
            bitmap: from_b64("certificate bitmap", &certificate.bitmap)?,
        })
    }
}

fn finality_proof(certificates: &[RpcCertificate]) -> WireResult<FinalityProof> {
    Ok(FinalityProof {
        certificates: certificates
            .iter()
            .map(Certificate::try_from)
            .collect::<WireResult<_>>()?,
    })
}

fn encode_finality_proof(proof: &FinalityProof) -> Vec<RpcCertificate> {
    proof
        .certificates
        .iter()
        .map(RpcCertificate::from)
        .collect()
}

impl From<&FooterProof> for RpcFooterProof {
    fn from(footer: &FooterProof) -> Self {
        Self {
            parent_slot: footer.parent_slot,
            parent_block_id: b58(footer.parent_block_id),
            fec_set_count: footer.fec_set_count,
            bank_hash: b58(footer.bank_hash),
            commitment_root: b58(footer.commitment_root),
            proof: footer.proof.iter().map(b58).collect(),
        }
    }
}

impl TryFrom<&RpcFooterProof> for FooterProof {
    type Error = WireError;

    fn try_from(footer: &RpcFooterProof) -> WireResult<Self> {
        Ok(Self {
            parent_slot: footer.parent_slot,
            parent_block_id: hash("footer parent block id", &footer.parent_block_id)?,
            fec_set_count: footer.fec_set_count,
            bank_hash: hash("footer bank hash", &footer.bank_hash)?,
            commitment_root: hash("footer commitment root", &footer.commitment_root)?,
            proof: footer
                .proof
                .iter()
                .map(|entry| from_b58::<MERKLE_PROOF_ENTRY_LEN>("footer proof entry", entry))
                .collect::<WireResult<Vec<MerkleProofEntry>>>()?,
        })
    }
}

impl From<&HistoryProof> for RpcHistoryProof {
    fn from(proof: &HistoryProof) -> Self {
        Self {
            block_history_root: b58(proof.block_history_root),
            opening: encode_slot_opening(&proof.opening),
            slot: proof.slot,
            block_id: b58(proof.block_id),
            commitment_root: b58(proof.commitment_root),
            proof: encode_inclusion_proof(&proof.proof),
        }
    }
}

impl TryFrom<&RpcHistoryProof> for HistoryProof {
    type Error = WireError;

    fn try_from(proof: &RpcHistoryProof) -> WireResult<Self> {
        Ok(Self {
            block_history_root: hash("block history root", &proof.block_history_root)?,
            opening: slot_opening("history opening", &proof.opening)?,
            slot: proof.slot,
            block_id: hash("history block id", &proof.block_id)?,
            commitment_root: hash("history commitment root", &proof.commitment_root)?,
            proof: inclusion_proof("history proof", &proof.proof)?,
        })
    }
}

// ---------------------------------------------------------------------------
// getFinalityProof

/// `getFinalityProof`: a block, its footer proof, and how it is final.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinalityResponse {
    pub slot: u64,
    pub block_id: Hash,
    pub footer: FooterProof,
    pub finality: Finality,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Finality {
    /// The block's own certificates.
    Certificates(FinalityProof),
    /// The certificates of a later finalized block, and the history path
    /// from it to the block.
    Descendant(Box<FinalizedDescendant>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinalizedDescendant {
    pub slot: u64,
    pub block_id: Hash,
    pub finality: FinalityProof,
    pub footer: FooterProof,
    pub history: HistoryProof,
}

/// A block verified final, directly or through a finalized descendant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifiedFinal {
    Block(VerifiedBlock),
    Ancestor(VerifiedAncestor),
}

impl CommittedBlock for VerifiedFinal {
    fn slot(&self) -> u64 {
        match self {
            Self::Block(block) => block.slot(),
            Self::Ancestor(ancestor) => ancestor.slot(),
        }
    }

    fn block_id(&self) -> &Hash {
        match self {
            Self::Block(block) => block.block_id(),
            Self::Ancestor(ancestor) => ancestor.block_id(),
        }
    }

    fn commitment_root(&self) -> &Hash {
        match self {
            Self::Block(block) => block.commitment_root(),
            Self::Ancestor(ancestor) => ancestor.commitment_root(),
        }
    }
}

impl FinalityResponse {
    /// Verifies that the block at `slot` (the one requested) is final, and
    /// returns it with its commitment root.
    ///
    /// For a block finalized through a descendant: verifies the descendant
    /// with its certificates and footer, then the block as its ancestor,
    /// then the block's own footer against the block id and commitment root
    /// the history gives.
    pub fn verify(&self, client: &LightClient, slot: u64) -> Result<VerifiedFinal, Error> {
        if self.slot != slot {
            return Err(Error::WrongBlock);
        }
        match &self.finality {
            Finality::Certificates(finality) => client
                .verify_block(self.slot, &self.block_id, finality, &self.footer)
                .map(VerifiedFinal::Block),
            Finality::Descendant(descendant) => {
                if descendant.slot <= self.slot {
                    return Err(Error::WrongBlock);
                }
                let verified = client.verify_block(
                    descendant.slot,
                    &descendant.block_id,
                    &descendant.finality,
                    &descendant.footer,
                )?;
                let ancestor = client.verify_ancestor(&verified, &descendant.history)?;
                if ancestor.slot() != self.slot || *ancestor.block_id() != self.block_id {
                    return Err(Error::WrongBlock);
                }
                self.footer.verify(&self.block_id)?;
                if self.footer.commitment_root != *ancestor.commitment_root() {
                    return Err(Error::FooterProofMismatch);
                }
                Ok(VerifiedFinal::Ancestor(ancestor))
            }
        }
    }
}

impl From<&FinalityResponse> for RpcFinalityProof {
    fn from(response: &FinalityResponse) -> Self {
        let (certificates, finalized_descendant) = match &response.finality {
            Finality::Certificates(finality) => (Some(encode_finality_proof(finality)), None),
            Finality::Descendant(descendant) => (
                None,
                Some(RpcFinalizedDescendant {
                    slot: descendant.slot,
                    block_id: b58(descendant.block_id),
                    certificates: encode_finality_proof(&descendant.finality),
                    footer: RpcFooterProof::from(&descendant.footer),
                    history: RpcHistoryProof::from(&descendant.history),
                }),
            ),
        };
        Self {
            slot: response.slot,
            block_id: b58(response.block_id),
            footer: RpcFooterProof::from(&response.footer),
            certificates,
            finalized_descendant,
        }
    }
}

impl TryFrom<&RpcFinalityProof> for FinalityResponse {
    type Error = WireError;

    fn try_from(response: &RpcFinalityProof) -> WireResult<Self> {
        let finality = match (&response.certificates, &response.finalized_descendant) {
            (Some(certificates), None) => Finality::Certificates(finality_proof(certificates)?),
            (None, Some(descendant)) => Finality::Descendant(Box::new(FinalizedDescendant {
                slot: descendant.slot,
                block_id: hash("descendant block id", &descendant.block_id)?,
                finality: finality_proof(&descendant.certificates)?,
                footer: FooterProof::try_from(&descendant.footer)?,
                history: HistoryProof::try_from(&descendant.history)?,
            })),
            _ => {
                return Err(WireError::new(
                    "finality proof",
                    "exactly one of certificates and finalizedDescendant must be set",
                ));
            }
        };
        Ok(Self {
            slot: response.slot,
            block_id: hash("block id", &response.block_id)?,
            footer: FooterProof::try_from(&response.footer)?,
            finality,
        })
    }
}

// ---------------------------------------------------------------------------
// getBlockHistoryProof

/// `getBlockHistoryProof`: an ancestor of the block at `at_slot`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockHistoryResponse {
    pub at_slot: u64,
    pub at_block_id: Hash,
    pub history: HistoryProof,
}

impl From<&BlockHistoryResponse> for RpcBlockHistoryProof {
    fn from(response: &BlockHistoryResponse) -> Self {
        Self {
            at_slot: response.at_slot,
            at_block_id: b58(response.at_block_id),
            history: RpcHistoryProof::from(&response.history),
        }
    }
}

impl TryFrom<&RpcBlockHistoryProof> for BlockHistoryResponse {
    type Error = WireError;

    fn try_from(response: &RpcBlockHistoryProof) -> WireResult<Self> {
        Ok(Self {
            at_slot: response.at_slot,
            at_block_id: hash("block id", &response.at_block_id)?,
            history: HistoryProof::try_from(&response.history)?,
        })
    }
}

// ---------------------------------------------------------------------------
// getTransactionProof

fn encode_receipt(receipt: &TransactionReceipt) -> RpcTransactionReceipt {
    RpcTransactionReceipt {
        version: receipt.version,
        status: match receipt.status {
            ReceiptStatus::Success => RpcReceiptStatus::Success,
            ReceiptStatus::Failed => RpcReceiptStatus::Failed,
        },
        signature: b58(receipt.signature),
        message_hash: b58(receipt.message_hash),
        fee: receipt.fee,
        instruction_trace_root: b58(receipt.instruction_trace_root),
        events_root: b58(receipt.events_root),
    }
}

fn decode_receipt(receipt: &RpcTransactionReceipt) -> WireResult<TransactionReceipt> {
    Ok(TransactionReceipt {
        version: receipt.version,
        status: match receipt.status {
            RpcReceiptStatus::Success => ReceiptStatus::Success,
            RpcReceiptStatus::Failed => ReceiptStatus::Failed,
        },
        signature: from_b58("receipt signature", &receipt.signature)?,
        message_hash: hash("receipt message hash", &receipt.message_hash)?,
        fee: receipt.fee,
        instruction_trace_root: hash(
            "receipt instruction trace root",
            &receipt.instruction_trace_root,
        )?,
        events_root: hash("receipt events root", &receipt.events_root)?,
    })
}

fn encode_instruction_record(record: &InstructionRecord) -> RpcInstructionRecord {
    RpcInstructionRecord {
        stack_height: record.stack_height,
        program_id: b58(record.program_id),
        accounts_hash: b58(record.accounts_hash),
        data_hash: b58(record.data_hash),
    }
}

fn decode_instruction_record(record: &RpcInstructionRecord) -> WireResult<InstructionRecord> {
    Ok(InstructionRecord {
        stack_height: record.stack_height,
        program_id: from_b58("instruction program id", &record.program_id)?,
        accounts_hash: hash("instruction accounts hash", &record.accounts_hash)?,
        data_hash: hash("instruction data hash", &record.data_hash)?,
    })
}

fn encode_event_record(event: &EventRecord) -> RpcEventRecord {
    RpcEventRecord {
        program_id: b58(event.program_id),
        instruction_index: event.instruction_index,
        kind: match event.kind {
            EventKind::LogData => RpcEventKind::LogData,
            EventKind::Log => RpcEventKind::Log,
        },
        payload: b64(&event.payload),
    }
}

fn decode_event_record(event: &RpcEventRecord) -> WireResult<EventRecord> {
    Ok(EventRecord {
        program_id: from_b58("event program id", &event.program_id)?,
        instruction_index: event.instruction_index,
        kind: match event.kind {
            RpcEventKind::LogData => EventKind::LogData,
            RpcEventKind::Log => EventKind::Log,
        },
        payload: from_b64("event payload", &event.payload)?,
    })
}

/// `getTransactionProof`: a transaction's receipt in a block, and the
/// requested instructions and events.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransactionResponse {
    pub slot: u64,
    pub block_id: Hash,
    pub transaction: TransactionProof,
    /// Each with a copy of `transaction`.
    pub instructions: Vec<InstructionProof>,
    /// Each with a copy of the proof of the instruction that emitted it.
    pub events: Vec<EventProof>,
}

impl From<&TransactionResponse> for RpcTransactionProof {
    fn from(response: &TransactionResponse) -> Self {
        let transaction = &response.transaction;
        Self {
            slot: response.slot,
            block_id: b58(response.block_id),
            receipts_root: b58(transaction.receipts_root),
            opening: encode_slot_opening(&transaction.opening),
            receipt: encode_receipt(&transaction.receipt),
            proof: encode_inclusion_proof(&transaction.proof),
            instructions: response
                .instructions
                .iter()
                .map(|instruction| RpcInstructionProof {
                    record: encode_instruction_record(&instruction.record),
                    proof: encode_inclusion_proof(&instruction.proof),
                })
                .collect(),
            events: response
                .events
                .iter()
                .map(|event| RpcEventProof {
                    event: encode_event_record(&event.event),
                    proof: encode_inclusion_proof(&event.proof),
                })
                .collect(),
        }
    }
}

impl TryFrom<&RpcTransactionProof> for TransactionResponse {
    type Error = WireError;

    fn try_from(response: &RpcTransactionProof) -> WireResult<Self> {
        let transaction = TransactionProof {
            receipts_root: hash("receipts root", &response.receipts_root)?,
            opening: slot_opening("receipts opening", &response.opening)?,
            receipt: decode_receipt(&response.receipt)?,
            proof: inclusion_proof("receipt proof", &response.proof)?,
        };
        let instructions = response
            .instructions
            .iter()
            .map(|instruction| {
                Ok(InstructionProof {
                    transaction: transaction.clone(),
                    record: decode_instruction_record(&instruction.record)?,
                    proof: inclusion_proof("instruction proof", &instruction.proof)?,
                })
            })
            .collect::<WireResult<Vec<_>>>()?;
        let events = response
            .events
            .iter()
            .map(|event| {
                let record = decode_event_record(&event.event)?;
                let instruction = instructions
                    .iter()
                    .find(|instruction| {
                        instruction.proof.leaf_index == u64::from(record.instruction_index)
                    })
                    .ok_or_else(|| {
                        WireError::new(
                            "event",
                            format!(
                                "no proof of instruction {} that emitted it",
                                record.instruction_index
                            ),
                        )
                    })?
                    .clone();
                Ok(EventProof {
                    instruction,
                    event: record,
                    proof: inclusion_proof("event proof", &event.proof)?,
                })
            })
            .collect::<WireResult<Vec<_>>>()?;
        Ok(Self {
            slot: response.slot,
            block_id: hash("block id", &response.block_id)?,
            transaction,
            instructions,
            events,
        })
    }
}

// ---------------------------------------------------------------------------
// getAccountDeltaProof

fn encode_delta_leaf(leaf: &DeltaLeafProof) -> RpcDeltaLeafProof {
    RpcDeltaLeafProof {
        address: b58(leaf.entry.address),
        account_hash: b58(leaf.entry.account_hash),
        proof: encode_inclusion_proof(&leaf.proof),
    }
}

fn decode_delta_leaf(leaf: &RpcDeltaLeafProof) -> WireResult<DeltaLeafProof> {
    Ok(DeltaLeafProof {
        entry: DeltaEntry::new(
            from_b58("delta leaf address", &leaf.address)?,
            hash("delta leaf account hash", &leaf.account_hash)?,
        ),
        proof: inclusion_proof("delta leaf proof", &leaf.proof)?,
    })
}

fn encode_unchanged(proof: &UnchangedProof) -> RpcUnchangedProof {
    match proof {
        UnchangedProof::EmptyTree => RpcUnchangedProof::EmptyTree,
        UnchangedProof::BeforeFirst { first } => RpcUnchangedProof::BeforeFirst {
            first: encode_delta_leaf(first),
        },
        UnchangedProof::AfterLast { last } => RpcUnchangedProof::AfterLast {
            last: encode_delta_leaf(last),
        },
        UnchangedProof::Between { left, right } => RpcUnchangedProof::Between {
            left: encode_delta_leaf(left),
            right: encode_delta_leaf(right),
        },
    }
}

fn decode_unchanged(proof: &RpcUnchangedProof) -> WireResult<UnchangedProof> {
    Ok(match proof {
        RpcUnchangedProof::EmptyTree => UnchangedProof::EmptyTree,
        RpcUnchangedProof::BeforeFirst { first } => UnchangedProof::BeforeFirst {
            first: decode_delta_leaf(first)?,
        },
        RpcUnchangedProof::AfterLast { last } => UnchangedProof::AfterLast {
            last: decode_delta_leaf(last)?,
        },
        RpcUnchangedProof::Between { left, right } => UnchangedProof::Between {
            left: decode_delta_leaf(left)?,
            right: decode_delta_leaf(right)?,
        },
    })
}

/// `getAccountDeltaProof`: whether a block changed an account.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccountDeltaResponse {
    pub slot: u64,
    pub block_id: Hash,
    pub delta: AccountDelta,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AccountDelta {
    Changed(AccountDeltaProof),
    Unchanged(AccountUnchangedProof),
}

impl From<&AccountDeltaResponse> for RpcAccountDeltaProof {
    fn from(response: &AccountDeltaResponse) -> Self {
        let (accounts_delta_root, opening, delta) = match &response.delta {
            AccountDelta::Changed(proof) => (
                proof.accounts_delta_root,
                &proof.opening,
                RpcAccountDelta::Changed(encode_delta_leaf(&proof.leaf)),
            ),
            AccountDelta::Unchanged(proof) => (
                proof.accounts_delta_root,
                &proof.opening,
                RpcAccountDelta::Unchanged(encode_unchanged(&proof.proof)),
            ),
        };
        Self {
            slot: response.slot,
            block_id: b58(response.block_id),
            accounts_delta_root: b58(accounts_delta_root),
            opening: encode_slot_opening(opening),
            delta,
        }
    }
}

impl TryFrom<&RpcAccountDeltaProof> for AccountDeltaResponse {
    type Error = WireError;

    fn try_from(response: &RpcAccountDeltaProof) -> WireResult<Self> {
        let accounts_delta_root = hash("accounts delta root", &response.accounts_delta_root)?;
        let opening = slot_opening("accounts delta opening", &response.opening)?;
        let delta = match &response.delta {
            RpcAccountDelta::Changed(leaf) => AccountDelta::Changed(AccountDeltaProof {
                accounts_delta_root,
                opening,
                leaf: decode_delta_leaf(leaf)?,
            }),
            RpcAccountDelta::Unchanged(proof) => AccountDelta::Unchanged(AccountUnchangedProof {
                accounts_delta_root,
                opening,
                proof: decode_unchanged(proof)?,
            }),
        };
        Ok(Self {
            slot: response.slot,
            block_id: hash("block id", &response.block_id)?,
            delta,
        })
    }
}

// ---------------------------------------------------------------------------
// getAccountStateProof

fn encode_membership(proof: &MembershipProof) -> RpcMembershipProof {
    RpcMembershipProof {
        entry_count: proof.entry_count,
        path: proof
            .path
            .iter()
            .map(|(bit, sibling)| RpcKeyedPathEntry {
                bit: *bit,
                sibling: b58(sibling),
            })
            .collect(),
    }
}

fn decode_membership(proof: &RpcMembershipProof) -> WireResult<MembershipProof> {
    Ok(MembershipProof {
        entry_count: proof.entry_count,
        path: proof
            .path
            .iter()
            .map(|entry| Ok((entry.bit, hash("keyed path sibling", &entry.sibling)?)))
            .collect::<WireResult<_>>()?,
    })
}

fn encode_base_leaf(proof: &BaseLeafProof) -> RpcBaseLeafProof {
    RpcBaseLeafProof {
        address: b58(proof.leaf.address),
        account_hash: b58(proof.leaf.account_hash),
        proof: encode_inclusion_proof(&proof.proof),
    }
}

fn decode_base_leaf(proof: &RpcBaseLeafProof) -> WireResult<BaseLeafProof> {
    Ok(BaseLeafProof {
        leaf: BaseLeaf::new(
            from_b58("base leaf address", &proof.address)?,
            hash("base leaf account hash", &proof.account_hash)?,
        ),
        proof: inclusion_proof("base leaf proof", &proof.proof)?,
    })
}

fn encode_base_absence(proof: &BaseAbsenceProof) -> RpcBaseAbsence {
    match proof {
        BaseAbsenceProof::EmptyTree => RpcBaseAbsence::EmptyTree,
        BaseAbsenceProof::BeforeFirst { first } => RpcBaseAbsence::BeforeFirst {
            first: encode_base_leaf(first),
        },
        BaseAbsenceProof::AfterLast { last } => RpcBaseAbsence::AfterLast {
            last: encode_base_leaf(last),
        },
        BaseAbsenceProof::Between { left, right } => RpcBaseAbsence::Between {
            left: encode_base_leaf(left),
            right: encode_base_leaf(right),
        },
    }
}

fn decode_base_absence(proof: &RpcBaseAbsence) -> WireResult<BaseAbsenceProof> {
    Ok(match proof {
        RpcBaseAbsence::EmptyTree => BaseAbsenceProof::EmptyTree,
        RpcBaseAbsence::BeforeFirst { first } => BaseAbsenceProof::BeforeFirst {
            first: decode_base_leaf(first)?,
        },
        RpcBaseAbsence::AfterLast { last } => BaseAbsenceProof::AfterLast {
            last: decode_base_leaf(last)?,
        },
        RpcBaseAbsence::Between { left, right } => BaseAbsenceProof::Between {
            left: decode_base_leaf(left)?,
            right: decode_base_leaf(right)?,
        },
    })
}

fn encode_account_lookup(lookup: &AccountLookup) -> RpcAccountLookup {
    match lookup {
        AccountLookup::Overlay { entry, proof } => RpcAccountLookup::Overlay {
            address: b58(entry.address),
            now: b58(entry.now),
            base: b58(entry.base),
            proof: encode_membership(proof),
        },
        AccountLookup::Base {
            overlay_absence,
            base,
        } => RpcAccountLookup::Base {
            overlay_absence: match overlay_absence.as_ref() {
                NonMembershipProof::EmptyTree => RpcOverlayAbsence::EmptyTree,
                NonMembershipProof::Neighbor { key, value, proof } => RpcOverlayAbsence::Neighbor {
                    key: b58(key),
                    value: b64(value),
                    proof: encode_membership(proof),
                },
            },
            base: match base {
                BaseLookup::Present(proof) => RpcBaseLookup::Present(encode_base_leaf(proof)),
                BaseLookup::Absent(proof) => RpcBaseLookup::Absent(encode_base_absence(proof)),
            },
        },
    }
}

fn decode_account_lookup(lookup: &RpcAccountLookup) -> WireResult<AccountLookup> {
    Ok(match lookup {
        RpcAccountLookup::Overlay {
            address,
            now,
            base,
            proof,
        } => AccountLookup::Overlay {
            entry: OverlayEntry::new(
                from_b58("overlay entry address", address)?,
                hash("overlay entry hash", now)?,
                hash("overlay entry base hash", base)?,
            ),
            proof: decode_membership(proof)?,
        },
        RpcAccountLookup::Base {
            overlay_absence,
            base,
        } => AccountLookup::Base {
            overlay_absence: Box::new(match overlay_absence {
                RpcOverlayAbsence::EmptyTree => NonMembershipProof::EmptyTree,
                RpcOverlayAbsence::Neighbor { key, value, proof } => {
                    let value = from_b64("overlay neighbor value", value)?;
                    let len = value.len();
                    let value: OverlayValue = value.try_into().map_err(|_| {
                        WireError::new(
                            "overlay neighbor value",
                            format!("{len} bytes, expected {OVERLAY_VALUE_LEN}"),
                        )
                    })?;
                    NonMembershipProof::Neighbor {
                        key: from_b58("overlay neighbor key", key)?,
                        value,
                        proof: decode_membership(proof)?,
                    }
                }
            }),
            base: match base {
                RpcBaseLookup::Present(proof) => BaseLookup::Present(decode_base_leaf(proof)?),
                RpcBaseLookup::Absent(proof) => BaseLookup::Absent(decode_base_absence(proof)?),
            },
        },
    })
}

/// `getAccountStateProof`: the state of an account at the end of a block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccountStateResponse {
    pub slot: u64,
    pub block_id: Hash,
    pub proof: AccountStateProof,
}

impl From<&AccountStateResponse> for RpcAccountStateProof {
    fn from(response: &AccountStateResponse) -> Self {
        let AccountProof { opening, lookup } = &response.proof.0;
        Self {
            slot: response.slot,
            block_id: b58(response.block_id),
            generation: opening.generation,
            base_root: b58(opening.base_root),
            overlay_root: b58(opening.overlay_root),
            slot_proof: encode_inclusion_proof(&opening.slot_proof),
            lookup: encode_account_lookup(lookup),
        }
    }
}

impl TryFrom<&RpcAccountStateProof> for AccountStateResponse {
    type Error = WireError;

    fn try_from(response: &RpcAccountStateProof) -> WireResult<Self> {
        Ok(Self {
            slot: response.slot,
            block_id: hash("block id", &response.block_id)?,
            proof: AccountStateProof(AccountProof {
                opening: AccountsStateOpening {
                    generation: response.generation,
                    base_root: hash("base root", &response.base_root)?,
                    overlay_root: hash("overlay root", &response.overlay_root)?,
                    slot_proof: inclusion_proof("accounts state slot proof", &response.slot_proof)?,
                },
                lookup: decode_account_lookup(&response.lookup)?,
            }),
        })
    }
}

// ---------------------------------------------------------------------------
// getValidatorSet

fn encode_signer(signer: &Signer) -> RpcSigner {
    RpcSigner {
        bls_pubkey_compressed: b58(signer.bls_pubkey_compressed),
        stake: signer.stake,
        vote_pubkey: b58(signer.vote_pubkey),
    }
}

fn decode_signer(signer: &RpcSigner) -> WireResult<Signer> {
    Ok(Signer {
        bls_pubkey_compressed: from_b58("signer BLS public key", &signer.bls_pubkey_compressed)?,
        stake: signer.stake,
        vote_pubkey: from_b58("signer vote account", &signer.vote_pubkey)?,
    })
}

/// A commitment slot opened in a block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenedAt {
    pub slot: u64,
    pub block_id: Hash,
    pub opening: SlotOpening,
}

impl From<&OpenedAt> for RpcSlotOpeningAt {
    fn from(opened: &OpenedAt) -> Self {
        Self {
            slot: opened.slot,
            block_id: b58(opened.block_id),
            opening: encode_slot_opening(&opened.opening),
        }
    }
}

impl TryFrom<&RpcSlotOpeningAt> for OpenedAt {
    type Error = WireError;

    fn try_from(opened: &RpcSlotOpeningAt) -> WireResult<Self> {
        Ok(Self {
            slot: opened.slot,
            block_id: hash("block id", &opened.block_id)?,
            opening: slot_opening("opening", &opened.opening)?,
        })
    }
}

/// `getValidatorSet`: the signer set of an epoch and the block of the
/// previous epoch that commits to it.
#[derive(Clone, Debug)]
pub struct ValidatorSetResponse {
    pub signer_set: SignerSet,
    pub committed_by: Option<OpenedAt>,
}

impl ValidatorSetResponse {
    /// Verifies that `block` (verified by the caller; the block named by
    /// `committed_by`) commits to this signer set in slot 3, and returns it.
    /// See [`LightClient::advance_epoch`] to switch a client to it.
    pub fn verify(
        &self,
        block: &impl CommittedBlock,
        epoch_schedule: &EpochSchedule,
    ) -> Result<&SignerSet, Error> {
        let committed_by = self
            .committed_by
            .as_ref()
            .ok_or(Error::SlotOpening(CommitmentSlot::ValidatorSetRoot))?;
        if committed_by.slot != block.slot() || committed_by.block_id != *block.block_id() {
            return Err(Error::WrongBlock);
        }
        verify_next_signer_set(
            block,
            epoch_schedule,
            &committed_by.opening,
            &self.signer_set,
        )?;
        Ok(&self.signer_set)
    }
}

impl From<&ValidatorSetResponse> for RpcValidatorSet {
    fn from(response: &ValidatorSetResponse) -> Self {
        Self {
            epoch: response.signer_set.epoch(),
            signers: response
                .signer_set
                .signers()
                .iter()
                .map(encode_signer)
                .collect(),
            validator_set_root: b58(response.signer_set.validator_set_root()),
            committed_by: response.committed_by.as_ref().map(RpcSlotOpeningAt::from),
        }
    }
}

impl TryFrom<&RpcValidatorSet> for ValidatorSetResponse {
    type Error = WireError;

    /// Fails as well if the signers are not a valid signer set, or do not
    /// hash to the `validatorSetRoot` of the response.
    fn try_from(response: &RpcValidatorSet) -> WireResult<Self> {
        let signers = response
            .signers
            .iter()
            .map(decode_signer)
            .collect::<WireResult<Vec<_>>>()?;
        let validator_set_root = hash("validator set root", &response.validator_set_root)?;
        let signer_set = SignerSet::from_root(response.epoch, signers, &validator_set_root)
            .map_err(|err| WireError::new("signer set", err.to_string()))?;
        Ok(Self {
            signer_set,
            committed_by: response
                .committed_by
                .as_ref()
                .map(OpenedAt::try_from)
                .transpose()?,
        })
    }
}

// ---------------------------------------------------------------------------
// getEpochBoundary

fn encode_epoch_boundary(boundary: &EpochBoundary) -> RpcEpochBoundaryInputs {
    RpcEpochBoundaryInputs {
        epoch: boundary.epoch,
        first_slot: boundary.first_slot,
        parent_slot: boundary.parent_slot,
        parent_block_id: b58(boundary.parent_block_id),
        parent_commitment_root: b58(boundary.parent_commitment_root),
        history_len: boundary.history_len,
    }
}

fn decode_epoch_boundary(boundary: &RpcEpochBoundaryInputs) -> WireResult<EpochBoundary> {
    Ok(EpochBoundary {
        epoch: boundary.epoch,
        first_slot: boundary.first_slot,
        parent_slot: boundary.parent_slot,
        parent_block_id: hash("boundary parent block id", &boundary.parent_block_id)?,
        parent_commitment_root: hash(
            "boundary parent commitment root",
            &boundary.parent_commitment_root,
        )?,
        history_len: boundary.history_len,
    })
}

/// `getEpochBoundary`: where the previous epoch ended, from a block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpochBoundaryResponse {
    pub slot: u64,
    pub block_id: Hash,
    pub proof: EpochBoundaryProof,
}

impl From<&EpochBoundaryResponse> for RpcEpochBoundary {
    fn from(response: &EpochBoundaryResponse) -> Self {
        Self {
            slot: response.slot,
            block_id: b58(response.block_id),
            boundary: encode_epoch_boundary(&response.proof.boundary),
            opening: encode_slot_opening(&response.proof.opening),
        }
    }
}

impl TryFrom<&RpcEpochBoundary> for EpochBoundaryResponse {
    type Error = WireError;

    fn try_from(response: &RpcEpochBoundary) -> WireResult<Self> {
        Ok(Self {
            slot: response.slot,
            block_id: hash("block id", &response.block_id)?,
            proof: EpochBoundaryProof {
                boundary: decode_epoch_boundary(&response.boundary)?,
                opening: slot_opening("epoch boundary opening", &response.opening)?,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn footer() -> FooterProof {
        FooterProof {
            parent_slot: 7,
            parent_block_id: Hash::new_from_array([1; 32]),
            fec_set_count: 3,
            bank_hash: Hash::new_from_array([2; 32]),
            commitment_root: Hash::new_from_array([3; 32]),
            proof: vec![[4; MERKLE_PROOF_ENTRY_LEN], [5; MERKLE_PROOF_ENTRY_LEN]],
        }
    }

    fn certificate(kind: CertificateKind) -> Certificate {
        Certificate {
            kind,
            signature: BlsSignature([6; 192]),
            bitmap: vec![1, 2, 3],
        }
    }

    #[test]
    fn test_finality_round_trip() {
        let opening = SlotOpening {
            siblings: [Hash::new_from_array([8; 32]); SLOT_OPENING_LEN],
        };
        for finality in [
            Finality::Certificates(FinalityProof {
                certificates: vec![
                    certificate(CertificateKind::Notarize),
                    certificate(CertificateKind::Finalize),
                ],
            }),
            Finality::Descendant(Box::new(FinalizedDescendant {
                slot: 12,
                block_id: Hash::new_from_array([9; 32]),
                finality: FinalityProof {
                    certificates: vec![certificate(CertificateKind::FastFinalize)],
                },
                footer: footer(),
                history: HistoryProof {
                    block_history_root: Hash::new_from_array([10; 32]),
                    opening,
                    slot: 10,
                    block_id: Hash::new_from_array([11; 32]),
                    commitment_root: Hash::default(),
                    proof: InclusionProof {
                        leaf_index: 0,
                        leaf_count: 3,
                        siblings: vec![Hash::new_from_array([12; 32])],
                    },
                },
            })),
        ] {
            let response = FinalityResponse {
                slot: 10,
                block_id: Hash::new_from_array([11; 32]),
                footer: footer(),
                finality,
            };
            let wire = RpcFinalityProof::from(&response);
            assert_eq!(FinalityResponse::try_from(&wire).unwrap(), response);
        }
    }

    #[test]
    fn test_decoding_errors() {
        let mut wire = RpcFinalityProof::from(&FinalityResponse {
            slot: 10,
            block_id: Hash::new_from_array([11; 32]),
            footer: footer(),
            finality: Finality::Certificates(FinalityProof {
                certificates: vec![certificate(CertificateKind::FastFinalize)],
            }),
        });
        // Both or neither finality form.
        let mut both = wire.clone();
        both.finalized_descendant = Some(RpcFinalizedDescendant {
            slot: 11,
            block_id: wire.block_id.clone(),
            certificates: vec![],
            footer: wire.footer.clone(),
            history: RpcHistoryProof::from(&HistoryProof {
                block_history_root: Hash::default(),
                opening: SlotOpening {
                    siblings: [Hash::default(); SLOT_OPENING_LEN],
                },
                slot: 0,
                block_id: Hash::default(),
                commitment_root: Hash::default(),
                proof: InclusionProof {
                    leaf_index: 0,
                    leaf_count: 1,
                    siblings: vec![],
                },
            }),
        });
        assert!(FinalityResponse::try_from(&both).is_err());
        let mut neither = wire.clone();
        neither.certificates = None;
        assert!(FinalityResponse::try_from(&neither).is_err());
        // A hash of the wrong length, a bad base-58 string, a short footer
        // entry, a short signature.
        wire.block_id = b58([1; 31]);
        assert_eq!(
            FinalityResponse::try_from(&wire).unwrap_err().field,
            "block id"
        );
        wire.block_id = "0OIl".to_string();
        assert!(FinalityResponse::try_from(&wire).is_err());
        wire.block_id = b58([1; 32]);
        wire.footer.proof[0] = b58([1; 19]);
        assert_eq!(
            FinalityResponse::try_from(&wire).unwrap_err().field,
            "footer proof entry"
        );
        wire.footer.proof[0] = b58([1; 20]);
        wire.certificates.as_mut().unwrap()[0].signature = b64([0; 96]);
        assert_eq!(
            FinalityResponse::try_from(&wire).unwrap_err().field,
            "certificate signature"
        );
    }

    #[test]
    fn test_event_needs_its_instruction() {
        let wire = RpcTransactionProof {
            slot: 1,
            block_id: b58([1; 32]),
            receipts_root: b58([2; 32]),
            opening: vec![b58([3; 32]); 3],
            receipt: encode_receipt(&TransactionReceipt::failed([4; 64], Hash::default(), 5)),
            proof: RpcInclusionProof {
                leaf_index: 0,
                leaf_count: 1,
                siblings: vec![],
            },
            instructions: vec![],
            events: vec![RpcEventProof {
                event: encode_event_record(&EventRecord {
                    program_id: [5; 32],
                    instruction_index: 2,
                    kind: EventKind::Log,
                    payload: b"x".to_vec(),
                }),
                proof: RpcInclusionProof {
                    leaf_index: 0,
                    leaf_count: 1,
                    siblings: vec![],
                },
            }],
        };
        assert_eq!(
            TransactionResponse::try_from(&wire).unwrap_err().field,
            "event"
        );
        let mut short_opening = wire.clone();
        short_opening.events.clear();
        short_opening.opening.pop();
        assert_eq!(
            TransactionResponse::try_from(&short_opening)
                .unwrap_err()
                .field,
            "receipts opening"
        );
    }
}
