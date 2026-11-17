//! Leaf encodings of the "Transaction Receipt Tree" SIMD.
//!
//! A block commits to one [`TransactionReceipt`] per transaction with
//! `receipts_root`, slot 1 of the commitment vector of the Block Commitment
//! Root SIMD. Each receipt of a successful transaction commits to the
//! instructions the transaction executed ([`InstructionRecord`]) and to the
//! data its programs emitted through the logging syscalls ([`EventRecord`]):
//!
//! ```text
//! receipts_root          = root(0x02, [receipt_0, receipt_1, ...])
//! instruction_trace_root = root(0x03, [record_0, record_1, ...])
//! events_root            = root(0x04, [event_0, event_1, ...])
//! ```
//!
//! All integers are little-endian. Every leaf is the concatenation of its
//! fields in the order listed in the SIMD, without padding; a variable-length
//! `payload` is the last field and runs to the end of the leaf. The trees are
//! the commitment trees of [`crate::commitment_tree`].
//!
//! This module only deals with bytes and hashes. Collecting the trace and the
//! events during execution is done by the program runtime and the SVM, and
//! accumulating receipts in block order by the bank.

use {
    crate::commitment_tree::{
        InclusionProof, TreeId, compute_root, compute_root_from_leaf_hashes, hash_leaf,
        make_inclusion_proof, make_inclusion_proof_from_leaf_hashes,
    },
    solana_hash::{HASH_BYTES, Hash},
    solana_sha256_hasher::{Hasher, hashv},
    std::sync::LazyLock,
};

/// Size of a public key in a leaf.
pub const PUBKEY_BYTES: usize = 32;
/// Size of a transaction signature in a leaf.
pub const SIGNATURE_BYTES: usize = 64;

/// `version` of the receipts defined by this module.
pub const RECEIPT_VERSION: u8 = 1;
/// Size of a serialized [`TransactionReceipt`].
pub const RECEIPT_LEN: usize = 1 + 1 + SIGNATURE_BYTES + HASH_BYTES + 8 + HASH_BYTES + HASH_BYTES;
/// Size of a serialized [`InstructionRecord`].
pub const INSTRUCTION_RECORD_LEN: usize = 1 + PUBKEY_BYTES + HASH_BYTES + HASH_BYTES;
/// Size of the fixed part of a serialized [`EventRecord`], before `payload`.
pub const EVENT_RECORD_HEADER_LEN: usize = PUBKEY_BYTES + 4 + 1;

/// Account flag bit: the account is a signer of the instruction.
pub const ACCOUNT_FLAG_SIGNER: u8 = 1 << 0;
/// Account flag bit: the account is writable by the instruction.
pub const ACCOUNT_FLAG_WRITABLE: u8 = 1 << 1;

static EMPTY_INSTRUCTION_TRACE_ROOT: LazyLock<Hash> =
    LazyLock::new(|| compute_root::<&[u8]>(TreeId::INSTRUCTION_TRACE, &[]));
static EMPTY_EVENTS_ROOT: LazyLock<Hash> =
    LazyLock::new(|| compute_root::<&[u8]>(TreeId::EVENTS, &[]));

/// `root(0x03, [])`: the trace root of a failed transaction.
pub fn empty_instruction_trace_root() -> Hash {
    *EMPTY_INSTRUCTION_TRACE_ROOT
}

/// `root(0x04, [])`: the events root of a failed transaction, or of a
/// successful transaction that emitted no event.
pub fn empty_events_root() -> Hash {
    *EMPTY_EVENTS_ROOT
}

/// `status` of a [`TransactionReceipt`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ReceiptStatus {
    Success = 0,
    Failed = 1,
}

/// The receipt of one transaction (170 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransactionReceipt {
    pub version: u8,
    pub status: ReceiptStatus,
    /// First signature of the transaction.
    pub signature: [u8; SIGNATURE_BYTES],
    /// Hash of the transaction message, as used for duplicate detection.
    pub message_hash: Hash,
    /// Lamports charged to the fee payer.
    pub fee: u64,
    pub instruction_trace_root: Hash,
    pub events_root: Hash,
}

impl TransactionReceipt {
    /// Receipt of a successful transaction.
    pub fn success(
        signature: [u8; SIGNATURE_BYTES],
        message_hash: Hash,
        fee: u64,
        instruction_trace_root: Hash,
        events_root: Hash,
    ) -> Self {
        Self {
            version: RECEIPT_VERSION,
            status: ReceiptStatus::Success,
            signature,
            message_hash,
            fee,
            instruction_trace_root,
            events_root,
        }
    }

    /// Receipt of a failed transaction: both sub-roots are empty-tree roots.
    pub fn failed(signature: [u8; SIGNATURE_BYTES], message_hash: Hash, fee: u64) -> Self {
        Self {
            version: RECEIPT_VERSION,
            status: ReceiptStatus::Failed,
            signature,
            message_hash,
            fee,
            instruction_trace_root: empty_instruction_trace_root(),
            events_root: empty_events_root(),
        }
    }

    /// Leaf bytes:
    /// `version || status || signature || message_hash || u64_le(fee) ||
    /// instruction_trace_root || events_root`.
    pub fn to_bytes(&self) -> [u8; RECEIPT_LEN] {
        let mut bytes = [0u8; RECEIPT_LEN];
        let mut writer = LeafWriter::new(&mut bytes);
        writer.put(&[self.version, self.status as u8]);
        writer.put(&self.signature);
        writer.put(self.message_hash.as_ref());
        writer.put(&self.fee.to_le_bytes());
        writer.put(self.instruction_trace_root.as_ref());
        writer.put(self.events_root.as_ref());
        writer.finish();
        bytes
    }

    /// `leaf(0x02, receipt)`
    pub fn leaf_hash(&self) -> Hash {
        hash_leaf(TreeId::TRANSACTION_RECEIPTS, &self.to_bytes())
    }
}

/// One account of an instruction, with the privileges it was invoked with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstructionAccountMeta {
    pub pubkey: [u8; PUBKEY_BYTES],
    pub is_signer: bool,
    pub is_writable: bool,
}

impl InstructionAccountMeta {
    /// `flags`: bit 0 is_signer, bit 1 is_writable.
    pub fn flags(&self) -> u8 {
        let mut flags = 0;
        if self.is_signer {
            flags |= ACCOUNT_FLAG_SIGNER;
        }
        if self.is_writable {
            flags |= ACCOUNT_FLAG_WRITABLE;
        }
        flags
    }
}

/// `accounts_hash = sha256(u16_le(num_accounts) || (pubkey || flags) ...)`
///
/// `num_accounts` must fit in a `u16`; the runtime caps the accounts of an
/// instruction far below that (`MAX_ACCOUNTS_PER_INSTRUCTION`).
pub fn accounts_hash<I>(accounts: I) -> Hash
where
    I: IntoIterator<Item = InstructionAccountMeta>,
    I::IntoIter: ExactSizeIterator,
{
    let accounts = accounts.into_iter();
    let num_accounts = u16::try_from(accounts.len())
        .expect("the number of accounts of an instruction fits in a u16");
    let mut hasher = Hasher::default();
    hasher.hash(&num_accounts.to_le_bytes());
    for account in accounts {
        hasher.hashv(&[&account.pubkey, &[account.flags()]]);
    }
    hasher.result()
}

/// `data_hash = sha256(instruction_data)`
pub fn data_hash(data: &[u8]) -> Hash {
    hashv(&[data])
}

/// One instruction of the trace (97 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstructionRecord {
    /// 1 for a top-level instruction, 2 for the first CPI level, ...
    pub stack_height: u8,
    pub program_id: [u8; PUBKEY_BYTES],
    pub accounts_hash: Hash,
    pub data_hash: Hash,
}

impl InstructionRecord {
    pub fn new<I>(
        stack_height: u8,
        program_id: [u8; PUBKEY_BYTES],
        accounts: I,
        instruction_data: &[u8],
    ) -> Self
    where
        I: IntoIterator<Item = InstructionAccountMeta>,
        I::IntoIter: ExactSizeIterator,
    {
        Self {
            stack_height,
            program_id,
            accounts_hash: accounts_hash(accounts),
            data_hash: data_hash(instruction_data),
        }
    }

    /// Leaf bytes: `stack_height || program_id || accounts_hash || data_hash`.
    pub fn to_bytes(&self) -> [u8; INSTRUCTION_RECORD_LEN] {
        let mut bytes = [0u8; INSTRUCTION_RECORD_LEN];
        let mut writer = LeafWriter::new(&mut bytes);
        writer.put(&[self.stack_height]);
        writer.put(&self.program_id);
        writer.put(self.accounts_hash.as_ref());
        writer.put(self.data_hash.as_ref());
        writer.finish();
        bytes
    }

    /// `leaf(0x03, record)`
    pub fn leaf_hash(&self) -> Hash {
        hash_leaf(TreeId::INSTRUCTION_TRACE, &self.to_bytes())
    }
}

/// `kind` of an [`EventRecord`]: the logging syscall that emitted it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EventKind {
    /// `sol_log_data`: the payload is [`log_data_payload`] of its fields.
    LogData = 0,
    /// `sol_log_`: the payload is the message bytes.
    Log = 1,
}

/// `payload` of a `sol_log_data` event:
/// `u32_le(count)`, then `u32_le(len) || bytes` for each field.
///
/// Field counts and lengths are bounded by the compute units the syscall
/// charges, far below `u32::MAX`.
pub fn log_data_payload<T: AsRef<[u8]>>(fields: &[T]) -> Vec<u8> {
    let total = fields
        .iter()
        .map(|field| 4 + field.as_ref().len())
        .sum::<usize>();
    let mut payload = Vec::with_capacity(4 + total);
    payload.extend_from_slice(&(fields.len() as u32).to_le_bytes());
    for field in fields {
        let field = field.as_ref();
        payload.extend_from_slice(&(field.len() as u32).to_le_bytes());
        payload.extend_from_slice(field);
    }
    payload
}

/// One event: a call to a logging syscall with program-chosen bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventRecord {
    /// Program that called the syscall.
    pub program_id: [u8; PUBKEY_BYTES],
    /// Index, in the instruction trace, of the record of the emitting
    /// instruction (execution order).
    pub instruction_index: u32,
    pub kind: EventKind,
    pub payload: Vec<u8>,
}

impl EventRecord {
    /// Leaf bytes: `program_id || u32_le(instruction_index) || kind || payload`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(EVENT_RECORD_HEADER_LEN + self.payload.len());
        bytes.extend_from_slice(&self.program_id);
        bytes.extend_from_slice(&self.instruction_index.to_le_bytes());
        bytes.push(self.kind as u8);
        bytes.extend_from_slice(&self.payload);
        bytes
    }

    /// `leaf(0x04, event)`, hashed without materializing the leaf bytes.
    pub fn leaf_hash(&self) -> Hash {
        hashv(&[
            &[crate::commitment_tree::LEAF_PREFIX, TreeId::EVENTS.0],
            &self.program_id,
            &self.instruction_index.to_le_bytes(),
            &[self.kind as u8],
            &self.payload,
        ])
    }
}

/// `receipts_root = root(0x02, receipts)`
pub fn receipts_root(receipts: &[TransactionReceipt]) -> Hash {
    let leaf_hashes: Vec<_> = receipts.iter().map(TransactionReceipt::leaf_hash).collect();
    compute_root_from_leaf_hashes(TreeId::TRANSACTION_RECEIPTS, &leaf_hashes)
}

/// `receipts_root` over receipts that are already hashed with
/// [`TransactionReceipt::leaf_hash`].
pub fn receipts_root_from_leaf_hashes(leaf_hashes: &[Hash]) -> Hash {
    compute_root_from_leaf_hashes(TreeId::TRANSACTION_RECEIPTS, leaf_hashes)
}

/// `instruction_trace_root = root(0x03, records)`
pub fn instruction_trace_root(records: &[InstructionRecord]) -> Hash {
    let leaf_hashes: Vec<_> = records.iter().map(InstructionRecord::leaf_hash).collect();
    compute_root_from_leaf_hashes(TreeId::INSTRUCTION_TRACE, &leaf_hashes)
}

/// `events_root = root(0x04, events)`
pub fn events_root(events: &[EventRecord]) -> Hash {
    let leaf_hashes: Vec<_> = events.iter().map(EventRecord::leaf_hash).collect();
    compute_root_from_leaf_hashes(TreeId::EVENTS, &leaf_hashes)
}

/// Proof of receipt `index` against [`receipts_root`].
pub fn receipt_inclusion_proof(
    receipts: &[TransactionReceipt],
    index: usize,
) -> Option<InclusionProof> {
    let leaf_hashes: Vec<_> = receipts.iter().map(TransactionReceipt::leaf_hash).collect();
    make_inclusion_proof_from_leaf_hashes(&leaf_hashes, index)
}

/// Proof of trace record `index` against [`instruction_trace_root`].
pub fn instruction_record_inclusion_proof(
    records: &[InstructionRecord],
    index: usize,
) -> Option<InclusionProof> {
    let leaves: Vec<_> = records.iter().map(InstructionRecord::to_bytes).collect();
    make_inclusion_proof(TreeId::INSTRUCTION_TRACE, &leaves, index)
}

/// Proof of event `index` against [`events_root`].
pub fn event_inclusion_proof(events: &[EventRecord], index: usize) -> Option<InclusionProof> {
    let leaf_hashes: Vec<_> = events.iter().map(EventRecord::leaf_hash).collect();
    make_inclusion_proof_from_leaf_hashes(&leaf_hashes, index)
}

/// Returns true if `receipt` is at `proof.leaf_index` of the tree with
/// `receipts_root`.
pub fn verify_receipt(proof: &InclusionProof, receipt: &TransactionReceipt, root: &Hash) -> bool {
    proof.verify(TreeId::TRANSACTION_RECEIPTS, &receipt.to_bytes(), root)
}

/// Returns true if `record` is at `proof.leaf_index` of the trace with
/// `instruction_trace_root`.
pub fn verify_instruction_record(
    proof: &InclusionProof,
    record: &InstructionRecord,
    instruction_trace_root: &Hash,
) -> bool {
    proof.verify(
        TreeId::INSTRUCTION_TRACE,
        &record.to_bytes(),
        instruction_trace_root,
    )
}

/// Returns true if `event` is at `proof.leaf_index` of the events tree with
/// `events_root`.
pub fn verify_event(proof: &InclusionProof, event: &EventRecord, events_root: &Hash) -> bool {
    proof.verify_leaf_hash(TreeId::EVENTS, event.leaf_hash(), events_root)
}

/// Writes fixed-size leaves field by field and checks that they are filled
/// exactly.
struct LeafWriter<'a> {
    bytes: &'a mut [u8],
    offset: usize,
}

impl<'a> LeafWriter<'a> {
    fn new(bytes: &'a mut [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn put(&mut self, field: &[u8]) {
        self.bytes[self.offset..self.offset + field.len()].copy_from_slice(field);
        self.offset += field.len();
    }

    fn finish(self) {
        debug_assert_eq!(self.offset, self.bytes.len());
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

    fn account(byte: u8, is_signer: bool, is_writable: bool) -> InstructionAccountMeta {
        InstructionAccountMeta {
            pubkey: [byte; 32],
            is_signer,
            is_writable,
        }
    }

    /// The trace, events and receipts of the test vectors below.
    fn vector_records() -> Vec<InstructionRecord> {
        vec![
            InstructionRecord::new(
                1,
                [0x11; 32],
                [account(0xa1, true, true), account(0xa2, false, true)],
                b"top",
            ),
            InstructionRecord::new(2, [0x22; 32], [account(0xa2, false, false)], b""),
            InstructionRecord::new(
                1,
                [0x33; 32],
                std::iter::empty::<InstructionAccountMeta>(),
                &[0, 1, 2, 3],
            ),
        ]
    }

    fn vector_events() -> Vec<EventRecord> {
        vec![
            EventRecord {
                program_id: [0x11; 32],
                instruction_index: 0,
                kind: EventKind::LogData,
                payload: log_data_payload(&[b"ab".as_slice(), b"", b"xyz"]),
            },
            EventRecord {
                program_id: [0x22; 32],
                instruction_index: 1,
                kind: EventKind::Log,
                payload: b"hello".to_vec(),
            },
        ]
    }

    fn vector_receipts() -> Vec<TransactionReceipt> {
        vec![
            TransactionReceipt::success(
                [0x01; 64],
                Hash::new_from_array([0x02; 32]),
                5000,
                instruction_trace_root(&vector_records()),
                events_root(&vector_events()),
            ),
            TransactionReceipt::failed([0x03; 64], Hash::new_from_array([0x04; 32]), 10_000),
            TransactionReceipt::success(
                [0x05; 64],
                Hash::new_from_array([0x06; 32]),
                0x0102_0304_0506_0708,
                instruction_trace_root(&vector_records()[..1]),
                empty_events_root(),
            ),
        ]
    }

    #[test]
    fn test_layouts() {
        assert_eq!(RECEIPT_LEN, 170);
        assert_eq!(INSTRUCTION_RECORD_LEN, 97);

        let receipt = vector_receipts()[2];
        let bytes = receipt.to_bytes();
        assert_eq!(bytes[0], 1);
        assert_eq!(bytes[1], 0);
        assert_eq!(&bytes[2..66], &[0x05; 64]);
        assert_eq!(&bytes[66..98], &[0x06; 32]);
        assert_eq!(&bytes[98..106], &0x0102_0304_0506_0708u64.to_le_bytes());
        assert_eq!(&bytes[106..138], receipt.instruction_trace_root.as_ref());
        assert_eq!(&bytes[138..170], empty_events_root().as_ref());
        assert_eq!(
            TransactionReceipt::failed([0; 64], Hash::default(), 0).to_bytes()[1],
            1
        );

        let record = vector_records()[0];
        let bytes = record.to_bytes();
        assert_eq!(bytes[0], 1);
        assert_eq!(&bytes[1..33], &[0x11; 32]);
        assert_eq!(&bytes[33..65], record.accounts_hash.as_ref());
        assert_eq!(&bytes[65..97], record.data_hash.as_ref());

        let event = &vector_events()[0];
        let bytes = event.to_bytes();
        assert_eq!(&bytes[..32], &[0x11; 32]);
        assert_eq!(&bytes[32..36], &0u32.to_le_bytes());
        assert_eq!(bytes[36], 0);
        assert_eq!(
            &bytes[37..],
            &[
                3, 0, 0, 0, 2, 0, 0, 0, b'a', b'b', 0, 0, 0, 0, 3, 0, 0, 0, b'x', b'y', b'z'
            ]
        );
        for event in vector_events() {
            assert_eq!(
                event.leaf_hash(),
                hash_leaf(TreeId::EVENTS, &event.to_bytes())
            );
        }
    }

    #[test]
    fn test_accounts_and_data_hash() {
        assert_eq!(
            accounts_hash([account(0xa1, true, true), account(0xa2, false, true)]),
            hashv(&[
                &2u16.to_le_bytes(),
                &[0xa1; 32],
                &[0b11],
                &[0xa2; 32],
                &[0b10],
            ])
        );
        assert_eq!(account(0, true, false).flags(), 0b01);
        assert_eq!(account(0, false, false).flags(), 0);
        assert_eq!(
            accounts_hash(std::iter::empty::<InstructionAccountMeta>()),
            hashv(&[&[0, 0]])
        );
        assert_eq!(data_hash(b""), hashv(&[b""]));
        // Duplicate accounts are hashed once per occurrence.
        assert_ne!(
            accounts_hash([account(1, false, false)]),
            accounts_hash([account(1, false, false), account(1, false, false)])
        );
    }

    #[test]
    fn test_empty_roots() {
        assert_eq!(
            empty_instruction_trace_root(),
            hashv(&[&[0x80, 0x03], &0u64.to_le_bytes(), &[0; 32]])
        );
        assert_eq!(
            empty_events_root(),
            hashv(&[&[0x80, 0x04], &0u64.to_le_bytes(), &[0; 32]])
        );
        assert_eq!(instruction_trace_root(&[]), empty_instruction_trace_root());
        assert_eq!(events_root(&[]), empty_events_root());
        // Empty block.
        assert_eq!(
            receipts_root(&[]),
            hashv(&[&[0x80, 0x02], &0u64.to_le_bytes(), &[0; 32]])
        );
        assert_eq!(receipts_root_from_leaf_hashes(&[]), receipts_root(&[]));
    }

    // Vectors computed with an independent Python implementation of the
    // encodings in the SIMD (hashlib.sha256) over `vector_records()`,
    // `vector_events()` and `vector_receipts()`.
    const TRACE_ROOT: &str = "cb59d464941be0adb370e61d8eaaac83d7f4a6fe31ddb6652dfbc0a24646c5d5";
    const EVENTS_ROOT: &str = "9001da36ca494d1e9b76d9d9a496a0b06769a7ab8fe459e2d1992585c278af43";
    const RECEIPTS_ROOT: &str = "2c38667086f86b2aead340b667a4d0e8a4cf0406d1f7504139a96c8ff6a4d26d";

    #[test]
    fn test_vectors() {
        assert_eq!(
            instruction_trace_root(&vector_records()),
            hash_from_hex(TRACE_ROOT)
        );
        assert_eq!(events_root(&vector_events()), hash_from_hex(EVENTS_ROOT));
        assert_eq!(
            receipts_root(&vector_receipts()),
            hash_from_hex(RECEIPTS_ROOT)
        );
    }

    #[test]
    fn test_proofs() {
        let receipts = vector_receipts();
        let root = receipts_root(&receipts);
        for (i, receipt) in receipts.iter().enumerate() {
            let proof = receipt_inclusion_proof(&receipts, i).unwrap();
            assert!(verify_receipt(&proof, receipt, &root));
            // A receipt does not verify at another position.
            let other = &receipts[(i + 1) % receipts.len()];
            assert!(!verify_receipt(&proof, other, &root));
        }
        assert!(receipt_inclusion_proof(&receipts, receipts.len()).is_none());

        let records = vector_records();
        let trace_root = instruction_trace_root(&records);
        assert_eq!(receipts[0].instruction_trace_root, trace_root);
        for (i, record) in records.iter().enumerate() {
            let proof = instruction_record_inclusion_proof(&records, i).unwrap();
            assert!(verify_instruction_record(&proof, record, &trace_root));
        }

        let events = vector_events();
        let events_root = events_root(&events);
        assert_eq!(receipts[0].events_root, events_root);
        for (i, event) in events.iter().enumerate() {
            let proof = event_inclusion_proof(&events, i).unwrap();
            assert!(verify_event(&proof, event, &events_root));
            let mut tampered = event.clone();
            tampered.payload.push(0);
            assert!(!verify_event(&proof, &tampered, &events_root));
        }
        // Trees are domain separated: an event proof does not open the trace.
        let proof = event_inclusion_proof(&events, 0).unwrap();
        assert!(!proof.verify_leaf_hash(
            TreeId::INSTRUCTION_TRACE,
            events[0].leaf_hash(),
            &events_root
        ));
    }

    #[test]
    fn test_large_event_is_committed_in_full() {
        // Larger than the default RPC log limit (10 KB): nothing truncates it.
        let payload = vec![0x5a; 100 * 1024];
        let event = EventRecord {
            program_id: [7; 32],
            instruction_index: 3,
            kind: EventKind::Log,
            payload: payload.clone(),
        };
        assert_eq!(
            event.to_bytes().len(),
            EVENT_RECORD_HEADER_LEN + payload.len()
        );
        let mut truncated = event.clone();
        truncated.payload.truncate(10 * 1024);
        assert_ne!(event.leaf_hash(), truncated.leaf_hash());
    }
}
