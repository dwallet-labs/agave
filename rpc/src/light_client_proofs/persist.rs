//! Files of the proof store, so that a restarted node still serves the
//! blocks it kept before.
//!
//! One file per rooted block (`blocks/<slot>`), per slot with certificates
//! (`certificates/<slot>`) and per signer set (`signer-sets/<epoch>`). A
//! file is written to a temporary name, synced and renamed, so a file is
//! whole or absent after a crash or a power loss. The directories are
//! synced once per root ([`Persistence::sync_directories`]), so a power
//! loss can lose the files of the latest root, never leave half of one. A
//! file that does not parse is skipped with a warning.
//!
//! The history leaves are kept in segment files (`history/<index>`, named
//! by the index of their first leaf): the frontier of the leaves before the
//! segment, then one 72-byte record per leaf, appended and synced at every
//! root. A new segment starts every [`HISTORY_SEGMENT_LEN`] leaves, and a
//! segment is removed when every leaf in it is older than the kept history.
//!
//! The accounts state of a block (trees shared between blocks in memory)
//! is not written. A block file has instead what stepping back from the
//! state after the block to the state before it needs, see
//! [`super::StateRecord`].

use {
    super::{ActiveFeatures, GenerationRecord, RetainedBlock, SlotCertificates, StateRecord},
    log::warn,
    solana_bls_signatures::Signature as BlsSignature,
    solana_clock::{Epoch, Slot},
    solana_hash::Hash,
    solana_light_client::{
        Certificate, CertificateKind, HISTORY_LEAF_LEN, history_leaf,
        solana_merkle_tree::{
            accounts_delta::{AccountsDelta, DeltaEntry},
            commitment_tree::{AppendOnlyFrontier, TreeId, hash_leaf},
            epoch_boundary::EpochBoundary,
            receipt_tree::{
                EventKind, EventRecord, InstructionRecord, ReceiptStatus, TransactionReceipt,
            },
            validator_set::Signer,
        },
    },
    solana_runtime::bank::{
        accounts_state::BaseId, block_commitment::CommitmentVector,
        transaction_receipts::RetainedReceipt,
    },
    solana_svm::transaction_execution_result::ReceiptLeaves,
    std::{
        collections::BTreeSet,
        fs,
        io::{self, Write},
        path::{Path, PathBuf},
        sync::{Arc, Mutex},
    },
};

/// First bytes of every file: format and version.
const MAGIC: &[u8; 8] = b"lcproof1";
/// Last bytes of every file: a file cut short does not end with them.
const END: &[u8; 4] = b"end.";

const BLOCKS: &str = "blocks";
const CERTIFICATES: &str = "certificates";
const SIGNER_SETS: &str = "signer-sets";
const HISTORY: &str = "history";

/// Number of history leaves per segment file.
pub(super) const HISTORY_SEGMENT_LEN: u64 = 4096;

/// A history leaf as kept: `(slot, block_id, commitment_root)`.
pub(super) type HistoryEntry = (Slot, Hash, Hash);

/// What the history segment files hold, kept by the one thread that writes
/// them.
#[derive(Default)]
pub(super) struct HistoryFiles {
    /// [`super::HistoryLog::restarts`] of the log the files follow.
    pub restarts: u64,
    /// The leaves before this index are in the files (or before them).
    pub next: u64,
    /// Frontier of the leaves `[0, next)`.
    frontier: AppendOnlyFrontier,
    /// First leaf index of every segment file, ascending.
    segments: Vec<u64>,
    /// Number of leaves in the last segment.
    last_len: u64,
}

fn history_leaf_hash((slot, block_id, commitment_root): &HistoryEntry) -> Hash {
    hash_leaf(
        TreeId::BLOCK_HISTORY,
        &history_leaf(*slot, block_id, commitment_root),
    )
}

/// A rooted block as written: what the store keeps of it, and what
/// ingesting it again needs.
pub(super) struct BlockRecord {
    pub block: RetainedBlock,
    /// `H(B)`: the block history before the block's own leaf.
    pub history: Option<AppendOnlyFrontier>,
    pub parent_slot: Slot,
}

pub(super) struct Persistence {
    dir: PathBuf,
    /// Directories with new or removed files since they were last synced.
    unsynced: Mutex<BTreeSet<&'static str>>,
}

impl Persistence {
    /// Opens (creating it if needed) the directory. `None`, with a warning,
    /// if it cannot be used: the store then keeps nothing on disk.
    pub fn open(dir: &Path) -> Option<Self> {
        for sub in [BLOCKS, CERTIFICATES, SIGNER_SETS, HISTORY] {
            if let Err(err) = fs::create_dir_all(dir.join(sub)) {
                warn!(
                    "light-client proofs: cannot use {}: {err}; proof data will not survive a \
                     restart",
                    dir.display()
                );
                return None;
            }
        }
        Some(Self {
            dir: dir.to_path_buf(),
            unsynced: Mutex::default(),
        })
    }

    /// Makes the files written and removed since the last call durable: a
    /// new file is on disk once its directory is.
    pub fn sync_directories(&self) {
        let unsynced = std::mem::take(&mut *self.unsynced.lock().unwrap());
        for sub in unsynced {
            let path = self.dir.join(sub);
            if let Err(err) = fs::File::open(&path).and_then(|directory| directory.sync_all()) {
                warn!("light-client proofs: cannot sync {}: {err}", path.display());
            }
        }
    }

    fn changed(&self, sub: &'static str) {
        self.unsynced.lock().unwrap().insert(sub);
    }

    fn path(&self, sub: &str, number: u64) -> PathBuf {
        self.dir.join(sub).join(number.to_string())
    }

    fn write(&self, sub: &'static str, number: u64, payload: Vec<u8>) {
        let path = self.path(sub, number);
        let temporary = path.with_extension("tmp");
        let mut bytes = Vec::with_capacity(MAGIC.len() + payload.len() + END.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&payload);
        bytes.extend_from_slice(END);
        let written = fs::File::create(&temporary)
            .and_then(|mut file| {
                file.write_all(&bytes)?;
                file.sync_all()
            })
            .and_then(|()| fs::rename(&temporary, &path));
        match written {
            Ok(()) => self.changed(sub),
            Err(err) => warn!(
                "light-client proofs: cannot write {}: {err}",
                path.display()
            ),
        }
    }

    fn remove(&self, sub: &'static str, number: u64) {
        let path = self.path(sub, number);
        match fs::remove_file(&path) {
            Ok(()) => self.changed(sub),
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => warn!(
                "light-client proofs: cannot remove {}: {err}",
                path.display()
            ),
        }
    }

    /// The numbers (slots or epochs) that have a file in `sub`, ascending.
    fn numbers(&self, sub: &str) -> Vec<u64> {
        let mut numbers: Vec<u64> = fs::read_dir(self.dir.join(sub))
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| entry.file_name().to_str()?.parse().ok())
            .collect();
        numbers.sort_unstable();
        numbers
    }

    /// The payload of a file, if it is whole.
    fn read(&self, sub: &'static str, number: u64) -> Option<Vec<u8>> {
        let path = self.path(sub, number);
        let bytes = fs::read(&path)
            .map_err(|err| warn!("light-client proofs: cannot read {}: {err}", path.display()))
            .ok()?;
        let payload = bytes
            .strip_prefix(MAGIC.as_slice())
            .and_then(|bytes| bytes.strip_suffix(END.as_slice()));
        if payload.is_none() {
            warn!(
                "light-client proofs: {} is not a whole proof file, skipping it",
                path.display()
            );
        }
        payload.map(<[u8]>::to_vec)
    }

    /// Every file of `sub` that parses, ascending by number.
    fn load<T>(
        &self,
        sub: &'static str,
        decode: impl Fn(&mut Reader) -> Option<T>,
    ) -> Vec<(u64, T)> {
        self.numbers(sub)
            .into_iter()
            .filter_map(|number| {
                let payload = self.read(sub, number)?;
                let mut reader = Reader(&payload);
                let value = decode(&mut reader).filter(|_| reader.0.is_empty());
                if value.is_none() {
                    warn!(
                        "light-client proofs: {} does not parse, skipping it",
                        self.path(sub, number).display()
                    );
                }
                Some((number, value?))
            })
            .collect()
    }

    pub fn write_block(&self, record: &BlockRecord) {
        let mut writer = Writer::default();
        encode_block(&mut writer, record);
        self.write(BLOCKS, record.block.slot, writer.0);
    }

    pub fn remove_block(&self, slot: Slot) {
        self.remove(BLOCKS, slot);
    }

    pub fn load_blocks(&self) -> Vec<BlockRecord> {
        self.load(BLOCKS, decode_block)
            .into_iter()
            .filter(|(slot, record)| *slot == record.block.slot)
            .map(|(_, record)| record)
            .collect()
    }

    pub fn write_certificates(&self, slot: Slot, certificates: &SlotCertificates) {
        let mut writer = Writer::default();
        encode_certificates(&mut writer, certificates);
        self.write(CERTIFICATES, slot, writer.0);
    }

    /// Removes the certificates of slots before `oldest`.
    pub fn remove_certificates_before(&self, oldest: Slot) {
        for slot in self.numbers(CERTIFICATES) {
            if slot >= oldest {
                break;
            }
            self.remove(CERTIFICATES, slot);
        }
    }

    pub fn load_certificates(&self) -> Vec<(Slot, SlotCertificates)> {
        self.load(CERTIFICATES, decode_certificates)
    }

    pub fn write_signer_set(&self, epoch: Epoch, signers: &[Signer]) {
        let mut writer = Writer::default();
        writer.len(signers.len());
        for signer in signers {
            writer.bytes(&signer.bls_pubkey_compressed);
            writer.u64(signer.stake);
            writer.bytes(&signer.vote_pubkey);
        }
        self.write(SIGNER_SETS, epoch, writer.0);
    }

    pub fn remove_signer_set(&self, epoch: Epoch) {
        self.remove(SIGNER_SETS, epoch);
    }

    pub fn load_signer_sets(&self) -> Vec<(Epoch, Vec<Signer>)> {
        self.load(SIGNER_SETS, |reader| {
            reader.vec(|reader| {
                Some(Signer {
                    bls_pubkey_compressed: reader.array()?,
                    stake: reader.u64()?,
                    vote_pubkey: reader.array()?,
                })
            })
        })
    }
}

// ---------------------------------------------------------------------------
// History segments.

impl Persistence {
    /// Reads the history segments: the frontier before the first kept leaf
    /// and the leaves, with the state to go on writing from. Segments are
    /// taken while each starts where the one before ended, with the frontier
    /// the leaves before it give; the rest is removed. A record cut short at
    /// the end of the last segment is cut off.
    pub fn load_history(
        &self,
    ) -> (
        HistoryFiles,
        Option<(AppendOnlyFrontier, Vec<HistoryEntry>)>,
    ) {
        let mut files = HistoryFiles::default();
        let mut base = None;
        let mut entries = Vec::new();
        let numbers = self.numbers(HISTORY);
        let mut rejected = None;
        for (position, first) in numbers.iter().copied().enumerate() {
            let path = self.path(HISTORY, first);
            let segment = fs::read(&path).ok().and_then(|bytes| {
                let mut reader = Reader(bytes.strip_prefix(MAGIC.as_slice())?);
                let frontier = decode_frontier(&mut reader)?;
                let fits = frontier.leaf_count() == first
                    && (base.is_none() || (first == files.next && frontier == files.frontier));
                let header_len = bytes.len().checked_sub(reader.0.len())?;
                fits.then(|| (frontier, reader.0.to_vec(), header_len))
            });
            let Some((frontier, records, header_len)) = segment else {
                rejected = Some(position);
                break;
            };
            if base.is_none() {
                files.frontier = frontier.clone();
                files.next = first;
                base = Some(frontier);
            }
            let whole = records.len() / HISTORY_LEAF_LEN;
            for record in records.as_chunks::<HISTORY_LEAF_LEN>().0 {
                let mut reader = Reader(record);
                let (Some(slot), Some(block_id), Some(commitment_root)) =
                    (reader.u64(), reader.hash(), reader.hash())
                else {
                    break;
                };
                let entry = (slot, block_id, commitment_root);
                if files
                    .frontier
                    .append_leaf_hash(history_leaf_hash(&entry))
                    .is_err()
                {
                    break;
                }
                files.next = files.next.saturating_add(1);
                entries.push(entry);
            }
            if records.len() % HISTORY_LEAF_LEN != 0 {
                let len = header_len.saturating_add(whole.saturating_mul(HISTORY_LEAF_LEN));
                let cut = fs::OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .and_then(|file| {
                        file.set_len(len as u64)?;
                        file.sync_all()
                    });
                if let Err(err) = cut {
                    warn!("light-client proofs: cannot cut {}: {err}", path.display());
                }
            }
            files.segments.push(first);
            files.last_len = whole as u64;
        }
        if let Some(position) = rejected {
            for first in &numbers[position..] {
                warn!(
                    "light-client proofs: {} does not continue the kept history, removing it",
                    self.path(HISTORY, *first).display()
                );
                self.remove(HISTORY, *first);
            }
        }
        let loaded = base.map(|base| (base, entries));
        (files, loaded)
    }

    /// Brings the segment files up to date. With `restart`, the files are
    /// replaced: the kept history then starts after the leaves of that
    /// frontier, and `entries` are all its leaves. Otherwise `entries` are
    /// the leaves from `files.next` on. Segments entirely before leaf
    /// `base_len` are removed.
    pub fn write_history(
        &self,
        files: &mut HistoryFiles,
        restart: Option<(u64, AppendOnlyFrontier)>,
        entries: &[HistoryEntry],
        base_len: u64,
    ) {
        if let Some((restarts, base)) = restart {
            for first in self.numbers(HISTORY) {
                self.remove(HISTORY, first);
            }
            *files = HistoryFiles {
                restarts,
                next: base.leaf_count(),
                frontier: base,
                segments: Vec::new(),
                last_len: 0,
            };
        }
        let mut entries = entries;
        while !entries.is_empty() {
            let segment = match files.segments.last() {
                Some(first) if files.last_len < HISTORY_SEGMENT_LEN => *first,
                _ => {
                    let mut writer = Writer::default();
                    writer.bytes(MAGIC);
                    encode_frontier(&mut writer, &files.frontier);
                    let path = self.path(HISTORY, files.next);
                    let created = fs::File::create(&path).and_then(|mut file| {
                        file.write_all(&writer.0)?;
                        file.sync_all()
                    });
                    if let Err(err) = created {
                        warn!(
                            "light-client proofs: cannot write {}: {err}",
                            path.display()
                        );
                        return;
                    }
                    self.changed(HISTORY);
                    files.segments.push(files.next);
                    files.last_len = 0;
                    files.next
                }
            };
            let room = HISTORY_SEGMENT_LEN.saturating_sub(files.last_len);
            let count = entries
                .len()
                .min(usize::try_from(room).unwrap_or(usize::MAX));
            let (now, later) = entries.split_at(count);
            let mut writer = Writer::default();
            for (slot, block_id, commitment_root) in now {
                writer.u64(*slot);
                writer.hash(block_id);
                writer.hash(commitment_root);
            }
            let path = self.path(HISTORY, segment);
            let appended = fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .and_then(|mut file| {
                    file.write_all(&writer.0)?;
                    file.sync_data()
                });
            if let Err(err) = appended {
                warn!(
                    "light-client proofs: cannot write {}: {err}",
                    path.display()
                );
                return;
            }
            for entry in now {
                if files
                    .frontier
                    .append_leaf_hash(history_leaf_hash(entry))
                    .is_err()
                {
                    return;
                }
            }
            files.next = files.next.saturating_add(count as u64);
            files.last_len = files.last_len.saturating_add(count as u64);
            entries = later;
        }
        // A segment ends where the next one starts.
        while files.segments.len() > 1 && files.segments[1] <= base_len {
            self.remove(HISTORY, files.segments.remove(0));
        }
    }
}

// ---------------------------------------------------------------------------
// Encoding: little-endian integers, 32-bit lengths, one-byte tags.

#[derive(Default)]
struct Writer(Vec<u8>);

impl Writer {
    fn u8(&mut self, value: u8) {
        self.0.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }

    fn hash(&mut self, hash: &Hash) {
        self.bytes(hash.as_ref());
    }

    /// A length. Nothing the store keeps of one block has `2^32` items;
    /// a longer one is written as `u32::MAX` items and fails to parse.
    fn len(&mut self, len: usize) {
        self.u32(u32::try_from(len).unwrap_or(u32::MAX));
    }

    fn var_bytes(&mut self, bytes: &[u8]) {
        self.len(bytes.len());
        self.bytes(bytes);
    }

    fn option<T>(&mut self, value: Option<T>, encode: impl FnOnce(&mut Self, T)) {
        match value {
            None => self.u8(0),
            Some(value) => {
                self.u8(1);
                encode(self, value);
            }
        }
    }
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take(&mut self, len: usize) -> Option<&[u8]> {
        let (taken, rest) = self.0.split_at_checked(len)?;
        self.0 = rest;
        Some(taken)
    }

    fn array<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N)?.try_into().ok()
    }

    fn u8(&mut self) -> Option<u8> {
        self.array::<1>().map(|[byte]| byte)
    }

    fn u32(&mut self) -> Option<u32> {
        self.array().map(u32::from_le_bytes)
    }

    fn u64(&mut self) -> Option<u64> {
        self.array().map(u64::from_le_bytes)
    }

    fn hash(&mut self) -> Option<Hash> {
        self.array().map(Hash::new_from_array)
    }

    fn var_bytes(&mut self) -> Option<Vec<u8>> {
        let len = usize::try_from(self.u32()?).ok()?;
        self.take(len).map(<[u8]>::to_vec)
    }

    fn string(&mut self) -> Option<String> {
        String::from_utf8(self.var_bytes()?).ok()
    }

    fn option<T>(&mut self, decode: impl FnOnce(&mut Self) -> Option<T>) -> Option<Option<T>> {
        match self.u8()? {
            0 => Some(None),
            1 => decode(self).map(Some),
            _ => None,
        }
    }

    /// A counted sequence. Items are read one by one, so a wrong count ends
    /// at the end of the input instead of allocating for it.
    fn vec<T>(&mut self, decode: impl Fn(&mut Self) -> Option<T>) -> Option<Vec<T>> {
        let len = self.u32()?;
        (0..len).map(|_| decode(self)).collect()
    }
}

fn encode_frontier(writer: &mut Writer, frontier: &AppendOnlyFrontier) {
    writer.u64(frontier.leaf_count());
    writer.len(frontier.subtree_roots().len());
    for root in frontier.subtree_roots() {
        writer.hash(root);
    }
}

fn decode_frontier(reader: &mut Reader) -> Option<AppendOnlyFrontier> {
    let leaf_count = reader.u64()?;
    let subtree_roots = reader.vec(Reader::hash)?;
    AppendOnlyFrontier::from_parts(leaf_count, subtree_roots).ok()
}

fn encode_receipt(writer: &mut Writer, retained: &RetainedReceipt) {
    let TransactionReceipt {
        version,
        status,
        signature,
        message_hash,
        fee,
        instruction_trace_root,
        events_root,
    } = &retained.receipt;
    writer.u8(*version);
    writer.u8(*status as u8);
    writer.bytes(signature);
    writer.hash(message_hash);
    writer.u64(*fee);
    writer.hash(instruction_trace_root);
    writer.hash(events_root);
    writer.option(retained.leaves.as_deref(), |writer, leaves| {
        writer.len(leaves.records.len());
        for record in &leaves.records {
            writer.u8(record.stack_height);
            writer.bytes(&record.program_id);
            writer.hash(&record.accounts_hash);
            writer.hash(&record.data_hash);
        }
        writer.len(leaves.events.len());
        for event in &leaves.events {
            writer.bytes(&event.program_id);
            writer.u32(event.instruction_index);
            writer.u8(event.kind as u8);
            writer.var_bytes(&event.payload);
        }
    });
}

fn decode_receipt(reader: &mut Reader) -> Option<RetainedReceipt> {
    let receipt = TransactionReceipt {
        version: reader.u8()?,
        status: match reader.u8()? {
            0 => ReceiptStatus::Success,
            1 => ReceiptStatus::Failed,
            _ => return None,
        },
        signature: reader.array()?,
        message_hash: reader.hash()?,
        fee: reader.u64()?,
        instruction_trace_root: reader.hash()?,
        events_root: reader.hash()?,
    };
    let leaves = reader.option(|reader| {
        let records = reader.vec(|reader| {
            Some(InstructionRecord {
                stack_height: reader.u8()?,
                program_id: reader.array()?,
                accounts_hash: reader.hash()?,
                data_hash: reader.hash()?,
            })
        })?;
        let events = reader.vec(|reader| {
            Some(EventRecord {
                program_id: reader.array()?,
                instruction_index: reader.u32()?,
                kind: match reader.u8()? {
                    0 => EventKind::LogData,
                    1 => EventKind::Log,
                    _ => return None,
                },
                payload: reader.var_bytes()?,
            })
        })?;
        Some(Arc::new(ReceiptLeaves { records, events }))
    })?;
    Some(RetainedReceipt { receipt, leaves })
}

fn encode_block(writer: &mut Writer, record: &BlockRecord) {
    let BlockRecord {
        block,
        history,
        parent_slot,
    } = record;
    let RetainedBlock {
        slot,
        epoch,
        block_id,
        features,
        commitment,
        history_len,
        receipts,
        accounts_delta,
        // Not written, see the module documentation.
        accounts_state: _,
        accounts_state_not_kept: _,
        state_record,
        epoch_boundary,
    } = block;
    writer.u64(*slot);
    writer.u64(*epoch);
    writer.u64(*parent_slot);
    writer.option(block_id.as_ref(), Writer::hash);
    let ActiveFeatures {
        block_commitment_root,
        transaction_receipt_tree,
        accounts_delta_root,
        epoch_validator_set_root,
        epoch_boundary_commitment,
        accounts_state_root,
    } = *features;
    for active in [
        block_commitment_root,
        transaction_receipt_tree,
        accounts_delta_root,
        epoch_validator_set_root,
        epoch_boundary_commitment,
        accounts_state_root,
    ] {
        writer.u8(u8::from(active));
    }
    match commitment {
        Ok((commitment_root, vector)) => {
            writer.u8(0);
            writer.hash(commitment_root);
            for value in &vector.0 {
                writer.hash(value);
            }
        }
        Err(reason) => {
            writer.u8(1);
            writer.var_bytes(reason.as_bytes());
        }
    }
    writer.option(*history_len, Writer::u64);
    writer.option(history.as_ref(), encode_frontier);
    match receipts {
        None => writer.u8(0),
        Some(Ok(receipts)) => {
            writer.u8(1);
            writer.len(receipts.len());
            for retained in receipts.iter() {
                encode_receipt(writer, retained);
            }
        }
        Some(Err(reason)) => {
            writer.u8(2);
            writer.var_bytes(reason.as_bytes());
        }
    }
    writer.option(accounts_delta.as_deref(), |writer, delta| {
        writer.len(delta.entries().len());
        for entry in delta.entries() {
            writer.bytes(&entry.address);
            writer.hash(&entry.account_hash);
        }
    });
    writer.option(epoch_boundary.as_ref(), |writer, boundary| {
        let EpochBoundary {
            epoch,
            first_slot,
            parent_slot,
            parent_block_id,
            parent_commitment_root,
            history_len,
        } = boundary;
        writer.u64(*epoch);
        writer.u64(*first_slot);
        writer.u64(*parent_slot);
        writer.hash(parent_block_id);
        writer.hash(parent_commitment_root);
        writer.u64(*history_len);
    });
    // Last, so that a file written before this was added still parses.
    writer.option(state_record.as_deref(), |writer, record| {
        let StateRecord {
            parent_slot,
            generation_in_use,
            generations,
        } = record;
        writer.u64(*parent_slot);
        writer.option(*generation_in_use, Writer::u64);
        writer.len(generations.len());
        for GenerationRecord { base, previous } in generations {
            writer.u64(base.epoch);
            writer.option(base.parent, |writer, (slot, bank_hash)| {
                writer.u64(slot);
                writer.hash(&bank_hash);
            });
            writer.option(previous.as_ref(), |writer, previous| {
                writer.len(previous.len());
                for hash in previous {
                    writer.option(hash.as_ref(), Writer::hash);
                }
            });
        }
    });
}

fn decode_block(reader: &mut Reader) -> Option<BlockRecord> {
    let slot = reader.u64()?;
    let epoch = reader.u64()?;
    let parent_slot = reader.u64()?;
    let block_id = reader.option(Reader::hash)?;
    let mut flag = || match reader.u8()? {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    };
    let features = ActiveFeatures {
        block_commitment_root: flag()?,
        transaction_receipt_tree: flag()?,
        accounts_delta_root: flag()?,
        epoch_validator_set_root: flag()?,
        epoch_boundary_commitment: flag()?,
        accounts_state_root: flag()?,
    };
    let commitment = match reader.u8()? {
        0 => {
            let commitment_root = reader.hash()?;
            let mut vector = CommitmentVector::default();
            for value in &mut vector.0 {
                *value = reader.hash()?;
            }
            Ok((commitment_root, vector))
        }
        1 => Err(reader.string()?),
        _ => return None,
    };
    let history_len = reader.option(Reader::u64)?;
    let history = reader.option(decode_frontier)?;
    let receipts = match reader.u8()? {
        0 => None,
        1 => Some(Ok(Arc::new(reader.vec(decode_receipt)?))),
        2 => Some(Err(reader.string()?)),
        _ => return None,
    };
    let accounts_delta = reader.option(|reader| {
        let entries =
            reader.vec(|reader| Some(DeltaEntry::new(reader.array()?, reader.hash()?)))?;
        AccountsDelta::new(entries).ok().map(Arc::new)
    })?;
    let epoch_boundary = reader.option(|reader| {
        Some(EpochBoundary {
            epoch: reader.u64()?,
            first_slot: reader.u64()?,
            parent_slot: reader.u64()?,
            parent_block_id: reader.hash()?,
            parent_commitment_root: reader.hash()?,
            history_len: reader.u64()?,
        })
    })?;
    let state_record = if reader.0.is_empty() {
        None
    } else {
        reader.option(|reader| {
            Some(Arc::new(StateRecord {
                parent_slot: reader.u64()?,
                generation_in_use: reader.option(Reader::u64)?,
                generations: reader.vec(|reader| {
                    Some(GenerationRecord {
                        base: BaseId {
                            epoch: reader.u64()?,
                            parent: reader
                                .option(|reader| Some((reader.u64()?, reader.hash()?)))?,
                        },
                        previous: reader
                            .option(|reader| reader.vec(|reader| reader.option(Reader::hash)))?,
                    })
                })?,
            }))
        })?
    };
    Some(BlockRecord {
        block: RetainedBlock {
            slot,
            epoch,
            block_id,
            features,
            commitment,
            history_len,
            receipts,
            accounts_delta,
            accounts_state: None,
            accounts_state_not_kept: true,
            state_record,
            epoch_boundary,
        },
        history,
        parent_slot,
    })
}

fn encode_certificate(writer: &mut Writer, certificate: &Certificate) {
    writer.bytes(&certificate.signature.0);
    writer.var_bytes(&certificate.bitmap);
}

fn decode_certificate(reader: &mut Reader, kind: CertificateKind) -> Option<Certificate> {
    Some(Certificate {
        kind,
        signature: BlsSignature(reader.array()?),
        bitmap: reader.var_bytes()?,
    })
}

fn encode_certificates(writer: &mut Writer, certificates: &SlotCertificates) {
    let SlotCertificates {
        notarize,
        finalize,
        fast_finalize,
    } = certificates;
    for list in [notarize, fast_finalize] {
        writer.len(list.len());
        for (block_id, certificate) in list {
            writer.hash(block_id);
            encode_certificate(writer, certificate);
        }
    }
    writer.option(finalize.as_ref(), encode_certificate);
}

fn decode_certificates(reader: &mut Reader) -> Option<SlotCertificates> {
    let list = |reader: &mut Reader, kind| {
        reader.vec(|reader| Some((reader.hash()?, decode_certificate(reader, kind)?)))
    };
    Some(SlotCertificates {
        notarize: list(reader, CertificateKind::Notarize)?,
        fast_finalize: list(reader, CertificateKind::FastFinalize)?,
        finalize: reader.option(|reader| decode_certificate(reader, CertificateKind::Finalize))?,
    })
}
