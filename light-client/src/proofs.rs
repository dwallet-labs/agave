//! One proof type and one verifier per question a client can ask about a
//! block whose `commitment_root` is verified ([`CommittedBlock`]).
//!
//! Every proof carries the value of the commitment slot it starts from and
//! the [`SlotOpening`] of that slot; the verifier opens the slot first, then
//! checks the item against the value.

use {
    crate::{
        commitment::{CommitmentSlot, CommittedBlock, SlotOpening, VerifiedAncestor},
        error::Error,
        signer_set::SignerSet,
    },
    solana_epoch_schedule::EpochSchedule,
    solana_hash::Hash,
    solana_merkle_tree::{
        accounts_delta::{DELETED_ACCOUNT_HASH, DeltaLeafProof, UnchangedProof},
        accounts_state::AccountProof,
        commitment_tree::InclusionProof,
        epoch_boundary::EpochBoundary,
        receipt_tree::{
            EventRecord, InstructionAccountMeta, InstructionRecord, RECEIPT_VERSION, ReceiptStatus,
            SIGNATURE_BYTES, TransactionReceipt, verify_event, verify_instruction_record,
            verify_receipt,
        },
    },
    solana_pubkey::Pubkey,
};

/// The epoch of `block`, which must be `expected`.
fn check_block_epoch(
    block: &impl CommittedBlock,
    epoch_schedule: &EpochSchedule,
    expected: u64,
) -> Result<(), Error> {
    let block_epoch = epoch_schedule.get_epoch(block.slot());
    if block_epoch == expected {
        Ok(())
    } else {
        Err(Error::BlockEpoch {
            slot: block.slot(),
            block_epoch,
            expected,
        })
    }
}

// ---------------------------------------------------------------------------
// Transaction Receipt Tree (slot 1).

/// Proof of a transaction's receipt: its status and fee, by signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransactionProof {
    /// Value of commitment slot 1.
    pub receipts_root: Hash,
    pub opening: SlotOpening,
    pub receipt: TransactionReceipt,
    /// Path of the receipt in the receipts tree. The leaf index is the
    /// transaction's position in the block.
    pub proof: InclusionProof,
}

/// A transaction proven to be in a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedTransaction {
    /// Position of the transaction in the block.
    pub index: u64,
    pub receipt: TransactionReceipt,
}

impl VerifiedTransaction {
    pub fn is_success(&self) -> bool {
        self.receipt.status == ReceiptStatus::Success
    }

    /// Lamports charged to the fee payer.
    pub fn fee(&self) -> u64 {
        self.receipt.fee
    }
}

impl TransactionProof {
    /// Verifies that the transaction whose first signature is `signature` is
    /// in `block`, and returns its receipt.
    pub fn verify(
        &self,
        block: &impl CommittedBlock,
        signature: &[u8; SIGNATURE_BYTES],
    ) -> Result<VerifiedTransaction, Error> {
        self.opening.verify(
            block.commitment_root(),
            CommitmentSlot::ReceiptsRoot,
            &self.receipts_root,
        )?;
        if self.receipt.version != RECEIPT_VERSION {
            return Err(Error::UnsupportedReceiptVersion(self.receipt.version));
        }
        if self.receipt.signature != *signature {
            return Err(Error::SignatureMismatch);
        }
        if !verify_receipt(&self.proof, &self.receipt, &self.receipts_root) {
            return Err(Error::ReceiptProof);
        }
        Ok(VerifiedTransaction {
            index: self.proof.leaf_index,
            receipt: self.receipt,
        })
    }
}

/// Proof of one instruction executed by a successful transaction, at top
/// level or through CPI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstructionProof {
    pub transaction: TransactionProof,
    pub record: InstructionRecord,
    /// Path of the record in the transaction's instruction trace. The leaf
    /// index is the record's position in execution order.
    pub proof: InclusionProof,
}

/// An instruction proven to be executed by a transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedInstruction {
    pub transaction: VerifiedTransaction,
    /// Position of the record in the trace (execution order).
    pub index: u64,
    pub record: InstructionRecord,
}

impl VerifiedInstruction {
    /// Whether the instruction is `program_id` invoked at `stack_height`
    /// with exactly these accounts (with the flags it was invoked with) and
    /// this data.
    pub fn matches(
        &self,
        stack_height: u8,
        program_id: &Pubkey,
        accounts: &[InstructionAccountMeta],
        data: &[u8],
    ) -> bool {
        // `accounts_hash` binds a u16 account count; no instruction has more.
        if u16::try_from(accounts.len()).is_err() {
            return false;
        }
        InstructionRecord::new(
            stack_height,
            program_id.to_bytes(),
            accounts.iter().copied(),
            data,
        ) == self.record
    }
}

impl InstructionProof {
    /// Verifies the transaction `signature` in `block` and that it executed
    /// this instruction.
    pub fn verify(
        &self,
        block: &impl CommittedBlock,
        signature: &[u8; SIGNATURE_BYTES],
    ) -> Result<VerifiedInstruction, Error> {
        let transaction = self.transaction.verify(block, signature)?;
        if !transaction.is_success() {
            return Err(Error::TransactionFailed);
        }
        if !verify_instruction_record(
            &self.proof,
            &self.record,
            &transaction.receipt.instruction_trace_root,
        ) {
            return Err(Error::InstructionProof);
        }
        Ok(VerifiedInstruction {
            transaction,
            index: self.proof.leaf_index,
            record: self.record,
        })
    }
}

/// Proof of one event emitted by a successful transaction, together with
/// the instruction that emitted it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventProof {
    /// The instruction whose trace record the event's `instruction_index`
    /// names.
    pub instruction: InstructionProof,
    pub event: EventRecord,
    /// Path of the event in the transaction's events tree. The leaf index is
    /// the event's position in emission order.
    pub proof: InclusionProof,
}

/// An event proven to be emitted by an instruction of a transaction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedEvent {
    pub instruction: VerifiedInstruction,
    /// Position of the event in emission order.
    pub index: u64,
    pub event: EventRecord,
}

impl EventProof {
    /// Verifies the transaction `signature` in `block`, the instruction, and
    /// that the instruction emitted this event: the event is in the
    /// transaction's events tree, its `instruction_index` is the position of
    /// the proven record and its `program_id` that record's program.
    pub fn verify(
        &self,
        block: &impl CommittedBlock,
        signature: &[u8; SIGNATURE_BYTES],
    ) -> Result<VerifiedEvent, Error> {
        let instruction = self.instruction.verify(block, signature)?;
        if !verify_event(
            &self.proof,
            &self.event,
            &instruction.transaction.receipt.events_root,
        ) {
            return Err(Error::EventProof);
        }
        if u64::from(self.event.instruction_index) != instruction.index
            || self.event.program_id != instruction.record.program_id
        {
            return Err(Error::EventInstructionMismatch);
        }
        Ok(VerifiedEvent {
            instruction,
            index: self.proof.leaf_index,
            event: self.event.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Accounts Delta Root (slot 2).

/// Proof of the state of an account at the end of a block that changed it,
/// or that the block deleted it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccountDeltaProof {
    /// Value of commitment slot 2.
    pub accounts_delta_root: Hash,
    pub opening: SlotOpening,
    pub leaf: DeltaLeafProof,
}

impl AccountDeltaProof {
    /// Verifies that `block` changed `address` and left it with the state
    /// whose [`account_hash`](crate::account_hash) is `account_hash` (not
    /// zero; see [`Self::verify_deleted`]).
    ///
    /// This is the state at the end of `block`; it says nothing about later
    /// blocks.
    pub fn verify(
        &self,
        block: &impl CommittedBlock,
        address: &Pubkey,
        account_hash: &Hash,
    ) -> Result<(), Error> {
        if *account_hash == DELETED_ACCOUNT_HASH {
            return Err(Error::ZeroAccountHash);
        }
        self.verify_hash(block, address, account_hash)
    }

    /// Verifies that `block` emptied `address` (its leaf carries the zero
    /// hash).
    pub fn verify_deleted(
        &self,
        block: &impl CommittedBlock,
        address: &Pubkey,
    ) -> Result<(), Error> {
        self.verify_hash(block, address, &DELETED_ACCOUNT_HASH)
    }

    fn verify_hash(
        &self,
        block: &impl CommittedBlock,
        address: &Pubkey,
        account_hash: &Hash,
    ) -> Result<(), Error> {
        self.opening.verify(
            block.commitment_root(),
            CommitmentSlot::AccountsDeltaRoot,
            &self.accounts_delta_root,
        )?;
        if self
            .leaf
            .verify_account(&address.to_bytes(), account_hash, &self.accounts_delta_root)
        {
            Ok(())
        } else {
            Err(Error::AccountDeltaProof)
        }
    }
}

/// Proof that a block did not change an account: two adjacent leaves around
/// the address, the first or the last leaf, or an empty tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccountUnchangedProof {
    /// Value of commitment slot 2.
    pub accounts_delta_root: Hash,
    pub opening: SlotOpening,
    pub proof: UnchangedProof,
}

impl AccountUnchangedProof {
    /// Verifies that `block` did not change `address`.
    pub fn verify(&self, block: &impl CommittedBlock, address: &Pubkey) -> Result<(), Error> {
        self.opening.verify(
            block.commitment_root(),
            CommitmentSlot::AccountsDeltaRoot,
            &self.accounts_delta_root,
        )?;
        if self
            .proof
            .verify(&address.to_bytes(), &self.accounts_delta_root)
        {
            Ok(())
        } else {
            Err(Error::AccountUnchangedProof)
        }
    }
}

// ---------------------------------------------------------------------------
// Accounts State Root (slot 5).

/// Proof of the current state of an account at the end of a block, or that
/// the address has no account: the opening of slot 5 to its generation, base
/// root and overlay root, and the overlay entry, or overlay non-membership
/// with the base leaf or its absence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccountStateProof(pub AccountProof);

impl AccountStateProof {
    /// Verifies that `address` has the state whose
    /// [`account_hash`](crate::account_hash) is `account_hash` (not zero; see
    /// [`Self::verify_nonexistent`]) at the end of `block`. Returns the
    /// generation (epoch of the base state) the proof is bound to.
    pub fn verify(
        &self,
        block: &impl CommittedBlock,
        address: &Pubkey,
        account_hash: &Hash,
    ) -> Result<u64, Error> {
        if *account_hash == Hash::default() {
            return Err(Error::ZeroAccountHash);
        }
        self.0
            .verify_current_state(block.commitment_root(), &address.to_bytes(), account_hash)
            .map_err(Error::AccountStateProof)
    }

    /// Verifies that `address` has no account at the end of `block`.
    /// Returns the generation the proof is bound to.
    pub fn verify_nonexistent(
        &self,
        block: &impl CommittedBlock,
        address: &Pubkey,
    ) -> Result<u64, Error> {
        self.0
            .verify_nonexistence(block.commitment_root(), &address.to_bytes())
            .map_err(Error::AccountStateProof)
    }
}

// ---------------------------------------------------------------------------
// Epoch Validator Set Root (slot 3).

/// Verifies that `next` is the signer set of the epoch after `block`'s: its
/// epoch is `epoch(block) + 1` and its `validator_set_root` is in slot 3 of
/// `block`.
pub fn verify_next_signer_set(
    block: &impl CommittedBlock,
    epoch_schedule: &EpochSchedule,
    opening: &SlotOpening,
    next: &SignerSet,
) -> Result<(), Error> {
    let block_epoch = epoch_schedule.get_epoch(block.slot());
    let expected = block_epoch.checked_add(1).ok_or(Error::NextEpoch {
        expected: block_epoch,
        actual: next.epoch(),
    })?;
    if next.epoch() != expected {
        return Err(Error::NextEpoch {
            expected,
            actual: next.epoch(),
        });
    }
    opening.verify(
        block.commitment_root(),
        CommitmentSlot::ValidatorSetRoot,
        &next.validator_set_root(),
    )
}

// ---------------------------------------------------------------------------
// Epoch Boundary Commitment (slot 4).

/// Proof of where the previous epoch ended on a block's chain: the six inputs
/// of `epoch_boundary` and the opening of slot 4.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpochBoundaryProof {
    pub boundary: EpochBoundary,
    pub opening: SlotOpening,
}

/// An epoch boundary proven by a block of `boundary.epoch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedEpochBoundary {
    pub boundary: EpochBoundary,
}

impl VerifiedEpochBoundary {
    /// `P`, the last block before the epoch, with its commitment root:
    /// end-of-epoch state is proven against it. `None` if `P` predates Block
    /// Commitment Root (zero commitment root); its slot and block id are
    /// still in `boundary`.
    pub fn last_block_before(&self) -> Option<VerifiedAncestor> {
        let boundary = &self.boundary;
        (boundary.parent_commitment_root != Hash::default()).then(|| {
            VerifiedAncestor::new(
                boundary.parent_slot,
                boundary.parent_block_id,
                boundary.parent_commitment_root,
            )
        })
    }
}

impl EpochBoundaryProof {
    /// Verifies the epoch boundary committed by `block`: every epoch before
    /// the block's is closed, the last block before it is `P`, and the epoch
    /// began at `F.slot`.
    ///
    /// Besides the opening, checks what holds for every boundary validators
    /// compute: it is for the block's epoch, `F` is in that epoch and not
    /// after the block, `P` is before `F` in an earlier epoch, `F` has a
    /// history, and a `P` without commitment root is the anchor leaf (history
    /// length 1).
    pub fn verify(
        &self,
        block: &impl CommittedBlock,
        epoch_schedule: &EpochSchedule,
    ) -> Result<VerifiedEpochBoundary, Error> {
        let boundary = &self.boundary;
        check_block_epoch(block, epoch_schedule, boundary.epoch)?;
        let consistent = epoch_schedule.get_epoch(boundary.first_slot) == boundary.epoch
            && boundary.first_slot <= block.slot()
            && boundary.parent_slot < boundary.first_slot
            && epoch_schedule.get_epoch(boundary.parent_slot) < boundary.epoch
            && boundary.history_len >= 1
            && (boundary.parent_commitment_root != Hash::default() || boundary.history_len == 1);
        if !consistent {
            return Err(Error::InvalidEpochBoundary);
        }
        self.opening.verify(
            block.commitment_root(),
            CommitmentSlot::EpochBoundary,
            &boundary.hash(),
        )?;
        Ok(VerifiedEpochBoundary {
            boundary: *boundary,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use {
        super::*,
        crate::{
            account::account_hash, commitment::tests::TestVector, signer_set::tests::test_signers,
        },
        solana_merkle_tree::{
            accounts_delta::{AccountsDelta, DeltaEntry},
            accounts_state::{
                AccountLookup, BaseLeaf, BaseTree, OverlayEntry, account_lookup,
                accounts_state_root, apply_block_changes, new_overlay,
            },
            commitment_tree::make_inclusion_proof,
            receipt_tree::{
                EventKind, event_inclusion_proof, events_root, instruction_record_inclusion_proof,
                instruction_trace_root, log_data_payload, receipt_inclusion_proof, receipts_root,
            },
        },
    };

    fn h(byte: u8) -> Hash {
        Hash::new_from_array([byte; 32])
    }

    fn meta(byte: u8, is_signer: bool, is_writable: bool) -> InstructionAccountMeta {
        InstructionAccountMeta {
            pubkey: [byte; 32],
            is_signer,
            is_writable,
        }
    }

    struct Block {
        records: Vec<InstructionRecord>,
        events: Vec<EventRecord>,
        receipts: Vec<TransactionReceipt>,
        vector: TestVector,
    }

    /// Three transactions; transaction 1 succeeded with three instructions
    /// and two events, transaction 2 failed.
    fn receipts_block() -> Block {
        let records = vec![
            InstructionRecord::new(
                1,
                [0x11; 32],
                [meta(1, true, true), meta(2, false, true)],
                b"top",
            ),
            InstructionRecord::new(2, [0x22; 32], [meta(2, false, true)], b"cpi"),
            InstructionRecord::new(1, [0x33; 32], [], b""),
        ];
        let events = vec![
            EventRecord {
                program_id: [0x22; 32],
                instruction_index: 1,
                kind: EventKind::LogData,
                payload: log_data_payload(&[b"x".as_slice(), b"yz"]),
            },
            EventRecord {
                program_id: [0x11; 32],
                instruction_index: 0,
                kind: EventKind::Log,
                payload: b"hello".to_vec(),
            },
        ];
        let receipts = vec![
            TransactionReceipt::success(
                [1; 64],
                h(1),
                5000,
                instruction_trace_root(&[]),
                events_root(&[]),
            ),
            TransactionReceipt::success(
                [2; 64],
                h(2),
                7000,
                instruction_trace_root(&records),
                events_root(&events),
            ),
            TransactionReceipt::failed([3; 64], h(3), 5000),
        ];
        let mut vector = TestVector::new();
        vector.set(CommitmentSlot::ReceiptsRoot, receipts_root(&receipts));
        Block {
            records,
            events,
            receipts,
            vector,
        }
    }

    fn transaction_proof(block: &Block, index: usize) -> TransactionProof {
        TransactionProof {
            receipts_root: receipts_root(&block.receipts),
            opening: block.vector.opening(CommitmentSlot::ReceiptsRoot),
            receipt: block.receipts[index],
            proof: receipt_inclusion_proof(&block.receipts, index).unwrap(),
        }
    }

    fn instruction_proof(block: &Block, index: usize) -> InstructionProof {
        InstructionProof {
            transaction: transaction_proof(block, 1),
            record: block.records[index],
            proof: instruction_record_inclusion_proof(&block.records, index).unwrap(),
        }
    }

    fn event_proof(block: &Block, index: usize) -> EventProof {
        let event = block.events[index].clone();
        EventProof {
            instruction: instruction_proof(block, event.instruction_index as usize),
            event,
            proof: event_inclusion_proof(&block.events, index).unwrap(),
        }
    }

    #[test]
    fn test_transaction_proof() {
        let block = receipts_block();
        let verified_block = block.vector.block(100);
        for (index, receipt) in block.receipts.iter().enumerate() {
            let proof = transaction_proof(&block, index);
            let verified = proof.verify(&verified_block, &receipt.signature).unwrap();
            assert_eq!(verified.index, index as u64);
            assert_eq!(verified.receipt, *receipt);
            assert_eq!(verified.is_success(), index != 2);
            assert_eq!(verified.fee(), receipt.fee);

            // Another signature.
            assert_eq!(
                proof.verify(&verified_block, &[9; 64]),
                Err(Error::SignatureMismatch)
            );
            // Tampered status or fee.
            let mut tampered = proof.clone();
            tampered.receipt.status = if index == 2 {
                ReceiptStatus::Success
            } else {
                ReceiptStatus::Failed
            };
            assert_eq!(
                tampered.verify(&verified_block, &receipt.signature),
                Err(Error::ReceiptProof)
            );
            let mut tampered = proof.clone();
            tampered.receipt.fee += 1;
            assert_eq!(
                tampered.verify(&verified_block, &receipt.signature),
                Err(Error::ReceiptProof)
            );
            // Tampered sibling.
            let mut tampered = proof.clone();
            tampered.proof.siblings[0] = h(0xee);
            assert_eq!(
                tampered.verify(&verified_block, &receipt.signature),
                Err(Error::ReceiptProof)
            );
            // Unknown version.
            let mut tampered = proof.clone();
            tampered.receipt.version = 2;
            assert_eq!(
                tampered.verify(&verified_block, &receipt.signature),
                Err(Error::UnsupportedReceiptVersion(2))
            );
            // Another block.
            assert_eq!(
                proof.verify(&TestVector::new().block(100), &receipt.signature),
                Err(Error::SlotOpening(CommitmentSlot::ReceiptsRoot))
            );
            // The opening of another slot.
            let mut tampered = proof.clone();
            tampered.opening = block.vector.opening(CommitmentSlot::AccountsDeltaRoot);
            assert_eq!(
                tampered.verify(&verified_block, &receipt.signature),
                Err(Error::SlotOpening(CommitmentSlot::ReceiptsRoot))
            );
        }
    }

    #[test]
    fn test_instruction_proof() {
        let block = receipts_block();
        let verified_block = block.vector.block(100);
        let signature = [2; 64];
        for (index, record) in block.records.iter().enumerate() {
            let proof = instruction_proof(&block, index);
            let verified = proof.verify(&verified_block, &signature).unwrap();
            assert_eq!(verified.index, index as u64);
            assert_eq!(verified.record, *record);
            assert_eq!(verified.transaction.index, 1);

            let mut tampered = proof.clone();
            tampered.record.data_hash = h(0xee);
            assert_eq!(
                tampered.verify(&verified_block, &signature),
                Err(Error::InstructionProof)
            );
            let mut tampered = proof.clone();
            tampered.proof.leaf_index = (index as u64 + 1) % 3;
            assert_eq!(
                tampered.verify(&verified_block, &signature),
                Err(Error::InstructionProof)
            );
        }
        let verified = instruction_proof(&block, 0)
            .verify(&verified_block, &signature)
            .unwrap();
        let program = Pubkey::new_from_array([0x11; 32]);
        let accounts = [meta(1, true, true), meta(2, false, true)];
        assert!(verified.matches(1, &program, &accounts, b"top"));
        assert!(!verified.matches(2, &program, &accounts, b"top"));
        assert!(!verified.matches(1, &program, &accounts, b"tOp"));
        assert!(!verified.matches(
            1,
            &program,
            &[meta(1, true, false), meta(2, false, true)],
            b"top"
        ));
        assert!(!verified.matches(1, &Pubkey::new_from_array([0x12; 32]), &accounts, b"top"));
        let too_many = vec![meta(1, false, false); usize::from(u16::MAX) + 1];
        assert!(!verified.matches(1, &program, &too_many, b"top"));

        // A failed transaction has no instructions: its trace root is the
        // empty tree's, so the record of another transaction cannot be shown
        // in it.
        let mut failed = instruction_proof(&block, 0);
        failed.transaction = transaction_proof(&block, 2);
        assert_eq!(
            failed.verify(&verified_block, &[3; 64]),
            Err(Error::TransactionFailed)
        );
    }

    #[test]
    fn test_event_proof() {
        let block = receipts_block();
        let verified_block = block.vector.block(100);
        let signature = [2; 64];
        for (index, event) in block.events.iter().enumerate() {
            let proof = event_proof(&block, index);
            let verified = proof.verify(&verified_block, &signature).unwrap();
            assert_eq!(verified.index, index as u64);
            assert_eq!(verified.event, *event);
            assert_eq!(
                verified.instruction.index,
                u64::from(event.instruction_index)
            );
            assert_eq!(verified.instruction.record.program_id, event.program_id);

            let mut tampered = proof.clone();
            tampered.event.payload.push(0);
            assert_eq!(
                tampered.verify(&verified_block, &signature),
                Err(Error::EventProof)
            );
            // Bound to another instruction than the one it names.
            let mut tampered = proof.clone();
            tampered.instruction = instruction_proof(&block, 2);
            assert_eq!(
                tampered.verify(&verified_block, &signature),
                Err(Error::EventInstructionMismatch)
            );
        }
        // An event attributed to the wrong program cannot be in the tree.
        let mut tampered = event_proof(&block, 0);
        tampered.event.program_id = [0x11; 32];
        assert_eq!(
            tampered.verify(&verified_block, &signature),
            Err(Error::EventProof)
        );
    }

    #[test]
    fn test_event_must_match_instruction_program() {
        // An events tree in which an event names instruction 0 but carries
        // another program id (which no valid block has): the event proof
        // verifies against the tree, the binding to the instruction does not.
        let records = vec![InstructionRecord::new(1, [0x11; 32], [], b"")];
        let events = vec![EventRecord {
            program_id: [0x99; 32],
            instruction_index: 0,
            kind: EventKind::Log,
            payload: b"x".to_vec(),
        }];
        let receipts = vec![TransactionReceipt::success(
            [5; 64],
            h(5),
            1,
            instruction_trace_root(&records),
            events_root(&events),
        )];
        let mut vector = TestVector::new();
        vector.set(CommitmentSlot::ReceiptsRoot, receipts_root(&receipts));
        let proof = EventProof {
            instruction: InstructionProof {
                transaction: TransactionProof {
                    receipts_root: receipts_root(&receipts),
                    opening: vector.opening(CommitmentSlot::ReceiptsRoot),
                    receipt: receipts[0],
                    proof: receipt_inclusion_proof(&receipts, 0).unwrap(),
                },
                record: records[0],
                proof: instruction_record_inclusion_proof(&records, 0).unwrap(),
            },
            event: events[0].clone(),
            proof: event_inclusion_proof(&events, 0).unwrap(),
        };
        assert_eq!(
            proof.verify(&vector.block(1), &[5; 64]),
            Err(Error::EventInstructionMismatch)
        );
    }

    fn delta_block() -> (AccountsDelta, TestVector) {
        let delta = AccountsDelta::new(vec![
            DeltaEntry::new([1; 32], h(0x81)),
            DeltaEntry::new([3; 32], DELETED_ACCOUNT_HASH),
            DeltaEntry::new([5; 32], h(0x85)),
        ])
        .unwrap();
        let mut vector = TestVector::new();
        vector.set(CommitmentSlot::AccountsDeltaRoot, delta.root());
        (delta, vector)
    }

    #[test]
    fn test_account_delta_proof() {
        let (delta, vector) = delta_block();
        let block = vector.block(100);
        let proof = |address: [u8; 32]| AccountDeltaProof {
            accounts_delta_root: delta.root(),
            opening: vector.opening(CommitmentSlot::AccountsDeltaRoot),
            leaf: delta.inclusion_proof(&address).unwrap(),
        };
        let address = Pubkey::new_from_array([1; 32]);
        let changed = proof([1; 32]);
        assert_eq!(changed.verify(&block, &address, &h(0x81)), Ok(()));
        assert_eq!(
            changed.verify(&block, &address, &h(0x82)),
            Err(Error::AccountDeltaProof)
        );
        assert_eq!(
            changed.verify(&block, &Pubkey::new_from_array([5; 32]), &h(0x81)),
            Err(Error::AccountDeltaProof)
        );
        assert_eq!(
            changed.verify_deleted(&block, &address),
            Err(Error::AccountDeltaProof)
        );
        assert_eq!(
            changed.verify(&block, &address, &Hash::default()),
            Err(Error::ZeroAccountHash)
        );
        let mut tampered = changed.clone();
        tampered.leaf.proof.siblings[0] = h(0xee);
        assert_eq!(
            tampered.verify(&block, &address, &h(0x81)),
            Err(Error::AccountDeltaProof)
        );
        assert_eq!(
            changed.verify(&TestVector::new().block(100), &address, &h(0x81)),
            Err(Error::SlotOpening(CommitmentSlot::AccountsDeltaRoot))
        );

        let deleted = proof([3; 32]);
        let address = Pubkey::new_from_array([3; 32]);
        assert_eq!(deleted.verify_deleted(&block, &address), Ok(()));
        assert_eq!(
            deleted.verify(&block, &address, &h(0x83)),
            Err(Error::AccountDeltaProof)
        );
    }

    #[test]
    fn test_account_unchanged_proof() {
        let (delta, vector) = delta_block();
        let block = vector.block(100);
        for byte in [0u8, 2, 4, 6, 0xff] {
            let address = Pubkey::new_from_array([byte; 32]);
            let proof = AccountUnchangedProof {
                accounts_delta_root: delta.root(),
                opening: vector.opening(CommitmentSlot::AccountsDeltaRoot),
                proof: delta.unchanged_proof(&address.to_bytes()).unwrap(),
            };
            assert_eq!(proof.verify(&block, &address), Ok(()));
            // Not for a changed address.
            for changed in [1u8, 3, 5] {
                assert_eq!(
                    proof.verify(&block, &Pubkey::new_from_array([changed; 32])),
                    Err(Error::AccountUnchangedProof)
                );
            }
        }
        // Two leaves around the address that are not adjacent.
        let address = Pubkey::new_from_array([2; 32]);
        let first = delta.inclusion_proof(&[1; 32]).unwrap();
        let last = delta.inclusion_proof(&[5; 32]).unwrap();
        let skipping = AccountUnchangedProof {
            accounts_delta_root: delta.root(),
            opening: vector.opening(CommitmentSlot::AccountsDeltaRoot),
            proof: UnchangedProof::Between {
                left: first,
                right: last,
            },
        };
        assert_eq!(
            skipping.verify(&block, &Pubkey::new_from_array([4; 32])),
            Err(Error::AccountUnchangedProof)
        );
        // An empty-tree claim against a non-empty tree.
        let empty = AccountUnchangedProof {
            accounts_delta_root: delta.root(),
            opening: vector.opening(CommitmentSlot::AccountsDeltaRoot),
            proof: UnchangedProof::EmptyTree,
        };
        assert_eq!(
            empty.verify(&block, &address),
            Err(Error::AccountUnchangedProof)
        );
    }

    #[test]
    fn test_account_state_proof() {
        // Base: accounts 2 and 4. Overlay: 4 changed, 6 created, 8 created
        // and deleted.
        let owner = Pubkey::new_unique();
        let state = |byte: u8, lamports: u64| {
            account_hash(
                lamports,
                &[byte],
                false,
                &owner,
                &Pubkey::new_from_array([byte; 32]),
            )
        };
        let base = BaseTree::new(vec![
            BaseLeaf::new([2; 32], state(2, 10)),
            BaseLeaf::new([4; 32], state(4, 10)),
        ])
        .unwrap();
        let mut overlay = new_overlay();
        apply_block_changes(
            &mut overlay,
            [
                ([4; 32], state(4, 10), state(4, 11)),
                ([6; 32], Hash::default(), state(6, 1)),
                ([8; 32], Hash::default(), state(8, 1)),
            ]
            .iter()
            .map(|(a, b, c)| (a, b, c)),
        );
        apply_block_changes(
            &mut overlay,
            [([8; 32], state(8, 1), Hash::default())]
                .iter()
                .map(|(a, b, c)| (a, b, c)),
        );
        let generation = 3;
        let mut vector = TestVector::new();
        vector.set(
            CommitmentSlot::AccountsStateRoot,
            accounts_state_root(generation, &base.root(), &overlay.root()),
        );
        let block = vector.block(100);
        let opening = solana_merkle_tree::accounts_state::AccountsStateOpening {
            generation,
            base_root: base.root(),
            overlay_root: overlay.root(),
            slot_proof: make_inclusion_proof(
                solana_merkle_tree::commitment_tree::TreeId::COMMITMENT_VECTOR,
                &vector.values,
                5,
            )
            .unwrap(),
        };
        let proof = |byte: u8| {
            AccountStateProof(AccountProof {
                opening: opening.clone(),
                lookup: account_lookup(&overlay, Some(&base), &[byte; 32]).unwrap(),
            })
        };
        let address = |byte: u8| Pubkey::new_from_array([byte; 32]);

        // Unchanged since the base: overlay non-membership and base leaf.
        let p2 = proof(2);
        assert!(matches!(p2.0.lookup, AccountLookup::Base { .. }));
        assert_eq!(
            p2.verify(&block, &address(2), &state(2, 10)),
            Ok(generation)
        );
        assert!(p2.verify(&block, &address(2), &state(2, 11)).is_err());
        assert!(p2.verify_nonexistent(&block, &address(2)).is_err());
        // Changed: overlay entry.
        let p4 = proof(4);
        assert!(matches!(p4.0.lookup, AccountLookup::Overlay { .. }));
        assert_eq!(
            p4.verify(&block, &address(4), &state(4, 11)),
            Ok(generation)
        );
        // The base state is not the current state.
        assert!(p4.verify(&block, &address(4), &state(4, 10)).is_err());
        assert_eq!(
            p4.verify(&block, &address(4), &Hash::default()),
            Err(Error::ZeroAccountHash)
        );
        // Created.
        assert_eq!(
            proof(6).verify(&block, &address(6), &state(6, 1)),
            Ok(generation)
        );
        // Created and deleted: no account (overlay entry with zero hash).
        let p8 = proof(8);
        assert_eq!(
            p8.0.lookup,
            AccountLookup::Overlay {
                entry: OverlayEntry::new([8; 32], Hash::default(), Hash::default()),
                proof: overlay.membership_proof(&[8; 32]).unwrap().1,
            }
        );
        assert_eq!(p8.verify_nonexistent(&block, &address(8)), Ok(generation));
        // Never existed: overlay non-membership and base absence.
        for byte in [0u8, 3, 5, 0xff] {
            let p = proof(byte);
            assert_eq!(p.verify_nonexistent(&block, &address(byte)), Ok(generation));
            assert!(p.verify(&block, &address(byte), &state(byte, 1)).is_err());
        }
        // Proof for another address.
        assert!(proof(3).verify_nonexistent(&block, &address(5)).is_err());
        // Another block.
        assert!(matches!(
            p2.verify(&TestVector::new().block(100), &address(2), &state(2, 10)),
            Err(Error::AccountStateProof(_))
        ));
    }

    #[test]
    fn test_next_signer_set() {
        let schedule = EpochSchedule::custom(64, 64, false);
        let (_, signers) = test_signers(&[5, 3, 2]);
        let next = SignerSet::new(4, signers.clone()).unwrap();
        let mut vector = TestVector::new();
        vector.set(CommitmentSlot::ValidatorSetRoot, next.validator_set_root());
        let opening = vector.opening(CommitmentSlot::ValidatorSetRoot);
        // A block of epoch 3.
        let block = vector.block(3 * 64 + 10);
        assert_eq!(
            verify_next_signer_set(&block, &schedule, &opening, &next),
            Ok(())
        );

        // The same signers bound to another epoch number.
        let wrong_epoch = SignerSet::new(5, signers.clone()).unwrap();
        assert_eq!(
            verify_next_signer_set(&block, &schedule, &opening, &wrong_epoch),
            Err(Error::NextEpoch {
                expected: 4,
                actual: 5
            })
        );
        // A block of another epoch, with the same vector.
        assert_eq!(
            verify_next_signer_set(&vector.block(4 * 64), &schedule, &opening, &next),
            Err(Error::NextEpoch {
                expected: 5,
                actual: 4
            })
        );
        // Other signers, or another stake.
        let (_, others) = test_signers(&[5, 3, 2]);
        assert_eq!(
            verify_next_signer_set(
                &block,
                &schedule,
                &opening,
                &SignerSet::new(4, others).unwrap()
            ),
            Err(Error::SlotOpening(CommitmentSlot::ValidatorSetRoot))
        );
        let mut more_stake = signers;
        more_stake[0].stake += 1;
        assert_eq!(
            verify_next_signer_set(
                &block,
                &schedule,
                &opening,
                &SignerSet::new(4, more_stake).unwrap()
            ),
            Err(Error::SlotOpening(CommitmentSlot::ValidatorSetRoot))
        );
        // The opening of another slot.
        assert_eq!(
            verify_next_signer_set(
                &block,
                &schedule,
                &vector.opening(CommitmentSlot::EpochBoundary),
                &next
            ),
            Err(Error::SlotOpening(CommitmentSlot::ValidatorSetRoot))
        );
    }

    #[test]
    fn test_epoch_boundary_proof() {
        let schedule = EpochSchedule::custom(64, 64, false);
        let boundary = EpochBoundary {
            epoch: 2,
            first_slot: 130,
            parent_slot: 100,
            parent_block_id: h(0x10),
            parent_commitment_root: h(0x20),
            history_len: 5,
        };
        let mut vector = TestVector::new();
        vector.set(CommitmentSlot::EpochBoundary, boundary.hash());
        let proof = EpochBoundaryProof {
            boundary,
            opening: vector.opening(CommitmentSlot::EpochBoundary),
        };
        for slot in [130, 160, 191] {
            let verified = proof.verify(&vector.block(slot), &schedule).unwrap();
            assert_eq!(verified.boundary, boundary);
            let last = verified.last_block_before().unwrap();
            assert_eq!(last.slot(), 100);
            assert_eq!(*last.block_id(), h(0x10));
            assert_eq!(*last.commitment_root(), h(0x20));
        }
        // A block of another epoch.
        assert_eq!(
            proof.verify(&vector.block(192), &schedule),
            Err(Error::BlockEpoch {
                slot: 192,
                block_epoch: 3,
                expected: 2
            })
        );
        // A block before `F`.
        assert_eq!(
            proof.verify(&vector.block(129), &schedule),
            Err(Error::InvalidEpochBoundary)
        );
        // Tampered inputs: each changes the hash, so the opening fails, or
        // is inconsistent.
        let tampered_inputs = [
            EpochBoundary {
                first_slot: 131,
                ..boundary
            },
            EpochBoundary {
                parent_slot: 99,
                ..boundary
            },
            EpochBoundary {
                parent_block_id: h(0x11),
                ..boundary
            },
            EpochBoundary {
                parent_commitment_root: h(0x21),
                ..boundary
            },
            EpochBoundary {
                history_len: 6,
                ..boundary
            },
        ];
        for tampered in tampered_inputs {
            let tampered = EpochBoundaryProof {
                boundary: tampered,
                opening: proof.opening,
            };
            assert_eq!(
                tampered.verify(&vector.block(160), &schedule),
                Err(Error::SlotOpening(CommitmentSlot::EpochBoundary))
            );
        }
        // Wrong epoch number in the boundary (the hash binds it).
        let mut vector3 = TestVector::new();
        let wrong_epoch = EpochBoundary {
            epoch: 3,
            first_slot: 194,
            ..boundary
        };
        vector3.set(CommitmentSlot::EpochBoundary, wrong_epoch.hash());
        let wrong = EpochBoundaryProof {
            boundary: wrong_epoch,
            opening: vector3.opening(CommitmentSlot::EpochBoundary),
        };
        assert!(wrong.verify(&vector3.block(200), &schedule).is_ok());
        assert_eq!(
            wrong.verify(&vector3.block(160), &schedule),
            Err(Error::BlockEpoch {
                slot: 160,
                block_epoch: 2,
                expected: 3
            })
        );
        // Inconsistent inputs are rejected even if a block committed to
        // them.
        for inconsistent in [
            EpochBoundary {
                first_slot: 120, // in epoch 1
                ..boundary
            },
            EpochBoundary {
                parent_slot: 130, // not before F
                ..boundary
            },
            EpochBoundary {
                parent_slot: 128, // in epoch 2
                ..boundary
            },
            EpochBoundary {
                history_len: 0,
                ..boundary
            },
            EpochBoundary {
                parent_commitment_root: Hash::default(), // anchor needs history_len 1
                ..boundary
            },
        ] {
            let mut vector = TestVector::new();
            vector.set(CommitmentSlot::EpochBoundary, inconsistent.hash());
            let proof = EpochBoundaryProof {
                boundary: inconsistent,
                opening: vector.opening(CommitmentSlot::EpochBoundary),
            };
            assert_eq!(
                proof.verify(&vector.block(160), &schedule),
                Err(Error::InvalidEpochBoundary),
                "{inconsistent:?}"
            );
        }
        // `P` predating Block Commitment Root: the anchor.
        let anchor = EpochBoundary {
            parent_commitment_root: Hash::default(),
            history_len: 1,
            ..boundary
        };
        let mut vector = TestVector::new();
        vector.set(CommitmentSlot::EpochBoundary, anchor.hash());
        let verified = EpochBoundaryProof {
            boundary: anchor,
            opening: vector.opening(CommitmentSlot::EpochBoundary),
        }
        .verify(&vector.block(160), &schedule)
        .unwrap();
        assert_eq!(verified.last_block_before(), None);
    }
}
