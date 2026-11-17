//! The light client: a trusted signer set and the cluster constants needed
//! to verify blocks with it.

use {
    crate::{
        block_id::FooterProof,
        certificate::FinalityProof,
        commitment::{
            BlockRef, CommittedBlock, HistoryProof, SlotOpening, VerifiedAncestor, VerifiedBlock,
        },
        error::Error,
        proofs::verify_next_signer_set,
        signer_set::SignerSet,
    },
    solana_epoch_schedule::EpochSchedule,
    solana_hash::Hash,
};

/// A light client: the cluster's epoch schedule and shred version, and the
/// signer set it trusts for one epoch.
///
/// It verifies blocks of that epoch with a finality proof and a footer proof,
/// and follows the validator set one epoch at a time. It never contacts the
/// network: proofs are fetched by the caller.
///
/// The starting signer set comes from a trusted source (a checkpoint shipped
/// with the software, or out of band). As the Epoch Validator Set Root SIMD
/// requires, callers must bound its age (weak subjectivity) and obtain a new
/// checkpoint when it is too old.
#[derive(Clone, Debug)]
pub struct LightClient {
    epoch_schedule: EpochSchedule,
    shred_version: u16,
    signer_set: SignerSet,
}

impl LightClient {
    pub fn new(epoch_schedule: EpochSchedule, shred_version: u16, signer_set: SignerSet) -> Self {
        Self {
            epoch_schedule,
            shred_version,
            signer_set,
        }
    }

    pub fn epoch_schedule(&self) -> &EpochSchedule {
        &self.epoch_schedule
    }

    /// The shred version included in every signed vote.
    pub fn shred_version(&self) -> u16 {
        self.shred_version
    }

    /// The trusted signer set.
    pub fn signer_set(&self) -> &SignerSet {
        &self.signer_set
    }

    /// The epoch whose blocks this client can verify.
    pub fn epoch(&self) -> u64 {
        self.signer_set.epoch()
    }

    /// Verifies that the block `(slot, block_id)` is final and opens its
    /// footer: `finality` must verify with the trusted signer set (so `slot`
    /// must be in its epoch), and `footer` must lead to `block_id`.
    pub fn verify_block(
        &self,
        slot: u64,
        block_id: &Hash,
        finality: &FinalityProof,
        footer: &FooterProof,
    ) -> Result<VerifiedBlock, Error> {
        finality.verify(
            &self.signer_set,
            &self.epoch_schedule,
            self.shred_version,
            slot,
            block_id,
        )?;
        footer.verify(block_id)?;
        if footer.parent_slot >= slot {
            return Err(Error::ParentNotBefore {
                slot,
                parent_slot: footer.parent_slot,
            });
        }
        Ok(VerifiedBlock::new(
            slot,
            *block_id,
            footer.bank_hash,
            footer.commitment_root,
            BlockRef {
                slot: footer.parent_slot,
                block_id: footer.parent_block_id,
            },
        ))
    }

    /// Verifies an ancestor of a verified block through its block history,
    /// giving the ancestor's commitment root without another certificate.
    pub fn verify_ancestor(
        &self,
        block: &impl CommittedBlock,
        proof: &HistoryProof,
    ) -> Result<VerifiedAncestor, Error> {
        proof.verify(block)
    }

    /// Moves to the next epoch: `block` is a verified block of the trusted
    /// epoch `E`, `opening` opens its slot 3, and `next` is the full signer
    /// set of epoch `E + 1`, which must hash to the opened
    /// `validator_set_root(E + 1)`. On success the client trusts `next`.
    pub fn advance_epoch(
        &mut self,
        block: &impl CommittedBlock,
        opening: &SlotOpening,
        next: SignerSet,
    ) -> Result<(), Error> {
        let epoch = self.signer_set.epoch();
        let block_epoch = self.epoch_schedule.get_epoch(block.slot());
        if block_epoch != epoch {
            return Err(Error::BlockEpoch {
                slot: block.slot(),
                block_epoch,
                expected: epoch,
            });
        }
        verify_next_signer_set(block, &self.epoch_schedule, opening, &next)?;
        self.signer_set = next;
        Ok(())
    }
}
