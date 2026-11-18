//! Light-client proof methods.
//!
//! Each method returns the JSON form ([`solana_rpc_client_api::light_client`])
//! of a proof that `solana-light-client` verifies; the `rpc` feature of that
//! crate converts it back. Proofs are served from the
//! [`LightClientProofStore`] only, for finalized blocks; every reason a proof
//! cannot be served is a [`LightClientProofError`], with its own error
//! code.
//!
//! | Method                 | Parameters                            |
//! |------------------------|---------------------------------------|
//! | `getFinalityProof`     | `slot`                                |
//! | `getBlockHistoryProof` | `slot`, `atSlot`                      |
//! | `getTransactionProof`  | `signature`, `{instructions, events}` |
//! | `getAccountDeltaProof` | `pubkey`, `slot`                      |
//! | `getAccountStateProof` | `pubkey`, `{slot}`                    |
//! | `getValidatorSet`      | `epoch`, `{atSlot}`                   |
//! | `getEpochBoundary`     | `epoch`, `{atSlot}`                   |

use {
    super::{JsonRpcRequestProcessor, verify_pubkey, verify_signature},
    crate::light_client_proofs::LightClientProofStore,
    jsonrpc_core::{Error, Result},
    jsonrpc_derive::rpc,
    solana_clock::{Epoch, Slot},
    solana_commitment_config::CommitmentConfig,
    solana_rpc_client_api::{
        custom_error::{LightClientProofError, RpcCustomError},
        light_client::*,
    },
};

fn proof_error(err: LightClientProofError) -> Error {
    RpcCustomError::LightClientProof(err).into()
}

impl JsonRpcRequestProcessor {
    fn light_client_proof_store(&self) -> Result<&LightClientProofStore> {
        self.light_client_proofs
            .as_deref()
            .filter(|_| self.config.enable_light_client_proofs)
            .ok_or_else(|| proof_error(LightClientProofError::ProofsDisabled))
    }
}

#[rpc]
pub trait LightClient {
    type Metadata;

    #[rpc(meta, name = "getFinalityProof")]
    fn get_finality_proof(&self, meta: Self::Metadata, slot: Slot) -> Result<RpcFinalityProof>;

    #[rpc(meta, name = "getBlockHistoryProof")]
    fn get_block_history_proof(
        &self,
        meta: Self::Metadata,
        slot: Slot,
        at_slot: Slot,
    ) -> Result<RpcBlockHistoryProof>;

    #[rpc(meta, name = "getTransactionProof")]
    fn get_transaction_proof(
        &self,
        meta: Self::Metadata,
        signature_str: String,
        config: Option<RpcTransactionProofConfig>,
    ) -> Result<RpcTransactionProof>;

    #[rpc(meta, name = "getAccountDeltaProof")]
    fn get_account_delta_proof(
        &self,
        meta: Self::Metadata,
        pubkey_str: String,
        slot: Slot,
    ) -> Result<RpcAccountDeltaProof>;

    #[rpc(meta, name = "getAccountStateProof")]
    fn get_account_state_proof(
        &self,
        meta: Self::Metadata,
        pubkey_str: String,
        config: Option<RpcAccountStateProofConfig>,
    ) -> Result<RpcAccountStateProof>;

    #[rpc(meta, name = "getValidatorSet")]
    fn get_validator_set(
        &self,
        meta: Self::Metadata,
        epoch: Epoch,
        config: Option<RpcCommittingBlockConfig>,
    ) -> Result<RpcValidatorSet>;

    #[rpc(meta, name = "getEpochBoundary")]
    fn get_epoch_boundary(
        &self,
        meta: Self::Metadata,
        epoch: Epoch,
        config: Option<RpcCommittingBlockConfig>,
    ) -> Result<RpcEpochBoundary>;
}

pub struct LightClientImpl;

impl LightClient for LightClientImpl {
    type Metadata = JsonRpcRequestProcessor;

    fn get_finality_proof(&self, meta: Self::Metadata, slot: Slot) -> Result<RpcFinalityProof> {
        debug!("get_finality_proof rpc request received: {slot}");
        let response = meta
            .light_client_proof_store()?
            .finality_proof(slot)
            .map_err(proof_error)?;
        Ok(RpcFinalityProof::from(&response))
    }

    fn get_block_history_proof(
        &self,
        meta: Self::Metadata,
        slot: Slot,
        at_slot: Slot,
    ) -> Result<RpcBlockHistoryProof> {
        debug!("get_block_history_proof rpc request received: {slot} at {at_slot}");
        let response = meta
            .light_client_proof_store()?
            .block_history_proof(slot, at_slot)
            .map_err(proof_error)?;
        Ok(RpcBlockHistoryProof::from(&response))
    }

    fn get_transaction_proof(
        &self,
        meta: Self::Metadata,
        signature_str: String,
        config: Option<RpcTransactionProofConfig>,
    ) -> Result<RpcTransactionProof> {
        debug!("get_transaction_proof rpc request received: {signature_str}");
        let signature = verify_signature(&signature_str)?;
        let config = config.unwrap_or_default();
        let store = meta.light_client_proof_store()?;
        let response = store
            .transaction_proof(
                &signature,
                config.instructions.as_ref(),
                config.events.as_ref(),
            )
            .map_err(|err| match err {
                // The store knows the transactions of finalized blocks only:
                // a transaction this node processed in a block it has not
                // finalized yet is not final, rather than not found.
                LightClientProofError::NotFound { .. } => {
                    let finalized_slot = store.highest_root().unwrap_or_default();
                    match meta
                        .bank(Some(CommitmentConfig::processed()))
                        .get_signature_status_slot(&signature)
                    {
                        Some((slot, _)) if slot > finalized_slot => {
                            LightClientProofError::BlockNotFinal {
                                slot,
                                finalized_slot,
                            }
                        }
                        _ => err,
                    }
                }
                err => err,
            })
            .map_err(proof_error)?;
        Ok(RpcTransactionProof::from(&response))
    }

    fn get_account_delta_proof(
        &self,
        meta: Self::Metadata,
        pubkey_str: String,
        slot: Slot,
    ) -> Result<RpcAccountDeltaProof> {
        debug!("get_account_delta_proof rpc request received: {pubkey_str} at {slot}");
        let pubkey = verify_pubkey(&pubkey_str)?;
        let response = meta
            .light_client_proof_store()?
            .account_delta_proof(&pubkey, slot)
            .map_err(proof_error)?;
        Ok(RpcAccountDeltaProof::from(&response))
    }

    fn get_account_state_proof(
        &self,
        meta: Self::Metadata,
        pubkey_str: String,
        config: Option<RpcAccountStateProofConfig>,
    ) -> Result<RpcAccountStateProof> {
        debug!("get_account_state_proof rpc request received: {pubkey_str}");
        let pubkey = verify_pubkey(&pubkey_str)?;
        let slot = config.unwrap_or_default().slot;
        let response = meta
            .light_client_proof_store()?
            .account_state_proof(&pubkey, slot)
            .map_err(proof_error)?;
        Ok(RpcAccountStateProof::from(&response))
    }

    fn get_validator_set(
        &self,
        meta: Self::Metadata,
        epoch: Epoch,
        config: Option<RpcCommittingBlockConfig>,
    ) -> Result<RpcValidatorSet> {
        debug!("get_validator_set rpc request received: {epoch}");
        let at_slot = config.unwrap_or_default().at_slot;
        let response = meta
            .light_client_proof_store()?
            .validator_set(epoch, at_slot)
            .map_err(proof_error)?;
        Ok(RpcValidatorSet::from(&response))
    }

    fn get_epoch_boundary(
        &self,
        meta: Self::Metadata,
        epoch: Epoch,
        config: Option<RpcCommittingBlockConfig>,
    ) -> Result<RpcEpochBoundary> {
        debug!("get_epoch_boundary rpc request received: {epoch}");
        let at_slot = config.unwrap_or_default().at_slot;
        let response = meta
            .light_client_proof_store()?
            .epoch_boundary(epoch, at_slot)
            .map_err(proof_error)?;
        Ok(RpcEpochBoundary::from(&response))
    }
}

#[cfg(test)]
mod tests;
