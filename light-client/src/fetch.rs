//! Fetching: a light client that asks a node for the proofs it needs and
//! verifies them.
//!
//! [`ProofFetcher`] holds a [`LightClient`] and a [`ProofSource`], the
//! node's seven proof methods (`getFinalityProof` and the others). Each of
//! its methods fetches what one question needs, verifies it, and returns
//! only verified values. It follows the validator set by itself: asked
//! about a block of a later epoch than the one it trusts, it first moves
//! there one epoch at a time, each step checked with the signer set of the
//! epoch before.
//!
//! The source is not trusted. A wrong answer fails verification
//! ([`FetchError::Verify`], [`FetchError::Wire`], [`FetchError::Mismatch`]);
//! a source can only withhold answers. Nothing here shows that a block is
//! the latest one: callers apply their own freshness rule.
//!
//! The crate has no transport of its own. With the `rpc-client` feature,
//! `solana_rpc_client::rpc_client::RpcClient` is a [`ProofSource`]; anything
//! else that can call the seven methods can implement the trait.

use {
    crate::{
        BlockRef, CommittedBlock, Error, LightClient, VerifiedAncestor, VerifiedEpochBoundary,
        VerifiedEvent, VerifiedInstruction, VerifiedTransaction,
        rpc::{
            AccountDelta, AccountDeltaResponse, AccountStateResponse, BlockHistoryResponse,
            EpochBoundaryResponse, Finality, FinalityResponse, TransactionResponse,
            ValidatorSetResponse, VerifiedFinal, WireError,
        },
    },
    solana_hash::Hash,
    solana_pubkey::Pubkey,
    solana_rpc_client_types::light_client::{
        RpcAccountDeltaProof, RpcAccountStateProof, RpcBlockHistoryProof, RpcEpochBoundary,
        RpcFinalityProof, RpcProofSelection, RpcTransactionProof, RpcValidatorSet,
    },
    std::{collections::BTreeMap, time::Duration},
};

/// JSON-RPC error codes of the proof methods (`LightClientProofError` on
/// the node).
pub mod error_code {
    pub const PROOFS_DISABLED: i64 = -32030;
    pub const FEATURE_INACTIVE: i64 = -32031;
    pub const BLOCK_NOT_FINAL: i64 = -32032;
    pub const BLOCK_NOT_RETAINED: i64 = -32033;
    pub const CERTIFICATES_NOT_AVAILABLE: i64 = -32034;
    pub const DATA_NOT_AVAILABLE: i64 = -32035;
    pub const BASE_TREE_NOT_READY: i64 = -32036;
    pub const NOT_FOUND: i64 = -32037;
}

/// Why a [`ProofSource`] gave no proof.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SourceError {
    /// The node cannot serve this yet and may be able to shortly: the block
    /// is not final on it, its certificates have not arrived, or a base
    /// tree is still being computed. Worth asking again.
    #[error("not ready: {0}")]
    NotReady(String),
    /// The node answered that it cannot serve this: proofs are disabled,
    /// the feature is not active for the block, the block is no longer
    /// retained, or there is no such item.
    #[error("not available: {0}")]
    NotAvailable(String),
    /// No answer from the node (connection, encoding).
    #[error("transport: {0}")]
    Transport(String),
}

impl SourceError {
    /// The error for a JSON-RPC error response of a proof method.
    pub fn from_rpc_error(code: i64, message: impl Into<String>) -> Self {
        let message = message.into();
        match code {
            error_code::BLOCK_NOT_FINAL
            | error_code::CERTIFICATES_NOT_AVAILABLE
            | error_code::BASE_TREE_NOT_READY => Self::NotReady(message),
            _ => Self::NotAvailable(message),
        }
    }
}

/// The node's proof methods. Answers are in their JSON form; the fetcher
/// decodes and verifies them.
pub trait ProofSource {
    /// `getFinalityProof(slot)`.
    fn finality_proof(&self, slot: u64) -> Result<RpcFinalityProof, SourceError>;

    /// `getBlockHistoryProof(slot, atSlot)`.
    fn block_history_proof(
        &self,
        slot: u64,
        at_slot: u64,
    ) -> Result<RpcBlockHistoryProof, SourceError>;

    /// `getTransactionProof(signature, {instructions, events})`.
    fn transaction_proof(
        &self,
        signature: &[u8; 64],
        instructions: Option<RpcProofSelection>,
        events: Option<RpcProofSelection>,
    ) -> Result<RpcTransactionProof, SourceError>;

    /// `getAccountDeltaProof(pubkey, slot)`.
    fn account_delta_proof(
        &self,
        address: &Pubkey,
        slot: u64,
    ) -> Result<RpcAccountDeltaProof, SourceError>;

    /// `getAccountStateProof(pubkey, {slot})`.
    fn account_state_proof(
        &self,
        address: &Pubkey,
        slot: Option<u64>,
    ) -> Result<RpcAccountStateProof, SourceError>;

    /// `getValidatorSet(epoch, {atSlot})`.
    fn validator_set(
        &self,
        epoch: u64,
        at_slot: Option<u64>,
    ) -> Result<RpcValidatorSet, SourceError>;

    /// `getEpochBoundary(epoch, {atSlot})`.
    fn epoch_boundary(
        &self,
        epoch: u64,
        at_slot: Option<u64>,
    ) -> Result<RpcEpochBoundary, SourceError>;
}

impl<S: ProofSource + ?Sized> ProofSource for &S {
    fn finality_proof(&self, slot: u64) -> Result<RpcFinalityProof, SourceError> {
        (**self).finality_proof(slot)
    }

    fn block_history_proof(
        &self,
        slot: u64,
        at_slot: u64,
    ) -> Result<RpcBlockHistoryProof, SourceError> {
        (**self).block_history_proof(slot, at_slot)
    }

    fn transaction_proof(
        &self,
        signature: &[u8; 64],
        instructions: Option<RpcProofSelection>,
        events: Option<RpcProofSelection>,
    ) -> Result<RpcTransactionProof, SourceError> {
        (**self).transaction_proof(signature, instructions, events)
    }

    fn account_delta_proof(
        &self,
        address: &Pubkey,
        slot: u64,
    ) -> Result<RpcAccountDeltaProof, SourceError> {
        (**self).account_delta_proof(address, slot)
    }

    fn account_state_proof(
        &self,
        address: &Pubkey,
        slot: Option<u64>,
    ) -> Result<RpcAccountStateProof, SourceError> {
        (**self).account_state_proof(address, slot)
    }

    fn validator_set(
        &self,
        epoch: u64,
        at_slot: Option<u64>,
    ) -> Result<RpcValidatorSet, SourceError> {
        (**self).validator_set(epoch, at_slot)
    }

    fn epoch_boundary(
        &self,
        epoch: u64,
        at_slot: Option<u64>,
    ) -> Result<RpcEpochBoundary, SourceError> {
        (**self).epoch_boundary(epoch, at_slot)
    }
}

/// Why a [`ProofFetcher`] returned nothing verified.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FetchError {
    /// The source gave no proof.
    #[error(transparent)]
    Source(#[from] SourceError),
    /// The source's answer does not decode.
    #[error("the proof does not decode: {0}")]
    Wire(#[from] WireError),
    /// The source's answer does not verify.
    #[error("the proof does not verify: {0}")]
    Verify(#[from] Error),
    /// The source's answer is about something other than what was asked.
    #[error("the answer does not match the request: {0}")]
    Mismatch(&'static str),
    /// The block is in an epoch before the one the client started from.
    /// Its signer set cannot be learned from a later one.
    #[error("epoch {epoch} is before the first epoch this client trusts ({first_trusted})")]
    EpochBeforeTrusted { epoch: u64, first_trusted: u64 },
}

/// How often a [`ProofFetcher`] asks again when the source is not ready
/// ([`SourceError::NotReady`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Retry {
    /// Requests after the first one.
    pub attempts: u32,
    /// Pause before each of them (the thread sleeps).
    pub delay: Duration,
}

impl Retry {
    /// Ask once.
    pub const NONE: Self = Self {
        attempts: 0,
        delay: Duration::ZERO,
    };
}

/// A transaction verified in a final block, with the instructions and
/// events that were asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FetchedTransaction {
    pub block: VerifiedFinal,
    pub transaction: VerifiedTransaction,
    pub instructions: Vec<VerifiedInstruction>,
    pub events: Vec<VerifiedEvent>,
}

/// What a final block did to an account.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccountChange {
    /// The block changed the account and left it with the state that has
    /// this [`account_hash`](crate::account_hash).
    Changed { account_hash: Hash },
    /// The block deleted the account.
    Deleted,
    /// The block did not change the account.
    Unchanged,
}

/// An ancestor of a verified block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FetchedAncestor {
    /// A block with a commitment root, which further proofs can open.
    Block(VerifiedAncestor),
    /// The last block before the commitments were activated: an ancestor,
    /// without a commitment root.
    Anchor(BlockRef),
}

/// A light client that fetches the proofs it needs from a [`ProofSource`].
/// See the [module documentation](self).
#[derive(Clone, Debug)]
pub struct ProofFetcher<S> {
    source: S,
    /// The client for each epoch it has trusted a signer set for, from the
    /// one it started with to the latest it has moved to.
    clients: BTreeMap<u64, LightClient>,
    retry: Retry,
}

impl<S: ProofSource> ProofFetcher<S> {
    /// A fetcher that trusts the signer set of `client` and learns later
    /// ones from `source`.
    pub fn new(client: LightClient, source: S) -> Self {
        Self {
            source,
            clients: BTreeMap::from([(client.epoch(), client)]),
            retry: Retry::NONE,
        }
    }

    /// Asks again when the source is not ready, as `retry` says.
    pub fn with_retry(mut self, retry: Retry) -> Self {
        self.retry = retry;
        self
    }

    pub fn source(&self) -> &S {
        &self.source
    }

    /// The client of the latest epoch this fetcher has moved to.
    pub fn client(&self) -> &LightClient {
        self.clients
            .last_key_value()
            .map(|(_, client)| client)
            .expect("a fetcher always has a client")
    }

    /// The latest epoch whose signer set this fetcher trusts.
    pub fn epoch(&self) -> u64 {
        self.client().epoch()
    }

    /// The first epoch whose signer set this fetcher trusts.
    pub fn first_trusted_epoch(&self) -> u64 {
        self.clients
            .first_key_value()
            .map(|(epoch, _)| *epoch)
            .expect("a fetcher always has a client")
    }

    /// Forgets the signer sets of epochs before `epoch` (never the latest
    /// one). Blocks of those epochs can no longer be verified.
    pub fn forget_epochs_before(&mut self, epoch: u64) {
        let latest = self.epoch();
        self.clients = self.clients.split_off(&epoch.min(latest));
    }

    fn ask<T>(&self, request: impl Fn(&S) -> Result<T, SourceError>) -> Result<T, FetchError> {
        let mut attempts_left = self.retry.attempts;
        loop {
            match request(&self.source) {
                Err(SourceError::NotReady(_)) if attempts_left > 0 => {
                    attempts_left = attempts_left.saturating_sub(1);
                    std::thread::sleep(self.retry.delay);
                }
                result => return Ok(result?),
            }
        }
    }

    /// Moves to `epoch`, one epoch at a time: for each step, the signer set
    /// of the next epoch, the block of the current epoch that commits to
    /// it, and that block's finality, verified with the current signer
    /// set. Does nothing if the fetcher is already there or later.
    pub fn sync_to_epoch(&mut self, epoch: u64) -> Result<(), FetchError> {
        while self.epoch() < epoch {
            // `epoch() < epoch`, so there is a next epoch.
            let next = self.epoch().saturating_add(1);
            let response = self.ask(|source| source.validator_set(next, None))?;
            let response = ValidatorSetResponse::try_from(&response)?;
            if response.signer_set.epoch() != next {
                return Err(FetchError::Mismatch("signer set of another epoch"));
            }
            let committed_by = response
                .committed_by
                .as_ref()
                .ok_or(FetchError::Mismatch("no block commits to the signer set"))?;
            let mut client = self.client().clone();
            if client.epoch_schedule().get_epoch(committed_by.slot) != client.epoch() {
                return Err(FetchError::Mismatch(
                    "the committing block is not in the trusted epoch",
                ));
            }
            // With the trusted signer set only: certificates of a later
            // epoch cannot be checked yet.
            let finality = self.ask(|source| source.finality_proof(committed_by.slot))?;
            let block =
                FinalityResponse::try_from(&finality)?.verify(&client, committed_by.slot)?;
            if *block.block_id() != committed_by.block_id {
                return Err(FetchError::Mismatch("the committing block has another id"));
            }
            client.advance_epoch(&block, &committed_by.opening, response.signer_set)?;
            self.clients.insert(next, client);
        }
        Ok(())
    }

    /// The client that verifies certificates of `slot`'s epoch, moving
    /// forward to it if needed.
    fn client_for(&mut self, slot: u64) -> Result<&LightClient, FetchError> {
        let epoch = self.client().epoch_schedule().get_epoch(slot);
        self.sync_to_epoch(epoch)?;
        let first_trusted = self.first_trusted_epoch();
        self.clients
            .get(&epoch)
            .ok_or(FetchError::EpochBeforeTrusted {
                epoch,
                first_trusted,
            })
    }

    /// The block at `slot`, verified final.
    pub fn block(&mut self, slot: u64) -> Result<VerifiedFinal, FetchError> {
        let response = self.ask(|source| source.finality_proof(slot))?;
        let response = FinalityResponse::try_from(&response)?;
        // The certificates are those of the block, or of the descendant it
        // is final through, which may be in a later epoch.
        let certified_slot = match &response.finality {
            Finality::Certificates(_) => slot,
            Finality::Descendant(descendant) => descendant.slot,
        };
        let client = self.client_for(certified_slot)?;
        Ok(response.verify(client, slot)?)
    }

    /// The block at `slot`, verified final, which must have the id the
    /// source named in another answer.
    fn block_with_id(&mut self, slot: u64, block_id: &Hash) -> Result<VerifiedFinal, FetchError> {
        let block = self.block(slot)?;
        if block.block_id() != block_id {
            return Err(FetchError::Mismatch("the proof is for another block id"));
        }
        Ok(block)
    }

    /// The transaction with this signature, in a final block: its receipt
    /// (status and fee), and the instructions and events selected.
    pub fn transaction(
        &mut self,
        signature: &[u8; 64],
        instructions: Option<RpcProofSelection>,
        events: Option<RpcProofSelection>,
    ) -> Result<FetchedTransaction, FetchError> {
        let response = self.ask(|source| {
            source.transaction_proof(signature, instructions.clone(), events.clone())
        })?;
        let response = TransactionResponse::try_from(&response)?;
        let block = self.block_with_id(response.slot, &response.block_id)?;
        let transaction = response.transaction.verify(&block, signature)?;
        let instructions = response
            .instructions
            .iter()
            .map(|proof| proof.verify(&block, signature))
            .collect::<Result<_, _>>()?;
        let events = response
            .events
            .iter()
            .map(|proof| proof.verify(&block, signature))
            .collect::<Result<_, _>>()?;
        Ok(FetchedTransaction {
            block,
            transaction,
            instructions,
            events,
        })
    }

    /// What the final block at `slot` did to the account at `address`.
    pub fn account_change(
        &mut self,
        address: &Pubkey,
        slot: u64,
    ) -> Result<(VerifiedFinal, AccountChange), FetchError> {
        let response = self.ask(|source| source.account_delta_proof(address, slot))?;
        let response = AccountDeltaResponse::try_from(&response)?;
        if response.slot != slot {
            return Err(FetchError::Mismatch("account delta of another slot"));
        }
        let block = self.block_with_id(slot, &response.block_id)?;
        let change = match &response.delta {
            AccountDelta::Changed(proof) if proof.leaf.entry.is_deletion() => {
                proof.verify_deleted(&block, address)?;
                AccountChange::Deleted
            }
            AccountDelta::Changed(proof) => {
                let account_hash = proof.leaf.entry.account_hash;
                proof.verify(&block, address, &account_hash)?;
                AccountChange::Changed { account_hash }
            }
            AccountDelta::Unchanged(proof) => {
                proof.verify(&block, address)?;
                AccountChange::Unchanged
            }
        };
        Ok((block, change))
    }

    /// Verifies the state of the account at `address` at the end of the
    /// final block at `slot`, or of the latest block the source has:
    /// `account_hash` is the [`account_hash`](crate::account_hash) of the
    /// account the caller holds, or `None` to verify that there is no
    /// account. Returns the block the state is proven in.
    pub fn account_state(
        &mut self,
        address: &Pubkey,
        slot: Option<u64>,
        account_hash: Option<&Hash>,
    ) -> Result<VerifiedFinal, FetchError> {
        let response = self.ask(|source| source.account_state_proof(address, slot))?;
        let response = AccountStateResponse::try_from(&response)?;
        if slot.is_some_and(|slot| slot != response.slot) {
            return Err(FetchError::Mismatch("account state of another slot"));
        }
        let block = self.block_with_id(response.slot, &response.block_id)?;
        match account_hash {
            Some(account_hash) => response.proof.verify(&block, address, account_hash)?,
            None => response.proof.verify_nonexistent(&block, address)?,
        };
        Ok(block)
    }

    /// The block at `slot` as an ancestor of the final block at `at_slot`,
    /// through the block history of the latter: one finality proof covers
    /// both.
    pub fn ancestor(
        &mut self,
        slot: u64,
        at_slot: u64,
    ) -> Result<(VerifiedFinal, FetchedAncestor), FetchError> {
        let response = self.ask(|source| source.block_history_proof(slot, at_slot))?;
        let response = BlockHistoryResponse::try_from(&response)?;
        if response.at_slot != at_slot || response.history.slot != slot {
            return Err(FetchError::Mismatch("history proof of other blocks"));
        }
        let block = self.block_with_id(at_slot, &response.at_block_id)?;
        let ancestor = match response.history.verify(&block) {
            Ok(ancestor) => FetchedAncestor::Block(ancestor),
            Err(Error::AnchorLeaf) => {
                FetchedAncestor::Anchor(response.history.verify_anchor(&block)?)
            }
            Err(err) => return Err(err.into()),
        };
        Ok((block, ancestor))
    }

    /// Where the epoch before `epoch` ended, from a final block of `epoch`.
    pub fn epoch_boundary(
        &mut self,
        epoch: u64,
    ) -> Result<(VerifiedFinal, VerifiedEpochBoundary), FetchError> {
        let response = self.ask(|source| source.epoch_boundary(epoch, None))?;
        let response = EpochBoundaryResponse::try_from(&response)?;
        let block = self.block_with_id(response.slot, &response.block_id)?;
        let epoch_schedule = self.client().epoch_schedule().clone();
        if epoch_schedule.get_epoch(block.slot()) != epoch {
            return Err(FetchError::Mismatch("epoch boundary of another epoch"));
        }
        let boundary = response.proof.verify(&block, &epoch_schedule)?;
        Ok((block, boundary))
    }
}

#[cfg(feature = "rpc-client")]
mod rpc_client {
    use {
        super::*,
        solana_rpc_client::{
            api::{
                client_error::{Error as ClientError, ErrorKind},
                request::RpcError,
            },
            rpc_client::RpcClient,
        },
        solana_rpc_client_types::light_client::{
            RpcAccountStateProofConfig, RpcCommittingBlockConfig, RpcTransactionProofConfig,
        },
        solana_signature::Signature,
    };

    fn source_error(err: ClientError) -> SourceError {
        match err.kind() {
            ErrorKind::RpcError(RpcError::RpcResponseError { code, message, .. }) => {
                SourceError::from_rpc_error(*code, message.clone())
            }
            _ => SourceError::Transport(err.to_string()),
        }
    }

    impl ProofSource for RpcClient {
        fn finality_proof(&self, slot: u64) -> Result<RpcFinalityProof, SourceError> {
            self.get_finality_proof(slot).map_err(source_error)
        }

        fn block_history_proof(
            &self,
            slot: u64,
            at_slot: u64,
        ) -> Result<RpcBlockHistoryProof, SourceError> {
            self.get_block_history_proof(slot, at_slot)
                .map_err(source_error)
        }

        fn transaction_proof(
            &self,
            signature: &[u8; 64],
            instructions: Option<RpcProofSelection>,
            events: Option<RpcProofSelection>,
        ) -> Result<RpcTransactionProof, SourceError> {
            self.get_transaction_proof(
                &Signature::from(*signature),
                RpcTransactionProofConfig {
                    instructions,
                    events,
                },
            )
            .map_err(source_error)
        }

        fn account_delta_proof(
            &self,
            address: &Pubkey,
            slot: u64,
        ) -> Result<RpcAccountDeltaProof, SourceError> {
            self.get_account_delta_proof(address, slot)
                .map_err(source_error)
        }

        fn account_state_proof(
            &self,
            address: &Pubkey,
            slot: Option<u64>,
        ) -> Result<RpcAccountStateProof, SourceError> {
            self.get_account_state_proof(address, RpcAccountStateProofConfig { slot })
                .map_err(source_error)
        }

        fn validator_set(
            &self,
            epoch: u64,
            at_slot: Option<u64>,
        ) -> Result<RpcValidatorSet, SourceError> {
            self.get_validator_set(epoch, RpcCommittingBlockConfig { at_slot })
                .map_err(source_error)
        }

        fn epoch_boundary(
            &self,
            epoch: u64,
            at_slot: Option<u64>,
        ) -> Result<RpcEpochBoundary, SourceError> {
            self.get_epoch_boundary(epoch, RpcCommittingBlockConfig { at_slot })
                .map_err(source_error)
        }
    }
}
