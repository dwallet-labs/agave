//! End-to-end tests of the light-client proofs on a local Alpenglow cluster.
//!
//! The validators run with all six light-client features active from
//! genesis (`LocalCluster` activates every feature), with the full RPC API
//! and `enable_light_client_proofs`, and 64-slot epochs (the shortest epoch
//! with an accounts state generation). The tests fetch proofs over JSON-RPC
//! and verify them with `solana-light-client`.
//!
//! The light client starts from the genesis config only: the signer set of
//! epoch 0 (computed from the genesis accounts), the epoch schedule and the
//! shred version (from the genesis hash). Everything else it accepts comes
//! from RPC responses and goes through `solana-light-client` verification.
//! Ground truth to compare verified values with (account states,
//! transaction statuses and fees, block contents, the rooted chain) comes
//! from the normal RPC methods.

#![allow(clippy::arithmetic_side_effects)]

use {
    agave_snapshots::snapshot_archive_info::SnapshotArchiveInfoGetter,
    bitvec::{order::Lsb0, vec::BitVec},
    log::*,
    serial_test::serial,
    solana_account::{Account, AccountSharedData},
    solana_clock::{Epoch, Slot},
    solana_commitment_config::CommitmentConfig,
    solana_core::validator::ValidatorConfig,
    solana_genesis_config::GenesisConfig,
    solana_hash::Hash,
    solana_instruction::{AccountMeta, Instruction},
    solana_keypair::{Keypair, keypair_from_seed},
    solana_light_client::{
        CertificateKind, CommittedBlock, EpochSchedule, Error as VerifyError, LightClient,
        SignerSet, account_hash,
        rpc::{
            AccountDelta, AccountDeltaResponse, AccountStateResponse, BlockHistoryResponse,
            EpochBoundaryResponse, Finality, FinalityResponse, TransactionResponse,
            ValidatorSetResponse, VerifiedFinal,
        },
        solana_merkle_tree::{
            accounts_state::{AccountLookup, BaseAbsenceProof, BaseLookup},
            receipt_tree::{EventKind, InstructionAccountMeta, ReceiptStatus, log_data_payload},
        },
    },
    solana_local_cluster::{
        cluster::Cluster,
        cluster_tests::TpuSender,
        integration_tests::{
            DEFAULT_NODE_STAKE, RUST_LOG_FILTER, ValidatorKeys, setup_snapshot_validator_config,
        },
        local_cluster::{ClusterConfig, LocalCluster},
        validator_configs::make_identical_validator_configs,
    },
    solana_native_token::LAMPORTS_PER_SOL,
    solana_net_utils::SocketAddrSpace,
    solana_poh_config::PohConfig,
    solana_program_binaries::bpf_loader_upgradeable_program_accounts,
    solana_pubkey::Pubkey,
    solana_rent::Rent,
    solana_rpc_client::rpc_client::RpcClient,
    solana_rpc_client_api::{
        client_error::{Error as ClientError, ErrorKind as ClientErrorKind},
        config::{RpcBlockConfig, RpcTransactionConfig},
        custom_error::{
            JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BASE_TREE_NOT_READY,
            JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_FINAL,
            JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_RETAINED,
            JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_CERTIFICATES_NOT_AVAILABLE,
            JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_DATA_NOT_AVAILABLE,
            JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_NOT_FOUND,
        },
        light_client::{
            RpcAccountStateProofConfig, RpcCommittingBlockConfig, RpcProofSelection,
            RpcTransactionProofConfig,
        },
        request::RpcError,
    },
    solana_runtime::bank::Bank,
    solana_sdk_ids::system_program,
    solana_signature::Signature,
    solana_signer::Signer,
    solana_signer_store::{Decoded, encode_base2},
    solana_system_transaction as system_transaction,
    solana_transaction::Transaction,
    solana_transaction_error::TransactionError,
    solana_transaction_status_client_types::{
        TransactionConfirmationStatus, TransactionDetails, UiTransactionEncoding,
        UiTransactionStatusMeta, option_serializer::OptionSerializer,
    },
    std::{
        collections::BTreeMap,
        fmt::{Debug, Display},
        num::NonZeroU64,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        thread::sleep,
        time::{Duration, Instant},
    },
};

/// The shortest epoch in which the accounts state root creates a generation
/// (`MIN_SLOTS_PER_EPOCH_FOR_GENERATION` of the Accounts State Root draft).
const SLOTS_PER_EPOCH: u64 = 64;

/// See `runtime/tests/fixtures/receipt_test_program/receipt_test_program.c`:
/// with data `[0, extra..]` it logs "outer", invokes itself with `[1]` (which
/// logs data `["inner", ""]` and transfers one lamport from account 0 to
/// account 1 through the system program), then logs data `["after",
/// extra]`; with data `[2]` it logs "failing" and fails.
const RECEIPT_TEST_PROGRAM: &[u8] =
    include_bytes!("../../runtime/tests/fixtures/receipt_test_program/receipt_test_program.so");

/// How long a transaction may take from being sent to being finalized, a
/// slot to be finalized, or a proof to become available.
const TIMEOUT: Duration = Duration::from_secs(90);

// ---------------------------------------------------------------------------
// Cluster.

struct TestCluster {
    cluster: LocalCluster,
    /// Address of the receipt test program, deployed in genesis.
    program: Pubkey,
    node_keys: Vec<ValidatorKeys>,
    tpu_sender: TpuSender,
}

fn light_client_validator_config() -> ValidatorConfig {
    let mut config = ValidatorConfig::default_for_test();
    // As in test_alpenglow_nodes_basic: all nodes start together.
    config.wait_for_supermajority = Some(0);
    // `default_for_test` has the full RPC API.
    assert!(config.rpc_config.full_api);
    config.rpc_config.enable_light_client_proofs = true;
    // Ground truth: getTransaction, getBlock.
    config.rpc_config.enable_rpc_transaction_history = true;
    // Log messages, to compare events with.
    config.rpc_config.enable_extended_tx_metadata_storage = true;
    config
}

/// Starts an Alpenglow cluster of `stakes.len()` validators with these
/// stakes, all in genesis, with 64-slot epochs (no warmup), a leader
/// schedule offset of one epoch (as on mainnet), and the receipt test
/// program deployed.
fn start_cluster(stakes: &[u64]) -> TestCluster {
    start_cluster_with(
        stakes,
        make_identical_validator_configs(&light_client_validator_config(), stakes.len()),
    )
}

fn start_cluster_with(stakes: &[u64], validator_configs: Vec<ValidatorConfig>) -> TestCluster {
    let num_nodes = stakes.len();
    let node_keys: Vec<_> = (0..num_nodes)
        .map(|i| ValidatorKeys {
            node_keypair: Arc::new(keypair_from_seed(&[i as u8 + 1; 32]).unwrap()),
            vote_keypair: Arc::new(Keypair::new()),
        })
        .collect();
    let program = Pubkey::new_unique();
    let mut config = ClusterConfig {
        validator_configs,
        validator_keys: Some(node_keys.iter().cloned().map(|keys| (keys, true)).collect()),
        node_stakes: stakes.to_vec(),
        slots_per_epoch: SLOTS_PER_EPOCH,
        stakers_slot_offset: SLOTS_PER_EPOCH,
        skip_warmup_slots: true,
        poh_config: PohConfig {
            target_tick_duration: PohConfig::default().target_tick_duration,
            hashes_per_tick: None,
            target_tick_count: None,
        },
        additional_accounts: bpf_loader_upgradeable_program_accounts(
            &program,
            RECEIPT_TEST_PROGRAM,
            &Pubkey::default(),
            &Rent::default(),
        )
        .into_iter()
        .map(|(address, account)| (address, AccountSharedData::from(account)))
        .collect(),
        ..ClusterConfig::default()
    };
    let cluster = LocalCluster::new(&mut config, SocketAddrSpace::Unspecified);
    assert_eq!(cluster.validators.len(), num_nodes);
    TestCluster {
        cluster,
        program,
        node_keys,
        tpu_sender: TpuSender::new(),
    }
}

impl TestCluster {
    fn node_pubkey(&self, node: usize) -> Pubkey {
        self.node_keys[node].node_keypair.pubkey()
    }

    /// An RPC client of `node`, at finalized commitment.
    fn rpc(&self, node: usize) -> RpcClient {
        let contact_info = &self.cluster.validators[&self.node_pubkey(node)]
            .info
            .contact_info;
        RpcClient::new_with_commitment(
            format!("http://{}", contact_info.rpc().unwrap()),
            CommitmentConfig::finalized(),
        )
    }

    fn genesis_config(&self) -> &GenesisConfig {
        &self.cluster.genesis_config
    }

    fn epoch_schedule(&self) -> &EpochSchedule {
        &self.cluster.genesis_config.epoch_schedule
    }

    /// Sends `transaction` to the TPU of the current leader until `rpc` has
    /// a status for it, then waits until it is finalized.
    fn send(&self, rpc: &RpcClient, transaction: &Transaction) -> SentTransaction {
        self.send_many(rpc, std::slice::from_ref(transaction))
            .pop()
            .unwrap()
    }

    /// Sends `transactions` together to the TPU of the current leader, again
    /// for those `rpc` has no status for, then waits until all are
    /// finalized.
    fn send_many(&self, rpc: &RpcClient, transactions: &[Transaction]) -> Vec<SentTransaction> {
        let signatures: Vec<_> = transactions.iter().map(|tx| tx.signatures[0]).collect();
        let wires: Vec<_> = transactions
            .iter()
            .map(|tx| wincode::serialize(tx).unwrap())
            .collect();
        let statuses = || {
            signatures
                .chunks(256)
                .flat_map(|chunk| rpc.get_signature_statuses(chunk).unwrap().value)
                .collect::<Vec<_>>()
        };
        let deadline = Instant::now() + TIMEOUT;
        loop {
            let leader = rpc.get_slot_leader().unwrap().to_string();
            let tpu = rpc
                .get_cluster_nodes()
                .unwrap()
                .into_iter()
                .find(|node| node.pubkey == leader)
                .and_then(|node| node.tpu_quic);
            if let Some(tpu) = tpu {
                let pending: Vec<_> = statuses()
                    .iter()
                    .zip(&wires)
                    .filter(|(status, _)| status.is_none())
                    .map(|(_, wire)| wire.clone())
                    .collect();
                // Keep the connection open while waiting, so that the
                // transactions are not dropped with it.
                let landed = self.tpu_sender.with_connection(tpu, |sender| {
                    for wire in pending {
                        self.tpu_sender.send_wire_transaction(sender, wire);
                    }
                    let resend = Instant::now() + Duration::from_secs(2);
                    loop {
                        if statuses().iter().all(Option::is_some) {
                            break true;
                        }
                        if Instant::now() > resend {
                            break false;
                        }
                        sleep(Duration::from_millis(100));
                    }
                });
                if landed {
                    break;
                }
            }
            assert!(
                Instant::now() < deadline,
                "transactions {signatures:?} did not land in {TIMEOUT:?}"
            );
        }
        let statuses = wait_for(
            format!("transactions {signatures:?} to be finalized"),
            TIMEOUT,
            || {
                statuses()
                    .into_iter()
                    .map(|status| {
                        status.filter(|status| {
                            status.confirmation_status
                                == Some(TransactionConfirmationStatus::Finalized)
                        })
                    })
                    .collect::<Option<Vec<_>>>()
            },
        );
        signatures
            .into_iter()
            .zip(statuses)
            .map(|(signature, status)| {
                info!(
                    "transaction {signature} finalized in slot {} ({:?})",
                    status.slot, status.err
                );
                SentTransaction {
                    signature,
                    slot: status.slot,
                    err: status.err,
                }
            })
            .collect()
    }
}

struct SentTransaction {
    signature: Signature,
    slot: Slot,
    err: Option<TransactionError>,
}

// ---------------------------------------------------------------------------
// Waiting.

/// Calls `f` until it returns a value; panics with `what` after `timeout`.
fn wait_for<T>(what: impl Display, timeout: Duration, mut f: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(value) = f() {
            return value;
        }
        assert!(
            Instant::now() < deadline,
            "timed out after {timeout:?} waiting for {what}"
        );
        sleep(Duration::from_millis(100));
    }
}

fn finalized_slot(rpc: &RpcClient) -> Slot {
    rpc.get_slot_with_commitment(CommitmentConfig::finalized())
        .unwrap()
}

/// Waits until the node has finalized `slot`; returns its finalized slot.
fn wait_for_finalized(rpc: &RpcClient, slot: Slot) -> Slot {
    wait_for(format!("slot {slot} to be finalized"), TIMEOUT, || {
        let finalized = finalized_slot(rpc);
        (finalized >= slot).then_some(finalized)
    })
}

/// The rooted slots in `[start, end]`, from `getBlocks`.
fn rooted_slots(rpc: &RpcClient, start: Slot, end: Slot) -> Vec<Slot> {
    rpc.get_blocks_with_commitment(start, Some(end), CommitmentConfig::finalized())
        .unwrap()
}

// ---------------------------------------------------------------------------
// Errors.

/// The JSON-RPC error code and message of `err`, if the node answered with
/// an error.
fn rpc_error(err: &ClientError) -> Option<(i64, &str)> {
    match err.kind() {
        ClientErrorKind::RpcError(RpcError::RpcResponseError { code, message, .. }) => {
            Some((*code, message.as_str()))
        }
        _ => None,
    }
}

/// Asserts that `result` is the light-client error `code`; returns the
/// message.
#[track_caller]
fn expect_proof_error<T: Debug>(result: Result<T, ClientError>, code: i64) -> String {
    match result {
        Ok(value) => panic!("expected error {code}, got {value:?}"),
        Err(err) => match rpc_error(&err) {
            Some((actual, message)) => {
                assert_eq!(actual, code, "{message}");
                message.to_string()
            }
            None => panic!("expected error {code}, got {err}"),
        },
    }
}

/// Calls the proof method `f` until the node has the block (it does not
/// fail with `BlockNotFinal`); the time waited is logged under `what`.
fn when_final<T>(what: impl Display, mut f: impl FnMut() -> Result<T, ClientError>) -> T {
    let start = Instant::now();
    let deadline = start + TIMEOUT;
    loop {
        match f() {
            Ok(value) => {
                let waited = start.elapsed();
                if waited > Duration::from_millis(1) {
                    debug!("{what}: available after {waited:?}");
                }
                return value;
            }
            Err(err) => match rpc_error(&err) {
                Some((JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_FINAL, _))
                    if Instant::now() < deadline =>
                {
                    sleep(Duration::from_millis(100))
                }
                _ => panic!("{what}: {err}"),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// The light client.

/// The light client, starting from the genesis config only, and statistics
/// of what it verified.
struct Verifier {
    client: LightClient,
    /// The client as it was in each earlier epoch, to verify older blocks
    /// with.
    past: BTreeMap<Epoch, LightClient>,
    /// For each epoch the client advanced to, the block of the previous
    /// epoch that committed to its signer set.
    committing: BTreeMap<Epoch, VerifiedFinal>,
    /// How each verified block was final.
    kinds: BTreeMap<&'static str, usize>,
    /// How often `getFinalityProof` of a block that `getBlocks` already
    /// listed as rooted failed, and why.
    transient: BTreeMap<i64, usize>,
    /// Blocks verified through a finalized descendant, and the descendant.
    through_descendant: BTreeMap<Slot, Slot>,
}

/// The signer set of epoch 0, from the genesis accounts (a bank built
/// locally from the genesis config; nothing from a node).
fn genesis_signer_set(genesis_config: &GenesisConfig) -> SignerSet {
    let bank = Bank::new_for_tests(genesis_config);
    let signers = bank
        .validator_set_signers(0)
        .expect("genesis has the stakes of epoch 0");
    SignerSet::new(0, signers).unwrap()
}

fn certificate_kinds(certificates: &[solana_light_client::Certificate]) -> &'static str {
    let kinds: Vec<_> = certificates
        .iter()
        .map(|certificate| certificate.kind)
        .collect();
    match kinds.as_slice() {
        [CertificateKind::FastFinalize] => "fast-finalize",
        [CertificateKind::Notarize, CertificateKind::Finalize] => "notarize+finalize",
        _ => "other",
    }
}

fn finality_kind(response: &FinalityResponse) -> &'static str {
    match &response.finality {
        Finality::Certificates(proof) => certificate_kinds(&proof.certificates),
        Finality::Descendant(descendant) => {
            match certificate_kinds(&descendant.finality.certificates) {
                "fast-finalize" => "through a descendant (fast-finalize)",
                "notarize+finalize" => "through a descendant (notarize+finalize)",
                _ => "through a descendant (other)",
            }
        }
    }
}

impl Verifier {
    fn from_genesis(genesis_config: &GenesisConfig) -> Self {
        let client = LightClient::new(
            genesis_config.epoch_schedule.clone(),
            solana_shred_version::compute_shred_version(&genesis_config.hash(), None),
            genesis_signer_set(genesis_config),
        );
        Self {
            client,
            past: BTreeMap::new(),
            committing: BTreeMap::new(),
            kinds: BTreeMap::new(),
            transient: BTreeMap::new(),
            through_descendant: BTreeMap::new(),
        }
    }

    /// The client that verifies blocks of the epoch of `slot`.
    fn client_for(&self, slot: Slot) -> &LightClient {
        let epoch = self.epoch_schedule().get_epoch(slot);
        self.past.get(&epoch).unwrap_or(&self.client)
    }

    fn epoch_schedule(&self) -> &EpochSchedule {
        self.client.epoch_schedule()
    }

    /// `getFinalityProof(slot)`, waiting while the block is not final or
    /// its certificates have not arrived.
    fn finality(&mut self, rpc: &RpcClient, slot: Slot) -> FinalityResponse {
        let transient = &mut self.transient;
        let response = wait_for(
            format!("a finality proof of slot {slot}"),
            TIMEOUT,
            || match rpc.get_finality_proof(slot) {
                Ok(response) => Some(response),
                Err(err) => match rpc_error(&err) {
                    Some((
                        code @ (JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_FINAL
                        | JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_CERTIFICATES_NOT_AVAILABLE),
                        _,
                    )) => {
                        *transient.entry(code).or_default() += 1;
                        None
                    }
                    // Seen once right after a block was rooted: the footer
                    // proof is read from blockstore, which did not have the
                    // complete block yet. Retried until the timeout, so a
                    // block that never becomes available still fails.
                    Some((code @ JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_DATA_NOT_AVAILABLE, _)) => {
                        info!("getFinalityProof({slot}), retrying: {err}");
                        *transient.entry(code).or_default() += 1;
                        None
                    }
                    _ => panic!("getFinalityProof({slot}): {err}"),
                },
            },
        );
        FinalityResponse::try_from(&response).unwrap()
    }

    /// Fetches and verifies the finality of the block at `slot`, first
    /// following the validator set to the epoch of `slot` if needed.
    fn verify_block(&mut self, rpc: &RpcClient, slot: Slot) -> VerifiedFinal {
        while self.client.epoch() < self.epoch_schedule().get_epoch(slot) {
            self.advance_epoch(rpc);
        }
        let response = self.finality(rpc, slot);
        self.verify_response(&response, slot)
    }

    fn verify_response(&mut self, response: &FinalityResponse, slot: Slot) -> VerifiedFinal {
        let verified = response
            .verify(self.client_for(slot), slot)
            .unwrap_or_else(|err| panic!("finality proof of slot {slot}: {err}: {response:?}"));
        assert_eq!(verified.slot(), slot);
        assert_eq!(*verified.block_id(), response.block_id);
        *self.kinds.entry(finality_kind(response)).or_default() += 1;
        if let Finality::Descendant(descendant) = &response.finality {
            self.through_descendant.insert(slot, descendant.slot);
        }
        verified
    }

    /// Moves the client from its epoch `E` to `E + 1`: `getValidatorSet(E +
    /// 1)` from a rooted block of `E` that the client can verify with the
    /// signer set of `E`, the finality of that block, and
    /// `LightClient::advance_epoch`. Returns the committing block.
    fn advance_epoch(&mut self, rpc: &RpcClient) -> VerifiedFinal {
        let epoch = self.client.epoch();
        let next = epoch + 1;
        let schedule = self.epoch_schedule().clone();
        // The node opens a block that is final with certificates of epoch
        // `E`: a block finalized only through a descendant in `E + 1` could
        // not be verified with the signer set of `E`.
        let response = when_final(format!("getValidatorSet({next})"), || {
            rpc.get_validator_set(next, RpcCommittingBlockConfig::default())
        });
        let response = ValidatorSetResponse::try_from(&response).unwrap();
        assert_eq!(response.signer_set.epoch(), next);
        let committed_by = response
            .committed_by
            .clone()
            .expect("a block of the previous epoch commits to the signer set");
        assert_eq!(schedule.get_epoch(committed_by.slot), epoch);
        let finality = self.finality(rpc, committed_by.slot);
        match &finality.finality {
            Finality::Certificates(_) => (),
            Finality::Descendant(descendant) => assert_eq!(
                schedule.get_epoch(descendant.slot),
                epoch,
                "block {} of epoch {epoch} is final only through block {} of a later epoch",
                committed_by.slot,
                descendant.slot
            ),
        }
        let block = self.verify_response(&finality, committed_by.slot);
        assert_eq!(*block.block_id(), committed_by.block_id);
        let signer_set = response.verify(&block, &schedule).unwrap().clone();
        self.past.insert(epoch, self.client.clone());
        self.client
            .advance_epoch(&block, &committed_by.opening, signer_set)
            .unwrap();
        assert_eq!(self.client.epoch(), next);
        self.committing.insert(next, block);
        info!(
            "light client advanced to epoch {next} through block {} ({} signers)",
            committed_by.slot,
            self.client.signer_set().len()
        );
        block
    }
}

// ---------------------------------------------------------------------------
// Ground truth.

fn hash_of(address: &Pubkey, account: &Account) -> Hash {
    account_hash(
        account.lamports,
        &account.data,
        account.executable,
        &account.owner,
        address,
    )
}

/// The finalized state of `address` (`None`: no account) and the slot of
/// the bank it is read from.
fn finalized_account(rpc: &RpcClient, address: &Pubkey) -> (Slot, Option<Account>) {
    let response = rpc
        .get_account_with_commitment(address, CommitmentConfig::finalized())
        .unwrap();
    (response.context.slot, response.value)
}

fn transaction_meta(rpc: &RpcClient, signature: &Signature) -> (Slot, UiTransactionStatusMeta) {
    let transaction = rpc
        .get_transaction_with_config(
            signature,
            RpcTransactionConfig {
                encoding: Some(UiTransactionEncoding::Base64),
                commitment: Some(CommitmentConfig::finalized()),
                max_supported_transaction_version: Some(0),
                min_context_slot: None,
            },
        )
        .unwrap();
    (transaction.slot, transaction.transaction.meta.unwrap())
}

/// The first signatures of the transactions of the block at `slot`, in
/// block order, and its parent slot.
fn block_signatures(rpc: &RpcClient, slot: Slot) -> (Vec<String>, Slot) {
    let block = rpc
        .get_block_with_config(
            slot,
            RpcBlockConfig {
                encoding: None,
                transaction_details: Some(TransactionDetails::Signatures),
                rewards: Some(false),
                commitment: Some(CommitmentConfig::finalized()),
                max_supported_transaction_version: Some(0),
            },
        )
        .unwrap();
    (block.signatures.unwrap(), block.parent_slot)
}

fn log_messages(meta: &UiTransactionStatusMeta) -> Vec<String> {
    match &meta.log_messages {
        OptionSerializer::Some(logs) => logs.clone(),
        _ => vec![],
    }
}

fn meta(pubkey: &Pubkey, is_signer: bool, is_writable: bool) -> InstructionAccountMeta {
    InstructionAccountMeta {
        pubkey: pubkey.to_bytes(),
        is_signer,
        is_writable,
    }
}

/// Fetches the proof of `sent` (with all its instructions and events),
/// verifies the block it names and the receipt, and checks the receipt
/// against `getSignatureStatuses`, `getTransaction` and `getBlock`.
fn verify_transaction(
    verifier: &mut Verifier,
    rpc: &RpcClient,
    sent: &SentTransaction,
) -> (VerifiedFinal, TransactionResponse, UiTransactionStatusMeta) {
    let signature = sent.signature;
    // A node that has not rooted the block yet does not know the
    // transaction (`NotFound`, not `BlockNotFinal`).
    wait_for_finalized(rpc, sent.slot);
    let response = when_final(format!("getTransactionProof({signature})"), || {
        rpc.get_transaction_proof(
            &signature,
            RpcTransactionProofConfig {
                instructions: Some(RpcProofSelection::All),
                events: Some(RpcProofSelection::All),
            },
        )
    });
    let response = TransactionResponse::try_from(&response).unwrap();
    assert_eq!(response.slot, sent.slot);
    let block = verifier.verify_block(rpc, response.slot);
    assert_eq!(*block.block_id(), response.block_id);
    let signature_bytes: [u8; 64] = signature.into();
    let verified = response
        .transaction
        .verify(&block, &signature_bytes)
        .unwrap();
    // Status and fee, against the node's transaction status.
    let (slot, status_meta) = transaction_meta(rpc, &signature);
    assert_eq!(slot, sent.slot);
    assert_eq!(verified.is_success(), sent.err.is_none());
    assert_eq!(status_meta.err.is_none(), sent.err.is_none());
    assert_eq!(
        verified.receipt.status,
        if sent.err.is_none() {
            ReceiptStatus::Success
        } else {
            ReceiptStatus::Failed
        }
    );
    assert_eq!(verified.fee(), status_meta.fee);
    // Position, against the block's transactions.
    let (signatures, _) = block_signatures(rpc, sent.slot);
    let position = signatures
        .iter()
        .position(|s| *s == signature.to_string())
        .unwrap();
    assert_eq!(verified.index, position as u64);
    assert_eq!(
        response.transaction.proof.leaf_count,
        signatures.len() as u64
    );
    (block, response, status_meta)
}

/// Verifies the account delta proof of `address` in the verified block
/// `block`: changed, with the state whose hash is `expected`, or unchanged.
fn verify_delta(
    rpc: &RpcClient,
    block: &VerifiedFinal,
    address: &Pubkey,
    expected: Option<&Hash>,
) -> AccountDeltaResponse {
    let slot = block.slot();
    let response = when_final(format!("getAccountDeltaProof({address}, {slot})"), || {
        rpc.get_account_delta_proof(address, slot)
    });
    let response = AccountDeltaResponse::try_from(&response).unwrap();
    assert_eq!(
        (response.slot, response.block_id),
        (slot, *block.block_id())
    );
    match (&response.delta, expected) {
        (AccountDelta::Changed(proof), Some(hash)) => {
            proof.verify(block, address, hash).unwrap();
            assert_eq!(proof.leaf.entry.account_hash, *hash);
        }
        (AccountDelta::Unchanged(proof), None) => proof.verify(block, address).unwrap(),
        (delta, expected) => panic!(
            "account {address} in block {slot}: expected {}, got {delta:?}",
            if expected.is_some() {
                "changed"
            } else {
                "unchanged"
            }
        ),
    }
    response
}

/// How often `getAccountStateProof` answered `BaseTreeNotReady`.
static BASE_TREE_NOT_READY: AtomicUsize = AtomicUsize::new(0);

/// `getAccountStateProof(address, slot)`, waiting while the base tree is
/// being computed.
fn account_state(rpc: &RpcClient, address: &Pubkey, slot: Slot) -> AccountStateResponse {
    let response = wait_for(
        format!("an account state proof of {address} at {slot}"),
        TIMEOUT,
        || match rpc
            .get_account_state_proof(address, RpcAccountStateProofConfig { slot: Some(slot) })
        {
            Ok(response) => Some(response),
            Err(err) => match rpc_error(&err) {
                Some((JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_FINAL, _)) => None,
                Some((JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BASE_TREE_NOT_READY, _)) => {
                    BASE_TREE_NOT_READY.fetch_add(1, Ordering::Relaxed);
                    None
                }
                _ => panic!("getAccountStateProof({address}, {slot}): {err}"),
            },
        },
    );
    let response = AccountStateResponse::try_from(&response).unwrap();
    assert_eq!(response.slot, slot);
    response
}

/// Verifies the state of `address` at the end of the verified block
/// `block` (`None`: no account); returns the generation.
fn verify_state(
    rpc: &RpcClient,
    epoch_schedule: &EpochSchedule,
    block: &VerifiedFinal,
    address: &Pubkey,
    expected: Option<&Hash>,
) -> Epoch {
    let response = account_state(rpc, address, block.slot());
    assert_eq!(response.block_id, *block.block_id());
    let generation = match expected {
        Some(hash) => response.proof.verify(block, address, hash).unwrap(),
        None => response.proof.verify_nonexistent(block, address).unwrap(),
    };
    // The generation follows the switch slot rule of the draft.
    let epoch = epoch_schedule.get_epoch(block.slot());
    let switch_slot = epoch_schedule.get_first_slot_in_epoch(epoch)
        + epoch_schedule.get_slots_in_epoch(epoch) / 2;
    let expected_generation = if block.slot() < switch_slot {
        epoch.checked_sub(1)
    } else {
        Some(epoch)
    };
    assert_eq!(
        Some(generation),
        expected_generation,
        "slot {}",
        block.slot()
    );
    generation
}

/// A rooted slot after `slot` (strictly), waiting for one.
fn next_rooted_slot(rpc: &RpcClient, slot: Slot) -> Slot {
    wait_for(format!("a rooted slot after {slot}"), TIMEOUT, || {
        let finalized = finalized_slot(rpc);
        (finalized > slot)
            .then(|| rooted_slots(rpc, slot + 1, finalized).first().copied())
            .flatten()
    })
}

// ---------------------------------------------------------------------------
// Tampering.

/// Checks that verification fails for a response with one field changed.
#[track_caller]
fn assert_rejected<T: Debug>(what: &str, result: Result<T, VerifyError>) -> VerifyError {
    match result {
        Ok(value) => panic!("tampered {what} verified: {value:?}"),
        Err(err) => {
            info!("tampered {what}: rejected with {err}");
            err
        }
    }
}

fn flip(hash: &mut Hash) {
    let mut bytes = hash.to_bytes();
    bytes[0] ^= 1;
    *hash = Hash::new_from_array(bytes);
}

// ---------------------------------------------------------------------------
// Tests.

/// One validator. Real blocks and certificates, verified from genesis;
/// transactions (a transfer, a program with CPI and events, a failing
/// program) proven with their receipts, instructions and events; account
/// deltas and states of the accounts they wrote; block history within and
/// across an epoch; the epoch boundary; tampered responses and the errors
/// for blocks that are not final.
#[test]
#[serial]
fn test_light_client_proofs_single_node() {
    agave_logger::setup_with_default(RUST_LOG_FILTER);
    let started = Instant::now();
    let tc = start_cluster(&[DEFAULT_NODE_STAKE]);
    let rpc = tc.rpc(0);
    let schedule = tc.epoch_schedule().clone();
    let mut verifier = Verifier::from_genesis(tc.genesis_config());
    info!("cluster started in {:?}", started.elapsed());

    // The signer set of epoch 0 served by the node is the one from genesis.
    let served = ValidatorSetResponse::try_from(&when_final("getValidatorSet(0)", || {
        rpc.get_validator_set(0, RpcCommittingBlockConfig::default())
    }))
    .unwrap();
    assert_eq!(
        served.signer_set.validator_set_root(),
        verifier.client.signer_set().validator_set_root()
    );
    assert!(served.committed_by.is_none());

    // Every rooted block so far verifies, and names its parent.
    let finalized = wait_for_finalized(&rpc, 8);
    let slots = rooted_slots(&rpc, 1, finalized);
    let mut first_blocks = BTreeMap::new();
    for slot in slots.iter().copied() {
        let block = verifier.verify_block(&rpc, slot);
        let VerifiedFinal::Block(verified) = block else {
            panic!("block {slot} has its own certificates on a single node");
        };
        let (_, parent_slot) = block_signatures(&rpc, slot);
        assert_eq!(verified.parent().slot, parent_slot);
        first_blocks.insert(slot, block);
    }
    // Parent block ids chain: the parent named by each footer is the
    // verified previous block.
    for (slot, block) in first_blocks.range(2..) {
        let VerifiedFinal::Block(verified) = block else {
            unreachable!()
        };
        if let Some(parent) = first_blocks.get(&verified.parent().slot) {
            assert_eq!(
                verified.parent().block_id,
                *parent.block_id(),
                "slot {slot}"
            );
        }
    }

    // Before the switch slot of epoch 0 (slot 32) generation -1 would be
    // used, which does not exist: slot 5 is zero.
    let early = *slots.last().unwrap();
    assert!(early < 32);
    let message = expect_proof_error(
        rpc.get_account_state_proof(
            &tc.cluster.funding_keypair.pubkey(),
            RpcAccountStateProofConfig { slot: Some(early) },
        ),
        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_DATA_NOT_AVAILABLE,
    );
    info!("account state before the first switch slot: {message}");
    // Epoch 0 is the genesis epoch: no epoch boundary.
    let message = expect_proof_error(
        rpc.get_epoch_boundary(0, RpcCommittingBlockConfig::default()),
        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_DATA_NOT_AVAILABLE,
    );
    info!("epoch boundary of epoch 0: {message}");

    // Not final yet: a block the node has processed but not rooted (when
    // one is seen; finalization is fast), and a slot far ahead of the root.
    let mut processed_not_final = 0;
    for _ in 0..50 {
        let processed = rpc
            .get_slot_with_commitment(CommitmentConfig::processed())
            .unwrap();
        if processed > finalized_slot(&rpc) {
            match rpc.get_finality_proof(processed) {
                Err(err) => {
                    assert_eq!(
                        rpc_error(&err).map(|(code, _)| code),
                        Some(JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_FINAL),
                        "{err}"
                    );
                    processed_not_final += 1;
                }
                // Rooted in between.
                Ok(response) => {
                    verifier.verify_response(
                        &FinalityResponse::try_from(&response).unwrap(),
                        processed,
                    );
                }
            }
        }
        sleep(Duration::from_millis(20));
    }
    info!("processed but not finalized blocks refused as not final: {processed_not_final}");
    // Not final yet: far ahead of the root.
    let finalized = finalized_slot(&rpc);
    let ahead = finalized + 1000;
    let message = expect_proof_error(
        rpc.get_finality_proof(ahead),
        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_FINAL,
    );
    assert!(
        message.contains(&format!("block {ahead} is not final")),
        "{message}"
    );
    expect_proof_error(
        rpc.get_account_delta_proof(&tc.program, ahead),
        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_FINAL,
    );
    expect_proof_error(
        rpc.get_block_history_proof(1, ahead),
        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_FINAL,
    );
    expect_proof_error(
        rpc.get_transaction_proof(
            &Signature::from([7u8; 64]),
            RpcTransactionProofConfig::default(),
        ),
        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_NOT_FOUND,
    );

    // Wait for generation 0 to be in use, so that every account written
    // from now on has a state proof in the block that wrote it.
    wait_for_finalized(&rpc, 34);

    // A: the mint funds a payer only this test writes.
    let mint = &tc.cluster.funding_keypair;
    let payer = Keypair::new();
    let recipient = Keypair::new().pubkey();
    let untouched = Keypair::new().pubkey();
    let blockhash = rpc.get_latest_blockhash().unwrap();
    let a = tc.send(
        &rpc,
        &system_transaction::transfer(mint, &payer.pubkey(), 10_000_000_000, blockhash),
    );
    assert_eq!(a.err, None);
    let (block_a, _, _) = verify_transaction(&mut verifier, &rpc, &a);
    let (read_at, payer_account) = finalized_account(&rpc, &payer.pubkey());
    assert!(read_at >= a.slot);
    let payer_hash = hash_of(&payer.pubkey(), &payer_account.unwrap());
    verify_delta(&rpc, &block_a, &payer.pubkey(), Some(&payer_hash));
    verify_delta(&rpc, &block_a, &untouched, None);
    verify_state(
        &rpc,
        &schedule,
        &block_a,
        &payer.pubkey(),
        Some(&payer_hash),
    );
    verify_state(&rpc, &schedule, &block_a, &untouched, None);

    // B: a transfer to a new account.
    let blockhash = rpc.get_latest_blockhash().unwrap();
    let b = tc.send(
        &rpc,
        &system_transaction::transfer(&payer, &recipient, 1_000_000, blockhash),
    );
    assert_eq!(b.err, None);
    let (block_b, proof_b, _) = verify_transaction(&mut verifier, &rpc, &b);
    // One instruction (the transfer), no events.
    assert_eq!(proof_b.instructions.len(), 1);
    let transfer = proof_b.instructions[0]
        .verify(&block_b, &b.signature.into())
        .unwrap();
    let mut transfer_data = vec![2, 0, 0, 0];
    transfer_data.extend_from_slice(&1_000_000u64.to_le_bytes());
    assert!(transfer.matches(
        1,
        &system_program::id(),
        &[
            meta(&payer.pubkey(), true, true),
            meta(&recipient, false, true)
        ],
        &transfer_data
    ));
    assert!(proof_b.events.is_empty());
    let (_, payer_account) = finalized_account(&rpc, &payer.pubkey());
    let payer_hash = hash_of(&payer.pubkey(), &payer_account.unwrap());
    let (_, recipient_account) = finalized_account(&rpc, &recipient);
    let recipient_hash = hash_of(&recipient, &recipient_account.unwrap());
    verify_delta(&rpc, &block_b, &payer.pubkey(), Some(&payer_hash));
    let delta_b = verify_delta(&rpc, &block_b, &recipient, Some(&recipient_hash));
    verify_state(&rpc, &schedule, &block_b, &recipient, Some(&recipient_hash));
    // A later block that did not write the recipient: unchanged there, and
    // the same state.
    let later = next_rooted_slot(&rpc, b.slot);
    let block_later = verifier.verify_block(&rpc, later);
    verify_delta(&rpc, &block_later, &recipient, None);
    verify_state(
        &rpc,
        &schedule,
        &block_later,
        &recipient,
        Some(&recipient_hash),
    );
    // The recipient did not exist at the end of block A.
    verify_state(&rpc, &schedule, &block_a, &recipient, None);

    // C: the program, with a CPI into itself and from there into the system
    // program, and three events.
    let blockhash = rpc.get_latest_blockhash().unwrap();
    let program_data = [0u8, 0xaa, 0xbb];
    let c = tc.send(
        &rpc,
        &Transaction::new_signed_with_payer(
            &[Instruction::new_with_bytes(
                tc.program,
                &program_data,
                vec![
                    AccountMeta::new(payer.pubkey(), true),
                    AccountMeta::new(recipient, false),
                    AccountMeta::new_readonly(system_program::id(), false),
                    AccountMeta::new_readonly(tc.program, false),
                ],
            )],
            Some(&payer.pubkey()),
            &[&payer],
            blockhash,
        ),
    );
    assert_eq!(c.err, None);
    let (block_c, proof_c, meta_c) = verify_transaction(&mut verifier, &rpc, &c);
    let signature_c: [u8; 64] = c.signature.into();
    let instructions: Vec<_> = proof_c
        .instructions
        .iter()
        .map(|instruction| instruction.verify(&block_c, &signature_c).unwrap())
        .collect();
    assert_eq!(
        instructions.iter().map(|i| i.index).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    let program_accounts = [
        meta(&payer.pubkey(), true, true),
        meta(&recipient, false, true),
        meta(&system_program::id(), false, false),
        meta(&tc.program, false, false),
    ];
    assert!(
        instructions[0].matches(1, &tc.program, &program_accounts, &program_data),
        "{:?}",
        instructions[0]
    );
    assert!(
        instructions[1].matches(2, &tc.program, &program_accounts, &[1]),
        "{:?}",
        instructions[1]
    );
    let mut one_lamport = vec![2, 0, 0, 0];
    one_lamport.extend_from_slice(&1u64.to_le_bytes());
    assert!(
        instructions[2].matches(
            3,
            &system_program::id(),
            &[
                meta(&payer.pubkey(), true, true),
                meta(&recipient, false, true)
            ],
            &one_lamport
        ),
        "{:?}",
        instructions[2]
    );
    let events: Vec<_> = proof_c
        .events
        .iter()
        .map(|event| event.verify(&block_c, &signature_c).unwrap())
        .collect();
    let expected_events = [
        (0, EventKind::Log, b"outer".to_vec()),
        (
            1,
            EventKind::LogData,
            log_data_payload(&[b"inner".as_slice(), b""]),
        ),
        (
            0,
            EventKind::LogData,
            log_data_payload(&[b"after".as_slice(), &[0xaa, 0xbb]]),
        ),
    ];
    assert_eq!(events.len(), expected_events.len());
    for (index, (event, (instruction, kind, payload))) in
        events.iter().zip(expected_events).enumerate()
    {
        assert_eq!(event.index, index as u64);
        assert_eq!(event.instruction.index, instruction);
        assert_eq!(event.event.kind, kind);
        assert_eq!(event.event.payload, payload);
        assert_eq!(event.event.program_id, tc.program.to_bytes());
    }
    // The events are the program's log lines.
    let logs = log_messages(&meta_c);
    assert!(logs.contains(&"Program log: outer".to_string()), "{logs:?}");
    assert_eq!(
        logs.iter()
            .filter(|line| line.starts_with("Program data: "))
            .count(),
        2,
        "{logs:?}"
    );
    // The recipient got one lamport through the CPI.
    let (_, recipient_account) = finalized_account(&rpc, &recipient);
    let recipient_account = recipient_account.unwrap();
    assert_eq!(recipient_account.lamports, 1_000_001);
    let recipient_hash_c = hash_of(&recipient, &recipient_account);
    verify_delta(&rpc, &block_c, &recipient, Some(&recipient_hash_c));
    verify_state(
        &rpc,
        &schedule,
        &block_c,
        &recipient,
        Some(&recipient_hash_c),
    );
    // The earlier state does not verify at C.
    let state_c = account_state(&rpc, &recipient, c.slot);
    assert_rejected(
        "account state (old hash)",
        state_c.proof.verify(&block_c, &recipient, &recipient_hash),
    );

    // D: the program fails. Charged a fee, no trace or events in the
    // receipt.
    let blockhash = rpc.get_latest_blockhash().unwrap();
    let d = tc.send(
        &rpc,
        &Transaction::new_signed_with_payer(
            &[Instruction::new_with_bytes(
                tc.program,
                &[2],
                vec![
                    AccountMeta::new(payer.pubkey(), true),
                    AccountMeta::new(recipient, false),
                    AccountMeta::new_readonly(system_program::id(), false),
                    AccountMeta::new_readonly(tc.program, false),
                ],
            )],
            Some(&payer.pubkey()),
            &[&payer],
            blockhash,
        ),
    );
    assert!(d.err.is_some());
    let (block_d, proof_d, meta_d) = verify_transaction(&mut verifier, &rpc, &d);
    assert!(meta_d.fee > 0);
    assert!(
        proof_d.instructions.is_empty(),
        "{:?}",
        proof_d.instructions
    );
    assert!(proof_d.events.is_empty(), "{:?}", proof_d.events);
    // The payer paid the fee; the recipient is unchanged.
    let (_, payer_account) = finalized_account(&rpc, &payer.pubkey());
    let payer_hash_d = hash_of(&payer.pubkey(), &payer_account.unwrap());
    verify_delta(&rpc, &block_d, &payer.pubkey(), Some(&payer_hash_d));
    verify_delta(&rpc, &block_d, &recipient, None);

    // E: an account emptied by a transfer of all its lamports, so deleted:
    // its delta leaf has the zero hash, and the accounts state shows that it
    // does not exist.
    let doomed = Keypair::new();
    let sink = Keypair::new().pubkey();
    let blockhash = rpc.get_latest_blockhash().unwrap();
    let fund = tc.send(
        &rpc,
        &system_transaction::transfer(&payer, &doomed.pubkey(), 1_000_000_000, blockhash),
    );
    assert_eq!(fund.err, None);
    let blockhash = rpc.get_latest_blockhash().unwrap();
    let drain = tc.send(
        &rpc,
        &Transaction::new_signed_with_payer(
            &[solana_system_interface::instruction::transfer(
                &doomed.pubkey(),
                &sink,
                1_000_000_000,
            )],
            Some(&payer.pubkey()),
            &[&payer, &doomed],
            blockhash,
        ),
    );
    assert_eq!(drain.err, None);
    let (block_drain, _, _) = verify_transaction(&mut verifier, &rpc, &drain);
    assert_eq!(finalized_account(&rpc, &doomed.pubkey()).1, None);
    let response = AccountDeltaResponse::try_from(
        &rpc.get_account_delta_proof(&doomed.pubkey(), drain.slot)
            .unwrap(),
    )
    .unwrap();
    let AccountDelta::Changed(deleted) = &response.delta else {
        panic!("the drain changed the account: {:?}", response.delta);
    };
    deleted
        .verify_deleted(&block_drain, &doomed.pubkey())
        .unwrap();
    assert_rejected(
        "deleted account claimed to exist",
        deleted.verify(&block_drain, &doomed.pubkey(), &payer_hash_d),
    );
    verify_state(&rpc, &schedule, &block_drain, &doomed.pubkey(), None);

    // A sysvar the runtime writes in every block, outside any transaction:
    // the clock, read at the root, is in the delta and the state of that
    // block.
    let clock_id = solana_sdk_ids::sysvar::clock::id();
    let (clock_slot, clock) = finalized_account(&rpc, &clock_id);
    let clock_hash = hash_of(&clock_id, &clock.unwrap());
    let block_clock = verifier.verify_block(&rpc, clock_slot);
    verify_delta(&rpc, &block_clock, &clock_id, Some(&clock_hash));
    verify_state(&rpc, &schedule, &block_clock, &clock_id, Some(&clock_hash));

    // The program account, unchanged since genesis. Generation 0 starts at
    // the genesis block, which has no parent: its base state is empty (the
    // draft: "If `F` is the genesis block there is no `P`, and the base
    // state is empty"), and the accounts created by genesis are changes of
    // the genesis block, in the overlay. An address without an account is
    // in neither.
    let (_, program_account) = finalized_account(&rpc, &tc.program);
    let program_hash = hash_of(&tc.program, &program_account.unwrap());
    verify_state(&rpc, &schedule, &block_d, &tc.program, Some(&program_hash));
    assert!(matches!(
        account_state(&rpc, &tc.program, d.slot).proof.0.lookup,
        AccountLookup::Overlay { .. }
    ));
    assert!(matches!(
        account_state(&rpc, &untouched, d.slot).proof.0.lookup,
        AccountLookup::Base {
            base: BaseLookup::Absent(BaseAbsenceProof::EmptyTree),
            ..
        }
    ));
    verify_delta(&rpc, &block_d, &tc.program, None);

    // Block history: A, B and C as ancestors of D.
    for (slot, block) in [(a.slot, &block_a), (b.slot, &block_b), (c.slot, &block_c)] {
        let response = BlockHistoryResponse::try_from(&when_final(
            format!("getBlockHistoryProof({slot}, {})", d.slot),
            || rpc.get_block_history_proof(slot, d.slot),
        ))
        .unwrap();
        assert_eq!(
            (response.at_slot, response.at_block_id),
            (d.slot, *block_d.block_id())
        );
        let ancestor = verifier
            .client
            .verify_ancestor(&block_d, &response.history)
            .unwrap();
        assert_eq!(ancestor.slot(), slot);
        assert_eq!(ancestor.block_id(), block.block_id());
        assert_eq!(ancestor.commitment_root(), block.commitment_root());
    }

    // Tampered responses.
    let finality_d = verifier.finality(&rpc, d.slot);
    let Finality::Certificates(certificates) = &finality_d.finality else {
        panic!("block {} has its own certificates on a single node", d.slot);
    };
    let tamper_certificate = |f: &dyn Fn(&mut solana_light_client::Certificate)| {
        let mut tampered = finality_d.clone();
        let Finality::Certificates(proof) = &mut tampered.finality else {
            unreachable!()
        };
        f(&mut proof.certificates[0]);
        tampered.verify(&verifier.client, d.slot)
    };
    // The bitmap of the only signer, cleared; and a signer that does not
    // exist.
    assert!(!certificates.certificates[0].bitmap.is_empty());
    assert_rejected(
        "certificate bitmap (no signer)",
        tamper_certificate(&|certificate| certificate.bitmap.iter_mut().for_each(|b| *b = 0)),
    );
    assert_rejected(
        "certificate bitmap (unknown signer)",
        tamper_certificate(&|certificate| certificate.bitmap[0] ^= 0b10),
    );
    assert_rejected(
        "certificate signature",
        tamper_certificate(&|certificate| certificate.signature.0[10] ^= 1),
    );
    let mut tampered = finality_d.clone();
    tampered.footer.proof[0][0] ^= 1;
    assert_eq!(
        assert_rejected(
            "footer proof node",
            tampered.verify(&verifier.client, d.slot)
        ),
        VerifyError::FooterProofMismatch
    );
    let mut tampered = finality_d.clone();
    flip(&mut tampered.footer.commitment_root);
    assert_rejected(
        "footer commitment root",
        tampered.verify(&verifier.client, d.slot),
    );
    let mut tampered = finality_d.clone();
    tampered.footer.parent_slot -= 1;
    assert_rejected(
        "footer parent slot",
        tampered.verify(&verifier.client, d.slot),
    );
    // The certificates of D do not finalize C.
    let mut tampered = finality_d.clone();
    tampered.slot = c.slot;
    assert_rejected("finality slot", tampered.verify(&verifier.client, c.slot));
    // An account hash in the delta proof, and a claimed hash.
    let AccountDelta::Changed(changed) = &delta_b.delta else {
        unreachable!()
    };
    let mut tampered = changed.clone();
    flip(&mut tampered.leaf.entry.account_hash);
    assert_rejected(
        "delta leaf account hash",
        tampered.verify(&block_b, &recipient, &recipient_hash),
    );
    let mut wrong = recipient_hash;
    flip(&mut wrong);
    assert_rejected(
        "claimed account hash",
        changed.verify(&block_b, &recipient, &wrong),
    );
    let mut tampered = changed.clone();
    if let Some(sibling) = tampered.leaf.proof.siblings.first_mut() {
        flip(sibling);
        assert_rejected(
            "delta proof node",
            tampered.verify(&block_b, &recipient, &recipient_hash),
        );
    }
    // The receipt: its status, its fee, a proof node.
    let mut tampered = proof_d.transaction.clone();
    tampered.receipt.status = ReceiptStatus::Success;
    assert_rejected(
        "receipt status",
        tampered.verify(&block_d, &d.signature.into()),
    );
    let mut tampered = proof_d.transaction.clone();
    tampered.receipt.fee -= 1;
    assert_rejected(
        "receipt fee",
        tampered.verify(&block_d, &d.signature.into()),
    );
    let mut tampered = proof_c.events[1].clone();
    tampered.event.payload.push(0);
    assert_rejected("event payload", tampered.verify(&block_c, &signature_c));
    // The receipt of C is not in D.
    assert_rejected(
        "receipt of another block",
        proof_c.transaction.verify(&block_d, &signature_c),
    );
    // A history proof node.
    let mut history =
        BlockHistoryResponse::try_from(&rpc.get_block_history_proof(a.slot, d.slot).unwrap())
            .unwrap()
            .history;
    if let Some(sibling) = history.proof.siblings.first_mut() {
        flip(sibling);
        assert_rejected(
            "history proof node",
            verifier.client.verify_ancestor(&block_d, &history),
        );
    }
    // An account state proof with a tampered overlay root.
    let mut tampered = state_c.clone();
    flip(&mut tampered.proof.0.opening.overlay_root);
    assert_rejected(
        "accounts state overlay root",
        tampered
            .proof
            .verify(&block_c, &recipient, &recipient_hash_c),
    );

    // Into epoch 1: the client follows the validator set, verifies a block
    // of epoch 1, proves A as its ancestor across the boundary, and checks
    // the boundary against the rooted chain.
    let first_slot_1 = schedule.get_first_slot_in_epoch(1);
    wait_for_finalized(&rpc, first_slot_1 + 4);
    if verifier.client.epoch() == 0 {
        verifier.advance_epoch(&rpc);
    }
    let boundary_block = verifier.committing[&1];
    assert_eq!(schedule.get_epoch(boundary_block.slot()), 0);
    let in_epoch_1 = next_rooted_slot(&rpc, first_slot_1 + 1);
    let block_e1 = verifier.verify_block(&rpc, in_epoch_1);
    let response = BlockHistoryResponse::try_from(&when_final(
        format!("getBlockHistoryProof({}, {in_epoch_1})", a.slot),
        || rpc.get_block_history_proof(a.slot, in_epoch_1),
    ))
    .unwrap();
    let ancestor = verifier
        .client
        .verify_ancestor(&block_e1, &response.history)
        .unwrap();
    assert_eq!(ancestor.commitment_root(), block_a.commitment_root());
    let response = EpochBoundaryResponse::try_from(&when_final("getEpochBoundary(1)", || {
        rpc.get_epoch_boundary(
            1,
            RpcCommittingBlockConfig {
                at_slot: Some(in_epoch_1),
            },
        )
    }))
    .unwrap();
    assert_eq!(
        (response.slot, response.block_id),
        (in_epoch_1, *block_e1.block_id())
    );
    let boundary = response.proof.verify(&block_e1, &schedule).unwrap();
    let rooted = rooted_slots(&rpc, 0, in_epoch_1);
    let last_of_epoch_0 = *rooted
        .iter()
        .filter(|slot| **slot < first_slot_1)
        .last()
        .unwrap();
    let first_of_epoch_1 = *rooted.iter().find(|slot| **slot >= first_slot_1).unwrap();
    assert_eq!(boundary.boundary.parent_slot, last_of_epoch_0);
    assert_eq!(boundary.boundary.first_slot, first_of_epoch_1);
    // P, the last block of epoch 0, is the block the boundary names: the
    // same as its history leaf in the block of epoch 1.
    let last_before = boundary.last_block_before().unwrap();
    let response = BlockHistoryResponse::try_from(&when_final(
        format!("getBlockHistoryProof({last_of_epoch_0}, {in_epoch_1})"),
        || rpc.get_block_history_proof(last_of_epoch_0, in_epoch_1),
    ))
    .unwrap();
    let block_p = verifier
        .client
        .verify_ancestor(&block_e1, &response.history)
        .unwrap();
    assert_eq!(last_before, block_p);
    if boundary_block.slot() == last_of_epoch_0 {
        assert_eq!(boundary_block.commitment_root(), block_p.commitment_root());
    }
    // A block of epoch 1 is verified only with the advanced signer set.
    let genesis_client = Verifier::from_genesis(tc.genesis_config());
    assert!(matches!(
        verifier
            .finality(&rpc, in_epoch_1)
            .verify(&genesis_client.client, in_epoch_1),
        Err(VerifyError::EpochMismatch { .. })
    ));
    // The recipient, still unchanged in epoch 1 (generation 0 until the
    // switch slot 96).
    verify_state(
        &rpc,
        &schedule,
        &block_e1,
        &recipient,
        Some(&recipient_hash_c),
    );

    // From the switch slot of epoch 1 on, generation 1: the recipient,
    // changed in epoch 0, is now in the base tree of generation 1 (built
    // from generation 0) and not in its overlay; the program is too; the
    // untouched address is in neither.
    let switch_slot_1 = first_slot_1 + SLOTS_PER_EPOCH / 2;
    wait_for_finalized(&rpc, switch_slot_1);
    let after_switch = next_rooted_slot(&rpc, switch_slot_1 - 1);
    let block_switch = verifier.verify_block(&rpc, after_switch);
    for (address, hash) in [
        (recipient, Some(&recipient_hash_c)),
        (tc.program, Some(&program_hash)),
        (untouched, None),
    ] {
        assert_eq!(
            verify_state(&rpc, &schedule, &block_switch, &address, hash),
            1
        );
        let lookup = account_state(&rpc, &address, after_switch).proof.0.lookup;
        match hash {
            Some(_) => assert!(
                matches!(
                    lookup,
                    AccountLookup::Base {
                        base: BaseLookup::Present(_),
                        ..
                    }
                ),
                "{address}: {lookup:?}"
            ),
            None => assert!(
                matches!(
                    lookup,
                    AccountLookup::Base {
                        base: BaseLookup::Absent(ref absence),
                        ..
                    } if !matches!(absence, BaseAbsenceProof::EmptyTree)
                ),
                "{address}: {lookup:?}"
            ),
        }
    }
    // The default block of getAccountStateProof: the latest finalized.
    let response = AccountStateResponse::try_from(
        &rpc.get_account_state_proof(&recipient, RpcAccountStateProofConfig::default())
            .unwrap(),
    )
    .unwrap();
    assert!(response.slot >= after_switch);
    let block_latest = verifier.verify_block(&rpc, response.slot);
    assert_eq!(response.block_id, *block_latest.block_id());
    response
        .proof
        .verify(&block_latest, &recipient, &recipient_hash_c)
        .unwrap();

    info!("finality kinds: {:?}", verifier.kinds);
    info!(
        "transient getFinalityProof errors (code: count): {:?}; BaseTreeNotReady: {}; final \
         through a descendant (slot: descendant): {:?}",
        verifier.transient,
        BASE_TREE_NOT_READY.load(Ordering::Relaxed),
        verifier.through_descendant
    );
    info!("test took {:?}", started.elapsed());
}

/// Four validators with unequal stakes. The light client follows the
/// validator set across two epoch boundaries with `getValidatorSet` and
/// `advance_epoch`, verifies blocks of epoch 2, proves a block of epoch 0
/// as their ancestor and checks both epoch boundaries against the rooted
/// chain. Transaction proofs are served the same by the leader and by the
/// nodes that replayed the block. With a quarter of the stake offline,
/// blocks are finalized by notarization and finalization certificates and
/// the offline leader's slots are skipped: proofs of a skipped slot are
/// refused as not final.
#[test]
#[serial]
fn test_light_client_epoch_handoff_four_nodes() {
    agave_logger::setup_with_default(RUST_LOG_FILTER);
    let started = Instant::now();
    // 40%, 25%, 20% and 15% of the stake.
    let stakes: Vec<u64> = [8, 5, 4, 3]
        .into_iter()
        .map(|units| units * DEFAULT_NODE_STAKE)
        .collect();
    let mut tc = start_cluster(&stakes);
    let schedule = tc.epoch_schedule().clone();
    // The light client asks a node that is not the bootstrap validator.
    let rpc = tc.rpc(2);
    let rpcs: Vec<_> = (0..stakes.len()).map(|node| tc.rpc(node)).collect();
    let mut verifier = Verifier::from_genesis(tc.genesis_config());
    info!("cluster started in {:?}", started.elapsed());

    // The genesis signer set: four signers, by stake.
    let stakes_of = |signer_set: &SignerSet| {
        signer_set
            .signers()
            .iter()
            .map(|signer| signer.stake)
            .collect::<Vec<_>>()
    };
    assert_eq!(stakes_of(verifier.client.signer_set()), stakes);
    let vote_account_of_rank = |signer_set: &SignerSet, rank: usize| {
        Pubkey::new_from_array(signer_set.signers()[rank].vote_pubkey)
    };
    assert_eq!(
        vote_account_of_rank(verifier.client.signer_set(), 3),
        tc.node_keys[3].vote_keypair.pubkey()
    );

    // More stake for node 3 (30 + 15 = 45 units, more than node 2's 40):
    // it activates in the next epoch, and is in the signer set two epochs
    // after the delegation, where node 3 ranks before node 2. Certificates
    // of that epoch are signed with the new ranks.
    let mint = &tc.cluster.funding_keypair;
    let stake_keypair = Keypair::new();
    let extra_stake = DEFAULT_NODE_STAKE * 3 / 2;
    // The stake program refuses to delegate while the rewards of an epoch
    // are being distributed, in the first slots of the epoch (stake error
    // 16): the whole transaction fails, and is sent again later.
    let mut attempts = 0;
    let delegation = loop {
        let blockhash = rpc.get_latest_blockhash().unwrap();
        let delegation =
            tc.send(
                &rpc,
                &Transaction::new_signed_with_payer(
                    &solana_stake_interface::instruction::create_account_and_delegate_stake(
                        &mint.pubkey(),
                        &stake_keypair.pubkey(),
                        &tc.node_keys[3].vote_keypair.pubkey(),
                        &solana_stake_interface::state::Authorized::auto(&mint.pubkey()),
                        &solana_stake_interface::state::Lockup::default(),
                        extra_stake
                            + Rent::default().minimum_balance(
                                solana_stake_interface::state::StakeStateV2::size_of(),
                            ),
                    ),
                    Some(&mint.pubkey()),
                    &[mint, &stake_keypair],
                    blockhash,
                ),
            );
        let rewards_active = matches!(
            &delegation.err,
            Some(TransactionError::InstructionError(1, err)) if format!("{err:?}") == "Custom(16)"
        );
        if rewards_active && attempts < 20 {
            attempts += 1;
            info!("delegation refused during reward distribution, sending again");
            sleep(Duration::from_millis(500));
            continue;
        }
        break delegation;
    };
    assert_eq!(delegation.err, None);
    let new_stakes_epoch = schedule.get_epoch(delegation.slot) + 2;
    info!(
        "stake delegated in slot {}: new signer set from epoch {new_stakes_epoch}",
        delegation.slot
    );

    // Blocks of epoch 0 verify, from every node.
    let finalized = wait_for_finalized(&rpc, 8);
    let mut epoch_0_blocks = BTreeMap::new();
    for slot in rooted_slots(&rpc, 1, finalized.min(SLOTS_PER_EPOCH - 1)) {
        let block = verifier.verify_block(&rpc, slot);
        epoch_0_blocks.insert(slot, block);
        for other in &rpcs {
            let response = verifier.finality(other, slot);
            assert_eq!(response.block_id, *block.block_id());
            verifier.verify_response(&response, slot);
        }
    }

    // A transfer in epoch 0, proven by every node: one produced the block,
    // the others replayed it.
    let recipient = Keypair::new().pubkey();
    let blockhash = rpc.get_latest_blockhash().unwrap();
    let t0 = tc.send(
        &rpc,
        &system_transaction::transfer(mint, &recipient, 1_000_000, blockhash),
    );
    let (block_t0, proof_t0, _) = verify_transaction(&mut verifier, &rpc, &t0);
    for (node, other) in rpcs.iter().enumerate() {
        wait_for_finalized(other, t0.slot);
        let response = TransactionResponse::try_from(
            &other
                .get_transaction_proof(&t0.signature, RpcTransactionProofConfig::default())
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            response.transaction, proof_t0.transaction,
            "node {node} serves another receipt proof"
        );
        response
            .transaction
            .verify(&block_t0, &t0.signature.into())
            .unwrap();
    }
    let (_, account) = finalized_account(&rpc, &recipient);
    let recipient_hash = hash_of(&recipient, &account.unwrap());
    verify_delta(&rpc, &block_t0, &recipient, Some(&recipient_hash));
    info!(
        "block of the transfer: {}, leader {}",
        t0.slot,
        rpc.get_slot_leaders(t0.slot, 1).unwrap()[0]
    );

    // A burst of transfers from eight payers, some failing: several
    // transactions per block, executed by the leader's banking threads in
    // parallel. Each receipt is at the transaction's position in the block,
    // and every node serves the same receipt proofs.
    let payers: Vec<_> = (0..8).map(|_| Keypair::new()).collect();
    let blockhash = rpc.get_latest_blockhash().unwrap();
    let funding = Transaction::new_signed_with_payer(
        &payers
            .iter()
            .map(|payer| {
                solana_system_interface::instruction::transfer(
                    &mint.pubkey(),
                    &payer.pubkey(),
                    1_000_000_000,
                )
            })
            .collect::<Vec<_>>(),
        Some(&mint.pubkey()),
        &[mint],
        blockhash,
    );
    assert_eq!(tc.send(&rpc, &funding).err, None);
    let blockhash = rpc.get_latest_blockhash().unwrap();
    let burst: Vec<_> = payers
        .iter()
        .flat_map(|payer| {
            (0..4u64).map(move |i| {
                // One in four overdraws the payer and fails.
                let lamports = if i == 3 { 2_000_000_000 } else { 1_000_000 + i };
                system_transaction::transfer(payer, &Pubkey::new_unique(), lamports, blockhash)
            })
        })
        .collect();
    let sent = tc.send_many(&rpc, &burst);
    let mut per_block = BTreeMap::<Slot, usize>::new();
    for (i, sent) in sent.iter().enumerate() {
        assert_eq!(sent.err.is_some(), i % 4 == 3, "{:?}", sent.err);
        *per_block.entry(sent.slot).or_default() += 1;
        let (_, proof, _) = verify_transaction(&mut verifier, &rpc, sent);
        for (node, other) in rpcs.iter().enumerate() {
            wait_for_finalized(other, sent.slot);
            let response = TransactionResponse::try_from(
                &other
                    .get_transaction_proof(&sent.signature, RpcTransactionProofConfig::default())
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(
                response.transaction, proof.transaction,
                "node {node} serves another receipt proof"
            );
        }
    }
    info!(
        "burst of {} transactions: per block {per_block:?}",
        sent.len()
    );

    // Epoch 0 -> 1 -> 2, and on to the epoch with the new stakes (the
    // client may have followed already).
    for epoch in 1..=new_stakes_epoch.max(2) {
        wait_for_finalized(&rpc, schedule.get_first_slot_in_epoch(epoch) + 4);
        if verifier.client.epoch() < epoch {
            verifier.advance_epoch(&rpc);
        }
        // On a slow machine the client has followed past this epoch while
        // verifying the blocks above; its set is then checked from the
        // epoch it is in.
        if verifier.client.epoch() > epoch {
            continue;
        }
        assert_eq!(verifier.client.epoch(), epoch);
        let signer_set = verifier.client.signer_set();
        if epoch < new_stakes_epoch {
            assert_eq!(stakes_of(signer_set), stakes, "epoch {epoch}");
        } else {
            // The stake program may delegate the rent reserve as well.
            let new_stakes = stakes_of(signer_set);
            assert_eq!(
                [new_stakes[0], new_stakes[1], new_stakes[3]],
                [stakes[0], stakes[1], stakes[2]],
                "epoch {epoch}"
            );
            let added = new_stakes[2] - stakes[3];
            assert!(
                (extra_stake..=extra_stake + LAMPORTS_PER_SOL).contains(&added),
                "epoch {epoch}: {new_stakes:?}"
            );
            assert_eq!(
                vote_account_of_rank(signer_set, 2),
                tc.node_keys[3].vote_keypair.pubkey()
            );
        }
    }
    // Blocks of the epoch with the new ranks verify (fast finalization: all
    // four signers, with node 3 at rank 2).
    let first_new_ranks = schedule.get_first_slot_in_epoch(new_stakes_epoch);
    let finalized = wait_for_finalized(&rpc, first_new_ranks + 8);
    for slot in rooted_slots(&rpc, first_new_ranks, finalized) {
        verifier.verify_block(&rpc, slot);
    }
    let first_slot_1 = schedule.get_first_slot_in_epoch(1);
    let first_slot_2 = schedule.get_first_slot_in_epoch(2);
    let in_epoch_2 = next_rooted_slot(&rpc, first_slot_2 + 1);
    let block_e2 = verifier.verify_block(&rpc, in_epoch_2);
    // The block of epoch 1 that committed to the signer set of epoch 2 is an
    // ancestor of the block of epoch 2.
    let committing = verifier.committing[&2];
    let response = BlockHistoryResponse::try_from(&when_final(
        format!("getBlockHistoryProof({}, {in_epoch_2})", committing.slot()),
        || rpc.get_block_history_proof(committing.slot(), in_epoch_2),
    ))
    .unwrap();
    let ancestor = verifier
        .client
        .verify_ancestor(&block_e2, &response.history)
        .unwrap();
    assert_eq!(ancestor.commitment_root(), committing.commitment_root());

    // Block history across two boundaries: a block of epoch 0.
    let (slot_e0, block_e0) = epoch_0_blocks.iter().next_back().unwrap();
    let response = BlockHistoryResponse::try_from(&when_final(
        format!("getBlockHistoryProof({slot_e0}, {in_epoch_2})"),
        || rpc.get_block_history_proof(*slot_e0, in_epoch_2),
    ))
    .unwrap();
    let ancestor_e0 = verifier
        .client
        .verify_ancestor(&block_e2, &response.history)
        .unwrap();
    assert_eq!(ancestor_e0.block_id(), block_e0.block_id());
    assert_eq!(ancestor_e0.commitment_root(), block_e0.commitment_root());
    // And the transfer's block.
    let response = BlockHistoryResponse::try_from(&when_final(
        format!("getBlockHistoryProof({}, {in_epoch_2})", t0.slot),
        || rpc.get_block_history_proof(t0.slot, in_epoch_2),
    ))
    .unwrap();
    let ancestor_t0 = verifier
        .client
        .verify_ancestor(&block_e2, &response.history)
        .unwrap();
    assert_eq!(ancestor_t0.block_id(), block_t0.block_id());
    assert_eq!(ancestor_t0.commitment_root(), block_t0.commitment_root());
    // The receipt and the account delta of epoch 0, through the ancestor.
    proof_t0
        .transaction
        .verify(&ancestor_t0, &t0.signature.into())
        .unwrap();
    verify_delta(
        &rpc,
        &VerifiedFinal::Ancestor(ancestor_t0),
        &recipient,
        Some(&recipient_hash),
    );

    // Epoch boundaries 1 and 2: the boundary of epoch 1 from a block of
    // epoch 1 proven as an ancestor, the boundary of epoch 2 from the
    // verified block of epoch 2.
    let rooted = rooted_slots(&rpc, 0, in_epoch_2);
    let in_epoch_1 = *rooted.iter().find(|slot| **slot > first_slot_1).unwrap();
    let response = BlockHistoryResponse::try_from(&when_final(
        format!("getBlockHistoryProof({in_epoch_1}, {in_epoch_2})"),
        || rpc.get_block_history_proof(in_epoch_1, in_epoch_2),
    ))
    .unwrap();
    let block_e1 = verifier
        .client
        .verify_ancestor(&block_e2, &response.history)
        .unwrap();
    let boundary_1 = EpochBoundaryResponse::try_from(&when_final("getEpochBoundary(1)", || {
        rpc.get_epoch_boundary(
            1,
            RpcCommittingBlockConfig {
                at_slot: Some(in_epoch_1),
            },
        )
    }))
    .unwrap()
    .proof
    .verify(&block_e1, &schedule)
    .unwrap();
    let boundary_2 = EpochBoundaryResponse::try_from(&when_final("getEpochBoundary(2)", || {
        rpc.get_epoch_boundary(
            2,
            RpcCommittingBlockConfig {
                at_slot: Some(in_epoch_2),
            },
        )
    }))
    .unwrap()
    .proof
    .verify(&block_e2, &schedule)
    .unwrap();
    // The default block of getEpochBoundary: the latest of the epoch.
    let response = EpochBoundaryResponse::try_from(
        &rpc.get_epoch_boundary(2, RpcCommittingBlockConfig::default())
            .unwrap(),
    )
    .unwrap();
    assert!(response.slot >= in_epoch_2);
    let block = verifier.verify_block(&rpc, response.slot);
    assert_eq!(
        response.proof.verify(&block, &schedule).unwrap(),
        boundary_2
    );
    let last_before = |first_slot| {
        *rooted
            .iter()
            .filter(|slot| **slot < first_slot)
            .last()
            .unwrap()
    };
    let first_from = |first_slot| *rooted.iter().find(|slot| **slot >= first_slot).unwrap();
    assert_eq!(boundary_1.boundary.parent_slot, last_before(first_slot_1));
    assert_eq!(boundary_1.boundary.first_slot, first_from(first_slot_1));
    assert_eq!(boundary_2.boundary.parent_slot, last_before(first_slot_2));
    assert_eq!(boundary_2.boundary.first_slot, first_from(first_slot_2));
    // The number of blocks in epoch 1 is the difference of the history
    // lengths.
    let blocks_in_epoch_1 = rooted
        .iter()
        .filter(|slot| (first_slot_1..first_slot_2).contains(*slot))
        .count() as u64;
    assert_eq!(
        boundary_1.boundary.blocks_until(&boundary_2.boundary),
        Some(blocks_in_epoch_1)
    );
    // P of epoch 2 is the last block of epoch 1, with the commitment root of
    // its history leaf.
    let p2 = boundary_2.last_block_before().unwrap();
    let response = BlockHistoryResponse::try_from(&when_final(
        format!("getBlockHistoryProof({}, {in_epoch_2})", p2.slot()),
        || rpc.get_block_history_proof(p2.slot(), in_epoch_2),
    ))
    .unwrap();
    assert_eq!(
        verifier
            .client
            .verify_ancestor(&block_e2, &response.history)
            .unwrap(),
        p2
    );
    info!(
        "finality kinds with every node online: {:?}",
        verifier.kinds
    );

    // A quarter of the stake goes offline: no fast finalization (80%) is
    // possible, notarization and finalization (60%) are.
    let offline = tc.node_pubkey(1);
    let _offline_info = tc.cluster.exit_node(&offline);
    let after_exit = finalized_slot(&rpc);
    info!("node 1 ({offline}) offline from slot {after_exit}");
    let mut slow_kinds = BTreeMap::<&'static str, usize>::new();
    let mut skipped = vec![];
    let mut last_checked = after_exit;
    let deadline = Instant::now() + TIMEOUT * 2;
    while skipped.is_empty() || slow_kinds.get("notarize+finalize").copied().unwrap_or(0) < 4 {
        assert!(
            Instant::now() < deadline,
            "no skipped slot or no slow finalization after node 1 went offline: {slow_kinds:?}"
        );
        let finalized = wait_for_finalized(&rpc, last_checked + 1);
        let slots = rooted_slots(&rpc, last_checked + 1, finalized);
        let mut expected = last_checked + 1;
        for slot in slots.iter().copied() {
            skipped.extend(expected..slot);
            expected = slot + 1;
            while verifier.client.epoch() < schedule.get_epoch(slot) {
                verifier.advance_epoch(&rpc);
            }
            let response = verifier.finality(&rpc, slot);
            verifier.verify_response(&response, slot);
            *slow_kinds.entry(finality_kind(&response)).or_default() += 1;
        }
        last_checked = finalized;
    }
    info!("finality kinds with node 1 offline: {slow_kinds:?}; skipped slots {skipped:?}");
    for slot in skipped.iter().copied().take(4) {
        let message = expect_proof_error(
            rpc.get_finality_proof(slot),
            JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_FINAL,
        );
        assert!(
            message.contains(&format!("block {slot} is not final")),
            "{message}"
        );
        expect_proof_error(
            rpc.get_account_delta_proof(&recipient, slot),
            JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_FINAL,
        );
        expect_proof_error(
            rpc.get_block_history_proof(slot, last_checked),
            JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_FINAL,
        );
    }
    // The slow finality proof, tampered: a finalization certificate alone,
    // or the bitmap of the notarization with a signer removed.
    let slow_slot = *rooted_slots(&rpc, after_exit + 1, last_checked)
        .iter()
        .rev()
        .find(|slot| {
            schedule.get_epoch(**slot) == verifier.client.epoch()
                && matches!(
                    &verifier.finality(&rpc, **slot).finality,
                    Finality::Certificates(proof) if proof.certificates.len() == 2
                )
        })
        .expect("a block finalized by notarization and finalization");
    let slow = verifier.finality(&rpc, slow_slot);
    let mut tampered = slow.clone();
    let Finality::Certificates(proof) = &mut tampered.finality else {
        unreachable!()
    };
    proof.certificates.remove(0);
    assert_rejected(
        "slow finality without notarization",
        tampered.verify(&verifier.client, slow_slot),
    );
    // The notarization bitmap with a signer removed, and with the offline
    // signer added (claiming its stake): the aggregate signature does not
    // match either. A bitmap with a signer beyond the set is malformed.
    let signers = verifier.client.signer_set().len();
    let offline_rank = verifier
        .client
        .signer_set()
        .signers()
        .iter()
        .position(|signer| signer.vote_pubkey == tc.node_keys[1].vote_keypair.pubkey().to_bytes())
        .expect("the offline validator is in the signer set");
    let tamper_ranks = |f: &dyn Fn(&mut BitVec<u8, Lsb0>)| {
        let mut tampered = slow.clone();
        let Finality::Certificates(proof) = &mut tampered.finality else {
            unreachable!()
        };
        let bitmap = &mut proof.certificates[0].bitmap;
        let Ok(Decoded::Base2(mut ranks)) = solana_signer_store::decode(bitmap, signers) else {
            panic!("a base-2 bitmap");
        };
        ranks.resize(signers, false);
        f(&mut ranks);
        *bitmap = encode_base2(&ranks).unwrap();
        tampered.verify(&verifier.client, slow_slot)
    };
    // The real bitmap, re-encoded, still verifies.
    tamper_ranks(&|_| ()).unwrap();
    assert!(matches!(
        assert_rejected(
            "notarization bitmap (a signer removed)",
            tamper_ranks(&|ranks| {
                let first = ranks.first_one().unwrap();
                ranks.set(first, false);
            })
        ),
        VerifyError::InvalidSignature { .. }
    ));
    assert!(matches!(
        assert_rejected(
            "notarization bitmap (the offline signer added)",
            tamper_ranks(&|ranks| ranks.set(offline_rank, true))
        ),
        VerifyError::InvalidSignature { .. }
    ));
    assert!(matches!(
        assert_rejected(
            "notarization bitmap (a signer beyond the set)",
            tamper_ranks(&|ranks| ranks.push(true))
        ),
        VerifyError::InvalidBitmap { .. }
    ));

    info!("finality kinds: {:?}", verifier.kinds);
    info!(
        "transient getFinalityProof errors (code: count): {:?}; BaseTreeNotReady: {}; final \
         through a descendant (slot: descendant): {:?}",
        verifier.transient,
        BASE_TREE_NOT_READY.load(Ordering::Relaxed),
        verifier.through_descendant
    );
    info!("test took {:?}", started.elapsed());
}

/// Four validators; the one with the least stake (15%, so blocks are still
/// fast-finalized while it is down) takes snapshots and is restarted from
/// its latest snapshot. After the restart it serves proofs of new blocks
/// (finality, transactions, account deltas and states from the accounts
/// state rebuilt from the snapshot), and refuses those of blocks rooted
/// before: `BlockNotRetained` for the blocks, `DataNotAvailable` for their
/// history leaves.
#[test]
#[serial]
fn test_light_client_restart_from_snapshot() {
    agave_logger::setup_with_default(RUST_LOG_FILTER);
    let started = Instant::now();
    let stakes: Vec<u64> = [8, 5, 4, 3]
        .into_iter()
        .map(|units| units * DEFAULT_NODE_STAKE)
        .collect();
    const RESTARTED: usize = 3;
    let snapshot_config = setup_snapshot_validator_config(NonZeroU64::new(40).unwrap(), 1);
    let mut validator_configs =
        make_identical_validator_configs(&light_client_validator_config(), stakes.len());
    validator_configs[RESTARTED].snapshot_config =
        snapshot_config.validator_config.snapshot_config.clone();
    validator_configs[RESTARTED].validator_exit_backpressure = snapshot_config
        .validator_config
        .validator_exit_backpressure
        .clone();
    let mut tc = start_cluster_with(&stakes, validator_configs);
    let schedule = tc.epoch_schedule().clone();
    let rpc = tc.rpc(0);
    let mut verifier = Verifier::from_genesis(tc.genesis_config());
    info!("cluster started in {:?}", started.elapsed());

    // Before the restart: a transfer, proven by the node to be restarted.
    let mint = tc.cluster.funding_keypair.insecure_clone();
    let recipient = Keypair::new().pubkey();
    let blockhash = rpc.get_latest_blockhash().unwrap();
    let t0 = tc.send(
        &rpc,
        &system_transaction::transfer(&mint, &recipient, 1_000_000, blockhash),
    );
    let rpc_restarted = tc.rpc(RESTARTED);
    let (block_t0, _, _) = verify_transaction(&mut verifier, &rpc_restarted, &t0);

    // A snapshot after the transfer.
    let snapshot = loop {
        let snapshot = tc.cluster.wait_for_next_full_snapshot(
            &snapshot_config
                .validator_config
                .snapshot_config
                .full_snapshot_archives_dir,
            Some(TIMEOUT),
        );
        if snapshot.slot() > t0.slot {
            break snapshot;
        }
    };
    info!("snapshot of node {RESTARTED} at slot {}", snapshot.slot());
    let pubkey = tc.node_pubkey(RESTARTED);
    let info = tc.cluster.exit_node(&pubkey);
    let finalized_before = finalized_slot(&rpc);
    let restarted_at = Instant::now();
    tc.cluster
        .restart_node(&pubkey, info, SocketAddrSpace::Unspecified);
    let rpc_restarted = tc.rpc(RESTARTED);
    info!(
        "node {RESTARTED} restarted in {:?}; the cluster had finalized slot {finalized_before}",
        restarted_at.elapsed()
    );

    // New blocks are served again: wait for the first block the restarted
    // node roots itself.
    let first_new = wait_for(
        format!("node {RESTARTED} to serve a finality proof after its restart"),
        TIMEOUT,
        || {
            let finalized = rpc_restarted
                .get_slot_with_commitment(CommitmentConfig::finalized())
                .ok()?;
            (finalized > finalized_before)
                .then(|| {
                    rooted_slots(&rpc_restarted, finalized_before + 1, finalized)
                        .into_iter()
                        .find(|slot| rpc_restarted.get_finality_proof(*slot).is_ok())
                })
                .flatten()
        },
    );
    info!(
        "node {RESTARTED} serves proofs again from slot {first_new}, {:?} after its restart",
        restarted_at.elapsed()
    );
    while verifier.client.epoch() < schedule.get_epoch(first_new) {
        wait_for_finalized(
            &rpc,
            schedule.get_first_slot_in_epoch(verifier.client.epoch() + 1) + 4,
        );
        verifier.advance_epoch(&rpc);
    }
    let block_new = verifier.verify_block(&rpc_restarted, first_new);
    // Right away, an account in the base tree (unchanged since genesis), from
    // the accounts state the node restored from its snapshot and is
    // rebuilding.
    let (_, program_account) = finalized_account(&rpc, &tc.program);
    let program_hash = hash_of(&tc.program, &program_account.unwrap());
    let response = account_state(&rpc_restarted, &tc.program, first_new);
    response
        .proof
        .verify(&block_new, &tc.program, &program_hash)
        .unwrap();

    // Blocks the node rooted before its restart, or while starting from the
    // snapshot: not retained.
    for slot in [t0.slot, snapshot.slot()] {
        let message = expect_proof_error(
            rpc_restarted.get_finality_proof(slot),
            JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_RETAINED,
        );
        info!("finality proof of slot {slot} after the restart: {message}");
        expect_proof_error(
            rpc_restarted.get_account_delta_proof(&recipient, slot),
            JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_RETAINED,
        );
    }
    // Blocks the cluster rooted while the node was down or catching up:
    // retained if the node rooted them itself after its start (with the
    // certificates it received since), then they verify; otherwise not
    // retained.
    let mut served = vec![];
    let mut not_retained = vec![];
    for slot in rooted_slots(&rpc_restarted, snapshot.slot() + 1, first_new) {
        match rpc_restarted.get_finality_proof(slot) {
            Ok(response) => {
                let response = FinalityResponse::try_from(&response).unwrap();
                verifier.verify_response(&response, slot);
                served.push(slot);
            }
            Err(err) => {
                assert_eq!(
                    rpc_error(&err).map(|(code, _)| code),
                    Some(JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_RETAINED),
                    "slot {slot}: {err}"
                );
                not_retained.push(slot);
            }
        }
    }
    // A prefix is not retained, the rest is.
    assert!(
        not_retained
            .iter()
            .all(|slot| served.first().is_none_or(|first| slot < first)),
        "not retained {not_retained:?}, served {served:?}"
    );
    info!(
        "after the restart: blocks {:?} to {:?} not retained, {:?} to {:?} served (the cluster \
         had finalized {finalized_before} before the restart)",
        not_retained.first(),
        not_retained.last(),
        served.first(),
        served.last()
    );
    let message = expect_proof_error(
        rpc_restarted.get_transaction_proof(&t0.signature, RpcTransactionProofConfig::default()),
        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_RETAINED,
    );
    info!(
        "transaction proof of {} after the restart: {message}",
        t0.signature
    );
    // Their history leaves are folded into the frontier the node restarted
    // from: no history proof, from the restarted node; the other nodes
    // still have them.
    let message = expect_proof_error(
        rpc_restarted.get_block_history_proof(t0.slot, first_new),
        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_DATA_NOT_AVAILABLE,
    );
    info!(
        "history proof of slot {} after the restart: {message}",
        t0.slot
    );
    let response =
        BlockHistoryResponse::try_from(&rpc.get_block_history_proof(t0.slot, first_new).unwrap())
            .unwrap();
    let ancestor = verifier
        .client
        .verify_ancestor(&block_new, &response.history)
        .unwrap();
    assert_eq!(ancestor.commitment_root(), block_t0.commitment_root());

    // After the restart: a transfer to the same account, proven by the
    // restarted node: the receipt, the delta, and the state from the accounts
    // state it rebuilt from the snapshot.
    let blockhash = rpc.get_latest_blockhash().unwrap();
    let t1 = tc.send(
        &rpc,
        &system_transaction::transfer(&mint, &recipient, 1_000_000, blockhash),
    );
    while verifier.client.epoch() < schedule.get_epoch(t1.slot) {
        verifier.advance_epoch(&rpc);
    }
    let (block_t1, _, _) = verify_transaction(&mut verifier, &rpc_restarted, &t1);
    let (_, account) = finalized_account(&rpc_restarted, &recipient);
    let account = account.unwrap();
    assert_eq!(account.lamports, 2_000_000);
    let recipient_hash = hash_of(&recipient, &account);
    verify_delta(&rpc_restarted, &block_t1, &recipient, Some(&recipient_hash));
    verify_state(
        &rpc_restarted,
        &schedule,
        &block_t1,
        &recipient,
        Some(&recipient_hash),
    );
    // An account the transfer did not write, unchanged since the snapshot:
    // from the base tree (or the overlay) of the restarted node.
    verify_state(
        &rpc_restarted,
        &schedule,
        &block_t1,
        &tc.program,
        Some(&program_hash),
    );
    // The same proofs from a node that was not restarted agree.
    verify_state(
        &rpc,
        &schedule,
        &block_t1,
        &recipient,
        Some(&recipient_hash),
    );

    info!("finality kinds: {:?}", verifier.kinds);
    info!(
        "transient getFinalityProof errors (code: count): {:?}; BaseTreeNotReady: {}; final \
         through a descendant (slot: descendant): {:?}",
        verifier.transient,
        BASE_TREE_NOT_READY.load(Ordering::Relaxed),
        verifier.through_descendant
    );
    info!("test took {:?}", started.elapsed());
}
