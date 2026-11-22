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
    agave_snapshots::{SnapshotArchiveKind, snapshot_archive_info::SnapshotArchiveInfoGetter},
    bitvec::{order::Lsb0, vec::BitVec},
    log::*,
    serial_test::serial,
    solana_account::{Account, AccountSharedData},
    solana_clock::{Epoch, Slot},
    solana_commitment_config::CommitmentConfig,
    solana_core::validator::ValidatorConfig,
    solana_download_utils::download_snapshot_archive,
    solana_genesis_config::GenesisConfig,
    solana_hash::Hash,
    solana_instruction::{AccountMeta, Instruction},
    solana_keypair::{Keypair, keypair_from_seed},
    solana_light_client::{
        CertificateKind, CommittedBlock, EpochSchedule, Error as VerifyError, LightClient,
        SignerSet, account_hash, history_leaf,
        rpc::{
            AccountDelta, AccountDeltaResponse, AccountStateResponse, BlockHistoryResponse,
            EpochBoundaryResponse, Finality, FinalityResponse, TransactionResponse,
            ValidatorSetResponse, VerifiedFinal,
        },
        solana_merkle_tree::{
            accounts_state::{AccountLookup, BaseAbsenceProof, BaseLookup},
            commitment_tree::{AppendOnlyFrontier, TreeId},
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
    solana_rpc::light_client_proofs::history_peer::{HistoryPeer, RpcHistoryPeer},
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
            JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_FEATURE_INACTIVE,
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
    serve_light_client_proofs(&mut config);
    config
}

/// Makes a node with `config` serve light-client proofs.
fn serve_light_client_proofs(config: &mut ValidatorConfig) {
    // `default_for_test` has the full RPC API.
    assert!(config.rpc_config.full_api);
    config.rpc_config.enable_light_client_proofs = true;
    // Every node of these clusters is this build.
    config.rpc_config.light_client_proof_gossip_marker = true;
    // Ground truth: getTransaction, getBlock.
    config.rpc_config.enable_rpc_transaction_history = true;
    // Log messages, to compare events with.
    config.rpc_config.enable_extended_tx_metadata_storage = true;
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
    start_cluster_with_accounts(stakes, validator_configs, vec![])
}

/// As [`start_cluster_with`], with more genesis accounts. They replace the
/// accounts genesis would otherwise have at the same addresses.
fn start_cluster_with_accounts(
    stakes: &[u64],
    validator_configs: Vec<ValidatorConfig>,
    genesis_accounts: Vec<(Pubkey, AccountSharedData)>,
) -> TestCluster {
    start_cluster_with_epochs(stakes, validator_configs, genesis_accounts, SLOTS_PER_EPOCH)
}

/// As [`start_cluster_with_accounts`], with epochs of `slots_per_epoch`
/// slots.
fn start_cluster_with_epochs(
    stakes: &[u64],
    validator_configs: Vec<ValidatorConfig>,
    genesis_accounts: Vec<(Pubkey, AccountSharedData)>,
    slots_per_epoch: u64,
) -> TestCluster {
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
        slots_per_epoch,
        stakers_slot_offset: slots_per_epoch,
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
        .chain(genesis_accounts)
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
        Self::from_signer_set(genesis_config, genesis_signer_set(genesis_config))
    }

    /// A client that trusts `signer_set` (of any epoch) to start from.
    fn from_signer_set(genesis_config: &GenesisConfig, signer_set: SignerSet) -> Self {
        let client = LightClient::new(
            genesis_config.epoch_schedule.clone(),
            solana_shred_version::compute_shred_version(&genesis_config.hash(), None),
            signer_set,
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
/// state rebuilt from the snapshot). It also still serves the blocks it
/// rooted before, from the proof data it kept on disk (finality,
/// transactions, account deltas, and their history from a block after the
/// restart), except their account states.
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

    // Blocks the node rooted before its restart: served again from the
    // proof data it kept on disk, with the same result as before.
    let (block_t0_again, _, _) = verify_transaction(&mut verifier, &rpc_restarted, &t0);
    assert_eq!(block_t0_again.block_id(), block_t0.block_id());
    assert_eq!(block_t0_again.commitment_root(), block_t0.commitment_root());
    let (_, account) = finalized_account(&rpc, &recipient);
    let recipient_hash = hash_of(&recipient, &account.unwrap());
    verify_delta(
        &rpc_restarted,
        &block_t0_again,
        &recipient,
        Some(&recipient_hash),
    );
    verifier.verify_block(&rpc_restarted, snapshot.slot());
    // Its accounts state is rebuilt from a bank rooted after the restart if
    // its generation is still held then; if not, the node says so.
    match rpc_restarted.get_account_state_proof(
        &recipient,
        RpcAccountStateProofConfig {
            slot: Some(t0.slot),
        },
    ) {
        Ok(response) => {
            AccountStateResponse::try_from(&response)
                .unwrap()
                .proof
                .verify(&block_t0_again, &recipient, &recipient_hash)
                .unwrap();
            info!(
                "account state of slot {} after the restart: rebuilt",
                t0.slot
            );
        }
        Err(err) => {
            let message = expect_proof_error::<()>(
                Err(err),
                JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_DATA_NOT_AVAILABLE,
            );
            assert!(message.contains("restarted"), "{message}");
            info!(
                "account state of slot {} after the restart: {message}",
                t0.slot
            );
        }
    }

    // Every block from the snapshot to the first new one: kept from before
    // the restart, or rooted by the node since.
    let slots = rooted_slots(&rpc_restarted, snapshot.slot() + 1, first_new);
    // Those in the generation of the first new block also have their
    // accounts state: taken from a bank, or rebuilt by undoing the changes
    // of the blocks after them.
    let generation_of = |slot: Slot| {
        let epoch = schedule.get_epoch(slot);
        let switch_slot =
            schedule.get_first_slot_in_epoch(epoch) + schedule.get_slots_in_epoch(epoch) / 2;
        if slot < switch_slot {
            epoch.checked_sub(1)
        } else {
            Some(epoch)
        }
    };
    // Those of the generation before, which no bank holds after a restart
    // past its switch slot, have theirs from the overlay the node wrote
    // when that generation ended, if the base leaves are still on disk (with
    // these short epochs they may not be); if not, the node says so.
    let mut states_served = 0;
    let mut ended_served = 0;
    let mut ended_not_served = 0;
    for slot in &slots {
        let block = verifier.verify_block(&rpc_restarted, *slot);
        if generation_of(*slot).is_none() {
            continue;
        }
        if generation_of(*slot) == generation_of(first_new) {
            verify_state(
                &rpc_restarted,
                &schedule,
                &block,
                &tc.program,
                Some(&program_hash),
            );
            states_served += 1;
            continue;
        }
        match rpc_restarted.get_account_state_proof(
            &tc.program,
            RpcAccountStateProofConfig { slot: Some(*slot) },
        ) {
            Ok(response) => {
                AccountStateResponse::try_from(&response)
                    .unwrap()
                    .proof
                    .verify(&block, &tc.program, &program_hash)
                    .unwrap();
                ended_served += 1;
            }
            Err(err) => {
                expect_proof_error::<()>(
                    Err(err),
                    JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_DATA_NOT_AVAILABLE,
                );
                ended_not_served += 1;
            }
        }
    }
    info!(
        "after the restart: the accounts state of {states_served} of those blocks is served, \
         and of {ended_served} of the {} blocks of the generation that ended",
        ended_served + ended_not_served
    );
    info!(
        "after the restart: {} blocks from {:?} to {:?} served (the cluster had finalized \
         {finalized_before} before the restart)",
        slots.len(),
        slots.first(),
        slots.last()
    );
    // The history goes on across the restart: a block from before it is
    // proven from a block after it, by the restarted node.
    let response = BlockHistoryResponse::try_from(
        &rpc_restarted
            .get_block_history_proof(t0.slot, first_new)
            .unwrap(),
    )
    .unwrap();
    let ancestor = verifier
        .client
        .verify_ancestor(&block_new, &response.history)
        .unwrap();
    assert_eq!(ancestor.commitment_root(), block_t0.commitment_root());

    // The history leaves, as a node that starts from a snapshot made
    // elsewhere asks another node for them, over JSON RPC: the frontier
    // and the leaves a node gives have the history root of the verified
    // block.
    let leaf_count = response.history.proof.leaf_count;
    let peer = RpcHistoryPeer::new(rpc.url());
    let mut frontier: Option<AppendOnlyFrontier> = None;
    let mut next = 1u64;
    while next < leaf_count {
        let page = peer.history_leaves(next, 100, frontier.is_none()).unwrap();
        assert!(page.leaf_count >= leaf_count);
        assert_eq!(page.start, next);
        let frontier = frontier.get_or_insert_with(|| {
            let roots = page
                .frontier
                .as_ref()
                .unwrap()
                .iter()
                .map(|root| root.parse().unwrap())
                .collect();
            AppendOnlyFrontier::from_parts(1, roots).unwrap()
        });
        for leaf in page.leaves.iter().take((leaf_count - next) as usize) {
            frontier
                .append(
                    TreeId::BLOCK_HISTORY,
                    &history_leaf(
                        leaf.slot,
                        &leaf.block_id.parse().unwrap(),
                        &leaf.commitment_root.parse().unwrap(),
                    ),
                )
                .unwrap();
            next += 1;
        }
    }
    assert_eq!(
        frontier.unwrap().root(TreeId::BLOCK_HISTORY),
        response.history.block_history_root
    );
    info!("history leaves 1 to {leaf_count} from node 0 have the verified history root");

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

/// A node restarted just after a switch slot. The banks it has after the
/// restart no longer hold the generation that the blocks before the switch
/// slot use; it serves their account states from the overlay it wrote when
/// that generation ended and from the base leaves on disk.
///
/// Epochs have 160 slots: generation 1 is used from slot 240 to slot 399,
/// and a node removes its base leaves when it replays the first block of
/// epoch 3, at slot 480. The node is stopped between the two, and started
/// again only when the cluster is past slot 480, so how long it was down
/// does not matter: the leaves are still on its disk when it starts, and
/// it opens them then, before it replays slot 480 while it catches up.
#[test]
#[serial]
fn test_light_client_restart_after_switch_slot() {
    agave_logger::setup_with_default(RUST_LOG_FILTER);
    let started = Instant::now();
    const EPOCH: u64 = 160;
    const FIRST_USE: Slot = EPOCH + EPOCH / 2;
    const SWITCH: Slot = 2 * EPOCH + EPOCH / 2;
    const BASE_REMOVED: Slot = 3 * EPOCH;
    let stakes: Vec<u64> = [8, 5, 4, 3]
        .into_iter()
        .map(|units| units * DEFAULT_NODE_STAKE)
        .collect();
    const RESTARTED: usize = 3;
    let snapshot_config = setup_snapshot_validator_config(NonZeroU64::new(16).unwrap(), 1);
    let mut validator_configs =
        make_identical_validator_configs(&light_client_validator_config(), stakes.len());
    validator_configs[RESTARTED].snapshot_config =
        snapshot_config.validator_config.snapshot_config.clone();
    validator_configs[RESTARTED].validator_exit_backpressure = snapshot_config
        .validator_config
        .validator_exit_backpressure
        .clone();
    let mut tc = start_cluster_with_epochs(&stakes, validator_configs, vec![], EPOCH);
    let schedule = tc.epoch_schedule().clone();
    let rpc = tc.rpc(0);
    let mut verifier = Verifier::from_genesis(tc.genesis_config());

    // A transfer in a block that uses generation 1, before the switch slot.
    wait_for_finalized(&rpc, 2 * EPOCH + 4);
    let mint = tc.cluster.funding_keypair.insecure_clone();
    let recipient = Keypair::new().pubkey();
    let blockhash = rpc.get_latest_blockhash().unwrap();
    let t0 = tc.send(
        &rpc,
        &system_transaction::transfer(&mint, &recipient, 1_000_000, blockhash),
    );
    assert!(t0.slot < SWITCH, "the transfer landed in slot {}", t0.slot);
    let (_, account) = finalized_account(&rpc, &recipient);
    let recipient_hash = hash_of(&recipient, &account.unwrap());
    let (_, program_account) = finalized_account(&rpc, &tc.program);
    let program_hash = hash_of(&tc.program, &program_account.unwrap());

    // The node is stopped as soon as it has rooted a block after the switch
    // slot, 76 slots (30 seconds) before it would remove the base leaves of
    // generation 1. No snapshot is waited for: on restart the node replays
    // its own ledger from its latest snapshot, whichever that is, up to
    // its last root, and the banks it has from there on hold generation 2
    // only.
    let pubkey = tc.node_pubkey(RESTARTED);
    wait_for_finalized(&tc.rpc(RESTARTED), SWITCH + 4);
    let rooted_before = finalized_slot(&tc.rpc(RESTARTED));
    let info = tc.cluster.exit_node(&pubkey);
    info!("node {RESTARTED} had finalized slot {rooted_before} when it was stopped");
    assert!(
        rooted_before + 8 < BASE_REMOVED,
        "the node was stopped too late for this test (slot {rooted_before})"
    );
    // The rest of the cluster goes on into epoch 3.
    wait_for_finalized(&rpc, BASE_REMOVED + 8);
    let finalized_before = finalized_slot(&rpc);
    tc.cluster
        .restart_node(&pubkey, info, SocketAddrSpace::Unspecified);
    let rpc_restarted = tc.rpc(RESTARTED);
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
    info!("node {RESTARTED} serves proofs again from slot {first_new}");

    // The account state at a block before the switch slot, from the
    // restarted node. It reads the ended generation back on a thread of its
    // own, and says "not available" until that is done.
    let state_after_restart = |address: &Pubkey, slot: Slot| {
        let response = wait_for(
            format!("the account state of {address} at {slot} after the restart"),
            TIMEOUT,
            || match rpc_restarted
                .get_account_state_proof(address, RpcAccountStateProofConfig { slot: Some(slot) })
            {
                Ok(response) => Some(response),
                Err(err) => match rpc_error(&err) {
                    Some((
                        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_DATA_NOT_AVAILABLE
                        | JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BASE_TREE_NOT_READY,
                        _,
                    )) => None,
                    _ => panic!("getAccountStateProof({address}, {slot}): {err}"),
                },
            },
        );
        AccountStateResponse::try_from(&response).unwrap()
    };
    // Every rooted block that uses generation 1.
    let slots = rooted_slots(&rpc, FIRST_USE, SWITCH - 1);
    assert!(slots.len() > 64, "{slots:?}");
    for slot in &slots {
        let block = verifier.verify_block(&rpc_restarted, *slot);
        // An account unchanged since genesis: in the base tree.
        let generation = state_after_restart(&tc.program, *slot)
            .proof
            .verify(&block, &tc.program, &program_hash)
            .unwrap();
        assert_eq!(generation, 1, "slot {slot}");
    }
    // An account changed in one of them: in the overlay.
    let block = verifier.verify_block(&rpc_restarted, t0.slot);
    let generation = state_after_restart(&recipient, t0.slot)
        .proof
        .verify(&block, &recipient, &recipient_hash)
        .unwrap();
    assert_eq!(generation, 1);
    // And not there in the block before.
    let before = *slots.iter().rev().find(|slot| **slot < t0.slot).unwrap();
    let block = verifier.verify_block(&rpc_restarted, before);
    state_after_restart(&recipient, before)
        .proof
        .verify_nonexistent(&block, &recipient)
        .unwrap();
    // The blocks of the generation in use since the switch slot too.
    let block = verifier.verify_block(&rpc_restarted, rooted_before);
    verify_state(
        &rpc_restarted,
        &schedule,
        &block,
        &recipient,
        Some(&recipient_hash),
    );
    info!(
        "after the restart: the accounts state of the {} blocks from {FIRST_USE} to {} is served \
         from the ended generation",
        slots.len(),
        SWITCH - 1
    );
    info!("test took {:?}", started.elapsed());
}

/// A node that joins from a snapshot another node made. Its blockstore
/// does not have the blocks before the snapshot, so it cannot build their
/// history leaves itself: it finds the other nodes in gossip, asks them for
/// the leaves, and then proves blocks from before its snapshot.
#[test]
#[serial]
fn test_light_client_history_from_peers_after_snapshot_download() {
    agave_logger::setup_with_default(RUST_LOG_FILTER);
    let started = Instant::now();
    let stakes: Vec<u64> = [8, 5, 4]
        .into_iter()
        .map(|units| units * DEFAULT_NODE_STAKE)
        .collect();
    let source_snapshots = setup_snapshot_validator_config(NonZeroU64::new(40).unwrap(), 1);
    let mut validator_configs =
        make_identical_validator_configs(&light_client_validator_config(), stakes.len());
    validator_configs[0].snapshot_config =
        source_snapshots.validator_config.snapshot_config.clone();
    validator_configs[0].validator_exit_backpressure = source_snapshots
        .validator_config
        .validator_exit_backpressure
        .clone();
    let mut tc = start_cluster_with(&stakes, validator_configs);
    let rpc = tc.rpc(0);
    let mut verifier = Verifier::from_genesis(tc.genesis_config());

    // A transfer, then a snapshot after it.
    let mint = tc.cluster.funding_keypair.insecure_clone();
    let recipient = Keypair::new().pubkey();
    let blockhash = rpc.get_latest_blockhash().unwrap();
    let t0 = tc.send(
        &rpc,
        &system_transaction::transfer(&mint, &recipient, 1_000_000, blockhash),
    );
    let (block_t0, _, _) = verify_transaction(&mut verifier, &rpc, &t0);
    let snapshot = loop {
        let snapshot = tc.cluster.wait_for_next_full_snapshot(
            &source_snapshots
                .validator_config
                .snapshot_config
                .full_snapshot_archives_dir,
            Some(TIMEOUT),
        );
        if snapshot.slot() > t0.slot + 8 {
            break snapshot;
        }
    };
    info!("snapshot of node 0 at slot {}", snapshot.slot());

    // The new node: the snapshot is downloaded from node 0, as a validator
    // that bootstraps does; no stake, no history peer named.
    let mut joiner_snapshots = setup_snapshot_validator_config(NonZeroU64::new(40).unwrap(), 1);
    let joiner_config = &mut joiner_snapshots.validator_config;
    serve_light_client_proofs(joiner_config);
    assert!(
        joiner_config
            .rpc_config
            .light_client_proof_history_peers
            .is_empty()
    );
    download_snapshot_archive(
        &tc.cluster.entry_point_info.rpc().unwrap(),
        &joiner_config.snapshot_config.full_snapshot_archives_dir,
        &joiner_config
            .snapshot_config
            .incremental_snapshot_archives_dir,
        (snapshot.slot(), *snapshot.hash()),
        SnapshotArchiveKind::Full,
        joiner_config
            .snapshot_config
            .maximum_full_snapshot_archives_to_retain,
        joiner_config
            .snapshot_config
            .maximum_incremental_snapshot_archives_to_retain,
        false,
        &mut None,
    )
    .unwrap();
    let joiner = tc.cluster.add_validator_listener(
        joiner_config,
        0,
        &ValidatorKeys::new(),
        true,
        SocketAddrSpace::Unspecified,
    );
    let rpc_joiner = RpcClient::new_with_commitment(
        format!(
            "http://{}",
            tc.cluster.validators[&joiner]
                .info
                .contact_info
                .rpc()
                .unwrap()
        ),
        CommitmentConfig::finalized(),
    );

    // A block the new node rooted itself.
    let first_new = wait_for("the new node to serve a finality proof", TIMEOUT, || {
        let finalized = rpc_joiner
            .get_slot_with_commitment(CommitmentConfig::finalized())
            .ok()?;
        rooted_slots(&rpc_joiner, snapshot.slot() + 1, finalized)
            .into_iter()
            .find(|slot| rpc_joiner.get_finality_proof(*slot).is_ok())
    });
    // The signer sets of the epochs up to that block come from node 0: the
    // new node does not have the blocks that commit to them.
    let schedule = tc.epoch_schedule().clone();
    while verifier.client.epoch() < schedule.get_epoch(first_new) {
        verifier.advance_epoch(&rpc);
    }
    let block_new = verifier.verify_block(&rpc_joiner, first_new);
    info!("the new node serves proofs from slot {first_new}");
    // It never had the block of the transfer: no finality proof of its own.
    assert!(t0.slot < snapshot.slot());
    let err = rpc_joiner.get_finality_proof(t0.slot).unwrap_err();
    assert!(
        matches!(
            rpc_error(&err),
            Some((
                JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_FINAL
                    | JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_RETAINED,
                _
            ))
        ),
        "{err}"
    );
    // The block of the transfer, proven from the new block by the new
    // node, with history leaves it took from the other nodes.
    let response = wait_for(
        "the new node to prove a block from before its snapshot",
        TIMEOUT,
        || match rpc_joiner.get_block_history_proof(t0.slot, first_new) {
            Ok(response) => Some(response),
            Err(err) => match rpc_error(&err) {
                Some((JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_DATA_NOT_AVAILABLE, _)) => None,
                _ => panic!("getBlockHistoryProof({}, {first_new}): {err}", t0.slot),
            },
        },
    );
    let response = BlockHistoryResponse::try_from(&response).unwrap();
    let ancestor = verifier
        .client
        .verify_ancestor(&block_new, &response.history)
        .unwrap();
    assert_eq!(ancestor.commitment_root(), block_t0.commitment_root());
    // The same proof as a node that was there all along gives.
    let from_node_0 =
        BlockHistoryResponse::try_from(&rpc.get_block_history_proof(t0.slot, first_new).unwrap())
            .unwrap();
    assert_eq!(response.history, from_node_0.history);
    info!("test took {:?}", started.elapsed());
}

/// A single node produces every block, and can root a block before its
/// broadcast has written the whole block to blockstore. Asked for each
/// block as soon as it is rooted, the node answers "block not final" until
/// then, never another error, and then serves a proof that verifies.
#[test]
#[serial]
fn test_light_client_finality_right_after_root() {
    agave_logger::setup_with_default(RUST_LOG_FILTER);
    let tc = start_cluster(&[DEFAULT_NODE_STAKE]);
    let rpc = tc.rpc(0);
    let mut verifier = Verifier::from_genesis(tc.genesis_config());
    let mut next = wait_for_finalized(&rpc, 4);
    // Stays within epoch 0, which the genesis signer set verifies.
    let end = SLOTS_PER_EPOCH - 8;
    let (mut served, mut not_final_first) = (0u64, 0u64);
    while next < end {
        let finalized = finalized_slot(&rpc).min(end - 1);
        if finalized < next {
            continue;
        }
        for slot in rooted_slots(&rpc, next, finalized) {
            let asked = Instant::now();
            let mut first = true;
            let response = loop {
                match rpc.get_finality_proof(slot) {
                    Ok(response) => break FinalityResponse::try_from(&response).unwrap(),
                    Err(err) => match rpc_error(&err) {
                        Some((
                            JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_FINAL
                            | JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_CERTIFICATES_NOT_AVAILABLE,
                            _,
                        )) => {
                            not_final_first += u64::from(first);
                            first = false;
                            assert!(
                                asked.elapsed() < TIMEOUT,
                                "getFinalityProof({slot}) never served: {err}"
                            );
                        }
                        _ => panic!("getFinalityProof({slot}): {err}"),
                    },
                }
            };
            verifier.verify_response(&response, slot);
            served += 1;
        }
        next = finalized + 1;
    }
    info!("{served} blocks served, {not_final_first} after a first \"not final\" answer");
    assert!(served > 0);
}

/// The six feature gates of the commitments.
fn commitment_features() -> [Pubkey; 6] {
    [
        agave_feature_set::block_commitment_root::id(),
        agave_feature_set::transaction_receipt_tree::id(),
        agave_feature_set::accounts_delta_root::id(),
        agave_feature_set::epoch_validator_set_root::id(),
        agave_feature_set::epoch_boundary_commitment::id(),
        agave_feature_set::accounts_state_root::id(),
    ]
}

fn feature_activated_at(rpc: &RpcClient, feature: &Pubkey) -> Option<Slot> {
    let account = rpc.get_account(feature).unwrap();
    solana_feature_gate_interface::from_account(&account)
        .unwrap()
        .activated_at
}

/// The six features are not active in genesis: their feature accounts are
/// pending, as after the activation transaction on a live cluster, and the
/// running node activates them at the first epoch boundary. Before it no
/// proof is served; from the first block of epoch 1 a client that trusts
/// the signer set of epoch 1 verifies blocks, transactions, account deltas
/// and states, follows the validator set into epoch 2 and checks the epoch
/// boundary.
#[test]
#[serial]
fn test_light_client_features_activated_on_running_cluster() {
    agave_logger::setup_with_default(RUST_LOG_FILTER);
    let started = Instant::now();
    let pending = commitment_features()
        .into_iter()
        .map(|feature| {
            (
                feature,
                solana_feature_gate_interface::create_account(
                    &solana_feature_gate_interface::Feature { activated_at: None },
                    Rent::default()
                        .minimum_balance(solana_feature_gate_interface::Feature::size_of()),
                ),
            )
        })
        .collect();
    let tc = start_cluster_with_accounts(
        &[DEFAULT_NODE_STAKE],
        make_identical_validator_configs(&light_client_validator_config(), 1),
        pending,
    );
    let rpc = tc.rpc(0);
    let schedule = tc.epoch_schedule().clone();
    let first_slot_1 = schedule.get_first_slot_in_epoch(1);
    let first_slot_2 = schedule.get_first_slot_in_epoch(2);
    let mint = tc.cluster.funding_keypair.insecure_clone();
    let recipient = Keypair::new().pubkey();
    info!("cluster started in {:?}", started.elapsed());

    // Epoch 0: the features are pending, and no proof is served.
    let finalized = wait_for_finalized(&rpc, 8);
    assert!(finalized < first_slot_1);
    for feature in commitment_features() {
        assert_eq!(feature_activated_at(&rpc, &feature), None);
    }
    let in_epoch_0 = *rooted_slots(&rpc, 4, finalized).last().unwrap();
    let message = expect_proof_error(
        rpc.get_finality_proof(in_epoch_0),
        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_FEATURE_INACTIVE,
    );
    info!("finality proof of slot {in_epoch_0}, before activation: {message}");
    let blockhash = rpc.get_latest_blockhash().unwrap();
    let t0 = tc.send(
        &rpc,
        &system_transaction::transfer(&mint, &recipient, 1_000_000, blockhash),
    );
    assert!(t0.slot < first_slot_1);
    let err = rpc
        .get_transaction_proof(&t0.signature, RpcTransactionProofConfig::default())
        .unwrap_err();
    info!("transaction proof before activation: {err}");

    // The epoch boundary activates them, on the running node.
    wait_for_finalized(&rpc, first_slot_1 + 4);
    for feature in commitment_features() {
        assert_eq!(feature_activated_at(&rpc, &feature), Some(first_slot_1));
    }
    let rooted = rooted_slots(&rpc, 0, first_slot_1 + 4);
    let last_of_epoch_0 = *rooted
        .iter()
        .filter(|slot| **slot < first_slot_1)
        .last()
        .unwrap();
    let first_of_epoch_1 = *rooted.iter().find(|slot| **slot >= first_slot_1).unwrap();
    expect_proof_error(
        rpc.get_finality_proof(last_of_epoch_0),
        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_FEATURE_INACTIVE,
    );

    // A client starts at the activation epoch: it trusts the signer set of
    // epoch 1, here computed from genesis. No block of epoch 0 commits to
    // it, and the node says so.
    let signers = Bank::new_for_tests(tc.genesis_config())
        .validator_set_signers(1)
        .expect("genesis has the stakes of epoch 1");
    let set_1 = SignerSet::new(1, signers).unwrap();
    let response = ValidatorSetResponse::try_from(
        &rpc.get_validator_set(1, RpcCommittingBlockConfig::default())
            .unwrap(),
    )
    .unwrap();
    assert_eq!(response.signer_set.epoch(), 1);
    assert_eq!(
        response.signer_set.validator_set_root(),
        set_1.validator_set_root()
    );
    assert!(response.committed_by.is_none());
    let mut verifier = Verifier::from_signer_set(tc.genesis_config(), set_1);

    // The first block of epoch 1 is the first with a commitment root.
    let block_f = verifier.verify_block(&rpc, first_of_epoch_1);
    info!("first block with the features: slot {first_of_epoch_1}");

    // A transfer in epoch 1: receipt and account delta.
    let blockhash = rpc.get_latest_blockhash().unwrap();
    let t1 = tc.send(
        &rpc,
        &system_transaction::transfer(&mint, &recipient, 1_000_000, blockhash),
    );
    assert_eq!(schedule.get_epoch(t1.slot), 1);
    let (block_t1, _, _) = verify_transaction(&mut verifier, &rpc, &t1);
    let (_, account) = finalized_account(&rpc, &recipient);
    let recipient_hash = hash_of(&recipient, &account.unwrap());
    verify_delta(&rpc, &block_t1, &recipient, Some(&recipient_hash));

    // The history starts at activation: the first block is an ancestor of
    // the transfer's block. The block before it is the anchor leaf: proven
    // as an ancestor, without a commitment root.
    let response = BlockHistoryResponse::try_from(&when_final(
        format!("getBlockHistoryProof({first_of_epoch_1}, {})", t1.slot),
        || rpc.get_block_history_proof(first_of_epoch_1, t1.slot),
    ))
    .unwrap();
    let ancestor = verifier
        .client
        .verify_ancestor(&block_t1, &response.history)
        .unwrap();
    assert_eq!(ancestor.commitment_root(), block_f.commitment_root());
    let response = BlockHistoryResponse::try_from(
        &rpc.get_block_history_proof(last_of_epoch_0, t1.slot)
            .unwrap(),
    )
    .unwrap();
    assert!(
        verifier
            .client
            .verify_ancestor(&block_t1, &response.history)
            .is_err()
    );
    let anchor = response.history.verify_anchor(&block_t1).unwrap();
    assert_eq!(anchor.slot, last_of_epoch_0);
    info!(
        "the anchor (slot {last_of_epoch_0}) is proven from slot {}",
        t1.slot
    );
    // The epoch boundary of the activation epoch: the anchor is the last
    // block before it.
    let response = EpochBoundaryResponse::try_from(
        &rpc.get_epoch_boundary(1, RpcCommittingBlockConfig::default())
            .unwrap(),
    )
    .unwrap();
    let block = verifier.verify_block(&rpc, response.slot);
    let boundary = response.proof.verify(&block, &schedule).unwrap();
    assert_eq!(boundary.boundary.parent_slot, last_of_epoch_0);
    assert_eq!(boundary.boundary.first_slot, first_of_epoch_1);

    // The account state. In the first half of the activation epoch the
    // generation of the epoch before would be used, which does not exist:
    // slot 5 is zero. From the switch slot on, generation 1.
    let message = expect_proof_error(
        rpc.get_account_state_proof(
            &recipient,
            RpcAccountStateProofConfig {
                slot: Some(t1.slot),
            },
        ),
        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_DATA_NOT_AVAILABLE,
    );
    info!("account state at slot {}: {message}", t1.slot);
    let switch_slot = first_slot_1 + SLOTS_PER_EPOCH / 2;
    assert!(t1.slot < switch_slot);
    wait_for_finalized(&rpc, switch_slot + 2);
    let after_switch = *rooted_slots(&rpc, switch_slot, switch_slot + 8)
        .first()
        .unwrap();
    let block = verifier.verify_block(&rpc, after_switch);
    let generation = verify_state(&rpc, &schedule, &block, &recipient, Some(&recipient_hash));
    assert_eq!(generation, 1);

    // Epoch 2: the validator set handoff from a block of epoch 1, the epoch
    // boundary and the account state.
    wait_for_finalized(&rpc, first_slot_2 + 4);
    let committing = verifier.advance_epoch(&rpc);
    assert_eq!(schedule.get_epoch(committing.slot()), 1);
    assert_eq!(verifier.client.epoch(), 2);
    let rooted = rooted_slots(&rpc, first_slot_1, first_slot_2 + 4);
    let last_of_epoch_1 = *rooted
        .iter()
        .filter(|slot| **slot < first_slot_2)
        .last()
        .unwrap();
    let first_of_epoch_2 = *rooted.iter().find(|slot| **slot >= first_slot_2).unwrap();
    let block_e2 = verifier.verify_block(&rpc, first_of_epoch_2);
    let response = EpochBoundaryResponse::try_from(&when_final("getEpochBoundary(2)", || {
        rpc.get_epoch_boundary(
            2,
            RpcCommittingBlockConfig {
                at_slot: Some(first_of_epoch_2),
            },
        )
    }))
    .unwrap();
    let boundary = response.proof.verify(&block_e2, &schedule).unwrap();
    assert_eq!(boundary.boundary.parent_slot, last_of_epoch_1);
    assert_eq!(boundary.boundary.first_slot, first_of_epoch_2);
    let generation = verify_state(
        &rpc,
        &schedule,
        &block_e2,
        &recipient,
        Some(&recipient_hash),
    );
    assert_eq!(generation, 1);
    info!(
        "finality kinds: {:?}; test took {:?}",
        verifier.kinds,
        started.elapsed()
    );
}

/// The fetching client against a real node, over JSON-RPC: it starts from
/// the genesis signer set, fetches and verifies a transaction, what its
/// block did to an account, and the account's state, and follows the
/// validator set into the next epoch by itself when asked about a block
/// there.
#[test]
#[serial]
fn test_light_client_fetcher_over_rpc() {
    use solana_light_client::fetch::{AccountChange, FetchedAncestor, ProofFetcher, Retry};
    agave_logger::setup_with_default(RUST_LOG_FILTER);
    let tc = start_cluster(&[DEFAULT_NODE_STAKE]);
    let rpc = tc.rpc(0);
    let schedule = tc.epoch_schedule().clone();
    let genesis_config = tc.genesis_config();
    let client = LightClient::new(
        schedule.clone(),
        solana_shred_version::compute_shred_version(&genesis_config.hash(), None),
        genesis_signer_set(genesis_config),
    );
    // The node answers "not final yet" for a moment after it roots a block.
    let mut fetcher = ProofFetcher::new(client, tc.rpc(0)).with_retry(Retry {
        attempts: 100,
        delay: Duration::from_millis(50),
    });
    let mint = tc.cluster.funding_keypair.insecure_clone();
    let recipient = Keypair::new().pubkey();
    let transfer = |lamports| {
        let blockhash = rpc.get_latest_blockhash().unwrap();
        let sent = tc.send(
            &rpc,
            &system_transaction::transfer(&mint, &recipient, lamports, blockhash),
        );
        wait_for_finalized(&rpc, sent.slot);
        sent
    };

    // Epoch 0.
    let t0 = transfer(1_000_000);
    assert_eq!(schedule.get_epoch(t0.slot), 0);
    let signature: [u8; 64] = t0.signature.into();
    let fetched0 = fetcher
        .transaction(
            &signature,
            Some(RpcProofSelection::All),
            Some(RpcProofSelection::All),
        )
        .unwrap();
    assert_eq!(fetched0.block.slot(), t0.slot);
    assert!(fetched0.transaction.is_success());
    assert_eq!(fetched0.instructions.len(), 1);
    let (_, account) = finalized_account(&rpc, &recipient);
    let hash0 = hash_of(&recipient, &account.unwrap());
    let (block, change) = fetcher.account_change(&recipient, t0.slot).unwrap();
    assert_eq!(block, fetched0.block);
    assert_eq!(
        change,
        AccountChange::Changed {
            account_hash: hash0
        }
    );
    assert_eq!(fetcher.epoch(), 0);

    // Epoch 1: asked about a transaction there, the fetcher first follows
    // the validator set.
    wait_for_finalized(&rpc, schedule.get_first_slot_in_epoch(1) + 4);
    let t1 = transfer(2_000_000);
    assert_eq!(schedule.get_epoch(t1.slot), 1);
    let signature: [u8; 64] = t1.signature.into();
    let fetched1 = fetcher.transaction(&signature, None, None).unwrap();
    assert_eq!(fetcher.epoch(), 1);
    assert_eq!(fetched1.block.slot(), t1.slot);
    assert!(fetched1.transaction.is_success());

    // The account's state at the end of that block, and an address with no
    // account.
    let (_, account) = finalized_account(&rpc, &recipient);
    let account = account.unwrap();
    assert_eq!(account.lamports, 3_000_000);
    let hash1 = hash_of(&recipient, &account);
    let block = fetcher
        .account_state(&recipient, Some(t1.slot), Some(&hash1))
        .unwrap();
    assert_eq!(block, fetched1.block);
    fetcher
        .account_state(&Keypair::new().pubkey(), Some(t1.slot), None)
        .unwrap();
    // The state it had after the first transfer is no longer its state.
    assert!(
        fetcher
            .account_state(&recipient, Some(t1.slot), Some(&hash0))
            .is_err()
    );

    // The first block as an ancestor of the second, and where epoch 0
    // ended.
    let (_, ancestor) = fetcher.ancestor(t0.slot, t1.slot).unwrap();
    let FetchedAncestor::Block(ancestor) = ancestor else {
        panic!("{ancestor:?}");
    };
    assert_eq!(ancestor.commitment_root(), fetched0.block.commitment_root());
    let (_, boundary) = fetcher.epoch_boundary(1).unwrap();
    assert!(boundary.boundary.parent_slot < schedule.get_first_slot_in_epoch(1));
    // A block of epoch 0 still verifies.
    assert_eq!(fetcher.block(t0.slot).unwrap(), fetched0.block);
}
