//! The light-client proof methods, called through JSON-RPC and verified with
//! `solana-light-client`.
//!
//! A chain of banks with the six light-client features activating at the
//! first epoch boundary (as in `light-client/tests/end_to_end.rs`), whose
//! blocks are written to blockstore as real shreds (block header, entries,
//! a version 2 footer with the bank hash and commitment root, and the final
//! tick): the block id of every block with the features is the double
//! Merkle root that blockstore computes, and footer proofs are read from
//! blockstore as for a running validator. The banks are rooted into the
//! proof store as root handling does. Certificates are signed with the
//! validators' BLS keys (`test_create_base2_unverified_certificate`) and
//! handed to the store as the consensus pool does; this harness has no
//! consensus pool.

#![allow(clippy::arithmetic_side_effects)]

use {
    super::{LightClient as _, *},
    crate::light_client_proofs::{LightClientProofConfig, LightClientProofStore},
    agave_bls_cert_verify::cert_verify::test_create_base2_unverified_certificate,
    agave_feature_set as feature_set,
    agave_votor_messages::{
        certificate::{Certificate as VotorCertificate, CertificateType},
        consensus_message::{Block, BlockId},
    },
    jsonrpc_core::MetaIoHandler,
    serde::de::DeserializeOwned,
    serde_json::{Value, json},
    solana_account::{AccountSharedData, ReadableAccount},
    solana_bls_signatures::keypair::Keypair as BlsKeypair,
    solana_entry::{
        block_component::{
            BlockComponent, BlockFooterV1, BlockFooterV2, BlockHeaderV1, VersionedBlockMarker,
        },
        entry::create_ticks,
    },
    solana_fee_calculator::FeeRateGovernor,
    solana_genesis_config::GenesisConfig,
    solana_hash::Hash,
    solana_instruction::{AccountMeta, Instruction},
    solana_keypair::Keypair,
    solana_ledger::{
        blockstore::Blockstore,
        blockstore_meta::BlockLocation,
        shred::{ProcessShredsStats, Shred, Shredder},
    },
    solana_light_client::{
        CommittedBlock, EpochSchedule, Error as VerifyError, LightClient, SignerSet, account_hash,
        rpc::{
            AccountDelta, AccountDeltaResponse, AccountStateResponse, BlockHistoryResponse,
            EpochBoundaryResponse, Finality, FinalityResponse, TransactionResponse,
            ValidatorSetResponse, VerifiedFinal,
        },
        solana_merkle_tree::receipt_tree::{EventKind, InstructionAccountMeta, log_data_payload},
    },
    solana_net_utils::SocketAddrSpace,
    solana_pubkey::Pubkey,
    solana_rpc_client_api::custom_error::{
        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_FINAL,
        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_RETAINED,
        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_CERTIFICATES_NOT_AVAILABLE,
        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_FEATURE_INACTIVE,
        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_NOT_FOUND,
        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_PROOFS_DISABLED,
    },
    solana_runtime::{
        bank::{Bank, test_utils::goto_end_of_slot},
        genesis_utils::{
            GenesisConfigInfo, ValidatorVoteKeypairs, create_genesis_config_with_vote_accounts,
            minimum_vote_account_balance_for_vat,
        },
        loader_utils::create_program_with_elf,
    },
    solana_sdk_ids::{bpf_loader_upgradeable, system_program},
    solana_signer::Signer,
    solana_system_transaction as system_transaction,
    solana_transaction::Transaction,
    std::sync::Arc,
};

const SLOTS_PER_EPOCH: u64 = 64;
const SHRED_VERSION: u16 = 4242;
const LAMPORTS_PER_SIGNATURE: u64 = 5000;

/// See `runtime/tests/fixtures/receipt_test_program/receipt_test_program.c`:
/// with data `[0, extra..]` it logs "outer", invokes itself with `[1]` (which
/// logs data `["inner", ""]` and transfers one lamport from account 0 to
/// account 1), then logs data `["after", extra]`.
const RECEIPT_TEST_PROGRAM: &[u8] = include_bytes!(
    "../../../../runtime/tests/fixtures/receipt_test_program/receipt_test_program.so"
);

fn all_features() -> [Pubkey; 6] {
    [
        feature_set::block_commitment_root::id(),
        feature_set::transaction_receipt_tree::id(),
        feature_set::accounts_delta_root::id(),
        feature_set::epoch_validator_set_root::id(),
        feature_set::epoch_boundary_commitment::id(),
        feature_set::accounts_state_root::id(),
    ]
}

fn genesis() -> (GenesisConfig, Keypair, Vec<ValidatorVoteKeypairs>) {
    let validators: Vec<_> = (0..3).map(|_| ValidatorVoteKeypairs::new_rand()).collect();
    let stakes = vec![
        50 * minimum_vote_account_balance_for_vat(100),
        30 * minimum_vote_account_balance_for_vat(100),
        20 * minimum_vote_account_balance_for_vat(100),
    ];
    let GenesisConfigInfo {
        mut genesis_config,
        mint_keypair,
        ..
    } = create_genesis_config_with_vote_accounts(1_000_000_000_000, &validators, stakes);
    genesis_config.epoch_schedule = EpochSchedule::custom(SLOTS_PER_EPOCH, SLOTS_PER_EPOCH, false);
    genesis_config.fee_rate_governor = FeeRateGovernor::new(LAMPORTS_PER_SIGNATURE, 0);
    for feature_id in all_features() {
        let lamports = genesis_config
            .rent
            .minimum_balance(solana_feature_gate_interface::Feature::size_of())
            .max(1);
        genesis_config.accounts.insert(
            feature_id,
            solana_feature_gate_interface::create_account(
                &solana_feature_gate_interface::Feature { activated_at: None },
                lamports,
            )
            .into(),
        );
    }
    (genesis_config, mint_keypair, validators)
}

fn new_child(parent: &Arc<Bank>, slot: u64) -> Arc<Bank> {
    Arc::new(Bank::new_from_parent(
        parent.clone(),
        *parent.leader(),
        slot,
    ))
}

fn data_shreds(shreds: Vec<Shred>) -> Vec<Shred> {
    shreds.into_iter().filter(Shred::is_data).collect()
}

fn component_shreds(
    slot: u64,
    parent_slot: u64,
    component: &BlockComponent,
    index: u32,
    is_last_in_slot: bool,
) -> Vec<Shred> {
    data_shreds(
        Shredder::new(slot, parent_slot, 0, SHRED_VERSION)
            .unwrap()
            .make_merkle_shreds_from_component(
                &Keypair::new(),
                component,
                is_last_in_slot,
                Hash::new_unique(),
                index,
                index,
                &mut ProcessShredsStats::default(),
            ),
    )
}

/// Freezes `bank` and writes its block to blockstore: a block header naming
/// the parent, some entries, a footer (version 2, binding the bank hash and
/// commitment root, if the bank has a commitment root) and the final tick.
/// The bank's block id is the double Merkle root blockstore computes.
fn freeze_and_store(blockstore: &Blockstore, bank: &Arc<Bank>) {
    goto_end_of_slot(bank.clone());
    bank.freeze();
    let slot = bank.slot();
    let parent = bank.parent().unwrap();
    let parent_slot = parent.slot();
    let header =
        BlockComponent::new_block_marker(VersionedBlockMarker::from_block_header(BlockHeaderV1 {
            parent_slot,
            parent_block_id: parent.block_id().unwrap(),
        }));
    let mut shreds = component_shreds(slot, parent_slot, &header, 0, false);
    let next = |shreds: &[Shred]| shreds.last().unwrap().index() + 1;
    let entries = BlockComponent::new_entry_batch(create_ticks(3, 1, Hash::new_unique())).unwrap();
    shreds.extend(component_shreds(
        slot,
        parent_slot,
        &entries,
        next(&shreds),
        false,
    ));
    let footer_v1 = BlockFooterV1 {
        bank_hash: bank.hash(),
        block_producer_time_nanos: 0,
        block_user_agent: vec![],
        block_final_cert: None,
        skip_reward_cert: None,
        notar_reward_cert: None,
    };
    let footer = match bank.commitment_root() {
        Some(commitment_root) => VersionedBlockMarker::from_block_footer_v2(
            BlockFooterV2::from_v1(footer_v1, commitment_root),
        ),
        None => VersionedBlockMarker::from_block_footer(footer_v1),
    };
    shreds.extend(component_shreds(
        slot,
        parent_slot,
        &BlockComponent::new_block_marker(footer),
        next(&shreds),
        false,
    ));
    let tick = BlockComponent::new_entry_batch(create_ticks(1, 1, Hash::new_unique())).unwrap();
    shreds.extend(component_shreds(
        slot,
        parent_slot,
        &tick,
        next(&shreds),
        true,
    ));
    blockstore.insert_shreds(shreds, true).unwrap();
    assert!(blockstore.is_full(slot));
    let block_id = blockstore
        .get_double_merkle_root(slot, BlockLocation::Original)
        .unwrap()
        .unwrap();
    bank.set_block_id(Some(BlockId::from(block_id)));
}

fn block_id(bank: &Bank) -> Hash {
    bank.block_id().unwrap().to_hash()
}

fn keypairs_by_rank(
    validators: &[ValidatorVoteKeypairs],
    signer_set: &SignerSet,
) -> Vec<BlsKeypair> {
    signer_set
        .signers()
        .iter()
        .map(|signer| {
            validators
                .iter()
                .find(|v| {
                    v.bls_keypair.public.to_bytes_compressed() == signer.bls_pubkey_compressed
                })
                .unwrap()
                .bls_keypair
                .clone()
        })
        .collect()
}

/// A certificate as the consensus pool completes it.
fn certificate(
    keypairs: &[BlsKeypair],
    cert_type: CertificateType,
    ranks: &[usize],
) -> VotorCertificate {
    let cert = test_create_base2_unverified_certificate(keypairs, SHRED_VERSION, cert_type, ranks);
    VotorCertificate {
        cert_type,
        signature: cert.signature,
        bitmap: cert.bitmap,
    }
}

fn block(bank: &Bank) -> Block {
    Block {
        slot: bank.slot(),
        block_id: bank.block_id().unwrap(),
    }
}

fn slow_certificates(bank: &Bank, keypairs: &[BlsKeypair]) -> [VotorCertificate; 2] {
    [
        certificate(keypairs, CertificateType::Notarize(block(bank)), &[0, 1]),
        certificate(keypairs, CertificateType::Finalize(bank.slot()), &[0, 1]),
    ]
}

fn fast_certificate(bank: &Bank, keypairs: &[BlsKeypair]) -> VotorCertificate {
    let ranks: Vec<usize> = (0..keypairs.len()).collect();
    certificate(keypairs, CertificateType::FinalizeFast(block(bank)), &ranks)
}

fn hash_of(account: &AccountSharedData, address: &Pubkey) -> Hash {
    account_hash(
        account.lamports(),
        account.data(),
        account.executable(),
        account.owner(),
        address,
    )
}

fn meta(pubkey: &Pubkey, is_signer: bool, is_writable: bool) -> InstructionAccountMeta {
    InstructionAccountMeta {
        pubkey: pubkey.to_bytes(),
        is_signer,
        is_writable,
    }
}

fn transfer(bank: &Bank, from: &Keypair, to: &Pubkey, lamports: u64) -> Transaction {
    let transaction = system_transaction::transfer(from, to, lamports, bank.last_blockhash());
    bank.process_transaction(&transaction).unwrap();
    transaction
}

/// The JSON-RPC handler of the light-client methods.
struct Rpc {
    io: MetaIoHandler<JsonRpcRequestProcessor>,
    meta: JsonRpcRequestProcessor,
}

impl Rpc {
    fn new(
        bank: Bank,
        enable: bool,
        store: Option<Arc<LightClientProofStore>>,
        blockstore: Option<Arc<Blockstore>>,
    ) -> Self {
        let mut meta = JsonRpcRequestProcessor::new_from_bank(bank, SocketAddrSpace::Unspecified)
            .with_light_client_proofs(store);
        meta.config.enable_light_client_proofs = enable;
        if let Some(blockstore) = blockstore {
            meta.blockstore = blockstore;
        }
        let mut io = MetaIoHandler::default();
        io.extend_with(LightClientImpl.to_delegate());
        Self { io, meta }
    }

    /// Calls `method`; the result, or the error code and data.
    fn call<T: DeserializeOwned>(
        &self,
        method: &str,
        params: Value,
    ) -> std::result::Result<T, (i64, Value)> {
        let request = json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params});
        let response = self
            .io
            .handle_request_sync(&request.to_string(), self.meta.clone())
            .expect("a response");
        let response: Value = serde_json::from_str(&response).unwrap();
        if let Some(error) = response.get("error") {
            return Err((
                error["code"].as_i64().unwrap(),
                error.get("data").cloned().unwrap_or(Value::Null),
            ));
        }
        Ok(serde_json::from_value(response["result"].clone()).unwrap())
    }

    fn error(&self, method: &str, params: Value) -> (i64, Value) {
        match self.call::<Value>(method, params) {
            Ok(result) => panic!("{method} succeeded: {result}"),
            Err(error) => error,
        }
    }

    fn finality(&self, slot: u64) -> FinalityResponse {
        let response: RpcFinalityProof = self.call("getFinalityProof", json!([slot])).unwrap();
        FinalityResponse::try_from(&response).unwrap()
    }
}

/// The chain, the store fed with it, and the validators' keys.
struct Chain {
    genesis_config: GenesisConfig,
    validators: Vec<ValidatorVoteKeypairs>,
    mint: Keypair,
    blockstore: Arc<Blockstore>,
    banks: std::collections::BTreeMap<u64, Arc<Bank>>,
    recipient: Pubkey,
    quiet: Pubkey,
    fresh: Pubkey,
    program: Pubkey,
    program_tx: Transaction,
    transfer_tx: Transaction,
}

impl Chain {
    fn bank(&self, slot: u64) -> &Arc<Bank> {
        &self.banks[&slot]
    }

    fn store(&self, retained_blocks: usize) -> Arc<LightClientProofStore> {
        let store = Arc::new(LightClientProofStore::new(
            LightClientProofConfig {
                retained_blocks,
                ..LightClientProofConfig::default()
            },
            self.blockstore.clone(),
        ));
        // Rooted from slot 62 on; slots 0 and 1 are roots from before.
        let rooted: Vec<_> = self
            .banks
            .range(62..)
            .map(|(_, bank)| bank.clone())
            .collect();
        store.on_rooted_banks(&rooted);
        store
    }

    fn rpc(&self, store: &Arc<LightClientProofStore>) -> Rpc {
        Rpc::new(
            Bank::new_for_tests(&self.genesis_config),
            true,
            Some(store.clone()),
            Some(self.blockstore.clone()),
        )
    }
}

fn build_chain() -> Chain {
    let (genesis_config, mint, validators) = genesis();
    let ledger_path = solana_ledger::get_tmp_ledger_path!();
    let blockstore = Arc::new(Blockstore::open(&ledger_path).unwrap());

    let mut bank0 = Bank::new_for_tests(&genesis_config);
    bank0.set_retain_light_client_proof_data_for_tests(true);
    let bank0 = Arc::new(bank0);
    let program = create_program_with_elf(
        &bank0,
        &bpf_loader_upgradeable::id(),
        &Pubkey::default(),
        RECEIPT_TEST_PROGRAM,
    );
    goto_end_of_slot(bank0.clone());
    bank0.freeze();
    bank0.set_block_id(Some(BlockId::from(Hash::new_unique())));

    let quiet = Pubkey::new_unique();
    let bank1 = new_child(&bank0, 1);
    transfer(&bank1, &mint, &quiet, 1_000_000);
    freeze_and_store(&blockstore, &bank1);
    let bank62 = new_child(&bank1, 62);
    transfer(&bank62, &mint, &Pubkey::new_unique(), 1_000_000);
    freeze_and_store(&blockstore, &bank62);
    assert_eq!(bank62.commitment_root(), None);

    // The features activate in bank 65, the first block of epoch 1.
    let bank65 = new_child(&bank62, 65);
    assert!(bank65.is_accounts_state_root_active());
    let recipient = Pubkey::new_unique();
    let transfer_tx = transfer(&bank65, &mint, &recipient, 1_000_000_000);
    let instruction = Instruction::new_with_bytes(
        program,
        &[0, 0xaa, 0xbb],
        vec![
            AccountMeta::new(mint.pubkey(), true),
            AccountMeta::new(recipient, false),
            AccountMeta::new_readonly(system_program::id(), false),
            AccountMeta::new_readonly(program, false),
        ],
    );
    let program_tx = Transaction::new_signed_with_payer(
        &[instruction],
        Some(&mint.pubkey()),
        &[&mint],
        bank65.last_blockhash(),
    );
    bank65.process_transaction(&program_tx).unwrap();
    freeze_and_store(&blockstore, &bank65);

    let bank66 = new_child(&bank65, 66);
    freeze_and_store(&blockstore, &bank66);
    let bank96 = new_child(&bank66, 96);
    let fresh = Pubkey::new_unique();
    transfer(&bank96, &mint, &fresh, 2_000_000);
    freeze_and_store(&blockstore, &bank96);
    assert_eq!(bank96.accounts_state_generation_in_use(), Some(1));
    let bank100 = new_child(&bank96, 100);
    freeze_and_store(&blockstore, &bank100);
    let bank130 = new_child(&bank100, 130);
    transfer(&bank130, &mint, &Pubkey::new_unique(), 1_000_000);
    freeze_and_store(&blockstore, &bank130);
    let bank160 = new_child(&bank130, 160);
    transfer(&bank160, &mint, &fresh, 3_000_000);
    freeze_and_store(&blockstore, &bank160);
    assert_eq!(bank160.accounts_state_generation_in_use(), Some(2));
    // The base trees of the account state proofs are ready (the RPC does
    // not wait for them).
    bank96.accounts_state_proof(&mint.pubkey()).unwrap();
    bank160.accounts_state_proof(&mint.pubkey()).unwrap();

    let banks: std::collections::BTreeMap<_, _> = [
        &bank0, &bank1, &bank62, &bank65, &bank66, &bank96, &bank100, &bank130, &bank160,
    ]
    .into_iter()
    .map(|bank| (bank.slot(), bank.clone()))
    .collect();
    blockstore.set_roots(banks.keys()).unwrap();

    Chain {
        genesis_config,
        validators,
        mint,
        blockstore,
        banks,
        recipient,
        quiet,
        fresh,
        program,
        program_tx,
        transfer_tx,
    }
}

#[test]
fn test_light_client_proofs_over_rpc() {
    let chain = build_chain();
    let store = chain.store(1000);
    let rpc = chain.rpc(&store);
    let bank = |slot| chain.bank(slot);

    // Before any certificate: no finality proof.
    let (code, data) = rpc.error("getFinalityProof", json!([100]));
    assert_eq!(
        code,
        JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_CERTIFICATES_NOT_AVAILABLE
    );
    assert_eq!(
        data,
        json!({"kind": "certificatesNotAvailable", "slot": 100})
    );

    // The consensus pool completes certificates: slow finality of 100, fast
    // finality of 96 (epoch 1) and 160 (epoch 2), and a notarization of a
    // block that lost (not used).
    let signers1 = bank(65).validator_set_signers(1).unwrap();
    let set1 = SignerSet::new(1, signers1).unwrap();
    let keys1 = keypairs_by_rank(&chain.validators, &set1);
    let set2 = SignerSet::new(2, bank(100).validator_set_signers(2).unwrap()).unwrap();
    let keys2 = keypairs_by_rank(&chain.validators, &set2);
    for certificate in slow_certificates(bank(100), &keys1) {
        store.on_certificate(&certificate);
    }
    store.on_certificate(&fast_certificate(bank(96), &keys1));
    store.on_certificate(&fast_certificate(bank(160), &keys2));
    store.on_certificate(&certificate(
        &keys1,
        CertificateType::Notarize(Block {
            slot: 66,
            block_id: BlockId::new_unique(),
        }),
        &[0, 1],
    ));

    let schedule = chain.genesis_config.epoch_schedule.clone();
    let mut client = LightClient::new(schedule.clone(), SHRED_VERSION, set1.clone());

    // getFinalityProof: slow (100), fast (96), through a descendant (65, 66).
    let finality100 = rpc.finality(100);
    assert!(matches!(finality100.finality, Finality::Certificates(_)));
    let v100 = finality100.verify(&client, 100).unwrap();
    assert_eq!(
        *v100.commitment_root(),
        bank(100).commitment_root().unwrap()
    );
    let VerifiedFinal::Block(v100_block) = v100 else {
        panic!("a directly finalized block");
    };
    assert_eq!(*v100_block.bank_hash(), bank(100).hash());
    assert_eq!(v100_block.parent().block_id, block_id(bank(96)));
    let v96 = rpc.finality(96).verify(&client, 96).unwrap();
    let finality65 = rpc.finality(65);
    let Finality::Descendant(descendant) = &finality65.finality else {
        panic!("65 has no certificates of its own");
    };
    assert_eq!(descendant.slot, 96);
    let v65 = finality65.verify(&client, 65).unwrap();
    assert!(matches!(v65, VerifiedFinal::Ancestor(_)));
    assert_eq!(*v65.block_id(), block_id(bank(65)));
    assert_eq!(*v65.commitment_root(), bank(65).commitment_root().unwrap());
    // The footer of 65 names 62 as its parent.
    assert_eq!(finality65.footer.parent_slot, 62);
    assert_eq!(finality65.footer.parent_block_id, block_id(bank(62)));
    // A response is checked against the slot asked for.
    assert_eq!(finality65.verify(&client, 66), Err(VerifyError::WrongBlock));
    let mut tampered = finality100.clone();
    tampered.footer.bank_hash = Hash::new_unique();
    assert_eq!(
        tampered.verify(&client, 100),
        Err(VerifyError::FooterProofMismatch)
    );

    // getBlockHistoryProof: an ancestor, and the anchor (62).
    let response: RpcBlockHistoryProof =
        rpc.call("getBlockHistoryProof", json!([66, 100])).unwrap();
    let history = BlockHistoryResponse::try_from(&response).unwrap();
    assert_eq!(
        (history.at_slot, history.at_block_id),
        (100, block_id(bank(100)))
    );
    let a66 = client.verify_ancestor(&v100, &history.history).unwrap();
    assert_eq!(*a66.commitment_root(), bank(66).commitment_root().unwrap());
    let response: RpcBlockHistoryProof =
        rpc.call("getBlockHistoryProof", json!([62, 100])).unwrap();
    let anchor = BlockHistoryResponse::try_from(&response).unwrap();
    assert_eq!(
        client.verify_ancestor(&v100, &anchor.history),
        Err(VerifyError::AnchorLeaf)
    );
    let anchor = anchor.history.verify_anchor(&v100).unwrap();
    assert_eq!((anchor.slot, anchor.block_id), (62, block_id(bank(62))));
    let (code, _) = rpc.error("getBlockHistoryProof", json!([100, 96]));
    assert_eq!(code, JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_NOT_FOUND);

    // getTransactionProof: the program transaction in 65, with all its
    // instructions and its second event; the transfer before it.
    let signature = chain.program_tx.signatures[0];
    let signature_bytes: [u8; 64] = signature.into();
    let response: RpcTransactionProof = rpc
        .call(
            "getTransactionProof",
            json!([signature.to_string(), {"instructions": "all", "events": {"indices": [1]}}]),
        )
        .unwrap();
    let transaction = TransactionResponse::try_from(&response).unwrap();
    assert_eq!(
        (transaction.slot, transaction.block_id),
        (65, block_id(bank(65)))
    );
    let verified = transaction
        .transaction
        .verify(&v65, &signature_bytes)
        .unwrap();
    assert!(verified.is_success());
    assert_eq!(verified.fee(), LAMPORTS_PER_SIGNATURE);
    assert_eq!(verified.index, 1);
    assert_eq!(transaction.instructions.len(), 3);
    let cpi = transaction.instructions[2]
        .verify(&v65, &signature_bytes)
        .unwrap();
    let mut transfer_data = vec![2, 0, 0, 0];
    transfer_data.extend_from_slice(&1u64.to_le_bytes());
    assert!(cpi.matches(
        3,
        &system_program::id(),
        &[
            meta(&chain.mint.pubkey(), true, true),
            meta(&chain.recipient, false, true)
        ],
        &transfer_data
    ));
    let top_level = transaction.instructions[0]
        .verify(&v65, &signature_bytes)
        .unwrap();
    assert_eq!(top_level.record.program_id, chain.program.to_bytes());
    assert_eq!(transaction.events.len(), 1);
    let event = transaction.events[0]
        .verify(&v65, &signature_bytes)
        .unwrap();
    assert_eq!(event.index, 1);
    assert_eq!(event.event.kind, EventKind::LogData);
    assert_eq!(
        event.event.payload,
        log_data_payload(&[b"inner".as_slice(), b""])
    );
    assert_eq!(event.instruction.record.stack_height, 2);
    // Not in block 100.
    assert!(
        transaction
            .transaction
            .verify(&v100, &signature_bytes)
            .is_err()
    );
    // Only the event: its instruction comes with it.
    let response: RpcTransactionProof = rpc
        .call(
            "getTransactionProof",
            json!([signature.to_string(), {"events": "all"}]),
        )
        .unwrap();
    let transaction = TransactionResponse::try_from(&response).unwrap();
    assert_eq!(transaction.events.len(), 3);
    assert_eq!(
        transaction
            .instructions
            .iter()
            .map(|instruction| instruction.proof.leaf_index)
            .collect::<Vec<_>>(),
        [0, 1]
    );
    for event in &transaction.events {
        event.verify(&v65, &signature_bytes).unwrap();
    }
    let transfer_signature = chain.transfer_tx.signatures[0];
    let response: RpcTransactionProof = rpc
        .call(
            "getTransactionProof",
            json!([transfer_signature.to_string()]),
        )
        .unwrap();
    let transfer = TransactionResponse::try_from(&response).unwrap();
    let verified = transfer
        .transaction
        .verify(&v65, &transfer_signature.into())
        .unwrap();
    assert_eq!(verified.index, 0);
    assert!(transfer.instructions.is_empty());
    let (code, _) = rpc.error(
        "getTransactionProof",
        json!([signature.to_string(), {"instructions": {"indices": [3]}}]),
    );
    assert_eq!(code, JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_NOT_FOUND);
    let (code, _) = rpc.error(
        "getTransactionProof",
        json!([solana_signature::Signature::from([7u8; 64]).to_string()]),
    );
    assert_eq!(code, JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_NOT_FOUND);

    // getAccountDeltaProof: changed in 65, unchanged in 65.
    let response: RpcAccountDeltaProof = rpc
        .call(
            "getAccountDeltaProof",
            json!([chain.recipient.to_string(), 65]),
        )
        .unwrap();
    let delta = AccountDeltaResponse::try_from(&response).unwrap();
    let AccountDelta::Changed(changed) = &delta.delta else {
        panic!("65 changed the recipient");
    };
    let recipient_hash = hash_of(
        &bank(65).get_account(&chain.recipient).unwrap(),
        &chain.recipient,
    );
    changed
        .verify(&v65, &chain.recipient, &recipient_hash)
        .unwrap();
    let response: RpcAccountDeltaProof = rpc
        .call("getAccountDeltaProof", json!([chain.quiet.to_string(), 65]))
        .unwrap();
    let AccountDelta::Unchanged(unchanged) =
        AccountDeltaResponse::try_from(&response).unwrap().delta
    else {
        panic!("65 did not change `quiet`");
    };
    unchanged.verify(&v65, &chain.quiet).unwrap();
    assert!(unchanged.verify(&v65, &chain.recipient).is_err());

    // getAccountStateProof at 96 (generation 1): an account changed since
    // the base, an account unchanged since the base, no account.
    for address in [chain.mint.pubkey(), chain.quiet, chain.fresh] {
        let response: RpcAccountStateProof = rpc
            .call(
                "getAccountStateProof",
                json!([address.to_string(), {"slot": 96}]),
            )
            .unwrap();
        let state = AccountStateResponse::try_from(&response).unwrap();
        let hash = hash_of(&bank(96).get_account(&address).unwrap(), &address);
        assert_eq!(state.proof.verify(&v96, &address, &hash), Ok(1));
    }
    let nobody = Pubkey::new_unique();
    let response: RpcAccountStateProof = rpc
        .call(
            "getAccountStateProof",
            json!([nobody.to_string(), {"slot": 96}]),
        )
        .unwrap();
    let state = AccountStateResponse::try_from(&response).unwrap();
    assert_eq!(state.proof.verify_nonexistent(&v96, &nobody), Ok(1));
    assert!(state.proof.verify_nonexistent(&v100, &nobody).is_err());

    // getEpochBoundary of epoch 1, from block 100.
    let response: RpcEpochBoundary = rpc
        .call("getEpochBoundary", json!([1, {"atSlot": 100}]))
        .unwrap();
    let boundary = EpochBoundaryResponse::try_from(&response).unwrap();
    assert_eq!(boundary.slot, 100);
    let boundary1 = boundary.proof.verify(&v100, &schedule).unwrap();
    assert_eq!(boundary1.boundary.first_slot, 65);
    assert_eq!(boundary1.boundary.parent_slot, 62);
    assert_eq!(boundary1.last_block_before(), None);
    let (code, _) = rpc.error("getEpochBoundary", json!([2, {"atSlot": 100}]));
    assert_eq!(code, JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_NOT_FOUND);

    // Epoch 2: block 160 cannot be verified with the signer set of epoch 1.
    let finality160 = rpc.finality(160);
    assert!(matches!(
        finality160.verify(&client, 160),
        Err(VerifyError::EpochMismatch { .. })
    ));
    // getValidatorSet of epoch 2, committed by block 100.
    let response: RpcValidatorSet = rpc.call("getValidatorSet", json!([2])).unwrap();
    let validator_set = ValidatorSetResponse::try_from(&response).unwrap();
    let committed_by = validator_set.committed_by.clone().unwrap();
    assert_eq!(
        (committed_by.slot, committed_by.block_id),
        (100, block_id(bank(100)))
    );
    let next = validator_set.verify(&v100, &schedule).unwrap().clone();
    assert_eq!(next.validator_set_root(), set2.validator_set_root());
    // Not committed by a block of another epoch.
    assert_eq!(
        validator_set.verify(&v96, &schedule).unwrap_err(),
        VerifyError::WrongBlock
    );
    client
        .advance_epoch(&v100, &committed_by.opening, next)
        .unwrap();
    assert_eq!(client.epoch(), 2);
    let (code, _) = rpc.error("getValidatorSet", json!([2, {"atSlot": 160}]));
    assert_eq!(code, JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_NOT_FOUND);

    let v160 = finality160.verify(&client, 160).unwrap();
    // Block 130 is final through 160.
    let v130 = rpc.finality(130).verify(&client, 130).unwrap();
    assert_eq!(
        *v130.commitment_root(),
        bank(130).commitment_root().unwrap()
    );
    // getEpochBoundary of epoch 2, default block: the latest of epoch 2.
    let response: RpcEpochBoundary = rpc.call("getEpochBoundary", json!([2])).unwrap();
    let boundary = EpochBoundaryResponse::try_from(&response).unwrap();
    assert_eq!(boundary.slot, 160);
    let boundary2 = boundary.proof.verify(&v160, &schedule).unwrap();
    let last = boundary2.last_block_before().unwrap();
    assert_eq!((last.slot(), *last.block_id()), (100, block_id(bank(100))));
    assert_eq!(
        boundary1.boundary.blocks_until(&boundary2.boundary),
        Some(4)
    );

    // getAccountStateProof, default block: the latest finalized (160).
    let response: RpcAccountStateProof = rpc
        .call("getAccountStateProof", json!([chain.fresh.to_string()]))
        .unwrap();
    let state = AccountStateResponse::try_from(&response).unwrap();
    assert_eq!(state.slot, 160);
    let hash = hash_of(&bank(160).get_account(&chain.fresh).unwrap(), &chain.fresh);
    assert_eq!(state.proof.verify(&v160, &chain.fresh, &hash), Ok(2));
}

#[test]
fn test_light_client_proof_errors() {
    let chain = build_chain();
    let store = chain.store(1000);
    let rpc = chain.rpc(&store);
    let some_pubkey = Pubkey::new_unique().to_string();

    // The feature is not active for block 62.
    let (code, data) = rpc.error("getFinalityProof", json!([62]));
    assert_eq!(code, JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_FEATURE_INACTIVE);
    assert_eq!(
        data,
        json!({"kind": "featureInactive", "slot": 62, "feature": "block_commitment_root"})
    );
    let (code, _) = rpc.error("getAccountDeltaProof", json!([some_pubkey, 62]));
    assert_eq!(code, JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_FEATURE_INACTIVE);

    // Unknown slots: a skipped slot below the root, a slot above it.
    for slot in [64, 1000] {
        let (code, data) = rpc.error("getFinalityProof", json!([slot]));
        assert_eq!(code, JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_FINAL);
        assert_eq!(
            data,
            json!({"kind": "blockNotFinal", "slot": slot, "finalizedSlot": 160})
        );
    }
    // A block that is frozen but not rooted.
    let bank161 = new_child(chain.bank(160), 161);
    freeze_and_store(&chain.blockstore, &bank161);
    for (method, params) in [
        ("getFinalityProof", json!([161])),
        ("getBlockHistoryProof", json!([160, 161])),
        ("getAccountDeltaProof", json!([some_pubkey, 161])),
        ("getAccountStateProof", json!([some_pubkey, {"slot": 161}])),
        ("getEpochBoundary", json!([2, {"atSlot": 161}])),
    ] {
        let (code, _) = rpc.error(method, params);
        assert_eq!(
            code, JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_FINAL,
            "{method}"
        );
    }
    // A root from before the store's first block.
    let (code, data) = rpc.error("getFinalityProof", json!([1]));
    assert_eq!(code, JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_RETAINED);
    assert_eq!(
        data,
        json!({
            "kind": "blockNotRetained",
            "slot": 1,
            "firstRetained": 62,
            "lastRetained": 160,
        })
    );

    // A store that keeps two blocks: older roots are not retained, and their
    // certificates are dropped.
    let small = chain.store(2);
    let small_rpc = chain.rpc(&small);
    let (code, data) = small_rpc.error("getAccountDeltaProof", json!([some_pubkey, 96]));
    assert_eq!(code, JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_RETAINED);
    assert_eq!(data["firstRetained"], json!(130));
    let (code, _) = small_rpc.error(
        "getTransactionProof",
        json!([chain.program_tx.signatures[0].to_string()]),
    );
    assert_eq!(code, JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_NOT_FOUND);
    // Its history still reaches back to 65: the leaves are kept longer.
    let response: RpcBlockHistoryProof = small_rpc
        .call("getBlockHistoryProof", json!([65, 160]))
        .unwrap();
    assert_eq!(response.history.slot, 65);

    // Proofs disabled: by configuration, or without a store.
    for rpc in [
        Rpc::new(
            Bank::new_for_tests(&chain.genesis_config),
            false,
            Some(store.clone()),
            None,
        ),
        Rpc::new(Bank::new_for_tests(&chain.genesis_config), true, None, None),
    ] {
        for (method, params) in [
            ("getFinalityProof", json!([100])),
            ("getBlockHistoryProof", json!([65, 100])),
            (
                "getTransactionProof",
                json!([chain.program_tx.signatures[0].to_string()]),
            ),
            ("getAccountDeltaProof", json!([some_pubkey, 65])),
            ("getAccountStateProof", json!([some_pubkey])),
            ("getValidatorSet", json!([2])),
            ("getEpochBoundary", json!([1])),
        ] {
            let (code, data) = rpc.error(method, params);
            assert_eq!(
                code, JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_PROOFS_DISABLED,
                "{method}"
            );
            assert_eq!(data, json!({"kind": "proofsDisabled"}));
        }
    }

    // Invalid parameters are JSON-RPC parameter errors.
    let (code, _) = rpc.error("getAccountDeltaProof", json!(["not a pubkey", 65]));
    assert_eq!(code, jsonrpc_core::ErrorCode::InvalidParams.code());
    let (code, _) = rpc.error(
        "getTransactionProof",
        json!([chain.program_tx.signatures[0].to_string(), {"instruction": "all"}]),
    );
    assert_eq!(code, jsonrpc_core::ErrorCode::InvalidParams.code());
}
