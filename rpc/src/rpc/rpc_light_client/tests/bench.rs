//! Measurements, not tests: run in release mode with
//!
//! ```text
//! cargo test --release -p solana-rpc --features agave-unstable-api,dev-context-only-utils \
//!     --lib bench_light_client -- --ignored --nocapture --test-threads 1
//! ```
//!
//! One block of `N` transfers (each from its own payer to a new account) at
//! slot 100, after the switch slot of epoch 1, on a chain built as in the
//! tests of the parent module, with `V` validators and `A` more accounts
//! in the genesis bank. Timed: executing and freezing the block with the six
//! features off, on, and on with the proof data retained (as on a node
//! that serves proofs); building the base tree of the accounts state;
//! taking the block into the proof store; building each proof; and
//! verifying each proof as a client. Sizes are those of the JSON answers.
//!
//! [`Mix::MainnetLike`] replaces the transfers by a block shaped like the
//! mainnet blocks sampled in `measurements/`: about 410 transactions of
//! which 36% fail, about 1,300 changed accounts, most of them accounts with
//! data that already existed, calls into an on-chain program that makes
//! nested calls and logs events, and an overlay that already holds the
//! accounts changed earlier in the epoch.

use {
    super::*,
    solana_light_client::rpc::AccountDelta,
    solana_rpc_client_api::light_client::{
        RpcAccountDeltaProof, RpcAccountStateProof, RpcBlockHistoryProof, RpcEpochBoundary,
        RpcFinalityProof, RpcTransactionProof, RpcValidatorSet,
    },
    std::{
        collections::HashMap,
        time::{Duration, Instant},
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// The six features are not active.
    Off,
    /// Active, as on a validator that serves no proofs.
    On,
    /// Active, and the bank keeps the receipts with their trace and events.
    Retain,
}

/// What the transactions of the measured block are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mix {
    /// Each a transfer from its own payer to a new account.
    Transfers,
    /// Of every 25 transactions: 9 fail (5 transfers of more than the payer
    /// has, 4 program calls that return an error), 5 call a program that
    /// calls itself, then the system program, and logs three events, 6 pay
    /// eight existing 165-byte accounts each, 3 create a 165-byte account,
    /// and 2 are transfers to new accounts.
    MainnetLike,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Transfer,
    FailingTransfer,
    FailingProgram,
    Program,
    MultiTransfer,
    CreateAccount,
}

impl Mix {
    fn kind(self, index: usize) -> Kind {
        match (self, index % 25) {
            (Mix::Transfers, _) => Kind::Transfer,
            (_, 0..=4) => Kind::FailingTransfer,
            (_, 5..=8) => Kind::FailingProgram,
            (_, 9..=13) => Kind::Program,
            (_, 14..=19) => Kind::MultiTransfer,
            (_, 20..=22) => Kind::CreateAccount,
            _ => Kind::Transfer,
        }
    }
}

/// Size of the data of the accounts the mix pays and creates: that of a
/// token account.
const DATA_LEN: usize = 165;
/// Accounts paid by one transaction that pays several.
const PAID_PER_TRANSACTION: usize = 8;

/// An address spread like a hash, unlike `Pubkey::new_unique`.
fn spread_pubkey() -> Pubkey {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let index = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Pubkey::from(solana_sha256_hasher::hashv(&[b"bench", &index.to_le_bytes()]).to_bytes())
}

struct Scenario {
    validators: usize,
    extra_accounts: usize,
    transfers: usize,
    mode: Mode,
    /// With a mode other than `Off`: the features that are active, or all
    /// six.
    only: Option<Vec<Pubkey>>,
    mix: Mix,
    /// Accounts changed earlier in the epoch: entries the overlay of the
    /// accounts state already holds when the measured block is frozen.
    overlay_entries: usize,
}

#[derive(Default)]
struct Measured {
    base_tree: Duration,
    execute: Duration,
    freeze: Duration,
    empty_freeze: Duration,
    proofs: Vec<(&'static str, Duration, usize, Duration)>,
    ingest: Duration,
    /// From the end of `ingest` until the block is durable on disk.
    durable: Duration,
    block_file_bytes: u64,
    changed_accounts: usize,
    failed: usize,
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

/// Median time of `f` over `runs` runs, and its last result.
fn timed<T>(runs: usize, mut f: impl FnMut() -> T) -> (Duration, T) {
    let mut samples = Vec::with_capacity(runs);
    let mut last = None;
    for _ in 0..runs {
        let start = Instant::now();
        let value = f();
        samples.push(start.elapsed());
        last = Some(value);
    }
    (median(samples), last.unwrap())
}

fn json_len(value: &impl serde::Serialize) -> usize {
    serde_json::to_vec(value).unwrap().len()
}

fn run(scenario: &Scenario) -> Measured {
    let Scenario {
        validators: validator_count,
        extra_accounts,
        transfers,
        mode,
        ref only,
        mix,
        overlay_entries,
    } = *scenario;
    let mut measured = Measured::default();

    let validators: Vec<_> = (0..validator_count)
        .map(|_| ValidatorVoteKeypairs::new_rand())
        .collect();
    let stakes = vec![minimum_vote_account_balance_for_vat(100); validator_count];
    let GenesisConfigInfo {
        mut genesis_config, ..
    } = create_genesis_config_with_vote_accounts(1_000_000_000_000, &validators, stakes);
    genesis_config.epoch_schedule = EpochSchedule::custom(SLOTS_PER_EPOCH, SLOTS_PER_EPOCH, false);
    genesis_config.fee_rate_governor = FeeRateGovernor::new(LAMPORTS_PER_SIGNATURE, 0);
    for feature_id in all_features() {
        let active = only.as_ref().is_none_or(|only| only.contains(&feature_id));
        if mode == Mode::Off || !active {
            genesis_config.accounts.remove(&feature_id);
        } else {
            solana_runtime::genesis_utils::activate_feature(&mut genesis_config, feature_id);
        }
    }
    let system_account = |lamports| solana_account::Account {
        lamports,
        owner: system_program::id(),
        ..solana_account::Account::default()
    };
    let payers: Vec<Keypair> = (0..transfers).map(|_| Keypair::new()).collect();
    for payer in &payers {
        genesis_config
            .accounts
            .insert(payer.pubkey(), system_account(10_000_000_000));
    }
    // One more than asked: `untouched` is proven from the base tree.
    let untouched = Pubkey::new_unique();
    genesis_config
        .accounts
        .insert(untouched, system_account(1_000_000_000));

    let ledger_path = solana_ledger::get_tmp_ledger_path!();
    let blockstore = Arc::new(Blockstore::open(&ledger_path).unwrap());
    let mut bank0 = Bank::new_for_tests(&genesis_config);
    bank0.set_retain_light_client_proof_data_for_tests(mode == Mode::Retain);
    let bank0 = Arc::new(bank0);
    // Stored in the genesis bank, not in the genesis config, whose size is
    // limited.
    let extra = AccountSharedData::from(system_account(1_000_000_000));
    for _ in 0..extra_accounts {
        bank0.store_account(&spread_pubkey(), &extra);
    }
    // For the mix: the program, and the accounts with data that are paid.
    let data_owner = Pubkey::new_unique();
    let data_account_lamports = genesis_config.rent.minimum_balance(DATA_LEN);
    let mut paid: Vec<Pubkey> = Vec::new();
    let mut program = Pubkey::default();
    if mix == Mix::MainnetLike {
        program = create_program_with_elf(
            &bank0,
            &bpf_loader_upgradeable::id(),
            &Pubkey::default(),
            RECEIPT_TEST_PROGRAM,
        );
        let data_account = AccountSharedData::from(solana_account::Account {
            lamports: data_account_lamports,
            data: vec![1; DATA_LEN],
            owner: data_owner,
            ..solana_account::Account::default()
        });
        paid = (0..transfers * PAID_PER_TRANSACTION)
            .map(|_| spread_pubkey())
            .collect();
        for address in &paid {
            bank0.store_account(address, &data_account);
        }
    }
    goto_end_of_slot(bank0.clone());
    bank0.freeze();
    bank0.set_block_id(Some(BlockId::from(Hash::new_unique())));
    let bank1 = new_child(&bank0, 1);
    freeze_and_store(&blockstore, &bank1);
    assert_eq!(bank1.is_block_commitment_root_active(), mode != Mode::Off);

    // The first block of epoch 1 starts the base tree of generation 1 in
    // the background; the block at the switch slot waits for it.
    let base_start = Instant::now();
    let bank64 = new_child(&bank1, 64);
    freeze_and_store(&blockstore, &bank64);
    let bank96 = new_child(&bank64, 96);
    for _ in 0..overlay_entries {
        bank96.store_account(&spread_pubkey(), &extra);
    }
    freeze_and_store(&blockstore, &bank96);
    measured.base_tree = base_start.elapsed();

    // The measured block.
    let bank100 = new_child(&bank96, 100);
    let blockhash = bank100.last_blockhash();
    let mut recipients: Vec<Pubkey> = Vec::with_capacity(transfers);
    let transactions: Vec<Transaction> = payers
        .iter()
        .enumerate()
        .map(|(index, payer)| {
            let paid = |offset: usize| paid[index * PAID_PER_TRANSACTION + offset];
            let program_call = |data: &[u8]| {
                let instruction = Instruction::new_with_bytes(
                    program,
                    data,
                    vec![
                        AccountMeta::new(payer.pubkey(), true),
                        AccountMeta::new(paid(0), false),
                        AccountMeta::new_readonly(system_program::id(), false),
                        AccountMeta::new_readonly(program, false),
                    ],
                );
                Transaction::new_signed_with_payer(
                    &[instruction],
                    Some(&payer.pubkey()),
                    &[payer],
                    blockhash,
                )
            };
            let (recipient, transaction) = match mix.kind(index) {
                Kind::Transfer => {
                    let recipient = Pubkey::new_unique();
                    let transaction =
                        system_transaction::transfer(payer, &recipient, 1_000_000_000, blockhash);
                    (recipient, transaction)
                }
                // More than the payer has.
                Kind::FailingTransfer => {
                    let recipient = Pubkey::new_unique();
                    let transaction = system_transaction::transfer(
                        payer,
                        &recipient,
                        1_000_000_000_000,
                        blockhash,
                    );
                    (recipient, transaction)
                }
                Kind::Program => (paid(0), program_call(&[0, 0xaa, 0xbb])),
                Kind::FailingProgram => (paid(0), program_call(&[2])),
                Kind::MultiTransfer => {
                    let instructions: Vec<_> = (0..PAID_PER_TRANSACTION)
                        .map(|offset| {
                            solana_system_interface::instruction::transfer(
                                &payer.pubkey(),
                                &paid(offset),
                                1_000_000,
                            )
                        })
                        .collect();
                    let transaction = Transaction::new_signed_with_payer(
                        &instructions,
                        Some(&payer.pubkey()),
                        &[payer],
                        blockhash,
                    );
                    (paid(0), transaction)
                }
                Kind::CreateAccount => {
                    let created = Keypair::new();
                    let instruction = solana_system_interface::instruction::create_account(
                        &payer.pubkey(),
                        &created.pubkey(),
                        data_account_lamports,
                        DATA_LEN as u64,
                        &data_owner,
                    );
                    let transaction = Transaction::new_signed_with_payer(
                        &[instruction],
                        Some(&payer.pubkey()),
                        &[payer, &created],
                        blockhash,
                    );
                    (created.pubkey(), transaction)
                }
            };
            recipients.push(recipient);
            transaction
        })
        .collect();
    let start = Instant::now();
    let results = bank100.process_transactions(transactions.iter());
    measured.execute = start.elapsed();
    let expected_failures = (0..transfers)
        .filter(|index| {
            matches!(
                mix.kind(*index),
                Kind::FailingTransfer | Kind::FailingProgram
            )
        })
        .count();
    measured.failed = results.iter().filter(|result| result.is_err()).count();
    assert_eq!(measured.failed, expected_failures, "{results:?}");
    // Registers the remaining ticks (the same work in every mode) and
    // freezes the bank.
    let start = Instant::now();
    goto_end_of_slot(bank100.clone());
    measured.freeze = start.elapsed();
    store_frozen(&blockstore, &bank100);
    measured.changed_accounts = bank100.accounts_delta().map_or(0, |delta| delta.len());
    assert_eq!(bank100.commitment_failure(), None);
    assert_eq!(bank100.commitment_root().is_some(), mode != Mode::Off);

    let bank101 = new_child(&bank100, 101);
    let start = Instant::now();
    goto_end_of_slot(bank101.clone());
    measured.empty_freeze = start.elapsed();
    store_frozen(&blockstore, &bank101);
    if mode != Mode::Retain {
        return measured;
    }

    // The proof store, writing to disk as on a node.
    let directory = solana_ledger::get_tmp_ledger_path!();
    let banks = [&bank1, &bank64, &bank96, &bank100, &bank101];
    blockstore
        .set_roots(
            banks
                .iter()
                .map(|bank| bank.slot())
                .collect::<Vec<_>>()
                .iter(),
        )
        .unwrap();
    let store = LightClientProofStore::new(
        LightClientProofConfig {
            persist_dir: Some(directory.clone()),
            ..LightClientProofConfig::default()
        },
        blockstore.clone(),
    );
    store.on_rooted_banks(&[bank1.clone(), bank64.clone(), bank96.clone()]);
    // The files of the earlier blocks are not part of what is timed.
    store.flush_writes();
    let start = Instant::now();
    store.on_rooted_banks(std::slice::from_ref(&bank100));
    measured.ingest = start.elapsed();
    let start = Instant::now();
    store.flush_writes();
    measured.durable = start.elapsed();
    measured.block_file_bytes = std::fs::metadata(directory.join("blocks").join("100"))
        .unwrap()
        .len();
    store.on_rooted_banks(std::slice::from_ref(&bank101));

    let set1 = SignerSet::new(1, bank64.validator_set_signers(1).unwrap()).unwrap();
    let by_key: HashMap<_, _> = validators
        .iter()
        .map(|v| (v.bls_keypair.public.to_bytes_compressed(), &v.bls_keypair))
        .collect();
    let keys1: Vec<BlsKeypair> = set1
        .signers()
        .iter()
        .map(|signer| by_key[&signer.bls_pubkey_compressed].clone())
        .collect();
    store.on_certificate(&fast_certificate(&bank100, &keys1));
    store.on_certificate(&fast_certificate(&bank101, &keys1));
    let schedule = genesis_config.epoch_schedule.clone();
    let client = LightClient::new(schedule.clone(), SHRED_VERSION, set1);

    // Each proof: time to build it and encode it as JSON, the size of the
    // JSON, and the time to decode and verify it.
    const RUNS: usize = 9;
    let mut record = |name, build: Duration, bytes: usize, verify: Duration| {
        measured.proofs.push((name, build, bytes, verify));
    };

    let (build, json) = timed(RUNS, || {
        RpcFinalityProof::from(&store.finality_proof(100).unwrap())
    });
    let (verify, block) = timed(RUNS, || {
        FinalityResponse::try_from(&json)
            .unwrap()
            .verify(&client, 100)
            .unwrap()
    });
    record(
        "finality (fast certificate)",
        build,
        json_len(&json),
        verify,
    );

    if transfers > 0 {
        // A transfer that succeeds, near the middle of the block.
        let middle = (transfers / 2..transfers)
            .find(|index| mix.kind(*index) == Kind::Transfer)
            .expect("a transfer in the second half of the block");
        let signature = transactions[middle].signatures[0];
        let signature_bytes: [u8; 64] = signature.into();
        let (build, json) = timed(RUNS, || {
            RpcTransactionProof::from(&store.transaction_proof(&signature, None, None).unwrap())
        });
        let (verify, _) = timed(RUNS, || {
            TransactionResponse::try_from(&json)
                .unwrap()
                .transaction
                .verify(&block, &signature_bytes)
                .unwrap()
        });
        record("transaction receipt", build, json_len(&json), verify);

        let all = RpcProofSelection::All;
        let (build, json) = timed(RUNS, || {
            RpcTransactionProof::from(
                &store
                    .transaction_proof(&signature, Some(&all), Some(&all))
                    .unwrap(),
            )
        });
        let (verify, _) = timed(RUNS, || {
            let response = TransactionResponse::try_from(&json).unwrap();
            for proof in &response.instructions {
                proof.verify(&block, &signature_bytes).unwrap();
            }
        });
        record(
            "transaction + its instruction",
            build,
            json_len(&json),
            verify,
        );

        let recipient = recipients[middle];
        let recipient_hash = hash_of(&bank100.get_account(&recipient).unwrap(), &recipient);
        let (build, json) = timed(RUNS, || {
            RpcAccountDeltaProof::from(&store.account_delta_proof(&recipient, 100).unwrap())
        });
        let (verify, _) = timed(RUNS, || {
            match AccountDeltaResponse::try_from(&json).unwrap().delta {
                AccountDelta::Changed(proof) => {
                    proof.verify(&block, &recipient, &recipient_hash).unwrap()
                }
                AccountDelta::Unchanged(_) => panic!("unchanged"),
            }
        });
        record("account changed in block", build, json_len(&json), verify);

        let (build, json) = timed(RUNS, || {
            RpcAccountStateProof::from(&store.account_state_proof(&recipient, Some(100)).unwrap())
        });
        let (verify, _) = timed(RUNS, || {
            AccountStateResponse::try_from(&json)
                .unwrap()
                .proof
                .verify(&block, &recipient, &recipient_hash)
                .unwrap()
        });
        record(
            "account state (changed this epoch)",
            build,
            json_len(&json),
            verify,
        );
    }

    let (build, json) = timed(RUNS, || {
        RpcAccountDeltaProof::from(&store.account_delta_proof(&untouched, 100).unwrap())
    });
    let (verify, _) = timed(RUNS, || {
        match AccountDeltaResponse::try_from(&json).unwrap().delta {
            AccountDelta::Unchanged(proof) => proof.verify(&block, &untouched).unwrap(),
            AccountDelta::Changed(_) => panic!("changed"),
        }
    });
    record("account unchanged in block", build, json_len(&json), verify);

    let untouched_hash = hash_of(&bank100.get_account(&untouched).unwrap(), &untouched);
    let (build, json) = timed(RUNS, || {
        RpcAccountStateProof::from(&store.account_state_proof(&untouched, Some(100)).unwrap())
    });
    let (verify, _) = timed(RUNS, || {
        AccountStateResponse::try_from(&json)
            .unwrap()
            .proof
            .verify(&block, &untouched, &untouched_hash)
            .unwrap()
    });
    record(
        "account state (from the base tree)",
        build,
        json_len(&json),
        verify,
    );

    let nobody = Pubkey::new_unique();
    let (build, json) = timed(RUNS, || {
        RpcAccountStateProof::from(&store.account_state_proof(&nobody, Some(100)).unwrap())
    });
    let (verify, _) = timed(RUNS, || {
        AccountStateResponse::try_from(&json)
            .unwrap()
            .proof
            .verify_nonexistent(&block, &nobody)
            .unwrap()
    });
    record("account does not exist", build, json_len(&json), verify);

    let (build, json) = timed(RUNS, || {
        RpcBlockHistoryProof::from(&store.block_history_proof(64, 100).unwrap())
    });
    let (verify, _) = timed(RUNS, || {
        client
            .verify_ancestor(
                &block,
                &BlockHistoryResponse::try_from(&json).unwrap().history,
            )
            .unwrap()
    });
    record(
        "ancestor (history of 3 blocks)",
        build,
        json_len(&json),
        verify,
    );

    let (build, json) = timed(RUNS, || {
        RpcEpochBoundary::from(&store.epoch_boundary(1, Some(100)).unwrap())
    });
    let (verify, _) = timed(RUNS, || {
        EpochBoundaryResponse::try_from(&json)
            .unwrap()
            .proof
            .verify(&block, &schedule)
            .unwrap()
    });
    record("epoch boundary", build, json_len(&json), verify);

    let (build, json) = timed(RUNS, || {
        RpcValidatorSet::from(&store.validator_set(2, Some(100)).unwrap())
    });
    let (verify, _) = timed(RUNS, || {
        ValidatorSetResponse::try_from(&json)
            .unwrap()
            .verify(&block, &schedule)
            .map(|set| set.epoch())
            .unwrap()
    });
    record("next validator set", build, json_len(&json), verify);

    measured
}

fn ms(duration: Duration) -> String {
    format!("{:.3}", duration.as_secs_f64() * 1e3)
}

fn print_proofs(title: &str, measured: &Measured) {
    println!("\n{title}");
    println!("| proof | build ms | JSON bytes | verify ms |");
    println!("|---|---|---|---|");
    for (name, build, bytes, verify) in &measured.proofs {
        println!("| {name} | {} | {bytes} | {} |", ms(*build), ms(*verify));
    }
}

/// The cost of a block with the features off, on, and on with the proof
/// data retained, for blocks of 0, 1,000 and 5,000 transfers.
#[test]
#[ignore = "measurement; run in release mode"]
fn bench_light_client_block_cost() {
    println!(
        "\n| transfers | mode | execute ms | end of slot + freeze ms | same, empty block ms |"
    );
    println!("|---|---|---|---|---|");
    for transfers in [0, 1_000, 5_000] {
        for mode in [Mode::Off, Mode::On, Mode::Retain] {
            // Median of five chains, each built from scratch.
            let runs: Vec<Measured> = (0..5)
                .map(|_| {
                    run(&Scenario {
                        validators: 3,
                        extra_accounts: 0,
                        transfers,
                        mode,
                        only: None,
                        mix: Mix::Transfers,
                        overlay_entries: 0,
                    })
                })
                .collect();
            let pick = |f: fn(&Measured) -> Duration| median(runs.iter().map(f).collect());
            println!(
                "| {transfers} | {mode:?} | {} | {} | {} |",
                ms(pick(|m| m.execute)),
                ms(pick(|m| m.freeze)),
                ms(pick(|m| m.empty_freeze)),
            );
        }
    }
}

/// Proofs of a block, by the number of transfers in it.
#[test]
#[ignore = "measurement; run in release mode"]
fn bench_light_client_proofs_by_block_size() {
    for transfers in [100, 1_000, 5_000] {
        let measured = run(&Scenario {
            validators: 3,
            extra_accounts: 0,
            transfers,
            mode: Mode::Retain,
            only: None,
            mix: Mix::Transfers,
            overlay_entries: 0,
        });
        print_proofs(
            &format!(
                "{transfers} transfers, 3 validators: taking the block into the store {} ms \
                 on the thread that roots, durable {} ms later, {} bytes on disk",
                ms(measured.ingest),
                ms(measured.durable),
                measured.block_file_bytes
            ),
            &measured,
        );
    }
}

/// The base tree and account state proofs, by the number of accounts.
#[test]
#[ignore = "measurement; run in release mode"]
fn bench_light_client_accounts() {
    for extra_accounts in [10_000, 100_000, 1_000_000] {
        let measured = run(&Scenario {
            validators: 3,
            extra_accounts,
            transfers: 100,
            mode: Mode::Retain,
            only: None,
            mix: Mix::Transfers,
            overlay_entries: 0,
        });
        print_proofs(
            &format!(
                "{extra_accounts} accounts: base tree ready {} ms after the epoch started \
                 (includes freezing two empty blocks)",
                ms(measured.base_tree)
            ),
            &measured,
        );
    }
}

/// Finality and validator set proofs, by the number of validators.
#[test]
#[ignore = "measurement; run in release mode"]
fn bench_light_client_validators() {
    for validators in [100, 1_000, 2_000] {
        let measured = run(&Scenario {
            validators,
            extra_accounts: 0,
            transfers: 100,
            mode: Mode::Retain,
            only: None,
            mix: Mix::Transfers,
            overlay_entries: 0,
        });
        print_proofs(&format!("{validators} validators"), &measured);
    }
}

/// The cost of a block of 5,000 transfers with each feature added to the
/// base feature alone, to split the total between the proposals.
#[test]
#[ignore = "measurement; run in release mode"]
fn bench_light_client_cost_by_feature() {
    use feature_set::{
        accounts_delta_root, accounts_state_root, block_commitment_root, epoch_boundary_commitment,
        epoch_validator_set_root, transaction_receipt_tree,
    };
    let base = block_commitment_root::id();
    let cases: [(&str, Mode, Vec<Pubkey>); 8] = [
        ("none", Mode::Off, vec![]),
        ("1 block commitment root", Mode::On, vec![base]),
        (
            "1 + 2 receipts",
            Mode::On,
            vec![base, transaction_receipt_tree::id()],
        ),
        (
            "1 + 3 accounts delta",
            Mode::On,
            vec![base, accounts_delta_root::id()],
        ),
        (
            "1 + 4 validator set",
            Mode::On,
            vec![base, epoch_validator_set_root::id()],
        ),
        (
            "1 + 5 epoch boundary",
            Mode::On,
            vec![base, epoch_boundary_commitment::id()],
        ),
        (
            "1 + 3 + 6 accounts state",
            Mode::On,
            vec![base, accounts_delta_root::id(), accounts_state_root::id()],
        ),
        ("all six", Mode::On, all_features().to_vec()),
    ];
    println!("\n| features | execute ms | end of slot + freeze ms | same, empty block ms |");
    println!("|---|---|---|---|");
    for (name, mode, only) in cases {
        let runs: Vec<Measured> = (0..7)
            .map(|_| {
                run(&Scenario {
                    validators: 3,
                    extra_accounts: 0,
                    transfers: 5_000,
                    mode,
                    only: Some(only.clone()),
                    mix: Mix::Transfers,
                    overlay_entries: 0,
                })
            })
            .collect();
        let pick = |f: fn(&Measured) -> Duration| median(runs.iter().map(f).collect());
        println!(
            "| {name} | {} | {} | {} |",
            ms(pick(|m| m.execute)),
            ms(pick(|m| m.freeze)),
            ms(pick(|m| m.empty_freeze)),
        );
    }
}

/// A block shaped like a mainnet block (see [`Mix::MainnetLike`]): its cost
/// with the features off, on, and on with the proof data kept, then its
/// proofs. 100,000 more accounts exist, and the overlay holds a million
/// accounts changed earlier in the epoch.
#[test]
#[ignore = "measurement; run in release mode"]
fn bench_light_client_mainnet_like_block() {
    use feature_set::accounts_state_root;
    let scenario = |mode, only: Option<Vec<Pubkey>>, overlay_entries| Scenario {
        validators: 3,
        extra_accounts: 100_000,
        transfers: 410,
        mode,
        only,
        mix: Mix::MainnetLike,
        overlay_entries,
    };
    let without_state: Vec<Pubkey> = all_features()
        .into_iter()
        .filter(|feature| *feature != accounts_state_root::id())
        .collect();
    let cases = [
        ("off", Mode::Off, None, 1_000_000),
        ("drafts 1 to 5", Mode::On, Some(without_state), 1_000_000),
        ("all six, empty overlay", Mode::On, None, 0),
        ("all six", Mode::On, None, 1_000_000),
        ("all six, proof data kept", Mode::Retain, None, 1_000_000),
    ];
    println!(
        "\n| features | transactions | failed | changed accounts | execute ms | end of slot + \
         freeze ms | same, empty block ms |"
    );
    println!("|---|---|---|---|---|---|---|");
    let mut retained = None;
    for (name, mode, only, overlay_entries) in cases {
        let runs: Vec<Measured> = (0..5)
            .map(|_| run(&scenario(mode, only.clone(), overlay_entries)))
            .collect();
        let pick = |f: fn(&Measured) -> Duration| median(runs.iter().map(f).collect());
        println!(
            "| {name} | 410 | {} | {} | {} | {} | {} |",
            runs[0].failed,
            runs[0].changed_accounts,
            ms(pick(|m| m.execute)),
            ms(pick(|m| m.freeze)),
            ms(pick(|m| m.empty_freeze)),
        );
        if mode == Mode::Retain {
            retained = runs.into_iter().next();
        }
    }
    let measured = retained.expect("the last mode");
    print_proofs(
        &format!(
            "mainnet-like block: taking the block into the store {} ms on the thread that \
             roots, durable {} ms later, {} bytes on disk",
            ms(measured.ingest),
            ms(measured.durable),
            measured.block_file_bytes
        ),
        &measured,
    );
}
