//! Per-transaction part of the "Transaction Receipt Tree" SIMD: the
//! `instruction_trace_root` and `events_root` of a successful transaction.
//!
//! The rest of the receipt (signature, message hash, fee, status) and the
//! accumulation of receipts in block order are done by the bank at commit.
//!
//! # Execution order
//!
//! The [`TransactionContext`] instruction trace is not in execution order:
//! all top-level instructions are configured up front at indices
//! `0..num_top_level`, and every CPI is appended after them when it is
//! pushed. The receipt commits to the instructions in execution order
//! (pre-order, depth first): a top-level instruction, everything it invoked,
//! then the next top-level instruction. Because CPIs are appended in push
//! order and a top-level instruction finishes before the next one starts, the
//! CPIs of top-level instruction `i` form a contiguous run of the CPI part of
//! the trace, in push order. [`execution_order`] recovers the order from the
//! caller index of every CPI.

use {
    solana_hash::Hash,
    solana_instruction_error::InstructionError,
    solana_merkle_tree::receipt_tree::{
        EventRecord, InstructionAccountMeta, InstructionRecord, events_root, instruction_trace_root,
    },
    solana_program_runtime::invoke_context::RecordedEvent,
    solana_transaction_context::transaction::TransactionContext,
};

/// Sub-roots of the receipt of a successful transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReceiptRoots {
    pub instruction_trace_root: Hash,
    pub events_root: Hash,
}

/// Indices in the [`TransactionContext`] trace, in execution order.
fn execution_order(
    transaction_context: &TransactionContext,
) -> Result<Vec<usize>, InstructionError> {
    let trace_len = transaction_context.get_instruction_trace_length();
    let num_cpis = transaction_context.number_of_cpis_in_trace();
    let num_top_level = trace_len
        .checked_sub(num_cpis)
        .ok_or(InstructionError::CallDepth)?;

    // Top-level ancestor of every CPI, indexed by `index_in_trace -
    // num_top_level`. A caller always precedes its callees in the trace.
    let mut root_of_cpi = Vec::with_capacity(num_cpis);
    for index_in_trace in num_top_level..trace_len {
        let caller = transaction_context
            .get_instruction_context_at_index_in_trace(index_in_trace)?
            .get_index_of_caller();
        let root = if caller < num_top_level {
            caller
        } else if caller < index_in_trace {
            root_of_cpi[caller - num_top_level]
        } else {
            return Err(InstructionError::CallDepth);
        };
        root_of_cpi.push(root);
    }

    let mut order = Vec::with_capacity(trace_len);
    let mut next_cpi = 0;
    for top_level in 0..num_top_level {
        order.push(top_level);
        while next_cpi < num_cpis && root_of_cpi[next_cpi] == top_level {
            order.push(num_top_level + next_cpi);
            next_cpi += 1;
        }
    }
    if next_cpi != num_cpis {
        // The CPIs are not grouped by top-level instruction in push order,
        // which the runtime guarantees.
        return Err(InstructionError::CallDepth);
    }
    Ok(order)
}

/// The instruction trace of a transaction in execution order, and, for every
/// index in the [`TransactionContext`] trace, the index of its record.
pub fn instruction_trace_records(
    transaction_context: &TransactionContext,
) -> Result<(Vec<InstructionRecord>, Vec<u32>), InstructionError> {
    let order = execution_order(transaction_context)?;
    let mut record_index_of_trace_index = vec![u32::MAX; order.len()];
    let mut records = Vec::with_capacity(order.len());
    for (record_index, index_in_trace) in order.into_iter().enumerate() {
        let instruction_context =
            transaction_context.get_instruction_context_at_index_in_trace(index_in_trace)?;
        let stack_height = u8::try_from(instruction_context.get_stack_height())
            .map_err(|_| InstructionError::CallDepth)?;
        let program_id = instruction_context.get_program_key()?.to_bytes();
        let accounts = instruction_context
            .instruction_accounts()
            .iter()
            .map(|account| {
                transaction_context
                    .get_key_of_account_at_index(account.index_in_transaction)
                    .map(|pubkey| InstructionAccountMeta {
                        pubkey: pubkey.to_bytes(),
                        is_signer: account.is_signer(),
                        is_writable: account.is_writable(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        records.push(InstructionRecord::new(
            stack_height,
            program_id,
            accounts,
            instruction_context.get_instruction_data(),
        ));
        record_index_of_trace_index[index_in_trace] = record_index as u32;
    }
    Ok((records, record_index_of_trace_index))
}

/// The events of a transaction, with `instruction_index` pointing at the
/// record of the emitting instruction.
pub fn event_records(
    recorded_events: Vec<RecordedEvent>,
    record_index_of_trace_index: &[u32],
) -> Result<Vec<EventRecord>, InstructionError> {
    recorded_events
        .into_iter()
        .map(|event| {
            let instruction_index = record_index_of_trace_index
                .get(event.index_in_trace)
                .copied()
                .filter(|index| *index != u32::MAX)
                .ok_or(InstructionError::CallDepth)?;
            Ok(EventRecord {
                program_id: event.program_id.to_bytes(),
                instruction_index,
                kind: event.kind,
                payload: event.payload,
            })
        })
        .collect()
}

/// `instruction_trace_root` and `events_root` of a transaction that executed
/// successfully with `transaction_context` and emitted `recorded_events`.
pub fn compute_receipt_roots(
    transaction_context: &TransactionContext,
    recorded_events: Vec<RecordedEvent>,
) -> Result<ReceiptRoots, InstructionError> {
    let (records, record_index_of_trace_index) = instruction_trace_records(transaction_context)?;
    let events = event_records(recorded_events, &record_index_of_trace_index)?;
    Ok(ReceiptRoots {
        instruction_trace_root: instruction_trace_root(&records),
        events_root: events_root(&events),
    })
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        solana_account::AccountSharedData,
        solana_merkle_tree::receipt_tree::{EventKind, empty_events_root, log_data_payload},
        solana_pubkey::Pubkey,
        solana_rent::Rent,
        solana_transaction_context::instruction_accounts::InstructionAccount,
    };

    struct Fixture {
        keys: Vec<Pubkey>,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                keys: (0..5).map(|_| Pubkey::new_unique()).collect(),
            }
        }

        fn transaction_context(&self, num_top_level: usize) -> TransactionContext<'static> {
            TransactionContext::new(
                self.keys
                    .iter()
                    .map(|key| (*key, AccountSharedData::default()))
                    .collect(),
                Rent::default(),
                5,
                64,
                num_top_level,
            )
        }

        fn meta(&self, index: usize, is_signer: bool, is_writable: bool) -> InstructionAccountMeta {
            InstructionAccountMeta {
                pubkey: self.keys[index].to_bytes(),
                is_signer,
                is_writable,
            }
        }
    }

    fn dedup_map(num_accounts: usize, accounts: &[InstructionAccount]) -> Vec<u8> {
        let mut map = vec![u8::MAX; num_accounts];
        for (index_in_instruction, account) in accounts.iter().enumerate() {
            let entry = &mut map[account.index_in_transaction as usize];
            if *entry == u8::MAX {
                *entry = index_in_instruction as u8;
            }
        }
        map
    }

    /// Configures all top-level instructions up front, as the program
    /// runtime does: instruction `i` runs program account `programs[i].0`
    /// with accounts `programs[i].1` and data `[i]`.
    fn configure_top_level(
        transaction_context: &mut TransactionContext,
        instructions: Vec<(u16, Vec<InstructionAccount>)>,
    ) {
        let num_accounts = transaction_context.get_number_of_accounts() as usize;
        for (i, (program, accounts)) in instructions.into_iter().enumerate() {
            let map = dedup_map(num_accounts, &accounts);
            transaction_context
                .configure_instruction_at_index(
                    i,
                    program,
                    accounts,
                    map,
                    std::borrow::Cow::Owned(vec![i as u8]),
                    None,
                )
                .unwrap();
        }
    }

    /// Executes the next top-level instruction, running `body` while it is
    /// on the stack.
    fn run_top_level(
        transaction_context: &mut TransactionContext,
        body: impl FnOnce(&mut TransactionContext),
    ) {
        transaction_context.push(true).unwrap();
        body(transaction_context);
        transaction_context.pop().unwrap();
    }

    fn run_cpi(
        transaction_context: &mut TransactionContext,
        program: u16,
        accounts: Vec<InstructionAccount>,
        data: Vec<u8>,
        body: impl FnOnce(&mut TransactionContext),
    ) {
        transaction_context
            .configure_next_cpi_for_tests(program, accounts, data)
            .unwrap();
        transaction_context.push(true).unwrap();
        body(transaction_context);
        transaction_context.pop().unwrap();
    }

    #[test]
    fn test_execution_order_and_records() {
        let fixture = Fixture::new();
        let mut tc = fixture.transaction_context(3);
        let writable_signer = InstructionAccount::new(0, true, true);
        let readonly = InstructionAccount::new(1, false, false);
        let mut events = Vec::new();

        // ix0 (prog 2) -> cpi A (prog 3) -> cpi B (prog 4); ix0 -> cpi C (prog 3)
        // ix1 (prog 4), no CPI
        // ix2 (prog 2) -> cpi D (prog 3)
        configure_top_level(
            &mut tc,
            vec![
                (2, vec![writable_signer, readonly]),
                (4, vec![]),
                (2, vec![readonly, readonly]),
            ],
        );
        run_top_level(&mut tc, |tc| {
            run_cpi(tc, 3, vec![readonly], vec![0xa], |tc| {
                run_cpi(tc, 4, vec![], vec![0xb], |tc| {
                    events.push(RecordedEvent {
                        program_id: fixture.keys[4],
                        index_in_trace: tc.get_current_instruction_index().unwrap(),
                        kind: EventKind::Log,
                        payload: b"deep".to_vec(),
                    });
                });
            });
            run_cpi(tc, 3, vec![writable_signer], vec![0xc], |_| {});
        });
        run_top_level(&mut tc, |_| {});
        run_top_level(&mut tc, |tc| {
            events.push(RecordedEvent {
                program_id: fixture.keys[2],
                index_in_trace: tc.get_current_instruction_index().unwrap(),
                kind: EventKind::LogData,
                payload: log_data_payload(&[b"x"]),
            });
            run_cpi(tc, 3, vec![], vec![0xd], |_| {});
        });

        assert_eq!(execution_order(&tc).unwrap(), vec![0, 3, 4, 5, 1, 2, 6]);

        let (records, map) = instruction_trace_records(&tc).unwrap();
        let k = &fixture.keys;
        let expected = vec![
            InstructionRecord::new(
                1,
                k[2].to_bytes(),
                [fixture.meta(0, true, true), fixture.meta(1, false, false)],
                &[0],
            ),
            InstructionRecord::new(2, k[3].to_bytes(), [fixture.meta(1, false, false)], &[0xa]),
            InstructionRecord::new(3, k[4].to_bytes(), [], &[0xb]),
            InstructionRecord::new(2, k[3].to_bytes(), [fixture.meta(0, true, true)], &[0xc]),
            InstructionRecord::new(1, k[4].to_bytes(), [], &[1]),
            InstructionRecord::new(
                1,
                k[2].to_bytes(),
                [fixture.meta(1, false, false), fixture.meta(1, false, false)],
                &[2],
            ),
            InstructionRecord::new(2, k[3].to_bytes(), [], &[0xd]),
        ];
        assert_eq!(records, expected);
        assert_eq!(map, vec![0, 4, 5, 1, 2, 3, 6]);

        let event_records = event_records(events.clone(), &map).unwrap();
        assert_eq!(
            event_records,
            vec![
                EventRecord {
                    program_id: k[4].to_bytes(),
                    instruction_index: 2,
                    kind: EventKind::Log,
                    payload: b"deep".to_vec(),
                },
                EventRecord {
                    program_id: k[2].to_bytes(),
                    instruction_index: 5,
                    kind: EventKind::LogData,
                    payload: log_data_payload(&[b"x"]),
                },
            ]
        );

        let roots = compute_receipt_roots(&tc, events).unwrap();
        assert_eq!(
            roots.instruction_trace_root,
            instruction_trace_root(&expected)
        );
        assert_eq!(roots.events_root, events_root(&event_records));
    }

    #[test]
    fn test_no_cpi_no_events() {
        let fixture = Fixture::new();
        let mut tc = fixture.transaction_context(2);
        configure_top_level(&mut tc, vec![(2, vec![]), (3, vec![])]);
        run_top_level(&mut tc, |_| {});
        run_top_level(&mut tc, |_| {});
        assert_eq!(execution_order(&tc).unwrap(), vec![0, 1]);
        let roots = compute_receipt_roots(&tc, vec![]).unwrap();
        assert_eq!(roots.events_root, empty_events_root());
        assert_eq!(
            roots.instruction_trace_root,
            instruction_trace_root(&[
                InstructionRecord::new(1, fixture.keys[2].to_bytes(), [], &[0]),
                InstructionRecord::new(1, fixture.keys[3].to_bytes(), [], &[1]),
            ])
        );
    }

    #[test]
    fn test_event_with_unknown_instruction() {
        let fixture = Fixture::new();
        let mut tc = fixture.transaction_context(1);
        configure_top_level(&mut tc, vec![(2, vec![])]);
        run_top_level(&mut tc, |_| {});
        let event = RecordedEvent {
            program_id: fixture.keys[2],
            index_in_trace: 1,
            kind: EventKind::Log,
            payload: vec![],
        };
        assert!(compute_receipt_roots(&tc, vec![event]).is_err());
    }
}
