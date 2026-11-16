/**
 * @brief Test program for the Transaction Receipt Tree: emits events with
 * sol_log_ and sol_log_data, and makes a nested CPI.
 *
 * Accounts: [0] payer (signer, writable), [1] recipient (writable),
 * [2] system program, [3] this program.
 *
 * Instruction data[0]:
 *   0 (outer): sol_log("outer"), CPI into this program with data [1],
 *              then sol_log_data(["after", data[1..]]).
 *   1 (inner): sol_log_data(["inner", ""]), then CPI into the system
 *              program: transfer 1 lamport from payer to recipient.
 *   2 (fail):  sol_log("failing") and fail.
 *
 * Build (platform tools): see build.sh next to this file.
 */
#include <solana_sdk.h>

static uint64_t outer(SolParameters *params) {
  sol_log("outer");

  uint8_t data[] = {1};
  SolAccountMeta metas[] = {
      {params->ka[0].key, true, true},
      {params->ka[1].key, true, false},
      {params->ka[2].key, false, false},
      {params->ka[3].key, false, false},
  };
  const SolInstruction instruction = {params->ka[3].key, metas,
                                      SOL_ARRAY_SIZE(metas), data,
                                      SOL_ARRAY_SIZE(data)};
  uint64_t result = sol_invoke(&instruction, params->ka, params->ka_num);
  if (result != SUCCESS) {
    return result;
  }

  static const char after[] = "after";
  SolBytes fields[2];
  fields[0].addr = (const uint8_t *)after;
  fields[0].len = sizeof(after) - 1;
  fields[1].addr = params->data + 1;
  fields[1].len = params->data_len - 1;
  sol_log_data(fields, 2);
  return SUCCESS;
}

static uint64_t inner(SolParameters *params) {
  static const char inner_name[] = "inner";
  SolBytes fields[2];
  fields[0].addr = (const uint8_t *)inner_name;
  fields[0].len = sizeof(inner_name) - 1;
  fields[1].addr = (const uint8_t *)inner_name;
  fields[1].len = 0;
  sol_log_data(fields, 2);

  /* SystemInstruction::Transfer { lamports: 1 } */
  uint8_t data[12] = {2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
  SolAccountMeta metas[] = {
      {params->ka[0].key, true, true},
      {params->ka[1].key, true, false},
  };
  const SolInstruction instruction = {params->ka[2].key, metas,
                                      SOL_ARRAY_SIZE(metas), data,
                                      SOL_ARRAY_SIZE(data)};
  return sol_invoke(&instruction, params->ka, params->ka_num);
}

extern uint64_t entrypoint(const uint8_t *input) {
  SolAccountInfo ka[4];
  SolParameters params = (SolParameters){.ka = ka};

  if (!sol_deserialize(input, &params, SOL_ARRAY_SIZE(ka))) {
    return ERROR_INVALID_ARGUMENT;
  }
  if (params.ka_num != 4 || params.data_len < 1) {
    return ERROR_NOT_ENOUGH_ACCOUNT_KEYS;
  }

  switch (params.data[0]) {
  case 0:
    return outer(&params);
  case 1:
    return inner(&params);
  case 2:
    sol_log("failing");
    return ERROR_CUSTOM_ZERO;
  default:
    return ERROR_INVALID_INSTRUCTION_DATA;
  }
}
