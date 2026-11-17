//! Implementation defined RPC server errors
use {
    crate::response::RpcSimulateTransactionResult,
    jsonrpc_core::{Error, ErrorCode},
    serde::{Deserialize, Serialize},
    solana_clock::Slot,
    solana_transaction_status_client_types::EncodeError,
    thiserror::Error,
};

// Keep in sync with https://github.com/solana-labs/solana-web3.js/blob/master/src/errors.ts
pub const JSON_RPC_SERVER_ERROR_BLOCK_CLEANED_UP: i64 = -32001;
pub const JSON_RPC_SERVER_ERROR_SEND_TRANSACTION_PREFLIGHT_FAILURE: i64 = -32002;
pub const JSON_RPC_SERVER_ERROR_TRANSACTION_SIGNATURE_VERIFICATION_FAILURE: i64 = -32003;
pub const JSON_RPC_SERVER_ERROR_BLOCK_NOT_AVAILABLE: i64 = -32004;
pub const JSON_RPC_SERVER_ERROR_NODE_UNHEALTHY: i64 = -32005;
pub const JSON_RPC_SERVER_ERROR_TRANSACTION_PRECOMPILE_VERIFICATION_FAILURE: i64 = -32006;
pub const JSON_RPC_SERVER_ERROR_SLOT_SKIPPED: i64 = -32007;
pub const JSON_RPC_SERVER_ERROR_NO_SNAPSHOT: i64 = -32008;
pub const JSON_RPC_SERVER_ERROR_LONG_TERM_STORAGE_SLOT_SKIPPED: i64 = -32009;
pub const JSON_RPC_SERVER_ERROR_KEY_EXCLUDED_FROM_SECONDARY_INDEX: i64 = -32010;
pub const JSON_RPC_SERVER_ERROR_TRANSACTION_HISTORY_NOT_AVAILABLE: i64 = -32011;
pub const JSON_RPC_SCAN_ERROR: i64 = -32012;
pub const JSON_RPC_SERVER_ERROR_TRANSACTION_SIGNATURE_LEN_MISMATCH: i64 = -32013;
pub const JSON_RPC_SERVER_ERROR_BLOCK_STATUS_NOT_AVAILABLE_YET: i64 = -32014;
pub const JSON_RPC_SERVER_ERROR_UNSUPPORTED_TRANSACTION_VERSION: i64 = -32015;
pub const JSON_RPC_SERVER_ERROR_MIN_CONTEXT_SLOT_NOT_REACHED: i64 = -32016;
pub const JSON_RPC_SERVER_ERROR_EPOCH_REWARDS_PERIOD_ACTIVE: i64 = -32017;
pub const JSON_RPC_SERVER_ERROR_SLOT_NOT_EPOCH_BOUNDARY: i64 = -32018;
pub const JSON_RPC_SERVER_ERROR_LONG_TERM_STORAGE_UNREACHABLE: i64 = -32019;
pub const JSON_RPC_SERVER_ERROR_FILTER_TRANSACTION_NOT_FOUND: i64 = -32020;
pub const JSON_RPC_SERVER_ERROR_NO_SLOT_HISTORY: i64 = -32021;
pub const JSON_RPC_SERVER_ERROR_LEADER_SCHEDULE_IDENTITY_NOT_FOUND: i64 = -32022;
// Light-client proofs (`getFinalityProof` and the other proof methods); see
// [`LightClientProofError`].
pub const JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_PROOFS_DISABLED: i64 = -32030;
pub const JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_FEATURE_INACTIVE: i64 = -32031;
pub const JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_FINAL: i64 = -32032;
pub const JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_RETAINED: i64 = -32033;
pub const JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_CERTIFICATES_NOT_AVAILABLE: i64 = -32034;
pub const JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_DATA_NOT_AVAILABLE: i64 = -32035;
pub const JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BASE_TREE_NOT_READY: i64 = -32036;
pub const JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_NOT_FOUND: i64 = -32037;

#[derive(Error, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum RpcCustomError {
    #[error("BlockCleanedUp")]
    BlockCleanedUp {
        slot: Slot,
        first_available_block: Slot,
    },
    #[error("SendTransactionPreflightFailure")]
    SendTransactionPreflightFailure {
        message: String,
        result: RpcSimulateTransactionResult,
    },
    #[error("TransactionSignatureVerificationFailure")]
    TransactionSignatureVerificationFailure,
    #[error("BlockNotAvailable")]
    BlockNotAvailable { slot: Slot },
    #[error("NodeUnhealthy")]
    NodeUnhealthy { num_slots_behind: Option<Slot> },
    #[error("TransactionPrecompileVerificationFailure")]
    TransactionPrecompileVerificationFailure(solana_transaction_error::TransactionError),
    #[error("SlotSkipped")]
    SlotSkipped { slot: Slot },
    #[error("NoSnapshot")]
    NoSnapshot,
    #[error("LongTermStorageSlotSkipped")]
    LongTermStorageSlotSkipped { slot: Slot },
    #[error("KeyExcludedFromSecondaryIndex")]
    KeyExcludedFromSecondaryIndex { index_key: String },
    #[error("TransactionHistoryNotAvailable")]
    TransactionHistoryNotAvailable,
    #[error("ScanError")]
    ScanError { message: String },
    #[error("TransactionSignatureLenMismatch")]
    TransactionSignatureLenMismatch,
    #[error("BlockStatusNotAvailableYet")]
    BlockStatusNotAvailableYet { slot: Slot },
    #[error("UnsupportedTransactionVersion")]
    UnsupportedTransactionVersion(u8),
    #[error("MinContextSlotNotReached")]
    MinContextSlotNotReached { context_slot: Slot },
    #[error("EpochRewardsPeriodActive")]
    EpochRewardsPeriodActive {
        slot: Slot,
        current_block_height: u64,
        rewards_complete_block_height: u64,
    },
    #[error("SlotNotEpochBoundary")]
    SlotNotEpochBoundary { slot: Slot },
    #[error("LongTermStorageUnreachable")]
    LongTermStorageUnreachable,
    #[error("FilterTransactionNotFound")]
    FilterTransactionNotFound { signature: String },
    #[error("NoSlotHistory")]
    NoSlotHistory,
    #[error("LeaderScheduleIdentityNotFound")]
    LeaderScheduleIdentityNotFound { identity: String },
    #[error("LightClientProof")]
    LightClientProof(LightClientProofError),
}

/// Why a light-client proof cannot be served. Each kind has its own error
/// code; the error's `data` is this value as JSON (`{"kind": .., ..}`).
#[derive(Clone, Debug, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum LightClientProofError {
    /// The node does not serve light-client proofs (it is started without
    /// `--enable-light-client-proofs`).
    #[error("light-client proofs are not enabled on this node")]
    ProofsDisabled,
    /// The feature that the proof needs is not active for the block.
    #[error("{feature} is not active for slot {slot}")]
    #[serde(rename_all = "camelCase")]
    FeatureInactive { slot: Slot, feature: String },
    /// The block is not finalized on this node yet, or is not on the
    /// finalized chain.
    #[error("block {slot} is not final on this node (latest finalized slot {finalized_slot})")]
    #[serde(rename_all = "camelCase")]
    BlockNotFinal { slot: Slot, finalized_slot: Slot },
    /// The block is final but this node does not keep its proof data (it is
    /// older than the retention window, or older than the node's start).
    #[error(
        "block {slot} is not retained by this node (retained: {first_retained:?} to \
         {last_retained:?})"
    )]
    #[serde(rename_all = "camelCase")]
    BlockNotRetained {
        slot: Slot,
        first_retained: Option<Slot>,
        last_retained: Option<Slot>,
    },
    /// This node has no certificates for the block, nor for any later
    /// finalized block it retains.
    #[error("no certificates for block {slot} or for a later finalized block on this node")]
    #[serde(rename_all = "camelCase")]
    CertificatesNotAvailable { slot: Slot },
    /// A part of the proof is missing on this node: a commitment failure, a
    /// block id or footer that blockstore no longer has, receipts that were
    /// not kept, a history leaf older than the retained history, ...
    #[error("proof data for block {slot} is not available: {reason}")]
    #[serde(rename_all = "camelCase")]
    DataNotAvailable { slot: Slot, reason: String },
    /// The base tree of the accounts state is still being computed; retry.
    #[error("the accounts state base tree for block {slot} is still being computed")]
    #[serde(rename_all = "camelCase")]
    BaseTreeNotReady { slot: Slot },
    /// The transaction, epoch or block asked for is not in what this node
    /// retains.
    #[error("{what} not found")]
    #[serde(rename_all = "camelCase")]
    NotFound { what: String },
}

impl LightClientProofError {
    /// The JSON-RPC error code of this error.
    pub fn code(&self) -> i64 {
        match self {
            Self::ProofsDisabled => JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_PROOFS_DISABLED,
            Self::FeatureInactive { .. } => JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_FEATURE_INACTIVE,
            Self::BlockNotFinal { .. } => JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_FINAL,
            Self::BlockNotRetained { .. } => JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BLOCK_NOT_RETAINED,
            Self::CertificatesNotAvailable { .. } => {
                JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_CERTIFICATES_NOT_AVAILABLE
            }
            Self::DataNotAvailable { .. } => JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_DATA_NOT_AVAILABLE,
            Self::BaseTreeNotReady { .. } => JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_BASE_TREE_NOT_READY,
            Self::NotFound { .. } => JSON_RPC_SERVER_ERROR_LIGHT_CLIENT_NOT_FOUND,
        }
    }
}

impl From<LightClientProofError> for RpcCustomError {
    fn from(err: LightClientProofError) -> Self {
        Self::LightClientProof(err)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeUnhealthyErrorData {
    pub num_slots_behind: Option<Slot>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MinContextSlotNotReachedErrorData {
    pub context_slot: Slot,
}

#[cfg_attr(test, derive(PartialEq))]
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EpochRewardsPeriodActiveErrorData {
    pub current_block_height: u64,
    pub rewards_complete_block_height: u64,
    pub slot: Option<u64>,
}

impl From<EncodeError> for RpcCustomError {
    fn from(err: EncodeError) -> Self {
        match err {
            EncodeError::UnsupportedTransactionVersion(version) => {
                Self::UnsupportedTransactionVersion(version)
            }
        }
    }
}

impl From<RpcCustomError> for Error {
    fn from(e: RpcCustomError) -> Self {
        match e {
            RpcCustomError::BlockCleanedUp {
                slot,
                first_available_block,
            } => Self {
                code: ErrorCode::ServerError(JSON_RPC_SERVER_ERROR_BLOCK_CLEANED_UP),
                message: format!(
                    "Block {slot} cleaned up, does not exist on node. First available block: \
                     {first_available_block}",
                ),
                data: None,
            },
            RpcCustomError::SendTransactionPreflightFailure { message, result } => Self {
                code: ErrorCode::ServerError(
                    JSON_RPC_SERVER_ERROR_SEND_TRANSACTION_PREFLIGHT_FAILURE,
                ),
                message,
                data: Some(serde_json::json!(result)),
            },
            RpcCustomError::TransactionSignatureVerificationFailure => Self {
                code: ErrorCode::ServerError(
                    JSON_RPC_SERVER_ERROR_TRANSACTION_SIGNATURE_VERIFICATION_FAILURE,
                ),
                message: "Transaction signature verification failure".to_string(),
                data: None,
            },
            RpcCustomError::BlockNotAvailable { slot } => Self {
                code: ErrorCode::ServerError(JSON_RPC_SERVER_ERROR_BLOCK_NOT_AVAILABLE),
                message: format!("Block not available for slot {slot}"),
                data: None,
            },
            RpcCustomError::NodeUnhealthy { num_slots_behind } => Self {
                code: ErrorCode::ServerError(JSON_RPC_SERVER_ERROR_NODE_UNHEALTHY),
                message: if let Some(num_slots_behind) = num_slots_behind {
                    format!("Node is behind by {num_slots_behind} slots")
                } else {
                    "Node is unhealthy".to_string()
                },
                data: Some(serde_json::json!(NodeUnhealthyErrorData {
                    num_slots_behind
                })),
            },
            RpcCustomError::TransactionPrecompileVerificationFailure(e) => Self {
                code: ErrorCode::ServerError(
                    JSON_RPC_SERVER_ERROR_TRANSACTION_PRECOMPILE_VERIFICATION_FAILURE,
                ),
                message: format!("Transaction precompile verification failure {e:?}"),
                data: None,
            },
            RpcCustomError::SlotSkipped { slot } => Self {
                code: ErrorCode::ServerError(JSON_RPC_SERVER_ERROR_SLOT_SKIPPED),
                message: format!(
                    "Slot {slot} was skipped, or missing due to ledger jump to recent snapshot"
                ),
                data: None,
            },
            RpcCustomError::NoSnapshot => Self {
                code: ErrorCode::ServerError(JSON_RPC_SERVER_ERROR_NO_SNAPSHOT),
                message: "No snapshot".to_string(),
                data: None,
            },
            RpcCustomError::LongTermStorageSlotSkipped { slot } => Self {
                code: ErrorCode::ServerError(JSON_RPC_SERVER_ERROR_LONG_TERM_STORAGE_SLOT_SKIPPED),
                message: format!("Slot {slot} was skipped, or missing in long-term storage"),
                data: None,
            },
            RpcCustomError::KeyExcludedFromSecondaryIndex { index_key } => Self {
                code: ErrorCode::ServerError(
                    JSON_RPC_SERVER_ERROR_KEY_EXCLUDED_FROM_SECONDARY_INDEX,
                ),
                message: format!(
                    "{index_key} excluded from account secondary indexes; this RPC method \
                     unavailable for key"
                ),
                data: None,
            },
            RpcCustomError::TransactionHistoryNotAvailable => Self {
                code: ErrorCode::ServerError(
                    JSON_RPC_SERVER_ERROR_TRANSACTION_HISTORY_NOT_AVAILABLE,
                ),
                message: "Transaction history is not available from this node".to_string(),
                data: None,
            },
            RpcCustomError::ScanError { message } => Self {
                code: ErrorCode::ServerError(JSON_RPC_SCAN_ERROR),
                message,
                data: None,
            },
            RpcCustomError::TransactionSignatureLenMismatch => Self {
                code: ErrorCode::ServerError(
                    JSON_RPC_SERVER_ERROR_TRANSACTION_SIGNATURE_LEN_MISMATCH,
                ),
                message: "Transaction signature length mismatch".to_string(),
                data: None,
            },
            RpcCustomError::BlockStatusNotAvailableYet { slot } => Self {
                code: ErrorCode::ServerError(JSON_RPC_SERVER_ERROR_BLOCK_STATUS_NOT_AVAILABLE_YET),
                message: format!("Block status not yet available for slot {slot}"),
                data: None,
            },
            RpcCustomError::UnsupportedTransactionVersion(version) => Self {
                code: ErrorCode::ServerError(JSON_RPC_SERVER_ERROR_UNSUPPORTED_TRANSACTION_VERSION),
                message: format!(
                    "Transaction version ({version}) is not supported by the requesting client. \
                     Please try the request again with the following configuration parameter: \
                     \"maxSupportedTransactionVersion\": {version}"
                ),
                data: None,
            },
            RpcCustomError::MinContextSlotNotReached { context_slot } => Self {
                code: ErrorCode::ServerError(JSON_RPC_SERVER_ERROR_MIN_CONTEXT_SLOT_NOT_REACHED),
                message: "Minimum context slot has not been reached".to_string(),
                data: Some(serde_json::json!(MinContextSlotNotReachedErrorData {
                    context_slot,
                })),
            },
            RpcCustomError::EpochRewardsPeriodActive {
                slot,
                current_block_height,
                rewards_complete_block_height,
            } => Self {
                code: ErrorCode::ServerError(JSON_RPC_SERVER_ERROR_EPOCH_REWARDS_PERIOD_ACTIVE),
                message: format!("Epoch rewards period still active at slot {slot}"),
                data: Some(serde_json::json!(EpochRewardsPeriodActiveErrorData {
                    current_block_height,
                    rewards_complete_block_height,
                    slot: Some(slot),
                })),
            },
            RpcCustomError::SlotNotEpochBoundary { slot } => Self {
                code: ErrorCode::ServerError(JSON_RPC_SERVER_ERROR_SLOT_NOT_EPOCH_BOUNDARY),
                message: format!(
                    "Rewards cannot be found because slot {slot} is not the epoch boundary. This \
                     may be due to gap in the queried node's local ledger or long-term storage"
                ),
                data: Some(serde_json::json!({
                    "slot": slot,
                })),
            },
            RpcCustomError::LongTermStorageUnreachable => Self {
                code: ErrorCode::ServerError(JSON_RPC_SERVER_ERROR_LONG_TERM_STORAGE_UNREACHABLE),
                message: "Failed to query long-term storage; please try again".to_string(),
                data: None,
            },
            RpcCustomError::FilterTransactionNotFound { signature } => Self {
                code: ErrorCode::ServerError(JSON_RPC_SERVER_ERROR_FILTER_TRANSACTION_NOT_FOUND),
                message: format!("Transaction {signature} not found"),
                data: None,
            },
            RpcCustomError::NoSlotHistory => Self {
                code: ErrorCode::ServerError(JSON_RPC_SERVER_ERROR_NO_SLOT_HISTORY),
                message: "No slot history".to_string(),
                data: None,
            },
            RpcCustomError::LeaderScheduleIdentityNotFound { identity } => Self {
                code: ErrorCode::ServerError(
                    JSON_RPC_SERVER_ERROR_LEADER_SCHEDULE_IDENTITY_NOT_FOUND,
                ),
                message: format!(
                    "Node {identity} was not in the leader schedule for specified epoch"
                ),
                data: None,
            },
            RpcCustomError::LightClientProof(err) => Self {
                code: ErrorCode::ServerError(err.code()),
                message: err.to_string(),
                data: serde_json::to_value(&err).ok(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use {
        crate::custom_error::EpochRewardsPeriodActiveErrorData, serde_json::Value,
        test_case::test_case,
    };

    #[test]
    fn test_light_client_proof_error_json() {
        use {
            super::{LightClientProofError, RpcCustomError},
            jsonrpc_core::{Error, ErrorCode},
        };
        let err = LightClientProofError::BlockNotRetained {
            slot: 5,
            first_retained: Some(10),
            last_retained: None,
        };
        let rpc_error = Error::from(RpcCustomError::from(err.clone()));
        assert_eq!(rpc_error.code, ErrorCode::ServerError(-32033));
        let data = rpc_error.data.unwrap();
        assert_eq!(
            data,
            serde_json::json!({
                "kind": "blockNotRetained",
                "slot": 5,
                "firstRetained": 10,
                "lastRetained": null,
            })
        );
        assert_eq!(
            serde_json::from_value::<LightClientProofError>(data).unwrap(),
            err
        );
        let rpc_error = Error::from(RpcCustomError::from(LightClientProofError::ProofsDisabled));
        assert_eq!(rpc_error.code, ErrorCode::ServerError(-32030));
        assert_eq!(
            rpc_error.data.unwrap(),
            serde_json::json!({"kind": "proofsDisabled"})
        );
    }

    #[test_case(serde_json::json!({
            "currentBlockHeight": 123,
            "rewardsCompleteBlockHeight": 456
        }); "Pre-3.0 schema")]
    #[test_case(serde_json::json!({
            "currentBlockHeight": 123,
            "rewardsCompleteBlockHeight": 456,
            "slot": 789
        }); "3.0+ schema")]
    fn test_deseriailze_epoch_rewards_period_active_error_data(serialized_data: Value) {
        let expected_current_block_height = serialized_data
            .get("currentBlockHeight")
            .map(|v| v.as_u64().unwrap())
            .unwrap();
        let expected_rewards_complete_block_height = serialized_data
            .get("rewardsCompleteBlockHeight")
            .map(|v| v.as_u64().unwrap())
            .unwrap();
        let expected_slot: Option<u64> = serialized_data.get("slot").map(|v| v.as_u64().unwrap());
        let actual: EpochRewardsPeriodActiveErrorData =
            serde_json::from_value(serialized_data).expect("Failed to deserialize test fixture");
        assert_eq!(
            actual,
            EpochRewardsPeriodActiveErrorData {
                current_block_height: expected_current_block_height,
                rewards_complete_block_height: expected_rewards_complete_block_height,
                slot: expected_slot,
            }
        );
    }
}
