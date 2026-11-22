//! History leaves from another node.
//!
//! A node that starts from a snapshot made elsewhere has the history
//! frontier of its first rooted block (it is in the bank), but not the
//! leaves before it, and its blockstore does not have those blocks. Without
//! the leaves it cannot give a history proof of an earlier block.
//!
//! Another node that serves light-client proofs has them, and gives them
//! with `getBlockHistoryLeaves`. The other node is not trusted: the leaves
//! it gives, appended to the frontier it gives, must have exactly the
//! history root of this node's own rooted block, or they are not used. A
//! node that gives wrong leaves can only make the fetch fail.

use {
    super::{HistoryLog, Inner, persist::HistoryEntry},
    log::*,
    serde::{Deserialize, Serialize},
    solana_clock::Slot,
    solana_hash::Hash,
    solana_light_client::{
        history_leaf,
        solana_merkle_tree::commitment_tree::{AppendOnlyFrontier, TreeId, hash_leaf},
    },
    std::{
        str::FromStr,
        sync::{
            Arc, OnceLock, RwLock,
            atomic::{AtomicBool, Ordering},
        },
        time::Duration,
    },
};

/// Most history leaves one `getBlockHistoryLeaves` call returns.
pub const MAX_HISTORY_LEAVES_PER_REQUEST: usize = 4096;

/// How many times every peer is asked before the fetch gives up.
const FETCH_ROUNDS: usize = 6;

/// Time between two rounds: a peer may be a few roots behind this node.
const FETCH_ROUND_INTERVAL: Duration = Duration::from_secs(5);

/// One history leaf, as `getBlockHistoryLeaves` gives it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcHistoryLeaf {
    pub slot: Slot,
    pub block_id: String,
    pub commitment_root: String,
}

/// Result of `getBlockHistoryLeaves`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcHistoryLeaves {
    /// Number of leaves of the history the node has, up to its latest
    /// rooted block.
    pub leaf_count: u64,
    /// Index of the first leaf given.
    pub start: u64,
    /// With `withFrontier`: the roots of the frontier of the leaves before
    /// `start`, largest subtree first.
    pub frontier: Option<Vec<String>>,
    pub leaves: Vec<RpcHistoryLeaf>,
}

/// A node that gives history leaves.
pub trait HistoryPeer: Send + Sync {
    /// Its name in the log.
    fn name(&self) -> String;

    fn history_leaves(
        &self,
        start: u64,
        limit: usize,
        with_frontier: bool,
    ) -> Result<RpcHistoryLeaves, String>;
}

/// A node reached over JSON RPC.
pub struct RpcHistoryPeer {
    url: String,
    /// Made by the thread that fetches, on first use.
    client: OnceLock<solana_rpc_client::rpc_client::RpcClient>,
}

impl RpcHistoryPeer {
    pub fn new(url: String) -> Self {
        Self {
            url,
            client: OnceLock::new(),
        }
    }
}

impl HistoryPeer for RpcHistoryPeer {
    fn name(&self) -> String {
        self.url.clone()
    }

    fn history_leaves(
        &self,
        start: u64,
        limit: usize,
        with_frontier: bool,
    ) -> Result<RpcHistoryLeaves, String> {
        self.client
            .get_or_init(|| {
                solana_rpc_client::rpc_client::RpcClient::new_with_timeout(
                    self.url.clone(),
                    Duration::from_secs(30),
                )
            })
            .send(
                solana_rpc_client_api::request::RpcRequest::Custom {
                    method: "getBlockHistoryLeaves",
                },
                serde_json::json!([start, limit, with_frontier]),
            )
            .map_err(|err| err.to_string())
    }
}

/// What a fetch is for: the history the store restarted from.
pub(super) struct FetchTarget {
    /// The frontier the log restarted at: the history of a rooted block of
    /// this node before its own leaf.
    pub frontier: AppendOnlyFrontier,
    /// `restarts` of the log after that restart.
    pub restarts: u64,
    /// Number of leaves to get before the frontier, at most.
    pub leaves: u64,
}

fn parse_hash(text: &str) -> Result<Hash, String> {
    Hash::from_str(text).map_err(|err| format!("not a hash ({err}): {text}"))
}

/// The frontier before `start` and the leaves `[start, target)` from
/// `peer`, checked: together they have the root of the target frontier.
fn fetch_from(
    peer: &dyn HistoryPeer,
    target: &FetchTarget,
    exit: &AtomicBool,
) -> Result<(AppendOnlyFrontier, Vec<HistoryEntry>), String> {
    let end = target.frontier.leaf_count();
    let start = end.saturating_sub(target.leaves);
    let mut base: Option<AppendOnlyFrontier> = None;
    let mut entries: Vec<HistoryEntry> = Vec::new();
    while start + (entries.len() as u64) < end {
        if exit.load(Ordering::Relaxed) {
            return Err("the store is closing".to_string());
        }
        let next = start + entries.len() as u64;
        let limit = usize::try_from(end - next)
            .unwrap_or(usize::MAX)
            .min(MAX_HISTORY_LEAVES_PER_REQUEST);
        let page = peer.history_leaves(next, limit, base.is_none())?;
        if page.leaf_count < end {
            return Err(format!(
                "it has {} history leaves, {end} are needed",
                page.leaf_count
            ));
        }
        if page.start != next || page.leaves.is_empty() || page.leaves.len() > limit {
            return Err(format!("unexpected answer for leaves from {next}"));
        }
        if base.is_none() {
            let roots = page
                .frontier
                .as_ref()
                .ok_or("no frontier in the answer")?
                .iter()
                .map(|root| parse_hash(root))
                .collect::<Result<Vec<_>, _>>()?;
            base = Some(
                AppendOnlyFrontier::from_parts(start, roots)
                    .map_err(|err| format!("not a frontier of {start} leaves: {err:?}"))?,
            );
        }
        for leaf in &page.leaves {
            entries.push((
                leaf.slot,
                parse_hash(&leaf.block_id)?,
                parse_hash(&leaf.commitment_root)?,
            ));
        }
    }
    let base = match base {
        Some(base) => base,
        // Nothing to get: the target is the frontier before `start`.
        None => target.frontier.clone(),
    };
    let mut all = base.clone();
    for (slot, block_id, commitment_root) in &entries {
        all.append_leaf_hash(hash_leaf(
            TreeId::BLOCK_HISTORY,
            &history_leaf(*slot, block_id, commitment_root),
        ))
        .map_err(|err| format!("{err:?}"))?;
    }
    if all.root(TreeId::BLOCK_HISTORY) != target.frontier.root(TreeId::BLOCK_HISTORY) {
        return Err("its leaves do not have this node's history root".to_string());
    }
    Ok((base, entries))
}

/// Puts the fetched leaves before the leaves the log got since it
/// restarted. `false` if the log is no longer the one the fetch was for.
fn splice(
    inner: &RwLock<Inner>,
    target: &FetchTarget,
    base: AppendOnlyFrontier,
    fetched: Vec<HistoryEntry>,
    retained_history: usize,
) -> bool {
    let mut inner = inner.write().unwrap();
    let log = &inner.history;
    if log.restarts != target.restarts || log.base_len() != target.frontier.leaf_count() {
        return false;
    }
    let mut entries = fetched;
    entries.extend(log.entries.iter().copied());
    let mut new = HistoryLog::from_parts(base, entries);
    if new.len() != log.len() || new.frontier_root() != log.frontier_root() {
        return false;
    }
    // The files follow one log: this is another.
    new.restarts = log.restarts.saturating_add(1);
    new.prune(retained_history);
    inner.history = new;
    true
}

/// Asks the peers, in turn, for the leaves before the target frontier,
/// until one gives them. Runs on a thread of the store.
pub(super) fn fetch_history(
    inner: &RwLock<Inner>,
    peers: &[Arc<dyn HistoryPeer>],
    target: FetchTarget,
    retained_history: usize,
    exit: &AtomicBool,
) {
    let end = target.frontier.leaf_count();
    for round in 0..FETCH_ROUNDS {
        for peer in peers {
            match fetch_from(peer.as_ref(), &target, exit) {
                Ok((base, entries)) => {
                    let count = entries.len();
                    if splice(inner, &target, base, entries, retained_history) {
                        info!(
                            "light-client proofs: took {count} history leaves before leaf {end} \
                             from {}",
                            peer.name()
                        );
                    } else {
                        info!(
                            "light-client proofs: the history changed while its leaves were \
                             fetched; not using them"
                        );
                    }
                    return;
                }
                Err(reason) => warn!(
                    "light-client proofs: no history leaves from {}: {reason}",
                    peer.name()
                ),
            }
            if exit.load(Ordering::Relaxed) {
                return;
            }
        }
        if round + 1 < FETCH_ROUNDS {
            let mut waited = Duration::ZERO;
            while waited < FETCH_ROUND_INTERVAL {
                if exit.load(Ordering::Relaxed) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
                waited += Duration::from_millis(100);
            }
        }
    }
    warn!(
        "light-client proofs: no peer gave the history leaves before leaf {end}; blocks before \
         it are served with their own certificates only"
    );
}
