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
//!
//! Which nodes are asked: those named in the configuration, if any.
//! Otherwise the nodes that publish an RPC address in gossip, with this
//! node's shred version ([`gossip_history_peers`]), as a node that
//! downloads a snapshot finds its source. A node that serves light-client
//! proofs says so in its contact info; those are the ones asked. If none
//! says so, any node with an RPC address is tried, a few in each round.

use {
    super::{HistoryLog, Inner, persist::HistoryEntry},
    log::*,
    serde::{Deserialize, Serialize},
    solana_clock::Slot,
    solana_gossip::cluster_info::ClusterInfo,
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

/// Gives the nodes to ask, each time a round of asking starts.
pub type HistoryPeerSource = Arc<dyn Fn() -> Vec<Arc<dyn HistoryPeer>> + Send + Sync>;

/// Nodes found in gossip that are asked in one round.
const GOSSIP_PEERS_PER_ROUND: usize = 16;

/// Time a node named in the configuration has to answer.
const NAMED_PEER_TIMEOUT: Duration = Duration::from_secs(30);

/// Time a node found in gossip has to answer: most do not serve this.
const GOSSIP_PEER_TIMEOUT: Duration = Duration::from_secs(5);

/// The nodes that publish an RPC address in gossip with this node's shred
/// version and say they serve light-client proofs; if none says so, all
/// those with an RPC address. At most [`GOSSIP_PEERS_PER_ROUND`] of them
/// per call, others on the next call.
pub fn gossip_history_peers(cluster_info: Arc<ClusterInfo>) -> HistoryPeerSource {
    let next = std::sync::atomic::AtomicUsize::new(0);
    Arc::new(move || {
        let shred_version = cluster_info.my_shred_version();
        let mut nodes: Vec<_> = cluster_info
            .rpc_peers()
            .into_iter()
            .filter(|node| node.shred_version() == shred_version && node.rpc().is_some())
            .collect();
        if nodes.iter().any(|node| node.serves_light_client_proofs()) {
            nodes.retain(|node| node.serves_light_client_proofs());
        }
        let mut urls: Vec<String> = nodes
            .iter()
            .filter_map(|node| Some(format!("http://{}", node.rpc()?)))
            .collect();
        // The order gossip gives changes; a sorted list is walked in turn.
        urls.sort_unstable();
        urls.dedup();
        let count = urls.len().min(GOSSIP_PEERS_PER_ROUND);
        let start = next.fetch_add(count, Ordering::Relaxed);
        (0..count)
            .map(|offset| {
                let url = urls[start.wrapping_add(offset) % urls.len()].clone();
                Arc::new(RpcHistoryPeer::with_timeout(url, GOSSIP_PEER_TIMEOUT))
                    as Arc<dyn HistoryPeer>
            })
            .collect()
    })
}

/// A node reached over JSON RPC.
pub struct RpcHistoryPeer {
    url: String,
    timeout: Duration,
    /// Made by the thread that fetches, on first use.
    client: OnceLock<solana_rpc_client::rpc_client::RpcClient>,
}

impl RpcHistoryPeer {
    pub fn new(url: String) -> Self {
        Self::with_timeout(url, NAMED_PEER_TIMEOUT)
    }

    pub fn with_timeout(url: String, timeout: Duration) -> Self {
        Self {
            url,
            timeout,
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
                    self.timeout,
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

/// Asks the peers `peers` gives, in turn, for the leaves before the target
/// frontier, until one gives them. `peers` is called once per round. Runs
/// on a thread of the store.
pub(super) fn fetch_history(
    inner: &RwLock<Inner>,
    peers: &(dyn Fn() -> Vec<Arc<dyn HistoryPeer>> + Send + Sync),
    target: FetchTarget,
    retained_history: usize,
    exit: &AtomicBool,
) {
    let end = target.frontier.leaf_count();
    for round in 0..FETCH_ROUNDS {
        for peer in &peers() {
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

#[cfg(test)]
mod tests {
    use {
        super::*, solana_gossip::contact_info::ContactInfo, solana_keypair::Keypair,
        solana_net_utils::SocketAddrSpace, solana_signer::Signer, std::net::Ipv4Addr,
    };

    /// The gossip source gives the nodes with an RPC address and this
    /// node's shred version, a bounded number per round, all of them over
    /// the rounds.
    #[test]
    fn test_gossip_history_peers() {
        let node = |shred_version: u16, rpc_port: Option<u16>| {
            let mut info = ContactInfo::new(
                Keypair::new().pubkey(),
                solana_time_utils::timestamp(),
                shred_version,
            );
            info.set_gossip((Ipv4Addr::LOCALHOST, 7000)).unwrap();
            if let Some(port) = rpc_port {
                info.set_rpc((Ipv4Addr::LOCALHOST, port)).unwrap();
            }
            info
        };
        let keypair = Arc::new(Keypair::new());
        let mut own = node(7, Some(8899));
        own.hot_swap_pubkey(keypair.pubkey());
        let cluster_info = Arc::new(ClusterInfo::new(own, keypair, SocketAddrSpace::Unspecified));
        let source = gossip_history_peers(cluster_info.clone());
        assert!(source().is_empty());

        // Not asked: no RPC address, another shred version.
        cluster_info.insert_info(node(7, None));
        cluster_info.insert_info(node(8, Some(9000)));
        cluster_info.insert_info(node(7, Some(9001)));
        cluster_info.insert_info(node(7, Some(9002)));
        let names = |peers: Vec<Arc<dyn HistoryPeer>>| -> Vec<String> {
            peers.iter().map(|peer| peer.name()).collect()
        };
        assert_eq!(
            names(source()),
            ["http://127.0.0.1:9001", "http://127.0.0.1:9002"]
        );

        // More nodes than one round asks: the next round goes on.
        for port in 9003..9003 + 2 * GOSSIP_PEERS_PER_ROUND as u16 {
            cluster_info.insert_info(node(7, Some(port)));
        }
        let source = gossip_history_peers(cluster_info.clone());
        let mut asked: Vec<String> = (0..3).flat_map(|_| names(source())).collect();
        assert_eq!(asked.len(), 3 * GOSSIP_PEERS_PER_ROUND);
        asked.sort_unstable();
        asked.dedup();
        assert_eq!(asked.len(), 2 * GOSSIP_PEERS_PER_ROUND + 2);

        // Nodes that say they serve light-client proofs: only they are
        // asked.
        for port in [9900, 9901] {
            let mut serving = node(7, Some(port));
            serving.set_serves_light_client_proofs(true);
            cluster_info.insert_info(serving);
        }
        let mut not_ours = node(8, Some(9902));
        not_ours.set_serves_light_client_proofs(true);
        cluster_info.insert_info(not_ours);
        let source = gossip_history_peers(cluster_info);
        assert_eq!(
            names(source()),
            ["http://127.0.0.1:9900", "http://127.0.0.1:9901"]
        );
    }
}
