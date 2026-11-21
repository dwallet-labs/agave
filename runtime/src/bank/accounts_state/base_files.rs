//! The leaves of base trees on disk.
//!
//! A base tree has one 64-byte leaf per account. On a cluster with a billion
//! accounts that is 64 GB per generation, and a bank holds two generations
//! for half of every epoch, so the leaves are not kept in memory: they are
//! written to a file as they are derived and mapped read-only
//! ([`BaseDir::writer`]). What stays in memory is what the tree keeps of its
//! inner nodes (32 bytes per 1024 accounts) and whatever pages of the file
//! the kernel caches.
//!
//! - **Files.** One per base, named by its identity (epoch, and slot and
//!   bank hash of the block whose state it is), next to the accounts
//!   directory. A file is written under a temporary name and renamed when
//!   complete. It is not synced: a file damaged by a power loss does not
//!   have the root it is checked against when it is opened again, and is
//!   then rebuilt.
//! - **Reuse.** A node that restarts from a snapshot finds the files of the
//!   snapshot's generations and uses them if their leaves have the base
//!   roots the snapshot carries ([`BaseDir::open`]), instead of scanning
//!   every account.
//! - **Removal.** When the base of a new epoch is requested, the files of
//!   epochs other than that one and the one before are removed
//!   ([`BaseDir::retain_epochs`]). A file is never removed because its tree
//!   is dropped: the next start may want it.
//! - **Sorting.** A scan of the accounts does not give them in address
//!   order. [`LeafSorter`] takes the leaves in any order, keeps a bounded
//!   amount in memory, spills the rest to 256 files by the first byte of
//!   the address, and gives them back sorted, one file at a time.
//!
//! Without a directory (it cannot be created) everything is held in memory,
//! with a warning.

use {
    super::BaseId,
    log::*,
    memmap2::Mmap,
    rayon::prelude::*,
    solana_clock::Epoch,
    solana_hash::Hash,
    solana_merkle_tree::accounts_state::{
        BASE_LEAF_LEN, BaseLeaf, BaseLeafStorage, BaseTree, BaseTreeError,
    },
    std::{
        fs::{self, File},
        io::{self, BufWriter, Read, Write},
        path::{Path, PathBuf},
        sync::{
            LazyLock, OnceLock,
            atomic::{AtomicU64, Ordering},
        },
    },
};

/// Name of the directory, next to (or inside) the accounts directory.
const DIR_NAME: &str = "accounts_state_bases";
const LEAVES_EXTENSION: &str = "leaves";
const TEMPORARY_EXTENSION: &str = "tmp";

/// Bytes of leaves a [`LeafSorter`] holds before it spills to files.
const SORTER_BUDGET_BYTES: usize = 256 << 20;
/// Number of files a [`LeafSorter`] spills to: one per first address byte.
const SORTER_BUCKETS: usize = 256;

/// The threads that hash base leaves. A base is computed once per epoch,
/// off the block path, with half an epoch to finish: a few threads.
static BASE_POOL: LazyLock<rayon::ThreadPool> = LazyLock::new(|| {
    let threads = std::thread::available_parallelism()
        .map(|cores| cores.get() / 4)
        .unwrap_or(1)
        .clamp(1, 8);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|index| format!("solAcctStHash{index:02}"))
        .build()
        .expect("a thread pool for base trees")
});

/// Runs `job` on the threads that hash base leaves.
pub(super) fn on_base_pool<T: Send>(job: impl FnOnce() -> T + Send) -> T {
    BASE_POOL.install(job)
}

/// The base tree over `storage`, with its chunks hashed in parallel.
pub(super) fn tree_from_storage(
    storage: Box<dyn BaseLeafStorage>,
) -> Result<BaseTree, BaseTreeError> {
    BaseTree::from_storage(storage, &|count, one| {
        on_base_pool(|| (0..count).into_par_iter().map(one).collect())
    })
}

static NEXT_TEMPORARY: AtomicU64 = AtomicU64::new(0);

/// A file mapped read-only.
struct MappedLeaves(Mmap);

impl BaseLeafStorage for MappedLeaves {
    fn bytes(&self) -> &[u8] {
        &self.0
    }
}

fn map_leaves(path: &Path) -> io::Result<Box<dyn BaseLeafStorage>> {
    let file = File::open(path)?;
    if file.metadata()?.len() == 0 {
        // An empty file cannot be mapped.
        return Ok(Box::new(Vec::new()));
    }
    // SAFETY: the file is only read through the map. It is written before
    // it gets this name and never changed after; removing it does not
    // invalidate the map.
    let map = unsafe { Mmap::map(&file)? };
    Ok(Box::new(MappedLeaves(map)))
}

/// Where base leaves are kept: a directory, or memory.
#[derive(Debug, Default)]
pub(super) struct BaseDir {
    path: Option<PathBuf>,
    /// The directory is created, and what an earlier process left unfinished
    /// removed, when it is first used.
    prepared: OnceLock<bool>,
}

impl BaseDir {
    /// The directory for a node whose accounts are in `accounts_path`: next
    /// to it if that is the `run` directory of an accounts directory (which
    /// is emptied at every start), inside it otherwise.
    pub fn for_accounts_path(accounts_path: &Path) -> Self {
        let is_run = accounts_path.file_name().is_some_and(|name| name == "run");
        let parent = match accounts_path.parent() {
            Some(parent) if is_run => parent,
            _ => accounts_path,
        };
        Self {
            path: Some(parent.join(DIR_NAME)),
            prepared: OnceLock::new(),
        }
    }

    /// The directory, created. `None`: leaves are held in memory.
    pub(super) fn path(&self) -> Option<&Path> {
        let path = self.path.as_deref()?;
        let usable = *self.prepared.get_or_init(|| {
            if let Err(err) = fs::create_dir_all(path) {
                warn!(
                    "accounts state: cannot use {}: {err}; base trees are held in memory",
                    path.display()
                );
                return false;
            }
            for entry in fs::read_dir(path).into_iter().flatten().flatten() {
                let leftover = entry.path();
                if leftover
                    .extension()
                    .is_some_and(|ext| ext == TEMPORARY_EXTENSION)
                {
                    let _ = fs::remove_file(leftover);
                }
            }
            true
        });
        usable.then_some(path)
    }

    fn file_name(id: &BaseId) -> String {
        match id.parent {
            Some((slot, bank_hash)) => {
                format!("{}-{slot}-{bank_hash}.{LEAVES_EXTENSION}", id.epoch)
            }
            None => format!("{}-none.{LEAVES_EXTENSION}", id.epoch),
        }
    }

    fn temporary(dir: &Path, what: &str) -> PathBuf {
        dir.join(format!(
            "{what}.{}-{}.{TEMPORARY_EXTENSION}",
            std::process::id(),
            NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed)
        ))
    }

    /// Takes the leaves of base `id`, in address order.
    pub fn writer(&self, id: &BaseId) -> LeafWriter {
        let file = self.path().and_then(|dir| {
            let name = Self::file_name(id);
            let temporary = Self::temporary(dir, &name);
            match File::create(&temporary) {
                Ok(file) => Some(LeafFile {
                    file: BufWriter::with_capacity(1 << 20, file),
                    temporary,
                    path: dir.join(name),
                }),
                Err(err) => {
                    warn!(
                        "accounts state: cannot create {}: {err}; holding the base tree in memory",
                        temporary.display()
                    );
                    None
                }
            }
        });
        LeafWriter {
            file,
            memory: Vec::new(),
        }
    }

    /// The leaves of base `id` written by an earlier process, if there are
    /// any. They are not checked: the caller compares the root of their
    /// tree with the one it knows.
    pub fn open(&self, id: &BaseId) -> Option<Box<dyn BaseLeafStorage>> {
        let path = self.path()?.join(Self::file_name(id));
        if !path.is_file() {
            return None;
        }
        map_leaves(&path)
            .map_err(|err| warn!("accounts state: cannot read {}: {err}", path.display()))
            .ok()
    }

    /// Removes the leaves of base `id`.
    pub fn remove(&self, id: &BaseId) {
        if let Some(dir) = self.path() {
            let _ = fs::remove_file(dir.join(Self::file_name(id)));
        }
    }

    /// Removes the leaves of every base whose epoch `keep` rejects, and
    /// files that are not base leaves.
    pub fn retain_epochs(&self, keep: impl Fn(Epoch) -> bool) {
        let Some(dir) = self.path() else {
            return;
        };
        for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != LEAVES_EXTENSION) {
                continue;
            }
            let epoch = path
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(|name| name.split('-').next())
                .and_then(|epoch| epoch.parse::<Epoch>().ok());
            if !epoch.is_some_and(&keep) {
                match fs::remove_file(&path) {
                    Ok(()) => info!("accounts state: removed {}", path.display()),
                    Err(err) => warn!("accounts state: cannot remove {}: {err}", path.display()),
                }
            }
        }
    }

    /// A sorter that spills to this directory.
    pub fn sorter(&self) -> LeafSorter {
        self.sorter_with_budget(SORTER_BUDGET_BYTES)
    }

    pub(super) fn sorter_with_budget(&self, budget: usize) -> LeafSorter {
        LeafSorter {
            spill: self
                .path()
                .map(|dir| Self::temporary(dir, "sort").with_extension("")),
            budget,
            held: 0,
            buckets: vec![Vec::new(); SORTER_BUCKETS],
            spilled: vec![false; SORTER_BUCKETS],
        }
    }
}

struct LeafFile {
    file: BufWriter<File>,
    temporary: PathBuf,
    path: PathBuf,
}

/// Takes the leaves of one base in address order: to its file, or to memory
/// if there is no directory.
pub(super) struct LeafWriter {
    file: Option<LeafFile>,
    memory: Vec<u8>,
}

impl LeafWriter {
    pub fn push(&mut self, leaf: BaseLeaf) -> io::Result<()> {
        match &mut self.file {
            Some(file) => file.file.write_all(&leaf.bytes()),
            None => {
                self.memory.extend_from_slice(&leaf.bytes());
                Ok(())
            }
        }
    }

    /// The leaves, readable: the file under its name, mapped.
    pub fn finish(mut self) -> io::Result<Box<dyn BaseLeafStorage>> {
        let Some(LeafFile {
            file,
            temporary,
            path,
        }) = self.file.take()
        else {
            return Ok(Box::new(std::mem::take(&mut self.memory)));
        };
        let finished = file
            .into_inner()
            .map_err(io::IntoInnerError::into_error)
            .and_then(|_| fs::rename(&temporary, &path))
            .and_then(|()| map_leaves(&path));
        if finished.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        finished
    }
}

impl Drop for LeafWriter {
    fn drop(&mut self) {
        // Not finished: nothing is left behind.
        if let Some(file) = self.file.take() {
            let _ = fs::remove_file(file.temporary);
        }
    }
}

/// Sorts base leaves by address without holding all of them in memory. See
/// the module documentation.
pub(super) struct LeafSorter {
    /// Prefix of the spill files (`<prefix>.<bucket>.tmp`). `None`: no
    /// directory, everything is held.
    spill: Option<PathBuf>,
    budget: usize,
    /// Bytes held in `buckets`.
    held: usize,
    /// Leaves by first address byte, in the order they were pushed.
    buckets: Vec<Vec<u8>>,
    spilled: Vec<bool>,
}

impl LeafSorter {
    fn spill_path(prefix: &Path, bucket: usize) -> PathBuf {
        let mut name = prefix.as_os_str().to_owned();
        name.push(format!(".{bucket}.{TEMPORARY_EXTENSION}"));
        PathBuf::from(name)
    }

    pub fn push(&mut self, address: &[u8; 32], account_hash: &Hash) -> io::Result<()> {
        let bucket = &mut self.buckets[address[0] as usize];
        bucket.extend_from_slice(address);
        bucket.extend_from_slice(account_hash.as_ref());
        self.held = self.held.saturating_add(BASE_LEAF_LEN);
        if self.held > self.budget && self.spill.is_some() {
            self.spill()?;
        }
        Ok(())
    }

    /// Appends what is held to the files. Each file is opened for the
    /// append only, so that the sorter never holds 256 open files.
    fn spill(&mut self) -> io::Result<()> {
        let Some(prefix) = &self.spill else {
            return Ok(());
        };
        for (index, bucket) in self.buckets.iter_mut().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(Self::spill_path(prefix, index))?;
            self.spilled[index] = true;
            file.write_all(bucket)?;
            bucket.clear();
            bucket.shrink_to_fit();
        }
        self.held = 0;
        Ok(())
    }

    /// Gives every leaf to `sink` in address order. A bucket (1/256 of the
    /// leaves) is read back and sorted at a time.
    pub fn into_sorted(
        mut self,
        mut sink: impl FnMut(BaseLeaf) -> Result<(), String>,
    ) -> Result<(), String> {
        // One buffer for every bucket: they are about the same size.
        let mut bytes = Vec::new();
        for index in 0..SORTER_BUCKETS {
            bytes.clear();
            if self.spilled[index]
                && let Some(prefix) = &self.spill
            {
                let path = Self::spill_path(prefix, index);
                File::open(&path)
                    .and_then(|mut file| file.read_to_end(&mut bytes))
                    .map_err(|err| format!("cannot read {}: {err}", path.display()))?;
                let _ = fs::remove_file(&path);
                self.spilled[index] = false;
            }
            bytes.extend_from_slice(&self.buckets[index]);
            self.buckets[index] = Vec::new();
            let (leaves, rest) = bytes.as_chunks_mut::<BASE_LEAF_LEN>();
            if !rest.is_empty() {
                return Err("a spilled bucket is not a whole number of leaves".into());
            }
            on_base_pool(|| leaves.par_sort_unstable());
            for leaf in leaves.iter() {
                let (address, account_hash) = leaf.split_at(32);
                sink(BaseLeaf::new(
                    address.try_into().expect("an address"),
                    Hash::new_from_array(account_hash.try_into().expect("a hash")),
                ))?;
            }
        }
        Ok(())
    }
}

impl Drop for LeafSorter {
    fn drop(&mut self) {
        if let Some(prefix) = &self.spill {
            for (index, spilled) in self.spilled.iter().enumerate() {
                if *spilled {
                    let _ = fs::remove_file(Self::spill_path(prefix, index));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use {super::*, solana_sha256_hasher::hash, tempfile::TempDir};

    fn leaf(i: u64) -> BaseLeaf {
        // Addresses spread over the buckets, not in the order of `i`.
        BaseLeaf::new(hash(&i.to_le_bytes()).to_bytes(), hash(&i.to_be_bytes()))
    }

    fn id(epoch: Epoch) -> BaseId {
        BaseId {
            epoch,
            parent: Some((epoch * 10, hash(&epoch.to_le_bytes()))),
        }
    }

    /// Leaves written to a file are read back by a later `BaseDir`, give the
    /// same tree, and are removed with their epoch.
    #[test]
    fn test_files_written_reopened_and_removed() {
        let accounts = TempDir::new().unwrap();
        let run = accounts.path().join("run");
        fs::create_dir(&run).unwrap();
        let mut leaves: Vec<_> = (0..5_000).map(leaf).collect();
        leaves.sort_unstable();
        let in_memory = BaseTree::new(leaves.clone()).unwrap();

        let dir = BaseDir::for_accounts_path(&run);
        assert_eq!(dir.path().unwrap(), accounts.path().join(DIR_NAME));
        assert!(dir.open(&id(3)).is_none());
        let mut writer = dir.writer(&id(3));
        for leaf in &leaves {
            writer.push(*leaf).unwrap();
        }
        let tree = tree_from_storage(writer.finish().unwrap()).unwrap();
        assert_eq!(tree, in_memory);
        assert_eq!(
            tree.inclusion_proof(&leaves[4_321].address),
            in_memory.inclusion_proof(&leaves[4_321].address)
        );
        // A writer that is dropped leaves nothing, and one left by a process
        // that died is removed by the next.
        let mut abandoned = dir.writer(&id(4));
        abandoned.push(leaves[0]).unwrap();
        drop(abandoned);
        let leftover = accounts.path().join(DIR_NAME).join("x.1-1.tmp");
        fs::write(&leftover, b"half").unwrap();
        // An empty base.
        let empty = tree_from_storage(dir.writer(&id(5)).finish().unwrap()).unwrap();
        assert_eq!(empty, BaseTree::empty());

        let later = BaseDir::for_accounts_path(&run);
        let reopened = tree_from_storage(later.open(&id(3)).unwrap()).unwrap();
        assert_eq!(reopened, in_memory);
        assert!(!leftover.exists());
        assert!(later.open(&id(4)).is_none());
        assert_eq!(
            tree_from_storage(later.open(&id(5)).unwrap()).unwrap(),
            BaseTree::empty()
        );

        // The map stays readable after its file is removed.
        later.retain_epochs(|epoch| epoch >= 5);
        assert!(later.open(&id(3)).is_none());
        assert!(later.open(&id(5)).is_some());
        assert_eq!(reopened.to_vec(), leaves);
        later.remove(&id(5));
        assert!(later.open(&id(5)).is_none());
    }

    /// The sorter gives the same order whether it spills or not.
    #[test]
    fn test_sorter_spills_and_sorts() {
        let accounts = TempDir::new().unwrap();
        let dir = BaseDir::for_accounts_path(accounts.path());
        assert_eq!(dir.path().unwrap(), accounts.path().join(DIR_NAME));
        let leaves: Vec<_> = (0..20_000).map(leaf).collect();
        let mut expected = leaves.clone();
        expected.sort_unstable();
        // 20,000 leaves are 1.28 MB: with 100 kB held, about 13 spills.
        for budget in [100_000, usize::MAX] {
            let mut sorter = dir.sorter_with_budget(budget);
            for leaf in &leaves {
                sorter.push(&leaf.address, &leaf.account_hash).unwrap();
            }
            let spilled = sorter.spilled.iter().filter(|spilled| **spilled).count();
            assert_eq!(spilled > 0, budget != usize::MAX);
            let mut sorted = Vec::new();
            sorter
                .into_sorted(|leaf| {
                    sorted.push(leaf);
                    Ok(())
                })
                .unwrap();
            assert_eq!(sorted, expected);
        }
        // A sorter that is dropped removes what it spilled.
        let mut sorter = dir.sorter_with_budget(1_000);
        for leaf in &leaves[..1_000] {
            sorter.push(&leaf.address, &leaf.account_hash).unwrap();
        }
        drop(sorter);
        assert_eq!(fs::read_dir(dir.path().unwrap()).unwrap().count(), 0);
    }

    /// Peak resident memory of the process, in bytes.
    fn peak_memory() -> u64 {
        // SAFETY: `getrusage` fills the struct it is given.
        let usage = unsafe {
            let mut usage = std::mem::zeroed::<libc::rusage>();
            libc::getrusage(libc::RUSAGE_SELF, &mut usage);
            usage
        };
        // Bytes on macOS, kilobytes on Linux.
        (usage.ru_maxrss as u64) * if cfg!(target_os = "macos") { 1 } else { 1024 }
    }

    /// Times the three ways a node gets a base tree, at a size given by
    /// `BASE_TREE_BENCH_LEAVES` (default ten million accounts), and reports
    /// the peak memory of the process.
    ///
    /// ```text
    /// BASE_TREE_BENCH_LEAVES=100000000 cargo test --release -p solana-runtime \
    ///     --features dev-context-only-utils --lib bench_base_tree -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn bench_base_tree() {
        use {
            solana_merkle_tree::accounts_state::{
                apply_block_changes, apply_overlay_into, new_overlay,
            },
            std::time::Instant,
        };
        let count: u64 = std::env::var("BASE_TREE_BENCH_LEAVES")
            .ok()
            .and_then(|count| count.parse().ok())
            .unwrap_or(10_000_000);
        let changed = count / 100;
        let accounts = match std::env::var("BASE_TREE_BENCH_DIR") {
            Ok(dir) => TempDir::new_in(dir).unwrap(),
            Err(_) => TempDir::new().unwrap(),
        };
        let dir = BaseDir::for_accounts_path(accounts.path());
        // Addresses from a mixing function: spread like hashes, and cheap.
        let leaf = |i: u64| {
            let mut address = [0u8; 32];
            let mut x = i.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x0123_4567_89ab_cdef;
            for part in address.chunks_mut(8) {
                x ^= x >> 30;
                x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                x ^= x >> 27;
                x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
                x ^= x >> 31;
                part.copy_from_slice(&x.to_le_bytes());
            }
            let mut account_hash = address;
            account_hash[0] |= 1;
            (address, Hash::new_from_array(account_hash))
        };
        println!(
            "{count} accounts, {} threads, files in {}",
            BASE_POOL.current_num_threads(),
            accounts.path().display()
        );

        // By scan: leaves in the order a scan gives them, through the
        // sorter, to a file, then hashed.
        let start = Instant::now();
        let mut sorter = dir.sorter();
        for i in 0..count {
            let (address, account_hash) = leaf(i);
            sorter.push(&address, &account_hash).unwrap();
        }
        let pushed = start.elapsed();
        println!("spilled: peak memory {} MB", peak_memory() >> 20);
        let mut writer = dir.writer(&id(1));
        sorter
            .into_sorted(|leaf| writer.push(leaf).map_err(|err| err.to_string()))
            .unwrap();
        let storage = writer.finish().unwrap();
        let sorted = start.elapsed();
        // Before the file is read back through the map: what the process
        // holds itself. After, the peak includes the pages of the file that
        // the kernel keeps and can drop.
        println!("sorted and written: peak memory {} MB", peak_memory() >> 20);
        let tree = tree_from_storage(storage).unwrap();
        let scanned = start.elapsed();
        assert_eq!(tree.len() as u64, count);
        println!(
            "after a scan: spill {:.1} s, sort and write {:.1} s, hash {:.1} s; peak memory {} MB",
            pushed.as_secs_f64(),
            (sorted - pushed).as_secs_f64(),
            (scanned - sorted).as_secs_f64(),
            peak_memory() >> 20
        );

        // By merge: an overlay that changes 1% of the accounts, closes some
        // of them and creates as many new ones.
        let mut overlay = new_overlay();
        let no_account = Hash::default();
        for i in 0..changed {
            let (address, before) = leaf(i * 100);
            let after = if i % 10 == 0 {
                no_account
            } else {
                Hash::new_from_array([7; 32])
            };
            apply_block_changes(&mut overlay, [(&address, &before, &after)]);
            let (created, state) = leaf(count + i);
            apply_block_changes(&mut overlay, [(&created, &no_account, &state)]);
        }
        let start = Instant::now();
        let mut writer = dir.writer(&id(2));
        apply_overlay_into(&tree, &overlay, |leaf| writer.push(leaf)).unwrap();
        let storage = writer.finish().unwrap();
        let written = start.elapsed();
        let next = tree_from_storage(storage).unwrap();
        let merged = start.elapsed();
        assert_eq!(next.len() as u64, count + changed - changed.div_ceil(10));
        println!(
            "by merge with {} overlay entries: write {:.1} s, hash {:.1} s; peak memory {} MB",
            overlay.len(),
            written.as_secs_f64(),
            (merged - written).as_secs_f64(),
            peak_memory() >> 20
        );

        // After a restart: the file is opened and hashed.
        let root = next.root();
        drop(next);
        let start = Instant::now();
        let reopened = tree_from_storage(dir.open(&id(2)).unwrap()).unwrap();
        assert_eq!(reopened.root(), root);
        println!(
            "after a restart, from the file: {:.1} s",
            start.elapsed().as_secs_f64()
        );

        // Proofs, of accounts spread over the tree.
        let proofs = 1_000;
        let start = Instant::now();
        for i in 0..proofs {
            let (address, _) = leaf(i * (count / proofs) + 1);
            let verified = match reopened.inclusion_proof(&address) {
                Some(proof) => proof.verify(&root),
                None => reopened
                    .absence_proof(&address)
                    .unwrap()
                    .verify(&address, &root),
            };
            assert!(verified);
        }
        println!(
            "a proof, built and verified: {:.0} us; peak memory {} MB",
            start.elapsed().as_micros() as f64 / proofs as f64,
            peak_memory() >> 20
        );
    }

    /// Without a usable directory, leaves are held in memory.
    #[test]
    fn test_memory_without_directory() {
        let file = tempfile::NamedTempFile::new().unwrap();
        for dir in [BaseDir::default(), BaseDir::for_accounts_path(file.path())] {
            assert!(dir.path().is_none());
            let mut leaves: Vec<_> = (0..100).map(leaf).collect();
            let mut sorter = dir.sorter_with_budget(64);
            for leaf in &leaves {
                sorter.push(&leaf.address, &leaf.account_hash).unwrap();
            }
            let mut writer = dir.writer(&id(1));
            sorter
                .into_sorted(|leaf| writer.push(leaf).map_err(|err| err.to_string()))
                .unwrap();
            leaves.sort_unstable();
            assert_eq!(
                tree_from_storage(writer.finish().unwrap()).unwrap(),
                BaseTree::new(leaves).unwrap()
            );
            assert!(dir.open(&id(1)).is_none());
        }
    }
}
