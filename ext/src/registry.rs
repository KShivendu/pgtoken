//! Backend-local artefact registry, backed by mmap'd files.
//!
//! A vocabulary owns up to two optional artefacts, both named after its id and both resolved
//! under the `pgtoken.table_dir` GUC: `<vocabulary_id>.tntt`, the frequency ranking the `freq`
//! codec encodes against, and `<vocabulary_id>.tnmap`, the `token_id -> bytes` mapping
//! `pgtoken.text` detokenizes with. A vocabulary may have either, both or neither.
//!
//! Resolution is by filesystem convention rather than through a SQL catalog on purpose: a catalog
//! lookup would need SPI on every read, which would make `pgtoken.tokens`'s read paths —
//! `tokens_out`, `tokens_send`, and the `int[]`/`bytea` casts — neither `IMMUTABLE` nor
//! `PARALLEL SAFE`. As it stands, reading a value depends only on its bytes and a write-once
//! file, so declaring those functions `IMMUTABLE` is honest.
//!
//! Files are mmap'd read-only, so their pages are shared across every backend through the OS
//! page cache instead of being duplicated per connection.
//!
//! # The cache contract, stated exactly
//!
//! The parsed form is cached per backend, **keyed by the resolved path**, and entries are never
//! invalidated. So the guarantee is narrow and worth spelling out: a given path is read at most
//! once per backend, and every later read of that path in that backend answers from the first
//! read. Replacing, truncating or deleting a file at a path some live backend has already read
//! is **outside the contract** — that backend keeps answering from the bytes it saw, so two
//! backends can disagree about the same value, and an index built by one can contradict a
//! sequential scan run by the other. (`Mmap` makes truncation worse than stale: it is undefined,
//! not merely wrong.) Repointing `pgtoken.table_dir` is not a way around this; it is why the key
//! is the path and not the bare id, so a repointed directory is a cache miss rather than a
//! silent hit on the old directory's contents.
//!
//! What keeps the contract keepable is that nothing reachable from SQL can put a second file at
//! one path: `write_table` and `write_map` both refuse to overwrite, and `create_vocabulary`
//! refuses to hand out an id whose artefact files already exist — including the orphans a
//! rolled-back `load_mapping` leaves behind, since the file survives a `ROLLBACK` that removes
//! the catalog row. The two rules are halves of one guarantee: a path names one artefact for the
//! life of the cluster.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use memmap2::Mmap;
use pgrx::prelude::*;

use pgtoken_core::tables::{ByteMap, RankTable};

thread_local! {
    static CACHE: RefCell<HashMap<PathBuf, Rc<RankTable>>> = RefCell::new(HashMap::new());
    static MAP_CACHE: RefCell<HashMap<PathBuf, Rc<ByteMap>>> = RefCell::new(HashMap::new());
}

/// Directory holding a vocabulary's artefact files. Set via the `pgtoken.table_dir` GUC.
///
/// Empty when the GUC is unset. Prefer [`require_dir`] wherever an empty answer would be
/// mistaken for a real directory.
pub fn table_dir() -> PathBuf {
    PathBuf::from(crate::guc_str(&crate::TABLE_DIR, ""))
}

/// The artefact directory, or an error naming the GUC that is unset.
///
/// Every caller about to ask a question about an artefact goes through this rather than testing
/// `table_dir()` itself, because the failure it prevents is not an obvious one. `table_path` and
/// `map_path` join onto an *empty* path when the GUC is unset, yielding the bare relative name
/// `<id>.tnmap`; `.exists()` then resolves that against the data directory and answers false. A
/// caller that skipped this helper would read "unset" as "absent" and tell the user to create the
/// artefact — advice that, followed, seals a second mapping while the real one sits unreachable
/// in the directory nobody pointed at.
///
/// `purpose` completes the sentence "pgtoken.table_dir is not set; cannot ...".
pub fn require_dir(purpose: &str) -> Result<PathBuf, String> {
    let dir = table_dir();
    if dir.as_os_str().is_empty() {
        return Err(format!("pgtoken.table_dir is not set; cannot {purpose}"));
    }
    Ok(dir)
}

pub fn table_path(vocabulary_id: u16) -> PathBuf {
    table_dir().join(format!("{vocabulary_id}.tntt"))
}

/// Path of a vocabulary's `token_id -> bytes` mapping. Sits beside the ranking, which is
/// `<id>.tntt`; the two are separate files because a vocabulary may have either, both, or
/// neither.
pub fn map_path(vocabulary_id: u16) -> PathBuf {
    table_dir().join(format!("{vocabulary_id}.tnmap"))
}

/// The first artefact file already on disk for `vocabulary_id`, if any.
///
/// Asked by `create_vocabulary` before it hands out an id, because an artefact file outlives the
/// catalog row that ordered it: `load_mapping` writes through the filesystem, outside the
/// transaction, so a `ROLLBACK` takes the row and leaves the file. An id whose file is still
/// there would hand the next vocabulary a mapping it never loaded — and, the file being
/// write-once, no way to load its own.
///
/// Answers `None` when `pgtoken.table_dir` is unset, which is correct rather than evasive: with
/// no directory there is nowhere an artefact could have been written (every writer refuses first)
/// and nowhere to look — the bare relative name would be tested against the data directory.
pub fn existing_artefact(vocabulary_id: u16) -> Option<PathBuf> {
    if table_dir().as_os_str().is_empty() {
        return None;
    }
    [table_path(vocabulary_id), map_path(vocabulary_id)]
        .into_iter()
        .find(|p| p.exists())
}

/// Open and mmap an artefact file at `path`, or `Ok(None)` if there is none.
///
/// Shared by the ranking and the mapping, since both are plain write-once files under
/// `table_dir` that are read once and cached forever. Absence comes back as `Ok(None)` rather
/// than an error string because "this vocabulary has no ranking / no mapping" is a normal state
/// its callers each want to phrase for themselves — and because folding the question into the
/// `open` saves them a separate `stat()` on every row.
fn read_mmap_at(path: &Path) -> Result<Option<Mmap>, String> {
    require_dir("read vocabulary artefacts")?;
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("cannot open {}: {e}", path.display())),
    };
    // Safety: opened read-only and treated as immutable by contract — see the module docs on what
    // that contract does and does not cover. A concurrent truncation would be undefined, which is
    // why artefact files are write-once and a changed artefact needs a new id.
    unsafe { Mmap::map(&file) }
        .map(Some)
        .map_err(|e| format!("cannot mmap {}: {e}", path.display()))
}

/// Load a vocabulary's ranking, from the backend-local cache when possible. `Ok(None)` means the
/// vocabulary has no ranking.
pub fn rank_table(vocabulary_id: u16) -> Result<Option<Rc<RankTable>>, String> {
    let path = table_path(vocabulary_id);
    if let Some(hit) = CACHE.with(|c| c.borrow().get(&path).cloned()) {
        return Ok(Some(hit));
    }
    let Some(mmap) = read_mmap_at(&path)? else {
        return Ok(None);
    };
    let table = Rc::new(
        RankTable::from_bytes(&mmap)
            .map_err(|e| format!("coding table {vocabulary_id} is not valid: {e}"))?,
    );
    CACHE.with(|c| c.borrow_mut().insert(path, table.clone()));
    Ok(Some(table))
}

/// Load a vocabulary's mapping, from the backend-local cache when possible. `Ok(None)` means the
/// vocabulary has no mapping.
///
/// `pgtoken.text` is the only caller: it reads a mapping back to detokenize a stored value. It
/// leans on the `Ok(None)` arm rather than testing `map_path(...).exists()` first, which on a
/// warm backend would be one wasted `stat()` per row — a million of them on a large index build.
pub fn byte_map(vocabulary_id: u16) -> Result<Option<Rc<ByteMap>>, String> {
    let path = map_path(vocabulary_id);
    if let Some(hit) = MAP_CACHE.with(|c| c.borrow().get(&path).cloned()) {
        return Ok(Some(hit));
    }
    let Some(mmap) = read_mmap_at(&path)? else {
        return Ok(None);
    };
    let m = Rc::new(
        ByteMap::from_bytes(&mmap)
            .map_err(|e| format!("mapping for vocabulary {vocabulary_id} is not valid: {e}"))?,
    );
    MAP_CACHE.with(|c| c.borrow_mut().insert(path, m.clone()));
    Ok(Some(m))
}

/// Write a freshly trained table to `$table_dir/<vocabulary_id>.tntt`.
///
/// Refuses to overwrite: a `vocabulary_id` is a permanent name for a specific table, because
/// stored values reference it and `decode` is declared `IMMUTABLE`. Silently replacing one
/// would change what existing rows decode to.
///
/// This is the weaker of the two writers and should adopt [`write_map`]'s pattern — temp file,
/// `sync_all`, `hard_link` into place — which fixes both of the problems left here, just not as
/// part of this change. First, TOCTOU: the `exists()` check below and the `write` that follows it
/// are two separate syscalls, so two concurrent `train` calls on one fresh vocabulary id can both
/// pass the check and race to write. Second, and worse, the create is not atomic with its
/// content: a crash or `ENOSPC` mid-`write` leaves a truncated `.tntt` that `train` then refuses
/// to replace.
pub fn write_table(vocabulary_id: u16, bytes: &[u8]) -> Result<PathBuf, String> {
    let dir = require_dir("write a vocabulary ranking")?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let path = table_path(vocabulary_id);
    if path.exists() {
        return Err(format!(
            "coding table {vocabulary_id} already exists at {}; stored values reference \
             vocabulary ids, so pick an unused id rather than replacing one",
            path.display()
        ));
    }
    std::fs::write(&path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(path)
}

/// Write a freshly built mapping. Refuses to overwrite, for the same reason `write_table` does.
///
/// The create is atomic **with its content**: the bytes go to a temp file in the same directory,
/// are `sync_all`'d, and only then get `hard_link`ed to the final name. Three properties fall out
/// of that ordering:
///
/// * Write-once survives the split. `hard_link` fails with `AlreadyExists` if the final name is
///   taken, so the check and the create remain one atomic syscall, exactly as `create_new` was —
///   two concurrent `load_mapping` calls on one fresh vocabulary still cannot both win. `rename`,
///   the more familiar idiom, is wrong here for precisely this reason: it clobbers.
/// * `<id>.tnmap` never exists while short. It appears only once its bytes are on disk, so the
///   truncated-file state that `ENOSPC` or a crash used to leave — refused by `load_mapping` as
///   "already exists", rejected by `pgtoken.text` as corrupt, and with no SQL remedy for either —
///   is unreachable.
/// * A crash before the link leaves no mapping at all, which is a state `load_mapping` can simply
///   be run again from. The temp is removed on every path, success or failure; the directory is
///   deliberately not fsync'd, because losing the link entirely lands on that same re-runnable
///   state while losing the *content* would not.
pub fn write_map(vocabulary_id: u16, bytes: &[u8]) -> Result<PathBuf, String> {
    let dir = require_dir("write a vocabulary mapping")?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let path = map_path(vocabulary_id);

    // Dotted and suffixed so it can never be mistaken for an artefact, and stamped with the
    // backend pid plus a clock reading so a temp left behind by a crashed backend cannot collide
    // with this one's.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = dir.join(format!(
        ".{vocabulary_id}.tnmap.{}.{stamp}.tmp",
        std::process::id()
    ));

    let staged = (|| -> Result<(), String> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|e| {
                format!(
                    "cannot create temporary mapping file {}: {e}",
                    tmp.display()
                )
            })?;
        file.write_all(bytes)
            .map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
        file.sync_all()
            .map_err(|e| format!("cannot flush {} to disk: {e}", tmp.display()))
    })();
    if let Err(e) = staged {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }

    let linked = std::fs::hard_link(&tmp, &path).map_err(|e| {
        // `hard_link` fails for plenty of reasons besides a real collision -- permission denied,
        // a vanished parent directory, a filesystem without hard links -- and only
        // `AlreadyExists` means "pick a different id". Reporting every failure as a collision
        // would send an operator chasing the one fix that cannot possibly help; this matters
        // especially here, since managed PostgreSQL is exactly where permission and directory
        // problems turn up.
        if e.kind() == std::io::ErrorKind::AlreadyExists {
            format!(
                "mapping for vocabulary {vocabulary_id} already exists at {}; stored values \
                 reference vocabulary ids, so pick an unused id rather than replacing one",
                path.display()
            )
        } else {
            format!("cannot create mapping file {}: {e}", path.display())
        }
    });
    // Unlink the temp whichever way the link went: on success the content now lives under its
    // real name, on failure there is nothing to keep.
    let _ = std::fs::remove_file(&tmp);
    linked?;
    Ok(path)
}

/// Report what a ranking file contains, without adding it to the cache. `Ok(None)` if there is
/// no ranking.
pub fn describe_table(vocabulary_id: u16) -> Result<Option<(u32, String, u64)>, String> {
    let Some(mmap) = read_mmap_at(&table_path(vocabulary_id))? else {
        return Ok(None);
    };
    let bytes: &[u8] = &mmap;
    let table = RankTable::from_bytes(bytes)
        .map_err(|e| format!("coding table {vocabulary_id} is not valid: {e}"))?;
    let digest = pgtoken_core::tables::digest_hex(&pgtoken_core::tables::table_digest(bytes));
    Ok(Some((table.k(), digest, bytes.len() as u64)))
}

/// Report what a mapping contains, without adding it to the cache. `Ok(None)` if there is no
/// mapping.
pub fn describe_map(vocabulary_id: u16) -> Result<Option<(u32, String, u64)>, String> {
    let Some(mmap) = read_mmap_at(&map_path(vocabulary_id))? else {
        return Ok(None);
    };
    let bytes: &[u8] = &mmap;
    let map = ByteMap::from_bytes(bytes)
        .map_err(|e| format!("mapping for vocabulary {vocabulary_id} is not valid: {e}"))?;
    let digest = pgtoken_core::tables::digest_hex(&pgtoken_core::tables::table_digest(bytes));
    Ok(Some((map.mapped(), digest, bytes.len() as u64)))
}

/// Turn a core-crate error into a Postgres `ERROR`.
pub fn bail(msg: impl std::fmt::Display) -> ! {
    error!("{msg}");
}
