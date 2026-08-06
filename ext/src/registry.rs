//! Backend-local coding-table registry, backed by mmap'd files.
//!
//! A value's header names its coding table by a small `vocabulary_id`, which this module
//! resolves to `$pgtoken.table_dir/<vocabulary_id>.tntt`. Resolution is by filesystem
//! convention rather than through a SQL catalog on purpose: a catalog lookup would need SPI on
//! every read, which would make `pgtoken.tokens`'s read paths — `tokens_out`, `tokens_send`, and
//! the `int[]`/`bytea` casts — neither `IMMUTABLE` nor `PARALLEL SAFE`. As it stands, reading a
//! value depends only on its bytes and an append-only file, so declaring those functions
//! `IMMUTABLE` is honest.
//!
//! Files are mmap'd read-only, so their pages are shared across every backend through the OS
//! page cache instead of being duplicated per connection. The parsed form is cached per
//! backend and never invalidated, because a `vocabulary_id`'s file is immutable by contract —
//! a changed table is a new id.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::rc::Rc;

use memmap2::Mmap;
use pgrx::prelude::*;

use pgtoken_core::tables::{ByteMap, RankTable};

thread_local! {
    static CACHE: RefCell<HashMap<u16, Rc<RankTable>>> = RefCell::new(HashMap::new());
    static MAP_CACHE: RefCell<HashMap<u16, Rc<ByteMap>>> = RefCell::new(HashMap::new());
}

/// Directory holding `<vocabulary_id>.tntt` files. Set via the `pgtoken.table_dir` GUC.
pub fn table_dir() -> PathBuf {
    PathBuf::from(crate::guc_str(&crate::TABLE_DIR, ""))
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

/// Open and mmap an artefact file at `path`. Shared by the ranking and the mapping, since both
/// are plain append-only files under `table_dir` that are read once and cached forever.
fn read_mmap_at(path: PathBuf) -> Result<Mmap, String> {
    if table_dir().as_os_str().is_empty() {
        return Err("pgtoken.table_dir is not set; needed to read vocabulary artefacts".into());
    }
    let file = File::open(&path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    // Safety: opened read-only and treated as immutable by contract. A concurrent truncation
    // would be undefined, which is why artefact files are write-once and a changed artefact gets
    // a new id.
    unsafe { Mmap::map(&file) }.map_err(|e| format!("cannot mmap {}: {e}", path.display()))
}

/// Load a coding table, from the backend-local cache when possible.
pub fn rank_table(vocabulary_id: u16) -> Result<Rc<RankTable>, String> {
    if let Some(hit) = CACHE.with(|c| c.borrow().get(&vocabulary_id).cloned()) {
        return Ok(hit);
    }
    let mmap = read_mmap_at(table_path(vocabulary_id))?;
    let table = Rc::new(
        RankTable::from_bytes(&mmap)
            .map_err(|e| format!("coding table {vocabulary_id} is not valid: {e}"))?,
    );
    CACHE.with(|c| c.borrow_mut().insert(vocabulary_id, table.clone()));
    Ok(table)
}

/// Load a mapping, from the backend-local cache when possible.
///
/// Cached and never invalidated, on the same contract as the ranking: the file is write-once, so
/// a changed mapping is a new vocabulary.
///
/// Unused for now: this task only loads and stores mappings. `pgtoken.text`, the follow-up task
/// that reads them back, is `byte_map`'s only caller.
#[allow(dead_code)]
pub fn byte_map(vocabulary_id: u16) -> Result<Rc<ByteMap>, String> {
    if let Some(hit) = MAP_CACHE.with(|c| c.borrow().get(&vocabulary_id).cloned()) {
        return Ok(hit);
    }
    let mmap = read_mmap_at(map_path(vocabulary_id))?;
    let m = Rc::new(
        ByteMap::from_bytes(&mmap)
            .map_err(|e| format!("mapping for vocabulary {vocabulary_id} is not valid: {e}"))?,
    );
    MAP_CACHE.with(|c| c.borrow_mut().insert(vocabulary_id, m.clone()));
    Ok(m)
}

/// Write a freshly trained table to `$table_dir/<vocabulary_id>.tntt`.
///
/// Refuses to overwrite: a `vocabulary_id` is a permanent name for a specific table, because
/// stored values reference it and `decode` is declared `IMMUTABLE`. Silently replacing one
/// would change what existing rows decode to.
///
/// TOCTOU: the `exists()` check below and the `write` that follows it are two separate syscalls,
/// so two concurrent `train` calls on one fresh vocabulary id can both pass the check and race to
/// write. `write_map` closes this with `OpenOptions::create_new`; this one wants the same
/// treatment, just not as part of this change.
pub fn write_table(vocabulary_id: u16, bytes: &[u8]) -> Result<PathBuf, String> {
    let dir = table_dir();
    if dir.as_os_str().is_empty() {
        return Err("pgtoken.table_dir is not set".into());
    }
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
/// Uses `create_new` rather than `exists()`-then-`write`, so the check and the write are one
/// atomic syscall: two concurrent `load_mapping` calls on one fresh vocabulary cannot both pass a
/// separate existence check and race to write, the way `write_table` above still can.
pub fn write_map(vocabulary_id: u16, bytes: &[u8]) -> Result<PathBuf, String> {
    let dir = table_dir();
    if dir.as_os_str().is_empty() {
        return Err("pgtoken.table_dir is not set".into());
    }
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let path = map_path(vocabulary_id);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| {
            format!(
                "mapping for vocabulary {vocabulary_id} already exists at {}; stored values \
                 reference vocabulary ids, so pick an unused id rather than replacing one: {e}",
                path.display()
            )
        })?;
    file.write_all(bytes)
        .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(path)
}

/// Report what a table file contains, without adding it to the cache.
pub fn describe_table(vocabulary_id: u16) -> Result<(u32, String, u64), String> {
    let mmap = read_mmap_at(table_path(vocabulary_id))?;
    let bytes: &[u8] = &mmap;
    let table = RankTable::from_bytes(bytes)
        .map_err(|e| format!("coding table {vocabulary_id} is not valid: {e}"))?;
    let digest = pgtoken_core::tables::digest_hex(&pgtoken_core::tables::table_digest(bytes));
    Ok((table.k(), digest, bytes.len() as u64))
}

/// Report what a mapping contains, without adding it to the cache.
pub fn describe_map(vocabulary_id: u16) -> Result<(u32, String, u64), String> {
    let mmap = read_mmap_at(map_path(vocabulary_id))?;
    let bytes: &[u8] = &mmap;
    let map = ByteMap::from_bytes(bytes)
        .map_err(|e| format!("mapping for vocabulary {vocabulary_id} is not valid: {e}"))?;
    let digest = pgtoken_core::tables::digest_hex(&pgtoken_core::tables::table_digest(bytes));
    Ok((map.mapped(), digest, bytes.len() as u64))
}

/// Turn a core-crate error into a Postgres `ERROR`.
pub fn bail(msg: impl std::fmt::Display) -> ! {
    error!("{msg}");
}
