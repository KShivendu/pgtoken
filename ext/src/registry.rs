//! Backend-local coding-table registry, backed by mmap'd files.
//!
//! A value's header names its coding table by a small `table_id`, which this module resolves
//! to `$pgtoken.table_dir/<table_id>.tntt`. Resolution is by filesystem convention rather than
//! through a SQL catalog on purpose: a catalog lookup would need SPI on every decode, which
//! would make `pgtoken.decode` neither `IMMUTABLE` nor `PARALLEL SAFE`. As it stands, decoding
//! depends only on the value's bytes and an append-only file, so declaring it `IMMUTABLE` is
//! honest.
//!
//! Files are mmap'd read-only, so their pages are shared across every backend through the OS
//! page cache instead of being duplicated per connection. The parsed form is cached per
//! backend and never invalidated, because a `table_id`'s file is immutable by contract — a
//! changed table is a new id.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::rc::Rc;

use memmap2::Mmap;
use pgrx::prelude::*;

use pgtoken_core::tables::RankTable;

thread_local! {
    static CACHE: RefCell<HashMap<u16, Rc<RankTable>>> = RefCell::new(HashMap::new());
}

/// Directory holding `<table_id>.tntt` files. Set via the `pgtoken.table_dir` GUC.
pub fn table_dir() -> PathBuf {
    PathBuf::from(crate::guc_str(&crate::TABLE_DIR, ""))
}

pub fn table_path(table_id: u16) -> PathBuf {
    table_dir().join(format!("{table_id}.tntt"))
}

fn read_mmap(table_id: u16) -> Result<Mmap, String> {
    let dir = table_dir();
    if dir.as_os_str().is_empty() {
        return Err("pgtoken.table_dir is not set; the freq codec needs a coding table".into());
    }
    let path = table_path(table_id);
    let file = File::open(&path)
        .map_err(|e| format!("cannot open coding table {}: {e}", path.display()))?;
    // Safety: opened read-only and treated as immutable by contract. A concurrent truncation
    // would be undefined, which is why table files are write-once and a changed table gets a
    // new id.
    unsafe { Mmap::map(&file) }
        .map_err(|e| format!("cannot mmap coding table {}: {e}", path.display()))
}

/// Load a coding table, from the backend-local cache when possible.
pub fn rank_table(table_id: u16) -> Result<Rc<RankTable>, String> {
    if let Some(hit) = CACHE.with(|c| c.borrow().get(&table_id).cloned()) {
        return Ok(hit);
    }
    let mmap = read_mmap(table_id)?;
    let table = Rc::new(
        RankTable::from_bytes(&mmap)
            .map_err(|e| format!("coding table {table_id} is not valid: {e}"))?,
    );
    CACHE.with(|c| c.borrow_mut().insert(table_id, table.clone()));
    Ok(table)
}

/// Write a freshly trained table to `$table_dir/<table_id>.tntt`.
///
/// Refuses to overwrite: a `table_id` is a permanent name for a specific table, because stored
/// values reference it and `decode` is declared `IMMUTABLE`. Silently replacing one would
/// change what existing rows decode to.
pub fn write_table(table_id: u16, bytes: &[u8]) -> Result<PathBuf, String> {
    let dir = table_dir();
    if dir.as_os_str().is_empty() {
        return Err("pgtoken.table_dir is not set".into());
    }
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let path = table_path(table_id);
    if path.exists() {
        return Err(format!(
            "coding table {table_id} already exists at {}; stored values reference table ids, \
             so pick an unused id rather than replacing one",
            path.display()
        ));
    }
    std::fs::write(&path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(path)
}

/// Report what a table file contains, without adding it to the cache.
pub fn describe_table(table_id: u16) -> Result<(u32, String, u64), String> {
    let mmap = read_mmap(table_id)?;
    let bytes: &[u8] = &mmap;
    let table = RankTable::from_bytes(bytes)
        .map_err(|e| format!("coding table {table_id} is not valid: {e}"))?;
    let digest = pgtoken_core::tables::digest_hex(&pgtoken_core::tables::table_digest(bytes));
    Ok((table.k(), digest, bytes.len() as u64))
}

/// Turn a core-crate error into a Postgres `ERROR`.
pub fn bail(msg: impl std::fmt::Display) -> ! {
    error!("{msg}");
}
