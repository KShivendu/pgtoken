//! Backend-local coding-table registry, backed by mmap'd files.
//!
//! A value's header names its coding table by a small `table_id`, which this module resolves
//! to `$pgtoken.table_dir/<table_id>.tntt`. Resolution is by filesystem convention rather
//! than through a SQL catalog on purpose: a catalog lookup would need SPI on every decode,
//! which would make `pgtoken.decode` neither `IMMUTABLE` nor `PARALLEL SAFE`. As it stands,
//! decoding a value depends only on the value's bytes and an append-only file, so declaring
//! it `IMMUTABLE` is honest. Table files are content-addressed and must never be edited in
//! place; write a new `table_id` instead.
//!
//! Files are mmap'd read-only, so the pages are shared across every backend through the OS
//! page cache instead of being duplicated per connection. The parsed form is cached per
//! backend, keyed by `(table_id, tokenizer, kind)`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::rc::Rc;

use memmap2::Mmap;
use pgrx::prelude::*;

use pgtoken_core::header::Tokenizer;
use pgtoken_core::tables::{AnsTable, RankTable};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Rank,
    Ans,
}

impl Kind {
    fn as_str(self) -> &'static str {
        match self {
            Kind::Rank => "rank",
            Kind::Ans => "ans",
        }
    }
}

#[derive(Clone)]
pub enum Loaded {
    Rank(Rc<RankTable>),
    Ans(Rc<AnsTable>),
}

thread_local! {
    /// Parsed tables, per backend. Never invalidated, because a `table_id`'s file is
    /// immutable by contract; a changed table is a new id.
    static CACHE: RefCell<HashMap<(u16, Tokenizer, Kind), Loaded>> =
        RefCell::new(HashMap::new());
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
        return Err(
            "pgtoken.table_dir is not set; the +freq and +ANS codecs need a coding table"
                .into(),
        );
    }
    let path = table_path(table_id);
    let file = File::open(&path)
        .map_err(|e| format!("cannot open coding table {}: {e}", path.display()))?;
    // Safety: the file is opened read-only and treated as immutable by contract. A
    // concurrent truncation would be undefined, which is why table files are append-only
    // and a changed table gets a new id.
    unsafe { Mmap::map(&file) }
        .map_err(|e| format!("cannot mmap coding table {}: {e}", path.display()))
}

fn load(table_id: u16, tokenizer: Tokenizer, kind: Kind) -> Result<Loaded, String> {
    if let Some(hit) =
        CACHE.with(|c| c.borrow().get(&(table_id, tokenizer, kind)).cloned())
    {
        return Ok(hit);
    }
    let mmap = read_mmap(table_id)?;
    let loaded = match kind {
        Kind::Rank => Loaded::Rank(Rc::new(
            RankTable::from_bytes(&mmap, tokenizer)
                .map_err(|e| format!("coding table {table_id} is not a valid rank table: {e}"))?,
        )),
        Kind::Ans => Loaded::Ans(Rc::new(
            AnsTable::from_bytes(&mmap, tokenizer)
                .map_err(|e| format!("coding table {table_id} is not a valid ANS table: {e}"))?,
        )),
    };
    CACHE.with(|c| {
        c.borrow_mut().insert((table_id, tokenizer, kind), loaded.clone());
    });
    Ok(loaded)
}

pub fn rank_table(table_id: u16, tokenizer: Tokenizer) -> Result<Rc<RankTable>, String> {
    match load(table_id, tokenizer, Kind::Rank)? {
        Loaded::Rank(t) => Ok(t),
        Loaded::Ans(_) => Err(format!("coding table {table_id} is an ANS table, not a rank table")),
    }
}

pub fn ans_table(table_id: u16, tokenizer: Tokenizer) -> Result<Rc<AnsTable>, String> {
    match load(table_id, tokenizer, Kind::Ans)? {
        Loaded::Ans(t) => Ok(t),
        Loaded::Rank(_) => Err(format!("coding table {table_id} is a rank table, not an ANS table")),
    }
}

/// Write a freshly trained table to `$table_dir/<table_id>.tntt`.
///
/// Refuses to overwrite: a `table_id` is a permanent name for a specific table, because
/// stored values reference it and `decode` is declared `IMMUTABLE`. Silently replacing
/// one would change what existing rows decode to.
pub fn write_table(table_id: u16, bytes: &[u8], kind: Kind) -> Result<PathBuf, String> {
    let dir = table_dir();
    if dir.as_os_str().is_empty() {
        return Err("pgtoken.table_dir is not set".into());
    }
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let path = table_path(table_id);
    if path.exists() {
        return Err(format!(
            "coding table {table_id} already exists at {}; stored values reference table ids, \
             so pick an unused id instead of replacing one ({} table)",
            path.display(),
            kind.as_str()
        ));
    }
    std::fs::write(&path, bytes)
        .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(path)
}

/// Report what a table file contains, without adding it to the cache.
pub fn describe_table(table_id: u16) -> Result<(String, String, u32, String, u64), String> {
    let mmap = read_mmap(table_id)?;
    let bytes: &[u8] = &mmap;
    if bytes.len() < pgtoken_core::tables::TABLE_HEADER_LEN {
        return Err(format!("coding table {table_id} is truncated"));
    }
    let kind = match bytes[5] {
        1 => "rank",
        2 => "ans",
        other => return Err(format!("coding table {table_id} has unknown kind {other}")),
    };
    let tokenizer = Tokenizer::from_u8(bytes[6])
        .map_err(|e| format!("coding table {table_id}: {e}"))?;
    let vocab = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let digest = pgtoken_core::tables::digest_hex(&pgtoken_core::tables::table_digest(bytes));
    Ok((kind.to_string(), tokenizer.as_str().to_string(), vocab, digest, bytes.len() as u64))
}

/// Turn a core-crate error into a Postgres `ERROR`.
pub fn bail(msg: impl std::fmt::Display) -> ! {
    error!("{msg}");
}
