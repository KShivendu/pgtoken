//! Where a vocabulary's artefacts live, and the per-backend cache in front of them.
//!
//! A vocabulary owns up to two optional artefacts: the frequency ranking the `freq` codec encodes
//! against, and the `token_id -> bytes` mapping `pgtoken.text` detokenizes with (formats in
//! `core/src/tables.rs`). They come from one of two places:
//!
//! * **A bundled preset** (`preset.rs`): compiled into the binary, found by the vocabulary id's
//!   reserved range. Nothing in the database is consulted.
//! * **The database**, for artefacts made by `train` and `load_mapping`. Each one is a SQL
//!   function, `pgtoken.artefact_<id>_ranking()` or `pgtoken.artefact_<id>_mapping()`, returning
//!   the bytes as a constant.
//!
//! # Why a function and not a table
//!
//! A function is a schema object. `pg_dump` writes it in the pre-data section, which every restore
//! finishes before it loads any table data, `pg_restore -j` included. A table row would load in
//! the data section instead, in no fixed order relative to the user tables whose `freq` values
//! need it to parse: tables in a schema that sorts before `pgtoken` load first, and `-j` loads them
//! in parallel. A function created after `CREATE EXTENSION` is not an extension member, so
//! `pg_dump` includes it. It also gets everything else a catalog row gets for free: it is
//! transactional (a rolled-back `train` leaves nothing), WAL-logged (so durable and on every
//! physical and logical replica's schema), and private to its database (ids are per database).
//!
//! Write-once is kept by `CREATE FUNCTION` itself, which refuses an existing name. Two concurrent
//! `train` calls on one vocabulary serialize on `pg_proc`'s unique index, and the loser errors.
//!
//! # The cache
//!
//! Parsed artefacts and declared sizes are cached per backend, keyed by vocabulary id, so a warm
//! encode or decode does a hash lookup and no SPI. Entries are dropped whenever this backend sees
//! a `pg_proc` invalidation (a `CREATE` or `DROP FUNCTION` anywhere in the database, including
//! one rolled back, at top level or in a savepoint) and at every transaction abort. A rolled-back
//! `train` therefore cannot leave a ranking behind in the cache that the catalog no longer has.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::thread::LocalKey;

use pgrx::prelude::*;

use pgtoken_core::tables::{digest_hex, table_digest, ByteMap, RankTable};

use crate::preset;
use crate::tokenizer::Tokenizer;

/// Which artefact. The `&str` is both the function-name suffix and the word in messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Artefact {
    Ranking,
    Mapping,
}

impl Artefact {
    pub fn as_str(self) -> &'static str {
        match self {
            Artefact::Ranking => "ranking",
            Artefact::Mapping => "mapping",
        }
    }
}

/// The unqualified name of the function holding a custom artefact.
pub fn function_name(vocabulary_id: u16, a: Artefact) -> String {
    format!("artefact_{vocabulary_id}_{}", a.as_str())
}

// ── the cache ─────────────────────────────────────────────────────────────────────────────

type Cache<T> = RefCell<HashMap<u16, (u64, T)>>;

thread_local! {
    /// Bumped by every invalidation. An entry is live only while its stamp equals this, so an
    /// invalidation never has to borrow a map, and cannot fail even if it lands mid-lookup.
    static GENERATION: Cell<u64> = const { Cell::new(0) };
    /// Bumped at every transaction end. Stamps the "no catalog row" answers, which are only
    /// trusted for the rest of the transaction that saw them.
    static XACT: Cell<u64> = const { Cell::new(0) };
    static RANKINGS: Cache<Rc<RankTable>> = RefCell::new(HashMap::new());
    static MAPPINGS: Cache<Rc<ByteMap>> = RefCell::new(HashMap::new());
    static SIZES: Cache<u32> = RefCell::new(HashMap::new());
    /// Vocabulary ids with no catalog row, stamped with the transaction that looked.
    static MISSING: Cache<u64> = RefCell::new(HashMap::new());
}

fn generation() -> u64 {
    GENERATION.with(|g| g.get())
}

/// Forget everything cached. Safe to call from an invalidation callback at any moment.
pub fn invalidate() {
    GENERATION.with(|g| g.set(g.get().wrapping_add(1)));
}

fn get<T: Clone>(cache: &'static LocalKey<Cache<T>>, id: u16) -> Option<T> {
    let now = generation();
    cache.with(|c| {
        c.borrow()
            .get(&id)
            .filter(|(stamp, _)| *stamp == now)
            .map(|(_, v)| v.clone())
    })
}

/// `stamp` is the generation read *before* loading, so an invalidation that arrived during the
/// load (SPI takes locks, and taking a lock processes invalidations) leaves the entry stale.
fn put<T>(cache: &'static LocalKey<Cache<T>>, id: u16, stamp: u64, value: T) {
    cache.with(|c| c.borrow_mut().insert(id, (stamp, value)));
}

unsafe extern "C-unwind" fn on_proc_invalidation(_arg: pg_sys::Datum, _cacheid: i32, _hash: u32) {
    invalidate();
}

unsafe extern "C-unwind" fn on_xact_event(
    event: pg_sys::XactEvent::Type,
    _arg: *mut core::ffi::c_void,
) {
    use pg_sys::XactEvent::*;
    match event {
        XACT_EVENT_ABORT | XACT_EVENT_PARALLEL_ABORT => {
            invalidate();
            XACT.with(|x| x.set(x.get().wrapping_add(1)));
        }
        XACT_EVENT_COMMIT | XACT_EVENT_PARALLEL_COMMIT | XACT_EVENT_PREPARE => {
            XACT.with(|x| x.set(x.get().wrapping_add(1)));
        }
        _ => {}
    }
}

/// Hook the cache up to PostgreSQL's invalidation and transaction events. Called from `_PG_init`.
pub fn register_callbacks() {
    // Safety: both callbacks only touch thread-local `Cell`s, never allocate in a memory context
    // and never raise, so they are safe to run wherever PostgreSQL calls them.
    unsafe {
        pg_sys::CacheRegisterSyscacheCallback(
            pg_sys::SysCacheIdentifier::PROCOID as i32,
            Some(on_proc_invalidation),
            pg_sys::Datum::from(0usize),
        );
        pg_sys::RegisterXactCallback(Some(on_xact_event), std::ptr::null_mut());
    }
}

// ── reading ───────────────────────────────────────────────────────────────────────────────

/// A custom artefact's bytes from its function, or `None` if it has none. One SPI round trip
/// to ask, one to fetch; only ever on a cache miss.
fn fetch(vocabulary_id: u16, a: Artefact) -> Result<Option<Vec<u8>>, String> {
    let name = function_name(vocabulary_id, a);
    let signature = format!("pgtoken.{name}()");
    let present = Spi::get_one_with_args::<bool>(
        "SELECT pg_catalog.to_regprocedure($1) IS NOT NULL",
        &[signature.as_str().into()],
    )
    .map_err(|e| e.to_string())?
    .unwrap_or(false);
    if !present {
        return Ok(None);
    }
    Spi::get_one::<Vec<u8>>(&format!("SELECT pgtoken.{name}()")).map_err(|e| {
        format!(
            "cannot read the {} of vocabulary {vocabulary_id}: {e}",
            a.as_str()
        )
    })
}

/// Whether a custom artefact exists. Presets are not asked about here; they are never custom.
pub fn exists(vocabulary_id: u16, a: Artefact) -> bool {
    let signature = format!("pgtoken.{}()", function_name(vocabulary_id, a));
    Spi::get_one_with_args::<bool>(
        "SELECT pg_catalog.to_regprocedure($1) IS NOT NULL",
        &[signature.as_str().into()],
    )
    .unwrap_or_else(|e| bail(e))
    .unwrap_or(false)
}

/// An artefact's raw bytes, preset or custom, for `vocabulary_info` and `vocabulary_codec`.
pub fn artefact_bytes(vocabulary_id: u16, a: Artefact) -> Result<Option<Vec<u8>>, String> {
    if let Some(p) = preset::for_id(vocabulary_id) {
        return Ok(Some(match a {
            Artefact::Ranking => p.ranking_bytes(),
            Artefact::Mapping => p.mapping_bytes(),
        }));
    }
    fetch(vocabulary_id, a)
}

/// A vocabulary's ranking, from the backend-local cache when possible. `Ok(None)` means the
/// vocabulary has no ranking.
pub fn rank_table(vocabulary_id: u16) -> Result<Option<Rc<RankTable>>, String> {
    if let Some(p) = preset::for_id(vocabulary_id) {
        return Ok(Some(p.ranking()));
    }
    if let Some(hit) = get(&RANKINGS, vocabulary_id) {
        return Ok(Some(hit));
    }
    let stamp = generation();
    let Some(bytes) = fetch(vocabulary_id, Artefact::Ranking)? else {
        return Ok(None);
    };
    let t = Rc::new(
        RankTable::from_bytes(&bytes)
            .map_err(|e| format!("ranking of vocabulary {vocabulary_id} is not valid: {e}"))?,
    );
    put(&RANKINGS, vocabulary_id, stamp, t.clone());
    Ok(Some(t))
}

/// A vocabulary's mapping, from the backend-local cache when possible. `Ok(None)` means the
/// vocabulary has no mapping.
pub fn byte_map(vocabulary_id: u16) -> Result<Option<Rc<ByteMap>>, String> {
    if let Some(p) = preset::for_id(vocabulary_id) {
        return Ok(Some(p.mapping()));
    }
    if let Some(hit) = get(&MAPPINGS, vocabulary_id) {
        return Ok(Some(hit));
    }
    let stamp = generation();
    let Some(bytes) = fetch(vocabulary_id, Artefact::Mapping)? else {
        return Ok(None);
    };
    let m = Rc::new(
        ByteMap::from_bytes(&bytes)
            .map_err(|e| format!("mapping of vocabulary {vocabulary_id} is not valid: {e}"))?,
    );
    put(&MAPPINGS, vocabulary_id, stamp, m.clone());
    Ok(Some(m))
}

/// A vocabulary's server tokenizer, or `Ok(None)` if text for it has to be tokenized by the client.
///
/// The one place every text path asks — text input, the `text -> tokens` cast, comparing a column
/// with a text literal, `pgtoken.tokenize` — so none of them names a particular tokenizer. Only
/// the o200k preset has one today. A tokenizer stored in the database (an artefact function, like
/// a ranking) would be fetched and cached here, keyed by vocabulary id.
pub fn tokenizer(vocabulary_id: u16) -> Result<Option<Rc<dyn Tokenizer>>, String> {
    Ok(preset::for_id(vocabulary_id).and_then(|p| p.tokenizer()))
}

/// A vocabulary's declared `vocab_size`, for the bound check on text and `int[]` input.
///
/// `None` while a `pg_dump` restore is still loading: a table in a schema that sorts before
/// `pgtoken` loads its rows before `pgtoken.vocabulary`'s. That answer is cached for the rest of
/// the transaction only, so a restore does not pay one SPI call per row, and a later transaction
/// sees the row once it has loaded.
pub fn vocab_size(vocabulary_id: u16) -> Option<u32> {
    if let Some(p) = preset::for_id(vocabulary_id) {
        return Some(p.vocab_size);
    }
    if let Some(size) = get(&SIZES, vocabulary_id) {
        return Some(size);
    }
    let xact = XACT.with(|x| x.get());
    if get(&MISSING, vocabulary_id) == Some(xact) {
        return None;
    }
    let stamp = generation();
    let size = crate::vocabulary::vocab_size_for(vocabulary_id);
    match size {
        Some(s) => put(&SIZES, vocabulary_id, stamp, s),
        None => put(&MISSING, vocabulary_id, stamp, xact),
    }
    size
}

/// Size, sha256 and a count for one artefact, for `vocabulary_info`. `Ok(None)` if absent.
pub fn describe(vocabulary_id: u16, a: Artefact) -> Result<Option<(u32, String, u64)>, String> {
    let Some(bytes) = artefact_bytes(vocabulary_id, a)? else {
        return Ok(None);
    };
    let count = match a {
        Artefact::Ranking => RankTable::from_bytes(&bytes).map(|t| t.k()),
        Artefact::Mapping => ByteMap::from_bytes(&bytes).map(|m| m.mapped()),
    }
    .map_err(|e| {
        format!(
            "{} of vocabulary {vocabulary_id} is not valid: {e}",
            a.as_str()
        )
    })?;
    Ok(Some((
        count,
        digest_hex(&table_digest(&bytes)),
        bytes.len() as u64,
    )))
}

// ── writing ───────────────────────────────────────────────────────────────────────────────

/// Store a custom artefact as `pgtoken.artefact_<id>_<kind>()`. Write-once: an existing one is
/// an error, raised by `CREATE FUNCTION` itself so that the check and the create are one step.
///
/// The bytes go in as base64, a third larger than raw rather than twice as large as hex. The
/// function is `IMMUTABLE` and its body fully qualified, so it means the same under any
/// `search_path`, including the empty one `pg_dump` restores with.
pub fn store(vocabulary_id: u16, a: Artefact, name: &str, bytes: &[u8]) {
    let fname = function_name(vocabulary_id, a);
    let ddl = Spi::get_one_with_args::<String>(
        "SELECT pg_catalog.format(\
           'CREATE FUNCTION pgtoken.%I() RETURNS bytea LANGUAGE sql IMMUTABLE PARALLEL SAFE AS %L', \
           $1::text, \
           'SELECT pg_catalog.decode(' \
             || pg_catalog.quote_literal(pg_catalog.encode($2, 'base64')) || ', ''base64'')')",
        &[fname.as_str().into(), bytes.into()],
    )
    .unwrap_or_else(|e| bail(e))
    .unwrap_or_else(|| bail("could not build the artefact function"));
    Spi::run(&ddl).unwrap_or_else(|e| bail(e));
    Spi::run(&format!(
        "COMMENT ON FUNCTION pgtoken.{fname}() IS \
         'pgtoken {} for vocabulary {}; write-once, do not edit or drop'",
        a.as_str(),
        name.replace('\'', "''")
    ))
    .unwrap_or_else(|e| bail(e));
}

/// Turn a core-crate error into a Postgres `ERROR`.
pub fn bail(msg: impl std::fmt::Display) -> ! {
    error!("{msg}");
}
