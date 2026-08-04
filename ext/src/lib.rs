//! `pgtoken`: token-native storage for PostgreSQL.
//!
//! Stores text as its BPE token IDs rather than UTF-8 bytes. The systems that read and
//! write this text (embedders, rerankers, LLM agents) work in token IDs, so a token-native
//! column hands them the IDs directly instead of re-tokenizing on every read, and takes
//! less space doing it.
//!
//! # Column shape
//!
//! ```sql
//! CREATE TABLE docs (
//!   id     bigint PRIMARY KEY,
//!   body   bytea    -- token-native, see STORAGE below
//! );
//! ALTER TABLE docs ALTER COLUMN body SET STORAGE EXTERNAL;
//! ```
//!
//! `STORAGE EXTERNAL` is deliberate: it tells Postgres not to compress the column. The
//! payload is already entropy-coded, so running `pglz` over it burns CPU for no gain.
//!
//! # The two read paths
//!
//! An agent reads `SELECT body` and gets the blob straight off the heap page, decoding it
//! client-side; the server runs no tokenizer at all. A human reads
//! `SELECT pgtoken.decode(body)` and pays one detokenize. That asymmetry is the point: the
//! tokenization cost moves to the edge where a human is involved, instead of recurring on
//! every machine read.
//!
//! # Why `bytea` and not a custom type
//!
//! This is v1. A dedicated token-native varlena type with `output = detokenize` would make
//! `SELECT body` render as text for humans while `body::bytea` stays the agent path, which
//! is nicer. It is deferred because `bytea` carries zero type-system risk and reaches real
//! measurements sooner. See the plan's deferred list.

use std::ffi::CString;

use pgrx::prelude::*;
use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting};

use pgtoken_core::header::{Codec, Tokenizer};
use pgtoken_core::tables::{AnsTable, RankTable};
use pgtoken_core::{tokenizer, value};

mod registry;

use registry::{ans_table, bail, rank_table, Kind};

pgrx::pg_module_magic!();

/// Directory holding `<table_id>.tntt` coding tables.
static TABLE_DIR: GucSetting<Option<CString>> = GucSetting::<Option<CString>>::new(None);
/// Default tokenizer for the one-argument convenience functions.
static DEFAULT_TOKENIZER: GucSetting<Option<CString>> =
    GucSetting::<Option<CString>>::new(Some(c"o200k"));
/// Default codec for the one-argument convenience functions.
static DEFAULT_CODEC: GucSetting<Option<CString>> =
    GucSetting::<Option<CString>>::new(Some(c"raw"));
/// Default coding table id for the one-argument convenience functions.
static DEFAULT_TABLE_ID: GucSetting<i32> = GucSetting::<i32>::new(0);

/// Read a string GUC, falling back to `default` when unset.
fn guc_str(g: &GucSetting<Option<CString>>, default: &str) -> String {
    g.get().and_then(|c| c.into_string().ok()).unwrap_or_else(|| default.to_string())
}

#[pg_guard]
pub extern "C-unwind" fn _PG_init() {
    GucRegistry::define_string_guc(
        c"pgtoken.table_dir",
        c"Directory holding <table_id>.tntt coding tables.",
        c"Resolved by filesystem convention rather than a SQL catalog, so that decoding a \
          value needs no SPI and pgtoken.decode can honestly be IMMUTABLE and PARALLEL SAFE.",
        &TABLE_DIR,
        GucContext::Sighup,
        GucFlags::default(),
    );
    GucRegistry::define_string_guc(
        c"pgtoken.default_tokenizer",
        c"Tokenizer used by the one-argument pgtoken.tokenize().",
        c"One of r50k, cl100k, o200k.",
        &DEFAULT_TOKENIZER,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_string_guc(
        c"pgtoken.default_codec",
        c"Codec used by the one-argument pgtoken.tokenize().",
        c"One of raw, raw16, raw24, freq, ans. 'raw' picks the narrowest packing the \
          tokenizer allows. 'freq' is the recommended default once a rank table exists: \
          most of ANS's ratio at the fastest decode.",
        &DEFAULT_CODEC,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"pgtoken.default_table_id",
        c"Coding table id used by the one-argument pgtoken.tokenize().",
        c"Ignored by the raw codecs, which take no table.",
        &DEFAULT_TABLE_ID,
        0,
        u16::MAX as i32,
        GucContext::Userset,
        GucFlags::default(),
    );
}

// ── helpers ──────────────────────────────────────────────────────────────────────────

fn parse_tokenizer(name: &str) -> Tokenizer {
    Tokenizer::parse(name).unwrap_or_else(|e| bail(e))
}

fn parse_codec(name: &str, tok: Tokenizer) -> Codec {
    value::resolve_codec(name, tok).unwrap_or_else(|e| bail(e))
}

fn table_id_u16(id: i32) -> u16 {
    u16::try_from(id).unwrap_or_else(|_| bail(format!("table_id {id} is out of range (0..65535)")))
}

/// Load whichever tables the codec needs, then run `f` with them.
///
/// Written as a callback because the tables are `Rc`-owned here but `value::Tables` borrows
/// them; this keeps the guards alive for the duration of the call.
fn with_tables<R>(
    codec: Codec,
    tok: Tokenizer,
    table_id: u16,
    f: impl FnOnce(value::Tables<'_>) -> R,
) -> R {
    match codec {
        Codec::Raw16 | Codec::Raw24 => f(value::Tables::none()),
        Codec::Freq => {
            let t = rank_table(table_id, tok).unwrap_or_else(|e| bail(e));
            f(value::Tables::with_rank(&t))
        }
        Codec::Ans => {
            let t = ans_table(table_id, tok).unwrap_or_else(|e| bail(e));
            f(value::Tables::with_ans(&t))
        }
    }
}

/// Same, but driven by a stored value's own header rather than by arguments.
fn with_tables_for_value<R>(v: &[u8], f: impl FnOnce(value::Tables<'_>) -> R) -> R {
    let (h, _) = value::describe(v).unwrap_or_else(|e| bail(e));
    with_tables(h.codec, h.tokenizer, h.table_id, f)
}

fn ids_from_sql(ids: Vec<Option<i32>>) -> Vec<u32> {
    ids.into_iter()
        .map(|o| match o {
            None => bail("token id array must not contain NULL"),
            Some(v) if v < 0 => bail(format!("token id {v} is negative")),
            Some(v) => v as u32,
        })
        .collect()
}

// ── encode ───────────────────────────────────────────────────────────────────────────

/// Tokenize text and encode it. The human write path: costs one tokenize pass.
///
/// `IMMUTABLE` because the result depends only on the arguments and the named coding table,
/// which is immutable by contract. That makes it usable in an expression index.
#[pg_extern(immutable, parallel_safe, strict, name = "encode")]
fn encode_with(text: &str, tokenizer: &str, codec: &str, table_id: i32) -> Vec<u8> {
    let tok = parse_tokenizer(tokenizer);
    let c = parse_codec(codec, tok);
    let tid = table_id_u16(table_id);
    with_tables(c, tok, tid, |tables| {
        value::encode_text(text, tok, c, tid, tables).unwrap_or_else(|e| bail(e))
    })
}

/// Convenience form driven by the `pgtoken.*` GUCs.
///
/// Only `STABLE`, not `IMMUTABLE`: it reads settings that can change within a session, so it
/// must not back an index. Use the four-argument form for that.
#[pg_extern(stable, parallel_safe, strict, name = "encode")]
fn encode_default(text: &str) -> Vec<u8> {
    encode_with(
        text,
        &guc_str(&DEFAULT_TOKENIZER, "o200k"),
        &guc_str(&DEFAULT_CODEC, "raw"),
        DEFAULT_TABLE_ID.get(),
    )
}

/// Encode token IDs a model already produced. The agent write path: no tokenizer involved.
#[pg_extern(immutable, parallel_safe, strict)]
fn encode_ids(ids: Vec<Option<i32>>, tokenizer: &str, codec: &str, table_id: i32) -> Vec<u8> {
    let tok = parse_tokenizer(tokenizer);
    let c = parse_codec(codec, tok);
    let tid = table_id_u16(table_id);
    let ids = ids_from_sql(ids);
    with_tables(c, tok, tid, |tables| {
        value::encode_ids(&ids, tok, c, tid, tables).unwrap_or_else(|e| bail(e))
    })
}

// ── decode ───────────────────────────────────────────────────────────────────────────

/// Decode to text. The human read path: costs one detokenize pass.
#[pg_extern(immutable, parallel_safe, strict)]
fn decode(v: &[u8]) -> String {
    with_tables_for_value(v, |tables| {
        value::decode_text(v, tables).unwrap_or_else(|e| bail(e))
    })
}

/// Decode to token IDs. Note this is usually *not* the fast agent path: `int4[]` costs
/// 4 bytes per token on the wire, more than the compressed blob. An agent should select the
/// `bytea` directly and decode client-side; this exists for SQL-side work.
#[pg_extern(immutable, parallel_safe, strict)]
fn token_ids(v: &[u8]) -> Vec<i32> {
    with_tables_for_value(v, |tables| {
        value::decode_ids(v, tables)
            .unwrap_or_else(|e| bail(e))
            .into_iter()
            .map(|id| id as i32)
            .collect()
    })
}

// ── inspect (header only, O(1), loads no table) ──────────────────────────────────────

#[pg_extern(immutable, parallel_safe, strict)]
fn token_count(v: &[u8]) -> i32 {
    let (h, _) = value::describe(v).unwrap_or_else(|e| bail(e));
    h.n_tokens as i32
}

#[pg_extern(immutable, parallel_safe, strict)]
fn describe(
    v: &[u8],
) -> TableIterator<
    'static,
    (
        name!(version, i32),
        name!(tokenizer, String),
        name!(codec, String),
        name!(table_id, i32),
        name!(n_tokens, i32),
        name!(payload_bytes, i32),
        name!(total_bytes, i32),
    ),
> {
    let (h, payload_len) = value::describe(v).unwrap_or_else(|e| bail(e));
    TableIterator::once((
        pgtoken_core::VERSION as i32,
        h.tokenizer.as_str().to_string(),
        h.codec.as_str().to_string(),
        h.table_id as i32,
        h.n_tokens as i32,
        payload_len as i32,
        v.len() as i32,
    ))
}

/// Re-encode under a different codec without going through text.
///
/// Cheaper and safer than detokenize-then-retokenize: token IDs are the invariant, so this
/// cannot change the text even if a tokenizer's behaviour ever shifted.
#[pg_extern(immutable, parallel_safe, strict)]
fn recode(v: &[u8], codec: &str, table_id: i32) -> Vec<u8> {
    let (h, _) = value::describe(v).unwrap_or_else(|e| bail(e));
    let to = parse_codec(codec, h.tokenizer);
    let to_tid = table_id_u16(table_id);
    with_tables_for_value(v, |from_tables| {
        with_tables(to, h.tokenizer, to_tid, |to_tables| {
            value::recode(v, to, to_tid, from_tables, to_tables).unwrap_or_else(|e| bail(e))
        })
    })
}

// ── coding tables ────────────────────────────────────────────────────────────────────

/// Train a frequency-rank table over the text a query returns, and store it as `table_id`.
///
/// The query must return one text column. Training reads every row, so run it against a
/// representative sample rather than the whole table if the table is large.
#[pg_extern(strict)]
fn train_freq_table(table_id: i32, tokenizer: &str, query: &str) -> String {
    train(table_id, tokenizer, query, Kind::Rank)
}

/// Train a static ANS unigram table the same way.
#[pg_extern(strict)]
fn train_ans_table(table_id: i32, tokenizer: &str, query: &str) -> String {
    train(table_id, tokenizer, query, Kind::Ans)
}

fn train(table_id: i32, tokenizer: &str, query: &str, kind: Kind) -> String {
    let tok = parse_tokenizer(tokenizer);
    let tid = table_id_u16(table_id);

    let mut ids: Vec<u32> = Vec::new();
    let mut rows = 0usize;
    Spi::connect(|client| {
        let tup = client.select(query, None, &[]).unwrap_or_else(|e| bail(e));
        for row in tup {
            let text: Option<String> = row.get(1).unwrap_or_else(|e| bail(e));
            if let Some(text) = text {
                ids.extend(tokenizer::encode(&text, tok));
                rows += 1;
            }
        }
    });
    if ids.is_empty() {
        bail("training query returned no text; a table cannot be trained on an empty corpus");
    }

    let bytes = match kind {
        Kind::Rank => RankTable::train(&ids, tok).to_bytes(),
        Kind::Ans => AnsTable::train(&ids, tok).to_bytes(),
    };
    let path = registry::write_table(tid, &bytes, kind).unwrap_or_else(|e| bail(e));
    format!(
        "trained {} table {tid} for {} on {rows} rows / {} tokens -> {}",
        match kind {
            Kind::Rank => "rank",
            Kind::Ans => "ans",
        },
        tok.as_str(),
        ids.len(),
        path.display()
    )
}

/// Report what a coding table file contains.
#[pg_extern(strict)]
fn table_info(
    table_id: i32,
) -> TableIterator<
    'static,
    (
        name!(kind, String),
        name!(tokenizer, String),
        name!(vocab, i32),
        name!(sha256, String),
        name!(file_bytes, i64),
    ),
> {
    let tid = table_id_u16(table_id);
    let (kind, tok, vocab, digest, len) =
        registry::describe_table(tid).unwrap_or_else(|e| bail(e));
    TableIterator::once((kind, tok, vocab as i32, digest, len as i64))
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    /// Raw codecs need no table, so these run without configuring `pgtoken.table_dir`.
    #[pg_test]
    fn raw_roundtrips_through_sql() {
        let got = Spi::get_one::<String>(
            "SELECT pgtoken.decode(pgtoken.encode('hello world', 'o200k', 'raw', 0))",
        )
        .expect("query failed");
        assert_eq!(got, Some("hello world".to_string()));
    }

    #[pg_test]
    fn roundtrips_every_tokenizer_and_raw_codec() {
        for tok in ["r50k", "cl100k", "o200k"] {
            for text in ["hello world", "भारत में", "🚀 café", "def f(x):\n  return 1\n"] {
                let sql = format!(
                    "SELECT pgtoken.decode(pgtoken.encode($${text}$$, '{tok}', 'raw', 0))"
                );
                let got = Spi::get_one::<String>(&sql).expect("query failed");
                assert_eq!(got.as_deref(), Some(text), "{tok} failed on {text:?}");
            }
        }
    }

    #[pg_test]
    fn token_count_reads_only_the_header() {
        let (n, total) = Spi::get_two::<i32, i32>(
            "SELECT pgtoken.token_count(v), length(v) \
             FROM (SELECT pgtoken.encode('hello world', 'r50k', 'raw', 0) AS v) s",
        )
        .expect("query failed");
        assert_eq!(n, Some(2), "expected 2 tokens for 'hello world' under r50k");
        // r50k uses raw16: 12-byte header + 2 bytes per token.
        assert_eq!(total, Some(12 + 4));
    }

    #[pg_test]
    fn describe_reports_the_header() {
        let (tok, codec) = Spi::get_two::<String, String>(
            "SELECT tokenizer, codec FROM pgtoken.describe(\
               pgtoken.encode('hello', 'o200k', 'raw', 0))",
        )
        .expect("query failed");
        assert_eq!(tok, Some("o200k".to_string()));
        // o200k exceeds uint16, so `raw` resolves to the 3-byte packing.
        assert_eq!(codec, Some("raw24".to_string()));
    }

    #[pg_test]
    fn token_ids_roundtrip_through_sql() {
        let got = Spi::get_one::<String>(
            "SELECT pgtoken.decode(\
               pgtoken.encode_ids(\
                 pgtoken.token_ids(pgtoken.encode('hello world', 'o200k', 'raw', 0)),\
                 'o200k', 'raw', 0))",
        )
        .expect("query failed");
        assert_eq!(got, Some("hello world".to_string()));
    }

    #[pg_test]
    fn encoding_is_canonical_in_sql() {
        // Equality, GROUP BY and hash joins on a token-native column are only correct if
        // identical text encodes to identical bytes.
        let same = Spi::get_one::<bool>(
            "SELECT pgtoken.encode('the quick brown fox', 'o200k', 'raw', 0) \
                  = pgtoken.encode('the quick brown fox', 'o200k', 'raw', 0)",
        )
        .expect("query failed");
        assert_eq!(same, Some(true));
    }

    #[pg_test]
    fn recode_preserves_text_between_raw_widths() {
        let got = Spi::get_one::<String>(
            "SELECT pgtoken.decode(\
               pgtoken.recode(pgtoken.encode('hello world', 'r50k', 'raw16', 0), 'raw24', 0))",
        )
        .expect("query failed");
        assert_eq!(got, Some("hello world".to_string()));
    }

    #[pg_test]
    fn is_smaller_than_the_utf8_it_replaces() {
        // The whole point, asserted end to end in SQL.
        let (raw, packed) = Spi::get_two::<i32, i32>(
            "SELECT octet_length(t), length(pgtoken.encode(t, 'r50k', 'raw', 0)) \
             FROM (SELECT repeat('the quick brown fox jumps over the lazy dog. ', 20) AS t) s",
        )
        .expect("query failed");
        let (raw, packed) = (raw.unwrap(), packed.unwrap());
        assert!(packed < raw, "token-native form ({packed} B) should beat UTF-8 ({raw} B)");
    }

    // A Postgres ERROR aborts the surrounding transaction, so pgrx reports it as a test
    // failure unless the expected message is declared. These three assert that malformed
    // input is refused rather than decoded into plausible-looking wrong text.

    #[pg_test(error = "value is 1 bytes, shorter than the 12-byte header")]
    fn rejects_a_truncated_value() {
        Spi::get_one::<String>("SELECT pgtoken.decode('\\x00'::bytea)").unwrap();
    }

    #[pg_test(error = "bad magic byte 0x00, expected 0xA7")]
    fn rejects_bad_magic() {
        // Long enough to pass the length check, so this exercises the magic check itself.
        Spi::get_one::<String>(
            "SELECT pgtoken.decode('\\x000000000000000000000000'::bytea)",
        )
        .unwrap();
    }

    #[pg_test(
        error = "pgtoken.table_dir is not set; the +freq and +ANS codecs need a coding table"
    )]
    fn table_codecs_error_without_a_table_dir() {
        // Failing loudly beats silently falling back to a raw codec, which would write a
        // value whose header claims a coding table it was not encoded with.
        Spi::get_one::<Vec<u8>>("SELECT pgtoken.encode('hello', 'o200k', 'freq', 1)").unwrap();
    }
}

#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec![]
    }
}
