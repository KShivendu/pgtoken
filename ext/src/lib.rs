//! `pgtoken`: store text in PostgreSQL as the token IDs your models already use.
//!
//! The extension compresses sequences of token IDs. It contains no tokenizer, and takes no
//! position on which one you use — tokenizers are widely available client-side, and a database
//! has no reason to hold an opinion about vocabularies. In exchange, any tokenizer works, the
//! server never spends CPU tokenizing, and no backend carries a multi-megabyte merge table.
//!
//! ```sql
//! CREATE TABLE documents (id bigserial PRIMARY KEY, body bytea);
//! ALTER TABLE documents ALTER COLUMN body SET STORAGE EXTERNAL;
//!
//! INSERT INTO documents (body) VALUES (pgtoken.encode('{24912,2375}'));
//! SELECT pgtoken.decode(body) FROM documents;
//! ```
//!
//! `STORAGE EXTERNAL` is deliberate: it tells PostgreSQL not to compress a payload that is
//! already compressed.
//!
//! An agent reads `SELECT body` and decodes the blob client-side, so the server does nothing
//! but hand over bytes. `pgtoken.decode` exists for SQL-side work; note it returns `int[]`,
//! which costs 4 bytes per token on the wire — more than the compressed blob it came from.

use std::ffi::CString;

use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting};
use pgrx::prelude::*;

use pgtoken_core::header::Codec;
use pgtoken_core::tables::RankTable;
use pgtoken_core::value;

mod registry;
mod typmod;
mod vocabulary;

use registry::{bail, rank_table};

pgrx::pg_module_magic!();

/// Directory holding `<table_id>.tntt` coding tables.
static TABLE_DIR: GucSetting<Option<CString>> = GucSetting::<Option<CString>>::new(None);
/// Codec used by the one-argument `pgtoken.encode`.
static DEFAULT_CODEC: GucSetting<Option<CString>> =
    GucSetting::<Option<CString>>::new(Some(c"raw"));
/// Coding table id used by the one-argument `pgtoken.encode`.
static DEFAULT_TABLE_ID: GucSetting<i32> = GucSetting::<i32>::new(0);

#[pg_guard]
pub extern "C-unwind" fn _PG_init() {
    GucRegistry::define_string_guc(
        c"pgtoken.table_dir",
        c"Directory holding <table_id>.tntt coding tables.",
        c"Resolved by filesystem convention rather than a SQL catalog, so decoding a value \
          needs no SPI and pgtoken.decode can honestly be IMMUTABLE and PARALLEL SAFE. \
          SIGHUP-scoped for the same reason: if a session could repoint it, two sessions could \
          decode one value differently.",
        &TABLE_DIR,
        GucContext::Sighup,
        GucFlags::default(),
    );
    GucRegistry::define_string_guc(
        c"pgtoken.default_codec",
        c"Codec used by the one-argument pgtoken.encode().",
        c"One of raw, raw16, raw24, freq. 'raw' picks the narrowest packing the data fits. \
          'freq' is recommended once a coding table exists.",
        &DEFAULT_CODEC,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"pgtoken.default_table_id",
        c"Coding table id used by the one-argument pgtoken.encode().",
        c"Ignored by the raw codecs, which take no table.",
        &DEFAULT_TABLE_ID,
        0,
        u16::MAX as i32,
        GucContext::Userset,
        GucFlags::default(),
    );
}

// ── helpers ──────────────────────────────────────────────────────────────────────────

fn guc_str(g: &GucSetting<Option<CString>>, default: &str) -> String {
    g.get()
        .and_then(|c| c.into_string().ok())
        .unwrap_or_else(|| default.to_string())
}

fn table_id_u16(id: i32) -> u16 {
    u16::try_from(id).unwrap_or_else(|_| bail(format!("table_id {id} is out of range (0..65535)")))
}

/// SQL `int[]` to token IDs. Rejects NULLs and negatives rather than coercing them, since
/// either would silently store a different sequence than the caller meant.
fn ids_from_sql(ids: Vec<Option<i32>>) -> Vec<u32> {
    ids.into_iter()
        .map(|o| match o {
            None => bail("token id array must not contain NULL"),
            Some(v) if v < 0 => bail(format!("token id {v} is negative")),
            Some(v) => v as u32,
        })
        .collect()
}

/// Run `f` with the coding table the codec needs, if any.
fn with_table<R>(codec: Codec, table_id: u16, f: impl FnOnce(Option<&RankTable>) -> R) -> R {
    if codec.needs_table() {
        let t = rank_table(table_id).unwrap_or_else(|e| bail(e));
        f(Some(&t))
    } else {
        f(None)
    }
}

// ── encode / decode ──────────────────────────────────────────────────────────────────

/// Encode token IDs. `IMMUTABLE`, so it can back an expression index.
#[pg_extern(immutable, parallel_safe, strict, name = "encode")]
fn encode_with(ids: Vec<Option<i32>>, codec: &str, table_id: i32) -> Vec<u8> {
    let ids = ids_from_sql(ids);
    let c = Codec::parse(codec).unwrap_or_else(|e| bail(e));
    let tid = table_id_u16(table_id);
    with_table(c, tid, |t| {
        value::encode(&ids, c, tid, t).unwrap_or_else(|e| bail(e))
    })
}

/// Convenience form driven by the `pgtoken.*` GUCs.
///
/// Only `STABLE`, not `IMMUTABLE`: it reads settings that can change within a session, so it
/// must not back an index. Use the three-argument form for that.
#[pg_extern(stable, parallel_safe, strict, name = "encode")]
fn encode_default(ids: Vec<Option<i32>>) -> Vec<u8> {
    encode_with(ids, &guc_str(&DEFAULT_CODEC, "raw"), DEFAULT_TABLE_ID.get())
}

/// Decode back to token IDs.
#[pg_extern(immutable, parallel_safe, strict)]
fn decode(v: &[u8]) -> Vec<i32> {
    let (h, _) = value::describe(v).unwrap_or_else(|e| bail(e));
    with_table(h.codec, h.vocabulary_id, |t| {
        value::decode(v, t)
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
        h.codec.as_str().to_string(),
        h.vocabulary_id as i32,
        h.n_tokens as i32,
        payload_len as i32,
        v.len() as i32,
    ))
}

/// Re-encode under a different codec. Cheaper than decode-then-encode from SQL, and cannot
/// change the IDs.
#[pg_extern(immutable, parallel_safe, strict)]
fn recode(v: &[u8], codec: &str, table_id: i32) -> Vec<u8> {
    let (h, _) = value::describe(v).unwrap_or_else(|e| bail(e));
    let ids = with_table(h.codec, h.vocabulary_id, |t| {
        value::decode(v, t).unwrap_or_else(|e| bail(e))
    });
    let to = Codec::parse(codec).unwrap_or_else(|e| bail(e));
    let to_tid = table_id_u16(table_id);
    with_table(to, to_tid, |t| {
        value::encode(&ids, to, to_tid, t).unwrap_or_else(|e| bail(e))
    })
}

// ── coding tables ────────────────────────────────────────────────────────────────────

/// Train a frequency table from a query returning `int[]` of token IDs, and store it as
/// `table_id`.
///
/// The table holds only the tokens the query actually contained, so nothing needs to declare a
/// vocabulary size. Tokens it never saw still encode losslessly, just a little wider.
#[pg_extern(strict)]
fn train(table_id: i32, query: &str) -> String {
    train_capped(table_id, query, -1)
}

/// As [`train`], with a cap on how many tokens get ranked. `-1` means no cap.
#[pg_extern(strict, name = "train")]
fn train_capped(table_id: i32, query: &str, max_ranks: i32) -> String {
    let tid = table_id_u16(table_id);
    let cap = if max_ranks < 0 {
        None
    } else {
        Some(max_ranks as usize)
    };

    let mut ids: Vec<u32> = Vec::new();
    let mut rows = 0usize;
    Spi::connect(|client| {
        let tup = client.select(query, None, &[]).unwrap_or_else(|e| bail(e));
        for row in tup {
            let arr: Option<Vec<Option<i32>>> = row.get(1).unwrap_or_else(|e| bail(e));
            if let Some(arr) = arr {
                ids.extend(
                    arr.into_iter()
                        .flatten()
                        .filter(|&v| v >= 0)
                        .map(|v| v as u32),
                );
                rows += 1;
            }
        }
    });
    if ids.is_empty() {
        bail("training query returned no token ids; the query must return int[] columns");
    }

    let table = RankTable::train(&ids, cap).unwrap_or_else(|e| bail(e));
    let k = table.k();
    let path = registry::write_table(tid, &table.to_bytes()).unwrap_or_else(|e| bail(e));
    format!(
        "trained table {tid} on {rows} rows / {} tokens, {k} ranked -> {}",
        ids.len(),
        path.display()
    )
}

/// Report what a coding table contains.
#[pg_extern(strict)]
fn table_info(
    table_id: i32,
) -> TableIterator<
    'static,
    (
        name!(ranked_tokens, i32),
        name!(sha256, String),
        name!(file_bytes, i64),
    ),
> {
    let (k, digest, len) =
        registry::describe_table(table_id_u16(table_id)).unwrap_or_else(|e| bail(e));
    TableIterator::once((k as i32, digest, len as i64))
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn token_count_reads_only_the_header() {
        let (n, total) = Spi::get_two::<i32, i32>(
            "SELECT pgtoken.token_count(v), length(v) \
             FROM (SELECT pgtoken.encode('{1,2,3}', 'raw16', 0) AS v) s",
        )
        .expect("query failed");
        assert_eq!(n, Some(3));
        assert_eq!(total, Some(12 + 6), "12-byte header plus 2 bytes per token");
    }

    #[pg_test]
    fn recode_preserves_ids() {
        let got = Spi::get_one::<Vec<i32>>(
            "SELECT pgtoken.decode(pgtoken.recode(pgtoken.encode('{1,2,3}','raw16',0),'raw24',0))",
        )
        .expect("query failed");
        assert_eq!(got, Some(vec![1, 2, 3]));
    }

    #[pg_test]
    fn is_smaller_than_the_ids_it_replaces() {
        // int[] costs 4 bytes per element plus array overhead; raw16 costs 2.
        let (arr, packed) = Spi::get_two::<i32, i32>(
            "SELECT pg_column_size(a), length(pgtoken.encode(a, 'raw16', 0)) \
             FROM (SELECT array_agg(i % 60000)::int[] AS a FROM generate_series(1,512) i) s",
        )
        .expect("query failed");
        let (arr, packed) = (arr.unwrap(), packed.unwrap());
        assert!(
            packed < arr,
            "packed ({packed} B) should beat int[] ({arr} B)"
        );
    }

    /// Train a coding table if it is not already there.
    ///
    /// Coding tables are files and `train` deliberately refuses to overwrite one, so a test
    /// that trains unconditionally passes once and fails on every re-run. Each test below owns
    /// a fixed id with fixed training data, so the file's contents are the same either way;
    /// this just makes getting there idempotent. The PL/pgSQL block's subtransaction is what
    /// lets the error be swallowed without poisoning the test's transaction.
    fn ensure_table(table_id: i32, corpus_sql: &str) {
        Spi::run(&format!(
            "DO $ensure$ BEGIN \
               PERFORM pgtoken.train({table_id}, $corpus${corpus_sql}$corpus$); \
             EXCEPTION WHEN OTHERS THEN NULL; \
             END $ensure$"
        ))
        .expect("ensure_table");
    }

    #[pg_test]
    fn trains_a_table_and_uses_it() {
        // A skewed corpus, so the frequency remap has something to exploit.
        const TID: i32 = 1001;
        ensure_table(
            TID,
            "SELECT ARRAY[7,7,7,7,3,3,199999]::int[] FROM generate_series(1,40)",
        );

        let ranked = Spi::get_one::<i32>(&format!(
            "SELECT ranked_tokens FROM pgtoken.table_info({TID})"
        ))
        .expect("table_info failed");
        assert_eq!(
            ranked,
            Some(3),
            "only the three distinct tokens should be ranked"
        );

        let got = Spi::get_one::<Vec<i32>>(&format!(
            "SELECT pgtoken.decode(pgtoken.encode('{{7,3,199999}}', 'freq', {TID}))"
        ))
        .expect("freq roundtrip failed");
        assert_eq!(got, Some(vec![7, 3, 199999]));
    }

    #[pg_test]
    fn freq_roundtrips_ids_the_table_never_saw() {
        // The sparse table's whole point: no vocabulary is declared, so an unseen id must
        // still come back exactly.
        const TID: i32 = 1002;
        ensure_table(
            TID,
            "SELECT ARRAY[1,1,1,2]::int[] FROM generate_series(1,20)",
        );

        let got = Spi::get_one::<Vec<i32>>(&format!(
            "SELECT pgtoken.decode(pgtoken.encode('{{1,2,999999,0,16000000}}', 'freq', {TID}))"
        ))
        .expect("query failed");
        assert_eq!(got, Some(vec![1, 2, 999999, 0, 16000000]));
    }

    // A Postgres ERROR aborts the transaction, so pgrx needs the expected message declared.

    #[pg_test(error = "token id array must not contain NULL")]
    fn rejects_null_in_the_id_array() {
        // Dropping or coercing a NULL would silently store a different sequence.
        Spi::get_one::<Vec<u8>>("SELECT pgtoken.encode('{1,NULL,3}', 'raw', 0)").unwrap();
    }

    #[pg_test(error = "value is 1 bytes, shorter than the 12-byte header")]
    fn rejects_a_truncated_value() {
        Spi::get_one::<Vec<i32>>("SELECT pgtoken.decode('\\x00'::bytea)").unwrap();
    }

    #[pg_test(error = "bad magic byte 0x00, expected 0xA7")]
    fn rejects_bad_magic() {
        Spi::get_one::<Vec<i32>>("SELECT pgtoken.decode('\\x000000000000000000000000'::bytea)")
            .unwrap();
    }

    #[pg_test(
        error = "cannot open coding table /tmp/pgtoken-pgrx-test-tables/1.tntt: \
                 No such file or directory (os error 2)"
    )]
    fn freq_errors_without_its_coding_table() {
        // Failing loudly beats falling back to a raw codec, which would write a value whose
        // header claims a coding table it was not encoded with. Table id 1 is never trained;
        // the other tests use pid-derived ids well above it.
        Spi::get_one::<Vec<u8>>("SELECT pgtoken.encode('{1,2,3}', 'freq', 1)").unwrap();
    }

    #[pg_test]
    fn train_refuses_to_overwrite() {
        // Stored values reference table ids and `decode` is IMMUTABLE, so replacing a table
        // would change what existing rows mean. Deliberately not idempotent.
        //
        // Checked through a PL/pgSQL exception block rather than `#[pg_test(error = ...)]`:
        // the message names a path that differs per machine, and the block's subtransaction
        // lets the test survive the error and carry on asserting.
        const TID: i32 = 1005;
        ensure_table(TID, "SELECT ARRAY[1,2]::int[]");

        Spi::run(
            "CREATE FUNCTION train_is_refused(tid int, q text) RETURNS bool AS $fn$
             BEGIN
               PERFORM pgtoken.train(tid, q);
               RETURN false;
             EXCEPTION WHEN OTHERS THEN
               RETURN true;
             END
             $fn$ LANGUAGE plpgsql",
        )
        .expect("helper");

        let refused = Spi::get_one::<bool>(&format!(
            "SELECT train_is_refused({TID}, $$SELECT ARRAY[1,2]::int[]$$)"
        ))
        .expect("query failed");
        assert_eq!(
            refused,
            Some(true),
            "a second train on the same id must be refused"
        );
    }
}

#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        // `pgtoken.table_dir` is SIGHUP-scoped, so it cannot be set with SET inside a test.
        // Giving the test cluster one lets the suite cover `train` and the `freq` codec rather
        // than only the table-free paths.
        vec!["pgtoken.table_dir = '/tmp/pgtoken-pgrx-test-tables'"]
    }
}
