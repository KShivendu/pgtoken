//! `pgtoken`: store text in PostgreSQL as the token IDs your models already use.
//!
//! The extension compresses sequences of token IDs. It contains no tokenizer, and takes no
//! position on which one you use — tokenizers are widely available client-side, and a database
//! has no reason to hold an opinion about vocabularies. In exchange, any tokenizer works, the
//! server never spends CPU tokenizing, and no backend carries a multi-megabyte merge table.
//!
//! ```sql
//! SELECT pgtoken.create_vocabulary('cl100k', 100277);
//! CREATE TABLE documents (id bigserial PRIMARY KEY, body tokens.cl100k);
//!
//! INSERT INTO documents (body) VALUES ('{24912,2375}');
//! SELECT body::int[] FROM documents;
//! ```
//!
//! A vocabulary's declared size is the only source of a storage width, and `pgtoken.tokens` —
//! aliased per vocabulary as `tokens.<name>` — is `STORAGE EXTERNAL` by construction, so
//! PostgreSQL never spends cycles recompressing a payload that is already compressed.
//!
//! An agent reads `SELECT body` and decodes the blob client-side. Casting to `int[]` exists for
//! SQL-side work; note it costs 4 bytes per token on the wire, more than the compressed blob it
//! came from.

use std::ffi::CString;

use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting};
use pgrx::prelude::*;

use pgtoken_core::tables::RankTable;
use pgtoken_core::value;

mod casts;
mod registry;
mod tokens;
mod typmod;
mod vocabulary;

use registry::bail;

pgrx::pg_module_magic!();

/// Directory holding `<table_id>.tntt` coding tables.
static TABLE_DIR: GucSetting<Option<CString>> = GucSetting::<Option<CString>>::new(None);

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
}

// ── helpers ──────────────────────────────────────────────────────────────────────────

fn guc_str(g: &GucSetting<Option<CString>>, default: &str) -> String {
    g.get()
        .and_then(|c| c.into_string().ok())
        .unwrap_or_else(|| default.to_string())
}

fn vocabulary_id_u16(id: i32) -> u16 {
    u16::try_from(id)
        .unwrap_or_else(|_| bail(format!("vocabulary_id {id} is out of range (0..65535)")))
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
        name!(vocabulary_id, i32),
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

extension_sql!(
    r#"
DROP FUNCTION pgtoken.token_count(bytea);
CREATE FUNCTION pgtoken.token_count(pgtoken.tokens) RETURNS int
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE
    AS 'MODULE_PATHNAME', 'token_count_wrapper';

DROP FUNCTION pgtoken.describe(bytea);
CREATE FUNCTION pgtoken.describe(pgtoken.tokens)
    RETURNS TABLE(version int, codec text, vocabulary_id int,
                  n_tokens int, payload_bytes int, total_bytes int)
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE
    AS 'MODULE_PATHNAME', 'describe_wrapper';
"#,
    name = "tokens_functions",
    requires = ["tokens_type", token_count, describe],
);

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
    let tid = vocabulary_id_u16(table_id);
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
        registry::describe_table(vocabulary_id_u16(table_id)).unwrap_or_else(|e| bail(e));
    TableIterator::once((k as i32, digest, len as i64))
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

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
    fn the_storage_policy_gucs_are_gone() {
        let n = Spi::get_one::<i64>(
            "SELECT count(*) FROM pg_settings \
             WHERE name IN ('pgtoken.default_codec', 'pgtoken.default_table_id')",
        )
        .expect("query failed");
        assert_eq!(n, Some(0));
    }

    #[pg_test]
    fn table_dir_guc_survives_and_stays_sighup() {
        // A session that could repoint it could make two sessions decode one value differently.
        let ctx = Spi::get_one::<String>(
            "SELECT context FROM pg_settings WHERE name = 'pgtoken.table_dir'",
        )
        .expect("query failed");
        assert_eq!(ctx, Some("sighup".to_string()));
    }

    #[pg_test]
    fn the_old_function_surface_is_gone() {
        let n = Spi::get_one::<i64>(
            "SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace \
             WHERE n.nspname = 'pgtoken' AND p.proname IN ('encode', 'decode', 'recode')",
        )
        .expect("query failed");
        assert_eq!(n, Some(0), "casts and typmod replace all three");
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
