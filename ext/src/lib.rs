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
mod mapping;
mod registry;
mod tokens;
mod typmod;
mod vocabulary;

use registry::bail;

pgrx::pg_module_magic!();

/// Directory holding `<vocabulary_id>.tntt` coding tables.
static TABLE_DIR: GucSetting<Option<CString>> = GucSetting::<Option<CString>>::new(None);

#[pg_guard]
pub extern "C-unwind" fn _PG_init() {
    GucRegistry::define_string_guc(
        c"pgtoken.table_dir",
        c"Directory holding <vocabulary_id>.tntt coding tables.",
        c"Resolved by filesystem convention rather than a SQL catalog, so reading a value \
          needs no SPI and pgtoken.tokens's read paths can honestly stay IMMUTABLE and \
          PARALLEL SAFE. SIGHUP-scoped for the same reason: if a session could repoint it, two \
          sessions could decode one value differently.",
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

// ── inspect (header only, O(1), loads no table) ──────────────────────────────────────

/// Deliberately exempt from `require_vocabulary`, unlike every value-returning path in
/// `tokens.rs` and `casts.rs`. This is a diagnostic: it exists precisely to let you inspect a
/// value that might be broken, and refusing to report on an unresolved value would remove the
/// only way to see that it *is* unresolved rather than merely learning that reading it failed.
#[pg_extern(immutable, parallel_safe, strict)]
fn token_count(v: &[u8]) -> i32 {
    let (h, _) = value::describe(v).unwrap_or_else(|e| bail(e));
    h.n_tokens as i32
}

/// Deliberately exempt from `require_vocabulary`, for the same reason as `token_count`: this is
/// a diagnostic, meant to tell you what a value contains when something about it is wrong, and an
/// unresolved `vocabulary_id` is exactly the kind of wrong it should be able to report rather than
/// refuse to look at.
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

/// Train a frequency ranking from a query returning `int[]`, and attach it to a vocabulary.
///
/// The ranking holds only the tokens the query actually contained, so unseen IDs still encode
/// losslessly, just a little wider. Write-once: stored payloads reference ranks and `decode` is
/// `IMMUTABLE`, so replacing one would change what existing rows mean.
#[pg_extern(strict, name = "train")]
fn train(name: &str, query: &str, max_ranks: default!(i32, -1)) -> String {
    let v = vocabulary::lookup_by_name(name)
        .unwrap_or_else(|| bail(format!("vocabulary {name:?} does not exist")));
    // Directory question first: with `pgtoken.table_dir` unset, `table_path` yields a bare
    // relative name whose `.exists()` is answered against the data directory, so this guard would
    // report "no ranking yet" for a vocabulary that has one. See `registry::require_dir`.
    registry::require_dir("tell whether a vocabulary already has a ranking")
        .unwrap_or_else(|e| bail(e));
    if registry::table_path(v.id).exists() {
        bail(format!("vocabulary {name} already has a ranking"));
    }
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
                        .filter(|&x| x >= 0)
                        .map(|x| x as u32),
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
    registry::write_table(v.id, &table.to_bytes()).unwrap_or_else(|e| bail(e));
    format!(
        "trained vocabulary {name} on {rows} rows / {} tokens, {k} ranked",
        ids.len()
    )
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

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
    fn train_takes_a_vocabulary_name() {
        // A pinned id, not the default auto-assignment: every test in this module that trains
        // shares one Postgres cluster and one `pgtoken.table_dir`, and each test's catalog insert
        // rolls back while the `.tntt` file it wrote does not. Two tests that both auto-assigned
        // id 1 would silently race for the same file, so every trained vocabulary here gets its
        // own id, disjoint from the others and from `tokens.rs`'s reserved 60001/60002.
        Spi::run(
            "SELECT pgtoken.create_vocabulary('tr1', 200019, compression => 'freq', \
                                              id => 61001)",
        )
        .expect("create");
        Spi::run(
            "SELECT pgtoken.train('tr1', \
               $$SELECT ARRAY[7,7,7,7,3,3,199999]::int[] FROM generate_series(1,40)$$)",
        )
        .expect("train");
        let ranked = Spi::get_one::<i32>("SELECT ranked FROM pgtoken.vocabulary_info('tr1')")
            .expect("query failed");
        assert_eq!(ranked, Some(3), "only the three distinct tokens are ranked");

        Spi::run(
            "CREATE TABLE tr1_docs (body tokens.tr1); \
             INSERT INTO tr1_docs (body) VALUES ('{7,3,199999}');",
        )
        .expect("insert");
        let got =
            Spi::get_one::<Vec<i32>>("SELECT body::int[] FROM tr1_docs").expect("query failed");
        assert_eq!(got, Some(vec![7, 3, 199999]));
    }

    #[pg_test]
    fn freq_roundtrips_ids_the_ranking_never_saw() {
        // The sparse ranking's whole point: an unseen id must still come back exactly.
        Spi::run(
            "SELECT pgtoken.create_vocabulary('tr2', 200019, compression => 'freq', \
                                              id => 61002); \
             SELECT pgtoken.train('tr2', \
               $$SELECT ARRAY[1,1,1,2]::int[] FROM generate_series(1,20)$$);",
        )
        .expect("setup");
        let got = Spi::get_one::<Vec<i32>>("SELECT '{1,2,199999,0}'::pgtoken.tokens('tr2')::int[]")
            .expect("query failed");
        assert_eq!(got, Some(vec![1, 2, 199999, 0]));
    }

    #[pg_test]
    fn freq_beats_raw_on_a_skewed_stream() {
        Spi::run(
            "SELECT pgtoken.create_vocabulary('sk_raw', 200019); \
             SELECT pgtoken.create_vocabulary('sk_freq', 200019, compression => 'freq', \
                                              id => 61003); \
             SELECT pgtoken.train('sk_freq', \
               $$SELECT array_agg(199999)::int[] FROM generate_series(1,64)$$);",
        )
        .expect("setup");
        let (raw, freq) = Spi::get_two::<i32, i32>(
            "SELECT length(a::pgtoken.tokens('sk_raw')::bytea), \
                    length(a::pgtoken.tokens('sk_freq')::bytea) \
             FROM (SELECT array_agg(199999)::int[] AS a FROM generate_series(1,512)) s",
        )
        .expect("query failed");
        let (raw, freq) = (raw.unwrap(), freq.unwrap());
        assert!(
            freq < raw,
            "freq ({freq} B) should beat raw ({raw} B) on a skewed stream"
        );
    }

    #[pg_test]
    fn describe_reports_the_width_the_vocabulary_chose() {
        Spi::run("SELECT pgtoken.create_vocabulary('d_small', 256)").expect("create");
        let codec = Spi::get_one::<String>(
            "SELECT codec FROM pgtoken.describe('{1,2,3}'::pgtoken.tokens('d_small'))",
        )
        .expect("query failed");
        assert_eq!(codec, Some("raw8".to_string()));
    }

    #[pg_test]
    fn token_count_reads_only_the_header() {
        Spi::run("SELECT pgtoken.create_vocabulary('tc', 60000)").expect("create");
        let (n, total) = Spi::get_two::<i32, i32>(
            "SELECT pgtoken.token_count(v), length(v::bytea) \
             FROM (SELECT '{1,2,3}'::pgtoken.tokens('tc') AS v) s",
        )
        .expect("query failed");
        assert_eq!(n, Some(3));
        assert_eq!(total, Some(12 + 6), "12-byte header plus 2 bytes per token");
    }

    #[pg_test]
    fn encoding_is_canonical_in_sql() {
        // The property under test is byte equality, so comparing `::bytea` says that directly.
        // `pgtoken.tokens` deliberately has no `=` operator of its own (adding one would need a
        // hash opclass to back GROUP BY / DISTINCT / hash joins too, or it would ship half an
        // equality story; that is a design pass of its own, not a side effect of this test).
        Spi::run("SELECT pgtoken.create_vocabulary('canon', 60000)").expect("create");
        let same = Spi::get_one::<bool>(
            "SELECT '{5,9,5,1}'::pgtoken.tokens('canon')::bytea \
               = '{5,9,5,1}'::pgtoken.tokens('canon')::bytea",
        )
        .expect("query failed");
        assert_eq!(same, Some(true));
    }

    #[pg_test]
    fn text_output_round_trips_to_identical_bytes() {
        // The property that makes rendering token IDs rather than hex safe for pg_dump.
        Spi::run("SELECT pgtoken.create_vocabulary('dump', 60000)").expect("create");
        let same = Spi::get_one::<bool>(
            "SELECT v::bytea = (v::text)::pgtoken.tokens('dump')::bytea \
             FROM (SELECT ('{' || string_agg((i % 60000)::text, ',') || '}') \
                            ::pgtoken.tokens('dump') AS v \
                   FROM generate_series(1,512) i) s",
        )
        .expect("query failed");
        assert_eq!(same, Some(true));
    }

    #[pg_test]
    fn vocabulary_info_reports_an_unfilled_ranking_as_null() {
        Spi::run("SELECT pgtoken.create_vocabulary('vi_bare', 300, compression => 'freq')")
            .expect("create");
        let ranked = Spi::get_one::<i32>("SELECT ranked FROM pgtoken.vocabulary_info('vi_bare')")
            .expect("query failed");
        assert_eq!(ranked, None, "an unfilled part is NULL, not an error");
    }

    #[pg_test(error = "vocabulary \"vi_corrupt\" has a ranking file at \
                 /tmp/pgtoken-pgrx-test-tables/61005.tntt but it could not be read: coding table \
                 61005 is not valid: table file has bad magic, expected TNTT")]
    fn vocabulary_info_reports_a_corrupt_ranking_as_an_error() {
        // A present-but-broken ranking is a fault, not the same "unfilled" NULL as never having
        // trained -- collapsing the two would hide a real problem behind a normal-looking status
        // row. A pinned id, disjoint from every other trained vocabulary in this module.
        Spi::run(
            "SELECT pgtoken.create_vocabulary('vi_corrupt', 300, compression => 'freq', \
                                              id => 61005)",
        )
        .expect("create");
        // pgrx does not guarantee test order, so this cannot assume some `train` test already
        // created the directory via `registry::write_table`'s `create_dir_all` -- it has to create
        // it itself, the same way, to be independent of what ran before it in this run.
        std::fs::create_dir_all("/tmp/pgtoken-pgrx-test-tables").expect("create the table dir");
        std::fs::write(
            "/tmp/pgtoken-pgrx-test-tables/61005.tntt",
            b"not a real coding table, just junk bytes",
        )
        .expect("write a junk ranking file");
        Spi::get_one::<i32>("SELECT ranked FROM pgtoken.vocabulary_info('vi_corrupt')").unwrap();
    }

    #[pg_test(error = "value is 1 bytes, shorter than the 12-byte header")]
    fn rejects_a_truncated_value() {
        Spi::get_one::<Vec<i32>>("SELECT '\\x00'::bytea::pgtoken.tokens::int[]").unwrap();
    }

    #[pg_test(error = "vocabulary tr_untrained has no ranking; run pgtoken.train first")]
    fn freq_errors_before_train() {
        Spi::run("SELECT pgtoken.create_vocabulary('tr_untrained', 300, compression => 'freq')")
            .expect("create");
        Spi::get_one::<Vec<i32>>("SELECT '{1,2}'::pgtoken.tokens('tr_untrained')::int[]").unwrap();
    }

    #[pg_test(error = "vocabulary tr_once already has a ranking")]
    fn train_refuses_to_replace_a_ranking() {
        // Stored payloads reference ranks and decode is IMMUTABLE, so replacing a ranking would
        // change what existing rows mean.
        Spi::run(
            "SELECT pgtoken.create_vocabulary('tr_once', 300, compression => 'freq', \
                                              id => 61004); \
             SELECT pgtoken.train('tr_once', $$SELECT ARRAY[1,2]::int[]$$);",
        )
        .expect("setup");
        Spi::get_one::<String>("SELECT pgtoken.train('tr_once', $$SELECT ARRAY[1,2]::int[]$$)")
            .unwrap();
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
