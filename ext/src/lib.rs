//! `pgtoken`: store text in PostgreSQL as the token IDs your models already use.
//!
//! The extension compresses sequences of token IDs. Bring any tokenizer: the server needs only a
//! vocabulary's size, plus a `token_id -> bytes` mapping to give text back. A vocabulary may also
//! have a server tokenizer (today only the o200k preset, built from its own mapping; see
//! `tokenizer.rs`), which lets plain text be inserted and compared. Values are always stored and
//! dumped as token IDs, so nothing on the read or restore path tokenizes.
//!
//! ```sql
//! SELECT pgtoken.create_vocabulary('cl100k', 100277);
//! CREATE TABLE documents (id bigserial PRIMARY KEY, body pgtoken.cl100k);
//!
//! INSERT INTO documents (body) VALUES ('{24912,2375}');
//! SELECT body::int[] FROM documents;
//! ```
//!
//! A vocabulary's declared size is the only source of a storage width, and `pgtoken.tokens` —
//! aliased per vocabulary as `pgtoken.<name>` — is `STORAGE EXTERNAL` by construction, so
//! PostgreSQL never spends cycles recompressing a payload that is already compressed.
//!
//! An agent reads `SELECT body` and decodes the blob client-side. Casting to `int[]` exists for
//! SQL-side work; note it costs 4 bytes per token on the wire, more than the compressed blob it
//! came from.

use pgrx::prelude::*;

use pgtoken_core::tables::RankTable;
use pgtoken_core::value;

mod casts;
mod mapping;
mod preset;
mod registry;
mod tokenizer;
mod tokens;
mod typmod;
mod vocabulary;

use registry::bail;

pgrx::pg_module_magic!();

#[pg_guard]
pub extern "C-unwind" fn _PG_init() {
    // Keeps the per-backend artefact cache honest across rollbacks; see registry.rs.
    registry::register_callbacks();
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
        // An unresolved text carrier is `raw8` with no vocabulary; see tokens.rs.
        if h.vocabulary_id == 0 && h.codec == pgtoken_core::header::Codec::Raw8 {
            "text".to_string()
        } else {
            h.codec.as_str().to_string()
        },
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
/// `IMMUTABLE`, so replacing one would change what existing rows mean. It is stored in the
/// database as `pgtoken.artefact_<id>_ranking()`; see registry.rs for why a function.
#[pg_extern(strict, name = "train")]
fn train(name: &str, query: &str, max_ranks: default!(i32, -1)) -> String {
    let v = vocabulary::lookup_by_name(name)
        .unwrap_or_else(|| bail(format!("vocabulary {name:?} does not exist")));
    refuse_preset(name, v);
    if registry::exists(v.id, registry::Artefact::Ranking) {
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
    registry::store(v.id, registry::Artefact::Ranking, name, &table.to_bytes());
    format!(
        "trained vocabulary {name} on {rows} rows / {} tokens, {k} ranked",
        ids.len()
    )
}

/// `train` and `load_mapping` refuse a preset vocabulary: its artefacts are bundled and frozen.
fn refuse_preset(name: &str, v: vocabulary::Vocabulary) {
    if let Some(p) = preset::by_code(v.preset) {
        bail(format!(
            "vocabulary {name} uses the bundled {} preset, whose ranking and mapping are fixed",
            p.name
        ));
    }
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
    fn there_is_no_table_dir_setting() {
        // Artefacts live in the database now, so there is no directory to configure.
        let n = Spi::get_one::<i64>(
            "SELECT count(*) FROM pg_settings WHERE name = 'pgtoken.table_dir'",
        )
        .expect("query failed");
        assert_eq!(n, Some(0));
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
            "CREATE TABLE tr1_docs (body pgtoken.tr1); \
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
            "SELECT length(pgtoken.tokens_send(a::pgtoken.tokens('sk_raw'))), \
                    length(pgtoken.tokens_send(a::pgtoken.tokens('sk_freq'))) \
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
            "SELECT pgtoken.token_count(v), length(pgtoken.tokens_send(v)) \
             FROM (SELECT '{1,2,3}'::pgtoken.tokens('tc') AS v) s",
        )
        .expect("query failed");
        assert_eq!(n, Some(3));
        assert_eq!(
            total,
            Some(5 + 6),
            "5-byte header (tag, flags, id, one-byte count) plus 2 bytes per token"
        );
    }

    #[pg_test]
    fn encoding_is_canonical_in_sql() {
        // The property under test is byte equality, so comparing the sent bytes says so directly.
        // `pgtoken.tokens` deliberately has no `=` operator of its own (adding one would need a
        // hash opclass to back GROUP BY / DISTINCT / hash joins too, or it would ship half an
        // equality story; that is a design pass of its own, not a side effect of this test).
        Spi::run("SELECT pgtoken.create_vocabulary('canon', 60000)").expect("create");
        let same = Spi::get_one::<bool>(
            "SELECT pgtoken.tokens_send('{5,9,5,1}'::pgtoken.tokens('canon')) \
               = pgtoken.tokens_send('{5,9,5,1}'::pgtoken.tokens('canon'))",
        )
        .expect("query failed");
        assert_eq!(same, Some(true));
    }

    #[pg_test]
    fn text_output_round_trips_to_identical_bytes() {
        // The property that makes rendering token IDs rather than hex safe for pg_dump.
        Spi::run("SELECT pgtoken.create_vocabulary('dump', 60000)").expect("create");
        let same = Spi::get_one::<bool>(
            "SELECT pgtoken.tokens_send(v) \
             = pgtoken.tokens_send((pgtoken.tokens_out(v)::text)::pgtoken.tokens('dump')) \
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

    #[pg_test(
        error = "vocabulary \"vi_corrupt\" has a ranking but it could not be read: \
                 ranking of vocabulary 61005 is not valid: table file has bad magic, expected TNTT"
    )]
    fn vocabulary_info_reports_a_corrupt_ranking_as_an_error() {
        // A present-but-broken ranking is a fault, not the same "unfilled" NULL as never having
        // trained -- collapsing the two would hide a real problem behind a normal-looking status
        // row. Only an owner editing the artefact function by hand can get here.
        Spi::run(
            "SELECT pgtoken.create_vocabulary('vi_corrupt', 300, compression => 'freq', \
                                              id => 61005); \
             CREATE FUNCTION pgtoken.artefact_61005_ranking() RETURNS bytea \
               LANGUAGE sql IMMUTABLE AS $$ SELECT 'not a real coding table'::bytea $$;",
        )
        .expect("setup");
        Spi::get_one::<i32>("SELECT ranked FROM pgtoken.vocabulary_info('vi_corrupt')").unwrap();
    }

    #[pg_test(error = "value is 1 bytes, shorter than the 5-byte minimum header")]
    fn rejects_a_truncated_value() {
        // `tokens_recv_bytes` is the door outside bytes come in through; there is no
        // `bytea -> tokens` cast, deliberately, because it only duplicated this function.
        Spi::get_one::<Vec<i32>>("SELECT pgtoken.tokens_recv_bytes('\\x11'::bytea)::int[]")
            .unwrap();
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

    #[pg_test]
    fn train_stores_the_ranking_in_a_dumpable_function() {
        // The artefact is a schema object, so pg_dump writes it in pre-data, before any table
        // data. It must not be an extension member, or pg_dump would leave it out.
        Spi::run(
            "SELECT pgtoken.create_vocabulary('tr_fn', 300, compression => 'freq'); \
             SELECT pgtoken.train('tr_fn', $$SELECT ARRAY[5,5,9]::int[]$$);",
        )
        .expect("setup");
        let (n, members) = Spi::get_two::<i64, i64>(
            "SELECT count(*), count(d.objid) FROM pg_proc p \
             JOIN pg_namespace n ON n.oid = p.pronamespace \
             LEFT JOIN pg_depend d ON d.classid = 'pg_proc'::regclass AND d.objid = p.oid \
                                  AND d.deptype = 'e' \
             WHERE n.nspname = 'pgtoken' AND p.proname = 'artefact_' || \
                   (SELECT id FROM pgtoken.vocabulary WHERE name = 'tr_fn') || '_ranking'",
        )
        .expect("query failed");
        assert_eq!(n, Some(1), "one ranking function");
        assert_eq!(
            members,
            Some(0),
            "not an extension member, so pg_dump includes it"
        );
    }

    #[pg_test]
    fn a_rolled_back_train_leaves_nothing_behind_not_even_in_the_cache() {
        // Train inside a savepoint, use the ranking (so this backend caches it), roll back, then
        // train again with the opposite ranking. Write-once must not count the rolled-back one,
        // and the encode after it must use the new ranking, not the cached old one.
        Spi::run("SELECT pgtoken.create_vocabulary('tr_rb', 300, compression => 'freq')")
            .expect("create");
        // In the rolled-back ranking 5 is rank 0 and 9 is rank 1; in the kept one, the reverse.
        Spi::run(
            "DO $$ BEGIN \
               PERFORM pgtoken.train('tr_rb', 'SELECT ARRAY[5,5,5,9]::int[]'); \
               PERFORM '{9}'::pgtoken.tokens('tr_rb')::int[]; \
               RAISE EXCEPTION 'roll back'; \
             EXCEPTION WHEN raise_exception THEN NULL; END $$",
        )
        .expect("rolled-back train");
        let ranked = Spi::get_one::<i32>("SELECT ranked FROM pgtoken.vocabulary_info('tr_rb')")
            .expect("query failed");
        assert_eq!(ranked, None, "the rolled-back ranking is gone");

        Spi::run("SELECT pgtoken.train('tr_rb', $$SELECT ARRAY[9,9,9,5]::int[]$$)")
            .expect("retrain after rollback");
        // One token, freq: the header, then the rank as one LEB128 byte.
        let sent =
            Spi::get_one::<Vec<u8>>("SELECT pgtoken.tokens_send('{9}'::pgtoken.tokens('tr_rb'))")
                .expect("query failed")
                .expect("NULL");
        assert_eq!(
            sent.last(),
            Some(&0),
            "9 is rank 0 in the kept ranking: {sent:?}"
        );
    }

    #[pg_test(
        error = "vocabulary o200k uses the bundled o200k preset, whose ranking and mapping are fixed"
    )]
    fn train_refuses_a_preset() {
        Spi::run("SELECT pgtoken.create_vocabulary('o200k')").expect("create");
        Spi::run("SELECT pgtoken.train('o200k', $$SELECT ARRAY[1]::int[]$$)").unwrap();
    }
}

#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec![]
    }
}
