//! Named vocabularies: the declared ID space a column's tokens belong to.
//!
//! A vocabulary is the only source of a storage width. `vocab_size` declares how many token IDs
//! the tokenizer has; the width is the narrowest packing that can address them. Nothing inspects
//! the data to choose a width, which is what makes a column's stride uniform and `slice` O(1).
//!
//! Immutable by construction: stored values reference a vocabulary id and `decode` is
//! `IMMUTABLE`, so changing what an id means would change what existing rows decode to.

use pgrx::prelude::*;

use crate::registry::bail;

/// Compression method, as stored in the catalog and packed into a typmod.
pub const COMPRESSION_RAW: u8 = 0;
pub const COMPRESSION_FREQ: u8 = 1;

/// Widest packing the format can address: `raw24`. A `vocab_size` needing 4 bytes would need a
/// `raw32` codec, which does not exist because no tokenizer approaches 16.7M tokens.
pub const MAX_VOCAB_SIZE: i64 = 16_777_216;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vocabulary {
    pub id: u16,
    pub compression: u8,
    pub width: u8,
}

/// The narrowest width that can address `vocab_size` distinct ids.
///
/// `ceil(bit_length(vocab_size - 1) / 8)`, floored at 1 so a single-token vocabulary still gets
/// a byte.
pub fn width_for(vocab_size: i32) -> i32 {
    let max_id = (vocab_size - 1) as u32;
    let bits = (32 - max_id.leading_zeros()).max(1);
    bits.div_ceil(8) as i32
}

/// Quote an identifier the way PostgreSQL would, so a name needing quotes cannot break the DDL.
fn quote_ident(name: &str) -> String {
    Spi::get_one_with_args::<String>("SELECT quote_ident($1)", &[name.into()])
        .unwrap_or_else(|e| bail(e))
        .unwrap_or_else(|| bail("quote_ident returned NULL"))
}

/// Whether `tokens.<name>` currently exists as a type.
///
/// A catalog row surviving `drop_vocabulary` does not imply the domain still does — the row is
/// permanent, the domain is not — so this is a real catalog lookup rather than inferred from
/// `lookup_by_name`. `to_regtype` returns NULL for a name that does not resolve rather than
/// raising, which is what makes it safe to call before deciding whether `DROP DOMAIN` or
/// `CREATE DOMAIN` would even apply.
fn domain_exists(name: &str) -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT to_regtype('tokens.' || quote_ident($1)) IS NOT NULL",
        &[name.into()],
    )
    .unwrap_or_else(|e| bail(e))
    .unwrap_or(false)
}

fn compression_code(name: &str) -> u8 {
    match name {
        "raw" => COMPRESSION_RAW,
        "freq" => COMPRESSION_FREQ,
        other => bail(format!(
            "unknown compression {other:?}; expected 'raw' or 'freq'"
        )),
    }
}

// `vocabulary_info`'s only caller; kept here beside `compression_code` so the two directions of
// the mapping stay adjacent and cannot drift apart.
fn compression_name(code: u8) -> &'static str {
    if code == COMPRESSION_FREQ {
        "freq"
    } else {
        "raw"
    }
}

extension_sql!(
    r#"
CREATE TABLE pgtoken.vocabulary (
    -- `int`, not `smallint`: the upper bound is 65535 because the value header's vocabulary_id
    -- is a u16 and the typmod packs the id into bits 0-15, and smallint stops at 32767. Do not
    -- "simplify" this CHECK to `id > 0` — the bound is what the storage format can address.
    -- 0 is excluded because the header reserves it to mean "no vocabulary".
    id          int      PRIMARY KEY CHECK (id BETWEEN 1 AND 65535),
    name        text     NOT NULL UNIQUE,
    vocab_size  int      NOT NULL CHECK (vocab_size >= 1),
    compression text     NOT NULL CHECK (compression IN ('raw', 'freq')),
    width       smallint NOT NULL CHECK (width BETWEEN 1 AND 4)
);

-- Rows must survive pg_dump: a restored database with an empty catalog would have columns whose
-- typmod names a vocabulary that no longer exists.
SELECT pg_catalog.pg_extension_config_dump('pgtoken.vocabulary', '');

-- The HINT spells the migration out in full, `USING` clause included, because the bare
-- `ALTER TABLE ... TYPE` form is refused: it coerces in assignment context, and moving token ids
-- between vocabularies takes an explicit cast (see `tokens::tokens_typmod_apply_impl`). A hint that
-- stopped at "ALTER TABLE ... TYPE" would walk the user straight into a second error. Inner
-- dollar-quoting keeps the SQL readable rather than doubling every quote.
-- The exact statement shape printed here is executed by
-- `tokens::tests::an_explicit_cast_moves_a_value_to_another_vocabulary`.
CREATE FUNCTION pgtoken.vocabulary_is_immutable() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'vocabulary % is immutable', OLD.name
        USING HINT = $h$create a new vocabulary, then: ALTER TABLE t ALTER COLUMN c TYPE pgtoken.tokens('<new>') USING c::pgtoken.tokens('<new>')$h$;
END
$$;

-- Guards direct UPDATE and DELETE, not just the extension's own paths.
CREATE TRIGGER vocabulary_is_immutable
    BEFORE UPDATE OR DELETE ON pgtoken.vocabulary
    FOR EACH ROW EXECUTE FUNCTION pgtoken.vocabulary_is_immutable();

-- Vocabularies also appear as domains, so a column reads `body tokens.o200k` and psql can
-- complete `tokens.<tab>`. Task 6 fills this schema.
CREATE SCHEMA tokens;
GRANT USAGE ON SCHEMA tokens TO PUBLIC;
"#,
    name = "vocabulary_catalog",
);

#[pg_extern]
fn create_vocabulary(
    name: &str,
    vocab_size: i32,
    compression: default!(&str, "'raw'"),
    id: default!(Option<i32>, "NULL"),
) -> i32 {
    if vocab_size < 1 {
        bail("vocab_size is required and must be at least 1");
    }
    if vocab_size as i64 > MAX_VOCAB_SIZE {
        bail(format!(
            "vocab_size {vocab_size} exceeds the {MAX_VOCAB_SIZE} this format can address"
        ));
    }
    if name.contains('\'') || name.contains('"') || name.contains('\\') {
        bail(format!(
            "vocabulary name {name:?} may not contain quotes or backslashes"
        ));
    }
    // Validate before touching the catalog, so a bad call leaves nothing behind.
    let _ = compression_code(compression);
    let width = width_for(vocab_size);

    if lookup_by_name(name).is_some() {
        // A catalog row survives `drop_vocabulary`, so a live vocabulary and a dropped-but-
        // reserved name both land here — and they need different messages. A user who just
        // dropped `name` and expects it freed for reuse gets nothing from "already exists"; tell
        // them the id is permanently spoken for instead of implying a same-shaped collision.
        if domain_exists(name) {
            bail(format!("vocabulary {name:?} already exists"));
        }
        bail(format!(
            "vocabulary {name:?} was dropped and its name is permanently reserved: stored \
             values may still reference its id, so neither the id nor the name can be recycled; \
             pick a different name"
        ));
    }

    let assigned = match id {
        Some(v) if v < 1 || v > u16::MAX as i32 => {
            bail(format!("vocabulary id {v} is out of range (1..65535)"))
        }
        Some(v) => v,
        None => Spi::get_one::<i32>("SELECT coalesce(max(id), 0)::int + 1 FROM pgtoken.vocabulary")
            .unwrap_or_else(|e| bail(e))
            .unwrap_or(1),
    };

    Spi::run_with_args(
        "INSERT INTO pgtoken.vocabulary (id, name, vocab_size, compression, width) \
         VALUES ($1, $2, $3, $4, $5::smallint)",
        &[
            assigned.into(),
            name.into(),
            vocab_size.into(),
            compression.into(),
            width.into(),
        ],
    )
    .unwrap_or_else(|e| bail(e));

    // The domain is what users write. It carries the packed typmod, so the base type name never
    // has to appear in a schema definition — and two vocabularies become two distinct types,
    // which makes mixing tokenizers a type error rather than a silent recode.
    Spi::run(&format!(
        "CREATE DOMAIN tokens.{} AS pgtoken.tokens('{}')",
        quote_ident(name),
        name
    ))
    .unwrap_or_else(|e| bail(e));

    assigned
}

/// Remove a vocabulary's domain.
///
/// The catalog row and its id stay forever — stored values may still reference the id, and
/// reusing one would change what those rows mean. The immutability trigger enforces that even
/// against a deliberate `DELETE`. Three states, three outcomes: no catalog row is a plain "does
/// not exist"; a row whose domain is already gone is an error naming that rather than a Postgres
/// "type does not exist" from a bare `DROP DOMAIN` that was never asked to run twice; and a row
/// with a live domain is the ordinary drop.
#[pg_extern]
fn drop_vocabulary(name: &str) {
    if lookup_by_name(name).is_none() {
        bail(format!("vocabulary {name:?} does not exist"));
    }
    if !domain_exists(name) {
        bail(format!(
            "vocabulary {name:?} was already dropped; its id stays reserved forever and cannot \
             be dropped again"
        ));
    }
    Spi::run(&format!("DROP DOMAIN tokens.{}", quote_ident(name))).unwrap_or_else(|e| bail(e));
}

/// Resolve a name to the three fields a typmod needs. Called at DDL time only.
pub fn lookup_by_name(name: &str) -> Option<Vocabulary> {
    Spi::connect(|client| {
        let rows = client
            .select(
                "SELECT id, compression, width::int FROM pgtoken.vocabulary WHERE name = $1",
                Some(1),
                &[name.into()],
            )
            .ok()?;
        let row = rows.first();
        let id: i32 = row.get(1).ok()??;
        let compression: String = row.get(2).ok()??;
        let width: i32 = row.get(3).ok()??;
        Some(Vocabulary {
            id: id as u16,
            compression: compression_code(&compression),
            width: width as u8,
        })
    })
}

/// Reverse lookup for `typmod_out`. Returns `None` rather than raising, because `typmod_out` can
/// run while formatting an error message in an already-aborted transaction.
pub fn name_for_id(id: u16) -> Option<String> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT name FROM pgtoken.vocabulary WHERE id = $1",
                Some(1),
                &[(id as i32).into()],
            )
            .ok()?
            .first()
            .get::<String>(1)
            .ok()?
    })
}

/// The declared ID space, for the out-of-vocabulary bound check on writes.
pub fn vocab_size_for(id: u16) -> Option<u32> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT vocab_size FROM pgtoken.vocabulary WHERE id = $1",
                Some(1),
                &[(id as i32).into()],
            )
            .ok()?
            .first()
            .get::<i32>(1)
            .ok()?
            .map(|v| v as u32)
    })
}

/// Report what a vocabulary declares and which optional parts are filled.
#[pg_extern]
#[allow(clippy::type_complexity)] // A 7-column TableIterator tuple; a type alias would just move the noise.
fn vocabulary_info(
    name: &str,
) -> TableIterator<
    'static,
    (
        name!(id, i32),
        name!(vocab_size, i32),
        name!(compression, String),
        name!(width, i32),
        name!(ranked, Option<i32>),
        name!(sha256, Option<String>),
        name!(file_bytes, Option<i64>),
    ),
> {
    let v =
        lookup_by_name(name).unwrap_or_else(|| bail(format!("vocabulary {name:?} does not exist")));
    let size = vocab_size_for(v.id)
        .unwrap_or_else(|| bail(format!("vocabulary {name:?} has no catalog row")));

    // An unfilled ranking is a normal state, not a fault, so a genuinely absent file becomes
    // NULLs rather than an error. Anything else — `table_dir` unset so the question cannot even
    // be asked, or a file that is there but unreadable/corrupt — is a real fault and must say so
    // rather than collapsing into the same NULLs, which would silently read as "never trained".
    if crate::registry::table_dir().as_os_str().is_empty() {
        bail("pgtoken.table_dir is not set; cannot tell whether any vocabulary has a ranking");
    }
    let path = crate::registry::table_path(v.id);
    let (ranked, sha256, file_bytes) = if !path.exists() {
        (None, None, None)
    } else {
        match crate::registry::describe_table(v.id) {
            Ok((k, digest, len)) => (Some(k as i32), Some(digest), Some(len as i64)),
            Err(e) => bail(format!(
                "vocabulary {name:?} has a ranking file at {} but it could not be read: {e}",
                path.display()
            )),
        }
    };

    TableIterator::once((
        v.id as i32,
        size as i32,
        compression_name(v.compression).to_string(),
        v.width as i32,
        ranked,
        sha256,
        file_bytes,
    ))
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    /// Create a vocabulary and return its id.
    ///
    /// Creating and reading back have to be separate statements. `create_vocabulary` inserts
    /// through SPI, and a scan in the same statement runs under a snapshot taken before that
    /// insert, so `... FROM pgtoken.vocabulary WHERE id = pgtoken.create_vocabulary(...)` reads
    /// zero rows — on an empty catalog the qual is never even evaluated.
    fn create(sql_args: &str) -> i32 {
        Spi::get_one::<i32>(&format!("SELECT pgtoken.create_vocabulary({sql_args})"))
            .expect("create_vocabulary failed")
            .expect("create_vocabulary returned NULL")
    }

    #[pg_test]
    fn width_is_derived_from_vocab_size() {
        // ceil(bit_length(vocab_size - 1) / 8), and every boundary around it.
        for (size, want) in [
            (1, 1),
            (256, 1),
            (257, 2),
            (32000, 2),
            (65536, 2),
            (65537, 3),
            (100277, 3),
            (200019, 3),
            (16777216, 3),
        ] {
            let id = create(&format!("'v{size}', {size}"));
            let got = Spi::get_one::<i32>(&format!(
                "SELECT width::int FROM pgtoken.vocabulary WHERE id = {id}"
            ))
            .expect("query failed");
            assert_eq!(
                got,
                Some(want),
                "vocab_size {size} should derive width {want}"
            );
        }
    }

    #[pg_test]
    fn defaults_to_raw_compression() {
        let id = create("'d1', 32000");
        let c = Spi::get_one::<String>(&format!(
            "SELECT compression FROM pgtoken.vocabulary WHERE id = {id}"
        ))
        .expect("query failed");
        assert_eq!(c, Some("raw".to_string()));
    }

    #[pg_test]
    fn accepts_freq_compression() {
        let id = create("'d2', 32000, compression => 'freq'");
        let c = Spi::get_one::<String>(&format!(
            "SELECT compression FROM pgtoken.vocabulary WHERE id = {id}"
        ))
        .expect("query failed");
        assert_eq!(c, Some("freq".to_string()));
    }

    #[pg_test]
    fn assigns_ids_from_one_upward() {
        // Id 0 is reserved in the header to mean "no vocabulary".
        let id = Spi::get_one::<i32>("SELECT pgtoken.create_vocabulary('first', 300)")
            .expect("query failed");
        assert!(id.unwrap() >= 1, "ids must start at 1");
    }

    #[pg_test]
    fn honours_an_explicit_id() {
        let id = Spi::get_one::<i32>("SELECT pgtoken.create_vocabulary('pinned', 300, id => 41)")
            .expect("query failed");
        assert_eq!(id, Some(41));
    }

    #[pg_test]
    fn accepts_an_id_across_the_whole_u16_range() {
        // The value header's vocabulary_id is a u16, so the catalog must address all of it.
        // A smallint column would have failed here with "smallint out of range".
        let id = Spi::get_one::<i32>("SELECT pgtoken.create_vocabulary('wide', 300, id => 40000)")
            .expect("query failed");
        assert_eq!(id, Some(40000));
        let found =
            Spi::get_one::<i32>("SELECT vocab_size FROM pgtoken.vocabulary WHERE id = 40000")
                .expect("query failed");
        assert_eq!(
            found,
            Some(300),
            "the row must be findable at an id above smallint's range"
        );
    }

    #[pg_test(error = "vocab_size is required and must be at least 1")]
    fn rejects_a_zero_vocab_size() {
        Spi::get_one::<i32>("SELECT pgtoken.create_vocabulary('bad', 0)").unwrap();
    }

    // `#[pg_test(error = ...)]` compares the whole message, not a substring, so the expected
    // text has to name the offending size too.
    #[pg_test(error = "vocab_size 20000000 exceeds the 16777216 this format can address")]
    fn rejects_a_vocab_size_needing_four_bytes() {
        Spi::get_one::<i32>("SELECT pgtoken.create_vocabulary('huge', 20000000)").unwrap();
    }

    #[pg_test(error = "unknown compression \"lz4\"; expected 'raw' or 'freq'")]
    fn rejects_an_unknown_compression() {
        Spi::get_one::<i32>("SELECT pgtoken.create_vocabulary('bad2', 300, compression => 'lz4')")
            .unwrap();
    }

    #[pg_test(error = "vocabulary \"dup\" already exists")]
    fn rejects_a_duplicate_name() {
        Spi::run("SELECT pgtoken.create_vocabulary('dup', 300)").expect("first");
        Spi::get_one::<i32>("SELECT pgtoken.create_vocabulary('dup', 400)").unwrap();
    }

    #[pg_test(error = "vocabulary v_immutable is immutable")]
    fn refuses_to_change_a_vocabulary() {
        Spi::run("SELECT pgtoken.create_vocabulary('v_immutable', 300)").expect("create");
        Spi::run("UPDATE pgtoken.vocabulary SET vocab_size = 400 WHERE name = 'v_immutable'")
            .unwrap();
    }

    /// A binary-coercible `pgtoken.tokens -> bytea` cast, so a test can inspect a stored value's
    /// actual byte length. Domains are binary-compatible with their base type, so this also
    /// works on a value whose declared column type is `tokens.<name>`. Created inside the test's
    /// own transaction, which pgrx rolls back, and skipped if one already exists.
    fn ensure_bytea_cast() {
        Spi::run(
            "DO $cast$ BEGIN \
               IF NOT EXISTS (SELECT 1 FROM pg_cast \
                              WHERE castsource = 'pgtoken.tokens'::regtype \
                                AND casttarget = 'bytea'::regtype) THEN \
                 CREATE CAST (pgtoken.tokens AS bytea) WITHOUT FUNCTION; \
               END IF; \
             END $cast$",
        )
        .expect("bytea cast");
    }

    #[pg_test]
    fn creating_a_vocabulary_emits_a_domain() {
        // `body::int[]` is the brief's original assertion, but that cast is Task 7's job (see
        // the "(from Task 7) `int[]` input" note in `tokens.rs`'s `encode_for`) and does not
        // exist yet at this point in the sequence. `::text` proves the same thing — a table
        // declared with the domain accepts a literal and reads it back — using a cast this task
        // can actually reach.
        Spi::run("SELECT pgtoken.create_vocabulary('dom1', 32000)").expect("create");
        Spi::run("CREATE TABLE dom1_docs (body tokens.dom1)").expect("create table");
        Spi::run("INSERT INTO dom1_docs (body) VALUES ('{1,2,3}')").expect("insert");
        let got = Spi::get_one::<String>("SELECT body::text FROM dom1_docs").expect("query failed");
        assert_eq!(got, Some("{1,2,3}".to_string()));
    }

    #[pg_test]
    fn the_domain_carries_the_vocabulary_typmod() {
        Spi::run("SELECT pgtoken.create_vocabulary('dom2', 256)").expect("create");
        let base = Spi::get_one::<String>(
            "SELECT format_type(typbasetype, typtypmod) FROM pg_type \
             WHERE oid = 'tokens.dom2'::regtype",
        )
        .expect("query failed");
        assert_eq!(base, Some("pgtoken.tokens('dom2')".to_string()));

        // `format_type` reading the catalog correctly is necessary but not sufficient: it could
        // render right while a write through the domain still stored the unresolved raw24
        // carrier. Prove the domain's typmod actually reaches the write path by checking a value
        // inserted through it lands at dom2's declared width (256 -> raw8, one byte per token)
        // rather than the wider unresolved encoding.
        ensure_bytea_cast();
        Spi::run("CREATE TABLE dom2_docs (body tokens.dom2)").expect("create table");
        Spi::run("INSERT INTO dom2_docs (body) VALUES ('{1,2,3}')").expect("insert");
        let len =
            Spi::get_one::<i32>("SELECT length(body::bytea) FROM dom2_docs").expect("query failed");
        assert_eq!(
            len,
            Some(12 + 3),
            "raw8: one byte per token, proving the domain's typmod reached the encoder"
        );
    }

    #[pg_test(error = "cannot store dom_b token ids in a tokens('dom_a') column")]
    fn distinct_vocabularies_are_distinct_types() {
        // A domain buys a distinct type per vocabulary for free, but the refusal below is not
        // proof of that by itself: Task 5's cross-vocabulary-assignment check on the *base*
        // type's length coercion cast fires for exactly this statement even without domains,
        // because coercing dom_b to dom_a reduces to the base type's cast (with dom_a's typmod)
        // before the domain's own (constraint-only) wrapping ever runs. Both mechanisms now cover
        // this, so asserting on the exact message — rather than merely `is_err()` — pins down
        // which one actually fires, instead of leaving it ambiguous.
        Spi::run(
            "SELECT pgtoken.create_vocabulary('dom_a', 32000); \
             SELECT pgtoken.create_vocabulary('dom_b', 32000); \
             CREATE TABLE dom_a_t (body tokens.dom_a); \
             CREATE TABLE dom_b_t (body tokens.dom_b); \
             INSERT INTO dom_b_t (body) VALUES ('{1,2}');",
        )
        .expect("setup");
        Spi::run("INSERT INTO dom_a_t (body) SELECT body FROM dom_b_t").unwrap();
    }

    #[pg_test]
    fn drop_vocabulary_removes_the_domain_but_reserves_the_id() {
        Spi::run("SELECT pgtoken.create_vocabulary('dom_gone', 300)").expect("create");
        Spi::run("SELECT pgtoken.drop_vocabulary('dom_gone')").expect("drop");
        let domains =
            Spi::get_one::<i64>("SELECT count(*) FROM pg_type WHERE typname = 'dom_gone'")
                .expect("query failed");
        assert_eq!(domains, Some(0), "the domain must be gone");
        let reserved =
            Spi::get_one::<i64>("SELECT count(*) FROM pgtoken.vocabulary WHERE name = 'dom_gone'")
                .expect("query failed");
        assert_eq!(reserved, Some(1), "the id stays reserved forever");
    }

    #[pg_test(
        error = "vocabulary \"dom_double\" was already dropped; its id stays reserved forever \
                 and cannot be dropped again"
    )]
    fn dropping_a_vocabulary_twice_names_the_real_state() {
        // The catalog row `drop_vocabulary`'s guard checks via `lookup_by_name` is permanent by
        // design, so it cannot tell a second drop from a first one — only `domain_exists` can.
        // Without that second check this fell through to Postgres's raw
        // `type "tokens.dom_double" does not exist` on the second call, which does not say the
        // vocabulary itself is fine and only its domain is already gone.
        Spi::run("SELECT pgtoken.create_vocabulary('dom_double', 300)").expect("create");
        Spi::run("SELECT pgtoken.drop_vocabulary('dom_double')").expect("first drop");
        Spi::run("SELECT pgtoken.drop_vocabulary('dom_double')").unwrap();
    }

    #[pg_test(
        error = "vocabulary \"dom_reserved\" was dropped and its name is permanently reserved: \
                 stored values may still reference its id, so neither the id nor the name can be \
                 recycled; pick a different name"
    )]
    fn creating_a_dropped_name_explains_the_reservation() {
        // Before this fix, re-creating a dropped name hit the same "already exists" used for an
        // ordinary name collision, which tells someone who just freed the name nothing about why
        // it is not actually free.
        Spi::run("SELECT pgtoken.create_vocabulary('dom_reserved', 300)").expect("create");
        Spi::run("SELECT pgtoken.drop_vocabulary('dom_reserved')").expect("drop");
        Spi::run("SELECT pgtoken.create_vocabulary('dom_reserved', 300)").unwrap();
    }
}
