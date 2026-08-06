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

fn compression_code(name: &str) -> u8 {
    match name {
        "raw" => COMPRESSION_RAW,
        "freq" => COMPRESSION_FREQ,
        other => bail(format!(
            "unknown compression {other:?}; expected 'raw' or 'freq'"
        )),
    }
}

// Task 8's `vocabulary_info` is the only caller; kept here beside `compression_code` so the two
// directions of the mapping stay adjacent and cannot drift apart.
#[allow(dead_code)]
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

CREATE FUNCTION pgtoken.vocabulary_is_immutable() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'vocabulary % is immutable', OLD.name
        USING HINT = 'create a new vocabulary and ALTER TABLE ... TYPE to move to it';
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
        bail(format!("vocabulary {name:?} already exists"));
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

    assigned
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
// Task 4's `typmod_out` is the first caller.
#[allow(dead_code)]
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
// Task 5's `tokens_in` is the first caller.
#[allow(dead_code)]
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
}
