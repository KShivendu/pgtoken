//! Named vocabularies: the declared ID space a column's tokens belong to.
//!
//! A vocabulary is the only source of a storage width. `vocab_size` declares how many token IDs
//! the tokenizer has; the width is the narrowest packing that can address them. Nothing inspects
//! the data to choose a width, which is what makes a column's stride uniform and `slice` O(1).
//!
//! Immutable by construction: stored values reference a vocabulary id and `decode` is
//! `IMMUTABLE`, so changing what an id means would change what existing rows decode to.

use pgrx::prelude::*;

use crate::preset::{self, Preset};
use crate::registry::{self, bail, Artefact};

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
    /// 0 for a vocabulary with its own artefacts, else a `preset::Preset::code`.
    pub preset: u8,
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
    width       smallint NOT NULL CHECK (width BETWEEN 1 AND 4),
    -- A bundled preset (see preset.rs) whose ranking and mapping this vocabulary uses, or NULL
    -- for one that brings its own through `train` and `load_mapping`.
    preset      text
);

-- Rows must survive pg_dump: a restored database with an empty catalog would have columns whose
-- typmod names a vocabulary that no longer exists.
SELECT pg_catalog.pg_extension_config_dump('pgtoken.vocabulary', '');

-- The HINT spells the migration out in full, `USING` clause included, because the bare
-- `ALTER TABLE ... TYPE` form is refused: it coerces in assignment context, and moving token ids
-- between vocabularies takes an explicit cast (see `tokens::tokens_typmod_apply_impl`). A hint that
-- stopped at "ALTER TABLE ... TYPE" would walk the user straight into a second error. Inner
-- dollar-quoting keeps the SQL readable rather than doubling every quote.
-- The exact statement shape printed here — the `tokens.<name>` domain spelling, since the docs
-- call the domain the intended surface, not the `pgtoken.tokens('<name>')` base-type one — is
-- executed by `tokens::tests::an_explicit_cast_moves_a_value_to_another_vocabulary`.
CREATE FUNCTION pgtoken.vocabulary_is_immutable() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'vocabulary % is immutable', OLD.name
        USING HINT = $h$create a new vocabulary, then: ALTER TABLE t ALTER COLUMN c TYPE tokens.<new> USING c::tokens.<new>$h$;
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

extension_sql!(
    r#"
-- Reading the catalog is not optional, so PUBLIC has to be able to do it. Every write through a
-- typmod'd column consults `pgtoken.vocabulary` from inside the extension's own C functions --
-- `vocab_size_for` on the encode path, `lookup_by_name` from `typmod_in` -- and SPI runs those
-- queries as the *invoking* role, not the owner. Under the default ACLs the extension is usable
-- by nobody but its owner: a plain `INSERT`, and even `'{0}'::pgtoken.tokens('v')`, fails with a
-- permission error naming a query the user never wrote. Both grants are needed; the table grant
-- alone leaves "permission denied for schema pgtoken", which fires first.
--
-- What is being exposed is metadata, not data: vocabulary names, declared sizes, compression and
-- widths, describing columns the role can already see.
GRANT USAGE ON SCHEMA pgtoken TO PUBLIC;
GRANT SELECT ON pgtoken.vocabulary TO PUBLIC;

-- Schema USAGE reaches the functions too, and PUBLIC holds EXECUTE on a function by default, so
-- the four that mutate a vocabulary are taken back explicitly. They are also stopped by ordinary
-- object permissions -- `create_vocabulary` needs INSERT on the catalog, `train` and
-- `load_mapping` need CREATE on schema pgtoken for the artefact function they write,
-- `drop_vocabulary` needs to own the domain -- but saying so here makes the intent plain. Grant
-- EXECUTE back to whoever administers vocabularies.
REVOKE EXECUTE ON FUNCTION pgtoken.create_vocabulary(text, int, text, int, text) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION pgtoken.drop_vocabulary(text) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION pgtoken.train(text, text, int) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION pgtoken.load_mapping(text, text) FROM PUBLIC;
"#,
    name = "privileges",
    // Emitted last, after every function it names exists. The alternative -- a `requires` list --
    // would have to reach private items in three other modules.
    finalize,
);

/// Create a vocabulary and its `tokens.<name>` domain, and return its id.
///
/// Two kinds. A **preset** vocabulary (`create_vocabulary('o200k')`, or any name with
/// `preset => 'o200k'`) uses a ranking and mapping bundled in the extension: it defaults to `freq`
/// and works with no further setup. Any other vocabulary declares its own `vocab_size` and brings
/// its own artefacts through `train` and `load_mapping`. A name that is a preset's name always
/// means that preset.
#[pg_extern]
fn create_vocabulary(
    name: &str,
    vocab_size: default!(Option<i32>, "NULL"),
    compression: default!(Option<&str>, "NULL"),
    id: default!(Option<i32>, "NULL"),
    preset: default!(Option<&str>, "NULL"),
) -> i32 {
    if name.contains('\'') || name.contains('"') || name.contains('\\') {
        bail(format!(
            "vocabulary name {name:?} may not contain quotes or backslashes"
        ));
    }
    let preset: Option<&'static Preset> = match preset {
        Some(p) => Some(preset::by_name(p).unwrap_or_else(|| {
            bail(format!(
                "unknown preset {p:?}; the bundled presets are {}",
                preset::names()
            ))
        })),
        None => preset::by_name(name),
    };
    let vocab_size = match (preset, vocab_size) {
        (Some(p), Some(v)) if v as i64 != p.vocab_size as i64 => bail(format!(
            "vocab_size {v} does not match the {} preset, which has {} ids",
            p.name, p.vocab_size
        )),
        (Some(p), _) => p.vocab_size as i32,
        (None, Some(v)) => v,
        (None, None) => bail(format!(
            "vocab_size is required, unless the vocabulary uses a preset ({})",
            preset::names()
        )),
    };
    if vocab_size < 1 {
        bail("vocab_size is required and must be at least 1");
    }
    if vocab_size as i64 > MAX_VOCAB_SIZE {
        bail(format!(
            "vocab_size {vocab_size} exceeds the {MAX_VOCAB_SIZE} this format can address"
        ));
    }
    // Validate before touching the catalog, so a bad call leaves nothing behind.
    let compression = compression.unwrap_or(if preset.is_some() { "freq" } else { "raw" });
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

    // Preset vocabularies take ids from their preset's reserved range, because a value's header
    // carries only the id and the read paths find the bundled artefacts by it (see preset.rs).
    let assigned = match id {
        Some(v) if v < 1 || v > u16::MAX as i32 => {
            bail(format!("vocabulary id {v} is out of range (1..65535)"))
        }
        Some(v) => {
            let owner = preset::for_id(v as u16);
            match (preset, owner) {
                (Some(p), o) if o.map(|o| o.code) != Some(p.code) => bail(format!(
                    "vocabulary id {v} is outside the {} preset's ids {}..{}",
                    p.name,
                    p.ids.start(),
                    p.ids.end()
                )),
                (None, Some(o)) => bail(format!(
                    "vocabulary id {v} is reserved for vocabularies on the {} preset ({}..{})",
                    o.name,
                    o.ids.start(),
                    o.ids.end()
                )),
                _ => {}
            }
            if preset.is_none() && has_custom_artefact(v as u16) {
                bail(format!(
                    "vocabulary id {v} already has an artefact function \
                     (pgtoken.artefact_{v}_ranking or _mapping) but no catalog row; drop it if \
                     nothing references it, or pick another id"
                ));
            }
            v
        }
        None => next_free_id(preset),
    };

    Spi::run_with_args(
        "INSERT INTO pgtoken.vocabulary (id, name, vocab_size, compression, width, preset) \
         VALUES ($1, $2, $3, $4, $5::smallint, $6)",
        &[
            assigned.into(),
            name.into(),
            vocab_size.into(),
            compression.into(),
            width.into(),
            preset.map(|p| p.name).into(),
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

    // A `pgtoken.text` that accepts this exact domain, so `text(body)` resolves to the function
    // instead of PostgreSQL's cast-to-`text` syntax.
    //
    // `typename(expr)` is a legal spelling of `CAST(expr AS typename)`, and PostgreSQL only falls
    // back to reading it that way when no function matches the argument type exactly. The real
    // `pgtoken.text` takes `pgtoken.tokens`, and a column takes the domain, so nothing matched and
    // `text(body)` silently returned the id list. Declaring one per domain gives it the exact match
    // it was looking for. An explicit `body::text` is still a cast, which is correct.
    //
    // Only helps when `pgtoken` is on `search_path`; without it no unqualified `text` is visible at
    // all. That is the population this protects, since anyone else already qualifies every call.
    Spi::run(&format!(
        "CREATE FUNCTION pgtoken.text(tokens.{ident}) RETURNS text \
         LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE \
         AS $wrap$ SELECT pgtoken.text($1::pgtoken.tokens) $wrap$",
        ident = quote_ident(name)
    ))
    .unwrap_or_else(|e| bail(e));

    assigned
}

/// Whether an artefact function exists for an id. Only a `TRUNCATE` of the catalog (which the
/// immutability trigger cannot see) leaves one behind without its row; a vocabulary created at
/// that id would inherit it.
fn has_custom_artefact(id: u16) -> bool {
    registry::exists(id, Artefact::Ranking) || registry::exists(id, Artefact::Mapping)
}

/// The id a new vocabulary gets when none is given.
///
/// For a preset, the lowest unused id in its range. Otherwise `max(id) + 1` below every preset
/// range, skipping any id that still has an artefact function (see [`has_custom_artefact`]).
fn next_free_id(preset: Option<&Preset>) -> i32 {
    if let Some(p) = preset {
        return Spi::get_one_with_args::<i32>(
            "SELECT min(i) FROM generate_series($1, $2) i \
             WHERE NOT EXISTS (SELECT 1 FROM pgtoken.vocabulary v WHERE v.id = i)",
            &[(*p.ids.start() as i32).into(), (*p.ids.end() as i32).into()],
        )
        .unwrap_or_else(|e| bail(e))
        .unwrap_or_else(|| {
            bail(format!(
                "all {} ids for the {} preset are in use",
                p.ids.len(),
                p.name
            ))
        });
    }
    let reserved = preset::ALL
        .iter()
        .map(|p| *p.ids.start() as i32)
        .min()
        .unwrap_or(u16::MAX as i32 + 1);
    let mut candidate = Spi::get_one_with_args::<i32>(
        "SELECT coalesce(max(id), 0)::int + 1 FROM pgtoken.vocabulary WHERE id < $1",
        &[reserved.into()],
    )
    .unwrap_or_else(|e| bail(e))
    .unwrap_or(1);
    while candidate < reserved && has_custom_artefact(candidate as u16) {
        candidate += 1;
    }
    if candidate >= reserved {
        bail(format!(
            "no vocabulary id below {reserved} is free; ids from {reserved} up are reserved for \
             presets"
        ));
    }
    candidate
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
    // The per-domain `pgtoken.text` overload must go first: a function's parameter type counts as
    // a dependency, so with it in place `DROP DOMAIN` fails even when no column uses the domain.
    Spi::run(&format!(
        "DROP FUNCTION IF EXISTS pgtoken.text(tokens.{ident}); DROP DOMAIN tokens.{ident}",
        ident = quote_ident(name)
    ))
    .unwrap_or_else(|e| bail(e));
}

/// Resolve a name to the three fields a typmod needs. Called at DDL time only.
pub fn lookup_by_name(name: &str) -> Option<Vocabulary> {
    Spi::connect(|client| {
        let rows = client
            .select(
                "SELECT id, compression, width::int, preset FROM pgtoken.vocabulary WHERE name = $1",
                Some(1),
                &[name.into()],
            )
            .ok()?;
        let row = rows.first();
        let id: i32 = row.get(1).ok()??;
        let compression: String = row.get(2).ok()??;
        let width: i32 = row.get(3).ok()??;
        let preset: Option<String> = row.get(4).ok()?;
        Some(Vocabulary {
            id: id as u16,
            compression: compression_code(&compression),
            width: width as u8,
            preset: preset
                .and_then(|p| preset::by_name(&p))
                .map_or(0, |p| p.code),
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
///
/// Both artefacts get a digest and a length, not just the ranking. The digest is sold as the way
/// two machines check they hold the same vocabulary, and a vocabulary is the pair: identical
/// rankings with different mappings are different vocabularies, and reporting one hash for the
/// two of them would let them compare equal.
///
/// A preset vocabulary reports the bundled artefacts: the mapping always, the ranking when it
/// uses `freq` (a `raw` vocabulary never reads one).
#[pg_extern]
#[allow(clippy::type_complexity)] // An 11-column TableIterator tuple; a type alias would just move the noise.
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
        name!(rank_sha256, Option<String>),
        name!(rank_bytes, Option<i64>),
        name!(mapped, Option<i32>),
        name!(map_sha256, Option<String>),
        name!(map_bytes, Option<i64>),
        name!(preset, Option<String>),
    ),
> {
    let v =
        lookup_by_name(name).unwrap_or_else(|| bail(format!("vocabulary {name:?} does not exist")));
    let size = vocab_size_for(v.id)
        .unwrap_or_else(|| bail(format!("vocabulary {name:?} has no catalog row")));
    let p = preset::by_code(v.preset);

    // An unfilled artefact is a normal state, not a fault, so an absent one becomes NULLs. One
    // that is there but does not parse is a real fault and must say so rather than collapse into
    // the same NULLs, which would silently read as "never filled".
    let describe = |a: Artefact| match registry::describe(v.id, a) {
        Ok(None) => (None, None, None),
        Ok(Some((n, digest, len))) => (Some(n as i32), Some(digest), Some(len as i64)),
        Err(e) => bail(format!(
            "vocabulary {name:?} has a {} but it could not be read: {e}",
            a.as_str()
        )),
    };
    let (ranked, rank_sha256, rank_bytes) = if p.is_some() && v.compression != COMPRESSION_FREQ {
        (None, None, None)
    } else {
        describe(Artefact::Ranking)
    };
    let (mapped, map_sha256, map_bytes) = describe(Artefact::Mapping);

    TableIterator::once((
        v.id as i32,
        size as i32,
        compression_name(v.compression).to_string(),
        v.width as i32,
        ranked,
        rank_sha256,
        rank_bytes,
        mapped,
        map_sha256,
        map_bytes,
        p.map(|p| p.name.to_string()),
    ))
}

/// Everything a client needs to encode and decode a vocabulary's values itself.
///
/// `codec` is the codec name the client codec takes (`raw8`, `raw16`, `raw24` or `freq`), and
/// `ranking` is the trained ranking's bytes for a `freq` vocabulary: NULL for `raw`, and NULL for
/// a `freq` vocabulary that has not been trained yet. The ranking comes back as bytes rather than
/// a path, so this works over any connection, including to a hosted server whose disk the client
/// cannot see.
#[pg_extern(stable, strict)]
fn vocabulary_codec(
    name: &str,
) -> TableIterator<
    'static,
    (
        name!(id, i32),
        name!(vocab_size, i32),
        name!(codec, String),
        name!(ranking, Option<Vec<u8>>),
    ),
> {
    let v =
        lookup_by_name(name).unwrap_or_else(|| bail(format!("vocabulary {name:?} does not exist")));
    let size = vocab_size_for(v.id)
        .unwrap_or_else(|| bail(format!("vocabulary {name:?} has no catalog row")));
    let ranking = if v.compression == COMPRESSION_FREQ {
        registry::artefact_bytes(v.id, Artefact::Ranking).unwrap_or_else(|e| bail(e))
    } else {
        None
    };
    TableIterator::once((
        v.id as i32,
        size as i32,
        crate::typmod::codec_for(v).as_str().to_string(),
        ranking,
    ))
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn vocabulary_codec_reports_a_raw_vocabulary() {
        Spi::run("SELECT pgtoken.create_vocabulary('vc_raw', 60000, id => 62401)").expect("create");
        let (codec, ranking) = Spi::get_two::<String, Vec<u8>>(
            "SELECT codec, ranking FROM pgtoken.vocabulary_codec('vc_raw')",
        )
        .expect("query failed");
        assert_eq!(codec.as_deref(), Some("raw16"));
        assert_eq!(ranking, None, "raw has no ranking");
    }

    #[pg_test]
    fn vocabulary_codec_ships_the_ranking_bytes() {
        Spi::run(
            "SELECT pgtoken.create_vocabulary('vc_freq', 300, compression => 'freq', \
                                              id => 62402)",
        )
        .expect("create");
        let before =
            Spi::get_one::<Vec<u8>>("SELECT ranking FROM pgtoken.vocabulary_codec('vc_freq')")
                .expect("query failed");
        assert_eq!(
            before, None,
            "an untrained freq vocabulary has no ranking yet"
        );

        Spi::run("SELECT pgtoken.train('vc_freq', $$SELECT ARRAY[5,5,5,9]::int[]$$)")
            .expect("train");
        let (id, codec, ranking) = Spi::get_three::<i32, String, Vec<u8>>(
            "SELECT id, codec, ranking FROM pgtoken.vocabulary_codec('vc_freq')",
        )
        .expect("query failed");
        assert_eq!(id, Some(62402));
        assert_eq!(codec.as_deref(), Some("freq"));
        let ranking = ranking.expect("a trained freq vocabulary ships its ranking");
        let table = pgtoken_core::RankTable::from_bytes(&ranking).expect("valid ranking bytes");
        assert_eq!(table.k(), 2, "two distinct tokens were ranked");
        let len = Spi::get_one::<i64>("SELECT rank_bytes FROM pgtoken.vocabulary_info('vc_freq')")
            .expect("query failed");
        assert_eq!(
            len,
            Some(ranking.len() as i64),
            "the whole ranking, byte for byte"
        );
    }

    #[pg_test(error = "vocabulary \"vc_missing\" does not exist")]
    fn vocabulary_codec_rejects_an_unknown_name() {
        Spi::run("SELECT * FROM pgtoken.vocabulary_codec('vc_missing')").unwrap();
    }

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
        // The long typmod form (name, id, compression, width) that pg_dump needs; see typmod.rs.
        let base = base.expect("format_type returned NULL");
        assert!(
            base.starts_with("pgtoken.tokens('dom2', ") && base.ends_with(", 'raw', 1)"),
            "{base}"
        );

        // `format_type` reading the catalog correctly is necessary but not sufficient: it could
        // render right while a write through the domain still stored the unresolved raw24
        // carrier. Prove the domain's typmod actually reaches the write path by checking a value
        // inserted through it lands at dom2's declared width (256 -> raw8, one byte per token)
        // rather than the wider unresolved encoding.
        ensure_bytea_cast();
        Spi::run("CREATE TABLE dom2_docs (body tokens.dom2)").expect("create table");
        Spi::run("INSERT INTO dom2_docs (body) VALUES ('{1,2,3}')").expect("insert");
        let len = Spi::get_one::<i32>("SELECT length(pgtoken.tokens_send(body)) FROM dom2_docs")
            .expect("query failed");
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
    fn unqualified_text_resolves_to_the_function_not_the_cast() {
        // `text(body)` is a legal spelling of `CAST(body AS text)`, and PostgreSQL only reads it
        // that way when no function matches the argument type. Before the per-domain overload it
        // silently returned the id list from a column someone had loaded a mapping for.
        create("'ovl_hit', 8, id => 62301");
        Spi::run(
            "CREATE TABLE ovl_hit_t (b tokens.ovl_hit); \
             INSERT INTO ovl_hit_t VALUES ('{0,1}'); \
             CREATE TEMP TABLE ovl_stage (id int, bytes bytea); \
             INSERT INTO ovl_stage VALUES (0, 'hello'), (1, ' world');",
        )
        .expect("setup");
        Spi::run("SELECT pgtoken.load_mapping('ovl_hit', 'SELECT id, bytes FROM ovl_stage')")
            .expect("load_mapping");

        // `SET search_path` is what makes the bare name visible at all; without it PostgreSQL sees
        // no unqualified `text` function and falls back to the cast no matter what we declare.
        let got = Spi::get_one::<String>(
            "SET search_path = public, pgtoken; SELECT text(b) FROM ovl_hit_t",
        )
        .expect("query failed");
        assert_eq!(
            got,
            Some("hello world".to_string()),
            "the overload must win"
        );

        let qualified =
            Spi::get_one::<String>("SELECT pgtoken.text(b) FROM ovl_hit_t").expect("query failed");
        assert_eq!(qualified, Some("hello world".to_string()));

        // An explicit cast is still a cast. Only the ambiguous spelling changed.
        let cast = Spi::get_one::<String>("SELECT b::text FROM ovl_hit_t").expect("query failed");
        assert_eq!(cast, Some("{0,1}".to_string()), "::text must stay a cast");
    }

    #[pg_test]
    fn the_overload_can_back_an_index() {
        // A wrapper that could not be indexed would be useless: backing a GIN index over
        // to_tsvector is most of why pgtoken.text exists.
        create("'ovl_idx', 8, id => 62302");
        Spi::run(
            "CREATE TABLE ovl_idx_t (b tokens.ovl_idx); \
             INSERT INTO ovl_idx_t VALUES ('{0,1}'); \
             CREATE TEMP TABLE ovl_idx_stage (id int, bytes bytea); \
             INSERT INTO ovl_idx_stage VALUES (0, 'hello'), (1, ' world');",
        )
        .expect("setup");
        Spi::run("SELECT pgtoken.load_mapping('ovl_idx', 'SELECT id, bytes FROM ovl_idx_stage')")
            .expect("load_mapping");
        Spi::run(
            "SET search_path = public, pgtoken; \
             CREATE INDEX ON ovl_idx_t USING gin (to_tsvector('english', text(b)))",
        )
        .expect("the overload must be IMMUTABLE enough to index");
        let hits = Spi::get_one::<i64>(
            "SET search_path = public, pgtoken; \
             SELECT count(*) FROM ovl_idx_t \
             WHERE to_tsvector('english', text(b)) @@ to_tsquery('world')",
        )
        .expect("query failed");
        assert_eq!(hits, Some(1));
    }

    #[pg_test]
    fn dropping_a_vocabulary_removes_its_overload_too() {
        // The overload depends on the domain, so a stale one would block DROP DOMAIN outright.
        create("'ovl_drop', 300, id => 62303");
        Spi::run("SELECT pgtoken.drop_vocabulary('ovl_drop')").expect("drop must succeed");
        let left = Spi::get_one::<i64>(
            "SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace \
             WHERE n.nspname = 'pgtoken' AND p.proname = 'text' \
             AND pg_get_function_identity_arguments(p.oid) LIKE '%ovl_drop%'",
        )
        .expect("query failed");
        assert_eq!(left, Some(0), "the overload must go with the domain");
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

    #[pg_test]
    fn vocabulary_info_digests_each_artefact_separately() {
        // Two vocabularies trained on the same corpus and mapped differently. One digest over
        // the ranking alone would call them equal, which is exactly the check the column is sold
        // for — "two machines verify they hold the same vocabulary".
        for (name, id, bytes) in [("vi_da", 62211, "alpha"), ("vi_db", 62212, "beta")] {
            Spi::run(&format!(
                "SELECT pgtoken.create_vocabulary('{name}', 8, compression => 'freq', id => {id}); \
                 SELECT pgtoken.train('{name}', $$SELECT ARRAY[1,1,2]::int[]$$); \
                 SELECT pgtoken.load_mapping('{name}', $$SELECT 1::int, '{bytes}'::bytea$$);"
            ))
            .expect("setup");
        }
        let (ra, rb) = Spi::get_two::<String, String>(
            "SELECT (SELECT rank_sha256 FROM pgtoken.vocabulary_info('vi_da')), \
                    (SELECT rank_sha256 FROM pgtoken.vocabulary_info('vi_db'))",
        )
        .expect("query failed");
        assert_eq!(ra, rb, "the same corpus must produce the same ranking");
        let (ma, mb) = Spi::get_two::<String, String>(
            "SELECT (SELECT map_sha256 FROM pgtoken.vocabulary_info('vi_da')), \
                    (SELECT map_sha256 FROM pgtoken.vocabulary_info('vi_db'))",
        )
        .expect("query failed");
        assert_eq!(
            ma.as_ref().map(|s| s.len()),
            Some(64),
            "a sha256 hex digest"
        );
        assert_ne!(ma, mb, "different mappings are different vocabularies");

        let (rank_bytes, map_bytes) = Spi::get_two::<i64, i64>(
            "SELECT rank_bytes, map_bytes FROM pgtoken.vocabulary_info('vi_da')",
        )
        .expect("query failed");
        assert!(rank_bytes.unwrap() > 0 && map_bytes.unwrap() > 0);
    }

    #[pg_test]
    fn a_non_owner_can_read_a_column_and_detokenize() {
        // Without a grant on `pgtoken.vocabulary` the extension is unusable by anyone but its
        // owner: SPI inside the type's own C functions runs as the invoking role, so even a
        // literal cast fails on a catalog the user never named.
        Spi::run(
            "SELECT pgtoken.create_vocabulary('grantee', 8, id => 62213); \
             CREATE TEMP TABLE g_stage (id int, bytes bytea); \
             INSERT INTO g_stage VALUES (0, 'Hello'), (1, ', '), (2, 'world'), (3, '!'); \
             SELECT pgtoken.load_mapping('grantee', 'SELECT id, bytes FROM g_stage'); \
             CREATE TABLE g_docs (body tokens.grantee); \
             INSERT INTO g_docs VALUES ('{0,1,2,3}'); \
             GRANT SELECT ON g_docs TO PUBLIC; \
             CREATE ROLE pgtoken_reader;",
        )
        .expect("setup");

        Spi::run("SET ROLE pgtoken_reader").expect("set role");
        let size =
            Spi::get_one::<i32>("SELECT vocab_size FROM pgtoken.vocabulary WHERE name = 'grantee'")
                .expect("a non-owner must be able to read the catalog");
        assert_eq!(size, Some(8));
        let text = Spi::get_one::<String>("SELECT pgtoken.text(body) FROM g_docs")
            .expect("a non-owner must be able to detokenize");
        assert_eq!(text, Some("Hello, world!".to_string()));
        Spi::run("RESET ROLE").expect("reset role");
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

    // ── presets ──────────────────────────────────────────────────────────────────────────

    #[pg_test]
    fn a_preset_works_with_no_setup() {
        // The whole quickstart: no vocab_size, no train, no load_mapping.
        let id = create("'o200k'");
        assert_eq!(id, 65280, "the first id in o200k's reserved range");
        Spi::run(
            "CREATE TABLE p_docs (body tokens.o200k); \
             INSERT INTO p_docs VALUES ('{13225,11,2375,0}');",
        )
        .expect("setup");
        let (text, ids, codec) = Spi::get_three::<String, Vec<i32>, String>(
            "SELECT pgtoken.text(body), body::int[], (pgtoken.describe(body)).codec FROM p_docs",
        )
        .expect("query failed");
        assert_eq!(text.as_deref(), Some("Hello, world!"));
        assert_eq!(ids, Some(vec![13225, 11, 2375, 0]));
        assert_eq!(codec.as_deref(), Some("freq"));

        let (size, compression, width) = Spi::get_three::<i32, String, i32>(
            "SELECT vocab_size, compression, width FROM pgtoken.vocabulary_info('o200k')",
        )
        .expect("query failed");
        assert_eq!(
            (size, compression.as_deref(), width),
            (Some(200019), Some("freq"), Some(3))
        );
        let (ranked, mapped) =
            Spi::get_two::<i32, i32>("SELECT ranked, mapped FROM pgtoken.vocabulary_info('o200k')")
                .expect("query failed");
        assert_eq!(ranked, Some(77463));
        assert!(mapped.unwrap() > 199_000, "{mapped:?}");
        let (preset, rank_sha) = Spi::get_two::<String, String>(
            "SELECT preset, rank_sha256 FROM pgtoken.vocabulary_info('o200k')",
        )
        .expect("query failed");
        assert_eq!(preset.as_deref(), Some("o200k"));
        assert_eq!(
            rank_sha.as_deref(),
            Some("743afb3cae0eac236f9bece91973ebe3038539b557ac3dcc955bb0d9a1de42af")
        );

        let (codec, ranking) = Spi::get_two::<String, Vec<u8>>(
            "SELECT codec, ranking FROM pgtoken.vocabulary_codec('o200k')",
        )
        .expect("query failed");
        assert_eq!(codec.as_deref(), Some("freq"));
        assert_eq!(
            ranking.map(|r| r.len()),
            Some(309_864),
            "the bundled ranking, byte for byte"
        );
    }

    #[pg_test]
    fn a_preset_can_back_another_name_and_raw_compression() {
        let chat = create("'chat', preset => 'o200k'");
        let raw = create("'chat_raw', preset => 'o200k', compression => 'raw'");
        assert_eq!((chat, raw), (65280, 65281));
        let (a, b, codec) = Spi::get_three::<String, String, String>(
            "SELECT pgtoken.text('{13225,0}'::tokens.chat), \
                    pgtoken.text('{13225,0}'::tokens.chat_raw), \
                    (pgtoken.describe('{13225,0}'::tokens.chat_raw)).codec",
        )
        .expect("query failed");
        assert_eq!(a.as_deref(), Some("Hello!"));
        assert_eq!(b.as_deref(), Some("Hello!"));
        assert_eq!(codec.as_deref(), Some("raw24"));
        let ranking =
            Spi::get_one::<Vec<u8>>("SELECT ranking FROM pgtoken.vocabulary_codec('chat_raw')")
                .expect("query failed");
        assert_eq!(ranking, None, "raw reads no ranking");
        let typ = Spi::get_one::<String>(
            "SELECT format_type(typbasetype, typtypmod) FROM pg_type \
             WHERE oid = 'tokens.chat'::regtype",
        )
        .expect("query failed");
        assert_eq!(
            typ.as_deref(),
            Some("pgtoken.tokens('chat', 65280, 'freq', 3, 'o200k')")
        );
    }

    #[pg_test]
    fn a_preset_value_needs_no_catalog_row() {
        // What a restore or a replica sees before pgtoken.vocabulary's rows exist: the typmod's
        // long form names the preset, and the read paths find the artefacts by id range.
        let text = Spi::get_one::<String>(
            "SELECT pgtoken.text('{13225,11,2375,0}'::pgtoken.tokens(nowhere, 65300, freq, 3, o200k))",
        )
        .expect("query failed");
        assert_eq!(text.as_deref(), Some("Hello, world!"));
    }

    #[pg_test(error = "vocab_size 50257 does not match the o200k preset, which has 200019 ids")]
    fn a_preset_refuses_a_different_vocab_size() {
        create("'o200k', 50257");
    }

    #[pg_test(error = "unknown preset \"cl100k\"; the bundled presets are 'o200k'")]
    fn an_unknown_preset_is_refused() {
        create("'x', preset => 'cl100k'");
    }

    #[pg_test(
        error = "vocabulary id 65300 is reserved for vocabularies on the o200k preset (65280..65535)"
    )]
    fn a_custom_vocabulary_cannot_take_a_preset_id() {
        create("'x', 300, id => 65300");
    }

    #[pg_test(error = "vocabulary id 5 is outside the o200k preset's ids 65280..65535")]
    fn a_preset_vocabulary_cannot_take_a_custom_id() {
        create("'x', preset => 'o200k', id => 5");
    }

    #[pg_test(error = "vocab_size is required, unless the vocabulary uses a preset ('o200k')")]
    fn a_custom_vocabulary_needs_a_size() {
        create("'x'");
    }

    #[pg_test]
    fn custom_ids_stay_below_the_preset_range() {
        create("'o200k'");
        let next = create("'after_preset', 300");
        assert!(next < 65280, "{next}");
    }
}
