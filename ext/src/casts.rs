//! Casts into and out of `pgtoken.tokens`.
//!
//! These replace `pgtoken.encode` and `pgtoken.decode`. `int[] → tokens` is an *assignment*
//! cast, so a plain `INSERT ... VALUES (ARRAY[1,2])` works while accidental coercions in
//! expressions still have to be spelled out. It takes the three-argument cast signature so it
//! receives the target column's typmod — without it there would be no width to encode at.

use pgrx::prelude::*;

use crate::registry::bail;
use crate::tokens::{decode_value, encode_for, require_vocabulary, validate};

/// SQL `int[]` to token IDs. Rejects NULLs and negatives rather than coercing them, since either
/// would silently store a different sequence than the caller meant.
pub fn ids_from_sql(ids: Vec<Option<i32>>) -> Vec<u32> {
    ids.into_iter()
        .map(|o| match o {
            None => bail("token id array must not contain NULL"),
            Some(v) if v < 0 => bail(format!("token id {v} is negative")),
            Some(v) => v as u32,
        })
        .collect()
}

/// The three-argument cast form: `typmod` is the target column's, `explicit` distinguishes an
/// explicit cast from an assignment. Both are supplied by PostgreSQL.
#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_from_int4array_impl(ids: Vec<Option<i32>>, typmod: i32, _explicit: bool) -> Vec<u8> {
    encode_for(&ids_from_sql(ids), typmod)
}

/// `pgtoken.tokens → int[]`: the direct replacement for the deleted `pgtoken.decode`.
///
/// Calls `require_vocabulary` first, like every other path that hands a value back to a caller.
/// Without it, a bare (typmod-less) column's unresolved values would decode to plausible-looking
/// ids here even though `::text` refuses them — the exact hole `tokens_send` was changed to close.
#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_to_int4array_impl(value: &[u8]) -> Vec<i32> {
    require_vocabulary(value);
    decode_value(value)
        .into_iter()
        .map(|id| id as i32)
        .collect()
}

/// `pgtoken.tokens → bytea`: the documented idiom for drivers that cannot reach binary mode.
///
/// Calls `require_vocabulary` first, then copies the bytes through untouched — it does not decode,
/// same as `tokens_send_impl`. Without the check, a client relying on this idiom would get a
/// plausible-looking blob from a bare column and never learn the column is misdeclared.
#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_to_bytea_impl(value: &[u8]) -> Vec<u8> {
    require_vocabulary(value);
    value.to_vec()
}

/// `bytea → pgtoken.tokens`: one of the trusted binary paths, alongside `RECEIVE`. It checks only
/// the 12-byte header through [`validate`] and hands the payload through untouched — it does
/// **not** scan the ids, so do not mistake it for a validating entry point. See `validate`'s doc
/// comment in `tokens.rs` for what a header check guarantees and what it deliberately does not.
#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_from_bytea_impl(value: &[u8]) -> Vec<u8> {
    validate(value)
}

extension_sql!(
    r#"
CREATE FUNCTION pgtoken.tokens_from_int4array(int[], integer, boolean)
    RETURNS pgtoken.tokens
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE
    AS 'MODULE_PATHNAME', 'tokens_from_int4array_impl_wrapper';

CREATE FUNCTION pgtoken.tokens_to_int4array(pgtoken.tokens) RETURNS int[]
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE
    AS 'MODULE_PATHNAME', 'tokens_to_int4array_impl_wrapper';

CREATE FUNCTION pgtoken.tokens_to_bytea(pgtoken.tokens) RETURNS bytea
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE
    AS 'MODULE_PATHNAME', 'tokens_to_bytea_impl_wrapper';

CREATE FUNCTION pgtoken.tokens_from_bytea(bytea) RETURNS pgtoken.tokens
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE
    AS 'MODULE_PATHNAME', 'tokens_from_bytea_impl_wrapper';

-- Assignment, not implicit: a plain INSERT of an int[] works, while accidental coercions in
-- expressions still have to be spelled out.
CREATE CAST (int[] AS pgtoken.tokens)
    WITH FUNCTION pgtoken.tokens_from_int4array(int[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (pgtoken.tokens AS int[])
    WITH FUNCTION pgtoken.tokens_to_int4array(pgtoken.tokens);

CREATE CAST (pgtoken.tokens AS bytea)
    WITH FUNCTION pgtoken.tokens_to_bytea(pgtoken.tokens);

CREATE CAST (bytea AS pgtoken.tokens)
    WITH FUNCTION pgtoken.tokens_from_bytea(bytea);
"#,
    name = "tokens_casts",
    requires = [
        "tokens_type",
        tokens_from_int4array_impl,
        tokens_to_int4array_impl,
        tokens_to_bytea_impl,
        tokens_from_bytea_impl
    ],
);

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn insert_from_an_int_array_needs_no_explicit_cast() {
        Spi::run(
            "SELECT pgtoken.create_vocabulary('c1', 32000); \
             CREATE TABLE c1_docs (body tokens.c1); \
             INSERT INTO c1_docs (body) VALUES (ARRAY[1,2,3]);",
        )
        .expect("insert failed");
        let got =
            Spi::get_one::<Vec<i32>>("SELECT body::int[] FROM c1_docs").expect("query failed");
        assert_eq!(got, Some(vec![1, 2, 3]));
    }

    #[pg_test]
    fn the_cast_honours_the_target_width() {
        Spi::run(
            "SELECT pgtoken.create_vocabulary('c_small', 256); \
             SELECT pgtoken.create_vocabulary('c_big', 200019); \
             CREATE TABLE c_small_t (body tokens.c_small); \
             CREATE TABLE c_big_t (body tokens.c_big); \
             INSERT INTO c_small_t (body) VALUES (ARRAY[1,2,3]); \
             INSERT INTO c_big_t (body) VALUES (ARRAY[1,2,3]);",
        )
        .expect("setup");
        let (small, big) = Spi::get_two::<i32, i32>(
            "SELECT (SELECT length(body::bytea) FROM c_small_t), \
                    (SELECT length(body::bytea) FROM c_big_t)",
        )
        .expect("query failed");
        assert_eq!(small, Some(12 + 3), "raw8 via the target typmod");
        assert_eq!(big, Some(12 + 9), "raw24 via the target typmod");
    }

    #[pg_test]
    fn bytea_roundtrips_both_ways() {
        Spi::run("SELECT pgtoken.create_vocabulary('c2', 200019)").expect("create");
        let got = Spi::get_one::<Vec<i32>>(
            "SELECT ('{7,8}'::pgtoken.tokens('c2')::bytea)::pgtoken.tokens::int[]",
        )
        .expect("query failed");
        assert_eq!(got, Some(vec![7, 8]));
    }

    #[pg_test]
    fn tokens_beat_the_int_array_they_replace() {
        // int[] costs 4 bytes per element plus array overhead; raw16 costs 2.
        Spi::run(
            "SELECT pgtoken.create_vocabulary('c3', 60000); \
             CREATE TABLE c3_docs (body tokens.c3); \
             INSERT INTO c3_docs (body) \
               SELECT array_agg(i % 60000)::int[] FROM generate_series(1,512) i;",
        )
        .expect("setup");
        let (arr, packed) = Spi::get_two::<i32, i32>(
            "SELECT pg_column_size(body::int[]), pg_column_size(body) FROM c3_docs",
        )
        .expect("query failed");
        let (arr, packed) = (arr.unwrap(), packed.unwrap());
        assert!(
            packed < arr,
            "packed ({packed} B) should beat int[] ({arr} B)"
        );
    }

    #[pg_test(error = "token id array must not contain NULL")]
    fn cast_rejects_null_elements() {
        Spi::run("SELECT pgtoken.create_vocabulary('c4', 300)").expect("create");
        Spi::get_one::<Vec<i32>>("SELECT ARRAY[1,NULL,3]::pgtoken.tokens('c4')::int[]").unwrap();
    }

    #[pg_test(error = "cannot read a pgtoken.tokens value that has no vocabulary")]
    fn int_array_cast_refuses_a_value_without_a_vocabulary() {
        // `::int[]` is the direct replacement for the deleted `pgtoken.decode`, so it has to
        // refuse an unresolved value exactly as `::text` does -- otherwise a client on a bare
        // column gets plausible-looking ids back and never learns the column is misdeclared.
        Spi::run("CREATE TABLE bare_ia (x pgtoken.tokens)").expect("create table");
        Spi::run("INSERT INTO bare_ia VALUES ('{1,2,3}')").expect("insert");
        Spi::get_one::<Vec<i32>>("SELECT x::int[] FROM bare_ia").unwrap();
    }

    #[pg_test(error = "cannot read a pgtoken.tokens value that has no vocabulary")]
    fn bytea_cast_refuses_a_value_without_a_vocabulary() {
        // `::bytea` is the documented idiom for drivers that cannot reach binary mode, so it has
        // to refuse an unresolved value too -- otherwise it is the one accepting read path left
        // on a bare column, reopening the hole `tokens_send` was changed to close.
        Spi::run("CREATE TABLE bare_ba (x pgtoken.tokens)").expect("create table");
        Spi::run("INSERT INTO bare_ba VALUES ('{1,2,3}')").expect("insert");
        Spi::get_one::<Vec<u8>>("SELECT x::bytea FROM bare_ba").unwrap();
    }
}
