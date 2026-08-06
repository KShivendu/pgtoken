//! The `pgtoken.tokens` type: token IDs stored compressed, rendered as `{1,2,3}`.
//!
//! Text I/O renders token IDs rather than the compressed bytes, so `psql` is readable and
//! `pg_dump` round-trips to byte-identical values. Binary I/O hands over the stored bytes
//! verbatim, which is the path an agent takes.
//!
//! The `CREATE TYPE` is hand-written in `extension_sql!` because pgrx 0.19 cannot generate one
//! with custom `SEND`/`RECEIVE`/`TYPMOD_IN`. The `#[pg_extern]` functions below therefore carry
//! an `_impl` suffix and speak `bytea`; the SQL block re-declares each one against
//! `pgtoken.tokens`, pointing at the symbol pgrx generated. That is ABI-safe because the type is
//! a varlena with the same C representation as `bytea`.
//!
//! ## Why a typmod arrives in two steps
//!
//! PostgreSQL does **not** hand a string literal's typmod to the type's input function. In
//! `coerce_type()` (`parse_coerce.c`) an unknown-type `Const` is fed to `typinput` with typmod
//! `-1` for every type but `interval` — "any length constraint will be applied later by our
//! caller" — and the caller then applies it through a *length coercion cast*, a `pg_cast` row
//! from the type to itself whose function takes `(value, typmod, is_explicit)`. Text casts
//! (`CoerceViaIO`) do the same. Only `COPY … FROM` in text format, and `RECEIVE`, get the real
//! typmod.
//!
//! That is fine for types whose typmod is a constraint (`varchar(3)`, `vector(3)`), but here the
//! typmod *chooses the storage width*, so `tokens_in` with typmod `-1` has nothing to encode
//! with. It therefore emits an **unresolved** value: `raw24`, the widest packing the format has,
//! carrying `vocabulary_id = 0`, which the header format already reserves to mean "no
//! vocabulary". `tokens_typmod_apply` then re-encodes it at the column's width. This is not an
//! auto width — nothing looks at the ids to pick `raw24`; it is used unconditionally, and a
//! value still wearing it cannot be read back.
//!
//! Reading an unresolved value is an error, so `'{1,2,3}'::pgtoken.tokens` (no typmod, hence no
//! length coercion) fails rather than silently picking a width. The one asymmetry: an
//! `INSERT` into a column declared as a bare `pgtoken.tokens` stores the unresolved value and
//! fails on the way back out, not on the way in — see
//! `a_typmodless_column_cannot_be_read_back`.
//!
//! `RECEIVE` is the exception that has to police itself. It *does* get the real typmod, but
//! nothing re-checks what it returns — `CopyFrom` calls `ReceiveFunctionCall` and stores the
//! result, with no length coercion behind it — so `tokens_recv` compares the incoming value's
//! vocabulary and codec against the column's typmod itself. See `check_against_typmod`.

use std::ffi::{CStr, CString};

use pgrx::prelude::*;

use pgtoken_core::header::Codec;
use pgtoken_core::value;

use crate::registry::{bail, rank_table, table_path};
use crate::typmod::{codec_for, unpack};
use crate::vocabulary::{name_for_id, vocab_size_for};

/// `vocabulary_id` of a value whose typmod has not been applied yet. The header format already
/// reserves 0 for "no vocabulary", and `create_vocabulary` never assigns it, so an unresolved
/// value cannot be mistaken for a real one.
const UNRESOLVED: u16 = 0;

/// Parse `{1,2,3}` into token IDs.
///
/// Rejects anything it does not fully understand rather than salvaging a prefix: a value that
/// silently loses tokens is worse than a failed INSERT.
pub fn parse_ids(s: &str) -> Vec<u32> {
    let t = s.trim();
    let inner = t
        .strip_prefix('{')
        .unwrap_or_else(|| bail("expected '{' at the start of a token id list"))
        .strip_suffix('}')
        .unwrap_or_else(|| bail("expected '}' at the end of a token id list"));

    if inner.trim().is_empty() {
        return Vec::new();
    }

    inner
        .split(',')
        .map(|field| {
            let f = field.trim();
            if f.is_empty() {
                bail("empty token id between commas");
            }
            match f.parse::<i64>() {
                Ok(v) if v < 0 => bail(format!("token id {v} is negative")),
                Ok(v) if v > u32::MAX as i64 => bail(format!("token id {v} is too large")),
                Ok(v) => v as u32,
                Err(_) => bail(format!("{f:?} is not a token id")),
            }
        })
        .collect()
}

/// Render token IDs as `{1,2,3}`, with no whitespace.
pub fn render_ids(ids: &[u32]) -> String {
    // Roughly seven characters per id: six digits and a separator.
    let mut s = String::with_capacity(2 + ids.len() * 7);
    s.push('{');
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&id.to_string());
    }
    s.push('}');
    s
}

/// A vocabulary's name, falling back to its raw id when the catalog has no row for it — which
/// happens when a value outlives the vocabulary it names, and while formatting an error inside an
/// already-aborted transaction.
fn vocabulary_label(id: u16) -> String {
    name_for_id(id).unwrap_or_else(|| id.to_string())
}

/// Asked to apply a typmod that names no vocabulary. Only reachable by calling
/// `tokens_typmod_apply_impl` or `encode_for` by hand: the planner skips the length coercion
/// entirely for a negative typmod, and `tokens_in` emits an unresolved value rather than failing.
///
/// A separate `-> !` function because `ereport!(ERROR, ...)` expands to two statements and so
/// cannot sit in expression position.
fn no_vocabulary_typmod() -> ! {
    ereport!(
        ERROR,
        PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
        "pgtoken.tokens requires a vocabulary",
        "A typmod is the only thing that supplies a vocabulary, and so a storage width. \
         There is no default width to fall back to."
    );
}

/// Asked to read a value whose typmod was never applied.
///
/// The one error here a user can hit in bulk: a million rows can go into a column declared as a
/// bare `pgtoken.tokens` before anything reads one back, because nothing in the write path can
/// tell that column from a value about to be length-coerced. So this names the symptom, the
/// cause and the fix, rather than only the symptom.
fn unresolved_value() -> ! {
    pgrx::pg_sys::panic::ErrorReport::new(
        PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
        "cannot read a pgtoken.tokens value that has no vocabulary",
        function_name!(),
    )
    .set_detail(
        "The value was written through a bare pgtoken.tokens, which declares no vocabulary and \
         therefore no storage width, so it was stored unresolved.",
    )
    .set_hint(
        "Declare the column as tokens.<name> or pgtoken.tokens('<name>'), then re-insert the rows.",
    )
    .report(PgLogLevel::ERROR);
    unreachable!()
}

/// Encode token IDs with no vocabulary and therefore no width of their own.
///
/// The only caller is `tokens_in` on a value PostgreSQL handed it with typmod `-1`; see the
/// module docs. `raw24` is not chosen by looking at the ids — it is the widest packing the format
/// has, used unconditionally so that any id the column may later accept fits — and the recorded
/// vocabulary is [`UNRESOLVED`], which makes the value unreadable until a typmod is applied.
fn encode_unresolved(ids: &[u32]) -> Vec<u8> {
    value::encode(ids, Codec::Raw24, UNRESOLVED, None).unwrap_or_else(|e| bail(e))
}

/// Raise if `v` never had a typmod applied. Every path that hands a value back to a caller goes
/// through this, so an unresolved value can be written but never read.
fn require_vocabulary(v: &[u8]) {
    let (h, _) = value::describe(v).unwrap_or_else(|e| bail(e));
    if h.vocabulary_id == UNRESOLVED {
        unresolved_value();
    }
}

/// Check a value arriving from outside against the column's typmod.
///
/// `RECEIVE` is the one write path with nothing behind it: `CopyFrom` calls
/// `ReceiveFunctionCall` and stores whatever comes back, with no length coercion to re-encode or
/// re-check it. So this is all that stands between a binary client and a column full of values
/// from the wrong vocabulary — which would decode under their own header rather than the
/// column's, i.e. silently mean different text.
fn check_against_typmod(v: &[u8], typmod: i32) {
    let Some(target) = unpack(typmod) else {
        // No vocabulary to check against. A bare `pgtoken.tokens` column accepts an unresolved
        // value here for the same reason `tokens_in` produces one; reading it back still fails.
        return;
    };
    let (h, _) = value::describe(v).unwrap_or_else(|e| bail(e));
    if h.vocabulary_id != target.id {
        bail(format!(
            "binary value belongs to vocabulary {}, but the column is {}",
            vocabulary_label(h.vocabulary_id),
            vocabulary_label(target.id)
        ));
    }
    // Right vocabulary, wrong packing is equally wrong: a column's stride has to be uniform for
    // the header's width to mean anything.
    let want = codec_for(target);
    if h.codec != want {
        bail(format!(
            "binary value is encoded {}, but vocabulary {} stores {}",
            h.codec.as_str(),
            vocabulary_label(target.id),
            want.as_str()
        ));
    }
}

/// Encode token IDs under a column's typmod.
///
/// The typmod is the only source of a codec and width. A missing one is an error, not a default:
/// there is no auto width, so there would be nothing to fall back to.
pub fn encode_for(ids: &[u32], typmod: i32) -> Vec<u8> {
    let Some(v) = unpack(typmod) else {
        no_vocabulary_typmod()
    };

    // The declared ID space is a cheap, exact check that these ids came from this tokenizer.
    let size = vocab_size_for(v.id)
        .unwrap_or_else(|| bail(format!("vocabulary id {} is not registered", v.id)));
    if let Some(&bad) = ids.iter().find(|&&id| id >= size) {
        let name = vocabulary_label(v.id);
        bail(format!(
            "token id {bad} is outside vocabulary {name} (size {size})"
        ));
    }

    let codec = codec_for(v);
    if codec.needs_table() {
        // Name the vocabulary rather than a path: the vocabulary is what the user typed.
        if !table_path(v.id).exists() {
            let name = vocabulary_label(v.id);
            bail(format!(
                "vocabulary {name} has no ranking; run pgtoken.train first"
            ));
        }
        let t = rank_table(v.id).unwrap_or_else(|e| bail(e));
        value::encode(ids, codec, v.id, Some(&t)).unwrap_or_else(|e| bail(e))
    } else {
        value::encode(ids, codec, v.id, None).unwrap_or_else(|e| bail(e))
    }
}

/// Decode a stored value, loading its ranking if the codec needs one.
pub fn decode_value(v: &[u8]) -> Vec<u32> {
    let (h, _) = value::describe(v).unwrap_or_else(|e| bail(e));
    if h.codec.needs_table() {
        let t = rank_table(h.vocabulary_id).unwrap_or_else(|e| bail(e));
        value::decode(v, Some(&t)).unwrap_or_else(|e| bail(e))
    } else {
        value::decode(v, None).unwrap_or_else(|e| bail(e))
    }
}

/// Validate a blob as a stored value and return it unchanged. Every path admitting outside bytes
/// goes through this — failing loudly beats storing something that decodes to plausible-looking
/// wrong IDs.
pub fn validate(v: &[u8]) -> Vec<u8> {
    require_vocabulary(v);
    v.to_vec()
}

// ── I/O functions ────────────────────────────────────────────────────────────────────
//
// Declared against `bytea` here and re-declared against `pgtoken.tokens` in the SQL below.

#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_in_impl(input: &CStr, _oid: pg_sys::Oid, typmod: i32) -> Vec<u8> {
    let s = input
        .to_str()
        .unwrap_or_else(|_| bail("token id input is not valid UTF-8"));
    let ids = parse_ids(s);
    // `COPY … FROM` in text format — how `pg_dump` restores — hands over the column's real
    // typmod, so encode straight to its width and skip the length coercion entirely. Every
    // string literal and every text cast arrives with -1 instead; see the module docs.
    match unpack(typmod) {
        Some(_) => encode_for(&ids, typmod),
        None => encode_unresolved(&ids),
    }
}

/// Length coercion: `pg_cast`'s `pgtoken.tokens -> pgtoken.tokens` function.
///
/// This, not `tokens_in`, is where a string literal acquires its vocabulary and its width. It has
/// to re-encode rather than merely check, because the typmod picks the packing.
#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_typmod_apply_impl(value: &[u8], typmod: i32, _is_explicit: bool) -> Vec<u8> {
    let Some(target) = unpack(typmod) else {
        // Unreachable through the planner: `coerce_type_typmod` returns the value untouched for a
        // negative typmod rather than calling this. Reachable by calling the function by hand.
        no_vocabulary_typmod()
    };
    let (h, _) = value::describe(value).unwrap_or_else(|e| bail(e));
    if h.vocabulary_id == target.id && h.codec == codec_for(target) {
        // Already in the column's encoding, so re-encoding could only produce the same bytes.
        return value.to_vec();
    }
    // A different vocabulary is deliberately allowed: `ALTER TABLE … ALTER COLUMN … TYPE
    // tokens('other')` is the migration path the vocabulary catalog's immutability HINT points
    // at. The ids survive; the width and the recorded vocabulary change, and `encode_for`'s
    // bound check refuses ids the new vocabulary cannot address.
    encode_for(&decode_value(value), typmod)
}

#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_out_impl(value: &[u8]) -> CString {
    require_vocabulary(value);
    let rendered = render_ids(&decode_value(value));
    CString::new(rendered).unwrap_or_else(|_| bail("rendered token ids contained a NUL"))
}

#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_send_impl(value: &[u8]) -> Vec<u8> {
    value.to_vec()
}

/// `RECEIVE`: read the rest of the binary buffer, check it against the column, store it as-is.
///
/// The typmod is load-bearing here, not decoration. Nothing re-checks a received value — there is
/// no length coercion behind `RECEIVE` — so if this hands back a value from another vocabulary or
/// at another width, that is what lands on disk.
#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_recv_impl(
    mut internal: pgrx::datum::Internal,
    _oid: pg_sys::Oid,
    typmod: i32,
) -> Vec<u8> {
    // Safety: for a type's `RECEIVE` function Postgres passes the `StringInfo` it is reading the
    // binary wire format from, so the `internal` datum is a `StringInfoData *`.
    let buf: &mut pg_sys::StringInfoData =
        unsafe { internal.get_mut() }.unwrap_or_else(|| bail("tokens_recv received a null buffer"));

    let cursor = buf.cursor;
    let total = buf.len;
    if cursor < 0 || total < 0 || cursor > total {
        bail(format!(
            "binary buffer has cursor {cursor} and length {total}"
        ));
    }
    let out = {
        // Safety: `data` is a Postgres-allocated buffer of `len` readable bytes, valid for this
        // call, and `cursor..len` is bounds-checked above.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                buf.data.add(cursor as usize).cast::<u8>(),
                (total - cursor) as usize,
            )
        };
        check_against_typmod(bytes, typmod);
        validate(bytes)
    };
    // Postgres reports "incorrect binary data format" unless the buffer is fully consumed.
    buf.cursor = total;
    out
}

/// Turn a blob back into the type, validating it. The `bytea` counterpart of `RECEIVE`, which
/// SQL cannot call directly.
#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_recv_bytes(value: &[u8]) -> Vec<u8> {
    validate(value)
}

// ── the type ─────────────────────────────────────────────────────────────────────────

extension_sql!(
    r#"
CREATE TYPE pgtoken.tokens;

CREATE FUNCTION pgtoken.tokens_in(cstring, oid, integer) RETURNS pgtoken.tokens
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE
    AS 'MODULE_PATHNAME', 'tokens_in_impl_wrapper';

CREATE FUNCTION pgtoken.tokens_out(pgtoken.tokens) RETURNS cstring
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE
    AS 'MODULE_PATHNAME', 'tokens_out_impl_wrapper';

CREATE FUNCTION pgtoken.tokens_send(pgtoken.tokens) RETURNS bytea
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE
    AS 'MODULE_PATHNAME', 'tokens_send_impl_wrapper';

CREATE FUNCTION pgtoken.tokens_recv(internal, oid, integer) RETURNS pgtoken.tokens
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE
    AS 'MODULE_PATHNAME', 'tokens_recv_impl_wrapper';

-- Named after the type, as PostgreSQL's own length-coercion functions are (`varchar`, `numeric`).
CREATE FUNCTION pgtoken.tokens(pgtoken.tokens, integer, boolean) RETURNS pgtoken.tokens
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE
    AS 'MODULE_PATHNAME', 'tokens_typmod_apply_impl_wrapper';

CREATE TYPE pgtoken.tokens (
    INPUT          = pgtoken.tokens_in,
    OUTPUT         = pgtoken.tokens_out,
    SEND           = pgtoken.tokens_send,
    RECEIVE        = pgtoken.tokens_recv,
    TYPMOD_IN      = pgtoken.tokens_typmod_in,
    TYPMOD_OUT     = pgtoken.tokens_typmod_out,
    INTERNALLENGTH = VARIABLE,
    ALIGNMENT      = int4,
    STORAGE        = external
);

-- The length coercion. PostgreSQL finds it by looking for a pg_cast row from the type to itself,
-- and only calls it because the function takes three arguments. Without this row a typmod is
-- silently ignored on every literal, cast and INSERT, and every value would stay unresolved.
CREATE CAST (pgtoken.tokens AS pgtoken.tokens)
    WITH FUNCTION pgtoken.tokens(pgtoken.tokens, integer, boolean) AS IMPLICIT;

DROP FUNCTION pgtoken.tokens_recv_bytes(bytea);
CREATE FUNCTION pgtoken.tokens_recv_bytes(bytea) RETURNS pgtoken.tokens
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE
    AS 'MODULE_PATHNAME', 'tokens_recv_bytes_wrapper';
"#,
    name = "tokens_type",
    // Item order in the generated SQL is not stable, so everything this block references has to
    // be named here. The cross-module entries are written `typmod::...`, not `crate::typmod::...`:
    // pgrx matches a `requires` path by asking whether the target's real module path
    // (`pgtoken::typmod`) *ends with* the path's module part, and `pgtoken::typmod` does not end
    // with `crate::typmod`.
    requires = [
        "vocabulary_catalog",
        tokens_in_impl,
        tokens_out_impl,
        tokens_send_impl,
        tokens_recv_impl,
        tokens_recv_bytes,
        tokens_typmod_apply_impl,
        typmod::tokens_typmod_in,
        typmod::tokens_typmod_out
    ],
);

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    fn vocab(name: &str, size: i32) {
        Spi::run(&format!(
            "SELECT pgtoken.create_vocabulary('{name}', {size})"
        ))
        .expect("create_vocabulary");
    }

    /// A binary-coercible `pgtoken.tokens -> bytea` cast, for the tests that need a value's
    /// actual datum bytes.
    ///
    /// `::bytea` does not exist yet — Task 7 adds it — and PostgreSQL will not I/O-coerce to
    /// `bytea`, because I/O coercion is only offered when one side is a string-category type and
    /// `bytea` is not one. `WITHOUT FUNCTION` is deliberate and is what makes these tests
    /// meaningful: a binary-coercible cast is a pure relabel, so `v::bytea` hands back the stored
    /// datum untouched instead of whatever a conversion function chose to build. Created inside
    /// the test's own transaction, which pgrx rolls back, and skipped if one already exists.
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
    fn text_input_and_output_roundtrip() {
        vocab("t_rt", 200019);
        let got = Spi::get_one::<String>("SELECT '{24912,2375}'::pgtoken.tokens('t_rt')::text")
            .expect("query failed");
        assert_eq!(got, Some("{24912,2375}".to_string()));
    }

    #[pg_test]
    fn empty_sequence_roundtrips() {
        vocab("t_empty", 300);
        let got = Spi::get_one::<String>("SELECT '{}'::pgtoken.tokens('t_empty')::text")
            .expect("query failed");
        assert_eq!(got, Some("{}".to_string()));
    }

    #[pg_test]
    fn input_tolerates_whitespace_but_output_emits_none() {
        vocab("t_ws", 300);
        let got = Spi::get_one::<String>("SELECT '{ 1, 2 , 3 }'::pgtoken.tokens('t_ws')::text")
            .expect("query failed");
        assert_eq!(got, Some("{1,2,3}".to_string()));
    }

    #[pg_test]
    fn the_vocabulary_fixes_the_width() {
        // Same ids, two declared sizes, two stored sizes. This is the whole point of
        // declaring vocab_size rather than inspecting the data.
        vocab("t_small", 256);
        vocab("t_big", 200019);
        ensure_bytea_cast();
        let (small, big) = Spi::get_two::<i32, i32>(
            "SELECT length('{1,2,3}'::pgtoken.tokens('t_small')::bytea), \
                    length('{1,2,3}'::pgtoken.tokens('t_big')::bytea)",
        )
        .expect("query failed");
        assert_eq!(small, Some(12 + 3), "raw8: one byte per token");
        assert_eq!(big, Some(12 + 9), "raw24: three bytes per token");
    }

    #[pg_test]
    fn column_type_shows_the_vocabulary_name() {
        vocab("t_shown", 300);
        Spi::run("CREATE TABLE shown (body pgtoken.tokens('t_shown'))").expect("create table");
        let t = Spi::get_one::<String>(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute \
             WHERE attrelid = 'shown'::regclass AND attname = 'body'",
        )
        .expect("query failed");
        assert_eq!(t, Some("pgtoken.tokens('t_shown')".to_string()));
    }

    #[pg_test]
    fn storage_is_external_so_postgres_does_not_recompress() {
        let storage = Spi::get_one::<String>(
            "SELECT typstorage::text FROM pg_type WHERE oid = 'pgtoken.tokens'::regtype",
        )
        .expect("query failed");
        assert_eq!(storage, Some("e".to_string()), "typstorage must be 'e'");
    }

    #[pg_test]
    fn binary_send_is_the_stored_bytes_verbatim() {
        // The whole performance claim: a client in binary mode gets exactly what is on disk.
        // Written through a real table so the left side is what the write path stored, and
        // compared against a binary-coercible relabel of that same datum.
        vocab("t_send", 200019);
        ensure_bytea_cast();
        Spi::run("CREATE TABLE sent (body pgtoken.tokens('t_send'))").expect("create table");
        Spi::run("INSERT INTO sent VALUES ('{24912,2375,1,70000}')").expect("insert");
        let (same, len) = Spi::get_two::<bool, i32>(
            "SELECT pgtoken.tokens_send(body) = body::bytea, length(body::bytea) FROM sent",
        )
        .expect("query failed");
        assert_eq!(same, Some(true));
        assert_eq!(len, Some(12 + 12), "raw24: four tokens at three bytes each");
    }

    #[pg_test]
    fn binary_roundtrips_through_recv_bytes() {
        vocab("t_recv", 200019);
        let got = Spi::get_one::<String>(
            "SELECT pgtoken.tokens_recv_bytes(\
               pgtoken.tokens_send('{5,9,70000}'::pgtoken.tokens('t_recv')))::text",
        )
        .expect("query failed");
        assert_eq!(got, Some("{5,9,70000}".to_string()));
    }

    #[pg_test]
    fn binary_copy_roundtrips_through_send_and_recv() {
        // The only route from SQL to a type's RECEIVE function: COPY BINARY calls SEND on the
        // way out and RECEIVE on the way in. `tokens_recv_bytes` covers the same validation but
        // not the StringInfo cursor handling, which Postgres checks itself right here.
        vocab("t_copy", 200019);
        Spi::run("CREATE TABLE copy_src (body pgtoken.tokens('t_copy'))").expect("src");
        Spi::run("INSERT INTO copy_src VALUES ('{24912,2375,1,70000}')").expect("insert");
        Spi::run("CREATE TABLE copy_dst (LIKE copy_src)").expect("dst");
        Spi::run("COPY copy_src TO '/tmp/pgtoken-copy-binary.bin' WITH (FORMAT binary)")
            .expect("copy out");
        Spi::run("COPY copy_dst FROM '/tmp/pgtoken-copy-binary.bin' WITH (FORMAT binary)")
            .expect("copy in");

        let (rendered, same) = Spi::get_two::<String, bool>(
            "SELECT d.body::text, pgtoken.tokens_send(d.body) = pgtoken.tokens_send(s.body) \
             FROM copy_dst d, copy_src s",
        )
        .expect("query failed");
        assert_eq!(rendered, Some("{24912,2375,1,70000}".to_string()));
        assert_eq!(
            same,
            Some(true),
            "recv must store what send produced, byte for byte"
        );
    }

    #[pg_test]
    fn bound_checks_a_vocabulary_id_above_the_smallint_range() {
        // The write path's `vocab_size_for` lookup is the first real caller of that SQL, whose
        // parameter cast Task 3 fixed but nothing could execute. An id above 32767 would have
        // died with "smallint out of range" before the fix.
        let id =
            Spi::get_one::<i32>("SELECT pgtoken.create_vocabulary('t_wide', 200019, id => 40000)")
                .expect("create failed")
                .expect("create returned NULL");
        assert_eq!(id, 40000, "the vocabulary must land above smallint's range");

        let got = Spi::get_one::<String>("SELECT '{1,70000}'::pgtoken.tokens('t_wide')::text")
            .expect("query failed");
        assert_eq!(got, Some("{1,70000}".to_string()));
    }

    #[pg_test]
    fn copy_text_input_uses_the_columns_typmod_directly() {
        // `COPY … FROM` in text format is the only caller that hands `tokens_in` a real typmod, so
        // it is the one path that never goes through the length coercion. `pg_dump` restores this
        // way, which makes it the path that has to produce byte-identical values.
        vocab("t_copytext", 256);
        ensure_bytea_cast();
        Spi::run("CREATE TABLE ct (body pgtoken.tokens('t_copytext'))").expect("create");
        Spi::run("COPY ct FROM PROGRAM 'echo ''{1,2,3}'''").expect("copy in");
        let (rendered, len) =
            Spi::get_two::<String, i32>("SELECT body::text, length(body::bytea) FROM ct")
                .expect("query failed");
        assert_eq!(rendered, Some("{1,2,3}".to_string()));
        assert_eq!(
            len,
            Some(12 + 3),
            "the column's typmod must reach tokens_in, giving raw8"
        );
    }

    #[pg_test(error = "cannot read a pgtoken.tokens value that has no vocabulary")]
    fn a_value_without_a_vocabulary_cannot_be_read() {
        // No typmod means no length coercion, so this value stays unresolved. There is no auto
        // width to fall back to, so reading it is an error rather than a guess.
        Spi::get_one::<String>("SELECT '{1,2,3}'::pgtoken.tokens::text").unwrap();
    }

    #[pg_test(error = "pgtoken.tokens requires a vocabulary")]
    fn applying_a_negative_typmod_by_hand_errors() {
        // The `_impl` functions stay callable from SQL (a known wart of hand-writing the type),
        // so the length coercion has to refuse a typmod that names no vocabulary rather than
        // trusting the planner never to pass one. This is the only reachable caller of that arm.
        vocab("t_bare_tm", 300);
        Spi::get_one::<Vec<u8>>(
            "SELECT pgtoken.tokens_typmod_apply_impl(\
               pgtoken.tokens_send('{1,2}'::pgtoken.tokens('t_bare_tm')), -1, false)",
        )
        .unwrap();
    }

    #[pg_test(error = "cannot read a pgtoken.tokens value that has no vocabulary")]
    fn a_typmodless_column_cannot_be_read_back() {
        // The one asymmetry the two-step coercion forces: PostgreSQL accepts the DDL, and the
        // INSERT stores an unresolved value, because nothing in the write path can tell a
        // typmod-less column from a value that is about to be coerced. The error lands on the
        // way out. Pinned here so it cannot change by accident.
        Spi::run("CREATE TABLE bare (x pgtoken.tokens)").expect("create table");
        Spi::run("INSERT INTO bare VALUES ('{1,2,3}')").expect("insert");
        Spi::get_one::<String>("SELECT x::text FROM bare").unwrap();
    }

    #[pg_test]
    fn altering_the_column_type_re_encodes_at_the_new_width() {
        // The migration path the vocabulary catalog's immutability HINT points at, and the only
        // other user of the length coercion.
        vocab("t_from", 200019);
        vocab("t_to", 256);
        ensure_bytea_cast();
        Spi::run("CREATE TABLE moved (body pgtoken.tokens('t_from'))").expect("create");
        Spi::run("INSERT INTO moved VALUES ('{1,2,3}')").expect("insert");
        let before =
            Spi::get_one::<i32>("SELECT length(body::bytea) FROM moved").expect("query failed");
        assert_eq!(before, Some(12 + 9), "raw24 to start with");

        Spi::run("ALTER TABLE moved ALTER COLUMN body TYPE pgtoken.tokens('t_to')")
            .expect("alter type");
        let (rendered, after) =
            Spi::get_two::<String, i32>("SELECT body::text, length(body::bytea) FROM moved")
                .expect("query failed");
        assert_eq!(rendered, Some("{1,2,3}".to_string()), "ids must survive");
        assert_eq!(after, Some(12 + 3), "re-encoded to raw8");
    }

    #[pg_test(error = "token id 300 is outside vocabulary t_bound (size 256)")]
    fn rejects_an_id_outside_the_vocabulary() {
        vocab("t_bound", 256);
        Spi::get_one::<String>("SELECT '{300}'::pgtoken.tokens('t_bound')::text").unwrap();
    }

    #[pg_test(error = "token id -1 is negative")]
    fn input_rejects_a_negative_id() {
        vocab("t_neg", 300);
        Spi::get_one::<String>("SELECT '{1,-1}'::pgtoken.tokens('t_neg')::text").unwrap();
    }

    #[pg_test(error = "expected '{' at the start of a token id list")]
    fn input_rejects_junk() {
        vocab("t_junk", 300);
        Spi::get_one::<String>("SELECT 'nope'::pgtoken.tokens('t_junk')::text").unwrap();
    }

    #[pg_test(error = "vocabulary t_freq has no ranking; run pgtoken.train first")]
    fn freq_needs_a_trained_ranking() {
        // Falling back to a raw codec would write a value whose header claims a ranking it was
        // not encoded with. Id 60001 is never trained by any test.
        Spi::run(
            "SELECT pgtoken.create_vocabulary('t_freq', 200019, compression => 'freq', \
                                              id => 60001)",
        )
        .expect("create_vocabulary");
        Spi::get_one::<String>("SELECT '{1,2,3}'::pgtoken.tokens('t_freq')::text").unwrap();
    }

    #[pg_test(error = "binary value belongs to vocabulary v_from, but the column is v_to")]
    fn recv_refuses_a_value_from_another_vocabulary() {
        // Nothing re-checks a received value, so RECEIVE is the last line of defence against a
        // binary client filling a column with ids that mean different text. Both vocabularies are
        // the same size, so the width matches and only the vocabulary differs.
        vocab("v_from", 200019);
        vocab("v_to", 200019);
        Spi::run("CREATE TABLE from_t (body pgtoken.tokens('v_from'))").expect("src");
        Spi::run("INSERT INTO from_t VALUES ('{1,2,70000}')").expect("insert");
        Spi::run("CREATE TABLE to_t (body pgtoken.tokens('v_to'))").expect("dst");
        Spi::run("COPY from_t TO '/tmp/pgtoken-copy-crossvocab.bin' WITH (FORMAT binary)")
            .expect("copy out");
        Spi::run("COPY to_t FROM '/tmp/pgtoken-copy-crossvocab.bin' WITH (FORMAT binary)").unwrap();
    }

    #[pg_test(error = "binary value is encoded raw16, but vocabulary w_to stores raw24")]
    fn recv_refuses_a_value_at_the_wrong_width() {
        // Right vocabulary, wrong packing. No two columns can differ this way — a vocabulary's
        // width is fixed — so the value has to be built by hand and the COPY BINARY stream framed
        // around it: an 11-byte signature, zeroed flags and header extension, then one tuple of
        // one field, then the -1 trailer. A large object is the only way to write raw bytes to a
        // file from SQL.
        let id = Spi::get_one::<i32>("SELECT pgtoken.create_vocabulary('w_to', 200019)")
            .expect("create failed")
            .expect("create returned NULL");
        Spi::run("CREATE TABLE w (body pgtoken.tokens('w_to'))").expect("create table");
        // Creating and exporting have to be separate statements, for the same reason
        // `vocabulary.rs`'s `create` helper splits: `lo_export` in the same statement runs under a
        // snapshot taken before `lo_from_bytea` inserted the object, and fails to find it.
        let loid = Spi::get_one::<String>(&format!(
            "SELECT lo_from_bytea(0, \
                 '\\x5047434f50590aff0d0a00'::bytea || int4send(0) || int4send(0) \
                 || int2send(1::smallint) || int4send(length(v)) || v \
                 || int2send((-1)::smallint))::text \
             FROM (SELECT pgtoken.encode('{{1,2,3}}', 'raw16', {id}) AS v) s"
        ))
        .expect("build the COPY BINARY stream")
        .expect("lo_from_bytea returned NULL");
        Spi::run(&format!(
            "SELECT lo_export({loid}, '/tmp/pgtoken-recv-wrongwidth.bin')"
        ))
        .expect("write the COPY BINARY stream");
        Spi::run("COPY w FROM '/tmp/pgtoken-recv-wrongwidth.bin' WITH (FORMAT binary)").unwrap();
    }

    #[pg_test(error = "bad magic byte 0x00, expected 0xA7")]
    fn recv_bytes_rejects_a_corrupt_buffer() {
        Spi::get_one::<String>(
            "SELECT pgtoken.tokens_recv_bytes('\\x000000000000000000000000'::bytea)::text",
        )
        .unwrap();
    }
}
