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
//! ## Text input
//!
//! Input that is not a strict id list — `{`, digits separated by commas, `}`, with optional
//! whitespace, or `{}` — is text: `'Hello, world!'`, and also `'{"role": "user"}'`, are text, while
//! `'{1,2,3}'` is ids. Text rides the same two steps. With no typmod it becomes an unresolved *text*
//! carrier (`raw8` with `vocabulary_id = 0`, the UTF-8 bytes as payload; the id carrier is always
//! `raw24`, so the two cannot be confused), and wherever a vocabulary is known — the length
//! coercion, `COPY` with the column's typmod, or the other operand of `=` — it is tokenized with
//! that vocabulary's server tokenizer ([`crate::registry::tokenizer`]). A vocabulary without one
//! refuses text. Output is always the id list, so `pg_dump` never needs a tokenizer to restore.
//!
//! Reading an unresolved value is an error, so `'{1,2,3}'::pgtoken.tokens` (no typmod, hence no
//! length coercion) fails rather than silently picking a width. The one asymmetry, and it is only
//! on the text path: an `INSERT` of a *literal* into a column declared as a bare `pgtoken.tokens`
//! stores the unresolved value and fails on the way back out, not on the way in — see
//! `a_typmodless_column_cannot_be_read_back`. The binary paths refuse it at write time, because
//! `RECEIVE` calls `validate`.
//!
//! `RECEIVE` is the exception that has to police itself. It *does* get the real typmod, but
//! nothing re-checks what it returns — `CopyFrom` calls `ReceiveFunctionCall` and stores the
//! result, with no length coercion behind it — so `tokens_recv` compares the incoming value's
//! vocabulary and codec against the column's typmod itself. See `check_against_typmod`.
//!
//! ## Which vocabulary a value may be stored under
//!
//! Token ids only mean something to the vocabulary that produced them, so moving a value between
//! vocabularies has to be something the user asked for rather than something an assignment does
//! quietly. The length coercion gets `is_explicit` for exactly that, and the three write paths
//! now agree:
//!
//! | path | different vocabulary |
//! | --- | --- |
//! | explicit cast (`::`, `ALTER … USING`) | allowed, re-encoded at the new width |
//! | assignment (`INSERT … SELECT`, bare `ALTER … TYPE`) | refused |
//! | `RECEIVE` (`COPY … BINARY`) | refused |
//!
//! ## What is checked
//!
//! Every write path validates every id against the vocabulary's declared `vocab_size`: text and
//! `int[]` input in `encode_for`, the binary paths in `validate`. The binary paths also check the
//! header (tag, flags, token count, payload length, vocabulary) and then store the payload
//! unchanged. The id check costs them one decode pass. Without it, a binary write could store an
//! id that `pg_dump` writes out but text input refuses on restore.

use std::ffi::{CStr, CString};

use pgrx::prelude::*;

use pgtoken_core::header::Codec;
use pgtoken_core::value;

use crate::preset;
use crate::registry::{bail, byte_map, rank_table, vocab_size};
use crate::typmod::{codec_for, pack, unpack};
use crate::vocabulary::{
    name_for_id, width_for, Vocabulary, COMPRESSION_FREQ, COMPRESSION_RAW, MAX_VOCAB_SIZE,
};

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
/// tell that column from a value about to be length-coerced. Those rows are recoverable in place,
/// though, not just by re-inserting: `ALTER TABLE t ALTER COLUMN x TYPE pgtoken.<name>` works,
/// because the unresolved-value exemption in [`tokens_typmod_apply_impl`] lets this bare
/// assignment form through, and `encode_for`'s `vocab_size` bound still applies on the way through.
/// So this names the symptom, the cause and the cheap fix, rather than only the symptom.
///
/// The exact statement shape the hint prints — no `USING`, since it is the bare assignment form
/// the exemption is for, not an explicit cast — is executed end to end, starting from an actually
/// unresolved value, by `tests::a_typmodless_columns_rows_repair_in_place_via_alter_table`. Change
/// the hint and change that test together.
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
        "Repair in place: ALTER TABLE t ALTER COLUMN x TYPE pgtoken.<name>. If the stored ids \
         turn out to be wrong for that vocabulary, re-insert the rows instead.",
    )
    .report(PgLogLevel::ERROR);
    unreachable!("unresolved_value must not return")
}

/// Asked to store a value under a vocabulary other than the one that produced its ids, by an
/// assignment rather than an explicit cast. See [`tokens_typmod_apply_impl`].
///
/// A user who reaches this from the vocabulary catalog's immutability error has already followed
/// one hint to get here, so this one has to end the chain: it names the target vocabulary and the
/// statement that works, rather than only explaining the refusal. The statement shape it prints —
/// the `pgtoken.<name>` domain spelling, not the `pgtoken.tokens('<name>')` base-type one, since the
/// docs call the domain the intended surface — is executed by
/// `tests::an_explicit_cast_moves_a_value_to_another_vocabulary`, and matches the one
/// `pgtoken.vocabulary_is_immutable`'s HINT prints.
fn cross_vocabulary_assignment(from: u16, to: u16) -> ! {
    let to = vocabulary_label(to);
    pgrx::pg_sys::panic::ErrorReport::new(
        PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
        format!(
            "cannot store {} token ids in a tokens('{to}') column",
            vocabulary_label(from)
        ),
        function_name!(),
    )
    .set_detail(
        "An assignment will not reinterpret token ids under a different vocabulary. An explicit \
         cast will, and says so at the call site.",
    )
    .set_hint(format!(
        "token ids are only meaningful to the vocabulary that produced them; to reinterpret them \
         anyway, cast explicitly: ALTER TABLE t ALTER COLUMN c TYPE pgtoken.{to} \
         USING c::pgtoken.{to}"
    ))
    .report(PgLogLevel::ERROR);
    unreachable!("cross_vocabulary_assignment must not return")
}

/// Encode token IDs with no vocabulary and therefore no width of their own.
///
/// The only caller is `tokens_in` on a value PostgreSQL handed it with typmod `-1`; see the
/// module docs. `raw24` is not chosen by looking at the ids — it is the widest packing the format
/// has, used unconditionally so that any id the column may later accept fits — and the recorded
/// vocabulary is [`UNRESOLVED`], which makes the value unreadable until a typmod is applied.
fn encode_unresolved(ids: &[u32]) -> Vec<u8> {
    // Bound the ids against the *format*, not against the carrier codec. Letting `raw24` refuse
    // them would surface "does not fit codec raw24" from a codec the user never chose and cannot
    // see, in the one place the project promises a width is never derived from the data.
    let max_id = (MAX_VOCAB_SIZE - 1) as u32;
    if let Some(&bad) = ids.iter().find(|&&id| id > max_id) {
        bail(format!(
            "token id {bad} exceeds the {max_id} this format can address"
        ));
    }
    value::encode(ids, Codec::Raw24, UNRESOLVED, None).unwrap_or_else(|e| bail(e))
}

/// Raise if `v` never had a typmod applied.
///
/// The four value-returning paths — text output, binary output (`SEND`) and both receive paths
/// (`RECEIVE`, `tokens_recv_bytes`) — call this, directly or via [`validate`], so an unresolved
/// value can be written but never read back through them. The `int[]` and `bytea` output casts in
/// `casts.rs`, and `pgtoken.text` below, call it too, for the same reason. `pgtoken.token_count`
/// and `pgtoken.describe` deliberately do not: they are diagnostics, and refusing to describe an
/// unresolved value would remove the only way to see that it *is* unresolved.
///
/// Reads the header and nothing else, which is what makes it affordable on `SEND`.
pub fn require_vocabulary(v: &[u8]) {
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
        // Nothing to check against. This is not a hole: `tokens_recv` also calls `validate`, which
        // refuses an unresolved value outright, so a binary write into a bare `pgtoken.tokens`
        // column fails here and now — unlike the text path, which stores the value and fails on
        // the way back out.
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
    // `freq_lz4` is how a `freq` column stores values that lz4 shrinks, so it counts as `freq`.
    let want = codec_for(target);
    if h.codec.family() != want {
        bail(format!(
            "binary value is encoded {}, but vocabulary {} stores {}",
            h.codec.as_str(),
            vocabulary_label(target.id),
            want.as_str()
        ));
    }
}

/// The unresolved carrier for text: `raw8`, no vocabulary, the UTF-8 bytes as payload.
const TEXT_CARRIER: Codec = Codec::Raw8;

/// Whether input is a strict token id list: `{`, digits separated by commas, `}`, with optional
/// whitespace, or `{}`. Anything else is text. Deliberately narrow, so that text which happens to
/// start with a brace (`{"role": "user"}`, `{1,-1}`) is tokenized rather than rejected.
pub fn is_id_list(s: &str) -> bool {
    let Some(inner) = s.trim().strip_prefix('{').and_then(|t| t.strip_suffix('}')) else {
        return false;
    };
    inner.trim().is_empty()
        || inner.split(',').all(|f| {
            let f = f.trim();
            !f.is_empty() && f.bytes().all(|b| b.is_ascii_digit())
        })
}

fn encode_unresolved_text(text: &str) -> Vec<u8> {
    let n = u32::try_from(text.len())
        .unwrap_or_else(|_| bail("text input is longer than this format can hold"));
    let mut out = Vec::with_capacity(pgtoken_core::header::MAX_HEADER_LEN + text.len());
    pgtoken_core::header::Header::new(TEXT_CARRIER, UNRESOLVED, n).write_to(&mut out);
    out.extend_from_slice(text.as_bytes());
    out
}

/// Tokenize text with a vocabulary's server tokenizer, or explain that it has none.
pub fn tokenize_for(text: &str, vocabulary_id: u16) -> Vec<u32> {
    match crate::registry::tokenizer(vocabulary_id) {
        Ok(Some(t)) => t.tokenize(text).unwrap_or_else(|e| bail(e)),
        Ok(None) => {
            let name = vocabulary_label(vocabulary_id);
            pgrx::pg_sys::panic::ErrorReport::new(
                PgSqlErrorCode::ERRCODE_FEATURE_NOT_SUPPORTED,
                format!("vocabulary \"{name}\" has no server tokenizer; send token ids"),
                function_name!(),
            )
            .set_detail(
                "Input that is not a token id list such as '{1,2,3}' is text, and only a \
                 vocabulary with a server tokenizer can store text. Tokenize on the client and \
                 send the ids, or use a vocabulary built on the o200k preset.",
            )
            .report(PgLogLevel::ERROR);
            unreachable!("ErrorReport at ERROR does not return")
        }
        Err(e) => bail(e),
    }
}

/// Text or an id list, as `tokens_in` and the `text -> tokens` cast receive it, to a value:
/// encoded under `typmod` when there is one, otherwise an unresolved carrier.
pub fn value_from_input(s: &str, typmod: i32) -> Vec<u8> {
    match (is_id_list(s), unpack(typmod)) {
        (true, Some(_)) => encode_for(&parse_ids(s), typmod),
        (true, None) => encode_unresolved(&parse_ids(s)),
        (false, Some(v)) => encode_for(&tokenize_for(s, v.id), typmod),
        (false, None) => encode_unresolved_text(s),
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

    // A cheap, exact check that these ids came from this tokenizer. Text and `int[]` input arrive
    // here. The binary paths make the same check in `validate`, so every stored id is one a
    // `pg_dump` restore will accept back as text.
    //
    // A missing catalog row means a pg_dump restore is still running: its schema section created
    // the column's type from the long typmod form, and tables in a schema that sorts before
    // `pgtoken` load their rows before `pgtoken.vocabulary`'s (rows are never deleted otherwise;
    // see `vocabulary_is_immutable`). Those values passed the real check when first written, so
    // fall back to what the typmod's width can address rather than refuse the restore. A preset
    // knows its size without the catalog. `vocab_size` is cached, so a warm encode runs no SPI.
    let size = vocab_size(v.id).unwrap_or(1u32 << (8 * u32::from(v.width.min(3))));
    if let Some(&bad) = ids.iter().find(|&&id| id >= size) {
        let name = vocabulary_label(v.id);
        bail(format!(
            "token id {bad} is outside vocabulary {name} (size {size})"
        ));
    }

    let codec = codec_for(v);
    if codec.needs_table() {
        // `rank_table` answers from the per-backend cache when warm, and reports absence as
        // `Ok(None)`. Name the vocabulary in the error: that is what the user typed.
        let t = rank_table(v.id)
            .unwrap_or_else(|e| bail(e))
            .unwrap_or_else(|| missing_ranking(v.id));
        value::encode(ids, codec, v.id, Some(&t)).unwrap_or_else(|e| bail(e))
    } else {
        value::encode(ids, codec, v.id, None).unwrap_or_else(|e| bail(e))
    }
}

/// The one phrasing for "this vocabulary needs a ranking and has none", shared by the encode and
/// decode paths so they cannot drift.
fn missing_ranking(vocabulary_id: u16) -> ! {
    let name = vocabulary_label(vocabulary_id);
    bail(format!(
        "vocabulary {name} has no ranking; run pgtoken.train first"
    ))
}

/// Decode a stored value, loading its ranking if the codec needs one.
pub fn decode_value(v: &[u8]) -> Vec<u32> {
    let (h, _) = value::describe(v).unwrap_or_else(|e| bail(e));
    if h.codec.needs_table() {
        let t = rank_table(h.vocabulary_id)
            .unwrap_or_else(|e| bail(e))
            .unwrap_or_else(|| missing_ranking(h.vocabulary_id));
        value::decode(v, Some(&t)).unwrap_or_else(|e| bail(e))
    } else {
        value::decode(v, None).unwrap_or_else(|e| bail(e))
    }
}

/// Whether a value is an unresolved carrier: one whose typmod was never applied. Reads the two
/// vocabulary id bytes (offsets 2-3, after the tag and flags), so the comparison operators can
/// afford it on every row.
fn is_unresolved(v: &[u8]) -> bool {
    v.len() >= pgtoken_core::header::MIN_LEN && u16::from_le_bytes([v[2], v[3]]) == UNRESOLVED
}

/// Give an unresolved carrier the vocabulary and width a typmod names: the one step that turns a
/// string literal into a storable value. Ids are bound-checked exactly as on `INSERT`; text is
/// tokenized with the vocabulary's server tokenizer.
pub fn resolve(unresolved: &[u8], typmod: i32) -> Vec<u8> {
    let Some(v) = unpack(typmod) else {
        no_vocabulary_typmod()
    };
    let (h, payload) = pgtoken_core::header::Header::parse(unresolved).unwrap_or_else(|e| bail(e));
    if h.codec == TEXT_CARRIER {
        let text = std::str::from_utf8(payload)
            .unwrap_or_else(|_| bail("unresolved text value is not valid UTF-8"));
        return encode_for(&tokenize_for(text, v.id), typmod);
    }
    encode_for(&decode_value(unresolved), typmod)
}

/// The typmod a stored value was written under, rebuilt from its header. The comparison
/// operators see two values and no column, so this is how a literal compared against a column
/// learns the column's vocabulary and codec.
fn typmod_of(h: &pgtoken_core::header::Header) -> i32 {
    let id = h.vocabulary_id;
    let (compression, width) = match h.codec {
        Codec::Raw8 => (COMPRESSION_RAW, 1),
        Codec::Raw16 => (COMPRESSION_RAW, 2),
        Codec::Raw24 => (COMPRESSION_RAW, 3),
        // Freq ignores the width, except as the bound when the catalog row is missing.
        Codec::Freq | Codec::FreqLz4 => (
            COMPRESSION_FREQ,
            vocab_size(id).map_or(3, |s| width_for(s as i32) as u8),
        ),
    };
    pack(Vocabulary {
        id,
        compression,
        width,
        preset: preset::for_id(id).map_or(0, |p| p.code),
    })
}

/// The last resolution a comparison operator made, kept in its `fn_extra` so that
/// `WHERE body = '{...}'` resolves the constant once per query rather than once per row.
#[derive(Default)]
struct ResolvedConstant {
    unresolved: Vec<u8>,
    /// The resolved side's vocabulary id and codec, which together pick the encoding.
    target: (u16, u8),
    resolved: Vec<u8>,
}

/// Compare two values, resolving an unresolved one against the other's vocabulary first.
///
/// Two resolved values are equal exactly when their bytes are (see below). A literal in
/// `WHERE body = '{13225,11,2375,0}'` reaches the operator unresolved, because PostgreSQL coerces
/// it to the operator's argument type, which carries no typmod. Comparing its carrier bytes would
/// never match, and the query would silently return no rows. So it takes the other operand's
/// vocabulary from that operand's header and is encoded the way a column of that vocabulary
/// would store it. Usually the planner has already done this (`tokens_cmp_support`); this path
/// covers what it cannot see, such as a parameter in a generic plan.
fn tokens_equal(a: &[u8], b: &[u8], fcinfo: pg_sys::FunctionCallInfo) -> bool {
    let (unresolved, resolved) = match (is_unresolved(a), is_unresolved(b)) {
        (true, true) => return a == b,
        (false, false) => return same(a, b),
        (true, false) => (a, b),
        (false, true) => (b, a),
    };
    let (h, _) = value::describe(resolved).unwrap_or_else(|e| bail(e));
    let target = (h.vocabulary_id, h.codec.family() as u8);
    // Safety: `fcinfo` is the live call's. `flinfo` is null only under `DirectFunctionCall`,
    // where there is no call site to cache on.
    let has_flinfo = !fcinfo.is_null() && unsafe { !(*fcinfo).flinfo.is_null() };
    if !has_flinfo {
        return same(&resolve(unresolved, typmod_of(&h)), resolved);
    }
    let mut cache = unsafe { pgrx::pg_func_extra(fcinfo, ResolvedConstant::default) };
    if cache.target != target || cache.unresolved != unresolved {
        let bytes = resolve(unresolved, typmod_of(&h));
        cache.unresolved = unresolved.to_vec();
        cache.target = target;
        cache.resolved = bytes;
    }
    same(&cache.resolved, resolved)
}

/// Whether two resolved values hold the same tokens: equal bytes, or equal once any `freq_lz4`
/// side is decompressed (see `pgtoken_core::value::canonical`).
fn same(a: &[u8], b: &[u8]) -> bool {
    value::same(a, b).unwrap_or_else(|e| bail(e))
}

/// Check a blob from a binary write path (`RECEIVE`, `tokens_recv_bytes`) and return it
/// unchanged.
///
/// What this guarantees: the tag (version and codec), flags and token count are sound, the raw
/// codecs' payload length matches the token count exactly, the vocabulary id is nonzero, i.e.
/// not [`UNRESOLVED`], and every token id is below the vocabulary's `vocab_size`.
///
/// The id check costs one decode pass, and it is what keeps a dump restorable. `pg_dump` writes
/// ids as text, and text input refuses an id outside the vocabulary. So a binary write that
/// stored `{60000}` in a 32000-token vocabulary would make that table's whole `COPY` fail on
/// restore. Checking here moves the error to the write, where the client can fix it.
///
/// When the vocabulary has no catalog row, the ids are not checked and the codec never runs.
/// That happens while a `pg_dump` restore is still loading (see [`encode_for`]), and for a
/// hand-framed value naming an unregistered id. Such a value can be stored in a bare column. Any
/// typmod'd column catches it, through the length coercion or [`check_against_typmod`].
pub fn validate(v: &[u8]) -> Vec<u8> {
    require_vocabulary(v);
    let (h, _) = value::describe(v).unwrap_or_else(|e| bail(e));
    if let Some(size) = vocab_size(h.vocabulary_id) {
        if let Some(&bad) = decode_value(v).iter().find(|&&id| id >= size) {
            bail(format!(
                "token id {bad} is outside vocabulary {} (size {size})",
                vocabulary_label(h.vocabulary_id)
            ));
        }
    }
    v.to_vec()
}

// ── I/O functions ────────────────────────────────────────────────────────────────────
//
// Declared against `bytea` here and re-declared against `pgtoken.tokens` in the SQL below.

#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_in_impl(input: &CStr, _oid: pg_sys::Oid, typmod: i32) -> Vec<u8> {
    let s = input
        .to_str()
        .unwrap_or_else(|_| bail("token input is not valid UTF-8"));
    // `COPY … FROM` in text format — how `pg_dump` restores — hands over the column's real
    // typmod, so encode straight to its width and skip the length coercion entirely. Every
    // string literal arrives with -1 instead; see the module docs.
    value_from_input(s, typmod)
}

/// Length coercion: `pg_cast`'s `pgtoken.tokens -> pgtoken.tokens` function.
///
/// This, not `tokens_in`, is where a string literal acquires its vocabulary and its width. It has
/// to re-encode rather than merely check, because the typmod picks the packing.
///
/// `is_explicit` is what PostgreSQL sets to `ccontext == COERCION_EXPLICIT` in
/// `build_coercion_expression`, and it is the only thing distinguishing "the user asked for this
/// reinterpretation" from "the user assigned one column to another and did not notice the
/// vocabularies differ". Changing vocabulary is therefore allowed only on an explicit cast:
///
/// ```sql
/// ALTER TABLE t ALTER COLUMN body TYPE pgtoken.tokens('o200k')             -- refused
/// ALTER TABLE t ALTER COLUMN body TYPE pgtoken.tokens('o200k')
///     USING body::pgtoken.tokens('o200k')                                  -- allowed
/// INSERT INTO o200k_t SELECT body FROM cl100k_t                            -- refused
/// INSERT INTO o200k_t SELECT body::pgtoken.tokens('o200k') FROM cl100k_t   -- allowed
/// ```
///
/// Without the split, `INSERT … SELECT` would silently re-stamp every value's `vocabulary_id`
/// while the byte-identical value arriving over `COPY … (FORMAT binary)` was refused by
/// [`check_against_typmod`]. Two write paths disagreeing about the project's central hazard is
/// worse than either policy alone.
#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_typmod_apply_impl(value: &[u8], typmod: i32, is_explicit: bool) -> Vec<u8> {
    let Some(target) = unpack(typmod) else {
        // Unreachable through the planner: `coerce_type_typmod` returns the value untouched for a
        // negative typmod rather than calling this. Reachable by calling the function by hand.
        no_vocabulary_typmod()
    };
    let (h, _) = value::describe(value).unwrap_or_else(|e| bail(e));
    if h.vocabulary_id == target.id && h.codec.family() == codec_for(target) {
        // Already in the column's encoding, so re-encoding could only produce the same bytes.
        return value.to_vec();
    }
    // [`UNRESOLVED`] is exempt: that is every string literal on its way to acquiring a vocabulary
    // for the first time, and an INSERT of a literal is an assignment. It is not a
    // reinterpretation, because the value never had a vocabulary to be reinterpreted from.
    if h.vocabulary_id == UNRESOLVED {
        return resolve(value, typmod);
    }
    if h.vocabulary_id != target.id && !is_explicit {
        cross_vocabulary_assignment(h.vocabulary_id, target.id);
    }
    // Same vocabulary, different packing stays permissive in both contexts: a lossless recode
    // within one ID space reinterprets nothing.
    encode_for(&decode_value(value), typmod)
}

#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_out_impl(value: &[u8]) -> CString {
    require_vocabulary(value);
    let rendered = render_ids(&decode_value(value));
    CString::new(rendered).unwrap_or_else(|_| bail("rendered token ids contained a NUL"))
}

/// `SEND`: hand the stored bytes over unchanged.
///
/// The header check is not a conversion — nothing is decoded, nothing is re-encoded, and the
/// bytes still leave verbatim, which is the whole claim. It is here because text output refuses an
/// unresolved value and binary output must refuse it too: one read path accepting what the others
/// refuse would leave a bare-column table unreadable in `psql` but readable over the wire, and the
/// error would stop being the thing that makes someone fix their column declaration.
///
/// Deliberately `require_vocabulary` and not `decode_value`: this parses a few header bytes, where a decode
/// would put the whole payload through the codec on every binary read.
#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_send_impl(value: &[u8]) -> Vec<u8> {
    require_vocabulary(value);
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

/// Detokenize a stored value, for serving an interface that knows nothing about token IDs.
///
/// `IMMUTABLE` on the same grounds as the read paths: it depends only on the value's bytes and a
/// write-once mapping. That is what lets it back a full-text index.
///
/// `decode_value` does not itself refuse an unresolved value — the raw codecs decode a
/// `vocabulary_id` of 0 the same as any other, and only `freq`'s header check cares about it — so
/// this calls [`require_vocabulary`] first, the same guard `tokens_out_impl` and
/// `tokens_send_impl` use, rather than let an unresolved value fall through to a confusing
/// "vocabulary 0 has no mapping" instead of the standard unresolved-value error. Pinned by
/// `tests::text_refuses_a_value_without_a_vocabulary` — delete the call above and only that test
/// fails, with the nonsense message this guard exists to prevent.
///
/// It is also the `tokens → text` cast (see `casts.rs`), so `body::text` is the same call.
#[pg_extern(immutable, parallel_safe, strict)]
fn text_impl(value: &[u8]) -> String {
    require_vocabulary(value);
    let ids = decode_value(value);
    let (h, _) = value::describe(value).unwrap_or_else(|e| bail(e));
    // `byte_map` answers from the per-backend cache when warm, so a large index build looks the
    // mapping up once, not once per row.
    let map = match byte_map(h.vocabulary_id) {
        Ok(Some(m)) => m,
        Ok(None) => {
            let name = name_for_id(h.vocabulary_id).unwrap_or_else(|| h.vocabulary_id.to_string());
            bail(format!(
                "vocabulary {name} has no mapping; run pgtoken.load_mapping first"
            ))
        }
        Err(e) => bail(e),
    };
    pgtoken_core::detok::to_text(&ids, &map).unwrap_or_else(|e| bail(e))
}

/// `pgtoken.tokenize(text, vocabulary)`: the ids a vocabulary's server tokenizer gives `text`.
/// `STABLE`, not `IMMUTABLE`, because it looks the vocabulary up by name.
#[pg_extern(stable, parallel_safe, strict)]
fn tokenize(text: &str, vocabulary: &str) -> Vec<i32> {
    let v = crate::vocabulary::lookup_by_name(vocabulary)
        .unwrap_or_else(|| bail(format!("vocabulary {vocabulary:?} does not exist")));
    tokenize_for(text, v.id)
        .into_iter()
        .map(|id| id as i32)
        .collect()
}

// ── equality and hashing ──────────────────────────────────────────────────────────────
//
// Two stored values are equal exactly when their canonical bytes are equal. Every codec but
// `freq_lz4` has one byte form per token list (pgtoken_core::value::tests::encoding_is_canonical),
// and the header carries the vocabulary id, so values from different vocabularies never compare
// equal. `freq_lz4` does not: lz4 can write the same bytes as many different blocks, and a newer
// lz4 library may pick another. So its canonical form is the plain `freq` value it decompresses
// to (pgtoken_core::value::canonical). Equality is a memcmp, decompressing only when the bytes
// differ and a side is `freq_lz4`; the hash opclass hashes the canonical bytes, so GROUP BY,
// DISTINCT and hash joins agree with it. There is no
// ORDER BY on purpose: the freq codec remaps ids to frequency rank before packing, so byte order
// says nothing about the ids or the text. Sort on pgtoken.text(body) when you want alphabetical.

#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_eq_impl(a: &[u8], b: &[u8], fcinfo: pg_sys::FunctionCallInfo) -> bool {
    tokens_equal(a, b, fcinfo)
}

#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_ne_impl(a: &[u8], b: &[u8], fcinfo: pg_sys::FunctionCallInfo) -> bool {
    !tokens_equal(a, b, fcinfo)
}

/// Hash the canonical bytes so equal values land in the same bucket. FNV-1a, 32-bit:
/// deterministic across backends, self-contained, and consistent with `tokens_eq` by
/// construction, since that compares canonical bytes too. Only a `freq_lz4` value is rewritten
/// first; every other value hashes its stored bytes, as before.
///
/// An unresolved value has no hash that agrees with its resolved form, because a hash sees one
/// value and so has no other operand to borrow a vocabulary from. Hashing one is therefore an
/// error rather than a silent miss. The planner resolves `body = '...'` before execution, so
/// this is only reachable through a long `IN` list of literals, which PostgreSQL 14+ hashes.
#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_hash_impl(value: &[u8]) -> i32 {
    mix(fnv1a(FNV_BASIS, &hash_input(value))) as i32
}

/// The hash operator class's `FUNCTION 2`, which `PARTITION BY HASH` needs: a 64-bit hash with a
/// seed. PostgreSQL requires that with seed 0 its low 32 bits equal [`tokens_hash_impl`]'s
/// result, so the low half is that same hash with the seed folded into its starting value, and
/// the high half is a second pass from a different starting value.
#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_hash_extended_impl(value: &[u8], seed: i64) -> i64 {
    let bytes = hash_input(value);
    let seed = seed as u64;
    let lo = mix(fnv1a(FNV_BASIS ^ seed as u32, &bytes));
    let hi = mix(fnv1a(
        0x050c_5d1f ^ (seed >> 32) as u32 ^ (seed as u32).rotate_left(16),
        &bytes,
    ));
    (((hi as u64) << 32) | lo as u64) as i64
}

const FNV_BASIS: u32 = 0x811c_9dc5;

/// murmur3's `fmix32`. FNV-1a alone has weak low bits (its lowest bit is just the XOR of the
/// input bytes' lowest bits), and PostgreSQL picks hash-index buckets, hash-join buckets and hash
/// partitions from the low bits. This spreads every input bit over every output bit.
fn mix(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

fn fnv1a(basis: u32, bytes: &[u8]) -> u32 {
    let mut h = basis;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// What both hash functions hash: the canonical form, so values `=` calls equal hash equal.
fn hash_input(value: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    if is_unresolved(value) {
        pgrx::pg_sys::panic::ErrorReport::new(
            PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
            "cannot hash a pgtoken.tokens value that has no vocabulary",
            function_name!(),
        )
        .set_hint(
            "Cast the literal to the column's type, e.g. body IN ('{1,2}'::pgtoken.o200k, ...).",
        )
        .report(PgLogLevel::ERROR);
    }
    value::canonical(value).unwrap_or_else(|e| bail(e))
}

/// Planner support for `=` and `<>`: resolve an unresolved constant at plan time.
///
/// `WHERE body = '{13225,11,2375,0}'` arrives as `tokens_eq(body, <unresolved constant>)`. If the
/// other operand is a column whose type names a vocabulary (a `pgtoken.<name>` domain or a
/// `pgtoken.tokens('<name>')` column), the constant is replaced by its resolved form. That way the
/// constant is resolved once per plan, `EXPLAIN` can print it (an unresolved value has no output
/// form), and the clause stays an operator, so a hash index can serve it. Anything else is left
/// alone for [`tokens_equal`] to resolve at run time.
///
/// Hand-written rather than `#[pg_extern]`, because its argument and result are planner nodes.
#[no_mangle]
pub extern "C" fn pg_finfo_tokens_cmp_support() -> &'static pg_sys::Pg_finfo_record {
    const V1: pg_sys::Pg_finfo_record = pg_sys::Pg_finfo_record { api_version: 1 };
    &V1
}

/// # Safety
/// Called by PostgreSQL's planner with a `SupportRequest*` node as its one argument.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn tokens_cmp_support(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let none = pg_sys::Datum::from(0usize);
    let Some(arg) = pgrx::pg_getarg_datum(fcinfo, 0) else {
        return none;
    };
    let req = arg.cast_mut_ptr::<pg_sys::Node>();
    if req.is_null() || (*req).type_ != pg_sys::NodeTag::T_SupportRequestSimplify {
        return none;
    }
    let f = (*(req as *mut pg_sys::SupportRequestSimplify)).fcall;
    let args = pgrx::PgList::<pg_sys::Node>::from_pg((*f).args);
    if args.len() != 2 {
        return none;
    }
    let (a, b) = (args.get_ptr(0).unwrap(), args.get_ptr(1).unwrap());
    let (const_first, c, other) = if let Some(c) = unresolved_const(a) {
        (true, c, b)
    } else if let Some(c) = unresolved_const(b) {
        (false, c, a)
    } else {
        return none;
    };
    let typmod = declared_typmod(other);
    if typmod < 0 {
        return none;
    }
    let bytes = <Vec<u8> as FromDatum>::from_datum((*c).constvalue, false)
        .unwrap_or_else(|| bail("constant has no value"));
    let resolved = resolve(&bytes, typmod)
        .into_datum()
        .unwrap_or_else(|| bail("resolved constant has no value"));
    let new_const = pg_sys::makeConst(
        (*c).consttype,
        -1,
        pg_sys::InvalidOid,
        -1,
        resolved,
        false,
        false,
    )
    .cast::<pg_sys::Node>();
    let mut new_args = pgrx::PgList::<pg_sys::Node>::new();
    if const_first {
        new_args.push(new_const);
        new_args.push(other);
    } else {
        new_args.push(other);
        new_args.push(new_const);
    }

    // Rebuild the operator, not a function call, so the clause still reads `body = ...` and an
    // index or selectivity estimate can recognise it.
    let opno = operator_for((*f).funcid, (*c).consttype);
    if opno == pg_sys::InvalidOid {
        let fx = pg_sys::makeFuncExpr(
            (*f).funcid,
            (*f).funcresulttype,
            new_args.into_pg(),
            (*f).funccollid,
            (*f).inputcollid,
            pg_sys::CoercionForm::COERCE_EXPLICIT_CALL,
        );
        return pg_sys::Datum::from(fx);
    }
    let op = pg_sys::palloc0(std::mem::size_of::<pg_sys::OpExpr>()).cast::<pg_sys::OpExpr>();
    (*op).xpr.type_ = pg_sys::NodeTag::T_OpExpr;
    (*op).opno = opno;
    (*op).opfuncid = (*f).funcid;
    (*op).opresulttype = (*f).funcresulttype;
    (*op).opretset = false;
    (*op).opcollid = (*f).funccollid;
    (*op).inputcollid = (*f).inputcollid;
    (*op).args = new_args.into_pg();
    (*op).location = (*f).location;
    pg_sys::Datum::from(op)
}

/// `n` as a constant holding an unresolved value, if it is one.
unsafe fn unresolved_const(n: *mut pg_sys::Node) -> Option<*mut pg_sys::Const> {
    if (*n).type_ != pg_sys::NodeTag::T_Const {
        return None;
    }
    let c = n.cast::<pg_sys::Const>();
    if (*c).constisnull {
        return None;
    }
    let bytes = <&[u8] as FromDatum>::from_datum((*c).constvalue, false)?;
    is_unresolved(bytes).then_some(c)
}

/// The typmod an expression's declared type carries: a column's own, or its domain's. Looks
/// through the relabelling PostgreSQL adds to pass a domain value to a base-type operator.
unsafe fn declared_typmod(mut n: *mut pg_sys::Node) -> i32 {
    while (*n).type_ == pg_sys::NodeTag::T_RelabelType {
        n = (*n.cast::<pg_sys::RelabelType>()).arg.cast();
    }
    let mut typmod = pg_sys::exprTypmod(n);
    pg_sys::getBaseTypeAndTypmod(pg_sys::exprType(n), &mut typmod);
    typmod
}

/// `pg_catalog.=` or `pg_catalog.<>` on `pgtoken.tokens`, whichever `funcid` implements.
unsafe fn operator_for(funcid: pg_sys::Oid, tokens: pg_sys::Oid) -> pg_sys::Oid {
    let name = pg_sys::get_func_name(funcid);
    if name.is_null() {
        return pg_sys::InvalidOid;
    }
    let op = match CStr::from_ptr(name).to_bytes() {
        b"tokens_eq" => c"=",
        b"tokens_ne" => c"<>",
        _ => return pg_sys::InvalidOid,
    };
    let mut names = std::ptr::null_mut::<pg_sys::List>();
    for part in [c"pg_catalog", op] {
        let s = pg_sys::makeString(pg_sys::pstrdup(part.as_ptr()));
        names = pg_sys::lappend(names, s.cast());
    }
    pg_sys::OpernameGetOprid(names, tokens, tokens)
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
    -- int4, because PostgreSQL refuses anything narrower for a variable-length type ("alignment
    -- \"c\" is invalid for variable-length type"). It costs little: a value under 127 bytes is
    -- stored with a 1-byte varlena header and packed unaligned anyway. Nothing in the format
    -- relies on it either; every field is read with byte loads.
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

-- `IMMUTABLE` is the point: it depends only on the value's bytes and a write-once mapping,
-- which is what lets it back a GIN index over `to_tsvector`.
-- Also the `pgtoken.tokens -> text` cast; see casts.rs.
CREATE FUNCTION pgtoken.text(pgtoken.tokens) RETURNS text
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE
    AS 'MODULE_PATHNAME', 'text_impl_wrapper';
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
        text_impl,
        typmod::tokens_typmod_in,
        typmod::tokens_typmod_out
    ],
);

// ── equality operators and the hash operator class ─────────────────────────────────────
//
// Re-declared against `pgtoken.tokens`, like the I/O functions above. `HASHES` on `=` plus a
// default hash operator class is what lets GROUP BY, DISTINCT and hash joins work without the
// caller naming an operator. The operators live in `pg_catalog`, which every search_path includes,
// so `WHERE body = '...'` works without `pgtoken` on it: in schema `pgtoken` they were invisible
// to a default search_path and the query failed with "operator does not exist". They are still
// members of the extension and go with it on DROP EXTENSION. No btree class, so no ORDER BY:
// byte order is meaningless here.
extension_sql!(
    r#"
CREATE FUNCTION pgtoken.tokens_cmp_support(internal) RETURNS internal
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE
    AS 'MODULE_PATHNAME', 'tokens_cmp_support';

CREATE FUNCTION pgtoken.tokens_eq(pgtoken.tokens, pgtoken.tokens) RETURNS boolean
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE SUPPORT pgtoken.tokens_cmp_support
    AS 'MODULE_PATHNAME', 'tokens_eq_impl_wrapper';

CREATE FUNCTION pgtoken.tokens_ne(pgtoken.tokens, pgtoken.tokens) RETURNS boolean
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE SUPPORT pgtoken.tokens_cmp_support
    AS 'MODULE_PATHNAME', 'tokens_ne_impl_wrapper';

CREATE FUNCTION pgtoken.tokens_hash(pgtoken.tokens) RETURNS integer
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE
    AS 'MODULE_PATHNAME', 'tokens_hash_impl_wrapper';

CREATE FUNCTION pgtoken.tokens_hash_extended(pgtoken.tokens, bigint) RETURNS bigint
    LANGUAGE c IMMUTABLE STRICT PARALLEL SAFE
    AS 'MODULE_PATHNAME', 'tokens_hash_extended_impl_wrapper';

CREATE OPERATOR pg_catalog.= (
    LEFTARG = pgtoken.tokens, RIGHTARG = pgtoken.tokens,
    FUNCTION = pgtoken.tokens_eq,
    COMMUTATOR = OPERATOR(pg_catalog.=),
    NEGATOR = OPERATOR(pg_catalog.<>),
    RESTRICT = eqsel, JOIN = eqjoinsel, HASHES
);

CREATE OPERATOR pg_catalog.<> (
    LEFTARG = pgtoken.tokens, RIGHTARG = pgtoken.tokens,
    FUNCTION = pgtoken.tokens_ne,
    COMMUTATOR = OPERATOR(pg_catalog.<>),
    NEGATOR = OPERATOR(pg_catalog.=),
    RESTRICT = neqsel, JOIN = neqjoinsel
);

CREATE OPERATOR CLASS pgtoken.tokens_hash_ops DEFAULT FOR TYPE pgtoken.tokens USING hash AS
    OPERATOR 1 pg_catalog.= (pgtoken.tokens, pgtoken.tokens),
    FUNCTION 1 pgtoken.tokens_hash(pgtoken.tokens),
    FUNCTION 2 pgtoken.tokens_hash_extended(pgtoken.tokens, bigint);
"#,
    name = "tokens_equality",
    requires = [
        "tokens_type",
        tokens_eq_impl,
        tokens_ne_impl,
        tokens_hash_impl,
        tokens_hash_extended_impl,
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
    /// The extension deliberately ships no `tokens -> bytea` cast: a text-mode client gets the ids
    /// straight from `SELECT body` at the same wire cost, so a hex form would only add a codec
    /// dependency. `pgtoken.tokens_send` is what callers use for the stored bytes.
    ///
    /// This test-local cast is not that function, and must not be: `WITHOUT FUNCTION` makes it a
    /// pure relabel, so `v::bytea` hands back the stored datum untouched rather than whatever a
    /// conversion function chose to build. That is what lets the send test compare against an
    /// independent reference instead of against itself. PostgreSQL will not I/O-coerce to `bytea`,
    /// since I/O coercion needs a string-category type on one side. Created inside the test's own
    /// transaction, which pgrx rolls back.
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
    fn equality_powers_dedup_and_group_by() {
        vocab("eq", 300);
        Spi::run("SET search_path TO pgtoken, tokens, public").expect("search_path");
        Spi::run("CREATE TABLE eq_t (body pgtoken.eq)").expect("create table");
        Spi::run("INSERT INTO eq_t VALUES ('{1,2,3}'), ('{1,2,3}'), ('{4,5}')").expect("insert");

        // Equal ids compare equal, different ids compare unequal, within one vocabulary.
        assert_eq!(
            Spi::get_one::<bool>(
                "SELECT '{1,2,3}'::pgtoken.tokens('eq') = '{1,2,3}'::pgtoken.tokens('eq')"
            )
            .expect("eq query"),
            Some(true)
        );
        assert_eq!(
            Spi::get_one::<bool>(
                "SELECT '{1,2,3}'::pgtoken.tokens('eq') <> '{4,5}'::pgtoken.tokens('eq')"
            )
            .expect("ne query"),
            Some(true)
        );

        // DISTINCT collapses the two identical rows: three rows, two distinct.
        assert_eq!(
            Spi::get_one::<i64>("SELECT count(*) FROM (SELECT DISTINCT body FROM eq_t) s")
                .expect("distinct query"),
            Some(2)
        );

        // GROUP BY buckets the duplicate together: the largest group has two rows.
        assert_eq!(
            Spi::get_one::<i64>(
                "SELECT count(*) FROM eq_t GROUP BY body ORDER BY count(*) DESC LIMIT 1"
            )
            .expect("group by query"),
            Some(2)
        );
    }

    #[pg_test]
    fn text_input_and_output_roundtrip() {
        vocab("t_rt", 200019);
        let got =
            Spi::get_one::<String>("SELECT '{24912,2375}'::pgtoken.tokens('t_rt')::int[]::text")
                .expect("query failed");
        assert_eq!(got, Some("{24912,2375}".to_string()));
    }

    #[pg_test]
    fn empty_sequence_roundtrips() {
        vocab("t_empty", 300);
        let got = Spi::get_one::<String>("SELECT '{}'::pgtoken.tokens('t_empty')::int[]::text")
            .expect("query failed");
        assert_eq!(got, Some("{}".to_string()));
    }

    #[pg_test]
    fn input_tolerates_whitespace_but_output_emits_none() {
        vocab("t_ws", 300);
        let got =
            Spi::get_one::<String>("SELECT '{ 1, 2 , 3 }'::pgtoken.tokens('t_ws')::int[]::text")
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
            "SELECT length(pgtoken.tokens_send('{1,2,3}'::pgtoken.tokens('t_small'))), \
                    length(pgtoken.tokens_send('{1,2,3}'::pgtoken.tokens('t_big')))",
        )
        .expect("query failed");
        assert_eq!(small, Some(5 + 3), "raw8: one byte per token");
        assert_eq!(big, Some(5 + 9), "raw24: three bytes per token");
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
        // The long typmod form (name, id, compression, width), so pg_dump output restores.
        let t = t.expect("format_type returned NULL");
        assert!(
            t.starts_with("pgtoken.tokens('t_shown', ") && t.ends_with(", 'raw', 2)"),
            "{t}"
        );
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
        assert_eq!(len, Some(5 + 12), "raw24: four tokens at three bytes each");
    }

    #[pg_test]
    fn binary_roundtrips_through_recv_bytes() {
        vocab("t_recv", 200019);
        let got = Spi::get_one::<String>(
            "SELECT pgtoken.tokens_recv_bytes(\
               pgtoken.tokens_send('{5,9,70000}'::pgtoken.tokens('t_recv')))::int[]::text",
        )
        .expect("query failed");
        assert_eq!(got, Some("{5,9,70000}".to_string()));
    }

    #[pg_test(error = "token id 60000 is outside vocabulary t_oov (size 32000)")]
    fn binary_write_refuses_an_id_outside_the_vocabulary() {
        // 32000 tokens fit in raw16, which can still hold id 60000. pg_dump would write that id
        // as text, and text input refuses it, so the restore would fail. Refuse it here instead.
        vocab("t_oov", 32000);
        let mut v = Spi::get_one::<Vec<u8>>(
            "SELECT pgtoken.tokens_send('{5,31999}'::pgtoken.tokens('t_oov'))",
        )
        .expect("send")
        .expect("NULL");
        let n = v.len();
        v[n - 2..].copy_from_slice(&60000u16.to_le_bytes());
        let hex: String = v.iter().map(|b| format!("{b:02x}")).collect();
        Spi::get_one::<String>(&format!(
            "SELECT pgtoken.tokens_recv_bytes(decode('{hex}', 'hex'))::int[]::text"
        ))
        .unwrap();
    }

    #[pg_test]
    fn hash_values_are_pinned() {
        // Hash indexes and hash partitions store these numbers on disk. A change to the hash, the
        // seed mixing or the canonical form would silently break them after an upgrade, so any
        // such change has to show up here first. o200k is a preset, so its vocabulary id is fixed.
        Spi::run("SELECT pgtoken.create_vocabulary('o200k')").expect("create");
        let row = Spi::get_three::<i32, i64, i64>(
            "SELECT pgtoken.tokens_hash(v), pgtoken.tokens_hash_extended(v, 0), \
                    pgtoken.tokens_hash_extended(v, 42) \
             FROM (SELECT '{1,2,3}'::pgtoken.o200k AS v) s",
        )
        .expect("hash");
        let lz = Spi::get_two::<String, i32>(
            "SELECT d.codec, pgtoken.tokens_hash(v) \
             FROM (SELECT repeat('hello world ', 400)::pgtoken.o200k AS v) s, pgtoken.describe(v) d",
        )
        .expect("hash lz4");
        assert_eq!(
            row,
            (
                Some(472_099_173),
                Some(696_413_631_665_121_637),
                Some(-4_771_386_950_555_120_711)
            )
        );
        // A `freq_lz4` value hashes its decompressed form, so this also pins `canonical()`.
        assert_eq!(lz, (Some("freq_lz4".to_string()), Some(579_235_454)));
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
            "SELECT d.body::int[]::text, pgtoken.tokens_send(d.body) = pgtoken.tokens_send(s.body) \
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

        let got =
            Spi::get_one::<String>("SELECT '{1,70000}'::pgtoken.tokens('t_wide')::int[]::text")
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
        let (rendered, len) = Spi::get_two::<String, i32>(
            "SELECT body::int[]::text, length(pgtoken.tokens_send(body)) FROM ct",
        )
        .expect("query failed");
        assert_eq!(rendered, Some("{1,2,3}".to_string()));
        assert_eq!(
            len,
            Some(5 + 3),
            "the column's typmod must reach tokens_in, giving raw8"
        );
    }

    #[pg_test(error = "cannot read a pgtoken.tokens value that has no vocabulary")]
    fn a_value_without_a_vocabulary_cannot_be_read() {
        // No typmod means no length coercion, so this value stays unresolved. There is no auto
        // width to fall back to, so reading it is an error rather than a guess.
        Spi::get_one::<String>("SELECT '{1,2,3}'::pgtoken.tokens::int[]::text").unwrap();
    }

    #[pg_test]
    fn send_does_not_run_the_codec() {
        // The only test standing between the hot path and a future "simplification" that decodes
        // and re-encodes. Because encoding is canonical, such a rewrite would return identical
        // bytes and every other test here would still pass.
        //
        // The value is hand-framed so its header names `freq` against vocabulary 60002, which no
        // test registers, so `tokens_recv_bytes` cannot check its ids and skips the codec: tag 12 (version 1, codec freq), flags 00, id 60002 as u16 LE (62ea), token
        // count 01, then one payload byte. A header-only check accepts it and hands the bytes
        // back; anything that touches the codec must load a ranking that does not exist, and dies
        // with "has no ranking".
        let same = Spi::get_one::<bool>(
            "SELECT pgtoken.tokens_send(pgtoken.tokens_recv_bytes(v)) = v \
             FROM (SELECT '\\x120062ea0101'::bytea AS v) s",
        )
        .expect("query failed");
        assert_eq!(same, Some(true), "send must copy, never decode");
    }

    #[pg_test(error = "token id 16777216 exceeds the 16777215 this format can address")]
    fn input_rejects_an_id_the_format_cannot_address() {
        // Caught before the carrier encode, so the message names the format rather than leaking
        // `raw24` — a codec the user never chose and cannot see from the column definition.
        vocab("t_huge", 200019);
        Spi::get_one::<String>("SELECT '{16777216}'::pgtoken.tokens('t_huge')::int[]::text")
            .unwrap();
    }

    #[pg_test(error = "cannot read a pgtoken.tokens value that has no vocabulary")]
    fn send_refuses_a_value_without_a_vocabulary() {
        // Binary output has to refuse what text output refuses. Otherwise a bare-column table is
        // unreadable in psql but readable over the wire, and the error stops being what forces
        // someone to fix the column declaration.
        Spi::get_one::<Vec<u8>>("SELECT pgtoken.tokens_send('{1,2,3}'::pgtoken.tokens)").unwrap();
    }

    #[pg_test(error = "pgtoken.tokens requires a vocabulary")]
    fn applying_a_negative_typmod_by_hand_errors() {
        // The length coercion is an ordinary function anyone can call, so it has to refuse a
        // typmod that names no vocabulary rather than trusting the planner never to pass one.
        // This is the only reachable caller of that arm.
        vocab("t_bare_tm", 300);
        Spi::get_one::<String>(
            "SELECT pgtoken.tokens('{1,2}'::pgtoken.tokens('t_bare_tm'), -1, false)::text",
        )
        .unwrap();
    }

    #[pg_test]
    fn hash_partitioning_on_a_tokens_column_works() {
        // PARTITION BY HASH needs the opclass's FUNCTION 2. With seed 0 its low 32 bits must
        // equal the plain hash, or hash indexes and partitions would disagree.
        vocab("t_part", 300);
        Spi::run(
            "CREATE TABLE parts (body pgtoken.t_part) PARTITION BY HASH (body);
             CREATE TABLE parts_0 PARTITION OF parts FOR VALUES WITH (MODULUS 2, REMAINDER 0);
             CREATE TABLE parts_1 PARTITION OF parts FOR VALUES WITH (MODULUS 2, REMAINDER 1);
             INSERT INTO parts SELECT ('{' || i || ',' || (i * 7 % 300) || '}')
               FROM generate_series(1, 200) i;",
        )
        .expect("partitioned table");
        let (a, b) = (
            Spi::get_one::<i64>("SELECT count(*) FROM parts_0")
                .unwrap()
                .unwrap(),
            Spi::get_one::<i64>("SELECT count(*) FROM parts_1")
                .unwrap()
                .unwrap(),
        );
        assert_eq!(a + b, 200);
        assert!(
            a > 50 && b > 50,
            "rows should spread over both partitions: {a} / {b}"
        );
        // Partition pruning finds the row by its value.
        let n = Spi::get_one::<i64>("SELECT count(*) FROM parts WHERE body = '{5,35}'")
            .unwrap()
            .unwrap();
        assert_eq!(n, 1);
        let low_matches = Spi::get_one::<bool>(
            "SELECT bool_and((pgtoken.tokens_hash_extended(body, 0) & 4294967295)
                             = (pgtoken.tokens_hash(body)::bigint & 4294967295))
               FROM parts",
        )
        .unwrap()
        .unwrap();
        assert!(
            low_matches,
            "seed 0 must give the plain hash in the low 32 bits"
        );
    }

    #[pg_test]
    fn no_bytea_helper_functions_are_left_installed() {
        // pgrx declares each `_impl` function against `bytea`. They must not survive the install:
        // `tokens_eq_impl(bytea, bytea)` would compare any bytes, outside the type's checks.
        let left = Spi::get_one::<String>(
            "SELECT string_agg(p.oid::regprocedure::text, ', ') FROM pg_proc p \
             JOIN pg_namespace n ON n.oid = p.pronamespace \
             WHERE n.nspname = 'pgtoken' AND p.proname LIKE '%\\_impl'",
        )
        .unwrap();
        assert_eq!(left, None, "still installed: {left:?}");
    }

    #[pg_test(error = "cannot read a pgtoken.tokens value that has no vocabulary")]
    fn a_typmodless_column_cannot_be_read_back() {
        // The one asymmetry the two-step coercion forces: PostgreSQL accepts the DDL, and the
        // INSERT stores an unresolved value, because nothing in the write path can tell a
        // typmod-less column from a value that is about to be coerced. The error lands on the
        // way out. Pinned here so it cannot change by accident.
        Spi::run("CREATE TABLE bare (x pgtoken.tokens)").expect("create table");
        Spi::run("INSERT INTO bare VALUES ('{1,2,3}')").expect("insert");
        Spi::get_one::<String>("SELECT x::int[]::text FROM bare").unwrap();
    }

    #[pg_test]
    fn a_typmodless_columns_rows_repair_in_place_via_alter_table() {
        // The repair `unresolved_value`'s HINT prints, run from the exact starting state the hint
        // addresses: a value that is actually unresolved, not merely under a different vocabulary.
        // `an_explicit_cast_moves_a_value_to_another_vocabulary` covers the same code path
        // (`tokens_typmod_apply_impl`) but starting from a *resolved* value under a different
        // vocabulary — a different starting state, and the difference is precisely the UNRESOLVED
        // exemption this repair depends on. Change the hint and change this test together.
        //
        Spi::run(
            "SELECT pgtoken.create_vocabulary('t_repair', 256, id => 62000); \
             CREATE TABLE unresolved_ur (x pgtoken.tokens); \
             INSERT INTO unresolved_ur VALUES ('{1,2,3}');",
        )
        .expect("setup");

        // Confirm the value really is unresolved before "fixing" it, rather than assuming it:
        // `describe` is exempt from `require_vocabulary` (see its doc comment in `lib.rs`), so it
        // can report `vocabulary_id 0` on a value `::text` would refuse to read at all.
        let vocabulary_id =
            Spi::get_one::<i32>("SELECT vocabulary_id FROM unresolved_ur u, pgtoken.describe(u.x)")
                .expect("query failed");
        assert_eq!(
            vocabulary_id,
            Some(0),
            "must be unresolved (vocabulary_id 0) before the repair, or the test proves nothing"
        );

        // The exact statement shape the HINT prints: no `USING`. This is the bare assignment form
        // that a resolved cross-vocabulary value would have refused (see
        // `assignment_refuses_a_value_from_another_vocabulary`); it goes through here only because
        // `tokens_typmod_apply_impl` exempts [`UNRESOLVED`] from that refusal.
        Spi::run("ALTER TABLE unresolved_ur ALTER COLUMN x TYPE pgtoken.t_repair").expect("repair");

        let (rendered, len) = Spi::get_two::<String, i32>(
            "SELECT x::int[]::text, length(pgtoken.tokens_send(x)) FROM unresolved_ur",
        )
        .expect("query failed");
        assert_eq!(
            rendered,
            Some("{1,2,3}".to_string()),
            "the ids must survive the repair"
        );
        // Was raw24 (5 + 9) while unresolved, since `encode_unresolved` always picks the widest
        // packing. raw8 (5 + 3) here proves the repair re-encoded at t_repair's own width rather
        // than merely re-labelling the same raw24 bytes under its vocabulary id.
        assert_eq!(
            len,
            Some(5 + 3),
            "must re-encode to the vocabulary's own width (raw8), not relabel the raw24 carrier"
        );
    }

    #[pg_test]
    fn an_explicit_cast_moves_a_value_to_another_vocabulary() {
        // The migration path, and the guarantee that the syntax we print is the syntax that works:
        // the statement below is the shape both `pgtoken.vocabulary_is_immutable`'s HINT and
        // `cross_vocabulary_assignment`'s HINT tell the user to run — the `pgtoken.<name>` domain
        // spelling, not the `pgtoken.tokens('<name>')` base-type one. Change either hint and change
        // this together.
        //
        // `USING` is not decoration: the bare `ALTER TABLE … TYPE` form coerces in assignment
        // context, which `assignment_refuses_a_value_from_another_vocabulary` shows is refused. The
        // explicit cast inside `USING` is the user saying they meant it.
        vocab("t_from", 200019);
        vocab("t_to", 256);
        ensure_bytea_cast();
        Spi::run("CREATE TABLE moved (body pgtoken.tokens('t_from'))").expect("create");
        Spi::run("INSERT INTO moved VALUES ('{1,2,3}')").expect("insert");
        let before = Spi::get_one::<i32>("SELECT length(pgtoken.tokens_send(body)) FROM moved")
            .expect("query failed");
        assert_eq!(before, Some(5 + 9), "raw24 to start with");

        Spi::run(
            "ALTER TABLE moved ALTER COLUMN body TYPE pgtoken.t_to \
             USING body::pgtoken.t_to",
        )
        .expect("alter type");
        let (rendered, after) = Spi::get_two::<String, i32>(
            "SELECT body::int[]::text, length(pgtoken.tokens_send(body)) FROM moved",
        )
        .expect("query failed");
        assert_eq!(rendered, Some("{1,2,3}".to_string()), "ids must survive");
        assert_eq!(after, Some(5 + 3), "re-encoded to raw8");
    }

    #[pg_test]
    fn an_explicit_cast_chain_changes_vocabulary() {
        // The same permission, reached without ALTER TABLE's machinery: `::` is COERCION_EXPLICIT.
        vocab("x_from", 200019);
        vocab("x_to", 200019);
        let got = Spi::get_one::<String>(
            "SELECT ('{1,2,70000}'::pgtoken.tokens('x_from'))::pgtoken.tokens('x_to')::int[]::text",
        )
        .expect("query failed");
        assert_eq!(got, Some("{1,2,70000}".to_string()));
    }

    #[pg_test(error = "cannot store a_from token ids in a tokens('a_to') column")]
    fn assignment_refuses_a_value_from_another_vocabulary() {
        // Without the is_explicit split this silently re-stamped every value's vocabulary_id,
        // while the byte-identical value arriving over COPY BINARY was refused — two write paths
        // disagreeing about the one hazard the project is built around. Both vocabularies are the
        // same size, so nothing but the vocabulary differs.
        vocab("a_from", 200019);
        vocab("a_to", 200019);
        Spi::run("CREATE TABLE a_from_t (body pgtoken.tokens('a_from'))").expect("src");
        Spi::run("INSERT INTO a_from_t VALUES ('{1,2,70000}')").expect("insert");
        Spi::run("CREATE TABLE a_to_t (body pgtoken.tokens('a_to'))").expect("dst");
        Spi::run("INSERT INTO a_to_t SELECT body FROM a_from_t").unwrap();
    }

    #[pg_test(error = "token id 300 is outside vocabulary t_bound (size 256)")]
    fn rejects_an_id_outside_the_vocabulary() {
        vocab("t_bound", 256);
        Spi::get_one::<String>("SELECT '{300}'::pgtoken.tokens('t_bound')::int[]::text").unwrap();
    }

    #[pg_test(error = "vocabulary \"t_neg\" has no server tokenizer; send token ids")]
    fn a_negative_id_is_not_an_id_list() {
        // '{1,-1}' is not a strict id list, so it is text, which t_neg cannot tokenize.
        vocab("t_neg", 300);
        Spi::get_one::<String>("SELECT '{1,-1}'::pgtoken.tokens('t_neg')::int[]::text").unwrap();
    }

    #[pg_test(error = "vocabulary \"t_junk\" has no server tokenizer; send token ids")]
    fn a_vocabulary_without_a_tokenizer_rejects_text() {
        vocab("t_junk", 300);
        Spi::run("CREATE TABLE junk_t (body pgtoken.t_junk)").expect("create");
        Spi::run("INSERT INTO junk_t VALUES ('nope')").unwrap();
    }

    fn ids(sql: &str) -> Vec<i32> {
        Spi::get_one::<Vec<i32>>(sql).expect(sql).expect("NULL")
    }

    #[pg_test]
    fn text_inserted_into_an_o200k_column_is_tokenized_on_the_server() {
        o200k_docs();
        Spi::run("INSERT INTO cmp_docs VALUES (3, 'Hello, world!')").expect("insert");
        let (out, text) = Spi::get_two::<String, String>(
            "SELECT pgtoken.tokens_out(body)::text, body::text FROM cmp_docs WHERE id = 3",
        )
        .expect("query");
        assert_eq!(
            out.as_deref(),
            Some("{13225,11,2375,0}"),
            "output stays ids"
        );
        assert_eq!(text.as_deref(), Some("Hello, world!"));
        // Stored exactly as if the ids had been sent.
        assert_eq!(
            Spi::get_one::<bool>(
                "SELECT a.body = b.body AND pgtoken.tokens_send(a.body) = pgtoken.tokens_send(b.body) \
                 FROM cmp_docs a, cmp_docs b WHERE a.id = 1 AND b.id = 3"
            )
            .expect("query"),
            Some(true)
        );
    }

    /// A line of code repeated: `freq` alone stores it at about 2 bytes a token, lz4 on top
    /// finds the repeats. Equality, hashing and GROUP BY must treat the lz4 form and the plain
    /// `freq` form of the same tokens as one value, since lz4 can write the same data many ways.
    #[pg_test]
    fn repeated_text_is_stored_as_freq_lz4_and_still_compares_by_tokens() {
        Spi::run(
            "SELECT pgtoken.create_vocabulary('o200k'); \
             CREATE TABLE rep (id int, body pgtoken.o200k); \
             INSERT INTO rep VALUES (1, repeat('CThostFtdcLimitPosiSField[\"TotalVolume\"] = \"int\"' || E'\\n', 100))",
        )
        .expect("setup");
        let (codec, n, bytes) = Spi::get_three::<String, i32, i32>(
            "SELECT d.codec, d.n_tokens, d.total_bytes FROM rep, pgtoken.describe(body) d",
        )
        .expect("describe");
        assert_eq!(codec.as_deref(), Some("freq_lz4"));
        let (n, bytes) = (n.unwrap(), bytes.unwrap());
        assert!(n > 1000 && bytes < 100, "{n} tokens in {bytes} bytes");
        assert_eq!(
            Spi::get_one::<bool>(
                "SELECT body::text = repeat('CThostFtdcLimitPosiSField[\"TotalVolume\"] = \"int\"' || E'\\n', 100) FROM rep"
            )
            .expect("text"),
            Some(true)
        );

        // The same tokens as a plain `freq` value, as an older build or another lz4 would write.
        let stored = Spi::get_one::<Vec<u8>>("SELECT pgtoken.tokens_send(body) FROM rep")
            .expect("send")
            .expect("NULL");
        let plain = pgtoken_core::value::canonical(&stored)
            .unwrap()
            .into_owned();
        assert_eq!(plain[0], pgtoken_core::header::Codec::Freq.tag());
        let hex: String = plain.iter().map(|b| format!("{b:02x}")).collect();
        Spi::run(&format!(
            "INSERT INTO rep VALUES (2, pgtoken.tokens_recv_bytes(decode('{hex}', 'hex')))"
        ))
        .expect("insert plain freq");

        let (bytes_differ, equal, same_hash) = Spi::get_three::<bool, bool, bool>(
            "SELECT pgtoken.tokens_send(a.body) <> pgtoken.tokens_send(b.body), a.body = b.body, \
                    pgtoken.tokens_hash(a.body) = pgtoken.tokens_hash(b.body) \
             FROM rep a, rep b WHERE a.id = 1 AND b.id = 2",
        )
        .expect("compare");
        assert_eq!(
            (bytes_differ, equal, same_hash),
            (Some(true), Some(true), Some(true))
        );
        assert_eq!(
            Spi::get_one::<i64>("SELECT count(*) FROM (SELECT DISTINCT body FROM rep) s")
                .expect("distinct"),
            Some(1)
        );
        Spi::run("SET enable_sort = off").expect("force hash aggregation");
        assert_eq!(
            Spi::get_one::<i64>("SELECT count(*) FROM (SELECT body FROM rep GROUP BY body) s")
                .expect("group by"),
            Some(1)
        );
        // Text cast to the column's type encodes to the lz4 form and matches both rows.
        assert_eq!(
            Spi::get_one::<i64>(
                "SELECT count(*) FROM rep WHERE body = repeat('CThostFtdcLimitPosiSField[\"TotalVolume\"] = \"int\"' || E'\\n', 100)::pgtoken.o200k"
            )
            .expect("where"),
            Some(2)
        );
    }

    #[pg_test]
    fn braces_alone_do_not_make_an_id_list() {
        Spi::run("SELECT pgtoken.create_vocabulary('o200k')").expect("create");
        assert_eq!(ids("SELECT '{1,2,3}'::pgtoken.o200k::int[]"), [1, 2, 3]);
        assert_eq!(ids("SELECT ' { 1 , 2 } '::pgtoken.o200k::int[]"), [1, 2]);
        assert_eq!(ids("SELECT '{}'::pgtoken.o200k::int[]"), Vec::<i32>::new());
        assert_eq!(ids("SELECT ''::pgtoken.o200k::int[]"), Vec::<i32>::new());
        for text in [
            "{\"role\": \"user\"}",
            "{1,-1}",
            "{1,,2}",
            "{a}",
            "1,2,3",
            "{1 2}",
        ] {
            let lit = text.replace('\'', "''");
            let got = ids(&format!("SELECT '{lit}'::pgtoken.o200k::int[]"));
            let want = ids(&format!("SELECT pgtoken.tokenize('{lit}', 'o200k')"));
            assert_eq!(got, want, "{text} is text");
            assert_eq!(
                Spi::get_one::<String>(&format!("SELECT '{lit}'::pgtoken.o200k::text"))
                    .expect("query")
                    .as_deref(),
                Some(text),
                "{text} round-trips as text"
            );
        }
    }

    #[pg_test]
    fn copy_text_input_tokenizes_with_the_columns_vocabulary() {
        // COPY hands tokens_in the column's typmod, so text is tokenized there and then.
        Spi::run(
            "SELECT pgtoken.create_vocabulary('o200k'); \
             CREATE TABLE copy_txt (id int, body pgtoken.o200k); \
             COPY copy_txt FROM PROGRAM 'printf \"1\\tHello, world!\\n2\\t{13225,11,2375,0}\\n\"';",
        )
        .expect("copy");
        let (n, same) = Spi::get_two::<i64, bool>(
            "SELECT count(*), bool_and(body = '{13225,11,2375,0}') FROM copy_txt",
        )
        .expect("query");
        assert_eq!((n, same), (Some(2), Some(true)));
    }

    #[pg_test]
    fn a_text_literal_compares_against_a_column() {
        o200k_docs();
        for sql in [
            "SELECT count(*) FROM cmp_docs WHERE body = 'Hello, world!'",
            "SELECT count(*) FROM cmp_docs WHERE body <> 'Hello'",
        ] {
            assert_eq!(Spi::get_one::<i64>(sql).expect(sql), Some(1), "{sql}");
        }
        Spi::run(
            "SET plan_cache_mode = force_generic_plan; \
             PREPARE qt(pgtoken.tokens) AS SELECT count(*) FROM cmp_docs WHERE body = $1;",
        )
        .expect("prepare");
        assert_eq!(
            Spi::get_one::<i64>("EXECUTE qt('Hello, world!')").expect("execute"),
            Some(1)
        );
    }

    #[pg_test(error = "vocabulary \"t_cmp_raw\" has no server tokenizer; send token ids")]
    fn comparing_text_with_a_column_that_has_no_tokenizer_says_so() {
        vocab("t_cmp_raw", 300);
        Spi::run(
            "CREATE TABLE cmp_raw (body pgtoken.t_cmp_raw); INSERT INTO cmp_raw VALUES ('{1}')",
        )
        .expect("setup");
        Spi::get_one::<i64>("SELECT count(*) FROM cmp_raw WHERE body = 'hello'").unwrap();
    }

    #[pg_test]
    fn the_text_cast_tokenizes_on_assignment() {
        // What a driver does when it types a string parameter as text rather than unknown.
        o200k_docs();
        Spi::run(
            "INSERT INTO cmp_docs SELECT 4, 'Hello, world!'::text; \
             INSERT INTO cmp_docs SELECT 5, '{13225}'::text;",
        )
        .expect("insert");
        assert_eq!(
            ids("SELECT body::int[] FROM cmp_docs WHERE id = 4"),
            [13225, 11, 2375, 0]
        );
        assert_eq!(
            ids("SELECT body::int[] FROM cmp_docs WHERE id = 5"),
            [13225]
        );
    }

    #[pg_test]
    fn a_bare_columns_text_repairs_by_tokenizing() {
        Spi::run(
            "SELECT pgtoken.create_vocabulary('o200k'); \
             CREATE TABLE bare_txt (x pgtoken.tokens); \
             INSERT INTO bare_txt VALUES ('Hello, world!');",
        )
        .expect("setup");
        let codec = Spi::get_one::<String>("SELECT codec FROM bare_txt b, pgtoken.describe(b.x)")
            .expect("describe");
        assert_eq!(codec.as_deref(), Some("text"));
        Spi::run("ALTER TABLE bare_txt ALTER COLUMN x TYPE pgtoken.o200k").expect("repair");
        assert_eq!(ids("SELECT x::int[] FROM bare_txt"), [13225, 11, 2375, 0]);
    }

    #[pg_test(error = "cannot read a pgtoken.tokens value that has no vocabulary")]
    fn unresolved_text_cannot_be_read() {
        Spi::get_one::<String>("SELECT pgtoken.tokens_out('Hello'::pgtoken.tokens)::text").unwrap();
    }

    #[pg_test]
    fn tokenize_matches_tiktoken() {
        Spi::run("SELECT pgtoken.create_vocabulary('o200k')").expect("create");
        assert_eq!(
            ids("SELECT pgtoken.tokenize('Hello, world!', 'o200k')"),
            [13225, 11, 2375, 0]
        );
        assert_eq!(
            ids("SELECT pgtoken.tokenize('🦜', 'o200k')"),
            [4103, 99, 250]
        );
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
        Spi::get_one::<String>("SELECT '{1,2,3}'::pgtoken.tokens('t_freq')::int[]::text").unwrap();
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
        //
        // `pgtoken.encode` is gone, and `w_to`'s own width is raw24 (its vocab_size needs three
        // bytes), so a raw16 value under its id cannot come from the type itself — it is built by
        // hand, one hex byte at a time: 10 (tag: version 1, codec raw16), 00 (flags), the
        // vocabulary id as a little-endian u16, 03 (token count, LEB128), then ids 1, 2, 3 as
        // little-endian u16 each.
        let loid = Spi::get_one::<String>(&format!(
            "SELECT lo_from_bytea(0, \
                 '\\x5047434f50590aff0d0a00'::bytea || int4send(0) || int4send(0) \
                 || int2send(1::smallint) || int4send(length(v)) || v \
                 || int2send((-1)::smallint))::text \
             FROM (SELECT decode( \
                     '1000' || \
                     lpad(to_hex({id} & 255), 2, '0') || \
                     lpad(to_hex(({id} >> 8) & 255), 2, '0') || \
                     '03' || \
                     '010002000300', \
                     'hex') AS v) s"
        ))
        .expect("build the COPY BINARY stream")
        .expect("lo_from_bytea returned NULL");
        Spi::run(&format!(
            "SELECT lo_export({loid}, '/tmp/pgtoken-recv-wrongwidth.bin')"
        ))
        .expect("write the COPY BINARY stream");
        Spi::run("COPY w FROM '/tmp/pgtoken-recv-wrongwidth.bin' WITH (FORMAT binary)").unwrap();
    }

    #[pg_test(
        error = "unknown value format tag 0x00, this build reads format version 1 (tags 0x10-0x14)"
    )]
    fn recv_bytes_rejects_a_corrupt_buffer() {
        Spi::get_one::<String>(
            "SELECT pgtoken.tokens_recv_bytes('\\x000000000000000000000000'::bytea)::text",
        )
        .unwrap();
    }

    #[pg_test(
        error = "unknown value format: tag 0xA7 is the pre-release 12-byte header, which this \
                 build no longer reads; re-insert the data"
    )]
    fn recv_bytes_rejects_the_old_12_byte_format() {
        // `'{1}'` under a raw24 vocabulary in the old layout: magic, version, codec, reserved,
        // id 1 LE, reserved, count 1 LE, then the id.
        Spi::get_one::<String>(
            "SELECT pgtoken.tokens_recv_bytes('\\xa70101000100000001000000000001'::bytea)::text",
        )
        .unwrap();
    }

    #[pg_test(error = "value has flags 0x01, which this build does not understand")]
    fn recv_bytes_rejects_an_unknown_flag() {
        Spi::get_one::<String>("SELECT pgtoken.tokens_recv_bytes('\\x1301010001'::bytea)::text")
            .unwrap();
    }

    #[pg_test(error = "payload is 2 bytes, expected 3")]
    fn recv_bytes_rejects_a_count_that_disagrees_with_the_payload() {
        Spi::get_one::<String>(
            "SELECT pgtoken.tokens_recv_bytes('\\x13000100030102'::bytea)::text",
        )
        .unwrap();
    }

    /// A vocabulary with a mapping, at a pinned id.
    fn mapped_vocab(name: &str, id: i32) {
        Spi::run(&format!(
            "SELECT pgtoken.create_vocabulary('{name}', 8, id => {id})"
        ))
        .expect("create_vocabulary");
        Spi::run(&format!(
            "CREATE TEMP TABLE stage_{id} (id int, bytes bytea);
             INSERT INTO stage_{id} VALUES
               (0, 'Hello'), (1, ', '), (2, 'world'), (3, '!'),
               (4, '\\xc3'::bytea), (5, '\\xa9'::bytea);"
        ))
        .expect("stage");
        Spi::run(&format!(
            "SELECT pgtoken.load_mapping('{name}', 'SELECT id, bytes FROM stage_{id}')"
        ))
        .expect("load_mapping");
    }

    #[pg_test]
    fn text_detokenizes() {
        mapped_vocab("t_txt", 62101);
        let got =
            Spi::get_one::<String>("SELECT pgtoken.text('{0,1,2,3}'::pgtoken.tokens('t_txt'))")
                .expect("query failed");
        assert_eq!(got, Some("Hello, world!".to_string()));
    }

    #[pg_test]
    fn text_joins_a_character_split_across_tokens() {
        mapped_vocab("t_split", 62102);
        let got = Spi::get_one::<String>("SELECT pgtoken.text('{4,5}'::pgtoken.tokens('t_split'))")
            .expect("query failed");
        assert_eq!(got, Some("é".to_string()), "id -> bytes, interpreted once");
    }

    #[pg_test]
    fn text_replaces_a_chunk_cut_mid_character() {
        // An artefact of chunking by token count: the partial character becomes U+FFFD.
        mapped_vocab("t_cut", 62103);
        let got = Spi::get_one::<String>("SELECT pgtoken.text('{0,4}'::pgtoken.tokens('t_cut'))")
            .expect("query failed");
        assert_eq!(got, Some("Hello\u{FFFD}".to_string()));
    }

    #[pg_test]
    fn text_of_part_of_a_character_is_a_replacement_character_not_empty() {
        // 🦜 is o200k ids [4103, 99, 250]. The first alone is an incomplete character, which
        // used to come back as '', indistinguishable from an empty chunk.
        Spi::run("SELECT pgtoken.create_vocabulary('o200k')").expect("create");
        let (first, whole) = Spi::get_two::<String, String>(
            "SELECT pgtoken.text('{4103}'::pgtoken.o200k), \
                    pgtoken.text('{4103,99,250}'::pgtoken.o200k)",
        )
        .expect("query failed");
        assert_eq!(first.as_deref(), Some("\u{FFFD}"));
        assert_eq!(whole.as_deref(), Some("🦜"));
    }

    /// A table `cmp_docs` with one o200k row, "Hello, world!".
    fn o200k_docs() {
        Spi::run(
            "SELECT pgtoken.create_vocabulary('o200k'); \
             CREATE TABLE cmp_docs (id int, body pgtoken.o200k); \
             INSERT INTO cmp_docs VALUES (1, '{13225,11,2375,0}'), (2, '{13225}');",
        )
        .expect("setup");
    }

    #[pg_test]
    fn an_untyped_literal_compares_against_a_column() {
        // The literal reaches `=` unresolved; it used to compare as raw24 carrier bytes and never
        // match. It works with the default search_path, since the operators live in pg_catalog.
        o200k_docs();
        Spi::run("SET search_path = \"$user\", public").expect("search_path");
        for (sql, want) in [
            (
                "SELECT count(*) FROM cmp_docs WHERE body = '{13225,11,2375,0}'",
                1,
            ),
            (
                "SELECT count(*) FROM cmp_docs WHERE '{13225,11,2375,0}' = body",
                1,
            ),
            (
                "SELECT count(*) FROM cmp_docs WHERE body <> '{13225,11,2375,0}'",
                1,
            ),
            ("SELECT count(*) FROM cmp_docs WHERE body = '{ 13225 }'", 1),
            ("SELECT count(*) FROM cmp_docs WHERE body = '{}'", 0),
        ] {
            assert_eq!(Spi::get_one::<i64>(sql).expect(sql), Some(want), "{sql}");
        }
        Spi::run("SET search_path = public, pgtoken").expect("search_path");
        assert_eq!(
            Spi::get_one::<i64>("SELECT count(*) FROM cmp_docs WHERE body = '{13225,11,2375,0}'")
                .expect("query"),
            Some(1)
        );
    }

    #[pg_test]
    fn explain_prints_the_resolved_constant() {
        // An unresolved value has no output form, so EXPLAIN used to fail with "cannot read a
        // pgtoken.tokens value that has no vocabulary". The planner now resolves it first.
        o200k_docs();
        let plan = Spi::connect(|c| {
            c.select(
                "EXPLAIN (COSTS OFF) SELECT * FROM cmp_docs WHERE body = '{13225,11,2375,0}'",
                None,
                &[],
            )
            .expect("explain")
            .map(|row| row.get::<String>(1).expect("line").unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\n")
        });
        assert!(plan.contains("{13225,11,2375,0}"), "{plan}");
    }

    #[pg_test]
    fn a_parameter_resolves_at_run_time() {
        // A generic plan keeps `$1` as a parameter, so the planner cannot resolve it; the operator
        // does, once per call site.
        o200k_docs();
        Spi::run(
            "SET plan_cache_mode = force_generic_plan; \
             PREPARE q(pgtoken.tokens) AS SELECT count(*) FROM cmp_docs WHERE body = $1;",
        )
        .expect("prepare");
        assert_eq!(
            Spi::get_one::<i64>("EXECUTE q('{13225,11,2375,0}')").expect("execute"),
            Some(1)
        );
        assert_eq!(
            Spi::get_one::<i64>("EXECUTE q('{13225}')").expect("execute"),
            Some(1)
        );
    }

    #[pg_test]
    fn a_literal_compares_inside_an_expression() {
        // Neither side is a column here, so only the run-time path can resolve the literal.
        o200k_docs();
        let got = Spi::get_one::<bool>(
            "SELECT bool_or(body = (SELECT '{13225}'::pgtoken.tokens)) FROM cmp_docs",
        )
        .expect("query");
        assert_eq!(got, Some(true));
    }

    #[pg_test(error = "token id 999999 is outside vocabulary o200k (size 200019)")]
    fn comparing_with_an_id_outside_the_vocabulary_says_so() {
        o200k_docs();
        Spi::get_one::<i64>("SELECT count(*) FROM cmp_docs WHERE body = '{999999}'").unwrap();
    }

    #[pg_test(error = "cannot hash a pgtoken.tokens value that has no vocabulary")]
    fn hashing_an_unresolved_value_is_an_error_not_a_miss() {
        Spi::get_one::<i32>("SELECT pgtoken.tokens_hash('{1,2}'::pgtoken.tokens)").unwrap();
    }

    #[pg_test]
    fn text_is_immutable_so_it_can_back_an_index() {
        // The whole reason the mapping is write-once.
        let vol = Spi::get_one::<String>(
            "SELECT provolatile::text FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace \
             WHERE n.nspname = 'pgtoken' AND p.proname = 'text'",
        )
        .expect("query failed");
        assert_eq!(vol, Some("i".to_string()));
    }

    #[pg_test]
    fn text_backs_a_full_text_index() {
        mapped_vocab("t_idx", 62104);
        Spi::run(
            "CREATE TABLE idx_docs (body pgtoken.t_idx);
             INSERT INTO idx_docs VALUES ('{0,1,2,3}');
             CREATE INDEX ON idx_docs USING gin (to_tsvector('english', pgtoken.text(body)));",
        )
        .expect("index creation must be legal");
        let hits = Spi::get_one::<i64>(
            "SELECT count(*) FROM idx_docs \
             WHERE to_tsvector('english', pgtoken.text(body)) @@ to_tsquery('english', 'world')",
        )
        .expect("query failed");
        assert_eq!(hits, Some(1));
    }

    #[pg_test(error = "vocabulary t_nomap has no mapping; run pgtoken.load_mapping first")]
    fn text_without_a_mapping_says_so() {
        Spi::run("SELECT pgtoken.create_vocabulary('t_nomap', 8, id => 62105)").expect("create");
        Spi::get_one::<String>("SELECT pgtoken.text('{0}'::pgtoken.tokens('t_nomap'))").unwrap();
    }

    #[pg_test(error = "token id 6 has no entry in the mapping")]
    fn text_errors_on_an_unmapped_id() {
        // The mapping and the ids disagree about their tokenizer — a fault, not an empty string.
        mapped_vocab("t_gap", 62106);
        Spi::get_one::<String>("SELECT pgtoken.text('{6}'::pgtoken.tokens('t_gap'))").unwrap();
    }

    #[pg_test]
    fn text_detokenizes_a_freq_encoded_value() {
        // `pgtoken.text` over a `freq` value is the only call that needs both artefacts at once:
        // `decode_value` loads the ranking to get the ids back, then the mapping turns them into
        // bytes. Every other `text` test uses a `raw` vocabulary, where the ranking never comes
        // into it.
        Spi::run(
            "SELECT pgtoken.create_vocabulary('t_fq', 8, compression => 'freq', id => 62107); \
             CREATE TEMP TABLE stage_fq (id int, bytes bytea); \
             INSERT INTO stage_fq VALUES (0, 'Hello'), (1, ', '), (2, 'world'), (3, '!'); \
             SELECT pgtoken.train('t_fq', $$SELECT ARRAY[2,2,2,0,0,1]::int[]$$); \
             SELECT pgtoken.load_mapping('t_fq', 'SELECT id, bytes FROM stage_fq');",
        )
        .expect("setup");
        let codec = Spi::get_one::<String>(
            "SELECT codec FROM pgtoken.describe('{0,1,2,3}'::pgtoken.tokens('t_fq'))",
        )
        .expect("query failed");
        assert_eq!(
            codec,
            Some("freq".to_string()),
            "the ranking must be in use"
        );
        let got =
            Spi::get_one::<String>("SELECT pgtoken.text('{0,1,2,3}'::pgtoken.tokens('t_fq'))")
                .expect("query failed");
        assert_eq!(got, Some("Hello, world!".to_string()));
    }

    #[pg_test(error = "cannot read a pgtoken.tokens value that has no vocabulary")]
    fn text_refuses_a_value_without_a_vocabulary() {
        // `decode_value` alone would happily decode this — a raw codec's vocabulary_id of 0 is
        // not special to it, only `freq`'s header check cares — so without its own
        // `require_vocabulary` call, `text_impl` would fall through to a nonsense "vocabulary 0
        // has no mapping" instead of the standard unresolved-value error every other read path
        // gives.
        Spi::get_one::<String>("SELECT pgtoken.text('{1,2,3}'::pgtoken.tokens)").unwrap();
    }
}
