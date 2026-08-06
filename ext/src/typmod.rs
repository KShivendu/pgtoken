//! Packing a resolved vocabulary into a type modifier.
//!
//! A typmod is one `int32`. Everything the write path needs lives in it, so writing a row never
//! re-resolves a vocabulary name — that happens once, in `typmod_in`, at DDL time.
//!
//! ```text
//! bits    field
//!  0–15   vocabulary id (1..65535; 0 only in the unset case)
//! 16–19   compression   0 = raw, 1 = freq
//! 20–22   width         1..4 bytes per token
//! ```

use std::ffi::{CStr, CString};

use pgrx::prelude::*;

use pgtoken_core::header::Codec;

use crate::registry::bail;
use crate::vocabulary::{lookup_by_name, name_for_id, Vocabulary, COMPRESSION_FREQ};

pub fn pack(v: Vocabulary) -> i32 {
    (v.id as i32) | ((v.compression as i32) << 16) | ((v.width as i32) << 20)
}

/// `None` for PostgreSQL's "no typmod" (`-1`) and anything else negative. A value with no
/// vocabulary has no width, so callers must treat `None` as an error rather than a default.
pub fn unpack(typmod: i32) -> Option<Vocabulary> {
    if typmod < 0 {
        return None;
    }
    Some(Vocabulary {
        id: (typmod & 0xFFFF) as u16,
        compression: ((typmod >> 16) & 0xF) as u8,
        width: ((typmod >> 20) & 0x7) as u8,
    })
}

/// The codec a vocabulary's compression and width select.
///
/// Pure on purpose, unlike the rest of this module: a plain `#[test]` calls this directly (see
/// `width_selects_the_raw_codec` below), and that test binary has no Postgres backend behind
/// it. `crate::registry::bail` reaches `pgrx::error!`, which needs real backend symbols
/// (`errstart`, `CurrentMemoryContext`, ...) that only exist inside a running Postgres process;
/// linking a plain `cargo test` binary against them is not possible. `panic!` needs nothing
/// from Postgres to compile or link, and pgrx's `#[pg_extern]` wrapper still catches it and
/// reports it as a normal `ERROR` to any SQL caller, so the width-4 case (unreachable while
/// `MAX_VOCAB_SIZE` stays at 3 bytes) is still not silently accepted.
pub fn codec_for(v: Vocabulary) -> Codec {
    if v.compression == COMPRESSION_FREQ {
        return Codec::Freq;
    }
    match v.width {
        1 => Codec::Raw8,
        2 => Codec::Raw16,
        3 => Codec::Raw24,
        w => panic!("width {w} needs a raw32 codec, which this build does not have"),
    }
}

#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_typmod_in(list: Array<&CStr>) -> i32 {
    let args: Vec<&CStr> = list.iter().flatten().collect();
    if args.len() != 1 {
        bail("pgtoken.tokens takes exactly one type modifier: a vocabulary name");
    }
    let name = args[0]
        .to_str()
        .unwrap_or_else(|_| bail("vocabulary name is not valid UTF-8"));
    let v =
        lookup_by_name(name).unwrap_or_else(|| bail(format!("vocabulary {name:?} does not exist")));
    pack(v)
}

#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_typmod_out(typmod: i32) -> CString {
    let rendered = match unpack(typmod) {
        None => String::new(),
        Some(v) => match name_for_id(v.id) {
            Some(name) => format!("('{name}')"),
            // Deliberately not an error: see typmod_out_falls_back_to_the_raw_id.
            None => format!("({})", v.id),
        },
    };
    CString::new(rendered).unwrap_or_else(|_| bail("rendered typmod contained a NUL"))
}

// Named `unit_tests`, not `tests`: pgrx_tests hardcodes the schema for `#[pg_test]` proxy
// functions to `tests` (see `pgrx-tests-0.19.2/src/framework.rs`), so the `#[pg_schema] mod`
// below must be the one literally named `tests` — matching the convention already used in
// `vocabulary.rs` and `lib.rs` — and this plain-Rust module needs a different name to avoid
// colliding with it.
#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::vocabulary::{Vocabulary, COMPRESSION_FREQ, COMPRESSION_RAW};
    use pgtoken_core::header::Codec;

    #[test]
    fn packs_and_unpacks_every_field() {
        for v in [
            Vocabulary {
                id: 1,
                compression: COMPRESSION_RAW,
                width: 1,
            },
            Vocabulary {
                id: 3,
                compression: COMPRESSION_RAW,
                width: 3,
            },
            Vocabulary {
                id: 65535,
                compression: COMPRESSION_FREQ,
                width: 2,
            },
        ] {
            let packed = pack(v);
            assert!(packed >= 0, "a typmod must not collide with -1");
            assert_eq!(unpack(packed), Some(v));
        }
    }

    #[test]
    fn no_typmod_unpacks_to_nothing() {
        assert_eq!(unpack(-1), None);
    }

    #[test]
    fn width_selects_the_raw_codec() {
        for (width, want) in [(1, Codec::Raw8), (2, Codec::Raw16), (3, Codec::Raw24)] {
            let v = Vocabulary {
                id: 1,
                compression: COMPRESSION_RAW,
                width,
            };
            assert_eq!(codec_for(v), want);
        }
        let v = Vocabulary {
            id: 1,
            compression: COMPRESSION_FREQ,
            width: 3,
        };
        assert_eq!(codec_for(v), Codec::Freq, "freq ignores the width");
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn typmod_in_resolves_a_name() {
        Spi::run("SELECT pgtoken.create_vocabulary('tm1', 32000)").expect("create");
        // `tokens_typmod_out` returns `cstring`, which Spi's `get_one` cannot hand back as a
        // Rust `String` directly (their OIDs don't match); cast to `text` for the test.
        let out = Spi::get_one::<String>(
            "SELECT pgtoken.tokens_typmod_out(pgtoken.tokens_typmod_in('{tm1}'::cstring[]))::text",
        )
        .expect("query failed");
        assert_eq!(out, Some("('tm1')".to_string()));
    }

    #[pg_test]
    fn typmod_out_falls_back_to_the_raw_id() {
        // typmod_out can run while formatting an error in an aborted transaction, so an
        // unresolvable id must print rather than raise.
        let out = Spi::get_one::<String>("SELECT pgtoken.tokens_typmod_out(9999)::text")
            .expect("query failed");
        assert!(out.unwrap().contains("9999"));
    }

    #[pg_test]
    fn typmod_out_resolves_a_name_above_the_smallint_range() {
        // Task 3 fixed `name_for_id`'s SQL to handle ids above smallint's range, but nothing
        // called it until this task, so it was only verified by inspection. Run it for real.
        let id =
            Spi::get_one::<i32>("SELECT pgtoken.create_vocabulary('tm_wide', 300, id => 40000)")
                .expect("create failed")
                .expect("create returned NULL");
        assert_eq!(id, 40000, "the vocabulary must land above smallint's range");

        let out = Spi::get_one::<String>(
            "SELECT pgtoken.tokens_typmod_out(pgtoken.tokens_typmod_in('{tm_wide}'::cstring[]))::text",
        )
        .expect("query failed");
        assert_eq!(out, Some("('tm_wide')".to_string()));
    }

    #[pg_test(error = "vocabulary \"nope\" does not exist")]
    fn typmod_in_rejects_an_unknown_name() {
        Spi::get_one::<i32>("SELECT pgtoken.tokens_typmod_in('{nope}'::cstring[])").unwrap();
    }

    #[pg_test(error = "pgtoken.tokens takes exactly one type modifier: a vocabulary name")]
    fn typmod_in_rejects_two_arguments() {
        Spi::get_one::<i32>("SELECT pgtoken.tokens_typmod_in('{a,b}'::cstring[])").unwrap();
    }
}
