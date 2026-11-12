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
//! 24–27   preset        0 = none, 1 = o200k (see preset.rs)
//! ```
//!
//! A preset vocabulary's id also falls in its preset's reserved range, because the read paths see
//! only a value's header and never a typmod. `tokens_typmod_in` keeps the two in agreement.

use std::ffi::{CStr, CString};

use pgrx::prelude::*;

use pgtoken_core::header::Codec;

use crate::preset;
use crate::registry::bail;
use crate::vocabulary::{
    lookup_by_name, name_for_id, Vocabulary, COMPRESSION_FREQ, COMPRESSION_RAW,
};

pub fn pack(v: Vocabulary) -> i32 {
    (v.id as i32)
        | ((v.compression as i32) << 16)
        | ((v.width as i32) << 20)
        | ((v.preset as i32) << 24)
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
        preset: ((typmod >> 24) & 0xF) as u8,
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

/// Parse a type modifier: `('name')` as users write it, or the long form
/// `('name', id, 'compression', width [, 'preset'])` that [`tokens_typmod_out`] prints.
///
/// The long form exists for `pg_dump`. A dump creates the `tokens.<name>` domains in its schema
/// section, but the `pgtoken.vocabulary` rows they name only load later, in the data section
/// (that is how `pg_extension_config_dump` tables restore). So during a restore the name does not
/// resolve yet, and the long form carries everything the typmod packs. When the name does
/// resolve, every field must match it, so the long form can never re-point a type at a different
/// vocabulary. When it does not, the id must not belong to another vocabulary either, and the
/// fields must agree with each other: a preset field needs an id in that preset's range and the
/// preset's width, and an id in a preset's range needs that preset field.
#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_typmod_in(list: Array<&CStr>) -> i32 {
    let args: Vec<&str> = list
        .iter()
        .flatten()
        .map(|a| {
            a.to_str()
                .unwrap_or_else(|_| bail("vocabulary name is not valid UTF-8"))
        })
        .collect();
    let (name, v) = match args.as_slice() {
        [name] => {
            return pack(
                lookup_by_name(name)
                    .unwrap_or_else(|| bail(format!("vocabulary {name:?} does not exist"))),
            )
        }
        [name, id, compression, width] => (name, parse_long_form(id, compression, width, None)),
        [name, id, compression, width, preset] => {
            (name, parse_long_form(id, compression, width, Some(preset)))
        }
        _ => bail(
            "pgtoken.tokens takes one type modifier, a vocabulary name \
             (or the long form pg_dump writes: name, id, compression, width [, preset])",
        ),
    };
    let v = v.unwrap_or_else(|e| {
        bail(format!(
            "invalid type modifier for vocabulary {name:?}: {e}"
        ))
    });
    match lookup_by_name(name) {
        Some(known) if known == v => {}
        Some(known) => bail(format!(
            "type modifier {} does not match vocabulary {name:?}, which is {}",
            render(name, v),
            render(name, known)
        )),
        None => {
            if let Some(other) = name_for_id(v.id) {
                bail(format!(
                    "type modifier names vocabulary {name:?} with id {}, but id {} \
                     belongs to vocabulary {other:?}",
                    v.id, v.id
                ));
            }
        }
    }
    pack(v)
}

/// The id, compression, width and preset of the long typmod form, checked for agreement with each
/// other. Pure, so a plain `#[test]` covers it.
fn parse_long_form(
    id: &str,
    compression: &str,
    width: &str,
    preset: Option<&str>,
) -> Result<Vocabulary, String> {
    let id: u16 = id
        .parse()
        .ok()
        .filter(|&i| i >= 1)
        .ok_or(format!("id {id:?} is not in 1..65535"))?;
    let compression = match compression {
        "raw" => COMPRESSION_RAW,
        "freq" => COMPRESSION_FREQ,
        other => return Err(format!("compression {other:?} is not 'raw' or 'freq'")),
    };
    let width: u8 = width
        .parse()
        .ok()
        .filter(|w| (1..=3).contains(w))
        .ok_or(format!("width {width:?} is not 1, 2 or 3"))?;
    let by_id = preset::for_id(id);
    let preset = match preset {
        None => {
            if let Some(p) = by_id {
                return Err(format!(
                    "id {id} is reserved for vocabularies on the {} preset",
                    p.name
                ));
            }
            0
        }
        Some(name) => {
            let p = preset::by_name(name).ok_or(format!(
                "unknown preset {name:?}; known: {}",
                preset::names()
            ))?;
            if by_id.map(|b| b.code) != Some(p.code) {
                return Err(format!(
                    "id {id} is outside the {} preset's ids {}..{}",
                    p.name,
                    p.ids.start(),
                    p.ids.end()
                ));
            }
            if width != p.width() {
                return Err(format!(
                    "width {width} does not match the {} preset's {}",
                    p.name,
                    p.width()
                ));
            }
            p.code
        }
    };
    Ok(Vocabulary {
        id,
        compression,
        width,
        preset,
    })
}

/// The long form, as both `typmod_out` and the mismatch error print it.
fn render(name: &str, v: Vocabulary) -> String {
    let preset = match preset::by_code(v.preset) {
        Some(p) => format!(", '{}'", p.name),
        None => String::new(),
    };
    format!(
        "('{}', {}, '{}', {}{preset})",
        name.replace('\'', "''"),
        v.id,
        compression_name(v.compression),
        v.width
    )
}

fn compression_name(code: u8) -> &'static str {
    if code == COMPRESSION_FREQ {
        "freq"
    } else {
        "raw"
    }
}

/// Print a typmod in the long form `('name', id, 'compression', width [, 'preset'])`, so `pg_dump` output
/// restores before the vocabulary catalog's rows do (see [`tokens_typmod_in`]).
#[pg_extern(immutable, parallel_safe, strict)]
fn tokens_typmod_out(typmod: i32) -> CString {
    let rendered = match unpack(typmod) {
        None => String::new(),
        Some(v) => match name_for_id(v.id) {
            Some(name) => render(&name, v),
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

    fn v(id: u16, compression: u8, width: u8, preset: u8) -> Vocabulary {
        Vocabulary {
            id,
            compression,
            width,
            preset,
        }
    }

    #[test]
    fn packs_and_unpacks_every_field() {
        for v in [
            v(1, COMPRESSION_RAW, 1, 0),
            v(3, COMPRESSION_RAW, 3, 0),
            v(65279, COMPRESSION_FREQ, 2, 0),
            v(65535, COMPRESSION_FREQ, 3, 1),
            v(65280, COMPRESSION_RAW, 3, 15),
        ] {
            let packed = pack(v);
            assert!(packed >= 0, "a typmod must not collide with -1");
            assert_eq!(unpack(packed), Some(v));
        }
    }

    #[test]
    fn the_long_form_parses_and_rejects_bad_fields() {
        assert_eq!(
            parse_long_form("6", "freq", "2", None),
            Ok(v(6, COMPRESSION_FREQ, 2, 0))
        );
        assert_eq!(
            parse_long_form("65280", "raw", "3", Some("o200k")),
            Ok(v(65280, COMPRESSION_RAW, 3, 1))
        );
        for (id, c, w, p) in [
            ("0", "raw", "2", None),
            ("65536", "raw", "2", None),
            ("x", "raw", "2", None),
            ("6", "zstd", "2", None),
            ("6", "raw", "4", None),
            ("6", "raw", "0", None),
            // A preset's ids need its preset field, and the preset field needs its ids and width.
            ("65300", "freq", "3", None),
            ("6", "freq", "3", Some("o200k")),
            ("65300", "freq", "2", Some("o200k")),
            ("65300", "freq", "3", Some("cl100k")),
        ] {
            assert!(parse_long_form(id, c, w, p).is_err(), "{id} {c} {w} {p:?}");
        }
    }

    #[test]
    fn no_typmod_unpacks_to_nothing() {
        assert_eq!(unpack(-1), None);
    }

    #[test]
    fn width_selects_the_raw_codec() {
        for (width, want) in [(1, Codec::Raw8), (2, Codec::Raw16), (3, Codec::Raw24)] {
            assert_eq!(codec_for(v(1, COMPRESSION_RAW, width, 0)), want);
        }
        assert_eq!(
            codec_for(v(1, COMPRESSION_FREQ, 3, 0)),
            Codec::Freq,
            "freq ignores the width"
        );
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
        let out = out.expect("typmod_out returned NULL");
        assert!(
            out.starts_with("('tm1', ") && out.ends_with(", 'raw', 2)"),
            "{out}"
        );
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
        assert_eq!(out, Some("('tm_wide', 40000, 'raw', 2)".to_string()));
    }

    #[pg_test(error = "vocabulary \"nope\" does not exist")]
    fn typmod_in_rejects_an_unknown_name() {
        Spi::get_one::<i32>("SELECT pgtoken.tokens_typmod_in('{nope}'::cstring[])").unwrap();
    }

    #[pg_test(
        error = "pgtoken.tokens takes one type modifier, a vocabulary name (or the long form pg_dump writes: name, id, compression, width [, preset])"
    )]
    fn typmod_in_rejects_two_arguments() {
        Spi::get_one::<i32>("SELECT pgtoken.tokens_typmod_in('{a,b}'::cstring[])").unwrap();
    }

    #[pg_test]
    fn the_long_form_round_trips_through_typmod_in() {
        Spi::run("SELECT pgtoken.create_vocabulary('tm_long', 50257, compression => 'freq', id => 41000)")
            .expect("create");
        let same = Spi::get_one::<bool>(
            "SELECT pgtoken.tokens_typmod_in('{tm_long}'::cstring[]) \
                  = pgtoken.tokens_typmod_in('{tm_long,41000,freq,2}'::cstring[])",
        )
        .expect("query failed");
        assert_eq!(same, Some(true));
    }

    #[pg_test(
        error = "type modifier ('tm_mis', 41001, 'freq', 2) does not match vocabulary \"tm_mis\", which is ('tm_mis', 41001, 'raw', 2)"
    )]
    fn the_long_form_must_match_a_known_vocabulary() {
        Spi::run("SELECT pgtoken.create_vocabulary('tm_mis', 50257, id => 41001)").expect("create");
        Spi::get_one::<i32>("SELECT pgtoken.tokens_typmod_in('{tm_mis,41001,freq,2}'::cstring[])")
            .unwrap();
    }

    #[pg_test]
    fn the_long_form_resolves_before_the_catalog_row_exists() {
        // What a pg_dump restore does: the domain is created before pgtoken.vocabulary's rows load.
        let packed = Spi::get_one::<i32>(
            "SELECT pgtoken.tokens_typmod_in('{not_loaded_yet,41002,freq,2}'::cstring[])",
        )
        .expect("query failed")
        .expect("NULL typmod");
        assert_eq!(packed & 0xFFFF, 41002);
    }

    #[pg_test(
        error = "type modifier names vocabulary \"other_name\" with id 41003, but id 41003 belongs to vocabulary \"tm_taken\""
    )]
    fn the_long_form_cannot_borrow_another_vocabularys_id() {
        Spi::run("SELECT pgtoken.create_vocabulary('tm_taken', 300, id => 41003)").expect("create");
        Spi::get_one::<i32>(
            "SELECT pgtoken.tokens_typmod_in('{other_name,41003,raw,2}'::cstring[])",
        )
        .unwrap();
    }

    #[pg_test]
    fn a_value_loads_before_the_catalog_row_exists() {
        // A pg_dump restore can load a table's rows before pgtoken.vocabulary's rows, when the
        // table's schema sorts before `pgtoken`. Input then checks ids against the width.
        let ids = Spi::get_one::<Vec<i32>>(
            "SELECT '{464,262,65535}'::pgtoken.tokens(not_loaded_yet, 41004, raw, 2)::int[]",
        )
        .expect("query failed");
        assert_eq!(ids, Some(vec![464, 262, 65535]));
    }

    #[pg_test]
    fn the_preset_long_form_round_trips_through_typmod_in() {
        Spi::run("SELECT pgtoken.create_vocabulary('tm_p', preset => 'o200k')").expect("create");
        let (same, out) = Spi::get_two::<bool, String>(
            "SELECT pgtoken.tokens_typmod_in('{tm_p}'::cstring[]) \
                  = pgtoken.tokens_typmod_in('{tm_p,65280,freq,3,o200k}'::cstring[]), \
                    pgtoken.tokens_typmod_out(pgtoken.tokens_typmod_in('{tm_p}'::cstring[]))::text",
        )
        .expect("query failed");
        assert_eq!(same, Some(true));
        assert_eq!(out.as_deref(), Some("('tm_p', 65280, 'freq', 3, 'o200k')"));
    }

    #[pg_test(
        error = "invalid type modifier for vocabulary \"tm_p2\": id 65280 is reserved for vocabularies on the o200k preset"
    )]
    fn the_long_form_cannot_drop_a_preset() {
        Spi::run("SELECT pgtoken.create_vocabulary('tm_p2', preset => 'o200k')").expect("create");
        Spi::get_one::<i32>("SELECT pgtoken.tokens_typmod_in('{tm_p2,65280,freq,3}'::cstring[])")
            .unwrap();
    }

    #[pg_test(
        error = "type modifier names vocabulary \"other\" with id 65280, but id 65280 belongs to vocabulary \"tm_p3\""
    )]
    fn the_preset_long_form_cannot_borrow_another_vocabularys_id() {
        Spi::run("SELECT pgtoken.create_vocabulary('tm_p3', preset => 'o200k')").expect("create");
        Spi::get_one::<i32>(
            "SELECT pgtoken.tokens_typmod_in('{other,65280,freq,3,o200k}'::cstring[])",
        )
        .unwrap();
    }
}
