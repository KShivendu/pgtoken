//! Vocabularies bundled with the extension: a frozen ranking and mapping compiled into the binary.
//!
//! `SELECT pgtoken.create_vocabulary('o200k')` gives a working `freq` vocabulary with
//! `pgtoken.text()` and no other setup, because both artefacts ship inside the shared library.
//! They are zstd-compressed (`ext/presets/*.zst`, built by `scripts/build_o200k_preset.py`) and
//! decompressed once per backend, on first use, with a pure-Rust decoder.
//!
//! A preset vocabulary depends on nothing in the database to encode, decode or detokenize:
//!
//! * The write path reads the preset from the column's typmod (bits 24-27, see `typmod.rs`).
//! * The read paths see only a value's 12-byte header, which carries the vocabulary id and no
//!   typmod. So each preset owns a fixed range of vocabulary ids, and an id in that range *is*
//!   the preset. o200k owns 65280..=65535, room for 256 vocabularies built on it.
//!
//! That is what makes preset columns survive `pg_dump`, `pg_restore -j` and replicas with nothing
//! to copy: the artefacts are in the binary on every server that has the extension installed.
//!
//! A bundled artefact is frozen forever. Rows encoded with it must decode the same in every later
//! release, so a better ranking ships as a new preset with a new name and a new id range, never
//! as a change to this one. `tests::the_bundled_ranking_is_frozen` fails if the bytes change.

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Read;
use std::ops::RangeInclusive;
use std::rc::Rc;

use pgtoken_core::tables::{ByteMap, RankTable};

use crate::tokenizer::{TiktokenBpe, Tokenizer, O200K_PATTERN};

/// How to build a preset's server tokenizer from its own mapping.
#[derive(Debug)]
pub struct TokenizerSpec {
    /// The pre-tokenizer regex.
    pub pattern: &'static str,
    /// Ids below this are mergeable ranks; the rest are special tokens, never produced from text.
    pub mergeable: u32,
}

/// One bundled vocabulary.
#[derive(Debug)]
pub struct Preset {
    /// Packed into typmod bits 24-27. 0 means "no preset", so codes start at 1.
    pub code: u8,
    pub name: &'static str,
    pub vocab_size: u32,
    /// Vocabulary ids reserved for vocabularies built on this preset.
    pub ids: RangeInclusive<u16>,
    ranking_zst: &'static [u8],
    mapping_zst: &'static [u8],
    /// `None` for a preset whose text must be tokenized client-side.
    tokenizer: Option<TokenizerSpec>,
}

pub const O200K: Preset = Preset {
    code: 1,
    name: "o200k",
    vocab_size: 200_019,
    ids: 65280..=65535,
    ranking_zst: include_bytes!("../presets/o200k.tntt.zst"),
    mapping_zst: include_bytes!("../presets/o200k.tnmap.zst"),
    // tiktoken's o200k_base: ranks 0..=199997, then special tokens (<|endoftext|> is 199999).
    tokenizer: Some(TokenizerSpec {
        pattern: O200K_PATTERN,
        mergeable: 199_998,
    }),
};

pub const ALL: &[&Preset] = &[&O200K];

pub fn by_code(code: u8) -> Option<&'static Preset> {
    ALL.iter().copied().find(|p| p.code == code)
}

pub fn by_name(name: &str) -> Option<&'static Preset> {
    ALL.iter().copied().find(|p| p.name == name)
}

/// The preset a vocabulary id belongs to, by its reserved range. This is how the read paths,
/// which have no typmod, tell a preset value from a custom one.
pub fn for_id(id: u16) -> Option<&'static Preset> {
    ALL.iter().copied().find(|p| p.ids.contains(&id))
}

/// The names `create_vocabulary` accepts for `preset =>`, for error messages.
pub fn names() -> String {
    ALL.iter()
        .map(|p| format!("'{}'", p.name))
        .collect::<Vec<_>>()
        .join(", ")
}

thread_local! {
    static RANKINGS: RefCell<HashMap<u8, Rc<RankTable>>> = RefCell::new(HashMap::new());
    static MAPPINGS: RefCell<HashMap<u8, Rc<ByteMap>>> = RefCell::new(HashMap::new());
    static TOKENIZERS: RefCell<HashMap<u8, Rc<dyn Tokenizer>>> = RefCell::new(HashMap::new());
}

fn unzstd(what: &str, compressed: &[u8]) -> Vec<u8> {
    // A panic, not an error string: these bytes are compiled in and checked by the tests below,
    // so failing here means a broken build. pgrx still reports it to SQL as an ordinary ERROR.
    let mut d = ruzstd::decoding::StreamingDecoder::new(compressed)
        .unwrap_or_else(|e| panic!("bundled {what} is not a zstd frame: {e}"));
    let mut out = Vec::new();
    d.read_to_end(&mut out)
        .unwrap_or_else(|e| panic!("bundled {what} does not decompress: {e}"));
    out
}

impl Preset {
    /// The storage width every vocabulary on this preset has.
    pub fn width(&self) -> u8 {
        crate::vocabulary::width_for(self.vocab_size as i32) as u8
    }

    /// The ranking's bytes, in the `.tntt` format of `core/src/tables.rs`.
    pub fn ranking_bytes(&self) -> Vec<u8> {
        unzstd(&format!("{} ranking", self.name), self.ranking_zst)
    }

    /// The mapping's bytes, in the `.tnmap` format of `core/src/tables.rs`.
    pub fn mapping_bytes(&self) -> Vec<u8> {
        unzstd(&format!("{} mapping", self.name), self.mapping_zst)
    }

    /// The parsed ranking, decompressed at most once per backend.
    pub fn ranking(&self) -> Rc<RankTable> {
        if let Some(hit) = RANKINGS.with(|c| c.borrow().get(&self.code).cloned()) {
            return hit;
        }
        let t = Rc::new(
            RankTable::from_bytes(&self.ranking_bytes())
                .unwrap_or_else(|e| panic!("bundled {} ranking is not valid: {e}", self.name)),
        );
        RANKINGS.with(|c| c.borrow_mut().insert(self.code, t.clone()));
        t
    }

    /// The parsed mapping, decompressed at most once per backend.
    pub fn mapping(&self) -> Rc<ByteMap> {
        if let Some(hit) = MAPPINGS.with(|c| c.borrow().get(&self.code).cloned()) {
            return hit;
        }
        let m = Rc::new(
            ByteMap::from_bytes(&self.mapping_bytes())
                .unwrap_or_else(|e| panic!("bundled {} mapping is not valid: {e}", self.name)),
        );
        MAPPINGS.with(|c| c.borrow_mut().insert(self.code, m.clone()));
        m
    }

    /// The server tokenizer, built from the bundled mapping at most once per backend, or `None`
    /// if this preset has none.
    pub fn tokenizer(&self) -> Option<Rc<dyn Tokenizer>> {
        let spec = self.tokenizer.as_ref()?;
        if let Some(hit) = TOKENIZERS.with(|c| c.borrow().get(&self.code).cloned()) {
            return Some(hit);
        }
        let t: Rc<dyn Tokenizer> = Rc::new(
            TiktokenBpe::from_mapping(&self.mapping(), spec.mergeable, spec.pattern)
                .unwrap_or_else(|e| panic!("bundled {} tokenizer is not valid: {e}", self.name)),
        );
        TOKENIZERS.with(|c| c.borrow_mut().insert(self.code, t.clone()));
        Some(t)
    }
}

// Plain tests: no PostgreSQL needed, they only touch the bundled bytes and the core codec.
#[cfg(test)]
mod unit_tests {
    use super::*;
    use pgtoken_core::header::Codec;
    use pgtoken_core::tables::{digest_hex, table_digest};

    /// From `ext/presets/o200k.json`, written by `scripts/build_o200k_preset.py`.
    const RANKING_SHA256: &str = "743afb3cae0eac236f9bece91973ebe3038539b557ac3dcc955bb0d9a1de42af";
    const MAPPING_SHA256: &str = "5c216d2c1a8df59564750fca6813223854b294ffa4f296e3f40a400668e14de7";

    #[test]
    fn the_bundled_artefacts_match_their_recorded_sha256() {
        let ranking = O200K.ranking_bytes();
        assert_eq!(ranking.len(), 309_864);
        assert_eq!(digest_hex(&table_digest(&ranking)), RANKING_SHA256);
        let mapping = O200K.mapping_bytes();
        assert_eq!(mapping.len(), 2_197_790);
        assert_eq!(digest_hex(&table_digest(&mapping)), MAPPING_SHA256);
    }

    #[test]
    fn the_bundled_artefacts_parse() {
        assert_eq!(O200K.ranking().k(), 77_463);
        let map = O200K.mapping();
        assert_eq!(map.vocab_size(), O200K.vocab_size);
        assert_eq!(map.get(13225), Some(&b"Hello"[..]));
        assert_eq!(O200K.width(), 3);
    }

    /// Golden bytes: a fixed id sequence encoded under the bundled ranking. Values on disk were
    /// written with this ranking, so if this ever changes, they would decode to different ids.
    #[test]
    fn the_bundled_ranking_is_frozen() {
        let ids = [13225u32, 11, 2375, 0, 220, 290, 200_018, 199_999, 12_345, 1];
        let got = pgtoken_core::value::encode(&ids, Codec::Freq, 65280, Some(&O200K.ranking()))
            .expect("encode");
        let hex: String = got.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, GOLDEN);
        let back = pgtoken_core::value::decode(&got, Some(&O200K.ranking())).expect("decode");
        assert_eq!(back, ids);
    }

    const GOLDEN: &str = "a701020000ff00000a00000051a001381f00280302010201e93b04d63b04bb5360";

    #[test]
    fn ids_resolve_to_their_preset() {
        assert_eq!(for_id(65280).map(|p| p.name), Some("o200k"));
        assert_eq!(for_id(65535).map(|p| p.name), Some("o200k"));
        assert!(for_id(65279).is_none());
        assert!(for_id(1).is_none());
        assert_eq!(by_code(1).map(|p| p.name), Some("o200k"));
        assert!(by_code(0).is_none());
    }
}
