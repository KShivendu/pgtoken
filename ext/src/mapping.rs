//! Loading a vocabulary's `token_id -> bytes` mapping.
//!
//! Input is rows, not a server-side file path: on managed PostgreSQL you cannot put a file on the
//! host, and a path would need superuser besides. Rows arrive over the normal connection and load
//! with plain `COPY`. Same shape `train` already has.
//!
//! The extension receives plain `id -> bytes` and concatenates. Every tokenizer-specific decoding
//! step — SentencePiece's `▁`, HuggingFace's ByteLevel, leading-space stripping — is resolved by
//! the user at export time, which is what lets this exist while the server still has no
//! tokenizer.

use pgrx::prelude::*;

use pgtoken_core::tables::ByteMap;

use crate::registry::{self, bail};
use crate::vocabulary;

/// Build a vocabulary's mapping from a query returning `(int, bytea)`.
///
/// Write-once: `pgtoken.text` is `IMMUTABLE` so it can back a full-text index, which is only
/// honest if the mapping it reads can never change.
#[pg_extern(strict)]
fn load_mapping(name: &str, query: &str) -> String {
    let v = vocabulary::lookup_by_name(name)
        .unwrap_or_else(|| bail(format!("vocabulary {name:?} does not exist")));
    if registry::map_path(v.id).exists() {
        bail(format!("vocabulary {name} already has a mapping"));
    }
    let vocab_size = vocabulary::vocab_size_for(v.id)
        .unwrap_or_else(|| bail(format!("vocabulary {name:?} has no catalog row")));

    let mut pairs: Vec<(u32, Vec<u8>)> = Vec::new();
    Spi::connect(|client| {
        let tup = client.select(query, None, &[]).unwrap_or_else(|e| bail(e));
        for row in tup {
            let id: Option<i32> = row.get(1).unwrap_or_else(|e| bail(e));
            let bytes: Option<Vec<u8>> = row.get(2).unwrap_or_else(|e| bail(e));
            match (id, bytes) {
                (Some(i), Some(b)) if i >= 0 => pairs.push((i as u32, b)),
                (Some(i), _) if i < 0 => bail(format!("token id {i} is negative")),
                _ => bail("mapping rows must not contain NULL"),
            }
        }
    });
    if pairs.is_empty() {
        bail("mapping query returned no rows; it must return (int, bytea)");
    }

    let map = ByteMap::build(&pairs, vocab_size).unwrap_or_else(|e| bail(e));
    let mapped = map.mapped();
    registry::write_map(v.id, &map.to_bytes()).unwrap_or_else(|e| bail(e));
    format!("mapped {mapped} of {vocab_size} ids for vocabulary {name}")
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    /// Create a vocabulary at a pinned id and stage a tiny mapping for it.
    ///
    /// The id must be pinned and disjoint from every other test's: mapping files survive the
    /// transaction rollback that resets the catalog, so an auto-assigned id would collide with
    /// another test's file.
    fn staged(name: &str, id: i32) {
        Spi::run(&format!(
            "SELECT pgtoken.create_vocabulary('{name}', 8, id => {id})"
        ))
        .expect("create_vocabulary");
        Spi::run(
            "CREATE TEMP TABLE stage (id int, bytes bytea);
             INSERT INTO stage VALUES
               (0, 'Hello'), (1, ', '), (2, 'world'), (3, '!'),
               (4, '\\xc3'::bytea), (5, '\\xa9'::bytea);",
        )
        .expect("stage");
    }

    #[pg_test]
    fn loads_a_mapping_and_reports_it() {
        staged("m_load", 62001);
        let msg = Spi::get_one::<String>(
            "SELECT pgtoken.load_mapping('m_load', 'SELECT id, bytes FROM stage')",
        )
        .expect("load_mapping");
        assert!(
            msg.unwrap().contains("6"),
            "the message should say how many ids it mapped"
        );
        let mapped = Spi::get_one::<i32>("SELECT mapped FROM pgtoken.vocabulary_info('m_load')")
            .expect("query failed");
        assert_eq!(mapped, Some(6));
    }

    #[pg_test]
    fn a_vocabulary_without_a_mapping_reports_null() {
        Spi::run("SELECT pgtoken.create_vocabulary('m_none', 8, id => 62002)").expect("create");
        let mapped = Spi::get_one::<i32>("SELECT mapped FROM pgtoken.vocabulary_info('m_none')")
            .expect("query failed");
        assert_eq!(mapped, None, "unmapped is a normal state, not an error");
    }

    #[pg_test(error = "vocabulary m_twice already has a mapping")]
    fn load_mapping_is_write_once() {
        // text() must stay IMMUTABLE to back an index, so the mapping cannot change.
        staged("m_twice", 62003);
        Spi::run("SELECT pgtoken.load_mapping('m_twice', 'SELECT id, bytes FROM stage')")
            .expect("first load");
        Spi::get_one::<String>(
            "SELECT pgtoken.load_mapping('m_twice', 'SELECT id, bytes FROM stage')",
        )
        .unwrap();
    }

    #[pg_test(error = "token id 99 is outside the vocabulary's declared size 8")]
    fn load_mapping_rejects_an_id_outside_vocab_size() {
        Spi::run("SELECT pgtoken.create_vocabulary('m_oob', 8, id => 62004)").expect("create");
        Spi::get_one::<String>(
            "SELECT pgtoken.load_mapping('m_oob', $$SELECT 99::int, 'x'::bytea$$)",
        )
        .unwrap();
    }

    #[pg_test(error = "vocabulary \"m_nope\" does not exist")]
    fn load_mapping_needs_a_real_vocabulary() {
        // Dollar-quoted, not a single-quoted literal with embedded (unescaped) quotes: the naive
        // 'SELECT 0, \'x\'::bytea' form is not valid SQL -- the outer parser sees the string
        // literal end at the first embedded quote and reports its own syntax error before
        // load_mapping ever runs, which is not what this test means to exercise.
        Spi::get_one::<String>("SELECT pgtoken.load_mapping('m_nope', $$SELECT 0, 'x'::bytea$$)")
            .unwrap();
    }

    #[pg_test(error = "mapping query returned no rows; it must return (int, bytea)")]
    fn load_mapping_rejects_an_empty_query() {
        Spi::run("SELECT pgtoken.create_vocabulary('m_empty', 8, id => 62005)").expect("create");
        Spi::get_one::<String>(
            "SELECT pgtoken.load_mapping('m_empty', $$SELECT 0::int, 'x'::bytea WHERE false$$)",
        )
        .unwrap();
    }
}
