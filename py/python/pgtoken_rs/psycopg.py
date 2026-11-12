"""psycopg 3 integration: read and write pgtoken columns as numpy arrays of token ids.

    import psycopg
    from pgtoken_rs.psycopg import register

    conn = psycopg.connect("postgresql://...")
    vocab = register(conn, "o200k")
    conn.execute("INSERT INTO docs (body) VALUES (%s)", [vocab.encode([9906, 11, 1917])])
    conn.execute("INSERT INTO docs (body) VALUES (%s)", ["Hello, world!"])  # o200k: server tokenizes
    ids = conn.execute("SELECT body FROM docs", binary=True).fetchone()[0]

A plain `str` parameter needs nothing from this module: psycopg sends it as text, and a vocabulary
with a server tokenizer (the o200k preset) tokenizes it on the way in. Binary COPY cannot carry
text, so there a `str` is refused; tokenize on the client, or INSERT the strings.

`register` asks the server for the vocabulary's id, codec and ranking with
`pgtoken.vocabulary_codec(name)`, so it works over any connection. It never reads the server's
disk. Values are encoded and decoded on the client with the extension's own Rust codec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from psycopg.adapt import Buffer, Dumper, Loader
from psycopg.pq import Format
from psycopg.types import TypeInfo

from . import Table
from . import decode as _decode
from . import encode as _encode

__all__ = ["Tokens", "Vocabulary", "register"]

_FREQ = 2  # codec id of `freq` in the value header

_STR_HINT = (
    "a str is text, not token ids; pass it as a query parameter as it is "
    "(INSERT ... VALUES (%s)) and the server tokenizes it, which works for vocabularies "
    "with a server tokenizer such as o200k"
)


class Tokens(bytes):
    """An encoded value, ready to send as a query parameter. Made by `Vocabulary.encode`."""


@dataclass(frozen=True)
class Vocabulary:
    """A vocabulary as the client sees it: enough to encode and decode its values."""

    name: str
    id: int
    vocab_size: int
    codec: str
    table: Table | None
    type_name: str  # the column type, e.g. 'tokens.o200k'; usable in `Copy.set_types`

    def encode(self, ids) -> Tokens:
        """Token ids (list or numpy array) -> an encoded value for this vocabulary."""
        if isinstance(ids, str):
            raise TypeError(_STR_HINT)
        arr = np.asarray(ids)
        if arr.ndim != 1:
            raise ValueError(f"expected a 1-D sequence of token ids, got shape {arr.shape}")
        if arr.size and arr.dtype.kind not in "iu":
            raise TypeError(f"token ids must be integers, got {arr.dtype}")
        if arr.size and (arr.min() < 0 or arr.max() >= self.vocab_size):
            bad = arr[(arr < 0) | (arr >= self.vocab_size)][0]
            raise ValueError(
                f"token id {bad} is outside vocabulary {self.name!r} (0..{self.vocab_size - 1})"
            )
        return Tokens(_encode(arr.astype(np.uint32), self.codec, self.id, self.table))

    def decode(self, value: Buffer) -> np.ndarray:
        """An encoded value -> a uint32 numpy array of token ids."""
        return _decode(bytes(value), self.table)


def register(conn, name: str) -> Vocabulary:
    """Fetch vocabulary `name` from the server and teach `conn` to read and write its values.

    After this, on `conn`:

    * a `Tokens` parameter (from `vocab.encode(ids)`) is sent in binary, already encoded;
    * a pgtoken column reads back as a uint32 numpy array, in binary or text results;
    * binary COPY takes plain id lists or arrays once you call
      `copy.set_types([..., vocab.type_name])`.

    Call it once per vocabulary you use. Several vocabularies can share one connection.
    """
    row = conn.execute(
        "SELECT id, vocab_size, codec, ranking FROM pgtoken.vocabulary_codec(%s)", [name]
    ).fetchone()
    vid, vocab_size, codec, ranking = row
    if codec == "freq" and ranking is None:
        raise ValueError(
            f"vocabulary {name!r} uses freq compression but has no ranking yet; "
            "run pgtoken.train first"
        )
    table = Table.from_bytes(bytes(ranking)) if ranking is not None else None

    domain_oid, domain, base_oid = conn.execute(
        "SELECT t::oid, t::text, 'pgtoken.tokens'::regtype::oid"
        " FROM (SELECT format('tokens.%%I', %s::text)::regtype AS t) s",
        [name],
    ).fetchone()
    vocab = Vocabulary(name, vid, vocab_size, codec, table, domain)

    adapters = conn.adapters
    adapters.types.add(TypeInfo(domain, domain_oid, 0, regtype=domain))

    # Query parameters: send the encoded bytes as `pgtoken.tokens`. The server checks the value's
    # vocabulary against the column's when it stores it.
    class TokensDumper(Dumper):
        format = Format.BINARY
        oid = base_oid

        def dump(self, obj):
            return obj

    adapters.register_dumper(Tokens, TokensDumper)

    # COPY after `set_types`: the dumper is picked by the column's oid, so it knows the vocabulary
    # and can take plain ids.
    class DomainDumper(Dumper):
        format = Format.BINARY
        oid = domain_oid

        def dump(self, obj):
            return obj if isinstance(obj, Tokens) else vocab.encode(obj)

    adapters.register_dumper(None, DomainDumper)

    # Results: a domain column is reported with its base type's oid, so one loader on
    # `pgtoken.tokens` reads every vocabulary. It picks the ranking by the vocabulary id in each
    # value's header. Registered vocabularies are shared through the loader class.
    tables = _shared_tables(adapters, base_oid)
    tables[vid] = table
    for oid in (base_oid, domain_oid):
        adapters.register_loader(oid, _make_binary_loader(tables))
        adapters.register_loader(oid, _TextLoader)
    return vocab


def _shared_tables(adapters, base_oid: int) -> dict:
    loader = adapters.get_loader(base_oid, Format.BINARY)
    return getattr(loader, "_pgtoken_tables", {})


def _make_binary_loader(tables: dict) -> type[Loader]:
    class TokensLoader(Loader):
        format = Format.BINARY
        _pgtoken_tables = tables

        def load(self, data: Buffer) -> np.ndarray:
            data = bytes(data)
            table = None
            if len(data) >= 6 and data[2] == _FREQ:
                vid = int.from_bytes(data[4:6], "little")
                if vid not in tables:
                    raise ValueError(
                        f"value uses vocabulary id {vid}, which is not registered on this "
                        "connection; call pgtoken_rs.psycopg.register(conn, name) for it"
                    )
                table = tables[vid]
            return _decode(data, table)

    return TokensLoader


class _TextLoader(Loader):
    """Text results render a value as '{1,2,3}'. Parse that, so non-binary queries work too."""

    format = Format.TEXT

    def load(self, data: Buffer) -> np.ndarray:
        inner = bytes(data)[1:-1]
        if not inner:
            return np.empty(0, dtype=np.uint32)
        return np.array(inner.split(b","), dtype=np.uint32)
