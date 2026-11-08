"""Rust-speed client codec for pgtoken values, using the extension's own `pgtoken-core`."""

from __future__ import annotations

import numpy as np

from ._native import Table, decode as _decode, encode as _encode

__all__ = ["Table", "load_table", "decode", "encode"]


def load_table(path: str) -> Table:
    """Load a trained `freq` ranking from the extension's `<vocabulary id>.tntt` file."""
    with open(path, "rb") as f:
        return Table.from_bytes(f.read())


def decode(value: bytes, table: Table | None = None) -> np.ndarray:
    """Stored value bytes -> uint32 token ids."""
    return np.frombuffer(_decode(value, table), dtype="<u4")


def encode(ids, codec: str, vocabulary_id: int, table: Table | None = None) -> bytes:
    """Token ids -> stored value bytes, ready for a binary INSERT or COPY."""
    return _encode(np.ascontiguousarray(ids, dtype=np.uint32), codec, vocabulary_id, table)
