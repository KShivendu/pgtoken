"""Client-side codec for the pgtoken value format.

This is what an agent would run: it reads the blob the extension stored and turns it back into
token IDs without the database doing any work. It is also what makes an honest read benchmark
possible, since the point of the format is that the client decodes.

Everything is vectorised with numpy. A per-value Python loop would be so much slower than the
Rust codec that it would dominate any measurement and make `freq` look far worse than it is.

Validated against the extension by `test_client.py`, which round-trips both directions.

## Value format

```
off  size  field
  0     1  tag = (version 1 << 4) | codec    codec 0=raw16 1=raw24 2=freq 3=raw8
  1     1  flags, must be zero
  2     2  vocabulary id (u16 LE; 0 = none)
  4   1-5  token count, unsigned LEB128, minimal
  *     -  payload: raw = count x width bytes; freq = one minimal LEB128 rank per token
```

## Coding table format

```
off  size  field
  0     4  magic "TNTT"
  4     1  version (1)
  5     1  kind (1 = rank)
  6     2  reserved
  8     4  k (u32 LE)
 12     -  token_of_rank: k x u32 LE
```
"""

from __future__ import annotations

import numpy as np

VERSION = 1
LEGACY_MAGIC = 0xA7  # first byte of the pre-release 12-byte layout

RAW16, RAW24, FREQ, RAW8 = 0, 1, 2, 3
CODEC_NAMES = {RAW16: "raw16", RAW24: "raw24", FREQ: "freq", RAW8: "raw8"}
CODEC_IDS = {v: k for k, v in CODEC_NAMES.items()}



class RankTable:
    """The sparse frequency table. Tokens it never saw map to `k + id`."""

    def __init__(self, token_of_rank: np.ndarray):
        self.token_of_rank = token_of_rank.astype(np.uint32)
        self.k = len(token_of_rank)
        # Inverse lookup as a sorted array plus searchsorted, not a dict. The table is sparse
        # over an unbounded ID space so a dense array would need a vocabulary size, which is
        # exactly what the format avoids — but a dict would force a Python-level loop per
        # token, which costs more than the rest of the codec put together.
        order = np.argsort(self.token_of_rank, kind="stable")
        self._sorted_tokens = self.token_of_rank[order]
        self._sorted_ranks = order.astype(np.uint32)

    @classmethod
    def load(cls, path: str) -> "RankTable":
        with open(path, "rb") as f:
            return cls.from_bytes(f.read(), what=path)

    @classmethod
    def from_bytes(cls, buf: bytes, what: str = "ranking") -> "RankTable":
        """Parse a ranking, e.g. the bytes `pgtoken.vocabulary_codec(name)` returns."""
        buf = bytes(buf)
        if buf[:4] != b"TNTT":
            raise ValueError(f"{what}: bad magic, expected TNTT")
        if buf[4] != 1:
            raise ValueError(f"{what}: unsupported table version {buf[4]}")
        if buf[5] != 1:
            raise ValueError(f"{what}: unsupported table kind {buf[5]}")
        k = int.from_bytes(buf[8:12], "little")
        if len(buf) - 12 != k * 4:
            raise ValueError(f"{what}: payload is {len(buf) - 12} bytes, expected {k * 4}")
        return cls(np.frombuffer(buf, dtype=np.uint32, count=k, offset=12))

    def ranks(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.uint32)
        if len(ids) == 0:
            return np.zeros(0, dtype=np.uint32)
        pos = np.searchsorted(self._sorted_tokens, ids)
        pos_clipped = np.minimum(pos, len(self._sorted_tokens) - 1)
        hit = self._sorted_tokens[pos_clipped] == ids
        # Ranked tokens take their rank; everything else takes the k + id fallback.
        return np.where(
            hit, self._sorted_ranks[pos_clipped], ids.astype(np.int64) + self.k
        ).astype(np.uint32)

    def tokens(self, ranks: np.ndarray) -> np.ndarray:
        ranks = ranks.astype(np.int64)
        out = np.where(ranks < self.k, 0, ranks - self.k).astype(np.uint32)
        inside = ranks < self.k
        out[inside] = self.token_of_rank[ranks[inside]]
        return out


# ── LEB128 ──────────────────────────────────────────────────────────────────────────────
# 7 bits per byte, low bits first, high bit set on every byte but the last. Matches
# core/src/codec/freq.rs, including its refusal of non-minimal and over-32-bit varints.


def leb_encode(values: np.ndarray) -> bytes:
    v = np.asarray(values, dtype=np.uint64)
    if len(v) == 0:
        return b""
    lens = 1 + sum((v >= (1 << (7 * k))).astype(np.int64) for k in range(1, 5))
    groups = (v[:, None] >> (7 * np.arange(5, dtype=np.uint64))[None, :]) & 0x7F
    cont = np.arange(5)[None, :] < (lens[:, None] - 1)
    b = (groups | (cont.astype(np.uint64) << 7)).astype(np.uint8)
    keep = np.arange(5)[None, :] < lens[:, None]
    return b[keep].tobytes()


def leb_decode(payload: bytes) -> np.ndarray:
    b = np.frombuffer(payload, dtype=np.uint8)
    if len(b) == 0:
        return np.zeros(0, dtype=np.uint32)
    if b[-1] & 0x80:
        raise ValueError("freq payload ends inside a varint")
    ends = np.flatnonzero((b & 0x80) == 0)
    starts = np.concatenate([[0], ends[:-1] + 1])
    lens = ends - starts + 1
    if (lens > 5).any():
        raise ValueError("freq payload has a varint larger than 32 bits")
    if ((lens > 1) & (b[ends] == 0)).any():
        raise ValueError("freq payload has a non-minimal varint")
    idx = starts[:, None] + np.arange(5)[None, :]
    mask = np.arange(5)[None, :] < lens[:, None]
    got = (b[np.minimum(idx, len(b) - 1)].astype(np.uint64) & 0x7F) * mask
    out = (got << (7 * np.arange(5, dtype=np.uint64))[None, :]).sum(axis=1)
    if (out > 0xFFFFFFFF).any():
        raise ValueError("freq payload has a varint larger than 32 bits")
    return out.astype(np.uint32)


def _read_varint(blob: bytes, off: int) -> tuple[int, int]:
    v = 0
    for i in range(5):
        if off + i >= len(blob):
            raise ValueError("token count ends inside a varint")
        b = blob[off + i]
        if i == 4 and b > 0x0F:
            raise ValueError("token count is larger than 32 bits")
        v |= (b & 0x7F) << (7 * i)
        if not b & 0x80:
            if b == 0 and i > 0:
                raise ValueError("token count is not minimally encoded")
            return v, off + i + 1
    raise AssertionError("unreachable")


# ── values ──────────────────────────────────────────────────────────────────────────────


def parse_header(blob: bytes) -> tuple[int, int, int, int]:
    """Return (codec, vocabulary_id, n_tokens, payload offset). Validates like the extension."""
    if not blob:
        raise ValueError("value is empty")
    tag = blob[0]
    if tag == LEGACY_MAGIC:
        raise ValueError("unknown value format: tag 0xA7 is the pre-release 12-byte header")
    if tag >> 4 != VERSION or (tag & 0x0F) not in CODEC_NAMES:
        raise ValueError(f"unknown value format tag 0x{tag:02X}")
    if len(blob) < 5:
        raise ValueError(f"value is {len(blob)} bytes, shorter than the 5-byte minimum header")
    if blob[1]:
        raise ValueError(f"value has flags 0x{blob[1]:02X}, which this client does not understand")
    codec = tag & 0x0F
    n, off = _read_varint(blob, 4)
    if codec != FREQ and len(blob) - off != n * {RAW8: 1, RAW16: 2, RAW24: 3}[codec]:
        raise ValueError("payload length disagrees with the token count")
    return codec, int.from_bytes(blob[2:4], "little"), n, off


def token_count(blob: bytes) -> int:
    """O(1): reads the header only."""
    return parse_header(blob)[2]


def _varint(n: int) -> bytes:
    out = bytearray()
    while n >= 0x80:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)


def encode(ids, codec: str, vocabulary_id: int = 0, table: RankTable | None = None) -> bytes:
    """Encode `ids` under `codec`.

    `codec` is never guessed here: a vocabulary's declared `vocab_size` is what picks a width on
    the extension side, and this client only ever mirrors that choice, never makes its own.
    """
    v = np.asarray(ids, dtype=np.uint32)
    cid = CODEC_IDS[codec]

    head = bytes([(VERSION << 4) | cid, 0]) + vocabulary_id.to_bytes(2, "little") + _varint(len(v))

    if cid == RAW8:
        if len(v) and v.max() > 0xFF:
            raise ValueError(f"token id {v.max()} does not fit raw8")
        return head + v.astype(np.uint8).tobytes()
    if cid == RAW16:
        if len(v) and v.max() > 0xFFFF:
            raise ValueError(f"token id {v.max()} does not fit raw16")
        return head + v.astype("<u2").tobytes()
    if cid == RAW24:
        if len(v) and v.max() > 0xFFFFFF:
            raise ValueError(f"token id {v.max()} does not fit raw24")
        b = np.zeros((len(v), 3), dtype=np.uint8)
        b[:, 0] = (v >> 16) & 0xFF  # big-endian, matching the extension
        b[:, 1] = (v >> 8) & 0xFF
        b[:, 2] = v & 0xFF
        return head + b.tobytes()
    if table is None:
        raise ValueError("freq needs a coding table")
    return head + leb_encode(table.ranks(v))


def decode(blob: bytes, table: RankTable | None = None) -> np.ndarray:
    codec, _vocabulary_id, n, off = parse_header(blob)
    payload = blob[off:]
    if codec == RAW8:
        return np.frombuffer(payload, dtype=np.uint8, count=n).astype(np.uint32)
    if codec == RAW16:
        return np.frombuffer(payload, dtype="<u2", count=n).astype(np.uint32)
    if codec == RAW24:
        b = np.frombuffer(payload, dtype=np.uint8, count=n * 3).reshape(n, 3).astype(np.uint32)
        return (b[:, 0] << 16) | (b[:, 1] << 8) | b[:, 2]
    if table is None:
        raise ValueError("freq needs a coding table")
    ranks = leb_decode(payload)
    if len(ranks) != n:
        raise ValueError(f"freq payload holds {len(ranks)} tokens but the header says {n}")
    return table.tokens(ranks)
