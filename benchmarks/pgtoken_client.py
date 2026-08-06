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
  0     1  magic 0xA7
  1     1  version (1)
  2     1  codec  0=raw16 1=raw24 2=freq 3=raw8
  3     1  reserved
  4     2  vocabulary id (u16 LE)
  6     2  reserved
  8     4  token count (u32 LE)
 12     -  payload
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

MAGIC = 0xA7
VERSION = 1
HEADER_LEN = 12

RAW16, RAW24, FREQ, RAW8 = 0, 1, 2, 3
CODEC_NAMES = {RAW16: "raw16", RAW24: "raw24", FREQ: "freq", RAW8: "raw8"}
CODEC_IDS = {v: k for k, v in CODEC_NAMES.items()}

_POW256 = np.array([1, 256, 65536, 16777216], dtype=np.uint64)


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
            buf = f.read()
        if buf[:4] != b"TNTT":
            raise ValueError(f"{path}: bad magic, expected TNTT")
        if buf[4] != 1:
            raise ValueError(f"{path}: unsupported table version {buf[4]}")
        if buf[5] != 1:
            raise ValueError(f"{path}: unsupported table kind {buf[5]}")
        k = int.from_bytes(buf[8:12], "little")
        if len(buf) - 12 != k * 4:
            raise ValueError(f"{path}: payload is {len(buf) - 12} bytes, expected {k * 4}")
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


# ── Stream VByte ────────────────────────────────────────────────────────────────────────
# ceil(n/4) control bytes, each packing four 2-bit (length-1) fields, then the data bytes
# little-endian. Matches the `stream-vbyte` crate the extension uses.


def svb_encode(values: np.ndarray) -> bytes:
    v = np.asarray(values, dtype=np.uint32)
    n = len(v)
    if n == 0:
        return b""
    lens = (1 + (v >= 256) + (v >= 65536) + (v >= 16777216)).astype(np.int64)

    nctrl = (n + 3) // 4
    fields = (lens - 1).astype(np.uint8)
    padded = np.zeros(nctrl * 4, dtype=np.uint8)
    padded[:n] = fields
    grouped = padded.reshape(nctrl, 4).astype(np.uint16)
    ctrl = (grouped[:, 0] | (grouped[:, 1] << 2) | (grouped[:, 2] << 4) | (grouped[:, 3] << 6))
    ctrl = ctrl.astype(np.uint8)

    all_bytes = ((v[:, None].astype(np.uint32) >> (8 * np.arange(4, dtype=np.uint32))) & 0xFF)
    keep = np.arange(4)[None, :] < lens[:, None]
    data = all_bytes.astype(np.uint8)[keep]
    return ctrl.tobytes() + data.tobytes()


def svb_decode(payload: bytes, n: int) -> np.ndarray:
    if n == 0:
        return np.zeros(0, dtype=np.uint32)
    nctrl = (n + 3) // 4
    ctrl = np.frombuffer(payload, dtype=np.uint8, count=nctrl)
    data = np.frombuffer(payload, dtype=np.uint8, offset=nctrl)

    shifts = (2 * (np.arange(n) % 4)).astype(np.uint8)
    lens = ((np.repeat(ctrl, 4)[:n] >> shifts) & 0b11).astype(np.int64) + 1
    ends = np.cumsum(lens)
    starts = ends - lens

    # Pad so the fixed 4-wide gather cannot run off the end on the final value.
    padded = np.concatenate([data, np.zeros(4, dtype=np.uint8)])
    idx = starts[:, None] + np.arange(4)[None, :]
    mask = np.arange(4)[None, :] < lens[:, None]
    got = padded[idx] * mask
    return (got.astype(np.uint64) @ _POW256).astype(np.uint32)


# ── values ──────────────────────────────────────────────────────────────────────────────


def parse_header(blob: bytes) -> tuple[int, int, int]:
    """Return (codec, vocabulary_id, n_tokens). Validates like the extension does."""
    if len(blob) < HEADER_LEN:
        raise ValueError(f"value is {len(blob)} bytes, shorter than the {HEADER_LEN}-byte header")
    if blob[0] != MAGIC:
        raise ValueError(f"bad magic byte 0x{blob[0]:02X}")
    if blob[1] != VERSION:
        raise ValueError(f"unsupported format version {blob[1]}")
    codec = blob[2]
    if codec not in CODEC_NAMES:
        raise ValueError(f"unknown codec id {codec}")
    if blob[3] or blob[6] or blob[7]:
        raise ValueError("reserved header bytes are not zero")
    return codec, int.from_bytes(blob[4:6], "little"), int.from_bytes(blob[8:12], "little")


def token_count(blob: bytes) -> int:
    """O(1): reads the header only."""
    return parse_header(blob)[2]


def encode(ids, codec: str, vocabulary_id: int = 0, table: RankTable | None = None) -> bytes:
    """Encode `ids` under `codec`.

    `codec` is never guessed here: a vocabulary's declared `vocab_size` is what picks a width on
    the extension side, and this client only ever mirrors that choice, never makes its own.
    """
    v = np.asarray(ids, dtype=np.uint32)
    cid = CODEC_IDS[codec]

    head = bytes([MAGIC, VERSION, cid, 0]) + vocabulary_id.to_bytes(2, "little") + b"\x00\x00"
    head += len(v).to_bytes(4, "little")

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
    return head + svb_encode(table.ranks(v))


def decode(blob: bytes, table: RankTable | None = None) -> np.ndarray:
    codec, _vocabulary_id, n = parse_header(blob)
    payload = blob[HEADER_LEN:]
    if codec == RAW8:
        return np.frombuffer(payload, dtype=np.uint8, count=n).astype(np.uint32)
    if codec == RAW16:
        return np.frombuffer(payload, dtype="<u2", count=n).astype(np.uint32)
    if codec == RAW24:
        b = np.frombuffer(payload, dtype=np.uint8, count=n * 3).reshape(n, 3).astype(np.uint32)
        return (b[:, 0] << 16) | (b[:, 1] << 8) | b[:, 2]
    if table is None:
        raise ValueError("freq needs a coding table")
    return table.tokens(svb_decode(payload, n))
