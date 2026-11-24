# Compression ratio vs value length

ratio = UTF-8 bytes / stored bytes (per-value headers and frame headers included; dictionaries, symbol tables and rank tables counted once per column, not per value). Cells are the median ratio over the values in the bucket; n is the number of values.

## c4 / cut

| method | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n values | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 8000 |
| median o200k tokens | 16 | 32 | 64 | 128 | 255 | 511 | 1022 | 2043 | 190 |
| utf8 (no compression) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| PostgreSQL text | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.26 | 1.33 | 1.41 | 1.00 |
| lz4 | 0.93 | 0.99 | 1.05 | 1.12 | 1.18 | 1.26 | 1.33 | 1.41 | 1.16 |
| zstd-3 | 1.03 | 1.26 | 1.45 | 1.61 | 1.78 | 1.93 | 2.05 | 2.14 | 1.71 |
| zstd-3 + dict (text) | 1.54 | 1.86 | 2.10 | 2.26 | 2.37 | 2.46 | 2.51 | 2.52 | 2.30 |
| FSST | 1.74 | 1.78 | 1.79 | 1.81 | 1.81 | 1.82 | 1.82 | 1.82 | 1.81 |
| Schmidt 16-bit (cl100k patched) | 1.91 | 1.92 | 1.92 | 1.92 | 1.92 | 1.93 | 1.92 | 1.92 | 1.92 |
| raw24 [o200k] | 1.43 | 1.51 | 1.54 | 1.57 | 1.59 | 1.60 | 1.60 | 1.60 | 1.57 |
| zstd-3 + dict (ID bytes) [o200k] | 1.72 | 2.08 | 2.35 | 2.51 | 2.63 | 2.70 | 2.74 | 2.77 | 2.54 |
| freq [o200k] | 2.33 | 2.62 | 2.75 | 2.83 | 2.88 | 2.92 | 2.92 | 2.93 | 2.83 |
| freq+lz4 (pgtoken) [o200k] | 2.33 | 2.63 | 2.78 | 2.89 | 2.95 | 3.00 | 3.03 | 3.02 | 2.89 |
| raw24 [qwen3] | 1.40 | 1.47 | 1.50 | 1.53 | 1.54 | 1.56 | 1.55 | 1.55 | 1.53 |
| zstd-3 + dict (ID bytes) [qwen3] | 1.73 | 2.08 | 2.34 | 2.51 | 2.63 | 2.70 | 2.75 | 2.77 | 2.54 |
| freq [qwen3] | 2.31 | 2.58 | 2.71 | 2.79 | 2.84 | 2.88 | 2.89 | 2.90 | 2.79 |
| freq+lz4 (pgtoken) [qwen3] | 2.31 | 2.59 | 2.74 | 2.86 | 2.92 | 2.98 | 3.00 | 3.00 | 2.87 |

Decode to o200k IDs, median ns per token (single core; text = decompress + HF tokenize):

| method | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| utf8 (no compression) | 134 | 87 | 60 | 45 | 39 | 36 | 35 | 35 | 43 |
| PostgreSQL text | 134 | 87 | 60 | 45 | 39 | 38 | 36 | 36 | 44 |
| lz4 | 146 | 93 | 63 | 48 | 41 | 38 | 36 | 36 | 46 |
| zstd-3 | 221 | 144 | 95 | 64 | 55 | 48 | 44 | 43 | 61 |
| zstd-3 + dict (text) | 164 | 106 | 73 | 56 | 48 | 45 | 43 | 42 | 53 |
| FSST | 122 | 82 | 57 | 44 | 38 | 36 | 34 | 34 | 42 |
| raw24 [o200k] | 82 | 41 | 21 | 11 | 6 | 3 | 2 | 1 | 10 |
| zstd-3 + dict (ID bytes) [o200k] | 342 | 177 | 94 | 52 | 31 | 21 | 12 | 9 | 51 |
| freq+lz4 (pgtoken) [o200k] | 82 | 43 | 23 | 13 | 8 | 6 | 5 | 5 | 13 |

## c4 / doc

| method | 1-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | 1024-2047 | 2048+ | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n values | 11 | 253 | 1728 | 3834 | 4624 | 4618 | 3247 | 1273 | 513 | 20101 |
| median o200k tokens | 13 | 27 | 50 | 93 | 182 | 363 | 682 | 1306 | 2856 | 241 |
| utf8 (no compression) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| PostgreSQL text | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.34 | 1.43 | 1.54 | 1.00 |
| lz4 | 0.88 | 0.96 | 1.03 | 1.09 | 1.18 | 1.26 | 1.34 | 1.43 | 1.54 | 1.22 |
| zstd-3 | 0.90 | 1.13 | 1.38 | 1.53 | 1.72 | 1.91 | 2.06 | 2.19 | 2.35 | 1.81 |
| zstd-3 + dict (text) | 1.16 | 1.60 | 1.96 | 2.18 | 2.35 | 2.48 | 2.55 | 2.60 | 2.66 | 2.39 |
| FSST | 1.63 | 1.64 | 1.76 | 1.80 | 1.81 | 1.82 | 1.84 | 1.84 | 1.83 | 1.82 |
| Schmidt 16-bit (cl100k patched) | 1.71 | 1.81 | 1.88 | 1.91 | 1.92 | 1.93 | 1.93 | 1.93 | 1.92 | 1.92 |
| raw24 [o200k] | 1.03 | 1.35 | 1.48 | 1.54 | 1.58 | 1.60 | 1.60 | 1.59 | 1.57 | 1.57 |
| zstd-3 + dict (ID bytes) [o200k] | 1.26 | 1.77 | 2.18 | 2.42 | 2.60 | 2.73 | 2.79 | 2.80 | 2.87 | 2.63 |
| freq [o200k] | 1.53 | 2.24 | 2.57 | 2.76 | 2.86 | 2.93 | 2.98 | 2.97 | 2.94 | 2.87 |
| freq+lz4 (pgtoken) [o200k] | 1.53 | 2.26 | 2.61 | 2.82 | 2.94 | 3.04 | 3.08 | 3.09 | 3.12 | 2.96 |
| raw24 [qwen3] | 0.90 | 1.31 | 1.43 | 1.50 | 1.53 | 1.55 | 1.56 | 1.56 | 1.53 | 1.53 |
| zstd-3 + dict (ID bytes) [qwen3] | 1.20 | 1.75 | 2.17 | 2.41 | 2.60 | 2.74 | 2.80 | 2.81 | 2.89 | 2.63 |
| freq [qwen3] | 1.55 | 2.20 | 2.54 | 2.72 | 2.82 | 2.90 | 2.94 | 2.94 | 2.91 | 2.84 |
| freq+lz4 (pgtoken) [qwen3] | 1.55 | 2.23 | 2.57 | 2.79 | 2.91 | 3.01 | 3.06 | 3.06 | 3.11 | 2.93 |

Decode to o200k IDs, median ns per token (single core; text = decompress + HF tokenize):

| method | 1-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | 1024-2047 | 2048+ | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| utf8 (no compression) | 259 | 102 | 65 | 50 | 42 | 38 | 35 | 35 | 35 | 40 |
| PostgreSQL text | 259 | 102 | 65 | 50 | 42 | 38 | 37 | 36 | 37 | 40 |
| lz4 | 279 | 109 | 70 | 53 | 44 | 40 | 37 | 36 | 37 | 42 |
| zstd-3 | 290 | 157 | 109 | 73 | 58 | 51 | 46 | 44 | 43 | 55 |
| zstd-3 + dict (text) | 302 | 120 | 80 | 61 | 51 | 46 | 43 | 42 | 43 | 49 |
| FSST | 240 | 96 | 62 | 48 | 41 | 37 | 35 | 35 | 35 | 39 |
| raw24 [o200k] | 154 | 49 | 27 | 15 | 8 | 5 | 3 | 2 | 1 | 6 |
| zstd-3 + dict (ID bytes) [o200k] | 592 | 213 | 118 | 70 | 41 | 25 | 17 | 11 | 8 | 32 |
| freq+lz4 (pgtoken) [o200k] | 142 | 50 | 29 | 17 | 10 | 7 | 5 | 5 | 5 | 9 |

## code / cut

| method | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n values | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 787 | 393 | 7180 |
| median o200k tokens | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 128 |
| utf8 (no compression) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| PostgreSQL text | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.71 | 2.18 | 2.30 | 1.00 |
| lz4 | 0.97 | 1.14 | 1.33 | 1.55 | 1.75 | 2.00 | 2.19 | 2.30 | 1.55 |
| zstd-3 | 1.00 | 1.31 | 1.61 | 1.98 | 2.34 | 2.78 | 3.16 | 3.40 | 2.01 |
| zstd-3 + dict (text) | 1.52 | 1.89 | 2.21 | 2.58 | 2.92 | 3.30 | 3.61 | 3.86 | 2.62 |
| FSST | 1.60 | 1.66 | 1.66 | 1.69 | 1.69 | 1.69 | 1.68 | 1.68 | 1.67 |
| Schmidt 16-bit (cl100k patched) | 1.79 | 1.77 | 1.78 | 1.80 | 1.80 | 1.80 | 1.80 | 1.80 | 1.79 |
| raw24 [o200k] | 1.25 | 1.32 | 1.34 | 1.39 | 1.40 | 1.40 | 1.40 | 1.42 | 1.37 |
| zstd-3 + dict (ID bytes) [o200k] | 1.61 | 1.97 | 2.31 | 2.73 | 3.07 | 3.45 | 3.69 | 3.83 | 2.72 |
| freq [o200k] | 2.12 | 2.31 | 2.41 | 2.52 | 2.56 | 2.57 | 2.58 | 2.58 | 2.46 |
| freq+lz4 (pgtoken) [o200k] | 2.12 | 2.35 | 2.53 | 2.79 | 3.07 | 3.35 | 3.56 | 3.74 | 2.83 |
| freq+lz4, C4 table [o200k] | 1.50 | 1.66 | 1.88 | 2.15 | 2.41 | 2.69 | 2.87 | 3.06 | 2.18 |
| raw24 [qwen3] | 1.26 | 1.32 | 1.34 | 1.39 | 1.40 | 1.41 | 1.41 | 1.41 | 1.37 |
| zstd-3 + dict (ID bytes) [qwen3] | 1.63 | 1.99 | 2.37 | 2.80 | 3.14 | 3.54 | 3.76 | 3.90 | 2.77 |
| freq [qwen3] | 2.13 | 2.33 | 2.42 | 2.54 | 2.58 | 2.58 | 2.60 | 2.60 | 2.48 |
| freq+lz4 (pgtoken) [qwen3] | 2.13 | 2.38 | 2.55 | 2.82 | 3.09 | 3.38 | 3.58 | 3.76 | 2.85 |
| freq+lz4, C4 table [qwen3] | 1.52 | 1.69 | 1.93 | 2.19 | 2.44 | 2.72 | 2.91 | 3.08 | 2.21 |

Decode to o200k IDs, median ns per token (single core; text = decompress + HF tokenize):

| method | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| utf8 (no compression) | 138 | 86 | 58 | 44 | 36 | 33 | 31 | 31 | 43 |
| PostgreSQL text | 138 | 86 | 58 | 44 | 36 | 34 | 33 | 33 | 43 |
| lz4 | 150 | 93 | 61 | 46 | 38 | 34 | 33 | 33 | 46 |
| zstd-3 | 163 | 134 | 87 | 61 | 49 | 41 | 37 | 36 | 59 |
| zstd-3 + dict (text) | 164 | 103 | 69 | 52 | 43 | 38 | 36 | 36 | 51 |
| FSST | 128 | 81 | 56 | 43 | 36 | 33 | 32 | 32 | 43 |
| raw24 [o200k] | 81 | 40 | 21 | 11 | 6 | 3 | 2 | 1 | 11 |
| zstd-3 + dict (ID bytes) [o200k] | 339 | 176 | 92 | 51 | 29 | 19 | 13 | 9 | 50 |
| freq+lz4 (pgtoken) [o200k] | 80 | 42 | 23 | 13 | 9 | 6 | 5 | 5 | 13 |

## dbpedia / abstract

| method | 1-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n values | 54 | 779 | 5290 | 17722 | 1348 | 70 | 3 | 25266 |
| median o200k tokens | 13 | 27 | 49 | 94 | 136 | 300 | 591 | 88 |
| utf8 (no compression) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| PostgreSQL text | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| lz4 | 1.11 | 1.04 | 1.08 | 1.13 | 1.14 | 1.20 | 1.29 | 1.12 |
| zstd-3 | 1.10 | 1.21 | 1.38 | 1.52 | 1.46 | 1.68 | 1.86 | 1.49 |
| zstd-3 + dict (text) | 1.75 | 2.27 | 2.44 | 2.33 | 2.10 | 2.46 | 2.06 | 2.33 |
| FSST | 1.65 | 1.74 | 1.79 | 1.77 | 1.60 | 1.60 | 1.29 | 1.77 |
| Schmidt 16-bit (cl100k patched) | 1.85 | 1.75 | 1.74 | 1.80 | 1.57 | 1.72 | 1.34 | 1.77 |
| raw24 [o200k] | 1.42 | 1.33 | 1.33 | 1.41 | 1.19 | 1.33 | 0.85 | 1.38 |
| zstd-3 + dict (ID bytes) [o200k] | 1.92 | 2.45 | 2.62 | 2.49 | 2.16 | 2.54 | 1.94 | 2.49 |
| freq [o200k] | 2.32 | 2.50 | 2.62 | 2.65 | 2.16 | 2.36 | 1.58 | 2.61 |
| freq+lz4 (pgtoken) [o200k] | 2.33 | 2.52 | 2.65 | 2.70 | 2.24 | 2.42 | 1.82 | 2.66 |
| freq+lz4, C4 table [o200k] | 2.08 | 2.20 | 2.33 | 2.49 | 2.00 | 2.26 | 1.71 | 2.43 |
| raw24 [qwen3] | 1.41 | 1.27 | 1.22 | 1.33 | 1.13 | 1.23 | 0.78 | 1.30 |
| zstd-3 + dict (ID bytes) [qwen3] | 1.89 | 2.39 | 2.59 | 2.48 | 2.14 | 2.53 | 2.03 | 2.48 |
| freq [qwen3] | 2.32 | 2.42 | 2.53 | 2.60 | 2.13 | 2.32 | 1.56 | 2.55 |
| freq+lz4 (pgtoken) [qwen3] | 2.33 | 2.44 | 2.57 | 2.65 | 2.21 | 2.36 | 1.88 | 2.61 |
| freq+lz4, C4 table [qwen3] | 2.13 | 2.21 | 2.30 | 2.47 | 2.01 | 2.26 | 1.77 | 2.41 |

Decode to o200k IDs, median ns per token (single core; text = decompress + HF tokenize):

| method | 1-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| utf8 (no compression) | 177 | 94 | 67 | 50 | 44 | 37 |  | 52 |
| PostgreSQL text | 177 | 94 | 67 | 50 | 44 | 37 |  | 52 |
| lz4 | 187 | 101 | 72 | 53 | 46 | 39 |  | 55 |
| zstd-3 | 196 | 152 | 109 | 74 | 63 | 51 |  | 78 |
| zstd-3 + dict (text) | 202 | 113 | 80 | 60 | 52 | 44 |  | 62 |
| FSST | 165 | 88 | 64 | 49 | 42 | 36 |  | 50 |
| raw24 [o200k] | 98 | 48 | 28 | 15 | 10 | 5 |  | 16 |
| zstd-3 + dict (ID bytes) [o200k] | 392 | 201 | 121 | 69 | 50 | 27 |  | 73 |
| freq+lz4 (pgtoken) [o200k] | 95 | 49 | 30 | 17 | 13 | 8 |  | 18 |

## dbpedia / cut

| method | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n values | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 8000 |
| median o200k tokens | 16 | 32 | 64 | 127 | 254 | 507 | 1013 | 2025 | 188 |
| utf8 (no compression) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| PostgreSQL text | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.24 | 1.31 | 1.38 | 1.00 |
| lz4 | 1.04 | 1.06 | 1.10 | 1.16 | 1.20 | 1.26 | 1.31 | 1.38 | 1.20 |
| zstd-3 | 1.03 | 1.25 | 1.45 | 1.58 | 1.73 | 1.87 | 1.97 | 2.06 | 1.68 |
| zstd-3 + dict (text) | 1.59 | 2.04 | 2.26 | 2.37 | 2.45 | 2.52 | 2.52 | 2.54 | 2.41 |
| FSST | 1.59 | 1.71 | 1.74 | 1.74 | 1.73 | 1.73 | 1.72 | 1.72 | 1.72 |
| Schmidt 16-bit (cl100k patched) | 1.65 | 1.70 | 1.76 | 1.76 | 1.76 | 1.77 | 1.77 | 1.77 | 1.75 |
| raw24 [o200k] | 1.13 | 1.25 | 1.36 | 1.38 | 1.40 | 1.42 | 1.42 | 1.42 | 1.39 |
| zstd-3 + dict (ID bytes) [o200k] | 1.64 | 2.15 | 2.39 | 2.53 | 2.62 | 2.68 | 2.67 | 2.69 | 2.55 |
| freq [o200k] | 1.87 | 2.35 | 2.52 | 2.57 | 2.61 | 2.64 | 2.63 | 2.63 | 2.56 |
| freq+lz4 (pgtoken) [o200k] | 1.94 | 2.40 | 2.57 | 2.64 | 2.69 | 2.74 | 2.73 | 2.74 | 2.64 |
| freq+lz4, C4 table [o200k] | 1.71 | 2.13 | 2.36 | 2.44 | 2.51 | 2.58 | 2.57 | 2.59 | 2.44 |
| raw24 [qwen3] | 1.08 | 1.17 | 1.28 | 1.30 | 1.32 | 1.34 | 1.34 | 1.35 | 1.31 |
| zstd-3 + dict (ID bytes) [qwen3] | 1.62 | 2.14 | 2.37 | 2.51 | 2.61 | 2.68 | 2.67 | 2.68 | 2.53 |
| freq [qwen3] | 1.84 | 2.30 | 2.46 | 2.51 | 2.55 | 2.59 | 2.58 | 2.58 | 2.51 |
| freq+lz4 (pgtoken) [qwen3] | 1.93 | 2.35 | 2.53 | 2.60 | 2.66 | 2.70 | 2.70 | 2.71 | 2.60 |
| freq+lz4, C4 table [qwen3] | 1.73 | 2.12 | 2.34 | 2.43 | 2.51 | 2.56 | 2.56 | 2.57 | 2.44 |

Decode to o200k IDs, median ns per token (single core; text = decompress + HF tokenize):

| method | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| utf8 (no compression) | 141 | 88 | 59 | 45 | 38 | 35 | 34 | 34 | 41 |
| PostgreSQL text | 141 | 88 | 59 | 45 | 38 | 36 | 35 | 35 | 41 |
| lz4 | 153 | 94 | 63 | 47 | 40 | 36 | 35 | 35 | 43 |
| zstd-3 | 163 | 140 | 93 | 64 | 52 | 45 | 42 | 41 | 56 |
| zstd-3 + dict (text) | 168 | 104 | 70 | 54 | 46 | 42 | 40 | 39 | 48 |
| FSST | 129 | 82 | 57 | 43 | 37 | 34 | 33 | 34 | 40 |
| raw24 [o200k] | 82 | 41 | 21 | 11 | 6 | 3 | 2 | 1 | 6 |
| zstd-3 + dict (ID bytes) [o200k] | 339 | 176 | 93 | 52 | 30 | 20 | 12 | 9 | 31 |
| freq+lz4 (pgtoken) [o200k] | 84 | 43 | 23 | 13 | 9 | 6 | 5 | 4 | 9 |

## wildchat / assistant

| method | 1-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | 1024-2047 | 2048+ | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n values | 2505 | 2446 | 4244 | 8350 | 15114 | 16927 | 6849 | 1133 | 80 | 57648 |
| median o200k tokens | 9 | 23 | 47 | 96 | 192 | 344 | 655 | 1214 | 2392 | 223 |
| utf8 (no compression) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| PostgreSQL text | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.49 | 1.74 | 2.25 | 1.00 |
| lz4 | 0.85 | 0.95 | 1.02 | 1.12 | 1.27 | 1.40 | 1.52 | 1.74 | 2.25 | 1.28 |
| zstd-3 | 0.87 | 1.07 | 1.33 | 1.51 | 1.72 | 1.96 | 2.20 | 2.51 | 3.21 | 1.73 |
| zstd-3 + dict (text) | 1.18 | 1.53 | 1.79 | 1.89 | 2.09 | 2.30 | 2.47 | 2.73 | 3.35 | 2.14 |
| FSST | 1.22 | 1.32 | 1.36 | 1.36 | 1.34 | 1.34 | 1.38 | 1.36 | 1.33 | 1.35 |
| Schmidt 16-bit (cl100k patched) | 1.60 | 1.69 | 1.73 | 1.73 | 1.69 | 1.72 | 1.78 | 1.76 | 1.66 | 1.72 |
| raw24 [o200k] | 1.06 | 1.38 | 1.49 | 1.50 | 1.45 | 1.44 | 1.46 | 1.42 | 1.28 | 1.45 |
| zstd-3 + dict (ID bytes) [o200k] | 1.43 | 1.81 | 2.14 | 2.23 | 2.40 | 2.60 | 2.76 | 3.02 | 3.43 | 2.45 |
| freq [o200k] | 1.76 | 2.22 | 2.43 | 2.40 | 2.36 | 2.37 | 2.39 | 2.32 | 2.05 | 2.35 |
| freq+lz4 (pgtoken) [o200k] | 1.76 | 2.22 | 2.44 | 2.48 | 2.57 | 2.69 | 2.79 | 2.91 | 3.26 | 2.58 |
| freq+lz4, C4 table [o200k] | 1.54 | 1.91 | 2.15 | 2.08 | 2.04 | 2.23 | 2.59 | 2.69 | 2.88 | 2.18 |
| raw24 [qwen3] | 1.06 | 1.40 | 1.53 | 1.56 | 1.54 | 1.53 | 1.47 | 1.39 | 1.21 | 1.51 |
| zstd-3 + dict (ID bytes) [qwen3] | 1.42 | 1.82 | 2.16 | 2.27 | 2.45 | 2.62 | 2.76 | 3.02 | 3.51 | 2.47 |
| freq [qwen3] | 1.76 | 2.29 | 2.54 | 2.55 | 2.51 | 2.51 | 2.44 | 2.35 | 2.02 | 2.47 |
| freq+lz4 (pgtoken) [qwen3] | 1.76 | 2.29 | 2.55 | 2.62 | 2.69 | 2.78 | 2.80 | 2.93 | 3.27 | 2.68 |
| freq+lz4, C4 table [qwen3] | 1.55 | 1.86 | 2.03 | 2.03 | 2.09 | 2.24 | 2.56 | 2.71 | 2.95 | 2.16 |

Decode to o200k IDs, median ns per token (single core; text = decompress + HF tokenize):

| method | 1-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | 1024-2047 | 2048+ | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| utf8 (no compression) | 220 | 108 | 71 | 52 | 42 | 38 | 35 | 33 | 31 | 42 |
| PostgreSQL text | 220 | 108 | 71 | 52 | 42 | 38 | 36 | 34 | 32 | 42 |
| lz4 | 242 | 117 | 76 | 55 | 44 | 40 | 36 | 34 | 32 | 44 |
| zstd-3 | 250 | 156 | 112 | 77 | 58 | 51 | 44 | 40 | 36 | 57 |
| zstd-3 + dict (text) | 262 | 130 | 86 | 64 | 52 | 46 | 42 | 39 | 36 | 51 |
| FSST | 198 | 98 | 66 | 50 | 41 | 38 | 35 | 34 | 31 | 41 |
| raw24 [o200k] | 141 | 56 | 28 | 14 | 7 | 5 | 3 | 2 | 1 | 6 |
| zstd-3 + dict (ID bytes) [o200k] | 600 | 238 | 127 | 69 | 39 | 26 | 17 | 11 | 8 | 34 |
| freq+lz4 (pgtoken) [o200k] | 141 | 56 | 30 | 17 | 10 | 8 | 6 | 5 | 5 | 10 |

## wildchat / conversation

| method | 1-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | 1024-2047 | 2048+ | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n values | 134 | 284 | 591 | 1163 | 2081 | 3754 | 4755 | 3861 | 3399 | 20022 |
| median o200k tokens | 15 | 22 | 48 | 97 | 192 | 379 | 724 | 1364 | 3067 | 685 |
| utf8 (no compression) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| PostgreSQL text | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.51 | 1.69 | 2.04 | 1.44 |
| lz4 | 0.95 | 0.97 | 1.05 | 1.15 | 1.26 | 1.40 | 1.53 | 1.69 | 2.04 | 1.51 |
| zstd-3 | 0.92 | 0.96 | 1.34 | 1.56 | 1.75 | 2.00 | 2.21 | 2.45 | 2.94 | 2.17 |
| zstd-3 + dict (text) | 1.44 | 1.43 | 1.86 | 2.04 | 2.12 | 2.33 | 2.47 | 2.71 | 3.17 | 2.48 |
| FSST | 1.17 | 1.25 | 1.34 | 1.38 | 1.36 | 1.37 | 1.39 | 1.37 | 1.34 | 1.37 |
| Schmidt 16-bit (cl100k patched) | 1.66 | 1.58 | 1.72 | 1.75 | 1.71 | 1.73 | 1.76 | 1.76 | 1.72 | 1.74 |
| raw24 [o200k] | 1.10 | 1.24 | 1.42 | 1.49 | 1.45 | 1.47 | 1.49 | 1.46 | 1.41 | 1.45 |
| zstd-3 + dict (ID bytes) [o200k] | 1.83 | 1.72 | 2.11 | 2.37 | 2.42 | 2.64 | 2.80 | 2.96 | 3.34 | 2.76 |
| freq [o200k] | 1.83 | 1.99 | 2.34 | 2.42 | 2.34 | 2.40 | 2.43 | 2.40 | 2.31 | 2.37 |
| freq+lz4 (pgtoken) [o200k] | 1.83 | 2.00 | 2.38 | 2.54 | 2.56 | 2.73 | 2.86 | 2.96 | 3.31 | 2.84 |
| freq+lz4, C4 table [o200k] | 1.83 | 1.73 | 2.26 | 2.38 | 2.26 | 2.44 | 2.68 | 2.79 | 3.11 | 2.64 |
| raw24 [qwen3] | 1.10 | 1.25 | 1.40 | 1.48 | 1.48 | 1.53 | 1.51 | 1.49 | 1.45 | 1.48 |
| zstd-3 + dict (ID bytes) [qwen3] | 1.77 | 1.65 | 2.16 | 2.35 | 2.44 | 2.63 | 2.78 | 2.96 | 3.36 | 2.76 |
| freq [qwen3] | 1.83 | 2.02 | 2.34 | 2.48 | 2.44 | 2.51 | 2.50 | 2.46 | 2.40 | 2.45 |
| freq+lz4 (pgtoken) [qwen3] | 1.83 | 2.03 | 2.38 | 2.59 | 2.63 | 2.78 | 2.86 | 2.99 | 3.33 | 2.86 |
| freq+lz4, C4 table [qwen3] | 1.77 | 1.68 | 2.09 | 2.18 | 2.16 | 2.34 | 2.61 | 2.77 | 3.09 | 2.58 |

Decode to o200k IDs, median ns per token (single core; text = decompress + HF tokenize):

| method | 1-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | 1024-2047 | 2048+ | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| utf8 (no compression) | 166 | 135 | 72 | 51 | 42 | 38 | 35 | 34 | 35 | 37 |
| PostgreSQL text | 166 | 135 | 72 | 51 | 42 | 38 | 36 | 35 | 36 | 37 |
| lz4 | 180 | 144 | 76 | 54 | 44 | 39 | 36 | 35 | 36 | 38 |
| zstd-3 | 182 | 157 | 101 | 74 | 58 | 50 | 44 | 41 | 41 | 46 |
| zstd-3 + dict (text) | 194 | 161 | 86 | 62 | 51 | 46 | 42 | 40 | 40 | 44 |
| FSST | 155 | 120 | 65 | 48 | 40 | 37 | 35 | 34 | 35 | 36 |
| raw24 [o200k] | 86 | 69 | 28 | 14 | 7 | 4 | 3 | 2 | 1 | 3 |
| zstd-3 + dict (ID bytes) [o200k] | 365 | 288 | 124 | 66 | 38 | 25 | 15 | 10 | 7 | 16 |
| freq+lz4 (pgtoken) [o200k] | 85 | 68 | 29 | 17 | 10 | 7 | 6 | 5 | 5 | 6 |

## wildchat / cut

| method | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n values | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 8000 |
| median o200k tokens | 16 | 32 | 64 | 128 | 256 | 512 | 1023 | 2046 | 190 |
| utf8 (no compression) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| PostgreSQL text | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.34 | 1.53 | 1.64 | 1.00 |
| lz4 | 0.92 | 0.99 | 1.08 | 1.19 | 1.29 | 1.41 | 1.53 | 1.64 | 1.24 |
| zstd-3 | 0.94 | 1.21 | 1.41 | 1.61 | 1.79 | 2.00 | 2.19 | 2.32 | 1.66 |
| zstd-3 + dict (text) | 1.32 | 1.59 | 1.81 | 2.02 | 2.15 | 2.32 | 2.48 | 2.61 | 2.07 |
| FSST | 1.35 | 1.35 | 1.36 | 1.36 | 1.36 | 1.35 | 1.36 | 1.36 | 1.36 |
| Schmidt 16-bit (cl100k patched) | 1.70 | 1.73 | 1.73 | 1.74 | 1.73 | 1.72 | 1.72 | 1.70 | 1.72 |
| raw24 [o200k] | 1.26 | 1.35 | 1.42 | 1.44 | 1.45 | 1.45 | 1.47 | 1.46 | 1.42 |
| zstd-3 + dict (ID bytes) [o200k] | 1.56 | 1.87 | 2.11 | 2.35 | 2.49 | 2.64 | 2.75 | 2.89 | 2.36 |
| freq [o200k] | 1.97 | 2.13 | 2.27 | 2.33 | 2.36 | 2.37 | 2.40 | 2.39 | 2.29 |
| freq+lz4 (pgtoken) [o200k] | 1.97 | 2.17 | 2.34 | 2.48 | 2.58 | 2.69 | 2.80 | 2.93 | 2.53 |
| freq+lz4, C4 table [o200k] | 1.82 | 1.97 | 2.05 | 2.20 | 2.26 | 2.36 | 2.55 | 2.63 | 2.25 |
| raw24 [qwen3] | 1.25 | 1.35 | 1.43 | 1.46 | 1.48 | 1.47 | 1.49 | 1.47 | 1.43 |
| zstd-3 + dict (ID bytes) [qwen3] | 1.52 | 1.85 | 2.10 | 2.34 | 2.48 | 2.62 | 2.76 | 2.88 | 2.37 |
| freq [qwen3] | 1.95 | 2.14 | 2.33 | 2.41 | 2.43 | 2.44 | 2.46 | 2.43 | 2.33 |
| freq+lz4 (pgtoken) [qwen3] | 1.95 | 2.16 | 2.37 | 2.53 | 2.62 | 2.72 | 2.83 | 2.93 | 2.54 |
| freq+lz4, C4 table [qwen3] | 1.73 | 1.87 | 1.97 | 2.13 | 2.18 | 2.34 | 2.51 | 2.62 | 2.22 |

Decode to o200k IDs, median ns per token (single core; text = decompress + HF tokenize):

| method | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| utf8 (no compression) | 144 | 89 | 60 | 46 | 39 | 36 | 34 | 35 | 43 |
| PostgreSQL text | 144 | 89 | 60 | 46 | 39 | 37 | 36 | 36 | 43 |
| lz4 | 157 | 96 | 64 | 48 | 41 | 37 | 36 | 36 | 45 |
| zstd-3 | 180 | 137 | 90 | 65 | 53 | 46 | 42 | 41 | 58 |
| zstd-3 + dict (text) | 171 | 108 | 73 | 56 | 48 | 44 | 41 | 40 | 52 |
| FSST | 130 | 82 | 56 | 44 | 38 | 35 | 34 | 34 | 42 |
| raw24 [o200k] | 84 | 42 | 21 | 11 | 6 | 3 | 2 | 1 | 6 |
| zstd-3 + dict (ID bytes) [o200k] | 343 | 178 | 95 | 53 | 31 | 20 | 12 | 8 | 32 |
| freq+lz4 (pgtoken) [o200k] | 83 | 43 | 23 | 14 | 9 | 6 | 5 | 5 | 10 |

## wildchat / messages

| method | 1-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | 1024-2047 | 2048+ | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n values | 32299 | 13290 | 10141 | 12067 | 17541 | 18825 | 8284 | 1808 | 587 | 114842 |
| median o200k tokens | 8 | 21 | 45 | 93 | 190 | 345 | 666 | 1254 | 2948 | 71 |
| utf8 (no compression) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| PostgreSQL text | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.48 | 1.77 | 2.04 | 1.00 |
| lz4 | 0.85 | 0.95 | 1.02 | 1.12 | 1.26 | 1.39 | 1.52 | 1.77 | 2.04 | 1.08 |
| zstd-3 | 0.87 | 1.07 | 1.32 | 1.51 | 1.72 | 1.96 | 2.20 | 2.54 | 2.97 | 1.38 |
| zstd-3 + dict (text) | 0.94 | 1.41 | 1.72 | 1.87 | 2.10 | 2.30 | 2.48 | 2.78 | 3.14 | 1.72 |
| FSST | 1.24 | 1.34 | 1.37 | 1.37 | 1.35 | 1.35 | 1.38 | 1.36 | 1.35 | 1.33 |
| Schmidt 16-bit (cl100k patched) | 1.56 | 1.73 | 1.75 | 1.74 | 1.71 | 1.73 | 1.78 | 1.77 | 1.73 | 1.70 |
| raw24 [o200k] | 1.14 | 1.35 | 1.44 | 1.47 | 1.45 | 1.44 | 1.45 | 1.42 | 1.35 | 1.37 |
| zstd-3 + dict (ID bytes) [o200k] | 1.20 | 1.66 | 2.00 | 2.19 | 2.40 | 2.60 | 2.78 | 3.04 | 3.14 | 2.02 |
| freq [o200k] | 1.61 | 2.07 | 2.31 | 2.36 | 2.35 | 2.37 | 2.39 | 2.33 | 2.14 | 2.15 |
| freq+lz4 (pgtoken) [o200k] | 1.61 | 2.08 | 2.33 | 2.43 | 2.56 | 2.69 | 2.78 | 2.95 | 3.02 | 2.29 |
| freq+lz4, C4 table [o200k] | 1.32 | 1.83 | 2.10 | 2.10 | 2.09 | 2.27 | 2.60 | 2.73 | 2.87 | 1.90 |
| raw24 [qwen3] | 1.18 | 1.38 | 1.46 | 1.51 | 1.52 | 1.52 | 1.46 | 1.39 | 1.29 | 1.41 |
| zstd-3 + dict (ID bytes) [qwen3] | 1.25 | 1.67 | 2.01 | 2.22 | 2.44 | 2.62 | 2.77 | 3.03 | 3.13 | 2.04 |
| freq [qwen3] | 1.65 | 2.12 | 2.37 | 2.47 | 2.49 | 2.50 | 2.42 | 2.35 | 2.14 | 2.20 |
| freq+lz4 (pgtoken) [qwen3] | 1.65 | 2.12 | 2.38 | 2.53 | 2.67 | 2.77 | 2.79 | 2.96 | 3.02 | 2.34 |
| freq+lz4, C4 table [qwen3] | 1.41 | 1.77 | 1.98 | 2.03 | 2.11 | 2.25 | 2.57 | 2.74 | 2.93 | 1.94 |

Decode to o200k IDs, median ns per token (single core; text = decompress + HF tokenize):

| method | 1-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | 1024-2047 | 2048+ | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| utf8 (no compression) | 238 | 115 | 73 | 52 | 42 | 38 | 35 | 33 | 33 | 62 |
| PostgreSQL text | 238 | 115 | 73 | 52 | 42 | 38 | 36 | 34 | 34 | 62 |
| lz4 | 264 | 125 | 78 | 55 | 44 | 40 | 36 | 34 | 34 | 65 |
| zstd-3 | 281 | 165 | 114 | 77 | 58 | 51 | 44 | 40 | 39 | 91 |
| zstd-3 + dict (text) | 281 | 139 | 89 | 64 | 51 | 46 | 42 | 39 | 38 | 74 |
| FSST | 213 | 104 | 68 | 50 | 41 | 38 | 35 | 34 | 33 | 57 |
| raw24 [o200k] | 158 | 63 | 30 | 15 | 7 | 5 | 3 | 2 | 1 | 20 |
| zstd-3 + dict (ID bytes) [o200k] | 648 | 267 | 133 | 70 | 39 | 26 | 16 | 11 | 8 | 90 |
| freq+lz4 (pgtoken) [o200k] | 156 | 63 | 32 | 17 | 11 | 8 | 6 | 5 | 5 | 22 |

## wildchat / user

| method | 1-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | 1024-2047 | 2048+ | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n values | 29794 | 10844 | 5897 | 3717 | 2427 | 1898 | 1435 | 675 | 507 | 57194 |
| median o200k tokens | 8 | 21 | 43 | 87 | 174 | 353 | 742 | 1335 | 3087 | 15 |
| utf8 (no compression) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| PostgreSQL text | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.46 | 1.81 | 1.98 | 1.00 |
| lz4 | 0.85 | 0.95 | 1.02 | 1.11 | 1.22 | 1.32 | 1.48 | 1.81 | 1.98 | 0.92 |
| zstd-3 | 0.87 | 1.07 | 1.32 | 1.50 | 1.72 | 1.93 | 2.20 | 2.64 | 2.89 | 0.93 |
| zstd-3 + dict (text) | 0.92 | 1.39 | 1.68 | 1.86 | 2.11 | 2.30 | 2.54 | 2.85 | 3.07 | 1.24 |
| FSST | 1.24 | 1.35 | 1.38 | 1.38 | 1.38 | 1.38 | 1.38 | 1.36 | 1.35 | 1.31 |
| Schmidt 16-bit (cl100k patched) | 1.56 | 1.73 | 1.76 | 1.75 | 1.77 | 1.78 | 1.77 | 1.77 | 1.75 | 1.68 |
| raw24 [o200k] | 1.14 | 1.34 | 1.40 | 1.41 | 1.45 | 1.45 | 1.41 | 1.41 | 1.36 | 1.27 |
| zstd-3 + dict (ID bytes) [o200k] | 1.18 | 1.63 | 1.93 | 2.13 | 2.40 | 2.57 | 2.88 | 3.06 | 3.09 | 1.49 |
| freq [o200k] | 1.59 | 2.04 | 2.22 | 2.26 | 2.33 | 2.35 | 2.37 | 2.35 | 2.19 | 1.86 |
| freq+lz4 (pgtoken) [o200k] | 1.59 | 2.05 | 2.24 | 2.32 | 2.51 | 2.61 | 2.74 | 3.00 | 3.00 | 1.89 |
| freq+lz4, C4 table [o200k] | 1.31 | 1.82 | 2.07 | 2.14 | 2.40 | 2.49 | 2.62 | 2.77 | 2.87 | 1.62 |
| raw24 [qwen3] | 1.20 | 1.37 | 1.41 | 1.41 | 1.44 | 1.44 | 1.38 | 1.38 | 1.30 | 1.29 |
| zstd-3 + dict (ID bytes) [qwen3] | 1.24 | 1.65 | 1.91 | 2.11 | 2.38 | 2.54 | 2.85 | 3.06 | 3.09 | 1.52 |
| freq [qwen3] | 1.64 | 2.09 | 2.25 | 2.29 | 2.37 | 2.40 | 2.36 | 2.36 | 2.18 | 1.90 |
| freq+lz4 (pgtoken) [qwen3] | 1.64 | 2.10 | 2.27 | 2.34 | 2.52 | 2.64 | 2.72 | 2.99 | 2.98 | 1.93 |
| freq+lz4, C4 table [qwen3] | 1.39 | 1.75 | 1.94 | 2.03 | 2.30 | 2.48 | 2.60 | 2.78 | 2.93 | 1.64 |

Decode to o200k IDs, median ns per token (single core; text = decompress + HF tokenize):

| method | 1-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | 1024-2047 | 2048+ | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| utf8 (no compression) | 242 | 116 | 74 | 52 | 41 | 37 | 34 | 33 | 34 | 154 |
| PostgreSQL text | 242 | 116 | 74 | 52 | 41 | 37 | 36 | 34 | 35 | 154 |
| lz4 | 267 | 127 | 79 | 55 | 43 | 39 | 36 | 34 | 35 | 168 |
| zstd-3 | 286 | 167 | 115 | 78 | 57 | 49 | 43 | 39 | 40 | 202 |
| zstd-3 + dict (text) | 285 | 141 | 90 | 63 | 50 | 45 | 41 | 38 | 40 | 183 |
| FSST | 215 | 106 | 69 | 50 | 40 | 37 | 34 | 33 | 35 | 138 |
| raw24 [o200k] | 161 | 65 | 31 | 15 | 8 | 5 | 2 | 2 | 1 | 92 |
| zstd-3 + dict (ID bytes) [o200k] | 653 | 272 | 139 | 73 | 41 | 26 | 15 | 10 | 8 | 384 |
| freq+lz4 (pgtoken) [o200k] | 158 | 64 | 33 | 18 | 11 | 8 | 6 | 5 | 5 | 92 |

## Gate: freq vs zstd-3 + dict on WildChat messages

Median message: 71 o200k tokens (bucket 64-127). Band = messages of 53 to 89 tokens (n=7678).

| set | freq+lz4 | freq | zstd-3 + dict | FSST | zstd-3 dict on IDs | W/L/T freq vs zstd-dict |
|---|---:|---:|---:|---:|---:|---|
| band | 2.40 | 2.36 | 1.82 | 1.38 | 2.14 | 7528/130/20 (win 44.2 B each, loss -9.5 B each, p=0) |
| band_english | 2.70 | 2.68 | 2.09 | 1.44 | 2.40 | 3588/66/10 (win 34.6 B each, loss -9.0 B each, p=0) |
| bucket | 2.43 | 2.36 | 1.87 | 1.37 | 2.19 | 11742/293/32 (win 54.8 B each, loss -11.9 B each, p=0) |
| all_messages | 2.29 | 2.15 | 1.72 | 1.33 | 2.02 | 108930/5049/863 (win 62.5 B each, loss -86.8 B each, p=0) |

**Gate PASSES**: at the median message length freq stores 2.40x vs zstd-3 + dict 1.82x.

## Where freq+lz4 (o200k) loses, by bucket

Share of values where freq+lz4 stores more bytes than zstd-3 + dict (text) / than FSST.

c4 / cut:

| vs | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zstd-3 + dict (text) | 0.0% | 0.2% | 0.7% | 0.4% | 0.6% | 0.2% | 0.3% | 0.2% | 0.3% |
| FSST | 3.4% | 0.6% | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.5% |

c4 / doc:

| vs | 1-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | 1024-2047 | 2048+ | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zstd-3 + dict (text) | 0.0% | 0.4% | 0.7% | 0.5% | 0.5% | 0.8% | 0.4% | 1.3% | 3.3% | 0.7% |
| FSST | 54.5% | 2.8% | 0.1% | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.1% |

code / cut:

| vs | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zstd-3 + dict (text) | 2.8% | 8.1% | 15.6% | 22.0% | 32.4% | 41.6% | 50.4% | 65.9% | 26.2% |
| FSST | 10.9% | 3.1% | 0.9% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 2.1% |

dbpedia / abstract:

| vs | 1-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zstd-3 + dict (text) | 3.7% | 27.3% | 33.7% | 13.9% | 21.0% | 67.1% | 100.0% | 19.0% |
| FSST | 13.0% | 1.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.1% |

dbpedia / cut:

| vs | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zstd-3 + dict (text) | 7.3% | 20.5% | 20.8% | 21.5% | 19.7% | 17.4% | 11.4% | 7.1% | 15.7% |
| FSST | 13.9% | 1.6% | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 1.9% |

wildchat / assistant:

| vs | 1-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | 1024-2047 | 2048+ | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zstd-3 + dict (text) | 2.5% | 1.4% | 1.5% | 1.8% | 3.0% | 5.2% | 15.3% | 38.0% | 71.2% | 5.5% |
| FSST | 17.5% | 1.6% | 0.5% | 0.1% | 0.1% | 0.1% | 0.0% | 0.4% | 0.0% | 0.9% |

wildchat / conversation:

| vs | 1-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | 1024-2047 | 2048+ | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zstd-3 + dict (text) | 0.0% | 0.4% | 0.8% | 1.4% | 3.8% | 4.2% | 8.7% | 15.5% | 29.0% | 11.3% |
| FSST | 3.0% | 0.4% | 0.0% | 0.1% | 0.1% | 0.1% | 0.0% | 0.0% | 0.0% | 0.1% |

wildchat / cut:

| vs | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zstd-3 + dict (text) | 0.7% | 1.1% | 1.1% | 2.5% | 4.6% | 7.2% | 9.5% | 12.8% | 4.9% |
| FSST | 1.1% | 0.1% | 0.1% | 0.2% | 0.1% | 0.0% | 0.0% | 0.0% | 0.2% |

wildchat / messages:

| vs | 1-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | 1024-2047 | 2048+ | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zstd-3 + dict (text) | 0.9% | 0.6% | 1.2% | 2.4% | 3.7% | 6.1% | 16.5% | 41.0% | 61.3% | 4.4% |
| FSST | 18.4% | 1.1% | 0.4% | 0.2% | 0.1% | 0.1% | 0.0% | 0.2% | 0.2% | 5.4% |

wildchat / user:

| vs | 1-15 | 16-31 | 32-63 | 64-127 | 128-255 | 256-511 | 512-1023 | 1024-2047 | 2048+ | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zstd-3 + dict (text) | 0.8% | 0.4% | 1.0% | 3.9% | 8.0% | 13.7% | 22.0% | 46.1% | 59.8% | 3.3% |
| FSST | 18.5% | 1.0% | 0.4% | 0.4% | 0.2% | 0.4% | 0.1% | 0.0% | 0.2% | 9.9% |

## Examples (3 real values per dataset: short, median, long)

### c4

| value | tokens | utf8 (no compression) | PostgreSQL text | lz4 | zstd-3 | zstd-3 + dict (text) | FSST | Schmidt 16-bit (cl100k patched) | raw24 [o200k] | zstd-3 + dict (ID bytes) [o200k] | freq [o200k] | freq+lz4 (pgtoken) [o200k] | freq+lz4, C4 table [o200k] | raw24 [qwen3] | zstd-3 + dict (ID bytes) [qwen3] | freq [qwen3] | freq+lz4 (pgtoken) [qwen3] | freq+lz4, C4 table [qwen3] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| doc: `Post a supportive message for Earleen L. Hughes.\nMATTHEW - NEW CREATURE IN CHRIST!!!\nHi Earleen - trying to learn more about Warren because I lost con` | 64 | 241 | 241 | 242 | 194 | 151 | 156 | 140 | 197 | 135 | 117 | 116 |  | 209 | 130 | 115 | 115 |  |
| doc: `Painting in the style of the Divisionists, Louis Pigniollet enjoyed painting the landscapes of the Fousges, Jura and Franche-Comte areas of France. Us` | 241 | 1055 | 1055 | 884 | 612 | 483 | 571 | 570 | 729 | 463 | 425 | 408 |  | 768 | 467 | 429 | 414 |  |
| doc: `I promised last report that I would try to fill you in on a number of topics. Since I am late getting this in, I had better just jump right in.\nUnfort` | 1400 | 6692 | 4867 | 4867 | 3156 | 2526 | 3533 | 3446 | 4206 | 2302 | 2089 | 2085 |  | 4287 | 2325 | 2096 | 2082 |  |

freq+lz4 (o200k) loses to the better of zstd-3 + dict and FSST on 155 of 20101 values. vs zstd-3 + dict (text): W/L/T 19943/142/16; vs FSST: W/L/T 20084/16/1

- loss: `L was u 1 o having level, F heard, e amongst n having g defense within w amongst a heard, s defense level, s amongst h within o as c the k having e ag` (1125 tokens, 4010 B): freq+lz4 1751 B, zstd-3 + dict 1372 B, FSST 2338 B
- loss: `In The First Match at CCL 2014, Kerala Strikers Team Facing Telegu Warriors. Kerala Strikers Vs Telegu Warriors Match Date is 26th January 2014. Time ` (432 tokens, 1409 B): freq+lz4 832 B, zstd-3 + dict 755 B, FSST 924 B
- loss: `mainly on what we look for. . .` (10 tokens, 31 B): freq+lz4 21 B, zstd-3 + dict 26 B, FSST 17 B

### code

| value | tokens | utf8 (no compression) | PostgreSQL text | lz4 | zstd-3 | zstd-3 + dict (text) | FSST | Schmidt 16-bit (cl100k patched) | raw24 [o200k] | zstd-3 + dict (ID bytes) [o200k] | freq [o200k] | freq+lz4 (pgtoken) [o200k] | freq+lz4, C4 table [o200k] | raw24 [qwen3] | zstd-3 + dict (ID bytes) [qwen3] | freq [qwen3] | freq+lz4 (pgtoken) [qwen3] | freq+lz4, C4 table [qwen3] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cut16: ` = cls._windows_illegal_name_trans_table\n        if not table:\n           ` | 16 | 74 | 74 | 70 | 71 | 45 | 39 | 34 | 53 | 43 | 29 | 29 | 47 | 53 | 44 | 29 | 29 | 46 |
| cut128: `tdcQryLockField["ExchangeID"] = "string"\n#锁定编号\nCThostFtdcQryLockField["LockSysID"] = "string"\n#开始时间\nCThostFtdcQryLockField["InsertTimeStart"] = "strin` | 128 | 371 | 371 | 219 | 209 | 195 | 382 | 260 | 390 | 147 | 270 | 152 | 179 | 396 | 139 | 264 | 144 | 170 |
| cut2048: ` {1}'.format(\n                alias, str(ex.response_text)))\n        for alert_policy in alert_policies.get('items'):\n            if alert_policy.get(` | 2048 | 9352 | 4421 | 4421 | 2986 | 2433 | 5188 | 4814 | 6150 | 2356 | 3288 | 2590 | 3306 | 5958 | 2338 | 3193 | 2564 | 3230 |

freq+lz4 (o200k) loses to the better of zstd-3 + dict and FSST on 1999 of 7180 values. vs zstd-3 + dict (text): W/L/T 5230/1881/69; vs FSST: W/L/T 6988/149/43

- loss: `__email__ = 'ongsp@ucsd.edu'\n__date__ = '1/24/14'\n\n\nimport unittest\nimport os\n\nfrom monty.os.path import which, zpath\nfrom monty.os import cd\n\ntest_di` (512 tokens, 2336 B): freq+lz4 664 B, zstd-3 + dict 649 B, FSST 1302 B
- loss: `_event", self.quit)\n\n    def __setattr__(self, name, value):\n        self.__dict__[name] = value\n\n    def quit(self, *args):\n        self.status = 0\n ` (512 tokens, 2243 B): freq+lz4 775 B, zstd-3 + dict 771 B, FSST 1168 B
- loss: `        self.assertAllClose(col_sum, tf_out_col)\n        self.assertAllClose(row_sum, tf_out_row)\n        self.assertAllClose(full_sum, tf_out_full)\n\n` (128 tokens, 402 B): freq+lz4 160 B, zstd-3 + dict 145 B, FSST 259 B

### dbpedia

| value | tokens | utf8 (no compression) | PostgreSQL text | lz4 | zstd-3 | zstd-3 + dict (text) | FSST | Schmidt 16-bit (cl100k patched) | raw24 [o200k] | zstd-3 + dict (ID bytes) [o200k] | freq [o200k] | freq+lz4 (pgtoken) [o200k] | freq+lz4, C4 table [o200k] | raw24 [qwen3] | zstd-3 + dict (ID bytes) [qwen3] | freq [qwen3] | freq+lz4 (pgtoken) [qwen3] | freq+lz4, C4 table [qwen3] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| abstract: `Lists of people\nPeople in these lists may have either unspecified traits, or specific characteristics (e. g. the people of Spain or the people of the ` | 44 | 222 | 222 | 189 | 144 | 95 | 127 | 122 | 137 | 94 | 76 | 76 | 76 | 137 | 95 | 76 | 76 | 75 |
| abstract: `Mary of Rome\nThe Apostle Paul's Epistle to the Romans (16:6) mentions a Mary. She is said to have treated Paul with special kindness, and to have "lab` | 88 | 401 | 401 | 381 | 272 | 186 | 218 | 214 | 269 | 177 | 144 | 144 | 149 | 275 | 165 | 144 | 144 | 147 |
| abstract: `Gáe Bulg\nThe Gáe Bulg (also Gáe Bulga, Gáe Bolg, Gáe Bolga), meaning "spear of mortal pain/death spear", "gapped/notched spear", or "belly spear", was` | 129 | 473 | 473 | 400 | 296 | 214 | 272 | 296 | 393 | 214 | 230 | 215 | 221 | 405 | 217 | 231 | 216 | 225 |

freq+lz4 (o200k) loses to the better of zstd-3 + dict and FSST on 4818 of 25266 values. vs zstd-3 + dict (text): W/L/T 20150/4800/316; vs FSST: W/L/T 25244/19/3

- loss: `Abercrombie River National Park\nThe Abercrombie River National Park is a protected national park that is located in the Central Tablelands region of N` (81 tokens, 341 B): freq+lz4 131 B, zstd-3 + dict 124 B, FSST 192 B
- loss: `1883\nYear 1883 (MDCCCLXXXIII) was a common year starting on Monday (link will display the full calendar) of the Gregorian calendar and a common year s` (49 tokens, 207 B): freq+lz4 75 B, zstd-3 + dict 58 B, FSST 117 B
- loss: `South San Jose Hills, California\nSouth San Jose Hills is an unincorporated community and census-designated place in Los Angeles County, California, Un` (60 tokens, 243 B): freq+lz4 81 B, zstd-3 + dict 69 B, FSST 131 B

### wildchat

| value | tokens | utf8 (no compression) | PostgreSQL text | lz4 | zstd-3 | zstd-3 + dict (text) | FSST | Schmidt 16-bit (cl100k patched) | raw24 [o200k] | zstd-3 + dict (ID bytes) [o200k] | freq [o200k] | freq+lz4 (pgtoken) [o200k] | freq+lz4, C4 table [o200k] | raw24 [qwen3] | zstd-3 + dict (ID bytes) [qwen3] | freq [qwen3] | freq+lz4 (pgtoken) [qwen3] | freq+lz4, C4 table [qwen3] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| assistant: `ceasing to dismay her.` | 7 | 22 | 22 | 28 | 27 | 27 | 13 | 14 | 26 | 26 | 17 | 17 | 17 | 23 | 23 | 15 | 15 | 15 |
| assistant: `好的，根据最新的天气预报，深圳今天的天气晴间多云，最高温度约30摄氏度，最低温度约为25摄氏度，相对湿度较大，预计降雨概率较低，持续时间较短。提醒您注意及时补水、做好防晒以及避免日晒过度。` | 71 | 274 | 274 | 258 | 240 | 221 | 238 | 248 | 218 | 171 | 142 | 141 | 195 | 188 | 151 | 128 | 125 | 162 |
| user: `Improve this code any way possible:  <template>\n  <div>\n    <div ref="listContainer" class="list-container">\n      <div\n        v-for="(item, index) i` | 711 | 2820 | 1402 | 1402 | 990 | 956 | 2084 | 1542 | 2139 | 875 | 1296 | 965 | 1179 | 2010 | 857 | 1235 | 943 | 1122 |

freq+lz4 (o200k) loses to the better of zstd-3 + dict and FSST on 10862 of 114842 values. vs zstd-3 + dict (text): W/L/T 108930/5049/863; vs FSST: W/L/T 107176/6221/1445

- loss: `Hello` (1 tokens, 5 B): freq+lz4 7 B, zstd-3 + dict 10 B, FSST 4 B
- loss: `hi` (1 tokens, 2 B): freq+lz4 7 B, zstd-3 + dict 7 B, FSST 2 B
- loss: `4` (1 tokens, 1 B): freq+lz4 6 B, zstd-3 + dict 6 B, FSST 1 B

## Default table: freq with the C4-trained table vs the in-domain table

| dataset / group | in-domain freq+lz4 | C4-table freq+lz4 | loss |
|---|---:|---:|---:|
| code / cut [o200k] | 2.83 | 2.18 | 23.0% lower ratio, +24.5% bytes |
| code / cut [qwen3] | 2.85 | 2.21 | 22.3% lower ratio, +23.3% bytes |
| dbpedia / abstract [o200k] | 2.66 | 2.43 | 8.8% lower ratio, +8.8% bytes |
| dbpedia / abstract [qwen3] | 2.61 | 2.41 | 7.7% lower ratio, +7.4% bytes |
| dbpedia / cut [o200k] | 2.64 | 2.44 | 7.4% lower ratio, +6.9% bytes |
| dbpedia / cut [qwen3] | 2.60 | 2.44 | 6.5% lower ratio, +6.1% bytes |
| wildchat / assistant [o200k] | 2.58 | 2.18 | 15.7% lower ratio, +17.0% bytes |
| wildchat / assistant [qwen3] | 2.68 | 2.16 | 19.2% lower ratio, +14.9% bytes |
| wildchat / conversation [o200k] | 2.84 | 2.64 | 7.1% lower ratio, +13.1% bytes |
| wildchat / conversation [qwen3] | 2.86 | 2.58 | 9.9% lower ratio, +11.4% bytes |
| wildchat / cut [o200k] | 2.53 | 2.25 | 11.1% lower ratio, +14.5% bytes |
| wildchat / cut [qwen3] | 2.54 | 2.22 | 12.5% lower ratio, +12.8% bytes |
| wildchat / messages [o200k] | 2.29 | 1.90 | 17.2% lower ratio, +14.9% bytes |
| wildchat / messages [qwen3] | 2.34 | 1.94 | 16.9% lower ratio, +13.1% bytes |
| wildchat / user [o200k] | 1.89 | 1.62 | 14.4% lower ratio, +8.9% bytes |
| wildchat / user [qwen3] | 1.93 | 1.64 | 14.9% lower ratio, +8.0% bytes |
