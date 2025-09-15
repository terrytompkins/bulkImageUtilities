# PNG Lossless Compression & Bundling Playbook

This guide distills best‑practice, **lossless** approaches for shrinking large batches of PNG images and (optionally) bundling them into resilient, S3‑friendly archives. It’s designed to be copy‑pasted into an agent’s context to drive code changes and experiments.

---

## Goals & Constraints

- **Keep PNG format** (no lossy conversion).
- **Maximize size reduction** with reasonable CPU use on an edge device.
- Support **resumable, fault‑tolerant uploads** (prefer multiple mid‑sized archives or individual files).
- Keep the process **scriptable** with clear CLI parameters that you can mix & match.

---

## Quick Recommendations (TL;DR)

1. **First pass (biggest win):** Losslessly recompress all PNGs:
   - Fast & effective: `oxipng -o4 -T 0 --strip safe -r <dir>`
   - Max squeeze (slower): `zopflipng --lossless -y --iterations=15 --filters=01234mepb`
2. **If you still want to bundle (operational reasons):**
   - Balanced: `tar + zstd` with long-range matching → `tar -I 'zstd -T0 -9 --long=27' -cvf batch01.tar.zst <dir>`
   - Tighter but slower: `7z` solid LZMA2 → `7z a -t7z -m0=lzma2 -mx=7 -ms=on -mmt=on batch01.7z <dir>`
   - Fastest legacy: `tar + pigz` → `tar -I 'pigz -9 -p 0' -cvf batch01.tar.gz <dir>`
3. **Archive size target:** ~**200–500 MB** each. Upload archives in parallel with S3 multipart.

---

## Why Start with Lossless PNG Recompression?

Your current PNGs are saved at **compression level 0** (near‑uncompressed). Re‑optimizing the same PNGs can produce **large lossless reductions** (often 40–70% depending on data) because tools redo PNG filtering and DEFLATE decisions and strip non‑essential chunks. This change alone may beat a **26% WebP** target without any format change.

---

## Tooling & Commands

### 1) Lossless PNG Recompression (per‑file)

#### A. `oxipng` (fast, multi‑core, great default)
- **Use when:** You want strong reduction with good speed on edge hardware.
- **Commands:**
```bash
# Bulk, recursive, in-place, keep lossless, remove non-essential metadata
oxipng -o4 -T 0 --strip safe -r /path/to/images

# For more squeeze (slower): try -o6 or -o7
oxipng -o6 -T 0 --strip safe -r /path/to/images
```

**Flags:**
- `-oN` — effort level (higher = smaller, slower). Try 4–6 first.
- `-T 0` — use all CPU cores.
- `--strip safe` — strip non‑essential metadata (lossless).

#### B. `zopflipng` (max ratio, much slower)
- **Use when:** You need the smallest possible PNGs and can afford CPU time.
- **Commands:**
```bash
# Parallel over many files (Linux)
find /path/to/images -name '*.png' -print0 | xargs -0 -P "$(nproc)" -I{} \
  zopflipng --lossless -y --iterations=15 --filters=01234mepb {} {}
```
**Flags:**
- `--lossless` — keeps pixel data identical.
- `--iterations=15` — more trials (slower but smaller).
- `--filters=01234mepb` — try many filter strategies.

#### C. `pngcrush` (classic, mid‑speed)
```bash
find /path/to/images -name '*.png' -print0 | xargs -0 -P "$(nproc)" -I{} \
  pngcrush -brute -reduce -ow {}
```
**Flags:**
- `-brute` — exhaustive compression trials.
- `-reduce` — lossless bit‑depth/color‑type reductions.
- `-ow` — overwrite files in place.

> **Practical tip:** Run **oxipng** first over the entire set (fast, big wins). Then sample 5–10% through **zopflipng** to see if the extra CPU is worth it.

---

### 2) Bundling (optional, for operational benefits)

Once each PNG is well‑compressed, bundling adds less ratio benefit but helps uploads and retry logic. Prefer **multiple mid‑sized archives** so an error doesn’t force a multi‑GB restart.

#### A. `tar + zstd` (balanced ratio/speed, recommended)
**Good default:**
```bash
# Create ~300 MB archives by batching files (see scripts below)
tar -I 'zstd -T0 -9 --long=27' -cvf batch01.tar.zst /path/to/batch01
```
**Stronger (slower):**
```bash
tar -I 'zstd -T0 -12 --long=27 --ultra' -cvf batch01.tar.zst /path/to/batch01
```
**Notes:**
- `-T0` — all cores.
- `-9` — solid trade‑off. Try 7–12; above 12 gains are small vs. CPU.
- `--long=27` — long-range matching (~128 MiB window) helps if batch images are similar across time/frames.
- Decompress: `tar -I zstd -xvf batch01.tar.zst`

**Optional: zstd dictionary (extra % on very similar files)**
```bash
# Train a dictionary on a representative subset
zstd --train /path/to/sample/*.png -o png.dict

# Use dictionary during compression
tar -I 'zstd -T0 -10 --long=27 --ultra --dict=png.dict' -cvf batch01.tar.zst /path/to/batch01
```

#### B. `7z` (LZMA2, solid) — tighter, slower
```bash
7z a -t7z -m0=lzma2 -mx=7 -ms=on -mmt=on batch01.7z /path/to/batch01
```
- `-mx=9` is the max but often too slow on edge devices. 6–7 is a good balance.
- `-ms=on` (solid) is critical for cross‑file redundancy.
- `-mmt=on` uses multiple threads.

#### C. `tar + pigz` (parallel gzip) — fastest, least CPU per byte
```bash
tar -I 'pigz -9 -p 0' -cvf batch01.tar.gz /path/to/batch01
```
- `-p 0` uses all cores.
- `pigz -11` (zopfli pass) can be smaller but is **much** slower.

---

## Chunking / Batch Sizing

- Target **200–500 MB** per archive for quick retries and good S3 multipart behavior.
- Batch by estimated total bytes, not by file count alone.
- Keep batches internally **homogeneous** (e.g., consecutive frames) to maximize cross‑file redundancy for `--long`/solid archives.

### Bash chunker (approximate 300 MB batches)
```bash
#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/data/images"
DST_DIR="/data/batches"
TARGET=$((300 * 1024 * 1024)) # 300 MB

mkdir -p "$DST_DIR"
batch_idx=1
batch_bytes=0
batch_dir=$(printf "%s/part%02d" "$DST_DIR" "$batch_idx")
mkdir -p "$batch_dir"

# Use LC_ALL=C for faster sort; remove if order matters different
find "$SRC_DIR" -type f -name '*.png' -print0 | while IFS= read -r -d '' f; do
  size=$(stat -c%s "$f")
  if (( batch_bytes + size > TARGET )) && (( batch_bytes > 0 )); then
    batch_idx=$((batch_idx+1))
    batch_bytes=0
    batch_dir=$(printf "%s/part%02d" "$DST_DIR" "$batch_idx")
    mkdir -p "$batch_dir"
  fi
  mv "$f" "$batch_dir/"
  batch_bytes=$((batch_bytes + size))
done
```

---

## S3 Upload Strategy (CLI‑friendly)

- Use **multipart uploads** (AWS CLI v2 auto‑multipart for files >64 MB).
- **Parallelize** uploads across archives (GNU Parallel or background jobs).
- Consider **S3 Transfer Acceleration** if the edge is far from the bucket region.
- Validate with checksums when desired.

### Parallel upload example (GNU Parallel)
```bash
# Upload all .tar.zst in current dir to a bucket prefix (4 parallel jobs)
ls *.tar.zst | parallel -j 4 "aws s3 cp {} s3://my-bucket/prefix/ --only-show-errors --no-progress"
```

---

## Benchmark Recipe (pick the winner on *your* hardware)

Choose a representative **1–2 GB** subset and time these:
```bash
# A) oxipng fast
/usr/bin/time -v sh -c "oxipng -o4 -T 0 --strip safe -r sample_set"

# B) oxipng stronger
/usr/bin/time -v sh -c "oxipng -o6 -T 0 --strip safe -r sample_set"

# C) zopflipng (sampled subset, slow)
/usr/bin/time -v sh -c "find sample_set -name '*.png' | head -n 500 | xargs -I{} zopflipng --lossless -y --iterations=15 --filters=01234mepb {} {}"

# D) tar+zstd after recompression
/usr/bin/time -v sh -c "tar -I 'zstd -T0 -9 --long=27' -cvf test.tar.zst sample_set"

# E) 7z after recompression
/usr/bin/time -v sh -c "7z a -t7z -m0=lzma2 -mx=7 -ms=on -mmt=on test.7z sample_set"

# F) pigz after recompression
/usr/bin/time -v sh -c "tar -I 'pigz -9 -p 0' -cvf test.tar.gz sample_set"
```
Compute **effective upload time**:
```
effective_time = compression_time + (output_bytes / link_throughput_bytes_per_sec)
```
Pick the setting with the lowest effective time that meets operational needs.

---

## Preset Matrix (copy into your scripts)

| Purpose | Tool | Command (replace `<dir>`, `<batch>`) |
|---|---|---|
| Fast bulk PNG shrink | **oxipng** | `oxipng -o4 -T 0 --strip safe -r <dir>` |
| Stronger shrink | **oxipng** | `oxipng -o6 -T 0 --strip safe -r <dir>` |
| Max squeeze (slow) | **zopflipng** | `zopflipng --lossless -y --iterations=15 --filters=01234mepb in.png out.png` |
| Classic optimizer | **pngcrush** | `pngcrush -brute -reduce -ow <file_or_glob>` |
| Balanced bundling | **tar+zstd** | `tar -I 'zstd -T0 -9 --long=27' -cvf <batch>.tar.zst <dir>` |
| Stronger zstd | **tar+zstd** | `tar -I 'zstd -T0 -12 --long=27 --ultra' -cvf <batch>.tar.zst <dir>` |
| zstd + dict | **tar+zstd** | `zstd --train <sample_glob> -o png.dict && tar -I 'zstd -T0 -10 --long=27 --ultra --dict=png.dict' -cvf <batch>.tar.zst <dir>` |
| Tight but slow | **7z/LZMA2** | `7z a -t7z -m0=lzma2 -mx=7 -ms=on -mmt=on <batch>.7z <dir>` |
| Fastest bundling | **tar+pigz** | `tar -I 'pigz -9 -p 0' -cvf <batch>.tar.gz <dir>` |

---

## End‑to‑End Example Pipeline (ready to adapt)

```bash
# 1) Recompress PNGs in place (lossless, fast)
oxipng -o4 -T 0 --strip safe -r /data/images

# 2) Split into ~300 MB batches
/path/to/chunker.sh  # (the Bash chunker from this doc)

# 3) Compress each batch with zstd (balanced)
for d in /data/batches/part*; do
  base=$(basename "$d")
  tar -I 'zstd -T0 -9 --long=27' -cvf "${base}.tar.zst" "$d"
done

# 4) Upload in parallel with AWS CLI
ls *.tar.zst | parallel -j 4 "aws s3 cp {} s3://my-bucket/prefix/ --only-show-errors --no-progress"
```

---

## Guardrails & Notes

- All PNG steps here are **lossless**; pixel data remains identical.
- `--strip safe` removes only non‑essential metadata (timestamps, text chunks, etc.).
- Very small files compress relatively worse; batch them with **similar** neighbors.
- For `zstd --long`, very large windows may require more RAM; `--long=27` (~128 MiB) is a good starting point.
- Measure with **representative** data; frame‑to‑frame similarity drives solid‑archive gains.
- If CPU is the bottleneck and network is decent, prefer **oxipng + tar+pigz**.
- If network is tight and CPU is available, consider **oxipng + tar+zstd (L12)** or **7z**.

---

## Drop‑in Parameter Blocks (for agents)

```bash
# PNG: fast lossless
OXIPNG_FAST="oxipng -o4 -T 0 --strip safe -r"

# PNG: stronger lossless
OXIPNG_STRONG="oxipng -o6 -T 0 --strip safe -r"

# PNG: maximal (slow)
ZOPFLIPNG_MAX="zopflipng --lossless -y --iterations=15 --filters=01234mepb"

# Bundle: zstd balanced
ZSTD_BAL="tar -I 'zstd -T0 -9 --long=27' -cvf"

# Bundle: zstd stronger
ZSTD_STRONG="tar -I 'zstd -T0 -12 --long=27 --ultra' -cvf"

# Bundle: zstd with dict (after training)
ZSTD_DICT="tar -I 'zstd -T0 -10 --long=27 --ultra --dict=png.dict' -cvf"

# Bundle: 7z solid (balanced)
SEVENZ_BAL="7z a -t7z -m0=lzma2 -mx=7 -ms=on -mmt=on"

# Bundle: pigz fast
PIGZ_FAST="tar -I 'pigz -9 -p 0' -cvf"
```

---

## Integration Tips for Your Script/Agent

- Expose **knobs**: PNG pass (`none|oxipng_fast|oxipng_strong|zopfli_max`), bundler (`none|zstd|7z|pigz`), archive target size (MB), parallelism (N).
- Implement **dry‑run** mode to print planned commands before execution.
- Capture and log: tool version, elapsed seconds, input bytes, output bytes, and computed **effective upload time** given a network throughput assumption.
- Fail fast on partial errors (e.g., checksum mismatch or non‑zero exit).

---

*End of playbook.*
