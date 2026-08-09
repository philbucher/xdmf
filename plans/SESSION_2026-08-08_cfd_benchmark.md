# Session summary — xdmf vs pyvista CFD-case benchmark (2026-08-08)

> **Status: historical record.** This documents an ad hoc benchmarking session. The xdmf Python
> bindings it exercised aren't on `main` — see `06_python_bindings.md` and
> `SESSION_2026-07-18_python_bindings.md`; they live on `origin/multiple-features`, not merged.

> **CORRECTION (2026-08-08).** This summary originally stated that `mesh_gen.py`,
> `cfd_benchmark.py` and `CFD_BENCHMARK_PLAN.md` "were not found in this working tree" and concluded
> the benchmark would have to be rebuilt from scratch (see "If this benchmark is redone", below).
> **That is wrong** — only the working tree was checked, not the branch. All of them are committed on
> `origin/multiple-features`:
>
> ```
> $ git ls-tree -r --name-only origin/multiple-features | grep -E "benchmark|mesh_gen|CFD"
> CFD_BENCHMARK_PLAN.md
> python/benchmarks/bench_common.py
> python/benchmarks/cfd_benchmark.py
> python/benchmarks/mesh_gen.py
> python/benchmarks/random_benchmark.py
> ```
>
> Retrieve with `git show origin/multiple-features:<path>`. The branch's `CFD_BENCHMARK_PLAN.md` also
> contains material this summary does not: the HDF5-with-`shuffle`+`deflate` measurements (which are
> the reason `DEFAULT_DEFLATE_LEVEL = 3`) and the Blosc-fails-in-ParaView analysis. Read it before
> redoing any of this. `ROADMAP.md` schedules both for cherry-picking into `plans/`.

## Goal

Benchmark xdmf's (not-yet-merged) Python bindings against `pyvista` for writing a "standard CFD
case": a hex volume domain plus 3 quad boundary patches (`inlet`/`outlet`/`sides`), each carrying a
velocity vector + pressure scalar point-data field. Three mesh sizes were targeted: 1e3, 1e5, 1e7
elements. xdmf was tested with both `Ascii` and `Binary` storage (`Ascii` skipped at 1e7 — too slow
to be a realistic option at that size). pyvista wrote zipped `.vtu`.

## What was actually measured

Two questions ended up mattering more than the raw write-time numbers:

1. **Is xdmf's raw payload actually smaller**, as its "write the mesh once, not per timestep" design
   suggests it should be? Verified empirically: yes — 940 MB vs. 1,736 MB for the domain zone alone,
   uncompressed, at the 1e7 case.
2. **Does that translate into a smaller final (compressed) archive?** Not reliably — see Findings.

### Data-type mismatch found along the way

`pyvista`/VTK uses `float64` for points and attribute data and `int64` for connectivity — pyvista
does **not** use `float32` anywhere in this path. This matters because it was initially assumed
(wrongly) that xdmf should add `f32` support to match a pyvista optimization; there is no such
optimization to match. (xdmf's own float precision is hardcoded to 8 bytes in `Values::precision()`
— see `03_values_and_f32.md` for the actual f32 plan, which is motivated by xdmf's own performance
goals, not by parity with pyvista.) xdmf's `Binary` storage narrows connectivity to `uint32`
(`Format::uint_precision()` in `src/xdmf_elements/data_item.rs`), which pyvista never does.

## Benchmark methodology: the "final archive" fairness rule

The real result of either pipeline is a compressed archive, not the raw write. Getting a fair
comparison took several corrections during the session (see Process notes) and settled on this rule:

**Apply the exact same set of external compressors, in the same way, to both tools' real default
output** — not xdmf-raw-plus-a-strong-codec vs. pyvista-default-plus-a-weak-codec, and not
pyvista-with-compression-stripped-out. Concretely:

- pyvista: write with its actual default (`grid.save(path, binary=True, compression='zlib')` — this
  *is* pyvista's default, not an artificial choice), then layer deflate / zstd-6 / bzip2 / lzma on
  top of that already-zlib-compressed `.vtu` set.
- xdmf: write raw `Binary` storage (no internal compression to disable — `Binary`/`Ascii` don't
  compress), then layer the same four codecs on top.
- Both sides measured as **total time** (write + final compression) and **final archive size**.

Compressors tried, in increasing strength/decreasing speed: `zipfile.ZIP_DEFLATED` (weak/fast
baseline), `zstd` CLI (levels 3–19, `--long=27`/`--long=30` for extended match window — fast, good
ratio), `bz2.compress(data, 9)` (mid-strength, stdlib), `lzma.compress(data)` (strongest, slowest,
stdlib).

## Findings (1e7-element case)

| final compressor | pyvista (zlib write + compress) | xdmf (raw write + compress) |
|---|---|---|
| deflate | ~24.7 s / 100.0 MB | ~23.8 s / 124.2 MB |
| zstd-6  | ~21.9 s / 78.0 MB  | ~8.7 s / 104.6 MB  |
| bzip2   | ~34.2 s / 74.4 MB  | ~78.8 s / 82.8 MB  |
| lzma    | ~110.7 s / 52.5 MB | ~191.8 s / 22.9 MB |

Takeaways:

- pyvista wins on final size for deflate, zstd, and bzip2, despite writing ~1.85x more raw bytes.
- xdmf only wins decisively with lzma, and only at a large time cost (~1.7x slower than pyvista+lzma).
- xdmf is consistently faster at low/medium compression (zstd-6 in particular: ~2.5x faster than
  pyvista for a similar-order size), just not smaller.

### Root cause: why "less raw data" doesn't mean "smaller compressed archive"

VTK's `int64` connectivity indices are mostly zero-padding: for a 10M-node mesh, indices fit in
~24 bits, so the top ~5 bytes of every `int64` index are always zero. That padding compresses away
almost for free. xdmf's tightly-packed `uint32` connectivity has no such free padding — it starts
from half the raw bytes but each byte carries real entropy. Proven directly with an isolated test:
the *same* connectivity values, stored once as `uint32` (320 MB raw) and once as `int64` (640 MB
raw), compressed with zstd-6 to 87.3 MB and 49.4 MB respectively — the tightly-packed, objectively
smaller representation produced the *larger* compressed output. Packing tighter concentrates entropy;
it doesn't reduce it.

## Process notes (corrections during the session, worth remembering for next time)

- **Don't guess at what a comparison tool does — verify.** The float32 suggestion above was
  initially framed as matching a pyvista behavior; it wasn't (pyvista never uses f32 there). Check
  the actual dtypes/behavior of the thing you're comparing against before reasoning about why one
  side is smaller or faster.
- **"Stop just running stuff. Explain what you are doing and then run."** — explicit standing
  instruction from the user for any exploratory/benchmarking work: narrate the plan for a tool call
  *before* invoking it, especially for long-running ones, so it can be aborted before it starts.
- **Fair comparisons need identical treatment on both sides, applied to real (not synthetic) output.**
  Two wrong turns before landing on the rule above: (1) comparing xdmf+zstd against pyvista's own
  default pipeline (different codecs on each side); (2) stripping pyvista's compression out entirely
  to get a "raw" baseline, which no longer represented pyvista's actual behavior. The rule that
  stuck: same external codec set, applied on top of each tool's real default output.
- **"Why" questions about a surprising result deserve a root-cause answer, not just more numbers.**
  The uint32-vs-int64 padding test above was built specifically because restating the size table a
  second time wasn't an answer to "why is xdmf bigger despite writing less data."

## Related plans

- `03_values_and_f32.md` — the actual (pyvista-independent) motivation and plan for adding `f32`
  support to `Values`.
- `06_python_bindings.md` — status of the Python bindings this benchmark depends on (not on `main`).
- `02_performance.md` — general performance-tuning plan; compression-format tradeoffs belong there
  if/when this benchmark is formalized into a reusable script.

## If this benchmark is redone

~~None of `mesh_gen.py`, `cfd_benchmark.py`, or `CFD_BENCHMARK_PLAN.md` survived into this checkout,
so redoing this means rebuilding from scratch.~~ **See the correction at the top: they are all on
`origin/multiple-features`.** Start from those, not from a blank page.

What genuinely does still need doing: the **codec sweep** (the table above) was run by hand in scratch
files and was never built into the driver, so it did not survive. Build it into `cfd_benchmark.py`
when the scripts are cherry-picked, so the numbers are reproducible by running one command.
`02_performance.md` part A schedules this.
