# CFD write benchmark: `xdmf` (Python bindings) vs `pyvista`

**Status: executed manually in an interactive session; scripts exist but the codec sweep
was done ad hoc in scratch. This doc captures the methodology so it can be reproduced or
extended.** See "Findings" at the end for what the sweep actually showed.

## Goal

Compare `xdmf`'s Python bindings (see `PYTHON_BINDINGS_PLAN.md`) against `pyvista`/VTK for
writing a "standard CFD case": a 3D hexahedral structured-but-unstructured volume domain
with quad boundary patches ("inlet", "outlet", "sides"), carrying a vector field (velocity)
and a scalar field (pressure) as point data, at multiple mesh sizes. Metric of interest:
**time to produce a final, compressed, distributable archive, and that archive's size** —
not raw write time in isolation.

## Test matrix

| case | mesh              |   elements |
|:----:|:------------------|-----------:|
| 1e3  | 10x10x10 hex      |      1,000 |
| 1e5  | 10x10x1000 hex    |    100,000 |
| 1e7  | 100x100x1000 hex  | 10,000,000 |

Boundary patch cell counts derive from the mesh: `inlet`/`outlet` = `nx*ny` quads each,
`sides` = `2*(ny*nz + nx*nz)` quads.

## Mesh generation

Fully vectorized numpy (no per-node/per-cell Python loops), so the 10M-element case builds
in seconds. Implemented in `python/benchmarks/mesh_gen.py`:

- Points: `np.meshgrid(xs, ys, zs, indexing='ij')` over a `LX x LY x LZ` duct, flattened.
- Node indexing: `_node_index(i, j, k, ny, nz) = i*(ny+1)*(nz+1) + j*(nz+1) + k`.
- Hex connectivity: 8 shifted `meshgrid` index arrays stacked and flattened — standard VTK
  hex node ordering.
- Boundary quads: `inlet`/`outlet` at `k=0`/`k=nz` (vary i,j); `sides` = the 4 `i=0`, `i=nx`,
  `j=0`, `j=ny` walls (vary the other two indices). Same `_node_index` formula, so boundary
  nodes are shared with the volume mesh (no duplicate points).
- Fields: parabolic-profile `velocity` (duct-flow-like), linear `pressure` drop along z.
- `build_case(nx, ny, nz) -> CfdCase` returns flat float64 points/velocity, flat uint64
  connectivity (hex cells first, then inlet/outlet/sides quads, in that order), and a
  `blocks` list of `(name, cell_indices)` for xdmf's block API.
- `extract_submesh(case, flat_start, flat_end, num_nodes_per_cell)` slices a contiguous
  run of same-size cells out of the global connectivity and remaps to a compact local point
  numbering (via `np.unique(..., return_inverse=True)`) — used to build pyvista's per-zone
  grids. **Gotcha**: since hex cells (8 nodes) and quad cells (4 nodes) are concatenated in
  one flat array, a global cell index does *not* map to a flat array offset via a single
  uniform stride once you're past the hex section — pass explicit flat byte-offsets
  (`case.num_hex*8`, `+= count*4` per zone) rather than deriving them from a cell-index list.

## Writing: xdmf

**One file per case (per storage mode)**, combining the hex domain + 3 boundary patches as
named blocks sharing one point array, via `write_mesh_with_blocks`. This is a deliberate
divergence from pyvista's 4-file layout — decided because it's the more natural way to use
xdmf's block API, not to give either side an advantage.

```python
writer = xdmf.TimeSeriesWriter(file_prefix, storage)  # storage: Ascii or Binary
data_writer = writer.write_mesh_with_blocks(points, connectivity, cell_types, blocks)
data_writer.write_data("0", [
    ("velocity", xdmf.DataAttribute.VECTOR, velocity),
    ("pressure", xdmf.DataAttribute.SCALAR, pressure),
], [])
```

Single time snapshot (not a time series) — decided because the benchmark is about
write/compress throughput, not time-series overhead. Storage modes: `Ascii`, `Binary`,
`Hdf5SingleFile`, `Hdf5MultipleFiles` at 1e3/1e5/1e7; **`Ascii` skipped at 1e7**
(impractically large/slow) — the other three run at every size, since HDF5 writing (unlike
ASCII) doesn't get slower in a way that makes 10M elements impractical.

On-disk layout: `{name}.xdmf2` (XML) + one of:
- `Binary`/`Ascii`: `{name}.bin/` or `{name}.txt/` (one file per array: `points.bin`,
  `block_{name}_cells.bin`, `data_t_0_point_data_{velocity,pressure}.bin`).
- `Hdf5SingleFile`: `{name}.h5` — a single file, mesh + data in internal groups.
- `Hdf5MultipleFiles`: `{name}.h5/` — a directory of files (`mesh.h5`, `data_t_0.h5`).

Note: the `Hdf5*` writers hold their `.h5` file(s) open for further writes, so the Python
object must be dropped (`del data_writer, writer`) before reading file sizes or zipping —
otherwise HDF5's file locking can make a concurrent read fail. `cfd_benchmark.py`'s
`run_xdmf` does this right after `write_data`.

## Writing: pyvista

**4 separate `.vtu` files** (domain + inlet + outlet + sides), each a self-contained
`UnstructuredGrid` built via the dict-of-arrays constructor, using `extract_submesh` to get
each boundary zone's local points/connectivity:

```python
grid = pv.UnstructuredGrid({pv.CellType.HEXAHEDRON: domain_conn}, points_3d)
grid.point_data["velocity"] = velocity_3d
grid.point_data["pressure"] = pressure
grid.save(path, binary=True)  # default: compression="zlib"
```

`pv.CellType` uses standard VTK ids (HEXAHEDRON=12, QUAD=9) — **distinct from**
`xdmf.CellType`, which uses XDMF/VTK-legacy discriminants (Hexahedron=9, Quadrilateral=5).
Don't mix them up.

## Metrics and the "final archive" methodology

For both tools, report: **write time**, **compress time**, **total time**, and **final
archive size** — the archive being what you'd actually ship/store, not an intermediate
per-file size.

**Critical for a fair comparison**: pyvista's `.save(..., binary=True)` already
zlib-compresses internally by default (confirmed via `compressor="vtkZLibDataCompressor"`
in the `.vtu` XML header). xdmf's plain `Binary` storage writes fully uncompressed raw
data — `Values::precision()` in `src/values.rs` hardcodes float precision to 8 bytes
always, and narrows uint precision to 4 bytes only for `Format::Binary`
(`Format::uint_precision()` in `src/xdmf_elements/data_item.rs`), but that's it: no
in-writer compression exists in the core crate for either storage mode.

This means the two tools' raw output already differ in kind, not just size, so:

1. **Don't** just zip xdmf's raw output with a strong codec and compare it to pyvista's
   *own* pre-compressed `.vtu` bundled with a weak codec (e.g. plain `zip`/deflate) — that
   over-credits xdmf for a codec choice pyvista never got to use.
2. **Do** apply the *same set* of external compressors, at the *final archive* step, to
   both: pyvista's real default output (already zlib-compressed `.vtu` files) AND xdmf's
   raw `Binary` output. I.e. `pyvista-write(zlib) -> {deflate,zstd,bzip2,lzma}` vs
   `xdmf-write(raw) -> {deflate,zstd,bzip2,lzma}`, same codec on both sides, same code
   path (`zipfile`/`tarfile` + the codec). This isolates "which pipeline produces a
   smaller/faster final archive" from "which codec did we only give one side."
3. Report **total time = write + final compression**, since that's what a user actually
   waits for.

Codecs worth testing at the final-archive step (available without extra deps beyond the
`zstd`/`xz` system binaries): `zipfile.ZIP_DEFLATED` (baseline, weak/fast), `zstd` CLI at a
moderate level (e.g. `-6`, with `--long` for cross-block matching — cheap insurance, didn't
change results here since the redundancy turned out to be short-range), `bz2.compress`
(mid ground), `lzma.compress` (slow but strongest). Don't bother with `zstd` levels beyond
~9-12 or `--long` window tuning unless a first pass shows it matters — diminishing/negative
returns were observed here (level 19 was both slower and worse than plain `lzma`).

## Findings: HDF5 storage modes (2026-07-18 run, all 3 cases, plain `zipfile.ZIP_DEFLATED`)

Once HDF5 storage was wired up for the Python bindings (see `PYTHON_BINDINGS_PLAN.md`),
`Hdf5SingleFile` and `Hdf5MultipleFiles` were added to `run_xdmf`'s loop in
`cfd_benchmark.py` (a `DATA_SUFFIX` dict maps each storage mode's on-disk suffix, replacing
the old `"Binary" ? ".bin" : ".txt"` special case). Full run:

| case | method       | write (s) | zip (s) | total (s) | raw (MB) | zip (MB) | ratio |
|:----:|:-------------|----------:|--------:|----------:|---------:|---------:|------:|
| 1e3  | binary       |     0.001 |   0.005 |     0.006 |     0.13 |     0.02 | 0.153 |
| 1e3  | hdf5 single  |     0.002 |   0.004 |     0.006 |     0.19 |     0.02 | 0.117 |
| 1e3  | hdf5 multi   |     0.002 |   0.004 |     0.005 |     0.18 |     0.02 | 0.117 |
| 1e5  | binary       |     0.029 |   0.288 |     0.317 |    11.19 |     1.53 | 0.136 |
| 1e5  | hdf5 single  |     0.019 |   0.309 |     0.328 |    15.61 |     1.57 | 0.101 |
| 1e5  | hdf5 multi   |     0.015 |   0.307 |     0.322 |    15.60 |     1.57 | 0.101 |
| 1e7  | binary       |     2.243 |  21.330 |    23.574 |   940.23 |   124.16 | 0.132 |
| 1e7  | hdf5 single  |     1.791 |  26.062 |    27.853 |  1308.64 |   129.79 | 0.099 |
| 1e7  | hdf5 multi   |     1.881 |  25.998 |    27.879 |  1308.64 |   129.78 | 0.099 |
| 1e7  | pyvista-vtu  |    18.017 |   5.686 |    23.703 |   186.82 |   100.04 | 0.536 |

(`ascii` skipped at 1e7 as before; full table including it at 1e3/1e5 in the earlier
findings section still holds. "hdf5 single"/"hdf5 multi" = `Hdf5SingleFile`/`Hdf5MultipleFiles`.)

- **Raw size**: HDF5 writes ~40% *more* raw bytes than `Binary` at the same case (1308.64 MB
  vs 940.23 MB at 1e7) — expected, since HDF5 has per-dataset/group metadata overhead
  (chunk headers, B-tree/object-header structures) that flat `Binary` files don't, and the
  core crate's HDF5 writer applies no dataset-creation filter (no `H5Pset_deflate`, no
  chunking tuned for compression — see `write_values`/`write_mesh` in `src/hdf5_writer.rs`),
  so every array is stored as a single fully-contiguous, uncompressed dataset.
- **Compressed size**: despite the larger raw input, the final zip is only ~4-5% bigger than
  `Binary`'s (129.8 MB vs 124.2 MB at 1e7) — `zipfile.ZIP_DEFLATED` eats most of HDF5's
  metadata overhead just fine, so the two storage modes end up close once compressed.
  `Hdf5SingleFile` and `Hdf5MultipleFiles` are indistinguishable in output size (same data,
  just split across a different number of files) and nearly identical in write/zip time.
- **Time**: HDF5 write itself is actually *faster* than `Binary`'s at 1e5/1e7 (e.g. 1.79s vs
  2.24s at 1e7) despite writing more bytes — plausibly HDF5's library-level buffered I/O
  beats the crate's own per-array `write_all` calls. But `zip_s` is correspondingly larger
  (more raw bytes to walk/deflate), so **total time is worse than `Binary`** at every size
  (27.9s vs 23.6s at 1e7) and still far worse than `pyvista-vtu`'s pre-compressed pipeline.
- **Bottom line (superseded below)**: at this point neither HDF5 mode looked like a win for
  the "smallest/fastest final archive" metric. That was before `src/hdf5_writer.rs` applied
  any compression — see the next section.

## Findings: HDF5 with compression (2026-07-18, same run, `shuffle()+deflate(6)` added to
every dataset in `src/hdf5_writer.rs`)

The "not attempted here" from the previous section got attempted: every dataset created by
`SingleFileHdf5Writer`/`MultipleFilesHdf5Writer` (`points`, `cells`, block connectivity,
and all point/cell attribute data) now goes through HDF5's byte-shuffle filter followed by
gzip/deflate at level 6 (zlib's own default, so it's a fair comparison point against
pyvista's default `zlibcompression`). This needed no new public API — it's baked into the
writer's dataset-creation calls, not user-configurable. Re-running the exact same benchmark:

| case | method       | write (s) | zip (s) | total (s) | raw (MB) | zip (MB) | ratio |
|:----:|:-------------|----------:|--------:|----------:|---------:|---------:|------:|
| 1e3  | binary       |     0.001 |   0.005 |     0.005 |     0.13 |     0.02 | 0.153 |
| 1e3  | hdf5 single  |     0.003 |   0.001 |     0.004 |     0.04 |     0.01 | 0.278 |
| 1e3  | hdf5 multi   |     0.003 |   0.001 |     0.004 |     0.04 |     0.01 | 0.283 |
| 1e5  | binary       |     0.029 |   0.284 |     0.313 |    11.19 |     1.53 | 0.136 |
| 1e5  | hdf5 single  |     0.107 |   0.005 |     0.112 |     0.34 |     0.10 | 0.306 |
| 1e5  | hdf5 multi   |     0.102 |   0.005 |     0.106 |     0.34 |     0.10 | 0.308 |
| 1e7  | binary       |     2.220 |  21.347 |    23.567 |   940.23 |   124.16 | 0.132 |
| 1e7  | hdf5 single  |    10.450 |   0.450 |    10.900 |    37.68 |     8.96 | 0.238 |
| 1e7  | hdf5 multi   |    10.388 |   0.448 |    10.836 |    37.68 |     8.96 | 0.238 |
| 1e7  | pyvista-vtu  |    17.769 |   5.951 |    23.720 |   186.82 |   100.04 | 0.536 |

("hdf5 single"/"hdf5 multi" = `Hdf5SingleFile`/`Hdf5MultipleFiles`, same as above.)

- **Raw size collapses**: at 1e7, HDF5's raw output drops from 1,308.64 MB (uncompressed) to
  37.68 MB — a ~35x reduction, and now far smaller than `Binary`'s 940.23 MB raw output too.
  This dataset is a synthetic structured duct (parabolic velocity profile, linear pressure
  drop, regular hex grid), so it's unusually compressible — real CFD results will compress
  less, but the shuffle filter specifically targets exactly this kind of smooth, structured
  float data (it reorders bytes so each output byte position holds the same significance
  byte across all values, which turns smoothly-varying floats into long runs deflate can
  chew through), so real gains should still be substantial, just not this extreme.
- **Total time now wins outright at scale**: 1e7 total time drops from 27.9s (uncompressed
  HDF5) to 10.9s — cheaper than both `Binary`'s 23.6s and `pyvista-vtu`'s 23.7s, *despite*
  `write_s` itself going up 5-6x (1.8s to 10.4s, compression is real CPU work at write time).
  It wins because there's now almost nothing left for the final zip pass to do (0.45s vs
  21.3s for `Binary`) — the expensive compression happened once, in the writer, on raw
  values, instead of twice (a redundant final zip pass on already-small data).
- **At small/medium sizes** (1e3, 1e5) HDF5 is now also the best or tied-best on both total
  time and size, though the effect is proportionally smaller — 1e5 total time drops from
  0.32s to 0.11s.
- **`Hdf5SingleFile` vs `Hdf5MultipleFiles`**: still indistinguishable in output size and
  time, as before — the choice between them should be driven by other concerns (single
  portable file vs. easy per-timestep file management), not compression.
- **Updated bottom line**: with `shuffle()+deflate(6)` in place, HDF5 is now the strongest
  xdmf storage mode for this benchmark's metric, especially at scale, and it's the first
  xdmf mode to beat `pyvista-vtu` on both time and size simultaneously at 10M elements. The
  level (6) and filter choice are hardcoded, not tunable — `Blosc`/`zstd` filters (available
  in `hdf5-metno` behind Cargo features, not currently enabled) or a higher deflate level
  are possible follow-ups if this compression ratio/speed trade-off ever needs adjusting.
- **Blosc follow-up tried and reverted (2026-07-19)**: swapped `shuffle()+deflate(6)` for
  `blosc_zstd(9, true)` (`hdf5-metno`'s `blosc-zstd` feature; the `blosc-src` dependency
  builds C-Blosc from source via the `cc` crate, no system lib/CMake needed, so this part of
  the build was not the problem) and opened the resulting `.h5` in both local ParaView
  installs (5.13.2 and 6.1.1, see memory `paraview-install-locations` for where they live and
  how to test headlessly with `pvpython`). **Both fail**: neither ships the Blosc HDF5 filter
  plugin, so reads error with `required filter 'blosc' is not registered` and the reader then
  segfaults on teardown instead of failing cleanly. Setting `HDF5_PLUGIN_PATH` to the Python
  `hdf5plugin` package's prebuilt `libh5blosc.so` gets past the "not registered" error, but
  decompression itself then fails (`Blosc decompression error`) and the process hangs —
  apparently an incompatibility between `hdf5-metno`'s vendored Blosc filter build and
  `hdf5plugin`'s prebuilt one, not just a missing-plugin problem. **Conclusion: don't adopt
  Blosc for this writer unless/until this is resolved and re-verified** — `shuffle()+deflate()`
  is the only filter choice confirmed to open in stock ParaView with no extra setup. The
  experimental change was reverted; `src/hdf5_writer.rs`/`Cargo.toml` are back to the
  `shuffle()+deflate(6)` state described above.

## Findings: pseudo-random (noisy) field data (2026-07-18, `random_benchmark.py`)

Every finding above uses `mesh_gen.build_case`'s fields: a parabolic velocity profile and a
linear pressure drop — smooth and spatially correlated, i.e. close to a best case for
shuffle+deflate. `random_benchmark.py` (new; see "File layout") writes the *same* mesh via
the *same* harness (`bench_common.py`, factored out of `cfd_benchmark.py` for this reuse —
no behavioral change, verified byte-identical output before/after) but replaces
`velocity`/`pressure` with independent per-point noise from a seeded
`np.random.default_rng(42)` (`SEED` in the script), at realistic magnitudes (velocity
fluctuations around 0 m/s, pressure around atmospheric ± 500 Pa) so results stay comparable
to the smooth case — reproducible: re-running the script writes byte-identical output
(verified via `md5sum` on a written array before writing this up).

| case | method       | write (s) | zip (s) | total (s) | raw (MB) | zip (MB) | ratio |
|:----:|:-------------|----------:|--------:|----------:|---------:|---------:|------:|
| 1e3  | binary       |     0.001 |   0.006 |     0.007 |     0.13 |     0.06 | 0.461 |
| 1e3  | hdf5 single  |     0.004 |   0.002 |     0.006 |     0.08 |     0.05 | 0.620 |
| 1e3  | hdf5 multi   |     0.003 |   0.002 |     0.005 |     0.07 |     0.05 | 0.638 |
| 1e5  | binary       |     0.029 |   0.454 |     0.483 |    11.19 |     5.14 | 0.459 |
| 1e5  | hdf5 single  |     0.208 |   0.081 |     0.289 |     3.61 |     3.38 | 0.938 |
| 1e5  | hdf5 multi   |     0.210 |   0.082 |     0.292 |     3.60 |     3.38 | 0.938 |
| 1e7  | binary       |     2.245 |  34.591 |    36.836 |   940.23 |   429.03 | 0.456 |
| 1e7  | hdf5 single  |    19.369 |   6.900 |    26.269 |   313.03 |   284.98 | 0.910 |
| 1e7  | hdf5 multi   |    19.123 |   7.049 |    26.171 |   313.02 |   284.98 | 0.910 |
| 1e7  | pyvista-vtu  |    27.654 |  20.042 |    47.695 |   603.25 |   419.30 | 0.695 |

(`ascii` skipped at 1e7 as before; "hdf5 single"/"hdf5 multi" as above.)

- **This is a much harder case for compression, as intended**: HDF5's own ratio jumps from
  ~0.10-0.24 (smooth case) to ~0.62-0.94 — barely shrinks at all past ~1e5 (`0.938` means the
  final zip pass squeezes out under 7% more). Independent per-point noise has no
  cross-point redundancy for either the shuffle filter or deflate to exploit; the residual
  compressibility that remains comes only from IEEE-754 floats sharing similar
  magnitude/exponent bytes (bounded value range), not from the data's structure.
- **HDF5 is still the size and time winner at scale, just by a smaller margin**: at 1e7,
  final size is 284.98 MB (hdf5) vs 429.03 MB (`binary`) vs 419.30 MB (`pyvista-vtu`) — HDF5
  wins because its *internal* deflate pass (at write time, over genuinely raw uncompressed
  values) still finds more redundancy than a second-pass zip over already-formatted output
  can, even on noisy data. Total time: 26.3s (hdf5) vs 36.8s (`binary`) vs 47.7s
  (`pyvista-vtu`) — HDF5 wins on both metrics simultaneously again, though by a much smaller
  margin than the smooth case (was 10.9s/37.7MB vs 23.6s/940MB there).
- **The real cost shows up as write-time CPU, not output size**: HDF5 `write_s` jumps from
  ~1.8-10.4s (smooth case) to ~19.1-19.4s at 1e7 — deflate has to work much harder to find
  the same (smaller) amount of redundancy in noisy input. This is the trade-off implied but
  not measured in the compressed-HDF5 section above: compression cost scales with how hard
  the data fights back, not just with data size.
- **Bottom line**: the smooth-CFD numbers were a best case, not representative of noisy
  field data, but the qualitative conclusion holds even in this harder, more realistic
  setting — HDF5 with `shuffle()+deflate(6)` remains the best xdmf storage mode for total
  time and final size at scale. The margin over `Binary`/`pyvista-vtu` shrinks considerably
  (from ~2x to a much narrower gap) once the data stops cooperating.

## Findings: Blosc (2026-07-19, `cfd_benchmark.py` + `random_benchmark.py`, `blosc_zstd`)

Followed up on the "possible follow-up" note above: temporarily swapped
`shuffle()+deflate(6)` for `.blosc_zstd(level, true)` (`hdf5-metno`'s `blosc-zstd` feature;
`blosc-src` builds C-Blosc from source via the `cc` crate, no system lib needed) and re-ran
both benchmark scripts through the Python bindings (`maturin develop --release`). **Not
adopted** — reverted immediately after, see the ParaView-compatibility finding below the
table. Numbers are still useful for understanding the trade-off if this is revisited.

1e7-element case only (smaller cases show the same shape, just noisier at this timescale):

| field  | method              | write (s) | zip (s) | total (s) | raw (MB) | zip (MB) | ratio |
|:-------|:--------------------|----------:|--------:|----------:|---------:|---------:|------:|
| smooth | shuffle+deflate(6)  |     10.45 |    0.45 |     10.90 |    37.68 |     8.96 | 0.238 |
| smooth | blosc_zstd(9)       |     65.59 |    0.58 |     66.16 |    29.72 |    13.23 | 0.445 |
| smooth | blosc_zstd(5)       |      4.85 |    0.78 |      5.63 |    40.25 |    21.09 | 0.524 |
| smooth | binary              |      2.23 |   21.73 |     23.96 |   940.23 |   124.16 | 0.132 |
| smooth | pyvista-vtu         |     18.13 |    5.98 |     24.11 |   186.82 |   100.04 | 0.536 |
| noisy  | shuffle+deflate(6)  |     19.37 |    6.90 |     26.27 |   313.03 |   284.98 | 0.910 |
| noisy  | blosc_zstd(5)       |      6.53 |    7.77 |     14.29 |   322.45 |   302.66 | 0.939 |
| noisy  | binary              |      2.23 |   35.66 |     37.89 |   940.23 |   429.03 | 0.456 |
| noisy  | pyvista-vtu         |     28.17 |   20.67 |     48.84 |   603.25 |   419.30 | 0.695 |

- **`blosc_zstd(9)` is strictly dominated by `deflate(6)`** on this data — slower *and* bigger
  (66.2s/13.23MB vs 10.9s/8.96MB). Max Blosc compression level with zstd is not a free lunch;
  don't reach for it.
- **`blosc_zstd(5)` is a genuine speed/size trade, not a strict win or loss**: on the smooth
  case it writes ~2.2x faster than `deflate(6)` in total (5.63s vs 10.90s) but produces a
  final archive ~2.35x *larger* (21.09MB vs 8.96MB) — Blosc's block-parallel design trades
  compression ratio for throughput, and at level 5 that trade lands well below deflate's
  ratio on this smooth, highly-structured data.
- **On noisy data the trade looks much better for Blosc**: total time drops 46% (14.29s vs
  26.27s) for only a 6% larger final archive (302.66MB vs 284.98MB) — makes sense, since
  there's little structure for *either* codec to exploit, so deflate's extra ratio on smooth
  data (its main advantage) mostly evaporates, while Blosc's multithreaded throughput
  advantage doesn't. If write-time CPU cost genuinely matters more than a few percent of
  final size, and the data resembles real (noisy) simulation output more than this repo's
  synthetic smooth fields, Blosc's speed edge is real.
- **None of this matters yet: Blosc output doesn't open in ParaView.** Tested both local
  installs (5.13.2, 6.1.1 — see memory `paraview-install-locations`) via headless `pvpython`:
  neither bundles the Blosc HDF5 filter plugin (`required filter 'blosc' is not registered`,
  followed by a segfault on teardown). Pointing `HDF5_PLUGIN_PATH` at the `hdf5plugin` Python
  package's prebuilt `libh5blosc.so` gets past that error but then fails to actually decompress
  (`Blosc decompression error`) and hangs. **Until that's resolved, the speed numbers above are
  academic** — `shuffle()+deflate(6)` is the only option that's actually usable end-to-end.

## Findings: tuning deflate level (2026-07-19, `cfd_benchmark.py` + `random_benchmark.py`)

Cheap thing to check before looking elsewhere: does moving `DEFLATE_LEVEL` off the current `6`
(zlib's own default) buy anything for free? Tried both directions — `9` (max, does raising it
close the size gap?) and `3` (low, does lowering it trade size for speed?) — temporarily
editing the constant and re-running both benchmark scripts each time, then reverting. Both
stay within core HDF5 (`shuffle()`+`deflate()`), so there's no ParaView-compatibility question
here unlike Blosc — this is a pure speed/size dial.

1e7-element case only:

| field  | level | write (s) | zip (s) | total (s) | raw (MB) | zip (MB) | ratio |
|:-------|------:|----------:|--------:|----------:|---------:|---------:|------:|
| smooth |     3 |      7.15 |    0.59 |      7.75 |    50.22 |    11.24 | 0.224 |
| smooth |     6 |     10.45 |    0.45 |     10.90 |    37.68 |     8.96 | 0.238 |
| smooth |     9 |     11.79 |    0.51 |     12.30 |    37.51 |     8.95 | 0.239 |
| noisy  |     3 |     13.60 |    7.31 |     20.91 |   326.00 |   288.64 | 0.885 |
| noisy  |     6 |     19.37 |    6.90 |     26.27 |   313.03 |   284.98 | 0.910 |
| noisy  |     9 |     96.72 |    7.29 |    104.01 |   312.37 |   284.34 | 0.910 |

- **Level 9 (ceiling): no real gain, ever.** On smooth data, 8.95MB vs 8.96MB (0.1% smaller)
  for ~13% slower writes. On noisy data, write time blows up 5x (96.72s vs 19.37s) for a 0.2%
  size reduction. Level 6 is already compressing smooth data near-optimally, and noisy data is
  mostly incompressible, so extra effort just burns CPU hunting for matches that aren't there.
  Intermediate levels 7/8 aren't worth testing — the ceiling already shows the size is
  saturated.
- **Level 3 (floor): a genuine, asymmetric trade.** On smooth data it's a real trade, not a
  free win — writes ~32% faster (7.15s vs 10.45s) but the archive is ~25% *larger* (11.24MB vs
  8.96MB). On noisy data it's close to a free win — writes ~30% faster (13.60s vs 19.37s,
  20% faster total) for only ~1.3% larger output (288.64MB vs 284.98MB). Makes sense: on noisy
  data there's little structure for *any* deflate level to find, so the extra compression
  effort in level 6 buys almost nothing — level 3 gets most of the same ratio for much less
  CPU. On smooth, highly-structured data, that extra effort is what's finding the real
  redundancy, so backing off costs real size.
- **Conclusion: `3` is the new default, not `6`.** The synthetic "smooth" case is the best case
  for deflate's compression ratio and probably the least representative of real CFD output —
  real velocity/pressure fields carry solver noise, floating-point roundoff, and turbulence
  fluctuations, so they behave more like the "noisy" case, where level 3's cost is nearly free.
  Write speed matters more than a few percent of size for the common case, so the library
  default (`hdf5_writer::DEFAULT_DEFLATE_LEVEL`) was changed from `6` to `3`. Since the two
  field types still pull in different directions, `deflate_level` is also caller-configurable
  for cases that skew smooth/structured and want the old ratio back:
  `DataStorage::Hdf5SingleFile { deflate_level: Option<u8> }` /
  `DataStorage::Hdf5MultipleFiles { deflate_level: Option<u8> }` in Rust (`None` = library
  default), and `xdmf.DataStorage.hdf5_single_file(level)` / `.hdf5_multiple_files(level)` in
  Python (the plain `.Hdf5SingleFile`/`.Hdf5MultipleFiles` attributes give the default). Blosc
  remains blocked on the ParaView plugin-compatibility issue documented above regardless of
  level.

## Findings (10M-element case, illustrative — re-run to get current numbers)

- xdmf genuinely writes **less raw data** than pyvista: xdmf's whole case (940 MB, all 4
  zones combined) is smaller than pyvista's domain zone **alone**, uncompressed (1,736 MB).
  The gap: VTK's `.vtu` needs explicit per-cell `offsets` (a redundant but real array) and a
  per-cell `types` byte, plus 8-byte (`int64`) connectivity indices where xdmf uses 4 bytes.
- That raw-size advantage does **not** reliably translate into a smaller or faster final
  archive. Applying the identical external codec to both pipelines' real output:

  | compressor | pyvista (zlib write + compress) | xdmf (raw write + compress) |
  |:-----------|--------------------------------:|-----------------------------:|
  | deflate    |                ~24.7s / 100.0 MB |              ~23.8s / 124.2 MB |
  | zstd-6     |                 ~21.9s / 78.0 MB |              ~8.7s / 104.6 MB |
  | bzip2      |                 ~34.2s / 74.4 MB |             ~78.8s / 82.8 MB |
  | lzma       |                 ~110.7s / 52.5 MB |            ~191.8s / 22.9 MB |

  pyvista wins on size in 3 of 4 codecs, because its zlib pass at write time already
  strips most of the easy redundancy (VTK's zero-padded `int64` connectivity compresses
  away almost for free — see next point), leaving a smaller, denser residual that a second
  pass, even a weak one, can still shave further. xdmf only wins decisively with `lzma`,
  where a single strong pass over genuinely raw data beats compressing twice through a
  weaker first codec (zlib discards redundancy lzma could have used had it seen the raw
  bytes). xdmf's write step is always ~8x faster (no compression happens there), so it
  wins on **total time** whenever the compressor itself is fast (zstd); it loses on total
  time whenever the compressor is slow (bzip2, lzma), because pyvista's second pass has
  much less data left to chew through.
- **Root cause of xdmf's worse-than-expected compression ratio**: packing connectivity
  tightly (`uint32`) removes free redundancy rather than adding useful data. Tested
  directly — the *same* connectivity index values, at the *same* zstd level:
  `uint32` (320 MB raw) -> 87.3 MB compressed; `int64` (640 MB raw, same values, just
  zero-padded) -> 49.4 MB compressed. For a 10M-node mesh, indices fit in ~24 bits, so
  `int64`'s top 4 bytes are always zero — pure, nearly-free compressible padding that a
  tightly packed format doesn't have. Packing tighter helps the raw file size but
  concentrates entropy, which *hurts* the post-compression size.
- No single tested option beats pyvista's default pipeline on both time and size at once;
  zstd-6 on xdmf is the best time/size trade if you're picking one thing to change.

## File layout

- `python/benchmarks/mesh_gen.py` — mesh/field generator (`build_case`, `extract_submesh`).
- `python/benchmarks/bench_common.py` — shared write/zip/report harness (`run_xdmf`,
  `run_pyvista`, `Result`, `print_summary`, etc.), factored out so `cfd_benchmark.py` and
  `random_benchmark.py` share it verbatim rather than duplicating ~150 lines; only the case
  (mesh + field values) passed in differs between the two driver scripts.
- `python/benchmarks/cfd_benchmark.py` — driver: runs xdmf (`Binary`, `Ascii`,
  `Hdf5SingleFile`, `Hdf5MultipleFiles`) and pyvista across the 3 cases with `mesh_gen`'s
  smooth fields, currently zips with plain `zipfile.ZIP_DEFLATED` only (the codec sweep
  above was done ad hoc against its output in scratch, not yet folded back into the script
  — see Next steps). Accepts an output dir as `sys.argv[1]` (defaults to
  `./benchmark_output`).
- `python/benchmarks/random_benchmark.py` — same mesh/harness, but `velocity`/`pressure` are
  replaced with independent per-point noise from a seeded `np.random.default_rng` (`SEED`
  constant in the script) at realistic magnitudes, so the run is a reproducible stand-in for
  noisy/measured field data rather than `mesh_gen`'s smooth, easily-compressible fields. See
  "Findings: pseudo-random (noisy) field data" above. Same CLI (`sys.argv[1]` output dir,
  defaults to `./random_benchmark_output`).

## Next steps (not yet done)

- Fold the multi-codec final-archive comparison (deflate/zstd/bzip2/lzma, applied
  identically to both tools' real output) into `cfd_benchmark.py` as a proper option/flag,
  rather than re-deriving it by hand in scratch each time.
- Consider whether `zstd`/`lzma` should be optional deps (system binary shellout, as done
  here, vs a Python package) if this becomes a tracked/repeatable benchmark rather than a
  one-off investigation.
