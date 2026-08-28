# M6 — Python bindings and PyPI wheels

> **Status (2026-08-18): Part 1 landed on `python-interface`, for the writer only.** The bindings
> were re-implemented against the current `main` API (not cherry-picked: the reference
> implementations on `origin/multiple-features` and `origin/reader` both predate the `TimeStep`
> builder of M7 and the `paraview.rs` value validation), and every *pre-merge* review finding below
> that still applies is addressed — see the "Review findings to fix on the way in" list, each of
> which now carries its resolution. **A post-merge review of the landed commit (`5166dce`) then
> found nine more; eight are fixed as of 2026-08-19 and one (6, message polish) is deliberately
> left alone — see "Post-merge review findings", each of which carries its resolution. Six of the
> fixes landed with the M4 work; 7 (the unchecked multiply in the core crate) was deferred behind
> it and done on this branch afterwards, where 6 was also reconsidered (an `i64` fix was written,
> then reverted back to `u8` to avoid duplicating the core's own bound and message).**
> **Status (2026-08-28): Part 2 (wheels on PyPI) is implemented** -- the `hdf5-static` feature, the
> abi3 build, `.github/workflows/release.yml` and the metadata, with the Linux spike measured; see
> Part 2's own status note. **The reader bindings and Part 3 (the pyvista re-run) are untouched.**
> Part 1 landed ahead of its place in the milestone order, so the ordering note below still
> holds in reverse: M2/M4/M5 will each change the Rust API this layer wraps, and this layer then has
> to follow.

`README.md`: *"Python interface, already exists in the 'multiple-features' branch. => this was
vibe-coded, needs to be double checked. Data must ideally not be copied when going from Rust <=>
Python"*, and *"Publishing wheels to pypi"*.

Decision 3 in `ROADMAP.md`: **static HDF5 built into the wheel**, so `pip install xdmf` gives working
HDF5 storage with no system library. Decision 7: the branch's `python/` is a **reference
implementation to review**, not commits to merge.

This is last because every earlier milestone breaks the Rust API this layer wraps.

## Part 1 — Review and re-land the bindings

The branch has `python/src/{lib,arrays,enums,error,writer}.rs` (~470 lines) plus `Cargo.toml`,
`pyproject.toml`, and `tests/test_basic.py` (8 tests). The architecture is sound and should be kept:

- Separate workspace crate (`python/`), `publish = false`, `crate-type = ["cdylib"]`, `[lib] name = "xdmf"`.
  Keeps `pyo3`/`numpy` off the crates.io `xdmf` dependency graph. Correct.
- `xdmf = { path = "..", default-features = false, features = ["hdf5"] }` — explicit rather than
  relying on the core crate's default features. Correct, and the comment explaining why should survive
  the re-land.
- Zero-copy numpy borrowing via `PyReadonlyArray1` + `Values::from(&[T])`. This is the requirement
  from `README.md` and it is met.

### Review findings to fix on the way in

These come from reading the branch code; each is a concrete change, not a general "review it".

**All of them are done** as of 2026-08-18, except 6 (blocks do not exist outside
`origin/multiple-features` yet — they are M4). 1 turned out to be obsolete rather than fixed, and 11
needed no change. What each resolution was is listed after the findings.

1. **`unsafe` in `arrays.rs`.** `UintArray::as_u64_slice` reinterprets `&[i64]` as `&[u64]` via
   `slice::from_raw_parts` after a sign check. The reasoning is correct, but the core crate has zero
   `unsafe` and `02_performance.md` proposes `#![forbid(unsafe_code)]` there. Use
   `bytemuck::cast_slice::<i64, u64>` instead — both are `Pod`, it is the same zero-cost
   reinterpretation, and the sign check stays. Removes the only `unsafe` in the project.
2. **`FloatArray::extract` swallows the real error.** Any extraction failure maps to
   `"expected a 1D numpy array with dtype float64"`, so a 2-D `float64` array reports a dtype problem.
   Report shape and dtype separately.
3. **`ValueGuard::extract` misreports `float32`.** It tries `FloatArray` (f64) then `UintArray`, so a
   `float32` array falls through and is rejected with *"expected uint64 or int64"*. `03_values_and_f32.md`
   fixes the substance (float32 becomes natively supported via `Values::F32`); the error path still
   needs to name the actual dtype it got.
4. **The GIL is held during writes.** The branch documents this as a deviation: `py.detach()` requires
   `Ungil`, hence `Send`, and `Box<dyn DataWriter>` has no `Send` bound. Fix it: add `Send` as a
   supertrait on the crate-private `DataWriter` trait. `AsciiWriter`/`AsciiInlineWriter`/`BinaryWriter`
   are trivially `Send`. **Verify `hdf5::File`** — if `hdf5-metno`'s handle is not `Send`, the HDF5
   backends cannot cross `detach` and the options are (a) keep the GIL only for HDF5, or (b) drop the
   idea. Check this early; a 10-second write that blocks every other Python thread is a real defect in
   a library whose whole selling point is large-data throughput.
   Dropping `#[pyclass(unsendable)]` follows from the same change.
5. **`cell_types` takes `Vec<PyCellType>`.** Fine for typical meshes but it copies and it is
   per-element. Accept a numpy `uint8`/`int64` array of raw codes as an alternative input, using the
   `CellType::from_code` mapping that `05_reader.md` adds anyway.
6. **Blocks take `Vec<usize>`** — same issue; accept numpy index arrays, which is how a caller
   producing blocks in numpy will actually have them.
7. **No `close()` / context manager.** The HDF5 backends hold their files open until the object is
   dropped, and `CFD_BENCHMARK_PLAN.md` already records tripping over this (`del data_writer, writer`
   before reading file sizes). Add `__enter__`/`__exit__` and an explicit `close()`, mapping to the
   `finish()` method that `02_performance.md` part B introduces on the Rust side. This is the single
   most user-visible ergonomic fix in the list.
8. **Error mapping needs rewriting** for the `Error` enum from `01_error_type.md`: validation variants
   → `ValueError`, `Error::Io` → `OSError`, `Error::Unsupported` → `NotImplementedError`,
   `Error::StorageRequiresFeature` → `RuntimeError`. Per-variant, not per-`ErrorKind`.
9. **No `.pyi` stubs.** Add them — an extension module with no stubs is invisible to every editor and
   type checker. `pyo3-stub-gen` or hand-written; hand-written is fine at this API size.
10. **Version skew.** `python/pyproject.toml` and `python/Cargo.toml` carry their own `version`,
    independent of the crate. Derive or check it in CI so a wheel cannot claim a version the crate
    does not have.
11. **pyo3/numpy 0.29** on the branch — re-check for the current release at implementation time; pyo3
    moves fast and the `allow_threads` → `detach` rename is exactly the kind of churn to expect.

**How each was resolved (2026-08-18).**

1. There is no `unsafe` and no `bytemuck`: `Values` has a variant per element type since M3, so an
   `int64` array is borrowed as `Values::I64` rather than reinterpreted as `u64`. The sign check the
   old code needed is the core crate's own connectivity validation.
2/3. One `describe()` helper names what the object actually is (`"a numpy array with dtype int16"`,
   `"list"`), and the accepted dtypes are listed per *position*: points (`float64`/`float32`),
   connectivity (the four index types), attribute data (all six). A dtype valid in one position and
   not in another therefore says which position it was wrong in.
4. `DataWriter` gained a `Send + Sync` supertrait (also required by pyo3, which asserts pyclass
   payloads are both), `hdf5::File` satisfies it, and `write_mesh`/`write_time_step` wrap the write
   in `Python::detach`. Measured: 4 threads writing 3 steps of 2M values each take 1.75 s against
   4.51 s sequentially (2.6×), so the GIL is genuinely released.
5. `cell_types` accepts a sequence of `CellType` *or* a numpy `uint8`/`uint64`/`int64` array of raw
   VTK codes. The code→`CellType` mapping is generated in `python/src/enums.rs` off the same variant
   list as the pyclass, so `xdmf::CellType::from_code` (a `05_reader.md` item) is not needed for it,
   and a `const` block pins the discriminants to the core enum's.
8. Per variant, onto the current `Error`: the five `Invalid*` variants → `ValueError`,
   `IntegerOutOfRange` → **`OverflowError`** (deliberately not `ValueError`: it is the one failure a
   caller may want to catch to react, e.g. by choosing another `DataStorage`), `Internal` →
   `RuntimeError`, `Io`/`Hdf5` → `OSError` (Python's `IOError` is an alias of it, so the
   reference implementation's distinction between the two was cosmetic).
10. `python/Cargo.toml` takes `version.workspace = true` and `pyproject.toml` declares
    `dynamic = ["version"]`, so the wheel version *is* the crate version and cannot skew.

**Deviations from this plan, and why.**

- **`write_data` became `write_time_step(time, point_data=None, cell_data=None)`**, taking all
  attributes of a step at once. The Rust API (M7, `08_write_data_builder.md`) hands a `TimeStep` to a
  closure so each attribute is written immediately and one buffer serves them all; `TimeStep` borrows
  its writer, which a pyclass cannot hold, and refilling one array for several fields is not how
  numpy is used anyway. The binding runs the closure itself over the arrays it borrowed, so a step is
  still all-or-nothing and the GIL is released for the whole step. A `TimeStep` pyclass would need a
  self-referential struct (i.e. the `unsafe` that finding 1 removed) to gain nothing.
- **The strict `[lints.clippy]` list moved to `[workspace.lints]`**, so it applies to the bindings
  crate too, and CI gained a `python-bindings` job (`cargo clippy -p xdmf-python`,
  `pip install ./python[test]`, `pytest`) — the other jobs build the root package only, so without it
  nothing would have checked this crate.
- **`is_hdf5_enabled()` is exposed**, since a Python caller picking a `DataStorage` has no
  `#[cfg]` to consult, and the fallback story in Part 2 depends on it.

**Tests:** `python/tests/test_writer.py`, 46 tests — all five storages, several time steps, dtype
round-trips for points/connectivity/data (including the `Precision` that ends up in the XML),
`(N, 3)` shapes, cell types as codes, the context manager releasing the HDF5 file, four threads
writing concurrently, every error mapping (including the three `paraview.rs` limits), and a stub test
that fails if the module grows a name `xdmf.pyi` does not declare.

### Post-merge review findings (2026-08-18) — resolved 2026-08-19

A review of the merged commit (`5166dce`, the one that added the bindings, #26), done in two independent
passes that were then reconciled. Both read the bindings against the core crate and then **built a
release wheel locally and reproduced each item against the installed module**, so every finding
below is a transcript, not a reading. The recorded output is from
`maturin build --release --compatibility linux` on CPython 3.12.

**All resolved on 2026-08-19**, eight by a fix and 6 by deciding against one — see the
resolutions after the list. Two items (1 and 7) are
silent-corruption bugs — output that is accepted, opens fine, and holds something other than what
the caller passed, which is the one failure mode `paraview.rs` exists to prevent. Ranked by what
will actually bite.

1. **A `(3, N)` point array is silently written as a wrong mesh.** `arrays.rs`'s module doc states
   that any dimensionality is accepted and only C-contiguity is checked. That is what makes the
   natural `(N, 3)` layout free — but the transposed "x-row / y-row / z-row" layout is *also*
   C-contiguous, so it is accepted and reinterpreted as interleaved xyz:

   ~~~text
   pts = [[0,1,1,0],[0,0,1,1],[0,0,0,0]]   # shape (3, 4), C-contiguous
   -> points.txt: 0 1 1 0  0 0 1 1  0 0 0 0
   -> points (0,1,1) (0,0,0) (1,1,0) (0,0,0)   # garbage geometry, no error
   ~~~

   Valid XDMF, opens in ParaView, wrong mesh. Fix: for `points`, require `ndim == 1 || shape[-1] == 3`
   — cheap, and it cannot reject anything the flat layout allows. Worth deciding at the same time
   whether `Vector`/`Tensor` attributes get the same check; points are the unambiguous case (always
   3 components) and the one a caller is most likely to have in column layout.

2. **`cell_types` rejects `int32`/`uint32` code arrays** (`python/src/enums.rs:145`). Only `uint8`,
   `uint64` and `int64` have arms; `np.array([4, 4], dtype=np.int32)` raises
   *"cell_types must be a sequence of xdmf.CellType values, or a numpy array of dtype uint8, uint64,
   or int64"* (verified). Two reasons this matters beyond taste: `meshio` hands back `int32`/`uint32`
   cell arrays, and `requires-python >= 3.9` + `numpy >= 1.21` allow NumPy 1.x on Windows, where the
   default integer dtype is `int32` — so the documented `write_mesh(pts, conn, np.array([4, 4]))`
   works on Linux and fails on Windows from identical source. The CI job builds the bindings on
   Linux only, so this cannot be caught there. It is also inconsistent with `IndexArray`, which takes
   all four 32/64-bit index types. Two extra arms (plus `u16`/`i16` if we want to be generous).

3. **A validation error in `write_mesh` permanently consumes the writer, then misreports why**
   (`python/src/writer.rs:79-84`). `self.inner.take()` runs *before* the arguments are extracted, so
   every binding-level rejection — wrong dtype, non-contiguous array, unknown or negative cell-type
   code — burns the writer:

   ~~~text
   w.write_mesh(coords.astype(np.uint64), conn, cts)
     -> ValueError: expected points as a numpy array with dtype float64 or float32, got ... uint64
   w.write_mesh(coords, conn, cts)              # the corrected retry
     -> RuntimeError: write_mesh was already called on this TimeSeriesWriter
   ~~~

   The second message is false, and in Python — unlike Rust, where `write_mesh(self)` makes the move
   visible at the call site — a caller reasonably expects a rejected call to leave the object usable.
   Fix: extract and validate first, `take()` only immediately before `writer.write_mesh(...)`. The
   core `write_mesh(self)` consuming the writer on *its own* errors is unavoidable; these checks are
   not.

   **This is `write_mesh` only — `write_time_step` is not affected** and needs no change.
   `PyTimeSeriesDataWriter` takes `self.inner.as_mut()` (`python/src/writer.rs:133`) rather than
   `take()`, so a rejected step leaves the writer usable and the time free. Verified against all six
   rejection classes (bad dtype, non-contiguous, not an array, numpy scalar, wrong length, second
   attribute bad) on one writer: after six consecutive failures it accepted `"0.0"` and `"1.0"`
   normally, no attribute leaked into the XML, and the case that failed *after* its first attribute
   had been written left no orphan heavy-data file — the core's `step.discard()` rollback reaches
   through the binding intact.

4. **`DataStorage` and `DataAttribute` are unhashable, uncopyable and unpicklable, and every pyclass
   reports `__module__ == "builtins"`** (`python/src/enums.rs:8`, `:181`, and the other pyclass
   attributes). `#[pyclass(eq)]` without `hash` leaves `tp_hash` NULL — CPython only inherits
   `tp_hash` when `tp_richcompare` is NULL too. Verified: `CellType` is hashable (it has `hash`), the
   other two raise `TypeError: unhashable type: 'builtins.DataStorage'`, and `copy.copy` /
   `pickle.dumps` fail with `cannot pickle 'builtins.DataStorage' object`. So `{DataStorage.Ascii:
   ".txt"}`, a `set` of storages, `functools.lru_cache` over one, and handing one to a
   `multiprocessing` worker all break — all natural things to do with what is otherwise an immutable
   value type. Fix: `hash` + `derive(Hash)` on both (needs `Hash` on the core `DataStorage`/
   `DataAttribute`), `module = "xdmf"` on every pyclass, and `__reduce__` if pickling is wanted.

5. **The contiguity rejection does not say which array was wrong** (`python/src/arrays.rs:16`,
   `:33-38`). `NOT_CONTIGUOUS` is a bare const, while every dtype message carries a `role`. With
   points, connectivity and N attributes in one call, *"array must be C-contiguous; call
   `numpy.ascontiguousarray()` on it first"* does not identify the offender — verified on a
   two-attribute step. `contiguous_slice` already has the call sites to thread `role` through, and
   the module doc's own goal is that a rejection names the real problem.

6. **`deflate_level: u8` splits one user mistake across two exception types**
   (`python/src/enums.rs:51`, `:59`). Verified: `hdf5_single_file(10)` raises
   `ValueError: invalid configuration: deflate level 10 is out of range, must be between 0 and 9` at
   writer construction, but `hdf5_single_file(-1)` and `hdf5_single_file(300)` raise
   `OverflowError: out of range integral type conversion attempted` from pyo3's `u8` conversion. The
   boundary between a good message and an opaque one sits at 255, not at 9. Fix: take an `i64` here
   and let the core's `validate_deflate_level` produce the one `ValueError`.

7. **Unchecked `num_entities * data_attribute.size()` — core crate** (`src/time_series_writer.rs:592`,
   reachable from `python/src/enums.rs`'s unvalidated `matrix(rows, cols)` / `generic(size)`). In a
   debug build this panics as `pyo3_runtime.PanicException`, which derives from `BaseException` and
   so escapes `except Exception`. In the **release wheel — what users actually get — it wraps**, and
   a size that wraps onto the real array length is accepted:

   ~~~text
   DataAttribute.generic(2**62)     on a 4-point mesh -> ValueError "size ... must be 0, but is 4"
   DataAttribute.generic(2**62 + 1) on a 4-point mesh -> ACCEPTED
     -> Dimensions="0 4611686018427387905 1" in the .xdmf2
   ~~~

   Only reachable with an absurd size, so low practical severity — but it is an unchecked multiply
   producing a corrupt file rather than an error, in the core crate. Fix: `checked_mul` there (which
   also covers the Rust API), optionally a bound on the two constructors.

8. **`describe()` calls a numpy *scalar* "a numpy array"** (`python/src/arrays.rs:21`). It branches
   on `.dtype` existing, which scalars also have, producing a self-contradictory message:

   ~~~text
   np.float64(1.0) -> ValueError: expected data of 'x' as a numpy array with dtype float64,
                      float32, uint64, uint32, int64, or int32, got a numpy array with dtype float64
   ~~~

   `arr[0]` yields exactly such a scalar, so this is an easy mistake to make and the message gives
   the user nothing to act on. Fix: only call it an array when it is an `ndarray`, otherwise report
   the Python type alongside the dtype.

9. **`cell_types!` does not actually enforce parity with `xdmf::CellType`** (`python/src/enums.rs:73-76`,
   `:115`). The comment claims "a cell type added to `xdmf::CellType` is a single edit here", but the
   `const` assert block and `From<PyCellType>` both iterate the *Python* variant list only — the
   `const` block pins the discriminants of the variants that are listed, not the completeness of the
   list. Add a variant to the core enum and Python silently cannot use it
   (`ValueError: unknown cell type code N`) with every CI job green. This is the one place the file
   departs from the core crate's own convention, where exhaustive matches over `Values` and in the
   `*_writer.rs` backends make the compiler point at each missing decision. Fix: generate a `match`
   over `xdmf::CellType` in the lookup direction.

**Smaller notes**, not worth a numbered item each:

- `__repr__` returns Rust `Debug`: `Hdf5SingleFile { deflate_level: None }`, `Matrix(2, 3)`.
  Informative, but not Python syntax and not round-trippable.
- `is_hdf5_enabled()`'s `false` branch, and the `else` in
  `test_is_hdf5_enabled_matches_the_hdf5_storages_working`, are unreachable: `python/Cargo.toml`
  hardcodes `features = ["hdf5"]` with no cargo feature to turn it off, so no wheel can be built
  without it. Either add the passthrough feature (Part 2's fallback story needs it anyway) or drop
  the branch.
- The `docs` CI job builds the root package only, so `-D warnings -D missing_docs` never covers
  `python/src/`.
- `test_type_stubs_cover_the_module_surface` reads the *source* `xdmf.pyi`, so it would not catch the
  stubs failing to ship. They do ship — the built wheel contains `xdmf/__init__.pyi` and
  `xdmf/py.typed` alongside `xdmf/__init__.py` and the `.so`, confirmed by inspecting it.

**What the review confirmed working**, so it does not need re-checking: all 46 pytest tests pass
against a locally built wheel; the wheel ships the stubs and `py.typed`; concurrent writers with the
GIL released produce correct files (verified with both 4 and 6 threads); `deflate_level` in `0..=9`
is validated at writer construction; the five `paraview.rs` limits surface as the documented
`OverflowError`/`ValueError`.

**How each was resolved (2026-08-19).** Verified the way they were found: rebuilt the wheel and
re-ran the reproduction for every item. The suite grew accordingly (from 46), each fix carrying its
own test — 6's covers what it does instead; 7's test is a Rust one, in `tests/time_series_writer.rs`,
since the bug is in the core crate.

1. `PointArray::validate_shape` (`python/src/arrays.rs`) rejects a trailing dimension that is not 3.
   Flat arrays are exempt — a 1-D array's only dimension is a count, not a component width — so
   `(12,)`, `(4, 3)` and `(2, 2, 3)` all still pass and `(3, 4)` names the fix
   (`numpy.ascontiguousarray(points.T)`). Points only: they are the unambiguous case at always 3
   components, whereas `Generic`/`Matrix` attributes have arbitrary ones.
2. `code_dtypes!` (`python/src/enums.rs`) generates the extraction over **every** integer dtype
   (`u8`/`u16`/`u32`/`u64`/`i8`/`i16`/`i32`/`i64`) off one list that also produces the message
   naming them, so the two cannot drift. "Any integer dtype" is a rule a caller can predict; the
   previous three-dtype list was not.
3. `write_mesh` (`python/src/writer.rs`) checks `self.inner.is_none()` up front — so a genuine
   second call is still reported as one — and `take()`s only inside the innermost dispatch arm,
   after every dtype, shape, contiguity and cell-type check has passed. A rejected call now leaves
   the writer usable.
4. `hash` + `derive(Hash)` and `module = "xdmf"` on all three pyclasses, which needed `Eq + Hash` on
   the core `DataStorage`/`DataAttribute` (`src/lib.rs`) — the only core change the bindings
   themselves needed (7 changes the core too, for a bug that is the core's own). **Pickling is
   deliberately not added:** a `__reduce__` needs a public reconstructor, which is API design rather
   than a fix, so `copy.copy`/`pickle.dumps` still raise. Worth revisiting if a caller wants to hand
   a `DataStorage` to a `multiprocessing` worker.
5. `contiguous_slice` takes a `role`, and `writer.rs`'s `data_role()` builds the same name the dtype
   message uses, so `"data of 'pressure' must be C-contiguous"` points at one array out of many.
6. **Not fixed, deliberately.** The fix was written (an `i64` parameter range-checked in the
   bindings) and then reverted: it made the bindings restate the core crate's own limit and its
   message, which `validate_deflate_level` (`src/lib.rs`) owns, and the only way to have Rust
   produce the error for -1/300 instead is to widen `DataStorage`'s `deflate_level` past the `u8`
   it should be. One duplicated bound and message is the worse trade for the two exception types.
   So `hdf5_single_file`/`hdf5_multiple_files` take a `u8` and pass it through: 10-255 reaches the
   core and comes back as its `ValueError` naming 0-9 (at `TimeSeriesWriter` construction, where
   the core validates), and -1/300 do not survive pyo3's argument conversion and are its
   `OverflowError`. Worth revisiting only if the core's field itself ever widens.
7. Deferred at the time — the fix lands in `src/time_series_writer.rs`, which the M4 submesh work
   was rewriting wholesale in the same tree. `main` independently landed the identical fix (word for
   word) while this branch carried its own copy, so rebasing this branch onto `main` merged the two
   without a conflict. `DataAttribute::size()` returns `Option<usize>` (`Matrix`'s own `n * m`
   is a caller-supplied product too, so it is `checked_mul`), and `write_attribute` rejects a
   component count that is zero or does not fit, and a `num_entities * size` that does not fit,
   with one `InvalidData` — before anything is written, like every other check there. Fixing it in
   the core crate covers the Rust API as well as `matrix(rows, cols)`/`generic(size)`.
   The zero case was found while fixing this one and is the same expression: `Generic(0)` made
   `exp_size` 0, which empty data matches, and `Values::dimensions`'s `len / size` then panicked
   with a division by zero — reachable from safe public API in *both* debug and release, unlike the
   wrap. Verified in release, the way the wrap was found: `generic(2**62 + 1)` on a 4-point mesh now
   raises `ValueError` instead of writing `Dimensions="0 4611686018427387905 1"`, and `generic(0)`
   raises instead of panicking.
8. `describe()` tells an array apart by casting to `PyUntypedArray` rather than by having a `dtype`,
   and reports anything else by its fully qualified type name — so a numpy scalar reads
   `got numpy.float64`, and a `list` still reads `got list`.
9. The `cell_types!` macro emits a second `const` block matching exhaustively over `xdmf::CellType`,
   bound as a `const fn` pointer so it is neither dead code nor callable. Verified by deleting a
   variant from the list: `error[E0004]: non-exhaustive patterns`. This is the property the core
   crate gets from its exhaustive matches over `Values`.

Of the smaller notes: `__repr__` still returns Rust `Debug`, the `is_hdf5_enabled()` false branch is
still unreachable, the `docs` job still skips `python/`, and the stub test still reads the source
`.pyi` — all left as recorded, none of them affecting what lands in a file.

### New surface to add

- **Reader bindings** (M5): `TimeSeriesReader`, `read_mesh` into numpy arrays, `read_step`, and a
  read-into-preallocated-array path for the large cases.
- **f32** (M3): `Values::F32` means `float32` numpy arrays become a first-class zero-copy input, and
  `with_reduced_precision()` should be exposed as a constructor keyword.
- **2-D array shapes.** Points as `(N, 3)` and vector fields as `(N, 3)` are the natural numpy layouts;
  the current bindings accept flat 1-D only. Accept both, since a C-contiguous `(N, 3)` array has
  exactly the flat memory layout the Rust API wants — so it stays zero-copy, it is purely a shape
  check. This removes a `reshape(-1)` from every realistic call site.

### Tests

Keep and extend `python/tests/test_basic.py` (currently 8 tests, covering both storage modes, `int64`
connectivity, negative-`int64` rejection, dtype rejection, non-contiguous rejection, double
`write_mesh`, and blocks). Add: context manager, reader round-trip, f32, `(N, 3)` shapes, every error
mapping, and a `h5py`-based content check (already done via `pytest.importorskip`, which is the right
pattern).

## Part 2 — Wheels on PyPI

> **Status (2026-08-28): implemented as planned, and the Linux spike below is measured.** What
> landed: `hdf5-static` on the core crate (`hdf5/static` + `hdf5/zlib`, i.e. HDF5 *and* zlib built
> from source), passed through by an `hdf5-static` feature on the bindings crate whose `hdf5`
> feature is now a passthrough too — so `--no-default-features` produces the fallback wheel, and
> `is_hdf5_enabled()`'s `false` branch is reachable at last (the `python-bindings` CI job runs both
> ways). `pyo3/abi3-py39`, `.github/workflows/release.yml` (tags + manual, five native runners, an
> sdist, publish on a tag through PyPI trusted publishing), a tag-vs-crate-version check, and the
> PyPI metadata in `pyproject.toml`, and an `include` list on both crates so the package (and the
> sdist maturin builds from it) carries only sources, readme, license and stubs.
>
> **The spike's five questions, answered on Linux x86_64 (host build, not the manylinux image):**
> 1. The static build works and needs only CMake on top of a C compiler — hence
>    `before-script-linux` installing it in the manylinux image.
> 2. zlib is built from source too (`libz-sys` static), so deflate needs no system zlib: the
>    resulting `.so` links `libc`/`libm`/`libgcc_s` and nothing else.
> 3. Build time is ~2 min for the whole release wheel including HDF5, so `sccache` in the action is
>    enough and no separate HDF5 cache is warranted.
> 4. The wheel is 2.5 MB. A non-issue.
> 5. The output is unchanged: `gzip` level 3 + `shuffle` on every dataset, superblock version 2,
>    read back by `h5py` (HDF5 2.0.0) — the vendored HDF5 is 2.2.0, and the file uses core filters
>    only, which is what ParaView needs. The whole Rust suite (275 tests) passes against the static
>    build, and the 86 pytest tests pass against the built wheel in a clean venv.
>
> **All five platforms verified in CI 2026-08-28**, then released: `v0.2.1` published
> `pip install xdmf` to PyPI through trusted publishing. Every wheel builds its own static HDF5,
> installs itself and passes the pytest suite -- Linux x86_64 2.52 MB, Linux aarch64 2.51 MB,
> macOS x86_64 2.14 MB, macOS arm64 1.92 MB, Windows x86_64 3.74 MB, sdist 0.14 MB. So the non-HDF5
> fallback wheel is not needed on any platform today. Of the 6.7 MB uncompressed extension module,
> 3.26 MB is HDF5 code against a 3.6 MB system `libhdf5.so` -- the wheel is small because it is a
> zip, not because the library is trimmed.

### Build configuration

- **abi3.** Build with `pyo3/abi3-py39` so one wheel per platform covers every Python ≥ 3.9 instead of
  one per (platform × Python version). This cuts the matrix by ~5× and is the difference between a
  CI job that is maintainable and one that is not. The bindings use nothing that abi3 forbids.
- **Platforms:** `manylinux_2_28` x86_64 + aarch64, macOS x86_64 + arm64, Windows x86_64. Plus an
  sdist.
- **Tooling:** `maturin-action` in `.github/workflows/release.yml`, triggered on tags and manually.
- **Publishing:** PyPI trusted publishing (OIDC), not a long-lived API token.

### The HDF5 problem — spike before building the matrix

This is the risk in the milestone. `hdf5-metno`'s `static` feature builds HDF5 from source with CMake.
Unknowns to settle in a **Linux-only spike, before any of the platform matrix is written**:

1. Does the `static` build work inside the manylinux image, and is `cmake` available or pip-installable
   there?
2. Does the static build vendor zlib, or does it need a static zlib alongside? Deflate is not optional
   here — it is the entire reason HDF5 storage wins (see `CFD_BENCHMARK_PLAN.md`).
3. What does it do to build time? A from-source HDF5 per platform per CI run is minutes, not seconds;
   caching strategy matters.
4. Wheel size — a statically linked HDF5 is not small. Check it is acceptable.
5. Does the resulting `.h5` output still open in stock ParaView? It should (the file format is the
   file format), but this is cheap to confirm and expensive to discover late.

Only after Linux is green, extend to macOS (both arches) and Windows — Windows CMake/HDF5 is
historically the awkward one, so budget for it separately.

**Fallback if static HDF5 proves impractical on some platform:** ship the non-HDF5 wheel there
(`Ascii`/`AsciiInline`/`Binary` still work, and `is_hdf5_enabled()` already exists to report it), and
document the gap. Do not block the whole release on the worst platform.

### Package name

Check `xdmf` is available on PyPI before anything else — the branch's `pyproject.toml` assumes it. If
taken, decide the alternative (`xdmf-rs` / `pyxdmf`) early, because it propagates into the module name,
the docs, and every example.

**It is free** (checked 2026-08-28), so the name stays `xdmf` and nothing propagates.

## Part 3 — Re-run the pyvista benchmark

Two prior records: `CFD_BENCHMARK_PLAN.md` on `origin/multiple-features` (the HDF5-with-compression
measurements) and `SESSION_2026-08-08_cfd_benchmark.md` (the external-codec sweep against pyvista, and
the entropy finding). Read both before re-running; they measure different things and only together do
they explain the results.

Re-run once M2 (performance) and M3 (f32) have landed:

- The `1e7` numbers should improve from the bulk binary encoding (`02_performance.md` part D) and from
  HDF5 chunk tuning (part E).
- **Do not expect f32 to be a straightforward win on the archive metric, and do not report it as one
  before measuring.** `03_values_and_f32.md` carries the measurement gate and the reasoning: halving
  the raw bytes is not the same as halving the compressed archive, and the same session demonstrated a
  case where a 2× smaller raw representation compressed 1.77× *worse*. The expectation for f64→f32 is
  the opposite sign (it discards high-entropy mantissa noise rather than low-entropy padding) — but
  that is a hypothesis to test, not a result to assume.
- **Set expectations for the headline comparison honestly.** At 1e7, xdmf's `Binary` output plus an
  external codec currently loses to pyvista on final size at deflate, zstd-6 and bzip2, and only wins
  with lzma. The answer to that is HDF5 with `shuffle`+`deflate`, which compresses the values at write
  time — it was the first xdmf mode to beat pyvista on time *and* size simultaneously at 10M elements.
  Make sure the re-run includes the HDF5 modes; the session-summary table only covers `Binary`.
- The pyvista comparison was never run in the wheel-installed configuration; do that here, since it is
  the configuration users will have.

Keep the fairness methodology exactly as documented — the same external codec set applied to each
tool's *real default* output, on both sides. Two variants were tried and discarded before that rule
stuck, and it is the reason the numbers are trustworthy. Likewise, verify the other tool's actual
behaviour before explaining a difference: pyvista uses `float64`/`int64` throughout this path and never
`float32`, and assuming otherwise already misdirected one round of analysis.
