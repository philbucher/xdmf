# Post-1.0 — MPI support (API draft)

`README.md`: *"MPI support! => for now I want to consider this and draft the API, so that we don't
miss anything. First with HDF, as this can already support MPI. With Ascii and Binary I would like to
write a single file, but that needs MPI-IO, which is currently not supported in rsmpi. I don't think
this will impact the API, but I want to mention it for completeness; in v1.1 we will only do MPI with
hdf."*

**This milestone produces a design, not code.** Its purpose is to make sure nothing in the 1.0 API
forecloses the parallel implementation. Decision 8 in `ROADMAP.md`: design around **one global grid,
with each rank writing a hyperslab** into shared HDF5 datasets.

## The chosen model

Every rank holds a piece of the mesh. The output is **one logical grid** — a single `points` dataset,
a single `connectivity` dataset, a single dataset per attribute per time step — with each rank writing
its own slice at a computed global offset. ParaView opens it as one seamless mesh, indistinguishable
from serial output.

The rejected alternative (one spatial-collection block per rank) needs no collectives and would reuse
the M4 block machinery, but it bakes the partitioning into the visualization: the user sees rank
boundaries as separate blocks and gets partition artifacts in filters that care about connectivity.
For a general-purpose output library the seamless result is the right default.

## API sketch

```rust
let writer = TimeSeriesWriter::new_parallel(file_name, storage, &comm)?;

let mut ts = writer.write_mesh(
    &local_points,          // this rank's owned points, flat xyz
    &local_connectivity,    // indices into *global* node numbering, see below
    &local_cell_types,
)?;

ts.write_data(time, point_data, cell_data)?;   // this rank's local slices
```

The parallel entry point is a **separate constructor**, not a parameter on the existing one — the
serial path must not grow an `Option<&Communicator>` it never uses, and the two have genuinely
different preconditions. Everything after construction is the same API, taking local data.

Feature-gated: `mpi = ["dep:mpi"]`, off by default. `rsmpi` requires an MPI installation and
`libclang` for bindgen, which must not leak into the default build or into the Python wheels
(`06_python_bindings.md`).

## The central open question: global node numbering

**This is the thing that must not be missed, and it is the reason to draft now.**

Connectivity indices are local on each rank but must be global in the file. Complicating it: with
ghost/halo nodes, a node can exist on several ranks, so "the union of all ranks' points" is not the
global point set.

Three sub-problems:

1. **Who owns a shared node.** If every rank writes all its points, shared nodes are duplicated: the
   mesh gets seams (visually mostly fine, filters less so) and the point arrays grow.
2. **What index a cell refers to.** A rank's cell references a node that may be owned by a neighbour.
3. **How the writer learns any of this.**

**Recommendation:** the parallel `write_mesh` takes an explicit global node id per local point, plus
enough information to determine ownership:

```rust
pub fn write_mesh_parallel(
    self,
    points: &[f64],
    global_node_ids: &[u64],    // one per local point; ownership derived (lowest rank wins) or explicit
    connectivity: &[u64],       // in terms of global_node_ids
    cell_types: &[CellType],
) -> Result<TimeSeriesDataWriter>
```

Rationale: any code that already runs in parallel *has* this information — arotau tracks ghost nodes
explicitly (`is_ghost()` appears in `arotau-core/src/output/xdmf.rs`). Reconstructing it inside the
writer would require geometric matching or an all-to-all, both worse than being told. Deriving
ownership from the ids (lowest rank owning a duplicated id) avoids a second array in the common case;
an explicit ownership mask can be added if someone needs a different rule.

The alternative — the writer assigns global ids by exscanning local point counts — is simpler but is
only correct when no node is shared. That is not the common case in FEM/CFD, so it should not be the
design.

**Point data follows ownership:** each rank contributes values only for the nodes it owns, otherwise
ghost values collide at the same global offset. This must be validated, not assumed — a
size mismatch between "local points" and "owned points" is the most likely user error in the whole
parallel API.

## Collectives the writer performs internally

Per phase, and each one is a place a bug becomes a deadlock rather than an error:

- **Sizing.** `Exscan` on owned-point count and on prepared-connectivity length → this rank's offset;
  `Allreduce` → global totals. Same per attribute at each step.
- **Dataset creation.** HDF5 requires *all* ranks to call dataset creation collectively with the same
  global shape. This means every rank must know every attribute's global size before any rank writes,
  which in turn means the set of attributes and their order must be identical on all ranks.
- **Consistency validation.** Ranks disagreeing about the number, names, order, or types of attributes
  — or about the time value — is a user error that otherwise manifests as a hang. Check it explicitly:
  `Allreduce` a hash of `(time, attribute names, attribute types)` and fail with a clear error naming
  the disagreement. Cheap; prevents the single most frustrating class of parallel bug.
- **Error propagation.** Any rank failing must not leave the others waiting in a collective.
  `Allreduce(max)` on an error flag after each phase, so all ranks fail together. Easy to forget,
  impossible to retrofit cleanly. **Design it in from the start.**

## Light data

Written by rank 0 only. The `Dimensions` in every `DataItem` are the *global* sizes, so
`TimeSeriesDataWriter` needs global `num_points`/`num_cells` for the XML while validation uses local
counts. Worth noting because the current struct conflates the two (`src/time_series_writer.rs:244`).

The streaming tail-patching writer from `02_performance.md` part B composes with this without change —
it is rank-0-local file I/O.

## HDF5 requirements

- `hdf5-metno` built with the `mpio` feature, against a parallel-enabled HDF5.
- Collective vs. independent transfer mode is a tuning knob (collective is usually better for the
  large contiguous writes here); benchmark rather than guess.
- **Compression and parallel writes interact badly.** The `shuffle() + deflate()` filters this crate
  relies on require chunked datasets, and filtered writes to a chunk touched by more than one rank
  are not supported in independent mode and are expensive in collective mode. Either align chunk
  boundaries with rank boundaries (constrains chunk size, which `02_performance.md` part E is
  otherwise free to tune) or accept uncompressed parallel output. **This is the second thing that must
  not be missed** — the compression is what makes the HDF5 backend win, and it is exactly what
  parallel writing puts at risk. Investigate before committing to the design.
- Reading is unaffected: stock serial ParaView reads a parallel-written HDF5 file normally.

## Ascii and Binary

No MPI-IO in `rsmpi`, so a single shared file is off the table for now — as `README.md` says. But
there is a good interim that needs **no MPI-IO at all** and is strictly better than what exists today:

- Each rank writes its own binary/ascii files (it already can — the backends are per-file).
- Rank 0 writes **one** XDMF file describing a spatial `Grid` collection, one sub-grid per rank,
  referencing the per-rank data files.

That is one openable file instead of N, and it reuses the M4 block machinery directly. It produces the
partitioned (blocky) view rather than the seamless one, which is the acknowledged trade for not having
MPI-IO. Worth doing in v1.1 alongside the HDF5 path, and it is a strict improvement on arotau's current
workaround (`_rank{N}` filename suffixes producing N independent XDMF files).

## Verification strategy — design it now

- CI job with OpenMPI from apt, `mpirun -n 4` writing a decomposed cube, then `pvpython` asserting the
  result is **identical to the serial output** for the same mesh: same point count, same cell count,
  same field values. That equivalence is the correctness criterion and it is only meaningful if the
  serial path is the reference — another reason the seamless-global-grid model was chosen.
- Test with a partition that has shared nodes, not only a trivially disjoint one; the shared-node case
  is where the design is actually being tested.
- Test the mismatched-attributes and mismatched-time cases, asserting a clean collective error rather
  than a hang, with a timeout in CI so a regression fails instead of hanging the runner.

## Does any of this constrain the 1.0 API?

Reviewed against the other plans — the answer is essentially no, with three notes to carry forward:

1. `TimeSeriesDataWriter` conflates global and local entity counts. Keeping `num_points`/`num_cells`
   as private fields (they are) means this can change without a breaking API change. **Do not expose
   them publicly at 1.0.**
2. The `DataWriter` trait's `write_mesh`/`write_data` signatures take whole slices and return a
   `DataContent`. A parallel writer needs offset and global-size parameters. The trait is
   crate-private, so it can be extended freely — **keep it crate-private**.
3. `06_python_bindings.md` proposes adding `Send` to `DataWriter` for GIL release. That is compatible
   with, and arguably a prerequisite for, anything parallel. No conflict.

The parallel path is otherwise additive: a new constructor, a new feature flag, no change to the
serial API.

## Explicitly out of scope

- Parallel *reading*. The reader (M5) is serial; a parallel reader is a separate design.
- Hybrid MPI + threads.
- Rebalancing or repartitioning on write — the writer takes the decomposition it is given.
