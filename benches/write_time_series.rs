//! M2 performance benchmarks (`plans/02_performance.md` part A).
//!
//! `cargo bench` runs the groups below; criterion writes reports to `target/criterion/`. The
//! 1e7 case is deliberately not here — `examples/bench_cfd.rs` runs it manually and reports a
//! table, since criterion's default sampling is unusable at that size.

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use temp_dir::TempDir;
use xdmf::{DataAttribute, DataStorage, TimeSeriesWriter, Values};

#[path = "common/counting_alloc.rs"]
mod counting_alloc;
#[path = "common/mesh.rs"]
mod mesh;

use counting_alloc::CountingAllocator;
use mesh::{CfdCase, build_case};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// `(label, nx, ny, nz)`. Kept to 1e3/1e5 per `plans/02_performance.md` part A; 1e7 belongs in
/// `examples/bench_cfd.rs` instead.
const SIZES: &[(&str, usize, usize, usize)] = &[("1e3", 10, 10, 10), ("1e5", 10, 10, 1000)];

/// Extra time steps written before the timed `write_data` call in [`bench_write_data`], so the
/// measurement reflects steady-state per-step cost rather than the very first call.
const WARMUP_STEPS: usize = 10;

fn storages() -> Vec<(&'static str, DataStorage)> {
    let mut storages = vec![
        ("ascii", DataStorage::Ascii),
        ("ascii_inline", DataStorage::AsciiInline),
        ("binary", DataStorage::Binary),
    ];
    if xdmf::is_hdf5_enabled() {
        storages.push((
            "hdf5_single_file",
            DataStorage::Hdf5SingleFile {
                deflate_level: None,
            },
        ));
        storages.push((
            "hdf5_multiple_files",
            DataStorage::Hdf5MultipleFiles {
                deflate_level: None,
            },
        ));
    }
    storages
}

/// The generator's velocity (vector, point data) and pressure (scalar, point data) as
/// `write_data` attribute triples, borrowing `case` with no copy.
fn point_data(case: &CfdCase) -> [(&'static str, DataAttribute, Values<'_>); 2] {
    [
        (
            "velocity",
            DataAttribute::Vector,
            case.velocity.as_slice().into(),
        ),
        (
            "pressure",
            DataAttribute::Scalar,
            case.pressure.as_slice().into(),
        ),
    ]
}

/// A synthetic per-cell scalar (there is no cell data in the generator itself), to exercise the
/// cell-data path alongside `point_data`.
fn cell_data(cell_ids: &[f64]) -> [(&'static str, DataAttribute, Values<'_>); 1] {
    [("cell_id", DataAttribute::Scalar, cell_ids.into())]
}

#[expect(
    clippy::expect_used,
    reason = "benchmark harness: a failure here is a bug in the harness setup, not caller input"
)]
fn bench_write_mesh(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_mesh");
    for &(size_label, nx, ny, nz) in SIZES {
        let case = build_case(nx, ny, nz);
        for (storage_label, storage) in storages() {
            group.bench_function(format!("{storage_label}/{size_label}"), |b| {
                b.iter_batched(
                    || TempDir::new().expect("failed to create temp dir"),
                    |tmp_dir| {
                        let writer = TimeSeriesWriter::new(tmp_dir.path().join("mesh"), storage)
                            .expect("failed to create writer");
                        // no `black_box` needed: `write_mesh` performs real file I/O, an
                        // observable side effect the optimizer cannot elide regardless
                        writer
                            .write_mesh(&case.points, &case.connectivity, &case.cell_types)
                            .expect("failed to write mesh");
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

#[expect(
    clippy::expect_used,
    reason = "benchmark harness: a failure here is a bug in the harness setup, not caller input"
)]
fn bench_write_data(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_data");
    for &(size_label, nx, ny, nz) in SIZES {
        let case = build_case(nx, ny, nz);
        let cell_ids: Vec<f64> = (0..case.num_cells()).map(|i| i as f64).collect();

        for (storage_label, storage) in storages() {
            group.bench_function(format!("{storage_label}/{size_label}"), |b| {
                b.iter_batched(
                    || {
                        let tmp_dir = TempDir::new().expect("failed to create temp dir");
                        let writer = TimeSeriesWriter::new(tmp_dir.path().join("data"), storage)
                            .expect("failed to create writer");
                        let mut ts_writer = writer
                            .write_mesh(&case.points, &case.connectivity, &case.cell_types)
                            .expect("failed to write mesh");
                        for step in 0..WARMUP_STEPS {
                            ts_writer
                                .write_data(
                                    &step.to_string(),
                                    point_data(&case),
                                    cell_data(&cell_ids),
                                )
                                .expect("failed to write warmup step");
                        }
                        (tmp_dir, ts_writer)
                    },
                    |(_tmp_dir, mut ts_writer)| {
                        ts_writer
                            .write_data(
                                &WARMUP_STEPS.to_string(),
                                point_data(&case),
                                cell_data(&cell_ids),
                            )
                            .expect("failed to write timed step");
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

/// The O(steps^2) light-data cost (`plans/02_performance.md` part B): total time per run should
/// be flat *per step* after part B lands. Today it grows superlinearly. Criterion reports total
/// time per `n`; dividing by `n` gives the per-step cost to compare across `n`.
///
/// Deliberately uses a tiny mesh (2x2x2, not one of [`SIZES`]): part B's own estimate assumes a
/// "~1 KB grid fragment" per step, and the O(N^2) blowup is quadratic in *that* size too. A
/// first cut of this bench used the 1e3 case (real vector+scalar fields on ~1300 points) and
/// `steps_scaling/5000` alone ran for 28+ minutes of CPU time without finishing one iteration —
/// impractical for a bench meant to be run routinely. What this variant measures is still the
/// same effect (total time vs. `n`), just at a payload size the plan's own numbers assume.
const TINY_CASE_DIM: usize = 2;

#[expect(
    clippy::expect_used,
    reason = "benchmark harness: a failure here is a bug in the harness setup, not caller input"
)]
fn bench_steps_scaling(c: &mut Criterion) {
    // AsciiInline makes the effect starkest: part B notes the *entire* inline mesh text, not
    // just the per-step grid fragment, is re-cloned and re-serialized on every single step.
    let storage = DataStorage::AsciiInline;
    let case = build_case(TINY_CASE_DIM, TINY_CASE_DIM, TINY_CASE_DIM);
    let cell_ids: Vec<f64> = (0..case.num_cells()).map(|i| i as f64).collect();

    let mut group = c.benchmark_group("steps_scaling");
    group.sample_size(10);
    for &n in &[10_usize, 100, 1000, 5000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let tmp_dir = TempDir::new().expect("failed to create temp dir");
                    let writer = TimeSeriesWriter::new(tmp_dir.path().join("scaling"), storage)
                        .expect("failed to create writer");
                    let ts_writer = writer
                        .write_mesh(&case.points, &case.connectivity, &case.cell_types)
                        .expect("failed to write mesh");
                    (tmp_dir, ts_writer)
                },
                |(_tmp_dir, mut ts_writer)| {
                    for step in 0..n {
                        ts_writer
                            .write_data(&step.to_string(), point_data(&case), cell_data(&cell_ids))
                            .expect("failed to write step");
                    }
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

/// Reports allocations for one steady-state `write_data` call per storage backend — the metric
/// `README.md` asks for ("temporary allocations should be avoided as much as possible"), which
/// wall-clock time alone does not show. Printed rather than folded into a criterion measurement:
/// criterion's wall-clock `Measurement` isn't the right shape for a count, and a custom
/// `criterion::measurement::Measurement` is more machinery than warranted before part C exists to
/// actually change this number.
#[expect(
    clippy::expect_used,
    reason = "benchmark harness: a failure here is a bug in the harness setup, not caller input"
)]
#[expect(
    clippy::print_stdout,
    reason = "allocation counts are this bench's reported metric, not incidental debug output"
)]
fn bench_allocations_per_step(_c: &mut Criterion) {
    let (_, nx, ny, nz) = SIZES[0];
    let case = build_case(nx, ny, nz);
    let cell_ids: Vec<f64> = (0..case.num_cells()).map(|i| i as f64).collect();

    println!(
        "allocations_per_step ({} points, {} cells, one write_data call after a warmup step):",
        case.num_points,
        case.num_cells()
    );
    for (storage_label, storage) in storages() {
        let tmp_dir = TempDir::new().expect("failed to create temp dir");
        let writer = TimeSeriesWriter::new(tmp_dir.path().join("alloc"), storage)
            .expect("failed to create writer");
        let mut ts_writer = writer
            .write_mesh(&case.points, &case.connectivity, &case.cell_types)
            .expect("failed to write mesh");
        ts_writer
            .write_data("0", point_data(&case), cell_data(&cell_ids))
            .expect("failed to write warmup step");

        let (result, allocations) = CountingAllocator::count(|| {
            ts_writer.write_data("1", point_data(&case), cell_data(&cell_ids))
        });
        result.expect("failed to write timed step");
        println!("  {storage_label}: {allocations}");
    }
}

criterion_group!(
    benches,
    bench_write_mesh,
    bench_write_data,
    bench_steps_scaling,
    bench_allocations_per_step
);
criterion_main!(benches);
