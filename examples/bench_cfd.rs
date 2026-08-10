//! Manual driver for the 1e7-cell case from `plans/02_performance.md` part A.
//!
//! Deliberately not part of `cargo bench` (`benches/write_time_series.rs`): criterion's default
//! sampling is unusable at this size, and a single run already takes minutes. Run explicitly:
//! `cargo run --release --example bench_cfd`.
//!
//! Reports wall time and raw bytes written per storage backend, in the table style
//! `CFD_BENCHMARK_PLAN.md` (`origin/multiple-features`) used. `Ascii`/`AsciiInline` are skipped
//! at this size, matching that document's precedent ("Ascii skipped at 1e7" — impractically
//! slow/large).
//!
//! **Not yet covered**: compressed-archive size. Part E's finding is that raw size can rank
//! backends in the wrong order once a codec is applied (`plans/02_performance.md` part E), so a
//! size claim from this driver alone should not be trusted — a codec sweep is follow-up work.

use std::{
    io::{Error as IoError, Result as IoResult},
    time::{Duration, Instant},
};

use xdmf::{DataAttribute, DataStorage, TimeSeriesWriter};

#[path = "../benches/common/mesh.rs"]
mod mesh;

use mesh::{CfdCase, build_case};

const NX: usize = 100;
const NY: usize = 100;
const NZ: usize = 1000;

fn fmt_duration(d: Duration) -> String {
    format!("{:.3}s", d.as_secs_f64())
}

struct Report {
    storage_label: &'static str,
    mesh_write_time: Duration,
    data_write_time: Duration,
    bytes: u64,
}

fn run_case(storage_label: &'static str, storage: DataStorage, case: &CfdCase) -> IoResult<Report> {
    let tmp_dir = temp_dir::TempDir::new()?;
    let cell_ids: Vec<f64> = (0..case.num_cells()).map(|i| i as f64).collect();

    let writer = TimeSeriesWriter::new(tmp_dir.path().join("cfd"), storage)?;

    let start = Instant::now();
    let mut ts_writer = writer.write_mesh(&case.points, &case.connectivity, &case.cell_types)?;
    let mesh_write_time = start.elapsed();

    let start = Instant::now();
    ts_writer.write_data(
        "0",
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
        ],
        [("cell_id", DataAttribute::Scalar, cell_ids.as_slice().into())],
    )?;
    let data_write_time = start.elapsed();

    // The HDF5 backends keep their file(s) open for further writes, so the writer must be
    // dropped before the on-disk size is measured (`CFD_BENCHMARK_PLAN.md` notes the same for
    // the Python bindings, which hold the same HDF5 handles).
    drop(ts_writer);

    let bytes =
        fs_extra::dir::get_size(tmp_dir.path()).map_err(|err| IoError::other(err.to_string()))?;

    Ok(Report {
        storage_label,
        mesh_write_time,
        data_write_time,
        bytes,
    })
}

#[expect(
    clippy::print_stdout,
    reason = "this example's entire purpose is a printed report table"
)]
fn main() -> IoResult<()> {
    println!("Building the 1e7 case ({NX}x{NY}x{NZ})...");
    let build_start = Instant::now();
    let case = build_case(NX, NY, NZ);
    println!(
        "  {} points, {} cells, built in {}\n",
        case.num_points,
        case.num_cells(),
        fmt_duration(build_start.elapsed())
    );

    let mut storages: Vec<(&str, DataStorage)> = vec![("binary", DataStorage::Binary)];
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
    } else {
        println!("(hdf5 feature disabled: skipping the two HDF5 storages)\n");
    }

    println!(
        "{:<22} {:>14} {:>14} {:>12}",
        "storage", "mesh write", "data write", "size"
    );
    for (label, storage) in storages {
        let report = run_case(label, storage, &case)?;
        println!(
            "{:<22} {:>14} {:>14} {:>12}",
            report.storage_label,
            fmt_duration(report.mesh_write_time),
            fmt_duration(report.data_write_time),
            humansize::format_size(report.bytes, humansize::DECIMAL)
        );
    }

    Ok(())
}
