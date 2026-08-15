//! Fixture generator for the `ParaView` compatibility smoke test (see `.github/workflows/paraview.yml`).
//! Writes a tiny XDMF time series with the requested `DataStorage` backend, plus an
//! `expected.json` recording the values written, so `tests/paraview_smoke/verify_with_pvpython.py`
//! can reopen the file in `ParaView` and check the two agree. The mesh mixes cell types
//! (Quadrilateral + Triangle) and the data fields mix `DataAttribute` variants (Scalar, Vector,
//! Tensor, Tensor6) so the verification script can also confirm `ParaView` reads back the correct
//! number of components per field, not just the right numeric values.
//!
//! Usage: `cargo run --example paraview_smoke -- <output_dir> <storage>`
//! `<storage>` is any string accepted by `xdmf::DataStorage::from_str` (e.g. `Hdf5SingleFile`).

use std::{
    env,
    io::{Error as IoError, ErrorKind::InvalidInput, Result as IoResult},
    path::Path,
};

use serde::Serialize;
use xdmf::{CellType, DataAttribute, DataStorage, TimeSeriesWriter};

const NUM_POINTS: usize = 5;
const NUM_CELLS: usize = 2;

// a quad and a triangle sharing an edge, to exercise a mixed-cell-type mesh
const COORDS: [f64; NUM_POINTS * 3] = [
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.5, 0.0,
];
const CONNECTIVITY: [u64; 7] = [0, 1, 2, 3, 1, 4, 2];
const CELL_TYPES: [CellType; NUM_CELLS] = [CellType::Quadrilateral, CellType::Triangle];
const REGION_ID: [u64; NUM_CELLS] = [100, 200];

#[derive(Serialize)]
struct ExpectedTimestep {
    time: f64,
    temperature: Vec<f64>,
    displacement: Vec<[f64; 3]>,
    velocity_gradient: Vec<[f64; 9]>,
    region_id: Vec<u64>,
    stress: Vec<[f64; 6]>,
}

#[derive(Serialize)]
struct Expected {
    timesteps: Vec<ExpectedTimestep>,
    xdmf_file: String,
    points: Vec<Vec<f64>>,
}

fn main() -> IoResult<()> {
    let args: Vec<String> = env::args().collect();
    let [_, output_dir, storage_arg] = args.as_slice() else {
        return Err(IoError::new(
            InvalidInput,
            "usage: paraview_smoke <output_dir> <storage>",
        ));
    };

    let storage: DataStorage = storage_arg
        .parse()
        .map_err(|e| IoError::new(InvalidInput, e))?;

    let output_dir = Path::new(output_dir);
    let base_path = output_dir.join(format!("fixture_{}", storage_arg.to_lowercase()));

    let xdmf_writer = TimeSeriesWriter::new(&base_path, storage)?;
    let mut xdmf_writer =
        xdmf_writer.write_mesh(COORDS.as_slice().into(), &CONNECTIVITY, &CELL_TYPES)?;

    let mut timesteps = Vec::new();
    for (step, scale) in [1.0, 2.0].into_iter().enumerate() {
        let temperature: Vec<f64> = [10.0, 11.0, 12.0, 13.0, 14.0]
            .into_iter()
            .map(|v| v * scale)
            .collect();

        let displacement: Vec<[f64; 3]> = (0..NUM_POINTS)
            .map(|i| [i as f64 * 0.1 * scale, i as f64 * 0.2 * scale, 0.0])
            .collect();

        let velocity_gradient: Vec<[f64; 9]> = (0..NUM_POINTS)
            .map(|i| std::array::from_fn(|j| (i * 9 + j) as f64 * scale))
            .collect();

        let stress: Vec<[f64; 6]> = (0..NUM_CELLS)
            .map(|i| {
                let base = (i + 1) as f64 * scale;
                [
                    base,
                    2.0 * base,
                    3.0 * base,
                    4.0 * base,
                    5.0 * base,
                    6.0 * base,
                ]
            })
            .collect();

        timesteps.push(ExpectedTimestep {
            time: step as f64,
            temperature: temperature.clone(),
            displacement: displacement.clone(),
            velocity_gradient: velocity_gradient.clone(),
            region_id: REGION_ID.to_vec(),
            stress: stress.clone(),
        });

        let point_data = [
            ("temperature", DataAttribute::Scalar, temperature.into()),
            (
                "displacement",
                DataAttribute::Vector,
                displacement
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<f64>>()
                    .into(),
            ),
            (
                "velocity_gradient",
                DataAttribute::Tensor,
                velocity_gradient
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<f64>>()
                    .into(),
            ),
        ];

        let cell_data = [
            (
                "region_id",
                DataAttribute::Scalar,
                REGION_ID.to_vec().into(),
            ),
            (
                "stress",
                DataAttribute::Tensor6,
                stress
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<f64>>()
                    .into(),
            ),
        ];

        xdmf_writer.write_data(&step.to_string(), point_data, cell_data)?;
    }

    let xdmf_file_name = base_path
        .with_extension("xdmf2")
        .file_name()
        .ok_or_else(|| IoError::new(InvalidInput, "invalid output file name"))?
        .to_string_lossy()
        .into_owned();

    let expected = Expected {
        timesteps,
        xdmf_file: xdmf_file_name,
        points: COORDS.chunks_exact(3).map(<[f64]>::to_vec).collect(),
    };

    let expected_json =
        serde_json::to_string(&expected).map_err(|e| IoError::new(InvalidInput, e.to_string()))?;
    std::fs::write(output_dir.join("expected.json"), expected_json)?;

    #[expect(
        clippy::print_stdout,
        reason = "CLI progress output expected from an example binary"
    )]
    {
        println!(
            "Wrote fixture to {}",
            base_path.with_extension("xdmf2").display()
        );
    }

    Ok(())
}
