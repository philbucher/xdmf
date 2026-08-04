//! Fixture generator for the `ParaView` compatibility smoke test (see `.github/workflows/paraview.yml`).
//! Writes a tiny XDMF time series with the requested `DataStorage` backend, plus an
//! `expected.json` recording the values written, so `tests/paraview_smoke/verify_with_pvpython.py`
//! can reopen the file in `ParaView` and check the two agree.
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

const COORDS: [f64; 12] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
const CONNECTIVITY: [u64; 6] = [0, 1, 2, 0, 2, 3];
const CELL_TYPES: [CellType; 2] = [CellType::Triangle, CellType::Triangle];
const REGION_ID: [u64; 2] = [100, 200];

#[derive(Serialize)]
struct ExpectedTimestep {
    time: f64,
    temperature: Vec<f64>,
    region_id: Vec<u64>,
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
    let mut xdmf_writer = xdmf_writer.write_mesh(&COORDS, (&CONNECTIVITY, &CELL_TYPES))?;

    let mut timesteps = Vec::new();
    for (step, scale) in [1.0, 2.0].into_iter().enumerate() {
        let temperature: Vec<f64> = [10.0, 11.0, 12.0, 13.0]
            .into_iter()
            .map(|v| v * scale)
            .collect();

        timesteps.push(ExpectedTimestep {
            time: step as f64,
            temperature: temperature.clone(),
            region_id: REGION_ID.to_vec(),
        });

        let point_data = [(
            "temperature".to_string(),
            (DataAttribute::Scalar, temperature.into()),
        )]
        .into_iter()
        .collect();

        let cell_data = [(
            "region_id".to_string(),
            (DataAttribute::Scalar, REGION_ID.to_vec().into()),
        )]
        .into_iter()
        .collect();

        xdmf_writer.write_data(&step.to_string(), Some(&point_data), Some(&cell_data))?;
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
