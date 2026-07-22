"""Writes a tiny xdmf/HDF5 time series used by the ParaView compatibility smoke test
(verify_with_pvpython.py). Run this with a *regular* Python that has the built `xdmf` wheel
installed (e.g. via `maturin develop`) -- the fixture is then read back by `pvpython`, which
has its own bundled interpreter and doesn't have `xdmf` (or necessarily numpy) available.

Usage: python write_fixture.py <output_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np

import xdmf


def main(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    coords = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        dtype=np.float64,
    )
    connectivity = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint64)
    cell_types = [xdmf.CellType.Triangle, xdmf.CellType.Triangle]

    file_path = output_dir / "paraview_smoke"
    writer = xdmf.TimeSeriesWriter(str(file_path), xdmf.DataStorage.Hdf5SingleFile)
    data_writer = writer.write_mesh(coords, connectivity, cell_types)

    expected = {"timesteps": []}
    for step, scale in enumerate([1.0, 2.0]):
        temperature = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64) * scale
        region_id = np.array([100, 200], dtype=np.uint64)
        data_writer.write_data(
            str(step),
            [("temperature", xdmf.DataAttribute.SCALAR, temperature)],
            [("region_id", xdmf.DataAttribute.SCALAR, region_id)],
        )
        expected["timesteps"].append(
            {
                "time": float(step),
                "temperature": temperature.tolist(),
                "region_id": region_id.tolist(),
            }
        )
    del data_writer

    # Path stored relative to expected.json's own directory, not as an absolute host path --
    # the verify step reads this file from inside a different (containerized) filesystem.
    expected["xdmf_file"] = file_path.with_suffix(".xdmf2").name
    expected["points"] = coords.reshape(-1, 3).tolist()

    (output_dir / "expected.json").write_text(json.dumps(expected, indent=2))
    print(f"Wrote fixture to {file_path.with_suffix('.xdmf2')}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
