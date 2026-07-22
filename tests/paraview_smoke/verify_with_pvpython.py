"""Run under `pvpython` (ParaView's bundled interpreter, not a regular Python) to check that a
xdmf/HDF5 time series -- written beforehand by write_fixture.py in a separate Python environment
that has the `xdmf` package -- actually opens and reads back correctly in ParaView. This is the
headless equivalent of the manual `pvpython` checks used to validate the HDF5 filter pipeline
(see the `paraview-install-locations` memory / CFD_BENCHMARK_PLAN.md).

Usage: pvpython verify_with_pvpython.py <expected.json>
"""

import json
import sys
from pathlib import Path

from paraview import servermanager
from paraview.simple import UpdatePipeline, XDMFReader
from vtk.util.numpy_support import vtk_to_numpy


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def main(expected_path: Path) -> None:
    expected = json.loads(expected_path.read_text())
    xdmf_file = expected_path.parent / expected["xdmf_file"]

    reader = XDMFReader(FileNames=[str(xdmf_file)])
    UpdatePipeline(time=expected["timesteps"][0]["time"], proxy=reader)

    data = servermanager.Fetch(reader)
    points = vtk_to_numpy(data.GetPoints().GetData())
    if points.tolist() != expected["points"]:
        fail(f"points mismatch: got {points.tolist()}, expected {expected['points']}")

    for step in expected["timesteps"]:
        UpdatePipeline(time=step["time"], proxy=reader)
        data = servermanager.Fetch(reader)

        temperature = vtk_to_numpy(data.GetPointData().GetArray("temperature"))
        if temperature.tolist() != step["temperature"]:
            fail(
                f"t={step['time']}: temperature mismatch: got {temperature.tolist()}, "
                f"expected {step['temperature']}"
            )

        region_id = vtk_to_numpy(data.GetCellData().GetArray("region_id"))
        if region_id.tolist() != step["region_id"]:
            fail(
                f"t={step['time']}: region_id mismatch: got {region_id.tolist()}, "
                f"expected {step['region_id']}"
            )

    print(f"OK: {len(expected['timesteps'])} timestep(s) verified against {xdmf_file}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
