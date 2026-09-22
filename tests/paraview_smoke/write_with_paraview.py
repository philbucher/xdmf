"""Run under `pvpython` to write a small XDMF2 time series with ParaView's own writer
(`vtkIOXdmf2.vtkXdmfWriter`), so `tests/paraview_smoke/read_paraview_fixture.rs`'s companion Rust
binary can check that this crate's `TimeSeriesReader` -- not just its writer -- interoperates with
a document this crate never produced a byte of.

This is the reverse of `examples/paraview_smoke.rs`: that binary writes with this crate and checks
`ParaView` reads it correctly; this script writes with `ParaView` and checks this crate reads it
correctly. The two are asymmetric on purpose, not just in direction. `vtkXdmfWriter` inlines small
arrays as `Format="XML"` text unless told otherwise -- `SetLightDataLimit(0)` forces every array to
`Format="HDF"`, the only kind `TimeSeriesReader` reads at all. And `vtkXdmfWriter` writes several
constructs this crate's own writer never produces, each accepted only after a real reader gap it
surfaced was fixed:

- `<Topology Dimensions="N">` instead of `NumberOfElements="N"`
- `AttributeType="None"` (the XDMF2 DTD's own "unspecified, infer from the data" value) on every
  `<Attribute>`, since nothing here marks an array as the active scalars/vectors
- `<Grid GridType="Uniform">` with no `Name` at all, on every per-step grid
- a `<DataItem Format="HDF">` holding the array directly under `<Geometry>`/`<Topology>`, rather
  than this crate's own `Reference="XML"` indirection to a named `DataItem` under `<Domain>`

One shape is deliberately avoided rather than chased: a quad cell here would come back as XDMF2's
generic "Polygon" `Mixed`-topology code, which this crate's `CellType` has no variant for at all --
every other code names a *fixed* point count, and `CellType` (shared with the writer, and with
`prepare_cells`'/`Values::dimensions`' arithmetic) is built on that assumption throughout. Working
around that would be a real feature (a variable-arity cell type), not a reader compatibility fix,
so the fixture instead mixes two shapes ParaView maps onto this crate's own fixed-arity codes
(`Tetrahedron`, `Triangle`) -- enough to still exercise the `Mixed` topology path.

Usage: `pvpython write_with_paraview.py <output_dir>`
"""

import sys
from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline
from vtkmodules.vtkIOXdmf2 import vtkXdmfWriter

TIMES = [0.0, 1.0]

# a tetrahedron and a triangle sharing an edge -- two cell types, of different dimensionality, that
# ParaView maps onto this crate's own fixed-arity `Mixed`-topology codes (see the module docstring)
COORDS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (2.0, 0.5, 0.0),
]
TETRAHEDRON_POINTS = [0, 1, 2, 3]
TRIANGLE_POINTS = [1, 4, 2]

# the 32-bit field covers both ends of its range, so a reader that gets the signedness wrong (the
# sign bit of a negative i32 dropped) cannot pass
LEVEL_I32 = np.array([-2_000_000_000, 2_000_000_000], dtype=np.int32)


class MixedCellSource(VTKPythonAlgorithmBase):
    """A two-timestep `vtkUnstructuredGrid` source, scripted rather than read from a file so the
    fixture has no binary asset of its own to keep in the repository."""

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=0, nOutputPorts=1, outputType="vtkUnstructuredGrid"
        )

    def RequestInformation(self, request, inInfoVec, outInfoVec):
        info = outInfoVec.GetInformationObject(0)
        info.Remove(vtkStreamingDemandDrivenPipeline.TIME_STEPS())
        for t in TIMES:
            info.Append(vtkStreamingDemandDrivenPipeline.TIME_STEPS(), t)
        info.Remove(vtkStreamingDemandDrivenPipeline.TIME_RANGE())
        info.Append(vtkStreamingDemandDrivenPipeline.TIME_RANGE(), TIMES[0])
        info.Append(vtkStreamingDemandDrivenPipeline.TIME_RANGE(), TIMES[-1])
        return 1

    def RequestData(self, request, inInfoVec, outInfoVec):
        info = outInfoVec.GetInformationObject(0)
        time = 0.0
        if info.Has(vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP()):
            time = info.Get(vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP())
        scale = time + 1.0

        output = vtkUnstructuredGrid.GetData(outInfoVec, 0)

        points = vtk.vtkPoints()
        for coord in COORDS:
            points.InsertNextPoint(*coord)
        output.SetPoints(points)

        tetrahedron = vtk.vtkTetra()
        for i, point in enumerate(TETRAHEDRON_POINTS):
            tetrahedron.GetPointIds().SetId(i, point)
        triangle = vtk.vtkTriangle()
        for i, point in enumerate(TRIANGLE_POINTS):
            triangle.GetPointIds().SetId(i, point)

        output.Allocate(2)
        output.InsertNextCell(tetrahedron.GetCellType(), tetrahedron.GetPointIds())
        output.InsertNextCell(triangle.GetCellType(), triangle.GetPointIds())

        temperature = numpy_support.numpy_to_vtk(
            np.array([10.0, 11.0, 12.0, 13.0, 14.0]) * scale
        )
        temperature.SetName("temperature")
        output.GetPointData().AddArray(temperature)

        # a fresh array each call (not a `numpy_to_vtk` of the module-level `LEVEL_I32`): VTK wraps
        # a numpy buffer's memory without copying it, and this method is called once per timestep,
        # so reusing the same numpy array across calls hands the writer two `vtkIntArray`s backed
        # by the same memory, which segfaults during `Write()`
        level = numpy_support.numpy_to_vtk(LEVEL_I32.copy())
        level.SetName("level_i32")
        output.GetCellData().AddArray(level)

        info.Set(output.DATA_TIME_STEP(), time)
        return 1


def main(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    xdmf_file = output_dir / "paraview_written.xmf"

    # kept in a variable rather than passed inline: `SetInputConnection` only takes the C++ pipeline
    # object's output port, not a reference to the Python wrapper -- an inline `MixedCellSource()`
    # gets garbage collected before `Write()` calls back into its `RequestData`, which segfaults
    source = MixedCellSource()

    writer = vtkXdmfWriter()
    writer.SetInputConnection(source.GetOutputPort())
    writer.SetFileName(str(xdmf_file))
    # forces every array to `Format="HDF"` -- see the module docstring
    writer.SetLightDataLimit(0)
    writer.SetWriteAllTimeSteps(True)
    writer.Write()

    print(f"Wrote fixture to {xdmf_file}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
