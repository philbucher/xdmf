//! `TimeSeriesWriter`/`TimeSeriesDataWriter` pyclasses wrapping the core crate's writer API.
//!
//! Both classes are ordinary (non-`unsendable`) pyclasses, since the core crate's `DataWriter`
//! trait is `Send + Sync`, which is what lets the writes here release the GIL.
//!
//! A Rust `TimeStep` borrows its writer, which a pyclass cannot hold, so the Python method takes
//! all attributes of a step at once and runs the closure itself.

use std::path::{Path, PathBuf};

use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{
    arrays::{IndexArray, PointArray, ValueArray, contiguous_slice},
    enums::{PyDataAttribute, PyDataStorage, extract_cell_types, extract_submesh_cells},
    error::to_py_err,
};

const ALREADY_CONSUMED: &str = "write_mesh was already called on this TimeSeriesWriter";
const ALREADY_CLOSED: &str = "this TimeSeriesDataWriter has already been closed";

/// One attribute of a time step: its name, what it describes, and the numpy array holding it.
type NamedData<'py> = (String, PyDataAttribute, Bound<'py, PyAny>);

/// One submesh: its name, and the `range`, numpy array or sequence of `int` naming its cells.
type NamedSubmesh<'py> = (String, Bound<'py, PyAny>);

/// Borrows a numpy array as the concrete slice type its dtype names, and evaluates `$body` with it.
///
/// One arm per dtype, so `$body` is compiled once per element type and the generic parameters of
/// `xdmf::TimeSeriesWriter::write_mesh` (`Coordinate`/`ConnectivityIndex`) are resolved statically.
/// Nesting two invocations covers the cross product of point and index dtypes.
macro_rules! dispatch_dtype {
    ($array:expr, $enum:ident, [$($variant:ident),+], |$slice:ident| $body:expr) => {
        match $array {
            $($enum::$variant(array) => {
                let $slice = contiguous_slice(&array)?;
                $body
            })+
        }
    };
}

/// Writer for time series data in XDMF format.
#[pyclass(name = "TimeSeriesWriter")]
#[derive(Debug)]
pub struct PyTimeSeriesWriter {
    inner: Option<xdmf::TimeSeriesWriter>,
    // kept as its own field rather than read off `inner`, so it survives `write_mesh` taking that
    file_name: PathBuf,
}

#[pymethods]
impl PyTimeSeriesWriter {
    #[new]
    fn new(file_name: PathBuf, data_storage: PyDataStorage) -> PyResult<Self> {
        let inner =
            xdmf::TimeSeriesWriter::new(&file_name, data_storage.into()).map_err(to_py_err)?;
        let file_name = inner.file_name().to_path_buf();

        Ok(Self {
            inner: Some(inner),
            file_name,
        })
    }

    /// The XDMF file this writer writes: the name it was given, with the `.xdmf2` extension on
    /// it. The heavy data takes the same base and its own storage's extension.
    #[getter]
    fn file_name(&self) -> &Path {
        &self.file_name
    }

    /// Write the mesh, returning the writer for the time step data.
    ///
    /// `points` is a numpy `float64`/`float32` array of x/y/z coordinates, `connectivity` a numpy
    /// `uint64`/`uint32`/`int64`/`int32` array of point indices, and `cell_types` either a
    /// sequence of `xdmf.CellType` or a numpy array of the XDMF topology type codes those values
    /// have -- *not* the VTK cell codes, which differ (a hexahedron is 9 here and 12 in VTK).
    ///
    /// Both arrays are stored at the dtype they are passed in, so the connectivity dtype caps the
    /// mesh size. Their shape only has to be C-contiguous: the natural `(N, 3)` layout for points
    /// is the same memory as the flat one and needs no reshape.
    ///
    /// Consumes this writer, matching the Rust API, and calling it twice raises `RuntimeError`. A
    /// call *rejected* here leaves the writer usable.
    fn write_mesh(
        &mut self,
        py: Python<'_>,
        points: &Bound<'_, PyAny>,
        connectivity: &Bound<'_, PyAny>,
        cell_types: &Bound<'_, PyAny>,
    ) -> PyResult<PyTimeSeriesDataWriter> {
        // checked up front too, so a genuine second call is reported as such rather than as
        // whatever its arguments happen to be
        if self.inner.is_none() {
            return Err(PyRuntimeError::new_err(ALREADY_CONSUMED));
        }

        let (points, connectivity, cell_types) =
            extract_mesh_args(points, connectivity, cell_types)?;

        let inner = dispatch_dtype!(points, PointArray, [F64, F32], |point_slice| {
            dispatch_dtype!(
                connectivity,
                IndexArray,
                [U64, U32, I64, I32],
                |index_slice| {
                    let writer = self
                        .inner
                        .take()
                        .ok_or_else(|| PyRuntimeError::new_err(ALREADY_CONSUMED))?;
                    py.detach(|| writer.write_mesh(point_slice, index_slice, &cell_types))
                        .map_err(to_py_err)
                }
            )
        })?;

        Ok(data_writer(inner))
    }

    /// Write the mesh split into named submeshes, returning the writer for the time step data, as
    /// [`write_mesh`](Self::write_mesh) does.
    ///
    /// `submeshes` is a sequence of `(name, cells)` pairs, `cells` naming which cells (indices
    /// into `cell_types`) belong to that submesh, as a `range`, a numpy integer array or a
    /// sequence of `int`. A `range(start, stop)` is taken as the block it names without its
    /// indices ever being built, so a submesh of a huge mesh costs two numbers. Every cell must be
    /// in at least one submesh; submeshes may overlap.
    fn write_mesh_with_submeshes(
        &mut self,
        py: Python<'_>,
        points: &Bound<'_, PyAny>,
        connectivity: &Bound<'_, PyAny>,
        cell_types: &Bound<'_, PyAny>,
        submeshes: Vec<NamedSubmesh<'_>>,
    ) -> PyResult<PyTimeSeriesDataWriter> {
        if self.inner.is_none() {
            return Err(PyRuntimeError::new_err(ALREADY_CONSUMED));
        }

        let (points, connectivity, cell_types) =
            extract_mesh_args(points, connectivity, cell_types)?;
        let submeshes = submeshes
            .into_iter()
            .map(|(name, cells)| Ok((name, extract_submesh_cells(&cells)?)))
            .collect::<PyResult<Vec<_>>>()?;

        let inner = dispatch_dtype!(points, PointArray, [F64, F32], |point_slice| {
            dispatch_dtype!(
                connectivity,
                IndexArray,
                [U64, U32, I64, I32],
                |index_slice| {
                    let writer = self
                        .inner
                        .take()
                        .ok_or_else(|| PyRuntimeError::new_err(ALREADY_CONSUMED))?;
                    py.detach(|| {
                        writer.write_mesh_with_submeshes(
                            point_slice,
                            index_slice,
                            &cell_types,
                            submeshes,
                        )
                    })
                    .map_err(to_py_err)
                }
            )
        })?;

        Ok(data_writer(inner))
    }
}

/// Extracts and validates the three arguments `write_mesh`/`write_mesh_with_submeshes` share,
/// before either takes ownership of the writer, so a call rejected here leaves it usable.
fn extract_mesh_args<'py>(
    points: &Bound<'py, PyAny>,
    connectivity: &Bound<'py, PyAny>,
    cell_types: &Bound<'py, PyAny>,
) -> PyResult<(PointArray<'py>, IndexArray<'py>, Vec<xdmf::CellType>)> {
    let points = PointArray::extract(points, "points")?;
    points.validate_shape()?;
    let connectivity = IndexArray::extract(connectivity, "connectivity")?;
    let cell_types = extract_cell_types(cell_types)?;
    Ok((points, connectivity, cell_types))
}

/// Wraps the core crate's data writer, copying the file name off it up front: `close()` drops the
/// writer, and the name should outlive that.
fn data_writer(inner: xdmf::TimeSeriesDataWriter) -> PyTimeSeriesDataWriter {
    PyTimeSeriesDataWriter {
        file_name: inner.file_name().to_path_buf(),
        inner: Some(inner),
    }
}

/// Writer for the per-step data, obtained from `TimeSeriesWriter.write_mesh`.
#[pyclass(name = "TimeSeriesDataWriter")]
#[derive(Debug)]
pub struct PyTimeSeriesDataWriter {
    inner: Option<xdmf::TimeSeriesDataWriter>,
    // as on `PyTimeSeriesWriter`, so `close()` does not take the name with it
    file_name: PathBuf,
}

#[pymethods]
impl PyTimeSeriesDataWriter {
    /// The XDMF file this writer writes, as `TimeSeriesWriter.file_name` reported it.
    #[getter]
    fn file_name(&self) -> &Path {
        &self.file_name
    }

    /// Write the point and cell data of one time step.
    ///
    /// `time` is the time as a string, leaving its formatting to the caller. `point_data` and
    /// `cell_data` are sequences of `(name, DataAttribute, array)`, `array` being a C-contiguous
    /// numpy array of dtype `float64`, `float32`, `uint64`, `uint32`, `int64` or `int32`, borrowed
    /// without a copy.
    ///
    /// The step is all-or-nothing: one rejected attribute writes nothing for this time and leaves
    /// the time available. A step needs at least one attribute.
    ///
    /// The arrays are borrowed and the write releases the GIL, so another thread must not modify
    /// an array while a write of it is running.
    #[pyo3(signature = (time, point_data=None, cell_data=None))]
    fn write_time_step(
        &mut self,
        py: Python<'_>,
        time: &str,
        point_data: Option<Vec<NamedData<'_>>>,
        cell_data: Option<Vec<NamedData<'_>>>,
    ) -> PyResult<()> {
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err(ALREADY_CLOSED))?;

        // The arrays are borrowed, and the borrows checked for contiguity, before the GIL is
        // released -- both need Python, and neither is allowed to fail once the write is under way.
        let point_arrays = borrow_arrays(point_data.unwrap_or_default())?;
        let cell_arrays = borrow_arrays(cell_data.unwrap_or_default())?;
        let point_values = to_values(&point_arrays)?;
        let cell_values = to_values(&cell_arrays)?;

        py.detach(|| {
            writer.write_time_step(time, |step| {
                for (name, attribute, values) in point_values {
                    step.point_data(name, attribute, values)?;
                }
                for (name, attribute, values) in cell_values {
                    step.cell_data(name, attribute, values)?;
                }
                Ok(())
            })
        })
        .map_err(to_py_err)
    }

    /// Close the writer, releasing any open file handles. This matters most for the HDF5
    /// backends, whose file otherwise stays open and locked until this object is collected.
    ///
    /// Safe to call more than once; writing after it raises `RuntimeError`.
    fn close(&mut self) {
        self.inner = None;
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (exc_type=None, exc_value=None, traceback=None))]
    fn __exit__(
        &mut self,
        exc_type: Option<Bound<'_, PyAny>>,
        exc_value: Option<Bound<'_, PyAny>>,
        traceback: Option<Bound<'_, PyAny>>,
    ) {
        let _unused = (exc_type, exc_value, traceback);
        self.close();
    }
}

/// Names an attribute's array in a rejection, the way `"points"` names the mesh's.
fn data_role(name: &str) -> String {
    format!("data of '{name}'")
}

/// Borrows every array of one category, keeping the borrows alive for the whole step.
fn borrow_arrays<'py>(
    data: Vec<NamedData<'py>>,
) -> PyResult<Vec<(String, PyDataAttribute, ValueArray<'py>)>> {
    data.into_iter()
        .map(|(name, attribute, array)| {
            let array = ValueArray::extract(&array, &data_role(&name))?;
            Ok((name, attribute, array))
        })
        .collect()
}

/// Views the borrowed arrays as `xdmf::Values`, in the shape the `TimeStep` methods take them.
fn to_values<'a>(
    data: &'a [(String, PyDataAttribute, ValueArray<'_>)],
) -> PyResult<Vec<(&'a str, xdmf::DataAttribute, xdmf::Values<'a>)>> {
    data.iter()
        .map(|(name, attribute, array)| {
            Ok((name.as_str(), (*attribute).into(), array.to_values()?))
        })
        .collect()
}
