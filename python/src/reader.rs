//! `TimeSeriesReader`/`TimeSeriesDataReader` pyclasses wrapping the core crate's reader API.
//!
//! Like the writer, reads release the GIL for the actual (potentially large) I/O via
//! `Python::detach`, and heavy data crosses back into Python with no copy: a filled `Vec<T>` is
//! handed to numpy via `IntoPyArray`, which takes ownership of the Rust allocation as the numpy
//! array's backing buffer rather than copying it.

use numpy::{IntoPyArray, PyArray1};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{
    enums::{PyCellType, PyDataAttribute},
    error::to_py_err,
};

const ALREADY_CONSUMED: &str = "read_mesh was already called on this TimeSeriesReader";

#[pyclass(name = "TimeSeriesReader")]
pub struct PyTimeSeriesReader {
    inner: Option<xdmf::TimeSeriesReader>,
}

#[pymethods]
impl PyTimeSeriesReader {
    #[new]
    fn new(file_name: &str) -> PyResult<Self> {
        let inner = xdmf::TimeSeriesReader::new(file_name).map_err(to_py_err)?;
        Ok(Self { inner: Some(inner) })
    }

    /// Number of points in the mesh.
    fn num_points(&self) -> PyResult<usize> {
        Ok(self.reader()?.num_points())
    }

    /// Number of cells in the mesh. For a point cloud (no cells written), equals `num_points`.
    fn num_cells(&self) -> PyResult<usize> {
        Ok(self.reader()?.num_cells())
    }

    /// Time step labels, in file order.
    fn times(&self) -> PyResult<Vec<String>> {
        Ok(self.reader()?.times().to_vec())
    }

    /// Reads the mesh, returning `(points, connectivity, cell_types, data_reader)`: `points` is a
    /// flat `float64` numpy array of length `3 * num_points`, `connectivity` a flat `uint64`
    /// array, and `cell_types` a list of `xdmf.CellType` (empty for a point cloud). Consumes this
    /// reader (matching the Rust API, where `read_mesh` takes `self` by value).
    #[allow(
        clippy::type_complexity,
        reason = "mirrors the Rust API's (points, connectivity, cell_types, data_reader) return shape"
    )]
    fn read_mesh<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<u64>>,
        Vec<PyCellType>,
        PyTimeSeriesDataReader,
    )> {
        let reader = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err(ALREADY_CONSUMED))?;

        let mut points = Vec::new();
        let mut connectivity = Vec::new();
        let mut cell_types = Vec::new();
        let data_reader = py
            .detach(|| reader.read_mesh(&mut points, &mut connectivity, &mut cell_types))
            .map_err(to_py_err)?;

        let py_cell_types = cell_types.into_iter().map(PyCellType::from).collect();

        Ok((
            points.into_pyarray(py),
            connectivity.into_pyarray(py),
            py_cell_types,
            PyTimeSeriesDataReader { inner: data_reader },
        ))
    }
}

impl PyTimeSeriesReader {
    fn reader(&self) -> PyResult<&xdmf::TimeSeriesReader> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err(ALREADY_CONSUMED))
    }
}

/// Metadata about one point/cell data field, without reading its heavy data. Lets a caller pick
/// the right dtype ahead of a `read_point_data`/`read_cell_data` call.
#[pyclass(name = "DataInfo", get_all)]
pub struct PyDataInfo {
    /// The field's name.
    name: String,
    /// The field's tensor shape.
    attribute: PyDataAttribute,
    /// The field's element dtype, as a numpy dtype name (`"float64"`, `"float32"`, or `"uint64"`).
    dtype: &'static str,
    /// Total number of elements, i.e. `num_entities * attribute.size()`.
    len: usize,
}

#[pymethods]
impl PyDataInfo {
    fn __repr__(&self) -> String {
        format!(
            "DataInfo(name={:?}, attribute={:?}, dtype={:?}, len={})",
            self.name, self.attribute.0, self.dtype, self.len
        )
    }
}

impl From<xdmf::DataInfo> for PyDataInfo {
    fn from(info: xdmf::DataInfo) -> Self {
        let dtype = match info.kind {
            xdmf::ValueKind::F32 => "float32",
            xdmf::ValueKind::F64 => "float64",
            xdmf::ValueKind::U64 => "uint64",
        };
        Self {
            name: info.name,
            attribute: PyDataAttribute(info.attribute),
            dtype,
            len: info.len,
        }
    }
}

/// Reader for per-step point/cell attribute data, obtained from
/// `TimeSeriesReader.read_mesh`.
#[pyclass(name = "TimeSeriesDataReader")]
pub struct PyTimeSeriesDataReader {
    inner: xdmf::TimeSeriesDataReader,
}

#[pymethods]
impl PyTimeSeriesDataReader {
    /// Number of time steps written.
    fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }

    /// Time step labels, in file order.
    fn times(&self) -> Vec<String> {
        self.inner.times().to_vec()
    }

    /// Number of point-data fields at `step`.
    fn num_point_data(&self, step: usize) -> PyResult<usize> {
        self.inner.num_point_data(step).map_err(to_py_err)
    }

    /// Number of cell-data fields at `step`.
    fn num_cell_data(&self, step: usize) -> PyResult<usize> {
        self.inner.num_cell_data(step).map_err(to_py_err)
    }

    /// Metadata for the point-data field at `index` within `step`.
    fn point_data_info(&self, step: usize, index: usize) -> PyResult<PyDataInfo> {
        self.inner
            .point_data_info(step, index)
            .map(PyDataInfo::from)
            .map_err(to_py_err)
    }

    /// Metadata for the cell-data field at `index` within `step`.
    fn cell_data_info(&self, step: usize, index: usize) -> PyResult<PyDataInfo> {
        self.inner
            .cell_data_info(step, index)
            .map(PyDataInfo::from)
            .map_err(to_py_err)
    }

    /// Index of the point-data field named `name` at `step`, for callers that think in names.
    fn point_data_index(&self, step: usize, name: &str) -> PyResult<usize> {
        self.inner.point_data_index(step, name).map_err(to_py_err)
    }

    /// Index of the cell-data field named `name` at `step`.
    fn cell_data_index(&self, step: usize, name: &str) -> PyResult<usize> {
        self.inner.cell_data_index(step, name).map_err(to_py_err)
    }

    /// Reads every point-data field at `step`, returning a list of `(name, DataAttribute, array)`
    /// triples in the same order as `point_data_info` reports them. The natural shape for
    /// scripts; `read_point_data` is the per-field alternative for a caller that only wants one
    /// named field.
    fn read_point_step(
        &mut self,
        py: Python<'_>,
        step: usize,
    ) -> PyResult<Vec<(String, PyDataAttribute, Py<PyAny>)>> {
        let data = py
            .detach(|| self.inner.read_point_step(step))
            .map_err(to_py_err)?;
        Ok(to_py_step(py, data))
    }

    /// Reads every cell-data field at `step`. See `read_point_step` for the shape.
    fn read_cell_step(
        &mut self,
        py: Python<'_>,
        step: usize,
    ) -> PyResult<Vec<(String, PyDataAttribute, Py<PyAny>)>> {
        let data = py
            .detach(|| self.inner.read_cell_step(step))
            .map_err(to_py_err)?;
        Ok(to_py_step(py, data))
    }

    /// Reads the point-data field at `index` within `step`, returning a numpy array of the dtype
    /// reported by `point_data_info` (widening a `float32` file into `float64` is allowed;
    /// narrowing is not -- see the Rust `read_point_data` docs for the exact rule).
    fn read_point_data(
        &mut self,
        py: Python<'_>,
        step: usize,
        index: usize,
    ) -> PyResult<Py<PyAny>> {
        let info = self.inner.point_data_info(step, index).map_err(to_py_err)?;
        read_one(py, &mut self.inner, step, index, info, true)
    }

    /// Reads the cell-data field at `index` within `step`. See `read_point_data` for the shape.
    fn read_cell_data(&mut self, py: Python<'_>, step: usize, index: usize) -> PyResult<Py<PyAny>> {
        let info = self.inner.cell_data_info(step, index).map_err(to_py_err)?;
        read_one(py, &mut self.inner, step, index, info, false)
    }
}

fn to_py_step(
    py: Python<'_>,
    data: Vec<(String, xdmf::DataAttribute, xdmf::Values<'static>)>,
) -> Vec<(String, PyDataAttribute, Py<PyAny>)> {
    data.into_iter()
        .map(|(name, attr, values)| (name, PyDataAttribute(attr), values_to_pyobject(py, values)))
        .collect()
}

fn values_to_pyobject(py: Python<'_>, values: xdmf::Values<'static>) -> Py<PyAny> {
    match values {
        xdmf::Values::F64(cow) => cow.into_owned().into_pyarray(py).into_any().unbind(),
        xdmf::Values::F32(cow) => cow.into_owned().into_pyarray(py).into_any().unbind(),
        xdmf::Values::U64(cow) => cow.into_owned().into_pyarray(py).into_any().unbind(),
    }
}

fn read_one(
    py: Python<'_>,
    reader: &mut xdmf::TimeSeriesDataReader,
    step: usize,
    index: usize,
    info: xdmf::DataInfo,
    is_point: bool,
) -> PyResult<Py<PyAny>> {
    match info.kind {
        xdmf::ValueKind::F64 => {
            let mut buf = Vec::new();
            py.detach(|| {
                if is_point {
                    reader.read_point_data::<f64>(step, index, &mut buf)
                } else {
                    reader.read_cell_data::<f64>(step, index, &mut buf)
                }
            })
            .map_err(to_py_err)?;
            Ok(buf.into_pyarray(py).into_any().unbind())
        }
        xdmf::ValueKind::F32 => {
            let mut buf = Vec::new();
            py.detach(|| {
                if is_point {
                    reader.read_point_data::<f32>(step, index, &mut buf)
                } else {
                    reader.read_cell_data::<f32>(step, index, &mut buf)
                }
            })
            .map_err(to_py_err)?;
            Ok(buf.into_pyarray(py).into_any().unbind())
        }
        xdmf::ValueKind::U64 => {
            let mut buf = Vec::new();
            py.detach(|| {
                if is_point {
                    reader.read_point_data::<u64>(step, index, &mut buf)
                } else {
                    reader.read_cell_data::<u64>(step, index, &mut buf)
                }
            })
            .map_err(to_py_err)?;
            Ok(buf.into_pyarray(py).into_any().unbind())
        }
    }
}
