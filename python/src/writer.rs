//! `TimeSeriesWriter`/`TimeSeriesDataWriter` pyclasses wrapping the core crate's writer API.
//!
//! Both classes are ordinary (non-`unsendable`) pyclasses: the core crate's `DataWriter` trait is
//! `Send` (see `src/lib.rs`), so `xdmf::TimeSeriesWriter`/`TimeSeriesDataWriter` are `Send` too,
//! which is what lets `write_mesh`/`write_data` release the GIL (`Python::detach`) for the actual
//! write -- otherwise every other Python thread would block for the duration of a large write,
//! defeating the point of a library whose selling point is large-data throughput.

use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{
    arrays::NumpyArray,
    enums::{PyDataAttribute, PyDataStorage, extract_cell_types},
    error::to_py_err,
};

const ALREADY_CONSUMED: &str = "write_mesh was already called on this TimeSeriesWriter";
const ALREADY_CLOSED: &str = "this TimeSeriesDataWriter has already been closed";

#[pyclass(name = "TimeSeriesWriter")]
pub struct PyTimeSeriesWriter {
    inner: Option<xdmf::TimeSeriesWriter>,
}

#[pymethods]
impl PyTimeSeriesWriter {
    #[new]
    fn new(file_name: &str, data_storage: PyDataStorage) -> PyResult<Self> {
        let inner =
            xdmf::TimeSeriesWriter::new(file_name, data_storage.into()).map_err(to_py_err)?;
        Ok(Self { inner: Some(inner) })
    }

    /// Write the mesh. `points` is a numpy `float64` or `float32` array (flat or `(N, 3)`).
    /// `connectivity` is a numpy `uint64`/`uint32`/`int64`/`int32` array of point indices -- a
    /// signed array is checked for negative values once, centrally, in the core crate, and is
    /// stored on disk as unsigned data at whichever width (32/64-bit) was passed in. `cell_types`
    /// is either a list of `xdmf.CellType` or a numpy array of raw cell type codes. Consumes this
    /// writer (matching the Rust API, where `write_mesh` takes `self` by value); calling it a
    /// second time raises.
    fn write_mesh(
        &mut self,
        py: Python<'_>,
        points: &Bound<'_, PyAny>,
        connectivity: &Bound<'_, PyAny>,
        cell_types: &Bound<'_, PyAny>,
    ) -> PyResult<PyTimeSeriesDataWriter> {
        let writer = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err(ALREADY_CONSUMED))?;

        let points_arr = NumpyArray::extract(points)?;
        let points_values = points_arr.to_values()?;
        let conn_arr = NumpyArray::extract(connectivity)?;
        let conn_values = conn_arr.to_values()?;
        let cell_types = extract_cell_types(cell_types)?;

        let inner = py
            .detach(|| writer.write_mesh(points_values, conn_values, &cell_types))
            .map_err(to_py_err)?;

        Ok(PyTimeSeriesDataWriter { inner: Some(inner) })
    }
}

#[pyclass(name = "TimeSeriesDataWriter")]
pub struct PyTimeSeriesDataWriter {
    inner: Option<xdmf::TimeSeriesDataWriter>,
}

#[pymethods]
impl PyTimeSeriesDataWriter {
    /// Write point/cell attribute data for one time step. `point_data`/`cell_data` are lists of
    /// `(name, DataAttribute, array)`, `array` being a contiguous numpy array of dtype
    /// `float64`, `float32`, `uint64`, `uint32`, `int64`, or `int32` -- borrowed with no copy (see
    /// `arrays.rs`).
    fn write_data<'py>(
        &mut self,
        py: Python<'py>,
        time: &str,
        point_data: Vec<(String, PyDataAttribute, Bound<'py, PyAny>)>,
        cell_data: Vec<(String, PyDataAttribute, Bound<'py, PyAny>)>,
    ) -> PyResult<()> {
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err(ALREADY_CLOSED))?;

        let point_named = extract_guards(point_data)?;
        let cell_named = extract_guards(cell_data)?;

        let point_values = build_values(&point_named)?;
        let cell_values = build_values(&cell_named)?;

        py.detach(|| writer.write_data(time, point_values, cell_values))
            .map_err(to_py_err)
    }

    /// Closes the writer, flushing and releasing any open file handles (most relevant for the
    /// HDF5 backends, whose file otherwise stays open, and thus locked/unflushed, until this
    /// object is garbage-collected). Safe to call more than once.
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
        let _ = (exc_type, exc_value, traceback);
        self.close();
    }
}

fn extract_guards<'py>(
    data: Vec<(String, PyDataAttribute, Bound<'py, PyAny>)>,
) -> PyResult<Vec<(String, PyDataAttribute, NumpyArray<'py>)>> {
    data.into_iter()
        .map(|(name, attr, obj)| {
            let guard = NumpyArray::extract(&obj)?;
            Ok((name, attr, guard))
        })
        .collect()
}

fn build_values<'g>(
    named: &'g [(String, PyDataAttribute, NumpyArray<'_>)],
) -> PyResult<Vec<(&'g str, xdmf::DataAttribute, xdmf::Values<'g>)>> {
    named
        .iter()
        .map(|(name, attr, guard)| Ok((name.as_str(), (*attr).into(), guard.to_values()?)))
        .collect()
}
