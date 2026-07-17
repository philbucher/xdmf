//! `TimeSeriesWriter`/`TimeSeriesDataWriter` pyclasses wrapping the core crate's writer API.

use std::collections::BTreeSet;

use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{
    arrays::{FloatArray, UintArray, ValueGuard},
    enums::{PyCellType, PyDataAttribute, PyDataStorage},
    error::to_py_err,
};

const ALREADY_CONSUMED: &str =
    "write_mesh/write_mesh_with_blocks was already called on this TimeSeriesWriter";

// `unsendable`: the underlying writers hold plain `File`/`BufWriter` handles, so instances stay
// pinned to the thread that created them.
//
// Writes run under the GIL for now (not released via `Python::detach`): the core crate's
// `Box<dyn DataWriter>` isn't `Send`, which `detach` requires for anything crossing it. Making
// it `Send` is possible (all current writer impls are), but changes a trait used by the
// `hdf5`-feature build too, so it's left as a follow-up rather than done in passing here.
#[pyclass(name = "TimeSeriesWriter", unsendable)]
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

    /// Write the mesh. Consumes this writer (matching the Rust API, where `write_mesh` takes
    /// `self` by value); calling it a second time raises.
    fn write_mesh(
        &mut self,
        points: &Bound<'_, PyAny>,
        connectivity: &Bound<'_, PyAny>,
        cell_types: Vec<PyCellType>,
    ) -> PyResult<PyTimeSeriesDataWriter> {
        let writer = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err(ALREADY_CONSUMED))?;

        let points_arr = FloatArray::extract(points)?;
        let points_slice = points_arr.as_slice()?;
        let conn_arr = UintArray::extract(connectivity)?;
        let conn_slice = conn_arr.as_u64_slice()?;
        let cell_types: Vec<xdmf::CellType> = cell_types.into_iter().map(Into::into).collect();

        let inner = writer
            .write_mesh(points_slice, (conn_slice, &cell_types))
            .map_err(to_py_err)?;

        Ok(PyTimeSeriesDataWriter { inner })
    }

    /// Write the mesh as a collection of named blocks. `blocks` is a list of
    /// `(name, cell_indices)` pairs, `cell_indices` being 0-based indices into `connectivity`
    /// (in the same order as `cell_types`). See `xdmf::TimeSeriesWriter::write_mesh_with_blocks`
    /// for the exact semantics (overlaps allowed, every cell must belong to at least one block).
    fn write_mesh_with_blocks(
        &mut self,
        points: &Bound<'_, PyAny>,
        connectivity: &Bound<'_, PyAny>,
        cell_types: Vec<PyCellType>,
        blocks: Vec<(String, Vec<usize>)>,
    ) -> PyResult<PyTimeSeriesDataWriter> {
        let writer = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err(ALREADY_CONSUMED))?;

        let points_arr = FloatArray::extract(points)?;
        let points_slice = points_arr.as_slice()?;
        let conn_arr = UintArray::extract(connectivity)?;
        let conn_slice = conn_arr.as_u64_slice()?;
        let cell_types: Vec<xdmf::CellType> = cell_types.into_iter().map(Into::into).collect();

        let block_sets: Vec<(String, BTreeSet<usize>)> = blocks
            .into_iter()
            .map(|(name, indices)| (name, indices.into_iter().collect()))
            .collect();
        let block_refs: Vec<(&str, &BTreeSet<usize>)> =
            block_sets.iter().map(|(name, set)| (name.as_str(), set)).collect();

        let inner = writer
            .write_mesh_with_blocks(points_slice, (conn_slice, &cell_types), &block_refs)
            .map_err(to_py_err)?;

        Ok(PyTimeSeriesDataWriter { inner })
    }
}

#[pyclass(name = "TimeSeriesDataWriter", unsendable)]
pub struct PyTimeSeriesDataWriter {
    inner: xdmf::TimeSeriesDataWriter,
}

#[pymethods]
impl PyTimeSeriesDataWriter {
    /// Write point/cell attribute data for one time step. `point_data`/`cell_data` are lists of
    /// `(name, DataAttribute, array)`, `array` being a contiguous 1D numpy array of dtype
    /// `float64`, `uint64`, or `int64` -- borrowed with no copy (see `arrays.rs`).
    fn write_data<'py>(
        &mut self,
        time: &str,
        point_data: Vec<(String, PyDataAttribute, Bound<'py, PyAny>)>,
        cell_data: Vec<(String, PyDataAttribute, Bound<'py, PyAny>)>,
    ) -> PyResult<()> {
        let point_named = extract_guards(point_data)?;
        let cell_named = extract_guards(cell_data)?;

        let point_values = build_values(&point_named)?;
        let cell_values = build_values(&cell_named)?;

        let point_refs: Vec<(&str, xdmf::DataAttribute, &xdmf::Values<'_>)> =
            point_values.iter().map(|(n, a, v)| (*n, *a, v)).collect();
        let cell_refs: Vec<(&str, xdmf::DataAttribute, &xdmf::Values<'_>)> =
            cell_values.iter().map(|(n, a, v)| (*n, *a, v)).collect();

        self.inner
            .write_data(time, point_refs, cell_refs)
            .map_err(to_py_err)
    }
}

fn extract_guards<'py>(
    data: Vec<(String, PyDataAttribute, Bound<'py, PyAny>)>,
) -> PyResult<Vec<(String, PyDataAttribute, ValueGuard<'py>)>> {
    data.into_iter()
        .map(|(name, attr, obj)| {
            let guard = ValueGuard::extract(&obj)?;
            Ok((name, attr, guard))
        })
        .collect()
}

fn build_values<'g>(
    named: &'g [(String, PyDataAttribute, ValueGuard<'_>)],
) -> PyResult<Vec<(&'g str, xdmf::DataAttribute, xdmf::Values<'g>)>> {
    named
        .iter()
        .map(|(name, attr, guard)| Ok((name.as_str(), attr.0, guard.to_values()?)))
        .collect()
}
