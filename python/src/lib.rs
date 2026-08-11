//! Python bindings for the `xdmf` crate, built with pyo3/numpy.
//!
//! Points/float attribute data and connectivity/uint attribute data are borrowed directly from
//! the numpy buffer with no copy wherever the input dtype allows it -- see `arrays.rs`. Writes and
//! reads release the GIL for the actual I/O -- see `writer.rs`/`reader.rs`.

mod arrays;
mod enums;
mod error;
mod reader;
mod writer;

use pyo3::prelude::*;

use crate::{
    enums::{PyCellType, PyDataAttribute, PyDataStorage},
    reader::{PyDataInfo, PyTimeSeriesDataReader, PyTimeSeriesReader},
    writer::{PyTimeSeriesDataWriter, PyTimeSeriesWriter},
};

#[pymodule]
fn xdmf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataStorage>()?;
    m.add_class::<PyCellType>()?;
    m.add_class::<PyDataAttribute>()?;
    m.add_class::<PyTimeSeriesWriter>()?;
    m.add_class::<PyTimeSeriesDataWriter>()?;
    m.add_class::<PyTimeSeriesReader>()?;
    m.add_class::<PyTimeSeriesDataReader>()?;
    m.add_class::<PyDataInfo>()?;
    Ok(())
}
