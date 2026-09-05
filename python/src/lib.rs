//! Python bindings for the `xdmf` crate, built with pyo3/numpy.
//!
//! Points, connectivity and per-step attribute data are borrowed straight from the numpy buffer
//! with no copy, and the writes themselves release the GIL.

mod arrays;
mod enums;
mod error;
mod writer;

use pyo3::prelude::*;

use crate::{
    enums::{PyCellType, PyDataAttribute, PyDataStorage},
    writer::{PyTimeSeriesDataWriter, PyTimeSeriesWriter},
};

/// Whether this build can write the HDF5 storages, mirroring `xdmf::is_hdf5_enabled`.
#[pyfunction]
fn is_hdf5_enabled() -> bool {
    // `::` qualified: the `#[pymodule]` below generates a module named `xdmf` too, which would
    // otherwise shadow the crate of that name here.
    ::xdmf::is_hdf5_enabled()
}

/// Write XDMF files with time-series data
///
/// The mesh and the data are passed as numpy arrays, which are borrowed rather than copied, and are
/// written at the dtype they are passed in. See `TimeSeriesWriter`.
#[pymodule]
fn xdmf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataStorage>()?;
    m.add_class::<PyCellType>()?;
    m.add_class::<PyDataAttribute>()?;
    m.add_class::<PyTimeSeriesWriter>()?;
    m.add_class::<PyTimeSeriesDataWriter>()?;
    m.add_function(wrap_pyfunction!(is_hdf5_enabled, m)?)?;

    // Stated explicitly so that `from xdmf import *` -- which the package's generated
    // `__init__.py` itself uses -- imports exactly this and not the extension module's own name.
    m.add(
        "__all__",
        vec![
            "CellType",
            "DataAttribute",
            "DataStorage",
            "TimeSeriesDataWriter",
            "TimeSeriesWriter",
            "is_hdf5_enabled",
        ],
    )
}
