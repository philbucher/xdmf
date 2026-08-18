//! Python-facing enums mirroring the core crate's `DataStorage`, `CellType`, `DataAttribute`.

use numpy::PyReadonlyArrayDyn;
use pyo3::{exceptions::PyValueError, prelude::*};

/// Heavy-data storage format. `Hdf5SingleFile`/`Hdf5MultipleFiles` are plain attributes for the
/// default (library-chosen) deflate compression level; use `hdf5_single_file(level)`/
/// `hdf5_multiple_files(level)` to pick a specific level (0-9) instead.
#[pyclass(name = "DataStorage", eq, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyDataStorage(xdmf::DataStorage);

#[pymethods]
impl PyDataStorage {
    #[classattr]
    #[pyo3(name = "Ascii")]
    fn ascii() -> Self {
        Self(xdmf::DataStorage::Ascii)
    }

    #[classattr]
    #[pyo3(name = "AsciiInline")]
    fn ascii_inline() -> Self {
        Self(xdmf::DataStorage::AsciiInline)
    }

    #[classattr]
    #[pyo3(name = "Hdf5SingleFile")]
    fn hdf5_single_file_default() -> Self {
        Self(xdmf::DataStorage::Hdf5SingleFile {
            deflate_level: None,
        })
    }

    #[classattr]
    #[pyo3(name = "Hdf5MultipleFiles")]
    fn hdf5_multiple_files_default() -> Self {
        Self(xdmf::DataStorage::Hdf5MultipleFiles {
            deflate_level: None,
        })
    }

    #[classattr]
    #[pyo3(name = "Binary")]
    fn binary() -> Self {
        Self(xdmf::DataStorage::Binary)
    }

    /// HDF5, all data in a single file, with a custom zlib/deflate compression level (0-9).
    #[staticmethod]
    fn hdf5_single_file(deflate_level: u8) -> Self {
        Self(xdmf::DataStorage::Hdf5SingleFile {
            deflate_level: Some(deflate_level),
        })
    }

    /// HDF5, one file per time step, with a custom zlib/deflate compression level (0-9).
    #[staticmethod]
    fn hdf5_multiple_files(deflate_level: u8) -> Self {
        Self(xdmf::DataStorage::Hdf5MultipleFiles {
            deflate_level: Some(deflate_level),
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

impl From<PyDataStorage> for xdmf::DataStorage {
    fn from(value: PyDataStorage) -> Self {
        value.0
    }
}

// Declares `PyCellType`, its `From` impl and the code lookup off one variant list, so a cell type
// added to `xdmf::CellType` is a single edit here. The `const` block additionally pins the
// discriminants to the core enum's: `eq_int` exposes them to Python as the raw VTK codes, and they
// are otherwise restated here with nothing tying the two lists together.
macro_rules! cell_types {
    ($($variant:ident = $code:literal,)+) => {
        /// Cell types as defined in the VTK file format, mirroring `xdmf::CellType`. Values match
        /// the VTK/XDMF discriminants exactly, so a raw numpy array of codes (see
        /// `extract_cell_types`) is an equivalent, cheaper-to-produce alternative to a list of
        /// these.
        #[pyclass(name = "CellType", eq, eq_int, frozen, hash, from_py_object)]
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(u8)]
        pub enum PyCellType {
            $($variant = $code,)+
        }

        const _: () = {
            $(assert!(PyCellType::$variant as u8 == xdmf::CellType::$variant as u8);)+
        };

        impl From<PyCellType> for xdmf::CellType {
            fn from(value: PyCellType) -> Self {
                match value {
                    $(PyCellType::$variant => Self::$variant,)+
                }
            }
        }

        /// The cell type with this VTK code, `None` if no cell type has it.
        fn cell_type_from_code(code: u64) -> Option<xdmf::CellType> {
            match code {
                $($code => Some(xdmf::CellType::$variant),)+
                _ => None,
            }
        }
    };
}

cell_types! {
    Vertex = 1,
    Edge = 2,
    Triangle = 4,
    Quadrilateral = 5,
    Tetrahedron = 6,
    Pyramid = 7,
    Wedge = 8,
    Hexahedron = 9,
    Edge3 = 34,
    Quadrilateral9 = 35,
    Triangle6 = 36,
    Quadrilateral8 = 37,
    Tetrahedron10 = 38,
    Pyramid13 = 39,
    Wedge15 = 40,
    Wedge18 = 41,
    Hexahedron20 = 48,
    Hexahedron24 = 49,
    Hexahedron27 = 50,
}

/// Accepts either a Python sequence of `CellType` values or a numpy integer array of raw VTK cell
/// codes (`uint8`, `uint64`, or `int64` -- copied into a `Vec` either way, since this runs once per
/// mesh rather than per time step, unlike the attribute data path in `arrays.rs`).
pub(crate) fn extract_cell_types(obj: &Bound<'_, PyAny>) -> PyResult<Vec<xdmf::CellType>> {
    if let Ok(list) = obj.extract::<Vec<PyCellType>>() {
        return Ok(list.into_iter().map(Into::into).collect());
    }

    let codes: Vec<u64> = if let Ok(array) = obj.extract::<PyReadonlyArrayDyn<'_, u8>>() {
        array
            .as_array()
            .iter()
            .map(|&code| u64::from(code))
            .collect()
    } else if let Ok(array) = obj.extract::<PyReadonlyArrayDyn<'_, u64>>() {
        array.as_array().iter().copied().collect()
    } else if let Ok(array) = obj.extract::<PyReadonlyArrayDyn<'_, i64>>() {
        array
            .as_array()
            .iter()
            .map(|&code| {
                u64::try_from(code).map_err(|_negative| {
                    PyValueError::new_err(format!("cell type code {code} is negative"))
                })
            })
            .collect::<PyResult<Vec<_>>>()?
    } else {
        return Err(PyValueError::new_err(
            "cell_types must be a sequence of xdmf.CellType values, or a numpy array of dtype \
             uint8, uint64, or int64",
        ));
    };

    codes
        .into_iter()
        .map(|code| {
            cell_type_from_code(code)
                .ok_or_else(|| PyValueError::new_err(format!("unknown cell type code {code}")))
        })
        .collect()
}

/// Type of the data (scalar, vector, tensor, etc.). `matrix`/`generic` carry a size, so this is a
/// wrapper struct with static constructors rather than a plain enum.
#[pyclass(name = "DataAttribute", eq, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyDataAttribute(xdmf::DataAttribute);

#[pymethods]
impl PyDataAttribute {
    /// Single value.
    #[classattr]
    #[pyo3(name = "SCALAR")]
    fn scalar() -> Self {
        Self(xdmf::DataAttribute::Scalar)
    }

    /// 3D vector (3 components).
    #[classattr]
    #[pyo3(name = "VECTOR")]
    fn vector() -> Self {
        Self(xdmf::DataAttribute::Vector)
    }

    /// 2nd order tensor in 3D (9 components).
    #[classattr]
    #[pyo3(name = "TENSOR")]
    fn tensor() -> Self {
        Self(xdmf::DataAttribute::Tensor)
    }

    /// Symmetric 2nd order tensor in 3D (6 components).
    #[classattr]
    #[pyo3(name = "TENSOR6")]
    fn tensor6() -> Self {
        Self(xdmf::DataAttribute::Tensor6)
    }

    /// Matrix with the given number of rows and columns.
    #[staticmethod]
    fn matrix(rows: usize, cols: usize) -> Self {
        Self(xdmf::DataAttribute::Matrix(rows, cols))
    }

    /// Generic data with the given size.
    #[staticmethod]
    fn generic(size: usize) -> Self {
        Self(xdmf::DataAttribute::Generic(size))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

impl From<PyDataAttribute> for xdmf::DataAttribute {
    fn from(value: PyDataAttribute) -> Self {
        value.0
    }
}
