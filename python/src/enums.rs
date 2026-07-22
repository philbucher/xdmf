//! Python-facing enums mirroring the core crate's `DataStorage`, `CellType`, `DataAttribute`.

use pyo3::prelude::*;

/// Heavy-data storage format. `Hdf5SingleFile`/`Hdf5MultipleFiles` are plain attributes for the
/// default (library-chosen) deflate compression level; use `hdf5_single_file(level)`/
/// `hdf5_multiple_files(level)` to pick a specific level (0-9) instead.
#[pyclass(name = "DataStorage", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyDataStorage(pub(crate) xdmf::DataStorage);

#[pymethods]
impl PyDataStorage {
    #[classattr]
    #[allow(non_snake_case, reason = "matches Python constant naming convention")]
    fn Ascii() -> Self {
        Self(xdmf::DataStorage::Ascii)
    }

    #[classattr]
    #[allow(non_snake_case, reason = "matches Python constant naming convention")]
    fn AsciiInline() -> Self {
        Self(xdmf::DataStorage::AsciiInline)
    }

    #[classattr]
    #[allow(non_snake_case, reason = "matches Python constant naming convention")]
    fn Hdf5SingleFile() -> Self {
        Self(xdmf::DataStorage::Hdf5SingleFile { deflate_level: None })
    }

    #[classattr]
    #[allow(non_snake_case, reason = "matches Python constant naming convention")]
    fn Hdf5MultipleFiles() -> Self {
        Self(xdmf::DataStorage::Hdf5MultipleFiles { deflate_level: None })
    }

    #[classattr]
    #[allow(non_snake_case, reason = "matches Python constant naming convention")]
    fn Binary() -> Self {
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

/// Cell types as defined in the VTK file format, mirroring `xdmf::CellType`. Values match the
/// VTK/XDMF discriminants exactly.
#[pyclass(name = "CellType", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum PyCellType {
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

impl From<PyCellType> for xdmf::CellType {
    fn from(value: PyCellType) -> Self {
        match value {
            PyCellType::Vertex => Self::Vertex,
            PyCellType::Edge => Self::Edge,
            PyCellType::Triangle => Self::Triangle,
            PyCellType::Quadrilateral => Self::Quadrilateral,
            PyCellType::Tetrahedron => Self::Tetrahedron,
            PyCellType::Pyramid => Self::Pyramid,
            PyCellType::Wedge => Self::Wedge,
            PyCellType::Hexahedron => Self::Hexahedron,
            PyCellType::Edge3 => Self::Edge3,
            PyCellType::Quadrilateral9 => Self::Quadrilateral9,
            PyCellType::Triangle6 => Self::Triangle6,
            PyCellType::Quadrilateral8 => Self::Quadrilateral8,
            PyCellType::Tetrahedron10 => Self::Tetrahedron10,
            PyCellType::Pyramid13 => Self::Pyramid13,
            PyCellType::Wedge15 => Self::Wedge15,
            PyCellType::Wedge18 => Self::Wedge18,
            PyCellType::Hexahedron20 => Self::Hexahedron20,
            PyCellType::Hexahedron24 => Self::Hexahedron24,
            PyCellType::Hexahedron27 => Self::Hexahedron27,
        }
    }
}

/// Type of the data (scalar, vector, tensor, etc.). `Matrix`/`Generic` carry a size, so this is
/// a wrapper struct with static constructors rather than a plain enum.
#[pyclass(name = "DataAttribute", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyDataAttribute(pub(crate) xdmf::DataAttribute);

#[pymethods]
impl PyDataAttribute {
    /// Single value.
    #[classattr]
    #[allow(non_snake_case, reason = "matches Python constant naming convention")]
    fn SCALAR() -> Self {
        Self(xdmf::DataAttribute::Scalar)
    }

    /// 3D vector (3 components).
    #[classattr]
    #[allow(non_snake_case, reason = "matches Python constant naming convention")]
    fn VECTOR() -> Self {
        Self(xdmf::DataAttribute::Vector)
    }

    /// 2nd order tensor in 3D (9 components).
    #[classattr]
    #[allow(non_snake_case, reason = "matches Python constant naming convention")]
    fn TENSOR() -> Self {
        Self(xdmf::DataAttribute::Tensor)
    }

    /// Symmetric 2nd order tensor in 3D (6 components).
    #[classattr]
    #[allow(non_snake_case, reason = "matches Python constant naming convention")]
    fn TENSOR6() -> Self {
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
