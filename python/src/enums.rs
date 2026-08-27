//! Python-facing enums mirroring the core crate's `DataStorage`, `CellType`, `DataAttribute`.

use std::{fmt::Display, ops::Range};

use numpy::PyReadonlyArrayDyn;
use pyo3::{exceptions::PyValueError, prelude::*};

/// Heavy-data storage format. `Hdf5SingleFile`/`Hdf5MultipleFiles` are plain attributes for the
/// default (library-chosen) deflate compression level; use `hdf5_single_file(level)`/
/// `hdf5_multiple_files(level)` to pick a specific level (0-9) instead.
#[pyclass(
    name = "DataStorage",
    module = "xdmf",
    eq,
    frozen,
    hash,
    from_py_object
)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
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
// added to `xdmf::CellType` is a single edit here -- and the two `const` blocks make it a required
// one: the first pins the discriminants to the core enum's (`eq_int` exposes them to Python as the
// raw codes, and they are otherwise restated here with nothing tying the two lists together),
// the second matches exhaustively over the core enum so a variant missing from the list is a
// compile error rather than a silent gap.
macro_rules! cell_types {
    ($($variant:ident = $code:literal,)+) => {
        /// Cell types, mirroring `xdmf::CellType`. The values are the XDMF topology type codes,
        /// *not* the VTK cell codes (a hexahedron is 9 here and 12 in VTK); they match the core
        /// enum's discriminants exactly, so a raw numpy array of codes (see `extract_cell_types`)
        /// is an equivalent, cheaper-to-produce alternative to a list of these.
        #[pyclass(name = "CellType", module = "xdmf", eq, eq_int, frozen, hash, from_py_object)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(u8)]
        pub enum PyCellType {
            $($variant = $code,)+
        }

        const _: () = {
            $(assert!(PyCellType::$variant as u8 == xdmf::CellType::$variant as u8);)+
        };

        // Exhaustive over the *core* enum, so a cell type added to `xdmf::CellType` stops this
        // crate compiling until it is added to the list above
        const _: fn(xdmf::CellType) -> u8 = |cell_type| match cell_type {
            $(xdmf::CellType::$variant => $code,)+
        };

        impl From<PyCellType> for xdmf::CellType {
            fn from(value: PyCellType) -> Self {
                match value {
                    $(PyCellType::$variant => Self::$variant,)+
                }
            }
        }

        /// The cell type with this XDMF topology type code, `None` if no cell type has it.
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

/// The non-negative codes an iterator of integers holds, rejecting the first negative one by name.
/// `role` names what the codes represent ("cell type code", "submesh cell index"), since this backs
/// more than one extraction and a bare "code" would not say which.
fn non_negative_codes<T>(codes: impl IntoIterator<Item = T>, role: &str) -> PyResult<Vec<u64>>
where
    T: Copy + Display,
    u64: TryFrom<T>,
{
    codes
        .into_iter()
        .map(|code| {
            u64::try_from(code)
                .map_err(|_negative| PyValueError::new_err(format!("{role} {code} is negative")))
        })
        .collect()
}

// Generates the raw-code extraction off one dtype list, so the accepted dtypes and the message
// naming them cannot drift apart -- the same reason `arrays.rs` generates its array enums.
//
// Every integer dtype is accepted, rather than the three the codes plausibly come in: `int32` is
// what NumPy 1.x defaults to on Windows and what `meshio` hands back, so a narrower list makes
// identical source work on one platform and fail on another.
macro_rules! code_dtypes {
    ($dtypes:literal, [$($ty:ty),+ $(,)?]) => {
        const INTEGER_ARRAY_DTYPES: &str = $dtypes;

        /// The codes `obj` holds, or `None` if it is not an integer numpy array at all. `role` is
        /// forwarded to `non_negative_codes`.
        fn extract_codes(obj: &Bound<'_, PyAny>, role: &str) -> Option<PyResult<Vec<u64>>> {
            $(
                if let Ok(array) = obj.extract::<PyReadonlyArrayDyn<'_, $ty>>() {
                    return Some(non_negative_codes(array.as_array().iter().copied(), role));
                }
            )+
            None
        }
    };
}

code_dtypes!(
    "uint8, uint16, uint32, uint64, int8, int16, int32, or int64",
    [u8, u16, u32, u64, i8, i16, i32, i64]
);

/// Accepts either a Python sequence of `CellType` values or a numpy integer array of raw XDMF
/// topology type codes (copied into a `Vec` either way, since this runs once per mesh rather than
/// per time step, unlike the attribute data path in `arrays.rs`).
pub(crate) fn extract_cell_types(obj: &Bound<'_, PyAny>) -> PyResult<Vec<xdmf::CellType>> {
    if let Ok(list) = obj.extract::<Vec<PyCellType>>() {
        return Ok(list.into_iter().map(Into::into).collect());
    }

    let Some(codes) = extract_codes(obj, "cell type code") else {
        return Err(PyValueError::new_err(format!(
            "cell_types must be a sequence of xdmf.CellType values, or a numpy array of dtype \
             {INTEGER_ARRAY_DTYPES}"
        )));
    };

    codes?
        .into_iter()
        .map(|code| {
            cell_type_from_code(code)
                .ok_or_else(|| PyValueError::new_err(format!("unknown cell type code {code}")))
        })
        .collect()
}

/// The cells of one submesh: a `range`, a Python sequence of `int`, or a numpy integer array of
/// any dtype -- the same acceptance rule as `extract_cell_types` and for the same reason (`int32`
/// is what `NumPy` 1.x defaults to on Windows and what mesh generators commonly hand back).
///
/// A `range` becomes the contiguous form of [`xdmf::SubmeshCells`] without its indices ever being
/// materialised, which is the whole point of it: a submesh covering a block of a 100M-cell mesh
/// costs two numbers here rather than the ~800 MB list of every index in it.
pub(crate) fn extract_submesh_cells(
    obj: &Bound<'_, PyAny>,
) -> PyResult<xdmf::SubmeshCells<'static>> {
    const ROLE: &str = "submesh cell index";

    if let Some(range) = extract_cell_range(obj, ROLE)? {
        return Ok(xdmf::SubmeshCells::Range(range));
    }

    // The typed array path goes first: a numpy array satisfies `PySequence_Check`, so extracting
    // it as a `Vec<i64>` also succeeds -- and would convert the whole index list one Python object
    // at a time, which is exactly what reading the buffer through `PyReadonlyArrayDyn` avoids. The
    // list arm is the fallback, for the plain sequences no dtype matches.
    let codes = if let Some(codes) = extract_codes(obj, ROLE) {
        codes?
    } else if let Ok(list) = obj.extract::<Vec<i64>>() {
        non_negative_codes(list, ROLE)?
    } else {
        return Err(PyValueError::new_err(format!(
            "submesh cells must be a range, a sequence of int, or a numpy array of dtype \
             {INTEGER_ARRAY_DTYPES}"
        )));
    };

    let indices = codes
        .into_iter()
        .map(|code| {
            usize::try_from(code).map_err(|_too_large| {
                PyValueError::new_err(format!("{ROLE} {code} does not fit usize"))
            })
        })
        .collect::<PyResult<Vec<usize>>>()?;

    Ok(xdmf::SubmeshCells::from(indices))
}

/// A `range(start, stop)` of step 1, as the half-open range the core crate takes.
///
/// `None` for anything else: a non-`range` object, or a `range` with any other step -- which is
/// not a block of consecutive cells, so it reads as the plain index sequence it also is, through
/// the path above. An empty range (`range(5, 3)`) is passed on as such, so that it is rejected as
/// an empty submesh exactly as an empty list is.
fn extract_cell_range(obj: &Bound<'_, PyAny>, role: &str) -> PyResult<Option<Range<usize>>> {
    let range_type = obj.py().import("builtins")?.getattr("range")?;

    if !obj.is_instance(&range_type)? {
        return Ok(None);
    }

    if obj.getattr("step")?.extract::<i64>()? != 1 {
        return Ok(None);
    }

    let start = obj.getattr("start")?.extract::<i64>()?;
    let stop = obj.getattr("stop")?.extract::<i64>()?;

    if start < 0 {
        return Err(PyValueError::new_err(format!("{role} {start} is negative")));
    }

    let to_usize = |value: i64| {
        usize::try_from(value).map_err(|_too_large| {
            PyValueError::new_err(format!("{role} {value} does not fit usize"))
        })
    };

    // a stop below the start is an empty range whatever it is, so it needs no check of its own
    Ok(Some(to_usize(start)?..to_usize(stop.max(start))?))
}

/// Type of the data (scalar, vector, tensor, etc.). `matrix`/`generic` carry a size, so this is a
/// wrapper struct with static constructors rather than a plain enum.
#[pyclass(
    name = "DataAttribute",
    module = "xdmf",
    eq,
    frozen,
    hash,
    from_py_object
)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
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
