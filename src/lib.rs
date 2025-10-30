//! A library for writing XDMF files, which are commonly used in scientific simulations for visualizing datasets on meshes, for example with [Paraview](https://www.paraview.org/).
//!
//! The [XDMF](https://www.xdmf.org/) (e**X**tensible **D**ata **M**odel and **F**ormat) stores the metadata in XML files and the actual data in different formats, most commonly in HDF5 files.
use std::{
    collections::BTreeMap,
    io::{Error as IoError, Result as IoResult},
    path::Path,
    str::FromStr,
};

use serde::{Deserialize, Serialize};
use xdmf_elements::{
    attribute,
    data_item::{DataContent, Format},
};

mod ascii_writer;
#[cfg(feature = "hdf5")]
mod hdf5_writer;

mod time_series_writer;
mod values;
pub mod xdmf_elements;

// Re-export types used in the public API
pub use time_series_writer::{TimeSeriesDataWriter, TimeSeriesWriter};
pub use values::Values;
pub use xdmf_elements::CellType;

/// Map for data, relates name to attribtue and values
pub type DataMap = BTreeMap<String, (DataAttribute, Values)>;

/// Type of storage used for the heavy data (e.g. ASCII or HDF5)
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum DataStorage {
    /// store the data in ASCII format, each set of data is stored in a separate file.
    Ascii,
    /// store the data in ASCII format, but inline in the XDMF file. This is only recommended for small datasets.
    AsciiInline,
    /// store the data in HDF5 format, all data in a single HDF5 file.
    Hdf5SingleFile,
    /// store the data in HDF5 format, one file per time step.
    Hdf5MultipleFiles,
}

impl FromStr for DataStorage {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ascii" => Ok(Self::Ascii),
            "asciiinline" | "ascii_inline" | "ascii-inline" => Ok(Self::AsciiInline),
            "hdf5singlefile" | "hdf5_single_file" | "hdf5-single-file" => Ok(Self::Hdf5SingleFile),
            "hdf5multiplefiles" | "hdf5_multiple_files" | "hdf5-multiple-files" => {
                Ok(Self::Hdf5MultipleFiles)
            }
            _ => Err(format!(
                "Invalid DataStorage variant: '{s}'. Valid options are: 'Ascii', 'AsciiInline', 'Hdf5SingleFile', 'Hdf5MultipleFiles'"
            )),
        }
    }
}

/// this trait defines the interface used to write the heavy data
pub(crate) trait DataWriter {
    fn format(&self) -> Format;

    fn data_storage(&self) -> DataStorage;

    fn write_mesh(&mut self, points: &[f64], cells: &[u64])
    -> IoResult<(DataContent, DataContent)>;

    fn write_data(
        &mut self,
        name: &str,
        center: attribute::Center,
        data: &Values,
    ) -> IoResult<DataContent>;

    fn write_data_initialize(&mut self, _time: &str) -> IoResult<()> {
        Ok(())
    }

    fn write_data_finalize(&mut self) -> IoResult<()> {
        Ok(())
    }

    // flush the writer, if applicable
    fn flush(&mut self) -> IoResult<()> {
        Ok(())
    }
}

/// Create a writer for the heavy data, based on the chosen data storage.
pub(crate) fn create_writer(
    file_name: &Path,
    data_storage: DataStorage,
) -> IoResult<Box<dyn DataWriter>> {
    match data_storage {
        DataStorage::Ascii => Ok(Box::new(ascii_writer::AsciiWriter::new(file_name)?)),
        DataStorage::AsciiInline => Ok(Box::new(ascii_writer::AsciiInlineWriter::new())),
        DataStorage::Hdf5SingleFile => {
            #[cfg(feature = "hdf5")]
            {
                Ok(Box::new(hdf5_writer::SingleFileHdf5Writer::new(file_name)?))
            }
            #[cfg(not(feature = "hdf5"))]
            {
                Err(IoError::other(
                    "Using Hdf5SingleFile DataStorage requires the hdf5 feature.",
                ))
            }
        }
        DataStorage::Hdf5MultipleFiles => {
            #[cfg(feature = "hdf5")]
            {
                Ok(Box::new(hdf5_writer::MultipleFilesHdf5Writer::new(
                    file_name,
                )?))
            }
            #[cfg(not(feature = "hdf5"))]
            {
                Err(IoError::other(
                    "Using Hdf5MultipleFiles DataStorage requires the hdf5 feature.",
                ))
            }
        }
    }
}

/// Check if the hdf5 feature is enabled.
pub const fn is_hdf5_enabled() -> bool {
    #[cfg(feature = "hdf5")]
    {
        true
    }
    #[cfg(not(feature = "hdf5"))]
    {
        false
    }
}

/// Type of the data (scalar, vector, tensor, etc.)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DataAttribute {
    /// single value
    Scalar,
    /// 3D vector (3 components)
    Vector,
    /// 2nd order tensor in 3D (9 components)
    Tensor,
    /// Symmetric 2nd order tensor in 3D (6 components)
    Tensor6,
    /// Matrix with specified number of rows and columns
    Matrix(usize, usize),
    /// Generic data with specified size
    Generic(usize),
}

impl DataAttribute {
    pub(crate) fn size(&self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Vector => 3,
            Self::Tensor => 9,
            Self::Tensor6 => 6,
            Self::Matrix(n, m) => n * m,
            Self::Generic(size) => *size,
        }
    }
}

impl From<DataAttribute> for attribute::AttributeType {
    fn from(data_attr: DataAttribute) -> Self {
        match data_attr {
            DataAttribute::Scalar => Self::Scalar,
            DataAttribute::Vector => Self::Vector,
            DataAttribute::Tensor => Self::Tensor,
            DataAttribute::Tensor6 => Self::Matrix, // writen as Matrix to get detected as symmetric tensor
            DataAttribute::Matrix(_, _) => Self::Matrix,
            DataAttribute::Generic(_) => Self::Matrix,
        }
    }
}

/// Create directories in a way that is safe for MPI applications.
///
/// This function will create the directory if it does not exist, and wait for it to appear in the filesystem.
/// This is particularly needed on systems such as clusters with slow filesystems, to ensure that
/// all processes can see the created directory before proceeding.
///
/// For more details check the [reference](https://github.com/KratosMultiphysics/Kratos/pull/9247).
/// Its a battle-tested solution tested with > 1000 processes
pub fn mpi_safe_create_dir_all(path: impl AsRef<Path> + std::fmt::Debug) -> IoResult<()> {
    if !&path.as_ref().exists() {
        std::fs::create_dir_all(&path).map_err(|e| {
            IoError::new(
                e.kind(),
                format!("Failed to create directory {path:?}: {e}"),
            )
        })?;
    }

    if !path.as_ref().exists() {
        // wait for the path to appear in the filesystem
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpi_safe_create_dir_all() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let dirs_to_create = tmp_dir.path().join("out/xdmf/test/folder/random/testing");

        // Try to create dirs from 100 threads concurrently
        let handles: Vec<_> = (0..100)
            .map(|_| {
                std::thread::spawn({
                    let dir_thread_local = dirs_to_create.clone();
                    move || mpi_safe_create_dir_all(dir_thread_local).unwrap()
                })
            })
            .collect();

        // join threads, will propagate errors if any
        for handle in handles {
            handle.join().unwrap();
        }

        // Check that the directory was created
        assert!(dirs_to_create.exists());
    }

    #[test]
    fn test_data_attribute() {
        let scalar = DataAttribute::Scalar;
        let vector = DataAttribute::Vector;
        let tensor = DataAttribute::Tensor;
        let tensor6 = DataAttribute::Tensor6;
        let matrix = DataAttribute::Matrix(3, 3);
        let generic = DataAttribute::Generic(5);

        assert_eq!(scalar.size(), 1);
        assert_eq!(vector.size(), 3);
        assert_eq!(tensor.size(), 9);
        assert_eq!(tensor6.size(), 6);
        assert_eq!(matrix.size(), 9);
        assert_eq!(generic.size(), 5);

        assert_eq!(attribute::AttributeType::Scalar, scalar.into());
        assert_eq!(attribute::AttributeType::Vector, vector.into());
        assert_eq!(attribute::AttributeType::Tensor, tensor.into());
        assert_eq!(attribute::AttributeType::Matrix, tensor6.into());
        assert_eq!(attribute::AttributeType::Matrix, matrix.into());
        assert_eq!(attribute::AttributeType::Matrix, generic.into());
    }

    #[test]
    fn test_data_storage_from_str() {
        // Test exact case matches
        assert_eq!("ascii".parse::<DataStorage>().unwrap(), DataStorage::Ascii);
        assert_eq!("Ascii".parse::<DataStorage>().unwrap(), DataStorage::Ascii);
        assert_eq!("ASCII".parse::<DataStorage>().unwrap(), DataStorage::Ascii);

        // Test AsciiInline variants
        assert_eq!(
            "asciiinline".parse::<DataStorage>().unwrap(),
            DataStorage::AsciiInline
        );
        assert_eq!(
            "ascii_inline".parse::<DataStorage>().unwrap(),
            DataStorage::AsciiInline
        );
        assert_eq!(
            "ascii-inline".parse::<DataStorage>().unwrap(),
            DataStorage::AsciiInline
        );

        // Test Hdf5SingleFile variants
        assert_eq!(
            "hdf5singlefile".parse::<DataStorage>().unwrap(),
            DataStorage::Hdf5SingleFile
        );
        assert_eq!(
            "hdf5_single_file".parse::<DataStorage>().unwrap(),
            DataStorage::Hdf5SingleFile
        );
        assert_eq!(
            "Hdf5-Single-File".parse::<DataStorage>().unwrap(),
            DataStorage::Hdf5SingleFile
        );

        // Test Hdf5MultipleFiles variants
        assert_eq!(
            "hdf5multiplefiles".parse::<DataStorage>().unwrap(),
            DataStorage::Hdf5MultipleFiles
        );
        assert_eq!(
            "hdf5_multiple_files".parse::<DataStorage>().unwrap(),
            DataStorage::Hdf5MultipleFiles
        );
        assert_eq!(
            "HDF5-Multiple-Files".parse::<DataStorage>().unwrap(),
            DataStorage::Hdf5MultipleFiles
        );

        // Test invalid input
        let err = "invalid".parse::<DataStorage>().unwrap_err();
        assert_eq!(
            err,
            "Invalid DataStorage variant: 'invalid'. Valid options are: 'Ascii', 'AsciiInline', 'Hdf5SingleFile', 'Hdf5MultipleFiles'"
        );

        let err = "".parse::<DataStorage>().unwrap_err();
        assert_eq!(
            err,
            "Invalid DataStorage variant: ''. Valid options are: 'Ascii', 'AsciiInline', 'Hdf5SingleFile', 'Hdf5MultipleFiles'"
        );
    }
}
