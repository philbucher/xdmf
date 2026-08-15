//! A library for writing XDMF files, which are commonly used in scientific simulations for visualizing datasets on meshes, for example with [Paraview](https://www.paraview.org/).
//!
//! The [XDMF](https://www.xdmf.org/) (e**X**tensible **D**ata **M**odel and **F**ormat) stores the metadata in XML files and the actual data in different formats, most commonly in HDF5 files.
use std::{path::Path, str::FromStr};

use serde::{Deserialize, Serialize};
use xdmf_elements::{
    attribute,
    data_item::{DataContent, Format},
};

mod ascii_writer;
mod binary_writer;
mod error;
#[cfg(feature = "hdf5")]
mod hdf5_writer;

mod reader;
mod time_series_writer;
mod values;
pub mod xdmf_elements;

// Re-export types used in the public API
pub use error::{Error, Result};
pub use reader::{DataInfo, TimeSeriesDataReader, TimeSeriesReader, ValueKind};
pub use time_series_writer::{TimeSeriesDataWriter, TimeSeriesWriter};
pub use values::Values;
pub use xdmf_elements::CellType;

/// Type of storage used for the heavy data (e.g. ASCII or HDF5)
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum DataStorage {
    /// store the data in ASCII format, each set of data is stored in a separate file.
    Ascii,
    /// store the data in ASCII format, but inline in the XDMF file. This is only recommended for small datasets.
    AsciiInline,
    /// store the data in HDF5 format, all data in a single HDF5 file.
    Hdf5SingleFile {
        /// zlib/deflate compression level applied to every dataset (0 = none, 9 = max).
        /// `None` uses the library default (currently 3, chosen for write speed over
        /// compression ratio).
        deflate_level: Option<u8>,
    },
    /// store the data in HDF5 format, one file per time step.
    Hdf5MultipleFiles {
        /// zlib/deflate compression level applied to every dataset (0 = none, 9 = max).
        /// `None` uses the library default (currently 3, chosen for write speed over
        /// compression ratio).
        deflate_level: Option<u8>,
    },
    /// store the data in uncompressed raw binary format, each set of data is stored in a separate file.
    Binary,
}

impl FromStr for DataStorage {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ascii" => Ok(Self::Ascii),
            "asciiinline" | "ascii_inline" | "ascii-inline" => Ok(Self::AsciiInline),
            "hdf5singlefile" | "hdf5_single_file" | "hdf5-single-file" => {
                Ok(Self::Hdf5SingleFile {
                    deflate_level: None,
                })
            }
            "hdf5multiplefiles" | "hdf5_multiple_files" | "hdf5-multiple-files" => {
                Ok(Self::Hdf5MultipleFiles {
                    deflate_level: None,
                })
            }
            "binary" => Ok(Self::Binary),
            _ => Err(format!(
                "Invalid DataStorage variant: '{s}'. Valid options are: 'Ascii', 'AsciiInline', 'Hdf5SingleFile', 'Hdf5MultipleFiles', 'Binary'"
            )),
        }
    }
}

/// this trait defines the interface used to write the heavy data
pub(crate) trait DataWriter: Send + Sync {
    fn format(&self) -> Format;

    fn data_storage(&self) -> DataStorage;

    fn write_mesh(
        &mut self,
        points: &Values<'_>,
        cells: &[u64],
    ) -> Result<(DataContent, DataContent)>;

    fn write_data(
        &mut self,
        name: &str,
        center: attribute::Center,
        data: &Values<'_>,
    ) -> Result<DataContent>;

    fn write_data_initialize(&mut self, _time: &str) -> Result<()> {
        Ok(())
    }

    fn write_data_finalize(&mut self) -> Result<()> {
        Ok(())
    }

    // flush the writer, if applicable
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    /// Validate that `data` can be represented by this backend's format, without mutating state
    /// or touching disk. Called for every attribute before `write_data_initialize` runs, so a
    /// value out of the backend's representable range is reported as an upfront caller error
    /// rather than as a mid-write failure that would otherwise leave the writer poisoned.
    fn validate_values(&self, _data: &Values<'_>) -> Result<()> {
        Ok(())
    }
}

// zlib/deflate only accepts levels 0-9; anything else is a caller mistake that should be
// rejected before a writer is constructed, rather than surfacing as a raw HDF5 error later
// (`H5Pset_deflate(): invalid deflate level`) from inside `write_mesh`.
fn validate_deflate_level(deflate_level: Option<u8>) -> Result<()> {
    if let Some(level) = deflate_level
        && level > 9
    {
        return Err(Error::InvalidConfiguration {
            reason: format!("deflate level {level} is out of range, must be between 0 and 9"),
        });
    }
    Ok(())
}

/// Create a writer for the heavy data, based on the chosen data storage.
pub(crate) fn create_writer(
    file_name: &Path,
    data_storage: DataStorage,
) -> Result<Box<dyn DataWriter>> {
    match data_storage {
        DataStorage::Ascii => Ok(Box::new(ascii_writer::AsciiWriter::new(file_name)?)),
        DataStorage::AsciiInline => Ok(Box::new(ascii_writer::AsciiInlineWriter::new())),
        DataStorage::Hdf5SingleFile { deflate_level } => {
            validate_deflate_level(deflate_level)?;
            #[cfg(feature = "hdf5")]
            {
                Ok(Box::new(hdf5_writer::SingleFileHdf5Writer::new(
                    file_name,
                    deflate_level.unwrap_or(hdf5_writer::DEFAULT_DEFLATE_LEVEL),
                )?))
            }
            #[cfg(not(feature = "hdf5"))]
            {
                Err(Error::InvalidConfiguration {
                    reason: format!(
                        "using {data_storage:?} DataStorage requires the 'hdf5' feature"
                    ),
                })
            }
        }
        DataStorage::Hdf5MultipleFiles { deflate_level } => {
            validate_deflate_level(deflate_level)?;
            #[cfg(feature = "hdf5")]
            {
                Ok(Box::new(hdf5_writer::MultipleFilesHdf5Writer::new(
                    file_name,
                    deflate_level.unwrap_or(hdf5_writer::DEFAULT_DEFLATE_LEVEL),
                )?))
            }
            #[cfg(not(feature = "hdf5"))]
            {
                Err(Error::InvalidConfiguration {
                    reason: format!(
                        "using {data_storage:?} DataStorage requires the 'hdf5' feature"
                    ),
                })
            }
        }
        DataStorage::Binary => Ok(Box::new(binary_writer::BinaryWriter::new(file_name)?)),
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
            DataAttribute::Tensor6 => Self::Matrix, // written as Matrix to get detected as symmetric tensor
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
pub fn mpi_safe_create_dir_all(path: impl AsRef<Path> + std::fmt::Debug) -> Result<()> {
    if !&path.as_ref().exists() {
        std::fs::create_dir_all(&path)
            .map_err(error::io_ctx("creating directory", path.as_ref()))?;
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
            DataStorage::Hdf5SingleFile {
                deflate_level: None
            }
        );
        assert_eq!(
            "hdf5_single_file".parse::<DataStorage>().unwrap(),
            DataStorage::Hdf5SingleFile {
                deflate_level: None
            }
        );
        assert_eq!(
            "Hdf5-Single-File".parse::<DataStorage>().unwrap(),
            DataStorage::Hdf5SingleFile {
                deflate_level: None
            }
        );

        // Test Hdf5MultipleFiles variants
        assert_eq!(
            "hdf5multiplefiles".parse::<DataStorage>().unwrap(),
            DataStorage::Hdf5MultipleFiles {
                deflate_level: None
            }
        );
        assert_eq!(
            "hdf5_multiple_files".parse::<DataStorage>().unwrap(),
            DataStorage::Hdf5MultipleFiles {
                deflate_level: None
            }
        );
        assert_eq!(
            "HDF5-Multiple-Files".parse::<DataStorage>().unwrap(),
            DataStorage::Hdf5MultipleFiles {
                deflate_level: None
            }
        );

        // Test Binary variant
        assert_eq!(
            "binary".parse::<DataStorage>().unwrap(),
            DataStorage::Binary
        );
        assert_eq!(
            "Binary".parse::<DataStorage>().unwrap(),
            DataStorage::Binary
        );

        // Test invalid input
        let err = "invalid".parse::<DataStorage>().unwrap_err();
        assert_eq!(
            err,
            "Invalid DataStorage variant: 'invalid'. Valid options are: 'Ascii', 'AsciiInline', 'Hdf5SingleFile', 'Hdf5MultipleFiles', 'Binary'"
        );

        let err = "".parse::<DataStorage>().unwrap_err();
        assert_eq!(
            err,
            "Invalid DataStorage variant: ''. Valid options are: 'Ascii', 'AsciiInline', 'Hdf5SingleFile', 'Hdf5MultipleFiles', 'Binary'"
        );
    }

    #[test]
    fn test_validate_deflate_level() {
        validate_deflate_level(None).unwrap();
        validate_deflate_level(Some(0)).unwrap();
        validate_deflate_level(Some(9)).unwrap();

        std::assert_matches!(
            validate_deflate_level(Some(10)).unwrap_err(),
            Error::InvalidConfiguration { reason } if reason.contains("deflate level 10")
        );
        std::assert_matches!(
            validate_deflate_level(Some(255)).unwrap_err(),
            Error::InvalidConfiguration { reason } if reason.contains("deflate level 255")
        );
    }

    #[test]
    fn create_writer_rejects_invalid_deflate_level() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("test.xdmf");

        for storage in [
            DataStorage::Hdf5SingleFile {
                deflate_level: Some(10),
            },
            DataStorage::Hdf5MultipleFiles {
                deflate_level: Some(10),
            },
        ] {
            let Err(err) = create_writer(&file_name, storage) else {
                panic!("expected an error for deflate_level 10");
            };
            std::assert_matches!(err, Error::InvalidConfiguration { reason } if reason.contains("deflate level 10"));
        }
    }
}
