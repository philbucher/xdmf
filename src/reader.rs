//! `TimeSeriesReader`/`TimeSeriesDataReader`: the reader counterpart to `TimeSeriesWriter`.
//!
//! Everything needed to locate and decode heavy data is driven off each `DataItem`'s own
//! `Format`/`Precision`/`NumberType`/`Endian`/`Dimensions` rather than off `DataStorage`; that is
//! what makes reading a file written by another tool (almost) free, since a foreign file carries
//! the same `DataItem` information, and it is why a document that mixes formats across `DataItem`s
//! just works.

use std::path::{Path, PathBuf};

use crate::{
    CellType, DataAttribute, Error, Result,
    values::ValueType,
    xdmf_elements::attribute::{self, Attribute, AttributeType},
};

mod ascii_reader;
mod binary_reader;
mod data_reader;
#[cfg(feature = "hdf5")]
mod hdf5_reader;
mod light_data;
mod topology;

use data_reader::SourceKind;
use light_data::LightData;

/// Parses an `.xdmf` file's light data (XML metadata). Does not touch heavy data until
/// [`read_mesh`](Self::read_mesh) is called.
pub struct TimeSeriesReader {
    light_data: LightData,
    xdmf_path: PathBuf,
}

impl TimeSeriesReader {
    /// Parses the light data of `file_name`.
    /// ```rust
    /// use xdmf::{TimeSeriesReader, TimeSeriesWriter};
    ///
    /// let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    /// let connectivity = [0, 1, 2];
    /// let cell_types = [xdmf::CellType::Triangle];
    /// TimeSeriesWriter::new("xdmf_reader_new", xdmf::DataStorage::AsciiInline)
    ///     .unwrap()
    ///     .write_mesh(&coords, &connectivity, &cell_types)
    ///     .unwrap();
    ///
    /// let reader = TimeSeriesReader::new("xdmf_reader_new.xdmf2").unwrap();
    /// assert_eq!(reader.num_points(), 3);
    /// assert_eq!(reader.num_cells(), 1);
    /// ```
    pub fn new(file_name: impl AsRef<Path>) -> Result<Self> {
        let xdmf_path = file_name.as_ref().to_path_buf();
        let light_data = light_data::parse(&xdmf_path)?;
        Ok(Self {
            light_data,
            xdmf_path,
        })
    }

    /// Number of points in the mesh.
    pub fn num_points(&self) -> usize {
        self.light_data.num_points
    }

    /// Number of cells in the mesh. For a point cloud (no cells written), this equals
    /// [`num_points`](Self::num_points).
    pub fn num_cells(&self) -> usize {
        self.light_data.num_cells
    }

    /// Time step labels, in file order. Empty for a mesh-only file with no time steps written.
    pub fn times(&self) -> &[String] {
        &self.light_data.times
    }

    /// Reads the mesh into the caller's buffers (each cleared first, capacity reused), and
    /// returns a [`TimeSeriesDataReader`] for reading per-step attribute data.
    ///
    /// A point cloud (no cells written by `write_mesh`) round-trips to an empty `cell_types`, not
    /// one `Vertex` cell per point.
    pub fn read_mesh(
        self,
        points: &mut Vec<f64>,
        connectivity: &mut Vec<u64>,
        cell_types: &mut Vec<CellType>,
    ) -> Result<TimeSeriesDataReader> {
        let base_dir = self.light_data.base_dir.as_path();

        let raw_points = data_reader::read_data_item(
            &self.light_data.geometry_data_item,
            base_dir,
            &self.xdmf_path,
        )?;
        *points = data_reader::into_f64(raw_points, &self.xdmf_path)?;

        let raw_connectivity = data_reader::read_data_item(
            &self.light_data.topology_data_item,
            base_dir,
            &self.xdmf_path,
        )?;
        let raw_connectivity = data_reader::into_u64(raw_connectivity, &self.xdmf_path)?;

        let (decoded_connectivity, decoded_cell_types) = topology::invert(
            self.light_data.topology_type,
            &raw_connectivity,
            self.light_data.num_points,
            self.light_data.num_cells,
            &self.xdmf_path,
        )?;
        *connectivity = decoded_connectivity;
        *cell_types = decoded_cell_types;

        Ok(TimeSeriesDataReader {
            base_dir: self.light_data.base_dir,
            xdmf_path: self.xdmf_path,
            times: self.light_data.times,
            steps: self.light_data.steps,
            num_points: self.light_data.num_points,
            num_cells: self.light_data.num_cells,
        })
    }
}

/// Reader for per-step point/cell attribute data, obtained from
/// [`TimeSeriesReader::read_mesh`].
pub struct TimeSeriesDataReader {
    base_dir: PathBuf,
    xdmf_path: PathBuf,
    times: Vec<String>,
    steps: Vec<Vec<Attribute>>,
    num_points: usize,
    num_cells: usize,
}

/// The element type a stored data field can be read as, mirroring [`crate::Values`]'s variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueKind {
    /// f32 values
    F32,
    /// f64 values
    F64,
    /// u64 values
    U64,
}

/// Metadata about one point/cell data field, without reading its heavy data. Lets a caller size
/// a buffer and pick a matching `T` for [`TimeSeriesDataReader::read_point_data`]/
/// [`read_cell_data`](TimeSeriesDataReader::read_cell_data) ahead of time.
#[derive(Clone, Debug)]
pub struct DataInfo {
    /// The field's name.
    pub name: String,
    /// The field's tensor shape.
    pub attribute: DataAttribute,
    /// The field's element type.
    pub kind: ValueKind,
    /// Total number of elements, i.e. `num_entities * attribute.size()`.
    pub len: usize,
}

impl TimeSeriesDataReader {
    /// Number of time steps written.
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Time step labels, in file order.
    pub fn times(&self) -> &[String] {
        &self.times
    }

    /// Number of point-data fields at `step`.
    pub fn num_point_data(&self, step: usize) -> Result<usize> {
        Ok(self.filtered(step, attribute::Center::Node)?.len())
    }

    /// Number of cell-data fields at `step`.
    pub fn num_cell_data(&self, step: usize) -> Result<usize> {
        Ok(self.filtered(step, attribute::Center::Cell)?.len())
    }

    /// Metadata for the point-data field at `index` within `step`.
    pub fn point_data_info(&self, step: usize, index: usize) -> Result<DataInfo> {
        self.data_info(step, attribute::Center::Node, index, self.num_points)
    }

    /// Metadata for the cell-data field at `index` within `step`.
    pub fn cell_data_info(&self, step: usize, index: usize) -> Result<DataInfo> {
        self.data_info(step, attribute::Center::Cell, index, self.num_cells)
    }

    /// Index of the point-data field named `name` at `step`, for callers that think in names.
    pub fn point_data_index(&self, step: usize, name: &str) -> Result<usize> {
        self.data_index(step, attribute::Center::Node, name)
    }

    /// Index of the cell-data field named `name` at `step`.
    pub fn cell_data_index(&self, step: usize, name: &str) -> Result<usize> {
        self.data_index(step, attribute::Center::Cell, name)
    }

    /// Reads the point-data field at `index` within `step` into `into`, replacing its contents.
    ///
    /// Widening succeeds (e.g. a `Precision="4"` file read into `Vec<f64>`); narrowing (a
    /// `Precision="8"` file into `Vec<f32>`) and a kind mismatch (float into `Vec<u64>` or vice
    /// versa) both error — check [`point_data_info`](Self::point_data_info) first if unsure.
    pub fn read_point_data<T: ValueType>(
        &mut self,
        step: usize,
        index: usize,
        into: &mut Vec<T>,
    ) -> Result<()> {
        self.read_data(step, attribute::Center::Node, index, into)
    }

    /// Reads the cell-data field at `index` within `step` into `into`. See
    /// [`read_point_data`](Self::read_point_data) for the widening/narrowing rules.
    pub fn read_cell_data<T: ValueType>(
        &mut self,
        step: usize,
        index: usize,
        into: &mut Vec<T>,
    ) -> Result<()> {
        self.read_data(step, attribute::Center::Cell, index, into)
    }

    fn step_attributes(&self, step: usize) -> Result<&[Attribute]> {
        self.steps
            .get(step)
            .map(Vec::as_slice)
            .ok_or_else(|| Error::InvalidData {
                reason: format!(
                    "step index {step} is out of bounds, there are {} steps",
                    self.steps.len()
                ),
            })
    }

    fn filtered(&self, step: usize, center: attribute::Center) -> Result<Vec<&Attribute>> {
        Ok(self
            .step_attributes(step)?
            .iter()
            .filter(|attr| attr.center == center)
            .collect())
    }

    fn data_info(
        &self,
        step: usize,
        center: attribute::Center,
        index: usize,
        num_entities: usize,
    ) -> Result<DataInfo> {
        let filtered = self.filtered(step, center)?;
        let attr = filtered.get(index).ok_or_else(|| Error::InvalidData {
            reason: format!(
                "{center:?}-data index {index} is out of bounds at step {step}, there are {} fields",
                filtered.len()
            ),
        })?;
        build_data_info(attr, num_entities, &self.xdmf_path)
    }

    fn data_index(&self, step: usize, center: attribute::Center, name: &str) -> Result<usize> {
        let filtered = self.filtered(step, center)?;
        filtered
            .iter()
            .position(|attr| attr.name == name)
            .ok_or_else(|| Error::InvalidData {
                reason: format!("no {center:?}-data named '{name}' at step {step}"),
            })
    }

    fn read_data<T: ValueType>(
        &mut self,
        step: usize,
        center: attribute::Center,
        index: usize,
        into: &mut Vec<T>,
    ) -> Result<()> {
        let filtered = self.filtered(step, center)?;
        let attr = filtered.get(index).ok_or_else(|| Error::InvalidData {
            reason: format!(
                "{center:?}-data index {index} is out of bounds at step {step}, there are {} fields",
                filtered.len()
            ),
        })?;
        let item = single_data_item(attr)?;
        if item.reference.is_some() {
            return Err(Error::Unsupported {
                reason: "Reference on an Attribute DataItem is not supported".to_string(),
            });
        }

        let source = data_reader::read_data_item(item, &self.base_dir, &self.xdmf_path)?;
        data_reader::assign(source, T::as_values_mut(into))
    }
}

fn single_data_item(attr: &Attribute) -> Result<&crate::xdmf_elements::data_item::DataItem> {
    if attr.data_items.len() != 1 {
        return Err(Error::Unsupported {
            reason: format!(
                "Attribute '{}' has {} DataItem children, only exactly one is supported",
                attr.name,
                attr.data_items.len()
            ),
        });
    }
    Ok(&attr.data_items[0])
}

fn build_data_info(attr: &Attribute, num_entities: usize, xdmf_path: &Path) -> Result<DataInfo> {
    let item = single_data_item(attr)?;
    let dims = item.dimensions.as_ref().ok_or_else(|| Error::InvalidFile {
        path: xdmf_path.to_path_buf(),
        reason: format!("Attribute '{}' DataItem has no Dimensions", attr.name),
    })?;
    let total_len: usize = dims.0.iter().product();
    if num_entities == 0 || !total_len.is_multiple_of(num_entities) {
        return Err(Error::InvalidFile {
            path: xdmf_path.to_path_buf(),
            reason: format!(
                "Attribute '{}' has {total_len} values, which does not divide evenly across {num_entities} entities",
                attr.name
            ),
        });
    }
    let component_size = total_len / num_entities;

    let data_attribute = match attr.attribute_type {
        AttributeType::Scalar if component_size == 1 => DataAttribute::Scalar,
        AttributeType::Vector if component_size == 3 => DataAttribute::Vector,
        AttributeType::Tensor if component_size == 9 => DataAttribute::Tensor,
        AttributeType::Tensor6 if component_size == 6 => DataAttribute::Tensor6,
        AttributeType::Matrix => DataAttribute::Generic(component_size),
        _ => {
            return Err(Error::InvalidFile {
                path: xdmf_path.to_path_buf(),
                reason: format!(
                    "Attribute '{}' has AttributeType={:?} but {component_size} values per entity",
                    attr.name, attr.attribute_type
                ),
            });
        }
    };

    let kind = match SourceKind::from_data_item(item, xdmf_path)? {
        SourceKind::F32 => ValueKind::F32,
        SourceKind::F64 => ValueKind::F64,
        SourceKind::U32 | SourceKind::U64 => ValueKind::U64,
    };

    Ok(DataInfo {
        name: attr.name.clone(),
        attribute: data_attribute,
        kind,
        len: total_len,
    })
}
