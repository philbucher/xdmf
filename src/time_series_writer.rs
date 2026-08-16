//! This module contains functionalities for writing a series of time steps to XDMF.
//!
//! The mesh is written only once and then referenced in each time step.
//! This is a significant advantage over VTK based formats, making it more efficient both in terms of storage size as well as write speed.
//!
//! The concept is inspired by the `TimeSeriesWriter` of [meshio](https://github.com/nschloe/meshio)

use std::{
    collections::{HashMap, HashSet},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use crate::{
    CellType, Coordinate, DataAttribute, DataStorage, DataWriter, Error, Result, Values,
    create_writer,
    error::io_ctx,
    mpi_safe_create_dir_all,
    xdmf_elements::{
        Information, Xdmf, attribute,
        data_item::{DataItem, NumberType},
        dimensions::Dimensions,
        geometry::{Geometry, GeometryType},
        grid::{CollectionType, Grid, GridType, Time},
        topology::{Topology, TopologyType},
    },
};

/// Writer for time series data in XDMF format.
pub struct TimeSeriesWriter {
    xdmf_file_name: PathBuf,
    writer: Box<dyn DataWriter>,
}

impl TimeSeriesWriter {
    /// Create a new `TimeSeriesWriter`.
    /// ```rust
    /// use xdmf::TimeSeriesWriter;
    /// let xdmf_writer = TimeSeriesWriter::new("name_xdmf_file", xdmf::DataStorage::AsciiInline)
    ///     .expect("failed to create XDMF writer");
    /// ```
    pub fn new(file_name: impl AsRef<Path>, data_storage: DataStorage) -> Result<Self> {
        let xdmf_file_name = file_name.as_ref().to_path_buf().with_extension("xdmf2");

        validate_file_name(&xdmf_file_name)?;

        // create the parent directory if it does not exist
        if let Some(parent) = xdmf_file_name.parent() {
            mpi_safe_create_dir_all(parent)?;
        }

        Ok(Self {
            xdmf_file_name,
            writer: create_writer(file_name.as_ref(), data_storage)?,
        })
    }

    /// Writes the mesh to the XDMF file, returning a `TimeSeriesDataWriter` for writing time steps.
    ///
    /// Sizes of the inputs are validated to ensure consistency with the mesh and defined cell types.
    ///
    /// The coordinates are taken as `f32` or `f64`, whichever the caller already holds, and are
    /// written at that precision.
    ///
    /// ```rust
    /// use xdmf::TimeSeriesWriter;
    /// let xdmf_writer = TimeSeriesWriter::new("xdmf_write_mesh", xdmf::DataStorage::AsciiInline)
    ///     .expect("failed to create XDMF writer");
    ///
    /// // define 3 points and 2 cells (a line and a triangle)
    /// let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    /// let connectivity = [0, 1, 0, 2, 1]; // line (0,1) and triangle (0,2,1)
    /// let cell_types = [xdmf::CellType::Edge, xdmf::CellType::Triangle];
    ///
    /// // write the mesh
    /// let mut ts_writer = xdmf_writer.write_mesh(&coords, &connectivity, &cell_types);
    /// # // hidden: doctests run in the crate root, so the example cleans up after itself
    /// # std::fs::remove_file("xdmf_write_mesh.xdmf2").expect("the example writes this file");
    /// ```
    pub fn write_mesh<C: Coordinate>(
        mut self,
        points: &[C],
        connectivity: &[u64],
        cell_types: &[CellType],
    ) -> Result<TimeSeriesDataWriter> {
        validate_points_and_cells(points.len(), connectivity, cell_types)?;

        let points = C::as_values(points);
        let num_points = points.len() / 3;
        let num_cells = if cell_types.is_empty() {
            num_points
        } else {
            cell_types.len()
        };

        let (topo_type, prepared_cells) = prepare_cells(connectivity, cell_types, num_points);

        let (points_data, cells_data) = self.writer.write_mesh(&points, &prepared_cells)?;

        let format = self.writer.format();

        let data_item_coords = DataItem {
            name: Some("coords".to_string()),
            dimensions: Some(Dimensions(vec![num_points, 3])),
            data: points_data,
            number_type: Some(points.number_type()),
            precision: Some(points.precision(format)),
            format: Some(format),
            endian: format.endian(),
            reference: None,
        };

        let data_item_connectivity = DataItem {
            name: Some("connectivity".to_string()),
            dimensions: Some(Dimensions(vec![prepared_cells.len()])),
            number_type: Some(NumberType::UInt),
            data: cells_data,
            format: Some(format),
            precision: Some(format.int_precision()),
            endian: format.endian(),
            reference: None,
        };

        let data_item_coords_ref =
            DataItem::new_reference(&data_item_coords, "/Xdmf/Domain/DataItem");
        let data_item_connectivity_ref =
            DataItem::new_reference(&data_item_connectivity, "/Xdmf/Domain/DataItem");

        let geometry = Geometry {
            geometry_type: GeometryType::XYZ,
            data_item: data_item_coords_ref,
        };
        let topology = Topology {
            topology_type: topo_type,
            number_of_elements: num_cells.to_string(),
            data_item: data_item_connectivity_ref,
        };

        let mut ts_writer = TimeSeriesDataWriter {
            xdmf_file_name: self.xdmf_file_name,
            writer: self.writer,
            grid: Grid::new_uniform("mesh", geometry, topology),
            data_items: vec![data_item_coords, data_item_connectivity],
            attributes: vec![],
            written_times: HashMap::new(),
            num_points,
            num_cells,
        };

        ts_writer.write_xdmf_file()?;

        Ok(ts_writer)
    }
}

// Validate that the points and cells are valid.
fn validate_points_and_cells(
    num_coordinates: usize,
    connectivity: &[u64],
    cell_types: &[CellType],
) -> Result<()> {
    // at least one point is required
    if num_coordinates == 0 {
        return Err(Error::InvalidMesh {
            reason: "at least one point is required".to_string(),
        });
    }

    // check that points are a multiple of 3 (x, y, z)
    if !num_coordinates.is_multiple_of(3) {
        return Err(Error::InvalidMesh {
            reason: format!(
                "points must have 3 dimensions, but {num_coordinates} is not a multiple of 3"
            ),
        });
    }

    // check cells connectivity indices
    let max_connectivity_index = connectivity.iter().max();

    if let Some(&max_index) = max_connectivity_index
        && max_index as usize >= num_coordinates / 3
    {
        return Err(Error::InvalidMesh {
            reason: format!(
                "connectivity index {max_index} is out of bounds, the mesh only has {} points",
                num_coordinates / 3
            ),
        });
    }

    // check that the number of connectivities matches the expected number based on the cell types
    let exp_num_points: usize = cell_types.iter().map(|ct| ct.num_points()).sum();
    if exp_num_points != connectivity.len() {
        return Err(Error::InvalidMesh {
            reason: format!(
                "size of connectivity ({}) does not match the number expected from the cell types ({exp_num_points})",
                connectivity.len()
            ),
        });
    }

    Ok(())
}

// Poly-cells need to additionally specify the number of points
fn poly_cell_points(cell_type: CellType) -> Option<u64> {
    // For polyvertex and polyline, need to add the number of points
    match cell_type {
        CellType::Vertex => {
            // polyvertex with one point
            Some(1)
        }
        CellType::Edge => {
            // polyline with two points
            Some(2)
        }
        _ => None,
    }
}

/// Prepare cells / connectivity for writing. The cell type is prepended to the connectivity list,
/// and for poly-cells, the number of points is also added.
/// TODO if all cells are the same, then the type information can be stored as `TopologyType`
fn prepare_cells(
    connectivity: &[u64],
    cell_types: &[CellType],
    num_points: usize,
) -> (TopologyType, Vec<u64>) {
    if cell_types.is_empty() {
        // if there are no cells, use polyvertex on nodes
        // this is required by paraview to visualize only points
        return (TopologyType::Polyvertex, (0..num_points as u64).collect());
    }

    let mut cells_with_types = Vec::with_capacity(connectivity.len() + cell_types.len());
    let mut index = 0_usize;

    for cell_type in cell_types {
        let num_points = cell_type.num_points();
        cells_with_types.push(*cell_type as u64);

        if let Some(n_points_poly) = poly_cell_points(*cell_type) {
            // poly-cells need to specify the number of points
            cells_with_types.push(n_points_poly);
        }

        cells_with_types.extend_from_slice(&connectivity[index..index + num_points]);

        index += num_points; // move index to the next cell
    }

    (TopologyType::Mixed, cells_with_types)
}

/// Writer for time series data in XDMF format. Can be used after writing the mesh with `TimeSeriesWriter::write_mesh`.
pub struct TimeSeriesDataWriter {
    xdmf_file_name: PathBuf,
    writer: Box<dyn DataWriter>,
    grid: Grid,
    data_items: Vec<DataItem>,
    attributes: Vec<(String, Vec<attribute::Attribute>)>,
    // Keyed on `f64::to_bits` of the parsed time, not the caller's string, so two spellings of
    // the same instant (e.g. "0.1" and "0.10") are recognized as the same duplicate.
    written_times: HashMap<u64, String>,
    num_points: usize,
    num_cells: usize,
}

impl TimeSeriesDataWriter {
    /// Write one time step, passing a [`TimeStep`] to `write_step` to write its data into.
    ///
    /// The step is completed when `write_step` returns: on `Ok` its `<Grid>` is added to the XDMF
    /// file, on `Err` the step is discarded and the heavy data already written for it is removed
    /// again. Should that removal fail in turn, the caller's error is still the one reported --
    /// so heavy data can be left behind, unreferenced by any `<Grid>`, without that being said.
    ///
    /// The time is accepted as a str to avoid dealing with formatting, thus leaving it to the
    /// user. It is validated once, up front, to reject duplicated times. Each attribute is
    /// validated as it is passed: its name, and its size against the mesh and the
    /// [`DataAttribute`] it declares.
    ///
    /// `write_step` may fail with any error type that this crate's [`Error`] converts into, so a
    /// caller can abort a step with an error of their own and get it back unchanged.
    ///
    /// The error type is inferred from the closure, which works as long as one type is implied --
    /// as in the example below, where every `?` is on an [`Error`]. A closure that mixes error
    /// types (say a `?` on an [`Error`] and one on the caller's own error) leaves nothing to
    /// infer from and needs the type stated, most readably as a return type on the closure:
    /// `|step| -> Result<(), MyError> { ... }`.
    ///
    /// A step contains exactly the attributes that were written successfully: if `write_step`
    /// ignores the error of a rejected attribute and returns `Ok` anyway, the step is written
    /// without that attribute rather than failing. Only a step left with no attributes at all is
    /// rejected, since a `<Grid>` without data is of no use. Propagating the error with `?` --
    /// as the example below does -- is therefore what keeps a step all-or-nothing.
    ///
    /// ```rust
    /// use xdmf::TimeSeriesWriter;
    /// let xdmf_writer = TimeSeriesWriter::new("xdmf_write_data", xdmf::DataStorage::AsciiInline)
    ///     .expect("failed to create XDMF writer");
    ///
    /// // define 3 points and 2 cells (a line and a triangle)
    /// let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    /// let connectivity = [0, 1, 0, 2, 1]; // line (0,1) and triangle (0,2,1)
    /// let cell_types = [xdmf::CellType::Edge, xdmf::CellType::Triangle];
    ///
    /// // write the mesh
    /// let mut time_series_writer = xdmf_writer
    ///     .write_mesh(&coords, &connectivity, &cell_types)
    ///     .expect("failed to write mesh");
    ///
    /// // each attribute is written as it is passed, so a single buffer can be refilled and
    /// // reused for every field of every time step
    /// let mut point_values = vec![0.0; 9];
    /// let cell_values = vec![0.0, 1.0];
    ///
    /// // write the data for 10 time steps
    /// for i in 0..10 {
    ///     time_series_writer
    ///         .write_time_step(&i.to_string(), |step| {
    ///             step.point_data("point_data", xdmf::DataAttribute::Vector, &point_values)?;
    ///
    ///             point_values.fill(i as f64); // refill the same buffer for the next attribute
    ///             step.point_data(
    ///                 "more_point_data",
    ///                 xdmf::DataAttribute::Vector,
    ///                 &point_values,
    ///             )?;
    ///
    ///             step.cell_data("cell_data", xdmf::DataAttribute::Scalar, &cell_values)
    ///         })
    ///         .expect("failed to write time step");
    /// }
    /// # // hidden: doctests run in the crate root, so the example cleans up after itself
    /// # std::fs::remove_file("xdmf_write_data.xdmf2").expect("the example writes this file");
    /// ```
    pub fn write_time_step<F, E>(&mut self, time: &str, write_step: F) -> Result<(), E>
    where
        F: FnOnce(&mut TimeStep<'_>) -> Result<(), E>,
        E: From<Error>,
    {
        let parsed_time = time
            .parse::<f64>()
            .map_err(|_parse_error| Error::InvalidTimeStep {
                time: time.to_string(),
                reason: "must be a valid float".to_string(),
            })?;

        // `f64::from_str` accepts "NaN"/"inf"/"infinity" and overflows large literals to infinity,
        // none of which name an instant a reader can place on a time line
        if !parsed_time.is_finite() {
            return Err(Error::InvalidTimeStep {
                time: time.to_string(),
                reason: "must be a finite float".to_string(),
            }
            .into());
        }

        // Zero is normalized because -0.0 and 0.0 are the same instant with different bit
        // patterns, which the duplicate check below would otherwise take for two different times.
        let time_bits = if parsed_time == 0.0 { 0.0 } else { parsed_time }.to_bits();

        // check if the time step has already been written, keyed on the parsed value rather
        // than the string so different spellings of the same instant are caught too (e.g "0.1" == "0.10")
        if let Some(existing) = self.written_times.get(&time_bits) {
            // naming the earlier spelling is only informative if it differs from this one
            let reason = if existing == time {
                "already written".to_string()
            } else {
                format!("already written (as '{existing}')")
            };
            return Err(Error::InvalidTimeStep {
                time: time.to_string(),
                reason,
            }
            .into());
        }

        let mut step = TimeStep {
            writer: self,
            time: time.to_string(),
            time_bits,
            attributes: Vec::new(),
            point_names: HashSet::new(),
            cell_names: HashSet::new(),
            initialized: false,
        };

        match write_step(&mut step) {
            Ok(()) => step.finish().map_err(E::from),
            Err(error) => {
                // The caller's error is the one they can act on, so it is returned even if the
                // cleanup fails too -- reporting "could not remove file" instead would hide why
                // the step failed in the first place.
                let _discard_result = step.discard();
                Err(error)
            }
        }
    }

    fn write_xdmf_file(&mut self) -> Result<()> {
        self.writer.flush()?;

        // create the XDMF structure
        let time_grids = self
            .attributes
            .iter()
            .map(|(time, attributes)| {
                let mut grid = self.grid.clone();

                match grid.grid_type {
                    GridType::Uniform => {
                        grid.name = format!("time_series-t{time}");
                        grid.time = Some(Time::new(time));
                        grid.attributes = Some(attributes.clone());
                        grid
                    }
                    _ => unimplemented!("Only Uniform grids are supported for time series"),
                }
            })
            .collect();

        let temporal_grid =
            Grid::new_collection("time_series", CollectionType::Temporal, Some(time_grids));

        // If there are no attributes aka time-data, write the grid directly
        let grid_to_write = if self.attributes.is_empty() {
            self.grid.clone()
        } else {
            temporal_grid
        };

        let mut xdmf = Xdmf {
            information: vec![
                Information::new("data_storage", format!("{:?}", self.writer.data_storage())),
                Information::new("version", env!("CARGO_PKG_VERSION")),
            ],
            ..Default::default()
        };
        xdmf.domains[0].grids.push(grid_to_write);
        xdmf.domains[0].data_items.extend(self.data_items.clone());

        // Write the XDMF file to a temporary file first to avoid access races
        let temp_xdmf_file_name = self.xdmf_file_name.with_extension("xdmf.tmp");

        let mut xdmf_file = BufWriter::new(
            std::fs::File::create(&temp_xdmf_file_name)
                .map_err(io_ctx("creating XDMF file", &temp_xdmf_file_name))?,
        );
        xdmf.write_to(&mut xdmf_file)
            .map_err(io_ctx("writing XDMF XML", &temp_xdmf_file_name))?;
        xdmf_file
            .flush()
            .map_err(io_ctx("flushing XDMF file", &temp_xdmf_file_name))?;

        std::fs::rename(&temp_xdmf_file_name, &self.xdmf_file_name)
            .map_err(io_ctx("renaming XDMF file", &temp_xdmf_file_name))
    }
}

/// A single time step being written, handed to the closure passed to
/// [`TimeSeriesDataWriter::write_time_step`].
///
/// Each [`point_data`](Self::point_data)/[`cell_data`](Self::cell_data) call writes its heavy data
/// before returning, so the caller's buffer is free again immediately and one buffer can serve
/// every field of the step. The light data (XML) is written once, after the closure returns.
///
/// A step needs at least one attribute; returning from the closure without writing any is an
/// error. Returning an error discards the step: no `<Grid>` is added to the XDMF file, the heavy
/// data already written for it is removed again, and the time stays available.
pub struct TimeStep<'a> {
    writer: &'a mut TimeSeriesDataWriter,
    time: String,
    time_bits: u64,
    attributes: Vec<attribute::Attribute>,
    // Point and cell names are tracked separately: the same name may be used for one of each
    point_names: HashSet<String>,
    cell_names: HashSet<String>,
    // Whether `write_data_initialize` has run. Deferred to the first attribute so that a step
    // which never writes anything leaves no trace at all
    initialized: bool,
}

impl TimeStep<'_> {
    /// Write one point attribute, immediately.
    pub fn point_data<'v>(
        &mut self,
        name: &str,
        attribute: DataAttribute,
        data: impl Into<Values<'v>>,
    ) -> Result<()> {
        self.write_attribute(name, attribute, data.into(), attribute::Center::Node)
    }

    /// Write one cell attribute, immediately.
    pub fn cell_data<'v>(
        &mut self,
        name: &str,
        attribute: DataAttribute,
        data: impl Into<Values<'v>>,
    ) -> Result<()> {
        self.write_attribute(name, attribute, data.into(), attribute::Center::Cell)
    }

    fn write_attribute(
        &mut self,
        name: &str,
        data_attribute: DataAttribute,
        values: Values<'_>,
        center: attribute::Center,
    ) -> Result<()> {
        let is_point_data = center == attribute::Center::Node;
        let (label, num_entities) = if is_point_data {
            (POINT_DATA, self.writer.num_points)
        } else {
            (CELL_DATA, self.writer.num_cells)
        };

        if !is_valid_data_name(name) {
            return Err(Error::InvalidData {
                reason: format!(
                    "data name '{name}' of {label} is not valid, must be non-empty and contain \
                     only alphanumeric characters, underscores or dashes"
                ),
            });
        }

        let seen_names = if is_point_data {
            &self.point_names
        } else {
            &self.cell_names
        };
        if seen_names.contains(name) {
            return Err(Error::InvalidData {
                reason: format!("name '{name}' of {label} is used more than once"),
            });
        }

        let exp_size = num_entities * data_attribute.size();
        if values.len() != exp_size {
            return Err(Error::InvalidData {
                reason: format!(
                    "size of {label} '{name}' must be {exp_size}, but is {}",
                    values.len()
                ),
            });
        }

        // reject values the backend's format cannot represent (e.g. binary's u64->u32 range)
        // before anything is written, so a caller mistake leaves no partial output behind
        self.writer.writer.validate_values(&values)?;

        if !self.initialized {
            self.writer.writer.write_data_initialize(&self.time)?;
            self.initialized = true;
        }

        let format = self.writer.writer.format();
        let data_item = DataItem {
            name: None,
            dimensions: Some(values.dimensions(data_attribute)),
            number_type: Some(values.number_type()),
            format: Some(format),
            precision: Some(values.precision(format)),
            endian: format.endian(),
            data: self.writer.writer.write_data(name, center, &values)?,
            reference: None,
        };

        self.attributes.push(attribute::Attribute {
            name: name.to_string(),
            attribute_type: data_attribute.into(),
            center,
            data_items: vec![data_item],
        });

        // recorded only once the attribute is actually written, so a rejected call can be
        // retried under the same name
        if is_point_data {
            self.point_names.insert(name.to_string());
        } else {
            self.cell_names.insert(name.to_string());
        }

        Ok(())
    }

    /// Complete the time step, adding its `<Grid>` to the XDMF file.
    fn finish(self) -> Result<()> {
        if self.attributes.is_empty() {
            let time = self.time.clone();
            // An attribute can fail *after* it initialized the backend, and a closure that
            // ignores that error still arrives here -- so the step is discarded rather than
            // simply dropped, otherwise the backend would stay initialized and every later
            // step would fail. The caller's error wins over a cleanup failure, as in `write_time_step`.
            let _discard_result = self.discard();
            return Err(Error::InvalidTimeStep {
                time,
                reason: format!("no data written, needs at least one {POINT_DATA} or {CELL_DATA}"),
            });
        }

        if let Err(error) = self.writer.writer.write_data_finalize() {
            // The step is not recorded, so the heavy data written for it is removed again --
            // otherwise it would stay behind with no `<Grid>` referencing it.
            let _discard_result = self.discard();
            return Err(error);
        }

        let TimeStep {
            writer,
            time,
            time_bits,
            attributes,
            ..
        } = self;

        writer.attributes.push((time.clone(), attributes));
        writer.written_times.insert(time_bits, time);

        writer.write_xdmf_file()
    }

    /// Abandon the time step, removing the heavy data already written for it.
    fn discard(self) -> Result<()> {
        // Nothing to undo if no attribute made it far enough to initialize the backend -- and
        // `write_data_discard` would reject the unbalanced call.
        if !self.initialized {
            return Ok(());
        }

        self.writer.writer.write_data_discard()
    }
}

// The labels below name the data category in error messages as a plain string instead of going
// through an `attribute::Center`: `attribute::center_to_data_tag` names HDF5 groups and on-disk
// file segments, and error prose should not change when that storage layout is renamed (or vice
// versa).

/// Label for point data in user-facing error messages, named after [`TimeStep::point_data`].
const POINT_DATA: &str = "point_data";
/// Label for cell data in user-facing error messages, named after [`TimeStep::cell_data`].
const CELL_DATA: &str = "cell_data";

fn is_valid_data_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    name.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Characters not allowed in the final path component of an XDMF file name.
const INVALID_FILE_NAME_CHARS: [char; 8] = ['?', '\0', ':', '*', '"', '<', '>', '|'];

/// Validate the file name for the XDMF file.
fn validate_file_name(file_name: &Path) -> Result<()> {
    // Only validate the final path component, the parent directories are not under our control
    // and may legitimately contain characters such as ':' (e.g. Windows drive letters). Since the
    // error carries the whole path, every reason below says which component it is about.
    let Some(name) = file_name.file_name() else {
        // e.g. an empty path, or one ending in ".."
        return Err(Error::InvalidFileName {
            path: file_name.to_path_buf(),
            reason: "path has no file name component".to_string(),
        });
    };

    let Some(name) = name.to_str() else {
        return Err(Error::InvalidFileName {
            path: file_name.to_path_buf(),
            reason: "file name component is not valid UTF-8".to_string(),
        });
    };

    // Check for invalid characters
    if name.chars().any(|c| INVALID_FILE_NAME_CHARS.contains(&c)) {
        return Err(Error::InvalidFileName {
            path: file_name.to_path_buf(),
            reason: format!(
                "file name component must not contain any of the following characters: \
                 {INVALID_FILE_NAME_CHARS:?}"
            ),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DataAttribute,
        xdmf_elements::{
            data_item::{DataContent, Format},
            grid::Grid,
        },
    };

    #[test]
    fn test_poly_cell_points() {
        assert_eq!(poly_cell_points(CellType::Vertex), Some(1));
        assert_eq!(poly_cell_points(CellType::Edge), Some(2));
        assert_eq!(poly_cell_points(CellType::Triangle), None);
        assert_eq!(poly_cell_points(CellType::Quadrilateral), None);
        assert_eq!(poly_cell_points(CellType::Tetrahedron), None);
        assert_eq!(poly_cell_points(CellType::Pyramid), None);
        assert_eq!(poly_cell_points(CellType::Wedge), None);
        assert_eq!(poly_cell_points(CellType::Hexahedron), None);
        assert_eq!(poly_cell_points(CellType::Edge3), None);
        assert_eq!(poly_cell_points(CellType::Quadrilateral9), None);
        assert_eq!(poly_cell_points(CellType::Triangle6), None);
        assert_eq!(poly_cell_points(CellType::Quadrilateral8), None);
        assert_eq!(poly_cell_points(CellType::Tetrahedron10), None);
        assert_eq!(poly_cell_points(CellType::Pyramid13), None);
        assert_eq!(poly_cell_points(CellType::Wedge15), None);
        assert_eq!(poly_cell_points(CellType::Wedge18), None);
        assert_eq!(poly_cell_points(CellType::Hexahedron20), None);
        assert_eq!(poly_cell_points(CellType::Hexahedron24), None);
        assert_eq!(poly_cell_points(CellType::Hexahedron27), None);
    }

    #[test]
    fn test_prepare_cells() {
        let (topo_type, cells_prep) = prepare_cells(
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            &[
                CellType::Vertex,
                CellType::Edge,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
            0,
        );

        assert_eq!(topo_type, TopologyType::Mixed);
        assert_eq!(
            cells_prep,
            vec![1, 1, 0, 2, 2, 1, 2, 4, 3, 4, 5, 5, 6, 7, 8, 9]
        );
    }

    #[test]
    fn prepare_cells_by_celltype() {
        assert_eq!(prepare_cells(&[5], &[CellType::Vertex], 0).1, vec![1, 1, 5]);

        assert_eq!(
            prepare_cells(&[5, 6], &[CellType::Edge], 0).1,
            vec![2, 2, 5, 6]
        );

        assert_eq!(
            prepare_cells(&[5, 6, 7], &[CellType::Triangle], 0).1,
            vec![4, 5, 6, 7]
        );

        assert_eq!(
            prepare_cells(&[5, 6, 7, 8], &[CellType::Quadrilateral], 0).1,
            vec![5, 5, 6, 7, 8]
        );

        assert_eq!(
            prepare_cells(&[5, 6, 7, 8], &[CellType::Tetrahedron], 0).1,
            vec![6, 5, 6, 7, 8]
        );

        assert_eq!(
            prepare_cells(&[5, 6, 7, 8, 9], &[CellType::Pyramid], 0).1,
            vec![7, 5, 6, 7, 8, 9]
        );

        assert_eq!(
            prepare_cells(&[5, 6, 7, 8, 9, 10], &[CellType::Wedge], 0).1,
            vec![8, 5, 6, 7, 8, 9, 10]
        );

        assert_eq!(
            prepare_cells(&[5, 6, 7, 8, 9, 10, 11, 12], &[CellType::Hexahedron], 0).1,
            vec![9, 5, 6, 7, 8, 9, 10, 11, 12]
        );

        assert_eq!(
            prepare_cells(&[5, 6, 7], &[CellType::Edge3], 0).1,
            vec![34, 5, 6, 7]
        );

        assert_eq!(
            prepare_cells(
                &[5, 6, 7, 8, 9, 10, 11, 12, 13],
                &[CellType::Quadrilateral9],
                0
            )
            .1,
            vec![35, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        );

        assert_eq!(
            prepare_cells(&[5, 6, 7, 8, 9, 10], &[CellType::Triangle6], 0).1,
            vec![36, 5, 6, 7, 8, 9, 10]
        );

        assert_eq!(
            prepare_cells(&[5, 6, 7, 8, 9, 10, 11, 12], &[CellType::Quadrilateral8], 0).1,
            vec![37, 5, 6, 7, 8, 9, 10, 11, 12]
        );

        assert_eq!(
            prepare_cells(
                &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                &[CellType::Tetrahedron10],
                0
            )
            .1,
            vec![38, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        );

        assert_eq!(
            prepare_cells(
                &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                &[CellType::Pyramid13],
                0
            )
            .1,
            vec![39, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        );

        assert_eq!(
            prepare_cells(
                &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                &[CellType::Wedge15],
                0
            )
            .1,
            vec![40, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        );

        assert_eq!(
            prepare_cells(
                &[
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
                ],
                &[CellType::Wedge18],
                0
            )
            .1,
            vec![
                41, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
            ]
        );

        assert_eq!(
            prepare_cells(
                &[
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
                ],
                &[CellType::Hexahedron20],
                0
            )
            .1,
            vec![
                48, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
            ]
        );

        assert_eq!(
            prepare_cells(
                &[
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                    26, 27, 28
                ],
                &[CellType::Hexahedron24],
                0
            )
            .1,
            vec![
                49, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                26, 27, 28
            ]
        );

        assert_eq!(
            prepare_cells(
                &[
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31
                ],
                &[CellType::Hexahedron27],
                0
            )
            .1,
            vec![
                50, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                26, 27, 28, 29, 30, 31
            ]
        );
    }

    #[test]
    fn test_prepare_cells_no_cells() {
        let (topo_type, cells_prep) = prepare_cells(&[], &[], 5);

        assert_eq!(topo_type, TopologyType::Polyvertex);
        assert_eq!(cells_prep, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_validate_points_and_cells() {
        // valid input, must not return an error
        validate_points_and_cells(
            33,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[
                CellType::Vertex,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        )
        .unwrap();
    }

    #[test]
    fn validate_points_and_cells_only_points() {
        // valid input, must not return an error
        validate_points_and_cells(33, &[], &[]).unwrap();
    }

    #[test]
    fn validate_points_and_cells_points_empty() {
        let res = validate_points_and_cells(
            0,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[
                CellType::Vertex,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        );

        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("at least one point")
        );
    }

    #[test]
    fn validate_points_and_cells_points_not_3d() {
        let res = validate_points_and_cells(
            22,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[
                CellType::Vertex,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        );

        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidMesh { reason } if reason.contains("22 is not a multiple of 3")
        );
    }

    #[test]
    fn validate_points_and_cells_conn_index_out_of_bounds() {
        let res = validate_points_and_cells(
            33,
            &[0, 1, 2, 3, 4, 5, 6, 70],
            &[
                CellType::Vertex,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        );

        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidMesh { reason }
                if reason.contains("connectivity index 70")
                    && reason.contains("only has 11 points")
        );
    }

    #[test]
    fn validate_points_and_cells_conn_mismatch() {
        let res = validate_points_and_cells(
            33,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[
                CellType::Vertex,
                CellType::Edge,
                CellType::Triangle,
                CellType::Quadrilateral,
            ],
        );

        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidMesh { reason }
                if reason.contains("connectivity (8)") && reason.contains("cell types (10)")
        );
    }

    #[test]
    fn time_series_writer_create_folder() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let subfolder = Path::new("out/xdmf"); // deliberately not creating this folder
        let xdmf_folder = tmp_dir.path().join(subfolder);
        let xdmf_file_path = xdmf_folder.join("test_output");

        assert!(!xdmf_folder.exists());

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();

        assert!(xdmf_folder.exists());
        assert_eq!(
            writer.xdmf_file_name,
            xdmf_file_path.with_extension("xdmf2")
        );
    }

    #[test]
    fn mpi_safe_create_dir_all_works() {
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
    fn test_validate_data() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();

        const NUM_POINTS: usize = 10;

        // write mesh
        let mut writer = writer
            .write_mesh(
                &[0.0; NUM_POINTS * 3],
                &[0, 2, 3, 4],
                &[CellType::Vertex; 4],
            )
            .unwrap();

        let values = vec![5.0; NUM_POINTS];

        // Valid time step
        writer
            .write_time_step("0.1", |step| {
                step.point_data("point_data1", DataAttribute::Scalar, &values)
            })
            .unwrap();

        // no data at all provided
        let res = writer.write_time_step("1.0", |_step| Ok(()));
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidTimeStep { time, reason }
                if time == "1.0" && reason.contains("no data written")
        );

        // Invalid time step (already exists)
        let res = writer.write_time_step("0.1", |_step| Ok(()));
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidTimeStep { time, reason }
                if time == "0.1" && reason == "already written"
        );

        // Invalid time step (not a float)
        let res = writer.write_time_step("invalid_time", |_step| Ok(()));
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidTimeStep { time, reason }
                if time == "invalid_time" && reason.contains("must be a valid float")
        );

        // Invalid time step (empty)
        let res = writer.write_time_step("", |_step| Ok(()));
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidTimeStep { time, reason }
                if time.is_empty() && reason.contains("must be a valid float")
        );
    }

    #[test]
    fn write_time_step_rejects_non_finite_times() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let mut writer = flaky_writer(tmp_dir.path().join("non_finite_times.xdmf2"), None);

        // all of these parse as a float, the last one by overflowing to infinity
        for time in ["NaN", "inf", "-infinity", "1e400"] {
            let res = writer.write_time_step(time, |step| {
                step.point_data("data", DataAttribute::Scalar, vec![0.0; 0])
            });
            std::assert_matches!(
                res.unwrap_err(),
                Error::InvalidTimeStep { time: rejected, reason }
                    if rejected == time && reason == "must be a finite float"
            );
        }

        assert!(writer.attributes.is_empty());
    }

    #[test]
    fn write_time_step_treats_negative_zero_as_the_time_already_written() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let mut writer = flaky_writer(tmp_dir.path().join("negative_zero.xdmf2"), None);

        writer
            .write_time_step("0.0", |step| {
                step.point_data("data", DataAttribute::Scalar, vec![0.0; 0])
            })
            .unwrap();

        // -0.0 is the same instant as 0.0, despite the two having different bit patterns
        let res = writer.write_time_step("-0.0", |_step| Ok(()));
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidTimeStep { time, reason }
                if time == "-0.0" && reason == "already written (as '0.0')"
        );
    }

    #[test]
    fn write_time_step_erroring_out_discards_the_data_it_already_wrote() {
        // A caller's own error type, as `write_time_step`'s closure may use: this crate's errors
        // convert into it, and it comes back out of `write_time_step` unchanged.
        #[derive(Debug)]
        enum CallerError {
            ChangedItsMind,
            Xdmf(Error),
        }

        impl From<Error> for CallerError {
            fn from(error: Error) -> Self {
                Self::Xdmf(error)
            }
        }

        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        // Ascii, so the heavy data of the abandoned step is a file that can be checked for
        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::Ascii).unwrap();

        const NUM_POINTS: usize = 10;

        let mut writer = writer
            .write_mesh(
                &[0.0; NUM_POINTS * 3],
                &[0, 2, 3, 4],
                &[CellType::Vertex; 4],
            )
            .unwrap();

        let values = vec![5.0; NUM_POINTS];

        // one attribute written, then the closure gives up -- with an error of its own, which
        // comes back unchanged rather than squeezed into this crate's `Error`
        let res = writer.write_time_step("0.1", |step| {
            step.point_data("abandoned", DataAttribute::Scalar, &values)?;
            Err(CallerError::ChangedItsMind)
        });
        std::assert_matches!(res.unwrap_err(), CallerError::ChangedItsMind);

        // the heavy data of the abandoned attribute was removed again, rather than being left
        // behind with nothing in the XDMF file referencing it
        let txt_dir = xdmf_file_path.with_extension("txt");
        assert!(!txt_dir.join("data_t_0.1_point_data_abandoned.txt").exists());

        // the backing writer is not poisoned: the same time can be used again (the abandoned
        // step never consumed it), and so can a different one
        writer
            .write_time_step("0.1", |step| {
                step.point_data("kept", DataAttribute::Scalar, &values)
            })
            .unwrap();
        writer
            .write_time_step("0.2", |step| {
                step.point_data("kept", DataAttribute::Scalar, &values)
            })
            .unwrap();

        // a rejected attribute reaches the closure's own error type through `From`, so `?` is
        // all it takes to mix the caller's errors with this crate's
        let res = writer.write_time_step("0.3", |step| {
            step.point_data("wrong_size", DataAttribute::Scalar, &[1.0])?;
            Err(CallerError::ChangedItsMind)
        });
        std::assert_matches!(
            res.unwrap_err(),
            CallerError::Xdmf(Error::InvalidData { reason })
                if reason == "size of point_data 'wrong_size' must be 10, but is 1"
        );

        // the abandoned step left no trace in the light data either
        let xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();
        assert!(!xdmf.contains("abandoned"));
        assert_eq!(xdmf.matches("<Grid Name=\"time_series-t").count(), 2);
    }

    #[test]
    fn write_time_step_erroring_out_before_writing_anything_needs_no_cleanup() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::Ascii).unwrap();

        let mut writer = writer
            .write_mesh(&[0.0; 3], &[0], &[CellType::Vertex])
            .unwrap();

        // the closure fails before any attribute is written, so `write_data_initialize` never
        // ran -- discarding must not report the unbalanced call as an internal error
        let res = writer.write_time_step("0.1", |_step| {
            Err(Error::InvalidData {
                reason: "nothing to write".to_string(),
            })
        });
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidData { reason } if reason == "nothing to write"
        );

        writer
            .write_time_step("0.1", |step| {
                step.point_data("data", DataAttribute::Scalar, &[1.0])
            })
            .unwrap();
    }

    #[test]
    fn write_time_step_mixes_value_types_within_one_step() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();

        const NUM_POINTS: usize = 4;

        let mut writer = writer
            .write_mesh(
                &[0.0; NUM_POINTS * 3],
                &[0, 1, 2, 3],
                &[CellType::Vertex; 4],
            )
            .unwrap();

        // f64 and u64 attributes in the same step -- the case a single generic type parameter
        // on the old list-shaped `write_data` could not express
        let floats = vec![1.5; NUM_POINTS];
        let ids: Vec<u64> = (0..NUM_POINTS as u64).collect();

        writer
            .write_time_step("0.0", |step| {
                step.point_data("floats", DataAttribute::Scalar, &floats)?;
                step.cell_data("ids", DataAttribute::Scalar, &ids)
            })
            .unwrap();

        let xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();
        assert!(xdmf.contains(r#"Name="floats" AttributeType="Scalar" Center="Node""#));
        assert!(xdmf.contains(r#"Name="ids" AttributeType="Scalar" Center="Cell""#));
        assert!(xdmf.contains(r#"NumberType="UInt""#));
    }

    #[test]
    fn write_time_step_reuses_a_single_buffer_across_attributes() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();

        const NUM_POINTS: usize = 3;

        let mut writer = writer
            .write_mesh(&[0.0; NUM_POINTS * 3], &[0, 1, 2], &[CellType::Vertex; 3])
            .unwrap();

        // the point of the whole builder: one allocation, refilled between attributes
        let mut buf = vec![0.0; NUM_POINTS];

        writer
            .write_time_step("0.0", |step| {
                buf.fill(1.0);
                step.point_data("first", DataAttribute::Scalar, &buf)?;

                buf.fill(2.0);
                step.point_data("second", DataAttribute::Scalar, &buf)
            })
            .unwrap();

        let xdmf = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();
        let one = "1.0000000000000000e0";
        let two = "2.0000000000000000e0";
        assert!(xdmf.contains(&format!(">{one} {one} {one}<")));
        assert!(xdmf.contains(&format!(">{two} {two} {two}<")));
    }

    #[test]
    fn test_validate_data_dedup_is_numeric_not_textual() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();

        const NUM_POINTS: usize = 10;

        let mut writer = writer
            .write_mesh(
                &[0.0; NUM_POINTS * 3],
                &[0, 2, 3, 4],
                &[CellType::Vertex; 4],
            )
            .unwrap();

        let values = vec![5.0; NUM_POINTS];
        let write_step = |writer: &mut TimeSeriesDataWriter, time: &str| -> Result<()> {
            writer.write_time_step(time, |step| {
                step.point_data("point_data1", DataAttribute::Scalar, &values)
            })
        };

        write_step(&mut writer, "0.1").unwrap();

        // a different spelling of the same numeric value is still a duplicate
        let res = write_step(&mut writer, "0.10");
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidTimeStep { time, reason }
                if time == "0.10" && reason == "already written (as '0.1')"
        );

        // a genuinely different value is accepted
        write_step(&mut writer, "0.2").unwrap();
    }

    #[test]
    fn test_validate_data_duplicate_names() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();

        const NUM_POINTS: usize = 10;

        let mut writer = writer
            .write_mesh(
                &[0.0; NUM_POINTS * 3],
                &[0, 2, 3, 4],
                &[CellType::Vertex; 4],
            )
            .unwrap();

        let values = vec![5.0; NUM_POINTS];

        let res = writer.write_time_step("0.0", |step| {
            step.point_data("duplicate", DataAttribute::Scalar, &values)?;
            step.point_data("duplicate", DataAttribute::Scalar, &values)
        });
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidData { reason }
                if reason.contains("name 'duplicate' of point_data is used more than once")
        );

        // the same name for point_data and cell_data is allowed, they are separate entities
        let cell_values = vec![5.0; 4];
        writer
            .write_time_step("0.0", |step| {
                step.point_data("data", DataAttribute::Scalar, &values)?;
                step.cell_data("data", DataAttribute::Scalar, &cell_values)
            })
            .unwrap();
    }

    #[test]
    fn test_validate_data_wrong_point_data_sizes() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();

        const NUM_POINTS: usize = 10;

        // write mesh
        let mut writer = writer
            .write_mesh(
                &[0.0; NUM_POINTS * 3],
                &[0, 2, 3, 4],
                &[CellType::Vertex; 4],
            )
            .unwrap();

        let mut err_for = |name: &str, attribute: DataAttribute, len: usize| -> Error {
            writer
                .write_time_step("0.0", |step| {
                    step.point_data(name, attribute, vec![5.0; len])
                })
                .unwrap_err()
        };

        std::assert_matches!(
            err_for("point_data_sca", DataAttribute::Scalar, NUM_POINTS - 1),
            Error::InvalidData { reason }
                if reason == "size of point_data 'point_data_sca' must be 10, but is 9"
        );
        std::assert_matches!(
            err_for("point_data_vec", DataAttribute::Vector, NUM_POINTS * 2),
            Error::InvalidData { reason }
                if reason == "size of point_data 'point_data_vec' must be 30, but is 20"
        );
        std::assert_matches!(
            err_for("point_data_ten", DataAttribute::Tensor, NUM_POINTS * 3),
            Error::InvalidData { reason }
                if reason == "size of point_data 'point_data_ten' must be 90, but is 30"
        );
        std::assert_matches!(
            err_for("point_data_ten6", DataAttribute::Tensor6, NUM_POINTS * 3),
            Error::InvalidData { reason }
                if reason == "size of point_data 'point_data_ten6' must be 60, but is 30"
        );
        std::assert_matches!(
            err_for(
                "point_data_mat",
                DataAttribute::Matrix(2, 1),
                NUM_POINTS * 3 - 1
            ),
            Error::InvalidData { reason }
                if reason == "size of point_data 'point_data_mat' must be 20, but is 29"
        );
    }

    #[test]
    fn test_validate_data_wrong_cell_data_sizes() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();

        const NUM_CELLS: usize = 4;

        // write mesh
        let mut writer = writer
            .write_mesh(
                &[0.0; 10 * 3],
                &[0, 2, 3, 4],
                &[CellType::Vertex; NUM_CELLS],
            )
            .unwrap();

        let mut err_for = |name: &str, attribute: DataAttribute, len: usize| -> Error {
            writer
                .write_time_step("0.0", |step| {
                    step.cell_data(name, attribute, vec![5.0; len])
                })
                .unwrap_err()
        };

        std::assert_matches!(
            err_for("cell_data_sca", DataAttribute::Scalar, NUM_CELLS - 1),
            Error::InvalidData { reason }
                if reason == "size of cell_data 'cell_data_sca' must be 4, but is 3"
        );
        std::assert_matches!(
            err_for("cell_data_vec", DataAttribute::Vector, NUM_CELLS * 2),
            Error::InvalidData { reason }
                if reason == "size of cell_data 'cell_data_vec' must be 12, but is 8"
        );
        std::assert_matches!(
            err_for("cell_data_ten", DataAttribute::Tensor, NUM_CELLS * 3),
            Error::InvalidData { reason }
                if reason == "size of cell_data 'cell_data_ten' must be 36, but is 12"
        );
        std::assert_matches!(
            err_for("cell_data_ten6", DataAttribute::Tensor6, NUM_CELLS * 3),
            Error::InvalidData { reason }
                if reason == "size of cell_data 'cell_data_ten6' must be 24, but is 12"
        );
        std::assert_matches!(
            err_for(
                "cell_data_mat",
                DataAttribute::Matrix(2, 1),
                NUM_CELLS * 3 - 1
            ),
            Error::InvalidData { reason }
                if reason == "size of cell_data 'cell_data_mat' must be 8, but is 11"
        );
    }

    #[test]
    fn test_validate_data_names() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_output.xdmf");

        let writer = TimeSeriesWriter::new(&xdmf_file_path, DataStorage::AsciiInline).unwrap();
        let mut writer = writer
            .write_mesh(&[0.0; 3], &[0], &[CellType::Vertex])
            .unwrap();

        let res = writer.write_time_step("0.0", |step| {
            step.cell_data("cell_data_ten", DataAttribute::Scalar, vec![0.0; 1])?;
            step.point_data("cell[_data]_ten", DataAttribute::Scalar, vec![0.0; 1])
        });
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidData { reason }
                if reason.contains("data name 'cell[_data]_ten' of point_data is not valid")
        );
    }

    #[test]
    fn test_is_valid_data_name() {
        assert!(is_valid_data_name("valid_name"));
        assert!(is_valid_data_name("valid-name"));
        assert!(is_valid_data_name("valid_name_123"));
        assert!(!is_valid_data_name("")); // empty name
        assert!(!is_valid_data_name("invalid name")); // space
        assert!(!is_valid_data_name("invalid@name")); // special character
        assert!(!is_valid_data_name("invalid#name")); // special character
        assert!(!is_valid_data_name("invalid$name")); // special character
        assert!(!is_valid_data_name("invalid%name")); // special character
        assert!(!is_valid_data_name("invalid^name")); // special character
        assert!(!is_valid_data_name("invalid&name")); // special character
        assert!(!is_valid_data_name("invalid*name")); // special character
        assert!(!is_valid_data_name("invalid(name")); // special character
        assert!(!is_valid_data_name("invalid)name")); // special character
        assert!(!is_valid_data_name("invalid+name")); // special character
        assert!(!is_valid_data_name("invalid=name")); // special character
        assert!(!is_valid_data_name("invalid{name")); // special character
        assert!(!is_valid_data_name("invalid}name")); // special character
        assert!(!is_valid_data_name("invalid[name")); // special character
        assert!(!is_valid_data_name("invalid]name")); // special character
        assert!(!is_valid_data_name("invalid|name")); // special character
        assert!(!is_valid_data_name("invalid:name")); // special character
        assert!(!is_valid_data_name("invalid;name")); // special character
        assert!(!is_valid_data_name("invalid'")); // single quote
        assert!(!is_valid_data_name("invalid\"name")); // double quote
        assert!(!is_valid_data_name("invalid,name")); // comma
        assert!(!is_valid_data_name("invalid.name")); // dot
        assert!(!is_valid_data_name("invalid?name")); // question mark
        assert!(!is_valid_data_name("invalid/name")); // forward slash
        assert!(!is_valid_data_name("invalid\\name")); // backslash
        assert!(!is_valid_data_name("invalid\0name")); // null-char
    }

    #[test]
    fn test_validate_file_name() {
        validate_file_name(Path::new("asdf.txt")).unwrap();
        validate_file_name(Path::new("valid-name.txt")).unwrap();
        validate_file_name(Path::new("valid_name.txt")).unwrap();
        validate_file_name(Path::new("valid_name-123.txt")).unwrap();

        // only the final component is validated, a parent may legitimately contain ':'
        validate_file_name(Path::new("C:/some:dir/valid_name.txt")).unwrap();

        let res = validate_file_name(Path::new("valid_name:123.txt"));
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidFileName { path, reason }
                if path == Path::new("valid_name:123.txt")
                    && reason.contains("file name component must not contain any of")
        );

        let res = validate_file_name(Path::new(""));
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidFileName { path, reason }
                if path == Path::new("") && reason == "path has no file name component"
        );
    }

    fn dummy_geometry() -> Geometry {
        Geometry {
            geometry_type: GeometryType::XYZ,
            data_item: DataItem {
                dimensions: Some(Dimensions(vec![5, 3])),
                data: "0 1 0 0 1.5 0 0.5 1.5 0.5 1 1.5 0 1 1 0".into(),
                number_type: Some(NumberType::Float),
                ..Default::default()
            },
        }
    }

    fn dummy_topology() -> Topology {
        Topology {
            topology_type: TopologyType::Triangle,
            number_of_elements: "2".into(),
            data_item: DataItem {
                dimensions: Some(Dimensions(vec![6])),
                number_type: Some(NumberType::Int),
                data: "0 1 2 2 3 4".into(),
                ..Default::default()
            },
        }
    }

    #[test]
    fn test_write_data_preserve_order() {
        struct DummyWriter;

        impl DataWriter for DummyWriter {
            fn format(&self) -> Format {
                Format::XML
            }

            fn data_storage(&self) -> DataStorage {
                DataStorage::AsciiInline
            }

            fn write_mesh(
                &mut self,
                _points: &Values<'_>,
                _cells: &[u64],
            ) -> Result<(DataContent, DataContent)> {
                Ok((
                    DataContent::Raw("points".to_string()),
                    DataContent::Raw("cells".to_string()),
                ))
            }

            fn write_data(
                &mut self,
                name: &str,
                _center: attribute::Center,
                _data: &Values<'_>,
            ) -> Result<DataContent> {
                Ok(DataContent::Raw(format!("data_for_{name}")))
            }
        }

        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let xdmf_file_path = tmp_dir.path().join("test_write_data_preserve_order.xdmf2");

        let mut writer = TimeSeriesDataWriter {
            xdmf_file_name: xdmf_file_path.clone(),
            writer: Box::new(DummyWriter),
            grid: Grid::new_uniform("test", dummy_geometry(), dummy_topology()),
            data_items: Vec::new(),
            num_points: 0,
            num_cells: 0,
            attributes: Vec::new(),
            written_times: HashMap::new(),
        };

        let write_step = |writer: &mut TimeSeriesDataWriter, time: &str| {
            writer
                .write_time_step(time, |step| {
                    step.point_data("scalar_data", DataAttribute::Scalar, vec![0.0; 0])
                })
                .unwrap();
        };

        write_step(&mut writer, "0.0");
        write_step(&mut writer, "1.0");
        write_step(&mut writer, "2.0");
        write_step(&mut writer, "10.0");

        // Check that the data are in the correct order

        let expected_xdmf = r#"
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name="time_series" GridType="Collection" CollectionType="Temporal">
            <Grid Name="time_series-t0.0" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Dimensions="5 3" NumberType="Float" Format="XML" Precision="4">0 1 0 0 1.5 0 0.5 1.5 0.5 1 1.5 0 1 1 0</DataItem>
                </Geometry>
                <Topology TopologyType="Triangle" NumberOfElements="2">
                    <DataItem Dimensions="6" NumberType="Int" Format="XML" Precision="4">0 1 2 2 3 4</DataItem>
                </Topology>
                <Time Value="0.0"/>
                <Attribute Name="scalar_data" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="0" NumberType="Float" Format="XML" Precision="8">data_for_scalar_data</DataItem>
                </Attribute>
            </Grid>
            <Grid Name="time_series-t1.0" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Dimensions="5 3" NumberType="Float" Format="XML" Precision="4">0 1 0 0 1.5 0 0.5 1.5 0.5 1 1.5 0 1 1 0</DataItem>
                </Geometry>
                <Topology TopologyType="Triangle" NumberOfElements="2">
                    <DataItem Dimensions="6" NumberType="Int" Format="XML" Precision="4">0 1 2 2 3 4</DataItem>
                </Topology>
                <Time Value="1.0"/>
                <Attribute Name="scalar_data" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="0" NumberType="Float" Format="XML" Precision="8">data_for_scalar_data</DataItem>
                </Attribute>
            </Grid>
            <Grid Name="time_series-t2.0" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Dimensions="5 3" NumberType="Float" Format="XML" Precision="4">0 1 0 0 1.5 0 0.5 1.5 0.5 1 1.5 0 1 1 0</DataItem>
                </Geometry>
                <Topology TopologyType="Triangle" NumberOfElements="2">
                    <DataItem Dimensions="6" NumberType="Int" Format="XML" Precision="4">0 1 2 2 3 4</DataItem>
                </Topology>
                <Time Value="2.0"/>
                <Attribute Name="scalar_data" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="0" NumberType="Float" Format="XML" Precision="8">data_for_scalar_data</DataItem>
                </Attribute>
            </Grid>
            <Grid Name="time_series-t10.0" GridType="Uniform">
                <Geometry GeometryType="XYZ">
                    <DataItem Dimensions="5 3" NumberType="Float" Format="XML" Precision="4">0 1 0 0 1.5 0 0.5 1.5 0.5 1 1.5 0 1 1 0</DataItem>
                </Geometry>
                <Topology TopologyType="Triangle" NumberOfElements="2">
                    <DataItem Dimensions="6" NumberType="Int" Format="XML" Precision="4">0 1 2 2 3 4</DataItem>
                </Topology>
                <Time Value="10.0"/>
                <Attribute Name="scalar_data" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="0" NumberType="Float" Format="XML" Precision="8">data_for_scalar_data</DataItem>
                </Attribute>
            </Grid>
        </Grid>
    </Domain>
    <Information Name="data_storage" Value="AsciiInline"/>
    <Information Name="version" Value="0.1.3"/>
</Xdmf>"#;

        let xdmf_file = xdmf_file_path.with_extension("xdmf2");
        let read_xdmf = std::fs::read_to_string(&xdmf_file).unwrap();

        // for debugging purposes, you can uncomment the line below to write the XDMF file to disk
        // std::fs::copy(xdmf_file, "time_series_writer_only_mesh.xdmf").unwrap();

        pretty_assertions::assert_eq!(expected_xdmf, read_xdmf);
    }

    // A backend that fails on demand, to exercise the failure paths of a time step without
    // depending on a real storage format: writing the attribute named "boom" fails, and so does
    // finalizing the time given as `fail_finalize_at`.
    struct FlakyWriter {
        write_time: Option<String>,
        fail_finalize_at: Option<&'static str>,
    }

    impl DataWriter for FlakyWriter {
        fn format(&self) -> Format {
            Format::XML
        }

        fn data_storage(&self) -> DataStorage {
            DataStorage::AsciiInline
        }

        fn write_mesh(
            &mut self,
            _points: &Values<'_>,
            _cells: &[u64],
        ) -> Result<(DataContent, DataContent)> {
            Ok((
                DataContent::Raw("points".to_string()),
                DataContent::Raw("cells".to_string()),
            ))
        }

        fn write_data(
            &mut self,
            name: &str,
            _center: attribute::Center,
            _data: &Values<'_>,
        ) -> Result<DataContent> {
            if name == "boom" {
                return Err(Error::Io {
                    operation: "writing data (simulated)",
                    path: PathBuf::from("boom"),
                    source: std::io::Error::other("simulated mid-write failure"),
                });
            }
            Ok(DataContent::Raw(format!("data_for_{name}")))
        }

        fn write_data_initialize(&mut self, time: &str) -> Result<()> {
            if self.write_time.is_some() {
                return Err(Error::Internal("writing data was already initialized"));
            }
            self.write_time = Some(time.to_string());
            Ok(())
        }

        fn write_data_finalize(&mut self) -> Result<()> {
            let Some(time) = self.write_time.as_deref() else {
                return Err(Error::Internal("writing data was not initialized"));
            };

            // the step stays open on this failure, like a backend that could not complete it
            if self.fail_finalize_at == Some(time) {
                return Err(Error::Io {
                    operation: "finalizing data (simulated)",
                    path: PathBuf::from("finalize"),
                    source: std::io::Error::other("simulated finalize failure"),
                });
            }

            self.write_time = None;
            Ok(())
        }

        fn write_data_discard(&mut self) -> Result<()> {
            // mirrors the real backends: the step is dropped and the writer is ready for the
            // next one (this backend has no heavy data to remove)
            if self.write_time.is_none() {
                return Err(Error::Internal("writing data was not initialized"));
            }
            self.write_time = None;
            Ok(())
        }
    }

    // A `TimeSeriesDataWriter` on top of `FlakyWriter`, assembled directly rather than through
    // `TimeSeriesWriter::write_mesh`, since only the time-step handling is under test.
    fn flaky_writer(
        xdmf_file_name: PathBuf,
        fail_finalize_at: Option<&'static str>,
    ) -> TimeSeriesDataWriter {
        TimeSeriesDataWriter {
            xdmf_file_name,
            writer: Box::new(FlakyWriter {
                write_time: None,
                fail_finalize_at,
            }),
            grid: Grid::new_uniform("test", dummy_geometry(), dummy_topology()),
            data_items: Vec::new(),
            num_points: 0,
            num_cells: 0,
            attributes: Vec::new(),
            written_times: HashMap::new(),
        }
    }

    #[test]
    fn write_data_survives_a_mid_write_failure() {
        // Fails while writing one attribute, after already having written an earlier one --
        // reproducing the shape of the original binary-backend bug (now caught upfront for that
        // specific case by `DataWriter::validate_values`, but the discard-on-error handling in
        // `TimeSeriesDataWriter::write_time_step` is generic and backend-agnostic, so it needs its own
        // backend-agnostic regression test).
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let mut writer = flaky_writer(tmp_dir.path().join("mid_write_failure.xdmf2"), None);

        // "ok" is written successfully before "boom" fails, so this genuinely fails partway
        // through the step, after `write_data_initialize` already ran.
        let res = writer.write_time_step("0.0", |step| {
            step.point_data("ok", DataAttribute::Scalar, vec![0.0; 0])?;
            step.point_data("boom", DataAttribute::Scalar, vec![0.0; 0])
        });
        std::assert_matches!(res.unwrap_err(), Error::Io { .. });

        // The failed step must not have consumed the time slot ("0.0" is retried, not a new
        // time) and must not have left the backing writer poisoned: `FlakyWriter` itself would
        // fail with `Error::Internal("writing data was already initialized")` here if
        // `write_time_step` had skipped the discard.
        writer
            .write_time_step("0.0", |step| {
                step.point_data("ok", DataAttribute::Scalar, vec![0.0; 0])
            })
            .unwrap();
    }

    #[test]
    fn write_time_step_discards_when_the_closure_swallows_an_attribute_error() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let mut writer = flaky_writer(tmp_dir.path().join("swallowed_error.xdmf2"), None);

        // The closure ignores the failure of "boom" and returns `Ok`, so the step ends up with
        // no attributes even though the failed write already initialized the backend -- the
        // empty step must therefore be discarded, not just dropped.
        let res = writer.write_time_step("0.0", |step| {
            let _write_result = step.point_data("boom", DataAttribute::Scalar, vec![0.0; 0]);
            Ok(())
        });
        std::assert_matches!(
            res.unwrap_err(),
            Error::InvalidTimeStep { time, reason }
                if time == "0.0" && reason.contains("no data written")
        );

        assert!(writer.attributes.is_empty());
        assert!(writer.written_times.is_empty());

        // the backing writer is not poisoned: `FlakyWriter` would fail with
        // `Error::Internal("writing data was already initialized")` here otherwise
        writer
            .write_time_step("0.0", |step| {
                step.point_data("ok", DataAttribute::Scalar, vec![0.0; 0])
            })
            .unwrap();
    }

    #[test]
    fn write_time_step_keeps_a_step_whose_closure_swallowed_an_attribute_error() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let mut writer = flaky_writer(tmp_dir.path().join("swallowed_error.xdmf2"), None);

        // As above, but one attribute did make it: a step holds exactly what was written
        // successfully, so swallowing the error of "boom" writes the step without it rather
        // than failing. Only a step left with nothing at all is rejected.
        writer
            .write_time_step("0.0", |step| {
                step.point_data("ok", DataAttribute::Scalar, vec![0.0; 0])?;
                let _write_result = step.point_data("boom", DataAttribute::Scalar, vec![0.0; 0]);
                // annotated because nothing else in this closure pins the error type
                Ok::<(), Error>(())
            })
            .unwrap();

        let (time, attributes) = writer.attributes.first().unwrap();
        assert_eq!(time, "0.0");
        assert_eq!(attributes.len(), 1);
        assert_eq!(attributes[0].name, "ok");
    }

    #[test]
    fn write_time_step_discards_when_finalizing_fails() {
        let tmp_dir = temp_dir::TempDir::new().unwrap();
        let mut writer = flaky_writer(tmp_dir.path().join("finalize_failure.xdmf2"), Some("0.0"));

        // every attribute is written, but completing the step fails
        let res = writer.write_time_step("0.0", |step| {
            step.point_data("ok", DataAttribute::Scalar, vec![0.0; 0])
        });
        std::assert_matches!(
            res.unwrap_err(),
            Error::Io {
                operation: "finalizing data (simulated)",
                ..
            }
        );

        // the step is not recorded, so its heavy data must not be kept either -- and the time
        // stays available
        assert!(writer.attributes.is_empty());
        assert!(writer.written_times.is_empty());

        // the step was discarded rather than left open, so a following step still works
        writer
            .write_time_step("1.0", |step| {
                step.point_data("ok", DataAttribute::Scalar, vec![0.0; 0])
            })
            .unwrap();
    }
}
