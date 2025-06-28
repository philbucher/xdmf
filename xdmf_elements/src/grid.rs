use serde::Serialize;

use crate::attribute::Attribute;
use crate::geometry::Geometry;
use crate::topology::Topology;

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Grid {
    Uniform(Uniform),
    Tree(Tree),
    Collection(Collection),
    Reference(Reference),
    TimeSeriesGrid(TimeSeriesGrid),
}

#[derive(Debug, Serialize)]
pub struct Uniform {
    #[serde(rename = "@Name")]
    pub name: String,

    #[serde(rename = "@GridType")]
    pub grid_type: GridType,

    #[serde(rename = "Geometry")]
    pub geometry: Geometry,

    #[serde(rename = "Topology")]
    pub topology: Topology,
}

#[derive(Debug, Serialize)]
pub struct Tree {
    #[serde(rename = "@Name")]
    pub name: String,

    #[serde(rename = "@GridType")]
    pub grid_type: GridType,

    #[serde(rename = "Grid")]
    pub grids: Vec<Grid>,
}

#[derive(Debug, Serialize)]
pub struct Collection {
    #[serde(rename = "@Name")]
    pub name: String,

    #[serde(rename = "@GridType")]
    pub grid_type: GridType,

    #[serde(rename = "@CollectionType")]
    pub collection_type: CollectionType,

    #[serde(rename = "Grid")]
    pub grids: Vec<Grid>,
}

#[derive(Debug, Serialize)]
pub struct Reference {
    #[serde(skip_serializing)]
    pub mesh_grid_name: String,

    #[serde(rename = "Time")]
    pub time: Time,

    #[serde(rename = "Attribute")]
    pub attributes: Vec<Attribute>,
}

#[derive(Debug, Serialize)]
pub struct Time {
    #[serde(rename = "@Value")]
    pub value: String,
}

impl Time {
    pub fn new(value: impl ToString) -> Self {
        Time {
            value: value.to_string(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct TimeSeriesGrid {
    #[serde(rename = "@Name")]
    pub name: String,

    #[serde(skip_serializing)]
    pub mesh_grid_name: String,

    #[serde(rename = "@GridType")]
    pub grid_type: GridType,

    #[serde(rename = "@CollectionType")]
    pub collection_type: CollectionType,

    #[serde(rename = "Grid")]
    pub grids: Vec<Grid>,
}

impl Grid {
    pub fn new_uniform(name: impl ToString, geometry: Geometry, topology: Topology) -> Self {
        Grid::Uniform(Uniform {
            name: name.to_string(),
            grid_type: GridType::Uniform,
            topology,
            geometry,
        })
    }

    pub fn new_collection(
        name: impl ToString,
        collection_type: CollectionType,
        grids: Option<Vec<Grid>>,
    ) -> Self {
        Grid::Collection(Collection {
            name: name.to_string(),
            grid_type: GridType::Collection,
            collection_type,
            grids: grids.unwrap_or_default(),
        })
    }

    pub fn new_tree(name: impl ToString, grids: Option<Vec<Grid>>) -> Self {
        Grid::Tree(Tree {
            name: name.to_string(),
            grid_type: GridType::Tree,
            grids: grids.unwrap_or_default(),
        })
    }

    pub fn new_time_series(name: impl ToString, mesh_grid: Uniform) -> Self {
        Grid::TimeSeriesGrid(TimeSeriesGrid {
            name: name.to_string(),
            mesh_grid_name: mesh_grid.name.clone(),
            grid_type: GridType::Collection,
            collection_type: CollectionType::Temporal,
            grids: vec![Grid::Uniform(mesh_grid)],
        })
    }
}

impl TimeSeriesGrid {
    pub fn create_new_time(&mut self, time: impl ToString) -> &mut Reference {
        let ref_time = Reference {
            mesh_grid_name: self.mesh_grid_name.clone(),
            time: Time::new(time),
            attributes: vec![],
        };

        self.grids.push(Grid::Reference(ref_time));

        match self.grids.last_mut() {
            Some(Grid::Reference(ref_grid)) => ref_grid,
            _ => unreachable!("Last grid is not a Reference type"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub enum GridType {
    #[default]
    Uniform,
    Collection,
    Tree,
    SubSet,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub enum CollectionType {
    #[default]
    Spatial,
    Temporal,
}
