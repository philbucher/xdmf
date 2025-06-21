use serde::Serialize;

use crate::geometry::Geometry;
use crate::topology::Topology;

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Grid {
    Uniform(Uniform),
    Tree(Tree),
    Collection(Collection),
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

    #[serde(rename = "Grid")]
    pub grids: Vec<Grid>,

    #[serde(rename = "@CollectionType")]
    pub collection_type: CollectionType,
}

impl Grid {
    pub fn new_uniform(name: impl ToString, geometry: Geometry, topology: Topology) -> Self {
        Grid::Uniform(Uniform {
            name: name.to_string(),
            grid_type: GridType::Uniform,
            topology: topology,
            geometry: geometry,
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
