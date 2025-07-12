use serde::Serialize;

use super::attribute::Attribute;
use super::geometry::Geometry;
use super::topology::Topology;

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)] // TODO remove this
pub enum Grid {
    Uniform(Uniform),
    Tree(Tree),
    Collection(Collection),
}

#[derive(Clone, Debug, Serialize)]
pub struct Uniform {
    #[serde(rename = "@Name")]
    pub name: String,

    #[serde(rename = "@GridType")]
    pub grid_type: GridType,

    #[serde(rename = "Geometry")]
    pub geometry: Geometry,

    #[serde(rename = "Topology")]
    pub topology: Topology,

    #[serde(rename = "Time", skip_serializing_if = "Option::is_none")]
    pub time: Option<Time>,

    #[serde(rename = "Attribute", skip_serializing_if = "Option::is_none")]
    pub attributes: Option<Vec<Attribute>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Tree {
    #[serde(rename = "@Name")]
    pub name: String,

    #[serde(rename = "@GridType")]
    pub grid_type: GridType,

    #[serde(rename = "Grid")]
    pub grids: Vec<Grid>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Collection {
    #[serde(rename = "@Name")]
    pub name: String,

    #[serde(rename = "@GridType")]
    pub grid_type: GridType,

    #[serde(rename = "@CollectionType")]
    pub collection_type: CollectionType,

    #[serde(rename = "Grid")]
    pub grids: Vec<Grid>,

    #[serde(rename = "Time", skip_serializing_if = "Option::is_none")]
    pub time: Option<Time>,
}

#[derive(Clone, Debug, Serialize)]
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

impl Grid {
    pub fn new_uniform(name: impl ToString, geometry: Geometry, topology: Topology) -> Self {
        Grid::Uniform(Uniform {
            name: name.to_string(),
            grid_type: GridType::Uniform,
            topology,
            geometry,
            time: None,
            attributes: None,
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
            time: None,
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
