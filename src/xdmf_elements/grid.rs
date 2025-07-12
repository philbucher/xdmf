use serde::Serialize;

use super::attribute::Attribute;
use super::geometry::Geometry;
use super::topology::Topology;

#[derive(Clone, Debug, Serialize)]
pub struct Grid {
    #[serde(rename = "@Name")]
    pub name: String,

    #[serde(rename = "@GridType")]
    pub grid_type: GridType,

    #[serde(rename = "@CollectionType", skip_serializing_if = "Option::is_none")]
    pub collection_type: Option<CollectionType>,

    #[serde(rename = "Geometry", skip_serializing_if = "Option::is_none")]
    pub geometry: Option<Geometry>,

    #[serde(rename = "Topology", skip_serializing_if = "Option::is_none")]
    pub topology: Option<Topology>,

    #[serde(rename = "Grid", skip_serializing_if = "Option::is_none")]
    pub grids: Option<Vec<Grid>>,

    #[serde(rename = "Time", skip_serializing_if = "Option::is_none")]
    pub time: Option<Time>,

    #[serde(rename = "Attribute", skip_serializing_if = "Option::is_none")]
    pub attributes: Option<Vec<Attribute>>,
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
        Grid {
            name: name.to_string(),
            grid_type: GridType::Uniform,
            collection_type: None,
            geometry: Some(geometry),
            topology: Some(topology),
            grids: None,
            time: None,
            attributes: None,
        }
    }

    pub fn new_collection(
        name: impl ToString,
        collection_type: CollectionType,
        grids: Option<Vec<Grid>>,
    ) -> Self {
        Grid {
            name: name.to_string(),
            grid_type: GridType::Collection,
            collection_type: Some(collection_type),
            geometry: None,
            topology: None,
            attributes: None,
            grids,
            time: None,
        }
    }

    pub fn new_tree(name: impl ToString, grids: Option<Vec<Grid>>) -> Self {
        Grid {
            name: name.to_string(),
            grid_type: GridType::Tree,
            collection_type: None,
            grids,
            geometry: None,
            topology: None,
            attributes: None,
            time: None,
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
