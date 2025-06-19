use serde::Serialize;

use crate::attribute::Attribute;
use crate::geometry::Geometry;
use crate::topology::Topology;

#[derive(Debug, Default, Serialize)]
pub struct Grid {
    #[serde(rename = "@Name")]
    pub name: String,

    #[serde(rename = "@GridType")]
    pub grid_type: GridType,

    #[serde(rename = "Geometry", skip_serializing_if = "Option::is_none")]
    pub geometry: Option<Geometry>,

    #[serde(rename = "Topology", skip_serializing_if = "Option::is_none")]
    pub topology: Option<Topology>,

    #[serde(rename = "Grid", skip_serializing_if = "Option::is_none")]
    pub grids: Option<Vec<Grid>>,

    #[serde(rename = "@CollectionType", skip_serializing_if = "Option::is_none")]
    pub collection_type: Option<CollectionType>,

    #[serde(rename = "Time", skip_serializing_if = "Option::is_none")]
    pub time: Option<String>,

    #[serde(rename = "Attribute", skip_serializing_if = "Option::is_none")]
    pub attributes: Option<Attribute>,
}

impl Grid {
    pub fn new_uniform(name: impl ToString, geometry: Geometry, topology: Topology) -> Self {
        Self {
            name: name.to_string(),
            grid_type: GridType::Uniform,
            topology: Some(topology),
            geometry: Some(geometry),
            ..Default::default()
        }
    }

    pub fn new_collection(
        name: impl ToString,
        collection_type: CollectionType,
        grids: Vec<Grid>,
    ) -> Self {
        Self {
            name: name.to_string(),
            grid_type: GridType::Collection,
            grids: Some(grids),
            collection_type: Some(collection_type),
            ..Default::default()
        }
    }

    pub fn new_tree(name: impl ToString, grids: Vec<Grid>) -> Self {
        Self {
            name: name.to_string(),
            grid_type: GridType::Tree,
            grids: Some(grids),
            ..Default::default()
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
