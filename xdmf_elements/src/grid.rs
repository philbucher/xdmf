use serde::Serialize;

use crate::attribute::Attribute;
use crate::data_item::DataItem;
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

    #[serde(rename = "Section", skip_serializing_if = "Option::is_none")]
    pub section: Option<Section>,

    #[serde(rename = "Time", skip_serializing_if = "Option::is_none")]
    pub time: Option<String>,

    #[serde(rename = "Attribute", skip_serializing_if = "Option::is_none")]
    pub attributes: Option<Attribute>,
}

impl Grid {
    pub fn new_uniform(name: impl ToString, geometry: Geometry, topology: Topology) -> Self {
        Grid {
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
        Grid {
            name: name.to_string(),
            grid_type: GridType::Collection,
            grids: Some(grids),
            collection_type: Some(collection_type),
            ..Default::default()
        }
    }

    pub fn new_tree(name: impl ToString, grids: Vec<Grid>) -> Self {
        Grid {
            name: name.to_string(),
            grid_type: GridType::Tree,
            grids: Some(grids),
            ..Default::default()
        }
    }

    pub fn new_subset(
        name: impl ToString,
        topology: Topology,
        geometry: Geometry,
        section: Section,
    ) -> Self {
        Grid {
            name: name.to_string(),
            grid_type: GridType::SubSet,
            topology: Some(topology),
            geometry: Some(geometry),
            section: Some(section),
            ..Default::default()
        }
    }
}

#[derive(Debug, Serialize)]
pub enum GridType {
    Uniform,
    Collection,
    Tree,
    SubSet,
}

impl Default for GridType {
    fn default() -> Self {
        GridType::Uniform
    }
}

#[derive(Debug, Serialize)]
pub enum CollectionType {
    Spatial,
    Temporal,
}

impl Default for CollectionType {
    fn default() -> Self {
        CollectionType::Spatial
    }
}

#[derive(Debug, Serialize)]
pub enum Section {
    DataItem(DataItem),
    All,
}
