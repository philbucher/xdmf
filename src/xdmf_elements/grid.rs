use serde::Serialize;

use super::{attribute::Attribute, geometry::Geometry, topology::Topology};

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

#[cfg(test)]
mod tests {
    use quick_xml::se::to_string;

    use super::*;
    use crate::xdmf_elements::{
        attribute::{Attribute, AttributeType, Center},
        data_item::{DataItem, NumberType},
        dimensions::Dimensions,
        geometry::{Geometry, GeometryType},
        grid::{CollectionType, Grid, Time},
        topology::{Topology, TopologyType},
    };

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

    fn dummy_attribute() -> Attribute {
        Attribute {
            name: String::from("Temperature"),
            attribute_type: AttributeType::Scalar,
            center: Center::Cell,
            data_items: vec![DataItem {
                dimensions: Some(Dimensions(vec![2])),
                data: "2 3".into(),
                number_type: Some(NumberType::Float),
                ..Default::default()
            }],
        }
    }

    #[test]
    fn grid_new_uniform() {
        let grid = Grid::new_uniform("test", dummy_geometry(), dummy_topology());
        assert_eq!(grid.name, "test");
        assert_eq!(grid.grid_type, GridType::Uniform);
        assert!(grid.geometry.is_some());
        assert!(grid.topology.is_some());
        assert!(grid.grids.is_none());
        assert!(grid.time.is_none());
        assert!(grid.attributes.is_none());
    }

    #[test]
    fn grid_new_collection() {
        let subgrid = Grid::new_uniform("sub", dummy_geometry(), dummy_topology());
        let grid = Grid::new_collection("coll", CollectionType::Spatial, Some(vec![subgrid]));
        assert_eq!(grid.name, "coll");
        assert_eq!(grid.grid_type, GridType::Collection);
        assert_eq!(grid.collection_type, Some(CollectionType::Spatial));
        assert!(grid.grids.is_some());
        assert_eq!(grid.grids.as_ref().unwrap().len(), 1);
        assert!(grid.geometry.is_none());
        assert!(grid.topology.is_none());
        assert!(grid.time.is_none());
        assert!(grid.attributes.is_none());
    }

    #[test]
    fn grid_new_tree() {
        let subgrid = Grid::new_uniform("sub", dummy_geometry(), dummy_topology());
        let grid = Grid::new_tree("tree", Some(vec![subgrid]));
        assert_eq!(grid.name, "tree");
        assert_eq!(grid.grid_type, GridType::Tree);
        assert!(grid.grids.is_some());
        assert_eq!(grid.grids.as_ref().unwrap().len(), 1);
        assert!(grid.geometry.is_none());
        assert!(grid.topology.is_none());
        assert!(grid.time.is_none());
        assert!(grid.attributes.is_none());
    }

    #[test]
    fn time_new() {
        let time = Time::new(42);
        assert_eq!(time.value, "42");
        let time_str = Time::new("2024-06-01");
        assert_eq!(time_str.value, "2024-06-01");
    }

    #[test]
    fn time_serialization() {
        let time = Time::new("2024-06-01");
        pretty_assertions::assert_eq!(to_string(&time).unwrap(), "<Time Value=\"2024-06-01\"/>");
    }

    #[test]
    fn grid_serialization() {
        let geometry = dummy_geometry();
        let topology = dummy_topology();
        let mut grid = Grid::new_uniform("serialize", geometry, topology);
        grid.time = Some(Time::new(1.23));
        grid.attributes = Some(vec![dummy_attribute()]);

        pretty_assertions::assert_eq!(
            to_string(&grid).unwrap(),
            "<Grid Name=\"serialize\" GridType=\"Uniform\">\
                <Geometry GeometryType=\"XYZ\">\
                    <DataItem Dimensions=\"5 3\" NumberType=\"Float\" Format=\"XML\" Precision=\"4\">0 1 0 0 1.5 0 0.5 1.5 0.5 1 1.5 0 1 1 0</DataItem>\
                </Geometry>\
                <Topology TopologyType=\"Triangle\" NumberOfElements=\"2\">\
                    <DataItem Dimensions=\"6\" NumberType=\"Int\" Format=\"XML\" Precision=\"4\">0 1 2 2 3 4</DataItem>\
                </Topology>\
                <Time Value=\"1.23\"/>\
                <Attribute Name=\"Temperature\" AttributeType=\"Scalar\" Center=\"Cell\">\
                    <DataItem Dimensions=\"2\" NumberType=\"Float\" Format=\"XML\" Precision=\"4\">2 3</DataItem>\
                </Attribute>\
            </Grid>"
        );
    }

    #[test]
    fn gridtype_default() {
        assert_eq!(GridType::default(), GridType::Uniform);
    }

    #[test]
    fn collectiontype_default() {
        assert_eq!(CollectionType::default(), CollectionType::Spatial);
    }
}
