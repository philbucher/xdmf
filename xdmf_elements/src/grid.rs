use serde::Serialize;

use crate::attribute::Attribute;
use crate::data_item::DataItem;
use crate::geometry::Geometry;
use crate::topology::Topology;

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)] // TODO remove this
pub enum Grid {
    Uniform(Uniform),
    Tree(Tree),
    Collection(Collection),
    TimeSeriesGrid(TimeSeriesGrid),
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

    #[serde(skip_serializing)]
    pub indices: Option<DataItem>,
}

impl Uniform {
    pub fn set_attributes(&mut self, attributes: &[Attribute]) {
        if self.attributes.is_none() {
            self.attributes = Some(vec![]);
        }

        let mut attributes = attributes.to_vec();

        // if indices are present, i.e. this is a subgrid, then they must be added to the attributes (before the actual dataitem)
        if let Some(indcs) = &self.indices {
            for a in attributes.iter_mut() {
                a.set_indices(indcs.clone());
            }
        }
        self.attributes = Some(attributes);
    }
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

impl Collection {
    pub fn set_attributes(&mut self, attributes: &Vec<Attribute>) {
        for grid in &mut self.grids {
            match grid {
                Grid::Uniform(uniform) => {
                    uniform.set_attributes(attributes);
                }
                Grid::Collection(collection) => {
                    collection.set_attributes(attributes);
                }
                _ => {}
            }
        }
    }
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

#[derive(Clone, Debug, Serialize)]
pub struct TimeSeriesGrid {
    #[serde(rename = "@Name")]
    pub name: String,

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
            time: None,
            attributes: None,
            indices: None,
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

    pub fn new_time_series(name: impl ToString, mesh_grid: Uniform) -> Self {
        Grid::TimeSeriesGrid(TimeSeriesGrid {
            name: name.to_string(),
            grid_type: GridType::Collection,
            collection_type: CollectionType::Temporal,
            grids: vec![Grid::Uniform(mesh_grid)],
        })
    }
}

impl TimeSeriesGrid {
    pub fn create_new_time(&mut self, time: impl ToString, attributes: &Vec<Attribute>) {
        let prev_grid = self
            .grids
            .last()
            .expect("Time series grid must have at least one grid");

        // let mut new_grid = if prev_grid.time.is_some() {
        //     prev_grid.clone()
        // } else {
        //     prev_grid
        // };
        let mut new_grid = prev_grid.clone();

        match new_grid {
            Grid::Uniform(ref mut grid) => {
                if grid.time.is_none() {
                    // remove the first grid if it has no time, i.e. this is the first time data is written
                    self.grids.remove(0);
                }
                grid.name = format!("{}-t{}", self.name, time.to_string());
                grid.time = Some(Time::new(time));
                grid.set_attributes(attributes);
            }
            Grid::Collection(ref mut grid) if grid.collection_type == CollectionType::Spatial => {
                if grid.time.is_none() {
                    // remove the first grid if it has no time, i.e. this is the first time data is written
                    self.grids.remove(0);
                }
                grid.name = format!("{}-t{}", self.name, time.to_string());
                grid.time = Some(Time::new(time));
                grid.set_attributes(attributes);
            }
            _ => panic!("First grid in time series must be a Uniform or (spatial) collection grid"),
        };

        self.grids.push(new_grid);
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
