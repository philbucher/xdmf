use serde::Serialize;

pub mod attribute;
pub mod data_item;
pub mod dimensions;
pub mod geometry;
pub mod grid;
pub mod topology;

use grid::Grid;

pub const XDMF_TAG: &str = "Xdmf";

#[derive(Debug, Serialize)]
pub struct Xdmf {
    #[serde(rename = "@Version")]
    pub version: String,

    #[serde(rename = "Domain")]
    pub domains: Vec<Domain>,
}

impl Xdmf {
    pub fn new(domain: Domain) -> Self {
        Self {
            version: "3.0".to_string(),
            domains: vec![domain],
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Domain {
    #[serde(rename = "Grid")]
    pub grids: Vec<Grid>,
}

impl Domain {
    pub fn new(grid: Grid) -> Self {
        Self { grids: vec![grid] }
    }
}
