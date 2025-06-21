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

    pub fn write_to(&self, writer: &mut impl std::io::Write) -> std::io::Result<()> {
        let mut file_writer = quick_xml::Writer::new_with_indent(writer, b' ', 4);
        file_writer
            .write_serializable(XDMF_TAG, self)
            .map_err(|e| std::io::Error::other(e))
    }
}

impl Default for Xdmf {
    fn default() -> Self {
        Self::new(Domain::default())
    }
}

#[derive(Debug, Default, Serialize)]
pub struct Domain {
    #[serde(rename = "Grid")]
    pub grids: Vec<Grid>,
}

impl Domain {
    pub fn new(grid: Grid) -> Self {
        Self { grids: vec![grid] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xdmf_new() {
        let domain = Domain::default();
        let xdmf = Xdmf::new(domain);

        assert_eq!(xdmf.version, "3.0");
        assert_eq!(xdmf.domains.len(), 1);
    }

    #[test]
    fn test_xdmf_default() {
        let xdmf = Xdmf::default();

        assert_eq!(xdmf.version, "3.0");
        assert_eq!(xdmf.domains.len(), 1);
    }

    #[test]
    fn test_domain_new() {
        let grid = Grid::new_uniform(
            "test_grid",
            geometry::Geometry {
                geometry_type: geometry::GeometryType::XYZ,
                data_item: data_item::DataItem {
                    dimensions: dimensions::Dimensions(vec![3]),
                    data: "1.0 2.0 3.0".to_string(),
                    number_type: data_item::NumberType::Float,
                    ..Default::default()
                },
            },
            topology::Topology {
                topology_type: topology::TopologyType::Triangle,
                number_of_elements: "1".to_string(),
                data_item: data_item::DataItem {
                    dimensions: dimensions::Dimensions(vec![3]),
                    number_type: data_item::NumberType::Int,
                    data: "0 1 2".to_string(),
                    ..Default::default()
                },
            },
        );
        let domain = Domain::new(grid);

        assert_eq!(domain.grids.len(), 1);
    }

    #[test]
    fn test_domain_default() {
        let domain = Domain::default();
        assert!(domain.grids.is_empty());
    }
}
