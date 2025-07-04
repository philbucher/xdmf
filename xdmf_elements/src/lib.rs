use serde::Serialize;

pub mod attribute;
pub mod data_item;
pub mod dimensions;
pub mod geometry;
pub mod grid;
pub mod topology;

use data_item::DataItem;
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
            .map_err(std::io::Error::other)
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

    #[serde(rename = "DataItem", skip_serializing_if = "Vec::is_empty")]
    pub data_items: Vec<DataItem>,
}

impl Domain {
    pub fn new(grid: Grid) -> Self {
        Self {
            grids: vec![grid],
            data_items: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum CellType {
    Triangle = 4,
    Quadrilateral = 5,
    Tetrahedron = 6,
    Pyramid = 7,
    Wedge = 8,
    Hexahedron = 9,
    Edge3 = 34,
    Quadrilateral9 = 35,
    Triangle6 = 36,
    Quadrilateral8 = 37,
    Tetrahedron10 = 38,
    Pyramid13 = 39,
    Wedge15 = 40,
    Wedge18 = 41,
    Hexahedron20 = 48,
    Hexahedron24 = 49,
    Hexahedron27 = 50,
}

impl CellType {
    pub fn num_points(&self) -> usize {
        match self {
            CellType::Triangle => 3,
            CellType::Quadrilateral => 4,
            CellType::Tetrahedron => 4,
            CellType::Pyramid => 5,
            CellType::Wedge => 6,
            CellType::Hexahedron => 8,
            CellType::Edge3 => 2,
            CellType::Quadrilateral9 => 9,
            CellType::Triangle6 => 6,
            CellType::Quadrilateral8 => 8,
            CellType::Tetrahedron10 => 10,
            CellType::Pyramid13 => 13,
            CellType::Wedge15 => 15,
            CellType::Wedge18 => 18,
            CellType::Hexahedron20 => 20,
            CellType::Hexahedron24 => 24,
            CellType::Hexahedron27 => 27,
        }
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
                    dimensions: Some(dimensions::Dimensions(vec![3])),
                    data: "1.0 2.0 3.0".to_string(),
                    number_type: Some(data_item::NumberType::Float),
                    ..Default::default()
                },
            },
            topology::Topology {
                topology_type: topology::TopologyType::Triangle,
                number_of_elements: "1".to_string(),
                data_item: data_item::DataItem {
                    dimensions: Some(dimensions::Dimensions(vec![3])),
                    number_type: Some(data_item::NumberType::Int),
                    data: "0 1 2".to_string(),
                    ..Default::default()
                },
            },
        );
        let domain = Domain::new(grid);

        assert_eq!(domain.grids.len(), 1);
        assert!(domain.data_items.is_empty());
    }

    #[test]
    fn test_domain_default() {
        let mut domain = Domain::default();
        assert!(domain.grids.is_empty());
        assert!(domain.data_items.is_empty());

        domain.data_items.push(DataItem::default());
        assert_eq!(domain.data_items.len(), 1);
    }
}
