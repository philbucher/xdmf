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

    /// # Errors
    ///
    /// TODO
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CellType {
    Vertex = 1,
    Edge = 2,
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
            Self::Vertex => 1,
            Self::Edge => 2,
            Self::Triangle => 3,
            Self::Quadrilateral => 4,
            Self::Tetrahedron => 4,
            Self::Pyramid => 5,
            Self::Wedge => 6,
            Self::Hexahedron => 8,
            Self::Edge3 => 3,
            Self::Quadrilateral9 => 9,
            Self::Triangle6 => 6,
            Self::Quadrilateral8 => 8,
            Self::Tetrahedron10 => 10,
            Self::Pyramid13 => 13,
            Self::Wedge15 => 15,
            Self::Wedge18 => 18,
            Self::Hexahedron20 => 20,
            Self::Hexahedron24 => 24,
            Self::Hexahedron27 => 27,
        }
    }
}

#[cfg(test)]
mod tests {
    use quick_xml::se::to_string;

    use super::*;

    #[test]
    fn xdmf_new() {
        let domain = Domain::default();
        let xdmf = Xdmf::new(domain);

        assert_eq!(xdmf.version, "3.0");
        assert_eq!(xdmf.domains.len(), 1);
    }

    #[test]
    fn xdmf_default() {
        let xdmf = Xdmf::default();

        assert_eq!(xdmf.version, "3.0");
        assert_eq!(xdmf.domains.len(), 1);
    }

    #[test]
    fn xdmf_serialization() {
        let xdmf = Xdmf::default();

        pretty_assertions::assert_eq!(
            to_string(&xdmf).unwrap(),
            "<Xdmf Version=\"3.0\"><Domain/></Xdmf>"
        );
    }

    #[test]
    fn domain_new() {
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
    fn domain_default() {
        let mut domain = Domain::default();
        assert!(domain.grids.is_empty());
        assert!(domain.data_items.is_empty());

        domain.data_items.push(DataItem::default());
        assert_eq!(domain.data_items.len(), 1);
    }

    #[test]
    fn domain_serialization() {
        let domain = Domain::default();
        pretty_assertions::assert_eq!(to_string(&domain).unwrap(), "<Domain/>");
    }
}
