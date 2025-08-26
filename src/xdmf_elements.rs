//! This module contains the main XDMF elements along with their serialization logic.
//!
//! The official documentaion for these can be found [here](https://www.xdmf.org/index.php/XDMF_Model_and_Format.html).

use serde::Serialize;

pub mod attribute;
pub mod data_item;
pub mod dimensions;
pub mod geometry;
pub mod grid;
pub mod topology;

use data_item::DataItem;
use grid::Grid;

/// Name of the root element of an XDMF file.
pub const XDMF_TAG: &str = "Xdmf";

/// The root element of an XDMF file. Specifies basic information and holds the domain(s).
#[derive(Debug, Serialize)]
pub struct Xdmf {
    #[serde(rename = "@Version")]
    #[doc(hidden)]
    pub version: String,

    #[serde(rename = "@xmlns:xi")]
    #[doc(hidden)]
    pub xinclude_url: String,

    #[serde(rename = "Domain")]
    #[doc(hidden)]
    pub domains: Vec<Domain>,

    #[serde(rename = "Information", skip_serializing_if = "Vec::is_empty")]
    #[doc(hidden)]
    pub information: Vec<Information>,
}

impl Xdmf {
    /// Create a new XDMF instance with a single domain
    pub fn new(domain: Domain) -> Self {
        Self {
            version: "2.0".to_string(),
            xinclude_url: "http://www.w3.org/2001/XInclude".to_string(),
            domains: vec![domain],
            information: vec![],
        }
    }

    /// Write the serialized XDMF to the given writer.
    ///
    /// "Pretty-printing" with 4 spaces for indentation is used to format the output, making it human-readable.
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

/// Stores application-specific metadata that doesn't fit into the standard data model.
///
/// The `Information` element is designed to hold additional, system- or code-specific
/// details that can be safely ignored by other components.
///
/// See <https://www.xdmf.org/index.php/XDMF_Model_and_Format.html#Information>
#[derive(Debug, Serialize)]
pub struct Information {
    #[serde(rename = "@Name")]
    #[doc(hidden)]
    pub name: String,

    #[serde(rename = "@Value")]
    #[doc(hidden)]
    pub value: String,
}

impl Information {
    /// Create a new information instance
    pub fn new(name: impl ToString, value: impl ToString) -> Self {
        Self {
            name: name.to_string(),
            value: value.to_string(),
        }
    }
}

/// Top level container for grids, represents a computational domain.
#[derive(Debug, Default, Serialize)]
pub struct Domain {
    #[serde(rename = "Grid")]
    #[doc(hidden)]
    pub grids: Vec<Grid>,

    #[serde(rename = "DataItem", skip_serializing_if = "Vec::is_empty")]
    #[doc(hidden)]
    pub data_items: Vec<DataItem>,
}

impl Domain {
    /// Create a new domain with a single grid
    pub fn new(grid: Grid) -> Self {
        Self {
            grids: vec![grid],
            data_items: Vec::new(),
        }
    }
}

/// Cell types as defined in the VTK file format.
///
/// See <https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf> for details.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CellType {
    #[doc(hidden)]
    Vertex = 1,
    #[doc(hidden)]
    Edge = 2,
    #[doc(hidden)]
    Triangle = 4,
    #[doc(hidden)]
    Quadrilateral = 5,
    #[doc(hidden)]
    Tetrahedron = 6,
    #[doc(hidden)]
    Pyramid = 7,
    #[doc(hidden)]
    Wedge = 8,
    #[doc(hidden)]
    Hexahedron = 9,
    #[doc(hidden)]
    Edge3 = 34,
    #[doc(hidden)]
    Quadrilateral9 = 35,
    #[doc(hidden)]
    Triangle6 = 36,
    #[doc(hidden)]
    Quadrilateral8 = 37,
    #[doc(hidden)]
    Tetrahedron10 = 38,
    #[doc(hidden)]
    Pyramid13 = 39,
    #[doc(hidden)]
    Wedge15 = 40,
    #[doc(hidden)]
    Wedge18 = 41,
    #[doc(hidden)]
    Hexahedron20 = 48,
    #[doc(hidden)]
    Hexahedron24 = 49,
    #[doc(hidden)]
    Hexahedron27 = 50,
}

impl CellType {
    /// The number of points for the given cell type.
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

        assert_eq!(xdmf.version, "2.0");
        assert_eq!(xdmf.domains.len(), 1);
    }

    #[test]
    fn xdmf_new_with_information() {
        let xdmf = Xdmf {
            information: vec![Information::new("the_name", "some_value")],
            ..Default::default()
        };

        assert_eq!(xdmf.version, "2.0");
        assert_eq!(xdmf.domains.len(), 1);
        assert_eq!(xdmf.information.len(), 1);
        assert_eq!(xdmf.information[0].name, "the_name");
        assert_eq!(xdmf.information[0].value, "some_value");
    }

    #[test]
    fn xdmf_default() {
        let xdmf = Xdmf::default();

        assert_eq!(xdmf.version, "2.0");
        assert_eq!(xdmf.domains.len(), 1);
    }

    #[test]
    fn xdmf_serialization() {
        let xdmf = Xdmf::default();

        pretty_assertions::assert_eq!(
            to_string(&xdmf).unwrap(),
            "<Xdmf Version=\"2.0\" xmlns:xi=\"http://www.w3.org/2001/XInclude\"><Domain/></Xdmf>"
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
                    data: "1.0 2.0 3.0".into(),
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
                    data: "0 1 2".into(),
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
