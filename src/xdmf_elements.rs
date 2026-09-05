//! The XDMF elements and their serialization.
//!
//! The format's own documentation is [here](https://www.xdmf.org/index.php/XDMF_Model_and_Format.html).

use serde::{Deserialize, Serialize};

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
#[derive(Debug, Serialize, Deserialize)]
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

/// Application-specific metadata outside the standard data model, which other readers may ignore.
///
/// See <https://www.xdmf.org/index.php/XDMF_Model_and_Format.html#Information>
#[derive(Debug, Serialize, Deserialize)]
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
#[derive(Debug, Default, Serialize, Deserialize)]
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

// Declares the cell types, their XDMF codes and their point counts as one list, so a new type
// cannot be added while missing from `ALL`, `from_code` or `num_points`.
macro_rules! define_cell_types {
    ($($variant:ident = $code:literal => $num_points:literal),+ $(,)?) => {
        /// The cell types this crate can write, i.e. the XDMF topology types it supports.
        ///
        /// The discriminants are the XDMF topology type codes, which is what a `Mixed` topology's
        /// connectivity carries per cell — they are *not* the VTK cell codes, which differ (a
        /// hexahedron is 9 here and 12 in VTK). What follows VTK is the node *ordering* within a
        /// cell, see <https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf>.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(u8)]
        pub enum CellType {
            $(
                #[doc(hidden)]
                $variant = $code,
            )+
        }

        impl CellType {
            /// Every cell type this crate knows, in the order they are declared.
            ///
            /// A slice rather than a sized array, so adding a cell type is not a breaking change
            /// for a caller naming the type.
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            /// The cell type a `Mixed`-topology connectivity's per-cell code decodes to, `None`
            /// for a code this crate does not know.
            pub(crate) fn from_code(code: u8) -> Option<Self> {
                match code {
                    $($code => Some(Self::$variant),)+
                    _ => None,
                }
            }

            /// The number of points for the given cell type.
            pub fn num_points(&self) -> usize {
                match self {
                    $(Self::$variant => $num_points,)+
                }
            }
        }
    };
}

// `variant = XDMF topology type code => points per cell`
define_cell_types!(
    Vertex         =  1 =>  1,
    Edge           =  2 =>  2,
    Triangle       =  4 =>  3,
    Quadrilateral  =  5 =>  4,
    Tetrahedron    =  6 =>  4,
    Pyramid        =  7 =>  5,
    Wedge          =  8 =>  6,
    Hexahedron     =  9 =>  8,
    Edge3          = 34 =>  3,
    Quadrilateral9 = 35 =>  9,
    Triangle6      = 36 =>  6,
    Quadrilateral8 = 37 =>  8,
    Tetrahedron10  = 38 => 10,
    Pyramid13      = 39 => 13,
    Wedge15        = 40 => 15,
    Wedge18        = 41 => 18,
    Hexahedron20   = 48 => 20,
    Hexahedron24   = 49 => 24,
    Hexahedron27   = 50 => 27,
);

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
                data_items: vec![data_item::DataItem {
                    dimensions: Some(dimensions::Dimensions(vec![3])),
                    data: "1.0 2.0 3.0".into(),
                    number_type: Some(data_item::NumberType::Float),
                    ..Default::default()
                }],
            },
            topology::Topology {
                topology_type: topology::TopologyType::Triangle,
                nodes_per_element: None,
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
    fn cell_type_from_code_round_trips_every_variant() {
        for &cell_type in CellType::ALL {
            assert_eq!(CellType::from_code(cell_type as u8), Some(cell_type));
        }
    }

    /// Two variants sharing a code would make `from_code` return the first of them for both, and
    /// a `Mixed` connectivity would read back as the wrong cell type.
    #[test]
    fn every_cell_type_has_its_own_code() {
        let mut codes: Vec<u8> = CellType::ALL.iter().map(|&c| c as u8).collect();
        let declared = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), declared);
    }

    /// A cell has at least one point, and the connectivity is walked in strides of that count, so
    /// a zero would make a mesh of such cells read as an endless one.
    #[test]
    fn every_cell_type_has_points() {
        for cell_type in CellType::ALL {
            assert!(cell_type.num_points() > 0, "{cell_type:?}");
        }
    }

    #[test]
    fn cell_type_from_code_rejects_unknown() {
        assert_eq!(CellType::from_code(0), None);
        assert_eq!(CellType::from_code(3), None);
        assert_eq!(CellType::from_code(255), None);
    }

    #[test]
    fn domain_serialization() {
        let domain = Domain::default();
        pretty_assertions::assert_eq!(to_string(&domain).unwrap(), "<Domain/>");
    }
}
