use serde::Serialize;

use crate::data_item::DataItem;

#[derive(Debug, Serialize)]
pub struct Geometry {
    #[serde(rename = "@GeometryType")]
    pub geometry_type: GeometryType,

    #[serde(rename = "DataItem")]
    pub data_item: DataItem,
}

#[derive(Debug, Serialize)]
pub enum GeometryType {
    XYZ,
    XY,
}
