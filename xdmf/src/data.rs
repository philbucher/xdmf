use xdmf_elements::attribute;

use crate::values::Values;

pub enum Data {
    PointData(PointDataImpl),
    CellData(CellDataImpl),
}

impl Data {
    pub fn new_point_data(attribute_type: attribute::AttributeType, values: Values) -> Self {
        Data::PointData(PointDataImpl {
            attribute_type,
            values,
        })
    }
    pub fn new_cell_data(attribute_type: attribute::AttributeType, values: Values) -> Self {
        Data::CellData(CellDataImpl {
            attribute_type,
            values,
        })
    }
    pub(crate) fn attribute_type(&self) -> attribute::AttributeType {
        match self {
            Data::PointData(data) => data.attribute_type,
            Data::CellData(data) => data.attribute_type,
        }
    }
    pub(crate) fn center(&self) -> attribute::Center {
        match self {
            Data::PointData(_) => attribute::Center::Node,
            Data::CellData(_) => attribute::Center::Cell,
        }
    }
    pub(crate) fn values(&self) -> &Values {
        match self {
            Data::PointData(data) => &data.values,
            Data::CellData(data) => &data.values,
        }
    }
}

pub struct PointDataImpl {
    attribute_type: attribute::AttributeType,
    values: Values,
}

pub struct CellDataImpl {
    attribute_type: attribute::AttributeType,
    values: Values,
}
