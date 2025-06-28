use xdmf_elements::attribute;

use crate::values::Values;

pub enum Data<'a> {
    PointData(PointDataImpl<'a>),
    CellData(CellDataImpl<'a>),
}

impl<'a> Data<'a> {
    pub fn new_point_data(
        name: impl ToString,
        attribute_type: attribute::AttributeType,
        values: Values<'a>,
    ) -> Self {
        Data::PointData(PointDataImpl {
            name: name.to_string(),
            attribute_type,
            values,
        })
    }
    pub fn new_cell_data(
        name: impl ToString,
        attribute_type: attribute::AttributeType,
        values: Values<'a>,
    ) -> Self {
        Data::CellData(CellDataImpl {
            name: name.to_string(),
            attribute_type,
            values,
        })
    }
    pub(crate) fn name(&self) -> String {
        match self {
            Data::PointData(data) => data.name.clone(),
            Data::CellData(data) => data.name.clone(),
        }
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
    pub(crate) fn values(&self) -> &Values<'a> {
        match self {
            Data::PointData(data) => &data.values,
            Data::CellData(data) => &data.values,
        }
    }
}

pub struct PointDataImpl<'a> {
    name: String,
    attribute_type: attribute::AttributeType,
    values: Values<'a>,
}

pub struct CellDataImpl<'a> {
    name: String,
    attribute_type: attribute::AttributeType,
    values: Values<'a>,
}
