//! Parses an `.xdmf2` file's light data and resolves it down to the pieces `TimeSeriesReader`
//! actually needs: the mesh's `Geometry`/`Topology` `DataItem`s (with any `Reference="XML"`
//! resolved) and the per-step `Attribute` lists.

use std::path::{Path, PathBuf};

use crate::{
    Error, Result,
    error::io_ctx,
    xdmf_elements::{
        Xdmf,
        attribute::Attribute,
        data_item::DataItem,
        geometry::GeometryType,
        grid::{CollectionType, GridType},
        topology::TopologyType,
    },
};

pub(crate) struct LightData {
    pub base_dir: PathBuf,
    pub num_points: usize,
    pub num_cells: usize,
    pub geometry_data_item: DataItem,
    pub topology_type: TopologyType,
    pub topology_data_item: DataItem,
    pub times: Vec<String>,
    pub steps: Vec<Vec<Attribute>>,
}

pub(crate) fn parse(path: &Path) -> Result<LightData> {
    let xml = std::fs::read_to_string(path).map_err(io_ctx("reading XDMF file", path))?;
    let xdmf: Xdmf = quick_xml::de::from_str(&xml).map_err(|parse_error| Error::InvalidFile {
        path: path.to_path_buf(),
        reason: format!("XML does not parse: {parse_error}"),
    })?;

    let base_dir = path.parent().map_or_else(PathBuf::new, Path::to_path_buf);

    if xdmf.domains.len() != 1 {
        return Err(Error::Unsupported {
            reason: format!(
                "{} Domain elements, only exactly one is supported",
                xdmf.domains.len()
            ),
        });
    }
    let mut domain = xdmf.domains;
    let domain = domain.remove(0);
    let all_data_items = domain.data_items;

    if domain.grids.len() != 1 {
        return Err(Error::Unsupported {
            reason: format!(
                "{} top-level Grid elements, only exactly one is supported",
                domain.grids.len()
            ),
        });
    }
    let mut grids = domain.grids;
    let root_grid = grids.remove(0);

    let (mesh_grid, step_grids) = match (root_grid.grid_type, root_grid.collection_type) {
        (GridType::Uniform, _) => (root_grid, Vec::new()),
        (GridType::Collection, Some(CollectionType::Temporal)) => {
            let sub_grids = root_grid.grids.unwrap_or_default();
            let mesh_grid = sub_grids
                .first()
                .cloned()
                .ok_or_else(|| Error::InvalidFile {
                    path: path.to_path_buf(),
                    reason: "temporal Collection Grid has no child Grids".to_string(),
                })?;
            (mesh_grid, sub_grids)
        }
        (grid_type, collection_type) => {
            return Err(Error::Unsupported {
                reason: format!(
                    "top-level GridType={grid_type:?}{} is not supported",
                    collection_type
                        .map(|c| format!(" CollectionType={c:?}"))
                        .unwrap_or_default()
                ),
            });
        }
    };

    let geometry = mesh_grid.geometry.ok_or_else(|| Error::InvalidFile {
        path: path.to_path_buf(),
        reason: "Grid has no Geometry".to_string(),
    })?;
    if geometry.geometry_type != GeometryType::XYZ {
        return Err(Error::Unsupported {
            reason: format!(
                "GeometryType={:?} is not supported, only XYZ is",
                geometry.geometry_type
            ),
        });
    }
    let geometry_data_item = resolve_reference(geometry.data_item, &all_data_items, path)?;

    let topology = mesh_grid.topology.ok_or_else(|| Error::InvalidFile {
        path: path.to_path_buf(),
        reason: "Grid has no Topology".to_string(),
    })?;
    let topology_data_item = resolve_reference(topology.data_item, &all_data_items, path)?;
    let num_cells: usize = topology
        .number_of_elements
        .parse()
        .map_err(|_parse_error| Error::InvalidFile {
            path: path.to_path_buf(),
            reason: format!(
                "Topology NumberOfElements=\"{}\" is not a valid integer",
                topology.number_of_elements
            ),
        })?;

    let num_points = geometry_data_item
        .dimensions
        .as_ref()
        .and_then(|d| d.0.first().copied())
        .ok_or_else(|| Error::InvalidFile {
            path: path.to_path_buf(),
            reason: "Geometry DataItem has no Dimensions".to_string(),
        })?;

    let times = step_grids
        .iter()
        .map(|grid| {
            grid.time
                .as_ref()
                .map(|time| time.value.clone())
                .ok_or_else(|| Error::InvalidFile {
                    path: path.to_path_buf(),
                    reason: "a Grid in the temporal Collection has no Time".to_string(),
                })
        })
        .collect::<Result<Vec<_>>>()?;

    let steps = step_grids
        .into_iter()
        .map(|grid| grid.attributes.unwrap_or_default())
        .collect();

    Ok(LightData {
        base_dir,
        num_points,
        num_cells,
        geometry_data_item,
        topology_type: topology.topology_type,
        topology_data_item,
        times,
        steps,
    })
}

/// Resolves exactly the `Reference="XML"` pattern this crate's own writer produces
/// (`/Xdmf/Domain/DataItem[@Name="..."]`). Any other `Reference` value, or any other
/// XPath-like expression, is `Unsupported` rather than guessed at.
fn resolve_reference(item: DataItem, all_data_items: &[DataItem], path: &Path) -> Result<DataItem> {
    let Some(reference) = item.reference.clone() else {
        return Ok(item);
    };
    if reference != "XML" {
        return Err(Error::Unsupported {
            reason: format!("DataItem Reference=\"{reference}\" is not supported, only \"XML\" is"),
        });
    }

    let text = item.text.as_deref().ok_or_else(|| Error::InvalidFile {
        path: path.to_path_buf(),
        reason: "DataItem Reference=\"XML\" has no text content".to_string(),
    })?;
    let name = text
        .strip_prefix("/Xdmf/Domain/DataItem[@Name=\"")
        .and_then(|rest| rest.strip_suffix("\"]"))
        .ok_or_else(|| Error::Unsupported {
            reason: format!(
                "DataItem Reference=\"XML\" expression '{text}' is not the '/Xdmf/Domain/DataItem[@Name=\"...\"]' pattern this crate supports"
            ),
        })?;

    all_data_items
        .iter()
        .find(|candidate| candidate.name.as_deref() == Some(name))
        .cloned()
        .ok_or_else(|| Error::InvalidFile {
            path: path.to_path_buf(),
            reason: format!(
                "Reference=\"XML\" points at DataItem[@Name=\"{name}\"], which does not exist under /Xdmf/Domain"
            ),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xdmf_elements::data_item::NumberType;

    fn coords_item() -> DataItem {
        DataItem {
            name: Some("coords".to_string()),
            text: Some("0 0 0".to_string()),
            number_type: Some(NumberType::Float),
            ..Default::default()
        }
    }

    #[test]
    fn resolve_reference_passes_through_when_no_reference() {
        let item = coords_item();
        let resolved = resolve_reference(item.clone(), &[], Path::new("x.xdmf2")).unwrap();
        assert_eq!(resolved.text, item.text);
    }

    #[test]
    fn resolve_reference_finds_named_data_item() {
        let all_data_items = vec![coords_item()];
        let reference_item = DataItem {
            text: Some("/Xdmf/Domain/DataItem[@Name=\"coords\"]".to_string()),
            reference: Some("XML".to_string()),
            ..Default::default()
        };
        let resolved =
            resolve_reference(reference_item, &all_data_items, Path::new("x.xdmf2")).unwrap();
        assert_eq!(resolved.text, Some("0 0 0".to_string()));
        assert!(resolved.reference.is_none());
    }

    #[test]
    fn resolve_reference_rejects_other_expressions() {
        let reference_item = DataItem {
            text: Some("/Xdmf/Domain/Grid[1]/Geometry/DataItem".to_string()),
            reference: Some("XML".to_string()),
            ..Default::default()
        };
        let err = resolve_reference(reference_item, &[], Path::new("x.xdmf2")).unwrap_err();
        std::assert_matches!(err, Error::Unsupported { reason } if reason.contains("Geometry/DataItem"));
    }

    #[test]
    fn resolve_reference_rejects_missing_target() {
        let reference_item = DataItem {
            text: Some("/Xdmf/Domain/DataItem[@Name=\"missing\"]".to_string()),
            reference: Some("XML".to_string()),
            ..Default::default()
        };
        let err = resolve_reference(reference_item, &[], Path::new("x.xdmf2")).unwrap_err();
        std::assert_matches!(err, Error::InvalidFile { reason, .. } if reason.contains("missing"));
    }
}
