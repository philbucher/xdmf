//! Parsing an XDMF document into `xdmf_elements` structs and navigating its shape: which grid
//! holds the mesh, how many submeshes it has, how many steps were written, and resolving the one
//! `Reference="XML"` `XPath` shape this crate's own writer emits.

use std::path::{Path, PathBuf};

use crate::{
    Error, Result,
    xdmf_elements::{
        Domain, Xdmf,
        data_item::{DataContent, DataItem},
        grid::{CollectionType, Grid, GridType},
    },
};

/// A parsed document plus the directory its heavy-data paths are relative to.
pub(super) struct Document {
    pub xdmf: Xdmf,
    pub base_dir: PathBuf,
}

impl Document {
    pub(super) fn open(file_name: &Path) -> Result<Self> {
        let xml = std::fs::read_to_string(file_name)
            .map_err(crate::error::io_ctx("reading XDMF file", file_name))?;

        let xdmf: Xdmf =
            quick_xml::de::from_str(&xml).map_err(|source| Error::InvalidDocument {
                reason: format!("could not parse XDMF XML: {source}"),
            })?;

        let base_dir = file_name
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);

        Ok(Self { xdmf, base_dir })
    }

    pub(super) fn domain(&self) -> Result<&Domain> {
        if self.xdmf.domains.len() > 1 {
            return Err(Error::Unsupported {
                reason: "multiple Domains are not supported".to_string(),
            });
        }

        self.xdmf
            .domains
            .first()
            .ok_or_else(|| Error::InvalidDocument {
                reason: "the document has no Domain".to_string(),
            })
    }

    /// Value of the `Information` element with the given name, at the document's own top level.
    pub(super) fn information(&self, name: &str) -> Option<&str> {
        self.xdmf
            .information
            .iter()
            .find(|info| info.name == name)
            .map(|info| info.value.as_str())
    }
}

/// How the document's single root `Grid` breaks down: which grids carry the mesh (one per named
/// submesh, or a single unnamed one without submeshes), and, for each, its grids in step order.
///
/// A submesh/mesh not yet carrying any time step has exactly one entry, with no `Time` and no
/// `Attribute`s -- the shape [`crate::TimeSeriesWriter::write_mesh`]/
/// [`crate::TimeSeriesWriter::write_mesh_with_submeshes`] themselves write, before any
/// [`crate::TimeSeriesDataWriter::write_time_step`] call.
pub(super) struct Analysis<'a> {
    /// Empty when the mesh has no submeshes.
    pub(super) submesh_names: Vec<String>,
    /// One entry per submesh (or a single entry without submeshes), each the grids of that
    /// submesh in step order.
    pub(super) submeshes: Vec<Vec<&'a Grid>>,
}

impl<'a> Analysis<'a> {
    pub(super) fn build(domain: &'a Domain) -> Result<Self> {
        if domain.grids.len() > 1 {
            return Err(Error::Unsupported {
                reason: "multiple root Grids in one Domain are not supported".to_string(),
            });
        }

        let root = domain.grids.first().ok_or_else(|| Error::InvalidDocument {
            reason: "the Domain has no Grid".to_string(),
        })?;

        match (root.grid_type, root.collection_type) {
            (GridType::Uniform, _) => Ok(Self {
                submesh_names: Vec::new(),
                submeshes: vec![vec![root]],
            }),
            (GridType::Collection, Some(CollectionType::Temporal)) => {
                let steps = collection_children(root)?;
                Ok(Self {
                    submesh_names: Vec::new(),
                    submeshes: vec![steps],
                })
            }
            (GridType::Collection, Some(CollectionType::Spatial)) => {
                let children = collection_children(root)?;

                let mut submesh_names = Vec::with_capacity(children.len());
                let mut submeshes = Vec::with_capacity(children.len());

                for child in children {
                    submesh_names.push(child.name.clone());
                    submeshes.push(submesh_steps(child)?);
                }

                Ok(Self {
                    submesh_names,
                    submeshes,
                })
            }
            _ => Err(Error::Unsupported {
                reason: format!(
                    "root Grid '{}' has an unexpected shape (GridType {:?}, CollectionType {:?})",
                    root.name, root.grid_type, root.collection_type
                ),
            }),
        }
    }

    pub(super) fn num_submeshes(&self) -> usize {
        self.submeshes.len()
    }

    pub(super) fn times(&self) -> Result<Vec<String>> {
        let Some(first_submesh) = self.submeshes.first() else {
            return Ok(Vec::new());
        };

        // no step has been written yet: the single grid carries no `Time`
        if first_submesh.len() == 1 && first_submesh[0].time.is_none() {
            return Ok(Vec::new());
        }

        first_submesh
            .iter()
            .map(|grid| {
                grid.time
                    .as_ref()
                    .map(|time| time.value.clone())
                    .ok_or_else(|| Error::InvalidDocument {
                        reason: format!("Grid '{}' has no Time", grid.name),
                    })
            })
            .collect()
    }
}

/// A submesh's own grid holding one step (a plain submesh, before any step was written) or the
/// per-submesh `Temporal` collection wrapping its steps.
fn submesh_steps(child: &Grid) -> Result<Vec<&Grid>> {
    match (child.grid_type, child.collection_type) {
        (GridType::Uniform, _) => Ok(vec![child]),
        (GridType::Collection, Some(CollectionType::Temporal)) => collection_children(child),
        _ => Err(Error::Unsupported {
            reason: format!(
                "submesh Grid '{}' has an unexpected shape (GridType {:?}, CollectionType {:?})",
                child.name, child.grid_type, child.collection_type
            ),
        }),
    }
}

fn collection_children(grid: &Grid) -> Result<Vec<&Grid>> {
    let children: Vec<&Grid> = grid.grids.iter().flatten().collect();

    if children.is_empty() {
        return Err(Error::InvalidDocument {
            reason: format!("collection Grid '{}' has no children", grid.name),
        });
    }

    Ok(children)
}

/// The one `Reference="XML"` shape this crate's writer emits: an `XPath` naming a `Domain`-level
/// `DataItem` by its `Name` attribute. Any other `Reference` value, or any other `XPath` shape, is
/// [`Error::Unsupported`] rather than guessed at.
pub(super) fn resolve_reference<'a>(item: &DataItem, domain: &'a Domain) -> Result<&'a DataItem> {
    let Some(reference) = &item.reference else {
        return Err(Error::Internal(
            "resolve_reference called on a DataItem with no Reference",
        ));
    };

    if reference != "XML" {
        return Err(Error::Unsupported {
            reason: format!("Reference=\"{reference}\" is not supported, only \"XML\" is"),
        });
    }

    let DataContent::Raw(xpath) = &item.data else {
        return Err(Error::InvalidDocument {
            reason: "a Reference DataItem has no XPath text".to_string(),
        });
    };

    const PREFIX: &str = "/Xdmf/Domain/DataItem[@Name=\"";
    const SUFFIX: &str = "\"]";

    let name = xpath
        .strip_prefix(PREFIX)
        .and_then(|rest| rest.strip_suffix(SUFFIX))
        .ok_or_else(|| Error::Unsupported {
            reason: format!(
                "XPath reference '{xpath}' is not supported, only \
                 /Xdmf/Domain/DataItem[@Name=\"...\"] is"
            ),
        })?;

    find_by_name(domain, name).ok_or_else(|| Error::InvalidDocument {
        reason: format!("reference '{xpath}' names a DataItem that does not exist"),
    })
}

/// Look up a `Domain`-level `DataItem` by its own `Name`, as a `submesh_cells_k`/`submesh_points_k`
/// entry in an `<Information>` list does -- a plain name, not an `XPath`.
pub(super) fn find_by_name<'a>(domain: &'a Domain, name: &str) -> Option<&'a DataItem> {
    domain
        .data_items
        .iter()
        .find(|item| item.name.as_deref() == Some(name))
}
