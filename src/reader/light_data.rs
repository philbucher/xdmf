//! Parsing an XDMF document into `xdmf_elements` structs and navigating its shape: which grid
//! holds the mesh, how many submeshes it has, how many steps were written, and resolving the one
//! `Reference="XML"` `XPath` shape this crate's own writer emits.

use std::path::{Path, PathBuf};

use super::hdf5_reader::FileCache;
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
    /// The heavy-data file a read last opened, held open for the next one -- see [`FileCache`].
    /// Lives here because it is exactly as long-lived as the paths it caches.
    pub files: FileCache,
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

        Ok(Self {
            xdmf,
            base_dir,
            files: FileCache::default(),
        })
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

/// Where one grid sits under the `Domain`: the `grids` indices to follow from the `Domain`'s
/// single root grid, so an empty path is the root grid itself.
///
/// Positions rather than `&Grid` references: [`Analysis`] is built once, in
/// [`TimeSeriesReader::new`](crate::TimeSeriesReader::new), and kept next to the [`Document`] it
/// describes -- which a borrow of that same document could not be.
#[derive(Clone, Debug)]
pub(super) struct GridPath(Vec<usize>);

impl GridPath {
    fn root() -> Self {
        Self(Vec::new())
    }

    fn child(&self, index: usize) -> Self {
        let mut path = self.0.clone();
        path.push(index);

        Self(path)
    }

    /// The grid this path names. Every path was produced by walking the same document, so a
    /// missing step means the document changed underneath the reader.
    pub(super) fn resolve<'a>(&self, domain: &'a Domain) -> Result<&'a Grid> {
        let mut grid = domain.grids.first().ok_or_else(|| Error::InvalidDocument {
            reason: "the Domain has no Grid".to_string(),
        })?;

        for &index in &self.0 {
            grid = grid
                .grids
                .as_deref()
                .and_then(|children| children.get(index))
                .ok_or(Error::Internal("a grid path no longer resolves"))?;
        }

        Ok(grid)
    }
}

/// How the document's single root `Grid` breaks down: which grids carry the mesh (one per named
/// submesh, or a single unnamed one without submeshes), and, for each, its grids in step order.
///
/// A submesh/mesh not yet carrying any time step has exactly one entry, with no `Time` and no
/// `Attribute`s -- the shape [`crate::TimeSeriesWriter::write_mesh`]/
/// [`crate::TimeSeriesWriter::write_mesh_with_submeshes`] themselves write, before any
/// [`crate::TimeSeriesDataWriter::write_time_step`] call.
pub(super) struct Analysis {
    /// Empty when the mesh has no submeshes.
    submesh_names: Vec<String>,
    /// One entry per submesh (or a single entry without submeshes), each the grids of that
    /// submesh in step order.
    submeshes: Vec<Vec<GridPath>>,
}

impl Analysis {
    pub(super) fn build(domain: &Domain) -> Result<Self> {
        if domain.grids.len() > 1 {
            return Err(Error::Unsupported {
                reason: "multiple root Grids in one Domain are not supported".to_string(),
            });
        }

        let root = domain.grids.first().ok_or_else(|| Error::InvalidDocument {
            reason: "the Domain has no Grid".to_string(),
        })?;
        let root_path = GridPath::root();

        match (root.grid_type, root.collection_type) {
            (GridType::Uniform, _) => Ok(Self {
                submesh_names: Vec::new(),
                submeshes: vec![vec![root_path]],
            }),
            (GridType::Collection, Some(CollectionType::Temporal)) => {
                let steps = collection_children(root, &root_path)?;
                Ok(Self {
                    submesh_names: Vec::new(),
                    submeshes: vec![steps.into_iter().map(|(_grid, path)| path).collect()],
                })
            }
            (GridType::Collection, Some(CollectionType::Spatial)) => {
                let children = collection_children(root, &root_path)?;

                let mut submesh_names = Vec::with_capacity(children.len());
                let mut submeshes = Vec::with_capacity(children.len());

                for (child, path) in children {
                    submesh_names.push(child.name.clone());
                    submeshes.push(submesh_steps(child, &path)?);
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

    /// Empty when the mesh has no submeshes.
    pub(super) fn submesh_names(&self) -> &[String] {
        &self.submesh_names
    }

    pub(super) fn num_submeshes(&self) -> usize {
        self.submeshes.len()
    }

    /// The grid a submesh's mesh itself is described by: its first, which every step's grid
    /// repeats the geometry and topology of.
    pub(super) fn mesh_grid<'a>(&self, submesh: usize, domain: &'a Domain) -> Result<&'a Grid> {
        self.grid_path(submesh, 0)?.resolve(domain)
    }

    /// The grid of one submesh's `step`-th time step.
    pub(super) fn step_grid<'a>(
        &self,
        submesh: usize,
        step: usize,
        domain: &'a Domain,
    ) -> Result<&'a Grid> {
        self.grid_path(submesh, step)?.resolve(domain)
    }

    fn grid_path(&self, submesh: usize, step: usize) -> Result<&GridPath> {
        let grids = self.submeshes.get(submesh).ok_or(Error::Internal(
            "a grid was asked for with a submesh index out of range",
        ))?;

        grids.get(step).ok_or_else(|| Error::InvalidDocument {
            reason: format!(
                "submesh {submesh} has only {} grids, so it has no step {step}",
                grids.len()
            ),
        })
    }

    pub(super) fn times(&self, domain: &Domain) -> Result<Vec<String>> {
        let Some(first_submesh) = self.submeshes.first() else {
            return Ok(Vec::new());
        };

        let grids = first_submesh
            .iter()
            .map(|path| path.resolve(domain))
            .collect::<Result<Vec<_>>>()?;

        // no step has been written yet: the single grid carries no `Time`
        if grids.len() == 1 && grids[0].time.is_none() {
            return Ok(Vec::new());
        }

        grids
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
fn submesh_steps(child: &Grid, path: &GridPath) -> Result<Vec<GridPath>> {
    match (child.grid_type, child.collection_type) {
        (GridType::Uniform, _) => Ok(vec![path.clone()]),
        (GridType::Collection, Some(CollectionType::Temporal)) => {
            Ok(collection_children(child, path)?
                .into_iter()
                .map(|(_grid, child_path)| child_path)
                .collect())
        }
        _ => Err(Error::Unsupported {
            reason: format!(
                "submesh Grid '{}' has an unexpected shape (GridType {:?}, CollectionType {:?})",
                child.name, child.grid_type, child.collection_type
            ),
        }),
    }
}

fn collection_children<'a>(grid: &'a Grid, path: &GridPath) -> Result<Vec<(&'a Grid, GridPath)>> {
    let children: Vec<(&Grid, GridPath)> = grid
        .grids
        .iter()
        .flatten()
        .enumerate()
        .map(|(index, child)| (child, path.child(index)))
        .collect();

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
