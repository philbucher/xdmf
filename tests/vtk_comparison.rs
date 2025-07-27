use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use vtkio::model::*;
use xdmf::TimeSeriesWriter;

fn create_mesh(
    num_nodes_x: usize,
    num_nodes_y: usize,
    num_nodes_z: usize,
) -> (Vec<f64>, Vec<usize>) {
    let min_x = 0.0;
    let max_x = 1.0 * num_nodes_x as f64;
    let min_y = 0.0;
    let max_y = 1.0 * num_nodes_y as f64;
    let min_z = 0.0;
    let max_z = 1.0 * num_nodes_z as f64;

    let mut coords = Vec::with_capacity(num_nodes_x * num_nodes_y * num_nodes_z);
    for i in 0..num_nodes_x {
        for j in 0..num_nodes_y {
            for k in 0..num_nodes_z {
                let x = min_x + i as f64 * (max_x - min_x) / (num_nodes_x - 1) as f64;
                let y = min_y + j as f64 * (max_y - min_y) / (num_nodes_y - 1) as f64;
                let z = min_z + k as f64 * (max_z - min_z) / (num_nodes_z - 1) as f64;
                coords.extend([x, y, z]);
            }
        }
    }

    let mut connectivities_hexa =
        Vec::with_capacity((num_nodes_x - 1) * (num_nodes_y - 1) * (num_nodes_z - 1));
    for i in 0..num_nodes_x - 1 {
        for j in 0..num_nodes_y - 1 {
            for k in 0..num_nodes_z - 1 {
                let node_index = |i: usize, j: usize, k: usize| {
                    i * num_nodes_y * num_nodes_z + j * num_nodes_z + k
                };
                connectivities_hexa.extend(vec![
                    node_index(i, j, k),
                    node_index(i + 1, j, k),
                    node_index(i + 1, j + 1, k),
                    node_index(i, j + 1, k),
                    node_index(i, j, k + 1),
                    node_index(i + 1, j, k + 1),
                    node_index(i + 1, j + 1, k + 1),
                    node_index(i, j + 1, k + 1),
                ]);
            }
        }
    }

    (coords, connectivities_hexa)
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum OutputType {
    XdmfXml,
    XdmfH5Single,
    XdmfH5Multiple,
    VtkAscii,
    VtkBinary,
    VtkXmlUncompressed,
    VtkXmlCompressed,
}

trait Case {
    fn write_mesh(
        &mut self,
        coordinates: Vec<f64>,
        connectivity: Vec<usize>,
    ) -> std::io::Result<()>;

    fn write_step(&mut self, time: f64, data: &[f64]) -> std::io::Result<()>;

    fn get_results(&self) -> Results;
}

struct XdmfCase {
    output_type: OutputType,
    writer: Option<TimeSeriesWriter>,
    time_write_mesh: Duration,
    time_write_steps: Duration,
}

impl XdmfCase {
    fn new(output_type: OutputType, base_path: &Path) -> Self {
        Self {
            output_type,
            writer: None,
            time_write_mesh: Duration::ZERO,
            time_write_steps: Duration::ZERO,
        }
    }
}

impl Case for XdmfCase {
    fn write_mesh(
        &mut self,
        coordinates: Vec<f64>,
        connectivity: Vec<usize>,
    ) -> std::io::Result<()> {
        let start = Instant::now();
        // Implement mesh writing logic here
        self.time_write_mesh = start.elapsed();
        Ok(())
    }

    fn write_step(&mut self, time: f64, data: &[f64]) -> std::io::Result<()> {
        let start = Instant::now();
        // Implement step writing logic here
        self.time_write_steps += start.elapsed();
        Ok(())
    }

    fn get_results(&self) -> Results {
        Results {
            output_type: self.output_type,
            time_write_mesh: Some(self.time_write_mesh),
            time_write_steps: self.time_write_steps,
            size: 0, // Placeholder for size calculation
        }
    }
}

struct VtkCase {
    output_type: OutputType,
    time_write_steps: Duration,
    coordinates: Vec<f64>,
    connectivity: Vec<usize>,
}

impl VtkCase {
    fn new(output_type: OutputType, base_path: &Path) -> Self {
        Self {
            output_type,
            time_write_steps: Duration::ZERO,
            coordinates: Vec::new(),
            connectivity: Vec::new(),
        }
    }
}

impl Case for VtkCase {
    fn write_mesh(
        &mut self,
        coordinates: Vec<f64>,
        connectivity: Vec<usize>,
    ) -> std::io::Result<()> {
        // saving for writing later
        self.coordinates = coordinates;
        self.connectivity = connectivity;
        Ok(())
    }

    fn write_step(&mut self, time: f64, data: &[f64]) -> std::io::Result<()> {
        let start = Instant::now();

        let vtk = create_vtk(
            self.coordinates.clone(),
            self.connectivity.clone(),
            0, // Placeholder for pressure
            0, // Placeholder for velocity
        );

        match self.output_type {
            OutputType::VtkAscii => {
                let path = PathBuf::from(format!("output_{}.vtk", time));
                vtk.export_ascii(&path).map_err(std::io::Error::from)?;
            }
            OutputType::VtkBinary => {
                let path = PathBuf::from(format!("output_{}.vtu", time));
                vtk.export_be(&path).map_err(std::io::Error::from)?;
            }
            OutputType::VtkXmlUncompressed => {
                let path = PathBuf::from(format!("output_{}.vtu", time));
                vtk.try_into_xml_format(vtkio::xml::Compressor::None, 0)
                    .unwrap()
                    .export(&path)
                    .unwrap();
            }
            OutputType::VtkXmlCompressed => {
                let path = PathBuf::from(format!("output_{}.vtu.gz", time));
                vtk.try_into_xml_format(vtkio::xml::Compressor::ZLib, 5)
                    .unwrap()
                    .export(&path)
                    .unwrap();
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Unsupported VTK type",
                ));
            }
        }

        Ok(())
    }

    fn get_results(&self) -> Results {
        Results {
            output_type: self.output_type,
            time_write_mesh: None,
            time_write_steps: self.time_write_steps,
            size: 0, // Placeholder for size calculation
        }
    }
}

struct Results {
    output_type: OutputType,
    time_write_mesh: Option<Duration>,
    time_write_steps: Duration,
    size: usize,
}

impl std::fmt::Display for Results {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Output Type: {:?}, Mesh Write Time: {:?}, Step Write Time: {:?}, Size: {}",
            self.output_type, self.time_write_mesh, self.time_write_steps, self.size
        )
    }
}

#[test]
fn compare_xdmf_to_vtk_formats() {
    const NUM_STEPS: usize = 1000;
    const NUM_NODES_X: usize = 10;
    const NUM_NODES_Y: usize = 10;
    const NUM_NODES_Z: usize = 1000;

    println!(
        "Running xdmf_vtk_comparison test with {} steps, {} nodes in x, {} nodes in y, {} nodes in z",
        NUM_STEPS, NUM_NODES_X, NUM_NODES_Y, NUM_NODES_Z
    );

    let (coords, connectivity) = create_mesh(NUM_NODES_X, NUM_NODES_Y, NUM_NODES_Z);

    let base_path = Path::new("tests/xdmf_vtk_comparison");
    if base_path.exists() {
        std::fs::remove_dir_all(base_path).unwrap();
    }
    std::fs::create_dir_all(base_path).unwrap();

    let mut cases: Vec<Box<dyn Case>> = vec![];

    cases.push(Box::new(XdmfCase::new(OutputType::XdmfXml, &base_path)));
    cases.push(Box::new(XdmfCase::new(
        OutputType::XdmfH5Single,
        &base_path,
    )));
    cases.push(Box::new(XdmfCase::new(
        OutputType::XdmfH5Multiple,
        &base_path,
    )));
    cases.push(Box::new(VtkCase::new(OutputType::VtkAscii, &base_path)));
    cases.push(Box::new(VtkCase::new(OutputType::VtkBinary, &base_path)));
    cases.push(Box::new(VtkCase::new(
        OutputType::VtkXmlUncompressed,
        &base_path,
    )));
    cases.push(Box::new(VtkCase::new(
        OutputType::VtkXmlCompressed,
        &base_path,
    )));

    for case in &mut cases {
        let (coords, connectivity) = create_mesh(NUM_NODES_X, NUM_NODES_Y, NUM_NODES_Z);
        case.write_mesh(coords, connectivity).unwrap();

        for step in 0..NUM_STEPS {
            let time = step as f64 * 0.1;
            let data: Vec<f64> = (0..coords.len()).map(|_| rand::random::<f64>()).collect();
            case.write_step(time, &data).unwrap();
        }
    }

    for case in &cases {
        let results = case.get_results();
        println!("{}", results);
    }
}

fn create_vtk(
    coordinates: Vec<f64>,
    connectivity: Vec<usize>,
    pressure: i32,
    velocity: i32,
) -> Vtk {
    let vertices = mesh
        .geometries()
        .iter()
        .flat_map(|geometry| {
            let nodes = geometry
                .nodes()
                .iter()
                .map(|node| node.lock().index() as u32)
                .collect::<Vec<u32>>();

            // Prepend the number of nodes to the connectivity list
            let mut result = vec![geometry.num_nodes() as u32];
            result.extend(nodes);
            result
        })
        .collect();

    Vtk {
        version: Version::new(),
        byte_order: ByteOrder::BigEndian,
        title: String::from("vtk output"),
        file_path: None,
        data: DataSet::inline(UnstructuredGridPiece {
            points: IOBuffer::F64(coordinates),
            cells: Cells {
                cell_verts: VertexNumbers::Legacy {
                    num_cells: mesh.num_geometries() as u32,
                    vertices,
                },
                types: cell_types,
            },
            data: Attributes {
                point: vec![
                    Attribute::Field {
                        name: String::from("FieldData"),
                        data_array: vec![FieldArray {
                            name: String::from("Pressure"),
                            elem: 1,
                            data: point_indices,
                        }],
                    },
                    Attribute::Field {
                        name: String::from("FieldData"),
                        data_array: vec![FieldArray {
                            name: String::from("Velocity"),
                            elem: 1,
                            data: is_ghost,
                        }],
                    },
                ],
                cell: vec![Attribute::Field {
                    name: String::from("FieldData"),
                    data_array: vec![FieldArray {
                        name: String::from("Index"),
                        elem: 1,
                        data: cell_indices,
                    }],
                }],
            },
        }),
    }
}
