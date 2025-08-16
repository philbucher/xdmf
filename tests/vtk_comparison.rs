#[cfg(test)]
mod tests {
    use std::{
        io::Write,
        path::{Path, PathBuf},
        time::{Duration, Instant},
    };

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
        XdmfAscii,
        XdmfAsciiInline,
        XdmfH5Single,
        XdmfH5Multiple,
        VtkAscii,
        VtkBinary,
        VtkXmlUncompressed,
        VtkXmlCompressedZlib1,
        VtkXmlCompressedZlib5,
        VtkXmlCompressedLZ4, // compression level not yet supported
        VtkXmlCompressedLZMA1,
        VtkXmlCompressedLZMA5,
    }

    trait Case {
        fn name(&self) -> String {
            format!("{:?}", self.output_type())
        }

        fn output_type(&self) -> OutputType;

        fn write_mesh(&mut self, coordinates: Vec<f64>, connectivity: Vec<usize>);

        fn write_step(&mut self, time: f64, data: &[f64]);

        fn get_results(&self) -> Results;
    }

    struct XdmfCase {
        output_type: OutputType,
        file_name: PathBuf,
        wdir: PathBuf,
        writer: Option<xdmf::TimeSeriesDataWriter>,
        time_write_mesh: Duration,
        time_write_steps: Duration,
    }

    impl XdmfCase {
        fn new(output_type: OutputType, base_path: &Path) -> Self {
            Self {
                output_type,
                file_name: base_path.join(format!("{output_type:?}/mesh.xdmf")),
                wdir: base_path.join(format!("{output_type:?}")),
                writer: None,
                time_write_mesh: Duration::ZERO,
                time_write_steps: Duration::ZERO,
            }
        }
    }

    impl Case for XdmfCase {
        fn output_type(&self) -> OutputType {
            self.output_type
        }

        fn write_mesh(&mut self, coordinates: Vec<f64>, connectivity: Vec<usize>) {
            let start = Instant::now();

            let writer = match self.output_type {
                OutputType::XdmfAscii => {
                    TimeSeriesWriter::new(&self.file_name, xdmf::DataStorage::Ascii).unwrap()
                }
                OutputType::XdmfAsciiInline => {
                    TimeSeriesWriter::new(&self.file_name, xdmf::DataStorage::AsciiInline).unwrap()
                }
                OutputType::XdmfH5Single => {
                    TimeSeriesWriter::new(&self.file_name, xdmf::DataStorage::Hdf5SingleFile)
                        .unwrap()
                }
                OutputType::XdmfH5Multiple => {
                    TimeSeriesWriter::new(&self.file_name, xdmf::DataStorage::Hdf5MultipleFiles)
                        .unwrap()
                }
                _ => {
                    panic!("Unsupported XDMF type: {:?}", self.output_type);
                }
            };

            let cell_types = vec![xdmf::CellType::Hexahedron; connectivity.len() / 8];
            let conn: Vec<u64> = connectivity.iter().map(|&x| x as u64).collect();

            let writer = writer
                .write_mesh(&coordinates, (&conn, &cell_types))
                .unwrap();

            self.writer = Some(writer);

            self.time_write_mesh = start.elapsed();
        }

        fn write_step(&mut self, time: f64, data: &[f64]) {
            let start = Instant::now();

            let point_data = vec![(
                "pressure".to_string(),
                (xdmf::DataAttribute::Scalar, data.to_vec().into()),
            )]
            .into_iter()
            .collect();

            self.writer
                .as_mut()
                .unwrap()
                .write_data(format!("{time}").as_str(), Some(&point_data), None)
                .unwrap();

            // Implement step writing logic here
            self.time_write_steps += start.elapsed();
        }

        fn get_results(&self) -> Results {
            Results {
                output_type: self.output_type,
                time_write_mesh: Some(self.time_write_mesh),
                time_write_steps: self.time_write_steps,
                folder: self.wdir.clone(),
            }
        }
    }

    struct VtkCase {
        output_type: OutputType,
        wdir: PathBuf,
        time_write_steps: Duration,
        coordinates: Vec<f64>,
        connectivity: Vec<usize>,
    }

    impl VtkCase {
        fn new(output_type: OutputType, base_path: &Path) -> Self {
            let wdir = base_path.join(format!("{output_type:?}"));
            std::fs::create_dir_all(&wdir).unwrap();
            Self {
                output_type,
                wdir,
                time_write_steps: Duration::ZERO,
                coordinates: Vec::new(),
                connectivity: Vec::new(),
            }
        }
    }

    impl Case for VtkCase {
        fn output_type(&self) -> OutputType {
            self.output_type
        }

        fn write_mesh(&mut self, coordinates: Vec<f64>, connectivity: Vec<usize>) {
            // saving for writing later
            self.coordinates = coordinates;
            self.connectivity = connectivity;
        }

        fn write_step(&mut self, time: f64, data: &[f64]) {
            let start = Instant::now();

            let vtk = create_vtk(
                self.coordinates.clone(),
                self.connectivity.clone(),
                data, // Placeholder for pressure
                      // 0,    // Placeholder for velocity
            );

            let out_file = self.wdir.join(format!("output_{time}.vtk"));

            match self.output_type {
                OutputType::VtkAscii => {
                    vtk.export_ascii(&out_file).unwrap();
                }
                OutputType::VtkBinary => {
                    vtk.export_be(&out_file).unwrap();
                }
                OutputType::VtkXmlUncompressed => {
                    std::fs::write(
                        &out_file,
                        vtk.try_into_xml_format(vtkio::xml::Compressor::None, 0)
                            .unwrap()
                            .to_string(),
                    )
                    .unwrap();
                }
                OutputType::VtkXmlCompressedZlib5 => {
                    std::fs::write(
                        &out_file,
                        vtk.try_into_xml_format(vtkio::xml::Compressor::ZLib, 5)
                            .unwrap()
                            .to_string(),
                    )
                    .unwrap();
                }
                OutputType::VtkXmlCompressedZlib1 => {
                    std::fs::write(
                        &out_file,
                        vtk.try_into_xml_format(vtkio::xml::Compressor::ZLib, 1)
                            .unwrap()
                            .to_string(),
                    )
                    .unwrap();
                }
                OutputType::VtkXmlCompressedLZ4 => {
                    std::fs::write(
                        &out_file,
                        vtk.try_into_xml_format(vtkio::xml::Compressor::LZ4, 1)
                            .unwrap()
                            .to_string(),
                    )
                    .unwrap();
                }
                OutputType::VtkXmlCompressedLZMA1 => {
                    std::fs::write(
                        &out_file,
                        vtk.try_into_xml_format(vtkio::xml::Compressor::LZMA, 1)
                            .unwrap()
                            .to_string(),
                    )
                    .unwrap();
                }
                OutputType::VtkXmlCompressedLZMA5 => {
                    std::fs::write(
                        &out_file,
                        vtk.try_into_xml_format(vtkio::xml::Compressor::LZMA, 5)
                            .unwrap()
                            .to_string(),
                    )
                    .unwrap();
                }
                _ => {
                    panic!("Unsupported VTK type: {:?}", self.output_type);
                }
            }

            self.time_write_steps += start.elapsed();
        }

        fn get_results(&self) -> Results {
            Results {
                output_type: self.output_type,
                time_write_mesh: None,
                time_write_steps: self.time_write_steps,
                folder: self.wdir.clone(),
            }
        }
    }

    struct Results {
        output_type: OutputType,
        time_write_mesh: Option<Duration>,
        time_write_steps: Duration,
        folder: PathBuf,
    }

    impl std::fmt::Display for Results {
        #[expect(clippy::use_debug, reason = "Ignoring clippy in tests")]
        #[expect(clippy::unwrap_in_result, reason = "Ignoring clippy in tests")]
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            // compute size of folder
            write!(
                f,
                "Output Type: {:?}, Total Time: {:?}, Mesh Write Time: {:#?}, Step Write Time: {:?}, Size: {}",
                self.output_type,
                self.time_write_steps + self.time_write_mesh.unwrap_or(Duration::ZERO),
                self.time_write_mesh,
                self.time_write_steps,
                humansize::format_size(
                    fs_extra::dir::get_size(&self.folder).unwrap(),
                    humansize::DECIMAL
                )
            )
        }
    }

    #[test]
    #[expect(clippy::print_stdout, reason = "Ignoring clippy in tests")]
    fn compare_xdmf_to_vtk_formats() {
        const NUM_STEPS: usize = 3;
        const NUM_NODES_X: usize = 10;
        const NUM_NODES_Y: usize = 10;

        const NUM_NODES_Z: usize = 10;

        println!(
            "Running xdmf_vtk_comparison test with {NUM_STEPS} steps, {NUM_NODES_X}/{NUM_NODES_Y}/{NUM_NODES_Z} nodes in X/Y/Z\n"
        );

        let base_path = Path::new("tests/xdmf_vtk_comparison");
        if base_path.exists() {
            std::fs::remove_dir_all(base_path).unwrap();
        }
        std::fs::create_dir_all(base_path).unwrap();

        let mut cases: Vec<Box<dyn Case>> = vec![
            Box::new(XdmfCase::new(OutputType::XdmfAscii, base_path)),
            Box::new(XdmfCase::new(OutputType::XdmfAsciiInline, base_path)),
            Box::new(VtkCase::new(OutputType::VtkAscii, base_path)),
            Box::new(VtkCase::new(OutputType::VtkBinary, base_path)),
            Box::new(VtkCase::new(OutputType::VtkXmlUncompressed, base_path)),
            Box::new(VtkCase::new(OutputType::VtkXmlCompressedZlib1, base_path)),
            Box::new(VtkCase::new(OutputType::VtkXmlCompressedZlib5, base_path)),
            Box::new(VtkCase::new(OutputType::VtkXmlCompressedLZ4, base_path)),
            Box::new(VtkCase::new(OutputType::VtkXmlCompressedLZMA1, base_path)),
            Box::new(VtkCase::new(OutputType::VtkXmlCompressedLZMA5, base_path)),
        ];

        if xdmf::is_hdf5_enabled() {
            cases.push(Box::new(XdmfCase::new(OutputType::XdmfH5Single, base_path)));
            cases.push(Box::new(XdmfCase::new(
                OutputType::XdmfH5Multiple,
                base_path,
            )));
        } else {
            println!("HDF5 feature is not enabled, skipping HDF5 cases.");
        }

        for case in &mut cases {
            println!("\nRunning case {} ...", case.name());

            let (coords, connectivity) = create_mesh(NUM_NODES_X, NUM_NODES_Y, NUM_NODES_Z);
            case.write_mesh(coords, connectivity);

            for step in 0..NUM_STEPS {
                print!(".");
                std::io::stdout().flush().unwrap(); // Forces print to appear immediately

                let time = step as f64 * 0.1;
                let data: Vec<f64> = (0..NUM_NODES_X * NUM_NODES_Y * NUM_NODES_Z)
                    .map(|i| (i as f64 + time) / 1000.0)
                    .collect();
                case.write_step(time, &data);
            }
            println!();
        }
        println!();

        for case in &cases {
            let results = case.get_results();
            println!("{results}");
        }
    }

    fn create_vtk(coordinates: Vec<f64>, connectivity: Vec<usize>, pressure: &[f64]) -> Vtk {
        let vertices: Vec<u32> = connectivity
            .chunks(8)
            .flat_map(|chunk| {
                let mut chunk_vec = vec![chunk.len() as u32];
                chunk_vec.extend(chunk.iter().map(|&x| x as u32));
                chunk_vec
            })
            .collect();

        let num_cells = connectivity.len() / 8;

        Vtk {
            version: Version { major: 4, minor: 2 },
            byte_order: ByteOrder::native(),
            title: String::from("vtk output"),
            file_path: None,
            data: DataSet::inline(UnstructuredGridPiece {
                points: IOBuffer::F64(coordinates),
                cells: Cells {
                    cell_verts: VertexNumbers::Legacy {
                        num_cells: num_cells as u32,
                        vertices,
                    },
                    types: vec![CellType::Hexahedron; num_cells],
                },
                data: Attributes {
                    point: vec![
                        Attribute::Field {
                            name: String::from("FieldData"),
                            data_array: vec![FieldArray {
                                name: String::from("Pressure"),
                                elem: 1,
                                data: IOBuffer::F64(pressure.to_vec()),
                            }],
                        },
                        // Attribute::Field {
                        //     name: String::from("FieldData"),
                        //     data_array: vec![FieldArray {
                        //         name: String::from("Velocity"),
                        //         elem: 1,
                        //         data: is_ghost,
                        //     }],
                        // },
                    ],
                    cell: vec![],
                },
            }),
        }
    }
}
