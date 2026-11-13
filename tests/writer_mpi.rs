//! End-to-end coverage for [`xdmf::mpi`]: every rank writes its own share of a mesh, then rank 0
//! writes the *same mesh* serially, in one shot, and checks the two outputs are the same files --
//! a stronger oracle than hand-computed expected values, since it cannot share a bug with the
//! implementation the way a hand-derived expectation can.
//!
//! Run under MPI by `mpi_test`, which launches each test with `mpirun` at every listed `np`.

#![cfg(feature = "mpi")]

// in a `cfg(test)` module, which is what lets `.clippy.toml`'s test exemptions (`unwrap`, `panic`)
// reach the helpers, not only the `#[mpi_test]` functions themselves
#[cfg(test)]
mod mpi_tests {
    use std::{
        collections::BTreeMap,
        fs,
        path::{Path, PathBuf},
    };

    use ::mpi::{
        topology::SimpleCommunicator,
        traits::{Communicator, CommunicatorCollectives, Root},
    };
    use mpi_test::mpi_test;
    use temp_dir::TempDir;
    use xdmf::{CellType, DataAttribute, DataStorage, Error, TimeSeriesReader, TimeSeriesWriter};

    const STORAGES: [DataStorage; 5] = [
        DataStorage::Hdf5SingleFile {
            deflate_level: None,
        },
        DataStorage::Hdf5MultipleFiles {
            deflate_level: None,
        },
        DataStorage::Ascii,
        DataStorage::AsciiInline,
        DataStorage::Binary,
    ];

    const TIMES: [&str; 2] = ["0.0", "0.5"];

    /// A whole mesh, the same on every rank: a chain of points along x, joined by `Edge`s and, if
    /// `mixed`, by a `Triangle` after every third edge.
    struct GlobalMesh {
        points: Vec<f64>,
        connectivity: Vec<u32>,
        cell_types: Vec<CellType>,
        /// Where each cell starts in `connectivity`.
        cell_starts: Vec<usize>,
    }

    impl GlobalMesh {
        fn new(num_points: usize, mixed: bool) -> Self {
            let points = (0..num_points)
                .flat_map(|id| [id as f64, (id % 3) as f64, 0.0])
                .collect();

            let mut mesh = Self {
                points,
                connectivity: Vec::new(),
                cell_types: Vec::new(),
                cell_starts: Vec::new(),
            };

            for i in 0..num_points - 1 {
                mesh.push(CellType::Edge, &[i, i + 1]);
                if mixed && i % 3 == 0 && i + 2 < num_points {
                    mesh.push(CellType::Triangle, &[i, i + 1, i + 2]);
                }
            }

            mesh
        }

        fn push(&mut self, cell_type: CellType, points: &[usize]) {
            self.cell_starts.push(self.connectivity.len());
            self.cell_types.push(cell_type);
            self.connectivity
                .extend(points.iter().map(|&point| point as u32));
        }

        fn num_points(&self) -> usize {
            self.points.len() / 3
        }

        fn num_cells(&self) -> usize {
            self.cell_types.len()
        }
    }

    /// The fields of one step, for any subset of points and cells -- so a rank's share and the whole
    /// mesh's come from the same formula.
    fn temperature(points: &[usize], step: usize) -> Vec<f64> {
        points
            .iter()
            .map(|&id| id as f64 * 10.0 + step as f64)
            .collect()
    }

    fn velocity(points: &[usize], step: usize) -> Vec<f32> {
        points
            .iter()
            .flat_map(|&id| [id as f32, -(id as f32), step as f32])
            .collect()
    }

    fn cell_index(cells: &[usize]) -> Vec<i32> {
        cells.iter().map(|&cell| cell as i32).collect()
    }

    fn stress(cells: &[usize], step: usize) -> Vec<f64> {
        cells
            .iter()
            .flat_map(|&cell| (0..6).map(move |k| (cell * 6 + k + step) as f64 * 0.25))
            .collect()
    }

    /// How the whole mesh is split over the ranks: per rank, the first point id it owns and how
    /// many, and how many cells (taken in global order, rank after rank).
    struct Partition {
        point_blocks: Vec<(usize, usize)>,
        cell_counts: Vec<usize>,
    }

    impl Partition {
        /// Points and cells as evenly as possible, in rank order.
        fn even(size: usize, num_points: usize, num_cells: usize) -> Self {
            let counts = |total: usize| -> Vec<usize> {
                (0..size)
                    .map(|r| total / size + usize::from(r < total % size))
                    .collect()
            };

            let mut start = 0;
            let point_blocks = counts(num_points)
                .into_iter()
                .map(|len| {
                    let block = (start, len);
                    start += len;
                    block
                })
                .collect();

            Self {
                point_blocks,
                cell_counts: counts(num_cells),
            }
        }

        /// The last rank owns no points and the first owns no cells -- every rank still holds
        /// something, but no rank holds both.
        fn with_empty_shares(size: usize, num_points: usize, num_cells: usize) -> Self {
            if size == 1 {
                return Self::even(size, num_points, num_cells);
            }

            let mut partition = Self::even(size - 1, num_points, num_cells);
            partition.point_blocks.push((num_points, 0));
            partition.cell_counts.insert(0, 0);
            partition
        }

        /// The point blocks in reverse rank order: the last rank owns ids `0..`.
        fn reversed(size: usize, num_points: usize, num_cells: usize) -> Self {
            let mut partition = Self::even(size, num_points, num_cells);
            let mut lens: Vec<usize> = partition.point_blocks.iter().map(|b| b.1).collect();
            lens.reverse();

            let mut start = 0;
            let mut blocks: Vec<(usize, usize)> = lens
                .into_iter()
                .map(|len| {
                    let block = (start, len);
                    start += len;
                    block
                })
                .collect();
            blocks.reverse();
            partition.point_blocks = blocks;
            partition
        }

        fn points_of(&self, rank: usize) -> Vec<usize> {
            let (start, len) = self.point_blocks[rank];
            (start..start + len).collect()
        }

        fn cells_of(&self, rank: usize) -> Vec<usize> {
            let start: usize = self.cell_counts[..rank].iter().sum();
            (start..start + self.cell_counts[rank]).collect()
        }
    }

    /// One rank's share of a [`GlobalMesh`] under a [`Partition`].
    struct LocalShare {
        points: Vec<usize>,
        cells: Vec<usize>,
        coordinates: Vec<f64>,
        global_ids: Vec<u64>,
        connectivity: Vec<u32>,
        cell_types: Vec<CellType>,
    }

    impl LocalShare {
        fn new(mesh: &GlobalMesh, partition: &Partition, rank: usize) -> Self {
            let points = partition.points_of(rank);
            let cells = partition.cells_of(rank);

            let coordinates = points
                .iter()
                .flat_map(|&id| mesh.points[id * 3..id * 3 + 3].to_vec())
                .collect();
            let connectivity = cells
                .iter()
                .flat_map(|&cell| {
                    let start = mesh.cell_starts[cell];
                    mesh.connectivity[start..start + mesh.cell_types[cell].num_points()].to_vec()
                })
                .collect();

            Self {
                global_ids: points.iter().map(|&id| id as u64).collect(),
                cell_types: cells.iter().map(|&cell| mesh.cell_types[cell]).collect(),
                points,
                cells,
                coordinates,
                connectivity,
            }
        }
    }

    /// Initialize MPI for one test, making a panic on any rank abort the whole job.
    ///
    /// Without this, a failing assertion on one rank unwinds into `MPI_Finalize` while the others
    /// wait in their next collective, and the test hangs instead of failing.
    fn initialize() -> (::mpi::environment::Universe, SimpleCommunicator) {
        let universe = mpi::initialize().unwrap();

        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            default_hook(info);
            SimpleCommunicator::world().abort(1);
        }));

        let world = universe.world();
        (universe, world)
    }

    /// A fresh directory every rank agrees on: rank 0 creates it and broadcasts its path. Only rank
    /// 0's `TempDir` removes it, so it must outlive every rank's use of it.
    fn shared_dir(comm: &SimpleCommunicator) -> (Option<TempDir>, PathBuf) {
        let tmp_dir = (comm.rank() == 0).then(|| TempDir::new().unwrap());

        let mut bytes = tmp_dir
            .as_ref()
            .map(|dir| dir.path().to_str().unwrap().as_bytes().to_vec())
            .unwrap_or_default();

        let root = comm.process_at_rank(0);
        let mut len = bytes.len() as u64;
        root.broadcast_into(&mut len);
        bytes.resize(len as usize, 0);
        root.broadcast_into(&mut bytes[..]);

        (tmp_dir, PathBuf::from(String::from_utf8(bytes).unwrap()))
    }

    /// Write `share` through [`xdmf::mpi`], with every step of [`TIMES`].
    fn write_parallel(
        comm: &SimpleCommunicator,
        file_name: &Path,
        storage: DataStorage,
        share: &LocalShare,
    ) {
        let writer = xdmf::mpi::TimeSeriesWriter::new(file_name, storage, comm).unwrap();
        let mut writer = writer
            .write_mesh(
                &share.coordinates,
                &share.global_ids,
                &share.connectivity,
                &share.cell_types,
            )
            .unwrap();

        for (step, time) in TIMES.iter().enumerate() {
            writer
                .write_time_step(*time, |data| {
                    data.point_data(
                        "temperature",
                        DataAttribute::Scalar,
                        temperature(&share.points, step),
                    )?;
                    data.point_data(
                        "velocity",
                        DataAttribute::Vector,
                        velocity(&share.points, step),
                    )?;
                    data.cell_data(
                        "cell_index",
                        DataAttribute::Scalar,
                        cell_index(&share.cells),
                    )?;
                    data.cell_data("stress", DataAttribute::Tensor6, stress(&share.cells, step))
                })
                .unwrap();
        }
    }

    /// Write the whole `mesh` through the serial writer, with the same steps as [`write_parallel`].
    fn write_serial(file_name: &Path, storage: DataStorage, mesh: &GlobalMesh) {
        let points: Vec<usize> = (0..mesh.num_points()).collect();
        let cells: Vec<usize> = (0..mesh.num_cells()).collect();

        let writer = TimeSeriesWriter::new(file_name, storage).unwrap();
        let mut writer = writer
            .write_mesh(&mesh.points, &mesh.connectivity, &mesh.cell_types)
            .unwrap();

        for (step, time) in TIMES.iter().enumerate() {
            writer
                .write_time_step(*time, |data| {
                    data.point_data(
                        "temperature",
                        DataAttribute::Scalar,
                        temperature(&points, step),
                    )?;
                    data.point_data("velocity", DataAttribute::Vector, velocity(&points, step))?;
                    data.cell_data("cell_index", DataAttribute::Scalar, cell_index(&cells))?;
                    data.cell_data("stress", DataAttribute::Tensor6, stress(&cells, step))
                })
                .unwrap();
        }
    }

    /// Every file under `dir`, by its path relative to `dir`.
    fn files_under(dir: &Path) -> BTreeMap<PathBuf, PathBuf> {
        let mut files = BTreeMap::new();
        let mut pending = vec![dir.to_path_buf()];

        while let Some(current) = pending.pop() {
            for entry in fs::read_dir(&current).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    pending.push(path);
                } else {
                    files.insert(path.strip_prefix(dir).unwrap().to_path_buf(), path);
                }
            }
        }

        files
    }

    /// Every dataset of an HDF5 file, by its path inside the file.
    fn datasets(file: &hdf5::File) -> BTreeMap<String, hdf5::Dataset> {
        let mut datasets = BTreeMap::new();
        let mut pending = vec![file.as_group().unwrap()];

        while let Some(group) = pending.pop() {
            for dataset in group.datasets().unwrap() {
                datasets.insert(dataset.name(), dataset);
            }
            pending.extend(group.groups().unwrap());
        }

        datasets
    }

    /// The two directories hold the same files: the light data and the ascii/binary heavy data
    /// byte for byte, the HDF5 heavy data dataset by dataset (an HDF5 file also records when each
    /// object was written, so it is never the same twice).
    fn assert_same_files(parallel: &Path, serial: &Path, context: &str) {
        let parallel_files = files_under(parallel);
        let serial_files = files_under(serial);

        assert_eq!(
            parallel_files.keys().collect::<Vec<_>>(),
            serial_files.keys().collect::<Vec<_>>(),
            "{context}"
        );

        for (name, parallel_file) in &parallel_files {
            let serial_file = &serial_files[name];

            if name.extension().is_some_and(|ext| ext == "h5") {
                let parallel_h5 = hdf5::File::open(parallel_file).unwrap();
                let serial_h5 = hdf5::File::open(serial_file).unwrap();
                let parallel_sets = datasets(&parallel_h5);
                let serial_sets = datasets(&serial_h5);

                assert_eq!(
                    parallel_sets.keys().collect::<Vec<_>>(),
                    serial_sets.keys().collect::<Vec<_>>(),
                    "{context}: {name:?}"
                );

                for (path, parallel_set) in &parallel_sets {
                    let serial_set = &serial_sets[path];
                    let what = format!("{context}: {name:?}:{path}");

                    assert_eq!(
                        parallel_set.dtype().unwrap().to_descriptor().unwrap(),
                        serial_set.dtype().unwrap().to_descriptor().unwrap(),
                        "{what}"
                    );
                    assert_eq!(parallel_set.shape(), serial_set.shape(), "{what}");
                    assert_eq!(parallel_set.chunk(), serial_set.chunk(), "{what}");
                    assert_eq!(parallel_set.filters(), serial_set.filters(), "{what}");
                    // read through `f64`, which every element type written here converts to exactly
                    assert_eq!(
                        parallel_set.read_raw::<f64>().unwrap(),
                        serial_set.read_raw::<f64>().unwrap(),
                        "{what}"
                    );
                }
            } else {
                assert_eq!(
                    fs::read(parallel_file).unwrap(),
                    fs::read(serial_file).unwrap(),
                    "{context}: {name:?}"
                );
            }
        }
    }

    /// How to split a mesh of so many points and cells over so many ranks.
    type PartitionFn = fn(usize, usize, usize) -> Partition;

    /// Every storage, partition and topology kind: write in parallel, then serially, and compare.
    #[mpi_test(np = [1, 2, 3])]
    fn parallel_output_matches_serial_output() {
        let (_universe, world) = initialize();
        let rank = world.rank() as usize;
        let size = world.size() as usize;

        let (_tmp_dir, dir) = shared_dir(&world);

        let partitions: [(&str, PartitionFn); 3] = [
            ("even", Partition::even),
            ("with_empty_shares", Partition::with_empty_shares),
            ("reversed", Partition::reversed),
        ];

        for storage in STORAGES {
            for (partition_name, partition) in partitions {
                for mixed in [false, true] {
                    let context =
                        format!("{storage:?}, {partition_name}, mixed: {mixed}, np {size}");
                    let case = dir.join(format!(
                        "{}_{partition_name}_{mixed}",
                        format!("{storage:?}").split(' ').next().unwrap()
                    ));

                    let mesh = GlobalMesh::new(5 * size + 2, mixed);
                    let partition = partition(size, mesh.num_points(), mesh.num_cells());
                    let share = LocalShare::new(&mesh, &partition, rank);

                    write_parallel(&world, &case.join("parallel/mesh"), storage, &share);

                    world.barrier();

                    if rank == 0 {
                        write_serial(&case.join("serial/mesh"), storage, &mesh);
                        assert_same_files(&case.join("parallel"), &case.join("serial"), &context);
                    }

                    world.barrier();
                }
            }
        }
    }

    /// Large enough for several compressed chunks, split so that no rank boundary falls on a chunk
    /// boundary: parallel HDF5 has to merge each shared chunk from every rank writing into it.
    #[mpi_test(np = [1, 3])]
    fn compressed_collective_writes_match_serial_ones() {
        let (_universe, world) = initialize();
        let rank = world.rank() as usize;
        let size = world.size() as usize;

        let (_tmp_dir, dir) = shared_dir(&world);
        let storage = DataStorage::Hdf5SingleFile {
            deflate_level: Some(6),
        };

        let mesh = GlobalMesh::new(150_001, false);
        let partition = Partition::even(size, mesh.num_points(), mesh.num_cells());
        let share = LocalShare::new(&mesh, &partition, rank);

        write_parallel(&world, &dir.join("parallel/mesh"), storage, &share);

        world.barrier();

        if rank == 0 {
            write_serial(&dir.join("serial/mesh"), storage, &mesh);
            assert_same_files(&dir.join("parallel"), &dir.join("serial"), "compressed");

            let file = hdf5::File::open(dir.join("parallel/mesh.h5")).unwrap();
            let velocity = file.dataset("data/t_0.5/1").unwrap();
            assert!(velocity.chunk().is_some());
            assert!(
                velocity
                    .filters()
                    .contains(&hdf5::filters::Filter::Deflate(6))
            );

            // and the reader agrees, as a last check that the file is a well-formed XDMF one
            let reader = TimeSeriesReader::new(dir.join("parallel/mesh.xdmf2")).unwrap();
            let mut read = Vec::new();
            reader
                .read_point_data::<f64>(1, "temperature", &mut read)
                .unwrap();
            let all: Vec<usize> = (0..mesh.num_points()).collect();
            assert_eq!(read, temperature(&all, 1));
        }

        world.barrier();
    }

    /// Each backend: in place, into one file or one per step, and gathered to rank 0.
    const ERROR_STORAGES: [DataStorage; 3] = [
        DataStorage::Hdf5SingleFile {
            deflate_level: None,
        },
        DataStorage::Hdf5MultipleFiles {
            deflate_level: None,
        },
        DataStorage::Ascii,
    ];

    /// A small mesh, written through [`xdmf::mpi`] into `file_name`, evenly partitioned.
    fn small_writer(
        comm: &SimpleCommunicator,
        file_name: &Path,
        storage: DataStorage,
    ) -> (xdmf::mpi::TimeSeriesDataWriter, LocalShare) {
        let size = comm.size() as usize;
        let mesh = GlobalMesh::new(4 * size, false);
        let partition = Partition::even(size, mesh.num_points(), mesh.num_cells());
        let share = LocalShare::new(&mesh, &partition, comm.rank() as usize);

        let writer = xdmf::mpi::TimeSeriesWriter::new(file_name, storage, comm)
            .unwrap()
            .write_mesh(
                &share.coordinates,
                &share.global_ids,
                &share.connectivity,
                &share.cell_types,
            )
            .unwrap();

        (writer, share)
    }

    /// The times rank 0 can read back out of the file, which is what tells a kept step from a
    /// discarded one. Only the HDF5 storages can be read back; for the others, the light data's
    /// `<Time>` elements are counted instead.
    fn written_times(file_name: &Path, storage: DataStorage) -> Vec<String> {
        if matches!(
            storage,
            DataStorage::Hdf5SingleFile { .. } | DataStorage::Hdf5MultipleFiles { .. }
        ) {
            return TimeSeriesReader::new(file_name).unwrap().times().to_vec();
        }

        let xml = fs::read_to_string(file_name).unwrap();
        xml.split("<Time Value=\"")
            .skip(1)
            .map(|rest| rest.split('"').next().unwrap().to_string())
            .collect()
    }

    #[mpi_test(np = [2, 3])]
    fn ranks_writing_different_times_fail_together() {
        let (_universe, world) = initialize();
        let rank = world.rank();
        let (_tmp_dir, dir) = shared_dir(&world);

        for storage in ERROR_STORAGES {
            let (mut writer, share) = small_writer(
                &world,
                &dir.join(format!("times_{}", storage_tag(storage))),
                storage,
            );
            let time = if rank == 1 { "1.0" } else { "0.0" };

            let result = writer.write_time_step(time, |data| {
                data.point_data(
                    "temperature",
                    DataAttribute::Scalar,
                    temperature(&share.points, 0),
                )
            });
            std::assert_matches!(
                result.unwrap_err(),
                Error::InvalidData { reason } if reason.contains("out of step")
            );

            // the writer is still usable, and the failed call left nothing behind
            writer
                .write_time_step("0.0", |data| {
                    data.point_data(
                        "temperature",
                        DataAttribute::Scalar,
                        temperature(&share.points, 0),
                    )
                })
                .unwrap();

            let file_name = writer.file_name().to_path_buf();
            drop(writer);
            world.barrier();
            if rank == 0 {
                assert_eq!(written_times(&file_name, storage), ["0.0"]);
            }
            world.barrier();
        }
    }

    #[mpi_test(np = [2, 3])]
    fn ranks_writing_different_fields_fail_together() {
        let (_universe, world) = initialize();
        let rank = world.rank();
        let (_tmp_dir, dir) = shared_dir(&world);

        for storage in ERROR_STORAGES {
            let (mut writer, share) = small_writer(
                &world,
                &dir.join(format!("fields_{}", storage_tag(storage))),
                storage,
            );
            let name = if rank == 1 { "pressure" } else { "temperature" };

            let result = writer.write_time_step("0.0", |data| {
                data.point_data(name, DataAttribute::Scalar, temperature(&share.points, 0))
            });
            std::assert_matches!(
                result.unwrap_err(),
                Error::InvalidData { reason } if reason.contains("out of step")
            );

            // a different number of fields: rank 1 writes a second one the others do not
            let result = writer.write_time_step("0.0", |data| {
                data.point_data(
                    "temperature",
                    DataAttribute::Scalar,
                    temperature(&share.points, 0),
                )?;
                if rank == 1 {
                    data.cell_data(
                        "cell_index",
                        DataAttribute::Scalar,
                        cell_index(&share.cells),
                    )?;
                }
                Ok::<(), Error>(())
            });
            assert!(result.is_err());

            writer
                .write_time_step("0.5", |data| {
                    data.point_data(
                        "temperature",
                        DataAttribute::Scalar,
                        temperature(&share.points, 1),
                    )
                })
                .unwrap();

            let file_name = writer.file_name().to_path_buf();
            drop(writer);
            world.barrier();
            if rank == 0 {
                assert_eq!(written_times(&file_name, storage), ["0.5"]);
            }
            world.barrier();
        }
    }

    /// A caller's own error type, to check that a failing closure's error comes back untouched.
    #[derive(Debug)]
    enum CallerError {
        Solver,
        Xdmf(Error),
    }

    impl From<Error> for CallerError {
        fn from(error: Error) -> Self {
            Self::Xdmf(error)
        }
    }

    #[mpi_test(np = [2, 3])]
    fn a_closure_failing_on_one_rank_discards_the_step_on_every_rank() {
        let (_universe, world) = initialize();
        let rank = world.rank();
        let (_tmp_dir, dir) = shared_dir(&world);

        for storage in ERROR_STORAGES {
            let (mut writer, share) = small_writer(
                &world,
                &dir.join(format!("closure_{}", storage_tag(storage))),
                storage,
            );

            // rank 1 fails after the first field, the others go on to write a second one
            let result = writer.write_time_step("0.0", |data| -> Result<(), CallerError> {
                data.point_data(
                    "temperature",
                    DataAttribute::Scalar,
                    temperature(&share.points, 0),
                )?;
                if rank == 1 {
                    return Err(CallerError::Solver);
                }
                data.cell_data(
                    "cell_index",
                    DataAttribute::Scalar,
                    cell_index(&share.cells),
                )?;
                Ok(())
            });

            if rank == 1 {
                std::assert_matches!(result.unwrap_err(), CallerError::Solver);
            } else {
                // the others either saw rank 1 go out of step at their second field, or -- if they
                // got to the end first -- were told the step failed there
                std::assert_matches!(
                    result.unwrap_err(),
                    CallerError::Xdmf(Error::InvalidData { .. } | Error::InvalidTimeStep { .. })
                );
            }

            // failing before writing anything at all
            let result = writer.write_time_step("0.0", |data| -> Result<(), CallerError> {
                if rank == 0 {
                    return Err(CallerError::Solver);
                }
                data.point_data(
                    "temperature",
                    DataAttribute::Scalar,
                    temperature(&share.points, 0),
                )?;
                Ok(())
            });
            if rank == 0 {
                std::assert_matches!(result.unwrap_err(), CallerError::Solver);
            } else {
                assert!(result.is_err());
            }

            // the time is free again, and the step written now is the only one in the file
            writer
                .write_time_step("0.0", |data| {
                    data.point_data(
                        "temperature",
                        DataAttribute::Scalar,
                        temperature(&share.points, 0),
                    )
                })
                .unwrap();

            let file_name = writer.file_name().to_path_buf();
            drop(writer);
            world.barrier();
            if rank == 0 {
                assert_eq!(written_times(&file_name, storage), ["0.0"]);
            }
            world.barrier();
        }
    }

    #[mpi_test(np = [2, 3])]
    fn a_wrong_field_size_on_one_rank_fails_every_rank() {
        let (_universe, world) = initialize();
        let rank = world.rank();
        let (_tmp_dir, dir) = shared_dir(&world);

        for storage in ERROR_STORAGES {
            let (mut writer, share) = small_writer(
                &world,
                &dir.join(format!("size_{}", storage_tag(storage))),
                storage,
            );

            let mut values = temperature(&share.points, 0);
            if rank == 1 {
                values.pop();
            }

            let result = writer.write_time_step("0.0", |data| {
                data.point_data("temperature", DataAttribute::Scalar, &values)
            });

            if rank == 1 {
                std::assert_matches!(
                    result.unwrap_err(),
                    Error::InvalidData { reason } if reason.starts_with("size of point_data")
                );
            } else {
                std::assert_matches!(
                    result.unwrap_err(),
                    Error::InvalidData { reason } if reason.starts_with("on rank 1: size of point_data")
                );
            }
        }
    }

    fn storage_tag(storage: DataStorage) -> String {
        format!("{storage:?}")
            .split(' ')
            .next()
            .unwrap()
            .to_string()
    }

    /// `write_mesh` with this rank's points given as `ids` (coordinates made up to match), and no
    /// cells but one edge on rank 0 so the mesh is not empty of cells.
    fn write_mesh_with_ids(
        comm: &SimpleCommunicator,
        file_name: &Path,
        storage: DataStorage,
        ids: &[u64],
        connectivity: &[u64],
    ) -> xdmf::Result<xdmf::mpi::TimeSeriesDataWriter> {
        let points: Vec<f64> = ids.iter().flat_map(|&id| [id as f64, 0.0, 0.0]).collect();
        let cell_types = vec![CellType::Edge; connectivity.len() / 2];

        xdmf::mpi::TimeSeriesWriter::new(file_name, storage, comm)?.write_mesh(
            &points,
            ids,
            connectivity,
            &cell_types,
        )
    }

    #[mpi_test(np = [3])]
    fn an_overlap_cancelling_out_a_gap_is_rejected() {
        let (_universe, world) = initialize();
        let rank = world.rank();
        let (_tmp_dir, dir) = shared_dir(&world);

        // ranks 0 and 1 both own 0..3, rank 2 owns 6..9: the count (9), the smallest id (0) and the
        // largest (8) all look right, but ids 3..6 are owned by nobody
        let ids: Vec<u64> = match rank {
            0 | 1 => vec![0, 1, 2],
            _ => vec![6, 7, 8],
        };
        let connectivity: &[u64] = if rank == 0 { &[0, 1] } else { &[] };

        for storage in ERROR_STORAGES {
            let result = write_mesh_with_ids(
                &world,
                &dir.join(format!("overlap_{}", storage_tag(storage))),
                storage,
                &ids,
                connectivity,
            );
            std::assert_matches!(
                result.unwrap_err(),
                Error::InvalidMesh { reason } if reason.contains("overlaps")
            );
        }
    }

    #[mpi_test(np = [2, 3])]
    fn a_point_owned_by_nobody_is_rejected() {
        let (_universe, world) = initialize();
        let rank = world.rank() as u64;
        let (_tmp_dir, dir) = shared_dir(&world);

        // every rank owns 3 ids, but rank 0 drops the last of its block
        let start = rank * 3;
        let end = if rank == 0 { 2 } else { start + 3 };
        let ids: Vec<u64> = (start..end).collect();
        let connectivity: &[u64] = if rank == 0 { &[0, 1] } else { &[] };

        for storage in ERROR_STORAGES {
            let result = write_mesh_with_ids(
                &world,
                &dir.join(format!("gap_{}", storage_tag(storage))),
                storage,
                &ids,
                connectivity,
            );
            std::assert_matches!(
                result.unwrap_err(),
                Error::InvalidMesh { reason } if reason.contains("gap")
            );
        }
    }

    #[mpi_test(np = [2, 3])]
    fn a_non_contiguous_block_is_rejected() {
        let (_universe, world) = initialize();
        let rank = world.rank();
        let (_tmp_dir, dir) = shared_dir(&world);

        let ids: Vec<u64> = if rank == 0 { vec![0, 1, 3] } else { vec![] };
        let connectivity: &[u64] = if rank == 0 { &[0, 1] } else { &[] };

        for storage in ERROR_STORAGES {
            let result = write_mesh_with_ids(
                &world,
                &dir.join(format!("contiguous_{}", storage_tag(storage))),
                storage,
                &ids,
                connectivity,
            );
            std::assert_matches!(
                result.unwrap_err(),
                Error::InvalidMesh { reason } if reason.contains("contiguous block") && reason.contains("rank 0")
            );
        }
    }

    #[mpi_test(np = [2, 3])]
    fn an_out_of_bounds_index_on_one_rank_fails_every_rank() {
        let (_universe, world) = initialize();
        let rank = world.rank();
        let size = world.size() as u64;
        let (_tmp_dir, dir) = shared_dir(&world);

        let start = rank as u64 * 2;
        let ids = [start, start + 1];
        // the last rank points one past the end of the mesh
        let connectivity = if rank as u64 == size - 1 {
            [start, 2 * size]
        } else {
            [start, start + 1]
        };

        for storage in ERROR_STORAGES {
            let result = write_mesh_with_ids(
                &world,
                &dir.join(format!("bounds_{}", storage_tag(storage))),
                storage,
                &ids,
                &connectivity,
            );

            let expected_prefix = if rank as u64 == size - 1 {
                "connectivity index".to_string()
            } else {
                format!("on rank {}: connectivity index", size - 1)
            };
            std::assert_matches!(
                result.unwrap_err(),
                Error::InvalidMesh { reason } if reason.starts_with(&expected_prefix)
            );
        }
    }

    #[mpi_test(np = [1, 2, 3])]
    fn a_step_whose_light_data_fails_to_write_stays_written_on_every_rank() {
        let (_universe, world) = initialize();
        let rank = world.rank();
        let (_tmp_dir, dir) = shared_dir(&world);

        for storage in ERROR_STORAGES {
            let (mut writer, share) = small_writer(
                &world,
                &dir.join(format!("light_{}", storage_tag(storage))),
                storage,
            );

            // a directory where the file goes makes renaming the new file into place fail, after
            // the step was added to the document -- as serially, it stays added
            let file_name = writer.file_name().to_path_buf();
            if rank == 0 {
                fs::remove_file(&file_name).unwrap();
                fs::create_dir(&file_name).unwrap();
            }
            world.barrier();

            let write = |writer: &mut xdmf::mpi::TimeSeriesDataWriter| {
                writer.write_time_step("0.0", |data| {
                    data.point_data(
                        "temperature",
                        DataAttribute::Scalar,
                        temperature(&share.points, 0),
                    )
                })
            };

            if rank == 0 {
                std::assert_matches!(write(&mut writer).unwrap_err(), Error::Io { .. });
            } else {
                std::assert_matches!(
                    write(&mut writer).unwrap_err(),
                    Error::Io { source, .. } if source.to_string().starts_with("on rank 0")
                );
            }

            std::assert_matches!(
                write(&mut writer).unwrap_err(),
                Error::InvalidTimeStep { reason, .. } if reason == "already written"
            );
        }
    }
}
