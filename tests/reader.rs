//! Round-trip tests for `TimeSeriesReader`, which reads `Format="HDF"` documents only.
#![cfg(feature = "hdf5")]

use std::ops::Range;

use float_cmp::assert_approx_eq;
use temp_dir::TempDir;
use xdmf::{CellType, DataAttribute, DataStorage, TimeSeriesReader, TimeSeriesWriter};

const STORAGES: [DataStorage; 2] = [
    DataStorage::Hdf5SingleFile {
        deflate_level: None,
    },
    DataStorage::Hdf5MultipleFiles {
        deflate_level: None,
    },
];

#[test]
fn round_trip_mesh_only() {
    for storage in STORAGES {
        let tmp_dir = TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("mesh");

        let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let connectivity = [0_u64, 1, 0, 2, 1, 1, 2, 3];
        let cell_types = [CellType::Edge, CellType::Triangle, CellType::Triangle];

        TimeSeriesWriter::new(&file_name, storage)
            .unwrap()
            .write_mesh(&coords, &connectivity, &cell_types)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to write mesh: {error}"));

        let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2"))
            .unwrap_or_else(|error| panic!("{storage:?}: failed to open: {error}"));

        assert_eq!(reader.num_points(), 4, "{storage:?}");
        assert_eq!(reader.num_cells(), 3, "{storage:?}");
        assert!(reader.times().is_empty(), "{storage:?}");
        assert!(reader.submesh_names().is_empty(), "{storage:?}");
        assert_eq!(reader.num_steps(), 0, "{storage:?}");

        let mut points = Vec::new();
        reader
            .read_points(&mut points)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to read points: {error}"));

        let mut read_connectivity: Vec<u64> = Vec::new();
        let mut read_cell_types = Vec::new();
        reader
            .read_topology(&mut read_connectivity, &mut read_cell_types)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to read topology: {error}"));

        assert_approx_eq!(&[f64], &points, &coords);
        assert_eq!(read_connectivity, connectivity, "{storage:?}");
        assert_eq!(read_cell_types, cell_types, "{storage:?}");
    }
}

/// `write_mesh` with no cells at all writes a `Polyvertex` topology with identity connectivity
/// (required for `ParaView` to show bare points); the reader does not try to tell that apart from a
/// caller-supplied mesh of that many `Vertex` cells in that order, since the file doesn't either --
/// it reads back as such a mesh, not as empty `connectivity`/`cell_types`.
#[test]
fn write_mesh_with_no_cells_reads_back_as_vertex_cells() {
    for storage in STORAGES {
        let tmp_dir = TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("mesh");

        let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        TimeSeriesWriter::new(&file_name, storage)
            .unwrap()
            .write_mesh(&coords, &[] as &[u32], &[])
            .unwrap_or_else(|error| panic!("{storage:?}: failed to write mesh: {error}"));

        let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap();
        assert_eq!(reader.num_points(), 3, "{storage:?}");
        assert_eq!(reader.num_cells(), 3, "{storage:?}");

        let mut points = Vec::new();
        reader
            .read_points(&mut points)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to read points: {error}"));

        let mut connectivity: Vec<u64> = Vec::new();
        let mut cell_types = Vec::new();
        reader
            .read_topology(&mut connectivity, &mut cell_types)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to read topology: {error}"));

        assert_approx_eq!(&[f64], &points, &coords);
        assert_eq!(connectivity, [0_u64, 1, 2], "{storage:?}");
        assert_eq!(cell_types, [CellType::Vertex; 3], "{storage:?}");
    }
}

#[test]
fn round_trip_every_cell_type() {
    const ALL: [CellType; 19] = [
        CellType::Vertex,
        CellType::Edge,
        CellType::Triangle,
        CellType::Quadrilateral,
        CellType::Tetrahedron,
        CellType::Pyramid,
        CellType::Wedge,
        CellType::Hexahedron,
        CellType::Edge3,
        CellType::Quadrilateral9,
        CellType::Triangle6,
        CellType::Quadrilateral8,
        CellType::Tetrahedron10,
        CellType::Pyramid13,
        CellType::Wedge15,
        CellType::Wedge18,
        CellType::Hexahedron20,
        CellType::Hexahedron24,
        CellType::Hexahedron27,
    ];

    // enough points for the widest cell type (Hexahedron27, 27 points), points are otherwise
    // structurally arbitrary since neither the writer nor the reader validates geometry
    let num_points = 27;
    let coords: Vec<f64> = (0..num_points * 3).map(|i| i as f64).collect();
    let mut connectivity = Vec::new();
    for cell_type in ALL {
        connectivity.extend(0..cell_type.num_points() as u64);
    }

    for storage in STORAGES {
        let tmp_dir = TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("mesh");

        TimeSeriesWriter::new(&file_name, storage)
            .unwrap()
            .write_mesh(&coords, &connectivity, &ALL)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to write mesh: {error}"));

        let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap();
        let mut read_connectivity: Vec<u64> = Vec::new();
        let mut read_cell_types = Vec::new();
        reader
            .read_topology(&mut read_connectivity, &mut read_cell_types)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to read topology: {error}"));

        assert_eq!(read_cell_types, ALL, "{storage:?}");
        assert_eq!(read_connectivity, connectivity, "{storage:?}");
    }
}

fn quad_mesh() -> ([f64; 12], [u64; 8], [CellType; 3]) {
    (
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        [0_u64, 1, 0, 2, 1, 1, 2, 3],
        [CellType::Edge, CellType::Triangle, CellType::Triangle],
    )
}

#[test]
fn round_trip_point_and_cell_data_over_several_steps() {
    for storage in STORAGES {
        let (coords, connectivity, cell_types) = quad_mesh();

        let tmp_dir = TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("mesh");

        let mut writer = TimeSeriesWriter::new(&file_name, storage)
            .unwrap()
            .write_mesh(&coords, &connectivity, &cell_types)
            .unwrap();

        for time in ["0.0", "1.5"] {
            writer
                .write_time_step(time, |step| {
                    let t: f64 = time.parse().unwrap();
                    step.point_data(
                        "temperature",
                        DataAttribute::Scalar,
                        vec![t, t + 1.0, t + 2.0, t + 3.0],
                    )?;
                    step.point_data(
                        "velocity",
                        DataAttribute::Vector,
                        (0..12).map(|i| t + i as f64).collect::<Vec<_>>(),
                    )?;
                    step.cell_data("pressure", DataAttribute::Scalar, vec![t, t + 1.0, t + 2.0])
                })
                .unwrap_or_else(|error| panic!("{storage:?}: failed to write step: {error}"));
        }

        let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap();
        assert_eq!(reader.times(), ["0.0", "1.5"], "{storage:?}");
        assert_eq!(reader.num_steps(), 2, "{storage:?}");

        for (step, time) in ["0.0", "1.5"].iter().enumerate() {
            let t: f64 = time.parse().unwrap();

            let point_info = reader.point_data_info(step).unwrap();
            assert_eq!(point_info.len(), 2, "{storage:?}");

            let mut temperature = Vec::new();
            reader
                .read_point_data::<f64>(step, "temperature", &mut temperature)
                .unwrap_or_else(|error| panic!("{storage:?}: {error}"));
            assert_approx_eq!(&[f64], &temperature, &[t, t + 1.0, t + 2.0, t + 3.0]);

            let mut velocity = Vec::new();
            reader
                .read_point_data::<f64>(step, "velocity", &mut velocity)
                .unwrap();
            let expected: Vec<f64> = (0..12).map(|i| t + i as f64).collect();
            assert_approx_eq!(&[f64], &velocity, &expected);

            let cell_info = reader.cell_data_info(step).unwrap();
            assert_eq!(cell_info.len(), 1, "{storage:?}");
            assert_eq!(cell_info[0].attribute, DataAttribute::Scalar);
            assert_eq!(cell_info[0].components, 1);
            assert_eq!(cell_info[0].len, 3);

            let mut pressure = Vec::new();
            reader
                .read_cell_data::<f64>(step, "pressure", &mut pressure)
                .unwrap();
            assert_approx_eq!(&[f64], &pressure, &[t, t + 1.0, t + 2.0]);
        }
    }
}

#[test]
fn reading_a_field_repeatedly_allocates_nothing_of_its_size() {
    // A field read back at the width it was written goes straight into the caller's buffer, so a
    // loop over the steps allocates once rather than once per step. Measured in bytes rather than
    // in allocation *count*, which the path/name handling makes brittle: what must not happen is
    // an array the size of the field, and that is what shows up here.
    const NUM_POINTS: usize = 50_000;
    let field_bytes = NUM_POINTS * size_of::<f64>();

    for storage in STORAGES {
        let tmp_dir = TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("mesh");

        let coords: Vec<f64> = (0..NUM_POINTS * 3).map(|i| i as f64).collect();
        let mut writer = TimeSeriesWriter::new(&file_name, storage)
            .unwrap()
            .write_mesh(&coords, &[] as &[u32], &[])
            .unwrap();

        let times = ["0.0", "1.0", "2.0"];
        for time in times {
            let t: f64 = time.parse().unwrap();
            writer
                .write_time_step(time, |step| {
                    step.point_data(
                        "temperature",
                        DataAttribute::Scalar,
                        (0..NUM_POINTS).map(|i| t + i as f64).collect::<Vec<_>>(),
                    )
                })
                .unwrap();
        }
        drop(writer);

        let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap();

        // the first read is what sizes the buffer, and is allowed to allocate it
        let mut temperature = Vec::new();
        reader
            .read_point_data::<f64>(0, "temperature", &mut temperature)
            .unwrap_or_else(|error| panic!("{storage:?}: {error}"));

        for (step, time) in times.iter().enumerate().skip(1) {
            let before = counting_allocator::allocated_bytes();
            reader
                .read_point_data::<f64>(step, "temperature", &mut temperature)
                .unwrap();
            let allocated = counting_allocator::allocated_bytes() - before;

            let t: f64 = time.parse().unwrap();
            assert_approx_eq!(f64, temperature[0], t);
            assert_eq!(temperature.len(), NUM_POINTS, "{storage:?}");
            assert!(
                allocated < field_bytes / 8,
                "{storage:?}: reading step {step} into a buffer that already fits it allocated \
                 {allocated} bytes, about the {field_bytes} bytes of the field itself -- it is \
                 going through an array of its own instead of filling the buffer"
            );
        }
    }
}

#[test]
fn re_reading_the_mesh_allocates_nothing_of_its_size() {
    // Points and connectivity are filled in place too, so a second read into the same buffers
    // costs nothing -- no `u64` staging array for the connectivity, whatever type the file holds.
    const NUM_POINTS: usize = 50_000;
    let num_cells = NUM_POINTS - 2;

    for storage in STORAGES {
        let tmp_dir = TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("mesh");

        let coords: Vec<f64> = (0..NUM_POINTS * 3).map(|i| i as f64).collect();
        let connectivity: Vec<u32> = (0..num_cells as u32)
            .flat_map(|i| [i, i + 1, i + 2])
            .collect();
        let cell_types = vec![CellType::Triangle; num_cells];

        let mut writer = TimeSeriesWriter::new(&file_name, storage)
            .unwrap()
            .write_mesh(&coords, &connectivity, &cell_types)
            .unwrap();
        writer
            .write_time_step("0.0", |step| {
                step.point_data("temperature", DataAttribute::Scalar, vec![0.0; NUM_POINTS])
            })
            .unwrap();
        drop(writer);

        let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap();

        // the first read of each is what sizes the buffers
        let mut points: Vec<f64> = Vec::new();
        let mut read_connectivity: Vec<u32> = Vec::new();
        let mut read_cell_types = Vec::new();
        reader
            .read_points(&mut points)
            .unwrap_or_else(|error| panic!("{storage:?}: {error}"));
        reader
            .read_topology(&mut read_connectivity, &mut read_cell_types)
            .unwrap_or_else(|error| panic!("{storage:?}: {error}"));

        assert_eq!(points.len(), NUM_POINTS * 3, "{storage:?}");
        assert_eq!(read_connectivity, connectivity, "{storage:?}");
        assert_eq!(read_cell_types, cell_types, "{storage:?}");

        let points_bytes = NUM_POINTS * 3 * size_of::<f64>();
        let before = counting_allocator::allocated_bytes();
        reader.read_points(&mut points).unwrap();
        let allocated = counting_allocator::allocated_bytes() - before;
        assert!(
            allocated < points_bytes / 8,
            "{storage:?}: re-reading the points allocated {allocated} bytes against the \
             {points_bytes} bytes of the array itself"
        );

        let connectivity_bytes = connectivity.len() * size_of::<u32>();
        let before = counting_allocator::allocated_bytes();
        reader
            .read_topology(&mut read_connectivity, &mut read_cell_types)
            .unwrap();
        let allocated = counting_allocator::allocated_bytes() - before;
        assert!(
            allocated < connectivity_bytes / 8,
            "{storage:?}: re-reading the topology allocated {allocated} bytes against the \
             {connectivity_bytes} bytes of the connectivity itself"
        );
    }
}

#[test]
fn re_reading_a_mesh_with_submeshes_holds_one_submesh_at_a_time() {
    // The submesh path cannot fill the caller's buffers straight from the file -- the mesh is
    // scattered back together out of arrays that are each some submesh's own -- but nothing of
    // the mesh's own size is ever held twice: the points are interleaved one direction at a time,
    // and the topology decodes one submesh at a time into a buffer it reuses for the next.
    const NUM_CELLS: usize = 20_000;
    const NUM_SUBMESHES: usize = 4;
    let num_points = NUM_CELLS + 7;

    for storage in STORAGES {
        let tmp_dir = TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("mesh");

        let coords: Vec<f64> = (0..num_points * 3).map(|i| i as f64).collect();
        let connectivity: Vec<u64> = (0..NUM_CELLS as u64)
            .flat_map(|cell| cell..cell + 8)
            .collect();
        let cell_types = vec![CellType::Hexahedron; NUM_CELLS];

        let per_submesh = NUM_CELLS / NUM_SUBMESHES;
        let submeshes: Vec<(String, Range<usize>)> = (0..NUM_SUBMESHES)
            .map(|block| {
                (
                    format!("block_{block}"),
                    block * per_submesh..(block + 1) * per_submesh,
                )
            })
            .collect();

        TimeSeriesWriter::new(&file_name, storage)
            .unwrap()
            .write_mesh_with_submeshes(&coords, &connectivity, &cell_types, submeshes)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to write mesh: {error}"));

        let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap();

        // the first read of each is what sizes the buffers
        let mut points: Vec<f64> = Vec::new();
        let mut read_connectivity: Vec<u64> = Vec::new();
        let mut read_cell_types = Vec::new();
        reader
            .read_points(&mut points)
            .unwrap_or_else(|error| panic!("{storage:?}: {error}"));
        reader
            .read_topology(&mut read_connectivity, &mut read_cell_types)
            .unwrap_or_else(|error| panic!("{storage:?}: {error}"));

        assert_approx_eq!(&[f64], &coords, &points);
        assert_eq!(read_connectivity, connectivity, "{storage:?}");
        assert_eq!(read_cell_types, cell_types, "{storage:?}");

        let points_bytes = num_points * 3 * size_of::<f64>();
        let before = counting_allocator::allocated_bytes();
        reader.read_points(&mut points).unwrap();
        let allocated = counting_allocator::allocated_bytes() - before;
        assert!(
            allocated < points_bytes / 2,
            "{storage:?}: re-reading the points allocated {allocated} bytes against the \
             {points_bytes} bytes of the array itself -- it is holding all three directions at \
             once instead of one"
        );

        let connectivity_bytes = connectivity.len() * size_of::<u64>();
        let before = counting_allocator::allocated_bytes();
        reader
            .read_topology(&mut read_connectivity, &mut read_cell_types)
            .unwrap();
        let allocated = counting_allocator::allocated_bytes() - before;
        assert!(
            allocated < connectivity_bytes / 2,
            "{storage:?}: re-reading the topology allocated {allocated} bytes against the \
             {connectivity_bytes} bytes of the connectivity itself -- it is holding every \
             submesh's decoded topology at once instead of one"
        );
    }
}

/// Counts the bytes Rust code allocates, so a test can assert that a read does not build an array
/// the size of the field it is reading. Only Rust allocations are seen: HDF5's own buffers come
/// from the C library's `malloc` and do not pass through here.
mod counting_allocator {
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        sync::atomic::{AtomicUsize, Ordering},
    };

    static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

    pub fn allocated_bytes() -> usize {
        ALLOCATED.load(Ordering::Relaxed)
    }

    pub struct Counting;

    // SAFETY: every method forwards to `System`, which upholds the contract; the counter is the
    // only thing added and touches no allocation state.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) };
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            ALLOCATED.fetch_add(new_size.saturating_sub(layout.size()), Ordering::Relaxed);
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }
}

#[global_allocator]
static ALLOCATOR: counting_allocator::Counting = counting_allocator::Counting;

#[test]
fn round_trip_f32_attributes() {
    for storage in STORAGES {
        let (coords, connectivity, cell_types) = quad_mesh();

        let tmp_dir = TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("mesh");

        let mut writer = TimeSeriesWriter::new(&file_name, storage)
            .unwrap()
            .write_mesh(&coords, &connectivity, &cell_types)
            .unwrap();

        writer
            .write_time_step("0.0", |step| {
                step.point_data(
                    "temperature",
                    DataAttribute::Scalar,
                    vec![1.0_f32, 2.0, 3.0, 4.0],
                )?;
                step.point_data(
                    "temperature_f64",
                    DataAttribute::Scalar,
                    vec![1.0, 2.0, 3.0, 4.0],
                )
            })
            .unwrap();

        let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap();

        let info = reader.point_data_info(0).unwrap();
        assert_eq!(info[0].precision, 4, "{storage:?}");

        // reading at the file's own type works
        let mut exact = Vec::new();
        reader
            .read_point_data::<f32>(0, "temperature", &mut exact)
            .unwrap_or_else(|error| panic!("{storage:?}: {error}"));
        assert_approx_eq!(&[f32], &exact, &[1.0, 2.0, 3.0, 4.0]);

        // widening is allowed
        let mut widened = Vec::new();
        reader
            .read_point_data::<f64>(0, "temperature", &mut widened)
            .unwrap();
        assert_approx_eq!(&[f64], &widened, &[1.0, 2.0, 3.0, 4.0]);

        // narrowing f64 file data into a f32 buffer must be rejected, not silently truncated
        let mut narrowed = Vec::new();
        std::assert_matches!(
            reader.read_point_data::<f32>(0, "temperature_f64", &mut narrowed),
            Err(xdmf::Error::NumberTypeMismatch { .. })
        );
    }
}

#[test]
fn round_trip_u64_attributes() {
    for storage in STORAGES {
        let (coords, connectivity, cell_types) = quad_mesh();

        let tmp_dir = TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("mesh");

        let mut writer = TimeSeriesWriter::new(&file_name, storage)
            .unwrap()
            .write_mesh(&coords, &connectivity, &cell_types)
            .unwrap();

        writer
            .write_time_step("0.0", |step| {
                step.cell_data(
                    "global_id",
                    DataAttribute::Scalar,
                    vec![100_u64, 200, u64::from(u32::MAX)],
                )
            })
            .unwrap();

        let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap();

        let mut ids = Vec::new();
        reader
            .read_cell_data::<u64>(0, "global_id", &mut ids)
            .unwrap_or_else(|error| panic!("{storage:?}: {error}"));
        assert_eq!(ids, [100, 200, u64::from(u32::MAX)], "{storage:?}");

        // narrowing u64 file data into a u32 buffer must be rejected, not silently truncated
        let mut narrowed = Vec::new();
        std::assert_matches!(
            reader.read_cell_data::<u32>(0, "global_id", &mut narrowed),
            Err(xdmf::Error::NumberTypeMismatch { .. })
        );
    }
}

#[test]
fn round_trip_data_attribute_shapes() {
    for storage in STORAGES {
        let (coords, connectivity, cell_types) = quad_mesh();

        let tmp_dir = TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("mesh");

        let mut writer = TimeSeriesWriter::new(&file_name, storage)
            .unwrap()
            .write_mesh(&coords, &connectivity, &cell_types)
            .unwrap();

        writer
            .write_time_step("0.0", |step| {
                step.point_data(
                    "tensor",
                    DataAttribute::Tensor,
                    (0..36).map(|i| i as f64).collect::<Vec<_>>(),
                )?;
                step.point_data(
                    "tensor6",
                    DataAttribute::Tensor6,
                    (0..24).map(|i| i as f64).collect::<Vec<_>>(),
                )?;
                step.point_data(
                    "generic",
                    DataAttribute::Generic(5),
                    (0..20).map(|i| i as f64).collect::<Vec<_>>(),
                )
            })
            .unwrap();

        let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap();

        let info = reader.point_data_info(0).unwrap();
        let find = |name: &str| info.iter().find(|i| i.name == name).unwrap();

        assert_eq!(
            find("tensor").attribute,
            DataAttribute::Tensor,
            "{storage:?}"
        );
        assert_eq!(find("tensor").components, 9);
        assert_eq!(find("tensor").len, 36);
        // Tensor6/Matrix/Generic all collapse to AttributeType::Matrix on write and are read back
        // as Generic(size), see DataInfo::attribute's doc
        assert_eq!(
            find("tensor6").attribute,
            DataAttribute::Generic(6),
            "{storage:?}"
        );
        assert_eq!(find("tensor6").components, 6);
        assert_eq!(find("tensor6").len, 24);
        assert_eq!(
            find("generic").attribute,
            DataAttribute::Generic(5),
            "{storage:?}"
        );

        let mut tensor = Vec::new();
        reader
            .read_point_data::<f64>(0, "tensor", &mut tensor)
            .unwrap();
        let expected: Vec<f64> = (0..36).map(|i| i as f64).collect();
        assert_approx_eq!(&[f64], &tensor, &expected);
    }
}

fn submesh_test_mesh() -> ([f64; 12], [u64; 8], [CellType; 3]) {
    quad_mesh()
}

#[test]
fn round_trip_contiguous_submeshes() {
    for storage in STORAGES {
        let (coords, connectivity, cell_types) = submesh_test_mesh();

        let tmp_dir = TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("mesh");

        let mut writer = TimeSeriesWriter::new(&file_name, storage)
            .unwrap()
            .write_mesh_with_submeshes(
                &coords,
                &connectivity,
                &cell_types,
                [("edge", &[0][..]), ("surface", &[1, 2][..])],
            )
            .unwrap_or_else(|error| panic!("{storage:?}: failed to write mesh: {error}"));

        writer
            .write_time_step("0.0", |step| {
                step.point_data("p", DataAttribute::Scalar, vec![10.0, 11.0, 12.0, 13.0])?;
                step.cell_data("c", DataAttribute::Scalar, vec![20.0, 21.0, 22.0])
            })
            .unwrap();

        let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap();
        assert_eq!(reader.submesh_names(), ["edge", "surface"], "{storage:?}");
        assert_eq!(reader.num_points(), 4, "{storage:?}");
        assert_eq!(reader.num_cells(), 3, "{storage:?}");

        let mut points = Vec::new();
        reader
            .read_points(&mut points)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to read points: {error}"));

        let mut read_connectivity: Vec<u64> = Vec::new();
        let mut read_cell_types = Vec::new();
        reader
            .read_topology(&mut read_connectivity, &mut read_cell_types)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to read topology: {error}"));

        assert_approx_eq!(&[f64], &points, &coords);
        assert_eq!(read_cell_types, cell_types, "{storage:?}");
        assert_eq!(read_connectivity, connectivity, "{storage:?}");

        // which cells/points each submesh holds, in mesh (global) indices
        assert_eq!(reader.submesh_cells(0).unwrap(), vec![0], "{storage:?}");
        assert_eq!(reader.submesh_cells(1).unwrap(), vec![1, 2], "{storage:?}");
        assert_eq!(reader.submesh_points(0).unwrap(), vec![0, 1], "{storage:?}");
        assert_eq!(
            reader.submesh_points(1).unwrap(),
            vec![0, 1, 2, 3],
            "{storage:?}"
        );
        std::assert_matches!(
            reader.submesh_cells(2),
            Err(xdmf::Error::InvalidDocument { .. })
        );

        let mut p = Vec::new();
        reader.read_point_data::<f64>(0, "p", &mut p).unwrap();
        assert_approx_eq!(&[f64], &p, &[10.0, 11.0, 12.0, 13.0]);

        let mut c = Vec::new();
        reader.read_cell_data::<f64>(0, "c", &mut c).unwrap();
        assert_approx_eq!(&[f64], &c, &[20.0, 21.0, 22.0]);
    }
}

#[test]
fn round_trip_scattered_overlapping_submeshes_with_an_unused_point() {
    for storage in STORAGES {
        // 5 points, point 4 unused by any cell; 4 cells, cell 3 in no submesh's own patch but the
        // "both" submesh scatters cells [3, 0], and "quads" overlaps "tris" on cell 1
        let coords = [
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 5.0, 5.0, 5.0,
        ];
        let connectivity = [0_u64, 1, 2, 1, 2, 3, 0, 1, 2, 3, 2, 1, 0];
        let cell_types = [
            CellType::Triangle,
            CellType::Triangle,
            CellType::Triangle,
            CellType::Quadrilateral,
        ];

        let tmp_dir = TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("mesh");

        let mut writer = TimeSeriesWriter::new(&file_name, storage)
            .unwrap()
            .write_mesh_with_submeshes(
                &coords,
                &connectivity,
                &cell_types,
                [
                    ("tris", &[0, 1, 2][..]),
                    ("both", &[3, 0][..]),
                    ("overlap", &[1][..]),
                ],
            )
            .unwrap_or_else(|error| panic!("{storage:?}: failed to write mesh: {error}"));

        writer
            .write_time_step("0.0", |step| {
                step.point_data("p", DataAttribute::Scalar, vec![1.0, 2.0, 3.0, 4.0, 5.0])?;
                step.cell_data("c", DataAttribute::Scalar, vec![10.0, 11.0, 12.0, 13.0])
            })
            .unwrap();

        let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap();
        assert_eq!(reader.num_points(), 5, "{storage:?}");
        assert_eq!(reader.num_cells(), 4, "{storage:?}");

        let mut points = Vec::new();
        reader
            .read_points(&mut points)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to read points: {error}"));

        let mut read_connectivity: Vec<u64> = Vec::new();
        let mut read_cell_types = Vec::new();
        reader
            .read_topology(&mut read_connectivity, &mut read_cell_types)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to read topology: {error}"));

        // the identity is preserved on the whole points array, unused points included
        assert_approx_eq!(&[f64], &points, &coords);
        assert_eq!(read_cell_types, cell_types, "{storage:?}");
        assert_eq!(read_connectivity, connectivity, "{storage:?}");

        // "both" lists its cells [3, 0] -- positional, not sorted ascending
        assert_eq!(
            reader.submesh_cells(0).unwrap(),
            vec![0, 1, 2],
            "{storage:?}"
        );
        assert_eq!(reader.submesh_cells(1).unwrap(), vec![3, 0], "{storage:?}");
        assert_eq!(reader.submesh_cells(2).unwrap(), vec![1], "{storage:?}");
        // but the points a submesh holds are always ascending, regardless of cell order
        assert_eq!(
            reader.submesh_points(1).unwrap(),
            vec![0, 1, 2, 3],
            "{storage:?}"
        );

        let mut p = Vec::new();
        reader.read_point_data::<f64>(0, "p", &mut p).unwrap();
        assert_approx_eq!(&[f64], &p, &[1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut c = Vec::new();
        reader.read_cell_data::<f64>(0, "c", &mut c).unwrap();
        assert_approx_eq!(&[f64], &c, &[10.0, 11.0, 12.0, 13.0]);
    }
}

#[test]
fn round_trip_a_submesh_with_deliberately_unordered_cells() {
    for storage in STORAGES {
        let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let connectivity = [0_u64, 1, 2, 1, 2, 3, 0, 1, 3];
        let cell_types = [CellType::Triangle, CellType::Triangle, CellType::Triangle];

        let tmp_dir = TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("mesh");

        // cells listed 2, 0, 1 -- deliberately not ascending
        let mut writer = TimeSeriesWriter::new(&file_name, storage)
            .unwrap()
            .write_mesh_with_submeshes(
                &coords,
                &connectivity,
                &cell_types,
                [("unordered", &[2, 0, 1][..])],
            )
            .unwrap_or_else(|error| panic!("{storage:?}: failed to write mesh: {error}"));

        writer
            .write_time_step("0.0", |step| {
                step.cell_data("c", DataAttribute::Scalar, vec![100.0, 101.0, 102.0])
            })
            .unwrap();

        let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap();

        let mut read_connectivity: Vec<u64> = Vec::new();
        let mut read_cell_types = Vec::new();
        reader
            .read_topology(&mut read_connectivity, &mut read_cell_types)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to read topology: {error}"));

        assert_eq!(read_cell_types, cell_types, "{storage:?}");
        assert_eq!(read_connectivity, connectivity, "{storage:?}");

        // cell data maps back by mesh cell id, not by the submesh's listed (unordered) position
        let mut c = Vec::new();
        reader.read_cell_data::<f64>(0, "c", &mut c).unwrap();
        assert_approx_eq!(&[f64], &c, &[100.0, 101.0, 102.0]);
    }
}

/// The no-cells case with submeshes: reads back as `Vertex` cells in identity order, same as
/// without submeshes (see [`write_mesh_with_no_cells_reads_back_as_vertex_cells`]) -- submesh
/// reassembly does not change that.
#[test]
fn write_mesh_with_no_cells_and_submeshes_reads_back_as_vertex_cells() {
    for storage in STORAGES {
        let tmp_dir = TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("mesh");

        let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];

        TimeSeriesWriter::new(&file_name, storage)
            .unwrap()
            .write_mesh_with_submeshes(
                &coords,
                &[] as &[u32],
                &[],
                [("low", &[0, 1][..]), ("high", &[2, 3][..])],
            )
            .unwrap_or_else(|error| panic!("{storage:?}: failed to write mesh: {error}"));

        let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap();
        assert_eq!(reader.num_points(), 4, "{storage:?}");
        assert_eq!(reader.num_cells(), 4, "{storage:?}");

        let mut points = Vec::new();
        reader
            .read_points(&mut points)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to read points: {error}"));

        let mut connectivity: Vec<u64> = Vec::new();
        let mut cell_types = Vec::new();
        reader
            .read_topology(&mut connectivity, &mut cell_types)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to read topology: {error}"));

        assert_approx_eq!(&[f64], &points, &coords);
        assert_eq!(connectivity, [0_u64, 1, 2, 3], "{storage:?}");
        assert_eq!(cell_types, [CellType::Vertex; 4], "{storage:?}");
    }
}

/// `Vertex` cells over a subset of the points, in an order and count that's not the mesh's
/// identity -- makes sure submesh renumbering (every submesh's own connectivity is a local
/// identity run `0..n`) still reassembles back to the real, non-identity global connectivity.
#[test]
fn round_trip_vertex_cells_with_submeshes() {
    for storage in STORAGES {
        let tmp_dir = TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("mesh");

        let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let connectivity = [2_u64, 0, 3];
        let cell_types = [CellType::Vertex; 3];

        TimeSeriesWriter::new(&file_name, storage)
            .unwrap()
            .write_mesh_with_submeshes(
                &coords,
                &connectivity,
                &cell_types,
                [("pair", &[0, 1][..]), ("single", &[2][..])],
            )
            .unwrap_or_else(|error| panic!("{storage:?}: failed to write mesh: {error}"));

        let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap();
        assert_eq!(reader.num_points(), 4, "{storage:?}");
        assert_eq!(reader.num_cells(), 3, "{storage:?}");

        let mut read_connectivity: Vec<u64> = Vec::new();
        let mut read_cell_types = Vec::new();
        reader
            .read_topology(&mut read_connectivity, &mut read_cell_types)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to read topology: {error}"));

        assert_eq!(read_cell_types, cell_types, "{storage:?}");
        assert_eq!(read_connectivity, connectivity, "{storage:?}");
    }
}

/// A selector naming positions past the end of the array it selects from -- a truncated or
/// hand-written document, since this crate's own writer never emits one -- is reported, not
/// indexed past the end of. Both selector shapes are hand-written into the document here, since
/// nothing the writer produces has a selection over a plain (submesh-less) mesh.
#[test]
fn a_selector_past_the_end_of_its_source_is_rejected() {
    let (coords, connectivity, cell_types) = quad_mesh();

    let tmp_dir = TempDir::new().unwrap();
    let file_name = tmp_dir.path().join("mesh");

    TimeSeriesWriter::new(&file_name, STORAGES[0])
        .unwrap()
        .write_mesh(&coords, &connectivity, &cell_types)
        .unwrap()
        .write_time_step("0.0", |step| {
            step.point_data("p", DataAttribute::Scalar, vec![10.0, 11.0, 12.0, 13.0])
        })
        .unwrap();

    let document_path = file_name.with_extension("xdmf2");
    let document = std::fs::read_to_string(&document_path).unwrap();

    // the field's own DataItem, wrapped below in a selection that reaches past its 4 values
    let source = document
        .lines()
        .find(|line| line.contains("t_0.0"))
        .unwrap()
        .trim();

    // a HyperSlab of 40 values out of the 4 the field holds, and a Coordinates selection through
    // the mesh's connectivity, whose Mixed type codes name positions the field does not have
    let selections = [
        "<DataItem Dimensions=\"3\" NumberType=\"Int\" Format=\"XML\" Precision=\"4\">0 1 \
         40</DataItem>",
        "<DataItem Reference=\"XML\">/Xdmf/Domain/DataItem[@Name=\"connectivity\"]</DataItem>",
    ];

    for (item_type, selector) in ["HyperSlab", "Coordinates"].iter().zip(&selections) {
        let selection = format!(
            "<DataItem ItemType=\"{item_type}\" Dimensions=\"4\" NumberType=\"Float\" \
             Precision=\"8\">{selector}{source}</DataItem>"
        );
        std::fs::write(&document_path, document.replace(source, &selection)).unwrap();

        let reader = TimeSeriesReader::new(&document_path).unwrap();
        let mut p = Vec::new();

        std::assert_matches!(
            reader.read_point_data::<f64>(0, "p", &mut p).unwrap_err(),
            xdmf::Error::InvalidDocument { reason }
                if reason.contains("of an array of only 4 values"),
            "{item_type}"
        );
    }
}

/// A submesh's `<Geometry>` selector names which mesh points it holds, while the mesh's point
/// count comes from the array that selector reads out of -- two independent statements of the
/// file's, so a hand-written or truncated document can have them disagree. Those indices are
/// written *at* when a field is scattered back together, so one past the end is reported when the
/// document is opened rather than indexed with.
#[test]
fn a_submesh_holding_a_point_the_mesh_does_not_have_is_rejected() {
    let (coords, connectivity, cell_types) = submesh_test_mesh();

    let tmp_dir = TempDir::new().unwrap();
    let file_name = tmp_dir.path().join("mesh");

    TimeSeriesWriter::new(&file_name, STORAGES[0])
        .unwrap()
        .write_mesh_with_submeshes(
            &coords,
            &connectivity,
            &cell_types,
            [("edge", &[0][..]), ("surface", &[1, 2][..])],
        )
        .unwrap();

    let document_path = file_name.with_extension("xdmf2");
    let document = std::fs::read_to_string(&document_path).unwrap();

    // the first submesh holds the mesh's points 0..2, as the HyperSlab its three coordinate
    // selections share -- widened here to reach past the 4 points the mesh has
    assert!(document.contains(">0 1 2<"), "{document}");
    std::fs::write(&document_path, document.replace(">0 1 2<", ">0 1 40<")).unwrap();

    std::assert_matches!(
        TimeSeriesReader::new(&document_path).unwrap_err(),
        xdmf::Error::InvalidDocument { reason }
            if reason == "submesh 0 holds point 4, but the mesh only has 4 points"
    );
}

/// Two submeshes may overlap, but not while disagreeing about what a shared cell is: the mesh's
/// cell offsets are built from one of the two answers, so writing the other one's (differently
/// sized) cell there would run over its neighbour's points.
#[test]
fn submeshes_disagreeing_about_a_shared_cell_type_are_rejected() {
    let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let connectivity = [0_u64, 1, 3, 2];
    let cell_types = [CellType::Quadrilateral];

    let tmp_dir = TempDir::new().unwrap();
    let file_name = tmp_dir.path().join("mesh");

    // both submeshes hold the one cell, so the document states its type twice
    TimeSeriesWriter::new(&file_name, STORAGES[0])
        .unwrap()
        .write_mesh_with_submeshes(
            &coords,
            &connectivity,
            &cell_types,
            [("a", &[0][..]), ("b", &[0][..])],
        )
        .unwrap();

    let document_path = file_name.with_extension("xdmf2");
    let document = std::fs::read_to_string(&document_path).unwrap();

    // `Tetrahedron` rather than a type of another size, so the disagreement itself is what is
    // caught and not the cell count it would otherwise imply
    let quadrilateral = "TopologyType=\"Quadrilateral\"";
    assert!(document.contains(quadrilateral), "{document}");
    std::fs::write(
        &document_path,
        document.replacen(quadrilateral, "TopologyType=\"Tetrahedron\"", 1),
    )
    .unwrap();

    let reader = TimeSeriesReader::new(&document_path).unwrap();
    let mut read_connectivity: Vec<u64> = Vec::new();
    let mut read_cell_types = Vec::new();

    std::assert_matches!(
        reader
            .read_topology(&mut read_connectivity, &mut read_cell_types)
            .unwrap_err(),
        xdmf::Error::InvalidDocument { reason }
            if reason.contains("submeshes disagree about mesh cell 0")
    );
}

/// A mesh whose steps were never written has no step 0: its single grid is the mesh itself, and
/// counting it as a step would contradict `num_steps`.
#[test]
fn a_step_index_on_a_mesh_without_steps_is_rejected() {
    for storage in STORAGES {
        let (coords, connectivity, cell_types) = quad_mesh();

        for with_submeshes in [false, true] {
            let tmp_dir = TempDir::new().unwrap();
            let file_name = tmp_dir.path().join("mesh");
            let writer = TimeSeriesWriter::new(&file_name, storage).unwrap();

            if with_submeshes {
                writer
                    .write_mesh_with_submeshes(
                        &coords,
                        &connectivity,
                        &cell_types,
                        [("edge", &[0][..]), ("surface", &[1, 2][..])],
                    )
                    .unwrap();
            } else {
                writer
                    .write_mesh(&coords, &connectivity, &cell_types)
                    .unwrap();
            }

            let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap();
            let label = format!("{storage:?}, submeshes: {with_submeshes}");

            assert_eq!(reader.num_steps(), 0, "{label}");

            std::assert_matches!(
                reader.point_data_info(0).unwrap_err(),
                xdmf::Error::InvalidDocument { reason }
                    if reason == "step index 0 is out of range, 0 steps were written",
                "{label}"
            );
            std::assert_matches!(
                reader.cell_data_info(0).unwrap_err(),
                xdmf::Error::InvalidDocument { .. },
                "{label}"
            );

            let mut values = Vec::new();
            std::assert_matches!(
                reader
                    .read_point_data::<f64>(0, "p", &mut values)
                    .unwrap_err(),
                xdmf::Error::InvalidDocument { .. },
                "{label}"
            );
        }
    }
}

/// The reader answers nothing at all about a document written with a storage it cannot read:
/// `new` is where that is reported, rather than the first call that reaches heavy data.
#[test]
fn opening_a_document_written_with_a_non_hdf5_storage_is_rejected() {
    let (coords, _, cell_types) = quad_mesh();
    // 32-bit, since `Binary` refuses to hold a 64-bit connectivity at all
    let connectivity = [0_u32, 1, 0, 2, 1, 1, 2, 3];

    for storage in [
        DataStorage::Ascii,
        DataStorage::AsciiInline,
        DataStorage::Binary,
    ] {
        let tmp_dir = TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("mesh");

        TimeSeriesWriter::new(&file_name, storage)
            .unwrap()
            .write_mesh(&coords, &connectivity, &cell_types)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to write mesh: {error}"));

        std::assert_matches!(
            TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap_err(),
            xdmf::Error::Unsupported { reason }
                if reason.contains(&format!("written with the {storage:?} storage")),
            "{storage:?}"
        );
    }
}

/// A document that names no storage at all is a foreign file: it is opened, and its `DataItem`s
/// are checked for `Format="HDF"` one by one as they are read.
#[test]
fn a_document_without_a_data_storage_information_is_opened() {
    let (coords, connectivity, cell_types) = quad_mesh();

    let tmp_dir = TempDir::new().unwrap();
    let file_name = tmp_dir.path().join("mesh");

    TimeSeriesWriter::new(&file_name, DataStorage::AsciiInline)
        .unwrap()
        .write_mesh(&coords, &connectivity, &cell_types)
        .unwrap();

    let document_path = file_name.with_extension("xdmf2");
    let document = std::fs::read_to_string(&document_path).unwrap();
    let information = document
        .lines()
        .find(|line| line.contains("data_storage"))
        .unwrap();
    std::fs::write(&document_path, document.replace(information, "")).unwrap();

    let reader = TimeSeriesReader::new(&document_path).unwrap();
    assert_eq!(reader.num_points(), 4);

    let mut points: Vec<f64> = Vec::new();
    std::assert_matches!(
        reader.read_points(&mut points).unwrap_err(),
        xdmf::Error::Unsupported { reason } if reason.contains("is not supported by this reader")
    );
}

/// The element type a mesh is read back at is the caller's choice, independently of what it was
/// written as: `f32` coordinates widen into a `Vec<f64>`, and a `u64` connectivity comes back as
/// `u32` when every index fits one.
#[test]
fn points_and_connectivity_are_read_at_the_requested_width() {
    for storage in STORAGES {
        let tmp_dir = TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("mesh");

        let coords: [f32; 12] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let connectivity = [0_u64, 1, 0, 2, 1, 1, 2, 3];
        let cell_types = [CellType::Edge, CellType::Triangle, CellType::Triangle];

        TimeSeriesWriter::new(&file_name, storage)
            .unwrap()
            .write_mesh(&coords, &connectivity, &cell_types)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to write mesh: {error}"));

        let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap();

        // written as f32, read back at both widths
        let mut narrow: Vec<f32> = Vec::new();
        reader.read_points(&mut narrow).unwrap();
        assert_approx_eq!(&[f32], &narrow, &coords);

        let mut wide: Vec<f64> = Vec::new();
        reader.read_points(&mut wide).unwrap();
        let expected: Vec<f64> = coords.iter().map(|&value| f64::from(value)).collect();
        assert_approx_eq!(&[f64], &wide, &expected);

        // written as u64, read back as u32 -- an index check, not a type check
        let mut cell_types_read = Vec::new();
        let mut narrow_connectivity: Vec<u32> = Vec::new();
        reader
            .read_topology(&mut narrow_connectivity, &mut cell_types_read)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to read topology: {error}"));

        assert_eq!(
            narrow_connectivity,
            [0_u32, 1, 0, 2, 1, 1, 2, 3],
            "{storage:?}"
        );
        assert_eq!(cell_types_read, cell_types, "{storage:?}");
    }
}

/// Narrowing the *coordinates* is refused, unlike the connectivity: they are the file's own
/// values, so the widening rule that governs field data governs them too.
#[test]
fn reading_f64_points_as_f32_is_rejected() {
    let tmp_dir = TempDir::new().unwrap();
    let file_name = tmp_dir.path().join("mesh");

    let (coords, connectivity, cell_types) = quad_mesh();
    TimeSeriesWriter::new(&file_name, STORAGES[0])
        .unwrap()
        .write_mesh(&coords, &connectivity, &cell_types)
        .unwrap();

    let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap();
    let mut points: Vec<f32> = Vec::new();

    std::assert_matches!(
        reader.read_points(&mut points).unwrap_err(),
        xdmf::Error::NumberTypeMismatch { reason } if reason.contains("the file holds f64")
    );
}

/// A mesh split by ranges is the same file as one split by the equivalent index lists, and reads
/// back as such.
#[test]
fn submeshes_given_as_ranges_round_trip() {
    for storage in STORAGES {
        let tmp_dir = TempDir::new().unwrap();
        let file_name = tmp_dir.path().join("mesh");

        let (coords, connectivity, cell_types) = submesh_test_mesh();

        TimeSeriesWriter::new(&file_name, storage)
            .unwrap()
            .write_mesh_with_submeshes(
                &coords,
                &connectivity,
                &cell_types,
                [("edge", 0..1), ("surface", 1..3)],
            )
            .unwrap_or_else(|error| panic!("{storage:?}: failed to write mesh: {error}"));

        let reader = TimeSeriesReader::new(file_name.with_extension("xdmf2")).unwrap();

        assert_eq!(reader.submesh_names(), ["edge", "surface"], "{storage:?}");
        assert_eq!(reader.submesh_cells(0).unwrap(), [0], "{storage:?}");
        assert_eq!(reader.submesh_cells(1).unwrap(), [1, 2], "{storage:?}");

        let mut points: Vec<f64> = Vec::new();
        reader.read_points(&mut points).unwrap();

        let mut read_connectivity: Vec<u64> = Vec::new();
        let mut read_cell_types = Vec::new();
        reader
            .read_topology(&mut read_connectivity, &mut read_cell_types)
            .unwrap_or_else(|error| panic!("{storage:?}: failed to read topology: {error}"));

        assert_approx_eq!(&[f64], &points, &coords);
        assert_eq!(read_connectivity, connectivity, "{storage:?}");
        assert_eq!(read_cell_types, cell_types, "{storage:?}");
    }
}
