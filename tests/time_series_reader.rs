//! Round-trip tests for `TimeSeriesReader`/`TimeSeriesDataReader`: for every storage mode, what
//! this crate writes must read back identically.

use float_cmp::assert_approx_eq;
use rstest::rstest;
use temp_dir::TempDir;
use xdmf::{CellType, DataAttribute, DataStorage, TimeSeriesReader, TimeSeriesWriter};

// Hdf5* cases are skipped at runtime (not filtered out of the case list) when the `hdf5` feature
// is off, mirroring `xdmf::is_hdf5_enabled()` checks elsewhere in the test suite.
fn skip_if_hdf5_disabled(storage: DataStorage) -> bool {
    let needs_hdf5 = matches!(
        storage,
        DataStorage::Hdf5SingleFile { .. } | DataStorage::Hdf5MultipleFiles { .. }
    );
    needs_hdf5 && !xdmf::is_hdf5_enabled()
}

#[rstest]
#[case(DataStorage::Ascii)]
#[case(DataStorage::AsciiInline)]
#[case(DataStorage::Binary)]
#[case(DataStorage::Hdf5SingleFile { deflate_level: None })]
#[case(DataStorage::Hdf5MultipleFiles { deflate_level: None })]
fn mesh_plus_data_multiple_steps_round_trips_for_every_storage_mode(#[case] storage: DataStorage) {
    if skip_if_hdf5_disabled(storage) {
        return;
    }

    let tmp_dir = TempDir::new().unwrap();
    let path = tmp_dir.path().join("out");

    let points = [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 1.0,
    ];
    let connectivity = [0_u64, 1, 2, 3, 0, 1, 2, 4];
    let cell_types = [CellType::Quadrilateral, CellType::Tetrahedron];

    let writer = TimeSeriesWriter::new(&path, storage).unwrap();
    let mut data_writer = writer
        .write_mesh(&points, &connectivity, &cell_types)
        .unwrap();

    let point_scalar_t0 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let cell_ids_t0 = vec![10_u64, 20];
    data_writer
        .write_data(
            "0.0",
            [(
                "temperature",
                DataAttribute::Scalar,
                point_scalar_t0.as_slice().into(),
            )],
            [(
                "cell_id",
                DataAttribute::Scalar,
                cell_ids_t0.as_slice().into(),
            )],
        )
        .unwrap();

    let point_scalar_t1 = vec![11.0, 12.0, 13.0, 14.0, 15.0];
    let cell_ids_t1 = vec![30_u64, 40];
    data_writer
        .write_data(
            "1.0",
            [(
                "temperature",
                DataAttribute::Scalar,
                point_scalar_t1.as_slice().into(),
            )],
            [(
                "cell_id",
                DataAttribute::Scalar,
                cell_ids_t1.as_slice().into(),
            )],
        )
        .unwrap();

    let reader = TimeSeriesReader::new(path.with_extension("xdmf2")).unwrap();
    assert_eq!(reader.num_points(), 5);
    assert_eq!(reader.num_cells(), 2);
    assert_eq!(reader.times(), ["0.0", "1.0"]);

    let mut read_points = Vec::new();
    let mut read_connectivity = Vec::new();
    let mut read_cell_types = Vec::new();
    let mut data_reader = reader
        .read_mesh(
            &mut read_points,
            &mut read_connectivity,
            &mut read_cell_types,
        )
        .unwrap();

    assert_approx_eq!(&[f64], &points, &read_points);
    assert_eq!(read_connectivity, connectivity);
    assert_eq!(read_cell_types, cell_types);

    assert_eq!(data_reader.num_steps(), 2);
    assert_eq!(data_reader.times(), ["0.0", "1.0"]);

    for (step, (expected_points, expected_cells)) in [
        (point_scalar_t0, cell_ids_t0),
        (point_scalar_t1, cell_ids_t1),
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(data_reader.num_point_data(step).unwrap(), 1);
        assert_eq!(data_reader.num_cell_data(step).unwrap(), 1);

        let point_index = data_reader.point_data_index(step, "temperature").unwrap();
        let info = data_reader.point_data_info(step, point_index).unwrap();
        assert_eq!(info.name, "temperature");
        assert_eq!(info.attribute, DataAttribute::Scalar);
        assert_eq!(info.len, 5);

        let mut read_temperature: Vec<f64> = Vec::new();
        data_reader
            .read_point_data(step, point_index, &mut read_temperature)
            .unwrap();
        assert_approx_eq!(&[f64], &expected_points, &read_temperature);

        let cell_index = data_reader.cell_data_index(step, "cell_id").unwrap();
        let mut read_cell_id: Vec<u64> = Vec::new();
        data_reader
            .read_cell_data(step, cell_index, &mut read_cell_id)
            .unwrap();
        assert_eq!(read_cell_id, expected_cells);

        let point_step = data_reader.read_point_step(step).unwrap();
        assert_eq!(point_step.len(), 1);
        assert_eq!(point_step[0].0, "temperature");
        assert_eq!(point_step[0].1, DataAttribute::Scalar);
        assert_approx_eq!(
            &[f64],
            &expected_points,
            point_step[0].2.as_slice::<f64>().unwrap()
        );

        let cell_step = data_reader.read_cell_step(step).unwrap();
        assert_eq!(cell_step.len(), 1);
        assert_eq!(cell_step[0].0, "cell_id");
        assert_eq!(cell_step[0].1, DataAttribute::Scalar);
        assert_eq!(cell_step[0].2.as_slice::<u64>().unwrap(), &expected_cells);
    }
}

#[rstest]
#[case(DataStorage::Ascii)]
#[case(DataStorage::AsciiInline)]
#[case(DataStorage::Binary)]
#[case(DataStorage::Hdf5SingleFile { deflate_level: None })]
#[case(DataStorage::Hdf5MultipleFiles { deflate_level: None })]
fn mesh_only_round_trips_for_every_storage_mode(#[case] storage: DataStorage) {
    if skip_if_hdf5_disabled(storage) {
        return;
    }

    let tmp_dir = TempDir::new().unwrap();
    let path = tmp_dir.path().join("out");

    let points = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let connectivity = [0_u64, 1, 2];
    let cell_types = [CellType::Triangle];

    TimeSeriesWriter::new(&path, storage)
        .unwrap()
        .write_mesh(&points, &connectivity, &cell_types)
        .unwrap();

    let reader = TimeSeriesReader::new(path.with_extension("xdmf2")).unwrap();
    assert_eq!(reader.num_points(), 3);
    assert_eq!(reader.num_cells(), 1);
    assert!(reader.times().is_empty());

    let mut read_points = Vec::new();
    let mut read_connectivity = Vec::new();
    let mut read_cell_types = Vec::new();
    let data_reader = reader
        .read_mesh(
            &mut read_points,
            &mut read_connectivity,
            &mut read_cell_types,
        )
        .unwrap();

    assert_approx_eq!(&[f64], &points, &read_points);
    assert_eq!(read_connectivity, connectivity);
    assert_eq!(read_cell_types, cell_types);
    assert_eq!(data_reader.num_steps(), 0);
}

#[rstest]
#[case(DataStorage::Ascii)]
#[case(DataStorage::AsciiInline)]
#[case(DataStorage::Binary)]
#[case(DataStorage::Hdf5SingleFile { deflate_level: None })]
#[case(DataStorage::Hdf5MultipleFiles { deflate_level: None })]
fn point_cloud_round_trips_with_empty_cell_types(#[case] storage: DataStorage) {
    if skip_if_hdf5_disabled(storage) {
        return;
    }

    let tmp_dir = TempDir::new().unwrap();
    let path = tmp_dir.path().join("out");

    let points = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];

    TimeSeriesWriter::new(&path, storage)
        .unwrap()
        .write_mesh(&points, &[], &[])
        .unwrap();

    let reader = TimeSeriesReader::new(path.with_extension("xdmf2")).unwrap();
    assert_eq!(reader.num_points(), 3);

    let mut read_points = Vec::new();
    let mut read_connectivity = Vec::new();
    let mut read_cell_types = Vec::new();
    reader
        .read_mesh(
            &mut read_points,
            &mut read_connectivity,
            &mut read_cell_types,
        )
        .unwrap();

    assert_approx_eq!(&[f64], &points, &read_points);
    assert!(read_connectivity.is_empty());
    assert!(read_cell_types.is_empty());
}

#[rstest]
#[case(DataStorage::Ascii)]
#[case(DataStorage::AsciiInline)]
#[case(DataStorage::Binary)]
#[case(DataStorage::Hdf5SingleFile { deflate_level: None })]
#[case(DataStorage::Hdf5MultipleFiles { deflate_level: None })]
fn f32_attribute_round_trips_and_widens_into_f64(#[case] storage: DataStorage) {
    if skip_if_hdf5_disabled(storage) {
        return;
    }

    let tmp_dir = TempDir::new().unwrap();
    let path = tmp_dir.path().join("out");

    let points = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let connectivity = [0_u64, 1, 2];
    let cell_types = [CellType::Triangle];

    let mut data_writer = TimeSeriesWriter::new(&path, storage)
        .unwrap()
        .write_mesh(&points, &connectivity, &cell_types)
        .unwrap();

    let pressure: Vec<f32> = vec![1.5, 2.5, 3.5];
    data_writer
        .write_data(
            "0.0",
            [(
                "pressure",
                DataAttribute::Scalar,
                pressure.as_slice().into(),
            )],
            [],
        )
        .unwrap();

    let reader = TimeSeriesReader::new(path.with_extension("xdmf2")).unwrap();
    let mut read_points = Vec::new();
    let mut read_connectivity = Vec::new();
    let mut read_cell_types = Vec::new();
    let mut data_reader = reader
        .read_mesh(
            &mut read_points,
            &mut read_connectivity,
            &mut read_cell_types,
        )
        .unwrap();

    let index = data_reader.point_data_index(0, "pressure").unwrap();
    let info = data_reader.point_data_info(0, index).unwrap();
    assert_eq!(info.kind, xdmf::ValueKind::F32);

    // exact match: same type, no widening
    let mut read_as_f32: Vec<f32> = Vec::new();
    data_reader
        .read_point_data(0, index, &mut read_as_f32)
        .unwrap();
    assert_eq!(read_as_f32, pressure);

    // widening: f32 file into a f64 buffer
    let mut read_as_f64: Vec<f64> = Vec::new();
    data_reader
        .read_point_data(0, index, &mut read_as_f64)
        .unwrap();
    let expected: Vec<f64> = pressure.iter().map(|&v| f64::from(v)).collect();
    assert_approx_eq!(&[f64], &expected, &read_as_f64);
}

#[test]
fn narrowing_f64_into_f32_buffer_is_rejected() {
    let tmp_dir = TempDir::new().unwrap();
    let path = tmp_dir.path().join("out");

    let points = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let connectivity = [0_u64, 1, 2];
    let cell_types = [CellType::Triangle];

    let mut data_writer = TimeSeriesWriter::new(&path, DataStorage::AsciiInline)
        .unwrap()
        .write_mesh(&points, &connectivity, &cell_types)
        .unwrap();

    let pressure = vec![1.5_f64, 2.5, 3.5];
    data_writer
        .write_data(
            "0.0",
            [(
                "pressure",
                DataAttribute::Scalar,
                pressure.as_slice().into(),
            )],
            [],
        )
        .unwrap();

    let reader = TimeSeriesReader::new(path.with_extension("xdmf2")).unwrap();
    let mut read_points = Vec::new();
    let mut read_connectivity = Vec::new();
    let mut read_cell_types = Vec::new();
    let mut data_reader = reader
        .read_mesh(
            &mut read_points,
            &mut read_connectivity,
            &mut read_cell_types,
        )
        .unwrap();

    let index = data_reader.point_data_index(0, "pressure").unwrap();
    let mut into: Vec<f32> = Vec::new();
    let err = data_reader
        .read_point_data(0, index, &mut into)
        .unwrap_err();
    std::assert_matches!(err, xdmf::Error::InvalidData { .. });
}

#[test]
fn out_of_bounds_step_is_invalid_data() {
    let tmp_dir = TempDir::new().unwrap();
    let path = tmp_dir.path().join("out");

    let points = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let connectivity = [0_u64, 1, 2];
    let cell_types = [CellType::Triangle];

    let reader = TimeSeriesWriter::new(&path, DataStorage::AsciiInline)
        .unwrap()
        .write_mesh(&points, &connectivity, &cell_types)
        .unwrap();
    drop(reader);

    let reader = TimeSeriesReader::new(path.with_extension("xdmf2")).unwrap();
    let mut read_points = Vec::new();
    let mut read_connectivity = Vec::new();
    let mut read_cell_types = Vec::new();
    let data_reader = reader
        .read_mesh(
            &mut read_points,
            &mut read_connectivity,
            &mut read_cell_types,
        )
        .unwrap();

    let err = data_reader.num_point_data(0).unwrap_err();
    std::assert_matches!(err, xdmf::Error::InvalidData { reason } if reason.contains("out of bounds"));
}
