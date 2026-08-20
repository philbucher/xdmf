//! End-to-end check for `DataStorage::Binary`: writes a small mesh + time series to raw binary
//! files, then independently re-parses both the XDMF metadata (`Format="Binary"`,
//! `Endian="Little"`, `Precision`) and the raw bytes on disk, verifying they decode back to
//! exactly the values that were written.

use temp_dir::TempDir;
use xdmf::TimeSeriesWriter;

#[test]
fn write_and_verify_binary() {
    fn read_f64_le(path: &std::path::Path) -> Vec<f64> {
        let bytes = std::fs::read(path).unwrap();
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect()
    }

    fn read_u32_le(path: &std::path::Path) -> Vec<u32> {
        let bytes = std::fs::read(path).unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect()
    }

    // 4 points, 2 triangles sharing an edge
    let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let connectivity = [0_u32, 1, 2, 0, 2, 3];
    let cell_types = [xdmf::CellType::Triangle, xdmf::CellType::Triangle];

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer = TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::Binary).unwrap();

    let mut xdmf_writer = xdmf_writer
        .write_mesh(&coords, &connectivity, &cell_types)
        .unwrap();

    xdmf_writer
        .write_time_step("0", |step| {
            step.point_data(
                "temperature",
                xdmf::DataAttribute::Scalar,
                vec![10.0, 11.0, 12.0, 13.0],
            )?;
            step.cell_data("region_id", xdmf::DataAttribute::Scalar, vec![100_u32, 200])
        })
        .unwrap();

    let xdmf_file = xdmf_file_path.with_extension("xdmf2");
    let xdmf_xml = std::fs::read_to_string(&xdmf_file).unwrap();

    // Metadata a reader needs to interpret the raw bytes correctly.
    assert!(xdmf_xml.contains(r#"Format="Binary""#));
    assert!(xdmf_xml.contains(r#"Endian="Little""#));
    assert!(!xdmf_xml.contains('\\'), "paths must use forward slashes");
    // UInt data (connectivity, region_id) must declare 4-byte precision...
    assert!(xdmf_xml.contains(r#"NumberType="UInt" Format="Binary" Precision="4""#));
    // ...while Float data (coords, temperature) keeps the natural 8-byte precision.
    assert!(xdmf_xml.contains(r#"NumberType="Float" Format="Binary" Precision="8""#));

    let bin_dir = xdmf_file_path.with_extension("bin");

    // Mesh geometry/topology, written once.
    assert_eq!(read_f64_le(&bin_dir.join("points.bin")), coords.to_vec());
    assert_eq!(
        read_u32_le(&bin_dir.join("cells.bin")),
        // every cell is a Triangle, so the type is carried once as a uniform `TopologyType`
        // rather than being prefixed per cell
        vec![0, 1, 2, 0, 2, 3]
    );

    // Time-step attribute data, named by the order it was written in -- "temperature" is the
    // step's first array, "region_id" its second.
    assert_eq!(
        read_f64_le(&bin_dir.join("data_t_0_0.bin")),
        vec![10.0, 11.0, 12.0, 13.0]
    );
    assert_eq!(read_u32_le(&bin_dir.join("data_t_0_1.bin")), vec![100, 200]);
}

#[test]
fn binary_rejects_64_bit_integer_data() {
    // ParaView reads 64-bit integers in `Format="Binary"` at the wrong stride, so the storage
    // refuses the *type* rather than narrowing it behind the caller's back: it would otherwise put
    // an i32 in the file where an i64 was passed. The error says which storages do take it.
    let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let connectivity = [0_u32, 1, 2];
    let cell_types = [xdmf::CellType::Triangle];

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer = TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::Binary).unwrap();
    let mut xdmf_writer = xdmf_writer
        .write_mesh(&coords, &connectivity, &cell_types)
        .unwrap();

    // the attribute's error propagates out of the closure, so the step is never written
    // rejected whatever it holds, so even a value that would fit in an i32 does not get through
    for value in [i64::from(i32::MAX) + 1, 7_i64] {
        let res = xdmf_writer.write_time_step("0", |step| {
            step.cell_data("region_id", xdmf::DataAttribute::Scalar, vec![value])
        });
        std::assert_matches!(
            res.unwrap_err(),
            xdmf::Error::InvalidData { reason }
                if reason.contains("cannot hold i64 data")
                    && reason.contains("use another DataStorage"),
            "Binary must refuse an i64 of {value}"
        );
    }

    // ...and u64 goes the same way
    let res = xdmf_writer.write_time_step("0", |step| {
        step.cell_data("region_id", xdmf::DataAttribute::Scalar, vec![7_u64])
    });
    std::assert_matches!(
        res.unwrap_err(),
        xdmf::Error::InvalidData { reason } if reason.contains("cannot hold u64 data")
    );

    // the rejected value is caught before any file for this step is opened, so nothing is left
    // on disk that the XDMF file doesn't reference
    let bin_dir = xdmf_file_path.with_extension("bin");
    assert!(!bin_dir.join("data_t_0_0.bin").exists());

    // the writer must not be left poisoned by the failed step: a following valid step succeeds
    xdmf_writer
        .write_time_step("1", |step| {
            step.cell_data("region_id", xdmf::DataAttribute::Scalar, vec![7_u32])
        })
        .unwrap();
}

#[test]
fn write_and_verify_binary_signed_integers() {
    fn read_i32_le(path: &std::path::Path) -> Vec<i32> {
        let bytes = std::fs::read(path).unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect()
    }

    // 3 points forming a triangle
    let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let connectivity = [0_u32, 1, 2];
    let cell_types = [xdmf::CellType::Triangle];

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer = TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::Binary).unwrap();
    let mut xdmf_writer = xdmf_writer
        .write_mesh(&coords, &connectivity, &cell_types)
        .unwrap();

    xdmf_writer
        .write_time_step("0", |step| {
            step.point_data("level_i32", xdmf::DataAttribute::Scalar, vec![-2_i32, 0, 2])?;
            step.point_data(
                "flag_u32",
                xdmf::DataAttribute::Scalar,
                vec![0_u32, 1, u32::MAX],
            )
        })
        .unwrap();

    // the 32-bit types are what this storage carries, at their own width and nothing else
    let xdmf_xml = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();
    assert_eq!(
        xdmf_xml
            .matches(r#"NumberType="Int" Format="Binary" Precision="4""#)
            .count(),
        1
    );
    assert_eq!(
        xdmf_xml
            .matches(r#"NumberType="UInt" Format="Binary" Precision="4""#)
            .count(),
        2, // the connectivity is u32 here too
    );

    let bin_dir = xdmf_file_path.with_extension("bin");
    assert_eq!(read_i32_le(&bin_dir.join("data_t_0_0.bin")), vec![-2, 0, 2]);
    assert_eq!(
        std::fs::read(bin_dir.join("data_t_0_1.bin")).unwrap().len(),
        12,
        "3 values at 4 bytes each, with nothing widened or narrowed on the way out"
    );
}

#[test]
fn write_and_verify_binary_f32() {
    fn read_f32_le(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    // same mesh as `write_and_verify_binary`, but held as f32 by the caller
    let coords: [f32; 12] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let connectivity = [0_u32, 1, 2, 0, 2, 3];
    let cell_types = [xdmf::CellType::Triangle, xdmf::CellType::Triangle];

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer = TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::Binary).unwrap();

    let mut xdmf_writer = xdmf_writer
        .write_mesh(&coords, &connectivity, &cell_types)
        .unwrap();

    xdmf_writer
        .write_time_step("0", |step| {
            step.point_data(
                "temperature",
                xdmf::DataAttribute::Scalar,
                vec![10.5_f32, 11.5, 12.5, 13.5],
            )
        })
        .unwrap();

    let xdmf_xml = std::fs::read_to_string(xdmf_file_path.with_extension("xdmf2")).unwrap();

    // 4-byte precision is what tells the reader how wide the raw floats on disk are, so it has to
    // follow the caller's type rather than the format
    assert!(xdmf_xml.contains(r#"NumberType="Float" Format="Binary" Precision="4""#));
    assert!(!xdmf_xml.contains(r#"NumberType="Float" Format="Binary" Precision="8""#));

    let bin_dir = xdmf_file_path.with_extension("bin");

    let points = bin_dir.join("points.bin");
    // half the bytes of the equivalent f64 mesh, and the values survive the round trip exactly
    // (all of them are representable in f32)
    assert_eq!(std::fs::metadata(&points).unwrap().len(), 12 * 4);
    float_cmp::assert_approx_eq!(&[f32], &read_f32_le(&points), &coords);

    float_cmp::assert_approx_eq!(
        &[f32],
        &read_f32_le(&bin_dir.join("data_t_0_0.bin")),
        &[10.5, 11.5, 12.5, 13.5]
    );
}
