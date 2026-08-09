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
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn read_u32_le(path: &std::path::Path) -> Vec<u32> {
        let bytes = std::fs::read(path).unwrap();
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    // 4 points, 2 triangles sharing an edge
    let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let connectivity = [0_u64, 1, 2, 0, 2, 3];
    let cell_types = [xdmf::CellType::Triangle, xdmf::CellType::Triangle];

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer = TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::Binary).unwrap();

    let mut xdmf_writer = xdmf_writer
        .write_mesh(&coords, (&connectivity, &cell_types))
        .unwrap();

    xdmf_writer
        .write_data(
            "0",
            [(
                "temperature",
                xdmf::DataAttribute::Scalar,
                vec![10.0, 11.0, 12.0, 13.0].into(),
            )],
            [(
                "region_id",
                xdmf::DataAttribute::Scalar,
                vec![100_u64, 200].into(),
            )],
        )
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
        vec![4, 0, 1, 2, 4, 0, 2, 3] // cell type tag (4=Triangle) prefixed per cell
    );

    // Time-step attribute data.
    assert_eq!(
        read_f64_le(&bin_dir.join("data_t_0_point_data_temperature.bin")),
        vec![10.0, 11.0, 12.0, 13.0]
    );
    assert_eq!(
        read_u32_le(&bin_dir.join("data_t_0_cell_data_region_id.bin")),
        vec![100, 200]
    );
}

#[test]
fn binary_write_data_rejects_u64_too_large_for_u32() {
    // binary format is only using 32-bit integers, due to a bug in the paraview reader that misreads 64-bit integers
    let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let connectivity = [0_u64, 1, 2];
    let cell_types = [xdmf::CellType::Triangle];

    let tmp_dir = TempDir::new().unwrap();
    let xdmf_file_path = tmp_dir.path().join("test_output");

    let xdmf_writer = TimeSeriesWriter::new(&xdmf_file_path, xdmf::DataStorage::Binary).unwrap();
    let mut xdmf_writer = xdmf_writer
        .write_mesh(&coords, (&connectivity, &cell_types))
        .unwrap();

    let res = xdmf_writer.write_data(
        "0",
        [],
        [(
            "region_id",
            xdmf::DataAttribute::Scalar,
            vec![u64::from(u32::MAX) + 1].into(),
        )],
    );
    assert!(
        res.unwrap_err()
            .to_string()
            .contains("does not fit in 32 bits")
    );
}
