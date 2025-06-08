use serde::Serialize;

#[derive(Debug)]
pub struct Dimensions(pub Vec<usize>);

impl Serialize for Dimensions {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let s = self
            .0
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        serializer.serialize_str(&s)
    }
}
