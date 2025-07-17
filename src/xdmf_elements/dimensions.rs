use serde::Serialize;

#[derive(Clone, Debug, PartialEq)]
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

#[cfg(test)]
mod tests {
    use quick_xml::se::to_string;

    use super::*;

    #[test]
    fn dimensions_serialize() {
        #[derive(Serialize)]
        pub(crate) struct XmlRoot<T>
        where
            T: Serialize,
        {
            #[serde(rename = "$value")]
            content: T,
        }

        let dimensions = XmlRoot {
            content: Dimensions(vec![2, 3, 4]),
        };
        assert_eq!(to_string(&dimensions).unwrap(), "<XmlRoot>2 3 4</XmlRoot>");
    }

    #[test]
    fn dimensions_compare() {
        let dimensions1 = Dimensions(vec![2, 3, 4]);
        let dimensions2 = Dimensions(vec![2, 3, 4]);
        let dimensions3 = Dimensions(vec![1, 2, 3]);

        assert_eq!(dimensions1, dimensions2);
        assert_ne!(dimensions1, dimensions3);
    }
}
