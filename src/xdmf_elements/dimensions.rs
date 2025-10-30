//! Implementation of the `Dimensions` struct and its serialization, used to specify the shape of data arrays.

use serde::{Deserialize, Serialize};

/// Represents the dimensions of a data array in XDMF format.
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

impl<'de> Deserialize<'de> for Dimensions {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let dimensions = s
            .split_whitespace()
            .map(|n| n.parse().map_err(serde::de::Error::custom))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self(dimensions))
    }
}

#[cfg(test)]
mod tests {
    use quick_xml::se::to_string;

    use super::*;

    #[test]
    fn dimensions_serialize() {
        #[derive(Serialize)]
        struct XmlRoot<T>
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

    #[test]
    fn dimensions_roundtrip() {
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct XmlRoot {
            #[serde(rename = "$value")]
            content: Dimensions,
        }

        let original = XmlRoot {
            content: Dimensions(vec![2, 3, 4]),
        };

        // Serialize to XML
        let xml = to_string(&original).unwrap();
        assert_eq!(xml, "<XmlRoot>2 3 4</XmlRoot>");

        // Deserialize back from XML
        let deserialized: XmlRoot = quick_xml::de::from_str(&xml).unwrap();
        assert_eq!(deserialized, original);
        assert_eq!(deserialized.content.0, vec![2, 3, 4]);
    }

    #[test]
    fn dimensions_deserialize_single_value() {
        #[derive(Deserialize, PartialEq, Debug)]
        struct XmlRoot {
            #[serde(rename = "$value")]
            content: Dimensions,
        }

        let xml = "<XmlRoot>42</XmlRoot>";
        let deserialized: XmlRoot = quick_xml::de::from_str(xml).unwrap();
        assert_eq!(deserialized.content.0, vec![42]);
    }

    #[test]
    fn dimensions_deserialize_multiple_values() {
        #[derive(Deserialize, PartialEq, Debug)]
        struct XmlRoot {
            #[serde(rename = "$value")]
            content: Dimensions,
        }

        let xml = "<XmlRoot>10 20 30 40</XmlRoot>";
        let deserialized: XmlRoot = quick_xml::de::from_str(xml).unwrap();
        assert_eq!(deserialized.content.0, vec![10, 20, 30, 40]);
    }
}
