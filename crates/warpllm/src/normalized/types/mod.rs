//! Shared primitives for the normalized types.

pub(crate) mod message;
pub(crate) mod request;
pub(crate) mod response;

pub(crate) use message::*;
pub(crate) use request::*;
pub(crate) use response::*;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocol::Protocol;

/// Namespaced passthrough bags, keyed by dialect name (`"openai_compat"`)
/// or provider name (`"deepseek"`). Renderers emit the dialect namespace
/// only when the target speaks that dialect, then overlay the target
/// provider's namespace — a field meant for one provider never leaks into
/// another. Provider names shadowing dialect names are reserved.
pub(crate) type ProviderExt = BTreeMap<String, Value>;

/// Exact JSON text, never a parsed tree at rest.
///
/// Tool arguments stay raw because key order is load-bearing: some
/// providers' prompt caches hash the request bytes, and a parse/reserialize
/// round trip can also destroy the not-quite-valid JSON models sometimes
/// emit. Parse on demand; re-serialize only from the raw form.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct RawJson(String);

impl RawJson {
    pub(crate) fn new(raw: impl Into<String>) -> Self {
        Self(raw.into())
    }

    /// For ingest from providers that send objects: serialize ONCE at the
    /// boundary (preserve_order keeps the provider's key order).
    #[allow(dead_code)] // staged: used by the first object-arguments dialect
    pub(crate) fn from_value(value: &Value) -> Self {
        Self(value.to_string())
    }

    #[allow(dead_code)] // staged: used once typed tools render
    pub(crate) fn parse(&self) -> Result<Value, serde_json::Error> {
        if self.0.trim().is_empty() {
            return Ok(Value::Object(Default::default()));
        }
        serde_json::from_str(&self.0)
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

/// Where a normalized value came from. Retained so adapters can one day
/// take a same-dialect passthrough fast path, and for diffing what was
/// received against what was sent. Not part of the canonical JSON form.
#[derive(Debug, Clone)]
#[allow(dead_code)] // staged: read once the passthrough fast path lands
pub(crate) struct IngestSource {
    /// The wire format this was ingested from. [`Protocol`] rather than a
    /// separate `Dialect` enum: the name that keys the `ext` bags, the name the
    /// registry YAML uses, and the name of the format are one string, so they
    /// are one type.
    pub protocol: Protocol,
    /// The inbound body as ingested — re-serialized from the typed wire
    /// form, NOT the original bytes: unknown-field order survives
    /// (preserve_order), but whitespace, numeric spelling, and typed-field
    /// order are canonicalized. Byte-exact capture needs the gateway
    /// boundary's raw bytes and lands with the passthrough fast path.
    pub body: Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_json_round_trips_and_parses() {
        let raw = RawJson::new(r#"{"z":1,"a":2}"#);
        assert_eq!(raw.as_str(), r#"{"z":1,"a":2}"#);
        assert_eq!(raw.parse().unwrap()["z"], 1);
        // from_value keeps the source order (preserve_order).
        let value = serde_json::json!({"z": 1, "a": 2});
        assert_eq!(RawJson::from_value(&value).as_str(), r#"{"z":1,"a":2}"#);
    }

    #[test]
    fn raw_json_empty_parses_to_empty_object() {
        assert_eq!(
            RawJson::new("").parse().unwrap(),
            Value::Object(Default::default())
        );
    }

    #[test]
    fn raw_json_invalid_errors_on_parse_only() {
        let raw = RawJson::new(r#"{"almost": json"#);
        assert_eq!(raw.as_str(), r#"{"almost": json"#);
        assert!(raw.parse().is_err());
    }
}
