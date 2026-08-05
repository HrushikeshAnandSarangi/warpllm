//! The shared JSON boundary used by foreign-language bindings.
//!
//! PyO3 and napi-rs should expose Rust, not each maintain the same serde
//! adapter. Keeping that adapter here makes both native modules mechanical.

use crate::{Client, ClientConfig, Error, Result};

/// A [`Client`] whose inputs and outputs are JSON strings.
///
/// This is intentionally small: ownership and async-runtime integration stay
/// language-specific, while parsing, validation, dispatch, and serialization
/// happen once in the core.
pub struct JsonClient {
    inner: Client,
}

impl JsonClient {
    pub fn new(config_json: &str) -> Result<Self> {
        let config: ClientConfig = serde_json::from_str(config_json)
            .map_err(|error| Error::InvalidInput(error.to_string()))?;
        Ok(Self {
            inner: Client::new(config)?,
        })
    }

    pub async fn chat_completion(&self, request_json: &str) -> Result<String> {
        let request = serde_json::from_str(request_json)
            .map_err(|error| Error::InvalidInput(error.to_string()))?;
        let response = self.inner.chat_completion(request).await?;
        serde_json::to_string(&response).map_err(|error| Error::Internal(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_config_is_a_core_error() {
        let error = JsonClient::new(r#"{"unknown": true}"#)
            .err()
            .expect("invalid configuration should fail");
        assert!(matches!(error, Error::InvalidInput(_)));
    }
}
