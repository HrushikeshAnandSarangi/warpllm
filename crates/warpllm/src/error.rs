use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid model string '{given}': no registered model spec")]
    InvalidModel { given: String },
    #[error("{}", missing_api_key_message(provider, *env_var))]
    MissingApiKey {
        provider: &'static str,
        /// The variable to set, when the registry names one for this model.
        ///
        /// `None` is not an oversight: an entry may declare no
        /// `env_api_key` at all, meaning it authenticates only with a key the
        /// caller supplies. The wire form keeps one `missing_api_key` code
        /// either way — the failure is the same, only the remedy differs — and
        /// carries `env_var: null` for that case.
        env_var: Option<&'static str>,
    },
    #[error("{provider} returned HTTP {status}: {message}")]
    Provider {
        provider: &'static str,
        status: u16,
        error_type: Option<String>,
        message: String,
    },
    #[error("network error talking to {provider}: {source}")]
    Network {
        provider: &'static str,
        #[source]
        source: reqwest::Error,
    },
    #[error("could not decode {provider} response: {message}")]
    Decode {
        provider: &'static str,
        message: String,
    },
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
}

/// A function rather than two `#[error]` strings because the REMEDY differs
/// while the failure does not. Naming a variable that no spec declares would
/// send someone off to set an environment variable nothing ever reads.
fn missing_api_key_message(provider: &str, env_var: Option<&str>) -> String {
    match env_var {
        Some(var) => format!("missing API key for {provider}: set the {var} environment variable"),
        None => format!(
            "missing API key for {provider}: it names no default environment variable, \
             so the key has to be passed explicitly"
        ),
    }
}

impl Error {
    /// Stable machine-readable form for the FFI boundary: `code` plus typed
    /// fields, so language wrappers can raise typed exceptions without
    /// parsing display strings. The `code` values are a public contract.
    pub fn to_wire_json(&self) -> String {
        let value = match self {
            Error::InvalidInput(_) | Error::InvalidModel { .. } => {
                json!({"code": "invalid_request", "message": self.to_string()})
            }
            Error::MissingApiKey { provider, env_var } => json!({
                "code": "missing_api_key",
                "provider": provider,
                "env_var": env_var,
                "message": self.to_string(),
            }),
            Error::Provider {
                provider,
                status,
                error_type,
                ..
            } => json!({
                "code": "provider_error",
                "provider": provider,
                "status": status,
                "error_type": error_type,
                "message": self.to_string(),
            }),
            Error::Network { provider, .. } => json!({
                "code": "connection_error",
                "provider": provider,
                "message": self.to_string(),
            }),
            Error::Decode { provider, .. } => json!({
                "code": "decode_error",
                "provider": provider,
                "message": self.to_string(),
            }),
            Error::NotImplemented(_) => {
                json!({"code": "not_implemented", "message": self.to_string()})
            }
        };
        value.to_string()
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    fn wire(err: &Error) -> serde_json::Value {
        serde_json::from_str(&err.to_wire_json()).unwrap()
    }

    #[test]
    fn provider_error_wire_format() {
        let v = wire(&Error::Provider {
            provider: "openai",
            status: 429,
            error_type: Some("rate_limit_exceeded".into()),
            message: "slow down".into(),
        });
        assert_eq!(v["code"], "provider_error");
        assert_eq!(v["provider"], "openai");
        assert_eq!(v["status"], 429);
        assert_eq!(v["error_type"], "rate_limit_exceeded");
        assert!(v["message"].as_str().unwrap().contains("HTTP 429"));
    }

    #[test]
    fn missing_key_wire_format() {
        let v = wire(&Error::MissingApiKey {
            provider: "openai",
            env_var: Some("OPENAI_API_KEY"),
        });
        assert_eq!(v["code"], "missing_api_key");
        assert_eq!(v["env_var"], "OPENAI_API_KEY");
    }

    #[test]
    fn not_implemented_wire_format() {
        let v = wire(&Error::NotImplemented("streaming"));
        assert_eq!(v["code"], "not_implemented");
        assert!(v["message"].as_str().unwrap().contains("streaming"));
    }
}
