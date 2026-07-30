//! Client configuration.

use serde::Deserialize;

/// Matches the OpenAI SDK's default request timeout.
pub(crate) const DEFAULT_TIMEOUT_SECS: u64 = 600;

/// Holds no API key today. Credentials resolve at request time, once routing
/// has picked a model and its spec names the variable to read — so a client is
/// never asked up front for keys a given request will not use. An embedder
/// keeping keys outside the environment supplies one via
/// [`crate::Client::with_api_key`]; carrying several provider keys at once is
/// this struct's job as it grows.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientConfig {
    /// Overrides the provider's default base URL (proxies, tests). Absent
    /// means each provider talks to its own API.
    pub base_url: Option<String>,
    pub timeout_secs: Option<u64>,
}
