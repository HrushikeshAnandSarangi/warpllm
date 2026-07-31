//! Client configuration.

use serde::Deserialize;

/// Matches the OpenAI SDK's default request timeout.
pub(crate) const DEFAULT_TIMEOUT_SECS: u64 = 600;

/// Holds no API key. Credentials resolve at request time from the environment,
/// once routing has picked a model and its spec names the variable to read — so
/// a client is never asked up front for keys a given request will not use.
///
/// The environment is deliberately the only source for now. Carrying keys here
/// is this struct's job if an embedder that keeps them elsewhere turns up; until
/// one does, there is nothing to override.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientConfig {
    /// Overrides the provider's default base URL (proxies, tests). Absent
    /// means each provider talks to its own API.
    pub base_url: Option<String>,
    pub timeout_secs: Option<u64>,
}
