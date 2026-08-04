//! Core engine for warpllm, a warp-speed, robust AI gateway.

mod client;
mod config;
mod error;
mod gateway;
mod http;
mod registry;

pub mod protocol;
pub mod types;

pub use client::Client;
pub use config::ClientConfig;
pub use error::{Error, Origin, Result};
/// The gateway's canonical form for an upstream failure — the error-side
/// counterpart to the request and response forms, and the payload every
/// provider-driven [`Error`] variant carries. `gateway` itself stays private,
/// exactly as `registry` does; this is its only public shape.
pub use gateway::types::ProviderError;
pub use protocol::openai_compat::chat_completions::types::*;
/// The registry's public face: a provider, a model, and the lookup that hands
/// back one of each. `registry` itself stays private — the roster, the schema,
/// and the loading are not API.
///
/// Read-only by construction: every field is private and there is no public
/// constructor, so [`fetch_model`] is the way to obtain either half.
pub use registry::{Capabilities, ModelSpec, ProviderSpec, fetch_model};
pub use types::{Api, Protocol};

/// Returns the warpllm version.
///
/// ```
/// let version = warpllm::version();
/// assert_eq!(version.split('.').count(), 3);
/// ```
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_manifest() {
        assert_eq!(version(), env!("CARGO_PKG_VERSION"));
    }
}
