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
pub use error::{Error, Result};
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

/// Round-trips a message through the async runtime. Placeholder for real
/// gateway calls; the sleep forces a genuine suspension point so bindings
/// exercise their runtime bridge instead of a resolved-future fast path.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] if `msg` is empty.
///
/// # Examples
///
/// ```
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let reply = warpllm::echo("ping").await?;
/// assert_eq!(reply, "ping");
/// # Ok::<(), warpllm::Error>(())
/// # }).unwrap();
/// ```
pub async fn echo(msg: &str) -> Result<String> {
    if msg.is_empty() {
        return Err(Error::InvalidInput("message must not be empty".into()));
    }
    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    Ok(msg.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_manifest() {
        assert_eq!(version(), env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn echo_round_trips() {
        assert_eq!(echo("hi").await.unwrap(), "hi");
    }

    #[tokio::test]
    async fn echo_rejects_empty() {
        assert!(matches!(echo("").await, Err(Error::InvalidInput(_))));
    }
}
