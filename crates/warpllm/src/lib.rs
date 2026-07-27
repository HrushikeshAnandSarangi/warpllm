//! Core engine for warpllm, a warp-speed, robust AI gateway.

mod client;
mod config;
mod error;
mod http;
mod normalized;
mod protocol;
mod registry;

pub mod types;

pub use client::Client;
pub use config::ClientConfig;
pub use error::{Error, Result};
pub use protocol::Protocol;
/// The registry's public face. `registry` itself stays private: it is where
/// the roster, the resolution, and the lookup live, and none of that is API.
///
/// Read-only by construction — every field is private and there is no public
/// constructor, so [`model_spec`] is the way to get one. The one exception is
/// [`serde::Deserialize`], which these types derive because the same
/// declaration is the registry's YAML schema; a spec deserialized from
/// anything other than the registry can be missing fields the accessors read
/// unconditionally, and those accessors will panic. Go through
/// [`model_spec`].
pub use registry::{Capabilities, ModelSpec, model_spec};
pub use types::openai_compat::chat::completions::*;

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
