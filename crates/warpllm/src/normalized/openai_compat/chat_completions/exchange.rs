//! The upstream half of a chat completion: normalized in, normalized out.

use crate::error::Result;
use crate::normalized;
use crate::protocol::openai_compat::chat_completions::transport::{self, Outcome};
use crate::protocol::openai_compat::error_from_body;

use super::{ingest_response, render_request};

/// Renders the normalized request for this dialect, posts it, and ingests the
/// reply — the one place this protocol's order is stated.
///
/// Normalized on both ends, deliberately: that is what lets every protocol
/// implement this same signature, so `client.rs` gains one match arm per
/// protocol and nothing else. The caller-side conversions stay with the client,
/// since they answer to the dialect warpllm was *called* with, not this one.
///
/// Stateless, taking the transport context loose rather than borrowing a client:
/// nothing here needs to outlive the call.
///
/// Error mapping happens here rather than in the transport because which
/// [`crate::Error`] a status becomes is a protocol-and-provider decision, while
/// reading the socket is not.
pub(crate) async fn exchange(
    request: &normalized::ChatRequest,
    http: &reqwest::Client,
    provider: &'static str,
    base_url: &str,
    api_key: &str,
) -> Result<normalized::ChatResponse> {
    let wire = render_request(request, provider)?;
    match transport::post(http, provider, base_url, api_key, &wire).await? {
        Outcome::Ok(response) => Ok(ingest_response(response)),
        Outcome::Status { status, body } => Err(error_from_body(provider, status, &body)),
    }
}
