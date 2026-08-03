//! Shared HTTP helpers used by every provider implementation.

use std::time::Duration;

use crate::error::{Error, Result};

/// Header spellings for the upstream's own request identifier, in the order
/// they are tried.
///
/// Two entries, both de-facto standards (OpenAI sends the first, Anthropic
/// the second), and the list stays short ON PURPOSE. A long table of
/// per-provider header names is the staleness liability the classifier
/// already refuses to take on — a spelling nobody here knows costs a `None`
/// on a diagnostic field, never a wrong answer.
const REQUEST_ID_HEADERS: [&str; 2] = ["x-request-id", "request-id"];

/// Maps a transport-level reqwest error to [`Error::Network`].
pub(crate) fn network_error(provider: &'static str, source: reqwest::Error) -> Error {
    Error::Network { provider, source }
}

/// A response read to the end: its status, its body, and the header
/// evidence that would otherwise be dropped on the floor.
///
/// The headers are read HERE rather than at the conversion layer because
/// this is the last place they exist — a body and a status alone cannot
/// answer "how long should I wait" or "what do I quote to the provider's
/// support desk".
pub(crate) struct ResponseParts {
    pub status: u16,
    pub body: String,
    /// `Retry-After`, when the provider sent one in delta-seconds form.
    pub retry_after: Option<Duration>,
    /// The upstream's own request identifier, for correlating with a
    /// provider's logs.
    pub request_id: Option<String>,
}

/// Reads the response body and the headers worth keeping, mapping read
/// failures to [`Error::Network`].
pub(crate) async fn read_response(
    provider: &'static str,
    response: reqwest::Response,
) -> Result<ResponseParts> {
    let status = response.status().as_u16();
    let headers = response.headers().clone();
    let body = response
        .text()
        .await
        .map_err(|e| network_error(provider, e))?;
    let header = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
    Ok(ResponseParts {
        status,
        body,
        retry_after: header("retry-after").and_then(parse_retry_after),
        request_id: REQUEST_ID_HEADERS
            .iter()
            .find_map(|name| header(name))
            .map(str::to_string),
    })
}

/// Parses `Retry-After`'s delta-seconds form.
///
/// RFC 9110 permits an HTTP-date as well, and this deliberately does NOT
/// parse it: that would need a date parser, and every provider warpllm
/// speaks to sends seconds. A date form yields `None` — the header is still
/// on the response, and a caller that gets `None` waits on its own policy
/// rather than on a misparsed instant.
fn parse_retry_after(value: &str) -> Option<Duration> {
    value.trim().parse::<u64>().ok().map(Duration::from_secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_after_reads_delta_seconds() {
        assert_eq!(parse_retry_after("30"), Some(Duration::from_secs(30)));
        assert_eq!(parse_retry_after("  30 "), Some(Duration::from_secs(30)));
        assert_eq!(parse_retry_after("0"), Some(Duration::ZERO));
    }

    /// The date form and anything malformed are `None`, never a guess — a
    /// misparsed instant is worse than no instant.
    #[test]
    fn retry_after_declines_forms_it_cannot_read() {
        for value in ["Wed, 21 Oct 2015 07:28:00 GMT", "soon", "", "-1", "1.5"] {
            assert_eq!(parse_retry_after(value), None, "{value}");
        }
    }
}
