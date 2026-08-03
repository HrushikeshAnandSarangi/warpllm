//! The gateway's canonical form for an upstream FAILURE.
//!
//! A provider answers with one of two things, and both are wire shapes that
//! have to be normalized: a success body, whose canonical form is
//! [`ChatResponse`](super::ChatResponse), and an error body, whose canonical
//! form is here. Same contract on both — converting in is lossless, and
//! everything the provider said is retained beside what warpllm made of it.
//!
//! What the failure MEANS is not here: it is the [`Error`](crate::Error)
//! variant this payload is wrapped in. One flat enum names every way a call
//! can fail, warpllm's own reasons and the provider's alike, so a caller
//! writes one `match` instead of a match inside a match.
//!
//! This is the only PUBLIC type in `gateway::types`. The request and
//! response forms next door are internal because their serde shape is an
//! unstable implementation detail; a plain struct of owned strings has no
//! such problem, and callers need it to diagnose a failure. The module tree
//! stays private either way — `crate::gateway` is `mod`, not `pub mod`, so
//! this reaches the outside only through the re-export in `lib.rs`, exactly
//! as the registry's specs do.

use std::fmt;
use std::time::Duration;

/// The longest upstream error body kept verbatim on a [`ProviderError`].
///
/// Bounded because an error body is provider-controlled and occasionally an
/// HTML page or a stack trace; unbounded, it would ride every error through
/// the FFI boundary. 2 KiB holds every real JSON envelope with room to spare.
const MAX_RAW_BODY: usize = 2_048;

/// Everything the provider said, carried by every provider-driven
/// [`Error`](crate::Error) variant.
///
/// PROMOTE BUT RETAIN, the same contract `FinishReason` follows next door.
/// The variant is what the failure means; this is the evidence it was read
/// from. A classification is a lossy summary, so the exact `code` that
/// produced it, the raw envelope it came in, and the header evidence that
/// lives nowhere in the body at all are all kept — a caller diagnosing an
/// unusual failure, or reporting one to a provider's support desk, needs the
/// thing the provider said, not warpllm's reading of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderError {
    /// Which upstream answered.
    pub provider: &'static str,
    /// The HTTP status it answered with, passed through unchanged.
    pub status: u16,
    /// The provider's own `message`, or the whole body when nothing parsed.
    pub message: String,
    /// The provider's `type` — a failure FAMILY, verbatim.
    pub error_type: Option<String>,
    /// The provider's `code` — the specific failure, verbatim. Usually the
    /// strongest evidence in the envelope, and the field the classifier
    /// prefers, so losing it would discard exactly what the answer was read
    /// from.
    pub provider_code: Option<String>,
    /// `Retry-After`, when the provider sent one in delta-seconds form.
    ///
    /// Reported as what the PROVIDER said, carrying no claim from warpllm
    /// that waiting will help. Providers stamp this on 429s fairly
    /// indiscriminately, so it can well arrive on a failure no delay
    /// resolves — a caller decides what to make of it.
    pub retry_after: Option<Duration>,
    /// The upstream's own request identifier, for correlating with a
    /// provider's logs.
    pub request_id: Option<String>,
    /// The error body verbatim, truncated to [`MAX_RAW_BODY`]. The last
    /// resort when a provider ships a failure warpllm has no vocabulary
    /// for.
    pub raw_body: String,
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} returned HTTP {}: {}",
            self.provider, self.status, self.message
        )
    }
}

impl ProviderError {
    /// Truncates on a character boundary, so a multi-byte body cannot
    /// panic the error path.
    pub(crate) fn bounded(body: &str) -> String {
        match body.char_indices().nth(MAX_RAW_BODY) {
            None => body.to_string(),
            Some((end, _)) => format!("{}…[truncated]", &body[..end]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ProviderError {
        ProviderError {
            provider: "openai",
            status: 429,
            message: "slow down".into(),
            error_type: None,
            provider_code: None,
            retry_after: None,
            request_id: None,
            raw_body: String::new(),
        }
    }

    #[test]
    fn display_names_the_provider_and_status() {
        assert_eq!(sample().to_string(), "openai returned HTTP 429: slow down");
    }

    /// A body big enough to matter is truncated, not dropped — and the
    /// truncation is marked so nobody reads a cut envelope as a whole one.
    #[test]
    fn a_huge_raw_body_is_bounded_and_marked() {
        let bounded = ProviderError::bounded(&"x".repeat(10_000));
        assert!(bounded.len() < 10_000);
        assert!(bounded.starts_with("xxx"));
        assert!(bounded.ends_with("[truncated]"), "{bounded}");
    }

    /// Truncation lands on a character boundary: a multi-byte body must not
    /// panic the error path.
    #[test]
    fn bounding_a_multibyte_body_does_not_panic() {
        let bounded = ProviderError::bounded(&"é".repeat(4_000));
        assert!(bounded.ends_with("[truncated]"));
        // A short one is returned whole, with no marker.
        assert_eq!(ProviderError::bounded("héllo"), "héllo");
    }
}
