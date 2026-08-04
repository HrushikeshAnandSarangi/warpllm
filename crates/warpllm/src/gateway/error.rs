//! What an upstream failure MEANS, and the seam that lets one provider
//! differ from the protocol it speaks.
//!
//! Provider-agnostic, and deliberately so: this module names no protocol and
//! no provider. It owns the VOCABULARY LOOKUP as a trait and the ORDER those
//! lookups run in, and nothing else. Extracting `type`/`code`/`message` out
//! of a body is a protocol's job, because only a protocol knows the shape of
//! its error envelope.
//!
//! Three layers meet here, and each knows strictly less than the one below:
//!
//! 1. [`Error`] and [`ProviderError`] — the canonical failures. No protocol,
//!    no provider.
//! 2. A PROTOCOL's vocabulary — the spellings every backend speaking it
//!    shares. `openai_compat` next door.
//! 3. A PROVIDER's vocabulary — only where that provider genuinely diverges
//!    from its protocol.
//!
//! A provider with nothing to add implements nothing, and inherits its
//! protocol whole. That is what keeps a new roster entry a YAML edit: adding
//! a backend that speaks `openai_compat` faithfully needs no Rust at all, and
//! cannot silently lose the taxonomy by forgetting to write some.

use crate::error::Error;
use crate::gateway::types::ProviderError;

/// The classifier's answer: the [`Error`] variant this failure IS.
///
/// A variant constructor rather than a separate kind enum — the flat
/// [`Error`] already names every failure, so a parallel vocabulary here would
/// be a second level reintroduced one module over.
pub(crate) type Classified = fn(Box<ProviderError>) -> Error;

/// What a set of error spellings MEANS, as three independent lookups.
///
/// Implemented twice over for any given exchange: once by the protocol, for
/// the vocabulary its whole ecosystem shares, and once by the provider, for
/// what only it does. [`classify`] interleaves them.
///
/// THREE METHODS, NOT ONE, and that is the whole design. A single
/// `classify`-shaped hook would let a provider's rule outrank every protocol
/// rule regardless of strength — so a provider that maps a bare status would
/// beat the protocol reading an explicit `code`, and a failure the provider
/// NAMED would lose to a guess about its status. Splitting the hook by signal
/// makes the strength ordering something [`classify`] enforces rather than
/// something each implementor has to remember.
///
/// Every method defaults to `None`, so an implementor writes only the lookups
/// it actually has.
// `from_*` taking `&self` trips `clippy::wrong_self_convention`, which reads
// the prefix as a constructor. These are lookups on a mapper — "what does this
// mapper make of this code" — and the name states the SIGNAL each one reads,
// which is what `classify`'s ordering is built around. Renaming to `by_*`
// would lose that reading for a lint about constructors that do not exist here.
#[allow(clippy::wrong_self_convention)]
pub(crate) trait ErrorMapper: Sync {
    /// The provider's `code` — the failure naming itself. Strongest.
    fn from_code(&self, _code: &str) -> Option<Classified> {
        None
    }

    /// An HTTP status that means one thing on this protocol or from this
    /// provider. Weaker than a `code`: a status is warpllm reading a number,
    /// a code is the upstream saying what happened.
    fn from_status(&self, _status: u16) -> Option<Classified> {
        None
    }

    /// The provider's `type` — a failure FAMILY. Weakest of the three,
    /// because a family covers several failures that need different remedies.
    fn from_type(&self, _error_type: &str) -> Option<Classified> {
        None
    }
}

/// Nothing at all — the vocabulary of a provider that matches its protocol.
pub(crate) struct MatchesProtocol;

impl ErrorMapper for MatchesProtocol {}

/// Resolves one failure against a provider and the protocol it speaks,
/// strongest signal first.
///
/// The tier order lives HERE, once, rather than in each protocol: which
/// signal outranks which is a property of HTTP error envelopes in general,
/// not of any one wire format. Within a tier the provider is asked first,
/// since it is the more specific authority on its own spellings — but a tier
/// is never skipped, so a provider's status rule cannot outrank the
/// protocol's reading of an explicit `code`.
///
/// The residual is pure HTTP semantics and belongs to no vocabulary: a 4xx
/// nobody recognized is still a client error, and a 5xx is still the server's.
/// Anything else stays [`Error::Unknown`] rather than becoming a guess.
pub(crate) fn classify(
    provider: &dyn ErrorMapper,
    protocol: &dyn ErrorMapper,
    status: u16,
    error_type: Option<&str>,
    code: Option<&str>,
) -> Classified {
    code.and_then(|code| provider.from_code(code))
        .or_else(|| code.and_then(|code| protocol.from_code(code)))
        .or_else(|| provider.from_status(status))
        .or_else(|| protocol.from_status(status))
        .or_else(|| error_type.and_then(|family| provider.from_type(family)))
        .or_else(|| error_type.and_then(|family| protocol.from_type(family)))
        .unwrap_or(match status {
            400 | 422 => Error::InvalidRequest,
            500..=599 => Error::ServerError,
            _ => Error::Unknown,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A protocol that answers on every tier, so a provider rule can be shown
    /// to win or lose against each one independently.
    struct Protocol;

    impl ErrorMapper for Protocol {
        fn from_code(&self, code: &str) -> Option<Classified> {
            (code == "known_code").then_some(Error::ContentFilter as Classified)
        }
        fn from_status(&self, status: u16) -> Option<Classified> {
            (status == 429).then_some(Error::RateLimited as Classified)
        }
        fn from_type(&self, error_type: &str) -> Option<Classified> {
            (error_type == "known_type").then_some(Error::ServerError as Classified)
        }
    }

    struct ProviderStatus;

    impl ErrorMapper for ProviderStatus {
        fn from_status(&self, status: u16) -> Option<Classified> {
            (status == 402).then_some(Error::QuotaExceeded as Classified)
        }
    }

    fn code_of(
        provider: &dyn ErrorMapper,
        status: u16,
        error_type: Option<&str>,
        code: Option<&str>,
    ) -> &'static str {
        classify(provider, &Protocol, status, error_type, code)(Box::new(ProviderError {
            provider: "demo",
            status,
            message: String::new(),
            error_type: None,
            provider_code: None,
            retry_after: None,
            request_id: None,
            raw_body: String::new(),
        }))
        .code()
    }

    /// A provider that adds nothing inherits its protocol whole — the case
    /// every faithful backend is in, and the reason adding one needs no Rust.
    #[test]
    fn a_provider_with_no_vocabulary_inherits_the_protocol() {
        let none = &MatchesProtocol;
        assert_eq!(code_of(none, 429, None, None), "rate_limited");
        assert_eq!(
            code_of(none, 200, None, Some("known_code")),
            "content_filter"
        );
        assert_eq!(
            code_of(none, 200, Some("known_type"), None),
            "provider_server_error"
        );
    }

    /// Within a tier the provider is the more specific authority and wins.
    #[test]
    fn a_provider_rule_beats_the_protocol_at_the_same_tier() {
        assert_eq!(code_of(&ProviderStatus, 402, None, None), "quota_exceeded");
    }

    /// THE case the split hook exists for. A provider status rule must NOT
    /// outrank the protocol reading an explicit `code`: the upstream named
    /// its own failure, and a rule about its status numbers is a weaker
    /// signal than that. A single `classify` hook would get this backwards.
    #[test]
    fn a_provider_status_rule_does_not_outrank_a_protocol_code() {
        assert_eq!(
            code_of(&ProviderStatus, 402, None, Some("known_code")),
            "content_filter",
            "a status guess outranked the failure the provider named"
        );
    }

    /// ...and it DOES outrank the protocol's weaker tiers, or a provider
    /// could never correct a family its protocol reads differently.
    #[test]
    fn a_provider_status_rule_outranks_a_protocol_type() {
        assert_eq!(
            code_of(&ProviderStatus, 402, Some("known_type"), None),
            "quota_exceeded"
        );
    }

    /// Nothing recognized anywhere: HTTP semantics answer, and only where
    /// they actually say something.
    #[test]
    fn the_residual_is_http_semantics_and_nothing_more() {
        let none = &MatchesProtocol;
        assert_eq!(code_of(none, 400, None, None), "provider_invalid_request");
        assert_eq!(code_of(none, 422, None, None), "provider_invalid_request");
        assert_eq!(code_of(none, 500, None, None), "provider_server_error");
        assert_eq!(code_of(none, 502, None, None), "provider_server_error");
        // A status that means nothing in particular must not be guessed at.
        assert_eq!(code_of(none, 418, None, None), "provider_unknown");
    }
}
