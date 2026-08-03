//! Ingesting an OpenAI-compatible ERROR body into the gateway's canonical
//! failure form.
//!
//! Sibling of [`api`](super::api) rather than a child of one of its
//! surfaces: the error envelope is protocol-wide, shared by every endpoint
//! this protocol serves, so it belongs to the protocol and not to one API
//! surface.
//!
//! This is an ingest conversion like any other — a wire shape in, a canonical
//! shape out — which is why it lives HERE and not beside the wire types.
//! `protocol::openai_compat::chat_completions::transport` deliberately hands
//! back a non-2xx as DATA, saying that deciding which [`Error`] it becomes
//! "would mean `protocol` reaching into the conversion layer". This module is
//! that conversion layer.
//!
//! Two jobs, and only one of them is this protocol's alone:
//!
//! * EXTRACTION — pulling `type`, `code`, and `message` out of
//!   `{"error": {…}}`. Nothing but a protocol can do this, because the
//!   envelope shape is what a protocol IS. A backend that shapes its errors
//!   differently is a different protocol, not a provider override.
//! * VOCABULARY — which spelling means which failure. Stated here as the
//!   baseline every backend on this protocol shares, and overridden per
//!   provider in [`provider_overrides`](super::provider_overrides) only where one genuinely
//!   diverges.

use std::time::Duration;

use serde_json::Value;

use crate::error::Error;
use crate::gateway::error::{Classified, ErrorMapper, classify};
use crate::gateway::types::ProviderError;

/// OpenAI-compatible error bodies look like
/// `{"error": {"message": ..., "type": ..., "code": ...}}`. Unparseable
/// bodies fall back to the raw text.
///
/// `retry_after` and `request_id` come from the response HEADERS, which is
/// the only place they exist — no error body carries either.
pub(crate) fn error_from_body(
    provider: &'static str,
    status: u16,
    body: &str,
    retry_after: Option<Duration>,
    request_id: Option<String>,
) -> Error {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let error = parsed.as_ref().map(|v| &v["error"]);
    let field = |name: &str| error.and_then(|e| e[name].as_str());
    let classified = classify(
        super::provider_overrides::error_mapper(provider),
        &OpenAiCompat,
        status,
        field("type"),
        field("code"),
    );
    classified(Box::new(ProviderError {
        provider,
        status,
        message: field("message")
            .map(str::to_string)
            .unwrap_or_else(|| body.to_string()),
        error_type: field("type").map(str::to_string),
        provider_code: field("code").map(str::to_string),
        retry_after,
        request_id,
        raw_body: ProviderError::bounded(body),
    }))
}

/// The error mapping every backend speaking this protocol shares.
///
/// PROTOCOL-LEVEL, not per-provider. DeepSeek, vLLM, SGLang and OpenRouter
/// all reuse OpenAI's envelope spellings, so gating these per provider would
/// mean a new roster entry silently lost the mapping until someone wrote Rust
/// for it. A provider states only its DELTA, next door.
///
/// Deliberately does NOT sniff `message`. Free text is the one part of an
/// error body providers reword without notice, and a substring match on it is
/// the exact staleness liability a capability table would be — it would
/// misclassify silently, which is worse than [`Error::Unknown`] saying
/// nothing. Unrecognized envelopes stay `Unknown` with the status and the raw
/// type intact for the caller to read.
pub(super) struct OpenAiCompat;

impl ErrorMapper for OpenAiCompat {
    fn from_code(&self, code: &str) -> Option<Classified> {
        Some(match code {
            "context_length_exceeded" => Error::ContextLengthExceeded,
            "content_filter" | "content_policy_violation" => Error::ContentFilter,
            "model_not_found" => Error::ModelNotFound,
            "rate_limit_exceeded" => Error::RateLimited,
            // NOT a rate limit, however much the 429 beside it looks like
            // one: the account is out of credit, and backing off does not buy
            // any. OpenAI ships both spellings.
            "insufficient_quota" | "billing_hard_limit_reached" => Error::QuotaExceeded,
            "invalid_api_key" => Error::Authentication,
            _ => return None,
        })
    }

    /// Only the statuses that mean ONE thing on this protocol.
    ///
    /// 404 is deliberately absent: it may name a model, a proxy route, or the
    /// endpoint itself, so it needs envelope evidence before becoming
    /// [`Error::ModelNotFound`]. 402 is absent because it is not this
    /// protocol's — DeepSeek documents it, and states it itself.
    fn from_status(&self, status: u16) -> Option<Classified> {
        Some(match status {
            401 => Error::Authentication,
            403 => Error::PermissionDenied,
            429 => Error::RateLimited,
            503 => Error::Overloaded,
            _ => return None,
        })
    }

    /// The `type` family, weakest of the three — and the tier that explains
    /// why the split matters. `invalid_request_error` is OpenAI's CATCH-ALL,
    /// sent for a bad API key (401), an unknown model (404), and a genuinely
    /// malformed body alike. Reading it over an unambiguous 401 would report
    /// a credential failure as a bad request and send a caller to fix their
    /// payload, which is why a decisive status is asked first.
    fn from_type(&self, error_type: &str) -> Option<Classified> {
        Some(match error_type {
            "rate_limit_error" => Error::RateLimited,
            "overloaded_error" => Error::Overloaded,
            "authentication_error" => Error::Authentication,
            "permission_error" => Error::PermissionDenied,
            "not_found_error" => Error::ModelNotFound,
            "invalid_request_error" => Error::InvalidRequest,
            "api_error" => Error::ServerError,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classified(status: u16, body: &str) -> Error {
        error_from_body("demo", status, body, None, None)
    }

    /// The classified variant, named by its stable wire slug — which makes
    /// the tables below assert on the public contract rather than on an
    /// internal spelling.
    fn kind_of(status: u16, body: &str) -> &'static str {
        classified(status, body).code()
    }

    fn detail_of(status: u16, body: &str) -> ProviderError {
        let error = classified(status, body);
        error
            .provider_error()
            .unwrap_or_else(|| panic!("expected a provider failure, got {error:?}"))
            .clone()
    }

    #[test]
    fn unparseable_error_body_is_preserved() {
        let body = "x".repeat(1_024);
        let err = error_from_body("openai", 503, &body, None, None);
        assert_eq!(err.provider_error().unwrap().message, body);
    }

    /// Every classified failure is provider-driven, whatever the variant —
    /// the grouping the flat enum gives up in its shape.
    #[test]
    fn a_classified_failure_is_always_provider_driven() {
        for (status, body) in [(429, "{}"), (400, "not json"), (402, "{}")] {
            assert_eq!(
                classified(status, body).origin(),
                crate::error::Origin::Provider,
                "{status}"
            );
        }
    }

    /// The seam, end to end: the same status classifies differently by
    /// provider, because DeepSeek documents 402 as an exhausted balance and
    /// this protocol has no reading of it. Proves the override table is
    /// actually reached — the unit tests next door only prove the vocabulary
    /// itself is right.
    #[test]
    fn a_providers_own_vocabulary_reaches_the_classifier() {
        assert!(matches!(
            error_from_body("deepseek", 402, "{}", None, None),
            Error::QuotaExceeded(_)
        ));
        // Anyone else gets the protocol, which says nothing about 402 and
        // falls to the HTTP residual.
        assert!(matches!(
            error_from_body("openai", 402, "{}", None, None),
            Error::Unknown(_)
        ));
    }

    /// ...and a provider's status rule still loses to an explicit `code`,
    /// through the real dispatch rather than a stub vocabulary.
    #[test]
    fn a_providers_status_rule_loses_to_a_code_it_did_not_claim() {
        let body = r#"{"error":{"message":"m","code":"invalid_api_key"}}"#;
        assert!(matches!(
            error_from_body("deepseek", 402, body, None, None),
            Error::Authentication(_)
        ));
    }

    /// `code` is the most specific signal, so it decides even when the status
    /// and family would say something else. A context overflow is the case
    /// that matters: it arrives as a 400 `invalid_request_error`, and calling
    /// it `InvalidRequest` would tell a router to give up on a request that a
    /// bigger-window model would serve.
    #[test]
    fn code_outranks_type_and_status() {
        for (status, error_type, code, expected) in [
            (
                400,
                "invalid_request_error",
                "context_length_exceeded",
                "context_length_exceeded",
            ),
            (
                400,
                "invalid_request_error",
                "content_filter",
                "content_filter",
            ),
            (
                404,
                "invalid_request_error",
                "model_not_found",
                "model_not_found",
            ),
            // The 429 says rate limit; the code says the account is out of
            // credit. The code wins, and the difference is the whole point —
            // one is worth waiting out, the other never is.
            (
                429,
                "invalid_request_error",
                "insufficient_quota",
                "quota_exceeded",
            ),
            (
                429,
                "invalid_request_error",
                "billing_hard_limit_reached",
                "quota_exceeded",
            ),
            (
                401,
                "invalid_request_error",
                "invalid_api_key",
                "authentication",
            ),
        ] {
            let body =
                format!(r#"{{"error":{{"message":"m","type":"{error_type}","code":"{code}"}}}}"#);
            assert_eq!(kind_of(status, &body), expected, "{code}");
        }
    }

    /// The real 401: OpenAI sends its catch-all `invalid_request_error`
    /// family for a bad API key, an unknown model, and a malformed body
    /// alike. An unambiguous status has to win, or a credential failure gets
    /// reported as a bad payload and the caller goes looking in the wrong
    /// place.
    #[test]
    fn an_unambiguous_status_outranks_the_catch_all_family() {
        for (status, expected) in [
            (401, "authentication"),
            (403, "permission_denied"),
            (429, "rate_limited"),
            (503, "overloaded"),
        ] {
            let body = r#"{"error":{"message":"m","type":"invalid_request_error"}}"#;
            assert_eq!(kind_of(status, body), expected, "{status}");
        }
        // ...but where the status says nothing specific, the family still
        // decides.
        let body = r#"{"error":{"message":"m","type":"invalid_request_error"}}"#;
        assert_eq!(kind_of(400, body), "provider_invalid_request");
    }

    /// A 404 is NOT decisive on this protocol: it may name a model, a proxy
    /// route, or the endpoint itself. Without envelope evidence it stays a
    /// generic client error rather than claiming the model does not exist.
    #[test]
    fn a_bare_404_is_not_read_as_a_missing_model() {
        assert_ne!(kind_of(404, "{}"), "model_not_found");
        // ...but a 404 that NAMES the failure is.
        let body = r#"{"error":{"message":"m","type":"not_found_error"}}"#;
        assert_eq!(kind_of(404, body), "model_not_found");
    }

    /// Where the status is not decisive, the family decides over the residual
    /// status ranges.
    #[test]
    fn type_outranks_a_non_decisive_status() {
        for (error_type, expected) in [
            ("rate_limit_error", "rate_limited"),
            ("overloaded_error", "overloaded"),
            ("authentication_error", "authentication"),
            ("permission_error", "permission_denied"),
            ("not_found_error", "model_not_found"),
            ("invalid_request_error", "provider_invalid_request"),
            ("api_error", "provider_server_error"),
        ] {
            // A status that would classify differently on its own.
            let body = format!(r#"{{"error":{{"message":"m","type":"{error_type}"}}}}"#);
            assert_eq!(kind_of(418, &body), expected, "{error_type}");
        }
    }

    /// Plenty of OpenAI-compatible backends send a bare status with no
    /// envelope at all. The status is the last signal, never a guess beyond
    /// it.
    #[test]
    fn status_classifies_an_empty_envelope() {
        for (status, expected) in [
            (400, "provider_invalid_request"),
            (422, "provider_invalid_request"),
            (401, "authentication"),
            (403, "permission_denied"),
            (429, "rate_limited"),
            (503, "overloaded"),
            (500, "provider_server_error"),
            (502, "provider_server_error"),
        ] {
            assert_eq!(kind_of(status, "not json"), expected, "{status}");
        }
    }

    /// An envelope nothing recognizes must say nothing rather than guess.
    /// `Unknown` is a real answer here — the status and the raw type are both
    /// retained for the caller to read.
    #[test]
    fn an_unrecognized_envelope_is_unknown_and_keeps_the_raw_type() {
        let body = r#"{"error":{"message":"m","type":"vendor_specific_error","code":"weird"}}"#;
        assert!(matches!(classified(418, body), Error::Unknown(_)));
        let detail = detail_of(418, body);
        assert_eq!(detail.status, 418);
        assert_eq!(detail.error_type.as_deref(), Some("vendor_specific_error"));
    }

    /// The classification must never consume the provider's own spelling:
    /// the variant is what it means, `error_type` is what it said.
    #[test]
    fn classifying_retains_the_providers_own_spelling() {
        let body = r#"{"error":{"message":"slow down","type":"rate_limit_error"}}"#;
        assert!(matches!(classified(429, body), Error::RateLimited(_)));
        let detail = detail_of(429, body);
        assert_eq!(detail.error_type.as_deref(), Some("rate_limit_error"));
        assert_eq!(detail.message, "slow down");
    }

    /// Free text is the one part of an error body providers reword without
    /// notice, so it must not influence classification — a message that says
    /// "rate limit" under a 400 is still an invalid request.
    #[test]
    fn the_message_never_influences_classification() {
        let body = r#"{"error":{"message":"rate limit exceeded, context length exceeded"}}"#;
        assert_eq!(kind_of(400, body), "provider_invalid_request");
    }

    /// PROMOTE BUT RETAIN: the `code` is what the classification was READ
    /// from, so discarding it would throw away the strongest evidence in the
    /// envelope and leave a caller unable to check warpllm's reading.
    #[test]
    fn the_provider_code_survives_the_classification_it_produced() {
        let body = r#"{"error":{"message":"m","type":"invalid_request_error","code":"insufficient_quota"}}"#;
        let detail = detail_of(429, body);
        assert_eq!(detail.provider_code.as_deref(), Some("insufficient_quota"));
        assert_eq!(detail.error_type.as_deref(), Some("invalid_request_error"));
        // ...and the whole envelope is there as a last resort.
        assert!(detail.raw_body.contains("insufficient_quota"));
    }

    /// A vendor code nothing recognizes is exactly when the raw evidence
    /// matters most: warpllm has no answer, so the caller needs everything
    /// the provider said in order to form one.
    #[test]
    fn an_unrecognized_code_is_still_retained() {
        let body = r#"{"error":{"message":"m","type":"vendor_error","code":"tpm_exhausted_v2"}}"#;
        let detail = detail_of(418, body);
        assert_eq!(detail.provider_code.as_deref(), Some("tpm_exhausted_v2"));
        assert!(detail.raw_body.contains("tpm_exhausted_v2"));
    }

    /// `Retry-After` and the upstream request id live ONLY in headers, so
    /// they have to be threaded in from the transport — a body-only error
    /// path cannot recover them.
    #[test]
    fn header_evidence_reaches_the_error() {
        let err = error_from_body(
            "demo",
            429,
            r#"{"error":{"message":"slow down"}}"#,
            Some(Duration::from_secs(42)),
            Some("req-abc".into()),
        );
        assert!(matches!(err, Error::RateLimited(_)));
        let detail = err.provider_error().unwrap();
        assert_eq!(detail.retry_after, Some(Duration::from_secs(42)));
        assert_eq!(detail.request_id.as_deref(), Some("req-abc"));
    }
}
