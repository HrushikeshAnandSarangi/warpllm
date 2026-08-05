//! The OpenAI error envelope as it goes OUT, and the response metadata an
//! SDK reads off the transport rather than out of the body.
//!
//! Sibling in spirit to [`chat_completions::types`](super::chat_completions::types):
//! shapes only. Deciding WHICH of these a [`crate::Error`] becomes is an
//! egress conversion and lives at `crate::gateway::openai_compat::error`,
//! mirroring the ingest direction that turns a provider's error body into a
//! [`crate::Error`].

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The `error` object, spelled exactly as OpenAI spells it.
///
/// Four fields, and no fifth. warpllm's own taxonomy — its `code` slugs,
/// `origin`, the upstream evidence — is deliberately absent: this is what
/// callers program against, and a field here is one warpllm owes
/// compatibility on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ErrorBody {
    pub message: String,
    /// The OpenAI error family, e.g. `invalid_request_error`, `server_error`.
    #[serde(rename = "type")]
    pub error_type: String,
    /// Which request parameter was at fault. warpllm does not model this, so
    /// it is always `None` — present because OpenAI's shape has it, and a
    /// client reading `param` should find the field rather than a hole.
    pub param: Option<String>,
    /// The failure's own slug. Free-form in OpenAI's schema, which is what
    /// lets it carry the distinction `type` and status cannot: a quota
    /// exhaustion and a rate limit are both 429 `rate_limit_error` unless
    /// `code` says `insufficient_quota`.
    pub code: Option<String>,
}

/// Which language-level OpenAI-compatible exception to construct.
///
/// Rust decides this from the normalized status. Bindings only map this
/// discriminant to their local class object; they do not interpret statuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "codegen", ts(rename_all = "snake_case"))]
pub enum ErrorClass {
    ApiConnection,
    ApiStatus,
    BadRequest,
    Authentication,
    PermissionDenied,
    NotFound,
    Conflict,
    UnprocessableEntity,
    RateLimit,
    InternalServer,
}

impl ErrorClass {
    pub fn from_status(status: Option<u16>) -> Self {
        match status {
            None => Self::ApiConnection,
            Some(400) => Self::BadRequest,
            Some(401) => Self::Authentication,
            Some(403) => Self::PermissionDenied,
            Some(404) => Self::NotFound,
            Some(409) => Self::Conflict,
            Some(422) => Self::UnprocessableEntity,
            Some(429) => Self::RateLimit,
            Some(500..) => Self::InternalServer,
            Some(_) => Self::ApiStatus,
        }
    }
}

/// A failure rendered for an OpenAI-compatible surface: the body, plus what
/// an SDK takes from the response instead of from inside it.
///
/// One value, two renderings, which is the point of it existing — the HTTP
/// gateway uses its status and body, while FFI bindings also use its selected
/// exception class. No language binding interprets a status independently.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct OpenAiError {
    pub class: ErrorClass,
    /// The status an OpenAI-compatible surface answers with.
    ///
    /// `None` means no HTTP exchange ever happened — warpllm never reached
    /// the provider — which is precisely the case the official SDKs raise
    /// `APIConnectionError` for, so the absence is the signal.
    pub status: Option<u16>,
    /// Response headers worth keeping: `retry-after` and `x-request-id`,
    /// which exist nowhere else. The gateway re-emits them as real headers;
    /// a binding hands them to the caller as `APIError.headers`.
    pub headers: BTreeMap<String, String>,
    pub error: ErrorBody,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_status_class_is_decided_in_rust() {
        assert_eq!(ErrorClass::from_status(None), ErrorClass::ApiConnection);
        assert_eq!(
            ErrorClass::from_status(Some(401)),
            ErrorClass::Authentication
        );
        assert_eq!(ErrorClass::from_status(Some(418)), ErrorClass::ApiStatus);
        assert_eq!(
            ErrorClass::from_status(Some(503)),
            ErrorClass::InternalServer
        );
    }
}
