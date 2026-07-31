//! The OpenAI-compatible protocol: every provider whose spec selects
//! `Protocol::OpenAiCompat` is spoken through here, parameterized by the
//! provider's name, base URL, and key.
//!
//! `openai_compat` is warpllm's OpenAI-*compatible* dialect: a permissive
//! superset (unknown-field passthrough) of the OpenAI request, which many
//! providers speak. The authoritative *OpenAI* shape is defined upstream in
//! <https://github.com/openai/openai-openapi>; we track it and contribute
//! changes there rather than fork it. A provider with its own wire format gets
//! a sibling protocol module here (e.g. `anthropic::messages`), and the
//! conversions that translate between the two via the normalized request live
//! under `crate::normalized`.
//!
//! Providers that speak this dialect but diverge in places — a response field
//! carrying meaning OpenAI has no name for, a parameter spelled differently —
//! do NOT get shapes of their own. They get an adapter over the conversions;
//! the shapes here stay the dialect's.
//!
//! The error envelope below is protocol-wide, shared by every endpoint, and is
//! the default a provider's adapter may replace.

pub mod chat_completions;

use serde_json::Value;

use crate::error::Error;

/// OpenAI-compatible error bodies look like
/// `{"error": {"message": ..., "type": ...}}`. Unparseable bodies fall back
/// to the raw text.
pub(crate) fn error_from_body(provider: &'static str, status: u16, body: &str) -> Error {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let error = parsed.as_ref().map(|v| &v["error"]);
    Error::Provider {
        provider,
        status,
        error_type: error.and_then(|e| e["type"].as_str()).map(str::to_string),
        message: error
            .and_then(|e| e["message"].as_str())
            .map(str::to_string)
            .unwrap_or_else(|| body.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unparseable_error_body_is_preserved() {
        let body = "x".repeat(1_024);
        let err = error_from_body("openai", 503, &body);

        match err {
            Error::Provider { message, .. } => assert_eq!(message, body),
            other => panic!("expected Provider error, got {other:?}"),
        }
    }
}
