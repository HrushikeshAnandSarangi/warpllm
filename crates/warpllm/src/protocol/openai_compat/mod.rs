//! The OpenAI-compatible protocol: every provider whose spec selects
//! `Protocol::OpenAiCompat` is spoken through here, parameterized by the
//! provider's name, base URL, and key. The error envelope below is
//! protocol-wide and shared by every endpoint.

pub(crate) mod chat;

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
