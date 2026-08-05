//! Maps [`warpllm::Error`] onto HTTP statuses and the OpenAI error envelope
//! (`{"error": {"message", "type", "code"}}`) so official SDKs raise their
//! proper typed exceptions.
//!
//! The whole envelope comes from [`warpllm::Error::to_openai`], shared with
//! the FFI form so the two OpenAI-compatible surfaces cannot disagree about
//! what a failure is. Every OpenAI-visible field is therefore identical
//! whether a caller reaches warpllm in-process or through this gateway.
//!
//! What this adds on top — `origin`, `warpllm_code`, and which provider
//! answered with what — is deliberate and does NOT hold for the bindings: an
//! HTTP caller can ignore a field it does not know, while a field on an SDK
//! error object is a promise to keep. `provider_status` is here because the
//! rendering NORMALIZES status, so the upstream's own is worth keeping where
//! someone debugging can still reach it.

use axum::http::header::RETRY_AFTER;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use serde_json::{Value, json};

pub fn openai_error_body(error: &warpllm::Error) -> (StatusCode, Value) {
    let rendered = error.to_openai();
    let mut envelope = json!(rendered.error);
    // Additive, and only here. An HTTP caller can ignore a field it does not
    // recognize, which is not true of an attribute on an SDK error object —
    // so warpllm's own taxonomy rides along on this surface and on no other.
    envelope["origin"] = json!(error.origin().as_str());
    envelope["warpllm_code"] = json!(error.code());
    if let Some(upstream) = error.provider_error() {
        envelope["provider"] = json!(upstream.provider);
        envelope["provider_status"] = json!(upstream.status);
        if let Some(provider_code) = &upstream.provider_code {
            envelope["provider_code"] = json!(provider_code);
        }
    }
    (status_code(&rendered), json!({ "error": envelope }))
}

/// The status to answer with.
///
/// `None` means warpllm never reached the provider, which has no OpenAI
/// spelling because it never happens to a client talking to OpenAI directly —
/// 502 is what a gateway says when the thing behind it did not answer. A
/// provider can also name a status no HTTP client could send, and answering
/// with that verbatim would make this response unparseable.
fn status_code(rendered: &warpllm::OpenAiError) -> StatusCode {
    rendered
        .status
        .and_then(|status| StatusCode::from_u16(status).ok())
        .unwrap_or(StatusCode::BAD_GATEWAY)
}

pub fn error_response(error: &warpllm::Error) -> Response {
    let (status, body) = openai_error_body(error);
    let mut response = (status, Json(body)).into_response();
    // Re-emit the upstream's own `Retry-After` as a real header, not just a
    // body field: standard HTTP clients and proxies back off on the header
    // and will never read warpllm's JSON.
    if let Some(upstream) = error.provider_error() {
        if let Some(retry_after) = upstream.retry_after {
            if let Ok(value) = HeaderValue::from_str(&retry_after.as_secs().to_string()) {
                response.headers_mut().insert(RETRY_AFTER, value);
            }
        }
    }
    response
}

pub fn invalid_request_response(message: String) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "invalid_request",
                "origin": "gateway",
            }
        })),
    )
        .into_response()
}
