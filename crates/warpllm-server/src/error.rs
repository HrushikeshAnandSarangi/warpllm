//! Maps [`warpllm::Error`] onto HTTP statuses and the OpenAI error envelope
//! (`{"error": {"message", "type", "code"}}`) so official SDKs raise their
//! proper typed exceptions.
//!
//! The status and `type` come from [`warpllm::Error::openai_status_and_type`],
//! shared with the FFI envelope so the two OpenAI-compatible surfaces cannot
//! disagree about what a failure is. What this adds on top — `code` as
//! warpllm's slug, `origin`, and the provider's evidence — is deliberate and
//! does NOT hold for the bindings: an HTTP caller can ignore a field it does
//! not know, while a field on an SDK error object is a promise to keep.

use axum::http::header::RETRY_AFTER;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use serde_json::{Value, json};

pub fn openai_error_body(error: &warpllm::Error) -> (StatusCode, Value) {
    let (status, error_type) = error.openai_status_and_type();
    // A provider can name a status no HTTP client could send; answering with
    // it verbatim would make this response unparseable, so it degrades to a
    // plain upstream failure.
    let status = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut envelope = json!({
        "message": error.to_string(),
        "type": error_type,
        // `type` keeps its OpenAI meaning so official SDKs raise the
        // exceptions they already raise; `code` and `origin` are warpllm's
        // flat taxonomy, additive fields an HTTP caller can ignore. The FFI
        // envelope carries neither — see this module's header.
        "code": error.code(),
        "origin": error.origin().as_str(),
    });
    // What the provider said — never what to do about it.
    if let Some(upstream) = error.provider_error() {
        if let Some(provider_code) = &upstream.provider_code {
            envelope["provider_code"] = json!(provider_code);
        }
        if let Some(request_id) = &upstream.request_id {
            envelope["request_id"] = json!(request_id);
        }
        if let Some(retry_after) = upstream.retry_after {
            envelope["retry_after_seconds"] = json!(retry_after.as_secs());
        }
    }
    (status, json!({ "error": envelope }))
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
