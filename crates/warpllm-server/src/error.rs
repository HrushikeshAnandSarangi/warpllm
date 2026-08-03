//! Maps [`warpllm::Error`] onto HTTP statuses and the OpenAI error envelope
//! (`{"error": {"message", "type", "code"}}`) so official SDKs raise their
//! proper typed exceptions. `code` values reuse the stable
//! [`warpllm::Error::to_wire_json`] contract.

use axum::http::header::RETRY_AFTER;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use serde_json::{Value, json};

pub fn openai_error_body(error: &warpllm::Error) -> (StatusCode, Value) {
    use warpllm::Error;
    // A provider-driven failure is answered with the upstream's own status
    // and family, so a genuinely unknown model surfaces here as the
    // provider's 404 rather than as a warpllm opinion about it. Matched
    // first: `warpllm::Error` is `non_exhaustive`, so a new provider
    // variant must land here without a code change over here.
    let (status, error_type) = match (error.provider_error(), error) {
        (Some(upstream), _) => (
            StatusCode::from_u16(upstream.status).unwrap_or(StatusCode::BAD_GATEWAY),
            upstream.error_type.as_deref().unwrap_or("api_error"),
        ),
        (None, Error::InvalidInput(_) | Error::InvalidModel { .. }) => {
            (StatusCode::BAD_REQUEST, "invalid_request_error")
        }
        (None, Error::MissingApiKey { .. }) => (StatusCode::UNAUTHORIZED, "invalid_request_error"),
        (None, Error::Network { .. } | Error::Decode { .. }) => {
            (StatusCode::BAD_GATEWAY, "api_error")
        }
        (None, Error::NotImplemented(_)) => (StatusCode::NOT_IMPLEMENTED, "api_error"),
        // Unreachable today; a new gateway-side variant lands here rather
        // than failing to compile a binary that must keep serving.
        (None, _) => (StatusCode::INTERNAL_SERVER_ERROR, "api_error"),
    };
    let mut envelope = json!({
        "message": error.to_string(),
        "type": error_type,
        // `type` keeps its OpenAI meaning so official SDKs raise the
        // exceptions they already raise; `code` and `origin` are warpllm's
        // flat taxonomy, spelled exactly as on the FFI wire form.
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
