//! `POST /chat/completions`: the four conversions between the
//! OpenAI-compatible wire shapes and the normalized forms, plus transport.

mod request;
mod response;

pub(crate) use request::{ingest_request, render_request};
pub(crate) use response::{ingest_response, render_response};

use serde_json::Value;

use crate::error::{Error, Result};
use crate::http::{network_error, read_body};
use crate::normalized::{ProviderExt, Role};
use crate::protocol::openai_compat::error_from_body;
use crate::types::openai_compat::chat::completions::{
    CreateChatCompletionRequest, CreateChatCompletionResponse, UnknownFields,
};

/// The ext-bag namespace for this dialect's passthrough fields.
pub(super) const NAMESPACE: &str = "openai_compat";

/// Flattens the ext bags relevant to this render target: the dialect
/// namespace first (the target speaks it), overlaid by the provider's own
/// namespace. Other providers' namespaces are ignored — a field meant for
/// one provider never leaks into another.
pub(super) fn merged_ext(ext: &ProviderExt, provider: &str) -> UnknownFields {
    let mut merged = UnknownFields::new();
    for namespace in [NAMESPACE, provider] {
        if let Some(Value::Object(fields)) = ext.get(namespace) {
            for (key, value) in fields {
                merged.insert(key.clone(), value.clone());
            }
        }
    }
    merged
}

/// Wire role string → normalized role. Unrecognized roles (e.g. OpenAI's
/// `"developer"`) map to the closest normalized role and return the raw
/// spelling so ingest can stash it for exact restoration.
pub(super) fn role_from_wire(role: String) -> (Role, Option<String>) {
    match role.as_str() {
        "system" => (Role::System, None),
        "user" => (Role::User, None),
        "assistant" => (Role::Assistant, None),
        "tool" => (Role::Tool, None),
        _ => (Role::User, Some(role)),
    }
}

pub(super) fn role_to_wire(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

pub(crate) async fn post(
    http: &reqwest::Client,
    provider: &'static str,
    base_url: &str,
    api_key: &str,
    body: &CreateChatCompletionRequest,
) -> Result<CreateChatCompletionResponse> {
    let response = http
        .post(format!(
            "{}/chat/completions",
            base_url.trim_end_matches('/')
        ))
        .bearer_auth(api_key)
        .json(body)
        .send()
        .await
        .map_err(|e| network_error(provider, e))?;

    let (status, text) = read_body(provider, response).await?;
    if !(200..300).contains(&status) {
        return Err(error_from_body(provider, status, &text));
    }

    serde_json::from_str(&text).map_err(|e| Error::Decode {
        provider,
        message: e.to_string(),
    })
}
