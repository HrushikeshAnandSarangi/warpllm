//! OpenAI-compatible request and response types — warpllm's permissive
//! superset of the OpenAI shapes (the authoritative OpenAI spec lives in
//! <https://github.com/openai/openai-openapi>; changes to it belong
//! upstream). Other providers translate to and from these, so callers see
//! one shape regardless of upstream.
//!
//! The response section is a field-for-field copy of the `chat.completion`
//! object, keeping upstream object names and field order:
//! - Response object: <https://developers.openai.com/api/reference/resources/chat>
//! - Request parameters: <https://platform.openai.com/docs/api-reference/chat/create>
//!
//! Naming: types matching a named schema in the official OpenAPI spec
//! (<https://github.com/openai/openai-openapi>) use that exact name; types the
//! spec leaves anonymous/inline keep a local descriptive name.
//!
//! Every struct captures fields it doesn't model in an `unknown_fields`
//! catch-all; see [`UnknownFields`].

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Catch-all for fields OpenAI introduces that this crate does not model yet.
///
/// Every request and response struct carries a `#[serde(flatten)]`
/// `unknown_fields` of this type, so a field added upstream still reaches
/// clients (and an unmodeled request parameter still reaches the provider)
/// instead of being silently dropped — clients can adopt new API fields
/// before this crate ships explicit support for them.
pub type UnknownFields = serde_json::Map<String, serde_json::Value>;

// Codegen policy: Rust and Serde are the contract. Codegen-only attributes may
// enable derives or compensate for representation details a generator cannot
// infer (such as output optionality). Do not add literal unions, schema enums,
// const values, ranges, or other constraints that are absent from the Rust
// field type; OpenAI-compatible providers extend these values independently.

// ---------------------------------------------------------------------------
// Response — the `chat.completion` object
// <https://developers.openai.com/api/reference/resources/chat>
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct CreateChatCompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub created: u64,
    /// Echoes the caller-supplied `provider/model` string.
    pub model: String,
    /// Conventionally `"chat.completion"`; compatible providers may differ.
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub moderation: Option<ChatCompletionModeration>,
    /// Provider-defined service tier identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub service_tier: Option<String>,
    /// Deprecated upstream but still returned; passed through as-is.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub usage: Option<CompletionUsage>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct Choice {
    /// Provider-defined reason that generation stopped.
    pub finish_reason: String,
    pub index: u32,
    /// Optional per the docs; `Option` also tolerates the explicit
    /// `"logprobs": null` some OpenAI-compatible backends emit.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub logprobs: Option<ChoiceLogprobs>,
    pub message: ChatCompletionResponseMessage,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// OpenAI documents both arrays as required; OpenAI-compatible backends
/// are looser — DeepSeek omits `refusal` entirely and can null `content` —
/// so both are optional here and absent fields stay off the wire.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChoiceLogprobs {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub content: Option<Vec<ChatCompletionTokenLogprob>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub refusal: Option<Vec<ChatCompletionTokenLogprob>>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionTokenLogprob {
    pub token: String,
    pub bytes: Option<Vec<u8>>,
    pub logprob: f64,
    pub top_logprobs: Vec<TopLogprob>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct TopLogprob {
    pub token: String,
    pub bytes: Option<Vec<u8>>,
    pub logprob: f64,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionResponseMessage {
    pub content: Option<String>,
    pub refusal: Option<String>,
    /// Conventionally `"assistant"`; compatible providers may differ.
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub annotations: Option<Vec<Annotation>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub audio: Option<ChatCompletionAudio>,
    /// Deprecated upstream in favor of `tool_calls`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub function_call: Option<FunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub tool_calls: Option<Vec<ChatCompletionMessageToolCallUnion>>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct Annotation {
    /// Conventionally `"url_citation"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    pub url_citation: AnnotationURLCitation,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

// Exact upstream name; OpenAI-shape fidelity outranks Rust acronym casing.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct AnnotationURLCitation {
    pub end_index: u32,
    pub start_index: u32,
    pub title: String,
    pub url: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// Deprecated upstream in favor of tool calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct FunctionCall {
    /// JSON-encoded arguments; model-generated, so may be invalid JSON.
    pub arguments: String,
    pub name: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// Function- or custom-shaped tool call.
// Untagged so the structs own the `type` field: an internal serde tag would be
// captured by their flattened `unknown_fields` and emitted twice. Each
// variant's required `function`/`custom` field keeps dispatch unambiguous even
// when a provider sends a nonstandard `type` string.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
#[serde(untagged)]
pub enum ChatCompletionMessageToolCallUnion {
    Function(ChatCompletionMessageToolCall),
    Custom(ChatCompletionMessageCustomToolCall),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionMessageToolCall {
    pub id: String,
    /// Conventionally `"function"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    pub function: Function,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct Function {
    /// JSON-encoded arguments; model-generated, so may be invalid JSON.
    pub arguments: String,
    pub name: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionMessageCustomToolCall {
    pub id: String,
    /// Conventionally `"custom"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    pub custom: Custom,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct Custom {
    pub input: String,
    pub name: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionAudio {
    pub id: String,
    /// Base64-encoded audio bytes.
    pub data: String,
    pub expires_at: u64,
    pub transcript: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct CompletionUsage {
    pub completion_tokens: u32,
    pub prompt_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct CompletionTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub accepted_prediction_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub audio_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub reasoning_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub rejected_prediction_tokens: Option<u32>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct PromptTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub audio_tokens: Option<u32>,
    /// Unadjusted number of prompt tokens written to cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub cache_write_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional))]
    pub cached_tokens: Option<u32>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// Moderation results for the request input and the generated output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionModeration {
    pub input: ModerationOutcome,
    pub output: ModerationOutcome,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

// The docs define one shared ModerationResults/Error pair used by both
// `input` and `output`.

/// Union of [`ChatCompletionModerationResults`] or
/// [`ChatCompletionModerationError`]. Untagged so the structs own the `type`
/// field exactly as received; their required fields are disjoint, so dispatch
/// is unambiguous. The spec leaves this union unnamed; only this enum's name is
/// Rust-side.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
#[serde(untagged)]
pub enum ModerationOutcome {
    ModerationResults(ChatCompletionModerationResults),
    Error(ChatCompletionModerationError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionModerationResults {
    pub model: String,
    pub results: Vec<ModerationResultBody>,
    /// Conventionally `"moderation_results"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// One verdict in `ChatCompletionModerationResults.results`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ModerationResultBody {
    pub categories: HashMap<String, bool>,
    /// Values are input types, e.g. `"text"` or `"image"`.
    pub category_applied_input_types: HashMap<String, Vec<String>>,
    pub category_scores: HashMap<String, f64>,
    pub flagged: bool,
    pub model: String,
    /// Conventionally `"moderation_result"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

/// Moderation error.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionModerationError {
    pub code: String,
    pub message: String,
    /// Conventionally `"error"`; compatible providers may differ.
    #[serde(rename = "type")]
    pub r#type: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

// ---------------------------------------------------------------------------
// Request — the `POST /v1/chat/completions` parameters
// <https://platform.openai.com/docs/api-reference/chat/create>
// ---------------------------------------------------------------------------

// Doc comments on these types are REPUBLISHED: ts-rs copies them verbatim into
// the generated `.d.ts` that ships to npm. Rationale meant for this crate goes
// in `//` comments like this one, which ts-rs does not copy — a `///` here
// would put warpllm's internals, and any rustdoc `[link]` syntax that means
// nothing in TypeScript, into a published declaration file.
//
// `unknown_fields` is `ts(skip)` here, same as on responses. The method taking
// this request is generic in TypeScript and Mapping-based in Python, so both
// languages accept extensions without weakening response field checking.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct CreateChatCompletionRequest {
    /// Model string in `provider/model` form, e.g. `"openai/gpt-5.6"`.
    pub model: String,
    pub messages: Vec<ChatCompletionRequestMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub stop: Option<Vec<String>>,
    // Not implemented.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "codegen", ts(optional = nullable))]
    pub stream: Option<bool>,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "codegen", derive(ts_rs::TS, schemars::JsonSchema))]
pub struct ChatCompletionRequestMessage {
    /// Provider-defined role; common values include `"system"` and `"user"`.
    pub role: String,
    pub content: String,
    #[serde(flatten)]
    #[cfg_attr(feature = "codegen", ts(skip))]
    pub unknown_fields: UnknownFields,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// OpenAI-compatible proxies often omit the optional response fields;
    /// every one of them must deserialize as absent, not error.
    #[test]
    fn minimal_response_body_deserializes() {
        let completion: CreateChatCompletionResponse = serde_json::from_str(
            r#"{
                "id": "chatcmpl-123",
                "choices": [{
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": "hi", "refusal": null, "role": "assistant"}
                }],
                "created": 1700000000,
                "model": "gpt-5.6",
                "object": "chat.completion"
            }"#,
        )
        .unwrap();
        assert!(completion.moderation.is_none());
        assert!(completion.service_tier.is_none());
        assert!(completion.system_fingerprint.is_none());
        assert!(completion.usage.is_none());
        assert!(completion.choices[0].logprobs.is_none());
        assert!(completion.choices[0].message.tool_calls.is_none());
        // Absent optionals must also stay off the wire when re-serialized.
        let wire = serde_json::to_value(&completion).unwrap();
        assert!(wire.get("usage").is_none());
        assert!(wire.get("moderation").is_none());
        assert!(wire["choices"][0].get("logprobs").is_none());
        assert!(wire["choices"][0]["message"].get("tool_calls").is_none());
    }

    /// Fields we don't model must be captured into `unknown_fields` — at
    /// every nesting level — and re-emitted verbatim, not dropped.
    #[test]
    fn unknown_fields_round_trip() {
        let body = serde_json::json!({
            "id": "chatcmpl-123",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "hi",
                    "refusal": null,
                    "role": "assistant",
                    "reasoning_content": "step by step"
                },
                "new_choice_field": true
            }],
            "created": 1700000000,
            "model": "gpt-5.6",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 1,
                "prompt_tokens": 2,
                "total_tokens": 3,
                "new_usage_field": 7
            },
            "new_top_level_field": "surprise"
        });

        let completion: CreateChatCompletionResponse =
            serde_json::from_value(body.clone()).unwrap();

        assert_eq!(completion.unknown_fields["new_top_level_field"], "surprise");
        assert_eq!(
            completion.choices[0].unknown_fields["new_choice_field"],
            true
        );
        assert_eq!(
            completion.choices[0].message.unknown_fields["reasoning_content"],
            "step by step"
        );
        assert_eq!(
            completion.usage.as_ref().unwrap().unknown_fields["new_usage_field"],
            7
        );
        assert_eq!(serde_json::to_value(&completion).unwrap(), body);
    }

    /// These are strings in the compatibility protocol, not closed enums.
    /// Providers routinely add values before any shared specification does.
    #[test]
    fn provider_defined_string_values_round_trip() {
        let body = serde_json::json!({
            "id": "chatcmpl-123",
            "choices": [{
                "finish_reason": "provider_finished_elsewhere",
                "index": 0,
                "message": {
                    "content": "hi",
                    "refusal": null,
                    "role": "critic",
                    "annotations": [{
                        "type": "provider_citation",
                        "url_citation": {
                            "end_index": 2,
                            "start_index": 0,
                            "title": "Example",
                            "url": "https://example.com"
                        }
                    }],
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "provider_function",
                        "function": {"arguments": "{}", "name": "search"}
                    }]
                }
            }],
            "created": 1,
            "model": "provider/model",
            "object": "provider.chat.result",
            "service_tier": "provider_experimental"
        });

        let completion: CreateChatCompletionResponse =
            serde_json::from_value(body.clone()).unwrap();

        assert_eq!(completion.object, "provider.chat.result");
        assert_eq!(completion.choices[0].message.role, "critic");
        assert_eq!(serde_json::to_value(&completion).unwrap(), body);
    }

    /// A body with every documented field must round-trip byte-for-byte
    /// (as JSON values), proving the deep copy is complete and lossless.
    #[test]
    fn full_response_body_round_trips() {
        let body = serde_json::json!({
            "id": "chatcmpl-123",
            "choices": [{
                "finish_reason": "tool_calls",
                "index": 0,
                "logprobs": {
                    "content": [{
                        "token": "Hi",
                        "bytes": [72, 105],
                        "logprob": -0.1,
                        "top_logprobs": [{"token": "Hi", "bytes": null, "logprob": -0.1}]
                    }],
                    "refusal": []
                },
                "message": {
                    "content": "Hello there!",
                    "refusal": null,
                    "role": "assistant",
                    "annotations": [{
                        "type": "url_citation",
                        "url_citation": {
                            "end_index": 5,
                            "start_index": 0,
                            "title": "Example",
                            "url": "https://example.com"
                        }
                    }],
                    "audio": {
                        "id": "audio-1",
                        "data": "aGk=",
                        "expires_at": 1700000600,
                        "transcript": "hi"
                    },
                    "function_call": {"arguments": "{}", "name": "legacy_fn"},
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"arguments": "{\"q\":1}", "name": "search"}
                        },
                        {
                            "id": "call-2",
                            "type": "custom",
                            "custom": {"input": "raw text", "name": "my_tool"}
                        }
                    ]
                }
            }],
            "created": 1_700_000_000,
            "model": "gpt-5.6",
            "object": "chat.completion",
            "moderation": {
                "input": {
                    "type": "moderation_results",
                    "model": "omni-moderation-latest",
                    "results": [{
                        "categories": {"violence": false},
                        "category_applied_input_types": {"violence": ["text"]},
                        "category_scores": {"violence": 0.001},
                        "flagged": false,
                        "model": "omni-moderation-latest",
                        "type": "moderation_result"
                    }]
                },
                "output": {
                    "type": "error",
                    "code": "moderation_unavailable",
                    "message": "try again"
                }
            },
            "service_tier": "default",
            "system_fingerprint": "fp_44709d6fcb",
            "usage": {
                "completion_tokens": 12,
                "prompt_tokens": 9,
                "total_tokens": 21,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 5,
                    "rejected_prediction_tokens": 0
                },
                "prompt_tokens_details": {
                    "audio_tokens": 0,
                    "cache_write_tokens": 2,
                    "cached_tokens": 3
                }
            }
        });

        let completion: CreateChatCompletionResponse =
            serde_json::from_value(body.clone()).unwrap();

        let message = &completion.choices[0].message;
        let tool_calls = message.tool_calls.as_ref().unwrap();
        assert!(matches!(
            &tool_calls[0],
            ChatCompletionMessageToolCallUnion::Function(f) if f.function.name == "search"
        ));
        assert!(matches!(
            &tool_calls[1],
            ChatCompletionMessageToolCallUnion::Custom(c) if c.custom.input == "raw text"
        ));
        let moderation = completion.moderation.as_ref().unwrap();
        assert!(matches!(
            &moderation.input,
            ModerationOutcome::ModerationResults(r) if !r.results[0].flagged
        ));
        assert!(matches!(
            &moderation.output,
            ModerationOutcome::Error(e) if e.code == "moderation_unavailable"
        ));
        let usage = completion.usage.as_ref().unwrap();
        assert_eq!(
            usage
                .prompt_tokens_details
                .as_ref()
                .unwrap()
                .cache_write_tokens,
            Some(2)
        );

        assert_eq!(serde_json::to_value(&completion).unwrap(), body);
    }

    /// Byte-level, not Value-level: some providers' prompt caches hash the
    /// request bytes, so unknown fields must keep the caller's key order
    /// (requires serde_json's `preserve_order` feature).
    #[test]
    fn unknown_field_order_is_preserved() {
        let mut request = CreateChatCompletionRequest {
            model: "openai/gpt-5.6".into(),
            ..Default::default()
        };
        request
            .unknown_fields
            .insert("zeta".into(), serde_json::json!(1));
        request
            .unknown_fields
            .insert("alpha".into(), serde_json::json!(2));

        let wire = serde_json::to_string(&request).unwrap();
        assert!(
            wire.find("zeta").unwrap() < wire.find("alpha").unwrap(),
            "insertion order lost: {wire}"
        );
    }
}
