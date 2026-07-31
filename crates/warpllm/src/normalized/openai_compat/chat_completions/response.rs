//! Response conversions: OpenAI-compatible wire → normalized (ingest) and
//! normalized → wire (render). Round trips are lossless with zero
//! permitted transformations: dialect-specific fields (`object`,
//! `service_tier`, choice `index`, `refusal`, …) ride
//! `ext["openai_compat"]` at their nesting level and are restored
//! verbatim.

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::normalized::{self, ContentBlock, FinishReason, IngestSource, RawJson};
use crate::protocol::Protocol;
use crate::protocol::openai_compat::chat_completions::types::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCallUnion,
    ChatCompletionResponseMessage, Choice, CompletionUsage, CreateChatCompletionResponse, Function,
    UnknownFields,
};

use super::super::{merged_ext, namespaced, role_from_wire, role_to_wire};

/// Permissive and infallible; the exhaustive destructures at every level
/// make dropping a newly-typed wire field a compile error.
pub(crate) fn ingest_response(response: CreateChatCompletionResponse) -> normalized::ChatResponse {
    // Wire structs are plain serde data; serialization cannot fail.
    let body = serde_json::to_value(&response).expect("wire response serializes");
    let CreateChatCompletionResponse {
        id,
        choices,
        created,
        model,
        object,
        moderation,
        service_tier,
        system_fingerprint,
        usage,
        unknown_fields,
    } = response;
    let mut compat = UnknownFields::new();
    compat.insert("object".into(), Value::String(object));
    if let Some(moderation) = moderation {
        compat.insert("moderation".into(), plain(&moderation));
    }
    if let Some(tier) = service_tier {
        compat.insert("service_tier".into(), Value::String(tier));
    }
    if let Some(fingerprint) = system_fingerprint {
        compat.insert("system_fingerprint".into(), Value::String(fingerprint));
    }
    compat.extend(unknown_fields);
    normalized::ChatResponse {
        id,
        model,
        created: Some(created),
        completions: choices.into_iter().map(ingest_choice).collect(),
        usage: usage.map(ingest_usage),
        ext: namespaced(compat),
        source: Some(IngestSource {
            protocol: Protocol::OpenAiCompat,
            body,
        }),
    }
}

fn ingest_choice(choice: Choice) -> normalized::Completion {
    let Choice {
        finish_reason,
        index,
        logprobs,
        message,
        unknown_fields,
    } = choice;
    let mut compat = UnknownFields::new();
    compat.insert("index".into(), Value::from(index));
    if let Some(logprobs) = logprobs {
        compat.insert("logprobs".into(), plain(&logprobs));
    }
    compat.extend(unknown_fields);
    normalized::Completion {
        message: ingest_message(message),
        finish_reason: FinishReason::from_raw(&finish_reason),
        finish_reason_raw: finish_reason,
        ext: namespaced(compat),
    }
}

fn ingest_message(message: ChatCompletionResponseMessage) -> normalized::Message {
    let ChatCompletionResponseMessage {
        content,
        refusal,
        role,
        annotations,
        audio,
        function_call,
        tool_calls,
        unknown_fields,
    } = message;
    let (role, raw_role) = role_from_wire(role);
    let mut compat = UnknownFields::new();
    if let Some(raw) = raw_role {
        compat.insert("role".into(), Value::String(raw));
    }
    if let Some(refusal) = refusal {
        compat.insert("refusal".into(), Value::String(refusal));
    }
    if let Some(annotations) = annotations {
        compat.insert("annotations".into(), plain(&annotations));
    }
    if let Some(audio) = audio {
        compat.insert("audio".into(), plain(&audio));
    }
    if let Some(function_call) = function_call {
        compat.insert("function_call".into(), plain(&function_call));
    }
    let mut blocks: Vec<ContentBlock> = content
        .map(|text| ContentBlock::Text { text, cache: None })
        .into_iter()
        .collect();
    match tool_calls {
        // `Some([])` is distinguishable from absent; stash it so render
        // re-emits the empty array byte-for-byte.
        Some(calls) if calls.is_empty() => {
            compat.insert("tool_calls".into(), Value::Array(Vec::new()));
        }
        Some(calls) => blocks.extend(calls.into_iter().map(ingest_tool_call)),
        None => {}
    }
    compat.extend(unknown_fields);
    normalized::Message {
        role,
        content: blocks,
        ext: namespaced(compat),
    }
}

/// A plain function call becomes a typed block; anything else — custom
/// tool calls, or calls carrying unknown fields at either level — passes
/// through as an `Unknown` block, re-emitted verbatim in array order.
fn ingest_tool_call(call: ChatCompletionMessageToolCallUnion) -> ContentBlock {
    match &call {
        ChatCompletionMessageToolCallUnion::Function(function_call)
            if function_call.r#type == "function"
                && function_call.unknown_fields.is_empty()
                && function_call.function.unknown_fields.is_empty() =>
        {
            ContentBlock::ToolCall {
                id: function_call.id.clone(),
                name: function_call.function.name.clone(),
                arguments: RawJson::new(function_call.function.arguments.clone()),
            }
        }
        _ => ContentBlock::Unknown(plain(&call)),
    }
}

fn ingest_usage(usage: CompletionUsage) -> normalized::Usage {
    let CompletionUsage {
        completion_tokens,
        prompt_tokens,
        total_tokens,
        completion_tokens_details,
        prompt_tokens_details,
        unknown_fields,
    } = usage;
    let mut compat = UnknownFields::new();
    let mut reasoning_tokens = None;
    let mut cache_read_tokens = None;
    let mut cache_write_tokens = None;
    // A details residue is stashed iff the wire object was present (even
    // empty), so presence itself survives the round trip.
    if let Some(details) = prompt_tokens_details {
        let mut residue = object(plain(&details));
        cache_read_tokens = residue.remove("cached_tokens").and_then(|v| v.as_u64());
        cache_write_tokens = residue
            .remove("cache_write_tokens")
            .and_then(|v| v.as_u64());
        compat.insert("prompt_tokens_details".into(), Value::Object(residue));
    }
    if let Some(details) = completion_tokens_details {
        let mut residue = object(plain(&details));
        reasoning_tokens = residue.remove("reasoning_tokens").and_then(|v| v.as_u64());
        compat.insert("completion_tokens_details".into(), Value::Object(residue));
    }
    compat.extend(unknown_fields);
    normalized::Usage {
        input_tokens: Some(u64::from(prompt_tokens)),
        output_tokens: Some(u64::from(completion_tokens)),
        total_tokens: Some(u64::from(total_tokens)),
        reasoning_tokens,
        cache_read_tokens,
        cache_write_tokens,
        ext: namespaced(compat),
    }
}

/// Infallible: dialect fields restore from ext (a hook that corrupted a
/// stashed value beyond its wire type falls back to dropping that field).
pub(crate) fn render_response(
    response: &normalized::ChatResponse,
    provider: &str,
) -> CreateChatCompletionResponse {
    let mut unknown_fields = merged_ext(&response.ext, provider);
    let object =
        take_string(&mut unknown_fields, "object").unwrap_or_else(|| "chat.completion".to_string());
    CreateChatCompletionResponse {
        id: response.id.clone(),
        choices: response
            .completions
            .iter()
            .enumerate()
            .map(|(position, completion)| render_choice(completion, position, provider))
            .collect(),
        created: response.created.unwrap_or(0),
        model: response.model.clone(),
        object,
        moderation: take_typed(&mut unknown_fields, "moderation"),
        service_tier: take_string(&mut unknown_fields, "service_tier"),
        system_fingerprint: take_string(&mut unknown_fields, "system_fingerprint"),
        usage: response.usage.as_ref().map(|u| render_usage(u, provider)),
        unknown_fields,
    }
}

fn render_choice(completion: &normalized::Completion, position: usize, provider: &str) -> Choice {
    let mut unknown_fields = merged_ext(&completion.ext, provider);
    let index = unknown_fields
        .remove("index")
        .and_then(|v| v.as_u64())
        .unwrap_or(position as u64) as u32;
    Choice {
        finish_reason: completion.finish_reason_raw.clone(),
        index,
        logprobs: take_typed(&mut unknown_fields, "logprobs"),
        message: render_message(&completion.message, provider),
        unknown_fields,
    }
}

fn render_message(message: &normalized::Message, provider: &str) -> ChatCompletionResponseMessage {
    let mut unknown_fields = merged_ext(&message.ext, provider);
    let role = match unknown_fields.remove("role") {
        Some(Value::String(raw)) => raw,
        _ => role_to_wire(message.role).to_string(),
    };
    let mut texts = Vec::new();
    let mut tool_calls = Vec::new();
    for block in &message.content {
        match block {
            ContentBlock::Text { text, .. } => texts.push(text.as_str()),
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => tool_calls.push(ChatCompletionMessageToolCallUnion::Function(
                ChatCompletionMessageToolCall {
                    id: id.clone(),
                    r#type: "function".into(),
                    function: Function {
                        arguments: arguments.as_str().to_string(),
                        name: name.clone(),
                        unknown_fields: UnknownFields::new(),
                    },
                    unknown_fields: UnknownFields::new(),
                },
            )),
            ContentBlock::Unknown(value) => {
                if let Ok(call) = serde_json::from_value(value.clone()) {
                    tool_calls.push(call);
                }
                // A non-tool-call Unknown block is cross-dialect residue
                // with no compat rendering; dropping it is the documented
                // lossy-out path.
            }
            // Reasoning/media blocks only exist cross-dialect; same story.
            _ => {}
        }
    }
    if !tool_calls.is_empty() {
        // Typed tool calls are authoritative over any stashed empty array.
        unknown_fields.remove("tool_calls");
    }
    ChatCompletionResponseMessage {
        // Same-dialect messages carry at most one text block, so the join
        // is exact; joining >1 only occurs cross-dialect.
        content: (!texts.is_empty()).then(|| texts.join("\n")),
        refusal: take_string(&mut unknown_fields, "refusal"),
        role,
        annotations: take_typed(&mut unknown_fields, "annotations"),
        audio: take_typed(&mut unknown_fields, "audio"),
        function_call: take_typed(&mut unknown_fields, "function_call"),
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        unknown_fields,
    }
}

fn render_usage(usage: &normalized::Usage, provider: &str) -> CompletionUsage {
    let mut unknown_fields = merged_ext(&usage.ext, provider);
    let prompt_details = render_details(
        unknown_fields.remove("prompt_tokens_details"),
        &[
            ("cached_tokens", usage.cache_read_tokens),
            ("cache_write_tokens", usage.cache_write_tokens),
        ],
    );
    let completion_details = render_details(
        unknown_fields.remove("completion_tokens_details"),
        &[("reasoning_tokens", usage.reasoning_tokens)],
    );
    CompletionUsage {
        completion_tokens: usage.output_tokens.unwrap_or(0) as u32,
        prompt_tokens: usage.input_tokens.unwrap_or(0) as u32,
        total_tokens: usage.total_tokens.unwrap_or(0) as u32,
        completion_tokens_details: completion_details,
        prompt_tokens_details: prompt_details,
        unknown_fields,
    }
}

/// Rebuilds a details object from its ext residue plus the typed fields
/// lifted out at ingest. Emitted iff the residue was present (preserving
/// wire presence) or a lifted field is set.
fn render_details<T: DeserializeOwned>(
    residue: Option<Value>,
    lifted: &[(&str, Option<u64>)],
) -> Option<T> {
    let residue = match residue {
        Some(Value::Object(fields)) => Some(fields),
        _ => None,
    };
    if residue.is_none() && lifted.iter().all(|(_, value)| value.is_none()) {
        return None;
    }
    let mut fields = residue.unwrap_or_default();
    for (key, value) in lifted {
        if let Some(tokens) = value {
            fields.insert((*key).to_string(), Value::from(*tokens));
        }
    }
    serde_json::from_value(Value::Object(fields)).ok()
}

fn take_string(fields: &mut UnknownFields, key: &str) -> Option<String> {
    match fields.remove(key) {
        Some(Value::String(value)) => Some(value),
        _ => None,
    }
}

fn take_typed<T: DeserializeOwned>(fields: &mut UnknownFields, key: &str) -> Option<T> {
    fields
        .remove(key)
        .and_then(|value| serde_json::from_value(value).ok())
}

/// Wire structs are plain serde data; serialization cannot fail.
fn plain<T: serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).expect("wire data serializes")
}

fn object(value: Value) -> UnknownFields {
    match value {
        Value::Object(fields) => fields,
        _ => UnknownFields::new(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn parse(body: Value) -> CreateChatCompletionResponse {
        serde_json::from_value(body).unwrap()
    }

    /// The maximal body: every documented field, unknown fields at every
    /// nesting level, function + custom tool calls, both moderation arms.
    fn maximal_body() -> Value {
        json!({
            "id": "chatcmpl-123",
            "choices": [{
                "finish_reason": "tool_calls",
                "index": 3,
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
                    "refusal": "no thanks",
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
                            "function": {"arguments": "{\"z\":1,\"a\":2}", "name": "search"}
                        },
                        {
                            "id": "call-2",
                            "type": "custom",
                            "custom": {"input": "raw text", "name": "my_tool"}
                        },
                        {
                            "id": "call-3",
                            "type": "function",
                            "function": {"arguments": "{}", "name": "extended"},
                            "vendor_extra": true
                        }
                    ],
                    "reasoning_content": "step by step"
                },
                "new_choice_field": true
            }],
            "created": 1_700_000_000,
            "model": "gpt-5.6-2024-08-06",
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
                "output": {"type": "error", "code": "moderation_unavailable", "message": "try again"}
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
                },
                "new_usage_field": 7
            },
            "new_top_level_field": "surprise"
        })
    }

    /// The mandated test: an OpenAI-compatible response must survive
    /// normalization and come back out with ZERO permitted
    /// transformations.
    #[test]
    fn openai_compat_response_round_trip_is_lossless() {
        let body = maximal_body();
        let normalized = ingest_response(parse(body.clone()));
        let rendered = render_response(&normalized, "openai");
        assert_eq!(serde_json::to_value(&rendered).unwrap(), body);

        // Spot-check the typed views the IR exposes along the way.
        assert_eq!(normalized.id, "chatcmpl-123");
        assert_eq!(normalized.model, "gpt-5.6-2024-08-06");
        assert_eq!(normalized.created, Some(1_700_000_000));
        let completion = &normalized.completions[0];
        assert_eq!(completion.finish_reason, FinishReason::ToolCalls);
        assert_eq!(completion.finish_reason_raw, "tool_calls");
        assert_eq!(completion.ext["openai_compat"]["index"], json!(3));
    }

    #[test]
    fn minimal_response_round_trips() {
        // Optional fields absent; content/refusal explicit null contrast.
        let body = json!({
            "id": "chatcmpl-1",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": null, "refusal": null, "role": "assistant"}
            }],
            "created": 1_700_000_000,
            "model": "deepseek-v4-flash",
            "object": "chat.completion"
        });
        let normalized = ingest_response(parse(body.clone()));
        assert!(normalized.completions[0].message.content.is_empty());
        let rendered = render_response(&normalized, "deepseek");
        assert_eq!(serde_json::to_value(&rendered).unwrap(), body);
    }

    #[test]
    fn plain_function_tool_call_becomes_toolcall_block() {
        let normalized = ingest_response(parse(maximal_body()));
        let content = &normalized.completions[0].message.content;
        // Text block first, then tool calls in array order.
        assert!(matches!(&content[0], ContentBlock::Text { text, .. } if text == "Hello there!"));
        match &content[1] {
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                assert_eq!(id, "call-1");
                assert_eq!(name, "search");
                // Byte-exact, key order untouched.
                assert_eq!(arguments.as_str(), "{\"z\":1,\"a\":2}");
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
        // Custom and extended calls pass through as Unknown, verbatim.
        assert!(
            matches!(&content[2], ContentBlock::Unknown(v) if v["id"] == "call-2" && v["custom"]["input"] == "raw text")
        );
        assert!(
            matches!(&content[3], ContentBlock::Unknown(v) if v["id"] == "call-3" && v["vendor_extra"] == true)
        );
    }

    #[test]
    fn empty_tool_calls_array_survives() {
        let body = json!({
            "id": "chatcmpl-1",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "hi",
                    "refusal": null,
                    "role": "assistant",
                    "tool_calls": []
                }
            }],
            "created": 1_700_000_000,
            "model": "gpt-5.6",
            "object": "chat.completion"
        });
        let normalized = ingest_response(parse(body.clone()));
        let rendered = render_response(&normalized, "openai");
        assert_eq!(serde_json::to_value(&rendered).unwrap(), body);
    }

    #[test]
    fn usage_maps_typed_token_fields() {
        let normalized = ingest_response(parse(maximal_body()));
        let usage = normalized.usage.as_ref().unwrap();
        assert_eq!(usage.input_tokens, Some(9));
        assert_eq!(usage.output_tokens, Some(12));
        assert_eq!(usage.total_tokens, Some(21));
        assert_eq!(usage.reasoning_tokens, Some(5));
        assert_eq!(usage.cache_read_tokens, Some(3));
        assert_eq!(usage.cache_write_tokens, Some(2));
        assert_eq!(usage.ext["openai_compat"]["new_usage_field"], json!(7));
    }

    #[test]
    fn empty_details_object_presence_survives() {
        let body = json!({
            "id": "chatcmpl-1",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": "hi", "refusal": null, "role": "assistant"}
            }],
            "created": 1_700_000_000,
            "model": "gpt-5.6",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 1,
                "prompt_tokens": 2,
                "total_tokens": 3,
                "prompt_tokens_details": {}
            }
        });
        let normalized = ingest_response(parse(body.clone()));
        let rendered = render_response(&normalized, "openai");
        assert_eq!(serde_json::to_value(&rendered).unwrap(), body);
    }

    #[test]
    fn dialect_extras_land_in_ext() {
        let normalized = ingest_response(parse(maximal_body()));
        let ext = &normalized.ext["openai_compat"];
        assert_eq!(ext["object"], "chat.completion");
        assert_eq!(ext["service_tier"], "default");
        assert_eq!(ext["system_fingerprint"], "fp_44709d6fcb");
        assert_eq!(ext["new_top_level_field"], "surprise");
        let message = &normalized.completions[0].message;
        assert_eq!(message.ext["openai_compat"]["refusal"], "no thanks");
        assert_eq!(
            message.ext["openai_compat"]["reasoning_content"],
            "step by step"
        );
    }

    #[test]
    fn ingest_populates_source() {
        let wire = parse(maximal_body());
        let normalized = ingest_response(wire.clone());
        let source = normalized.source.as_ref().unwrap();
        assert_eq!(source.protocol, Protocol::OpenAiCompat);
        assert_eq!(source.body, serde_json::to_value(&wire).unwrap());
    }
}
