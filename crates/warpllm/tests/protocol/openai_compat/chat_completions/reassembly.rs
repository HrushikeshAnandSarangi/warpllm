//! What a caller actually does with a stream: iterate the chunks and add them
//! up. The shapes can be field-perfect and still be unusable, so this walks
//! recorded transcripts the way a consumer would and checks the total.
//!
//! Two claims per transcript, and they fail for different reasons:
//! - every event survives a deserialize → reserialize round trip, which is
//!   the permissiveness claim, one provider at a time;
//! - the events add up to the completion the same request would have returned
//!   unstreamed, which is the claim that the delta shapes carry enough to be
//!   reassembled at all.
//!
//! `reassemble` below is a consumer's-eye reference, not warpllm's future
//! ingest: it drops each chunk's `unknown_fields` rather than deciding how a
//! per-chunk extension should merge, and it reads only the fields a client
//! needs. When streaming reaches the gateway, that decision gets made there,
//! against these same transcripts.
//!
//! The transcripts are hand-built from the API reference and provider docs
//! until captured ones replace them; a real `curl -N` capture is strictly
//! better and drops in unchanged.

use std::collections::BTreeMap;

use warpllm::CreateChatCompletionResponse;
use warpllm::protocol::openai_compat::chat_completions::types::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCallUnion,
    ChatCompletionResponseMessage, ChatCompletionStreamResponseDelta, Choice, CompletionUsage,
    CreateChatCompletionStreamResponse, Function, UnknownFields,
};

fn transcript(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/tests/protocol/openai_compat/chat_completions/fixtures/transcript/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

/// The `data:` payloads of an SSE body, `[DONE]` excluded.
fn payloads(sse: &str) -> Vec<&str> {
    sse.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| *payload != "[DONE]")
        .collect()
}

fn chunks(sse: &str) -> Vec<CreateChatCompletionStreamResponse> {
    payloads(sse)
        .into_iter()
        .map(|payload| {
            serde_json::from_str(payload).unwrap_or_else(|e| panic!("{payload} failed: {e}"))
        })
        .collect()
}

#[derive(Default)]
struct AccumulatedChoice {
    role: Option<String>,
    content: Option<String>,
    refusal: Option<String>,
    finish_reason: Option<String>,
    tool_calls: BTreeMap<u32, AccumulatedToolCall>,
}

#[derive(Default)]
struct AccumulatedToolCall {
    id: Option<String>,
    r#type: Option<String>,
    name: Option<String>,
    arguments: String,
}

/// Chunks in, the completion they add up to out.
fn reassemble(chunks: &[CreateChatCompletionStreamResponse]) -> CreateChatCompletionResponse {
    let first = chunks.first().expect("a stream has at least one chunk");
    let mut accumulated: BTreeMap<u32, AccumulatedChoice> = BTreeMap::new();
    let mut usage = None;
    let mut service_tier = None;

    for chunk in chunks {
        for choice in &chunk.choices {
            let entry = accumulated.entry(choice.index).or_default();
            absorb(entry, &choice.delta);
            if let Some(reason) = &choice.finish_reason {
                entry.finish_reason = Some(reason.clone());
            }
        }
        // Both are `null` on every chunk but the one that carries them, so the
        // last stated value wins rather than the last chunk's.
        if let Some(Some(totals)) = &chunk.usage {
            usage = Some(totals.clone());
        }
        if let Some(Some(tier)) = &chunk.service_tier {
            service_tier = Some(tier.clone());
        }
    }

    CreateChatCompletionResponse {
        id: first.id.clone(),
        choices: accumulated.into_iter().map(complete).collect(),
        created: first.created,
        model: first.model.clone(),
        object: "chat.completion".into(),
        moderation: None,
        service_tier: service_tier.map(Some),
        system_fingerprint: first.system_fingerprint.clone(),
        usage,
        unknown_fields: UnknownFields::new(),
    }
}

fn absorb(entry: &mut AccumulatedChoice, delta: &ChatCompletionStreamResponseDelta) {
    if let Some(role) = &delta.role {
        entry.role = Some(role.clone());
    }
    if let Some(Some(fragment)) = &delta.content {
        entry.content.get_or_insert_default().push_str(fragment);
    }
    if let Some(Some(fragment)) = &delta.refusal {
        entry.refusal.get_or_insert_default().push_str(fragment);
    }
    for call in delta.tool_calls.iter().flatten() {
        // `index`, not `id`: the id arrives once, on whichever fragment
        // happens to open the call.
        let accumulating = entry.tool_calls.entry(call.index).or_default();
        if let Some(id) = &call.id {
            accumulating.id = Some(id.clone());
        }
        if let Some(kind) = &call.r#type {
            accumulating.r#type = Some(kind.clone());
        }
        let Some(function) = &call.function else {
            continue;
        };
        if let Some(name) = &function.name {
            accumulating.name = Some(name.clone());
        }
        if let Some(fragment) = &function.arguments {
            accumulating.arguments.push_str(fragment);
        }
    }
}

fn complete((index, accumulated): (u32, AccumulatedChoice)) -> Choice {
    let tool_calls: Vec<_> = accumulated
        .tool_calls
        .into_values()
        .map(|call| {
            ChatCompletionMessageToolCallUnion::Function(ChatCompletionMessageToolCall {
                id: call.id.unwrap_or_default(),
                r#type: call.r#type.unwrap_or_else(|| "function".into()),
                function: Function {
                    arguments: call.arguments,
                    name: call.name.unwrap_or_default(),
                    unknown_fields: UnknownFields::new(),
                },
                unknown_fields: UnknownFields::new(),
            })
        })
        .collect();

    Choice {
        finish_reason: accumulated.finish_reason.unwrap_or_default(),
        index,
        logprobs: None,
        message: ChatCompletionResponseMessage {
            content: accumulated.content,
            refusal: accumulated.refusal,
            role: accumulated.role.unwrap_or_else(|| "assistant".into()),
            annotations: None,
            audio: None,
            function_call: None,
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            unknown_fields: UnknownFields::new(),
        },
        unknown_fields: UnknownFields::new(),
    }
}

const TRANSCRIPTS: [&str; 2] = ["openai-text", "deepseek-tool-call"];

/// Every event of every transcript, verbatim in and verbatim out.
#[test]
fn every_streamed_event_round_trips_losslessly() {
    for name in TRANSCRIPTS {
        let sse = transcript(&format!("{name}.sse"));
        let payloads = payloads(&sse);
        assert!(payloads.len() > 1, "{name} has no events");
        for payload in payloads {
            let value: serde_json::Value = serde_json::from_str(payload).unwrap();
            let chunk: CreateChatCompletionStreamResponse = serde_json::from_value(value.clone())
                .unwrap_or_else(|e| panic!("{name}: {payload} failed: {e}"));
            assert_eq!(
                serde_json::to_value(&chunk).unwrap(),
                value,
                "lossy round trip in {name}"
            );
        }
    }
}

/// ...and the events add up to the completion the request would have returned
/// unstreamed. Content arrives in fragments, tool-call arguments arrive split
/// mid-JSON, and `usage` arrives after the choice has already finished.
#[test]
fn a_transcript_adds_up_to_its_completion() {
    for name in TRANSCRIPTS {
        let assembled = reassemble(&chunks(&transcript(&format!("{name}.sse"))));
        let expected: serde_json::Value =
            serde_json::from_str(&transcript(&format!("{name}.completion.json"))).unwrap();
        assert_eq!(
            serde_json::to_value(&assembled).unwrap(),
            expected,
            "{name} did not add up"
        );
    }
}

/// The reassembly above is only evidence if a client can read the pieces it
/// reads: a chunk that opens a tool call names it, and the fragments that
/// follow carry only an index and more argument text.
#[test]
fn a_tool_call_is_reassembled_from_fragments_by_index() {
    let chunks = chunks(&transcript("deepseek-tool-call.sse"));
    let fragments: Vec<_> = chunks
        .iter()
        .flat_map(|chunk| &chunk.choices)
        .filter_map(|choice| choice.delta.tool_calls.as_ref())
        .flatten()
        .collect();

    assert_eq!(fragments.len(), 3, "the call arrived in three pieces");
    assert_eq!(fragments[0].id.as_deref(), Some("call_0_5f3a91c2"));
    assert!(fragments[1].id.is_none(), "the id arrives once");
    assert!(fragments.iter().all(|fragment| fragment.index == 0));
    let arguments: String = fragments
        .iter()
        .filter_map(|fragment| fragment.function.as_ref()?.arguments.as_deref())
        .collect();
    assert_eq!(arguments, r#"{"city":"Seoul"}"#);
    let usage: &CompletionUsage = chunks
        .last()
        .unwrap()
        .usage
        .as_ref()
        .unwrap()
        .as_ref()
        .expect("the final chunk carries the totals");
    assert_eq!(usage.total_tokens, 111);
}
