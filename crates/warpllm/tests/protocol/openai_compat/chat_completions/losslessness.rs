//! The fixtures next door prove the chunks we thought to write down survive.
//! This proves the ones we didn't: generated chunks, every optional field in
//! every state the wire allows, unknown fields at every level, asserting that
//! deserialize → reserialize is the identity.
//!
//! The contract it states, exactly:
//!
//! > For any chunk whose fields are within the nullability OpenAI documents,
//! > warpllm re-emits what it received, byte for byte as JSON values.
//!
//! One consequence of "within the nullability OpenAI documents", deliberate
//! and generated accordingly: a field the spec marks optional but NOT
//! nullable — `role`, `tool_calls`, a tool call's `id` — is absent or a value
//! here, never null. A provider that sends `null` for one anyway is
//! understood, and normalized to absent. That is a permissive read of
//! out-of-spec input, not a round trip of it, and the test below it says so.
//!
//! Everywhere the spec DOES allow a null, one is generated and one must come
//! back. There is no exception left: `ChoiceLogprobs` held the last one until
//! its arrays became `Option<Option<_>>` like the rest.

use proptest::prelude::*;
use serde_json::{Map, Value, json};
use warpllm::protocol::openai_compat::chat_completions::types::CreateChatCompletionStreamResponse;

/// Optional and nullable: absent, `null`, or a value — three states the wire
/// distinguishes and this type has to as well.
fn optional_nullable(value: BoxedStrategy<Value>) -> BoxedStrategy<Option<Value>> {
    prop_oneof![Just(None), Just(Some(Value::Null)), value.prop_map(Some),].boxed()
}

/// Optional, not nullable: absent or a value.
fn optional(value: BoxedStrategy<Value>) -> BoxedStrategy<Option<Value>> {
    prop_oneof![Just(None), value.prop_map(Some)].boxed()
}

fn text() -> BoxedStrategy<Value> {
    "[a-zA-Z0-9 {}\":,_-]{0,24}".prop_map(Value::from).boxed()
}

/// Fields no specification names, which must reach the caller verbatim. The
/// `x_` prefix keeps them from colliding with a field the struct models — a
/// collision would be captured by the typed field and prove nothing.
fn unknown_fields() -> BoxedStrategy<Map<String, Value>> {
    prop::collection::vec((0u8..8, text()), 0..3)
        .prop_map(|entries| {
            entries
                .into_iter()
                .map(|(n, value)| (format!("x_{n}"), value))
                .collect()
        })
        .boxed()
}

/// Assembles an object from named slots, dropping the absent ones.
fn object(slots: Vec<(&str, Option<Value>)>, unknown: Map<String, Value>) -> Value {
    let mut map = Map::new();
    for (name, value) in slots {
        if let Some(value) = value {
            map.insert(name.to_owned(), value);
        }
    }
    map.extend(unknown);
    Value::Object(map)
}

fn tool_call() -> BoxedStrategy<Value> {
    (
        0u32..4,
        optional(text()),
        optional(
            (optional(text()), optional(text()))
                .prop_map(|(arguments, name)| {
                    object(vec![("arguments", arguments), ("name", name)], Map::new())
                })
                .boxed(),
        ),
        optional(text()),
        unknown_fields(),
    )
        .prop_map(|(index, id, function, kind, unknown)| {
            object(
                vec![
                    ("index", Some(json!(index))),
                    ("id", id),
                    ("function", function),
                    ("type", kind),
                ],
                unknown,
            )
        })
        .boxed()
}

fn delta() -> BoxedStrategy<Value> {
    (
        optional_nullable(text()),
        optional(
            (optional(text()), optional(text()))
                .prop_map(|(arguments, name)| {
                    object(vec![("arguments", arguments), ("name", name)], Map::new())
                })
                .boxed(),
        ),
        optional_nullable(text()),
        optional(text()),
        optional(
            prop::collection::vec(tool_call(), 0..3)
                .prop_map(Value::from)
                .boxed(),
        ),
        unknown_fields(),
    )
        .prop_map(
            |(content, function_call, refusal, role, tool_calls, unknown)| {
                object(
                    vec![
                        ("content", content),
                        ("function_call", function_call),
                        ("refusal", refusal),
                        ("role", role),
                        ("tool_calls", tool_calls),
                    ],
                    unknown,
                )
            },
        )
        .boxed()
}

fn token_logprob() -> BoxedStrategy<Value> {
    (
        text(),
        prop_oneof![
            Just(Value::Null),
            prop::collection::vec(any::<u8>(), 0..4).prop_map(Value::from)
        ],
        -20.0f64..0.0,
    )
        .prop_map(|(token, bytes, logprob)| {
            json!({
                "token": token,
                "bytes": bytes,
                "logprob": logprob,
                "top_logprobs": [],
            })
        })
        .boxed()
}

fn logprobs() -> BoxedStrategy<Value> {
    // Optional AND nullable: upstream documents both arrays as required and
    // nullable, and DeepSeek omits them, so all three states occur.
    let tokens = || {
        optional_nullable(
            prop::collection::vec(token_logprob(), 0..2)
                .prop_map(Value::from)
                .boxed(),
        )
    };
    (tokens(), tokens(), unknown_fields())
        .prop_map(|(content, refusal, unknown)| {
            object(vec![("content", content), ("refusal", refusal)], unknown)
        })
        .boxed()
}

fn choice() -> BoxedStrategy<Value> {
    (
        delta(),
        prop_oneof![Just(Value::Null), text()],
        0u32..4,
        optional_nullable(logprobs()),
        unknown_fields(),
    )
        .prop_map(|(delta, finish_reason, index, logprobs, unknown)| {
            object(
                vec![
                    ("delta", Some(delta)),
                    ("finish_reason", Some(finish_reason)),
                    ("index", Some(json!(index))),
                    ("logprobs", logprobs),
                ],
                unknown,
            )
        })
        .boxed()
}

fn usage() -> BoxedStrategy<Value> {
    (0u32..1000, 0u32..1000, 0u32..2000)
        .prop_map(|(completion_tokens, prompt_tokens, total_tokens)| {
            json!({
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
            })
        })
        .boxed()
}

fn chunk() -> BoxedStrategy<Value> {
    (
        text(),
        prop::collection::vec(choice(), 0..3),
        0u64..2_000_000_000,
        text(),
        text(),
        optional_nullable(text()),
        optional(text()),
        optional_nullable(usage()),
        unknown_fields(),
    )
        .prop_map(
            |(
                id,
                choices,
                created,
                model,
                object_type,
                service_tier,
                system_fingerprint,
                usage,
                unknown,
            )| {
                object(
                    vec![
                        ("id", Some(id)),
                        ("choices", Some(Value::from(choices))),
                        ("created", Some(json!(created))),
                        ("model", Some(model)),
                        ("object", Some(object_type)),
                        ("service_tier", service_tier),
                        ("system_fingerprint", system_fingerprint),
                        ("usage", usage),
                    ],
                    unknown,
                )
            },
        )
        .boxed()
}

/// The generator's boundary, stated as behavior rather than as prose: a null
/// where OpenAI documents no null is understood — the alternative is rejecting
/// a chunk over a field nobody read — and comes back out as absent. Every
/// field the spec DOES mark nullable keeps its null, which is what the
/// property above sweeps.
#[test]
fn an_out_of_spec_null_is_read_and_normalized_to_absent() {
    let chunk: CreateChatCompletionStreamResponse = serde_json::from_value(json!({
        "id": "chatcmpl-123",
        "choices": [{
            "delta": {"role": null, "tool_calls": null},
            "finish_reason": null,
            "index": 0
        }],
        "created": 1_700_000_000,
        "model": "gpt-5.6",
        "object": "chat.completion.chunk",
        "system_fingerprint": null
    }))
    .expect("an out-of-spec null must not fail the whole chunk");

    let wire = serde_json::to_value(&chunk).unwrap();
    assert!(wire.get("system_fingerprint").is_none());
    assert!(wire["choices"][0]["delta"].get("role").is_none());
    assert!(wire["choices"][0]["delta"].get("tool_calls").is_none());
    // ...while the in-spec null on the same chunk is still a null.
    assert_eq!(wire["choices"][0]["finish_reason"], Value::Null);
}

proptest! {
    #[test]
    fn any_in_spec_chunk_survives_a_round_trip(body in chunk()) {
        let parsed: CreateChatCompletionStreamResponse =
            serde_json::from_value(body.clone())
                .unwrap_or_else(|e| panic!("{body} failed to deserialize: {e}"));
        prop_assert_eq!(serde_json::to_value(&parsed).unwrap(), body);
    }
}
