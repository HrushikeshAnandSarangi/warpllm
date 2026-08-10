//! The one check no fixture can stand in for: what providers actually put on
//! the wire today. Recorded transcripts age, and a provider that adds a field
//! or starts omitting one says nothing until something reads its real output.
//!
//! Opt-in and ignored by default, because CI needs no API keys and this needs
//! several:
//!
//! ```text
//! OPENAI_API_KEY=... DEEPSEEK_API_KEY=... \
//!   cargo test -p warpllm --test live_stream -- --ignored --nocapture
//! ```
//!
//! Every provider whose key is absent is skipped and named. A chunk that fails
//! to round-trip is printed in full: that output is a fixture, and belongs
//! under `fixtures/transcript/` so the keyless suite catches it from then on.
//!
//! This speaks HTTP directly rather than through warpllm, since warpllm has no
//! streaming transport yet — which is exactly why it is worth running now. The
//! REQUEST is built from [`CreateChatCompletionRequest`], so what it proves
//! about that shape is real: `stream_options` rides along in `unknown_fields`,
//! and the provider accepts what warpllm would have sent.

use std::time::Duration;

use warpllm::protocol::openai_compat::chat_completions::types::CreateChatCompletionStreamResponse;
use warpllm::{ChatCompletionRequestMessage, CreateChatCompletionRequest, fetch_model};

/// One model per provider on the roster. Cheap and short-answered on purpose:
/// this is a shape check, not a capability survey.
const MODELS: [&str; 3] = [
    "openai/gpt-5-nano",
    "deepseek/deepseek-v4-flash",
    "openrouter/~deepseek/deepseek-v4-flash-latest",
];

#[tokio::test]
#[ignore = "needs live provider API keys; run with --ignored"]
async fn every_configured_provider_streams_chunks_warpllm_can_hold() {
    let mut checked = Vec::new();
    let mut skipped = Vec::new();

    for model_str in MODELS {
        let (provider, model) = fetch_model(model_str).expect("roster entry");
        let Some(key) = provider
            .env_api_key()
            .and_then(|name| std::env::var(name).ok())
        else {
            skipped.push(model_str);
            continue;
        };

        let mut request = CreateChatCompletionRequest {
            model: model.model().to_owned(),
            messages: vec![ChatCompletionRequestMessage {
                role: "user".into(),
                content: "Reply with exactly: hello".into(),
                unknown_fields: Default::default(),
            }],
            max_tokens: Some(16),
            stream: Some(true),
            ..Default::default()
        };
        // Unmodeled, and reaching the provider anyway — the catch-all doing
        // the job it exists for, on a request rather than a reply.
        request.unknown_fields.insert(
            "stream_options".into(),
            serde_json::json!({"include_usage": true}),
        );

        let body = reqwest::Client::new()
            .post(format!("{}/chat/completions", provider.base_url()))
            .bearer_auth(key)
            .json(&request)
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .unwrap_or_else(|e| panic!("{model_str}: {e}"))
            .text()
            .await
            .unwrap_or_else(|e| panic!("{model_str}: {e}"));

        assert_chunks_round_trip(model_str, &body);
        checked.push(model_str);
    }

    println!("streamed: {checked:?}");
    println!("skipped for want of a key: {skipped:?}");
    assert!(
        !checked.is_empty(),
        "no provider key was set, so nothing was checked: {skipped:?}"
    );
}

/// Every event of a live SSE body, parsed and re-emitted verbatim.
fn assert_chunks_round_trip(model_str: &str, body: &str) {
    let payloads: Vec<&str> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| *payload != "[DONE]")
        .collect();
    assert!(
        payloads.len() > 1,
        "{model_str} answered with no stream:\n{body}"
    );

    let mut text = String::new();
    let mut finished = false;
    for payload in payloads {
        let value: serde_json::Value = serde_json::from_str(payload)
            .unwrap_or_else(|e| panic!("{model_str} sent a non-JSON event: {payload}: {e}"));
        let chunk: CreateChatCompletionStreamResponse = serde_json::from_value(value.clone())
            .unwrap_or_else(|e| {
                panic!("{model_str} sent a chunk warpllm cannot hold: {e}\n{value:#}")
            });
        assert_eq!(
            serde_json::to_value(&chunk).unwrap(),
            value,
            "{model_str} lost a field on re-serialization; add this event to \
             fixtures/transcript/ and fix the shape:\n{value:#}"
        );
        for choice in &chunk.choices {
            if let Some(Some(fragment)) = &choice.delta.content {
                text.push_str(fragment);
            }
            finished |= choice.finish_reason.is_some();
        }
    }

    assert!(finished, "{model_str} never sent a finish_reason");
    assert!(!text.trim().is_empty(), "{model_str} streamed no content");
}
