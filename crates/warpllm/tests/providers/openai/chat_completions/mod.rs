use crate::openai_common::{client_for, openai_completion_body, request, with_openai_key};
use serde_json::json;
use warpllm::{Error, Origin};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[test]
fn openai_happy_path() {
    with_openai_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer sk-test-openai"))
            .respond_with(ResponseTemplate::new(200).set_body_json(openai_completion_body()))
            .expect(1)
            .mount(&server)
            .await;

        let completion = client_for(&server)
            .chat_completion(request("openai/gpt-5.6"))
            .await
            .unwrap();

        assert_eq!(completion.id, "chatcmpl-123");
        assert_eq!(completion.object, "chat.completion");
        // Echoes the caller's provider-prefixed string, not the upstream name.
        assert_eq!(completion.model, "openai/gpt-5.6");
        assert_eq!(
            completion.choices[0].message.content.as_deref(),
            Some("Hello there!")
        );
        assert_eq!(completion.choices[0].finish_reason, "stop");
        assert_eq!(completion.service_tier.as_deref(), Some("default"));
        assert_eq!(
            completion.system_fingerprint.as_deref(),
            Some("fp_44709d6fcb")
        );
        let usage = completion.usage.as_ref().unwrap();
        assert_eq!(usage.total_tokens, 21);
        let prompt_details = usage.prompt_tokens_details.as_ref().unwrap();
        assert_eq!(prompt_details.cached_tokens, Some(3));
        assert_eq!(prompt_details.cache_write_tokens, Some(2));
        assert_eq!(
            usage
                .completion_tokens_details
                .as_ref()
                .unwrap()
                .reasoning_tokens,
            Some(5)
        );

        let sent: serde_json::Value =
            serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
        // The provider prefix must be stripped from the outbound model.
        assert_eq!(sent["model"], "gpt-5.6");
        assert_eq!(sent["messages"][0]["content"], "hi");
        assert!(sent.get("stream").is_none());
    });
}

#[test]
fn unknown_request_fields_are_forwarded() {
    with_openai_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(openai_completion_body()))
            .expect(1)
            .mount(&server)
            .await;

        let mut req = request("openai/gpt-5.6");
        // Vendor extensions pass through verbatim...
        req.unknown_fields.insert("vendor_beta".into(), json!(40));
        req.unknown_fields.insert("seed".into(), json!(7));
        // ...including params OpenAI doesn't document (top_k): the provider
        // is the authority on its own parameters.
        req.unknown_fields.insert("top_k".into(), json!(40));
        req.messages[0]
            .unknown_fields
            .insert("name".into(), json!("alice"));
        client_for(&server).chat_completion(req).await.unwrap();

        let sent: serde_json::Value =
            serde_json::from_slice(&server.received_requests().await.unwrap()[0].body).unwrap();
        assert_eq!(sent["vendor_beta"], 40);
        assert_eq!(sent["seed"], 7);
        assert_eq!(sent["top_k"], 40);
        assert_eq!(sent["messages"][0]["name"], "alice");
    });
}

#[test]
fn openai_error_statuses_map_to_provider_error() {
    with_openai_key(async {
        for (status, error_type, message, code) in [
            (
                401,
                "invalid_request_error",
                "Incorrect API key provided",
                "authentication",
            ),
            (
                429,
                "rate_limit_exceeded",
                "Rate limit reached",
                "rate_limited",
            ),
            (
                500,
                "server_error",
                "The server had an error",
                "provider_server_error",
            ),
        ] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(status).set_body_json(json!({
                    "error": {"message": message, "type": error_type}
                })))
                .mount(&server)
                .await;

            let err = client_for(&server)
                .chat_completion(request("openai/gpt-5.6"))
                .await
                .unwrap_err();

            // The failure classified onto its own variant, and the
            // provider's own spelling retained beside it.
            assert_eq!(err.code(), code, "{status} classified wrong");
            assert_eq!(err.origin(), Origin::Provider);
            let upstream = err.provider_error().expect("a provider failure");
            assert_eq!(upstream.provider, "openai");
            assert_eq!(upstream.status, status);
            assert_eq!(upstream.error_type.as_deref(), Some(error_type));
            assert_eq!(upstream.message, message);
        }
    });
}

#[test]
fn malformed_success_body_maps_to_decode_error() {
    with_openai_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&server)
            .await;

        let err = client_for(&server)
            .chat_completion(request("openai/gpt-5.6"))
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                Error::Decode {
                    provider: "openai",
                    ..
                }
            ),
            "{err:?}"
        );
    });
}

#[test]
fn unparseable_error_body_falls_back_to_raw_text() {
    with_openai_key(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(503).set_body_string("upstream overloaded"))
            .mount(&server)
            .await;

        let err = client_for(&server)
            .chat_completion(request("openai/gpt-5.6"))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Overloaded(_)), "{err:?}");
        let upstream = err.provider_error().expect("a provider failure");
        assert_eq!(upstream.status, 503);
        assert_eq!(upstream.error_type, None);
        assert_eq!(upstream.message, "upstream overloaded");
        // Nothing parsed, so the raw body is the only evidence.
        assert_eq!(upstream.raw_body, "upstream overloaded");
    });
}

#[test]
fn stream_true_is_rejected_before_any_request() {
    with_openai_key(async {
        let server = MockServer::start().await;
        // No mock mounted: a request reaching the server would 404 into a
        // Provider error, so getting NotImplemented proves we rejected early.
        let mut req = request("openai/gpt-5.6");
        req.stream = Some(true);

        let err = client_for(&server).chat_completion(req).await.unwrap_err();
        assert!(
            matches!(&err, Error::NotImplemented("streaming")),
            "{err:?}"
        );
        assert!(server.received_requests().await.unwrap().is_empty());
    });
}

#[test]
fn response_unknowns_and_tool_calls_round_trip_to_caller() {
    with_openai_key(async {
        let mut body = openai_completion_body();
        body["choices"][0]["message"]["tool_calls"] = json!([
            {
                "id": "call-1",
                "type": "function",
                "function": {"arguments": "{\"z\":1,\"a\":2}", "name": "search"}
            },
            {
                "id": "call-2",
                "type": "custom",
                "custom": {"input": "raw text", "name": "my_tool"}
            }
        ]);
        body["choices"][0]["message"]["reasoning_content"] = json!("step by step");
        body["choices"][0]["vendor_choice_field"] = json!(true);
        body["vendor_top_field"] = json!("surprise");

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body.clone()))
            .expect(1)
            .mount(&server)
            .await;

        let completion = client_for(&server)
            .chat_completion(request("openai/gpt-5.6"))
            .await
            .unwrap();

        // The full pipeline (ingest → normalized → render) must hand the
        // caller exactly what the provider sent, model echo aside. The
        // explicit `"logprobs": null` is dropped by the wire types themselves
        // (Option + skip_serializing_if, predating normalization).
        let mut expected = body;
        expected["model"] = json!("openai/gpt-5.6");
        expected["choices"][0]
            .as_object_mut()
            .unwrap()
            .remove("logprobs");
        assert_eq!(serde_json::to_value(&completion).unwrap(), expected);
    });
}

#[test]
fn invalid_model_strings_are_rejected() {
    with_openai_key(async {
        let server = MockServer::start().await;
        let client = client_for(&server);

        let err = client
            .chat_completion(request("gpt-5.6"))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("no registered model spec"),
            "{err}"
        );

        let err = client
            .chat_completion(request("mistral/large"))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("no registered model spec"),
            "{err}"
        );
    });
}
