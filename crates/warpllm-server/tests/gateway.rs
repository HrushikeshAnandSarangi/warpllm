//! End-to-end gateway tests: a real axum server on an ephemeral port in
//! front of a wiremock "OpenAI" upstream.

use std::future::Future;
use std::sync::Arc;

use serde_json::{Value, json};
use warpllm_server::{AppState, router};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// The gateway-held provider key, placed in the environment by
/// [`with_gateway_key`] — the client's only key source.
const GATEWAY_KEY: &str = "sk-gateway";

/// Runs `body` holding temp-env's lock, with `OPENAI_API_KEY` set to `key`.
///
/// EVERY test in this binary goes through this, including the ones that never
/// reach the upstream. Building a gateway READS the environment — `Client::new`
/// resolves its providers once, up front — so a test that sets nothing is still
/// a reader, and a reader running beside a writer is the data race that made
/// `set_var` unsafe in edition 2024. Only a shared lock rules it out.
///
/// These used to split: key-reading tests took the lock and the rest stayed
/// `#[tokio::test]` for parallelism, which was sound while keys resolved at
/// request time and nothing else touched the environment. It stopped being
/// sound the moment construction started reading.
///
/// A runtime per test rather than `#[tokio::test]` because `async_with_vars`
/// cannot hold the lock across an await.
fn with_env<F: Future<Output = ()>>(key: Option<&str>, body: F) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    temp_env::with_var("OPENAI_API_KEY", key, || runtime.block_on(body));
}

/// The gateway holding its provider key, for the tests that reach upstream.
fn with_gateway_key<F: Future<Output = ()>>(body: F) {
    with_env(Some(GATEWAY_KEY), body);
}

/// No provider key at all, for the tests that must fail before key resolution
/// (bad route, `stream: true`, unknown model, health). Proving that from an
/// environment where the key is ABSENT is a stronger claim than proving it from
/// one where the key merely went unused.
fn without_key<F: Future<Output = ()>>(body: F) {
    with_env(None, body);
}

/// Serves the gateway against the given upstream, returning its base URL.
async fn spawn_app(upstream_uri: &str) -> String {
    let client = warpllm::Client::new(warpllm::ClientConfig {
        base_url: Some(upstream_uri.to_string()),
        timeout_secs: Some(5),
    })
    .unwrap();
    let app = router(AppState {
        client: Arc::new(client),
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

fn request_body() -> Value {
    json!({
        "model": "openai/gpt-5.6",
        "messages": [{"role": "user", "content": "hi"}]
    })
}

fn completion_body() -> Value {
    json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "gpt-5.6-2024-08-06",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there!", "refusal": null},
            "finish_reason": "stop",
            "logprobs": null
        }]
    })
}

#[test]
fn non_stream_happy_path_uses_gateway_key_and_echoes_model() {
    with_gateway_key(async {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", format!("Bearer {GATEWAY_KEY}")))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
            .expect(1)
            .mount(&upstream)
            .await;
        let gateway = spawn_app(&upstream.uri()).await;

        // No Authorization header needed: the gateway holds the provider key.
        let response = reqwest::Client::new()
            .post(format!("{gateway}/v1/chat/completions"))
            .json(&request_body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["model"], "openai/gpt-5.6");
        assert_eq!(body["choices"][0]["message"]["content"], "Hello there!");

        let sent: Value =
            serde_json::from_slice(&upstream.received_requests().await.unwrap()[0].body).unwrap();
        assert_eq!(sent["model"], "gpt-5.6");
    });
}

#[test]
fn unprefixed_route_is_404() {
    without_key(async {
        let upstream = MockServer::start().await;
        let gateway = spawn_app(&upstream.uri()).await;

        let response = reqwest::Client::new()
            .post(format!("{gateway}/chat/completions"))
            .json(&request_body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 404);
    });
}

#[test]
fn caller_bearer_is_ignored_never_forwarded() {
    with_gateway_key(async {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(header("authorization", format!("Bearer {GATEWAY_KEY}")))
            .respond_with(ResponseTemplate::new(200).set_body_json(completion_body()))
            .expect(1)
            .mount(&upstream)
            .await;
        let gateway = spawn_app(&upstream.uri()).await;

        // The caller sends its own bearer; the upstream must still see the
        // gateway's key (the mock 404s any other Authorization value).
        let response = reqwest::Client::new()
            .post(format!("{gateway}/v1/chat/completions"))
            .bearer_auth("sk-caller")
            .json(&request_body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
    });
}

/// An upstream failure, reported in OpenAI's vocabulary rather than the
/// provider's. The status here already matches what OpenAI would send, so it
/// survives unchanged — what the gateway proves is that the body a caller
/// reads is the same one the in-process SDK hands back.
#[test]
fn an_upstream_failure_is_reported_in_openai_vocabulary() {
    with_gateway_key(async {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "30")
                    .insert_header("x-request-id", "req-upstream-1")
                    .set_body_json(json!({
                        "error": {"message": "Rate limit reached", "type": "rate_limit_exceeded"}
                    })),
            )
            .mount(&upstream)
            .await;
        let gateway = spawn_app(&upstream.uri()).await;

        let response = reqwest::Client::new()
            .post(format!("{gateway}/v1/chat/completions"))
            .json(&request_body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 429);
        // The upstream's own Retry-After is re-emitted as a real header:
        // standard clients and proxies back off on the header and never
        // read warpllm's JSON.
        assert_eq!(
            response.headers().get("retry-after").unwrap(),
            "30",
            "the upstream's Retry-After did not survive the gateway"
        );

        let body: Value = response.json().await.unwrap();
        // `type` and `code` are OpenAI's own spellings, identical to what a
        // caller reaching warpllm in-process would see. warpllm's taxonomy
        // rides beside them, on this surface only.
        assert_eq!(body["error"]["type"], "rate_limit_error");
        assert_eq!(body["error"]["code"], "rate_limit_exceeded");
        assert_eq!(body["error"]["origin"], "provider");
        assert_eq!(body["error"]["warpllm_code"], "rate_limited");
        assert_eq!(body["error"]["provider"], "openai");
    });
}

/// The finding this split exists for, end to end through the HTTP gateway:
/// a quota exhaustion arrives as a 429 and reads exactly like a rate limit,
/// so a caller reading only the status backs off against a billing problem.
#[test]
fn quota_exhaustion_is_not_reported_as_a_rate_limit() {
    with_gateway_key(async {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429).set_body_json(json!({
                "error": {
                    "message": "You exceeded your current quota",
                    "type": "insufficient_quota",
                    "code": "insufficient_quota"
                }
            })))
            .mount(&upstream)
            .await;
        let gateway = spawn_app(&upstream.uri()).await;

        let response = reqwest::Client::new()
            .post(format!("{gateway}/v1/chat/completions"))
            .json(&request_body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 429);
        let body: Value = response.json().await.unwrap();
        assert_eq!(
            body["error"]["code"], "insufficient_quota",
            "reported as a rate limit, a backoff loop never resolves this"
        );
        assert_eq!(body["error"]["origin"], "provider");
        // warpllm's own name for it stays reachable for anyone debugging.
        assert_eq!(body["error"]["warpllm_code"], "quota_exceeded");
    });
}

#[test]
fn stream_requests_are_501_before_upstream() {
    without_key(async {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&upstream)
            .await;
        let gateway = spawn_app(&upstream.uri()).await;

        let mut body = request_body();
        body["stream"] = json!(true);
        let response = reqwest::Client::new()
            .post(format!("{gateway}/v1/chat/completions"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 501);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["code"], "not_implemented");
    });
}

#[test]
fn invalid_model_and_invalid_json_are_400s() {
    without_key(async {
        let upstream = MockServer::start().await;
        let gateway = spawn_app(&upstream.uri()).await;
        let client = reqwest::Client::new();

        // An unregistered name is rejected by the roster, which is checked
        // before credentials — so this stays a 400 about the model even with
        // no key in the environment, rather than becoming a 401.
        let response = client
            .post(format!("{gateway}/v1/chat/completions"))
            .json(&json!({"model": "gpt-5.6", "messages": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["code"], "invalid_request");
        assert_eq!(body["error"]["origin"], "gateway");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("no registered model spec")
        );

        let response = client
            .post(format!("{gateway}/v1/chat/completions"))
            .header("content-type", "application/json")
            .body("not json")
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["code"], "invalid_request");
        assert_eq!(body["error"]["origin"], "gateway");
    });
}

/// Exercises the `serve` entry point the binary and bindings share: boots
/// on a free port, answers `/health`, and exits cleanly on shutdown.
#[test]
fn serve_boots_answers_health_and_shuts_down_gracefully() {
    // `serve` builds the client itself, so this reads the environment too —
    // and boots with no providers, which is not an error.
    without_key(async {
        // Reserve a free port, then release it for serve to claim. Racy in
        // principle, harmless in practice for a test.
        let port = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let config = warpllm_server::config::ServerConfig {
            host: "127.0.0.1".into(),
            port,
            timeout_secs: 5,
        };
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(warpllm_server::serve(config, async {
            shutdown_rx.await.ok();
        }));

        let client = reqwest::Client::new();
        let url = format!("http://127.0.0.1:{port}/health");
        let mut health = None;
        for _ in 0..50 {
            match client.get(&url).send().await {
                Ok(response) => {
                    health = Some(response);
                    break;
                }
                Err(_) => tokio::time::sleep(std::time::Duration::from_millis(20)).await,
            }
        }
        assert_eq!(health.expect("server came up").status(), 200);

        shutdown_tx.send(()).unwrap();
        server.await.unwrap().unwrap();
    });
}

#[test]
fn health_reports_ok() {
    without_key(async {
        let upstream = MockServer::start().await;
        let gateway = spawn_app(&upstream.uri()).await;

        let response = reqwest::Client::new()
            .get(format!("{gateway}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["status"], "ok");
        assert_eq!(body["version"], warpllm::version());
    });
}
