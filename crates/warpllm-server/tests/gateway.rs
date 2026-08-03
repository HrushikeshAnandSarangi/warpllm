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

/// Runs `body` with the provider key in the environment.
///
/// Only the tests that actually reach the upstream need this: everything else
/// here fails before key resolution (bad route, `stream: true`, unknown model),
/// so they stay `#[tokio::test]` and keep running in parallel.
///
/// Env mutation is process-global and `unsafe` since edition 2024, hence
/// temp-env's locking helper and a runtime per test rather than
/// `async_with_vars`, which cannot hold the lock across an await.
fn with_gateway_key<F: Future<Output = ()>>(body: F) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    temp_env::with_var("OPENAI_API_KEY", Some(GATEWAY_KEY), || {
        runtime.block_on(body)
    });
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

#[tokio::test]
async fn unprefixed_route_is_404() {
    let upstream = MockServer::start().await;
    let gateway = spawn_app(&upstream.uri()).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway}/chat/completions"))
        .json(&request_body())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 404);
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

#[test]
fn upstream_status_and_error_type_pass_through() {
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
        // `type` keeps its OpenAI meaning for official SDKs; `code` and
        // `origin` are warpllm's flat taxonomy beside it.
        assert_eq!(body["error"]["type"], "rate_limit_exceeded");
        assert_eq!(body["error"]["code"], "rate_limited");
        assert_eq!(body["error"]["origin"], "provider");
        assert_eq!(body["error"]["retry_after_seconds"], 30);
        assert_eq!(body["error"]["request_id"], "req-upstream-1");
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
            body["error"]["code"], "quota_exceeded",
            "reported as a rate limit, a backoff loop never resolves this"
        );
        assert_eq!(body["error"]["origin"], "provider");
        // The code that produced the classification is retained.
        assert_eq!(body["error"]["provider_code"], "insufficient_quota");
    });
}

#[tokio::test]
async fn stream_requests_are_501_before_upstream() {
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
}

#[tokio::test]
async fn invalid_model_and_invalid_json_are_400s() {
    let upstream = MockServer::start().await;
    let gateway = spawn_app(&upstream.uri()).await;
    let client = reqwest::Client::new();

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
}

/// Exercises the `serve` entry point the binary and bindings share: boots
/// on a free port, answers `/health`, and exits cleanly on shutdown.
#[tokio::test]
async fn serve_boots_answers_health_and_shuts_down_gracefully() {
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
}

#[tokio::test]
async fn health_reports_ok() {
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
}
