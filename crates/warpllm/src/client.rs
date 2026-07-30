//! The client: one pooled HTTP connection set, one entrypoint.

use std::time::Duration;

use crate::config::{ClientConfig, DEFAULT_TIMEOUT_SECS};
use crate::error::{Error, Result};
use crate::protocol;
use crate::protocol::Protocol;
use crate::registry::{ModelSpec, model_spec};
use crate::types::openai_compat::chat::completions::{
    CreateChatCompletionRequest, CreateChatCompletionResponse,
};

pub struct Client {
    http: reqwest::Client,
    config: ClientConfig,
    /// `with_api_key` override; wins over every provider's env var.
    api_key_override: Option<String>,
}

impl Client {
    /// API keys resolve at request time from the routed provider's env var
    /// (e.g. `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`), so constructing a
    /// client never requires credentials.
    pub fn new(config: ClientConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(
                config.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS),
            ))
            .build()
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        Ok(Self {
            http,
            config,
            api_key_override: None,
        })
    }

    /// A copy of this client that authenticates with `api_key` instead of
    /// the environment's — for every provider. Cheap — the connection pool
    /// is shared — so an embedder holding keys somewhere other than the
    /// environment can call it per request.
    ///
    /// warpllm's own gateway does not: it holds the provider keys itself and
    /// ignores the caller's `Authorization` header entirely.
    #[must_use]
    pub fn with_api_key(&self, api_key: impl Into<String>) -> Self {
        Self {
            http: self.http.clone(),
            config: self.config.clone(),
            api_key_override: Some(api_key.into()),
        }
    }

    pub async fn chat_completion(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse> {
        if request.stream == Some(true) {
            return Err(Error::NotImplemented("streaming"));
        }
        let requested_model = request.model.clone();
        let spec = model_spec(&requested_model)?;
        let api_key = self.api_key(spec)?;
        let mut completion = match spec.protocol() {
            Protocol::OpenAiCompat => {
                self.openai_compat_chat(spec, &requested_model, request, &api_key)
                    .await?
            }
        };
        // Echo the caller's provider-prefixed string, not the upstream name.
        completion.model = requested_model;
        Ok(completion)
    }

    /// The OpenAI-compatible pipeline: ingest to the normalized form,
    /// render for the provider, post, and convert the response back out.
    ///
    /// `&'static ModelSpec` — which is all [`model_spec`] hands out — because
    /// the errors below and the protocol layer both name a provider with a
    /// `&'static str`, and a spec's accessors borrow from the spec.
    async fn openai_compat_chat(
        &self,
        spec: &'static ModelSpec,
        requested_model: &str,
        request: CreateChatCompletionRequest,
        api_key: &str,
    ) -> Result<CreateChatCompletionResponse> {
        use protocol::openai_compat::chat::completions as endpoint;

        if !spec.capabilities().supports_endpoint("/chat/completions") {
            return Err(Error::InvalidInput(format!(
                "{}: {} does not serve /chat/completions",
                spec.provider(),
                requested_model
            )));
        }
        // `wire_model`, not `model`: under a wildcard entry the upstream name
        // is whatever the caller asked for, which the spec cannot know.
        let normalized = endpoint::ingest_request(request, spec.wire_model(requested_model));
        let wire = endpoint::render_request(&normalized, spec.provider())?;
        let wire_response = endpoint::post(
            &self.http,
            spec.provider(),
            self.base_url(spec),
            api_key,
            &wire,
        )
        .await?;
        let response = endpoint::ingest_response(wire_response);
        Ok(endpoint::render_response(&response, spec.provider()))
    }

    /// An explicit key first, then the model's own environment variable if it
    /// names one.
    ///
    /// A spec with no `env_api_key` has no second source: it authenticates
    /// only through [`Client::with_api_key`], and without one the error says
    /// so rather than naming a variable that does not exist.
    fn api_key(&self, spec: &'static ModelSpec) -> Result<String> {
        if let Some(key) = &self.api_key_override {
            return Ok(key.clone());
        }
        let env_var = spec.env_api_key();
        env_var
            .and_then(|var| std::env::var(var).ok())
            .ok_or(Error::MissingApiKey {
                provider: spec.provider(),
                env_var,
            })
    }

    /// A configured `base_url` overrides the provider default (proxies,
    /// tests); otherwise each provider talks to its own API.
    fn base_url(&self, spec: &'static ModelSpec) -> &str {
        self.config.base_url.as_deref().unwrap_or(spec.base_url())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::Capabilities;
    use crate::types::openai_compat::chat::completions::ChatCompletionRequestMessage;

    fn spec_for(model: &str) -> &'static ModelSpec {
        model_spec(model).unwrap()
    }

    /// A wildcard spec, which the SHIPPED roster deliberately has none of —
    /// so the only way to exercise the wildcard path through the client is to
    /// build one. Leaked because `openai_compat_chat` takes `&'static`, which
    /// costs nothing in a test process that is about to exit.
    fn wildcard_spec(base_url: &str) -> &'static ModelSpec {
        Box::leak(Box::new(ModelSpec {
            provider: Some("demo".into()),
            wildcard: true,
            // What a `*` key resolves to: no upstream name of its own.
            model: Some("*".into()),
            base_url: Some(base_url.into()),
            env_api_key: Some("DEMO_API_KEY".into()),
            protocol: Some(Protocol::OpenAiCompat),
            capabilities: Capabilities {
                supported_endpoints: Some(vec!["/chat/completions".to_string()]),
                ..Capabilities::blank()
            },
        }))
    }

    /// A model that names no environment variable cannot fall back to one, so
    /// the error must say that rather than send someone off to set a variable
    /// nothing reads.
    #[test]
    fn a_model_with_no_env_api_key_asks_for_an_explicit_one() {
        let spec = Box::leak(Box::new(ModelSpec {
            env_api_key: None,
            ..(*wildcard_spec("https://api.demo.test")).clone()
        }));
        let err = Client::new(ClientConfig::default())
            .unwrap()
            .api_key(spec)
            .unwrap_err();
        match &err {
            Error::MissingApiKey { provider, env_var } => {
                assert_eq!(*provider, "demo");
                assert_eq!(*env_var, None, "named a variable that does not exist");
            }
            other => panic!("expected MissingApiKey, got {other:?}"),
        }
        let message = err.to_string();
        assert!(
            message.contains("names no default environment variable"),
            "{message}"
        );
        assert!(!message.contains("set the"), "{message}");

        // The wire form keeps one code, with a null where the variable would be.
        let wire: serde_json::Value = serde_json::from_str(&err.to_wire_json()).unwrap();
        assert_eq!(wire["code"], "missing_api_key");
        assert!(wire["env_var"].is_null());
    }

    /// And an explicitly supplied key is all such a model ever needed.
    #[test]
    fn an_explicit_key_serves_a_model_with_no_env_api_key() {
        let spec = Box::leak(Box::new(ModelSpec {
            env_api_key: None,
            ..(*wildcard_spec("https://api.demo.test")).clone()
        }));
        let client = Client::new(ClientConfig::default())
            .unwrap()
            .with_api_key("sk-explicit");
        assert_eq!(client.api_key(spec).unwrap(), "sk-explicit");
    }

    /// The one line that makes wildcards work at all: what ships upstream is
    /// the name the caller asked for, never the entry's `*`. A provider that
    /// received `"model": "*"` would reject every wildcard-routed request.
    #[tokio::test]
    async fn a_wildcard_sends_the_requested_name_upstream() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": "gpt-9",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop"
                }]
            })))
            .mount(&server)
            .await;

        let client = Client::new(ClientConfig::default()).unwrap();
        let request = CreateChatCompletionRequest {
            model: "demo/gpt-9".into(),
            messages: vec![ChatCompletionRequestMessage {
                role: "user".into(),
                content: "hi".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        client
            .openai_compat_chat(wildcard_spec(&server.uri()), "demo/gpt-9", request, "k")
            .await
            .unwrap();

        let sent = &server.received_requests().await.unwrap()[0];
        let body: serde_json::Value = serde_json::from_slice(&sent.body).unwrap();
        assert_eq!(
            body["model"], "gpt-9",
            "a wildcard shipped its own `*` instead of the requested name"
        );
    }

    /// The same path for a concrete entry still ships the ENTRY's name, which
    /// is what lets a routing alias differ from the provider's own name.
    #[tokio::test]
    async fn a_concrete_entry_sends_its_own_name_upstream() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": "gpt-5.6",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop"
                }]
            })))
            .mount(&server)
            .await;

        let client = Client::new(ClientConfig {
            base_url: Some(server.uri()),
            ..Default::default()
        })
        .unwrap();
        let request = CreateChatCompletionRequest {
            model: "openai/gpt-5.6".into(),
            messages: vec![ChatCompletionRequestMessage {
                role: "user".into(),
                content: "hi".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        client
            .openai_compat_chat(spec_for("openai/gpt-5.6"), "openai/gpt-5.6", request, "k")
            .await
            .unwrap();

        let sent = &server.received_requests().await.unwrap()[0];
        let body: serde_json::Value = serde_json::from_slice(&sent.body).unwrap();
        assert_eq!(body["model"], "gpt-5.6");
    }

    #[test]
    fn base_url_defaults_to_each_providers_api() {
        let client = Client::new(ClientConfig::default()).unwrap();
        assert_eq!(
            client.base_url(spec_for("openai/gpt-5.6")),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            client.base_url(spec_for("deepseek/deepseek-v4-flash")),
            "https://api.deepseek.com"
        );
    }

    #[test]
    fn configured_base_url_wins_over_the_default() {
        let client = Client::new(ClientConfig {
            base_url: Some("http://localhost:9999".into()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            client.base_url(spec_for("openai/gpt-5.6")),
            "http://localhost:9999"
        );
    }
}
