//! The client: one pooled HTTP connection set, one entrypoint.

use std::time::Duration;

use crate::config::{ClientConfig, DEFAULT_TIMEOUT_SECS};
use crate::error::{Error, Result};
use crate::protocol;
use crate::protocol::{Api, Protocol};
use crate::registry::{ModelSpec, ProviderSpec, fetch_model};
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
        let (provider, model) = fetch_model(&requested_model)?;
        let api_key = self.api_key(provider)?;
        let mut completion = match provider.protocol() {
            Protocol::OpenAiCompat => {
                self.openai_compat_chat(provider, model, &requested_model, request, &api_key)
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
    /// `&'static` specs — which is all [`fetch_model`] hands out — because the
    /// errors below and the protocol layer both name a provider with a
    /// `&'static str`, and a spec's accessors borrow from the spec.
    async fn openai_compat_chat(
        &self,
        provider: &'static ProviderSpec,
        model: &'static ModelSpec,
        requested_model: &str,
        request: CreateChatCompletionRequest,
        api_key: &str,
    ) -> Result<CreateChatCompletionResponse> {
        use protocol::openai_compat::chat::completions as endpoint;

        if !provider.supports_api(Api::ChatCompletions) {
            return Err(Error::InvalidInput(format!(
                "{}: {} does not serve chat completions",
                provider.name(),
                requested_model
            )));
        }
        // The ENTRY's name, not the caller's string: they differ whenever
        // warpllm's routing alias differs from the provider's own name.
        let normalized = endpoint::ingest_request(request, model.model());
        let wire = endpoint::render_request(&normalized, provider.name())?;
        let wire_response = endpoint::post(
            &self.http,
            provider.name(),
            self.base_url(provider),
            api_key,
            &wire,
        )
        .await?;
        let response = endpoint::ingest_response(wire_response);
        Ok(endpoint::render_response(&response, provider.name()))
    }

    /// An explicit key first, then the provider's own environment variable if
    /// it names one.
    ///
    /// A provider with no `env_api_key` has no second source: it authenticates
    /// only through [`Client::with_api_key`], and without one the error says
    /// so rather than naming a variable that does not exist.
    fn api_key(&self, provider: &'static ProviderSpec) -> Result<String> {
        if let Some(key) = &self.api_key_override {
            return Ok(key.clone());
        }
        let env_var = provider.env_api_key();
        env_var
            .and_then(|var| std::env::var(var).ok())
            .ok_or(Error::MissingApiKey {
                provider: provider.name(),
                env_var,
            })
    }

    /// A configured `base_url` overrides the provider default (proxies,
    /// tests); otherwise each provider talks to its own API.
    fn base_url(&self, provider: &'static ProviderSpec) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or(provider.base_url())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::Capabilities;
    use crate::types::openai_compat::chat::completions::ChatCompletionRequestMessage;

    /// The two halves the client works from, for a model the shipped roster
    /// does have.
    fn pair_for(model: &str) -> (&'static ProviderSpec, &'static ModelSpec) {
        fetch_model(model).unwrap()
    }

    /// Leaked because the client takes `&'static` specs, which costs nothing
    /// in a test process that is about to exit.
    fn demo_provider(base_url: &str, env_api_key: Option<&str>) -> &'static ProviderSpec {
        Box::leak(Box::new(ProviderSpec {
            name: "demo".into(),
            base_url: base_url.into(),
            env_api_key: env_api_key.map(str::to_string),
            protocol: Protocol::OpenAiCompat,
            supported_apis: vec![Api::ChatCompletions],
        }))
    }

    /// A provider that names no environment variable cannot fall back to one,
    /// so the error must say that rather than send someone off to set a
    /// variable nothing reads.
    #[test]
    fn a_provider_with_no_env_api_key_asks_for_an_explicit_one() {
        let err = Client::new(ClientConfig::default())
            .unwrap()
            .api_key(demo_provider("https://api.demo.test", None))
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

    /// And an explicitly supplied key is all such a provider ever needed.
    #[test]
    fn an_explicit_key_serves_a_provider_with_no_env_api_key() {
        let client = Client::new(ClientConfig::default())
            .unwrap()
            .with_api_key("sk-explicit");
        assert_eq!(
            client
                .api_key(demo_provider("https://api.demo.test", None))
                .unwrap(),
            "sk-explicit"
        );
    }

    /// What ships upstream is the ENTRY's name, never the string the caller
    /// routed with. That is the whole point of `model:` — a warpllm alias may
    /// differ from the provider's own name for the same model.
    #[tokio::test]
    async fn an_alias_sends_the_entrys_own_name_upstream() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": "demo-chat-20240101",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop"
                }]
            })))
            .mount(&server)
            .await;

        let aliased = Box::leak(Box::new(ModelSpec {
            provider: "demo".into(),
            model: "demo-chat-20240101".into(),
            capabilities: Capabilities::blank(),
        }));
        let client = Client::new(ClientConfig::default()).unwrap();
        let request = CreateChatCompletionRequest {
            model: "demo/chat".into(),
            messages: vec![ChatCompletionRequestMessage {
                role: "user".into(),
                content: "hi".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        client
            .openai_compat_chat(
                demo_provider(&server.uri(), Some("DEMO_API_KEY")),
                aliased,
                "demo/chat",
                request,
                "k",
            )
            .await
            .unwrap();

        let sent = &server.received_requests().await.unwrap()[0];
        let body: serde_json::Value = serde_json::from_slice(&sent.body).unwrap();
        assert_eq!(
            body["model"], "demo-chat-20240101",
            "the caller's routing string shipped instead of the entry's name"
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
        let (provider, model) = pair_for("openai/gpt-5.6");
        client
            .openai_compat_chat(provider, model, "openai/gpt-5.6", request, "k")
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
            client.base_url(pair_for("openai/gpt-5.6").0),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            client.base_url(pair_for("deepseek/deepseek-v4-flash").0),
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
            client.base_url(pair_for("openai/gpt-5.6").0),
            "http://localhost:9999"
        );
    }
}
