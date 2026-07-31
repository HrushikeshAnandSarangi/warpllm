//! The client: one pooled HTTP connection set, one entrypoint.

use std::time::Duration;

use crate::config::{ClientConfig, DEFAULT_TIMEOUT_SECS};
use crate::error::{Error, Result};
use crate::normalized::openai_compat;
use crate::protocol::openai_compat::chat_completions::types::{
    CreateChatCompletionRequest, CreateChatCompletionResponse,
};
use crate::protocol::{Api, Protocol};
use crate::registry::{ProviderSpec, fetch_model};

pub struct Client {
    http: reqwest::Client,
    config: ClientConfig,
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
        Ok(Self { http, config })
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
        if !provider.supports_api(Api::ChatCompletions) {
            return Err(Error::InvalidInput(format!(
                "{}: {} does not serve chat_completions",
                provider.name(),
                requested_model
            )));
        }
        let api_key = self.api_key(provider)?;

        // Ingest answers to the dialect warpllm was CALLED with, which is
        // openai_compat and only ever will be for this entrypoint. The ENTRY's
        // model name goes in, not the caller's string: they differ whenever
        // warpllm's routing alias differs from the provider's own name.
        let normalized = openai_compat::chat_completions::ingest_request(request, model.model());
        // One arm per protocol, each `&ChatRequest -> ChatResponse`. Adding a
        // protocol is a line here plus its own `exchange`.
        let response = match provider.protocol() {
            Protocol::OpenAiCompat => {
                openai_compat::chat_completions::exchange(
                    &normalized,
                    &self.http,
                    provider.name(),
                    self.base_url(provider),
                    &api_key,
                )
                .await?
            }
        };
        let mut completion =
            openai_compat::chat_completions::render_response(&response, provider.name());
        // Echo the caller's provider-prefixed string, not the upstream echo.
        completion.model = requested_model;
        Ok(completion)
    }

    /// The routed provider's own environment variable, read at request time.
    /// The environment is the only source: a provider with no `env_api_key`
    /// therefore has none at all, and the error names the roster rather than a
    /// variable that does not exist.
    fn api_key(&self, provider: &'static ProviderSpec) -> Result<String> {
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
    use crate::protocol::openai_compat::chat_completions::types::ChatCompletionRequestMessage;
    use crate::registry::{Capabilities, ModelSpec};

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

    /// A provider that names no environment variable has no key source at all,
    /// so the error must say that rather than send someone off to set a
    /// variable nothing reads.
    #[test]
    fn a_provider_with_no_env_api_key_names_the_roster() {
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
            message.contains("names no environment variable"),
            "{message}"
        );
        assert!(!message.contains("set the"), "{message}");

        // The wire form keeps one code, with a null where the variable would be.
        let wire: serde_json::Value = serde_json::from_str(&err.to_wire_json()).unwrap();
        assert_eq!(wire["code"], "missing_api_key");
        assert!(wire["env_var"].is_null());
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
        // What `chat_completion` does, minus the routing it already proved:
        // the SPEC's model name is what ingest is handed.
        let normalized = openai_compat::chat_completions::ingest_request(request, aliased.model());
        openai_compat::chat_completions::exchange(
            &normalized,
            &client.http,
            "demo",
            &server.uri(),
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
        let normalized = openai_compat::chat_completions::ingest_request(request, model.model());
        openai_compat::chat_completions::exchange(
            &normalized,
            &client.http,
            provider.name(),
            client.base_url(provider),
            "k",
        )
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
