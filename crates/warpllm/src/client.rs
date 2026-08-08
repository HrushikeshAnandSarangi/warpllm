//! The client: one pooled HTTP connection set, one entrypoint.

use std::time::Duration;

use crate::config::{ClientConfig, DEFAULT_TIMEOUT_SECS};
use crate::credentials::Credentials;
use crate::error::{Error, Result};
use crate::gateway::openai_compat;
use crate::protocol::openai_compat::chat_completions::types::{
    CreateChatCompletionRequest, CreateChatCompletionResponse,
};
use crate::registry::{ProviderSpec, fetch_model};
use crate::types::{Api, Protocol};

pub struct Client {
    http: reqwest::Client,
    config: ClientConfig,
    credentials: Credentials,
}

impl Client {
    /// Reads the environment once and keeps the providers it can authenticate.
    ///
    /// Constructing a client still never *requires* credentials: an environment
    /// holding none builds a client that reaches no provider, and each request
    /// says which variable it wanted. What construction does is answer that
    /// question up front, so it can be logged at the moment a caller is set up
    /// to read it rather than discovered one failed request at a time.
    pub fn new(config: ClientConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(
                config.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS),
            ))
            .build()
            .map_err(|e| Error::Internal(e.to_string()))?;
        Ok(Self {
            http,
            config,
            credentials: Credentials::from_env(),
        })
    }

    /// Serves one OpenAI-compatible chat completion.
    ///
    /// A request is admitted only when BOTH halves hold: the roster registers
    /// the model, and this client holds a key for the provider serving it. They
    /// are checked in that order deliberately — a name nothing registers is a
    /// typo, and hearing "set `DEEPSEEK_API_KEY`" for `deepseek/typo` would send
    /// someone after a credential they already have.
    pub async fn chat_completions(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse> {
        if request.stream == Some(true) {
            return Err(Error::NotImplemented("streaming"));
        }
        let requested_model = request.model.clone();
        // Half one: the roster. Closed, so an unregistered name stops here.
        let (provider, model) = fetch_model(&requested_model)?;
        if !provider.supports_api(Api::ChatCompletions) {
            return Err(Error::InvalidInput(format!(
                "{}: {} does not serve chat_completions",
                provider.name(),
                requested_model
            )));
        }
        // Half two: this client. A registered model whose provider was not
        // authenticated at construction never reaches the network.
        let api_key = self.api_key(provider)?;

        // Ingest answers to the protocol warpllm was CALLED with, which is
        // openai_compat and only ever will be for this entrypoint. The ENTRY's
        // model name goes in, not the caller's string: they differ whenever
        // warpllm's routing alias differs from the provider's own name.
        let normalized =
            openai_compat::api::chat_completions::ingest_request(request, model.model());
        // One arm per protocol, each `&ChatRequest -> ChatResponse`. Adding a
        // protocol is a line here plus its own `exchange`.
        let response = match provider.protocol() {
            Protocol::OpenAiCompat => {
                openai_compat::api::chat_completions::exchange(
                    &normalized,
                    &self.http,
                    provider.name(),
                    self.base_url(provider),
                    api_key,
                )
                .await?
            }
        };
        let mut completion =
            openai_compat::api::chat_completions::render_response(&response, provider.name());
        // Echo the caller's provider-prefixed string, not the upstream echo.
        completion.model = requested_model;
        Ok(completion)
    }

    /// The routed provider's key, from the snapshot this client took of the
    /// environment when it was built.
    ///
    /// A miss is not "the variable is unset now" but "it was unset then", and
    /// the error still names the variable to set, because that is the remedy
    /// either way. A provider with no `env_api_key` has no key source at all,
    /// so the error names the roster rather than a variable nothing reads.
    fn api_key(&self, provider: &'static ProviderSpec) -> Result<&str> {
        self.credentials
            .get(provider.name())
            .ok_or(Error::MissingApiKey {
                provider: provider.name(),
                env_var: provider.env_api_key(),
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
    use crate::credentials::with_env;
    use crate::protocol::openai_compat::chat_completions::types::ChatCompletionRequestMessage;
    use crate::registry::{Capabilities, ModelSpec};

    /// A client built under the environment lock.
    ///
    /// `Client::new` reads the environment, so every construction in this
    /// binary has to hold temp-env's lock even when the test has no opinion
    /// about what is set — see [`with_env`] for why.
    fn client(config: ClientConfig) -> Client {
        with_env(&[], || Client::new(config).unwrap())
    }

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
        let err = client(ClientConfig::default())
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

        // The wire form keeps one code either way. The remedy rides in the
        // message rather than a warpllm-specific `env_var` field, which the
        // OpenAI envelope has no place for.
        let wire: serde_json::Value = serde_json::from_str(&err.to_openai_json()).unwrap();
        assert_eq!(wire["status"], 401);
        // OpenAI's spelling for an unusable key, not warpllm's.
        assert_eq!(wire["error"]["code"], "invalid_api_key");
        assert!(
            wire["error"]["message"]
                .as_str()
                .unwrap()
                .contains("names no environment variable")
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
        let client = client(ClientConfig::default());
        let request = CreateChatCompletionRequest {
            model: "demo/chat".into(),
            messages: vec![ChatCompletionRequestMessage {
                role: "user".into(),
                content: "hi".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        // What `chat_completions` does, minus the routing it already proved:
        // the SPEC's model name is what ingest is handed.
        let normalized =
            openai_compat::api::chat_completions::ingest_request(request, aliased.model());
        openai_compat::api::chat_completions::exchange(
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

        let client = client(ClientConfig {
            base_url: Some(server.uri()),
            ..Default::default()
        });
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
        let normalized =
            openai_compat::api::chat_completions::ingest_request(request, model.model());
        openai_compat::api::chat_completions::exchange(
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
        let client = client(ClientConfig::default());
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
        let client = client(ClientConfig {
            base_url: Some("http://localhost:9999".into()),
            ..Default::default()
        });
        assert_eq!(
            client.base_url(pair_for("openai/gpt-5.6").0),
            "http://localhost:9999"
        );
    }
}
