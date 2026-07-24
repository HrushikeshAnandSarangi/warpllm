//! The client: one pooled HTTP connection set, one entrypoint.

use std::time::Duration;

use crate::config::{ClientConfig, DEFAULT_TIMEOUT_SECS};
use crate::error::{Error, Result};
use crate::model::{ModelSpec, Protocol, parse_model};
use crate::protocol;
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
    /// is shared — so gateways call it per request to forward each caller's
    /// bearer token upstream.
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
        let spec = parse_model(&requested_model)?;
        let api_key = self.api_key(&spec)?;
        let mut completion = match spec.protocol {
            Protocol::OpenAiCompat => self.openai_compat_chat(&spec, request, &api_key).await?,
        };
        // Echo the caller's provider-prefixed string, not the upstream name.
        completion.model = requested_model;
        Ok(completion)
    }

    /// The OpenAI-compatible pipeline: ingest to the normalized form,
    /// render for the provider, post, and convert the response back out.
    async fn openai_compat_chat(
        &self,
        spec: &ModelSpec<'_>,
        request: CreateChatCompletionRequest,
        api_key: &str,
    ) -> Result<CreateChatCompletionResponse> {
        use protocol::openai_compat::chat::completions as endpoint;

        if !spec.capabilities.supports_endpoint("/chat/completions") {
            return Err(Error::InvalidInput(format!(
                "{}: {} does not serve /chat/completions",
                spec.provider, spec.model
            )));
        }
        let normalized = endpoint::ingest_request(request, spec.model);
        let wire = endpoint::render_request(&normalized, spec.provider)?;
        let wire_response = endpoint::post(
            &self.http,
            spec.provider,
            self.base_url(spec),
            api_key,
            &wire,
        )
        .await?;
        let response = endpoint::ingest_response(wire_response);
        Ok(endpoint::render_response(&response, spec.provider))
    }

    fn api_key(&self, spec: &ModelSpec<'_>) -> Result<String> {
        if let Some(key) = &self.api_key_override {
            return Ok(key.clone());
        }
        std::env::var(spec.env_key)
            .ok()
            .ok_or(Error::MissingApiKey {
                provider: spec.provider,
                env_var: spec.env_key,
            })
    }

    /// A configured `base_url` overrides the provider default (proxies,
    /// tests); otherwise each provider talks to its own API.
    fn base_url(&self, spec: &ModelSpec<'_>) -> &str {
        self.config.base_url.as_deref().unwrap_or(spec.base_url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_for(model: &str) -> ModelSpec<'_> {
        parse_model(model).unwrap()
    }

    #[test]
    fn base_url_defaults_to_each_providers_api() {
        let client = Client::new(ClientConfig::default()).unwrap();
        assert_eq!(
            client.base_url(&spec_for("openai/gpt-4o")),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            client.base_url(&spec_for("deepseek/deepseek-v4-flash")),
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
            client.base_url(&spec_for("openai/gpt-4o")),
            "http://localhost:9999"
        );
    }
}
