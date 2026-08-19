//! Load-balanced client: distributes requests across provider/model pairs.
//!
//! [`BalancedClient`] wraps a [`Client`] and adds smooth weighted
//! round-robin selection via [`Balancer`](crate::balancer::Balancer). The
//! candidate set is fixed at construction; each call picks one candidate and
//! delegates to the inner client's normal 4-gate validation.

use crate::balancer::Balancer;
use crate::client::{ChatCompletionStream, Client};
use crate::error::{Error, Result};
use crate::protocol::openai_compat::chat_completions::types::{
    CreateChatCompletionRequest, CreateChatCompletionResponse,
};
use crate::registry::fetch_model;
use std::fmt;

/// A client that load-balances across multiple provider/model pairs.
///
/// Built from a [`Client`] reference and a list of `(model_str, weight)` pairs.
/// Each incoming request is routed to the next candidate chosen by smooth
/// weighted round-robin, then handed to the inner client's normal validation
/// and execution path.
///
/// The [`Balancer`] is stateful (per-candidate `current_weight`), so
/// `BalancedClient` is not `Sync` in the general sense — but the balancer's
/// atomics make it safe to share across threads anyway.
///
/// # Example
///
/// ```no_run
/// use warpllm::{BalancedClient, Client, ClientConfig};
///
/// let client = Client::new(ClientConfig::default()).unwrap();
/// let balanced = BalancedClient::new(&client, &[
///     ("openai/gpt-5.6", 3),
///     ("deepseek/deepseek-v4-pro", 1),
/// ]).unwrap();
/// // Use balanced.chat_completions(request) instead of client.chat_completions(request)
/// ```
pub struct BalancedClient<'a> {
    client: &'a Client,
    balancer: Balancer,
}

impl fmt::Debug for BalancedClient<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BalancedClient")
            .field("balancer", &self.balancer)
            .finish()
    }
}

impl<'a> BalancedClient<'a> {
    /// Creates a new balanced client.
    ///
    /// # Arguments
    ///
    /// * `client` — The underlying client used for every request.
    /// * `candidates` — Non-empty list of `(model_str, weight)` pairs. Each
    ///   `model_str` must exist in the registry. Weight determines the relative
    ///   proportion of requests routed to that candidate.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidInput`] if `candidates` is empty.
    /// - [`Error::InvalidModel`] if any `model_str` is not in the roster.
    pub fn new(client: &'a Client, candidates: &[(&str, u32)]) -> Result<Self> {
        if candidates.is_empty() {
            return Err(Error::InvalidInput(
                "balanced client requires at least one candidate".into(),
            ));
        }
        let mut resolved = Vec::with_capacity(candidates.len());
        for &(model_str, weight) in candidates {
            let (provider, model) = fetch_model(model_str)?;
            resolved.push(crate::balancer::Candidate {
                model_str: model_str.to_string(),
                provider,
                model,
                weight,
            });
        }
        Ok(Self {
            client,
            balancer: Balancer::new(resolved),
        })
    }

    /// Performs a non-streaming chat completion via the next balanced candidate.
    ///
    /// The request's `model` field is overwritten with the selected candidate's
    /// `model_str` before the inner client processes it — the caller's model
    /// name is the *group* name, and each candidate is a concrete provider/model
    /// within that group.
    pub async fn chat_completions(
        &self,
        mut request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse> {
        let candidate = self.balancer.select();
        request.model.clone_from(&candidate.model_str);
        self.client.chat_completions(request).await
    }

    /// Performs a streaming chat completion via the next balanced candidate.
    ///
    /// Same model-rewriting as [`Self::chat_completions`].
    pub async fn chat_completions_stream(
        &self,
        mut request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionStream> {
        let candidate = self.balancer.select();
        request.model.clone_from(&candidate.model_str);
        self.client.chat_completions_stream(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::balancer::Candidate;
    use crate::{Api, Client, ClientConfig, ModelSpec, ProviderSpec, SupportedApi};

    fn leaked_spec_pair(name: &str) -> (&'static ProviderSpec, &'static ModelSpec) {
        let provider = Box::leak(Box::new(ProviderSpec {
            name: name.to_string(),
            base_url: format!("https://api.{name}.test"),
            env_api_key: None,
        }));
        let model = Box::leak(Box::new(ModelSpec {
            provider: name.to_string(),
            model: name.to_string(),
            supported_apis: vec![SupportedApi {
                api: Api::OpenAiCompatChatCompletions,
            }],
            capabilities: crate::Capabilities::blank(),
            deprecation_date: None,
        }));
        (provider, model)
    }

    #[test]
    fn empty_candidates_rejected() {
        let client = Client::new(ClientConfig::default()).unwrap();
        let err = BalancedClient::new(&client, &[]).unwrap_err();
        assert!(err.to_string().contains("at least one candidate"));
    }

    #[test]
    fn unknown_model_rejected() {
        let client = Client::new(ClientConfig::default()).unwrap();
        let err = BalancedClient::new(&client, &[("nope/nope", 1)]).unwrap_err();
        assert!(err.to_string().contains("no registered model"));
    }

    #[test]
    fn balancer_distribution_from_public_interface() {
        // Directly test the balancer that BalancedClient wraps.
        let (p_a, m_a) = leaked_spec_pair("a");
        let (p_b, m_b) = leaked_spec_pair("b");
        let balancer = Balancer::new(vec![
            Candidate {
                model_str: "a/test".into(),
                provider: p_a,
                model: m_a,
                weight: 3,
            },
            Candidate {
                model_str: "b/test".into(),
                provider: p_b,
                model: m_b,
                weight: 1,
            },
        ]);
        let mut counts = [0u32; 2];
        for _ in 0..1000 {
            let c = balancer.select();
            if c.provider.name() == "a" {
                counts[0] += 1;
            } else {
                counts[1] += 1;
            }
        }
        assert_eq!(counts[0], 750);
        assert_eq!(counts[1], 250);
    }
}
