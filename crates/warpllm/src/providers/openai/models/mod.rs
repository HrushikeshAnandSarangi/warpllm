//! OpenAI's model roster. Entries inherit [`BASE`] with `..BASE` and
//! restate only what differs, so a model is one line plus its deltas.
//! Models split out here as real, documented differences are verified —
//! don't invent per-model data.

use crate::model::{ANY_MODEL, Capabilities, ModelSpec, Protocol};

/// What holds for every OpenAI model, per
/// <https://platform.openai.com/docs/api-reference>. Endpoint entries grow
/// as warpllm implements them. Also the spec any unlisted model resolves
/// to — which is currently every OpenAI model.
const BASE: ModelSpec<'static> = ModelSpec {
    provider: "openai",
    model: ANY_MODEL,
    base_url: "https://api.openai.com/v1",
    env_key: "OPENAI_API_KEY",
    protocol: Protocol::OpenAiCompat,
    capabilities: Capabilities {
        supported_endpoints: &["/chat/completions", "/responses"],
        max_input_tokens: None,
        max_output_tokens: None,
        max_concurrent_requests: None,
    },
};

pub(crate) const MODELS: &[ModelSpec<'static>] = &[BASE];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_base_spec_covers_unlisted_models() {
        let base = MODELS.iter().find(|s| s.model == ANY_MODEL).unwrap();
        assert_eq!(base.provider, "openai");
        assert_eq!(base.base_url, "https://api.openai.com/v1");
        assert!(base.capabilities.supports_endpoint("/chat/completions"));
        assert!(!base.capabilities.supports_endpoint("/batch"));
        // Undocumented, NOT unlimited.
        assert_eq!(base.capabilities.max_concurrent_requests, None);
    }
}
