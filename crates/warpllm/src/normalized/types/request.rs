//! The normalized chat request.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{CacheHint, IngestSource, Message, ProviderExt};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct ChatRequest {
    /// Upstream model name, provider prefix already stripped.
    pub model: String,

    /// Full conversation, system messages included (hoisted per target).
    pub messages: Vec<Message>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    #[serde(default)]
    pub params: GenerationParams,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    /// Request-level cache hint the renderer applies to the LAST cacheable
    /// block, so callers get caching without hand-placing breakpoints.
    /// Block-level hints take precedence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheHint>,

    #[serde(default)]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_include_usage: Option<bool>,

    #[serde(default, skip_serializing_if = "ProviderExt::is_empty")]
    pub ext: ProviderExt,

    /// Where this request came from; see [`IngestSource`].
    #[serde(skip)]
    #[allow(dead_code)] // staged: read once the passthrough fast path lands
    pub source: Option<IngestSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ToolDef {
    /// Verbatim; length/charset limits differ per provider.
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema, verbatim. Providers that reject some keywords prune in
    /// their renderer, not here.
    pub input_schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheHint>,
    /// Provider built-in tools (web search, computer use) ride here rather
    /// than masquerading as function tools.
    #[serde(default, skip_serializing_if = "ProviderExt::is_empty")]
    pub ext: ProviderExt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub(crate) enum ToolChoice {
    Auto,
    None,
    Required,
    Tool { name: String },
}

/// Store what the caller sent, unconverted. Only the params with a typed
/// home on every supported wire live here; everything else (seed, top_k,
/// penalties, ...) rides the namespaced ext bags untouched until a second
/// dialect needs to render them differently.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct GenerationParams {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
}

/// Unified over provider thinking budgets / reasoning-effort dialects,
/// carrying BOTH so cross-dialect conversion is an explicit render action.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct ReasoningConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// "minimal" | "low" | "medium" | "high" | "xhigh".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
    /// Reason but don't return the reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::{ContentBlock, Dialect, Role};
    use super::*;

    #[test]
    fn chat_request_round_trips_and_skips_source() {
        let mut request = ChatRequest {
            model: "gpt-4o".into(),
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "hi".into(),
                    cache: None,
                }],
                ext: ProviderExt::new(),
            }],
            params: GenerationParams {
                temperature: Some(0.7),
                stop: vec!["END".into()],
                ..Default::default()
            },
            stream: false,
            source: Some(IngestSource {
                dialect: Dialect::OpenAiCompat,
                body: json!({"model": "openai/gpt-4o"}),
            }),
            ..Default::default()
        };
        request
            .ext
            .insert("openai_compat".into(), json!({"top_k": 40}));

        let value = serde_json::to_value(&request).unwrap();
        assert!(value.get("source").is_none(), "source must not serialize");
        assert_eq!(value["params"]["temperature"], 0.7);

        let back: ChatRequest = serde_json::from_value(value.clone()).unwrap();
        assert!(back.source.is_none());
        assert_eq!(serde_json::to_value(&back).unwrap(), value);
    }
}
