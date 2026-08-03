//! Messages and content blocks: one uniform parts vector for every role.
//! genai's history is the warning here — it started with segregated
//! variants (Text | Parts | ToolCalls | ToolResponses) and had to collapse
//! them into one uniform vector. Start uniform.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{ProviderExt, RawJson};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Role {
    /// Covers OpenAI `system` AND `developer` (original spelling in ext).
    System,
    User,
    Assistant,
    /// ToolResult blocks only. OpenAI: one wire message per result;
    /// Anthropic: consecutive tool messages merge into one user turn.
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
    #[serde(default, skip_serializing_if = "ProviderExt::is_empty")]
    pub ext: ProviderExt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub(crate) enum ContentBlock {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache: Option<CacheHint>,
    },
    Image {
        source: MediaSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache: Option<CacheHint>,
    },
    Audio {
        source: MediaSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        format: Option<String>,
    },
    Document {
        source: MediaSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache: Option<CacheHint>,
    },

    ToolCall {
        /// Verbatim. Providers constrain id formats differently; renderers
        /// sanitize REVERSIBLY and report the mapping in
        /// `RenderReport::call_id_map` — never mutate it here.
        id: String,
        name: String,
        arguments: RawJson,
    },

    ToolResult {
        call_id: String,
        /// Blocks, not String: some providers permit images in results.
        content: Vec<ContentBlock>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },

    /// INVARIANT (enforced by upstream providers): a run of consecutive
    /// Reasoning blocks must be sent back exactly as received — same
    /// order, same content, signatures untouched. Treat the run as an
    /// opaque unit; typed provenance-tagged blocks exist so renderers can
    /// reconstitute the native shape (or know they can't).
    Reasoning {
        detail: ReasoningDetail,
        /// Native protocol this block originated from, e.g.
        /// "anthropic-claude-v1", "openai-responses-v1".
        #[serde(skip_serializing_if = "Option::is_none")]
        provenance: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },

    /// Forward-compat: pass through when source == target, else policy.
    #[serde(untagged)]
    Unknown(Value),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub(crate) enum ReasoningDetail {
    /// Visible reasoning text; `signature` is cryptographic and opaque —
    /// echo byte-exact or omit the block.
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Summarized reasoning (OpenAI o-series summaries).
    Summary { summary: String },
    /// Encrypted / redacted payload. Never inspect; replay verbatim.
    Encrypted { data: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum MediaSource {
    Url { url: String },
    Base64 { media_type: String, data: String },
    ProviderFile { id: String },
}

/// Prompt-cache breakpoint hint. Positional — it lives on blocks because
/// some providers' cache_control is positional.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CacheHint {
    /// "5m", "1h". None = provider default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn round_trip(block: &ContentBlock) -> Value {
        let value = serde_json::to_value(block).unwrap();
        let back: ContentBlock = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(&back).unwrap(), value);
        value
    }

    #[test]
    fn every_block_variant_round_trips() {
        let blocks = [
            ContentBlock::Text {
                text: "hi".into(),
                cache: Some(CacheHint {
                    ttl: Some("5m".into()),
                }),
            },
            ContentBlock::Image {
                source: MediaSource::Url {
                    url: "https://example.com/a.png".into(),
                },
                detail: Some("high".into()),
                cache: None,
            },
            ContentBlock::Audio {
                source: MediaSource::Base64 {
                    media_type: "audio/wav".into(),
                    data: "aGk=".into(),
                },
                format: Some("wav".into()),
            },
            ContentBlock::Document {
                source: MediaSource::ProviderFile {
                    id: "file-1".into(),
                },
                title: Some("doc".into()),
                cache: None,
            },
            ContentBlock::ToolCall {
                id: "call-1".into(),
                name: "search".into(),
                arguments: RawJson::new(r#"{"q":1}"#),
            },
            ContentBlock::ToolResult {
                call_id: "call-1".into(),
                content: vec![ContentBlock::Text {
                    text: "found".into(),
                    cache: None,
                }],
                is_error: false,
            },
            ContentBlock::Reasoning {
                detail: ReasoningDetail::Text {
                    text: "thinking".into(),
                    signature: Some("sig".into()),
                },
                provenance: Some("anthropic-claude-v1".into()),
                id: None,
            },
        ];
        for block in &blocks {
            round_trip(block);
        }
    }

    #[test]
    fn unrecognized_block_json_becomes_unknown_verbatim() {
        let value = json!({"type": "hologram", "payload": {"z": 1, "a": 2}});
        let block: ContentBlock = serde_json::from_value(value.clone()).unwrap();
        assert!(matches!(&block, ContentBlock::Unknown(v) if *v == value));
        assert_eq!(serde_json::to_value(&block).unwrap(), value);
    }

    #[test]
    fn tool_call_arguments_stay_byte_exact() {
        let value = round_trip(&ContentBlock::ToolCall {
            id: "call-1".into(),
            name: "search".into(),
            arguments: RawJson::new(r#"{"z": 1,"a":2 }"#),
        });
        assert_eq!(value["arguments"], r#"{"z": 1,"a":2 }"#);
    }
}
