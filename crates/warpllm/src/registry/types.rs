//! What the registry holds, and how a caller reads it.
//!
//! Two levels, two types, because they answer two questions. A
//! [`ProviderSpec`] is how an API is reached and what it serves — one host,
//! one credential, one wire format, one list of surfaces. A [`ModelSpec`] is
//! one routable name under that provider, carrying only what differs between
//! models of the same one. [`crate::fetch_model`] hands back one of each and
//! merges nothing.
//!
//! These are READ SURFACES, not the YAML schema: `load` next door owns the
//! schema and does the settling, which is why nothing here is an `Option`
//! meaning "not answered yet". A roster that leaves a required field unset
//! fails to load, so a spec that exists is a spec that is complete, and the
//! accessors below can hand back values rather than possibilities.
//!
//! [`ProviderSpec::env_api_key`] is the one genuine `Option`, and it means
//! what it says: a provider may legitimately name no environment variable.
//! The three [`Capabilities`] limits are the others, and there `None` means
//! undocumented — never unlimited.
//!
//! Fields are `pub(crate)` so the loader can build these. They are private to
//! everyone else: outside this crate a spec is read-only.

use std::collections::HashMap;

use crate::types::{Api, Protocol};

/// The resolved roster: providers, and every routable `model_str` under them.
///
/// Two `HashMap`s rather than one merged table. They are keyed at different
/// levels and looked up in sequence — the model row names its provider, and
/// the provider row is fetched by that name — so a provider's transport is
/// stored once no matter how many models it serves.
#[derive(Debug, Default)]
pub(crate) struct Registry {
    /// Keyed by provider name, the first segment of a `model_str`.
    pub(crate) providers: HashMap<String, ProviderSpec>,
    /// Keyed by the whole `model_str`, prefix included.
    pub(crate) models: HashMap<String, ModelSpec>,
}

/// One provider: where its API is, how to authenticate, what protocol it
/// speaks, and which surfaces it serves.
///
/// Everything here is true of every model the provider serves. What varies
/// per model is in [`ModelSpec`], and the split is what keeps a provider's
/// transport stated exactly once.
#[derive(Debug, Clone)]
pub struct ProviderSpec {
    pub(crate) name: String,
    pub(crate) base_url: String,
    pub(crate) env_api_key: Option<String>,
    pub(crate) protocol: Protocol,
    pub(crate) supported_apis: Vec<Api>,
}

impl ProviderSpec {
    /// The provider's name — its key in the roster, and the first segment of
    /// every `model_str` it serves.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The provider's API root, version prefix included and no trailing
    /// slash; an endpoint appends its own path.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The environment variable warpllm reads this provider's API key from, if
    /// the roster names one. Currently the only key source there is.
    ///
    /// `None` therefore means this provider cannot be authenticated: there is
    /// no variable to read and nothing to suggest setting. The roster still
    /// accepts it — a provider entry can land before the key plumbing it needs
    /// does — and a request to one says exactly that rather than naming a
    /// variable nothing reads.
    pub fn env_api_key(&self) -> Option<&str> {
        self.env_api_key.as_deref()
    }

    /// The wire format this provider speaks.
    pub fn protocol(&self) -> Protocol {
        self.protocol
    }

    /// The API surfaces this provider serves — never empty, since a provider
    /// that served nothing could not be routed to and would fail to load.
    pub fn supported_apis(&self) -> &[Api] {
        &self.supported_apis
    }

    /// Whether this provider serves `api`.
    ///
    /// Each variant is its own claim: a provider declaring
    /// [`Api::ChatCompletions`] has said nothing about
    /// [`Api::ChatCompletionsStream`].
    pub fn supports_api(&self, api: Api) -> bool {
        self.supported_apis.contains(&api)
    }
}

/// One routable model: the name it ships upstream, and its published limits.
///
/// Deliberately thin. Everything shared with the provider's other models
/// lives in [`ProviderSpec`], so an entry here is only what makes this model
/// different from its siblings — which for most models is nothing at all.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// The provider serving this model — the key its [`ProviderSpec`] is
    /// filed under, and the first segment of this model's own key.
    pub(crate) provider: String,
    /// Upstream model name — what ships on the wire. Defaults to the key's
    /// last segment, so it differs only when warpllm's routing alias differs
    /// from the provider's own model name.
    pub(crate) model: String,
    pub(crate) capabilities: Capabilities,
}

impl ModelSpec {
    /// The model name as it ships upstream, which differs from the
    /// `model_str` whenever warpllm's routing alias differs from the
    /// provider's own name for it.
    ///
    /// Always a real name: the roster registers every routable model by name,
    /// so there is no entry that serves many and pins none.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// What this model's published limits are.
    pub fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }
}

/// A model's published limits. Deliberately NOT coupled to the request shape:
/// parameter support is passthrough — the provider is the authority and
/// rejects what it doesn't accept. A field is added here only when a real
/// consumer need arrives with it.
///
/// The one type that IS its own YAML schema: it maps one-to-one onto a
/// `capabilities:` block with nothing to settle, so a second struct to
/// deserialize into would be a copy to keep in step. `deny_unknown_fields` is
/// what turns a contributor's `max_input_token:` typo into an error instead of
/// a silently ignored line.
///
/// No `Default`, deliberately. A derived `Default` on a public struct is
/// public too, and the loader's blank starting point is not something a caller
/// should be able to conjure — so it gets `Capabilities::blank`, which is
/// `pub(crate)` and therefore unreachable from outside.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capabilities {
    pub(crate) max_input_tokens: Option<u32>,
    pub(crate) max_output_tokens: Option<u32>,
    /// Requests this model will serve at once. Unset means undocumented,
    /// NOT unlimited. Account tier can move these, so treat the roster
    /// value as the default a config surface would later override.
    pub(crate) max_concurrent_requests: Option<u32>,
}

impl Capabilities {
    /// Nothing recorded — what an entry with no `capabilities:` block
    /// deserializes to.
    pub(crate) const fn blank() -> Self {
        Self {
            max_input_tokens: None,
            max_output_tokens: None,
            max_concurrent_requests: None,
        }
    }

    /// Largest documented input context, in tokens.
    ///
    /// `None` means the registry has no published figure for this model — it
    /// never means unlimited. These three stay `Option` precisely because
    /// undocumented and unbounded are different claims, and the registry
    /// refuses to guess between them.
    pub fn max_input_tokens(&self) -> Option<u32> {
        self.max_input_tokens
    }

    /// Largest documented output length, in tokens. `None` means
    /// undocumented, not unlimited.
    pub fn max_output_tokens(&self) -> Option<u32> {
        self.max_output_tokens
    }

    /// Documented ceiling on requests served at once. `None` means
    /// undocumented, not unlimited. Account tier can move this, so treat it
    /// as a default rather than a hard limit.
    pub fn max_concurrent_requests(&self) -> Option<u32> {
        self.max_concurrent_requests
    }
}
