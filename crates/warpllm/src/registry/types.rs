//! What a registry entry is, and how a caller reads one.
//!
//! These types are BOTH the YAML schema and the resolved spec. One struct
//! serves the entry a contributor writes, the accumulator the merge walks a
//! path into, and the row a caller reads. That is why every field is an
//! `Option`: in the YAML a missing field means "inherit from my namespace",
//! and mid-merge it means "no ancestor has said yet".
//!
//! By the time a spec reaches a caller most of that is no longer open.
//! `finish` in the parent module refuses to produce a spec whose `base_url`,
//! `protocol`, or `supported_endpoints` is still `None`, so a roster that
//! would produce one fails to load. That is what lets those accessors resolve
//! without a fallback: read their `Option`s as "not answered YET", never as
//! "may be absent at runtime".
//!
//! `env_api_key` is the exception, and the only field whose `Option` survives
//! into the public API. A model may legitimately name no environment
//! variable — see [`ModelSpec::env_api_key`].
//!
//! Fields are `pub(crate)` so the loader next door can merge them. They are
//! private to everyone else: outside this crate a spec is read-only, and
//! [`crate::model_spec`] is the only way to obtain one.

use std::collections::HashMap;

use crate::protocol::Protocol;

/// Every routable `model_str` -> its spec, already merged along its path, so
/// nothing at runtime walks a chain.
///
/// A `HashMap` rather than the sorted, binary-searched slice this used to be:
/// the roster is headed well past the ~16 entries where a hash starts winning,
/// and the table is now built once behind a `LazyLock` instead of emitted as a
/// `const`, so nothing is left that the ordering was buying.
#[derive(Debug, Default)]
pub(crate) struct Registry {
    pub(crate) specs: HashMap<String, ModelSpec>,
}

impl Registry {
    /// The spec filed under `model_str`, or `None` if nothing is.
    pub(crate) fn get(&self, model_str: &str) -> Option<&ModelSpec> {
        self.specs.get(model_str)
    }
}

/// What a model can do. Deliberately NOT coupled to the request shape:
/// parameter support is passthrough — the provider is the authority and
/// rejects what it doesn't accept. A field is added here only when a real
/// consumer need arrives with it.
///
/// `deny_unknown_fields` is what turns a contributor's `max_input_token:`
/// typo into an error instead of a silently ignored line.
///
/// No `Default`, deliberately. A derived `Default` on a public struct is
/// public too, and `Capabilities::default()` would hand a caller a value whose
/// accessors panic. The loader wants a blank one, so it gets
/// [`Capabilities::blank`] — `pub(crate)`, therefore unreachable from outside.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capabilities {
    /// Endpoint paths under the provider's `base_url` this model serves,
    /// e.g. `"/chat/completions"`. A list REPLACES rather than extends the
    /// one it inherits, which is the only way an entry can say it serves
    /// FEWER endpoints than its namespace.
    pub(crate) supported_endpoints: Option<Vec<String>>,
    pub(crate) max_input_tokens: Option<u32>,
    pub(crate) max_output_tokens: Option<u32>,
    /// Requests this model will serve at once. Unset means undocumented,
    /// NOT unlimited. Account tier can move these, so treat the roster
    /// value as the default a config surface would later override.
    pub(crate) max_concurrent_requests: Option<u32>,
}

impl Capabilities {
    /// Nothing answered yet — the starting point of a merge, and what an
    /// entry with no `capabilities:` block deserializes to.
    pub(crate) const fn blank() -> Self {
        Self {
            supported_endpoints: None,
            max_input_tokens: None,
            max_output_tokens: None,
            max_concurrent_requests: None,
        }
    }

    /// Endpoint paths this model serves, relative to [`ModelSpec::base_url`]
    /// — never empty, since a model that served nothing could not be routed
    /// to and would fail to load.
    pub fn supported_endpoints(&self) -> &[String] {
        required(
            self.supported_endpoints.as_deref(),
            "capabilities.supported_endpoints",
        )
    }

    /// Whether this model serves `endpoint`, e.g. `"/chat/completions"`.
    pub fn supports_endpoint(&self, endpoint: &str) -> bool {
        self.supported_endpoints()
            .iter()
            .any(|serves| serves == endpoint)
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

/// Everything warpllm knows about one model: how to reach it and what it
/// can do. The spec is per MODEL, not per provider, because that is the
/// granularity at which support diverges — two models from the same
/// provider can differ — so this is the only layer to maintain.
///
/// Obtain one with [`crate::model_spec`] and read it through the accessors;
/// the fields are private because their `Option` is a merge-time state, not a
/// question left open for callers.
///
/// No `Default`, for the reason given on [`Capabilities`]. `Deserialize` is
/// the one hole left: this declaration doubles as the registry's YAML schema,
/// so the impl cannot be withheld without a second struct to maintain. A spec
/// deserialized from anything but the registry can be missing fields, and the
/// accessors panic on those rather than inventing a value.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    /// The first segment of the entry's key; also names the provider in
    /// errors and its `ext` namespace.
    ///
    /// `skip`ped rather than deserialized: the key already says which
    /// provider an entry belongs to, and with `deny_unknown_fields` this
    /// makes a hand-written `provider:` line an error rather than a second,
    /// disagreeing source of truth.
    #[serde(skip)]
    pub(crate) provider: Option<String>,
    /// Whether this entry came from a `*` key, and so serves every name
    /// under its namespace that nothing more specific claims.
    ///
    /// `skip`ped for the same reason as `provider`: the key decides it, and a
    /// hand-written line saying otherwise would be a second source of truth.
    #[serde(skip)]
    pub(crate) wildcard: bool,
    /// Upstream model name — what ships on the wire. Defaults to the key's
    /// last segment, so set it only when warpllm's routing alias differs
    /// from the provider's own model name.
    ///
    /// A wildcard entry has no fixed one, and holds `*`. Read
    /// [`ModelSpec::wire_model`] instead of this whenever the spec might be
    /// one.
    pub(crate) model: Option<String>,
    /// API root including any version prefix (e.g. `/v1`); endpoint impls
    /// append only their own path (e.g. `/chat/completions`).
    pub(crate) base_url: Option<String>,
    /// Environment variable the API key is read from BY DEFAULT — the
    /// registry's suggestion, not the only source. An explicit key already
    /// wins over it (`Client::with_api_key`), and client configuration will
    /// be able to override it per provider.
    ///
    /// Unlike the other transport fields this one is genuinely OPTIONAL, and
    /// stays `Option` all the way out to the accessor. An entry that names
    /// none authenticates only with an explicitly supplied key, which is the
    /// right answer for a model whose credential is per-caller rather than
    /// per-deployment. It still inherits from its namespace, so omitting it
    /// under a namespace that names one changes nothing — dropping it means
    /// declaring it nowhere in the chain.
    pub(crate) env_api_key: Option<String>,
    pub(crate) protocol: Option<Protocol>,
    /// Names `blank` rather than relying on `#[serde(default)]`, which would
    /// need `Capabilities: Default` — the public constructor this type does
    /// not want.
    #[serde(default = "Capabilities::blank")]
    pub(crate) capabilities: Capabilities,
}

impl ModelSpec {
    /// Nothing answered yet — what a merge starts from before any ancestor
    /// has been applied.
    pub(crate) const fn blank() -> Self {
        Self {
            provider: None,
            wildcard: false,
            model: None,
            base_url: None,
            env_api_key: None,
            protocol: None,
            capabilities: Capabilities::blank(),
        }
    }

    /// The provider that serves this model — the first segment of the
    /// `model_str` it was looked up by.
    pub fn provider(&self) -> &str {
        required(self.provider.as_deref(), "provider")
    }

    /// The model name as it ships upstream, which differs from the
    /// `model_str` whenever warpllm's routing alias differs from the
    /// provider's own name for it.
    ///
    /// `"*"` when [`ModelSpec::is_wildcard`] — the entry serves many names
    /// and pins none of them. Use [`ModelSpec::wire_model`] unless you
    /// already know the spec is a concrete entry.
    pub fn model(&self) -> &str {
        required(self.model.as_deref(), "model")
    }

    /// Whether this spec came from a `*` key, and so serves every name under
    /// its namespace that no exact entry claims.
    ///
    /// A wildcard is opt-in per namespace, and it is the one thing that lets
    /// an unlisted name reach a provider: everywhere else an unregistered
    /// model is an error.
    pub fn is_wildcard(&self) -> bool {
        self.wildcard
    }

    /// The name to send upstream when routing `model_str` — the entry's own
    /// [`model`](ModelSpec::model) normally, and `model_str`'s last segment
    /// when this is a wildcard.
    ///
    /// Pass the same string [`crate::model_spec`] was given. A wildcard entry
    /// cannot answer this on its own: what ships upstream is whatever the
    /// caller asked for, minus the provider prefix.
    pub fn wire_model<'a>(&'a self, model_str: &'a str) -> &'a str {
        if self.wildcard {
            model_str
                .rsplit_once('/')
                .map_or(model_str, |(_, name)| name)
        } else {
            self.model()
        }
    }

    /// The provider's API root, version prefix included and no trailing
    /// slash; an endpoint appends its own path.
    pub fn base_url(&self) -> &str {
        required(self.base_url.as_deref(), "base_url")
    }

    /// The environment variable warpllm reads this model's API key from by
    /// default, if the registry names one.
    ///
    /// `None` is a real answer, not a gap: some models authenticate only with
    /// a key the caller supplies, so there is no variable to read and nothing
    /// to suggest setting. Either way this is never the only source — an
    /// explicit key (`Client::with_api_key`) wins over it.
    pub fn env_api_key(&self) -> Option<&str> {
        self.env_api_key.as_deref()
    }

    /// The wire format this model speaks.
    pub fn protocol(&self) -> Protocol {
        required(self.protocol, "protocol")
    }

    /// What this model can do.
    pub fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }
}

/// Reads a field the registry has already settled.
///
/// Every `Option` above means "no ancestor has said yet" during the merge,
/// never "may be absent at runtime": `finish` in the parent module refuses to
/// produce a spec with any of these unset, and `the_shipped_registry_loads_and_lints`
/// holds the roster to it, so nothing that could reach the panic below gets
/// past CI. One helper so that invariant is stated once instead of at ten call
/// sites.
fn required<T>(value: Option<T>, field: &str) -> T {
    value.unwrap_or_else(|| {
        unreachable!("the registry produced a spec with no `{field}`; `finish` rejects those")
    })
}
