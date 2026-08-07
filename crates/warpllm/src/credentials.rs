//! Which providers this process can authenticate, worked out once.
//!
//! The roster says which providers exist and which environment variable each
//! reads its key from. This turns that into the shorter list that matters to a
//! running client: the providers whose variable is actually set. A model may be
//! registered and still be unreachable, because the key for its provider is not
//! in this environment — and the two are different failures with different
//! remedies.
//!
//! Read once, at construction. A snapshot rather than a live read, so the set of
//! providers a client can reach cannot change under it mid-flight, and so the
//! answer can be logged at the one moment a caller is set up to see it. The cost
//! is the other side of that coin: a key exported after the client was built is
//! not picked up, and a rotated key needs a new client.

use std::collections::HashMap;

use crate::registry;

/// The API keys this process holds, keyed by provider name.
///
/// A `HashMap`, matching the registry's own tables: this answers one question,
/// "the key for this provider", and answers it on every request. Iteration
/// order is unspecified and varies run to run, so everything that RENDERS the
/// set goes through [`Credentials::names`] instead.
pub(crate) struct Credentials {
    keys: HashMap<&'static str, String>,
}

impl Credentials {
    /// Reads every roster provider's key out of the environment.
    ///
    /// A provider is included only if the roster names a variable for it AND
    /// that variable holds a non-empty value. Both exclusions are the same
    /// judgement: warpllm has no key for this provider, so a request routed to
    /// it should say so rather than send `Authorization: Bearer ` upstream and
    /// report back whatever the provider makes of it.
    ///
    /// The environment is the only source. [`crate::ClientConfig`] deliberately
    /// carries no key, so there is nothing here to merge or override.
    pub(crate) fn from_env() -> Self {
        let credentials = Self {
            keys: registry::providers()
                .filter_map(|provider| {
                    let key = std::env::var(provider.env_api_key()?).ok()?;
                    (!key.is_empty()).then_some((provider.name(), key))
                })
                .collect(),
        };

        // The names only. A key never reaches a log, in full or in part.
        if credentials.keys.is_empty() {
            tracing::warn!(
                "no provider API keys found in the environment; every request \
                 will fail until one is set"
            );
        } else {
            tracing::info!(providers = ?credentials.names(), "providers available");
        }
        credentials
    }

    /// This provider's key, or `None` when the environment held none at
    /// construction — which is what makes "warpllm cannot reach this provider"
    /// answerable without a request.
    pub(crate) fn get(&self, provider: &str) -> Option<&str> {
        self.keys.get(provider).map(String::as_str)
    }

    /// The providers a request can be routed to, in name order.
    ///
    /// Sorted here rather than held sorted, because the map's order is
    /// unspecified: this is the only thing standing between a hash-seeded
    /// iteration and a log line whose provider list reshuffles between runs.
    /// Called once per client at construction, never on the request path.
    fn names(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = self.keys.keys().copied().collect();
        names.sort_unstable();
        names
    }
}

/// Names only, never values. Derived `Debug` would print every API key the
/// process holds the moment anything upstream formats a [`crate::Client`].
impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credentials")
            .field("providers", &self.names())
            .finish()
    }
}

/// Runs `body` holding temp-env's global lock, with EVERY roster variable
/// unset and then `vars` applied over the top.
///
/// EVERY test in this binary that builds a [`Credentials`] — which now means
/// every test that builds a [`crate::Client`] — has to go through this, even
/// the ones setting nothing. Env mutation is process-global and `unsafe` since
/// edition 2024 precisely because one test writing a variable while another
/// reads it is a data race, and only a shared lock rules that out. An empty
/// `vars` is the "reads the environment, sets nothing" case.
///
/// Clearing the whole roster first is what keeps the AMBIENT environment out of
/// the result. Naming only the variables a test cares about would leave a
/// contributor who exports a third provider's key failing an assertion about
/// which providers are available — and would quietly add every new provider to
/// every snapshot as the roster grows.
#[cfg(test)]
pub(crate) fn with_env<T>(vars: &[(&str, Option<&str>)], body: impl FnOnce() -> T) -> T {
    let mut settings: std::collections::BTreeMap<String, Option<String>> = registry::providers()
        .filter_map(|provider| provider.env_api_key())
        .map(|var| (var.to_string(), None))
        .collect();
    for (var, value) in vars {
        settings.insert((*var).to_string(), value.map(str::to_string));
    }
    temp_env::with_vars(settings.into_iter().collect::<Vec<_>>(), body)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPENAI: &str = "OPENAI_API_KEY";
    const DEEPSEEK: &str = "DEEPSEEK_API_KEY";

    /// A key for one provider makes that provider available and says nothing
    /// about any other — the roster is not the availability list.
    #[test]
    fn one_key_admits_one_provider() {
        with_env(&[(OPENAI, Some("sk-openai")), (DEEPSEEK, None)], || {
            let credentials = Credentials::from_env();
            assert_eq!(credentials.names(), vec!["openai"]);
            assert_eq!(credentials.get("openai"), Some("sk-openai"));
            assert_eq!(credentials.get("deepseek"), None);
        });
    }

    /// Every provider whose variable is set is available at once, in name
    /// order.
    #[test]
    fn every_set_key_admits_its_provider() {
        with_env(
            &[(OPENAI, Some("sk-openai")), (DEEPSEEK, Some("sk-deepseek"))],
            || {
                let credentials = Credentials::from_env();
                assert_eq!(credentials.names(), vec!["deepseek", "openai"]);
                assert_eq!(credentials.get("deepseek"), Some("sk-deepseek"));
            },
        );
    }

    /// An empty environment is not an error. Construction succeeds with nothing
    /// available, and the failure lands on the request that needed a key —
    /// where the message can name the provider and the variable to set.
    #[test]
    fn an_empty_environment_yields_no_providers() {
        with_env(&[(OPENAI, None), (DEEPSEEK, None)], || {
            let credentials = Credentials::from_env();
            assert!(credentials.names().is_empty());
            assert_eq!(credentials.get("openai"), None);
        });
    }

    /// `OPENAI_API_KEY=` is a variable that is set and a key that is not.
    /// Treating it as present would send `Bearer ` upstream and turn a local
    /// configuration mistake into a remote authentication failure.
    #[test]
    fn an_empty_value_is_not_a_key() {
        with_env(&[(OPENAI, Some("")), (DEEPSEEK, None)], || {
            assert!(Credentials::from_env().names().is_empty());
        });
    }

    /// A provider warpllm does not serve has no key here, whatever the
    /// environment holds.
    #[test]
    fn an_unknown_provider_is_never_available() {
        with_env(&[(OPENAI, Some("sk-openai"))], || {
            assert_eq!(Credentials::from_env().get("mistral"), None);
        });
    }

    /// Debug is what a panic or a `tracing` field would print. It must name the
    /// providers and none of their keys.
    #[test]
    fn debug_redacts_the_keys() {
        with_env(
            &[(OPENAI, Some("sk-secret-value")), (DEEPSEEK, None)],
            || {
                let rendered = format!("{:?}", Credentials::from_env());
                assert!(rendered.contains("openai"), "{rendered}");
                assert!(!rendered.contains("sk-secret-value"), "{rendered}");
            },
        );
    }
}
