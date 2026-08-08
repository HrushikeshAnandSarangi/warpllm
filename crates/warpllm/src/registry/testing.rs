//! Roster fixtures shared by the three modules' tests.
//!
//! Here rather than duplicated because the line numbers matter, and because
//! `supported_apis` is required of every model: a case that only needs a
//! routable name says [`model`] rather than spelling out three lines of
//! surface it has no opinion about.

use super::load::load;
use super::types::Registry;

/// One provider with no models yet — the base most cases diverge from.
///
/// Five lines, ending on its `models:` key, so a case that appends entries
/// knows its first key lands on line 6 — which is what the sort check's error
/// message is asserted against. A [`model`] entry is three lines, so the
/// second key lands on line 9.
pub(super) const PROVIDER: &str = "\
providers:
  demo:
    base_url: \"https://api.demo.test/v1\"
    env_api_key: DEMO_API_KEY
    models:
";

/// A second provider, complete with a model, sorted after `demo` so it can be
/// appended to a [`with`] roster without tripping the order check.
///
/// Written without the `\` line continuation the fixture above uses: that
/// escape swallows the following line's indentation, and this block's two
/// leading spaces are what make it a sibling of `demo` rather than a
/// top-level key.
pub(super) const OTHER_PROVIDER: &str = "  other:
    base_url: \"https://api.other.test\"
    env_api_key: OTHER_API_KEY
    models:
      other/plain:
        supported_apis:
          - {api: openai_compat_chat_completions}
";

/// The one line every model entry must carry, indented to sit under a key at
/// six spaces.
///
/// `supported_apis` is required and inherits nothing, so there is no such
/// thing as a bare `demo/plain: {}` any more. Cases that have no opinion about
/// surfaces say so through [`model`] rather than repeating this.
pub(super) const CHAT: &str =
    "        supported_apis:\n          - {api: openai_compat_chat_completions}\n";

/// The minimal valid entry for `key`: it serves chat completions and says
/// nothing else.
pub(super) fn model(key: &str) -> String {
    format!("      {key}:\n{CHAT}")
}

/// The minimal valid entries for several keys, in the order given — which is
/// the order they will appear in the file, so a case testing the sort check
/// can hand them over deliberately unsorted.
pub(super) fn models(keys: &[&str]) -> String {
    keys.iter().map(|key| model(key)).collect()
}

pub(super) fn with(models: &str) -> String {
    format!("{PROVIDER}{models}")
}

/// A roster expected to be valid AND clean, resolved. Anything the shipped file
/// has to satisfy, a fixture is held to as well — so a fixture cannot quietly
/// rely on something the roster itself would be rejected for.
pub(super) fn clean(yaml: &str) -> Registry {
    super::lint::check(yaml).unwrap_or_else(|e| panic!("{e}"));
    load(yaml).unwrap_or_else(|e| panic!("{e}"))
}

/// Sorted, because the tables are `HashMap`s and an assertion on their
/// contents must not depend on iteration order.
pub(super) fn keys(registry: &Registry) -> Vec<&str> {
    sorted(registry.models.keys())
}

pub(super) fn providers(registry: &Registry) -> Vec<&str> {
    sorted(registry.providers.keys())
}

fn sorted<'a>(keys: impl Iterator<Item = &'a String>) -> Vec<&'a str> {
    let mut keys: Vec<&str> = keys.map(String::as_str).collect();
    keys.sort_unstable();
    keys
}
