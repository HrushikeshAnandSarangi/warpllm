# Contributing to warpllm

Thank you for your interest in contributing to warpllm. Welcome to the community building a warp speed AI gateway.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment:

* Be respectful and constructive in all interactions
* Welcome newcomers and help them get started
* Focus on what's best for the community and project
* Accept constructive criticism gracefully
* Show empathy towards other community members

Report any unacceptable behavior to us through our [Discord](https://discord.gg/tSSQTxFnsC).

## How to contribute

### Prerequisites

You need Git and Rust + cargo for any contribution — the core is Rust, and the
bindings compile it too. Add the others only for the package you're touching:

* **Git** and **Rust + cargo** — always
* **Python 3.10+ and uv** — only for `bindings/python`
* **Node 22+ and npm** — only for `bindings/node`

Reading and signing the [warpllm Individual Contributor License Agreement](https://cla-assistant.io/warpllm/warpllm) is mandatory before submitting PRs. You can read the full text in [CLA.md](CLA.md). Expedite the process by signing it sooner.

### Areas of contribution

We welcome contributions in several areas:

* **Model/provider integrations**: Improve the AI Gateway by maintaining or adding more models and providers. Usually an edit to `registry/specs.yaml` — see [Adding a provider or model](#adding-a-provider-or-model).
* **Adding protocols**: Sometimes we see new protocols outside of the ones we support (erhm.. OpenAI-Compatible API). These live in `protocol/`, with conversions in `gateway/`.
* **Documentation**: Improve guides, examples, and API docs
* **Testing**: Increase test coverage always helps
* **Examples**: Create demos and use cases on how to use warpllm
* **Bug Fixes**: Fix reported issues
* **Performance**: Simplify code, reduce latency, or reduce memory usage

## Project structure

warpllm is a single Cargo workspace. The Rust core does the work; the Python and
Node packages are thin bindings over that same core, so a fix in `crates/warpllm`
reaches all three languages at once.

```
crates/warpllm/          The SDK. Everything below is a module of this crate.
  registry/              Which providers and models exist.
    specs.yaml           The roster itself — adding a model is an edit here.
  protocol/              Wire shapes: what a provider's API actually sends
                         and receives, per wire format (not per provider).
  gateway/               warpllm's own request/response types, and the
                         conversions between them and the wire shapes.
  types.rs               The vocabulary both layers share: `Protocol` (which
                         wire format) and `Api` (which surface).
  client.rs              Routes a request: look up the model, pick the
                         protocol, send it, convert the response back.
crates/warpllm-server/   The OpenAI-compatible HTTP gateway (unreleased),
                         an axum server wrapping the SDK.
bindings/python/         PyO3 + maturin. Rust glue in src/, the importable
                         package in python/warpllm/, tests in tests/.
bindings/node/           napi-rs. Rust glue in src/, TypeScript in src-ts/,
                         tests in __test__/.
```

The two ideas worth knowing before you read the code:

* **The registry is the roster, and it fails closed.** A model warpllm doesn't
  know is an error, never a guess at some upstream default. The header comment
  in [`specs.yaml`](crates/warpllm/src/registry/specs.yaml) explains the
  provider/model split and the rules the lint enforces — read it before adding
  either.
* **`Protocol` has one variant per wire format, not per provider.** Providers
  that speak the same dialect share it, and a provider that diverges is handled
  in that dialect's conversions. This is why adding an OpenAI-compatible
  provider is usually a registry edit and no new Rust.

## Development Setup

Clone the repo, then verify the toolchain works before changing anything. The
Rust toolchain is pinned by `rust-toolchain.toml`, so `cargo` installs the right
version on first use.

```bash
git clone https://github.com/warpllm/warpllm.git
cd warpllm
cargo test --workspace
```

The bindings each build a native module from the Rust core, so they need a
working `cargo` too — but you only need to set one up if you're changing that
language's package:

```bash
# Python
cd bindings/python && uv sync --locked && uv run pytest

# Node
cd bindings/node && npm ci && ./node_modules/.bin/napi build --platform && npm test
```

**No API keys are needed to develop or run the tests.** The suites run against
mock HTTP servers. Keys are only read at request time, from the routed
provider's environment variable (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, …), if
you want to make a real call by hand.

### Before you open a PR

These are exactly what CI runs, so running them locally is the whole gate:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Plus the suite for any binding you touched, from the commands above. Tests are
part of the change, not a follow-up — if you add a provider, add the case that
routes to it.

### Adding a provider or model

The common path (if the provider speaks a protocol warpllm already knows), in order:

1.  Add the entry to
    [`crates/warpllm/src/registry/specs.yaml`](crates/warpllm/src/registry/specs.yaml),
    following the rules in that file's header.
1.  Run `cargo test -p warpllm`. The registry has both load-time gates and
    lints, and this is where a bad entry gets caught.
1.  Add a test under `crates/warpllm/tests/providers/`, alongside the existing
    `openai` and `deepseek` cases.

## Community

### Communication Channels

* **To report bugs and feature requests**: [GitHub Issues](https://github.com/warpllm/warpllm/issues)
* **To chat with the warpllm team (questions, ideas, reports)**:
[Discord](https://discord.gg/tSSQTxFnsC)
* **To discuss amongst the community**: [Reddit](https://www.reddit.com/r/warpllm/)

### Getting Help

* Check existing documentation and examples
* Search closed issues for similar problems
* Ask on Discord for quick questions

### Recognition

We value all contributions! Contributors are:

* Listed in release notes
* Mentioned in our README

## Questions?

If you have any questions, ask them away at any of these channels:

[![Discord](https://img.shields.io/badge/Discord-warpllm-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/tSSQTxFnsC)
[![Reddit](https://img.shields.io/badge/Reddit-r%2Fwarpllm-FF4500?style=for-the-badge&logo=reddit&logoColor=white)](https://www.reddit.com/r/warpllm/)
