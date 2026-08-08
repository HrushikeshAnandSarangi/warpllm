# warpllm

A warp-speed, robust AI gateway written for rust, node, and python applications - built for planet scale by the community.

[![Discord](https://img.shields.io/badge/Discord-warpllm-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/payU7Vzuq5)
[![Reddit](https://img.shields.io/badge/Reddit-r%2Fwarpllm-FF4500?style=for-the-badge&logo=reddit&logoColor=white)](https://www.reddit.com/r/warpllm/)

[![crates.io](https://img.shields.io/crates/v/warpllm?logo=rust&label=crates.io)](https://crates.io/crates/warpllm)
[![PyPI](https://img.shields.io/pypi/v/warpllm?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/warpllm/)
[![npm](https://img.shields.io/npm/v/%40warpllm%2Fwarpllm?logo=npm&label=npm)](https://www.npmjs.com/package/@warpllm/warpllm)

## Mission

This project is to lay out the most resilient open source productionization layer for AI-deployments. Designed for you if you want:

1.  To work with multiple AI providers or your own models.
1.  To keep your AI services up and running with 0 downtime.
1.  Speed (minimal overhead latency).
1.  A granular view of your metrics (uptime, P95 latency, costs, etc).
1.  Control over:
    1.  Where your data goes.
    1.  Your AI budget across providers.

## Status

> [!IMPORTANT]
> The published packages are **0.2.0**, which adds the provider registry —
> DeepSeek and OpenRouter alongside OpenAI, reached through `provider/model`
> routing strings. It is a **breaking** release: `chat_completion` is now
> `chat_completions`, and most of the public surface moved. See the
> [changelog](CHANGELOG.md) before upgrading from `0.1.x`.
>
> The OpenAI-compatible HTTP gateway has landed on `main` but is **not
> released yet**. Streaming is not implemented.

Install the published SDK:

```bash
cargo add warpllm                # rust
pip install warpllm              # python
npm install @warpllm/warpllm     # node
```

| | Released (0.2.0) | On `main` |
| --- | --- | --- |
| OpenAI chat completions, non-streaming | Yes | Yes |
| `provider/model` routing strings | Provider registry | Provider registry |
| DeepSeek, OpenRouter | Yes | Yes |
| OpenAI-compatible HTTP gateway | — | Unreleased |
| Streaming | — | — |
| Failover, load balancing, caching, metrics | — | — |

Unlisted models are rejected rather than guessed at, so routing a name warpllm
doesn't know is an error, not a surprise upstream bill.

## Layers

1.  **An SDK** - provide a request and we translate it to work with different providers and models out of box.
1.  [Unreleased] **A proxy** - run a self-hosted proxy that speaks the OpenAI API:
    1.  [Coming Soon] **Failover** - define multiple models to handle outages / errors
    1.  [Coming Soon] **Load Balancing** - define a % of requests to be handled per model
    1.  [Coming Soon] **Prompt Response Caching** - define a TTL and avoid paying twice for the same prompt

## Key focus points

1.  **Native SDK support** - Written once in rust, compiled for maximum performance, available for rust/typescript/python.
1.  **Self hostable** - Avoid vendor lock-in (e.g. from cloud provider or model provider), or data leaving your infra.
1.  **Warp-speed execution** - What we named ourselves after. Machine level code, faster than a typescript or python native library.
1.  **Compact file size** - Pre-compiled into binary format, not verbose text files.

## Roadmap

The roadmap lives in [GitHub issues](https://github.com/warpllm/warpllm/issues) — one issue per item, so direction is discussed where the work happens. Add a comment if you see something missing, or if something there matters enough to you that it should move up.

## Contributing

We're excited to have you join us. See the
[contribution guide](CONTRIBUTING.md) for how to get started.

A big thank you to the contributors below who have helped build this AI gateway to this point!

<a href="https://github.com/warpllm/warpllm/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=warpllm/warpllm" />
</a>

## License

The warpllm core is open source under the [Apache License 2.0](https://github.com/warpllm/warpllm/blob/main/LICENSE).
