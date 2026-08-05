import { Client as NativeClient } from '../index.js'

import { throwFromWire } from './errors.js'
import type {
  CreateChatCompletionRequest,
  CreateChatCompletionResponse,
} from './generated/types.js'

/** Constructor options. Mirrors Rust's `ClientConfig`. */
export interface WarpLLMOptions {
  baseUrl?: string
  /** Request timeout in seconds (default 600, matching the OpenAI SDK). */
  timeout?: number
}

/**
 * Model strings are `provider/model`, e.g. `"openai/gpt-5.6"`. API keys come
 * from the environment (`OPENAI_API_KEY`); a provider's key is only required
 * when a request targets that provider.
 */
export class WarpLLM {
  private readonly native: NativeClient

  constructor(options: WarpLLMOptions = {}) {
    try {
      this.native = new NativeClient(
        JSON.stringify({ base_url: options.baseUrl, timeout_secs: options.timeout }),
      )
    } catch (err) {
      throwFromWire(err)
    }
  }

  /**
   * One method, mirroring Rust's `client.chat_completion(request)`.
   *
   * The request crosses verbatim — its fields are Rust's, so nothing here
   * renames them and nothing here has to learn a field warpllm gains.
   *
   * Generic in the request so a parameter warpllm does not model still
   * type-checks: TypeScript skips excess-property checking when the target is
   * a type parameter, which is what lets `{ ...req, seed: 7 }` compile without
   * a cast and without an index signature on the generated declaration. An
   * index signature would make every `interface` in the official `openai`
   * package unassignable here, since TypeScript gives interfaces no implicit
   * one.
   */
  async chatCompletion<T extends CreateChatCompletionRequest>(
    request: T,
  ): Promise<CreateChatCompletionResponse> {
    let raw: string
    try {
      raw = await this.native.chatCompletion(JSON.stringify(request))
    } catch (err) {
      throwFromWire(err)
    }
    return JSON.parse(raw) as CreateChatCompletionResponse
  }
}
