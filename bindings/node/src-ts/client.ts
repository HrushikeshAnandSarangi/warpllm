import { Client as NativeClient } from '../index.js'

import { throwFromWire } from './errors.js'
import type {
  CreateChatCompletionRequest,
  CreateChatCompletionResponse,
  WarpLLMOptions,
} from './types.js'

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
   */
  async chatCompletion(
    request: CreateChatCompletionRequest,
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
