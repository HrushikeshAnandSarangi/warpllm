/**
 * Every failure warpllm reports, carrying what the official OpenAI SDK
 * carries on `APIError` — and nothing else.
 *
 * warpllm's own taxonomy (its `code` slugs, `origin`, the provider evidence)
 * deliberately stays in Rust. You program against this class directly, so a
 * property named here is one warpllm owes compatibility on, and that taxonomy
 * is not settled enough to promise.
 *
 * Branch on `code`, never on `status`. The statuses lie in both directions:
 * a 403 permission failure and a 401 bad key are both credential problems,
 * while one 429 is a rate limit and another is a billing failure that no
 * amount of backing off will clear — OpenAI spells that second one
 * `insufficient_quota`.
 *
 * ```ts
 * try {
 *   await client.chatCompletion({ model: 'openai/gpt-5.6', messages })
 * } catch (err) {
 *   if (err instanceof WarpLLMError && err.code === 'insufficient_quota') topUp()
 * }
 * ```
 */
export class WarpLLMError extends Error {
  /** HTTP status of the response that caused the error. */
  readonly status: number | undefined

  /** OpenAI error family, e.g. `"invalid_request_error"`, `"api_error"`. */
  readonly type: string | undefined

  /**
   * The failure's own slug — the provider's when an upstream named it, so a
   * quota exhaustion stays `insufficient_quota`. Free-form, per OpenAI.
   */
  readonly code: string | null | undefined

  /** Which request parameter was at fault. warpllm does not model this yet. */
  readonly param: string | null | undefined

  /** The upstream's request id, when it sent one. */
  readonly requestID: string | null | undefined

  constructor(message: string, wire: WireError = {}) {
    super(message)
    this.name = 'WarpLLMError'
    this.status = wire.status
    this.type = wire.error?.type
    this.code = wire.error?.code
    this.param = wire.error?.param
    this.requestID = wire.request_id
  }
}

/**
 * What crosses the FFI — `Error::to_openai_json` in Rust.
 *
 * Shaped like the arguments the OpenAI SDK builds an `APIError` from: the
 * error object exactly as OpenAI spells it, beside what an SDK would
 * otherwise read off the HTTP response.
 */
interface WireError {
  status?: number
  request_id?: string | null
  error?: {
    message?: string
    type?: string
    param?: string | null
    code?: string | null
  }
}

/** Turns the native layer's wire-format JSON message into a `WarpLLMError`. */
export function throwFromWire(err: unknown): never {
  const raw = err instanceof Error ? err.message : String(err)
  let wire: WireError
  try {
    wire = JSON.parse(raw) as WireError
  } catch {
    // Not our JSON — surface it whole rather than inventing a shape for it.
    throw new WarpLLMError(raw)
  }
  throw new WarpLLMError(wire.error?.message ?? raw, wire)
}
