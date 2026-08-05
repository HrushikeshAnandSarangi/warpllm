/**
 * The errors warpllm raises, matching the shape and the dispatch of the
 * official OpenAI SDK.
 *
 * warpllm's own failure taxonomy is not here and never crosses from Rust. It
 * does its work before this point: warpllm knows a quota exhaustion from a
 * rate limit, and knows DeepSeek's 402 for an exhausted balance means what
 * OpenAI calls 429 `insufficient_quota` — so what reaches you is what OpenAI
 * would have said, whichever provider served the request.
 *
 * ```ts
 * try {
 *   await client.chatCompletion({ model: 'deepseek/deepseek-chat', messages })
 * } catch (err) {
 *   if (err instanceof RateLimitError) {
 *     // A 429 covers two failures that need opposite handling. Waiting fixes
 *     // one of them; only `code` says which.
 *     if (err.code === 'insufficient_quota') await topUp()
 *     else await backOff()
 *   }
 * }
 * ```
 */
export class APIError extends Error {
  /** HTTP status of the response that caused the error. */
  readonly status: number | undefined

  /** Response headers worth keeping — `retry-after`, `x-request-id`. */
  readonly headers: Record<string, string> | undefined

  /** The error object as it came off the wire. */
  readonly error: ErrorBody | undefined

  /**
   * The failure's own slug, free-form per OpenAI. The only field that
   * separates failures sharing a status — `insufficient_quota` from a plain
   * rate limit, both of them 429.
   */
  readonly code: string | null | undefined

  /** Which request parameter was at fault. warpllm does not model this yet. */
  readonly param: string | null | undefined

  /** OpenAI error family, e.g. `invalid_request_error`, `server_error`. */
  readonly type: string | undefined

  /** The upstream's request id, when it sent one. */
  readonly requestID: string | null | undefined

  constructor(
    status: number | undefined,
    error: ErrorBody | undefined,
    message: string | undefined,
    headers: Record<string, string> | undefined,
  ) {
    super(message ?? error?.message ?? 'Unknown error')
    this.name = new.target.name
    this.status = status
    this.headers = headers
    this.error = error
    this.code = error?.code
    this.param = error?.param
    this.type = error?.type
    this.requestID = headers?.['x-request-id']
  }

  /**
   * Picks the class for a status, exactly as the OpenAI SDK's own factory
   * does. Keyed on status alone — which is why `code` carries everything a
   * status cannot say.
   */
  static generate(
    status: number | undefined,
    error: ErrorBody | undefined,
    message: string | undefined,
    headers: Record<string, string> | undefined,
  ): APIError {
    // No status means no response ever arrived.
    if (status === undefined) return new APIConnectionError(status, error, message, headers)
    const Class = BY_STATUS[status] ?? (status >= 500 ? InternalServerError : APIError)
    return new Class(status, error, message, headers)
  }
}

/** The `error` object, spelled as OpenAI spells it. */
export interface ErrorBody {
  message: string
  type: string
  param: string | null
  code: string | null
}

/** warpllm never reached the provider, so there is no status and no body. */
export class APIConnectionError extends APIError {}

export class BadRequestError extends APIError {}
export class AuthenticationError extends APIError {}
export class PermissionDeniedError extends APIError {}
export class NotFoundError extends APIError {}
export class ConflictError extends APIError {}
export class UnprocessableEntityError extends APIError {}

/**
 * A 429 — and NOT only a rate limit. An exhausted quota arrives here too,
 * because OpenAI reports both under one status and one class. Backing off
 * clears the first and never clears the second, so read `code` before
 * retrying: `insufficient_quota` means somebody has to pay.
 */
export class RateLimitError extends APIError {}

export class InternalServerError extends APIError {}

type Constructor = new (
  status: number | undefined,
  error: ErrorBody | undefined,
  message: string | undefined,
  headers: Record<string, string> | undefined,
) => APIError

const BY_STATUS: Record<number, Constructor> = {
  400: BadRequestError,
  401: AuthenticationError,
  403: PermissionDeniedError,
  404: NotFoundError,
  409: ConflictError,
  422: UnprocessableEntityError,
  429: RateLimitError,
}

/** What crosses the FFI — Rust's `Error::to_openai`. */
interface Wire {
  status?: number | null
  headers?: Record<string, string>
  error?: ErrorBody
}

/**
 * Rebuilds the error from the native layer's JSON message.
 *
 * Nothing is interpreted here. Rust already decided what the failure is and
 * what OpenAI calls it; this only chooses the class the status implies.
 */
export function throwFromWire(err: unknown): never {
  const raw = err instanceof Error ? err.message : String(err)
  let wire: Wire
  try {
    wire = JSON.parse(raw) as Wire
  } catch {
    // Not our JSON — surface it whole rather than inventing a shape for it.
    throw new APIError(undefined, undefined, raw, undefined)
  }
  throw APIError.generate(wire.status ?? undefined, wire.error, wire.error?.message, wire.headers)
}
