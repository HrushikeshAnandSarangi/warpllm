import type { ErrorBody, ErrorClass, OpenAiError } from './generated/types.js'

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

const BY_CLASS: Record<ErrorClass, Constructor> = {
  api_connection: APIConnectionError,
  api_status: APIError,
  bad_request: BadRequestError,
  authentication: AuthenticationError,
  permission_denied: PermissionDeniedError,
  not_found: NotFoundError,
  conflict: ConflictError,
  unprocessable_entity: UnprocessableEntityError,
  rate_limit: RateLimitError,
  internal_server: InternalServerError,
}

/**
 * Rebuilds the error from the native layer's JSON message.
 *
 * Nothing is interpreted here. Rust already decided what the failure is and
 * what OpenAI calls it; this only instantiates the class named by the wire
 * discriminant.
 */
export function throwFromWire(err: unknown): never {
  const raw = err instanceof Error ? err.message : String(err)
  let wire: OpenAiError
  try {
    wire = JSON.parse(raw) as OpenAiError
  } catch {
    // Not our JSON — surface it whole rather than inventing a shape for it.
    throw new APIError(undefined, undefined, raw, undefined)
  }
  const Class = BY_CLASS[wire.class] ?? APIError
  throw new Class(wire.status ?? undefined, wire.error, wire.error.message, wire.headers)
}
