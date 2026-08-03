export class WarpLLMError extends Error {
  /**
   * Stable machine-readable slug, one per failure, e.g. `"rate_limited"`.
   * warpllm's own failures and the provider's live in one flat space, so
   * `code` alone identifies a failure.
   */
  code: string

  /**
   * `"provider"` means an upstream answered and said no; `"gateway"` means
   * the call never got that far. Locally thrown errors are always
   * `"gateway"`.
   */
  origin: 'gateway' | 'provider'

  constructor(message: string, code = 'error', origin: 'gateway' | 'provider' = 'gateway') {
    super(message)
    this.name = new.target.name
    this.code = code
    this.origin = origin
  }
}

/**
 * warpllm rejected the call before sending it. Distinct from a provider
 * rejecting the request, which throws {@link APIStatusError} with
 * `code === 'provider_invalid_request'` — one is fixed by editing the call,
 * the other may just need a different model.
 */
export class InvalidRequestError extends WarpLLMError {
  constructor(message: string) {
    super(message, 'invalid_request')
  }
}

export class APIConnectionError extends WarpLLMError {
  provider?: string

  constructor(message: string, provider?: string) {
    super(message, 'connection_error')
    this.provider = provider
  }
}

export interface APIStatusErrorOptions {
  status?: number
  provider?: string
  errorType?: string
  retryAfterSeconds?: number
  providerCode?: string
  requestId?: string
  providerRaw?: string
  code?: string
  origin?: 'gateway' | 'provider'
}

/**
 * An upstream provider returned an error.
 *
 * Two groups of properties:
 *
 * - What happened — `code`, the normalized taxonomy (`"rate_limited"`,
 *   `"quota_exceeded"`, `"context_length_exceeded"`, …), and `origin`.
 *   Branch on `code` rather than on the status: a quota exhaustion and a
 *   rate limit both arrive as a 429 and are not the same failure.
 * - What the provider said — `errorType`, `providerCode`, `requestId`,
 *   `retryAfterSeconds`, and `providerRaw`.
 *
 * Nothing here says what to DO about a failure; that is the caller's policy.
 * `retryAfterSeconds` is the provider's own figure, carrying no claim from
 * warpllm that waiting will help.
 */
export class APIStatusError extends WarpLLMError {
  status?: number
  provider?: string
  errorType?: string
  retryAfterSeconds?: number
  providerCode?: string
  requestId?: string
  providerRaw?: string

  constructor(message: string, opts: APIStatusErrorOptions = {}) {
    super(message, opts.code ?? 'error', opts.origin ?? 'provider')
    this.status = opts.status
    this.provider = opts.provider
    this.errorType = opts.errorType
    this.retryAfterSeconds = opts.retryAfterSeconds
    this.providerCode = opts.providerCode
    this.requestId = opts.requestId
    this.providerRaw = opts.providerRaw
  }
}

export class AuthenticationError extends APIStatusError {}

export class PermissionDeniedError extends APIStatusError {}

/**
 * Requests arrived faster than the account may send them. Deliberately NOT
 * thrown for quota exhaustion — see {@link QuotaExceededError}.
 */
export class RateLimitError extends APIStatusError {}

/**
 * The account is out of credit or past a spend cap.
 *
 * A sibling of {@link RateLimitError}, never a subclass, because the two need
 * opposite handling and providers report them under the same 429. Catching
 * this as a rate limit is how a billing failure turns into an infinite
 * backoff loop, so `instanceof RateLimitError` must not match it.
 */
export class QuotaExceededError extends APIStatusError {}

export class NotImplementedError extends WarpLLMError {
  constructor(message: string) {
    super(message, 'not_implemented')
  }
}

interface WireError {
  code?: string
  message?: string
  provider?: string
  status?: number
  origin?: 'gateway' | 'provider'
  error_type?: string | null
  retry_after_seconds?: number | null
  provider_code?: string | null
  request_id?: string | null
  provider_raw?: string
}

/**
 * Codes that earn an error class of their own. Anything absent throws
 * `APIStatusError`, which still carries `code` — a new failure kind is never
 * a crash, just a less specific class. Mirrors the Rust `Error` variants,
 * which is why the two lists are the same shape.
 */
const BY_CODE: Record<string, new (message: string, opts: APIStatusErrorOptions) => APIStatusError> =
  {
    authentication: AuthenticationError,
    permission_denied: PermissionDeniedError,
    rate_limited: RateLimitError,
    quota_exceeded: QuotaExceededError,
  }

/** Translates the native layer's wire-format JSON message into typed errors. */
export function throwFromWire(err: unknown): never {
  const raw = err instanceof Error ? err.message : String(err)
  let wire: WireError
  try {
    wire = JSON.parse(raw) as WireError
  } catch {
    throw new WarpLLMError(raw)
  }

  const message = wire.message ?? raw
  switch (wire.code) {
    case 'not_implemented':
      throw new NotImplementedError(message)
    case 'invalid_request':
      throw new InvalidRequestError(message)
    case 'missing_api_key':
      throw new AuthenticationError(message, {
        provider: wire.provider,
        code: wire.code,
        origin: 'gateway',
      })
    case 'connection_error':
      throw new APIConnectionError(message, wire.provider)
    default:
      break
  }
  if (wire.origin === 'provider') {
    // Branch on the normalized code, never the raw status. The statuses lie
    // in both directions: a 403 permission failure and a 401 bad key are
    // both credential problems, while one 429 is a rate limit and another is
    // a billing failure.
    const Class = (wire.code && BY_CODE[wire.code]) || APIStatusError
    throw new Class(message, {
      code: wire.code,
      status: wire.status,
      provider: wire.provider,
      errorType: wire.error_type ?? undefined,
      retryAfterSeconds: wire.retry_after_seconds ?? undefined,
      providerCode: wire.provider_code ?? undefined,
      requestId: wire.request_id ?? undefined,
      providerRaw: wire.provider_raw,
    })
  }
  throw new WarpLLMError(message, wire.code)
}
