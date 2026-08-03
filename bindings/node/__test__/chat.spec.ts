import { afterEach, beforeEach, expect, test } from 'vitest'

import {
  APIStatusError,
  AuthenticationError,
  InvalidRequestError,
  NotImplementedError,
  QuotaExceededError,
  RateLimitError,
  WarpLLM,
} from '../dist/index.js'
import { MockServer } from './mock-server.js'

const MESSAGES = [{ role: 'user' as const, content: 'hi' }]

const OPENAI_COMPLETION = {
  id: 'chatcmpl-123',
  object: 'chat.completion',
  created: 1_700_000_000,
  model: 'gpt-5.6-2024-08-06',
  choices: [
    {
      index: 0,
      message: { role: 'assistant', content: 'Hello there!' },
      finish_reason: 'stop',
    },
  ],
  usage: {
    prompt_tokens: 9,
    completion_tokens: 12,
    total_tokens: 21,
    prompt_tokens_details: { cached_tokens: 3, cache_write_tokens: 2, audio_tokens: 0 },
    completion_tokens_details: {
      reasoning_tokens: 5,
      audio_tokens: 0,
      accepted_prediction_tokens: 0,
      rejected_prediction_tokens: 0,
    },
  },
  service_tier: 'default',
  system_fingerprint: 'fp_44709d6fcb',
}

let server: MockServer
let client: WarpLLM

beforeEach(async () => {
  server = await MockServer.start()
  // The native client reads OPENAI_API_KEY at construction, so set it first.
  process.env.OPENAI_API_KEY = 'sk-test-openai'
  client = new WarpLLM({ baseUrl: server.url, timeout: 5 })
})

afterEach(async () => {
  await server.close()
})

test('openai happy path', async () => {
  server.respondWith(200, OPENAI_COMPLETION)

  const completion = await client.chat.completions.create({
    model: 'openai/gpt-5.6',
    messages: MESSAGES,
  })

  expect(completion.choices[0].message.content).toBe('Hello there!')
  expect(completion.choices[0].finish_reason).toBe('stop')
  expect(completion.model).toBe('openai/gpt-5.6')
  expect(completion.usage?.total_tokens).toBe(21)
  expect(completion.service_tier).toBe('default')
  expect(completion.system_fingerprint).toBe('fp_44709d6fcb')
  expect(completion.usage?.prompt_tokens_details?.cached_tokens).toBe(3)
  expect(completion.usage?.prompt_tokens_details?.cache_write_tokens).toBe(2)
  expect(completion.usage?.completion_tokens_details?.reasoning_tokens).toBe(5)

  const sent = server.requests[0]
  expect(sent.url).toBe('/chat/completions')
  expect(sent.headers.authorization).toBe('Bearer sk-test-openai')
  // Provider prefix stripped from the outbound model.
  expect((sent.body as { model: string }).model).toBe('gpt-5.6')
})

test('401 rejects with AuthenticationError', async () => {
  server.respondWith(401, {
    error: { message: 'Incorrect API key provided', type: 'invalid_request_error' },
  })

  const err = await client.chat.completions
    .create({ model: 'openai/gpt-5.6', messages: MESSAGES })
    .catch((e: unknown) => e)

  expect(err).toBeInstanceOf(AuthenticationError)
  expect((err as AuthenticationError).status).toBe(401)
  expect((err as AuthenticationError).provider).toBe('openai')
  expect((err as AuthenticationError).message).toContain('Incorrect API key')
  // The normalized taxonomy, and the provider's own spelling beside it.
  expect((err as AuthenticationError).code).toBe('authentication')
  expect((err as AuthenticationError).errorType).toBe('invalid_request_error')
})

// A quota exhaustion arrives as a 429 and reads exactly like a rate limit,
// but no amount of backing off buys credit. `instanceof RateLimitError` must
// NOT match it — that is how a billing failure becomes an infinite retry
// loop.
test('quota exhaustion does not reject with RateLimitError', async () => {
  server.respondWith(429, {
    error: {
      message: 'You exceeded your current quota',
      type: 'invalid_request_error',
      code: 'insufficient_quota',
    },
  })

  const err = (await client.chat.completions
    .create({ model: 'openai/gpt-5.6', messages: MESSAGES })
    .catch((e: unknown) => e)) as QuotaExceededError

  expect(err).toBeInstanceOf(QuotaExceededError)
  expect(err).not.toBeInstanceOf(RateLimitError)
  expect(err.status).toBe(429)
  expect(err.code).toBe('quota_exceeded')
  // The evidence the classification was read from survives.
  expect(err.providerCode).toBe('insufficient_quota')
})

test('a rate limit carries the provider’s retry-after and request id', async () => {
  server.respondWith(
    429,
    { error: { message: 'Rate limit reached', type: 'rate_limit_error' } },
    { 'retry-after': '30', 'x-request-id': 'req-abc' },
  )

  const err = (await client.chat.completions
    .create({ model: 'openai/gpt-5.6', messages: MESSAGES })
    .catch((e: unknown) => e)) as RateLimitError

  expect(err).toBeInstanceOf(RateLimitError)
  expect(err.code).toBe('rate_limited')
  // Both live only in headers, so they prove the transport kept them.
  expect(err.retryAfterSeconds).toBe(30)
  expect(err.requestId).toBe('req-abc')
})

// A context overflow must not read as a plain bad request: the remedy is a
// shorter prompt or a bigger model, not a corrected payload.
test('context overflow is classified', async () => {
  server.respondWith(400, {
    error: {
      message: 'maximum context length is 8192 tokens',
      type: 'invalid_request_error',
      code: 'context_length_exceeded',
    },
  })

  const err = await client.chat.completions
    .create({ model: 'openai/gpt-5.6', messages: MESSAGES })
    .catch((e: unknown) => e)

  expect(err).toBeInstanceOf(APIStatusError)
  expect((err as APIStatusError).code).toBe('context_length_exceeded')
})

// The two halves of one flat code space. A provider rejecting the request
// and warpllm rejecting it read almost alike, and the remedy is not the
// same — one edits the payload, the other may just need a different model.
test('origin separates the provider’s failures from warpllm’s', async () => {
  server.respondWith(400, {
    error: { message: 'bad payload', type: 'invalid_request_error' },
  })

  const upstream = (await client.chat.completions
    .create({ model: 'openai/gpt-5.6', messages: MESSAGES })
    .catch((e: unknown) => e)) as APIStatusError
  expect(upstream.origin).toBe('provider')
  expect(upstream.code).toBe('provider_invalid_request')

  // ...and warpllm's own rejection never left the process.
  const local = (await client.chat.completions
    .create({ model: 'mistral/large', messages: MESSAGES })
    .catch((e: unknown) => e)) as InvalidRequestError
  expect(local.origin).toBe('gateway')
  expect(local.code).toBe('invalid_request')
})

test('invalid model rejects unsupported provider', async () => {
  const err = await client.chat.completions
    .create({ model: 'mistral/large', messages: MESSAGES })
    .catch((e: unknown) => e)

  expect(err).toBeInstanceOf(InvalidRequestError)
  expect((err as InvalidRequestError).message).toContain('no registered model spec')
})

test('bare model name is rejected', async () => {
  const err = await client.chat.completions
    .create({ model: 'gpt-5.6', messages: MESSAGES })
    .catch((e: unknown) => e)

  expect(err).toBeInstanceOf(InvalidRequestError)
  expect((err as InvalidRequestError).message).toContain('no registered model spec')
})

test('stream: true rejects as not implemented', async () => {
  const err = await client.chat.completions
    .create({ model: 'openai/gpt-5.6', messages: MESSAGES, stream: true })
    .catch((e: unknown) => e)

  expect(err).toBeInstanceOf(NotImplementedError)
  expect(server.requests).toHaveLength(0)
})
