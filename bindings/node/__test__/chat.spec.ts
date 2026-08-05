import { afterEach, beforeEach, expect, test } from 'vitest'

import { WarpLLM, WarpLLMError } from '../dist/index.js'
import { MockServer } from './mock-server.js'

const MESSAGES = [{ role: 'user', content: 'hi' }]

const request = (model = 'openai/gpt-5.6', extra: Record<string, unknown> = {}) => ({
  model,
  messages: MESSAGES,
  ...extra,
})

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

/** Every failure is one class now, so every assertion reads `code`. */
const failure = async (req: Parameters<WarpLLM['chatCompletion']>[0]) => {
  const err = await client.chatCompletion(req).catch((e: unknown) => e)
  expect(err).toBeInstanceOf(WarpLLMError)
  return err as WarpLLMError
}

test('openai happy path', async () => {
  server.respondWith(200, OPENAI_COMPLETION)

  const completion = await client.chatCompletion(request())

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

// The request is forwarded verbatim rather than rebuilt field by field, so a
// parameter the wrapper does not model still reaches the provider. The old
// wrapper copied a fixed list of keys and silently dropped the rest.
test('a request field the wrapper does not model still goes upstream', async () => {
  server.respondWith(200, OPENAI_COMPLETION)

  // No cast: an unmodelled OpenAI parameter has to type-check, or the
  // wrapper does not accept an OpenAI-compatible request.
  await client.chatCompletion({ ...request(), max_tokens: 64, seed: 7 })

  expect(server.requests[0].body).toMatchObject({ max_tokens: 64, seed: 7 })
})

test('401 reports authentication', async () => {
  server.respondWith(401, {
    error: {
      message: 'Incorrect API key provided',
      type: 'invalid_request_error',
      code: 'invalid_api_key',
    },
  })

  const err = await failure(request())

  expect(err.status).toBe(401)
  expect(err.message).toContain('Incorrect API key')
  // The provider's own slug reaches the caller, not warpllm's spelling of
  // it — warpllm would have called this one `authentication`.
  expect(err.code).toBe('invalid_api_key')
  expect(err.type).toBe('invalid_request_error')
})

// A quota exhaustion arrives as a 429 and reads exactly like a rate limit,
// but no amount of backing off buys credit. A retry loop keyed on
// `code === 'rate_limited'` must not fire here — that is how a billing
// failure becomes an infinite retry loop.
test('quota exhaustion is not reported as a rate limit', async () => {
  server.respondWith(429, {
    error: {
      message: 'You exceeded your current quota',
      type: 'invalid_request_error',
      code: 'insufficient_quota',
    },
  })

  const err = await failure(request())

  expect(err.code).toBe('insufficient_quota')
  expect(err.code).not.toBe('rate_limit_exceeded')
  expect(err.status).toBe(429)
})

test('a rate limit carries the provider’s request id', async () => {
  server.respondWith(
    429,
    { error: { message: 'Rate limit reached', type: 'rate_limit_error' } },
    { 'retry-after': '30', 'x-request-id': 'req-abc' },
  )

  const err = await failure(request())

  expect(err.type).toBe('rate_limit_error')
  // Lives only in a header, so it proves the transport kept it.
  expect(err.requestID).toBe('req-abc')
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

  expect((await failure(request())).code).toBe('context_length_exceeded')
})

// The two halves of one flat code space. A provider rejecting the request
// and warpllm rejecting it read almost alike, and the remedy is not the
// same — one edits the payload, the other may just need a different model.
test('code separates the provider’s rejection from warpllm’s', async () => {
  server.respondWith(400, {
    error: { message: 'bad payload', type: 'invalid_request_error' },
  })

  const upstream = await failure(request())
  // ...and warpllm's own rejection never left the process.
  const local = await failure(request('mistral/large'))

  expect(upstream.type).toBe('invalid_request_error')
  expect(local.type).toBe('invalid_request_error')
  expect(upstream.code).toBe('provider_invalid_request')
  expect(local.code).toBe('invalid_request')
})

test('invalid model rejects unsupported provider', async () => {
  expect((await failure(request('mistral/large'))).message).toContain(
    'no registered model spec',
  )
})

test('bare model name is rejected', async () => {
  expect((await failure(request('gpt-5.6'))).message).toContain('no registered model spec')
})

test('stream: true reports not implemented', async () => {
  const err = await failure(request('openai/gpt-5.6', { stream: true }))

  expect(err.code).toBe('not_implemented')
  expect(server.requests).toHaveLength(0)
})
