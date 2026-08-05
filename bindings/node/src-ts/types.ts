export interface WarpLLMOptions {
  baseUrl?: string
  /** Request timeout in seconds (default 600, matching the OpenAI SDK). */
  timeout?: number
}

// ---------------------------------------------------------------------------
// Request — field-for-field with Rust's `CreateChatCompletionRequest`, which
// is what crosses the FFI. Names are Rust's (`max_tokens`, not `maxTokens`):
// the wrapper forwards the object verbatim, so a rename here would be a
// mapping table to keep in sync for no gain.
// ---------------------------------------------------------------------------

export interface ChatCompletionRequestMessage {
  /**
   * `"system"`, `"user"`, or `"assistant"` — but deliberately not a union.
   * The gateway forwards roles it does not model (`"tool"`, `"developer"`),
   * so narrowing here would reject calls the provider accepts.
   */
  role: string
  content: string
  [key: string]: unknown
}

/**
 * The OpenAI chat-completions parameters.
 *
 * Named fields are the ones warpllm reads; everything else — `tools`,
 * `response_format`, `seed`, whatever OpenAI adds next — is forwarded to the
 * provider untouched, which is why the index signature is here rather than a
 * closed list that would reject a call the provider accepts.
 *
 * The cost is that a misspelled optional parameter is not caught here; it
 * reaches the provider, which rejects it by name.
 */
export interface CreateChatCompletionRequest {
  /** `provider/model`, e.g. `"openai/gpt-5.6"`. */
  model: string
  messages: ChatCompletionRequestMessage[]
  temperature?: number
  max_tokens?: number
  top_p?: number
  stop?: string[]
  /** Not yet implemented; `true` rejects with code `not_implemented`. */
  stream?: boolean
  [key: string]: unknown
}

// ---------------------------------------------------------------------------
// Response — field-for-field copy of the `chat.completion` object.
// Object names and field order mirror the upstream reference:
// https://developers.openai.com/api/reference/resources/chat
// ---------------------------------------------------------------------------

export interface CreateChatCompletionResponse {
  id: string
  choices: Choice[]
  created: number
  /** Echoes the caller-supplied `provider/model` string. */
  model: string
  /** Always `"chat.completion"`. */
  object: string
  moderation?: Moderation
  service_tier?: 'auto' | 'default' | 'flex' | 'scale' | 'priority'
  /** Deprecated upstream but still returned; passed through as-is. */
  system_fingerprint?: string
  usage?: CompletionUsage
}

export interface Choice {
  finish_reason: 'stop' | 'length' | 'tool_calls' | 'content_filter' | 'function_call'
  index: number
  /**
   * Optional per the docs; some OpenAI-compatible backends emit an explicit
   * `"logprobs": null`, which the gateway normalizes to absent.
   */
  logprobs?: ChoiceLogprobs
  message: ChatCompletionMessage
}

/**
 * OpenAI documents both arrays as required; OpenAI-compatible backends are
 * looser — DeepSeek omits `refusal` entirely and can null `content` — so
 * both are optional here.
 */
export interface ChoiceLogprobs {
  content?: ChatCompletionTokenLogprob[] | null
  refusal?: ChatCompletionTokenLogprob[] | null
}

export interface ChatCompletionTokenLogprob {
  token: string
  bytes: number[] | null
  logprob: number
  top_logprobs: TopLogprob[]
}

export interface TopLogprob {
  token: string
  bytes: number[] | null
  logprob: number
}

export interface ChatCompletionMessage {
  content: string | null
  refusal: string | null
  role: 'assistant'
  annotations?: Annotation[]
  audio?: ChatCompletionAudio
  /** Deprecated upstream in favor of `tool_calls`. */
  function_call?: FunctionCall
  tool_calls?: ChatCompletionMessageToolCall[]
}

export interface Annotation {
  type: 'url_citation'
  url_citation: AnnotationURLCitation
}

export interface AnnotationURLCitation {
  end_index: number
  start_index: number
  title: string
  url: string
}

/** Deprecated upstream in favor of tool calls. */
export interface FunctionCall {
  /** JSON-encoded arguments; model-generated, so may be invalid JSON. */
  arguments: string
  name: string
}

/** Union discriminated by `type`. */
export type ChatCompletionMessageToolCall =
  | ChatCompletionMessageFunctionToolCall
  | ChatCompletionMessageCustomToolCall

export interface ChatCompletionMessageFunctionToolCall {
  id: string
  type: 'function'
  function: {
    /** JSON-encoded arguments; model-generated, so may be invalid JSON. */
    arguments: string
    name: string
  }
}

export interface ChatCompletionMessageCustomToolCall {
  id: string
  type: 'custom'
  custom: {
    input: string
    name: string
  }
}

export interface ChatCompletionAudio {
  id: string
  /** Base64-encoded audio bytes. */
  data: string
  expires_at: number
  transcript: string
}

export interface CompletionUsage {
  completion_tokens: number
  prompt_tokens: number
  total_tokens: number
  completion_tokens_details?: CompletionTokensDetails
  prompt_tokens_details?: PromptTokensDetails
}

export interface CompletionTokensDetails {
  accepted_prediction_tokens?: number
  audio_tokens?: number
  reasoning_tokens?: number
  rejected_prediction_tokens?: number
}

export interface PromptTokensDetails {
  audio_tokens?: number
  /** Unadjusted number of prompt tokens written to cache. */
  cache_write_tokens?: number
  cached_tokens?: number
}

/** Moderation results for the request input and the generated output. */
export interface Moderation {
  input: ModerationResults | Error
  output: ModerationResults | Error
}

// The docs define one shared ModerationResults/Error pair used by both
// `input` and `output`; both unions are discriminated by `type`.

export interface ModerationResults {
  type: 'moderation_results'
  model: string
  results: ModerationResult[]
}

/**
 * One verdict in `ModerationResults.results`. The docs leave this element
 * object unnamed; named here after its `type` string.
 */
export interface ModerationResult {
  categories: Record<string, boolean>
  category_applied_input_types: Record<string, string[]>
  category_scores: Record<string, number>
  flagged: boolean
  model: string
  type: 'moderation_result'
}

/**
 * Moderation error. Exact upstream name; it shadows the global `Error` type
 * in this module and wherever it is imported unaliased.
 */
export interface Error {
  type: 'error'
  code: string
  message: string
}
