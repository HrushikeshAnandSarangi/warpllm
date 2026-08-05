// Compile-only compatibility check for the package's intentionally public
// type names. Rust may refactor internal DTO names without changing this API.
import type {
  Annotation,
  AnnotationURLCitation,
  ChatCompletionAudio,
  ChatCompletionMessage,
  ChatCompletionMessageCustomToolCall,
  ChatCompletionMessageFunctionToolCall,
  ChatCompletionMessageToolCall,
  ChatCompletionRequestMessage,
  ChatCompletionTokenLogprob,
  Choice,
  ChoiceLogprobs,
  CompletionTokensDetails,
  CompletionUsage,
  CreateChatCompletionRequest,
  CreateChatCompletionResponse,
  Error,
  ErrorBody,
  FunctionCall,
  Moderation,
  ModerationResult,
  ModerationResults,
  PromptTokensDetails,
  TopLogprob,
} from '../src-ts/index.js'

export type PublishedTypeSurface = [
  Annotation,
  AnnotationURLCitation,
  ChatCompletionAudio,
  ChatCompletionMessage,
  ChatCompletionMessageCustomToolCall,
  ChatCompletionMessageFunctionToolCall,
  ChatCompletionMessageToolCall,
  ChatCompletionRequestMessage,
  ChatCompletionTokenLogprob,
  Choice,
  ChoiceLogprobs,
  CompletionTokensDetails,
  CompletionUsage,
  CreateChatCompletionRequest,
  CreateChatCompletionResponse,
  Error,
  ErrorBody,
  FunctionCall,
  Moderation,
  ModerationResult,
  ModerationResults,
  PromptTokensDetails,
  TopLogprob,
]

// This name was historically the union, not only the function variant.
export const customToolCall: ChatCompletionMessageToolCall = {
  id: 'call-1',
  type: 'provider_custom',
  custom: { name: 'shell', input: 'pwd' },
}

// Rust's request deserializer accepts explicit null for optional values.
export const nullableRequest: CreateChatCompletionRequest = {
  model: 'openai/gpt-5.6',
  messages: [{ role: 'critic', content: 'review this' }],
  temperature: null,
}
