// Stable package facade. The declarations come from Rust; this list decides
// which wire-shape names are part of the npm package's public API.
export type {
  Annotation,
  AnnotationURLCitation,
  ChatCompletionAudio,
  ChatCompletionMessageCustomToolCall,
  ChatCompletionRequestMessage,
  ChatCompletionTokenLogprob,
  Choice,
  ChoiceLogprobs,
  CompletionTokensDetails,
  CompletionUsage,
  CreateChatCompletionRequest,
  CreateChatCompletionResponse,
  ErrorBody,
  FunctionCall,
  PromptTokensDetails,
  TopLogprob,
} from './generated/types.js'

export type {
  ChatCompletionModeration as Moderation,
  ChatCompletionModerationError as Error,
  ChatCompletionModerationResults as ModerationResults,
  ChatCompletionMessageToolCall as ChatCompletionMessageFunctionToolCall,
  ChatCompletionMessageToolCallUnion as ChatCompletionMessageToolCall,
  ChatCompletionResponseMessage as ChatCompletionMessage,
  ModerationResultBody as ModerationResult,
} from './generated/types.js'
