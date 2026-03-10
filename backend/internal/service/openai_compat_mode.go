package service

import "github.com/gin-gonic/gin"

// OpenAICompatibilityMode marks which client-facing protocol should be emitted.
type OpenAICompatibilityMode string

const (
	OpenAICompatibilityModeResponses       OpenAICompatibilityMode = ""
	OpenAICompatibilityModeChatCompletions OpenAICompatibilityMode = "chat_completions"
	OpenAICompatibilityModeCompletions     OpenAICompatibilityMode = "completions"
)

const openAICompatibilityModeContextKey = "openai_compatibility_mode"

// SetOpenAICompatibilityMode stores the client-facing compatibility mode.
func SetOpenAICompatibilityMode(c *gin.Context, mode OpenAICompatibilityMode) {
	if c == nil {
		return
	}
	c.Set(openAICompatibilityModeContextKey, string(mode))
}

// GetOpenAICompatibilityMode returns the client-facing compatibility mode.
func GetOpenAICompatibilityMode(c *gin.Context) OpenAICompatibilityMode {
	if c == nil {
		return OpenAICompatibilityModeResponses
	}
	if value, ok := c.Get(openAICompatibilityModeContextKey); ok {
		if mode, ok := value.(string); ok {
			return OpenAICompatibilityMode(mode)
		}
	}
	return OpenAICompatibilityModeResponses
}

// IsOpenAIChatCompletionsCompatibility reports whether the legacy Chat
// Completions compatibility layer is active for the current request.
func IsOpenAIChatCompletionsCompatibility(c *gin.Context) bool {
	return GetOpenAICompatibilityMode(c) == OpenAICompatibilityModeChatCompletions
}

// IsOpenAICompletionsCompatibility reports whether the legacy Completions
// compatibility layer is active for the current request.
func IsOpenAICompletionsCompatibility(c *gin.Context) bool {
	return GetOpenAICompatibilityMode(c) == OpenAICompatibilityModeCompletions
}
