package apicompat

import (
	"encoding/json"
	"fmt"
	"strings"
)

// OpenAICompletionsStreamState tracks state while converting Responses SSE
// events into legacy Completions chunks.
type OpenAICompletionsStreamState struct {
	ChatState *OpenAIChatCompletionsStreamState
}

// NewOpenAICompletionsStreamState returns an initialized stream state.
func NewOpenAICompletionsStreamState(model string) *OpenAICompletionsStreamState {
	return &OpenAICompletionsStreamState{
		ChatState: NewOpenAIChatCompletionsStreamState(model),
	}
}

// OpenAICompletionsToResponses converts a legacy Completions request into a
// Responses API request body by first translating it into a Chat Completions
// request, then reusing the Chat Completions -> Responses path.
func OpenAICompletionsToResponses(body []byte) ([]byte, error) {
	chatBody, err := OpenAICompletionsToChatCompletions(body)
	if err != nil {
		return nil, err
	}
	return OpenAIChatCompletionsToResponses(chatBody)
}

// OpenAICompletionsToChatCompletions converts a legacy Completions request into
// a legacy Chat Completions request body.
func OpenAICompletionsToChatCompletions(body []byte) ([]byte, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("parse completions request: %w", err)
	}

	out := map[string]any{}

	if model := strings.TrimSpace(stringFromAny(raw["model"])); model != "" {
		out["model"] = model
	}
	if stream, ok := raw["stream"].(bool); ok {
		out["stream"] = stream
	}
	if temperature, ok := float64FromAny(raw["temperature"]); ok {
		out["temperature"] = temperature
	}
	if topP, ok := float64FromAny(raw["top_p"]); ok {
		out["top_p"] = topP
	}
	if maxTokens, ok := intFromAny(raw["max_tokens"]); ok {
		out["max_tokens"] = maxTokens
	}
	if frequencyPenalty, ok := float64FromAny(raw["frequency_penalty"]); ok {
		out["frequency_penalty"] = frequencyPenalty
	}
	if presencePenalty, ok := float64FromAny(raw["presence_penalty"]); ok {
		out["presence_penalty"] = presencePenalty
	}
	if stop, ok := normalizeOpenAIStopSequences(raw["stop"]); ok {
		out["stop"] = stop
	}

	if prompt, ok := buildOpenAICompletionsInput(raw["prompt"]); ok {
		out["messages"] = []any{
			map[string]any{
				"role":    "user",
				"content": prompt,
			},
		}
	}

	return json.Marshal(out)
}

// ResponsesToOpenAICompletion converts a final Responses JSON body into a
// legacy Completions JSON body by reusing the existing Responses -> Chat
// Completions conversion and flattening the resulting assistant message.
func ResponsesToOpenAICompletion(body []byte) ([]byte, error) {
	chatBody, err := ResponsesToOpenAIChatCompletion(body)
	if err != nil {
		return nil, err
	}
	return OpenAIChatCompletionToCompletion(chatBody)
}

// OpenAIChatCompletionToCompletion converts a Chat Completions JSON body into a
// legacy Completions JSON body.
func OpenAIChatCompletionToCompletion(body []byte) ([]byte, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("parse chat completions body: %w", err)
	}

	rawChoices, _ := raw["choices"].([]any)
	choices := make([]any, 0, len(rawChoices))
	for _, item := range rawChoices {
		choice, ok := item.(map[string]any)
		if !ok {
			continue
		}
		message, _ := choice["message"].(map[string]any)
		choices = append(choices, map[string]any{
			"text":          flattenOpenAIChatContent(message["content"]),
			"index":         choiceIndexValue(choice["index"]),
			"logprobs":      nil,
			"finish_reason": choice["finish_reason"],
		})
	}
	if len(choices) == 0 {
		choices = append(choices, map[string]any{
			"text":          "",
			"index":         0,
			"logprobs":      nil,
			"finish_reason": nil,
		})
	}

	out := map[string]any{
		"id":      stringFromAny(raw["id"]),
		"object":  "text_completion",
		"created": createdAtFromMap(raw),
		"model":   stringFromAny(raw["model"]),
		"choices": choices,
	}
	if usage, ok := raw["usage"].(map[string]any); ok && len(usage) > 0 {
		out["usage"] = usage
	}

	return json.Marshal(out)
}

// ResponsesEventToOpenAICompletions converts one Responses SSE payload into
// zero or more legacy Completions chunk payloads by reusing the existing
// Responses -> Chat Completions stream conversion and flattening each chunk.
func ResponsesEventToOpenAICompletions(
	data []byte,
	state *OpenAICompletionsStreamState,
) ([][]byte, bool, error) {
	if len(data) == 0 {
		return nil, false, nil
	}

	if state == nil {
		state = NewOpenAICompletionsStreamState("")
	}
	if state.ChatState == nil {
		state.ChatState = NewOpenAIChatCompletionsStreamState("")
	}

	chatPayloads, done, err := ResponsesEventToOpenAIChatCompletions(data, state.ChatState)
	if err != nil {
		return nil, false, err
	}
	if len(chatPayloads) == 0 {
		return nil, done, nil
	}

	out := make([][]byte, 0, len(chatPayloads))
	for _, payload := range chatPayloads {
		converted, err := OpenAIChatCompletionChunkToCompletionChunk(payload)
		if err != nil {
			return nil, false, err
		}
		out = append(out, converted)
	}
	return out, done, nil
}

func buildOpenAICompletionsInput(raw any) (string, bool) {
	switch value := raw.(type) {
	case string:
		return value, true
	case []any:
		parts := make([]string, 0, len(value))
		for _, item := range value {
			switch v := item.(type) {
			case string:
				parts = append(parts, v)
			default:
				encoded, err := json.Marshal(v)
				if err != nil {
					continue
				}
				parts = append(parts, string(encoded))
			}
		}
		if len(parts) == 0 {
			return "", false
		}
		return strings.Join(parts, "\n\n"), true
	case nil:
		return "", false
	default:
		encoded, err := json.Marshal(value)
		if err != nil {
			return "", false
		}
		return string(encoded), true
	}
}

func openAICompletionFinishReason(status, incompleteReason string) string {
	switch strings.TrimSpace(status) {
	case "incomplete":
		switch strings.TrimSpace(incompleteReason) {
		case "max_output_tokens":
			return "length"
		case "content_filter":
			return "content_filter"
		default:
			return "stop"
		}
	default:
		return "stop"
	}
}

// OpenAIChatCompletionChunkToCompletionChunk converts a Chat Completions chunk
// payload into a legacy Completions chunk payload.
func OpenAIChatCompletionChunkToCompletionChunk(body []byte) ([]byte, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("parse chat completions chunk: %w", err)
	}

	rawChoices, _ := raw["choices"].([]any)
	choices := make([]any, 0, len(rawChoices))
	for _, item := range rawChoices {
		choice, ok := item.(map[string]any)
		if !ok {
			continue
		}
		delta, _ := choice["delta"].(map[string]any)
		choices = append(choices, map[string]any{
			"text":          flattenOpenAIChatContent(delta["content"]),
			"index":         choiceIndexValue(choice["index"]),
			"logprobs":      nil,
			"finish_reason": choice["finish_reason"],
		})
	}
	if len(choices) == 0 {
		choices = append(choices, map[string]any{
			"text":          "",
			"index":         0,
			"logprobs":      nil,
			"finish_reason": nil,
		})
	}

	out := map[string]any{
		"id":      stringFromAny(raw["id"]),
		"object":  "text_completion",
		"created": createdAtFromMap(raw),
		"model":   stringFromAny(raw["model"]),
		"choices": choices,
	}
	if usage, ok := raw["usage"].(map[string]any); ok && len(usage) > 0 {
		out["usage"] = usage
	}
	return json.Marshal(out)
}

func normalizeOpenAIStopSequences(raw any) (any, bool) {
	switch value := raw.(type) {
	case string:
		value = strings.TrimSpace(value)
		if value == "" {
			return nil, false
		}
		return value, true
	case []string:
		out := make([]any, 0, len(value))
		for _, item := range value {
			item = strings.TrimSpace(item)
			if item != "" {
				out = append(out, item)
			}
		}
		if len(out) == 0 {
			return nil, false
		}
		return out, true
	case []any:
		out := make([]any, 0, len(value))
		for _, item := range value {
			text := strings.TrimSpace(stringFromAny(item))
			if text != "" {
				out = append(out, text)
			}
		}
		if len(out) == 0 {
			return nil, false
		}
		return out, true
	default:
		return nil, false
	}
}

func choiceIndexValue(raw any) int {
	if index, ok := intFromAny(raw); ok {
		return index
	}
	return 0
}
