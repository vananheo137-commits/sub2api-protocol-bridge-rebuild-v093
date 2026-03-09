package apicompat

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestOpenAICompletionsToResponses_Basic(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.1",
		"stream":true,
		"max_tokens":256,
		"temperature":0.7,
		"top_p":0.8,
		"frequency_penalty":0.2,
		"presence_penalty":0.1,
		"stop":["END","DONE"],
		"prompt":"hello from completions"
	}`)

	converted, err := OpenAICompletionsToResponses(body)
	require.NoError(t, err)

	var parsed map[string]any
	require.NoError(t, json.Unmarshal(converted, &parsed))
	require.Equal(t, "gpt-5.1", parsed["model"])
	require.Equal(t, true, parsed["stream"])
	require.Equal(t, false, parsed["store"])
	require.EqualValues(t, 256, parsed["max_output_tokens"])
	require.EqualValues(t, 0.7, parsed["temperature"])
	require.EqualValues(t, 0.8, parsed["top_p"])
	require.EqualValues(t, 0.2, parsed["frequency_penalty"])
	require.EqualValues(t, 0.1, parsed["presence_penalty"])
	require.Equal(t, []any{"END", "DONE"}, parsed["stop"])

	input, ok := parsed["input"].([]any)
	require.True(t, ok)
	require.Len(t, input, 1)

	message, ok := input[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "user", message["role"])

	content, ok := message["content"].([]any)
	require.True(t, ok)
	require.Len(t, content, 1)

	part, ok := content[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "input_text", part["type"])
	require.Equal(t, "hello from completions", part["text"])
}

func TestResponsesToOpenAICompletion_Basic(t *testing.T) {
	body := []byte(`{
		"id":"resp_456",
		"created_at":1700000000,
		"model":"gpt-5.1",
		"status":"completed",
		"output":[
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"completion text"}]}
		],
		"usage":{"input_tokens":9,"output_tokens":4,"total_tokens":13}
	}`)

	converted, err := ResponsesToOpenAICompletion(body)
	require.NoError(t, err)

	var parsed map[string]any
	require.NoError(t, json.Unmarshal(converted, &parsed))
	require.Equal(t, "text_completion", parsed["object"])
	require.Equal(t, "resp_456", parsed["id"])
	require.Equal(t, "gpt-5.1", parsed["model"])

	choices, ok := parsed["choices"].([]any)
	require.True(t, ok)
	require.Len(t, choices, 1)

	choice, ok := choices[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "completion text", choice["text"])
	require.Equal(t, "stop", choice["finish_reason"])

	usage, ok := parsed["usage"].(map[string]any)
	require.True(t, ok)
	require.EqualValues(t, 9, usage["prompt_tokens"])
	require.EqualValues(t, 4, usage["completion_tokens"])
	require.EqualValues(t, 13, usage["total_tokens"])
}

func TestResponsesEventToOpenAICompletions_DoneFallback(t *testing.T) {
	state := NewOpenAICompletionsStreamState("gpt-5.1")

	payloads, done, err := ResponsesEventToOpenAICompletions([]byte(`{
		"type":"response.output_text.done",
		"output_index":0,
		"content_index":0,
		"text":"hello from done"
	}`), state)
	require.NoError(t, err)
	require.False(t, done)
	require.Len(t, payloads, 1)
	require.Contains(t, string(payloads[0]), `"text":"hello from done"`)
}

func TestOpenAIChatCompletionChunkToCompletionChunk_Basic(t *testing.T) {
	body := []byte(`{
		"id":"chatcmpl_123",
		"object":"chat.completion.chunk",
		"created":1700000000,
		"model":"gpt-5.1",
		"choices":[
			{"index":0,"delta":{"content":"hello"},"finish_reason":null}
		]
	}`)

	converted, err := OpenAIChatCompletionChunkToCompletionChunk(body)
	require.NoError(t, err)

	var parsed map[string]any
	require.NoError(t, json.Unmarshal(converted, &parsed))
	require.Equal(t, "text_completion", parsed["object"])
	require.Equal(t, "chatcmpl_123", parsed["id"])
	require.Equal(t, "gpt-5.1", parsed["model"])

	choices, ok := parsed["choices"].([]any)
	require.True(t, ok)
	require.Len(t, choices, 1)

	choice, ok := choices[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "hello", choice["text"])
	require.EqualValues(t, 0, choice["index"])
}
