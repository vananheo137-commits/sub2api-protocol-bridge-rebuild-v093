package apicompat

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestOpenAIChatCompletionsToResponses_Basic(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.1",
		"stream":true,
		"max_tokens":512,
		"messages":[
			{"role":"user","content":"hello"},
			{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"list_files","arguments":"{\"path\":\"/tmp\"}"}}]},
			{"role":"tool","tool_call_id":"call_1","content":"done"}
		],
		"tools":[{"type":"function","function":{"name":"list_files","description":"List files","parameters":{"type":"object"}}}]
	}`)

	converted, err := OpenAIChatCompletionsToResponses(body)
	require.NoError(t, err)

	var parsed map[string]any
	require.NoError(t, json.Unmarshal(converted, &parsed))
	require.Equal(t, "gpt-5.1", parsed["model"])
	require.Equal(t, true, parsed["stream"])
	require.Equal(t, false, parsed["store"])
	require.EqualValues(t, 512, parsed["max_output_tokens"])

	input, ok := parsed["input"].([]any)
	require.True(t, ok)
	require.Len(t, input, 4)

	first, ok := input[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "user", first["role"])

	second, ok := input[1].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "assistant", second["role"])

	third, ok := input[2].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "function_call", third["type"])
	require.Equal(t, "call_1", third["call_id"])

	fourth, ok := input[3].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "function_call_output", fourth["type"])
	require.Equal(t, "done", fourth["output"])

	tools, ok := parsed["tools"].([]any)
	require.True(t, ok)
	require.Len(t, tools, 1)
}

func TestResponsesToOpenAIChatCompletion_Basic(t *testing.T) {
	body := []byte(`{
		"id":"resp_123",
		"created_at":1700000000,
		"model":"gpt-5.1",
		"status":"completed",
		"output":[
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello world"}]},
			{"type":"function_call","call_id":"call_1","name":"list_files","arguments":"{}"}
		],
		"usage":{"input_tokens":11,"output_tokens":22,"total_tokens":33,"input_tokens_details":{"cached_tokens":2}}
	}`)

	converted, err := ResponsesToOpenAIChatCompletion(body)
	require.NoError(t, err)

	var parsed map[string]any
	require.NoError(t, json.Unmarshal(converted, &parsed))
	require.Equal(t, "chat.completion", parsed["object"])
	require.Equal(t, "resp_123", parsed["id"])
	require.Equal(t, "gpt-5.1", parsed["model"])

	choices, ok := parsed["choices"].([]any)
	require.True(t, ok)
	require.Len(t, choices, 1)
	choice, ok := choices[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "tool_calls", choice["finish_reason"])

	message, ok := choice["message"].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "assistant", message["role"])
	require.Equal(t, "hello world", message["content"])

	toolCalls, ok := message["tool_calls"].([]any)
	require.True(t, ok)
	require.Len(t, toolCalls, 1)

	usage, ok := parsed["usage"].(map[string]any)
	require.True(t, ok)
	require.EqualValues(t, 11, usage["prompt_tokens"])
	require.EqualValues(t, 22, usage["completion_tokens"])
}

func TestResponsesEventToOpenAIChatCompletions_DoneFallback(t *testing.T) {
	state := NewOpenAIChatCompletionsStreamState("gpt-5.1")

	payloads, done, err := ResponsesEventToOpenAIChatCompletions([]byte(`{
		"type":"response.output_text.done",
		"output_index":0,
		"content_index":0,
		"text":"hello from done"
	}`), state)
	require.NoError(t, err)
	require.False(t, done)
	require.Len(t, payloads, 1)
	require.Contains(t, string(payloads[0]), `"content":"hello from done"`)
}
