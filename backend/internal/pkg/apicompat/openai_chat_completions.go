package apicompat

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// OpenAIChatCompletionsStreamState tracks state while converting Responses SSE
// events into legacy Chat Completions chunks.
type OpenAIChatCompletionsStreamState struct {
	ResponseID string
	CreatedAt  int64
	Model      string

	AssistantRoleEmitted     bool
	NextToolCallIndex        int
	OutputIndexToToolCall    map[int]int
	ArgumentsDeltaSeen       map[int]bool
	OutputTextDeltaSeen      map[string]bool
	ToolCallAnnouncementSeen map[int]bool
	SawToolCall              bool
}

// NewOpenAIChatCompletionsStreamState returns an initialized stream state.
func NewOpenAIChatCompletionsStreamState(model string) *OpenAIChatCompletionsStreamState {
	return &OpenAIChatCompletionsStreamState{
		Model:                    strings.TrimSpace(model),
		OutputIndexToToolCall:    make(map[int]int),
		ArgumentsDeltaSeen:       make(map[int]bool),
		OutputTextDeltaSeen:      make(map[string]bool),
		ToolCallAnnouncementSeen: make(map[int]bool),
	}
}

// OpenAIChatCompletionsToResponses converts a legacy Chat Completions request
// into a Responses API request body.
func OpenAIChatCompletionsToResponses(body []byte) ([]byte, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("parse chat completions request: %w", err)
	}

	out := map[string]any{
		"store": false,
	}

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
	if frequencyPenalty, ok := float64FromAny(raw["frequency_penalty"]); ok {
		out["frequency_penalty"] = frequencyPenalty
	}
	if presencePenalty, ok := float64FromAny(raw["presence_penalty"]); ok {
		out["presence_penalty"] = presencePenalty
	}
	if maxTokens, ok := intFromAny(raw["max_tokens"]); ok {
		out["max_output_tokens"] = maxTokens
	} else if maxTokens, ok := intFromAny(raw["max_completion_tokens"]); ok {
		out["max_output_tokens"] = maxTokens
	}
	if stop, ok := normalizeOpenAIStopSequences(raw["stop"]); ok {
		out["stop"] = stop
	}
	if effort := strings.TrimSpace(stringFromAny(raw["reasoning_effort"])); effort != "" {
		out["reasoning"] = map[string]any{"effort": effort}
	}
	if parallelToolCalls, ok := raw["parallel_tool_calls"].(bool); ok {
		out["parallel_tool_calls"] = parallelToolCalls
	}

	if textSettings := buildOpenAIChatTextSettings(raw["response_format"], raw["text"]); len(textSettings) > 0 {
		out["text"] = textSettings
	}

	if tools, ok := buildOpenAIChatTools(raw["tools"]); ok {
		out["tools"] = tools
	}
	if toolChoice, ok := buildOpenAIChatToolChoice(raw["tool_choice"]); ok {
		out["tool_choice"] = toolChoice
	}

	input, err := buildOpenAIChatInput(raw["messages"])
	if err != nil {
		return nil, err
	}
	if len(input) > 0 {
		out["input"] = input
	}

	return json.Marshal(out)
}

// ResponsesToOpenAIChatCompletion converts a final Responses JSON body into a
// legacy Chat Completions JSON body.
func ResponsesToOpenAIChatCompletion(body []byte) ([]byte, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("parse responses body: %w", err)
	}

	message := map[string]any{
		"role":    "assistant",
		"content": nil,
	}

	var contentBuilder strings.Builder
	var reasoningBuilder strings.Builder
	toolCalls := make([]any, 0)

	if outputs, ok := raw["output"].([]any); ok {
		for _, item := range outputs {
			output, ok := item.(map[string]any)
			if !ok {
				continue
			}
			switch strings.TrimSpace(stringFromAny(output["type"])) {
			case "message":
				if text := extractResponsesMessageText(output["content"]); text != "" {
					contentBuilder.WriteString(text)
				}
			case "reasoning":
				if reasoning := extractResponsesReasoningText(output["summary"]); reasoning != "" {
					reasoningBuilder.WriteString(reasoning)
				}
			case "function_call":
				toolCalls = append(toolCalls, map[string]any{
					"id":   stringFromAny(output["call_id"]),
					"type": "function",
					"function": map[string]any{
						"name":      stringFromAny(output["name"]),
						"arguments": stringFromAny(output["arguments"]),
					},
				})
			}
		}
	}

	if contentBuilder.Len() > 0 {
		message["content"] = contentBuilder.String()
	}
	if reasoningBuilder.Len() > 0 {
		message["reasoning_content"] = reasoningBuilder.String()
	}
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
	}

	finishReason := openAIChatFinishReason(
		stringFromAny(raw["status"]),
		extractIncompleteReason(raw["incomplete_details"]),
		len(toolCalls) > 0,
	)

	choice := map[string]any{
		"index":                0,
		"message":              message,
		"finish_reason":        finishReason,
		"native_finish_reason": finishReason,
	}

	out := map[string]any{
		"id":      stringFromAny(raw["id"]),
		"object":  "chat.completion",
		"created": createdAtFromMap(raw),
		"model":   stringFromAny(raw["model"]),
		"choices": []any{choice},
	}

	if usage := buildOpenAIChatUsage(raw["usage"]); len(usage) > 0 {
		out["usage"] = usage
	}

	return json.Marshal(out)
}

// ResponsesEventToOpenAIChatCompletions converts one Responses SSE payload into
// zero or more Chat Completions chunk payloads. The returned boolean indicates
// whether the stream reached a terminal event and should emit [DONE].
func ResponsesEventToOpenAIChatCompletions(
	data []byte,
	state *OpenAIChatCompletionsStreamState,
) ([][]byte, bool, error) {
	if len(data) == 0 {
		return nil, false, nil
	}

	var evt ResponsesStreamEvent
	if err := json.Unmarshal(data, &evt); err != nil {
		return nil, false, fmt.Errorf("parse responses stream event: %w", err)
	}

	if state == nil {
		state = NewOpenAIChatCompletionsStreamState("")
	}
	if state.CreatedAt == 0 {
		state.CreatedAt = time.Now().Unix()
	}

	switch evt.Type {
	case "response.created":
		if evt.Response != nil {
			if state.ResponseID == "" {
				state.ResponseID = evt.Response.ID
			}
			if state.Model == "" {
				state.Model = evt.Response.Model
			}
		}
		return nil, false, nil
	case "response.output_text.delta":
		if evt.Delta == "" {
			return nil, false, nil
		}
		state.OutputTextDeltaSeen[openAIChatOutputTextKey(evt)] = true
		delta := map[string]any{
			"content": evt.Delta,
		}
		if !state.AssistantRoleEmitted {
			delta["role"] = "assistant"
			state.AssistantRoleEmitted = true
		}
		return marshalOpenAIChatChunks([]map[string]any{
			buildOpenAIChatChunk(state, delta, nil, nil),
		}, false)
	case "response.output_text.done":
		if evt.Text == "" {
			return nil, false, nil
		}
		if state.OutputTextDeltaSeen[openAIChatOutputTextKey(evt)] {
			return nil, false, nil
		}
		delta := map[string]any{
			"content": evt.Text,
		}
		if !state.AssistantRoleEmitted {
			delta["role"] = "assistant"
			state.AssistantRoleEmitted = true
		}
		return marshalOpenAIChatChunks([]map[string]any{
			buildOpenAIChatChunk(state, delta, nil, nil),
		}, false)
	case "response.reasoning_summary_text.delta":
		if evt.Delta == "" {
			return nil, false, nil
		}
		delta := map[string]any{
			"reasoning_content": evt.Delta,
		}
		if !state.AssistantRoleEmitted {
			delta["role"] = "assistant"
			state.AssistantRoleEmitted = true
		}
		return marshalOpenAIChatChunks([]map[string]any{
			buildOpenAIChatChunk(state, delta, nil, nil),
		}, false)
	case "response.reasoning_summary_text.done":
		delta := map[string]any{
			"reasoning_content": "\n\n",
		}
		if !state.AssistantRoleEmitted {
			delta["role"] = "assistant"
			state.AssistantRoleEmitted = true
		}
		return marshalOpenAIChatChunks([]map[string]any{
			buildOpenAIChatChunk(state, delta, nil, nil),
		}, false)
	case "response.output_item.added":
		if evt.Item == nil || evt.Item.Type != "function_call" {
			return nil, false, nil
		}
		idx := ensureOpenAIChatToolCallIndex(state, evt.OutputIndex)
		state.ToolCallAnnouncementSeen[evt.OutputIndex] = true
		state.SawToolCall = true
		delta := map[string]any{
			"tool_calls": []any{
				map[string]any{
					"index": idx,
					"id":    evt.Item.CallID,
					"type":  "function",
					"function": map[string]any{
						"name":      evt.Item.Name,
						"arguments": "",
					},
				},
			},
		}
		if !state.AssistantRoleEmitted {
			delta["role"] = "assistant"
			state.AssistantRoleEmitted = true
		}
		return marshalOpenAIChatChunks([]map[string]any{
			buildOpenAIChatChunk(state, delta, nil, nil),
		}, false)
	case "response.function_call_arguments.delta":
		idx := ensureOpenAIChatToolCallIndex(state, evt.OutputIndex)
		state.ArgumentsDeltaSeen[evt.OutputIndex] = true
		state.SawToolCall = true
		return marshalOpenAIChatChunks([]map[string]any{
			buildOpenAIChatChunk(state, map[string]any{
				"tool_calls": []any{
					map[string]any{
						"index": idx,
						"function": map[string]any{
							"arguments": evt.Delta,
						},
					},
				},
			}, nil, nil),
		}, false)
	case "response.function_call_arguments.done":
		if state.ArgumentsDeltaSeen[evt.OutputIndex] {
			return nil, false, nil
		}
		idx := ensureOpenAIChatToolCallIndex(state, evt.OutputIndex)
		state.SawToolCall = true
		return marshalOpenAIChatChunks([]map[string]any{
			buildOpenAIChatChunk(state, map[string]any{
				"tool_calls": []any{
					map[string]any{
						"index": idx,
						"function": map[string]any{
							"arguments": evt.Arguments,
						},
					},
				},
			}, nil, nil),
		}, false)
	case "response.output_item.done":
		if evt.Item == nil || evt.Item.Type != "function_call" {
			return nil, false, nil
		}
		if state.ToolCallAnnouncementSeen[evt.OutputIndex] {
			return nil, false, nil
		}
		idx := ensureOpenAIChatToolCallIndex(state, evt.OutputIndex)
		state.SawToolCall = true
		delta := map[string]any{
			"tool_calls": []any{
				map[string]any{
					"index": idx,
					"id":    evt.Item.CallID,
					"type":  "function",
					"function": map[string]any{
						"name":      evt.Item.Name,
						"arguments": evt.Item.Arguments,
					},
				},
			},
		}
		if !state.AssistantRoleEmitted {
			delta["role"] = "assistant"
			state.AssistantRoleEmitted = true
		}
		return marshalOpenAIChatChunks([]map[string]any{
			buildOpenAIChatChunk(state, delta, nil, nil),
		}, false)
	case "response.completed", "response.done", "response.incomplete", "response.failed":
		state.SawToolCall = state.SawToolCall || eventHasToolCalls(evt.Response)
		finishReason := openAIChatFinishReason(
			responseStatusFromEvent(evt),
			incompleteReasonFromEvent(evt.Response),
			state.SawToolCall,
		)
		return marshalOpenAIChatChunks([]map[string]any{
			buildOpenAIChatChunk(state, map[string]any{}, &finishReason, buildOpenAIChatUsageFromResponse(evt.Response)),
		}, true)
	default:
		return nil, false, nil
	}
}

func buildOpenAIChatInput(raw any) ([]any, error) {
	messages, ok := raw.([]any)
	if !ok || len(messages) == 0 {
		return nil, nil
	}

	input := make([]any, 0, len(messages))
	for _, item := range messages {
		message, ok := item.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("messages must be objects")
		}

		role := normalizeOpenAIChatRole(stringFromAny(message["role"]))
		if role == "tool" {
			toolCallID := strings.TrimSpace(stringFromAny(message["tool_call_id"]))
			if toolCallID == "" {
				return nil, fmt.Errorf("tool message missing tool_call_id")
			}
			input = append(input, map[string]any{
				"type":    "function_call_output",
				"call_id": toolCallID,
				"output":  flattenOpenAIChatContent(message["content"]),
			})
			continue
		}

		content, err := convertOpenAIChatContent(role, message["content"])
		if err != nil {
			return nil, err
		}
		toolCalls := buildOpenAIChatFunctionCalls(message["tool_calls"])

		if len(content) > 0 || (role == "assistant" && len(toolCalls) > 0) {
			input = append(input, map[string]any{
				"role":    role,
				"content": content,
			})
		}
		input = append(input, toolCalls...)
	}

	return input, nil
}

func convertOpenAIChatContent(role string, raw any) ([]any, error) {
	partType := "input_text"
	if role == "assistant" {
		partType = "output_text"
	}

	switch content := raw.(type) {
	case string:
		if content == "" {
			return nil, nil
		}
		return []any{map[string]any{"type": partType, "text": content}}, nil
	case []any:
		parts := make([]any, 0, len(content))
		for _, item := range content {
			part, ok := item.(map[string]any)
			if !ok {
				continue
			}
			switch strings.TrimSpace(stringFromAny(part["type"])) {
			case "text", "input_text", "output_text":
				text := stringFromAny(part["text"])
				if text == "" {
					continue
				}
				parts = append(parts, map[string]any{
					"type": partType,
					"text": text,
				})
			case "image_url":
				imageURL := ""
				if image, ok := part["image_url"].(map[string]any); ok {
					imageURL = stringFromAny(image["url"])
				}
				if imageURL == "" {
					imageURL = stringFromAny(part["image_url"])
				}
				if imageURL == "" {
					continue
				}
				parts = append(parts, map[string]any{
					"type":      "input_image",
					"image_url": imageURL,
				})
			case "input_image":
				next := map[string]any{"type": "input_image"}
				if imageURL := stringFromAny(part["image_url"]); imageURL != "" {
					next["image_url"] = imageURL
				}
				if fileID := stringFromAny(part["file_id"]); fileID != "" {
					next["file_id"] = fileID
				}
				if detail := stringFromAny(part["detail"]); detail != "" {
					next["detail"] = detail
				}
				if len(next) > 1 {
					parts = append(parts, next)
				}
			case "file", "input_file":
				next := map[string]any{"type": "input_file"}
				if file, ok := part["file"].(map[string]any); ok {
					if fileData := stringFromAny(file["file_data"]); fileData != "" {
						next["file_data"] = fileData
					}
					if filename := stringFromAny(file["filename"]); filename != "" {
						next["filename"] = filename
					}
				}
				if fileData := stringFromAny(part["file_data"]); fileData != "" {
					next["file_data"] = fileData
				}
				if filename := stringFromAny(part["filename"]); filename != "" {
					next["filename"] = filename
				}
				if len(next) > 1 {
					parts = append(parts, next)
				}
			}
		}
		return parts, nil
	case map[string]any:
		return convertOpenAIChatContent(role, []any{content})
	case nil:
		return nil, nil
	default:
		return nil, fmt.Errorf("unsupported message content type")
	}
}

func buildOpenAIChatFunctionCalls(raw any) []any {
	toolCalls, ok := raw.([]any)
	if !ok || len(toolCalls) == 0 {
		return nil
	}

	out := make([]any, 0, len(toolCalls))
	for _, item := range toolCalls {
		toolCall, ok := item.(map[string]any)
		if !ok {
			continue
		}
		if strings.TrimSpace(stringFromAny(toolCall["type"])) != "function" {
			continue
		}
		function, ok := toolCall["function"].(map[string]any)
		if !ok {
			continue
		}
		callID := strings.TrimSpace(stringFromAny(toolCall["id"]))
		out = append(out, map[string]any{
			"type":      "function_call",
			"call_id":   callID,
			"id":        callID,
			"name":      stringFromAny(function["name"]),
			"arguments": stringFromAny(function["arguments"]),
		})
	}
	return out
}

func buildOpenAIChatTools(raw any) ([]any, bool) {
	tools, ok := raw.([]any)
	if !ok || len(tools) == 0 {
		return nil, false
	}

	out := make([]any, 0, len(tools))
	for _, item := range tools {
		tool, ok := item.(map[string]any)
		if !ok {
			continue
		}
		toolType := strings.TrimSpace(stringFromAny(tool["type"]))
		if toolType != "function" {
			out = append(out, tool)
			continue
		}
		function, ok := tool["function"].(map[string]any)
		if !ok {
			continue
		}
		next := map[string]any{"type": "function"}
		if name := strings.TrimSpace(stringFromAny(function["name"])); name != "" {
			next["name"] = name
		}
		if description := stringFromAny(function["description"]); description != "" {
			next["description"] = description
		}
		if parameters, ok := function["parameters"]; ok && parameters != nil {
			next["parameters"] = parameters
		}
		if strict, ok := function["strict"].(bool); ok {
			next["strict"] = strict
		}
		out = append(out, next)
	}

	if len(out) == 0 {
		return nil, false
	}
	return out, true
}

func buildOpenAIChatToolChoice(raw any) (any, bool) {
	switch value := raw.(type) {
	case string:
		value = strings.TrimSpace(value)
		if value == "" {
			return nil, false
		}
		return value, true
	case map[string]any:
		choiceType := strings.TrimSpace(stringFromAny(value["type"]))
		if choiceType == "" {
			return nil, false
		}
		if choiceType != "function" {
			return value, true
		}
		function, _ := value["function"].(map[string]any)
		name := strings.TrimSpace(stringFromAny(function["name"]))
		out := map[string]any{"type": "function"}
		if name != "" {
			out["name"] = name
		}
		return out, true
	default:
		return nil, false
	}
}

func buildOpenAIChatTextSettings(responseFormatRaw, textRaw any) map[string]any {
	out := map[string]any{}

	if responseFormat, ok := responseFormatRaw.(map[string]any); ok {
		switch strings.TrimSpace(stringFromAny(responseFormat["type"])) {
		case "text":
			out["format"] = map[string]any{"type": "text"}
		case "json_schema":
			if schema, ok := responseFormat["json_schema"].(map[string]any); ok {
				format := map[string]any{"type": "json_schema"}
				if name := stringFromAny(schema["name"]); name != "" {
					format["name"] = name
				}
				if strict, ok := schema["strict"].(bool); ok {
					format["strict"] = strict
				}
				if value, ok := schema["schema"]; ok && value != nil {
					format["schema"] = value
				}
				out["format"] = format
			}
		}
	}

	if text, ok := textRaw.(map[string]any); ok {
		if verbosity := strings.TrimSpace(stringFromAny(text["verbosity"])); verbosity != "" {
			out["verbosity"] = verbosity
		}
	}

	return out
}

func flattenOpenAIChatContent(raw any) string {
	switch value := raw.(type) {
	case string:
		return value
	case []any:
		parts := make([]string, 0, len(value))
		for _, item := range value {
			part, ok := item.(map[string]any)
			if !ok {
				continue
			}
			switch strings.TrimSpace(stringFromAny(part["type"])) {
			case "text", "input_text", "output_text":
				if text := stringFromAny(part["text"]); text != "" {
					parts = append(parts, text)
				}
			}
		}
		if len(parts) > 0 {
			return strings.Join(parts, "\n\n")
		}
	case map[string]any:
		if text := stringFromAny(value["text"]); text != "" {
			return text
		}
	}

	if raw == nil {
		return ""
	}
	encoded, err := json.Marshal(raw)
	if err != nil {
		return ""
	}
	return string(encoded)
}

func extractResponsesMessageText(raw any) string {
	content, ok := raw.([]any)
	if !ok || len(content) == 0 {
		return ""
	}

	var builder strings.Builder
	for _, item := range content {
		part, ok := item.(map[string]any)
		if !ok {
			continue
		}
		if strings.TrimSpace(stringFromAny(part["type"])) != "output_text" {
			continue
		}
		builder.WriteString(stringFromAny(part["text"]))
	}
	return builder.String()
}

func extractResponsesReasoningText(raw any) string {
	summary, ok := raw.([]any)
	if !ok || len(summary) == 0 {
		return ""
	}

	var builder strings.Builder
	for _, item := range summary {
		part, ok := item.(map[string]any)
		if !ok {
			continue
		}
		if strings.TrimSpace(stringFromAny(part["type"])) != "summary_text" {
			continue
		}
		builder.WriteString(stringFromAny(part["text"]))
	}
	return builder.String()
}

func buildOpenAIChatUsage(raw any) map[string]any {
	usage, ok := raw.(map[string]any)
	if !ok || len(usage) == 0 {
		return nil
	}

	out := map[string]any{}
	if promptTokens, ok := intFromAny(usage["input_tokens"]); ok {
		out["prompt_tokens"] = promptTokens
	}
	if completionTokens, ok := intFromAny(usage["output_tokens"]); ok {
		out["completion_tokens"] = completionTokens
	}
	if totalTokens, ok := intFromAny(usage["total_tokens"]); ok {
		out["total_tokens"] = totalTokens
	}
	if inputDetails, ok := usage["input_tokens_details"].(map[string]any); ok {
		if cachedTokens, ok := intFromAny(inputDetails["cached_tokens"]); ok {
			out["prompt_tokens_details"] = map[string]any{"cached_tokens": cachedTokens}
		}
	}
	if outputDetails, ok := usage["output_tokens_details"].(map[string]any); ok {
		if reasoningTokens, ok := intFromAny(outputDetails["reasoning_tokens"]); ok {
			out["completion_tokens_details"] = map[string]any{"reasoning_tokens": reasoningTokens}
		}
	}
	if _, ok := out["total_tokens"]; !ok {
		promptTokens, _ := intFromAny(usage["input_tokens"])
		completionTokens, _ := intFromAny(usage["output_tokens"])
		out["total_tokens"] = promptTokens + completionTokens
	}
	return out
}

func buildOpenAIChatUsageFromResponse(resp *ResponsesResponse) map[string]any {
	if resp == nil || resp.Usage == nil {
		return nil
	}

	out := map[string]any{
		"prompt_tokens":     resp.Usage.InputTokens,
		"completion_tokens": resp.Usage.OutputTokens,
		"total_tokens":      resp.Usage.TotalTokens,
	}
	if resp.Usage.InputTokensDetails != nil {
		out["prompt_tokens_details"] = map[string]any{
			"cached_tokens": resp.Usage.InputTokensDetails.CachedTokens,
		}
	}
	if resp.Usage.OutputTokensDetails != nil {
		out["completion_tokens_details"] = map[string]any{
			"reasoning_tokens": resp.Usage.OutputTokensDetails.ReasoningTokens,
		}
	}
	return out
}

func buildOpenAIChatChunk(
	state *OpenAIChatCompletionsStreamState,
	delta map[string]any,
	finishReason *string,
	usage map[string]any,
) map[string]any {
	choice := map[string]any{
		"index":         0,
		"delta":         delta,
		"finish_reason": nil,
	}
	if finishReason != nil {
		choice["finish_reason"] = *finishReason
		choice["native_finish_reason"] = *finishReason
	}

	chunk := map[string]any{
		"id":      openAIChatResponseID(state),
		"object":  "chat.completion.chunk",
		"created": openAIChatCreatedAt(state),
		"model":   openAIChatModel(state),
		"choices": []any{choice},
	}
	if len(usage) > 0 {
		chunk["usage"] = usage
	}
	return chunk
}

func marshalOpenAIChatChunks(chunks []map[string]any, done bool) ([][]byte, bool, error) {
	if len(chunks) == 0 {
		return nil, done, nil
	}

	out := make([][]byte, 0, len(chunks))
	for _, chunk := range chunks {
		payload, err := json.Marshal(chunk)
		if err != nil {
			return nil, false, err
		}
		out = append(out, payload)
	}
	return out, done, nil
}

func ensureOpenAIChatToolCallIndex(state *OpenAIChatCompletionsStreamState, outputIndex int) int {
	if idx, ok := state.OutputIndexToToolCall[outputIndex]; ok {
		return idx
	}
	idx := state.NextToolCallIndex
	state.NextToolCallIndex++
	state.OutputIndexToToolCall[outputIndex] = idx
	return idx
}

func openAIChatOutputTextKey(evt ResponsesStreamEvent) string {
	return fmt.Sprintf("%d:%d", evt.OutputIndex, evt.ContentIndex)
}

func responseStatusFromEvent(evt ResponsesStreamEvent) string {
	if evt.Response != nil && evt.Response.Status != "" {
		return evt.Response.Status
	}
	switch evt.Type {
	case "response.incomplete":
		return "incomplete"
	case "response.failed":
		return "failed"
	default:
		return "completed"
	}
}

func incompleteReasonFromEvent(resp *ResponsesResponse) string {
	if resp == nil || resp.IncompleteDetails == nil {
		return ""
	}
	return strings.TrimSpace(resp.IncompleteDetails.Reason)
}

func eventHasToolCalls(resp *ResponsesResponse) bool {
	if resp == nil {
		return false
	}
	for _, item := range resp.Output {
		if item.Type == "function_call" {
			return true
		}
	}
	return false
}

func openAIChatFinishReason(status, incompleteReason string, hasToolCalls bool) string {
	switch strings.TrimSpace(status) {
	case "completed":
		if hasToolCalls {
			return "tool_calls"
		}
		return "stop"
	case "incomplete":
		switch strings.TrimSpace(incompleteReason) {
		case "max_output_tokens":
			return "length"
		case "content_filter":
			return "content_filter"
		default:
			return "stop"
		}
	case "failed":
		return "stop"
	default:
		return "stop"
	}
}

func extractIncompleteReason(raw any) string {
	details, ok := raw.(map[string]any)
	if !ok {
		return ""
	}
	return strings.TrimSpace(stringFromAny(details["reason"]))
}

func createdAtFromMap(raw map[string]any) int64 {
	if createdAt, ok := int64FromAny(raw["created_at"]); ok && createdAt > 0 {
		return createdAt
	}
	return time.Now().Unix()
}

func openAIChatResponseID(state *OpenAIChatCompletionsStreamState) string {
	if state == nil {
		return ""
	}
	return state.ResponseID
}

func openAIChatCreatedAt(state *OpenAIChatCompletionsStreamState) int64 {
	if state != nil && state.CreatedAt > 0 {
		return state.CreatedAt
	}
	return time.Now().Unix()
}

func openAIChatModel(state *OpenAIChatCompletionsStreamState) string {
	if state == nil {
		return ""
	}
	return state.Model
}

func normalizeOpenAIChatRole(role string) string {
	role = strings.TrimSpace(role)
	switch role {
	case "assistant", "system", "developer", "tool":
		return role
	case "":
		return "user"
	default:
		return role
	}
}

func stringFromAny(value any) string {
	switch v := value.(type) {
	case string:
		return v
	default:
		return ""
	}
}

func float64FromAny(value any) (float64, bool) {
	switch v := value.(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int32:
		return float64(v), true
	case int64:
		return float64(v), true
	default:
		return 0, false
	}
}

func intFromAny(value any) (int, bool) {
	switch v := value.(type) {
	case float64:
		return int(v), true
	case float32:
		return int(v), true
	case int:
		return v, true
	case int32:
		return int(v), true
	case int64:
		return int(v), true
	default:
		return 0, false
	}
}

func int64FromAny(value any) (int64, bool) {
	switch v := value.(type) {
	case float64:
		return int64(v), true
	case float32:
		return int64(v), true
	case int:
		return int64(v), true
	case int32:
		return int64(v), true
	case int64:
		return v, true
	default:
		return 0, false
	}
}
