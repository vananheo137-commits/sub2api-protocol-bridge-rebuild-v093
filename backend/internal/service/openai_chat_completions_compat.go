package service

import (
	"bufio"
	"fmt"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
)

func (s *OpenAIGatewayService) convertResponsesBodyToChatCompletions(body []byte) ([]byte, error) {
	converted, err := apicompat.ResponsesToOpenAIChatCompletion(body)
	if err != nil {
		return nil, fmt.Errorf("convert responses body to chat completions: %w", err)
	}
	return converted, nil
}

func (s *OpenAIGatewayService) convertOpenAIResponsesEventToChatCompletions(
	state *apicompat.OpenAIChatCompletionsStreamState,
	data []byte,
) ([][]byte, bool, error) {
	converted, done, err := apicompat.ResponsesEventToOpenAIChatCompletions(data, state)
	if err != nil {
		return nil, false, fmt.Errorf("convert responses event to chat completions: %w", err)
	}
	return converted, done, nil
}

func writeOpenAIChatCompletionsSSEPayloads(bufferedWriter *bufio.Writer, payloads [][]byte, done bool) error {
	return writeOpenAICompatibilitySSEPayloads(bufferedWriter, payloads, done)
}
