package service

import (
	"bufio"
	"fmt"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
)

func (s *OpenAIGatewayService) convertResponsesBodyToCompletions(body []byte) ([]byte, error) {
	converted, err := apicompat.ResponsesToOpenAICompletion(body)
	if err != nil {
		return nil, fmt.Errorf("convert responses body to completions: %w", err)
	}
	return converted, nil
}

func (s *OpenAIGatewayService) convertOpenAIResponsesEventToCompletions(
	state *apicompat.OpenAICompletionsStreamState,
	data []byte,
) ([][]byte, bool, error) {
	converted, done, err := apicompat.ResponsesEventToOpenAICompletions(data, state)
	if err != nil {
		return nil, false, fmt.Errorf("convert responses event to completions: %w", err)
	}
	return converted, done, nil
}

func writeOpenAICompletionsSSEPayloads(bufferedWriter *bufio.Writer, payloads [][]byte, done bool) error {
	return writeOpenAICompatibilitySSEPayloads(bufferedWriter, payloads, done)
}
