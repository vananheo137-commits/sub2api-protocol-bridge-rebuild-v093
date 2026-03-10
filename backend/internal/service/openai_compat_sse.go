package service

import (
	"bufio"
	"bytes"
)

func writeOpenAICompatibilitySSEPayloads(bufferedWriter *bufio.Writer, payloads [][]byte, done bool) error {
	if bufferedWriter == nil {
		return nil
	}
	for _, payload := range payloads {
		if len(bytes.TrimSpace(payload)) == 0 {
			continue
		}
		if _, err := bufferedWriter.WriteString("data: "); err != nil {
			return err
		}
		if _, err := bufferedWriter.Write(payload); err != nil {
			return err
		}
		if _, err := bufferedWriter.WriteString("\n\n"); err != nil {
			return err
		}
	}
	if done {
		if _, err := bufferedWriter.WriteString("data: [DONE]\n\n"); err != nil {
			return err
		}
	}
	return nil
}
