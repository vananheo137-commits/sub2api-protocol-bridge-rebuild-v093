package service

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNormalizeOpenAICompatibilityRequestBody_DoesNotRewriteRequestedModel(t *testing.T) {
	body := []byte(`{"model":"g5-codex","stream":true,"prompt":"hello"}`)

	normalized, err := NormalizeOpenAICompatibilityRequestBody(body)
	require.NoError(t, err)
	require.JSONEq(t, string(body), string(normalized))
}

func TestNormalizeOpenAICompatibilityRequestBody_LeavesUnknownModelUntouched(t *testing.T) {
	body := []byte(`{"model":"custom-non-openai-model","stream":true}`)

	normalized, err := NormalizeOpenAICompatibilityRequestBody(body)
	require.NoError(t, err)
	require.JSONEq(t, string(body), string(normalized))
}
