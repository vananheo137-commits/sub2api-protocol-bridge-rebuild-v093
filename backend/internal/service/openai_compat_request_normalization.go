package service

import (
	"fmt"

	"github.com/tidwall/gjson"
)

// NormalizeOpenAICompatibilityRequestBody only validates compatibility JSON.
// Model rewriting is intentionally disabled here so requests keep the user-
// supplied model unless an explicit account/group model mapping applies later.
func NormalizeOpenAICompatibilityRequestBody(body []byte) ([]byte, error) {
	if len(body) == 0 {
		return body, nil
	}
	if !gjson.ValidBytes(body) {
		return nil, fmt.Errorf("invalid json body")
	}
	return body, nil
}
