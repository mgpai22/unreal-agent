package messagesapi

import (
	"encoding/json/jsontext"
	"encoding/json/v2"
	"fmt"
	"maps"
	"strings"

	"github.com/unreallabsai/unreal-agent/harness/llm"
	"github.com/unreallabsai/unreal-agent/internal/apijson"
)

type blockContentBuilder struct {
	kind      string
	fields    map[string]any
	text      strings.Builder
	signature strings.Builder
	input     strings.Builder
	finalized bool
}

func newBlockContentBuilder(raw jsontext.Value) (*blockContentBuilder, error) {
	var fields map[string]any
	if err := apijson.Unmarshal(raw, &fields); err != nil {
		return nil, err
	}
	kind, _ := fields["type"].(string)
	block := &blockContentBuilder{kind: kind, fields: fields}
	switch kind {
	case "text":
		text, ok := fields["text"].(string)
		if !ok && fields["text"] != nil {
			return nil, fmt.Errorf("text must be a string")
		}
		block.text.WriteString(text)
	case "thinking":
		thinking, ok := fields["thinking"].(string)
		if !ok && fields["thinking"] != nil {
			return nil, fmt.Errorf("thinking must be a string")
		}
		signature, ok := fields["signature"].(string)
		if !ok && fields["signature"] != nil {
			return nil, fmt.Errorf("signature must be a string")
		}
		block.text.WriteString(thinking)
		block.signature.WriteString(signature)
	case "tool_use", "redacted_thinking", "fallback":
	default:
		return nil, fmt.Errorf("unsupported output block type %q", kind)
	}
	return block, nil
}

func (block *blockContentBuilder) handleDelta(raw jsontext.Value) (llm.Delta, error) {
	if block.finalized {
		return llm.Delta{}, fmt.Errorf("content block is finalized, cannot handle delta %q", raw)
	}
	var delta struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		Thinking    string `json:"thinking"`
		Signature   string `json:"signature"`
		PartialJSON string `json:"partial_json"`
	}
	if err := json.Unmarshal(raw, &delta); err != nil {
		return llm.Delta{}, err
	}
	switch {
	case delta.Type == "text_delta" && block.kind == "text":
		block.text.WriteString(delta.Text)
		return llm.Delta{Kind: llm.DeltaText, Text: delta.Text}, nil
	case delta.Type == "thinking_delta" && block.kind == "thinking":
		block.text.WriteString(delta.Thinking)
		return llm.Delta{Kind: llm.DeltaReasoning, Text: delta.Thinking}, nil
	case delta.Type == "signature_delta" && block.kind == "thinking":
		block.signature.WriteString(delta.Signature)
		return llm.Delta{}, nil
	case delta.Type == "input_json_delta" && block.kind == "tool_use":
		block.input.WriteString(delta.PartialJSON)
		return llm.Delta{Kind: llm.DeltaToolCall, Text: delta.PartialJSON}, nil
	default:
		return llm.Delta{}, fmt.Errorf("unsupported delta %q for %q block", delta.Type, block.kind)
	}
}

func (block *blockContentBuilder) toolName() string {
	name, _ := block.fields["name"].(string)
	return name
}

func (block *blockContentBuilder) handleStop() {
	block.finalized = true
}

func (block *blockContentBuilder) build() (map[string]any, error) {
	fields := maps.Clone(block.fields)
	switch block.kind {
	case "text":
		fields["text"] = block.text.String()
	case "thinking":
		fields["thinking"] = block.text.String()
		fields["signature"] = block.signature.String()
	case "tool_use":
		if block.input.Len() != 0 {
			input := jsontext.Value(block.input.String())
			if input.Kind() != jsontext.KindBeginObject || !input.IsValid() {
				return nil, fmt.Errorf("streamed tool input is not a complete JSON object")
			}
			fields["input"] = input
		}
	}
	return fields, nil
}
