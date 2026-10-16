package messagesapi

import (
	"bytes"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"fmt"
	"io"
	"net/http"

	"github.com/unreallabsai/unreal-agent/harness/llm"
)

type streamingState struct {
	message  *messageBuilder
	body     []byte
	terminal bool
	apiError *APIError
	onDelta  func(llm.Delta)
}

func (state *streamingState) observe(data []byte) error {
	if state.terminal {
		return nil
	}
	if state.apiError != nil {
		return state.apiError
	}
	var event struct {
		Type         string         `json:"type"`
		Index        *int           `json:"index"`
		Message      jsontext.Value `json:"message"`
		ContentBlock jsontext.Value `json:"content_block"`
		Delta        jsontext.Value `json:"delta"`
		Usage        jsontext.Value `json:"usage"`
	}
	if err := json.Unmarshal(data, &event); err != nil {
		return fmt.Errorf("decode Messages stream event: %w", err)
	}
	switch event.Type {
	case "message_start":
		if state.message != nil {
			return fmt.Errorf("duplicate message_start")
		}
		message, err := newMessageBuilder(event.Message)
		if err != nil {
			return err
		}
		state.message = message
	case "message_delta", "message_stop":
		if state.message == nil {
			return fmt.Errorf("%s before message_start", event.Type)
		}
		if event.Type == "message_delta" {
			return state.message.update(event.Delta, event.Usage)
		}
		body, err := state.message.build()
		if err != nil {
			return err
		}
		state.body, state.terminal = body, true
		state.message = nil
	case "content_block_start", "content_block_delta", "content_block_stop":
		if state.message == nil {
			return fmt.Errorf("%s before message_start", event.Type)
		}
		if event.Index == nil || *event.Index < 0 {
			return fmt.Errorf("%s requires a nonnegative index", event.Type)
		}
		if event.Type == "content_block_start" {
			if err := state.message.startBlock(*event.Index, event.ContentBlock); err != nil {
				return err
			}
			if state.onDelta != nil {
				if block := state.message.blocks[*event.Index]; block != nil && block.kind == "tool_use" {
					state.onDelta(llm.Delta{Index: *event.Index, Kind: llm.DeltaToolCall, Name: block.toolName()})
				}
			}
			return nil
		}
		block := state.message.blocks[*event.Index]
		if block == nil {
			return fmt.Errorf("%s for nonexistent block %d", event.Type, *event.Index)
		}
		switch event.Type {
		case "content_block_delta":
			delta, err := block.handleDelta(event.Delta)
			if err != nil {
				return fmt.Errorf("content block %d: %w", *event.Index, err)
			}
			if state.onDelta != nil && (delta.Text != "" || delta.Name != "") {
				delta.Index = *event.Index
				state.onDelta(delta)
			}
		case "content_block_stop":
			block.handleStop()
		}
	case "error":
		state.body = bytes.Clone(data)
		state.apiError = providerError(http.StatusOK, data, nil)
	case "":
		return fmt.Errorf("messages stream event has no type")
	}
	return nil
}

func (state *streamingState) unwrap() ([]byte, error) {
	if state.apiError != nil {
		return state.body, state.apiError
	}
	if !state.terminal {
		return nil, fmt.Errorf("messages stream ended without message_stop: %w", io.ErrUnexpectedEOF)
	}
	return state.body, nil
}
