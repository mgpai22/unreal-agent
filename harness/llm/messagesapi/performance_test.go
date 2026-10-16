package messagesapi

import (
	"encoding/json/jsontext"
	"fmt"
	"strings"
	"testing"

	"github.com/unreallabsai/unreal-agent/harness/llm"
)

func BenchmarkRequestBody(b *testing.B) {
	for _, kind := range []string{"tool_results", "thinking"} {
		b.Run(kind, func(b *testing.B) {
			request := validRequest()
			for index := range 32 {
				id := fmt.Sprint(index)
				if kind == "thinking" {
					request.Input = append(request.Input, llm.Item{Type: llm.ItemProvider, Data: llm.ProviderItem{Type: "thinking",
						Raw: jsontext.Value(`{"type":"thinking","thinking":"","signature":"` + strings.Repeat("x", 16<<10) + `","provider_state":{"v":2}}`),
					}})
				}
				request.Input = append(request.Input, toolCall(id), toolResult(id, strings.Repeat("x", 4<<10), false))
			}
			b.ReportAllocs()
			for b.Loop() {
				if _, err := requestBody(request, false, CacheTTL5m); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkDecodeResponse(b *testing.B) {
	body := responseBody("end_turn", `[{"type":"thinking","thinking":"","signature":"`+strings.Repeat("x", 256<<10)+`"},{"type":"text","text":"Done"}]`)
	b.ReportAllocs()
	b.SetBytes(int64(len(body)))
	for b.Loop() {
		if _, err := decodeResponse(body); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeToolResponse(b *testing.B) {
	input := `{"values":[` + strings.Repeat(`{"number":9007199254740993,"text":"value"},`, 1023) + `{"number":9007199254740993,"text":"value"}]}`
	body := responseBody("tool_use", `[{"type":"tool_use","id":"a","name":"capture","input":`+input+`}]`)
	b.ReportAllocs()
	b.SetBytes(int64(len(body)))
	for b.Loop() {
		if _, err := decodeResponse(body); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkStreamDelta(b *testing.B) {
	for _, test := range []struct{ name, kind, raw string }{
		{"text", "text", `{"type":"text_delta","text":"A short text chunk."}`},
		{"thinking", "thinking", `{"type":"thinking_delta","thinking":"A short thinking chunk."}`},
		{"signature", "thinking", `{"type":"signature_delta","signature":"A short signature chunk."}`},
		{"input_json", "tool_use", `{"type":"input_json_delta","partial_json":"{\"value\":\"alpha\"}"}`},
	} {
		b.Run(test.name, func(b *testing.B) {
			raw := jsontext.Value(test.raw)
			b.ReportAllocs()
			for b.Loop() {
				block := blockContentBuilder{kind: test.kind}
				if _, err := block.handleDelta(raw); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
