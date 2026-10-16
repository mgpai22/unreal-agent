package llm

import "context"

type DeltaKind string

const (
	DeltaText      DeltaKind = "text"
	DeltaReasoning DeltaKind = "reasoning"
	// DeltaToolCall carries the tool name on the block's first delta and argument JSON fragments after it.
	DeltaToolCall DeltaKind = "tool_call"
)

// Delta is one streamed fragment of the response; Index is the provider's output block index.
type Delta struct {
	Index int
	Kind  DeltaKind
	Name  string
	Text  string
}

type RequestOptions struct {
	CacheKey string
	// OnDelta, when set, receives fragments as they stream. A retried attempt streams again from the start.
	OnDelta func(Delta)
}

type Adapter interface {
	Respond(context.Context, Request, RequestOptions) (Response, error)
}
