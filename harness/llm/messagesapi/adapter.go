package messagesapi

import (
	"context"
	"encoding/json/v2"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"
	"uuid"

	"github.com/unreallabsai/unreal-agent/harness/llm"
	"github.com/unreallabsai/unreal-agent/harness/primitives"
	"github.com/unreallabsai/unreal-agent/internal/anthropicapi"
)

const DefaultMaxAttempts = primitives.DefaultRemoteMaxAttempts

type CacheTTL string

const (
	CacheTTL5m CacheTTL = "5m"
	CacheTTL1h CacheTTL = "1h"
)

func (ttl CacheTTL) Valid() bool {
	return ttl == "" || anthropicapi.CacheControlEphemeralTtl(ttl).Valid()
}

type APIError struct {
	StatusCode int
	Type       string
	Message    string
	RequestID  string
}

func (err *APIError) Error() string {
	if err.Type != "" {
		return fmt.Sprintf("messages API error %s: %s", err.Type, err.Message)
	}
	return fmt.Sprintf("messages API request failed with status %d: %s", err.StatusCode, err.Message)
}

type Exchange struct {
	RequestBody  []byte
	StatusCode   int
	ResponseBody []byte
}

type Config struct {
	Endpoint string
	Headers  map[string][]string
	// Nil enables server-side fallback.
	Fallback *bool
	CacheTTL CacheTTL
	// Nil uses DefaultMaxAttempts.
	MaxAttempts *int
	// Trace borrows read-only bodies: request JSON and assembled response JSON or provider error.
	Trace func(Exchange)
}

type adapter struct {
	remote      *primitives.RemoteClient
	endpoint    string
	headers     http.Header
	maxAttempts int
	fallback    bool
	cacheTTL    CacheTTL
	trace       func(Exchange)
}

var _ llm.Adapter = (*adapter)(nil)

func NewAdapter(remote *primitives.RemoteClient, config Config) (llm.Adapter, error) {
	if remote == nil {
		return nil, errors.New("remote client must be set")
	}
	if strings.TrimSpace(config.Endpoint) == "" {
		return nil, errors.New("messages API endpoint must be set")
	}
	if !config.CacheTTL.Valid() {
		return nil, fmt.Errorf("unsupported cache TTL %q", config.CacheTTL)
	}
	maxAttempts := DefaultMaxAttempts
	if config.MaxAttempts != nil {
		maxAttempts = *config.MaxAttempts
	}
	if maxAttempts <= 0 {
		return nil, errors.New("max attempts must be positive")
	}
	fallback := true
	if config.Fallback != nil {
		fallback = *config.Fallback
	}
	headers := make(http.Header, len(config.Headers)+1)
	for name, values := range config.Headers {
		for _, value := range values {
			headers.Add(name, value)
		}
	}
	headers.Set("Accept", "text/event-stream")
	if fallback {
		headers.Add("Anthropic-Beta", "server-side-fallback-2026-07-01")
	}
	return &adapter{
		remote: remote, endpoint: config.Endpoint, headers: headers, maxAttempts: maxAttempts,
		fallback: fallback, cacheTTL: config.CacheTTL, trace: config.Trace,
	}, nil
}

func (adapter *adapter) Respond(ctx context.Context, request llm.Request, options llm.RequestOptions) (llm.Response, error) {
	body, err := requestBody(request, adapter.fallback, adapter.cacheTTL)
	if err != nil {
		return llm.Response{}, err
	}
	result := adapter.exchange(ctx, body, options)
	if adapter.trace != nil && len(result.body) != 0 {
		adapter.trace(Exchange{RequestBody: body, StatusCode: result.status, ResponseBody: result.body})
	}
	if result.err != nil {
		return llm.Response{}, result.err
	}
	return decodeResponse(result.body)
}

func (adapter *adapter) remoteRequest(body []byte) primitives.RemoteRequest {
	request := primitives.DefaultRemoteRequest("llm.messagesapi", primitives.CorrelationID(uuid.New().String()), adapter.endpoint)
	request.Method = http.MethodPost
	request.Headers = adapter.headers.Clone()
	request.Body = body
	request.ResponseIdleTimeout = 30 * time.Minute
	request.RetryPolicy.MaxAttempts = 1
	request.SSE = &primitives.RemoteSSEOptions{MaxFrameSize: 256 << 20, FrameDelimiter: primitives.SSEFrameDelimiterStrip}
	return request
}

func providerError(status int, body []byte, headers http.Header) *APIError {
	var envelope struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
		RequestID string `json:"request_id"`
	}
	result := &APIError{StatusCode: status, RequestID: headers.Get("Request-Id")}
	if err := json.Unmarshal(body, &envelope); err == nil {
		result.Type, result.Message = envelope.Error.Type, envelope.Error.Message
		if envelope.RequestID != "" {
			result.RequestID = envelope.RequestID
		}
	}
	if result.Message == "" {
		result.Message = strings.TrimSpace(string(body))
		if result.Message == "" {
			result.Message = http.StatusText(status)
		}
	}
	return result
}

func remoteFailureError(ctx context.Context, event primitives.PrimitiveEvent) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	failure, ok := event.Result.(primitives.PrimitiveFailureResult)
	if !ok {
		return errors.New("remote request failed with an invalid result")
	}
	return errors.New(failure.Error)
}
