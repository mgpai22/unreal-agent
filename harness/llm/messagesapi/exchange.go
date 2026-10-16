package messagesapi

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"mime"
	"net/http"
	"time"

	"github.com/unreallabsai/unreal-agent/harness/llm"
	"github.com/unreallabsai/unreal-agent/harness/primitives"
)

type responseAttempt struct {
	status  int
	headers http.Header
	body    []byte
	err     error
	retry   bool
}

func (adapter *adapter) exchange(ctx context.Context, body []byte, options llm.RequestOptions) responseAttempt {
	events := make(chan primitives.PrimitiveEvent)
	for attempt := 1; ; attempt++ {
		if err := ctx.Err(); err != nil {
			return responseAttempt{err: err}
		}
		request := adapter.remoteRequest(body)
		result := adapter.exchangeAttempt(ctx, request, events, options.OnDelta)
		if !result.retry || attempt >= adapter.maxAttempts {
			return result
		}
		if err := ctx.Err(); err != nil {
			result.err = err
			return result
		}
		now := time.Now()
		delay := responseRetryDelay(request.RetryPolicy, attempt, result.headers, now, rand.Float64())
		primitives.ScheduleTimer(ctx, primitives.TimerRequest{
			Source: request.Source, CorrelationID: request.CorrelationID + ":backoff", Deadline: now.Add(delay),
		}, events)
		event := <-events
		switch event.Type {
		case primitives.PrimitiveEventTimerFired:
			continue
		case primitives.PrimitiveEventCanceled:
			result.err = ctx.Err()
			if result.err == nil {
				result.err = context.Canceled
			}
		case primitives.PrimitiveEventFailed:
			result.err = remoteFailureError(ctx, event)
		default:
			result.err = fmt.Errorf("unexpected retry timer event %q", event.Type)
		}
		return result
	}
}

func (adapter *adapter) exchangeAttempt(ctx context.Context, request primitives.RemoteRequest, events chan primitives.PrimitiveEvent, onDelta func(llm.Delta)) responseAttempt {
	attemptContext, cancel := context.WithCancel(ctx)
	defer cancel()
	adapter.remote.SendRequest(attemptContext, request, events)
	var result responseAttempt
	state := streamingState{onDelta: onDelta}
	var streaming bool
	var parseErr error
	for {
		event := <-events
		var transportErr error
		switch event.Type {
		case primitives.PrimitiveEventRemoteResponseStarted:
			started := event.Result.(primitives.RemoteResponseStartedResult)
			result.status, result.headers = started.StatusCode, http.Header(started.Headers)
			contentType := result.headers.Get("Content-Type")
			mediaType, _, err := mime.ParseMediaType(contentType)
			if err != nil && contentType != "" {
				parseErr = fmt.Errorf("invalid response content type: %w", err)
			} else if result.status >= 200 && result.status < 300 && mediaType != "text/event-stream" {
				parseErr = fmt.Errorf("expected text/event-stream response, got content type %q", contentType)
			}
			if parseErr != nil {
				cancel()
			}
			streaming = result.status >= 200 && result.status < 300 && mediaType == "text/event-stream"
			continue
		case primitives.PrimitiveEventRemoteOutput:
			if parseErr != nil || state.terminal || state.apiError != nil {
				continue
			}
			output := event.Result.(primitives.RemoteOutputResult)
			if streaming {
				data := bytes.TrimSpace(primitives.SSEData(output.Data))
				if len(data) != 0 {
					parseErr = state.observe(data)
				}
				if state.terminal || state.apiError != nil || parseErr != nil {
					// Drain the primitive's terminal event before returning or retrying.
					cancel()
				}
			} else {
				const limit = 1 << 20
				if len(result.body)+len(output.Data) > limit {
					parseErr = fmt.Errorf("messages API response exceeds %d bytes", limit)
					cancel()
				} else {
					result.body = append(result.body, output.Data...)
				}
			}
			continue
		case primitives.PrimitiveEventRemoteCompleted:
		case primitives.PrimitiveEventFailed:
			transportErr = remoteFailureError(ctx, event)
		case primitives.PrimitiveEventCanceled:
			transportErr = ctx.Err()
			if transportErr == nil {
				transportErr = context.Canceled
			}
		default:
			continue
		}
		switch {
		case parseErr != nil:
			result.err = parseErr
		case state.terminal:
			result.body, result.err = state.unwrap()
		case ctx.Err() != nil:
			result.err = ctx.Err()
		case state.apiError != nil:
			result.body = state.body
			apiError := providerError(result.status, result.body, result.headers)
			result.err, result.retry = apiError, retryableResponseError(apiError, result.headers)
		case result.status != 0 && (result.status < 200 || result.status >= 300):
			apiError := providerError(result.status, result.body, result.headers)
			result.err, result.retry = apiError, retryableResponseError(apiError, result.headers)
		case transportErr != nil:
			result.err, result.retry = transportErr, event.Type == primitives.PrimitiveEventFailed
		case streaming:
			result.body, result.err = state.unwrap()
			result.retry = errors.Is(result.err, io.ErrUnexpectedEOF)
		}
		return result
	}
}
