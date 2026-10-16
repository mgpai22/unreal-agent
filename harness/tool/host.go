package tool

import (
	"context"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/unreallabsai/unreal-agent/harness/llm"
	"github.com/unreallabsai/unreal-agent/harness/operation"
)

// HostTool is a model-visible tool whose calls the host executes. Each call is
// a durable remote-job operation, so a finished call is never run again on resume.
type HostTool struct {
	Name        string
	Description string
	Parameters  map[string]any
}

// HostFunc runs one tool call on the host. It is the host's gate: an error
// becomes the model-visible result text.
type HostFunc func(ctx context.Context, name string, arguments jsontext.Value) (string, error)

const (
	HostPlanType    operation.RemoteJobPlanType    = "host_tool"
	HostPlanVersion operation.RemoteJobPlanVersion = 1
)

type hostPlan struct {
	Name      string
	Arguments jsontext.Value
}

type hostRegistry struct {
	definitions []Definition
	names       map[string]struct{}
}

// NewHostRegistry exposes only the given host tools; no built-in tool is visible or resolvable.
func NewHostRegistry(tools ...HostTool) Registry {
	registry := &hostRegistry{names: make(map[string]struct{}, len(tools))}
	for _, current := range tools {
		registry.names[current.Name] = struct{}{}
		registry.definitions = append(registry.definitions, Definition{Tool: llm.Tool{
			Type: llm.ToolFunction, Name: current.Name, Description: current.Description, Parameters: current.Parameters,
		}})
	}
	return registry
}

func (registry *hostRegistry) StaticDefinitions() []Definition { return registry.definitions }

func (registry *hostRegistry) Resolve(name string) (Translator, bool) {
	if _, exists := registry.names[name]; !exists {
		return nil, false
	}
	return hostTranslator{name: name}, true
}

func (*hostRegistry) RegisterSkill(Skill) (RegistrationID, error) {
	return RegistrationID{}, errors.New("host registry does not accept skills")
}

func (*hostRegistry) UnregisterSkill(RegistrationID) {}

func (*hostRegistry) Skills() []Skill { return nil }

type hostTranslator struct {
	name string
}

func (translator hostTranslator) Translate(ctx Context, call llm.ToolCall) CallStatus {
	arguments := strings.TrimSpace(call.Arguments)
	if arguments == "" {
		arguments = "{}"
	}
	value := jsontext.Value(arguments)
	if !value.IsValid() {
		return ErrorStatus(fmt.Sprintf("%s arguments are not valid JSON", translator.name), 0)
	}
	data, err := json.Marshal(hostPlan{Name: translator.name, Arguments: value})
	if err != nil {
		return ErrorStatus(fmt.Sprintf("encode %s call: %v", translator.name, err), 0)
	}
	spec, err := operation.NewRemoteJobSpec(operation.RemoteJobPlan{Type: HostPlanType, Version: HostPlanVersion, Data: data})
	if err != nil {
		return ErrorStatus(fmt.Sprintf("build %s operation: %v", translator.name, err), 0)
	}
	return CallStatus{WaitingFor: []operation.ID{ctx.Submit(spec)}}
}

func (translator hostTranslator) TranslateResult(callID string, status CallStatus, operations []operation.Operation) (Result, error) {
	if status.Error != "" {
		return hostResult{callID: callID, text: status.Error}, nil
	}
	if len(operations) != 1 {
		return nil, fmt.Errorf("%s call %q has %d operations, want 1", translator.name, callID, len(operations))
	}
	state, err := operation.DecodeRemoteJobState(operations[0])
	if err != nil {
		return nil, err
	}
	switch operations[0].Status {
	case operation.StatusCompleted:
		return hostResult{callID: callID, text: state.TerminalResult}, nil
	case operation.StatusFailed, operation.StatusCanceled:
		return hostResult{callID: callID, text: state.TerminalError}, nil
	default:
		return hostResult{callID: callID, text: "Tool is running.", running: true}, nil
	}
}

type hostResult struct {
	callID  string
	text    string
	running bool
}

func (result hostResult) ToLLMResult() llm.ToolResult {
	return llm.ToolResult{CallID: result.callID, Running: result.running,
		Output: []llm.ToolResultOutput{{Kind: llm.ToolResultText, Value: result.text}}}
}

// HostExecutor is the operation.RemoteJobHandler that runs host tool calls.
type HostExecutor struct {
	ctx     context.Context
	run     HostFunc
	updates chan operation.Operation

	mu      sync.Mutex
	cancels map[operation.ID]context.CancelFunc
}

var _ operation.RemoteJobHandler = (*HostExecutor)(nil)

func NewHostExecutor(ctx context.Context, run HostFunc) *HostExecutor {
	return &HostExecutor{ctx: ctx, run: run, updates: make(chan operation.Operation), cancels: make(map[operation.ID]context.CancelFunc)}
}

func (*HostExecutor) RemoteJobPlanType() operation.RemoteJobPlanType        { return HostPlanType }
func (*HostExecutor) RemoteJobPlanVersion() operation.RemoteJobPlanVersion  { return HostPlanVersion }
func (executor *HostExecutor) RemoteJobUpdates() <-chan operation.Operation { return executor.updates }

func (executor *HostExecutor) AddRemoteJob(current operation.Operation) error {
	state, err := operation.DecodeRemoteJobState(current)
	if err != nil {
		return err
	}
	var plan hostPlan
	if err := json.Unmarshal(state.Plan.Data, &plan); err != nil {
		return fmt.Errorf("decode host tool plan: %w", err)
	}
	ctx, cancel := context.WithCancel(executor.ctx)
	executor.mu.Lock()
	executor.cancels[current.ID] = cancel
	executor.mu.Unlock()
	go func() {
		defer executor.forget(current.ID)
		output, runErr := executor.run(ctx, plan.Name, plan.Arguments)
		var step operation.Step
		switch {
		case ctx.Err() != nil:
			step, err = operation.CancelRemoteJob(current)
		case runErr != nil:
			step, err = operation.FailRemoteJob(current, runErr)
		default:
			state.TerminalResult = output
			step, err = operation.UpdateRemoteJob(current, state, operation.StatusCompleted)
		}
		if err != nil {
			if step, err = operation.FailRemoteJob(current, err); err != nil {
				return
			}
		}
		select {
		case executor.updates <- *step.Operation:
		case <-executor.ctx.Done():
		}
	}()
	return nil
}

func (executor *HostExecutor) CancelRemoteJob(id operation.ID, _ string) error {
	executor.mu.Lock()
	defer executor.mu.Unlock()
	if cancel, exists := executor.cancels[id]; exists {
		cancel()
	}
	return nil
}

func (executor *HostExecutor) forget(id operation.ID) {
	executor.mu.Lock()
	defer executor.mu.Unlock()
	executor.cancels[id]()
	delete(executor.cancels, id)
}
