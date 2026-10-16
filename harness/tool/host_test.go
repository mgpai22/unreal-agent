package tool

import (
	"context"
	"encoding/json/jsontext"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/unreallabsai/unreal-agent/harness/llm"
	"github.com/unreallabsai/unreal-agent/harness/operation"
)

type stubContext struct {
	next int
}

func (stub *stubContext) Submit(spec operation.Spec) operation.ID {
	stub.next++
	return operation.ID(fmt.Sprintf("op-%d", stub.next))
}

func TestHostRegistryExposesOnlyHostTools(t *testing.T) {
	registry := NewHostRegistry(
		HostTool{Name: "host_echo", Description: "Echo text back from the host.", Parameters: map[string]any{"type": "object"}},
		HostTool{Name: "host_delete", Description: "Delete a file on the host.", Parameters: map[string]any{"type": "object"}},
	)
	got := map[string]bool{}
	for _, definition := range registry.StaticDefinitions() {
		got[definition.Tool.Name] = true
	}
	if len(got) != 2 || !got["host_echo"] || !got["host_delete"] {
		t.Fatalf("definitions = %v", got)
	}
	for _, name := range []string{BashName, ViewImageName, SkillUseName} {
		if _, exists := registry.Resolve(name); exists {
			t.Fatalf("built-in %q resolves", name)
		}
	}
}

func hostOperation(t *testing.T, callID, name, args string) operation.Operation {
	t.Helper()
	registry := NewHostRegistry(HostTool{Name: name})
	translator, exists := registry.Resolve(name)
	if !exists {
		t.Fatalf("%s does not resolve", name)
	}
	status := translator.Translate(&stubContext{}, llm.ToolCall{CallID: callID, Name: name, Arguments: args})
	if status.Error != "" || len(status.WaitingFor) != 1 {
		t.Fatalf("status = %+v", status)
	}
	spec, err := operation.NewRemoteJobSpec(operation.RemoteJobPlan{
		Type: HostPlanType, Version: HostPlanVersion,
		Data: jsontext.Value(fmt.Sprintf(`{"Name":%q,"Arguments":%s}`, name, args)),
	})
	if err != nil {
		t.Fatal(err)
	}
	return operation.Operation{
		MaxOutputLength: spec.MaxOutputLength, ID: status.WaitingFor[0],
		Type: spec.Type, Version: spec.Version, Status: operation.StatusReady, State: spec.State,
	}
}

func TestHostExecutorAllowAndDeny(t *testing.T) {
	run := func(ctx context.Context, name string, arguments jsontext.Value) (string, error) {
		if name != "host_echo" {
			return "", fmt.Errorf("DENIED by host policy: %s is not allowed", name)
		}
		return "echo:ok", nil
	}
	for _, test := range []struct {
		name       string
		args       string
		wantStatus operation.Status
		wantText   string
	}{
		{"host_echo", `{"text":"ok"}`, operation.StatusCompleted, "echo:ok"},
		{"host_delete", `{"path":"/tmp/x"}`, operation.StatusFailed, "DENIED by host policy: host_delete is not allowed"},
	} {
		t.Run(test.name, func(t *testing.T) {
			executor := NewHostExecutor(context.Background(), run)
			current := hostOperation(t, "call-1", test.name, test.args)
			if err := executor.AddRemoteJob(current); err != nil {
				t.Fatal(err)
			}
			select {
			case updated := <-executor.RemoteJobUpdates():
				if updated.Status != test.wantStatus {
					t.Fatalf("status = %q, want %q", updated.Status, test.wantStatus)
				}
				registry := NewHostRegistry(HostTool{Name: test.name})
				translator, _ := registry.Resolve(test.name)
				result, err := translator.TranslateResult("call-1", CallStatus{}, []operation.Operation{updated})
				if err != nil {
					t.Fatal(err)
				}
				got := result.ToLLMResult()
				if len(got.Output) != 1 || got.Output[0].Value != test.wantText {
					t.Fatalf("result = %+v, want %q", got.Output, test.wantText)
				}
			case <-time.After(10 * time.Second):
				t.Fatal("no executor update")
			}
		})
	}
}

func TestHostExecutorCancel(t *testing.T) {
	started := make(chan struct{}, 1)
	executor := NewHostExecutor(context.Background(), func(ctx context.Context, _ string, _ jsontext.Value) (string, error) {
		started <- struct{}{}
		<-ctx.Done()
		return "", errors.New("unreachable")
	})
	current := hostOperation(t, "call-1", "host_echo", `{"text":"x"}`)
	if err := executor.AddRemoteJob(current); err != nil {
		t.Fatal(err)
	}
	<-started
	if err := executor.CancelRemoteJob(current.ID, "stop"); err != nil {
		t.Fatal(err)
	}
	select {
	case updated := <-executor.RemoteJobUpdates():
		if updated.Status != operation.StatusCanceled {
			t.Fatalf("status = %q, want canceled", updated.Status)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("no executor update")
	}
}
