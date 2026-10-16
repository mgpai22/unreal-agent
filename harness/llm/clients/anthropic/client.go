package anthropic

import (
	"errors"
	"strings"

	"github.com/unreallabsai/unreal-agent/harness/llm"
	"github.com/unreallabsai/unreal-agent/harness/llm/messagesapi"
	"github.com/unreallabsai/unreal-agent/harness/primitives"
)

const DefaultBaseURL = "https://api.anthropic.com/v1"

type Config struct {
	APIKey       string
	BaseURL      string
	Fallback     *bool
	CacheTTL     messagesapi.CacheTTL
	MaxAttempts  *int
	Trace        func(Exchange)
	ExtraHeaders map[string]string
}

type Exchange = messagesapi.Exchange

type Client struct {
	llm.Adapter
	remote *primitives.RemoteClient
}

var _ llm.Adapter = (*Client)(nil)

func NewClient(config Config) (*Client, error) {
	if config.APIKey == "" {
		return nil, errors.New("anthropic API key must be set")
	}
	baseURL := strings.TrimRight(strings.TrimSpace(config.BaseURL), "/")
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	remote := primitives.NewRemoteClient()
	headers := map[string][]string{
		"X-Api-Key":         {config.APIKey},
		"Anthropic-Version": {"2023-06-01"},
		"Content-Type":      {"application/json"},
	}
	for name, value := range config.ExtraHeaders {
		headers[name] = []string{value}
	}
	adapter, err := messagesapi.NewAdapter(remote, messagesapi.Config{
		Endpoint: baseURL + "/messages",
		Headers:  headers,
		Trace:    config.Trace, MaxAttempts: config.MaxAttempts, Fallback: config.Fallback,
		CacheTTL: config.CacheTTL,
	})
	if err != nil {
		return nil, errors.Join(err, remote.Close())
	}
	return &Client{Adapter: adapter, remote: remote}, nil
}

func (client *Client) Close() error {
	return client.remote.Close()
}
