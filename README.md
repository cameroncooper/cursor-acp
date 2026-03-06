# cursor-acp

ACP adapter that lets [Zed](https://zed.dev) (and other ACP clients) use the [Cursor](https://cursor.com) agent.

`cursor-acp` speaks the [Agent Client Protocol (ACP)](https://agentclientprotocol.com) over stdio and delegates to the Cursor headless CLI (`cursor-agent`).

## Features

- Full ACP agent lifecycle: initialize, authenticate, sessions (create / load / list / resume), prompt
- Real-time streaming of assistant messages, thinking, tool calls, diffs, shell output, and plans
- Mode enforcement: **Ask** and **Plan** modes are sandboxed read-only; **Agent** mode allows edits
- Managed edits: file changes routed through the client for native accept/reject review UI
- Session persistence with full event history replay
- Token usage reporting (`UsageUpdate`) and live session title updates (`SessionInfoUpdate`)
- Typed error handling for auth, unsupported models, workspace trust, and CLI failures

## Requirements

- [Cursor](https://cursor.com) installed (provides `cursor-agent`)
- Rust stable (if building from source)

## Install

### Option A: Download a release binary

Download the archive for your platform from [GitHub Releases](https://github.com/cameroncooper/cursor-acp/releases), extract it, and place `cursor-acp` on your `PATH`.

### Option B: Build from source

```bash
cargo install --git https://github.com/cameroncooper/cursor-acp --locked
```

Or clone and build locally:

```bash
git clone https://github.com/cameroncooper/cursor-acp.git
cd cursor-acp
cargo install --path . --locked
```

## Use with Zed

### 1. Ensure `cursor-agent` is available

`cursor-acp` launches `cursor-agent`. Either:

- have `cursor-agent` on your `PATH`, or
- set `CURSOR_AGENT_BIN` to the full path in the Zed config below.

### 2. Add a custom agent in Zed settings

Open Zed settings (`cmd+,` → JSON) and add:

```json
{
  "agent_servers": {
    "cursor-acp": {
      "type": "custom",
      "command": "cursor-acp",
      "args": [],
      "env": {
        "RUST_LOG": "info"
      }
    }
  }
}
```

If `cursor-agent` is not on `PATH`:

```json
{
  "agent_servers": {
    "cursor-acp": {
      "type": "custom",
      "command": "cursor-acp",
      "args": [],
      "env": {
        "CURSOR_AGENT_BIN": "/absolute/path/to/cursor-agent",
        "RUST_LOG": "info"
      }
    }
  }
}
```

### 3. Start a thread

1. Open the Agent panel in Zed.
2. Click **+** and select **cursor-acp**.
3. If prompted, authenticate by running `cursor-agent login` in your terminal.

### Optional: keybinding

```json
[
  {
    "bindings": {
      "cmd-alt-r": [
        "agent::NewExternalAgentThread",
        { "agent": { "custom": { "name": "cursor-acp" } } }
      ]
    }
  }
]
```

### Debugging

Use the command palette: **dev: open acp logs** to inspect ACP traffic.

## Modes

| Mode    | Behavior |
|---------|----------|
| Agent   | Full read/write access, tool execution |
| Plan    | Read-only, planning and analysis only |
| Ask     | Read-only, Q&A and explanations only |

Ask and Plan modes pass `--mode` and `--sandbox enabled` to `cursor-agent`, preventing any file writes.

## Environment variables

| Variable | Description |
|----------|-------------|
| `CURSOR_AGENT_BIN` | Path to `cursor-agent` binary |
| `CURSOR_AGENT_PATH` | Fallback if `CURSOR_AGENT_BIN` is not set |
| `CURSOR_ACP_MANAGED_EDITS` | Enable managed edits for client-side review UI (default: `1`) |
| `CURSOR_ACP_SESSIONS_FILE` | Custom path for session persistence file |
| `RUST_LOG` | Tracing level filter (`debug`, `info`, `warn`) |

## Development

```bash
cargo test                # run unit + integration tests
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

The test suite includes unit tests for CLI parsing/error classification, stream deduplication, diff extraction, and ACP protocol integration tests using an in-process connection harness with fixture scripts.

## License

[MIT](LICENSE)
