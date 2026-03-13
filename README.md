# cursor-acp

A thin ACP proxy that wraps Cursor's native `agent acp` server, adding session persistence, a Plan UI, and edit tracking to make Cursor work as a first-class agent in [Zed](https://zed.dev) and other [ACP](https://agentclientprotocol.com) clients.

## Why a proxy?

Cursor ships a built-in ACP server (`cursor-agent acp`) that handles modes, auth, streaming, tool calls, and permissions natively. But it doesn't provide everything an editor like Zed needs for a polished experience — there's no session history, the todo list uses a custom protocol Zed can't display, and file edits bypass Zed's review UI. This proxy sits between the editor and `agent acp`, forwarding everything transparently while filling in those gaps.

## Features

- **Session persistence** — sessions are stored locally so you can browse, reload, and continue past conversations. Sessions are listed with titles, timestamps, and full event history replay.
- **Plan UI** — Cursor's `_cursor/update_todos` requests and `createPlan`/`updateTodos` tool calls are translated into standard ACP Plan entries that Zed renders natively with status tracking.
- **Edit tracking** — file edits made by Cursor are routed through Zed's `fs/write_text_file` mechanism, populating the ActionLog so the "Edits" panel shows changed files with accept/reject controls.
- **Model selection** — available models are fetched from Cursor CLI and injected into session responses. Switching models restarts the agent process transparently.
- **Session ID remapping** — after model switches or session loads, the proxy keeps Zed's session ID in sync with the child process so updates, file operations, and permissions all route correctly.
- **Transparent proxy** — all other ACP messages pass through untouched.

## Requirements

- [Cursor CLI](https://cursor.com) installed (provides `cursor-agent` / `agent`)
- Rust stable (if building from source)

## Install

### Option A: Download a release binary

Grab the archive for your platform from [GitHub Releases](https://github.com/cameroncooper/cursor-acp/releases), extract it, and place `cursor-acp` on your `PATH`.

### Option B: Install with Homebrew

```bash
brew install cameroncooper/tap/cursor-acp
```

### Option C: Build from source

```bash
cargo install --git https://github.com/cameroncooper/cursor-acp --locked
```

## Use with Zed

### 1. Ensure `cursor-agent` is available

`cursor-acp` spawns `cursor-agent acp`. Either:

- have `cursor-agent` (or `agent`) on your `PATH`, or
- set `CURSOR_AGENT_BIN` to the full path.

If you have Cursor installed but not the CLI, you can install it with:

```bash
cursor --install-agent-cli
```

### 2. Add a custom agent in Zed settings

Open Zed settings (`cmd+,` → JSON) and add:

```json
{
  "agent_servers": {
    "Cursor CLI": {
      "type": "custom",
      "command": "cursor-acp",
      "args": [],
      "env": {}
    }
  }
}
```

If `cursor-agent` is not on `PATH`:

```json
{
  "agent_servers": {
    "Cursor CLI": {
      "type": "custom",
      "command": "cursor-acp",
      "args": [],
      "env": {
        "CURSOR_AGENT_BIN": "/absolute/path/to/cursor-agent"
      }
    }
  }
}
```

### 3. Start a thread

1. Open the Agent panel in Zed.
2. Click **+** and select **Cursor CLI**.
3. If prompted, click **Authenticate** in Zed (it will run `cursor-agent login`), or run `cursor-agent login` (or `agent login`) in your terminal.

### Debugging

Use the command palette: **dev: open acp logs** to inspect ACP traffic.
Set `RUST_LOG=debug` in the env config for proxy-level logging.

## Environment variables

| Variable | Description |
|----------|-------------|
| `CURSOR_AGENT_BIN` | Path to `cursor-agent` binary (default: `cursor-agent`) |
| `CURSOR_AGENT_PATH` | Fallback if `CURSOR_AGENT_BIN` is not set |
| `RUST_LOG` | Tracing level filter (`debug`, `info`, `warn`) |
| `CURSOR_ACP_SESSIONS_FILE` | Custom path for the session index (default: `~/.cursor/cursor-acp/sessions.json`) |
| `CURSOR_ACP_WRITE_PLAN_FILE` | Emit plan markdown via `fs/write_text_file` (default: enabled; set `0`/`false` to disable) |
| `CURSOR_ACP_WRITE_PLAN_FILE_MESSAGE` | Emit a chat message with a `file:///` plan link (default: enabled; set `0`/`false` to disable) |
| `CURSOR_ACP_LINK_CURSOR_PLAN_FILE` | Prefer linking to existing `.cursor/plans` markdown when available (default: disabled) |
| `CURSOR_ACP_WORKSPACE_ROOT` | Force an absolute workspace root used for all `session/new` forwarding and child restarts |
| `CURSOR_ACP_ALLOW_ROOT_WORKSPACE` | Allow root (`/` on Unix, drive root on Windows) as workspace (default: disabled) |

## Architecture

```
Zed (stdin/stdout) ←→ cursor-acp proxy ←→ cursor-agent acp (child process)
                         │
                         ├── session/list, session/load → local session store
                         ├── tool_call kind:edit → fs/read_text_file + fs/write_text_file
                         ├── _cursor/update_todos, createPlan, updateTodos → Plan UI
                         ├── session/set_model → child restart with --model
                         └── everything else → passthrough
```

Sessions are persisted at `~/.cursor/cursor-acp/`:
- `sessions.json` — session index (id, title, cwd, timestamps)
- `history/<session-id>.jsonl` — full event history per session

## Development

```bash
cargo test
cargo clippy --all-targets -- -D warnings
```

## License

[MIT](LICENSE)
