# Clanker Command Center — How It Works (walkthrough for a Rust-curious developer)

## The big picture

Everything hangs off one idea: **a conversation is just a `Vec<ChatMessage>` that you POST to an OpenAI-compatible server, get a reply back, and maybe loop on.** The shapes over that same loop:

- a prompt with no tools — send once, print, done.
- a prompt with tools — send, and if the reply contains `tool_calls`, execute them, append the results, send again (up to `max_iterations`).
- `clanker`/`tui` — the same, but the message list lives in a `ChatSession` that persists every turn to SQLite so you can resume later.

There is no mode to be in: whether the loop runs at all follows from whether any tool is set to something other than `never`. A new clanker starts with none, so it talks until `/tools on` gives it what `clank tools` allows.

One vocabulary note that pays off everywhere below, because the words nest and are easy to conflate:

- a **request** is one POST — stateless, so it carries the *entire* message array every time
- an **iteration** is one request plus running whatever tools it asked for
- a **turn** is one thing you typed through to a final answer: one request with no tools, one to `max_iterations` iterations with them
- a **clanker** is the saved conversation those turns accumulate into. The code calls it a session — `ChatSession`, the `sessions` table — and that boundary is deliberate: users learn one noun, the storage keeps the one it was written with

## The modules, what they actually do

**`main.rs`** — entry point. `Cli` is a `clap` derive struct: each subcommand (`Ask`, `Agent`, `Session`, `Login`, …) is a variant of the `Commands` enum with its flags. `main` is `#[tokio::main] async fn main()`; it parses, then `match cli.command` dispatches to a `cmd_*` function. Note the pattern every command follows at the top: `load_config()?` → resolve overrides → `Client::new(config)?`.

**`config.rs`** — `Config` is a plain `serde` struct with `#[serde(default = ...)]` on every field, so a config file with one key (`{"temperature": 1.5}`) is valid and the rest come from seeds. The API key is *not* in the file — `get_api_key()`/`set_api_key()` go through the `keyring` crate to the OS keychain. Also defines the tool-access model — `ToolAccess` (`ask`/`allow`/`never`) and `ToolAccessSettings`, which holds only the tools that differ from their default so a tool added later arrives with its own default already in force — and `SessionGates`, an `Arc<Mutex<ToolAccessSettings>>` + `Arc<AtomicBool>` pair that lets a mid-turn `/tools` or `/sandbox` change reach the running agent loop.

**`crypto.rs`** — AES-256-GCM over message content, titles and tool calls, with the key in the OS keychain. Metadata (roles, model names, timestamps) stays in the clear so the store can query on it. Lose that keychain entry and the database is unreadable — it is the one piece of state with no recovery path.

**`client.rs`** — the HTTP layer. `ChatMessage` is the one struct that flows everywhere: `role`, `content`, `tool_calls`, `tool_call_id`, plus `reasoning`/`reasoning_details` for thinking models. `Client::chat()` does a buffered POST to `{base_url}/chat/completions`; `chat_stream()` is the streaming version. The streaming path is the densest part: an `SseDecoder` splits the byte stream into `data:` lines (working on bytes so a multi-byte UTF-8 char split across network chunks survives), and a `StreamAccumulator` reassembles content deltas and — the fiddly bit — tool-call arguments that arrive fragmented across chunks, keyed by `index` so calls come out in order. The accumulator also catches the `usage` chunk, which is why a streaming request sends `stream_options.include_usage`: without asking, a stream ends having never said what it cost.

**`error_log.rs`** — a rolling hundred-entry log at `~/.clank/errors.log` recording the *shape* of a failed request (role sequence, tool-call and reasoning counts) but no message text. `CLANK_DEBUG_REQUESTS=1` additionally dumps the full failing body to a file, which is how a run of provider 400s was eventually traced to a missing `"type": "function"` discriminator on tool calls.

**`agent.rs`** — the heart. `run_agent_turn` is the loop:

```rust
while iteration < max_iterations {
    for text in steering.take() {          // typed since the last request
        messages.push(ChatMessage { role: "user", content: text, .. });
    }
    let message = request_turn(...).await?;   // send history + tool defs
    messages.push(message);
    if !message.has_tool_calls() { return Ok(final_response); }
    for tool_call in tool_calls {
        // maybe ask user for approval
        // execute_tool(name, arguments, sandbox).await
        messages.push(ChatMessage { role: "tool", content: result, tool_call_id: ... });
    }
}
```

The drain at the top is **steering**: a message typed while a turn runs joins *that* turn rather than waiting for the next one. Top-of-iteration is the only legal place for it — the previous iteration's tool results have completed their pairing, and the next request has not been built yet. A user message wedged between a `tool_calls` message and its results makes the whole request invalid.

`request_turn` takes streaming as a per-turn parameter rather than asking the client for it: streaming shapes how the *next request* is made, so a turn snapshots it at the start, the same way it snapshots model, effort and temperature. Approval and sandbox are the deliberate exceptions — those are read from `SessionGates` before every tool call, because revoking a permission is no use if it waits politely for the current work to finish.

It also takes a `UsageTracker`, which is where the token count comes from: the provider reports usage per *request*, and the loop above makes as many as it takes, so the tracker sums them across the turn. Shaped like `Steering` — an `Arc<Mutex<_>>` cheap to clone — for the same reason: the turn runs on its own task, and the caller needs a handle it can still read afterwards. That is what lets a cancelled turn still be charged for the requests that had already come back, since the abort destroys the task but not the shared count.

`normalize_system_prompt` prepends the system prompt fresh each turn, because Anthropic requires `system` to sit at position 0 and `/tools` can turn tools on mid-conversation.

**`compact.rs`** — folding the older part of a history into a summary, so a long conversation stops resending everything it has ever said. Three pieces: `seam` picks where to cut, `render`/`request_text` turn the span into a plain transcript, and `compact` makes one non-streaming request to the compactor model and hands back the summary plus what it cost.

`seam` is the part with the constraint in it. The cut lands on a *user* message and only ever there — a `tool` message whose `tool_calls` parent was folded away references a call the provider can no longer see, and the request is rejected outright. A user message never sits in the middle of that pairing, so cutting at one leaves both halves intact on the far side. The last couple of user turns are never folded, which is what keeps the exchange in progress verbatim.

Nothing is deleted anywhere in this. The summary and the index it covers go on the *session* row (`compacted_seq`, `compaction_summary`), and `ChatSession::request_messages` is the only reader: it returns the whole history when nothing has been compacted, and summary-then-tail when something has. So the transcript, `clank clankers show`, and everything else still see every message ever written; only what goes over the wire changes.

**`tools.rs`** — the six tools the model can call, defined as JSON schemas (`get_tool_definitions`) and executed in `execute_tool`: `write_file`, `read_file`, `list_files`, `replace_in_file`, `run_terminal_command`, and `web_fetch`. Results are returned as `serde_json::Value` objects and fed back to the model.

The sandbox lives here: `sandbox_bound` canonicalizes the current directory, `resolve_for_sandbox` canonicalizes the target's nearest *existing* ancestor (so `..` and symlinks resolve without creating anything on the way), and `sandbox_refusal` rejects writes outside it. Reads are deliberately unbounded — they mutate nothing, and confining them would break reading a file under `/etc`.

`web_fetch` exists for one reason: the agent could already reach the web through `run_terminal_command`, but a page is mostly markup, and converting it to text first costs two to four times fewer tokens for the same content. It does not prompt, and that is now a *default* (`default_access` gives the `web` category `allow`) rather than a name checked inside the gate — so it shows up as a row in `clank tools` you can change, instead of being invisibly exempt. The same function gives `terminal` `never`: the shell is the one tool bounded by nothing but what the user can do, so it is not offered until asked for.

**`session.rs`** — `ChatSession` owns the message history *and* the SQLite `Connection`, plus per-session settings (model, kind, effort, temperature, tool access, sandbox, verbose, highlight, streaming, working directory). `persist_pending()` writes only messages added since `saved_len` — that watermark is how resume doesn't duplicate history. Settings are snapshots: written to the DB at creation, then `/model` etc. mutate that row. `total_tokens` sits on the same row but is the one value that *accumulates* rather than being replaced — `add_tokens` is an additive `UPDATE`, so two writers can't lose each other's turn. `prompt_tokens` beside it is the opposite: the size of the *last* request, overwritten each turn, because it is a measurement of how big the history has grown and successive measurements of the same growing thing don't add up to anything. A reported `0` is never stored — providers that break nothing out of their total send one every time, and taking it would erase a real measurement and stop the clanker ever reaching its compaction threshold.

**`store.rs`** — raw SQL: `sessions` and `messages` tables, `create_session`, `append_message`, `load_messages`, `list_sessions`, prefix lookup for `--resume`. Message content, tool calls, and reasoning are encrypted via `crypto::encrypt_opt` before insert; metadata (model, roles, timestamps) is in the clear. `ensure_column` migrates old DBs by `ALTER TABLE ADD COLUMN`.

It also owns what the launch screen needs. `last_messages` reads every session's most recent message in one query, and `last_state` turns that plus the `activity` column into what a row should report. That column has to exist because a turn's messages are only written when it *finishes*: from storage alone, a request in flight looks exactly like a turn that failed, so the running process writes down what it is doing, and null means "nothing to say, read the messages."

**`conversation.rs`** — the TUI's worker. A `Conversation` wraps a `ChatSession` in a `tokio::spawn`ed task driven by a `Command` channel and reporting through an `Event` channel. This is what lets the TUI keep rendering while a turn runs, steer or queue messages typed mid-turn, and cancel (abort the turn task — which, via `kill_on_drop`, also reaps a running tool subprocess).

It is also where `$` commands run, spawned rather than awaited so that a thirty-second command cannot freeze the `select!` loop handling everything else, cancellation included.

Compaction is the other thing the worker drives, and it runs *between* turns — `run_turn` checks the threshold and compacts before the user's message is even recorded, so a cancellation leaves nothing half-started and the message about to be sent is not part of what gets summarized. It gets its own `select!` loop for the same reason a turn does: it is a round trip to a provider that can hang, and getting out of it has to stay possible. That is also why the command loop was flattened into `apply` plus a single queue drain — a message typed during a compaction has to go somewhere, and the drain after the match is now the one place that takes it, whether it was left there by a `Send` or by the compaction that was running when it arrived.

`absorb` changed shape for this: it used to skip as many messages as the session already held, which is only correct while the array handed to the turn *is* the session's history. Once a compacted clanker sends a summary and a tail instead, the count the turn started from is the only right answer, so `run_turn` passes it in.

Closing a `Conversation` has three shapes, and the difference is what the front end going away means. `shutdown` waits for the worker. `leave` waits only briefly and then lets go, so an idle worker's last writes land while a turn in flight carries on unwatched — it finishes, absorbs and persists as usual, then clears the session's activity and releases its claim. Abandoning it instead would be the one genuinely lossy option: its tool calls have already touched the disk, and a turn is only written when it ends.

The third is parking, and it is what makes the launch screen a monitor. Backing out of a working session moves the whole `Chat` — worker, `App`, transcript cache — into the event loop's parked list rather than letting go of it. Its events are drained into its `App` on the tick, so the transcript stays whole and an approval that arrives while you are elsewhere is sitting in the box when you press Enter on its row; reopening is `Vec::remove` and a screen swap, with no claim to retake and nothing to reload. A parked session that goes idle is `leave`d, which is what releases its claim. The claim itself is untouched by any of this: a session another process holds is refused exactly as it was, and a parked one is refused to everyone else too, because this process is still holding it.

Only cancelling (`Esc`) and quitting abort a turn, and quitting has to, since a worker is a task and tasks go with the process.

**`ui.rs` / `terminal_ui.rs` / `tui/`** — front ends. `AgentUi` is a trait with `event()` and `approve()`; the CLI's `TerminalAgentUi` prints, the TUI's `ChannelUi` forwards to the worker's event channel. This is the key decoupling: `agent.rs` never prints anything, it just calls `ui.event(...)`, so the same loop drives CLI, TUI, and tests.

`ui.rs` also holds `classify`, which turns typed input into a `Submission` — `$ cargo test`, `/model gpt-5`, or an ordinary message — and `conversation.rs`'s `command_for` maps the ones needing the worker into `Command`s. Both matches are exhaustive on purpose: adding a `Submission` forces a decision in every front end rather than letting one quietly drift. The `COMMANDS` table beside it is the single list of what a command *is*: `/help` renders it, the typo check scans it, and the TUI's input box reads it twice over — `command_span` for which part of a half-typed line to colour, `command_prefix`/`command_matches`/`command_syntax` for the row of candidates above it that Tab completes from.

**`tui/`** — `mod.rs` is the event loop and key handling, `app.rs` the state a screen renders from, `render.rs` the conversation view, `picker.rs` the launch and deployment screens. The launch screen lists every session in one flat list, re-reading every couple of seconds so one running in another terminal stays current, and gives each a small braille square hashed from its id — the left half of which is the mark in the reply gutter once you are inside that session. `picker.rs` also owns `Deployment`, the form a clanker is created from: it holds the id whose mark you are looking at *before* anything is written, and `plan()` is where a name, a model and a temperature are validated, so a bad one is a line on that screen rather than a clanker that exists and fails on its first turn.

## Trace `clank "hi"` end to end

1. `Cli::parse()` → `Commands::Ask { prompt: "hi", .. }` → `cmd_ask`.
2. `load_config()` → `Config` from `~/.clank/config.json` (or defaults).
3. `resolve_model`, `resolve_temperature`, `resolve_effort_level` merge CLI flags over config.
4. `Client::new(config)` — pulls the API key from the keychain.
5. Build `vec![ChatMessage { role: "user", content: "hi", .. }]`.
6. `client.chat(...)` → `build_request` → POST `https://openrouter.ai/api/v1/chat/completions` with `Authorization: Bearer <key>`, `stream: true`.
7. The stream yields `StreamEvent::Content` deltas → printed as they arrive, then `Done`.
8. `cmd_ask` prints the wrapped reply.

## Trace `clank "write fib to fib.rs" --tools` — the delta

Same start, but `cmd_agent` calls `agent::run_agent` → `run_agent_turn`, and now `tools` are included in the request. The reply likely comes back with `tool_calls` instead of text. Then:

- the turn is offered only the tools that are not `never` (`offered_tools`), and each call then goes through the gate — `allow` runs, `ask` calls `ui.approve(...)`, `never` is refused where it stands — → `execute_tool("write_file", r#"{"filepath":"fib.rs",...}"#, true)`
- the result JSON is appended as `role: "tool"` with the matching `tool_call_id`
- loop repeats until the model replies with text and no tool calls, or the cap hits → `"Agent exceeded max iterations"`.

That `tool_call_id` threading is the whole trick: providers reject a `tool` message that doesn't reference the call that produced it.

## Trace `$ cargo test` in the TUI — the one with no model in it

Worth following because it exercises the worker without touching the API:

1. `classify` sees the leading `$` → `Submission::Shell("cargo test")`.
2. `command_for` maps it to `Command::Shell`, which reaches the worker.
3. The worker emits `ShellStarted` and spawns the command in the *session's* directory.
4. The box appears immediately with a spinner; the event loop keeps running, so you can still type and cancel.
5. `ShellFinished` carries the output back, capped, keeping the **end** — a failing build says what went wrong on its last lines.
6. `Ctrl-S` appends it to the conversation as a user message *without* starting a turn, so it reaches the model together with whatever you type next. `Ctrl-D` throws it away.

## Rust patterns they'll keep seeing

- `#[derive(Parser)]` / `#[derive(Subcommand)]` — clap generates the CLI from structs.
- `#[derive(Serialize, Deserialize)]` + `#[serde(default, skip_serializing_if = "Option::is_none")]` — config and wire formats are plain structs.
- `Result<T, anyhow::Error>` + `?` everywhere; `anyhow!` for ad-hoc errors.
- `async`/`await` + `tokio` for all I/O; `tokio::time::timeout` for timeouts; `tokio::process::Command` with `kill_on_drop(true)` for the terminal tool.
- `impl Stream<Item = Result<StreamEvent>>` + `async_stream::try_stream!` for the SSE stream.
- `Arc<Mutex<…>>` handles (`SessionGates`, `Steering`) for the two things a running turn has to be able to see change underneath it.
- `tokio::select!` with `biased;` in the worker, so a finishing turn is handled before a late command.
- Exhaustive `match` with no `_` arm wherever two front ends must agree — the compiler is what keeps them from drifting.
- Tests are extensive and read like documentation — `client.rs`'s chunk-reassembly tests are the easiest way to understand the streaming format.

## Suggested reading order

1. `main.rs` (skim the `cmd_ask` function — smallest complete path)
2. `config.rs` (the `Config` struct + `load_config`)
3. `client.rs` (the `ChatMessage` struct, then `chat`, then the streaming section)
4. `agent.rs` (`run_agent_turn` — the loop)
5. `tools.rs` (what a tool actually is)
6. `session.rs` + `store.rs` (persistence)
7. `conversation.rs` + the `tui/` dir last — it's the most complex front end.

## The one-sentence takeaway

**Once you understand that every feature is just "build a `Vec<ChatMessage>`, send it, append the reply (or execute its tool calls and append those), repeat," you understand the whole tool — everything else is configuration, persistence, deciding who may change that vector while the loop is running, and pretty front ends around it.**
