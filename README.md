# ⣕⢛ Clanker Command Center (WIP)

An OpenAI-compatible CLI frontend for any LLM provider, with agentic tool capabilities, written in Rust. Defaults to OpenRouter, but works with any OpenAI-compatible service (OrcaRouter, Together, Groq, self-hosted gateways, etc) via `clank endpoint` — see [Using other providers](#using-other-providers).

CCC is most stable on Linux at the moment!

## Features

- **Fast & lightweight** — Compiled Rust binary, single executable with no runtime dependencies
- **One noun** — a clanker: a saved conversation with tools, without them, or with exactly the ones you allow. A one-off prompt, a line-based conversation, or the full-screen TUI, all over the same thing
- **Streaming responses** — Replies appear as they're generated rather than all at once
- **File operations** — LLM can read, write, and modify local files
- **Per-tool permissions** — every tool asks, runs freely, or isn't offered at all, set globally or per clanker with `clank tools`
- **Model selection** — Choose from configured provider's models
- **Agentic loops** — Multi-turn execution with tool calling
- **Persistent clankers** — `clanker`/`tui` conversations are saved to SQLite and resumable across restarts
- **Secure credential storage** — API keys live in your OS keychain, not a plaintext file

## Manual Installation

### Prerequisites
- Rust 1.70+ (install from [rustup.rs](https://rustup.rs))
- An API key for your provider — defaults to OpenRouter, get one from [openrouter.ai/keys](https://openrouter.ai/keys) (or see [Using other providers](#using-other-providers))

### Build from Source

```bash
cargo build --release
```

The binary will be at `target/release/clank` (or `clank.exe` on Windows).

### Install Globally

```bash
cargo install --path .
```

Then use `clank` from anywhere:

```bash
clank login
clank "Hello"                       # a one-off question
clank "Fix the build" --tools       # ...with tools
clank                               # the full-screen UI
```

## Usage

### Commands

#### `login`
Set up or update your API key. The key is stored in your OS keychain (macOS Keychain, Windows Credential Manager, or the Linux Secret Service) rather than in a plaintext config file.

```bash
clank login
```

#### `logout`
Remove your stored API key from the OS keychain.

```bash
clank logout
```

#### `status`
Check your configuration.

```bash
clank status
```

#### `models`
List available models from your configured provider (shows first 20).

```bash
clank models
```

#### `model [name]`
View or set the persistent default model, so you don't need to pass `-m` on every call.

```bash
# Show the current default
clank model

# Set the default model
clank model anthropic/claude-opus-4.5

# Clear the default (falls back to openrouter/auto)
clank model --clear
```

Once set, `ask`, `clanker`, and `agent` all use this default unless overridden with `-m`/`--model` for that specific call.

**Per-clanker models.** A clanker remembers the model it's using. Passing
`--model` when resuming, or running `/model` inside the TUI, switches it *and*
records it — so resuming later with no flag picks up where you left off rather
than reverting. Each stored reply still records the model that produced it,
even though the transcript itself — in `clanker`, `tui`, and `clankers show`
alike — no longer prints a model label on every line; `/model` shows what the
clanker is using *now*.

#### `max-iterations [value]`
View or set the persistent default for how many tool-calling iterations `agent` may run before giving up.

```bash
# Show the current default
clank max-iterations

# Set the default
clank max-iterations 20

# Clear it — a clanker with tools then needs a cap set per call
# per clanker (/max-iterations) to run at all; it does not fall back to 20
clank max-iterations --clear
```

Ships at 20 on a fresh install (no `config.json` yet), but once cleared it
stays cleared — nothing silently reintroduces a number. Overridden per call
with `--max-iterations` on `agent`, or persistently per clanker with
`/max-iterations` inside a `clanker`/`tui` conversation — see
[Per-clanker models](#model-name) for how that precedence works.

#### `temperature [value]`
View or set the persistent default sampling temperature (0-2) sent to models that support it.

```bash
# Show the current default
clank temperature

# Set the default
clank temperature 1.2

# Clear it — requests are then sent with no temperature field at all, and
# the provider uses its own default, rather than this falling back to 0.7
clank temperature --clear
```

Ships at 0.7 on a fresh install, same caveat as `max-iterations` once
cleared. Overridden per call with `--temperature` on `ask`, `clanker`, or
`agent`, or persistently per clanker with `/temperature` (or its `/temp`
shorthand) inside a `clanker`/`tui` conversation — see [Per-clanker
models](#model-name) for how that precedence works.

In `tui`, the clanker's current value shows in the settings row as 🌡
`<value>` (or `default` when nullified — see [Clanker
Persistence](#clanker-persistence)), color-coded cool-to-hot (cyan → yellow
→ orange → pink) as it rises from 0.

#### `verbose [on|off]`
View or set whether new clankers start showing full tool-call detail — arguments, results, and the model's own thinking.

```bash
# Show the current setting
clank verbose

# Start new clankers verbose
clank verbose on
```

Off by default. This is the *starting* value: a clanker snapshots it at creation, and `/verbose` from then on toggles that clanker, which is remembered per clanker rather than changing this. `clank "..." --tools -v` is unaffected — it's a per-run flag.

#### `highlight [on|off]`
View or set whether new clankers band your own messages in the transcript, so
they stand out when scrolling back through a long turn.

```bash
# Show the current setting
clank highlight

# Start new clankers without the band
clank highlight off
```

On by default, and the *starting* value like `verbose`: a clanker snapshots it
at creation, and `/highlight` changes that one clanker from then on.

The band is derived from your terminal's own background — one faint step
lighter on a dark theme, darker on a light one (NOT working too well atm) — rather than a fixed colour, so
it stays a tint of whatever you're using rather than a bar drawn over it. If
your terminal doesn't answer the query that asks, no band is drawn at all
rather than one guessed at.

#### `selection [on|off]`
View or set whether the launch screen bands its selected row.

```bash
clank selection off
```

On by default. Global only, with no per-clanker counterpart and no slash
command: the launch screen belongs to no clanker, so there is nothing to
override it with.

#### `sandbox [on|off]`
View or set whether the agent's file-writing tools are confined to your current working directory.

```bash
# Show the current setting
clank sandbox

# Let the agent write anywhere it has permission to
clank sandbox off
```

On by default, and the bound is the working directory alone — not your home directory, which would let an agent write across every project you keep under `~`. It bounds `write_file` and `replace_in_file` only: reads are never restricted, since they change nothing and confining them would break ordinary work like reading a file under `/etc`. The bound is checked against the path a write *resolves to*, so `..` and symlinks can't be used to step outside it, and `clank` writes its own `~/.clank` state directly so that keeps working at any setting.

This is the persistent default; a clanker snapshots it at creation, and `/sandbox` changes the clanker you're in. It's a separate axis from `tools`: a tool's state decides whether you're *asked* first, the sandbox decides whether the write is allowed at all.

#### `effort [value]`
View or set the persistent default reasoning effort sent to models that support it. Applies to `ask`, `clanker`, and `agent`. Usually `low`, `medium`, or `high`, but not checked against a fixed list — models vary in what they accept, and an unsupported value just gets rejected by the API.

```bash
# Show the current effort level
clank effort

# Set the default
clank effort high

# Clear it (falls back to the provider default)
clank effort --clear
```

Overridden per call with `--effort-level` on `ask`, `clanker`, or `agent`, or
persistently per clanker with `/effort` inside a `clanker`/`tui` conversation.

When an effort level is set, `ask`, `clanker`, and `agent` label responses as `<model> (<effort>)` instead of just `<model>`, so you can see which effort level produced a given answer.

In `tui`, the clanker's current value shows in the settings row as 🧠
`<level>`, color-coded calm-to-intense (cyan → yellow → red) as it rises
from `low`.

#### `endpoint [url]`
View or set the API base URL, so you can point `clank` at any OpenAI-compatible service instead of OpenRouter (OrcaRouter, Together, Groq, a self-hosted gateway, etc).

```bash
# Show the current endpoint
clank endpoint

# Point at OrcaRouter
clank endpoint https://api.orcarouter.ai/v1

# Clear it (falls back to the OpenRouter default)
clank endpoint --clear
```

Switching endpoints doesn't switch your API key or default model automatically — run `clank login` to set the new provider's key, and `clank model` to set a model it actually serves.

#### `effort-style [value]`
View or set how the reasoning effort level (`clank effort`) is serialized in requests, since providers disagree on the shape:

- `nested` (default) — sends `reasoning: { effort: "<level>" }`, as OpenRouter expects.
- `flat` — sends `reasoning_effort: "<level>"` at the top level, as OrcaRouter expects.
- `none` — omits effort entirely, for providers that reject unrecognized fields.

```bash
clank effort-style
clank effort-style flat
clank effort-style --clear
```

#### `headers`
View or manage extra HTTP headers sent with every API request, useful for providers with optional attribution headers (e.g. OpenRouter's `HTTP-Referer`/`X-Title`).

```bash
# Show current extra headers
clank headers

# Set a header
clank headers set HTTP-Referer https://myapp.example.com
clank headers set X-Title "My App"

# Remove one
clank headers unset HTTP-Referer
```

#### `tools`
What the agent can do, and what it may do without asking. Bare `clank tools`
lists every tool with its state:

```bash
$ clank tools
  read_file              ask    read     · Read a file from disk
  list_files             ask    read     · List a directory
  web_fetch              allow  web      · Fetch a web page as text
  write_file             ask    write    · Write or overwrite a file
  replace_in_file        ask    write    · Replace a string inside a file
  run_terminal_command   never  terminal · Run a shell command
```

Each tool is in one of three states:

| State | Means |
|---|---|
| `ask` | The turn stops and asks before every call. The default for everything that touches your machine |
| `allow` | It runs without asking |
| `never` | It isn't offered to the model at all — it costs no tokens and cannot be called. A tool set to `never` mid-turn is refused on the spot, not merely dropped from the next request. This is where `run_terminal_command` starts |

```bash
# One tool
clank tools never run_terminal_command
clank tools allow read_file

# A whole category: read, write, terminal, web
clank tools allow read

# Everything
clank tools ask all
clank tools off              # every tool to never
clank tools on               # every tool back to its default
```

**Two tools don't default to `ask`, for opposite reasons.**

`run_terminal_command` defaults to **`never`** — not offered to the model at
all. Every other tool is bounded by what it is *for*, and the sandbox bounds
the writes on top of that; the shell is bounded by nothing except what you
can do yourself. Turn it on when you want it:

```bash
clank tools ask run_terminal_command     # globally
❯ /tools ask run_terminal_command        # or just this clanker
```

`web_fetch` defaults to **`allow`** — it reads a page and changes nothing,
and a prompt per page is exactly the friction that would send the model back
to curling raw HTML through the shell. That used to be hardcoded and
invisible; it's a row you can see and change now.

So `on` is not the same as `ask all`: it restores each tool's *own* default,
shell off and web free, rather than making everything prompt.

**A clanker with every tool `never` has no tools at all**, which is what
"ask mode" used to be. There's no separate switch for it — and that is what a
new clanker starts as, so having tools is something you turn on with
`/tools on` rather than something you have to remember to turn off.

**Per-clanker tools.** These commands set the policy tools follow *once a
clanker has them* — a new clanker starts with none, and `/tools on` is what
applies this. A clanker remembers its own too: running `/tools` inside it (see
[`clanker`](#clanker) or [the full-screen UI](#clank-with-no-command))
switches and records them for that clanker alone, the same way `/model` does
for models — so resuming later picks up where you left off rather than
reverting to the configured default.

Settings written before tools had states of their own are read as what they
meant: a category that asked becomes `ask` for the tools in it, one that
didn't becomes `allow`. Nothing is migrated and nothing is rewritten. The
shell is the exception — it takes its new default of `never` whatever the
old gate said, since the old model had no way to express "not offered" and
an upgrade shouldn't leave you with a shell a fresh install doesn't have.

#### A prompt on its own

```bash
clank "What's the capital of France?"

# With tools — read and write files, run commands
clank "Fix the failing test in src/parser.rs" --tools

# Keep it as a clanker you can go back to
clank "Refactor the whole src/ directory" --tools --save

# Specify a model
clank "Explain quantum computing" -m anthropic/claude-opus-4.5

# Override temperature or effort level for this call only
clank "Write a haiku" --temperature 1.2
clank "Design a lock-free queue" --effort-level high
```

A prompt with no `--tools` is answered with no tools at all — it is a
question, not a task. With `--tools` the run gets whatever `clank tools`
allows, and `--verbose` shows each call as it happens.

This replaces the old `ask` and `agent` commands, which differed by exactly
the thing that is now a flag.

**A subcommand wins over a prompt**, so a one-word prompt that happens to be
a subcommand name (`clank status`) runs the subcommand. Put it after `--` to
force the prompt: `clank -- status`.

##### Keeping a run

By default a one-off does the work and leaves nothing behind. `--save` keeps
it as a clanker, so it appears in the picker and in `clank clankers list`,
reports `working`/`failed`/`replied` while it runs, and can be reopened later
with `clank clanker --resume <id>` or the full-screen UI. It is named after
the prompt, and the id is printed when it starts. It works with or without
`--tools`: a saved run with no tools is a clanker with no tools, which is a
perfectly good thing to come back to.

Only one process may run a clanker at a time. A run claims it before reading
its history, and a second process is refused rather than interleaving two
sets of turns into a history neither of them wrote. The claim expires by
itself, so a clanker whose runner died is available again with nothing to
clean up.

#### `clanker`
Start an interactive, persistent conversation — the line-based counterpart to
`tui`, with the same experience minus the full-screen UI. It's saved
automatically as you go (see [Clanker Persistence](#clanker-persistence)), so
you can pick it back up later.

A new clanker starts with **no tools** — it can only talk. `/tools on` gives
it the ones `clank tools` allows, and `/tools <state> <target>` tunes them
from there. There is no mode to be in: what a clanker can do is just what its
tools are set to. See [`tools`](#tools). Also supported, matching the TUI
exactly:

| Command | Does |
|---|---|
| `/help` | List every in-clanker command and what it does. The same list in both front ends, generated from the one the parser uses, so it cannot drift from what actually works |
| `/models` | Browse the models the endpoint offers and pick one. **TUI only** — it is a cursor moving through a list, which the line-based prompt has nowhere to draw. `clank models` lists them here instead |
| `/model <name>` | Switch the model for the rest of the clanker, and remember it |
| `/model` | Show the model currently in use |
| `/effort` | Show the reasoning effort level currently in use |
| `/effort <level>` | Switch reasoning effort for the rest of the clanker, and remember it |
| `/effort clear` | Nullify it — no effort field is sent at all until set again |
| `/effort default` | Read the *currently* configured default effort and save that to the clanker |
| `/verbose <on\|off>` | Show the model's thinking and full tool call arguments/results, or a one-line notice per call. Bare `/verbose` shows the current setting |
| `/stream <on\|off>` | Stream this clanker's replies token-by-token, or wait for the whole reply. Bare `/stream` shows the current setting. Overrides `clank stream` for this clanker |
| `/max-iterations <n>` | Switch the tool-calling iteration cap per turn (only matters when it has tools), and remember it |
| `/max-iterations clear` | Nullify it — a clanker with tools then errors on any turn until a cap is set again |
| `/max-iterations default` | Read the *currently* configured default cap and save that to the clanker |
| `/temperature <n>` (or `/temp <n>`) | Switch the sampling temperature for the rest of the clanker, and remember it |
| `/temperature clear` (or `/temp clear`) | Nullify it — requests are then sent with no temperature field |
| `/temperature default` (or `/temp default`) | Read the *currently* configured default temperature and save that to the clanker |
| `/temperature` (or `/temp`) | Show the temperature currently in use |
| `/tools <ask\|allow\|never> <tool\|category\|all>` | Switch what a tool may do for the rest of the clanker, and remember it. Takes effect immediately — including partway through a running turn, from its next tool call |
| `/tools on` / `/tools off` | Tools on, as `clank tools` allows them, or every tool off |
| `/tools` | List every tool and what it may do |
| `/sandbox <on\|off>` | Confine the agent's file writes to the working directory, or allow them anywhere. Takes effect immediately, including partway through a running turn |
| `/sandbox` | Show whether writes are currently confined |
| `/status` | Show every setting this clanker is running with — model, effort, temperature, iteration cap, sandbox, verbose, highlighting, streaming, what each tool may do, and the directory it runs in. The clanker-scoped counterpart to `clank status` |
| `/highlight <on\|off>` | Band your own messages in the transcript, or don't. Bare `/highlight` shows the current setting |
| `/clanker title <new title>` | Rename this clanker. Bare `/clanker` (or `/clanker title`) shows its current name |
| `/send`, `/discard` | Answer the `$` command box — the same as `Ctrl-S` and `Ctrl-D`. Typed forms exist because terminals claim chords: Zed's takes `Ctrl-S` |
| `/allow`, `/deny` | Answer a tool approval — the same as `Ctrl-Y` and `Ctrl-N`. Without a way to answer, a turn waits on a decision it can never be given |
| `/back` | Return to the launch screen — the same as `Ctrl-B`, which tmux claims as its own prefix |

A mistyped invocation of one of these (`/effort` with no value, `/tools
bogus off`), or a misspelled command name (`/mode` for `/model`), is
reported as an error rather than sent to the model — see the note under
`tui`'s command table for the exact boundary.

A new clanker needs a name. Pass one with `--title`, or you'll be asked for it before the clanker starts — starting one is meant to be deliberate, so there's no untitled path and a blank answer is refused. A resumed clanker keeps the name it has, and `--title` is ignored with a note.

```bash
clank clanker --title "Fix the parser"
# Type exit to quit

# Omit --title and you'll be prompted for one
clank clanker

# Override the default model for a new clanker (ignored when resuming —
# a resumed clanker always keeps its own saved model)
clank clanker -m anthropic/claude-opus-4.5

# Override the default max tool-calling iterations per turn
clank clanker --max-iterations 30

# Override the default reasoning effort for a new clanker (ignored when
# resuming — a resumed clanker always keeps its own saved value)
clank clanker --effort-level high

# Resume a previous clanker by id (or a unique prefix of it) — works
# whatever tools it was last left with
clank clanker --resume a1b2c3d4

# Or omit the id to pick from a numbered list of all your saved clankers
clank clanker --resume
```

#### `clank` with no command
A full-screen terminal UI. Unlike the line-based `clanker`, it owns the
screen, which is what lets the input box stay live while a reply streams in,
tool approvals appear inline, and a running turn be interrupted. Otherwise
the two are functionally identical — same commands, same settings, same
saved clankers, interchangeably resumable from either.

It's not a subcommand — there are no flags. Run `clank` with nothing else on
the command line, and it opens on a **launch screen**: start a new clanker,
or pick up any saved one. Every clanker is in one list, newest first, and
each row shows the directory it will resume in — `.` for the one you are
already in, a `~`-relative path for anything under home, and a full path for
anything above it. That directory matters because it is where the clanker
resumes and what its sandbox is bounded to.

```bash
clank
```

A new clanker starts with **no tools**, and `/tools on` gives it the ones
`clank tools` allows. There is no mode to pick on the way in and none to
switch between once you are there — a clanker is one thing, and what it can
do is what its tools say. A resumed clanker picks back up with the tools,
model and effort level it was last left with.

Choosing "Deploy clanker" from the launch screen opens **Clanker
Deployment**: a form for everything the clanker starts with, and the first
thing you want it to do. A name is required — starting a clanker is meant to
be deliberate, so there's no untitled path. The clanker is kept from the
moment you deploy it, whether or not anything is ever said in it.

Under **Settings** are the values the clanker is created with, each seeded
from the configured default so pressing Enter straight away deploys what a
new clanker always was:

| | |
|---|---|
| `Tools` | Whether it deploys with tools — the ones `clank tools` allows, exactly as `/tools on` would give it. Off by default: what a clanker may do to your machine is a decision worth making, not one to inherit from a config file you set months ago |
| `Model` | The model it runs, instead of `clank model`'s default |
| `Effort` | Reasoning effort, cycling through `default` (no field sent), `low`, `medium`, `high` — plus whatever `clank effort` is set to, if that's something else |
| `Temperature` | Emptying it is a real setting: no temperature field is sent at all, and the row says `none sent` |
| `Sandbox` | Whether the agent's file writes are confined to the directory you're deploying from |

Under **Initial Orders** is the first message, sent the moment the clanker
opens — so a clanker can be deployed already working rather than opened and
then told what to do. Leave it empty and it opens waiting for you, as it
always did.

Everything here can still be changed from inside the clanker with the usual
`/model`, `/tools`, `/effort`, `/temperature` and `/sandbox`; this is where
it's cheaper to say up front.

That screen also shows the mark the new clanker will carry, and `Tab` rolls
another. Because the mark is hashed from the clanker id, and the id is fixed
once the clanker exists, this is the only moment it can be chosen rather than
dealt — so keep pressing `Tab` until you get one you like, then fill the form
in and hit Enter.

Every clanker carries a small square of braille dots, hashed from its id and
the same for the life of the clanker. It says nothing you can type — the
clanker id isn't shown on the launch screen at all — and exists purely so a
row you have seen before is recognisable while the list refreshes and rows
move under you. Inside that clanker the same mark leads its title row —
`<mark> <name>  <directory>` — and sits in the gutter beside every reply, and
the CLI draws it too, so a reply is tied to the clanker it came from wherever
you read it. Names repeat and directories are shared; the mark is the thing
that says *which* clanker you are looking at.

A clanker being run by another process can be seen but not opened: only one
process may hold a clanker at a time, since two appending turns to one
history would interleave them irreparably. Opening one says so. The hold is
released when that process finishes, and expires by itself within half a
minute if it died instead.

Each row also shows what that clanker is doing, re-read every couple of
seconds so one you're running in another terminal stays current:

| | Meaning |
|---|---|
| 🔨 / 💬 | Whether that clanker has tools, in the column beside its mark. The hammer is the one the transcript puts in front of every tool call |
| spinner, yellow | Working — a request is in flight right now. The same animation and colour a conversation shows for itself |
| `?` yellow | Waiting on an approval nobody has answered. A clanker you are *inside* says the same thing in its settings row — `? waiting` in place of the working spinner, since a turn stopped at a gate is not moving |
| `✗` red | The last turn ended in an error — worth resuming to see why |
| `✓` green | The model answered; the turn ran to completion |
| `⎚` grey | Held by another process — it can be seen but not opened. Only shown when nothing else already implies a live process: a working or waiting clanker says so with its own badge |
| `⋯` cyan | Something was sent and nothing came back, and no process is saying otherwise |
| `⚑` yellow | Stopped part-way — after a tool result with no answer, or on a tool call that never ran |
| (blank) | Created, never used |

…followed by a one-line preview: normally the clanker's last message (a tool
call shows the tool it asked for), but a clanker waiting on an approval shows
*what* it's asking about instead — `needs approval — run_terminal_command: rm
-rf build` — since that's the row you'd want to act on.

The first three come from the process running the clanker, which is the only
thing that knows them: a turn's messages are only written when it *finishes*,
so from storage alone a request in flight looks exactly like a turn that
failed. The rest are read from the messages themselves, and are what a
clanker nobody is running can tell you.

Clankers started or deleted elsewhere appear and disappear as the list
refreshes, and the cursor follows the clanker it was on rather than the row
number, so rows moving underneath it can't quietly select a different
conversation.

A process killed outright leaves its last word behind until something opens
that clanker again.

**Launch screen**

| Key | Does |
|---|---|
| `↑` / `↓` (or `k` / `j`) | Move the selection (section labels are skipped) |
| `Enter` | Open the selected row |
| `r` | Rename the selected clanker |
| `d` | Delete the selected clanker (asks to confirm) |
| `q` | Quit |

**Clanker Deployment** (after choosing `Deploy clanker`)

| Key | Does |
|---|---|
| `↑` / `↓` | Move between the form's fields |
| `←` / `→` (or `Space`) | Change the field you're on, where it holds one of a set — `Tools`, `Effort`, `Sandbox`. Typing goes into the rest |
| `Tab` | Roll a different mark. It is hashed from the clanker id, which is fixed once the clanker exists, so this screen is the only place it can be chosen |
| `Enter` | Deploy it. A blank name, a blank model, or a temperature that isn't a number is refused here — with the reason under the form, and nothing created |
| `Esc` | Back to the list, having created nothing |

Opening a clanker whose directory no longer exists asks whether to resume in
the current one instead, repointing the clanker there — the same thing
`clank clanker --resume <id> --here` does from the shell.

**In a conversation**

| Key | Does |
|---|---|
| `Enter` | Send. If a turn is already running, see **Sending while a turn is running** below — with tools the message joins that turn; without them it waits and becomes the next one |
| `Esc` | Cancel the in-flight turn (kills a running tool command too) |
| `Alt-Enter` / `Shift-Enter` | Insert a newline instead of sending. `Alt-Enter` works everywhere; `Shift-Enter` needs a terminal that supports the kitty keyboard protocol (kitty, WezTerm, Ghostty, foot, recent Alacritty), because the older input protocol can't tell `Shift-Enter` apart from `Enter` at all |
| `↑` / `↓` | Recall previous messages into the input box |
| `Tab` | Complete a slash command being typed: fills in as much of the name as every match shares, then steps through the matches one press at a time. Only ever touches a command *name* — in a message, in a path, or once you're onto a command's arguments, it does nothing |
| `$ <command>` | Run a shell command yourself, in the clanker's directory. No model call, no tokens, no approval — you typed it. Output appears in its own box and waits for you to decide whether the model should see it |
| `Ctrl-S` / `Ctrl-D` | Send that output to the conversation, or discard it. Sending waits for your next message rather than prompting a reply. Different keys from the approval's on purpose, since both boxes can be open at once. Use `/send` and `/discard` where something has claimed the chord — Zed's terminal does |
| `Ctrl-Y` / `Ctrl-N` | Allow or deny a tool approval. It gets its own box above the prompt rather than taking the prompt over, so you can keep typing — and keep sending — while a decision waits, which is why it's a chord rather than a bare `y`/`n`. Use `/allow` and `/deny` where the chords are claimed; without a way to answer, a turn waits forever |
| `PgUp` / `PgDn` / `End` | Scroll the transcript; `End` re-pins to the newest |
| Mouse wheel | Also scrolls the transcript — `↑`/`↓` stay dedicated to prompt history |
| `Ctrl-Shift-V` / `Shift-Insert` / middle-click | Paste, using your terminal's own paste binding. Multi-line pastes land in the input box as text rather than sending a message per line. `Ctrl-V` is **not** a paste key in most terminals — it never reaches your clipboard |
| `Ctrl-B` | Back to the launch screen (the clanker is saved). A turn still running is **not** cancelled — it keeps working, shows as `working` in the list, and Enter on its row puts you back in it where it got to. Use `Esc` first if you meant to stop it. `/back` does the same — tmux takes `Ctrl-B` as its own prefix |
| `Ctrl-C` | Quit. This ends a running turn: the work happens inside this process, so nothing survives it leaving |

**Backing out of a running turn.** `Ctrl-B` leaves the screen, not the turn.
The clanker keeps working — and keeps its place in `clank`, transcript and
all — so the launch screen becomes a monitor for it: its row animates as
`working` with the tool it is running on the line beneath, and `?` when it
wants a decision from you. Press Enter on that row and you are back in the
clanker exactly where it got to, mid-turn, with the approval box waiting and
the input box live. Answer it, steer it, or back out again.

That is why the turn is not cancelled on the way out: its tool calls have
already touched your files, and a turn is only recorded once it ends, so
stopping it halfway would leave a clanker whose history disagrees with what
was done. Several clankers can be working at once this way — back out of one,
start or resume another, and both keep going.

Three things worth knowing:

- **It is still claimed while it works.** Your `clank` is holding it, so
  another terminal cannot open it, exactly as before. What changed is that
  *this* `clank` can hand the screen back, because it never let go.
- **A clanker that finishes while you are away is let go of** — its claim is
  released, so it opens the ordinary way (and can be opened from anywhere)
  once its row says `✓`.
- **Quitting is not backing out.** All of this lives inside `clank`, so
  `Ctrl-C` — and closing the terminal — ends every turn wherever it has got
  to, and what they had done is lost from the record. Clankers that outlive
  the process are a different feature and do not exist yet.

**Commands.** Type these in the message box instead of a message:

| Command | Does |
|---|---|
| `/help` | List every in-clanker command and what it does. The same list in both front ends, generated from the one the parser uses, so it cannot drift from what actually works |
| `/models` | Browse the models the endpoint offers and pick one. Type to filter, arrows to move, Enter to set, Esc to cancel. The list is fetched when you ask for it, so it reflects what the endpoint has now |
| `/model <name>` | Switch the model for the rest of the clanker, and remember it |
| `/model` | Show the model currently in use |
| `/effort` | Show the reasoning effort level currently in use |
| `/effort <level>` | Switch reasoning effort for the rest of the clanker, and remember it |
| `/effort clear` | Nullify it — no effort field is sent at all until set again |
| `/effort default` | Read the *currently* configured default effort and save that to the clanker |
| `/verbose <on\|off>` | Show the model's thinking and full tool call arguments/results, or a one-line notice per call. Bare `/verbose` shows the current setting |
| `/stream <on\|off>` | Stream this clanker's replies token-by-token, or wait for the whole reply. Bare `/stream` shows the current setting. Overrides `clank stream` for this clanker |
| `/max-iterations <n>` | Switch the tool-calling iteration cap per turn (only matters when it has tools), and remember it |
| `/max-iterations clear` | Nullify it — a clanker with tools then errors on any turn until a cap is set again |
| `/max-iterations default` | Read the *currently* configured default cap and save that to the clanker |
| `/temperature <n>` (or `/temp <n>`) | Switch the sampling temperature for the rest of the clanker, and remember it |
| `/temperature clear` (or `/temp clear`) | Nullify it — requests are then sent with no temperature field |
| `/temperature default` (or `/temp default`) | Read the *currently* configured default temperature and save that to the clanker |
| `/temperature` (or `/temp`) | Show the temperature currently in use |
| `/tools <ask\|allow\|never> <tool\|category\|all>` | Switch what a tool may do for the rest of the clanker, and remember it. Takes effect immediately — including partway through a running turn, from its next tool call |
| `/tools on` / `/tools off` | Tools on, as `clank tools` allows them, or every tool off |
| `/tools` | List every tool and what it may do |
| `/sandbox <on\|off>` | Confine the agent's file writes to the working directory, or allow them anywhere. Takes effect immediately, including partway through a running turn |
| `/sandbox` | Show whether writes are currently confined |
| `/status` | Show every setting this clanker is running with — model, effort, temperature, iteration cap, sandbox, verbose, highlighting, streaming, what each tool may do, and the directory it runs in. The clanker-scoped counterpart to `clank status` |
| `/highlight <on\|off>` | Band your own messages in the transcript, or don't. Bare `/highlight` shows the current setting |
| `/clanker title <new title>` | Rename this clanker. Bare `/clanker` (or `/clanker title`) shows its current name |

Only recognized commands are intercepted — including a *mistyped* one.
`/tools maybe read`, or a bare `/effort` with no value, is reported as an
error rather than sent to the model, since a line naming a known command is
confidently meant as one. So is a misspelled command name: `/mode gpt-5`
answers with `Did you mean /model?` instead of quietly asking the model
about it.

A message that merely starts with a slash is still sent as normal text
whenever it isn't close to a command — paths (`/etc/hosts`), and words that
merely extend a command name (`/verbosely`), both go through untouched.

**The box shows you which it is before you send.** As you type, the leading
`/command` turns cyan the moment it names a real one — the name only, not
its argument, so `/effort` stays lit while you type the level after it. A
path, a half-typed name, or a misspelling stays the colour of ordinary text.
It answers "is this a command as spelled?", not "will this parse" — a lit
`/tools maybe read` still comes back as a usage error, and an unlit
`/mdoel` is still caught as a typo when you send it.

**And a row above it says what you could be typing.** While the name is
still ambiguous the row lists every command that starts with what's there
(`/m` → `models model max-iterations`), with a `+N` when more match than the
row can hold; `Tab` fills in what they all share and then steps through them,
marking the one it has landed on. Once the name is settled the row turns into
that command's form — `/tools [on|off | <ask|allow|never> <target>]` — for
as long as you're typing its arguments. Nothing appears above an ordinary
message.

Both are **TUI only**: `clank clanker` reads a line at a time and can neither
restyle what has been printed nor draw above the line you are on.

All of the above persist to the clanker, so they stick across
`Ctrl-B`/`--resume` too.

A clanker records the directory it was started in. Resuming moves the process back into it, because that directory is the sandbox's boundary and what the clanker's relative paths resolve against — resuming somewhere else would silently rebind both to wherever your shell happened to be. If the directory no longer exists, resuming stops and says so — the CLI with an error, the TUI by asking whether to resume here instead and repoint the clanker. `clank clanker --resume <id> --here` resumes in the current directory and repoints the clanker, for a project that moved. Clankers saved before this was tracked have no recorded directory and resume wherever they're run, as they always did.

Clankers are saved exactly as the other commands save them, so a `tui`
clanker can be resumed with `clank clanker --resume` and vice versa — mode,
model, and effort level all carry over either way. A clanker is kept from the
moment you name it, whether or not anything is ever said in it — naming one
is the deliberate act of starting it. (Clankers created before names were
required can still be untitled; one of those is discarded if you open it and
leave without saying anything, rather than leaving an empty "Untitled" in
your list.)

#### `stream [on|off]`
Whether replies stream in as they're generated. On by default. Turn it off for
providers that handle streaming — particularly streaming alongside tool calls —
badly; the CLI then waits for the whole reply as it used to.

```bash
clank stream          # show the current setting
clank stream off      # wait for complete replies
clank stream on
```

#### `timeout [name] [seconds]`
How long the client waits, in four places. Bare `clank timeout` shows them all.

```bash
clank timeout                      # show all four
clank timeout stream-idle          # show one
clank timeout stream-idle 180      # set it
```

| | Default | Bounds |
|---|---|---|
| `connect` | 20s | Connecting: DNS, TCP and TLS. Independent of how long the provider then takes to answer |
| `request` | 300s | A whole non-streaming reply. It has no partial progress to show, so it gets one generous ceiling |
| `stream-idle` | 90s | The gap *between* streamed chunks. A long reply legitimately keeps sending, so there is no total ceiling — this catches a connection that has stalled rather than a model still thinking |
| `command` | 30s | A terminal command the agent runs, when the model names no `timeout_secs` of its own |

`stream-idle` is the one worth raising behind a slow provider: it is what
ends a turn that was still coming. A timeout of `0` is refused, since it
would fail every call before it started.

#### `clankers`
List, inspect, or delete your saved clankers — the same ones the launch
screen shows, whether they were started with `clank clanker`, the
full-screen UI, or a one-off run kept with `--save`.

```bash
# List all saved clankers (id prefix, tools, state, model, title)
clank clankers list
#   a1b2c3d4  [🔨]     working   openrouter/auto  Fix the Windows build
#             run_terminal_command: cargo test --all
#   b2c3d4e5  [💬]     replied   openrouter/auto  Notes on the picker

# Show a clanker's full message history
clank clankers show a1b2c3d4

# Delete a saved clanker
clank clankers delete a1b2c3d4
```

The state column is the same one the launch screen shows, from the same
derivation: `working`, `approval`, `failed`, `stopped`, `replied`, `no reply`
or `new`. A clanker waiting on an approval also prints what it is asking
about on the line beneath. Without the TUI's launch screen this is the only
way to see a clanker running in another terminal.

The first three come from the process running the clanker, which is the only
thing that knows them — a turn's messages are written when it *finishes*, so
from storage alone a request in flight looks exactly like a turn that failed.

Those three are only believed while that process is still there to back them
up. A running clanker re-stamps a heartbeat every few seconds; if the stamps
stop for longer than half a minute, whatever it last claimed is ignored and
the state is read from the messages instead. That is what stops a detached run
killed by a `kill -9`, an OOM or a reboot from leaving a row that insists it is
`working` for ever — it settles to `no reply`, which is the truth. The window
is deliberately several heartbeats wide: a briefly starved process is not a
dead one, and calling a live run dead is the worse mistake.

## Concepts

Five nouns do most of the work in this codebase, and they nest. Getting them
straight explains why some settings take effect immediately and others wait,
why a running turn is invisible in the database, and where a message typed
mid-turn can legally go.

### The ladder

**Message** — the atom. A role (`user`, `assistant`, `tool`), content, and
optionally `tool_calls` or the `tool_call_id` answering one. Stored as a row
per message, ordered by `seq` within its clanker.

**Request** — one HTTP POST to `/chat/completions`. Stateless, which is the
load-bearing part: it carries the *entire* message array every time, because
the provider remembers nothing between calls. This is the unit that gets
billed, times out, and rejects malformed message shapes.

**Iteration** — one lap of the agent loop: a request, plus running whatever
tools it asked for, plus appending those results to the array. Capped by
`max-iterations`. Only with tools — a clanker without them makes a single request and has
no loop.

**Turn** — one thing you typed, through to a final answer. One request with
no tools; one to `max-iterations` iterations with them, ending when the model
stops asking for tools, the cap is reached, or you cancel. A turn is also the
persistence unit: its messages are written when it *finishes*, which is why a
turn in progress leaves no trace in the messages table and why clankers carry
a separate `activity` column for the picker to read.

**Clanker** — a saved conversation: its messages, plus a title, a model, a
mode, a settings snapshot, and the directory it belongs to. Survives exit,
resumes by id.

So messages make up a request, requests make up an iteration, iterations make
up a turn, and turns make up a clanker.

### Alongside them

**Conversation** — the runtime that drives a clanker: it takes commands,
emits events, and runs the agent loop on its own task so the interface stays
responsive while a turn works. The clanker is the state; the conversation is
the thing moving it. It exists only while the process runs, and nothing stops
two processes from driving one clanker — see the note in TODO.

**Activity** — the only thing stored about a turn that hasn't finished:
working, awaiting approval, failed, or null for "nothing to say, read the
messages."

### Running a command yourself

`$ cargo test` runs it here and now, in the clanker's directory, without
involving the model. There is no approval prompt: you typed it, so there is
nothing to approve.

A box appears as soon as you press Enter. The command sits on its first line
with a spinner after it while it runs, and its output fills in underneath:

```
┌──────────────────────────────────────────────────┐
│$ cargo test ⠹                                    │
└──────────────────────────────────────────────────┘

┌ Ctrl-S send with next message · Ctrl-D discard ──┐
│$ cargo test                                      │
│running 319 tests                                 │
│test result: ok. 319 passed; 0 failed             │
└──────────────────────────────────────────────────┘
```

A non-zero exit shows in red beside the command. The border carries only what
you can act on.

`Ctrl-S` puts the output into the conversation; `Ctrl-D` leaves it out. Either
way the command stays in the transcript, marked sent or not — the decision is
about what the *model* sees, not what you see.

**Sending does not prompt a reply.** The output is added and waits, so it
reaches the model together with whatever you type next. That is the point of
the feature: `$ cargo test`, send, "fix these failures" is one turn, where
replying to bare output first would cost two. Sending during a turn joins that
turn, the same as any message.

Running a second command before answering the first drops that output from the
conversation, but it stays in the transcript marked `not sent` — nothing you
were deciding about disappears without a trace. Output the model never saw is
also **dimmed**, the same grey an unset setting wears, so scrolling back you
can see which results the conversation is actually working from without
reading the label on every one. A second command *while* one is
still running is refused: there is one box, and two results would land on top
of each other.

`Ctrl-S` was historically XOFF, the key that freezes a terminal until
`Ctrl-Q`. It works here because raw mode turns software flow control off — but
anything layered above the terminal can still claim it, which is what `/send`
and `/discard` are for. If a terminal ever does lock up, `Ctrl-Q` releases it.

**Commands get no stdin.** Anything that wants input — `sudo` without a
cached credential, `git commit` with no `-m`, a script calling `read` — gets
end-of-file immediately and fails with its own error, rather than blocking
until it is killed. Use another terminal for anything interactive.

Output is capped, keeping the end rather than the beginning, since a failing
build says what went wrong on its last lines. Commands are killed after 30
seconds, and `$` is TUI-only for now.

### Sending while a turn is running

You do not have to wait for a turn to finish before typing. What happens next
depends on whether there is a loop to join.

When the clanker **has tools**, the message joins the turn already running. A turn is many
requests, and the message array for each one is built fresh, so the loop takes
whatever you have typed at the top of its next iteration and includes it in
that request. The model sees it before deciding what to do next, which means
you can redirect work in progress — "actually, skip the tests", "check the
Windows path too" — rather than waiting for it to finish and correcting it
afterwards.

The timing is bounded by what the turn is doing. If a tool call is running,
your message lands as soon as that call finishes and the loop comes back
around, because that is the first legal place to put it: a message carrying
tool calls must be followed by their results, and nothing may come between.
It never interrupts a request already in flight — there is no such thing in
an OpenAI-compatible API.

With **no tools** there is one request and no loop, so there is no seam to
inject into. The message waits and becomes its own turn when the current one
finishes. The same fallback applies with tools if the turn ends before the
loop takes what you typed — the iteration cap was reached, the turn failed,
or you cancelled.

A box appears above the message input as soon as something is waiting, and
disappears when the last one leaves. It lists the messages in the order they
will be taken, and its title says where they are headed — `joining this turn`
with tools, `next turn` without them:

```
╭─ joining this turn ────────────────────────╮
│check the Windows path too                  │
│and skip the slow tests                     │
╰────────────────────────────────────────────╯
╭────────────────────────────────────────────╮
│                                            │
╰────────────────────────────────────────────╯
 ⠋ working · agent · sonnet-5 · 🧠 high
```

Each message drops out of the box as it is consumed and appears in the
transcript at the point it actually joined, so the conversation reads in the
order the model saw it. Past five waiting, the rest are summarised as
`+N more`. Cancelling a turn with `Esc` drops anything still waiting.

### Where a setting lives

| Scope | Changed by | Read |
|---|---|---|
| Global default | `clank <setting> <value>` | when a clanker is created |
| Clanker | `/model`, `/temperature`, … | stored on the clanker row |
| Per-turn snapshot | — | once, when a turn starts |
| Live gates | `/tools`, `/sandbox` | before every tool call |

The last two rows are the distinction worth knowing. Model, effort,
temperature and streaming are snapshotted when a turn starts, so changing one
mid-turn applies to the *next* turn. Approval and sandbox are re-read before
every single tool call, so changing one mid-turn applies to the turn that is
running — which is the entire point, since revoking permission is not much
use if it waits politely for the current work to finish.

## Agentic Tools

A run with tools — `clank "..." --tools`, or a clanker that has them — gives the LLM access to these:

### `write_file`
Write or append content to a file.

### `read_file`
Read the contents of a file.

### `list_files`
List files in a directory.

### `replace_in_file`
Replace text in an existing file.

### `run_terminal_command`
Execute a shell command and return the output. Supports custom working directory and timeout.

**Off by default** — its state is `never`, so it isn't offered to the model
at all until you say otherwise. It's the only tool whose reach isn't bounded
by what it's for: the file tools touch files and the sandbox bounds where,
while a shell command can do anything you can. `clank tools ask
run_terminal_command` turns it on globally, `/tools ask run_terminal_command`
for one clanker.

### `web_fetch`
Fetch an `http`/`https` URL and return the page as readable text rather than HTML.

The agent can already reach the web through `run_terminal_command` — it can run
`curl`. This exists because the raw page is mostly markup: converting first cuts
a documentation page to between a half and a quarter of its size (measured: 4.0×
on docs.rs, 3.8× on MDN, 2.0× on the Rust book), and whatever is fetched stays in
the conversation for the rest of the turn.

**It does not ask for approval by default.** It reads a page and changes
nothing, and a prompt on every page is the friction that would send the model
back to curling raw markup through `run_terminal_command` — which does prompt,
and can then do anything. This is a default rather than an exemption, so
`clank tools ask web_fetch` turns it on if you want it.

Refuses anything that isn't `http` or `https` (notably `file:`, which would read
the disk through a tool the sandbox doesn't cover), refuses content types it
can't read as text, caps a page at 1 MB, and times out after 30 seconds. What
comes back is labelled untrusted: it is the only tool result that originates
neither with you nor with your machine, so a page telling the agent to run
something is an attack, and the approval prompt is what stands in the way.

## Configuration

Configuration is stored at `~/.clank/config.json`. `tools` holds only the
tools whose state differs from the default, so a tool added in a later
version arrives with its own default already in force rather than needing
the file rewritten — and `run_terminal_command` being absent below means it
is `never`, not that nothing was said about it:

```json
{
  "base_url": "https://openrouter.ai/api/v1",
  "default_model": "anthropic/claude-opus-4.5",
  "tools": {
    "read_file": "allow"
  },
  "max_iterations": 20,
  "temperature": 0.7,
  "effort_level": "high",
  "effort_style": "nested",
  "extra_headers": {},
  "sandbox": true,
  "verbose": false,
  "highlight": true,
  "selection": true,
  "stream": true
}
```

- The file is created the first time you change a setting, not on first run — until then every value comes from the defaults above, which `clank status` will show you. You can also write it by hand: any keys you leave out fall back to their defaults, so a file containing only `{"temperature": 1.5}` is valid, and the next `clank` setting command fills in the rest around what you wrote.
- If the file can't be parsed, commands stop with the parse position rather than silently reverting to defaults, and nothing is written over it — a malformed config would otherwise send your API key to the default endpoint instead of the one you configured, and the next setting command would overwrite everything else you'd set. Fix it, or delete it to start over.
- Your API key is **not** in this file — `clank login`/`logout` store and remove it from the OS keychain instead (see [Security](#security)). If you have an old config with a plaintext `api_key` field, the next command that loads config transparently migrates it into the OS keychain and rewrites the file without it.
- `base_url` is managed via `clank endpoint` and is the API endpoint used by every command. Defaults to OpenRouter; point it at any OpenAI-compatible service.
- `default_model` is managed via `clank model` and is used by `ask`, `clanker`, and `agent` when `-m`/`--model` isn't passed, and always by `tui`, which has no flags at all.
- `tools` settings control what the agent may do, and what it may do without asking. Managed via `clank tools`.
- `max_iterations` is managed via `clank max-iterations` and is the default for `clanker` and one-off runs when `--max-iterations` isn't passed, and for `tui`, which has no flags at all. `null` (after `clank max-iterations --clear`) means a clanker with tools has no cap until one is set somewhere — it does not fall back to 20.
- `temperature` is managed via `clank temperature` and is the default for `ask`, `clanker`, and `agent` when `--temperature` isn't passed, and for `tui`, which has no flags at all. `null` (after `clank temperature --clear`) means requests are sent with no `temperature` field at all — it does not fall back to 0.7.
- `verbose` is managed via `clank verbose` and is the value new clankers start with; `/verbose` changes the clanker you're in, not this.
- `highlight` is managed via `clank highlight` and is the value new clankers start with for banding your own messages; `/highlight` changes the clanker you're in, not this.
- `selection` is managed via `clank selection` and controls the band on the launch screen's selected row. Global only — that screen belongs to no clanker, so there is no per-clanker counterpart and no slash command.
- `effort_level` is managed via `clank effort-level` and is sent for `ask`, `clanker`, and `agent` when set, shaped according to `effort_style`.
- `effort_style` is managed via `clank effort-style` and controls whether the effort level is sent flat, nested, or omitted (see [`effort-style`](#effort-style-value)).
- `extra_headers` is managed via `clank headers` and is merged into every API request.

### Using other providers

Clanker Command Center talks to any service exposing an OpenAI-compatible `/chat/completions` and `/models` API over `Authorization: Bearer` auth — this covers OpenRouter, OrcaRouter, Together, Groq, Fireworks, and self-hosted gateways (vLLM, Ollama's OpenAI shim, LM Studio). It does not cover providers with a different auth scheme or URL shape, like Azure OpenAI.

To switch to OrcaRouter, for example:

```bash
clank endpoint https://api.orcarouter.ai/v1
clank login                          # enter your OrcaRouter key
clank model orcarouter/auto          # or any model OrcaRouter serves
clank effort-style flat              # OrcaRouter expects reasoning as a top-level field
```

Only one provider is active at a time today — switching back to OpenRouter means re-running `clank endpoint`, `clank login`, `clank model`, and `clank effort-style` for it. Named provider profiles (switch between saved providers with one command) are tracked in `TODO.md`.

## Clanker Persistence

`clanker` and `tui` conversations are saved automatically to a SQLite database at `~/.clank/chats.db`. Every message (yours, the assistant's, and any tool calls and results) is written as the conversation happens, so you don't lose anything if you exit or your terminal closes — including a turn you cancelled partway through.

**Settings are a snapshot, not a live link to your config.** A clanker's row — model, effort level, max iterations, temperature, what each tool may do — is written to the database the moment it's created, before your first message, not after. `tui` has no flags at all, so a clanker it creates is always a straight snapshot of your persistent config defaults; `clank clanker` is the only place a brand new clanker can start away from those defaults, via its `--model`/`--effort-level`/`--max-iterations`/`--temperature` flags. That snapshot can itself be `None` for effort/max-iterations/temperature, if nothing is configured anywhere — same as `ask`/`agent`, which merge a `--flag` with the config default the same way but only ever for that one call, never a clanker.

From then on, the clanker's settings are entirely its own: `/model` and `/tools` changes always write a concrete value straight back to that same row; `/effort`, `/max-iterations`, and `/temperature` additionally support two different resets, since a clanker can also nullify these three:

- **`/setting clear`** nullifies it outright, with no fallback substituted anywhere: `/effort clear` and `/temperature clear` mean no effort/temperature field is sent in the request at all (the provider uses its own default); `/max-iterations clear` means a clanker with tools has no cap, so any turn that actually needs one fails immediately with an error telling you to set one, rather than the loop running unbounded or guessing a number.
- **`/setting default`** is a one-time snapshot instead: it reads whatever the global default currently is and saves that concrete value to the clanker right now — frozen from that point on, exactly like typing the value itself, and distinct from `clear` even when the global default happens to be unset (an `/effort default` with no global default configured saves `None` explicitly, the same as `clear` would, but as a deliberate choice rather than an indefinite fallback).

Either way, every outgoing request from a clanker reads its own stored settings directly, never your global config — including for a value that's currently `None`. Later changing a global default with `clank model`/`clank temperature`/etc. never reaches into any clanker that already exists, whether that clanker has an explicit value, is nullified, or was created before you ever set the global default at all. The global defaults themselves work the same way: `clank max-iterations --clear`/`clank temperature --clear` null them out too (see [`max-iterations`](#max-iterations-value) and [`temperature`](#temperature-value)), and nothing brings them back except setting one explicitly again.

Each clanker gets an id (a UUID) and a title derived from your first message (or one you choose up front, in the TUI). Use:

- `clank clankers list` to see saved clankers (shown by 8-character id prefix, kind, state, model, and title)
- `clank clankers show <id>` to view a clanker's full transcript
- `clank clankers delete <id>` to remove one
- `clank clanker --resume <id>` to continue a saved clanker — works whether or not it has tools, since that is just what its tools are set to
- `clank clanker --resume` with no id to pick one from a numbered list of all your saved clankers

Any unique prefix of a clanker's id works wherever a full id is expected. A
prefix matching more than one clanker is refused, and the candidates listed
with their titles — `clankers delete` resolves ids the same way, so guessing
between them would eventually delete the wrong conversation.

## Examples

### Generate and save code

```bash
clank "Write a function that calculates fibonacci numbers and save it to math.rs" --tools
```

### Multi-file project setup

```bash
clank "Create a basic Rust project structure with Cargo.toml, src/main.rs, and src/lib.rs" --tools
```

### Fix existing code

```bash
clank "Read main.rs, find any issues, and write a corrected version" --tools
```

### Using different models

```bash
# Claude for code review
clank "Read app.rs and provide detailed code review feedback" --tools -m anthropic/claude-opus-4.5

# GPT-4 for complex logic
clank "Create an algorithm to solve the traveling salesman problem" --tools -m openai/gpt-4o

# Adaptive routing (default)
clank "Generate boilerplate code" --tools -m openrouter/auto
```

## Building for Different Platforms

```bash
# Build for Windows (from macOS/Linux)
rustup target add x86_64-pc-windows-gnu
cargo build --release --target x86_64-pc-windows-gnu

# Build for macOS (from other platforms)
rustup target add x86_64-apple-darwin
cargo build --release --target x86_64-apple-darwin

# Build for Linux (from other platforms)
rustup target add x86_64-unknown-linux-gnu
cargo build --release --target x86_64-unknown-linux-gnu
```

## Troubleshooting

### "API key not configured"

Run `clank login` and enter your key from [openrouter.ai/keys](https://openrouter.ai/keys).

### "Model not found"

Run `clank models` to see available models, then use the correct model ID with `-m`.

### Build errors on macOS

```bash
xcode-select --install
```

### Build errors on Linux

```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# Fedora/RHEL
sudo dnf groupinstall "Development Tools"
```

## Security

- The agent's file-writing tools (`write_file`, `replace_in_file`) are confined to your current working directory by default, checked against the path a write resolves to so `..` and symlinks can't step outside it. Turn it off per clanker with `/sandbox off` or globally with `clank sandbox off`. Reads and terminal commands are not bounded this way — a terminal command runs whatever you approve. This gates the agent's tools only; `clank` writes its own `~/.clank` state directly and is unaffected
- API keys are stored in your OS keychain (macOS Keychain, Windows Credential Manager, or the Linux Secret Service via `keyring`), not in a plaintext file. An older `~/.clank/config.json` with a plaintext `api_key` field is migrated into the keychain automatically the next time you run any `clank` command, and the field is stripped from the file afterward
- `clanker`/`tui` history is stored in `~/.clank/chats.db` with message content, tool calls, reasoning, and titles encrypted at rest (AES-256-GCM, key held in your OS keychain under a separate `db_encryption_key` entry) — but the surrounding clanker metadata (roles, model names, effort levels, timestamps) is stored in the clear, and rows written before encryption existed stay plaintext until they're next written. The key lives in the same keychain `clank` already uses, so this protects the file at rest (backups, drive theft) rather than against someone who can run `clank` as you; avoid pasting secrets into a clanker if you plan to share the database file
- The last 100 LLM API errors (a non-2xx response, a stalled/dropped connection, a malformed stream) are kept at `~/.clank/errors.log`, so a confusing one can be looked back at without having to catch and copy it in the moment — plain text, one line per entry, oldest dropped as new ones come in
- Each of those entries records the shape of the request that failed — role sequence, tool-call and reasoning counts — but no message text. To capture the request itself, set `CLANK_DEBUG_REQUESTS=1`: the failing request's full JSON body is written to `~/.clank/failed-request.json` (only the most recent one, overwritten each time) and the log entry names the file. **That file contains the entire conversation verbatim** — every message, tool call and tool result — so it's off by default, and worth deleting once you're done with it

## Development

To modify the code:

```bash
# Run in debug mode
cargo run -- ask "Hello"

# Run tests
cargo test

# Format code
cargo fmt

# Lint
cargo clippy
```

## Performance

```bash
time clank "Hello"
# real    0m0.015s
```

## License

MIT

## Support

For issues with a specific provider's API itself (rate limits, billing, model availability), see that provider's own docs — e.g. [openrouter.ai/docs](https://openrouter.ai/docs) for the default OpenRouter endpoint.
