
**TODOs**
* connect providers directly, like Anthropic, OpenAI, etc..

Clanker re-design — API surface and theme, cheapest first
* Tutorial option on the picker screen. Last only because it is undefined.
   A static screen of keys and concepts is the smallest thing on this list
   and belongs first; an interactive walkthrough — a scripted clanker that
   talks you through spawning one, approving a tool call, backing out — is
   the largest, needs a screen mode with its own state, and is mostly
   writing content rather than code.

* prompt caching
* what else could be added to verbose mode?
* Live raw request/response screen?
* Skills? implement Agent Skill Standard: agentskills.io

NEXT:
* Vim mode
* $ command UI re-think
  What exists: a box above the prompt showing the command and, once it exits,
  its output, which you then send to the conversation or discard (Ctrl-S /
  Ctrl-D, or /send and /discard). Three things wrong with it, worth designing
  as one rather than patching separately: output is captured rather than
  streamed, so the box sits empty and then fills all at once — fine for a
  quick command, poor for anything long, and fixing it needs a different
  execution path and an event per chunk. A command wanting input on stdin has
  nowhere to type it. And the whole feature is TUI-only, because the CLI's
  blocking prompt loop has no box to put it in, so `$` there would have to
  mean something different — probably run-and-print with no send step.
* change status and /status to config and /config or maybe settings and /settings. In the in-session print out of the session config/settings should say something about how the session config/settings override the global ones
* running on windows resulted in some terminal freezes, especially when logging and launching
* --headless: built, then taken back out again. The implementation is in
  51f4ef7 if it's wanted back — flag on `ask` and `agent`, a refusal when any
  approval gate is on, and a `CLANK_HEADLESS` env marker stopping a headless
  run from launching another. Removed because two of its three legs are built
  wrong, not because the need isn't real:
  - Most of what it does is detectable, not declarable. There is no TTY check
    anywhere in the codebase — the spinner writes `\r` frames unconditionally
    and `colored` does its own isatty internally — so the flag asks you to
    declare something `std::io::IsTerminal` answers for free. Detection should
    come first, with a flag to force either way.
  - Its entry price is a global, persistent safety downgrade: `clank approval
    all off` writes to config, so disabling gates for one detached run leaves
    every later interactive run ungated too. There is no per-run alternative.
    A `--yes` that ungates one invocation and touches nothing else is the
    conventional shape, and is worth having on its own.
  - What is genuinely not inferable, and worth keeping when this comes back:
    refusing up front rather than being denied at every tool call, and the
    nesting marker.
  Also unresolved when it went: the startup check reads the global config
  while a resumed run uses the session's own saved gates, so
  `--headless --resume` could pass the check and then deny everything. And
  the flag is named for its cause (no terminal) rather than its policy (run
  unattended, never ask), which is why "should --headless also block X?" kept
  being an awkward question.
* ability to edit messages in the transcript?
* nothing clank prints is machine-readable, which is what stops it being
  driven by another program. `clank ask` is nearly there — one reply on
  stdout, errors on stderr, non-zero exit, and `colored` drops ANSI when not
  a TTY — but `clank agent` interleaves "Starting agent task...", every tool
  call and its status, and the reply onto one stream with nothing separating
  them, so a caller can check the exit code and nothing else. There is no
  --json on any command. Two ways out, pulling against each other: a --json
  mode emitting one object per run (reply, tool calls, stop reason, later
  token counts), which is what a caller actually wants; or progress to stderr
  and stdout for the result alone, which is the Unix convention and nearly
  free, but contradicts the idea that progress is a record worth keeping in a
  log — logging would become `> log 2>&1` and lose the separation again.
  Probably both. This is a precondition for anything like --headless coming
  back: safe to run unattended is not the same as usable as an API.
* nothing bounds what gets sent to the model. Every turn ships the whole
  history: `truncate_output` (conversation.rs) only clips shell output, and no
  other trimming exists anywhere. Cost and latency grow with session length
  until a long chat overruns the context window outright. Two answers to weigh:
  a sliding window (drop or elide the oldest turns — cheap, predictable, loses
  what it drops) or compaction (summarize older turns into one synthetic
  message — keeps the gist, costs an extra call, and the summary can be wrong
  in ways nothing catches). Probably both eventually, with a threshold that
  picks. Two things have to land first: a token counter, since neither can be
  triggered without knowing where you are (wanted in-session anyway — part of
  verbose?), and a decision about `prompt caching` above, which this collides
  with — a cache hit needs a stable prefix, and both windowing and compaction
  rewrite the front of the history, so the two features have to be designed
  against each other rather than one after the other. Storage is unaffected:
  the DB keeps every message either way, this is only about what each request
  carries. Not the same problem as the TUI render cost, which is local.
* how to handle $'s that need an answer to std in
* is it possible to create a session and not enter the terminal but keep the process open, or create/open a session and back out of the terminal but the process stays open so the agent can keep wokring?
* what happens if you --resume an 'elsewhere' session then ctrl-c? Where does your terminal land to?
* add a clank ask with no "" text and it immediately asks for input from the user and asks with that input
* if `$` gets reverted, keep the stdin fix in tools.rs. `run_terminal_command`
  never set stdin, so the child inherited this process's — for the TUI, a
  terminal in raw mode the event loop is already reading. Any command wanting
  input blocked until the 30s timeout with its prompt trapped in the piped
  stdout, while it and the TUI fought over keystrokes. It is a pre-existing bug
  in the *agent's* tool, not something `$` introduced: a model running
  `sudo apt install` hangs the same way. `.stdin(Stdio::null())` makes those
  fail in milliseconds with their own error instead (`sudo: a password is
  required`, exit 1).
* a steered message is lost if the turn it joined is cancelled. `absorb` runs
  only on the arm where a turn completes (conversation.rs), so cancelling
  discards everything the task accumulated — which is intended, except that a
  steered message is the one *user* message in that set. The turn's opening
  message is pushed onto the session before the task spawns and survives; a
  steered one lives only in the task's copy and does not. On screen you see
  both, in the database only the first, and a resume shows the difference.
  Low severity (explicit cancel only, and the turn was discarded anyway) but
  asymmetric. The obvious fix — have the worker push onto the session when it
  accepts the message — collides with `absorb`, which reconciles by skipping
  the messages the session already has and taking the rest from the task's
  copy; pushing in both places double-counts. Needs thought, not a patch.
* `clank agent --resume` removed for now; `--session` stays. The
  implementation is in 771f556~1 if it comes back. Two things to carry over
  with it rather than reintroduce:
  - The "Ignoring --model: resumed sessions keep their saved settings" notes
    printed *before* the working-directory check that can abort the run, so
    a resume whose directory had been deleted told you about a flag on a run
    that never happened. Print them after the checks.
  - It could not do the interactive pick that `clank session --resume` can:
    `resolve_resume_target` handles the "pick" sentinel for both, but agent's
    flag was declared without `num_args = 0..=1`, so bare `--resume` was a
    clap error and the picker branch was unreachable.
  Also worth deciding when it returns: it called `set_agentic(true)` before
  running anything, so resuming an ask-kind session converted it permanently
  even if the run failed immediately.
* two terminals on one session: the guard landed in b9c2499, the useful half
  of it hasn't. A session is claimed by one process at a time — an atomic
  conditional UPDATE, scoped to an owner token so a revived zombie can
  neither renew nor release someone else's, expiring by itself so a crash
  doesn't lock the session for good. A second process is refused. That was
  the necessary part: without it both write colliding `seq` values and the
  history reloads with its turns shuffled and tool results detached from
  their calls, which providers reject, so the session stops opening at all.
  What's left is being able to *do* something with a held session:
  - Read-only viewing. Open a claimed session, scroll its transcript, no
    input. Not free: `Chat` holds a non-optional `Conversation`, so this
    means an optional worker plus guarding every command path through
    `handle_key` and the event loop — call it 150-250 lines across the TUI,
    the worker and the picker. The picker should badge held sessions too, so
    you know before you open one.
  - Answering an approval for a run you are not attached to. This is the
    real goal, and it is the proper answer to the thing --headless was
    solving badly: a detached run needs its gates off globally *because
    nobody can answer it*. If another terminal can, it doesn't. Needs the
    full `ApprovalRequest` persisted (only three fields — tool_name,
    category, arguments), a response column, and a request id so an answer
    to an earlier prompt can't land on a later one. The holder then has to
    take an answer from either side: easy for the TUI worker, which already
    awaits a oneshot and can select against a poll; awkward for the CLI,
    whose approval blocks on a stdin read that would have to become
    selectable. A pending answer should expire with the claim.
  - UNIQUE(session_id, seq) on messages, so the collision is unpersistable
    even if the claim is somehow bypassed — defence in depth rather than
    prevention alone. Now that persist_pending is one transaction a
    violation rolls the batch back cleanly. Costs a table rebuild (SQLite
    can't add a constraint in place) and any database already holding
    duplicates has to have them resolved before the index will build.
* picker refresh decrypts every session title and every session's last message
  every 2s, regardless of how many rows are on screen. Fine now — it scales with
  how many sessions you've accumulated, not with the screen — but that's the
  thing that would eventually want attention. Options if it does: only decrypt
  rows that are visible, or cache by (session id, updated_at) so an unchanged
  session isn't decrypted again.
* the picker renders every row into a fixed area with no scrolling, so once the
  list is taller than the terminal the extra sessions are simply not drawn —
  and there's no indication they exist. The two blank lines between sections
  cost two more rows. Wants a scroll offset that follows the selection, and
  probably some hint that the list continues past the edge.
* East Asian Width Ambiguous glyphs, part done. The transcript's reply
  avatar was `●` (U+25CF), which some terminals draw two cells wide, shifting
  every wrapped line under it out of line with the gutter. Both front ends now
  draw the session's braille mark instead — braille is the one block where
  every pattern is Neutral — so the avatar is fixed. Still Ambiguous and still
  unfixed: `—` on notices and `·` in the settings bar. Safe replacements are
  ✦ ⏺ ◉ ✻ ❖ ⟡ ✧ ❉ ✱ ⌾ ⬤, or anything in the braille block.
* the client's timeouts are hardcoded in client.rs and can't be configured:
  CONNECT_TIMEOUT 20s, REQUEST_TIMEOUT 300s, STREAM_IDLE_TIMEOUT 90s, plus
  tools.rs's 30s default for a terminal command when the model doesn't give one.
  The 90s stream idle one has stalled real turns twice. Would follow the same
  shape as sandbox/verbose: seeded config fields, `clank <name> <value>`.
* project-scoped sessions via a .clank/ folder, like .git: walk up from cwd to
  find it, sessions live there. Bigger than storing working_dir (which is done):
  it changes WHERE state lives. Costs to weigh first — storage splits from one
  global chats.db into many plus a global fallback for sessions started outside
  any project, so you carry both mechanisms; existing sessions need migrating;
  `clank sessions` from outside a project can no longer list everything, which
  matters when you resume by id; conversation history moves inside repos, so it
  gets committed by anyone who doesn't gitignore it (encrypted, but present and
  shareable); and auto-creating .clank/ wherever you happen to run litters
  directories, while requiring `clank init` adds a new concept. The stored
  working_dir is the data you'd migrate from.
* should messages table be expanded to include tool calls, errors, etc.. OR errors logged somewhere
* [agent/ask] [verbose]
 <current os user>: 
* [model] [effort]
  AI:
* confirmation modal for deleting session where you type in name of session

FUTURE:
* android/ios app?
