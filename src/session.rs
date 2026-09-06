//! Conversation state and its persistence, independent of any front end.
//!
//! A [`ChatSession`] owns the message history for one `chat`/`agent-chat`
//! session plus the database handle backing it, and knows how to create,
//! resume, and durably record turns. It does no I/O with the user and holds
//! no opinion about how a conversation is displayed or driven, so the CLI
//! loops and any future GUI can share the same bookkeeping instead of each
//! reimplementing "append, persist, name the session".

use crate::client::ChatMessage;
use crate::config::ToolAccessSettings;
use crate::store::{self, SessionSummary, StoredMessage, KIND_AGENT_CHAT, KIND_CHAT};
use anyhow::Result;
use rusqlite::Connection;
use std::path::Path;

/// Keeps a session's `heartbeat` column fresh for as long as this is held,
/// and gives up the claim when it's dropped.
///
/// Exists because an `activity` is a claim about a *live process* — "a
/// request is in flight", "somebody is being asked a question" — written by
/// a process that then has to survive long enough to take it back. A run
/// killed by an OOM, a reboot or a `kill -9` never does, and the row goes on
/// insisting it is working for ever. Nothing was watching a detached run, so
/// nothing corrected it either.
///
/// A ticking timestamp fixes that without needing to identify the process:
/// no PIDs (which get reused), no platform-specific liveness check. If the
/// stamp is fresh, someone is there; if it stopped, they aren't, whatever
/// their activity last claimed. See [`store::heartbeat_is_live`].
///
/// Ticks on its own task rather than from the turn loop, because the state
/// that most needs to be believed — waiting on an approval — is exactly the
/// one where the loop is blocked and doing nothing.
pub struct Heartbeat {
    conn: Connection,
    session_id: String,
    /// Identifies this claim, so renewing and releasing only ever touch a
    /// claim this process actually holds. Random per claim rather than a
    /// PID, which the OS reuses.
    owner: String,
    /// `None` when there is no Tokio runtime to tick on. The claim is still
    /// real — it just expires on its own instead of being renewed.
    ticker: Option<tokio::task::JoinHandle<()>>,
}

impl Heartbeat {
    /// Takes the session for this process, or reports that someone else has
    /// it. `Ok(None)` means a live claim is already held.
    ///
    /// The claim is a single conditional write, so it cannot be split into a
    /// check and a take. Only the renewal below needs a runtime, which is why
    /// the claim itself is attempted unconditionally: a test without a
    /// runtime still gets a real claim, and simply lets it lapse.
    pub fn claim(session_id: String) -> Result<Option<Self>> {
        let owner = uuid::Uuid::new_v4().to_string();
        let conn = store::open_db()?;
        if !store::claim_session(&conn, &session_id, &owner)? {
            return Ok(None);
        }

        // Its own handle: this writes on a timer, from a task, while the
        // caller's connection is busy with whatever the turn is doing.
        let ticker = if tokio::runtime::Handle::try_current().is_ok() {
            let ticking = store::open_db()?;
            let id = session_id.clone();
            let mine = owner.clone();
            Some(tokio::spawn(async move {
                let mut interval = tokio::time::interval(store::HEARTBEAT_INTERVAL);
                loop {
                    interval.tick().await;
                    match store::renew_session_claim(&ticking, &id, &mine) {
                        // Starved past the stale window, and the session has
                        // been taken by someone else. Stop renewing rather
                        // than stamping over a claim that is no longer ours.
                        Ok(false) => break,
                        Ok(true) => {}
                        // A failed write proves nothing; the next tick may
                        // well succeed, and the cost of being wrong for one
                        // interval is a row that briefly looks abandoned.
                        Err(_) => {}
                    }
                }
            }))
        } else {
            None
        };

        Ok(Some(Heartbeat {
            conn,
            session_id,
            owner,
            ticker,
        }))
    }
}

impl Heartbeat {
    /// The token identifying this claim, for binding a session's writes to it.
    pub fn owner(&self) -> &str {
        &self.owner
    }
}

impl Drop for Heartbeat {
    fn drop(&mut self) {
        if let Some(ticker) = &self.ticker {
            ticker.abort();
        }
        // Best-effort, and only an optimisation: it makes a clean exit
        // register at once instead of after the staleness window. The exits
        // this whole mechanism exists for never reach here at all. Scoped to
        // our own claim, so a late exit cannot release someone else's.
        let _ = store::release_session_claim(&self.conn, &self.session_id, &self.owner);
    }
}

/// A fresh session id.
///
/// Here rather than inline at each caller because the TUI generates one
/// *before* it creates anything: a session's mark is hashed from its id, so
/// rolling a new id is how you get a different mark, and the naming screen
/// lets you keep rolling until you like the one you are about to spawn.
pub fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// The title a session carries until it is given a real one. Compared
/// against rather than a flag, because a flag has to be reconstructed
/// whenever a session is reopened and the stored title does not.
pub const UNTITLED: &str = "Untitled";

pub struct ChatSession {
    conn: Connection,
    id: String,
    /// The claim this session's writes belong to, when the caller took one.
    ///
    /// A process starved past the staleness window loses the session to
    /// whoever claims it next. It has no way to notice on its own — the turn
    /// carries on and persists at the end of it, into a session someone else
    /// is now writing to, which is the `seq` collision the claim exists to
    /// prevent, reached the long way round. Checked inside the write
    /// transaction so the answer cannot go stale before the insert.
    ///
    /// `None` for sessions built without a claim, which is every test and no
    /// real caller; those write unchecked, as they did before.
    claim_owner: Option<String>,
    /// The session's current title. "Untitled" until [`Self::persist_pending`]
    /// derives a real one from the first user message.
    title: String,
    model: String,
    /// `KIND_CHAT` or `KIND_AGENT_CHAT`, kept in step with the tools by
    /// [`Self::sync_kind`]. A cache of what [`Self::is_agentic`] derives, so
    /// a row read without being opened — the launch screen, `clank clankers
    /// list` — can say the same thing.
    kind: String,
    effort_level: Option<String>,
    /// Whether this session's TUI view currently shows verbose tool detail.
    /// Purely a display setting — the agent loop never reads it.
    verbose: bool,
    highlight: bool,
    /// This session's tool-calling iteration cap per turn, while in agent
    /// mode. Starts as a snapshot of the configured default (merged with any
    /// `--max-iterations` given at creation), mutable from inside it with
    /// `/max-iterations <n>`. `/max-iterations clear` nullifies it back to
    /// `None` — turns then fall back to whatever the configured default is
    /// *at the time each one runs*, not frozen to what it was at creation or
    /// at the last explicit set. `/max-iterations default` is the concrete
    /// counterpart: it reads the currently configured default once and
    /// saves that as a new explicit `Some` value, same as typing the number
    /// itself.
    max_iterations: Option<usize>,
    /// This session's sampling temperature — same deal as `max_iterations`.
    temperature: Option<f32>,
    /// Running total of tokens spent across this session's turns. `0` for a
    /// new session and for one written before this was tracked — see
    /// [`crate::store::SessionSummary::total_tokens`].
    total_tokens: i64,
    /// What each tool may do in this session, always concrete (unlike
    /// `effort_level`/`max_iterations`/`temperature` there's no "unset,
    /// defer to config" state once a turn actually needs to check them).
    tool_access: ToolAccessSettings,
    /// Whether this session confines the agent's file writes to the working
    /// directory. Always concrete, like `tool_access` — a tool about to write
    /// needs a yes or no, not "defer to config".
    sandbox: bool,
    /// Whether this session streams replies token-by-token. Snapshotted from
    /// the configured default at creation, like `sandbox`.
    stream: bool,
    /// The directory this session was started in — the sandbox's boundary
    /// and what its relative paths resolve against. `None` for a session
    /// recorded before this was tracked.
    working_dir: Option<String>,
    messages: Vec<ChatMessage>,
    /// Whether the session has been given a title derived from a user
    /// message yet. Sessions start as "Untitled".
    title_set: bool,
    /// How many of `messages` have been written to the database. Everything
    /// from here on is pending; see [`ChatSession::persist_pending`].
    saved_len: usize,
}

/// What moving into a session's recorded directory did.
pub enum EnteredDir {
    /// The process moved into the session's directory.
    Moved(String),
    /// Already there, or the session recorded no directory — sessions
    /// written before that was tracked resume wherever they're run, as they
    /// always did.
    Unchanged,
    /// The recorded directory is gone. The caller decides what to do: the
    /// session's sandbox boundary can't be honoured, so continuing means
    /// running against whatever directory happens to be current.
    Missing(String),
}

/// Moves the process into the session's recorded working directory.
///
/// The directory is the sandbox's boundary and what the session's relative
/// paths resolve against, so a session resumed somewhere else is bounded by
/// wherever the shell happened to be — which is not what anyone means by
/// resuming it. Moving the process is what keeps the boundary a property of
/// the session rather than of the terminal.
pub fn enter_working_dir(session: &ChatSession) -> Result<EnteredDir> {
    let Some(recorded) = session.working_dir() else {
        return Ok(EnteredDir::Unchanged);
    };
    let recorded = recorded.to_string();
    if !Path::new(&recorded).is_dir() {
        return Ok(EnteredDir::Missing(recorded));
    }
    if std::env::current_dir().is_ok_and(|cwd| cwd == Path::new(&recorded)) {
        return Ok(EnteredDir::Unchanged);
    }
    std::env::set_current_dir(&recorded)?;
    Ok(EnteredDir::Moved(recorded))
}

impl ChatSession {
    /// Starts a new session, registering it in the database. `effort_level`,
    /// `max_iterations`, and `temperature` are each a snapshot of the
    /// configured default at creation time (already merged with any
    /// `--flag` the caller was given) — like `tool_access`, they're written
    /// immediately rather than re-resolved on every resume. Any of the three
    /// can already be `None` here, if nothing is configured anywhere; that's
    /// still a real snapshot, not "unset by omission".
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        conn: Connection,
        // See `new_id`: the caller chooses it so the TUI can show what it
        // will look like before creating anything.
        id: String,
        model: String,
        kind: &str,
        effort_level: Option<String>,
        max_iterations: Option<usize>,
        temperature: Option<f32>,
        tool_access: ToolAccessSettings,
        sandbox: bool,
        verbose: bool,
        highlight: bool,
        stream: bool,
        working_dir: Option<String>,
    ) -> Result<Self> {
        store::create_session(
            &conn,
            &id,
            &model,
            kind,
            effort_level.as_deref(),
            max_iterations.map(|n| n as i64),
            temperature.map(|n| n as f64),
            &tool_access,
            sandbox,
            verbose,
            highlight,
            stream,
            working_dir.as_deref(),
        )?;
        Ok(ChatSession {
            conn,
            id,
            title: UNTITLED.to_string(),
            model,
            kind: kind.to_string(),
            effort_level,
            verbose,
            highlight,
            max_iterations,
            temperature,
            tool_access,
            sandbox,
            stream,
            working_dir,
            total_tokens: 0,
            messages: Vec::new(),
            title_set: false,
            saved_len: 0,
            claim_owner: None,
        })
    }

    /// Reopens a saved session. `model` overrides the model it was created
    /// with (a `--model` flag on resume); pass `summary.model` to keep it.
    /// Every other setting comes straight off `summary`, the session's own
    /// persisted value; there's no config fallback to re-resolve here — a
    /// `None` `max_iterations`/`temperature` stays `None` (nullified, or a
    /// row written before this was tracked; either way a caller resolves it
    /// against the configured default per turn, not here).
    ///
    /// Returns the stored history alongside the session so a caller can
    /// render the prior transcript — including the per-message model/effort
    /// each turn was produced with, which the in-memory history drops.
    pub fn resume(
        conn: Connection,
        summary: &SessionSummary,
        model: String,
    ) -> Result<(Self, Vec<StoredMessage>)> {
        let history = store::load_messages(&conn, &summary.id)?;
        let messages: Vec<ChatMessage> = history.iter().map(|sm| sm.message.clone()).collect();
        // A session keeps its name across a reopen, so the stored title is
        // the evidence — not whether anyone has spoken in it yet. Inferring
        // it from the messages made a named-but-empty session look unnamed
        // the moment it was reopened, which both deleted it on the way out
        // and let the first message overwrite the name with a derived one.
        let title_set = summary.title != UNTITLED || messages.iter().any(|m| m.role == "user");
        let saved_len = messages.len();

        // Resuming with a different model (a `--model` flag) is a real
        // switch, not a one-off override: record it so the session reports
        // what it's actually using and doesn't silently revert next time.
        if model != summary.model {
            store::set_session_model(&conn, &summary.id, &model)?;
        }

        Ok((
            ChatSession {
                conn,
                id: summary.id.clone(),
                title: summary.title.clone(),
                model,
                kind: summary.kind.clone(),
                effort_level: summary.effort_level.clone(),
                verbose: summary.verbose,
                highlight: summary.highlight,
                max_iterations: summary.max_iterations.map(|n| n as usize),
                sandbox: summary.sandbox,
                stream: summary.stream,
                working_dir: summary.working_dir.clone(),
                temperature: summary.temperature.map(|n| n as f32),
                tool_access: summary.tool_access.clone(),
                total_tokens: summary.total_tokens,
                messages,
                title_set,
                saved_len,
                claim_owner: None,
            },
            history,
        ))
    }

    /// Switches the model for subsequent turns and records it. Messages
    /// already sent keep the model they were produced with, since each row
    /// carries its own.
    pub fn set_model(&mut self, model: String) -> Result<()> {
        if self.model == model {
            return Ok(());
        }
        store::set_session_model(&self.conn, &self.id, &model)?;
        self.model = model;
        Ok(())
    }

    /// Switches between plain and agent (tool-calling) mode and records it,
    /// so the switch sticks on resume and in `sessions list`. Doesn't touch
    /// message history either way: the agent system prompt isn't stored —
    /// some providers (Anthropic among them) require a `system`-role
    /// message to sit at the very start of the conversation, and `/tools`
    /// can flip this on at any point mid-conversation, so there's no
    /// position here that's guaranteed to stay valid as the conversation
    /// grows around it. `agent::request_turn` prepends it fresh on every
    /// turn that actually needs it instead.
    /// Whether this session has tools at all.
    ///
    /// Derived rather than stored: having tools *is* having at least one
    /// tool that is not `never`, and a flag beside that would be a second
    /// answer to the same question, free to disagree with it. The `kind`
    /// column is kept in step by [`Self::set_tool_access`] as a cache, for
    /// the listings that read a row without opening it.
    pub fn is_agentic(&self) -> bool {
        self.tool_access.any_tools()
    }

    /// Writes `kind` to match the tools, so a row read without being opened
    /// — the launch screen, `clank clankers list` — says the same thing the
    /// session would.
    fn sync_kind(&mut self) -> Result<()> {
        let kind = if self.is_agentic() {
            KIND_AGENT_CHAT
        } else {
            KIND_CHAT
        };
        if self.kind == kind {
            return Ok(());
        }
        store::set_session_kind(&self.conn, &self.id, kind)?;
        self.kind = kind.to_string();
        Ok(())
    }

    /// Switches the reasoning effort for subsequent turns and records it.
    /// `None` clears the override, falling back to whatever the configured
    /// default is next time the session is opened.
    pub fn set_effort_level(&mut self, effort_level: Option<String>) -> Result<()> {
        if self.effort_level == effort_level {
            return Ok(());
        }
        store::set_session_effort_level(&self.conn, &self.id, effort_level.as_deref())?;
        self.effort_level = effort_level;
        Ok(())
    }

    /// Toggles whether this session's TUI view shows verbose tool detail,
    /// and records it so it's remembered on resume.
    pub fn set_verbose(&mut self, verbose: bool) -> Result<()> {
        if self.verbose == verbose {
            return Ok(());
        }
        store::set_session_verbose(&self.conn, &self.id, verbose)?;
        self.verbose = verbose;
        Ok(())
    }

    /// Whether this session's TUI view currently shows verbose tool detail.
    /// Whether this session bands your own messages.
    pub fn highlight(&self) -> bool {
        self.highlight
    }

    /// Switches it and records it, so a resume comes back the same.
    pub fn set_highlight(&mut self, highlight: bool) -> Result<()> {
        if self.highlight == highlight {
            return Ok(());
        }
        store::set_session_highlight(&self.conn, &self.id, highlight)?;
        self.highlight = highlight;
        Ok(())
    }

    pub fn verbose(&self) -> bool {
        self.verbose
    }

    /// Switches the tool-calling iteration cap per turn (only used when it
    /// has tools)
    /// and records it. `None` nullifies it (`/max-iterations clear`) — a
    /// turn then falls back to whatever the configured default is when it
    /// actually runs, not to anything frozen here.
    pub fn set_max_iterations(&mut self, max_iterations: Option<usize>) -> Result<()> {
        if self.max_iterations == max_iterations {
            return Ok(());
        }
        store::set_session_max_iterations(&self.conn, &self.id, max_iterations.map(|n| n as i64))?;
        self.max_iterations = max_iterations;
        Ok(())
    }

    /// This session's `/max-iterations` override, if one is set.
    pub fn max_iterations(&self) -> Option<usize> {
        self.max_iterations
    }

    /// Switches the sampling temperature for subsequent turns and records
    /// it. `None` nullifies it, same deal as [`Self::set_max_iterations`].
    pub fn set_temperature(&mut self, temperature: Option<f32>) -> Result<()> {
        if self.temperature == temperature {
            return Ok(());
        }
        store::set_session_temperature(&self.conn, &self.id, temperature.map(|n| n as f64))?;
        self.temperature = temperature;
        Ok(())
    }

    /// This session's `/temperature` override, if one is set.
    pub fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    /// Total tokens spent across every turn this session has run.
    pub fn total_tokens(&self) -> i64 {
        self.total_tokens
    }

    /// Adds to this session's running token total and records it
    /// immediately — there's nothing to snapshot per turn here, unlike
    /// `temperature`/`max_iterations`: it's a running count each turn
    /// contributes to, not a setting a turn starts with.
    pub fn add_tokens(&mut self, tokens: i64) -> Result<()> {
        if tokens == 0 {
            return Ok(());
        }
        store::add_session_tokens(&self.conn, &self.id, tokens)?;
        self.total_tokens += tokens;
        Ok(())
    }

    /// Binds this session's writes to a claim, so they stop if it is lost.
    pub fn writes_under_claim(&mut self, owner: String) {
        self.claim_owner = Some(owner);
    }

    /// Switches what this session's tools may do, and records it.
    pub fn set_tool_access(&mut self, tool_access: ToolAccessSettings) -> Result<()> {
        if self.tool_access == tool_access {
            return Ok(());
        }
        store::set_session_tool_access(&self.conn, &self.id, &tool_access)?;
        self.tool_access = tool_access;
        self.sync_kind()?;
        Ok(())
    }

    /// What each of this session's tools may do.
    pub fn tool_access(&self) -> &ToolAccessSettings {
        &self.tool_access
    }

    /// Switches whether the agent's file writes are confined to the working
    /// directory, and records it.
    pub fn set_sandbox(&mut self, sandbox: bool) -> Result<()> {
        if self.sandbox == sandbox {
            return Ok(());
        }
        store::set_session_sandbox(&self.conn, &self.id, sandbox)?;
        self.sandbox = sandbox;
        Ok(())
    }

    /// Whether this session confines the agent's file writes to the working
    /// directory.
    pub fn sandbox(&self) -> bool {
        self.sandbox
    }

    /// Switches whether this session streams replies, and records it.
    pub fn set_stream(&mut self, stream: bool) -> Result<()> {
        if self.stream == stream {
            return Ok(());
        }
        store::set_session_stream(&self.conn, &self.id, stream)?;
        self.stream = stream;
        Ok(())
    }

    /// Whether this session streams replies token-by-token.
    pub fn stream(&self) -> bool {
        self.stream
    }

    /// Says what this session's process is doing, for anything watching the
    /// list, or clears it with `None`. Best-effort: failing to announce a
    /// state is no reason to interrupt the turn producing it.
    pub fn set_activity(&self, activity: Option<store::Activity>, detail: Option<&str>) {
        let _ = store::set_session_activity(&self.conn, &self.id, activity, detail);
    }

    /// The directory this session was started in, if it recorded one.
    pub fn working_dir(&self) -> Option<&str> {
        self.working_dir.as_deref()
    }

    /// Repoints this session at `working_dir`, for a project that moved.
    pub fn set_working_dir(&mut self, working_dir: String) -> Result<()> {
        store::set_session_working_dir(&self.conn, &self.id, &working_dir)?;
        self.working_dir = Some(working_dir);
        Ok(())
    }

    /// The session's current title — "Untitled" until the first user
    /// message names it.
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Sets the title explicitly — naming a new session up front, or
    /// renaming an existing one. Marks it as no longer eligible for the
    /// usual derive-from-first-message step, so a later `persist_pending`
    /// won't silently overwrite a name that was actually chosen.
    pub fn set_title(&mut self, title: String) -> Result<()> {
        store::set_session_title(&self.conn, &self.id, &title)?;
        self.title = title;
        self.title_set = true;
        Ok(())
    }

    /// The full session id. The CLI mostly shows [`ChatSession::short_id`],
    /// but the whole id is what a caller needs to address a session later.
    #[allow(dead_code)]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The first 8 characters of the id — what `--resume` and `sessions
    /// list` show, since any unique prefix resolves.
    pub fn short_id(&self) -> &str {
        &self.id[..8]
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// The effort level new turns are recorded with. Callers that already
    /// hold the config value don't need this; one handed a session alone
    /// does.
    pub fn effort_level(&self) -> Option<&str> {
        self.effort_level.as_deref()
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// The history as the agent loop wants it: a `&mut Vec` it can append
    /// assistant and tool turns to. Anything added this way is pending until
    /// [`ChatSession::persist_pending`] runs.
    pub fn messages_mut(&mut self) -> &mut Vec<ChatMessage> {
        &mut self.messages
    }

    /// Appends a message in memory only. Use when several turns will be
    /// written together (an agent turn produces assistant + tool messages).
    pub fn push(&mut self, message: ChatMessage) {
        self.messages.push(message);
    }

    pub fn push_user(&mut self, text: String) {
        self.push(ChatMessage {
            role: "user".to_string(),
            content: Some(text),
            tool_calls: None,
            tool_call_id: None,
            ..Default::default()
        });
    }

    /// The counterpart to [`ChatSession::push_user`]. The agent loop appends
    /// assistant turns itself through `messages_mut`, so this is for callers
    /// driving a conversation directly.
    #[allow(dead_code)]
    pub fn push_assistant(&mut self, text: String) {
        self.push(ChatMessage {
            role: "assistant".to_string(),
            content: Some(text),
            tool_calls: None,
            tool_call_id: None,
            ..Default::default()
        });
    }

    /// Writes every message added since the last call, tagging each with the
    /// model and effort level in force now, and names the session from its
    /// first user message if it doesn't have a title yet.
    ///
    /// Attempts every pending message even if one fails, so a single bad
    /// write doesn't silently drop the rest of a turn; the first error is
    /// returned once the rest have been tried.
    pub fn persist_pending(&mut self) -> Result<()> {
        // All of the pending messages or none of them. Advancing
        // `saved_len` past one that failed to save used to drop it for
        // good, and saving the ones after it left a hole in `seq` — the
        // transcript then loaded cleanly with a turn missing from the
        // middle and nothing anywhere to say so. Silent gaps are worse
        // than a loud failure.
        //
        // Leaving `saved_len` alone on failure is what makes a retry work:
        // `cmd_agent` persists once before the turn and once after, so a
        // user message that could not be written the first time is written
        // by the second attempt, together with the reply it prompted,
        // rather than leaving an answer to a question nobody can see.
        let tx = self.conn.unchecked_transaction()?;

        // Inside the transaction: if this process has lost the session, the
        // turn it just ran belongs to nobody and writing it would collide
        // with whoever holds the session now.
        if let Some(owner) = &self.claim_owner {
            if !store::claim_is_held_by(&tx, &self.id, owner)? {
                anyhow::bail!(
                    "Session {} was taken over by another process while this turn ran, \
                     so its messages were not saved — writing them now would interleave \
                     with the turns that process is writing.",
                    &self.id[..8.min(self.id.len())]
                );
            }
        }

        for (seq, message) in self.messages.iter().enumerate().skip(self.saved_len) {
            store::append_message(
                &tx,
                &self.id,
                seq,
                message,
                &self.model,
                self.effort_level.as_deref(),
            )?;
        }

        // Derived from the first user message, so it belongs in the same
        // commit as the message that decides it.
        let title = (!self.title_set && self.messages.iter().any(|m| m.role == "user"))
            .then(|| store::derive_title(&self.messages));
        if let Some(title) = &title {
            store::set_session_title(&tx, &self.id, title)?;
        }

        tx.commit()?;

        // Only once the write is durable does the in-memory copy agree that
        // it happened.
        self.saved_len = self.messages.len();
        if let Some(title) = title {
            self.title = title;
            self.title_set = true;
        }
        Ok(())
    }

    /// Deletes the session if it was never named and nothing was ever said
    /// in it, reporting whether it did.
    ///
    /// A session row is created up front so messages have somewhere to go,
    /// which means opening a conversation and backing out without typing
    /// would leave an empty "Untitled" behind — clutter, now that the launch
    /// screen lists every session.
    ///
    /// Naming one is enough to keep it, though. Typing a title and
    /// confirming is a deliberate act: the session is something the user
    /// decided to start, whether or not they got as far as saying anything
    /// in it. Only backing out of the naming screen with a blank title, and
    /// then saying nothing, reads as "never mind".
    pub fn discard_if_unused(&self) -> Result<bool> {
        // The stored title, not `title_set`: this runs when a front end
        // goes away, which is exactly when the flag is least trustworthy —
        // it is rebuilt on every reopen, and getting it wrong here deletes
        // a conversation rather than mislabelling one.
        if self.title != UNTITLED || self.messages.iter().any(|m| m.role == "user") {
            return Ok(false);
        }
        store::delete_session(&self.conn, &self.id)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn highlight_persists_so_a_later_resume_looks_the_same() {
        let mut session = memory_session();
        assert!(session.highlight(), "on unless the config said otherwise");

        session.set_highlight(false).unwrap();
        assert!(!session.highlight());

        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert!(!summary.highlight, "recorded, not just held in memory");
    }
    use super::*;
    use crate::store::KIND_CHAT;

    /// An in-memory database with the same schema `store::open_db` builds,
    /// so sessions can be exercised without touching the real one.
    fn memory_conn() -> Connection {
        crate::crypto::seed_test_key();
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "
            CREATE TABLE sessions (
                id              TEXT PRIMARY KEY,
                title           TEXT NOT NULL,
                model           TEXT NOT NULL,
                kind            TEXT NOT NULL,
                effort_level    TEXT,
                verbose         INTEGER NOT NULL DEFAULT 0,
                highlight       INTEGER NOT NULL DEFAULT 1,
                max_iterations  INTEGER,
                temperature     REAL,
                approval_read      INTEGER NOT NULL DEFAULT 1,
                approval_write     INTEGER NOT NULL DEFAULT 1,
                approval_terminal  INTEGER NOT NULL DEFAULT 1,
                tool_access        TEXT,
                sandbox            INTEGER NOT NULL DEFAULT 1,
                stream             INTEGER NOT NULL DEFAULT 1,
                working_dir        TEXT,
                activity           TEXT,
                activity_detail    TEXT,
                heartbeat          INTEGER,
                claim_owner        TEXT,
                total_tokens       INTEGER NOT NULL DEFAULT 0,
                created_at      INTEGER NOT NULL,
                updated_at      INTEGER NOT NULL
            );
            CREATE TABLE messages (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                seq               INTEGER NOT NULL,
                role              TEXT NOT NULL,
                content           TEXT,
                tool_calls        TEXT,
                tool_call_id      TEXT,
                model             TEXT,
                effort_level      TEXT,
                reasoning_details TEXT,
                reasoning         TEXT
            );
            ",
        )
        .unwrap();
        conn
    }

    fn memory_session() -> ChatSession {
        ChatSession::create(
            memory_conn(),
            new_id(),
            "test-model".to_string(),
            KIND_CHAT,
            Some("high".to_string()),
            Some(20),
            Some(0.7),
            ToolAccessSettings::default(),
            true,
            false,
            true,
            true,
            None,
        )
        .unwrap()
    }

    #[test]
    fn persists_pending_messages_once() {
        let mut session = memory_session();
        session.push_user("hello".to_string());
        session.push_assistant("hi there".to_string());
        session.persist_pending().unwrap();

        let stored = store::load_messages(&session.conn, session.id()).unwrap();
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].message.role, "user");
        assert_eq!(stored[1].message.content, Some("hi there".to_string()));

        // A second call must not duplicate anything already written.
        session.persist_pending().unwrap();
        let stored = store::load_messages(&session.conn, session.id()).unwrap();
        assert_eq!(stored.len(), 2);
    }

    #[test]
    fn tags_each_message_with_model_and_effort() {
        let mut session = memory_session();
        session.push_user("hello".to_string());
        session.persist_pending().unwrap();

        let stored = store::load_messages(&session.conn, session.id()).unwrap();
        assert_eq!(stored[0].model.as_deref(), Some("test-model"));
        assert_eq!(stored[0].effort_level.as_deref(), Some("high"));
    }

    #[test]
    fn titles_session_from_first_user_message() {
        let mut session = memory_session();
        session.push(ChatMessage {
            role: "system".to_string(),
            content: Some("system prompt".to_string()),
            tool_calls: None,
            tool_call_id: None,
            ..Default::default()
        });
        session.persist_pending().unwrap();
        // A system-only session has nothing to name itself after yet.
        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert_eq!(summary.title, "Untitled");

        session.push_user("Write me a snake game".to_string());
        session.persist_pending().unwrap();
        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert_eq!(summary.title, "Write me a snake game");
    }

    #[test]
    fn title_is_not_rewritten_by_later_turns() {
        let mut session = memory_session();
        session.push_user("first question".to_string());
        session.persist_pending().unwrap();
        session.push_user("second question".to_string());
        session.persist_pending().unwrap();

        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert_eq!(summary.title, "first question");
    }

    #[test]
    fn set_model_persists_so_a_later_resume_picks_it_up() {
        let mut session = memory_session();
        session.push_user("hello".to_string());
        session.persist_pending().unwrap();

        session.set_model("second-model".to_string()).unwrap();
        assert_eq!(session.model(), "second-model");

        // The sessions row must reflect it, not just the in-memory session.
        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert_eq!(summary.model, "second-model");
    }

    #[test]
    fn having_tools_follows_from_the_tools_and_the_kind_column_follows_that() {
        // `memory_session` starts from the built-in defaults, where every
        // tool asks — so it has tools.
        let mut session = memory_session();
        assert!(session.is_agentic());
        session.push_user("hello".to_string());
        session.persist_pending().unwrap();
        let messages_before = session.messages().len();

        session.set_tool_access(ToolAccessSettings::none()).unwrap();
        assert!(!session.is_agentic(), "every tool never means no tools");

        // The column follows, so a row read without being opened — the
        // launch screen, `clank clankers list` — agrees with the session.
        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert_eq!(summary.kind, crate::store::KIND_CHAT);

        // And back again, without touching history either way: no system
        // prompt is stored, because a provider-valid position for one cannot
        // be guaranteed when tools can come and go mid-conversation.
        // `agent::normalize_system_prompt` prepends it per turn instead.
        session
            .set_tool_access(ToolAccessSettings::defaults())
            .unwrap();
        assert!(session.is_agentic());
        assert_eq!(session.messages().len(), messages_before);
        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert_eq!(summary.kind, crate::store::KIND_AGENT_CHAT);
    }

    #[test]
    fn set_effort_level_persists_and_clears() {
        let mut session = memory_session();
        assert_eq!(session.effort_level(), Some("high"));

        session.set_effort_level(Some("low".to_string())).unwrap();
        assert_eq!(session.effort_level(), Some("low"));
        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert_eq!(summary.effort_level, Some("low".to_string()));

        session.set_effort_level(None).unwrap();
        assert_eq!(session.effort_level(), None);
        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert_eq!(summary.effort_level, None);
    }

    #[test]
    fn set_verbose_persists_so_a_later_resume_picks_it_up() {
        let mut session = memory_session();
        assert!(!session.verbose());

        session.set_verbose(true).unwrap();
        assert!(session.verbose());
        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert!(summary.verbose);
    }

    #[test]
    fn set_max_iterations_persists_and_clears() {
        let mut session = memory_session();
        // The 20 `memory_session` created it with — a snapshot, not "unset".
        assert_eq!(session.max_iterations(), Some(20));

        session.set_max_iterations(Some(30)).unwrap();
        assert_eq!(session.max_iterations(), Some(30));
        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert_eq!(summary.max_iterations, Some(30));

        // Nullifying is a session-layer concept again — a caller resolves
        // `/max-iterations default` to a concrete value itself, same as any
        // other explicit number, but `clear` passes `None` straight through.
        session.set_max_iterations(None).unwrap();
        assert_eq!(session.max_iterations(), None);
        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert_eq!(summary.max_iterations, None);
    }

    #[test]
    fn set_temperature_persists_and_clears() {
        let mut session = memory_session();
        assert_eq!(session.temperature(), Some(0.7));

        session.set_temperature(Some(1.5)).unwrap();
        assert_eq!(session.temperature(), Some(1.5));
        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert_eq!(summary.temperature, Some(1.5));

        session.set_temperature(None).unwrap();
        assert_eq!(session.temperature(), None);
        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert_eq!(summary.temperature, None);
    }

    #[test]
    fn add_tokens_sums_and_persists() {
        let mut session = memory_session();
        assert_eq!(session.total_tokens(), 0);

        session.add_tokens(120).unwrap();
        session.add_tokens(30).unwrap();
        assert_eq!(session.total_tokens(), 150);
        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert_eq!(summary.total_tokens, 150);

        // A resumed session picks the running total back up rather than
        // starting over.
        let (resumed, _) =
            ChatSession::resume(session.conn, &summary, summary.model.clone()).unwrap();
        assert_eq!(resumed.total_tokens(), 150);
    }

    fn session_in(dir: Option<&str>) -> ChatSession {
        ChatSession::create(
            memory_conn(),
            new_id(),
            "test-model".to_string(),
            KIND_CHAT,
            None,
            Some(20),
            Some(0.7),
            ToolAccessSettings::default(),
            true,
            false,
            true,
            true,
            dir.map(str::to_string),
        )
        .unwrap()
    }

    #[test]
    fn a_session_records_the_directory_it_started_in() {
        let dir = std::env::temp_dir().display().to_string();
        let session = session_in(Some(&dir));
        assert_eq!(session.working_dir(), Some(dir.as_str()));

        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert_eq!(summary.working_dir.as_deref(), Some(dir.as_str()));
    }

    #[test]
    fn a_session_without_a_recorded_directory_resumes_where_it_is() {
        // Rows written before this was tracked. Refusing to resume them, or
        // moving the process somewhere arbitrary, would break sessions that
        // worked yesterday.
        let session = session_in(None);
        assert!(matches!(
            enter_working_dir(&session).unwrap(),
            EnteredDir::Unchanged
        ));
    }

    #[test]
    fn a_missing_directory_is_reported_rather_than_ignored() {
        // The session's sandbox is anchored to a directory that isn't there,
        // so neither front end can honour it — and quietly rebinding the
        // bound to whatever is current is the one outcome worth refusing.
        let session = session_in(Some("/clank-no-such-directory-exists"));
        assert!(matches!(
            enter_working_dir(&session).unwrap(),
            EnteredDir::Missing(_)
        ));
        // Nothing moved.
        assert_ne!(
            std::env::current_dir().unwrap().display().to_string(),
            "/clank-no-such-directory-exists"
        );
    }

    #[test]
    fn repointing_a_session_records_the_new_directory() {
        let mut session = session_in(Some("/clank-no-such-directory-exists"));
        session.set_working_dir("/tmp".to_string()).unwrap();

        assert_eq!(session.working_dir(), Some("/tmp"));
        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert_eq!(summary.working_dir.as_deref(), Some("/tmp"));
    }

    #[test]
    fn a_new_session_snapshots_the_configured_verbose_default() {
        // The configured default is a starting value, not a live one: it is
        // written into the session at creation, and `/verbose` from then on
        // changes the session rather than the configuration.
        let conn = memory_conn();
        let session = ChatSession::create(
            conn,
            new_id(),
            "test-model".to_string(),
            KIND_CHAT,
            None,
            Some(20),
            Some(0.7),
            ToolAccessSettings::default(),
            true,
            true,
            true,
            true,
            None,
        )
        .unwrap();

        assert!(session.verbose());
        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert!(
            summary.verbose,
            "it has to survive a reload, not just live in memory"
        );
    }

    #[test]
    fn a_turn_that_outlived_its_claim_is_not_written() {
        // A process starved past the staleness window loses the session to
        // whoever claims it next, and finds out only here. Persisting anyway
        // would collide with the turns that process is writing.
        let mut session = memory_session();
        session.writes_under_claim("mine".to_string());
        store::claim_session(&session.conn, session.id(), "mine").unwrap();
        session.push_user("before".to_string());
        session.persist_pending().expect("held: should save");

        // Taken over, which can only happen once our claim has gone stale —
        // a fresh one is not claimable, so backdate it the way a starved
        // process's would be.
        session
            .conn
            .execute(
                "UPDATE sessions SET heartbeat = 0 WHERE id = ?1",
                rusqlite::params![session.id()],
            )
            .unwrap();
        assert!(
            store::claim_session(&session.conn, session.id(), "usurper").unwrap(),
            "the stale claim should be takeable"
        );
        session.push_user("after".to_string());
        let error = session
            .persist_pending()
            .expect_err("a lost claim must not write")
            .to_string();
        assert!(error.contains("taken over"), "{error}");

        let stored = store::load_messages(&session.conn, session.id()).unwrap();
        assert_eq!(stored.len(), 1, "only what was written while we held it");
    }

    #[test]
    fn a_session_bound_to_no_claim_writes_as_before() {
        // Every test builds one of these, and so does any caller that never
        // took a claim; the check must not turn those into failures.
        let mut session = memory_session();
        session.push_user("unclaimed".to_string());
        session.persist_pending().expect("no claim, no check");
        assert_eq!(
            store::load_messages(&session.conn, session.id())
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn a_failed_save_keeps_the_messages_pending_for_the_next_attempt() {
        // `cmd_agent` persists once before the turn and once after, and
        // carries on when the first one fails. That is only safe if the
        // failure left the messages pending: otherwise the question is
        // dropped and the transcript keeps the answer alone.
        let mut session = memory_session();
        session.push_user("the question".to_string());

        // Fault injection: with the table gone the append cannot succeed.
        session
            .conn
            .execute("ALTER TABLE messages RENAME TO messages_hidden", [])
            .unwrap();
        assert!(session.persist_pending().is_err(), "the save should fail");

        session
            .conn
            .execute("ALTER TABLE messages_hidden RENAME TO messages", [])
            .unwrap();
        session.push_assistant("the answer".to_string());
        session.persist_pending().expect("the retry should succeed");

        let stored = store::load_messages(&session.conn, session.id()).unwrap();
        let text: Vec<&str> = stored
            .iter()
            .filter_map(|m| m.message.content.as_deref())
            .collect();
        assert_eq!(
            text,
            vec!["the question", "the answer"],
            "the retry must save what the failed attempt did not"
        );
    }

    #[test]
    fn a_failed_save_writes_nothing_at_all() {
        // Partial success would leave a hole in `seq`, which reads back as a
        // transcript that is simply missing a turn.
        let mut session = memory_session();
        session.push_user("first".to_string());
        session.persist_pending().unwrap();

        session.push_user("second".to_string());
        session.push_assistant("third".to_string());
        session
            .conn
            .execute("ALTER TABLE messages RENAME TO messages_hidden", [])
            .unwrap();
        assert!(session.persist_pending().is_err());
        session
            .conn
            .execute("ALTER TABLE messages_hidden RENAME TO messages", [])
            .unwrap();

        let stored = store::load_messages(&session.conn, session.id()).unwrap();
        assert_eq!(stored.len(), 1, "only the message saved before the fault");
    }

    #[test]
    fn a_failed_save_does_not_claim_the_title_it_could_not_write() {
        let mut session = memory_session();
        session.push_user("name me from this".to_string());
        session
            .conn
            .execute("ALTER TABLE messages RENAME TO messages_hidden", [])
            .unwrap();
        assert!(session.persist_pending().is_err());
        assert_eq!(session.title(), "Untitled", "title followed a failed write");
    }

    #[test]
    fn set_tool_access_persists_and_updates_the_summary() {
        let mut session = memory_session();
        assert_eq!(*session.tool_access(), ToolAccessSettings::default());

        let custom = ToolAccessSettings::default()
            .with("run_terminal_command", crate::config::ToolAccess::Never)
            .unwrap();
        session.set_tool_access(custom.clone()).unwrap();
        assert_eq!(*session.tool_access(), custom);
        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert_eq!(summary.tool_access, custom);
    }

    #[test]
    fn title_tracks_live_once_derived_from_the_first_message() {
        let mut session = memory_session();
        assert_eq!(session.title(), "Untitled");

        session.push_user("Write me a snake game".to_string());
        session.persist_pending().unwrap();
        assert_eq!(session.title(), "Write me a snake game");
    }

    #[test]
    fn set_title_persists_and_blocks_later_auto_derivation() {
        let mut session = memory_session();
        session.set_title("My chosen title".to_string()).unwrap();
        assert_eq!(session.title(), "My chosen title");
        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        assert_eq!(summary.title, "My chosen title");

        // A later user message must not clobber the chosen title.
        session.push_user("hello".to_string());
        session.persist_pending().unwrap();
        assert_eq!(session.title(), "My chosen title");
    }

    #[test]
    fn resume_picks_up_the_persisted_title_verbose_max_iterations_and_temperature() {
        let mut session = memory_session();
        session.push_user("hello".to_string());
        session.persist_pending().unwrap();
        session.set_verbose(true).unwrap();
        session.set_max_iterations(Some(30)).unwrap();
        session.set_temperature(Some(1.5)).unwrap();

        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        let ChatSession { conn, .. } = session;

        let (resumed, _) = ChatSession::resume(conn, &summary, summary.model.clone()).unwrap();
        assert_eq!(resumed.title(), "hello");
        assert!(resumed.verbose());
        assert_eq!(resumed.max_iterations(), Some(30));
        assert_eq!(resumed.temperature(), Some(1.5));
    }

    #[test]
    fn resuming_with_a_different_model_records_the_switch() {
        let mut session = memory_session();
        session.push_user("hello".to_string());
        session.persist_pending().unwrap();
        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        let ChatSession { conn, .. } = session;

        let (resumed, _) =
            ChatSession::resume(conn, &summary, "switched-model".to_string()).unwrap();
        assert_eq!(resumed.model(), "switched-model");

        // And resuming again with no override keeps the switched model
        // rather than reverting to the original.
        let summary = store::find_session(&resumed.conn, resumed.id())
            .unwrap()
            .unwrap();
        assert_eq!(summary.model, "switched-model");
    }

    #[test]
    fn a_named_but_empty_session_survives_being_reopened_and_left() {
        // The TUI refuses a blank name, so every session it creates has one.
        let mut session = memory_session();
        session.set_title("Plan the migration".to_string()).unwrap();
        let id = session.id().to_string();

        // Open it from the picker and back straight out again without
        // saying anything — which is what `discard_if_unused` runs on.
        // Same connection, since the in-memory database lives in it.
        let summary = store::find_session(&session.conn, &id).unwrap().unwrap();
        let conn = std::mem::replace(&mut session.conn, Connection::open_in_memory().unwrap());
        drop(session);
        let (reopened, _) = ChatSession::resume(conn, &summary, summary.model.clone()).unwrap();
        assert!(
            !reopened.discard_if_unused().unwrap(),
            "a session you took the trouble to name must not be deleted \
             just because you have not typed in it yet"
        );

        assert!(
            store::find_session(&reopened.conn, &id).unwrap().is_some(),
            "it is gone from the picker"
        );
    }

    #[test]
    fn reopening_a_named_session_does_not_let_the_first_message_rename_it() {
        // The same flag, the other way round: with `title_set` inferred from
        // the messages, a named session you had not typed in yet came back
        // looking unnamed, and `persist_pending` then derived a title over
        // the one you chose.
        let mut session = memory_session();
        session.set_title("Plan the migration".to_string()).unwrap();
        let id = session.id().to_string();

        let summary = store::find_session(&session.conn, &id).unwrap().unwrap();
        let conn = std::mem::replace(&mut session.conn, Connection::open_in_memory().unwrap());
        drop(session);
        let (mut reopened, _) = ChatSession::resume(conn, &summary, summary.model.clone()).unwrap();

        reopened.push_user("something else entirely".to_string());
        reopened.persist_pending().unwrap();

        assert_eq!(
            reopened.title(),
            "Plan the migration",
            "the name you gave it should outlast the first thing you say"
        );
    }

    #[test]
    fn an_unused_session_is_discarded() {
        let session = memory_session();
        assert!(session.discard_if_unused().unwrap());
        assert!(store::find_session(&session.conn, session.id())
            .unwrap()
            .is_none());
    }

    #[test]
    fn a_named_session_is_kept_even_with_nothing_said_in_it() {
        // Typing a title and confirming is a deliberate act — the session is
        // one the user decided to start, whether or not they got as far as
        // saying anything in it.
        let mut session = memory_session();
        session.set_title("Plan the migration".to_string()).unwrap();

        assert!(!session.discard_if_unused().unwrap());
        assert!(store::find_session(&session.conn, session.id())
            .unwrap()
            .is_some());
    }

    #[test]
    fn a_session_with_only_a_system_prompt_still_counts_as_unused() {
        // Nothing writes a system prompt into history any more — the agent
        // one is prepended per request — but sessions created before that
        // have one sitting in theirs, and opening one of those and leaving
        // must still discard it.
        let mut session = memory_session();
        session.push(ChatMessage {
            role: "system".to_string(),
            content: Some("system prompt".to_string()),
            tool_calls: None,
            tool_call_id: None,
            ..Default::default()
        });
        session.persist_pending().unwrap();
        assert!(session.discard_if_unused().unwrap());
    }

    #[test]
    fn a_used_session_is_kept() {
        let mut session = memory_session();
        session.push_user("hello".to_string());
        session.persist_pending().unwrap();
        assert!(!session.discard_if_unused().unwrap());
        assert!(store::find_session(&session.conn, session.id())
            .unwrap()
            .is_some());
    }

    #[test]
    fn resume_restores_history_and_keeps_appending_in_order() {
        let mut session = memory_session();
        session.push_user("hello".to_string());
        session.push_assistant("hi there".to_string());
        session.persist_pending().unwrap();

        let summary = store::find_session(&session.conn, session.id())
            .unwrap()
            .unwrap();
        let ChatSession { conn, .. } = session;

        let (mut resumed, history) =
            ChatSession::resume(conn, &summary, summary.model.clone()).unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(resumed.messages().len(), 2);

        // Resuming must not re-write the history it just loaded.
        resumed.push_user("follow up".to_string());
        resumed.persist_pending().unwrap();
        let stored = store::load_messages(&resumed.conn, resumed.id()).unwrap();
        assert_eq!(stored.len(), 3);
        assert_eq!(stored[2].message.content, Some("follow up".to_string()));
    }
}
