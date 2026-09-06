//! The full-screen terminal front end.
//!
//! Owns the screen rather than scrolling through it, which is what makes a
//! persistent input box, live streaming text, and inline tool approval
//! possible — all things the line-based CLI structurally can't do.
//!
//! It drives a [`Conversation`] worker and renders the events it emits; it
//! never calls the agent loop itself. A GUI would attach at the same place.
//!
//! Three screens: a launch list, a sessions browser, and the conversation
//! itself. Only the conversation owns a worker. Navigating away from an idle
//! one shuts it down and waits for its last writes; navigating away from one
//! mid-turn *parks* it — the whole `Chat`, worker and transcript included,
//! moves to the event loop's parked list and keeps running, since its tool
//! calls have already touched the disk and a turn is only recorded once it
//! ends. That is what makes the launch screen a monitor: parked sessions
//! report `working` and `?` through the database like any running session,
//! and opening one hands the same screen back rather than building a new
//! one. They stay claimed while they run, so no other terminal can reach
//! them. Quitting still ends everything: a worker is a task, and tasks go
//! with the process.

mod app;
mod picker;
mod render;
/// The session mark, so the CLI can draw the same one the TUI does.
pub(crate) use render::{busy_frame, identicon_mark};

use crate::client::Client;
use crate::config::ToolAccessSettings;
use crate::conversation::{command_for, Command, Conversation};
use crate::session::{self, ChatSession};
use crate::store::{self, SessionSummary, StoredMessage, KIND_AGENT_CHAT, KIND_CHAT};
use crate::ui::response_label;
use anyhow::Result;
use app::{App, ShellState, TranscriptItem};
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event as TermEvent, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers,
    KeyboardEnhancementFlags, MouseEvent, MouseEventKind, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
};
use crossterm::{execute, terminal};
use futures_util::StreamExt;
use picker::{Activation, Deployment, Picker, Plan, SessionRow};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io::{self, Stdout};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// How often the screen redraws while idle, driving the spinner.
const TICK: Duration = Duration::from_millis(100);
/// How often the picker screen re-reads where each session left off. Slower
/// than the animation tick on purpose: it's a database read, and a session's
/// last message doesn't change faster than a turn takes.
const SESSION_REFRESH: Duration = Duration::from_secs(2);

/// Everything needed to start conversations on demand, since with a launch
/// screen the TUI no longer receives a ready-made session.
pub struct Context {
    pub client: Arc<Client>,
    /// Model for new sessions; a resumed one keeps its own.
    pub default_model: String,
    pub effort_level: Option<String>,
    pub max_iterations: Option<usize>,
    pub temperature: Option<f32>,
    pub tool_access: ToolAccessSettings,
    /// The configured default for confining the agent's file writes, taken
    /// as this session's starting value.
    pub sandbox: bool,
    /// The configured default for showing full tool-call detail, taken as
    /// this session's starting value.
    pub verbose: bool,
    /// The configured default for banding your own messages, same deal.
    pub highlight: bool,
    /// Whether the launch screen bands its selected row. Global only — the
    /// launch screen belongs to no session, so there is nothing to override.
    pub selection: bool,
    /// The configured default for streaming replies, same deal.
    pub stream: bool,
}

enum Screen {
    /// Boxed for the same reason `Chat` is: it dwarfs the other variants,
    /// and every `Screen` would otherwise carry its footprint.
    Launch(Box<Picker>),
    /// Setting a clanker up before it exists — reached by choosing "Deploy
    /// clanker" on the launch screen. Boxed for the same reason as the
    /// others: it is a form with several fields, and every `Screen` would
    /// otherwise carry them.
    Deploy(Box<Deployment>),
    Chat(Box<Chat>),
}

struct Chat {
    app: App,
    conversation: Conversation,
    /// Rendered transcript blocks, kept between frames. Lives here rather
    /// than on `App`, which is deliberately free of rendering types.
    transcript_cache: render::TranscriptCache,
}

/// Sessions left working with nobody watching them, kept whole — worker,
/// transcript, approval box and all — so reopening one picks the screen back
/// up rather than building a new one from the database.
///
/// Only ever this process's own. A session another terminal claims is
/// refused exactly as it always was: these are the ones *we* still hold, and
/// holding them is what makes handing the screen back possible.
type Parked = Vec<Box<Chat>>;

/// Runs the TUI until the user quits. Always opens on the launch screen —
/// there's no flag to skip straight into a new or resumed session, so this
/// is the one and only way in.
pub async fn run(context: Context) -> Result<()> {
    let mut screen = Screen::Launch(Box::new(launch_picker()?));

    let mut terminal = enter()?;
    // Restore the terminal even on the way out of an error, so a failure
    // never leaves the user with a broken shell.
    let result = event_loop(&mut terminal, &context, &mut screen).await;
    if let Screen::Chat(chat) = screen {
        // Cancelled rather than left to finish: the process is going, so a
        // turn cannot outlive this wait however long it is given, and the
        // abort is what reaps a tool subprocess still running. A no-op when
        // nothing is in flight.
        chat.conversation.send(Command::Cancel);
        chat.conversation.shutdown().await;
    }
    leave(&mut terminal)?;
    result
}

fn load_sessions() -> Result<Vec<SessionRow>> {
    let conn = store::open_db()?;
    // Every session's last message in one query, then matched up here —
    // a query per row would make opening the picker scale with how many
    // sessions have accumulated.
    let mut last = store::last_messages(&conn)?;
    Ok(store::list_sessions(&conn)?
        .into_iter()
        .map(|summary| {
            let mut row = SessionRow::from(summary);
            row.last = last.remove(&row.id);
            row
        })
        .collect())
}

/// Each conversation gets its own database handle, so sessions can be
/// opened and closed over the life of the TUI without threading one
/// connection through every screen.
/// The launch screen, grouped by whether each session belongs to the
/// directory the process is in right now.
fn launch_picker() -> Result<Picker> {
    Ok(Picker::launch(load_sessions()?, current_dir().as_deref()))
}

fn current_dir() -> Option<String> {
    std::env::current_dir()
        .ok()
        .map(|dir| dir.display().to_string())
}

fn open_new(context: &Context, plan: Plan) -> Result<Chat> {
    // A clanker deployed with tools gets the ones `clank tools` allows —
    // the same set `/tools on` would hand it — and one deployed without gets
    // none at all. The configured access says what tools may do *once they
    // are on*, which is why "on" is a choice made on the deployment screen
    // rather than inherited from a config file you set months ago.
    let tool_access = if plan.tools {
        context.tool_access.clone()
    } else {
        ToolAccessSettings::none()
    };
    // Written to match the tools it is created with, so a row read without
    // being opened says the same thing the clanker would — the same job
    // `ChatSession::sync_kind` does for every later change.
    let kind = if tool_access.any_tools() {
        KIND_AGENT_CHAT
    } else {
        KIND_CHAT
    };

    let mut session = ChatSession::create(
        store::open_db()?,
        plan.id,
        plan.model,
        kind,
        plan.effort,
        context.max_iterations,
        plan.temperature,
        tool_access,
        plan.sandbox,
        context.verbose,
        context.highlight,
        context.stream,
        std::env::current_dir()
            .ok()
            .map(|dir| dir.display().to_string()),
    )?;
    session.set_title(plan.title)?;
    // The agent system prompt is not written into history: `agent.rs`'s
    // `normalize_system_prompt` strips any stored copy and prepends a fresh
    // one on every turn that has tools, which is what lets tools be turned
    // on and off part-way through without leaving a stale instruction
    // behind.
    start_chat(context, session, Vec::new())
}

fn open_resumed(context: &Context, summary: &SessionSummary) -> Result<Chat> {
    let (session, history) =
        ChatSession::resume(store::open_db()?, summary, summary.model.clone())?;
    // The session's directory is its sandbox boundary, so opening one moves
    // the process into it. Unlike the CLI there's nowhere to refuse to: the
    // TUI can open any session from its picker, so a directory that's gone
    // is reported in the transcript and the session opens where it is —
    // loudly, because its bound is now the wrong one.
    let entered = session::enter_working_dir(&session)?;
    let mut chat = start_chat(context, session, history)?;
    match entered {
        session::EnteredDir::Moved(dir) => chat
            .app
            .transcript
            .push(TranscriptItem::Notice(format!("Working directory: {dir}"))),
        session::EnteredDir::Unchanged => {}
        session::EnteredDir::Missing(dir) => {
            chat.app.transcript.push(TranscriptItem::Error(format!(
                "This clanker was started in {dir}, which no longer exists — \
                 it is running in the current directory instead, so its sandbox \
                 and relative paths point somewhere else than when it was saved."
            )))
        }
    }
    Ok(chat)
}

/// Why a row couldn't be opened, when the caller can do something about it.
enum OpenFailure {
    /// The session's directory is gone. Offerable: resuming here and
    /// repointing is a real answer, so the picker asks rather than refusing.
    MissingDir(String),
    Other(anyhow::Error),
}

fn open_row(context: &Context, row: &SessionRow) -> std::result::Result<Chat, OpenFailure> {
    let summary = (|| {
        let conn = store::open_db()?;
        store::find_session(&conn, &row.id)?
            .ok_or_else(|| anyhow::anyhow!("Clanker {} no longer exists", row.short_id()))
    })()
    .map_err(OpenFailure::Other)?;

    if let Some(dir) = summary.working_dir.as_deref() {
        if !std::path::Path::new(dir).is_dir() {
            return Err(OpenFailure::MissingDir(dir.to_string()));
        }
    }
    open_resumed(context, &summary).map_err(OpenFailure::Other)
}

/// Resumes `row` in the current directory, recording it as the session's own.
fn open_row_here(context: &Context, row: &SessionRow) -> Result<Chat> {
    let conn = store::open_db()?;
    let summary = store::find_session(&conn, &row.id)?
        .ok_or_else(|| anyhow::anyhow!("Clanker {} no longer exists", row.short_id()))?;
    let (mut session, history) = ChatSession::resume(conn, &summary, summary.model.clone())?;
    if let Some(cwd) = std::env::current_dir()
        .ok()
        .map(|d| d.display().to_string())
    {
        session.set_working_dir(cwd)?;
    }
    start_chat(context, session, history)
}

/// Builds the chat screen for a session this process has taken.
///
/// Claims first, and refuses if the claim is already held: two processes
/// appending turns to one history write colliding `seq` values, which reload
/// as a conversation with its turns shuffled and its tool results detached
/// from the calls they answer. Nothing detects that and nothing repairs it,
/// so it has to be prevented rather than warned about.
fn start_chat(
    context: &Context,
    session: ChatSession,
    history: Vec<StoredMessage>,
) -> Result<Chat> {
    let Some(claim) = crate::session::Heartbeat::claim(session.id().to_string())? else {
        // One line, no paragraph breaks, and short: this surfaces as the
        // picker's notice, which is a single unwrapped `Line` — it renders
        // `\n` as nothing at all rather than as a break, and anything past
        // the terminal's width is clipped rather than folded, so the
        // sentence that matters has to be the first one. The half-minute
        // staleness window is in the README rather than here for that
        // reason. Both causes are named because they read as different
        // problems: someone else has it, or you left a turn running in it.
        anyhow::bail!(
            "Clanker {} is in use — another terminal, or a turn still finishing",
            &session.id()[..8]
        );
    };
    let mut app = App::new(
        session.model().to_string(),
        session.effort_level().map(str::to_string),
        // The full id: the mark in the reply gutter hashes it, and the
        // picker hashes the same to draw this session's row.
        session.id().to_string(),
    );
    app.verbose = session.verbose();
    app.highlight = session.highlight();
    app.max_iterations = session.max_iterations();
    app.temperature = session.temperature();
    app.total_tokens = session.total_tokens();
    app.tool_access = session.tool_access().clone();
    app.sandbox = session.sandbox();
    app.stream = session.stream();
    app.working_dir = session.working_dir().map(str::to_string);
    app.title = session.title().to_string();
    seed_transcript(&mut app, &history);

    let conversation = Conversation::spawn(
        Arc::clone(&context.client),
        session,
        context.max_iterations,
        context.temperature,
        context.effort_level.clone(),
        context.tool_access.clone(),
        claim,
    );
    Ok(Chat {
        app,
        conversation,
        transcript_cache: render::TranscriptCache::default(),
    })
}

/// Replays a resumed session into the transcript so the TUI opens showing
/// the conversation so far.
fn seed_transcript(app: &mut App, history: &[StoredMessage]) {
    for stored in history {
        let message = &stored.message;
        match message.role.as_str() {
            "user" => {
                if let Some(text) = &message.content {
                    app.transcript.push(TranscriptItem::User(text.clone()));
                    // So Up/Down can recall prompts from before this resume,
                    // not just what's typed in the current sitting.
                    app.input_history.push(text.clone());
                }
            }
            "assistant" => {
                // Ahead of the reply it led to, matching the live ordering.
                // Pushed even when the reply itself had no visible text —
                // a turn that only called a tool still thought first.
                if let Some(thought) = message.thinking_text() {
                    app.transcript.push(TranscriptItem::Thinking(thought));
                }
                if let Some(text) = &message.content {
                    if !text.trim().is_empty() {
                        // Each stored message knows the model that produced
                        // it, so a session whose model changed part-way
                        // replays with each reply correctly attributed.
                        let label = stored
                            .model
                            .as_ref()
                            .map(|model| response_label(model, &stored.effort_level));
                        app.transcript.push(TranscriptItem::Assistant {
                            text: text.clone(),
                            streaming: false,
                            label,
                        });
                    }
                }
            }
            // Tool results and the system prompt are bookkeeping; a resumed
            // view shows the conversation, not the plumbing.
            _ => {}
        }
    }
}

type Tui = Terminal<CrosstermBackend<Stdout>>;

/// Whether the terminal accepted the keyboard enhancement flags, so teardown
/// knows whether it has a push to undo. A static because both [`leave`] and
/// the panic hook have to see it and neither is handed any state.
static ENHANCED_KEYS: AtomicBool = AtomicBool::new(false);

fn enter() -> Result<Tui> {
    // Before raw mode and the alternate screen: this writes an escape
    // sequence and waits for the terminal's reply, which would otherwise
    // race the renderer for the same stream.
    render::detect_band();

    terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    // Without this, a terminal delivers a paste as plain keystrokes, and
    // any embedded newline reads as a real Enter — submitting each pasted
    // line as its own message instead of landing in the input box as text.
    // Mouse capture is what lets the scroll wheel move the transcript
    // instead of the terminal's own (unrelated) native scrollback.
    execute!(
        stdout,
        terminal::EnterAlternateScreen,
        EnableBracketedPaste,
        EnableMouseCapture
    )?;

    // Shift-Enter can't be seen at all under the legacy input protocol: a
    // bare Enter arrives as a carriage return, which has nowhere to carry a
    // modifier, so Shift-Enter is byte-identical to Enter. (Alt-Enter works
    // because Alt has always been encoded as an escape prefix, which *is*
    // distinguishable.) The kitty keyboard protocol reports the modifier
    // properly, so ask for its disambiguation flag wherever the terminal
    // advertises support and leave everything alone where it doesn't —
    // Alt-Enter still covers those.
    let enhanced = terminal::supports_keyboard_enhancement().unwrap_or(false);
    if enhanced {
        execute!(
            stdout,
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        )?;
    }
    ENHANCED_KEYS.store(enhanced, Ordering::Relaxed);

    // A panic while in raw mode would otherwise leave the terminal unusable
    // with no echo and no cursor, so restore first, then panic normally.
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = terminal::disable_raw_mode();
        if ENHANCED_KEYS.load(Ordering::Relaxed) {
            let _ = execute!(io::stdout(), PopKeyboardEnhancementFlags);
        }
        let _ = execute!(
            io::stdout(),
            DisableBracketedPaste,
            DisableMouseCapture,
            terminal::LeaveAlternateScreen
        );
        previous(info);
    }));

    Ok(Terminal::new(CrosstermBackend::new(stdout))?)
}

fn leave(terminal: &mut Tui) -> Result<()> {
    terminal::disable_raw_mode()?;
    // Popped before the rest so the terminal is back on its own protocol
    // even if a later command fails.
    if ENHANCED_KEYS.swap(false, Ordering::Relaxed) {
        execute!(terminal.backend_mut(), PopKeyboardEnhancementFlags)?;
    }
    execute!(
        terminal.backend_mut(),
        DisableBracketedPaste,
        DisableMouseCapture,
        terminal::LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;
    Ok(())
}

/// What woke the loop. Resolving the select into this first keeps the
/// screen borrow short, so handling can then mutate it freely.
enum Wake {
    Key(TermEvent),
    Conversation(Option<crate::conversation::Event>),
    Tick,
}

/// Waits on the conversation worker when one exists, and otherwise never
/// resolves, so the same `select!` serves every screen.
async fn next_conversation_event(screen: &mut Screen) -> Option<crate::conversation::Event> {
    match screen {
        Screen::Chat(chat) => chat.conversation.next_event().await,
        _ => std::future::pending().await,
    }
}

fn draw(terminal: &mut Tui, screen: &mut Screen, tick: usize, selection: bool) -> Result<()> {
    terminal.draw(|frame| match screen {
        Screen::Launch(p) => picker::draw(
            frame,
            p,
            "CLANKER COMMAND CENTER",
            current_dir().as_deref(),
            selection,
            "↑/↓ move · Enter open · r rename · d delete · q quit",
            tick,
        ),
        Screen::Deploy(deployment) => picker::draw_deployment(frame, deployment),
        Screen::Chat(chat) => render::draw(frame, &chat.app, &mut chat.transcript_cache, tick),
    })?;
    Ok(())
}

/// Drives the screens, and owns whatever they leave running.
///
/// The parked sessions are cleaned up here rather than by the caller so that
/// every way out of the loop — quitting, or an error on the way to a frame —
/// goes through the same shutdown.
async fn event_loop(terminal: &mut Tui, context: &Context, screen: &mut Screen) -> Result<()> {
    let mut parked = Parked::new();
    let result = run_screens(terminal, context, screen, &mut parked).await;
    // Cancelled rather than left to finish, for the reason the foreground
    // conversation is: the process is going, and no task outlives it.
    for chat in parked {
        chat.conversation.send(Command::Cancel);
        chat.conversation.shutdown().await;
    }
    result
}

async fn run_screens(
    terminal: &mut Tui,
    context: &Context,
    screen: &mut Screen,
    parked: &mut Parked,
) -> Result<()> {
    let mut keys = EventStream::new();
    let mut ticker = tokio::time::interval(TICK);
    // A frame that overruns the tick must not queue up the ticks it missed:
    // the default burst behaviour would fire them back to back, so a
    // transcript slow enough to draw would spend all its time drawing.
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut last_refresh = std::time::Instant::now();
    let mut tick = 0usize;
    let mut quit = false;
    // Conversation events arrive per streamed token, and a redraw costs the
    // whole transcript. Drawing on each one makes a reply cost the transcript
    // times the number of tokens in it — so they only mark the screen stale
    // and the ticker below decides when to spend a frame. Keystrokes are not
    // coalesced: typing has to feel immediate.
    let mut stale = false;

    draw(terminal, screen, tick, context.selection)?;

    while !quit {
        let wake = tokio::select! {
            Some(Ok(event)) = keys.next() => Wake::Key(event),
            event = next_conversation_event(screen) => Wake::Conversation(event),
            _ = ticker.tick() => Wake::Tick,
        };

        let mut dirty = false;
        match wake {
            Wake::Key(TermEvent::Key(key)) if key.kind == KeyEventKind::Press => {
                quit = handle_key(context, screen, parked, key).await?;
                dirty = true;
            }
            Wake::Key(TermEvent::Paste(text)) => {
                if let Screen::Chat(chat) = screen {
                    chat.app.paste(&text);
                }
                dirty = true;
            }
            Wake::Key(TermEvent::Resize(_, _)) => dirty = true,
            Wake::Key(TermEvent::Mouse(mouse)) => {
                if let Screen::Chat(chat) = screen {
                    handle_mouse_scroll(&mut chat.app, mouse);
                    dirty = true;
                }
            }
            Wake::Key(_) => {}
            Wake::Conversation(Some(event)) => {
                if let Screen::Chat(chat) = screen {
                    chat.app.apply(event);
                }
                stale = true;
            }
            // The worker stopped on its own; nothing more will arrive.
            Wake::Conversation(None) => {}
            Wake::Tick => {
                // Whatever the deltas since the last frame added, drawn once.
                if stale {
                    stale = false;
                    dirty = true;
                }
                // A `$` command run outside a turn leaves `busy` false, so
                // without the second half its spinner sits on one frame.
                if matches!(screen, Screen::Chat(chat)
                    if chat.app.busy || chat.app.pending_shell.is_some())
                {
                    tick = tick.wrapping_add(1);
                    dirty = true;
                }
                // The picker animates too, but only while it has something
                // to animate: a list of idle sessions shouldn't repaint ten
                // times a second for nothing.
                if matches!(screen, Screen::Launch(p) if p.has_working_session()) {
                    tick = tick.wrapping_add(1);
                    dirty = true;
                }
                // Parked sessions carry on talking with nothing drawing
                // them. Their events are applied anyway: an approval that
                // arrives while you are elsewhere has to be waiting in the
                // box when you come back, and a transcript missing the
                // middle of a turn is worse than one that lagged.
                for chat in parked.iter_mut() {
                    while let Some(event) = chat.conversation.try_next_event() {
                        chat.app.apply(event);
                    }
                }
                // One that has finished has nothing left to watch, and its
                // claim is only keeping it from being opened anywhere else.
                let mut i = 0;
                while i < parked.len() {
                    // The approval check is belt and braces: an approval
                    // pauses a turn without ending it, so `busy` covers it —
                    // but letting go of a session that is waiting on an
                    // answer would have the worker deny it on the way out,
                    // which is the one outcome nobody asked for.
                    if parked[i].app.busy || parked[i].app.pending_approval.is_some() {
                        i += 1;
                    } else {
                        parked.remove(i).conversation.leave().await;
                    }
                }
                // Sessions move on while the picker is open — including ones
                // running in another terminal — so it re-reads rather than
                // showing whatever was true when it was opened.
                if let Screen::Launch(p) = screen {
                    if last_refresh.elapsed() >= SESSION_REFRESH {
                        last_refresh = std::time::Instant::now();
                        if p.refresh(load_sessions()?, current_dir().as_deref()) {
                            dirty = true;
                        }
                    }
                }
            }
        }

        if dirty && !quit {
            // This frame shows everything up to now, including deltas that
            // were only marked stale — so they must not ask for another.
            stale = false;
            draw(terminal, screen, tick, context.selection)?;
        }
    }

    Ok(())
}

/// Handles one keypress. Returns whether the TUI should exit.
async fn handle_key(
    context: &Context,
    screen: &mut Screen,
    parked: &mut Parked,
    key: KeyEvent,
) -> Result<bool> {
    // Quit works from anywhere, including mid-turn.
    if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('c')) {
        return Ok(true);
    }

    match screen {
        Screen::Launch(_) => handle_picker_key(context, screen, parked, key).await,
        Screen::Deploy(_) => handle_deploy_key(context, screen, key),
        Screen::Chat(chat) => {
            if handle_chat_key(&mut chat.app, &chat.conversation, key) {
                let busy = chat.app.busy;
                let Screen::Chat(chat) =
                    std::mem::replace(screen, Screen::Launch(Box::new(launch_picker()?)))
                else {
                    unreachable!("just matched Chat")
                };
                if busy {
                    // Left working, and parked whole rather than let go of:
                    // the worker keeps its claim, the approval it is about
                    // to ask still has somewhere to be answered, and coming
                    // back is this same screen rather than a fresh one.
                    parked.push(chat);
                } else {
                    // Idle, so this is instant, and waiting is what lets an
                    // unused session be discarded before the list is built.
                    chat.conversation.leave().await;
                    // The list was loaded before that flushed this session,
                    // so refresh it to show the up-to-date title.
                    *screen = Screen::Launch(Box::new(launch_picker()?));
                }
            }
            Ok(false)
        }
    }
}

async fn handle_picker_key(
    context: &Context,
    screen: &mut Screen,
    parked: &mut Parked,
    key: KeyEvent,
) -> Result<bool> {
    let Screen::Launch(p) = screen else {
        unreachable!("picker screen only")
    };

    // Cleared before the key is acted on, not after: a notice reports what
    // the *last* keypress did, and leaving it up makes a stale failure look
    // like a fresh one. Anything below is free to set a new one, which is
    // why this is the first thing rather than the last. Nothing used to
    // clear it at all — survivable while the only way to get one was a
    // session deleted from under the list, and much less so now that
    // opening a session another terminal holds is a normal thing to hit.
    p.notice = None;

    // A pending rename swallows everything until it's answered.
    if p.renaming.is_some() {
        match key.code {
            KeyCode::Esc => p.cancel_rename(),
            KeyCode::Enter => {
                if let Some((id, title)) = p.confirm_rename() {
                    // A parked session has a worker holding the same title
                    // in memory, so it is told rather than written around —
                    // otherwise the two disagree and the name you gave it is
                    // the one that isn't on screen when you go back in.
                    // Unlike deleting, there is nothing unsafe here to
                    // refuse: renaming a running session is a fine thing to
                    // want, it just has two places to land.
                    match parked.iter().find(|chat| chat.app.session_id == id) {
                        Some(chat) => chat.conversation.send(Command::SetTitle(title.clone())),
                        None => {
                            let conn = store::open_db()?;
                            store::set_session_title(&conn, &id, &title)?;
                        }
                    }
                    p.apply_rename(&id, title);
                }
            }
            KeyCode::Backspace => p.rename_backspace(),
            KeyCode::Char(c) if is_typed_char(&key) => p.rename_insert_char(c),
            _ => {}
        }
        return Ok(false);
    }

    // A pending repoint swallows everything until it's answered.
    if p.confirming_repoint.is_some() {
        let action = match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => p.resolve_repoint(true),
            _ => p.resolve_repoint(false),
        };
        if let Some(Activation::Repoint(row)) = action {
            match open_row_here(context, &row) {
                Ok(chat) => *screen = Screen::Chat(Box::new(chat)),
                Err(e) => {
                    if let Screen::Launch(p) = screen {
                        p.notice = Some(e.to_string());
                    }
                }
            }
        }
        return Ok(false);
    }

    // A pending delete swallows everything until it's answered.
    if p.confirming_delete.is_some() {
        let action = match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => p.resolve_delete(true),
            _ => p.resolve_delete(false),
        };
        if let Some(Activation::Delete(row)) = action {
            // Refused at the last moment rather than at the prompt, because
            // this is where the row being deleted is known for certain. A
            // worker is still writing turns into it; deleting the row from
            // under one leaves it writing into nothing.
            if parked.iter().any(|chat| chat.app.session_id == row.id) {
                p.notice = Some(format!(
                    "Clanker {} is still working — it can be deleted once it stops",
                    &row.id[..8]
                ));
            } else {
                let conn = store::open_db()?;
                store::delete_session(&conn, &row.id)?;
                p.remove_session(&row.id);
            }
        }
        return Ok(false);
    }

    match key.code {
        KeyCode::Char('q') => return Ok(true),
        KeyCode::Up | KeyCode::Char('k') => p.move_up(),
        KeyCode::Down | KeyCode::Char('j') => p.move_down(),
        // Rename and delete used to live on the separate browser; with one
        // list they belong here.
        KeyCode::Char('r') => p.begin_rename(),
        KeyCode::Char('d') => p.begin_delete(),
        KeyCode::Enter => {
            let Some(activation) = p.activate() else {
                return Ok(false);
            };
            match activation {
                Activation::NewSession => {
                    *screen = Screen::Deploy(Box::new(Deployment::new(
                        session::new_id(),
                        context.default_model.clone(),
                        context.effort_level.clone(),
                        context.temperature,
                        context.sandbox,
                    )))
                }
                Activation::Resume(row) => {
                    match parked.iter().position(|chat| chat.app.session_id == row.id) {
                        // One we parked is already ours and already claimed,
                        // so there is nothing to open: the screen is handed
                        // straight back, mid-turn and mid-approval if that is
                        // where it got to.
                        Some(at) => *screen = Screen::Chat(parked.remove(at)),
                        None => match open_row(context, &row) {
                            Ok(chat) => *screen = Screen::Chat(Box::new(chat)),
                            // Neither failure is fatal: a session that won't
                            // open is one row in a list of them, and taking
                            // the whole TUI down would lose access to every
                            // other session too.
                            Err(OpenFailure::MissingDir(dir)) => {
                                if let Screen::Launch(p) = screen {
                                    p.begin_repoint(row, dir);
                                }
                            }
                            Err(OpenFailure::Other(e)) => {
                                if let Screen::Launch(p) = screen {
                                    p.notice = Some(e.to_string());
                                }
                            }
                        },
                    }
                }
                // Delete and repoint are resolved by their confirmation
                // flows, not here.
                Activation::Delete(_) | Activation::Repoint(_) => {}
            }
        }
        _ => {}
    }
    Ok(false)
}

/// Handles a keypress on the Clanker Deployment screen. Returns whether the
/// TUI should exit (always `false` — quitting from here isn't supported,
/// same as any other picker screen).
fn handle_deploy_key(context: &Context, screen: &mut Screen, key: KeyEvent) -> Result<bool> {
    let Screen::Deploy(deployment) = screen else {
        unreachable!("deployment screen only")
    };

    match key.code {
        KeyCode::Esc => *screen = Screen::Launch(Box::new(launch_picker()?)),
        KeyCode::Enter => match deployment.plan() {
            Ok(plan) => {
                let orders = plan.orders.clone();
                let chat = open_new(context, plan)?;
                // Sent rather than typed for you: it goes to the worker the
                // way any message does, so the transcript, the activity the
                // picker reads, and the turn itself all begin exactly as if
                // you had opened the clanker and typed it.
                if let Some(orders) = orders {
                    chat.conversation.send(Command::Send(orders));
                }
                *screen = Screen::Chat(Box::new(chat));
            }
            // Nothing is created and nothing is lost: the form stays as it
            // is with the reason under it, which is the only place the fix
            // can be made.
            Err(error) => deployment.error = Some(error),
        },
        // Another id, which means another mark: the one thing about a
        // clanker you cannot change afterwards, so it is worth being able to
        // roll for one you like before it exists. Tab rather than a letter
        // because most of this screen is fields you type into.
        KeyCode::Tab => deployment.id = session::new_id(),
        KeyCode::Up => deployment.move_up(),
        KeyCode::Down => deployment.move_down(),
        KeyCode::Left => deployment.change(-1),
        KeyCode::Right => deployment.change(1),
        KeyCode::Backspace => deployment.backspace(),
        KeyCode::Char(c) if is_typed_char(&key) => deployment.type_char(c),
        _ => {}
    }
    Ok(false)
}

/// How many transcript lines one wheel notch moves — a finer step than
/// PageUp/PageDown's 5, since a notch is closer to a nudge than a page.
const MOUSE_SCROLL_STEP: u16 = 3;

/// Scrolls the transcript with the wheel, the mouse counterpart to
/// PageUp/PageDown. Left `app.scroll_back` untouched (and the input box
/// alone) for any other mouse event — clicks aren't wired to anything yet.
fn handle_mouse_scroll(app: &mut App, mouse: MouseEvent) {
    match mouse.kind {
        MouseEventKind::ScrollUp => {
            app.scroll_back = app.scroll_back.saturating_add(MOUSE_SCROLL_STEP);
        }
        MouseEventKind::ScrollDown => {
            app.scroll_back = app.scroll_back.saturating_sub(MOUSE_SCROLL_STEP);
        }
        _ => {}
    }
}

/// Whether a `Char` keypress is someone typing, rather than a chord that
/// happens to carry a letter. Without this every unhandled Ctrl-combination
/// types its bare letter — Ctrl-V, the paste chord users reach for first,
/// put a stray `v` in the input box instead of doing nothing.
///
/// Only CONTROL disqualifies a keypress. SHIFT is how capitals arrive, and
/// ALT composes real characters on some layouts and terminals, so neither
/// can be treated as "not typing".
fn is_typed_char(key: &KeyEvent) -> bool {
    !key.modifiers.contains(KeyModifiers::CONTROL)
}

/// Answers a pending approval. Shared by the chord and by `/allow`, `/deny` —
/// the typed forms exist because a terminal that claims `Ctrl-Y` would
/// otherwise leave a turn waiting on a decision with no way to give it.
fn answer_approval(app: &mut App, send: &mut impl FnMut(Command), allowed: bool) {
    if app.pending_approval.is_none() {
        return;
    }
    send(Command::Approve(allowed));
    app.approval_answered(allowed);
}

/// Settles a finished `$` command, sending its output to the model or not.
/// Does nothing when no command is waiting, so the keys and the commands are
/// both inert the rest of the time.
fn settle_shell(app: &mut App, send: &mut impl FnMut(Command), sent: bool) {
    if let Some(text) = app.settle_shell(sent) {
        send(Command::Include(text));
    }
}

/// Handles a keypress in the conversation. Returns whether to leave it and
/// go back to the launch screen.
/// Acts on one submitted line: what the worker must be told, and what
/// this side shows for itself.
///
/// Split out of the key handler and given a `send` sink rather than the
/// `Conversation` so it can be driven from a test. It was not, and the
/// first thing that went wrong here was invisible to every test in the
/// suite: `/models` was routed to the worker *and* needed to open a box,
/// and the dispatch below treats those as alternatives, so the box never
/// opened and the fetched list arrived with nowhere to land.
/// Returns whether the chat screen should be left (`/back`).
fn dispatch_submission(app: &mut App, text: &str, send: &mut impl FnMut(Command)) -> bool {
    let submission = app::classify(text);

    // One box, one command. Two in flight would race on the same
    // slot — whichever finished last would land on top of the
    // other's output, and the first would be lost with no record
    // that it ran.
    if matches!(submission, app::Submission::Shell(_))
        && matches!(app.pending_shell, Some(ShellState::Running { .. }))
    {
        app.transcript.push(TranscriptItem::Notice(
            "A command is still running — wait for it to finish".to_string(),
        ));
        return false;
    }

    // The one submission that is both. Everything else either
    // goes to the worker or is answered locally, so the dispatch
    // below treats those as alternatives — which silently
    // swallowed this: the fetch was sent, the box was never
    // opened, and the list arrived with nowhere to land.
    if matches!(submission, app::Submission::BrowseModels) {
        app.model_browser = Some(app::ModelBrowser::Loading);
    }

    match command_for(&submission) {
        // Everything that changes session state is the worker's
        // to apply; it replies with the event that updates the
        // view, so the two can't disagree.
        Some(command) => send(command),
        // The rest are read-only, answered from state this side
        // already holds. `command_for` is the exhaustive match,
        // so a new submission variant is classified there first.
        None => {
            match submission {
                // Round-tripped rather than read locally, so the
                // answer reflects what the session actually holds.
                app::Submission::ShowModel => send(Command::SetModel(app.model.clone())),
                app::Submission::ShowHelp => {
                    app.transcript
                        .push(TranscriptItem::Help(crate::ui::help_rows()));
                }
                // Opened above, then dispatched to the worker
                // by `command_for` — so this arm never runs. It
                // exists because the match is exhaustive, which
                // is what will make the next person adding a
                // submission think about which side it belongs
                // on.
                app::Submission::BrowseModels => {}
                app::Submission::ShowEffort => {
                    app.transcript
                        .push(TranscriptItem::Notice(crate::ui::effort_notice(
                            app.effort_level.as_deref(),
                            false,
                        )));
                }
                app::Submission::ShowStatus => {
                    let tool_access = app.tool_access.clone();
                    let rows = crate::ui::session_settings_rows(&crate::ui::SessionSettings {
                        id: app.short_id(),
                        title: &app.title,
                        model: &app.model,
                        effort_level: app.effort_level.as_deref(),
                        temperature: app.temperature,
                        max_iterations: app.max_iterations,
                        verbose: app.verbose,
                        highlight: app.highlight,
                        sandbox: app.sandbox,
                        stream: app.stream,
                        working_dir: app.working_dir.as_deref(),
                        tool_access: &tool_access,
                        total_tokens: app.total_tokens,
                    });
                    app.transcript.push(TranscriptItem::SessionStatus(rows));
                }
                app::Submission::ShowHighlight => {
                    app.transcript
                        .push(TranscriptItem::Notice(crate::ui::highlight_notice(
                            app.highlight,
                            false,
                        )));
                }
                app::Submission::ShowVerbose => {
                    app.transcript
                        .push(TranscriptItem::Notice(crate::ui::verbose_notice(
                            app.verbose,
                            false,
                        )));
                }
                app::Submission::ShowTemperature => {
                    app.transcript
                        .push(TranscriptItem::Notice(crate::ui::temperature_notice(
                            app.temperature,
                            false,
                        )));
                }
                app::Submission::ShowTitle => {
                    app.transcript
                        .push(TranscriptItem::Notice(crate::ui::title_notice(
                            &app.title, false,
                        )));
                }
                app::Submission::ShowStream => {
                    app.transcript
                        .push(TranscriptItem::Notice(crate::ui::stream_notice(
                            app.stream, false,
                        )));
                }
                app::Submission::ShowSandbox => {
                    app.transcript
                        .push(TranscriptItem::Notice(crate::ui::sandbox_notice(
                            app.sandbox,
                            false,
                        )));
                }
                app::Submission::ShowTools => {
                    app.transcript.push(TranscriptItem::ToolStatus {
                        access: app.tool_access.clone(),
                        changed: false,
                    });
                }
                // Typed equivalents of the box's chords, for a
                // terminal or multiplexer that has claimed them.
                app::Submission::SendShell => settle_shell(app, send, true),
                app::Submission::DiscardShell => settle_shell(app, send, false),
                app::Submission::AllowTool => answer_approval(app, send, true),
                app::Submission::DenyTool => answer_approval(app, send, false),
                app::Submission::Back => return true,
                app::Submission::UnknownCommand(message) => {
                    app.transcript.push(TranscriptItem::Error(message));
                }
                // Listed rather than caught by `_`, so adding
                // a submission has to be considered on this side
                // too — a catch-all here would silently ignore a
                // new read-only one.
                app::Submission::Message(_)
                | app::Submission::SetModel(_)
                | app::Submission::SetEffort(_)
                | app::Submission::ResetEffort
                | app::Submission::SetVerbose(_)
                | app::Submission::SetHighlight(_)
                | app::Submission::SetStream(_)
                | app::Submission::SetTitle(_)
                | app::Submission::SetSandbox(_)
                | app::Submission::SetMaxIterations(_)
                | app::Submission::ResetMaxIterations
                | app::Submission::SetTemperature(_)
                | app::Submission::ResetTemperature
                | app::Submission::Shell(_)
                | app::Submission::SetToolAccess { .. }
                | app::Submission::ResetToolAccess => {
                    unreachable!("command_for routes these to the worker")
                }
            }
        }
    }
    false
}

fn handle_chat_key(app: &mut App, conversation: &Conversation, key: KeyEvent) -> bool {
    // The browser holds the keyboard while it is open. Unlike an approval —
    // which arrives unbidden, and so was deliberately kept out of the input
    // box — this is something you asked for a keystroke ago, and there is
    // nothing else to be typing at it.
    if app.model_browser.is_some() {
        match key.code {
            KeyCode::Esc => app.model_browser = None,
            KeyCode::Up => app.browser_move(false),
            KeyCode::Down => app.browser_move(true),
            KeyCode::Backspace => app.browser_filter_pop(),
            KeyCode::Enter => {
                if let Some(model) = app.model_browser.as_ref().and_then(|b| b.highlighted()) {
                    conversation.send(Command::SetModel(model));
                }
                app.model_browser = None;
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.browser_filter_push(c)
            }
            _ => {}
        }
        return false;
    }

    // Ctrl-B backs out to the launch screen; plain Esc stays reserved for
    // cancelling a turn, which is needed far more often.
    if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('b')) {
        return true;
    }

    // Answering an approval, from anywhere. Deliberately a chord rather than
    // a bare y/n: the input box stays live while a decision is owed, so a
    // plain letter has to keep meaning the letter. This is why the approval
    // no longer needs a mode of its own.
    if key.modifiers.contains(KeyModifiers::CONTROL) && app.pending_approval.is_some() {
        if let KeyCode::Char(c) = key.code {
            match c {
                'y' => {
                    answer_approval(app, &mut |c| conversation.send(c), true);
                    return false;
                }
                'n' => {
                    answer_approval(app, &mut |c| conversation.send(c), false);
                    return false;
                }
                _ => {}
            }
        }
    }

    // Deciding a finished `$` command. Deliberately different keys from the
    // approval's: both boxes can be open at once, and an approval can arrive
    // between reading your output and reaching for the key, so one shared
    // chord would act on whichever happened to be open.
    //
    // Ctrl-S is XOFF on a terminal with flow control on, which would freeze
    // the display until Ctrl-Q. It reaches us because raw mode clears IXON —
    // see `enter`. Anything above our tty (tmux, an ssh chain) can still
    // claim it, which is what `/send` and `/discard` are for.
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        if let KeyCode::Char(c) = key.code {
            match c {
                's' => {
                    settle_shell(app, &mut |c| conversation.send(c), true);
                    return false;
                }
                'd' => {
                    settle_shell(app, &mut |c| conversation.send(c), false);
                    return false;
                }
                _ => {}
            }
        }
    }

    match key.code {
        // Alt-Enter always inserts a newline. Shift-Enter does too wherever
        // the terminal can report the modifier — see `enter`, which asks
        // for that reporting when the terminal supports it.
        KeyCode::Enter
            if key
                .modifiers
                .intersects(KeyModifiers::ALT | KeyModifiers::SHIFT) =>
        {
            app.insert_char('\n')
        }
        KeyCode::Enter => {
            if let Some(text) = app.take_input() {
                if dispatch_submission(app, &text, &mut |c| conversation.send(c)) {
                    return true;
                }
            }
        }
        KeyCode::Esc => {
            if app.busy {
                conversation.send(Command::Cancel);
            }
        }
        // Only ever completes a command name — see `complete_command` — so
        // Tab stays free for whatever it means to the terminal everywhere
        // else, and pressing it in the middle of a message does nothing.
        KeyCode::Tab => app.complete_command(),
        KeyCode::Backspace => app.backspace(),
        KeyCode::Left => app.move_left(),
        KeyCode::Right => app.move_right(),
        KeyCode::Up => app.history_up(),
        KeyCode::Down => app.history_down(),
        KeyCode::PageUp => app.scroll_back = app.scroll_back.saturating_add(5),
        KeyCode::PageDown => app.scroll_back = app.scroll_back.saturating_sub(5),
        KeyCode::End => app.scroll_back = 0,
        KeyCode::Char(c) if is_typed_char(&key) => app.insert_char(c),
        _ => {}
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mouse(kind: MouseEventKind) -> MouseEvent {
        MouseEvent {
            kind,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::empty(),
        }
    }

    fn key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, modifiers)
    }

    fn test_context() -> Context {
        Context {
            client: Arc::new(Client::for_test(crate::config::Config::default())),
            default_model: "m".to_string(),
            effort_level: None,
            max_iterations: Some(20),
            temperature: None,
            tool_access: ToolAccessSettings::default(),
            sandbox: true,
            verbose: false,
            highlight: true,
            selection: true,
            stream: true,
        }
    }

    /// A deployment screen holding `name`, as the launch screen would build
    /// it from the configured defaults.
    fn deploy_screen(name: &str) -> Screen {
        let mut deployment = Deployment::new(
            session::new_id(),
            "test-model".to_string(),
            None,
            Some(0.7),
            true,
        );
        deployment.name = name.to_string();
        Screen::Deploy(Box::new(deployment))
    }

    #[test]
    fn a_blank_title_does_not_deploy_a_clanker() {
        // Naming it is the deliberate act of creating one, so Enter on an
        // empty name has nothing to do — it must not fall through to
        // deploying an untitled clanker.
        //
        // Only the refused paths are tested here: a deploy that succeeds
        // reaches the database, and a unit test has no business writing to
        // the user's own.
        let context = test_context();
        let mut screen = deploy_screen("   ");

        handle_deploy_key(&context, &mut screen, KeyEvent::from(KeyCode::Enter)).unwrap();

        let Screen::Deploy(deployment) = &screen else {
            panic!("a blank name should leave you on the deployment screen")
        };
        assert_eq!(deployment.error.as_deref(), Some("A name is required"));

        // And typing takes the complaint back down: it is about to stop
        // being true.
        handle_deploy_key(&context, &mut screen, KeyEvent::from(KeyCode::Char('a'))).unwrap();
        let Screen::Deploy(deployment) = &screen else {
            unreachable!()
        };
        assert_eq!(deployment.error, None);
    }

    #[test]
    fn a_temperature_that_is_not_a_number_is_refused_here() {
        // Caught on the form rather than at creation: the alternative is a
        // clanker that exists and fails on its first turn.
        let context = test_context();
        let mut screen = deploy_screen("Parser work");
        let Screen::Deploy(deployment) = &mut screen else {
            unreachable!()
        };
        deployment.focus = picker::Field::Temperature;
        deployment.temperature = "warm".to_string();

        handle_deploy_key(&context, &mut screen, KeyEvent::from(KeyCode::Enter)).unwrap();

        let Screen::Deploy(deployment) = &screen else {
            panic!("a bad temperature should leave you on the deployment screen")
        };
        assert!(
            deployment
                .error
                .as_deref()
                .is_some_and(|e| e.contains("Temperature")),
            "{:?}",
            deployment.error
        );
    }

    #[test]
    fn tab_rolls_a_different_clanker_and_leaves_the_form_alone() {
        // The mark is hashed from the id, so a new id is the only way to a
        // new mark — and it has to happen before creation, since afterwards
        // the id is what everything else refers to.
        let context = test_context();
        let mut screen = deploy_screen("Parser work");
        let Screen::Deploy(deployment) = &screen else {
            unreachable!()
        };
        let before = deployment.id.clone();

        handle_deploy_key(&context, &mut screen, KeyEvent::from(KeyCode::Tab)).unwrap();

        let Screen::Deploy(deployment) = &screen else {
            panic!("Tab left the deployment screen")
        };
        assert_ne!(deployment.id, before, "Tab must roll a different id");
        assert_eq!(
            deployment.name, "Parser work",
            "and must not touch what was typed"
        );

        // Typing is not rerolling: only Tab moves the mark, or it would
        // change under you as you named the thing.
        let rolled = deployment.id.clone();
        handle_deploy_key(&context, &mut screen, KeyEvent::from(KeyCode::Char('!'))).unwrap();
        let Screen::Deploy(deployment) = &screen else {
            unreachable!()
        };
        assert_eq!(deployment.id, rolled);
        assert_eq!(deployment.name, "Parser work!");
    }

    #[test]
    fn the_form_walks_its_fields_and_changes_only_the_focused_one() {
        let context = test_context();
        let mut screen = deploy_screen("Parser work");

        // Down from the name lands on Tools, which Space and ←/→ toggle —
        // and typing must not reach it, or a stray letter would silently
        // arm a clanker.
        handle_deploy_key(&context, &mut screen, KeyEvent::from(KeyCode::Down)).unwrap();
        let Screen::Deploy(deployment) = &screen else {
            unreachable!()
        };
        assert_eq!(deployment.focus, picker::Field::Tools);
        assert!(!deployment.tools, "a fresh form deploys with no tools");

        handle_deploy_key(&context, &mut screen, KeyEvent::from(KeyCode::Right)).unwrap();
        let Screen::Deploy(deployment) = &screen else {
            unreachable!()
        };
        assert!(deployment.tools);
        assert_eq!(
            deployment.name, "Parser work",
            "the name is not the focused field any more"
        );

        handle_deploy_key(&context, &mut screen, KeyEvent::from(KeyCode::Char('x'))).unwrap();
        let Screen::Deploy(deployment) = &screen else {
            unreachable!()
        };
        assert_eq!(
            deployment.name, "Parser work",
            "a letter aimed at a choice field types nowhere"
        );
    }

    #[test]
    fn a_control_chord_does_not_type_its_letter() {
        // Ctrl-V is the paste chord people reach for first. Most terminals
        // don't treat it as paste, so it arrives here as an ordinary
        // keypress — and used to leave a stray `v` in the input box.
        assert!(!is_typed_char(&key(
            KeyCode::Char('v'),
            KeyModifiers::CONTROL
        )));
    }

    #[test]
    fn ordinary_typing_still_types() {
        assert!(is_typed_char(&key(
            KeyCode::Char('v'),
            KeyModifiers::empty()
        )));
        // Capitals arrive carrying SHIFT, and ALT composes real characters
        // on some layouts — neither means "not typing".
        assert!(is_typed_char(&key(KeyCode::Char('V'), KeyModifiers::SHIFT)));
        assert!(is_typed_char(&key(KeyCode::Char('e'), KeyModifiers::ALT)));
    }

    #[test]
    fn wheel_scrolls_the_transcript_by_a_wheel_step() {
        let mut app = App::new("m".to_string(), None, "id".to_string());
        assert_eq!(app.scroll_back, 0);

        handle_mouse_scroll(&mut app, mouse(MouseEventKind::ScrollUp));
        assert_eq!(app.scroll_back, MOUSE_SCROLL_STEP);

        handle_mouse_scroll(&mut app, mouse(MouseEventKind::ScrollUp));
        assert_eq!(app.scroll_back, MOUSE_SCROLL_STEP * 2);

        handle_mouse_scroll(&mut app, mouse(MouseEventKind::ScrollDown));
        assert_eq!(app.scroll_back, MOUSE_SCROLL_STEP);
    }

    #[test]
    fn wheel_does_not_scroll_past_the_newest_message() {
        let mut app = App::new("m".to_string(), None, "id".to_string());
        handle_mouse_scroll(&mut app, mouse(MouseEventKind::ScrollDown));
        assert_eq!(app.scroll_back, 0, "can't scroll below the newest content");
    }

    #[test]
    fn other_mouse_events_are_ignored() {
        let mut app = App::new("m".to_string(), None, "id".to_string());
        handle_mouse_scroll(
            &mut app,
            mouse(MouseEventKind::Down(crossterm::event::MouseButton::Left)),
        );
        assert_eq!(app.scroll_back, 0);
    }
}

#[cfg(test)]
mod dispatch_tests {
    use super::*;
    use crate::tui::app::{App, ModelBrowser};

    /// Runs one submitted line and reports what the worker was told.
    fn dispatch(app: &mut App, text: &str) -> Vec<Command> {
        let mut sent = Vec::new();
        dispatch_submission(app, text, &mut |command| sent.push(command));
        sent
    }

    fn app() -> App {
        App::new("m".to_string(), None, "abcd1234".to_string())
    }

    #[test]
    fn asking_for_models_both_fetches_and_opens_the_box() {
        // The bug this exists for: `/models` needs the worker to fetch *and*
        // this side to open the browser, but the dispatch treats those as
        // alternatives. It was routed to the worker, the box never opened,
        // and the list arrived with nowhere to land — so `/models` displayed
        // nothing at all. Every unit test passed, because none of them ran
        // the wiring.
        let mut app = app();
        let sent = dispatch(&mut app, "/models");

        assert!(
            matches!(app.model_browser, Some(ModelBrowser::Loading)),
            "the box should open at once, not when the list lands"
        );
        assert!(
            matches!(sent[..], [Command::ListModels]),
            "and the fetch should be on its way: {sent:?}"
        );
    }

    #[test]
    fn a_plain_message_goes_to_the_worker_untouched() {
        let mut app = app();
        let sent = dispatch(&mut app, "hello there");
        assert!(matches!(sent[..], [Command::Send(_)]), "{sent:?}");
        assert!(app.model_browser.is_none());
    }

    #[test]
    fn a_read_only_command_is_answered_here_and_not_sent() {
        let mut app = app();
        let sent = dispatch(&mut app, "/help");
        assert!(sent.is_empty(), "nothing to ask the worker: {sent:?}");
        assert!(matches!(
            app.transcript.last(),
            Some(TranscriptItem::Help(_))
        ));
    }

    #[test]
    fn back_reports_that_the_screen_should_be_left() {
        let mut app = app();
        let mut sent = Vec::new();
        assert!(dispatch_submission(&mut app, "/back", &mut |c| sent.push(c)));
        assert!(!dispatch_submission(&mut app, "/help", &mut |c| sent.push(c)));
    }

    #[test]
    fn a_misspelled_command_is_reported_rather_than_sent() {
        let mut app = app();
        let sent = dispatch(&mut app, "/mdoel gpt-5");
        assert!(sent.is_empty(), "{sent:?}");
        assert!(matches!(
            app.transcript.last(),
            Some(TranscriptItem::Error(_))
        ));
    }
}
