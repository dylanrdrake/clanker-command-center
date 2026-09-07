//! The boundary between what the app *does* and how that is presented.
//!
//! The agent loop in [`crate::agent`] is pure orchestration: it decides what
//! to call and in what order, and reports progress by emitting
//! [`AgentEvent`]s to an [`AgentUi`] rather than printing. Anything that
//! needs a decision from the user goes through [`AgentUi::approve`].
//!
//! That keeps a second front end (a GUI, a web server, a test harness) from
//! having to fork the loop just to render it differently: it implements this
//! trait instead. [`crate::terminal_ui`] is the CLI's implementation.

use crate::config::{ToolAccess, ToolAccessSettings};
use anyhow::Result;
use std::future::Future;

/// Formats a model name with its effort level for display, e.g.
/// "openrouter/auto (high)", or just the model name when no effort is set.
pub fn response_label(model: &str, effort_level: &Option<String>) -> String {
    match effort_level {
        Some(effort) => format!("{} ({})", model, effort),
        None => model.to_string(),
    }
}

/// The one argument that best identifies what a tool call is doing — the
/// path for a file tool, the command for `run_terminal_command` — so a
/// terse, non-verbose notice can name it without dumping the full argument
/// JSON. `None` if `arguments` isn't a JSON object or has none of these.
pub fn primary_argument(arguments: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(arguments).ok()?;
    let object = value.as_object()?;
    ["filepath", "command", "dirpath"]
        .iter()
        .find_map(|key| object.get(*key).and_then(|v| v.as_str()))
        .map(str::to_string)
}

/// Flattens a possibly long/multi-line value onto one line, truncated to
/// `max` characters, for a compact preview — shared by both front ends so
/// neither drifts from the other's idea of "too long to show in full".
pub fn summarize(text: &str, max: usize) -> String {
    let flat: String = text
        .chars()
        .map(|c| if c == '\n' || c == '\t' { ' ' } else { c })
        .collect();
    let flat = flat.trim();
    if flat.chars().count() > max {
        let kept: String = flat.chars().take(max).collect();
        format!("{kept}…")
    } else {
        flat.to_string()
    }
}

/// Splits a tool call's JSON arguments/result into `(field, value)` pairs,
/// each value flattened and truncated for a single display line — the
/// per-field detail both the approval prompt and verbose tool-call notices
/// show. Empty if `text` isn't a JSON object.
pub fn json_fields(text: &str) -> Vec<(String, String)> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(text) else {
        return Vec::new();
    };
    let Some(object) = value.as_object() else {
        return Vec::new();
    };
    object
        .iter()
        .map(|(key, value)| {
            let shown = value
                .as_str()
                .map(|s| s.to_string())
                .unwrap_or_else(|| value.to_string());
            (key.clone(), summarize(&shown, 100))
        })
        .collect()
}

/// [`json_fields`] for a tool call's arguments, plus a `working_dir` entry
/// for a `run_terminal_command` call that didn't specify one — that means
/// it runs in the current directory, which otherwise wouldn't show up in
/// the notice at all.
pub fn tool_call_fields(name: &str, arguments: &str) -> Vec<(String, String)> {
    let mut fields = json_fields(arguments);
    if name == "run_terminal_command" && !fields.iter().any(|(key, _)| key == "working_dir") {
        if let Ok(cwd) = std::env::current_dir() {
            fields.push(("working_dir".to_string(), cwd.display().to_string()));
        }
    }
    fields
}

/// Interprets a typed answer to an approval prompt. Anything other than an
/// explicit yes denies the action — a blank answer included, matching a
/// conventional `[y/N]:` prompt's default. Shared so the CLI's stdin prompt
/// and the TUI's input-box prompt agree on what counts as "yes".
pub fn parse_yes_no(input: &str) -> bool {
    let response = input.trim().to_lowercase();
    response == "y" || response == "yes"
}

/// What a submitted line turned out to be. Shared between the TUI's input
/// box and the CLI's `clanker` loop, so `/model`, `/tools`, etc. behave
/// identically in both.
///
/// Recognized commands are intercepted, and so is a line that names one of
/// them but doesn't invoke it validly — see [`Submission::UnknownCommand`].
/// That's confidently a failed command, not text meant for the model, so it
/// is never sent either. Anything else is an ordinary message, including
/// text that merely starts with a slash but doesn't name a known command at
/// all, since paths like `/etc/hosts` are common enough in a coding tool
/// that swallowing them would be worse than never catching a genuine typo.
#[derive(Debug, Clone, PartialEq)]
pub enum Submission {
    Message(String),
    /// `$ <command>` — run it here, now, without involving the model. The
    /// user typed it, so there is nothing to approve.
    Shell(String),
    /// Put the last command's output into the conversation, or don't. Typed
    /// forms of the box's `Ctrl-S`/`Ctrl-D`, so the decision survives a
    /// terminal or multiplexer that has claimed those chords.
    SendShell,
    DiscardShell,
    /// `/highlight <on|off>` — whether this session bands your own messages.
    SetHighlight(bool),
    /// Prints it without changing it.
    ShowHighlight,
    /// Answer a tool approval. Typed forms of `Ctrl-Y`/`Ctrl-N`, and not
    /// optional: where those chords are intercepted — Zed's terminal claims
    /// several — an approval would otherwise be unanswerable and the turn
    /// stuck with nothing but cancelling to get out of it.
    AllowTool,
    DenyTool,
    /// Typed form of `Ctrl-B`. tmux takes `Ctrl-B` as its own prefix, so
    /// without this there is no way back to the launch screen from inside a
    /// tmux session.
    Back,
    SetModel(String),
    ShowModel,
    /// `None` nullifies the override (`/effort clear`) — no effort field is
    /// sent, regardless of the configured default, until set again.
    SetEffort(Option<String>),
    /// `/effort default` — reads the *currently* configured default and
    /// saves that concrete value to the session now, distinct from
    /// [`Submission::SetEffort`]`(None)`, which nullifies instead.
    ResetEffort,
    /// Shows full tool-call detail for this session, or stops. Bare
    /// `/verbose` reads it rather than flipping it — every other setting
    /// answers a bare command with its current value, and a toggle can't be
    /// written down as an instruction without knowing where it started.
    SetVerbose(bool),
    /// Prints/shows whether this session is showing full detail, without
    /// changing it.
    ShowVerbose,
    /// Prints/shows this session's sampling temperature without changing it.
    ShowTemperature,
    /// Prints/shows the reasoning effort level without changing it.
    ShowEffort,
    /// Opens the model browser. TUI only: it is a list you move a cursor
    /// through, which the CLI's blocking prompt has nowhere to put.
    BrowseModels,
    /// Streams replies token-by-token for this session, or waits for the
    /// whole reply. The configured default seeds it; this changes the
    /// session, like `/verbose` and `/sandbox`.
    SetStream(bool),
    /// Prints/shows whether this session streams, without changing it.
    ShowStream,
    /// `None` nullifies the override (`/max-iterations clear`) — turns fall
    /// back to whatever the configured default is at the time each one
    /// runs, regardless of what it was when this session started.
    SetMaxIterations(Option<usize>),
    /// `/max-iterations default` — reads the *currently* configured default
    /// and saves that concrete value to the session now, distinct from
    /// [`Submission::SetMaxIterations`]`(None)`, which nullifies instead.
    ResetMaxIterations,
    /// `None` nullifies the override (`/temperature clear`), same deal as
    /// [`Submission::SetMaxIterations`].
    SetTemperature(Option<f32>),
    /// `/temperature default` — same deal as [`Submission::ResetMaxIterations`].
    ResetTemperature,
    /// `target` is a tool's name, a category, or "all" — checked against
    /// the tool table by whoever applies it, since a typo there deserves a
    /// message rather than a silent no-op.
    SetToolAccess {
        target: String,
        access: ToolAccess,
    },
    /// `/tools on` — every tool back to what it does by default.
    ResetToolAccess,
    /// Prints/shows what each tool may do, without changing anything.
    ShowTools,
    /// Confines the agent's file writes to the working directory, or lets
    /// them go anywhere. The read tools are unaffected either way.
    SetSandbox(bool),
    /// Prints/shows whether writes are currently confined, without changing
    /// it.
    ShowSandbox,
    /// Prints/shows every setting this session is running with, without
    /// changing any of them. Named for `clank status`, which does the same
    /// job one scope out: global configuration there, this session here.
    ShowStatus,
    /// Renames this clanker. `/clanker` is the namespace for acting on the
    /// session itself, as opposed to the settings it runs with.
    SetTitle(String),
    /// Prints/shows this session's name without changing it.
    ShowTitle,
    /// Prints/shows every in-session command and what it does.
    ShowHelp,
    /// `/compact` — fold the earlier conversation into a summary now,
    /// without waiting for it to grow past the configured threshold.
    Compact,
    /// A line that named a known command (`/effort`, `/max-iterations`,
    /// `/temperature`/`/temp`, `/tools`, `/sandbox`) but wasn't a valid
    /// invocation of
    /// it — a missing/invalid argument, a tool or category that is not one,
    /// too many or too few words, and so on. Unlike a `/<word>` that isn't a
    /// recognized command at all (kept as an ordinary [`Submission::Message`],
    /// since paths like `/etc/hosts` are common in a coding tool), this is
    /// confidently a failed command: never sent to the model, and reported
    /// by the front end as an error instead. Carries a usage hint for
    /// whichever command was misused.
    UnknownCommand(String),
}

/// Everything `/status` reports, gathered from whichever front end is
/// asking. Both hold the same state — the TUI in its `App`, the CLI in its
/// `ChatSession` — so the shape lives here and the rendering is shared,
/// rather than each growing its own list that drifts from the other.
pub struct SessionSettings<'a> {
    pub id: &'a str,
    pub title: &'a str,
    pub model: &'a str,
    pub effort_level: Option<&'a str>,
    pub temperature: Option<f32>,
    pub max_iterations: Option<usize>,
    pub verbose: bool,
    pub highlight: bool,
    pub sandbox: bool,
    pub stream: bool,
    pub working_dir: Option<&'a str>,
    pub tool_access: &'a ToolAccessSettings,
    pub total_tokens: i64,
    /// The model that compacts this clanker's history, and the prompt size
    /// that sets it going. Configuration rather than session state — every
    /// other row here is something the clanker owns — but they change what a
    /// turn does, and `/status` is where someone looks to find out why a
    /// turn paused to summarize itself.
    pub compactor: &'a str,
    pub compact_at: Option<u64>,
}

/// A token count with `,`-grouped thousands, since a long-running clanker's
/// total is exactly the kind of number that's unreadable as a bare string of
/// digits.
pub fn format_tokens(n: i64) -> String {
    let sign = if n < 0 { "-" } else { "" };
    let digits = n.unsigned_abs().to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, ch) in digits.chars().enumerate() {
        // Not `.is_multiple_of()`: that needs a newer Rust than the
        // README's stated 1.70 floor.
        #[allow(clippy::manual_is_multiple_of)]
        let at_group_boundary = i > 0 && (digits.len() - i) % 3 == 0;
        if at_group_boundary {
            grouped.push(',');
        }
        grouped.push(ch);
    }
    format!("{sign}{grouped}")
}

/// `/status` as label/value rows, ready for either front end to draw.
///
/// A setting that isn't set says what that *means* rather than showing an
/// empty cell — a nullified temperature sends no field at all, which is a
/// different thing from one that happens to equal the default.
pub fn session_settings_rows(settings: &SessionSettings) -> Vec<(String, String)> {
    let on_off = |value: bool| if value { "on" } else { "off" }.to_string();

    vec![
        ("ID".to_string(), settings.id.to_string()),
        ("Title".to_string(), settings.title.to_string()),
        (
            "Tools".to_string(),
            // Read from the tool states rather than from a mode flag beside
            // them: having tools *is* having at least one tool that is not
            // `never`, and two places to look would eventually disagree.
            if settings.tool_access.any_tools() {
                "on".to_string()
            } else {
                "off — nothing is offered to the model".to_string()
            },
        ),
        ("Model".to_string(), settings.model.to_string()),
        (
            "Effort".to_string(),
            settings
                .effort_level
                .map(str::to_string)
                .unwrap_or_else(|| "none sent".to_string()),
        ),
        (
            "Temperature".to_string(),
            settings
                .temperature
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none sent".to_string()),
        ),
        (
            "Max iterations".to_string(),
            settings
                .max_iterations
                .map(|value| value.to_string())
                .unwrap_or_else(|| "not set".to_string()),
        ),
        ("Sandbox".to_string(), on_off(settings.sandbox)),
        ("Verbose".to_string(), on_off(settings.verbose)),
        ("Highlight".to_string(), on_off(settings.highlight)),
        ("Streaming".to_string(), on_off(settings.stream)),
        (
            "Tokens".to_string(),
            format!("🪙 {}", format_tokens(settings.total_tokens)),
        ),
        (
            "Compactor".to_string(),
            match settings.compact_at {
                Some(threshold) => format!(
                    "{} at {} prompt tokens",
                    settings.compactor,
                    format_tokens(threshold as i64)
                ),
                None => format!("{} — /compact only, never automatic", settings.compactor),
            },
        ),
        (
            "Directory".to_string(),
            settings
                .working_dir
                .map(str::to_string)
                // Sessions predating this resume wherever they're run, and
                // saying so beats an empty cell that looks like a bug.
                .unwrap_or_else(|| "not recorded".to_string()),
        ),
        (
            "Each tool".to_string(),
            // Named individually rather than by category: the categories are
            // a way to set several at once, not a thing a session is in, and
            // a row that said "write: ask" would not tell you *what* asks.
            // `/tools` on its own is the fuller listing.
            settings
                .tool_access
                .rows()
                .iter()
                .map(|(name, _, access)| format!("{name} {}", access.label()))
                .collect::<Vec<_>>()
                .join(" · "),
        ),
    ]
}

/// What each tool may do, as label/value rows — the body of `clank tools`
/// and of `/tools`, so the two cannot describe the same state differently.
///
/// Ordered as [`crate::tools::TOOLS`] is: what only reads first, what
/// changes your machine last, so the dangerous end is where the eye lands.
pub fn tool_rows(access: &ToolAccessSettings) -> Vec<(String, String)> {
    crate::tools::TOOLS
        .iter()
        .map(|tool| {
            (
                tool.name.to_string(),
                format!(
                    "{:<6} {:<8} · {}",
                    access.access(tool.name).label(),
                    tool.category,
                    tool.summary
                ),
            )
        })
        .collect()
}

/// One line naming what an approval is asking about, for a list that has a
/// row per session.
///
/// Shared by both front ends deliberately: a session shows the same row in
/// the picker whether it's being run from the TUI or the CLI, and two copies
/// of this would eventually disagree about what that row says. The tool
/// alone is too vague to act on — `write_file` is a different decision
/// depending on the file — so it carries the same file-or-command the
/// transcript shows.
pub fn approval_summary(request: &ApprovalRequest) -> String {
    match primary_argument(&request.arguments) {
        Some(detail) => format!("{}: {detail}", request.tool_name),
        None => request.tool_name.clone(),
    }
}

/// How this session's name reads back to the user.
pub fn title_notice(title: &str, changed: bool) -> String {
    let verb = if changed { "renamed to" } else { "is" };
    format!("Clanker {verb} {title}")
}

/// How the streaming setting reads back to the user.
pub fn stream_notice(stream: bool, changed: bool) -> String {
    let verb = if changed { "set to" } else { "is" };
    let state = if stream {
        "on — replies arrive token by token"
    } else {
        "off — replies arrive whole"
    };
    format!("Streaming {verb} {state}")
}

/// How the verbose setting reads back to the user.
pub fn verbose_notice(verbose: bool, changed: bool) -> String {
    let verb = if changed { "set to" } else { "is" };
    let state = if verbose {
        "on — showing tool call detail and the model's thinking"
    } else {
        "off — showing a one-line notice per tool call"
    };
    format!("Verbose {verb} {state}")
}

/// How the temperature reads back to the user. `None` is a real value —
/// requests go out with no temperature field — so it says that rather than
/// showing a blank.
/// How the message band reads back to the user.
pub fn highlight_notice(highlight: bool, changed: bool) -> String {
    let verb = if changed { "is now" } else { "is" };
    let state = if highlight { "on" } else { "off" };
    format!("Message highlighting {verb} {state}")
}

pub fn temperature_notice(temperature: Option<f32>, changed: bool) -> String {
    let verb = if changed { "set to" } else { "is" };
    match temperature {
        Some(value) => format!("Temperature {verb} {value}"),
        None => format!("Temperature {verb} none sent — the provider uses its own default"),
    }
}

/// How the effort level reads back to the user. Shaped like
/// [`temperature_notice`]: both are optional, and both mean "send no field
/// at all" when unset rather than "send some default".
pub fn effort_notice(effort_level: Option<&str>, changed: bool) -> String {
    let verb = if changed { "set to" } else { "is" };
    match effort_level {
        Some(level) => format!("Effort {verb} {level}"),
        None => format!("Effort {verb} none sent — the provider uses its own default"),
    }
}

/// How the sandbox setting reads back to the user. `changed` picks "set to"
/// over "is", the same distinction every other setting's notice makes.
pub fn sandbox_notice(sandbox: bool, changed: bool) -> String {
    let verb = if changed { "set to" } else { "is" };
    let state = if sandbox {
        "on — writes confined to the working directory"
    } else {
        "off — writes allowed anywhere"
    };
    format!("Sandbox {verb} {state}")
}

/// What a clanker says when it starts compacting. Named with the model
/// doing it: the compactor is a global setting a user may not have looked at
/// in a while, and a pause explained by an unfamiliar model name is a pause
/// they can act on.
pub fn compacting_notice(model: &str) -> String {
    format!("Compacting the earlier conversation with {model}...")
}

/// What it says when the summary lands. `folded` is how many messages the
/// summary now stands in for — the whole span since the start, not just what
/// this pass added, because that is what the next request leaves out.
pub fn compacted_notice(folded: usize) -> String {
    let plural = if folded == 1 { "message" } else { "messages" };
    format!(
        "Compacted — the first {folded} {plural} are now sent as a summary.          They are still here to scroll back through."
    )
}

pub fn classify(text: &str) -> Submission {
    let trimmed = text.trim();

    // Only a leading `$` runs anything: "it cost me $5" is a message. A bare
    // `$` is one too — there is no command in it to run.
    if let Some(command) = trimmed.strip_prefix('$') {
        let command = command.trim();
        if !command.is_empty() {
            return Submission::Shell(command.to_string());
        }
    }

    match trimmed.strip_prefix("/model") {
        // "/models-are-great" is a message, not a malformed command.
        Some("") => return Submission::ShowModel,
        Some(rest) if rest.starts_with(char::is_whitespace) => {
            let name = rest.trim();
            return if name.is_empty() {
                Submission::ShowModel
            } else {
                Submission::SetModel(name.to_string())
            };
        }
        _ => {}
    }

    if let Some(rest) = trimmed.strip_prefix("/highlight") {
        if rest.trim().is_empty() {
            return Submission::ShowHighlight;
        }
    }
    if let Some(value) = argument(trimmed, "/highlight") {
        if let Ok(enabled) = parse_bool(value) {
            return Submission::SetHighlight(enabled);
        }
    }
    if let Some(rest) = trimmed.strip_prefix("/models") {
        if rest.trim().is_empty() {
            return Submission::BrowseModels;
        }
    }

    if bare_command(trimmed, "/allow") {
        return Submission::AllowTool;
    }
    if bare_command(trimmed, "/deny") {
        return Submission::DenyTool;
    }
    if bare_command(trimmed, "/back") {
        return Submission::Back;
    }
    if bare_command(trimmed, "/send") {
        return Submission::SendShell;
    }
    if bare_command(trimmed, "/discard") {
        return Submission::DiscardShell;
    }
    if let Some(rest) = trimmed.strip_prefix("/verbose") {
        if rest.trim().is_empty() {
            return Submission::ShowVerbose;
        }
    }

    if let Some(value) = argument(trimmed, "/verbose") {
        if let Ok(enabled) = parse_bool(value) {
            return Submission::SetVerbose(enabled);
        }
    }

    if let Some(rest) = trimmed.strip_prefix("/effort") {
        if rest.trim().is_empty() {
            return Submission::ShowEffort;
        }
    }

    if let Some(value) = argument(trimmed, "/effort") {
        // "clear" nullifies — no effort field is sent at all, regardless of
        // the configured default, until set again. "default" is a distinct
        // action: it reads whatever the default currently is and saves that
        // concrete value to the session now. Anything else is passed
        // through as typed rather than checked against a fixed
        // low/medium/high allowlist — models vary in what they actually
        // accept, and this is a live per-session override, easy to correct
        // if wrong, not worth gatekeeping the way the persistent global
        // default is.
        if value.eq_ignore_ascii_case("clear") {
            return Submission::SetEffort(None);
        }
        if value.eq_ignore_ascii_case("default") {
            return Submission::ResetEffort;
        }
        return Submission::SetEffort(Some(value.to_string()));
    }

    if let Some(value) = argument(trimmed, "/max-iterations") {
        if value.eq_ignore_ascii_case("clear") {
            return Submission::SetMaxIterations(None);
        }
        if value.eq_ignore_ascii_case("default") {
            return Submission::ResetMaxIterations;
        }
        // A value that isn't recognized above and isn't a positive number
        // falls through below, same as any other malformed command — no
        // distinct error variant, matching how the rest of `classify`
        // degrades.
        if let Ok(n) = value.parse::<usize>() {
            if n > 0 {
                return Submission::SetMaxIterations(Some(n));
            }
        }
    }

    // "/temp" is accepted as a shorthand for "/temperature".
    for name in ["/temperature", "/temp"] {
        if let Some(rest) = trimmed.strip_prefix(name) {
            if rest.trim().is_empty() {
                return Submission::ShowTemperature;
            }
        }
    }

    if let Some(value) = argument(trimmed, "/temperature").or_else(|| argument(trimmed, "/temp")) {
        if value.eq_ignore_ascii_case("clear") {
            return Submission::SetTemperature(None);
        }
        if value.eq_ignore_ascii_case("default") {
            return Submission::ResetTemperature;
        }
        // A value that isn't recognized above and isn't a non-negative
        // number falls through below, same as any other malformed command —
        // no distinct error variant, matching how the rest of `classify`
        // degrades.
        if let Ok(n) = value.parse::<f32>() {
            if n >= 0.0 && n.is_finite() {
                return Submission::SetTemperature(Some(n));
            }
        }
    }

    if let Some(rest) = trimmed.strip_prefix("/tools") {
        if rest.trim().is_empty() {
            return Submission::ShowTools;
        }
    }

    if let Some(rest) = trimmed.strip_prefix("/stream") {
        if rest.trim().is_empty() {
            return Submission::ShowStream;
        }
    }

    if let Some(value) = argument(trimmed, "/stream") {
        if let Ok(enabled) = parse_bool(value) {
            return Submission::SetStream(enabled);
        }
    }

    if let Some(rest) = trimmed.strip_prefix("/sandbox") {
        if rest.trim().is_empty() {
            return Submission::ShowSandbox;
        }
    }

    // Compared exactly rather than with `bare_command`, which treats any
    // trailing text as ignorable — fine for `/back`, wrong for a namespace
    // where the trailing text is the subcommand.
    if trimmed == "/clanker" {
        return Submission::ShowTitle;
    }

    if let Some(rest) = argument(trimmed, "/clanker") {
        // `/clanker title` reads the name; anything after it sets one. A
        // blank title is refused the same way the naming screen refuses it —
        // a session always has a name.
        if rest.trim() == "title" {
            return Submission::ShowTitle;
        }
        if let Some(title) = rest.strip_prefix("title") {
            if title.starts_with(char::is_whitespace) && !title.trim().is_empty() {
                return Submission::SetTitle(title.trim().to_string());
            }
        }
    }

    if let Some(rest) = trimmed.strip_prefix("/help") {
        if rest.trim().is_empty() {
            return Submission::ShowHelp;
        }
    }

    if let Some(rest) = trimmed.strip_prefix("/compact") {
        if rest.trim().is_empty() {
            return Submission::Compact;
        }
    }

    if let Some(rest) = trimmed.strip_prefix("/status") {
        if rest.trim().is_empty() {
            return Submission::ShowStatus;
        }
    }

    if let Some(value) = argument(trimmed, "/sandbox") {
        if let Ok(enabled) = parse_bool(value) {
            return Submission::SetSandbox(enabled);
        }
    }

    if let Some(rest) = argument(trimmed, "/tools") {
        let mut words = rest.split_whitespace();
        match (words.next(), words.next(), words.next()) {
            // Every tool back to its default, which is not the same as
            // setting them all to one state: the web tool's default is to
            // run without asking, and "on" has to keep meaning that.
            (Some("on"), None, None) => return Submission::ResetToolAccess,
            (Some("off"), None, None) => {
                return Submission::SetToolAccess {
                    target: "all".to_string(),
                    access: ToolAccess::Never,
                }
            }
            (Some(state), Some(target), None) => {
                if let Some(access) = ToolAccess::parse(state) {
                    return Submission::SetToolAccess {
                        target: target.to_string(),
                        access,
                    };
                }
            }
            _ => {}
        }
    }

    // A line naming a known command's word is confidently an attempted
    // command even when nothing above could parse it — unlike a `/<word>`
    // that isn't one of these at all (a path like `/etc/hosts`, say), which
    // stays an ordinary message. See `command_usage`.
    if let Some(word) = command_word(trimmed) {
        if let Some(usage) = command_usage(word) {
            return Submission::UnknownCommand(format!(
                "Unrecognized /{word} usage. Usage: {usage}"
            ));
        }
        // ...and so is one that merely comes close to naming it. A typo is
        // the case this whole check exists for: `/mode anthropic/...` used
        // to reach the model as text, which reads as the model ignoring a
        // command rather than as a mistake the user can see and fix.
        if let Some(nearest) = nearest_command(word) {
            return Submission::UnknownCommand(match command_usage(nearest) {
                Some(usage) => {
                    format!("Unrecognized command /{word}. Did you mean /{nearest}? Usage: {usage}")
                }
                None => format!("Unrecognized command /{word}. Did you mean /{nearest}?"),
            });
        }
    }

    Submission::Message(text.to_string())
}

/// The leading `/word` of `trimmed`, when it starts with a slash at all —
/// `"model"` for both `"/model"` and `"/model anthropic/opus"`. Stops only
/// at whitespace, not at a second slash, so a path like `/etc/hosts` yields
/// `"etc/hosts"` rather than `"etc"` — which is what keeps it from ever
/// matching a name in [`command_usage`].
fn command_word(trimmed: &str) -> Option<&str> {
    let rest = trimmed.strip_prefix('/')?;
    Some(rest.split_whitespace().next().unwrap_or(rest))
}

/// The stretch of `text` that names a command — the `/` and the word after
/// it, not its arguments — when that word is one this session knows.
///
/// For a front end that can style what is being typed, so a recognized
/// command looks different from a message before it is sent. It answers the
/// narrower question than [`classify`] does: whether the *name* is real, not
/// whether the whole line parses. `/effort` is a command as soon as it is
/// spelled, and stays one while its value is still being typed — a highlight
/// that came and went between `/effort` and `/effort high` would flicker at
/// exactly the moment it is being read.
///
/// Measured in `char`s from the start of `text`, since the callers that draw
/// it work in columns rather than bytes. Leading whitespace is skipped the
/// same way [`classify`] trims it, so a line that runs is a line that lights
/// up.
pub fn command_span(text: &str) -> Option<std::ops::Range<usize>> {
    let trimmed = text.trim_start();
    let word = command_word(trimmed)?;
    if !COMMANDS.iter().any(|c| c.word == word) {
        return None;
    }
    let start = text.chars().count() - trimmed.chars().count();
    // The slash, then the word.
    Some(start..start + 1 + word.chars().count())
}

/// Every in-session command: the word, how it is invoked, what it does, and
/// whether invoking it wrongly is an error.
///
/// The single place this list lives. `/help` renders it, [`command_usage`]
/// reads the syntax out of it for the "you invoked this wrongly" message,
/// and [`nearest_command`] scans the words to spot a typo. Keeping them on
/// one table is the point: a command added to `classify` and forgotten here
/// is invisible to `/help`, and one described here but never parsed is a
/// promise the session does not keep.
///
/// Every command is listed, including the ones that cannot be invoked
/// wrongly, because typo-spotting scans the same words: `/mdoel` is caught
/// as a near miss even though `/model` itself never fails to parse.
struct Command {
    word: &'static str,
    /// How `/help` lists it. Brackets mark an argument that can be left off,
    /// which for most of these means "show the current setting".
    syntax: &'static str,
    blurb: &'static str,
    /// The form quoted back when the command is named but not validly
    /// invoked. `None` for the bare commands, which are never read as
    /// malformed — a line starting `/back` with other words after it is
    /// likelier prose than a mistake, and swallowing a message costs more
    /// than missing one. Separate from `syntax` because an error should
    /// state the form that was wanted, not advertise that the argument was
    /// optional all along.
    usage: Option<&'static str>,
}

const COMMANDS: [Command; 20] = [
    Command {
        word: "help",
        syntax: "/help",
        blurb: "Show this list",
        usage: Some("/help"),
    },
    Command {
        word: "models",
        syntax: "/models",
        blurb: "Browse and pick from the models the endpoint offers",
        usage: Some("/models"),
    },
    Command {
        word: "model",
        syntax: "/model [name]",
        blurb: "Show the model, or switch it for this clanker",
        usage: None,
    },
    Command {
        word: "effort",
        syntax: "/effort <level> | clear | default",
        blurb: "Set the reasoning effort level",
        usage: Some("/effort <level> | clear | default"),
    },
    Command {
        word: "max-iterations",
        syntax: "/max-iterations <n> | clear | default",
        blurb: "Cap the tool-calling loop per turn, when it has tools",
        usage: Some("/max-iterations <n> | clear | default (n must be a positive integer)"),
    },
    Command {
        word: "temperature",
        syntax: "/temperature [<n> | clear | default]",
        blurb: "Show or set the sampling temperature",
        usage: Some("/temperature <n> | clear | default (n must be 0 or greater)"),
    },
    Command {
        word: "temp",
        syntax: "/temp [<n> | clear | default]",
        blurb: "Short for /temperature",
        usage: Some("/temp <n> | clear | default (n must be 0 or greater)"),
    },
    Command {
        word: "tools",
        syntax: "/tools [on|off | <ask|allow|never> <target>]",
        blurb: "Show what each tool may do, or change it",
        usage: Some("/tools on|off | <ask|allow|never> <tool|category|all>"),
    },
    Command {
        word: "sandbox",
        syntax: "/sandbox [on|off]",
        blurb: "Confine the agent's writes to the working directory",
        usage: Some("/sandbox <on|off>"),
    },
    Command {
        word: "verbose",
        syntax: "/verbose [on|off]",
        blurb: "Show tool arguments and results, not just the call",
        usage: Some("/verbose <on|off>"),
    },
    Command {
        word: "highlight",
        syntax: "/highlight [on|off]",
        blurb: "Band your own messages in the transcript",
        usage: Some("/highlight <on|off>"),
    },
    Command {
        word: "stream",
        syntax: "/stream [on|off]",
        blurb: "Stream replies token-by-token, or wait for the whole one",
        usage: Some("/stream <on|off>"),
    },
    Command {
        word: "compact",
        syntax: "/compact",
        blurb: "Fold the earlier conversation into a summary now",
        usage: Some("/compact"),
    },
    Command {
        word: "status",
        syntax: "/status",
        blurb: "Show every setting this clanker runs with",
        usage: Some("/status"),
    },
    Command {
        word: "clanker",
        syntax: "/clanker [title <new title>]",
        blurb: "Show or change this clanker's name",
        usage: Some("/clanker title <new title>"),
    },
    Command {
        word: "send",
        syntax: "/send",
        blurb: "Send a $ command's output to the conversation (Ctrl-S)",
        usage: None,
    },
    Command {
        word: "discard",
        syntax: "/discard",
        blurb: "Throw a $ command's output away (Ctrl-D)",
        usage: None,
    },
    Command {
        word: "allow",
        syntax: "/allow",
        blurb: "Approve the waiting tool call (Ctrl-Y)",
        usage: None,
    },
    Command {
        word: "deny",
        syntax: "/deny",
        blurb: "Refuse the waiting tool call (Ctrl-N)",
        usage: None,
    },
    Command {
        word: "back",
        syntax: "/back",
        blurb: "Return to the launch screen (Ctrl-B) — TUI only",
        usage: None,
    },
];

/// The name a line is still in the middle of typing: `"eff"` for `"/eff"`,
/// `""` for a bare `"/"`.
///
/// `None` once the line has moved past the name — a space after it settles
/// which command it is — and for anything that isn't a `/` line at all. That
/// is the whole of what completion may rewrite: a name mid-typing, never an
/// argument, and never prose.
pub fn command_prefix(text: &str) -> Option<&str> {
    let rest = text.trim_start().strip_prefix('/')?;
    (!rest.contains(char::is_whitespace)).then_some(rest)
}

/// Every command name starting with `prefix`, in the order [`help_rows`]
/// lists them, so what completion offers and what `/help` prints read the
/// same way down the screen.
pub fn command_matches(prefix: &str) -> Vec<&'static str> {
    COMMANDS
        .iter()
        .map(|c| c.word)
        .filter(|word| word.starts_with(prefix))
        .collect()
}

/// The form of whatever command `text` names, for showing above the box
/// while its arguments are typed. The same string `/help` lists, rather than
/// the terser one an error quotes: this is a reminder of the shape, not a
/// complaint that the shape was got wrong.
pub fn command_syntax(text: &str) -> Option<&'static str> {
    let word = command_word(text.trim_start())?;
    COMMANDS.iter().find(|c| c.word == word).map(|c| c.syntax)
}

/// The commands as `/help` shows them: how to type it, and what it does.
pub fn help_rows() -> Vec<(String, String)> {
    COMMANDS
        .iter()
        .map(|c| (c.syntax.to_string(), c.blurb.to_string()))
        .collect()
}

/// How far a word may stray from a command name and still be read as a
/// misspelling of it, given its length. Two edits covers the ordinary slips
/// — a dropped letter, a doubled one, a transposition (`/mdoel`,
/// `/tolos`) — but on a short word two edits is most of the word, which
/// is how `/usr` ends up two from `ask`. Short words get one edit only.
fn max_distance_for(word_length: usize) -> usize {
    if word_length >= 5 {
        2
    } else {
        1
    }
}

/// The command `word` looks like a misspelling of, if any.
///
/// Deliberately conservative, because a false positive costs more than a
/// miss — it swallows a message someone meant to send. Three things are
/// refused outright:
///
/// - anything holding a `/`: that's a path like `etc/hosts`, never a command;
/// - anything under three characters, too close to everything to tell apart;
/// - a word that *extends* the command it matched. `/verbosely` is `verbose`
///   plus a real suffix — a different word, not a misspelling — and words
///   like it have always been ordinary messages. This is checked against
///   the match itself rather than against every command, or `/temperatur`
///   would be thrown out for beginning with the unrelated `/temp`.
///
/// Within what's left it takes the single best match rather than the first
/// in range. The one knowingly-accepted overlap is a bare `/tmp`, one edit
/// from `/temp` — a path segment, but not one anybody types alone as a
/// whole message, and the error names exactly what it thought you meant.
fn nearest_command(word: &str) -> Option<&'static str> {
    let length = word.chars().count();
    if word.contains('/') || length < 3 {
        return None;
    }
    let max_distance = max_distance_for(length);
    let (_, nearest) = COMMANDS
        .iter()
        .map(|c| (edit_distance(word, c.word), c.word))
        // 0 would mean an exact match, which every caller above has already
        // handled — suggesting a word to itself would be nonsense.
        .filter(|(distance, _)| (1..=max_distance).contains(distance))
        .min_by_key(|(distance, _)| *distance)?;

    (!(word.starts_with(nearest) && length > nearest.chars().count())).then_some(nearest)
}

/// Levenshtein distance, iterating over `char`s so a multi-byte character
/// counts as one edit rather than as its bytes. Only ever run over one
/// short word against nine short names, so the straightforward two-row
/// implementation is well within its keep.
fn edit_distance(a: &str, b: &str) -> usize {
    let b_chars: Vec<char> = b.chars().collect();
    let mut previous: Vec<usize> = (0..=b_chars.len()).collect();
    let mut current = vec![0; b_chars.len() + 1];

    for (i, a_char) in a.chars().enumerate() {
        current[0] = i + 1;
        for (j, b_char) in b_chars.iter().enumerate() {
            let substitution = previous[j] + usize::from(a_char != *b_char);
            let deletion = previous[j + 1] + 1;
            let insertion = current[j] + 1;
            current[j + 1] = substitution.min(deletion).min(insertion);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[b_chars.len()]
}

/// The usage hint for a known command word, if invoking it wrong is even
/// possible. `/model` and `/verbose` accept every
/// invocation they can be given (bare, or with any trailing text at all),
/// so they have nothing invalid to report and aren't listed here — by the
/// time [`classify`] reaches this check, any of those words would already
/// have returned its own [`Submission`] above.
///
/// The hint is spelled with the word it was asked about, not a canonical
/// one, so it always names the command the reader just typed or was just
/// pointed at: `/temp` is answered about `/temp`, never about
/// `/temperature`.
fn command_usage(word: &str) -> Option<String> {
    COMMANDS
        .iter()
        .find(|c| c.word == word)
        .and_then(|c| c.usage)
        .map(str::to_string)
}

/// "clear" (case-insensitive) resets an override; anything else is the new
/// value to set.
/// Accepts the same words the CLI's own `clank stream`/`clank sandbox`
/// flags do, so `/sandbox` in a clanker reads the same way.
pub fn parse_bool(s: &str) -> Result<bool, String> {
    match s.to_lowercase().as_str() {
        "true" | "on" | "yes" | "1" => Ok(true),
        "false" | "off" | "no" | "0" => Ok(false),
        _ => Err(format!(
            "Invalid boolean value: '{}'. Use true/false, on/off, yes/no, or 1/0",
            s
        )),
    }
}

/// The trimmed text after `/name `, when `trimmed` is `/name` followed by
/// whitespace and something non-empty. `None` for a bare `/name` (nothing
/// sensible to do without a value) or for text that isn't this command at
/// all. The caller decides what `None` means from there — a bare `/model`
/// still shows the current model, while a bare `/effort`/`/max-iterations`/
/// `/temperature`/`/tools` falls through to [`command_usage`] and is
/// reported as a failed command rather than reaching the model as text.
fn argument<'a>(trimmed: &'a str, name: &str) -> Option<&'a str> {
    let rest = trimmed.strip_prefix(name)?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let value = rest.trim();
    (!value.is_empty()).then_some(value)
}

/// Whether `trimmed` is exactly `/name`, or `/name` followed by whitespace
/// (any trailing text is ignored — neither command takes an argument).
/// Requiring that separator, like `/model` does, is what keeps
/// `/agentic-issue` a message rather than a malformed command.
fn bare_command(trimmed: &str, name: &str) -> bool {
    match trimmed.strip_prefix(name) {
        Some(rest) => rest.is_empty() || rest.starts_with(char::is_whitespace),
        None => false,
    }
}

/// Something the agent loop wants to report as it runs. A front end decides
/// what (if any) of this to surface — the CLI, for instance, shows most of
/// it only in verbose mode.
///
/// Every event carries enough to render it standalone even where the CLI
/// happens not to use all of it: a front end that lists tool calls as they
/// resolve needs the `name` on a denial or a result to match it back to the
/// call it belongs to, which a purely sequential transcript doesn't.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum AgentEvent {
    /// A new pass through the tool-calling loop has begun. 1-based.
    IterationStarted { iteration: usize },
    /// A request to the model is in flight; nothing will happen until it
    /// resolves. Paired with exactly one `RequestFinished`.
    RequestStarted,
    /// The in-flight request resolved, successfully or not.
    RequestFinished,
    /// A fragment of the reply, as it streams in. Only emitted when
    /// streaming is on. The deltas of a turn concatenate to exactly the
    /// `AssistantMessage` that follows them, so a front end renders one or
    /// the other — never both.
    AssistantDelta { text: String },
    /// The model's own thinking for this turn, when it returned any.
    /// Emitted before the reply (and before any tool call) it led to, so a
    /// front end can show the reasoning in the order it happened. Front
    /// ends gate this behind `/verbose`: it's the same class of detail as a
    /// tool call's arguments.
    Thinking { text: String },
    /// The model produced visible text for the user. Always emitted at the
    /// end of a turn, streaming or not, with the complete text.
    AssistantMessage {
        model: String,
        effort_level: Option<String>,
        text: String,
    },
    /// Something went wrong that the user should see but that doesn't end
    /// the session — a failed request, a message that couldn't be saved.
    Error { message: String },
    /// The model asked to run a tool. Emitted before any approval prompt.
    ToolCallStarted { name: String, arguments: String },
    /// The user declined to let a tool run.
    ToolCallDenied { name: String },
    /// A tool ran (or failed); `result` is the JSON handed back to the model.
    ToolCallCompleted { name: String, result: String },
    /// A message typed while the turn was running has joined it, and will
    /// be part of the next request. Emitted in place of the usual
    /// `UserMessage`, which only brackets the start of a turn.
    Steered { text: String },
    /// The model answered without requesting tools, so the turn is over.
    TurnFinished,
}

/// A tool call waiting on a yes/no decision before it runs.
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    pub tool_name: String,
    /// Which [`crate::agent::ToolCategory`]-style bucket the tool falls in
    /// ("read", "write", "terminal", or "unknown"), for front ends that want
    /// to describe the action rather than just name the tool.
    pub category: &'static str,
    /// The tool's arguments, as the raw JSON string the model produced.
    pub arguments: String,
}

/// How the agent loop talks to whoever is driving it.
///
/// Both methods return futures so an implementation can await real work — a
/// GUI answering `approve` from a channel once someone clicks a button, say —
/// rather than blocking the executor.
///
/// They're written as explicit `-> impl Future + Send` rather than `async fn`
/// because an `async fn` in a trait gives its future no `Send` bound, which
/// makes the whole agent loop un-spawnable from any generic context. The TUI
/// runs the loop on a background task, so `Send` is required.
pub trait AgentUi {
    /// Report progress. Implementations should not block for long here.
    fn event(&mut self, event: AgentEvent) -> impl Future<Output = ()> + Send;

    /// Ask whether a tool may run. Returning `Ok(false)` denies it and lets
    /// the loop continue; returning `Err` aborts the turn.
    fn approve(&mut self, request: ApprovalRequest) -> impl Future<Output = Result<bool>> + Send;
}

#[cfg(test)]
mod tests {

    #[test]
    fn classify_recognizes_highlight() {
        assert_eq!(classify("/highlight on"), Submission::SetHighlight(true));
        assert_eq!(classify("/highlight off"), Submission::SetHighlight(false));
        assert_eq!(classify("/highlight"), Submission::ShowHighlight);
        // A bad value is a usage error, not a message reaching the model.
        assert!(matches!(
            classify("/highlight maybe"),
            Submission::UnknownCommand(_)
        ));
    }

    #[test]
    fn a_leading_dollar_runs_a_command() {
        assert_eq!(
            classify("$ cargo test"),
            Submission::Shell("cargo test".to_string())
        );
        assert_eq!(
            classify("$git status"),
            Submission::Shell("git status".to_string())
        );
    }

    #[test]
    fn a_dollar_that_is_not_a_command_stays_a_message() {
        // Only a leading `$` runs anything, and a bare one names nothing to
        // run.
        assert_eq!(
            classify("it cost me $5"),
            Submission::Message("it cost me $5".to_string())
        );
        assert_eq!(classify("$"), Submission::Message("$".to_string()));
        // `Message` carries what was typed, trailing space and all.
        assert!(matches!(classify("$   "), Submission::Message(_)));
    }

    #[test]
    fn every_chord_has_a_typed_equivalent() {
        // Not a convenience. Zed's terminal claims Ctrl-S and tmux claims
        // Ctrl-B, and an approval whose keys are intercepted would leave a
        // turn waiting on a decision with no way to give it.
        assert_eq!(classify("/allow"), Submission::AllowTool);
        assert_eq!(classify("/deny"), Submission::DenyTool);
        assert_eq!(classify("/back"), Submission::Back);
    }

    #[test]
    fn send_and_discard_are_typeable() {
        // The path that keeps the feature working when a multiplexer has
        // claimed Ctrl-S — and the one nobody exercises by habit.
        assert_eq!(classify("/send"), Submission::SendShell);
        assert_eq!(classify("/discard"), Submission::DiscardShell);
    }
    use super::*;

    #[test]
    fn parse_yes_no_accepts_only_explicit_yes() {
        assert!(parse_yes_no("y"));
        assert!(parse_yes_no("yes"));
        assert!(parse_yes_no("  YES  \n"));
        assert!(parse_yes_no("Y\n"));
    }

    #[test]
    fn parse_yes_no_denies_everything_else() {
        assert!(!parse_yes_no("n"));
        assert!(!parse_yes_no("no"));
        assert!(!parse_yes_no(""));
        assert!(!parse_yes_no("\n"));
        assert!(!parse_yes_no("maybe"));
        // Fails closed: a stray answer is a denial, never an approval.
        assert!(!parse_yes_no("yep"));
    }

    #[test]
    fn classify_recognizes_the_model_command() {
        assert_eq!(
            classify("/model anthropic/claude-opus-4.5"),
            Submission::SetModel("anthropic/claude-opus-4.5".to_string())
        );
        assert_eq!(classify("/model"), Submission::ShowModel);
        assert_eq!(classify("  /model   "), Submission::ShowModel);
    }

    #[test]
    fn classify_leaves_ordinary_text_and_paths_alone() {
        assert_eq!(
            classify("what does /etc/hosts do?"),
            Submission::Message("what does /etc/hosts do?".to_string())
        );
        // A leading slash that isn't a command must still reach the model,
        // since paths are common input in a coding tool.
        assert_eq!(
            classify("/usr/bin/env"),
            Submission::Message("/usr/bin/env".to_string())
        );
        // Not the command, just a word starting with it.
        assert_eq!(
            classify("/modelling is fun"),
            Submission::Message("/modelling is fun".to_string())
        );
    }

    #[test]
    fn a_command_span_covers_the_name_and_nothing_else() {
        // The name only: the value is still being typed, and colouring it
        // would say the whole line had been understood.
        assert_eq!(command_span("/effort high"), Some(0..7));
        assert_eq!(command_span("/help"), Some(0..5));
        // Trimmed the same way `classify` trims, so what lights up and what
        // runs agree.
        assert_eq!(command_span("  /help"), Some(2..7));
    }

    #[test]
    fn a_command_span_holds_off_until_the_name_is_real() {
        // Mid-word, before it names anything.
        assert_eq!(command_span("/hel"), None);
        // Past it: a longer word is a different word.
        assert_eq!(command_span("/helpful"), None);
        // Paths are the case this must never claim.
        assert_eq!(command_span("/etc/hosts"), None);
        assert_eq!(command_span("/usr/bin/env"), None);
        // A slash anywhere but the front is just prose.
        assert_eq!(command_span("what does /help do"), None);
        assert_eq!(command_span("hello"), None);
        assert_eq!(command_span("/"), None);
    }

    #[test]
    fn a_command_prefix_is_only_ever_the_name_being_typed() {
        assert_eq!(command_prefix("/eff"), Some("eff"));
        assert_eq!(command_prefix("  /eff"), Some("eff"));
        // A bare slash is a name nothing has been typed into yet.
        assert_eq!(command_prefix("/"), Some(""));
        // Past the name: the space settled which command it is, and there
        // is nothing left here to complete.
        assert_eq!(command_prefix("/effort hi"), None);
        assert_eq!(command_prefix("/effort "), None);
        assert_eq!(command_prefix("hello"), None);
        assert_eq!(command_prefix("$ ls"), None);
    }

    #[test]
    fn matches_are_offered_in_the_order_help_prints_them() {
        // Two lists in different orders would read as two different sets.
        assert_eq!(command_matches("mo"), ["models", "model"]);
        assert_eq!(command_matches("temp"), ["temperature", "temp"]);
        assert_eq!(command_matches("help"), ["help"]);
        assert!(command_matches("nonesuch").is_empty());
        assert_eq!(command_matches("").len(), COMMANDS.len());
    }

    #[test]
    fn the_syntax_shown_is_the_one_help_lists() {
        assert_eq!(command_syntax("/model"), Some("/model [name]"));
        // Read off the name, whatever is being typed after it.
        assert_eq!(
            command_syntax("/model anthropic/opus"),
            Some("/model [name]")
        );
        assert_eq!(command_syntax("/etc/hosts"), None);
        assert_eq!(command_syntax("hello"), None);
    }

    #[test]
    fn every_command_help_lists_can_be_seen_as_you_type_it() {
        // The two would drift apart silently: a command added to `COMMANDS`
        // but matched by a hand-written list here would work, list itself in
        // `/help`, and stay stubbornly uncoloured in the box.
        for (syntax, _) in help_rows() {
            let word = syntax.split_whitespace().next().unwrap().to_string();
            assert_eq!(
                command_span(&word),
                Some(0..word.chars().count()),
                "{syntax} does not light up"
            );
        }
    }

    #[test]
    fn a_command_span_counts_characters_not_bytes() {
        // The callers draw in columns. A non-breaking space is one column
        // and two bytes, and `trim_start` counts it as whitespace — so this
        // does name a command, and measured in bytes the span would land a
        // column to the right of the word it belongs to.
        assert_eq!(classify("\u{a0}/help"), Submission::ShowHelp);
        assert_eq!(command_span("\u{a0}/help"), Some(1..6));
    }

    #[test]
    fn agent_and_ask_are_no_longer_commands() {
        // There is no mode to switch any more — `/tools on|off` is the
        // whole of it — so these are ordinary messages again. `/ask` is
        // short enough that the typo-catcher leaves it alone too.
        assert_eq!(
            classify("/agent"),
            Submission::Message("/agent".to_string())
        );
        assert_eq!(classify("/ask"), Submission::Message("/ask".to_string()));
        assert_eq!(
            classify("/agentic-issue"),
            Submission::Message("/agentic-issue".to_string())
        );
    }

    #[test]
    fn classify_recognizes_verbose() {
        // Bare reads rather than flips, matching `/sandbox` and the global
        // `clank verbose` — a toggle can't be written down as an
        // instruction without knowing where it started.
        assert_eq!(classify("/verbose"), Submission::ShowVerbose);
        assert_eq!(classify("  /verbose  "), Submission::ShowVerbose);
        assert_eq!(classify("/verbose on"), Submission::SetVerbose(true));
        assert_eq!(classify("/verbose off"), Submission::SetVerbose(false));
        assert_eq!(
            classify("/verbose maybe"),
            Submission::UnknownCommand(
                "Unrecognized /verbose usage. Usage: /verbose <on|off>".to_string()
            )
        );
        assert_eq!(
            classify("/verbosely"),
            Submission::Message("/verbosely".to_string())
        );
    }

    #[test]
    fn every_setting_command_answers_when_asked_bare() {
        // The parity that `/effort` was missing: if a command sets a value,
        // typing it alone should report that value rather than scolding you
        // for not passing one.
        for command in [
            "/effort",
            "/temperature",
            "/temp",
            "/model",
            "/verbose",
            "/highlight",
            "/sandbox",
            "/stream",
            "/tools",
            "/clanker",
        ] {
            let shown = classify(command);
            assert!(
                !matches!(
                    shown,
                    Submission::UnknownCommand(_) | Submission::Message(_)
                ),
                "bare {command} should report its value, got {shown:?}"
            );
        }
    }

    #[test]
    fn classify_recognizes_effort() {
        assert_eq!(
            classify("/effort high"),
            Submission::SetEffort(Some("high".to_string()))
        );
        // "clear" nullifies — no effort field is sent until set again.
        assert_eq!(classify("/effort clear"), Submission::SetEffort(None));
        // "default" is a distinct action: it reads whatever the configured
        // default currently is and saves that concrete value now.
        assert_eq!(classify("/effort default"), Submission::ResetEffort);
        // Case-insensitive, like a keyword rather than a literal value.
        assert_eq!(classify("/effort CLEAR"), Submission::SetEffort(None));
        assert_eq!(classify("/effort DEFAULT"), Submission::ResetEffort);

        // Anything else passes through as typed, case included — not
        // checked against a fixed low/medium/high list, since models vary
        // in what reasoning-effort values they actually accept.
        assert_eq!(
            classify("/effort HIGH"),
            Submission::SetEffort(Some("HIGH".to_string()))
        );
        assert_eq!(
            classify("/effort minimal"),
            Submission::SetEffort(Some("minimal".to_string()))
        );

        // Bare shows the current value, the same as every other setting
        // command. It used to be a usage error, which left `/effort` the
        // only one of them you could not simply ask.
        assert_eq!(classify("/effort"), Submission::ShowEffort);
        assert_eq!(classify("  /effort  "), Submission::ShowEffort);
        // Not the command, just a word starting with it.
        assert_eq!(
            classify("/effortless"),
            Submission::Message("/effortless".to_string())
        );
    }

    #[test]
    fn classify_recognizes_max_iterations() {
        assert_eq!(
            classify("/max-iterations 30"),
            Submission::SetMaxIterations(Some(30))
        );
        // "clear" nullifies — turns fall back to the configured default.
        assert_eq!(
            classify("/max-iterations clear"),
            Submission::SetMaxIterations(None)
        );
        // "default" is a distinct action: it reads whatever the configured
        // default currently is and saves that concrete value now.
        assert_eq!(
            classify("/max-iterations default"),
            Submission::ResetMaxIterations
        );
        // Case-insensitive, like a keyword rather than a literal value.
        assert_eq!(
            classify("/max-iterations CLEAR"),
            Submission::SetMaxIterations(None)
        );
        assert_eq!(
            classify("/max-iterations DEFAULT"),
            Submission::ResetMaxIterations
        );

        // Zero and non-numeric values aren't valid iteration counts, so —
        // like a bare invocation — they're reported as a failed command
        // rather than silently reaching the model as text.
        let max_iterations_usage = "Unrecognized /max-iterations usage. Usage: \
            /max-iterations <n> | clear | default (n must be a positive integer)";
        assert_eq!(
            classify("/max-iterations 0"),
            Submission::UnknownCommand(max_iterations_usage.to_string())
        );
        assert_eq!(
            classify("/max-iterations banana"),
            Submission::UnknownCommand(max_iterations_usage.to_string())
        );
        // Bare, with nothing to act on, is reported the same way.
        assert_eq!(
            classify("/max-iterations"),
            Submission::UnknownCommand(max_iterations_usage.to_string())
        );
    }

    #[test]
    fn classify_recognizes_temperature() {
        assert_eq!(
            classify("/temperature 1.5"),
            Submission::SetTemperature(Some(1.5))
        );
        // Zero is a valid (deterministic) temperature, unlike max-iterations.
        assert_eq!(
            classify("/temperature 0"),
            Submission::SetTemperature(Some(0.0))
        );
        // "clear" nullifies — turns fall back to the configured default.
        assert_eq!(
            classify("/temperature clear"),
            Submission::SetTemperature(None)
        );
        // "default" is a distinct action: it reads whatever the configured
        // default currently is and saves that concrete value now.
        assert_eq!(
            classify("/temperature default"),
            Submission::ResetTemperature
        );
        // Case-insensitive, like a keyword rather than a literal value.
        assert_eq!(
            classify("/temperature CLEAR"),
            Submission::SetTemperature(None)
        );
        assert_eq!(
            classify("/temperature DEFAULT"),
            Submission::ResetTemperature
        );

        // Negative and non-numeric values aren't valid temperatures, so —
        // like a bare invocation — they're reported as a failed command
        // rather than silently reaching the model as text.
        let temperature_usage = "Unrecognized /temperature usage. Usage: \
            /temperature <n> | clear | default (n must be 0 or greater)";
        assert_eq!(
            classify("/temperature -1"),
            Submission::UnknownCommand(temperature_usage.to_string())
        );
        assert_eq!(
            classify("/temperature banana"),
            Submission::UnknownCommand(temperature_usage.to_string())
        );
        // Bare reads the current value instead — see
        // `bare_temperature_reads_it_rather_than_failing`.
    }

    #[test]
    fn classify_recognizes_the_temp_shorthand() {
        assert_eq!(classify("/temp 1.5"), Submission::SetTemperature(Some(1.5)));
        assert_eq!(classify("/temp clear"), Submission::SetTemperature(None));
        // Same rules as the full name: a malformed or bare invocation is a
        // failed command, not text sent to the model.
        // Answered about `/temp`, the word actually typed — not about
        // `/temperature`, which the reader didn't write.
        let temp_usage = "Unrecognized /temp usage. Usage: \
            /temp <n> | clear | default (n must be 0 or greater)";
        assert_eq!(
            classify("/temp banana"),
            Submission::UnknownCommand(temp_usage.to_string())
        );
        // Bare reads the current value; only a malformed one is reported.
    }

    #[test]
    fn classify_recognizes_tools() {
        let set = |target: &str, access| Submission::SetToolAccess {
            target: target.to_string(),
            access,
        };
        assert_eq!(
            classify("/tools allow read"),
            set("read", ToolAccess::Allow)
        );
        assert_eq!(
            classify("/tools never run_terminal_command"),
            set("run_terminal_command", ToolAccess::Never)
        );
        assert_eq!(classify("/tools ask all"), set("all", ToolAccess::Ask));

        // `off` is every tool at once; `on` is every tool back to its own
        // default, which is a different thing — the web tool's default is
        // to get on with it, and turning tools back on must not start
        // prompting for that.
        assert_eq!(classify("/tools off"), set("all", ToolAccess::Never));
        assert_eq!(classify("/tools on"), Submission::ResetToolAccess);

        // A target that names nothing is still a command — it is reported
        // by whoever applies it, which is where the tool table lives.
        assert_eq!(classify("/tools ask bogus"), set("bogus", ToolAccess::Ask));

        // An unknown state, or the wrong number of words, is a failed
        // command rather than text sent to the model.
        let usage = "Unrecognized /tools usage. Usage: \
            /tools on|off | <ask|allow|never> <tool|category|all>";
        assert_eq!(
            classify("/tools maybe read"),
            Submission::UnknownCommand(usage.to_string())
        );
        assert_eq!(
            classify("/tools ask"),
            Submission::UnknownCommand(usage.to_string())
        );
        assert_eq!(
            classify("/tools ask read too"),
            Submission::UnknownCommand(usage.to_string())
        );

        // Bare — with or without trailing whitespace — lists them.
        assert_eq!(classify("/tools"), Submission::ShowTools);
        assert_eq!(classify("  /tools   "), Submission::ShowTools);
    }

    #[test]
    fn bare_temperature_reads_it_rather_than_failing() {
        assert_eq!(classify("/temperature"), Submission::ShowTemperature);
        assert_eq!(classify("/temp"), Submission::ShowTemperature);
        assert_eq!(classify("  /temp   "), Submission::ShowTemperature);
        // Setting still works either way round.
        assert_eq!(classify("/temp 1.5"), Submission::SetTemperature(Some(1.5)));
        assert_eq!(
            classify("/temperature clear"),
            Submission::SetTemperature(None)
        );
    }

    #[test]
    fn temperature_notice_says_what_none_means() {
        assert_eq!(temperature_notice(Some(0.7), false), "Temperature is 0.7");
        assert_eq!(
            temperature_notice(None, true),
            "Temperature set to none sent — the provider uses its own default"
        );
    }

    #[test]
    fn classify_recognizes_the_stream_command() {
        assert_eq!(classify("/stream off"), Submission::SetStream(false));
        assert_eq!(classify("/stream on"), Submission::SetStream(true));
        assert_eq!(classify("/stream yes"), Submission::SetStream(true));
        // Bare reads it, like every other setting.
        assert_eq!(classify("/stream"), Submission::ShowStream);
        assert_eq!(classify("  /stream   "), Submission::ShowStream);
        // A value that isn't a boolean is a failed command, not a message.
        assert_eq!(
            classify("/stream sometimes"),
            Submission::UnknownCommand(
                "Unrecognized /stream usage. Usage: /stream <on|off>".to_string()
            )
        );
        assert_eq!(nearest_command("strem"), Some("stream"));
    }

    #[test]
    fn the_on_off_notices_say_what_the_setting_does() {
        // Not just "on"/"off": the point of a bare read is to tell someone
        // what the setting will actually do.
        assert_eq!(
            verbose_notice(true, false),
            "Verbose is on — showing tool call detail and the model's thinking"
        );
        assert_eq!(
            verbose_notice(false, true),
            "Verbose set to off — showing a one-line notice per tool call"
        );
        assert_eq!(
            stream_notice(true, false),
            "Streaming is on — replies arrive token by token"
        );
        assert_eq!(
            stream_notice(false, true),
            "Streaming set to off — replies arrive whole"
        );
    }

    #[test]
    fn classify_recognizes_the_clanker_command() {
        assert_eq!(
            classify("/clanker title Fix the parser"),
            Submission::SetTitle("Fix the parser".to_string())
        );
        // Bare, either way round, reads the name.
        assert_eq!(classify("/clanker"), Submission::ShowTitle);
        assert_eq!(classify("  /clanker   "), Submission::ShowTitle);
        assert_eq!(classify("/clanker title"), Submission::ShowTitle);
        assert_eq!(classify("/clanker title   "), Submission::ShowTitle);

        // An unknown subcommand is a failed command, not a message.
        assert_eq!(
            classify("/clanker rename x"),
            Submission::UnknownCommand(
                "Unrecognized /clanker usage. Usage: /clanker title <new title>".to_string()
            )
        );
        // "titles" is not "title".
        assert_eq!(
            classify("/clanker titles"),
            Submission::UnknownCommand(
                "Unrecognized /clanker usage. Usage: /clanker title <new title>".to_string()
            )
        );
        assert_eq!(nearest_command("clankr"), Some("clanker"));
    }

    #[test]
    fn a_title_keeps_the_spacing_inside_it() {
        // Only the ends are trimmed: the name is whatever was typed.
        assert_eq!(
            classify("/clanker title  Fix  the   parser  "),
            Submission::SetTitle("Fix  the   parser".to_string())
        );
    }

    #[test]
    fn an_approval_summary_names_what_is_being_asked() {
        // A row that says only "needs approval" tells you nothing you can
        // act on; the file or command is the decision.
        let request = |tool: &str, arguments: &str| ApprovalRequest {
            tool_name: tool.to_string(),
            category: "write",
            arguments: arguments.to_string(),
        };

        assert_eq!(
            approval_summary(&request("write_file", r#"{"filepath":"src/main.rs"}"#)),
            "write_file: src/main.rs"
        );
        assert_eq!(
            approval_summary(&request(
                "run_terminal_command",
                r#"{"command":"rm -rf build"}"#
            )),
            "run_terminal_command: rm -rf build"
        );
        // Arguments with nothing worth naming fall back to the tool alone
        // rather than to something misleading.
        assert_eq!(approval_summary(&request("list_files", "{}")), "list_files");
        assert_eq!(
            approval_summary(&request("read_file", "not json")),
            "read_file"
        );
    }

    #[test]
    fn title_notice_distinguishes_a_rename_from_a_read() {
        assert_eq!(
            title_notice("Fix the parser", false),
            "Clanker is Fix the parser"
        );
        assert_eq!(
            title_notice("Fix the parser", true),
            "Clanker renamed to Fix the parser"
        );
    }

    #[test]
    fn classify_recognizes_the_status_command() {
        assert_eq!(classify("/status"), Submission::ShowStatus);
        assert_eq!(classify("  /status   "), Submission::ShowStatus);
        // It takes no argument, so anything after it is a mistake rather
        // than a message.
        assert_eq!(
            classify("/status verbose"),
            Submission::UnknownCommand("Unrecognized /status usage. Usage: /status".to_string())
        );
        assert_eq!(nearest_command("statu"), Some("status"));
    }

    #[test]
    fn session_settings_rows_say_what_unset_means() {
        // A nullified setting isn't blank — it does something specific, and
        // the readout has to distinguish "sends nothing" from "happens to
        // match the default".
        let tool_access = ToolAccessSettings::default();
        let rows = session_settings_rows(&SessionSettings {
            id: "abc123",
            title: "Untitled",
            model: "openrouter/auto",
            effort_level: None,
            temperature: None,
            max_iterations: None,
            verbose: false,
            highlight: true,
            sandbox: true,
            stream: true,
            working_dir: None,
            tool_access: &tool_access,
            total_tokens: 0,
            compactor: "openrouter/auto",
            compact_at: None,
        });
        let value = |label: &str| {
            rows.iter()
                .find(|(l, _)| l == label)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| panic!("no {label} row"))
        };

        assert_eq!(value("ID"), "abc123");
        assert_eq!(value("Tools"), "on");
        assert_eq!(value("Effort"), "none sent");
        assert_eq!(value("Temperature"), "none sent");
        assert_eq!(value("Max iterations"), "not set");
        assert_eq!(value("Sandbox"), "on");
        assert_eq!(
            value("Each tool"),
            "read_file ask · list_files ask · web_fetch allow · write_file ask \
             · replace_in_file ask · run_terminal_command never"
        );
        assert_eq!(value("Directory"), "not recorded");
        assert_eq!(value("Tokens"), "🪙 0");
    }

    #[test]
    fn session_settings_rows_report_a_configured_session() {
        let tool_access = ToolAccessSettings::default()
            .with("read", ToolAccess::Allow)
            .unwrap();
        let rows = session_settings_rows(&SessionSettings {
            id: "abc123",
            title: "Fix the parser",
            model: "anthropic/claude-sonnet-5",
            effort_level: Some("high"),
            temperature: Some(0.7),
            max_iterations: Some(20),
            verbose: true,
            highlight: true,
            sandbox: false,
            stream: false,
            working_dir: Some("/home/dev/project"),
            tool_access: &tool_access,
            total_tokens: 12345,
            compactor: "openrouter/auto",
            compact_at: Some(60_000),
        });
        let value = |label: &str| {
            rows.iter()
                .find(|(l, _)| l == label)
                .map(|(_, v)| v.clone())
                .unwrap()
        };

        assert_eq!(value("Tools"), "on");
        assert_eq!(value("Effort"), "high");
        assert_eq!(value("Sandbox"), "off");
        assert_eq!(value("Verbose"), "on");
        // A tool waved through reads differently from one that still asks,
        // and each is named — which category it fell in is not the question
        // anyone is asking here.
        assert_eq!(
            value("Each tool"),
            "read_file allow · list_files allow · web_fetch allow · write_file ask \
             · replace_in_file ask · run_terminal_command never"
        );
        assert_eq!(value("Directory"), "/home/dev/project");
        assert_eq!(value("Tokens"), "🪙 12,345");
    }

    #[test]
    fn format_tokens_groups_thousands() {
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(999), "999");
        assert_eq!(format_tokens(1000), "1,000");
        assert_eq!(format_tokens(1_234_567), "1,234,567");
        assert_eq!(format_tokens(-42), "-42");
    }

    #[test]
    fn classify_recognizes_the_sandbox_command() {
        assert_eq!(classify("/sandbox off"), Submission::SetSandbox(false));
        assert_eq!(classify("/sandbox on"), Submission::SetSandbox(true));
        // Same boolean words every other on/off setting takes.
        assert_eq!(classify("/sandbox false"), Submission::SetSandbox(false));
        assert_eq!(classify("/sandbox 1"), Submission::SetSandbox(true));
        // Bare shows the current setting, matching `/tools`.
        assert_eq!(classify("/sandbox"), Submission::ShowSandbox);
        assert_eq!(classify("  /sandbox   "), Submission::ShowSandbox);
        // A value that isn't a boolean is a failed command, not a message.
        assert_eq!(
            classify("/sandbox maybe"),
            Submission::UnknownCommand(
                "Unrecognized /sandbox usage. Usage: /sandbox <on|off>".to_string()
            )
        );
        // And it joins the near-miss set like every other command word.
        assert_eq!(nearest_command("sandbix"), Some("sandbox"));
    }

    #[test]
    fn sandbox_notice_says_what_changed_and_what_it_means() {
        assert_eq!(
            sandbox_notice(true, true),
            "Sandbox set to on — writes confined to the working directory"
        );
        assert_eq!(
            sandbox_notice(false, false),
            "Sandbox is off — writes allowed anywhere"
        );
    }

    #[test]
    fn classify_recognizes_help() {
        assert_eq!(classify("/help"), Submission::ShowHelp);
        assert_eq!(classify("  /help  "), Submission::ShowHelp);
        // Takes no argument, so anything after it is a mistake rather than
        // a message — the same rule `/status` follows.
        assert!(matches!(
            classify("/help model"),
            Submission::UnknownCommand(_)
        ));
        // And a word that merely starts with it is prose.
        assert_eq!(
            classify("/helpful hints please"),
            Submission::Message("/helpful hints please".to_string())
        );
    }

    #[test]
    fn help_lists_every_command_the_classifier_knows() {
        // The table is the single source for `/help`, the usage quoted in an
        // error, and typo-spotting. A command parsed by `classify` but left
        // out of it is invisible to all three, so this walks the table and
        // checks each entry actually is a command.
        for (syntax, blurb) in help_rows() {
            assert!(syntax.starts_with('/'), "{syntax} is not a command");
            assert!(!blurb.is_empty(), "{syntax} has no description");

            let word = syntax
                .trim_start_matches('/')
                .split([' ', '<', '['])
                .next()
                .unwrap()
                .to_string();
            assert!(
                !matches!(classify(&format!("/{word}")), Submission::Message(_)),
                "/{word} is listed by /help but reaches the model as a message"
            );
        }
    }

    #[test]
    fn a_misspelling_of_help_is_caught_like_any_other() {
        // Only worth asserting because `/help` was added to the table after
        // the near-miss machinery already existed; being in the table is
        // what puts it in scope.
        assert!(matches!(
            classify("/halp"),
            Submission::UnknownCommand(message) if message.contains("/help")
        ));
        // Not `/hlep`, though: a transposition is two edits, and a
        // four-letter word is allowed only one before the guess is refused
        // as too likely to swallow a real message.
        assert_eq!(classify("/hlep"), Submission::Message("/hlep".to_string()));
    }

    #[test]
    fn classify_catches_a_misspelled_command() {
        // The case this exists for, taken from a real session: `/mode ...`
        // reached the model as text, which reads as the model ignoring a
        // command rather than as a typo the user can see and fix.
        assert_eq!(
            classify("/mode anthropic/claude-sonnet-5"),
            Submission::UnknownCommand(
                "Unrecognized command /mode. Did you mean /model?".to_string()
            )
        );
        // A transposition is two edits, still within reach, and a command
        // that *can* be misused carries its usage along with the guess.
        assert_eq!(
            classify("/tolos allow read"),
            Submission::UnknownCommand(
                "Unrecognized command /tolos. Did you mean /tools? \
                 Usage: /tools on|off | <ask|allow|never> <tool|category|all>"
                    .to_string()
            )
        );
    }

    #[test]
    fn a_suggestion_spells_its_usage_the_way_it_was_suggested() {
        // `/tmp` is nearest to the `/temp` shorthand, so the hint has to
        // talk about `/temp` — naming `/temperature` here would answer
        // about a word the reader was never pointed at.
        assert_eq!(
            classify("/tmp"),
            Submission::UnknownCommand(
                "Unrecognized command /tmp. Did you mean /temp? \
                 Usage: /temp <n> | clear | default (n must be 0 or greater)"
                    .to_string()
            )
        );
    }

    #[test]
    fn classify_leaves_paths_and_unrelated_slash_words_alone() {
        // The regression the whole hybrid guards against: a leading slash
        // is not on its own enough to swallow a message.
        for text in [
            "/etc/hosts",
            "/etc/hosts, what's in it?",
            "/usr/bin/env python",
            "/home/dylan/code",
            "/x",
        ] {
            assert_eq!(
                classify(text),
                Submission::Message(text.to_string()),
                "{text} should stay a message"
            );
        }
    }

    #[test]
    fn nearest_command_refuses_to_guess_at_a_path() {
        // A `/` anywhere in the word means a path, however close its first
        // segment lands to a command name.
        assert_eq!(nearest_command("mode/dark"), None);
        // Too short to tell apart from anything.
        assert_eq!(nearest_command("md"), None);
        // Unrelated words stay unrelated. `usr` is two edits from `ask`,
        // which on a three-letter word is most of the word — hence the
        // length-relative bound.
        assert_eq!(nearest_command("etc"), None);
        assert_eq!(nearest_command("usr"), None);
        // A word that extends its own match is a different word, not a
        // misspelling — these have always been ordinary messages.
        assert_eq!(nearest_command("verbosely"), None);
        assert_eq!(nearest_command("models-are-great"), None);
        assert_eq!(nearest_command("tempo"), None);
        // But extending some *other*, unrelated command is no reason to
        // throw a real typo away: `temperatur` begins with `temp`.
        assert_eq!(nearest_command("temperatur"), Some("temperature"));
        // The knowingly-accepted overlap, called out so a change is loud.
        assert_eq!(nearest_command("tmp"), Some("temp"));
        // And the best match wins, not merely the first in range.
        assert_eq!(nearest_command("temperatur"), Some("temperature"));
    }

    #[test]
    fn edit_distance_counts_characters_not_bytes() {
        assert_eq!(edit_distance("model", "model"), 0);
        assert_eq!(edit_distance("mode", "model"), 1);
        assert_eq!(edit_distance("mdoel", "model"), 2);
        // A multi-byte character is one edit, not one per byte.
        assert_eq!(edit_distance("café", "cafe"), 1);
    }

    #[test]
    fn primary_argument_finds_a_file_path() {
        assert_eq!(
            primary_argument(r#"{"filepath":"src/main.rs","content":"x"}"#),
            Some("src/main.rs".to_string())
        );
    }

    #[test]
    fn primary_argument_finds_a_terminal_command() {
        assert_eq!(
            primary_argument(r#"{"command":"cargo test","timeout_secs":30}"#),
            Some("cargo test".to_string())
        );
    }

    #[test]
    fn primary_argument_finds_a_directory() {
        assert_eq!(
            primary_argument(r#"{"dirpath":"src"}"#),
            Some("src".to_string())
        );
    }

    #[test]
    fn primary_argument_is_none_without_a_recognized_key() {
        assert_eq!(
            primary_argument(r#"{"search":"foo","replace":"bar"}"#),
            None
        );
        assert_eq!(primary_argument("not json"), None);
        assert_eq!(primary_argument("{}"), None);
    }

    #[test]
    fn tool_call_fields_adds_the_default_working_dir_for_a_terminal_command() {
        let fields = tool_call_fields("run_terminal_command", r#"{"command":"cargo test"}"#);
        let working_dir = fields.iter().find(|(key, _)| key == "working_dir");
        assert!(working_dir.is_some(), "{fields:?}");
        assert_eq!(
            working_dir.unwrap().1,
            std::env::current_dir().unwrap().display().to_string()
        );
    }

    #[test]
    fn tool_call_fields_keeps_an_explicit_working_dir_as_is() {
        let fields = tool_call_fields(
            "run_terminal_command",
            r#"{"command":"ls","working_dir":"/tmp"}"#,
        );
        assert_eq!(
            fields,
            vec![
                ("command".to_string(), "ls".to_string()),
                ("working_dir".to_string(), "/tmp".to_string()),
            ]
        );
    }

    #[test]
    fn tool_call_fields_leaves_other_tools_unchanged() {
        let fields = tool_call_fields("write_file", r#"{"filepath":"a.rs","content":"x"}"#);
        assert!(
            !fields.iter().any(|(key, _)| key == "working_dir"),
            "{fields:?}"
        );
    }
}
