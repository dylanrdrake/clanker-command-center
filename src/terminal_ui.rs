//! The CLI's [`AgentUi`]: renders agent progress to stdout and asks for tool
//! approval on stdin. This is the only place the agent loop's output is
//! formatted, so a different front end can present the same events however
//! it likes without touching the loop itself.

use crate::spinner::Spinner;
use crate::ui::{
    json_fields, parse_yes_no, primary_argument, response_label, tool_call_fields, AgentEvent,
    AgentUi, ApprovalRequest,
};
use crate::wrap;
use anyhow::Result;
use colored::*;
use std::future::Future;
use std::io::{self, Write};
use unicode_width::UnicodeWidthStr;

/// Holds the session for this process, and writes what it is doing while a
/// turn runs, for anything watching the list of sessions.
///
/// The two jobs travel together because they have the same lifetime, but
/// they are not equally optional: reporting is a courtesy, while the claim is
/// what stops a second process appending turns to the same history. See
/// [`Self::claim`].
///
/// Holds its own database handle rather than borrowing the session: the turn
/// already has that borrowed mutably for the whole call, and an approval
/// prompt happens in the middle of it.
pub struct ActivityWriter {
    conn: rusqlite::Connection,
    session_id: String,
    /// Held, not used: it renews while this writer is alive and gives the
    /// claim up when it drops.
    _claim: crate::session::Heartbeat,
}

impl ActivityWriter {
    /// Claims the session, returning `Ok(None)` if another live process
    /// already holds it.
    ///
    /// A caller must not run the session without one of these. Two processes
    /// appending to one history write colliding `seq` values, and the result
    /// reloads as a conversation whose turns are shuffled and whose tool
    /// results no longer follow the calls they answer — which most providers
    /// reject outright, so the session stops being resumable at all. Nothing
    /// detects it and nothing repairs it.
    pub fn claim(session_id: String) -> Result<Option<Self>> {
        let Some(claim) = crate::session::Heartbeat::claim(session_id.clone())? else {
            return Ok(None);
        };
        Ok(Some(ActivityWriter {
            conn: crate::store::open_db()?,
            session_id,
            _claim: claim,
        }))
    }

    /// The claim this writer holds, for binding a session's writes to it.
    pub fn claim_owner(&self) -> &str {
        self._claim.owner()
    }

    /// The session being watched, so a front end can derive its mark.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    fn set(&self, activity: Option<crate::store::Activity>, detail: Option<&str>) {
        let _ = crate::store::set_session_activity(&self.conn, &self.session_id, activity, detail);
    }
}

pub struct TerminalAgentUi {
    /// Mirrors the `-v` flag: gates the full argument/result dump. The
    /// marker-and-name notice below it — matching what the TUI always shows
    /// — is not gated, so a run with tools isn't silent about
    /// tool calls the way it used to be.
    verbose: bool,
    /// Whether a reply is prefixed with `model (effort):`. On for one-shot
    /// one-off runs, where there's no other way to see what answered; off
    /// for `clanker`, matching the TUI transcript, which dropped the same
    /// label — current model there is `/model`'s job, not every reply's.
    show_model_label: bool,
    /// Live only between `RequestStarted` and `RequestFinished`.
    spinner: Option<Spinner>,
    /// Set for a clanker, which is watchable from the picker; `None` for a
    /// one-off run, which nobody is monitoring.
    activity: Option<ActivityWriter>,
    /// The call's arguments, held from `ToolCallStarted` to whichever event
    /// settles it, so the notice that settles it can still name the file
    /// or command being acted on, and so `ToolCallCompleted` can tell a
    /// denied call (already reported by `ToolCallDenied`) from one that
    /// actually ran. Tool calls run one at a time, so there's never more
    /// than one in flight to track.
    pending_arguments: Option<String>,
    /// Whether the current tool-call header line (`🔨 name  detail`) is
    /// still open — printed without a trailing newline, waiting for
    /// whichever event resolves it to close it with a trailing status
    /// marker on the same line, the CLI's equivalent of the TUI
    /// transcript's trailing ✓/✗/?. A verbose dump between the header and
    /// its resolution closes it early instead, since the marker can no
    /// longer land on that same (now scrolled-past) line.
    tool_header_open: bool,
    /// Whether the current call showed an approval prompt (closing the
    /// header with `?`). Its own typed `y`/`N` answer already says how it
    /// resolved, so — unlike a call that ran with no prompt at all —
    /// `ToolCallCompleted`/`ToolCallDenied` draw no further marker for it.
    approval_shown: bool,
    /// This session's braille mark, the same one the TUI's gutter and the
    /// picker's rows draw. `None` for a one-off run, which has no
    /// session to be identified by; that falls back to a plain marker.
    mark: Option<String>,
    /// What the last turn found of CLANKERS.md, so a notice is printed only
    /// when it appears, changes or goes.
    instructions: Option<crate::instructions::Seen>,
}

impl TerminalAgentUi {
    /// Starts reporting what this session is doing while turns run, so a
    /// CLI session shows up in the picker the way a TUI one does.
    pub fn watch(&mut self, activity: ActivityWriter) {
        // Taken here because this is the moment the UI learns it belongs to
        // a session at all — a one-shot run never calls it.
        self.mark_as(activity.session_id());
        self.activity = Some(activity);
    }

    /// Draws replies with this session's mark rather than the one-off
    /// fallback. For a front end whose activity something else reports — a
    /// line-mode clanker, whose worker does — and so never calls
    /// [`Self::watch`].
    pub fn mark_as(&mut self, session_id: &str) {
        self.mark = Some(crate::tui::identicon_mark(session_id));
    }

    pub fn new(verbose: bool, show_model_label: bool) -> Self {
        TerminalAgentUi {
            verbose,
            show_model_label,
            spinner: None,
            pending_arguments: None,
            tool_header_open: false,
            approval_shown: false,
            activity: None,
            mark: None,
            instructions: None,
        }
    }

    /// Flips the `-v`-equivalent detail level live, for `/verbose` in a
    /// `clanker` loop. Takes effect from the next event on.
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }
}

impl TerminalAgentUi {
    /// Draws one agent event.
    ///
    /// Inherent rather than only reachable through [`AgentUi`], because the
    /// two ways this front end is driven arrive by different routes: the
    /// one-off run calls the loop inline and gets events
    /// through the trait, while a `clanker` running on a
    /// [`crate::conversation::Conversation`] sees them re-emitted as
    /// `Event::Agent(..)` and never touches the trait at all. Both render
    /// identically because both land here.
    pub async fn render_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::Steered { text } => {
                // Echoed back the way the prompt would have shown it, so the
                // transcript reads in the order the model saw it rather than
                // the message appearing to have come from nowhere.
                println!("\n{} {}", "❯".green().bold(), text);
            }
            AgentEvent::IterationStarted { iteration } => {
                if self.verbose {
                    println!("{}", format!("\n[Iteration {}]", iteration).bright_black());
                }
            }
            AgentEvent::RequestStarted => {
                self.spinner = Some(Spinner::start("Thinking..."));
            }
            AgentEvent::RequestFinished => {
                if let Some(spinner) = self.spinner.take() {
                    spinner.stop().await;
                }
            }
            // Deliberately ignored: a scrolling terminal can't re-wrap
            // text it has already printed, so the CLI buffers and renders
            // the complete `AssistantMessage` below instead.
            AgentEvent::AssistantDelta { .. } => {}
            AgentEvent::AssistantMessage {
                model,
                effort_level,
                text,
            } => {
                if self.show_model_label {
                    let label = format!("{}:", response_label(&model, &effort_level));
                    println!("{} {}", label.cyan(), wrap::wrap(&text));
                } else {
                    // The session's own mark, the same one the TUI's
                    // gutter and the picker's rows draw, so a reply is tied
                    // to the session it came from wherever you read it.
                    // Braille rather than the `●` this replaced: every
                    // pattern in that block is East Asian Width Neutral,
                    // while `●` is Ambiguous and some terminals give it two
                    // cells, shifting the wrapped lines under it out of
                    // line with the gutter.
                    //
                    // The indent is measured from the mark rather than
                    // fixed: a session's mark is two cells where the
                    // one-shot fallback is one, and a wrapped line has to
                    // start under the text either way.
                    let mark = self.mark.as_deref().unwrap_or("⠶");
                    let indent = " ".repeat(UnicodeWidthStr::width(mark) + 1);
                    println!("{} {}", mark.cyan(), wrap::wrap_indented(&text, &indent));
                }
                // One blank line after every transcript unit, matching
                // the TUI, which spaces its items the same way
                // regardless of what kind each one is.
                println!();
            }
            AgentEvent::Thinking { text } => {
                if self.verbose {
                    // The spinner is still animating on the current line
                    // here: the reply has resolved, but `RequestFinished`
                    // hasn't been emitted yet. Printing over it lands
                    // mid-line and then gets half-overwritten by the next
                    // redraw, so clear it first and let the thinking
                    // start its own line. The request it was tracking is
                    // already done, so `RequestFinished` simply finds
                    // nothing left to stop.
                    if let Some(spinner) = self.spinner.take() {
                        spinner.stop().await;
                    }
                    // Same marker-plus-hanging-indent shape the
                    // assistant's own reply uses, one step dimmer.
                    println!(
                        "{} {}",
                        "💭".bright_black(),
                        wrap::wrap_indented(&text, "   ").bright_black().italic()
                    );
                    println!();
                }
            }
            AgentEvent::ToolCallStarted { name, arguments } => {
                // Printed without a trailing newline — closed by
                // whichever of approval/denial/completion resolves it
                // next, with a trailing status marker, so the whole
                // call reads as one line: the CLI's equivalent of the
                // TUI transcript's gutter-marker-plus-trailing-status
                // row instead of repeating the tool's name on its own
                // line every time.
                print_tool_header(&name, &arguments);
                self.tool_header_open = true;
                self.approval_shown = false;
                if self.verbose {
                    println!();
                    self.tool_header_open = false;
                    print_fields(&tool_call_fields(&name, &arguments));
                }
                self.pending_arguments = Some(arguments);
            }
            AgentEvent::ToolCallDenied { name: _ } => {
                // Consumes the pending call so the ToolCallCompleted
                // that always follows a denial knows not to report the
                // same call again as if it had succeeded. A denial only
                // ever happens after an approval prompt — the typed `N`
                // that produced it already says how this resolved, so
                // there's no separate marker to draw underneath it.
                let _ = self.pending_arguments.take();
                println!();
            }
            AgentEvent::ToolCallCompleted { name: _, result } => {
                // Only the non-denied path reaches here with a pending
                // entry; a denial already reported itself and cleared it.
                if self.pending_arguments.take().is_none() {
                    return;
                }
                // A call that went through an approval prompt already
                // has its answer on screen (the typed `y`); only a call
                // that ran with no prompt at all still needs its own
                // closing marker.
                if !self.approval_shown {
                    self.close_tool_header(if crate::ui::tool_failed(&result) {
                        "✗".red()
                    } else {
                        "✓".green()
                    });
                }
                if self.verbose {
                    print_fields(&json_fields(&result));
                }
                println!();
            }
            AgentEvent::Error { message } => {
                println!("{} {}", "✗".red(), message);
                println!();
            }
            AgentEvent::Instructions { seen } => {
                if let Some(notice) = crate::ui::instructions_notice(self.instructions, seen) {
                    println!("{}", notice.bright_black());
                }
                self.instructions = seen;
            }
            AgentEvent::TurnFinished => {
                if self.verbose {
                    println!("{}", "✓ Agent finished".green());
                    println!();
                }
            }
        }
    }
}

impl AgentUi for TerminalAgentUi {
    fn event(&mut self, event: AgentEvent) -> impl Future<Output = ()> + Send {
        self.render_agent_event(event)
    }

    async fn approve(&mut self, request: ApprovalRequest) -> Result<bool> {
        self.prompt_approval(request)
    }
}

impl TerminalAgentUi {
    /// The blocking stdin prompt behind [`AgentUi::approve`], kept separate
    /// so the async wrapper stays trivial.
    fn prompt_approval(&mut self, request: ApprovalRequest) -> Result<bool> {
        // Announced before the prompt blocks on stdin: this is exactly the
        // state worth seeing from another terminal, and it's the one a
        // blocking loop would otherwise never report.
        if let Some(activity) = &self.activity {
            activity.set(
                Some(crate::store::Activity::AwaitingApproval),
                Some(&crate::ui::approval_summary(&request)),
            );
        }
        let answered = self.ask_approval(request);
        if let Some(activity) = &self.activity {
            activity.set(Some(crate::store::Activity::Working), None);
        }
        answered
    }

    fn ask_approval(&mut self, request: ApprovalRequest) -> Result<bool> {
        // Closes the tool-call header with a trailing `?`, matching the
        // TUI transcript's `AwaitingApproval` marker, before the prompt
        // itself — which has no TUI transcript equivalent (that lives in
        // the TUI's separate approval modal instead) — continues below.
        self.close_tool_header("?".yellow());
        self.approval_shown = true;

        let category_label = match request.category {
            "read" => "Read from disk",
            "write" => "Write to disk",
            "terminal" => "Terminal command",
            _ => "Unknown action",
        };

        println!("\n{} {} requested:", "⚠".yellow(), category_label);
        println!("  Tool: {}", request.tool_name.cyan());

        // Parse and display arguments nicely
        if let Ok(args) = serde_json::from_str::<serde_json::Value>(&request.arguments) {
            if let Some(obj) = args.as_object() {
                for (key, value) in obj {
                    let display_value = if key == "content" {
                        // Truncate long content
                        let s = value
                            .as_str()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| value.to_string());
                        if s.len() > 100 {
                            format!("{}... ({} chars)", &s[..100], s.len())
                        } else {
                            s
                        }
                    } else {
                        value
                            .as_str()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| value.to_string())
                    };
                    println!("  {}: {}", key, display_value.bright_black());
                }
            }
        }

        print!("\n{} ", "Allow? [y/N]:".blue());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        Ok(parse_yes_no(&input))
    }

    /// Closes the currently open tool-call header line with a trailing
    /// status marker — landing right after the name/detail on the same
    /// line, matching the TUI transcript's trailing ✓/? — or, if the line
    /// was already closed by a verbose dump, prints the marker on its own
    /// short indented line instead, rather than repeating the tool's name.
    /// Only ever called for `?` (always) and `✓` on a call that ran with no
    /// approval prompt — one that went through a prompt has its answer on
    /// screen already and draws no further marker.
    fn close_tool_header(&mut self, marker: ColoredString) {
        if self.tool_header_open {
            println!(" {marker}");
            self.tool_header_open = false;
        } else {
            println!("  {marker}");
        }
    }
}

/// Prints `🔨 name  detail` with no trailing newline — `detail` is the file
/// path or command the call is acting on when its arguments have one, the
/// same terse identification the TUI always shows for a tool call,
/// regardless of `-v`. Left open for [`TerminalAgentUi::close_tool_header`]
/// to finish with a trailing status marker.
fn print_tool_header(name: &str, arguments: &str) {
    match primary_argument(arguments) {
        Some(detail) => print!(
            "{} {}  {}",
            "🔨".magenta(),
            name.bold(),
            detail.bright_black()
        ),
        None => print!("{} {}", "🔨".magenta(), name.bold()),
    }
    let _ = io::stdout().flush();
}

/// The verbose-only per-field breakdown under a tool notice, matching the
/// TUI's indentation for the same data.
fn print_fields(fields: &[(String, String)]) {
    for (key, shown) in fields {
        println!("     {}  {}", key.bright_black(), shown);
    }
}

/// Every tool with what it may do, one line each, in the column layout
/// `clank tools` and `/tools` share.
pub fn tool_lines(access: &crate::config::ToolAccessSettings) -> Vec<String> {
    let rows = crate::ui::tool_rows(access);
    // Measured rather than fixed. The width used to be the longest built-in
    // plus a space; a tool from a server is named `server/tool` and routinely
    // longer than that, which ran the name into the column beside it.
    let width = rows
        .iter()
        .map(|(name, _)| name.chars().count())
        .max()
        .unwrap_or(0)
        .max(20);
    rows.iter()
        .map(|(name, value)| format!("  {name:<width$} {}", value.bright_black()))
        .collect()
}

/// What a line-mode clanker knows about its session, which the worker owns.
///
/// Read off the session once, before it moves into the worker, and kept
/// current from the worker's events after that — the same arrangement the
/// TUI's `App` has. It is what the read-only commands (`/model`, `/status`,
/// …) are answered from, and what tells a change from a repeat when a
/// setting's event says what it ended up as.
#[derive(Debug, Clone)]
pub struct SessionView {
    pub short_id: String,
    pub title: String,
    pub model: String,
    pub effort_level: Option<String>,
    pub temperature: Option<f32>,
    pub max_iterations: Option<usize>,
    pub verbose: bool,
    pub highlight: bool,
    pub sandbox: bool,
    pub stream: bool,
    pub working_dir: Option<String>,
    pub tool_access: crate::config::ToolAccessSettings,
    pub total_tokens: i64,
}

impl SessionView {
    pub fn of(session: &crate::session::ChatSession) -> Self {
        SessionView {
            short_id: session.short_id().to_string(),
            title: session.title().to_string(),
            model: session.model().to_string(),
            effort_level: session.effort_level().map(str::to_string),
            temperature: session.temperature(),
            max_iterations: session.max_iterations(),
            verbose: session.verbose(),
            highlight: session.highlight(),
            sandbox: session.sandbox(),
            stream: session.stream(),
            working_dir: session.working_dir().map(str::to_string),
            tool_access: session.tool_access().clone(),
            total_tokens: session.total_tokens(),
        }
    }

    /// Takes in what `event` says about the session, and returns the lines
    /// to print for it — none for an event that changes nothing a line-mode
    /// clanker shows. The agent's own progress, approvals and
    /// [`Event::Ready`](crate::conversation::Event::Ready) are the caller's:
    /// they need the terminal UI or the worker, which this has neither of.
    pub fn apply(&mut self, event: &crate::conversation::Event) -> Vec<String> {
        use crate::conversation::Event;
        use crate::ui;

        fn set_or_is(changed: bool) -> &'static str {
            if changed {
                "set to"
            } else {
                "is"
            }
        }
        fn done(text: String) -> String {
            format!("{} {text}", "✓".green())
        }
        fn info(text: String) -> String {
            text.blue().to_string()
        }

        match event {
            Event::ModelChanged {
                model,
                effort_level,
            } => {
                let changed = *model != self.model;
                self.model = model.clone();
                self.effort_level = effort_level.clone();
                vec![done(format!("Model {} {model}", set_or_is(changed)))]
            }
            Event::EffortChanged { effort_level } => {
                let changed = *effort_level != self.effort_level;
                self.effort_level = effort_level.clone();
                let label = effort_level.as_deref().unwrap_or("default");
                vec![done(format!("Effort {} {label}", set_or_is(changed)))]
            }
            Event::MaxIterationsChanged { max_iterations } => {
                let changed = *max_iterations != self.max_iterations;
                self.max_iterations = *max_iterations;
                let label = max_iterations.map_or_else(|| "default".to_string(), |n| n.to_string());
                vec![done(format!(
                    "Max iterations {} {label}",
                    set_or_is(changed)
                ))]
            }
            Event::TemperatureChanged { temperature } => {
                let changed = *temperature != self.temperature;
                self.temperature = *temperature;
                let label = temperature.map_or_else(|| "default".to_string(), |t| t.to_string());
                vec![done(format!("Temperature {} {label}", set_or_is(changed)))]
            }
            Event::ToolAccessChanged { access } => {
                let changed = *access != self.tool_access;
                self.tool_access = access.clone();
                // The whole list, not just the row that moved: naming one
                // state tells you neither what else is set nor what it was
                // set from, and this is the readout people check before
                // walking away.
                let verb = if changed { "set to" } else { "are" };
                let mut lines = vec![done(format!("Tools {verb}:"))];
                lines.extend(tool_lines(access));
                lines
            }
            Event::VerboseChanged { verbose } => {
                self.verbose = *verbose;
                vec![info(ui::verbose_notice(*verbose, true))]
            }
            Event::HighlightChanged { highlight } => {
                // Recorded either way: the CLI draws no band, but the
                // setting belongs to the session, so switching it here is
                // what the TUI picks up on its next resume.
                self.highlight = *highlight;
                vec![info(ui::highlight_notice(*highlight, true))]
            }
            Event::StreamChanged { stream } => {
                self.stream = *stream;
                vec![info(ui::stream_notice(*stream, true))]
            }
            Event::SandboxChanged { sandbox } => {
                self.sandbox = *sandbox;
                vec![info(ui::sandbox_notice(*sandbox, true))]
            }
            // Only a change is announced: the worker says what the title is
            // after every turn, which the TUI takes in silently.
            Event::TitleChanged { title } if *title != self.title => {
                self.title = title.clone();
                vec![info(ui::title_notice(title, true))]
            }
            Event::TitleChanged { .. } => Vec::new(),
            Event::TokensUsed { total_tokens } => {
                self.total_tokens = *total_tokens;
                Vec::new()
            }
            Event::Compacting { model } => vec![info(ui::compacting_notice(model))],
            Event::Compacted { folded } => vec![info(ui::compacted_notice(*folded))],
            Event::CompactionSkipped { reason } => vec![done(reason.clone())],
            Event::McpReconnected { summary } => vec![info(summary.clone())],
            Event::Cancelled => vec![format!("{} Cancelled", "✗".red())],
            // The caller's — see above.
            Event::Agent(_) | Event::ApprovalRequested(_) | Event::Ready => Vec::new(),
            // Nothing to show. The message echoed is the line just typed; a
            // blocking prompt is never busy while it reads, and nothing can
            // queue behind a turn it cannot type during; `$` and the model
            // browser are TUI-only, so neither of their answers arrives.
            Event::UserMessage(_)
            | Event::Busy(_)
            | Event::Queued { .. }
            | Event::ShellStarted { .. }
            | Event::ShellFinished { .. }
            | Event::ModelsListed(_)
            | Event::ModelsUnavailable(_) => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ToolAccessSettings;
    use crate::conversation::Event;

    fn view() -> SessionView {
        SessionView {
            short_id: "a1b2c3d4".to_string(),
            title: "Fix the parser".to_string(),
            model: "model-a".to_string(),
            effort_level: None,
            temperature: None,
            max_iterations: Some(20),
            verbose: false,
            highlight: true,
            sandbox: true,
            stream: true,
            working_dir: None,
            tool_access: ToolAccessSettings::none(),
            total_tokens: 0,
        }
    }

    /// Joined, so a test reads the words rather than the escape codes
    /// around them.
    fn shown(view: &mut SessionView, event: Event) -> String {
        view.apply(&event).join("\n")
    }

    #[test]
    fn a_setting_says_whether_it_changed() {
        // The worker's event says what a setting ended up as, not whether it
        // moved — telling the two apart is the view's job, because it is
        // the one holding what it was before.
        let mut view = view();
        let changed = shown(
            &mut view,
            Event::ModelChanged {
                model: "model-b".to_string(),
                effort_level: Some("high".to_string()),
            },
        );
        assert!(changed.contains("Model set to model-b"), "{changed}");
        assert_eq!(view.model, "model-b");
        assert_eq!(view.effort_level.as_deref(), Some("high"));

        let repeated = shown(
            &mut view,
            Event::ModelChanged {
                model: "model-b".to_string(),
                effort_level: Some("high".to_string()),
            },
        );
        assert!(repeated.contains("Model is model-b"), "{repeated}");

        let nullified = shown(
            &mut view,
            Event::MaxIterationsChanged {
                max_iterations: None,
            },
        );
        assert!(
            nullified.contains("Max iterations set to default"),
            "{nullified}"
        );
        assert_eq!(view.max_iterations, None);
    }

    #[test]
    fn the_title_is_announced_only_when_it_changes() {
        let mut view = view();
        let same = Event::TitleChanged {
            title: "Fix the parser".to_string(),
        };
        assert!(view.apply(&same).is_empty());
        let renamed = shown(
            &mut view,
            Event::TitleChanged {
                title: "Fix the lexer".to_string(),
            },
        );
        assert!(renamed.contains("Fix the lexer"), "{renamed}");
        assert_eq!(view.title, "Fix the lexer");
    }

    #[test]
    fn a_change_to_the_tools_shows_the_whole_list() {
        let mut view = view();
        let lines = view.apply(&Event::ToolAccessChanged {
            access: ToolAccessSettings::defaults(),
        });
        assert!(lines[0].contains("Tools set to:"), "{}", lines[0]);
        assert!(lines.iter().any(|line| line.contains("read_file")));
        assert_eq!(view.tool_access, ToolAccessSettings::defaults());
    }

    #[test]
    fn what_a_blocking_prompt_has_no_use_for_prints_nothing() {
        let mut view = view();
        for event in [
            Event::Ready,
            Event::Busy(true),
            Event::UserMessage("the line just typed".to_string()),
            Event::Queued {
                text: "x".to_string(),
            },
        ] {
            assert!(view.apply(&event).is_empty(), "{event:?}");
        }
        // Kept for `/status`, but not announced after every turn.
        assert!(view
            .apply(&Event::TokensUsed {
                total_tokens: 1_234
            })
            .is_empty());
        assert_eq!(view.total_tokens, 1_234);
    }
}
