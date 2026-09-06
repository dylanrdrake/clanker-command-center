//! The launch screen: everything you can start or return to, in one list.
//!
//! There was a second screen listing all sessions, which meant the launch
//! screen could only offer a handful and hand off to it. One list grouped by
//! where each session lives says more in fewer keystrokes — the sessions for
//! the directory you're in are the ones you almost always want, and the rest
//! are still right there.
//!
//! Kept free of I/O: the caller loads sessions and acts on the
//! [`Activation`] returned when a row is chosen.

use super::render::{band, draw_rule, home_relative, identicon, pad_to};
use crate::store::{mode_label, Activity, LastMessage, LastState, SessionSummary, KIND_AGENT_CHAT};
use ratatui::prelude::*;
use ratatui::widgets::Paragraph;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionRow {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub updated_at: i64,
    /// Where the session was started, which is its sandbox boundary. `None`
    /// for one recorded before that was tracked.
    pub working_dir: Option<String>,
    /// Where the session left off, from its most recent stored message.
    /// `None` for one that has never been used.
    pub last: Option<LastMessage>,
    /// What its process says it's doing, when it said anything.
    pub activity: Option<Activity>,
    /// The line that goes with that — for an approval, what is being asked.
    pub activity_detail: Option<String>,
    /// When the process running it last checked in, so a state it claimed and
    /// then died holding can be told apart from one that's still true.
    pub heartbeat: Option<i64>,
    /// Total tokens spent across this session's turns so far.
    pub total_tokens: i64,
}

impl From<SessionSummary> for SessionRow {
    fn from(summary: SessionSummary) -> Self {
        SessionRow {
            id: summary.id,
            kind: summary.kind,
            title: summary.title,
            updated_at: summary.updated_at,
            working_dir: summary.working_dir,
            last: None,
            activity: summary.activity,
            activity_detail: summary.activity_detail,
            heartbeat: summary.heartbeat,
            total_tokens: summary.total_tokens,
        }
    }
}

impl SessionRow {
    pub fn short_id(&self) -> &str {
        &self.id[..8.min(self.id.len())]
    }

    pub fn is_agentic(&self) -> bool {
        self.kind == KIND_AGENT_CHAT
    }
}

/// A row on the launch screen.
///
/// `NewSession` carries nothing and `Resume` carries a whole row, which
/// clippy reads as a size imbalance worth boxing. It isn't: there is exactly
/// one `NewSession` in the list and one `Resume` per session, so boxing
/// would trade a single row's worth of unused space for an allocation and a
/// pointer chase on every session there is.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchItem {
    NewSession,
    Resume(SessionRow),
}

/// What choosing a row means to the caller.
#[derive(Debug, Clone)]
pub enum Activation {
    NewSession,
    Resume(SessionRow),
    Delete(SessionRow),
    /// Resume a session whose recorded directory is gone, in the current
    /// one, repointing it there.
    Repoint(SessionRow),
}

impl SessionRow {
    /// What to show for this session.
    ///
    /// The process's own word wins when it gave one: it knows things the
    /// stored messages can't say, like that a request is in flight or that
    /// somebody is being asked a question. Otherwise the last message
    /// speaks, which is all a session nobody is running can offer.
    pub fn last_state(&self) -> LastState {
        crate::store::last_state(self.activity, self.heartbeat, self.last.as_ref())
    }

    /// Whether a live process holds this session, and so whether opening it
    /// will be refused. Distinct from being *busy*: a session sitting idle
    /// at a prompt in another terminal is held but not working.
    pub fn is_held(&self) -> bool {
        crate::store::heartbeat_is_live(self.heartbeat)
    }
}

impl SessionRow {
    /// The line to show after the state, when there is one worth showing.
    ///
    /// A pending approval displaces the conversation preview: what the
    /// session last said matters far less than what it is stuck asking, and
    /// that is the row someone reading this list needs to act on.
    pub fn preview(&self) -> Option<String> {
        if self.activity == Some(Activity::AwaitingApproval) {
            if let Some(detail) = &self.activity_detail {
                return Some(format!("needs approval — {detail}"));
            }
        }
        self.last
            .as_ref()
            .map(|last| last.preview.clone())
            .filter(|preview| !preview.is_empty())
    }
}

/// The glyph and colour for a session's state.
///
/// The glyph carries the meaning and the colour only reinforces it, so the
/// list still reads on a terminal without colour.
///
/// Every glyph here is East Asian Width *Neutral* or *Narrow*, which is what
/// keeps the column aligned. The obvious circles — `●`, `◐`, `○`, `•` — are
/// *Ambiguous*, and a terminal may draw those two cells wide while drawing
/// the rest one, which pushed some rows a column right of the others.
fn state_badge(state: LastState, held: bool, tick: usize) -> (String, Style) {
    // Held by another process and not otherwise saying so. `Working` and
    // `AwaitingApproval` are already gated on a live heartbeat, so those two
    // only ever occur while someone holds the session and their own glyphs
    // already imply it. Every other state can belong to a session sitting
    // idle at a prompt in another terminal, which looks openable and isn't
    // — so it gets a glyph of its own rather than a `✓` you can't act on.
    if held && !matches!(state, LastState::Working | LastState::AwaitingApproval) {
        // A screen, because that is the fact: another terminal has it open.
        // Not a padlock — every lock glyph (🔒 🔓 🔏 🔐) is East Asian Width
        // *Wide*, and `⚿` is *Ambiguous*, so any of them would push this row
        // a column right of the others on some terminals and not on
        // others. If the font lacks this one it draws tofu, which is ugly
        // but still one cell, so the column survives either way.
        return (
            format!("{:<BADGE_WIDTH$}", "⎚"),
            Style::new().dark_gray().bold(),
        );
    }

    let (glyph, style): (String, Style) = match state {
        LastState::New => (" ".to_string(), Style::new()),
        // The conversation's own spinner, frame for frame and in the same
        // yellow, so a busy session animates identically whether you're
        // watching it from the list or sitting inside it.
        LastState::Working => (super::render::busy_frame(tick), Style::new().yellow()),
        LastState::AwaitingApproval => ("?".to_string(), Style::new().yellow().bold()),
        LastState::Failed => ("✗".to_string(), Style::new().red().bold()),
        LastState::Replied => ("✓".to_string(), Style::new().green()),
        LastState::NoReply => ("⋯".to_string(), Style::new().cyan()),
        LastState::Interrupted => ("⚑".to_string(), Style::new().yellow()),
    };
    (format!("{glyph:<BADGE_WIDTH$}"), style)
}

// Fixed columns, so the preview can be given whatever the line has left.
const MARK_WIDTH: usize = 2 + ICON_WIDTH + 1; // selection marker, mark, gap
                                               // One glyph and a gutter. `mode_label` returns a two-column emoji that
                                               // `column` pads as one `char`, so the cell draws a column wider than this
                                               // says — harmlessly, since every row carries one and they stay aligned with
                                               // each other.
const KIND_WIDTH: usize = 3;
const TITLE_WIDTH: usize = 24;
const DIR_WIDTH: usize = 24;
const WHEN_WIDTH: usize = 8;
/// Same misalignment tradeoff as `KIND_WIDTH`: `🪙 ` is a two-column glyph
/// that `column` pads as one `char`, drawing a column wider than this says —
/// harmless, since every row carries exactly one.
const TOKENS_WIDTH: usize = 9;

/// Below this a preview says too little to be worth the clutter.
const MIN_PREVIEW: usize = 12;
/// Every badge is a single cell — see `state_badge` — and padded to this so
/// the column stays straight.
const BADGE_WIDTH: usize = 2;

/// The mark is two braille cells: 4 dots across by 4 down. One cell is 2
/// dots wide by 4 tall in a character box about half as wide as it is tall,
/// so the dots already sit on a square lattice — it's the glyph that's a
/// 1:2 rectangle. Two of them side by side makes the block square as well.
const ICON_WIDTH: usize = 2;

/// A list with a moving selection. Empty lists are allowed (a fresh install
/// has no sessions), in which case there is nothing to activate.
#[derive(Debug, Default)]
pub struct Picker {
    pub items: Vec<LaunchItem>,
    pub selected: usize,
    /// Set while a delete is awaiting y/n confirmation, since dropping a
    /// saved conversation shouldn't be one keystroke away.
    pub confirming_delete: Option<SessionRow>,
    /// Set while a rename is in progress: the row being renamed, and the
    /// text typed so far (pre-filled with its current title).
    pub renaming: Option<(SessionRow, String)>,
    /// Set while a repoint is awaiting y/n: the row, and the directory it
    /// was recorded in and can no longer be opened from.
    pub confirming_repoint: Option<(SessionRow, String)>,
    /// Why the last attempt to open a session failed, shown in place of the
    /// key hints. Opening can fail for reasons the user needs to act on —
    /// a session whose directory is gone, or one deleted from under the
    /// list — and the picker is where they still have other choices.
    pub notice: Option<String>,
}

impl Picker {
    /// Every session, newest first, in one list.
    ///
    /// Not split by directory. The split put the sessions you were most
    /// likely to want under a heading and the rest under another, which
    /// reads well until you are looking for one and have to decide which
    /// half it is in. The directory each session belongs to is a column on
    /// its own row instead, where it can be read without being grouped by.
    pub fn launch(all: Vec<SessionRow>, _cwd: Option<&str>) -> Self {
        let mut items = vec![LaunchItem::NewSession];
        items.extend(all.into_iter().map(LaunchItem::Resume));

        Picker {
            items,
            selected: 0,
            confirming_delete: None,
            renaming: None,
            confirming_repoint: None,
            notice: None,
        }
    }

    /// Folds freshly-read state into the rows already on screen, reporting
    /// whether anything changed.
    ///
    /// The whole list is rebuilt, so sessions started or deleted elsewhere
    /// appear and disappear — a view you leave open to watch is not much use
    /// if it only knows the sessions that existed when you opened it.
    ///
    /// Rebuilding is why the selection is restored by session id rather than
    /// left on its row number: rows can be inserted, removed or regrouped
    /// underneath it, and an index would quietly come to rest on a different
    /// conversation.
    pub fn refresh(&mut self, latest: Vec<SessionRow>, cwd: Option<&str>) -> bool {
        let rebuilt = Picker::launch(latest, cwd);
        if rebuilt.items == self.items {
            return false;
        }

        // The selection follows the session, not the row number: rebuilding
        // can insert, remove or regroup rows, and a cursor that stayed on an
        // index would silently come to rest on a different conversation.
        let selected = self.selected_session().map(|row| row.id.clone());
        let previous = self.selected;
        self.items = rebuilt.items;

        self.selected = selected
            .and_then(|id| {
                self.items
                    .iter()
                    .position(|item| matches!(item, LaunchItem::Resume(row) if row.id == id))
            })
            // Whatever was selected is gone — deleted from another process,
            // say. Stay about where the eye was rather than jumping home.
            .unwrap_or_else(|| previous.min(self.items.len().saturating_sub(1)));
        if !self.selectable(self.selected) {
            self.move_up();
        }
        true
    }

    /// Whether any row is mid-request, which is the only reason this screen
    /// needs to redraw between refreshes.
    pub fn has_working_session(&self) -> bool {
        self.items.iter().any(|item| {
            matches!(item, LaunchItem::Resume(row) if row.last_state() == LastState::Working)
        })
    }

    /// Whether a row can hold the cursor. Only an index past the end can't,
    /// now that every row is a choice.
    fn selectable(&self, index: usize) -> bool {
        self.items.get(index).is_some()
    }

    pub fn move_up(&mut self) {
        let mut index = self.selected;
        while index > 0 {
            index -= 1;
            if self.selectable(index) {
                self.selected = index;
                return;
            }
        }
    }

    pub fn move_down(&mut self) {
        let mut index = self.selected;
        while index + 1 < self.items.len() {
            index += 1;
            if self.selectable(index) {
                self.selected = index;
                return;
            }
        }
    }

    pub fn selected_session(&self) -> Option<&SessionRow> {
        match self.items.get(self.selected) {
            Some(LaunchItem::Resume(row)) => Some(row),
            _ => None,
        }
    }

    /// What the currently selected row does when chosen.
    pub fn activate(&self) -> Option<Activation> {
        match self.items.get(self.selected)? {
            LaunchItem::NewSession => Some(Activation::NewSession),
            LaunchItem::Resume(row) => Some(Activation::Resume(row.clone())),
        }
    }

    /// Begins a delete, which then needs confirming. Only meaningful on a
    /// session row.
    pub fn begin_delete(&mut self) {
        if let Some(row) = self.selected_session() {
            self.confirming_delete = Some(row.clone());
        }
    }

    /// Offers to resume a session here when its own directory is gone.
    pub fn begin_repoint(&mut self, row: SessionRow, missing: String) {
        self.confirming_repoint = Some((row, missing));
    }

    /// Resolves a pending repoint; `true` means resume here and repoint.
    pub fn resolve_repoint(&mut self, confirmed: bool) -> Option<Activation> {
        let (row, _) = self.confirming_repoint.take()?;
        confirmed.then_some(Activation::Repoint(row))
    }

    /// Resolves a pending confirmation; `true` means go ahead.
    pub fn resolve_delete(&mut self, confirmed: bool) -> Option<Activation> {
        let row = self.confirming_delete.take()?;
        confirmed.then_some(Activation::Delete(row))
    }

    /// Drops a row after it's been deleted, keeping the selection in range.
    ///
    /// Takes the section label with it when that was the last row under it:
    /// a header with nothing beneath reads as a section that failed to load
    /// rather than one that's empty.
    pub fn remove_session(&mut self, id: &str) {
        self.items
            .retain(|item| !matches!(item, LaunchItem::Resume(row) if row.id == id));
        if self.selected >= self.items.len() {
            self.selected = self.items.len().saturating_sub(1);
        }
        if !self.selectable(self.selected) {
            self.move_up();
        }
    }

    /// Begins renaming the selected session, pre-filling the input with its
    /// current title so it can be edited rather than retyped from scratch.
    /// Only meaningful on a session row.
    pub fn begin_rename(&mut self) {
        if let Some(row) = self.selected_session() {
            self.renaming = Some((row.clone(), row.title.clone()));
        }
    }

    pub fn rename_insert_char(&mut self, c: char) {
        if let Some((_, input)) = &mut self.renaming {
            input.push(c);
        }
    }

    pub fn rename_backspace(&mut self) {
        if let Some((_, input)) = &mut self.renaming {
            input.pop();
        }
    }

    /// Cancels an in-progress rename without saving anything.
    pub fn cancel_rename(&mut self) {
        self.renaming = None;
    }

    /// Confirms the rename, returning the session id and its new title to
    /// persist. A blank (post-trim) title isn't meaningful, so it's
    /// rejected rather than saved — the rename stays open for another try
    /// instead of silently discarding it on a stray Enter.
    pub fn confirm_rename(&mut self) -> Option<(String, String)> {
        let (row, input) = self.renaming.as_ref()?;
        let title = input.trim().to_string();
        if title.is_empty() {
            return None;
        }
        let id = row.id.clone();
        self.renaming = None;
        Some((id, title))
    }

    /// Reflects a persisted rename in the row itself, so the list shows it
    /// without a reload.
    pub fn apply_rename(&mut self, id: &str, title: String) {
        for item in &mut self.items {
            if let LaunchItem::Resume(row) = item {
                if row.id == id {
                    row.title = title;
                    return;
                }
            }
        }
    }
}

pub fn draw(
    frame: &mut Frame,
    picker: &Picker,
    title: &str,
    dir: Option<&str>,
    // Whether the selected row is banded. Global rather than per-session:
    // this screen belongs to no session, so there is nothing to override.
    selection: bool,
    hint: &str,
    tick: usize,
) {
    let areas = Layout::vertical([
        Constraint::Length(1), // title
        Constraint::Length(1), // rule
        Constraint::Min(1),    // list
        Constraint::Length(1), // footer/hint
    ])
    .split(frame.area());

    let mut lines: Vec<Line> = Vec::new();
    // Set by a row that wants a blank line after it. Only "Spawn clanker"
    // does, to keep it off the top of the list proper.
    let mut trailing_blank = false;
    // Which built line the cursor is on, which is what the list has to keep
    // on screen. Not the item index: headers push a blank line ahead of
    // themselves, so the two drift apart as sections go by.
    let mut selected_line = 0usize;
    let width = areas[2].width as usize;
    for (index, item) in picker.items.iter().enumerate() {
        let selected = index == picker.selected;
        let marker = if selected { "❯ " } else { "  " };
        let base = if selected {
            Style::new().bold()
        } else {
            Style::new()
        };

        let mut spans = vec![Span::styled(marker, Style::new().cyan().bold())];
        match item {
            LaunchItem::NewSession => {
                spans.push(Span::styled("Spawn clanker", base.green()));
                // Still set apart from the sessions below it, now that no
                // heading does that: it is the one row that isn't one.
                trailing_blank = true;
            }
            LaunchItem::Resume(row) => {
                // Identity first, then name, then state: what the session
                // is, what it's called, then what it's doing.
                let (mark, mark_style) = identicon(&row.id);
                spans.push(Span::styled(mark, mark_style));
                spans.push(Span::raw(" "));

                spans.push(Span::styled(column(&row.title, TITLE_WIDTH), base));

                let (glyph, style) = state_badge(row.last_state(), row.is_held(), tick);
                spans.push(Span::styled(format!("{glyph} "), style));

                spans.push(Span::styled(
                    column(mode_label(row.is_agentic()), KIND_WIDTH),
                    if row.is_agentic() {
                        Style::new().yellow()
                    } else {
                        Style::new().cyan()
                    },
                ));

                spans.push(Span::styled(
                    column(
                        &format!("🪙 {}", format_tokens_compact(row.total_tokens)),
                        TOKENS_WIDTH,
                    ),
                    Style::new().dark_gray(),
                ));

                // Now that the list isn't grouped by directory, every row
                // carries its own. `.` for the one you are in, which is the
                // common case and the one worth making shortest; otherwise
                // `~`-relative, or absolute for anything above home.
                let row_dir = match &row.working_dir {
                    Some(d) if Some(d.as_str()) == dir => ".".to_string(),
                    Some(d) => home_relative(d),
                    None => "dir not recorded".to_string(),
                };
                spans.push(Span::styled(
                    column(&row_dir, DIR_WIDTH),
                    Style::new().dark_gray(),
                ));

                spans.push(Span::styled(
                    column(&relative_time(row.updated_at), WHEN_WIDTH),
                    Style::new().dark_gray(),
                ));

                let used = MARK_WIDTH
                    + TITLE_WIDTH
                    + BADGE_WIDTH
                    + 1
                    + KIND_WIDTH
                    + TOKENS_WIDTH
                    + DIR_WIDTH
                    + WHEN_WIDTH;

                // Whatever is left of the line goes to what was last said,
                // so the row describes where the session got to rather than
                // only when it was touched. Dropped entirely when the
                // terminal is too narrow to say anything useful.
                if let Some(preview) = row.preview() {
                    let room = width.saturating_sub(used + 2);
                    if room >= MIN_PREVIEW {
                        spans.push(Span::styled(
                            format!("  {}", truncate(&preview, room)),
                            Style::new().dark_gray().italic(),
                        ));
                    }
                }
            }
        }
        let mut line = Line::from(spans);
        // The same band the transcript puts behind your own messages: the
        // cursor is easy to lose in a list where several rows are animating.
        if selected && selection {
            pad_to(&mut line, width);
            line.style = line.style.patch(band());
        }
        if selected {
            selected_line = lines.len();
        }
        lines.push(line);
        if std::mem::take(&mut trailing_blank) {
            lines.push(Line::raw(""));
        }
    }

    if picker.items.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no saved clankers yet",
            Style::new().dark_gray().italic(),
        )));
    }

    // A pending delete or rename takes over the hint line, so the question
    // (or the text being typed) is right where the answer goes.
    let footer = if let Some((_, input)) = &picker.renaming {
        Line::from(vec![
            Span::styled(" rename to: ", Style::new().yellow().bold()),
            Span::raw(input.clone()),
            Span::styled("▏", Style::new().yellow()),
        ])
    } else if let Some(row) = &picker.confirming_delete {
        Line::from(vec![
            Span::styled(
                format!(
                    " delete clanker {} ({})? ",
                    row.short_id(),
                    truncate(&row.title, 30)
                ),
                Style::new().red().bold(),
            ),
            Span::styled("y / n", Style::new().red()),
        ])
    } else if let Some((_, missing)) = &picker.confirming_repoint {
        Line::from(vec![
            Span::styled(
                format!(
                    " {} is gone — resume here instead? ",
                    home_relative(missing)
                ),
                Style::new().yellow().bold(),
            ),
            Span::styled("y / n", Style::new().yellow()),
        ])
    } else {
        Line::from(Span::styled(format!(" {hint}"), Style::new().dark_gray()))
    };

    let mut heading = vec![Span::styled(title.to_string(), Style::new().bold())];
    // The directory the list is grouped around: "In this directory" means
    // nothing without saying which.
    if let Some(dir) = dir {
        heading.push(Span::styled(
            format!("  {}", home_relative(dir)),
            Style::new().dark_gray(),
        ));
    }
    // Everything that doesn't fit used to be silently dropped: past a
    // screenful, the extra sessions were neither drawn nor hinted at, and
    // the cursor could be moved onto a row nobody could see. The offset is
    // derived from the selection rather than remembered, so there is no
    // scroll state to get out of step with a list that regroups under it
    // when a session is deleted or moves between sections.
    let visible = areas[2].height as usize;
    let total = lines.len();
    let offset = if total <= visible {
        0
    } else {
        // Centred once scrolling starts, but only then: near the top the
        // cursor should move down the screen rather than drag the list.
        selected_line
            .saturating_sub(visible / 2)
            .min(total - visible)
    };

    // Counted, not just marked: "more below" says nothing about whether
    // it's one session or forty.
    let hidden_above = offset;
    let hidden_below = total.saturating_sub(offset + visible);
    let rule_hint = match (hidden_above, hidden_below) {
        (0, 0) => None,
        (0, below) => Some(format!("{below} more below ")),
        (above, 0) => Some(format!("{above} more above ")),
        (above, below) => Some(format!("{above} above · {below} below ")),
    };

    frame.render_widget(Paragraph::new(Line::from(heading)), areas[0]);
    draw_rule(frame, areas[1], rule_hint.as_deref());
    frame.render_widget(
        Paragraph::new(Text::from(lines)).scroll((offset as u16, 0)),
        areas[2],
    );
    // A failed open replaces the key hints: it's the thing that just
    // happened, and the hints are still discoverable by pressing anything.
    match &picker.notice {
        Some(notice) => frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!("✗ {notice}"),
                Style::new().red(),
            ))),
            areas[3],
        ),
        None => frame.render_widget(Paragraph::new(footer), areas[3]),
    }
}

/// The prompt shown before a clanker is spawned, so it starts with a real
/// name instead of "Untitled". Leaving it blank falls back to the usual
/// behavior: derived from the first message once there is one.
///
/// Also where its mark is chosen. The mark is hashed from the session id, so
/// it cannot be changed once the session exists — this screen holds an id
/// that has not been written yet, and Tab rolls another, which is the only
/// point at which the thing you will be looking at for the next week is
/// yours to pick rather than dealt to you.
pub fn draw_naming(frame: &mut Frame, input: &str, id: &str) {
    let areas = Layout::vertical([
        Constraint::Length(1), // title
        Constraint::Length(1), // rule
        Constraint::Min(1),    // content
        Constraint::Length(1), // hint
    ])
    .split(frame.area());

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled("clank", Style::new().bold()))),
        areas[0],
    );
    draw_rule(frame, areas[1], None);

    // The mark this session will carry for the rest of its life, shown
    // before it has one: it is hashed from an id that does not have to be
    // kept, so this is the only moment it can be chosen rather than dealt.
    let (mark, mark_style) = identicon(id);
    let lines = vec![
        Line::raw(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled(mark, mark_style),
            Span::styled("  Tab for another", Style::new().dark_gray()),
        ]),
        Line::raw(""),
        Line::from(vec![
            Span::styled("  Clanker name  ", Style::new().dark_gray()),
            Span::raw(input.to_string()),
            Span::styled("▏", Style::new().yellow()),
        ]),
    ];
    frame.render_widget(Paragraph::new(Text::from(lines)), areas[2]);

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            " A name is required · Tab reroll · Enter spawn · Esc cancel",
            Style::new().dark_gray(),
        ))),
        areas[3],
    );
}

/// A token count squeezed into the few characters a list row can spare —
/// `format_tokens` in `ui.rs` is for `/status`, where there's room for every
/// digit; this is for a column that has to sit beside four others.
fn format_tokens_compact(n: i64) -> String {
    let n = n.max(0);
    if n < 1_000 {
        return n.to_string();
    }
    if n < 1_000_000 {
        return format!("{:.1}k", n as f64 / 1_000.0);
    }
    format!("{:.1}M", n as f64 / 1_000_000.0)
}

/// One cell of the row grid: the text truncated to fit and padded out to
/// `width`, always leaving a two-space gutter so a full-width value can't
/// run into the column after it.
fn column(text: &str, width: usize) -> String {
    let text = truncate(text, width.saturating_sub(2));
    format!("{text:<width$}")
}

/// At most `max` characters, the ellipsis included — it replaces the last
/// character kept rather than being added past the limit, so a caller that
/// sized a column or the room left on a line gets something that fits it.
fn truncate(text: &str, max: usize) -> String {
    let flat: String = text
        .chars()
        .map(|c| if c == '\n' { ' ' } else { c })
        .collect();
    if flat.chars().count() <= max {
        return flat;
    }
    match max {
        0 => String::new(),
        _ => format!("{}…", flat.chars().take(max - 1).collect::<String>()),
    }
}

/// Coarse "how long ago", enough to tell yesterday's work from this
/// morning's without a date library.
fn relative_time(timestamp: i64) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let seconds = now.saturating_sub(timestamp);
    match seconds {
        s if s < 60 => "just now".to_string(),
        s if s < 3600 => format!("{}m ago", s / 60),
        s if s < 86_400 => format!("{}h ago", s / 3600),
        s if s < 2_592_000 => format!("{}d ago", s / 86_400),
        s => format!("{}mo ago", s / 2_592_000),
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn the_kind_column_says_what_it_can_do_not_what_it_is_stored_as() {
        // The stored kind is still "chat"/"agent_chat" — a cache of what
        // the tools add up to, for rows read without being opened — but
        // there are no modes any more, so what a reader sees is whether it
        // has tools.
        let without = row("00000001", crate::store::KIND_CHAT, "t");
        let with = row("00000002", KIND_AGENT_CHAT, "t");
        assert_eq!(
            column(mode_label(without.is_agentic()), KIND_WIDTH).trim_end(),
            "💬"
        );
        assert_eq!(
            column(mode_label(with.is_agentic()), KIND_WIDTH).trim_end(),
            "🔨"
        );
    }

    #[test]
    fn truncate_never_exceeds_its_limit() {
        // The ellipsis takes the place of a kept character. Returning max + 1
        // used to be enough to wrap a row whose preview was sized to the
        // space left on the line.
        assert_eq!(truncate("abcdefgh", 4).chars().count(), 4);
        assert_eq!(truncate("abcdefgh", 4), "abc…");
        assert_eq!(truncate("abcd", 4), "abcd");
        assert_eq!(truncate("abc", 4), "abc");
        assert_eq!(truncate("abc", 1), "…");
        assert_eq!(truncate("abc", 0), "");
    }

    #[test]
    fn truncate_counts_characters_not_bytes() {
        assert_eq!(truncate("ünïcödé test", 6).chars().count(), 6);
    }

    #[test]
    fn truncate_flattens_newlines() {
        assert_eq!(truncate("two\nlines", 20), "two lines");
    }

    #[test]
    fn column_pads_short_values_and_keeps_a_gutter() {
        assert_eq!(column("chat", 7), "chat   ");
        // A value wider than its column still can't touch the next one.
        let cell = column("~/code/some/very/long/path", 24);
        assert_eq!(cell.chars().count(), 24);
        assert!(cell.ends_with("  "), "no gutter left in {cell:?}");
    }
    #[test]
    fn a_notice_replaces_the_key_hints() {
        // A session that can't be opened has to say so where the user still
        // has other choices, rather than taking the whole TUI down.
        let mut picker = picker_of(vec![]);
        assert!(picker.notice.is_none());
        picker.notice = Some("abc123 was started in /gone, which no longer exists.".to_string());
        assert!(picker.notice.as_deref().unwrap().contains("/gone"));
    }

    use super::*;

    const HERE: &str = "/work/project";

    fn row(id: &str, kind: &str, title: &str) -> SessionRow {
        row_in(id, kind, title, None)
    }

    fn row_in(id: &str, kind: &str, title: &str, dir: Option<&str>) -> SessionRow {
        SessionRow {
            id: format!("{id}-0000-0000-0000-000000000000"),
            kind: kind.to_string(),
            title: title.to_string(),
            updated_at: 0,
            working_dir: dir.map(str::to_string),
            last: None,
            activity: None,
            activity_detail: None,
            heartbeat: None,
            total_tokens: 0,
        }
    }

    /// Marks a row as being run right now: an activity is only believed
    /// while a process is checking in to back it up.
    fn running(mut row: SessionRow, activity: Activity) -> SessionRow {
        row.activity = Some(activity);
        row.heartbeat = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0),
        );
        row
    }

    fn with_last(mut row: SessionRow, role: &str, tool_calls: bool, preview: &str) -> SessionRow {
        row.last = Some(LastMessage {
            role: role.to_string(),
            has_tool_calls: tool_calls,
            preview: preview.to_string(),
        });
        row
    }

    /// A picker over `rows`, as seen from `HERE`.
    fn picker_of(rows: Vec<SessionRow>) -> Picker {
        Picker::launch(rows, Some(HERE))
    }

    #[test]
    fn the_cursor_steps_over_headers() {
        let mut picker = picker_of(vec![
            row_in("00000001", "chat", "here", Some(HERE)),
            row_in("00000002", "chat", "away", Some("/elsewhere")),
        ]);
        // new → (skip header) → here → (skip header) → away
        assert!(matches!(picker.activate(), Some(Activation::NewSession)));
        picker.move_down();
        assert_eq!(picker.selected_session().unwrap().title, "here");
        picker.move_down();
        assert_eq!(picker.selected_session().unwrap().title, "away");
        // ...and back up the same way, never landing on a label.
        picker.move_up();
        assert_eq!(picker.selected_session().unwrap().title, "here");
        picker.move_up();
        assert!(matches!(picker.activate(), Some(Activation::NewSession)));
    }

    #[test]
    fn the_last_message_says_where_a_session_stopped() {
        let base = || row_in("00000001", "chat", "t", Some(HERE));

        // Never used.
        assert_eq!(base().last_state(), LastState::New);
        // Ran to completion.
        assert_eq!(
            with_last(base(), "assistant", false, "here you go").last_state(),
            LastState::Replied
        );
        // Sent, nothing came back — running elsewhere, or it failed. The
        // messages table can't tell those apart, and this doesn't pretend to.
        assert_eq!(
            with_last(base(), "user", false, "do the thing").last_state(),
            LastState::NoReply
        );
        // Stopped part-way: a tool result with no answer after it...
        assert_eq!(
            with_last(base(), "tool", false, "{}").last_state(),
            LastState::Interrupted
        );
        // ...or a tool call that never ran.
        assert_eq!(
            with_last(base(), "assistant", true, "read_file").last_state(),
            LastState::Interrupted
        );
    }

    #[test]
    fn a_running_process_speaks_over_the_stored_messages() {
        // The whole reason for the column: from the messages alone, a
        // request in flight and a turn that failed are the same row.
        let sent = with_last(
            row_in("00000001", "chat", "t", Some(HERE)),
            "user",
            false,
            "do the thing",
        );
        assert_eq!(sent.last_state(), LastState::NoReply);

        let in_flight = running(sent.clone(), Activity::Working);
        assert_eq!(in_flight.last_state(), LastState::Working);

        let asking = running(sent.clone(), Activity::AwaitingApproval);
        assert_eq!(asking.last_state(), LastState::AwaitingApproval);

        let mut broken = sent.clone();
        broken.activity = Some(Activity::Failed);
        assert_eq!(broken.last_state(), LastState::Failed);
    }

    #[test]
    fn a_row_left_working_by_a_dead_process_stops_saying_so() {
        // A detached run killed mid-turn never clears its activity. Without
        // a heartbeat to back it, the claim is ignored and the row reports
        // what its messages actually show.
        let sent = with_last(
            row_in("00000001", "chat", "t", Some(HERE)),
            "user",
            false,
            "do the thing",
        );

        let mut abandoned = sent.clone();
        abandoned.activity = Some(Activity::Working);
        abandoned.heartbeat = None;
        assert_eq!(abandoned.last_state(), LastState::NoReply);

        abandoned.heartbeat = Some(0); // the epoch: as stale as it gets
        assert_eq!(abandoned.last_state(), LastState::NoReply);
    }

    #[test]
    fn with_nothing_said_the_messages_still_speak() {
        // No process running it, so the column is empty and the row falls
        // back to what was stored — which is the common case.
        let answered = with_last(
            row_in("00000001", "chat", "t", Some(HERE)),
            "assistant",
            false,
            "here you go",
        );
        assert_eq!(answered.activity, None);
        assert_eq!(answered.last_state(), LastState::Replied);
    }

    #[test]
    fn the_working_badge_animates_and_fills_its_column() {
        // It draws the conversation's own busy frame, so a running session
        // looks the same from the list as from inside it — and every frame
        // still has to be exactly the width the other badges are padded to,
        // or this row sits a column off from the rest.
        let frames: Vec<String> = (0..24)
            .map(|tick| state_badge(LastState::Working, false, tick).0)
            .collect();

        let mut distinct = frames.clone();
        distinct.sort();
        distinct.dedup();
        assert!(
            distinct.len() > 8,
            "it should look random, not cycle through a handful: {distinct:?}"
        );
        for frame in &frames {
            assert_eq!(
                frame.chars().count(),
                BADGE_WIDTH,
                "{frame:?} does not fill the badge column"
            );
        }
    }

    #[test]
    fn a_still_list_has_nothing_to_animate() {
        // The picker redraws itself while this is true, so it must only be
        // true when something is actually running.
        let idle = picker_of(vec![row_in("00000001", "chat", "one", Some(HERE))]);
        assert!(!idle.has_working_session());

        let busy_row = running(
            row_in("00000002", "chat", "two", Some(HERE)),
            Activity::Working,
        );
        assert!(picker_of(vec![busy_row]).has_working_session());
    }

    /// Renders the naming screen off-screen, the same way.
    fn naming_to_string(input: &str, id: &str, width: u16, height: u16) -> String {
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(width, height)).unwrap();
        terminal
            .draw(|frame| draw_naming(frame, input, id))
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn the_first_row_spawns_a_clanker() {
        let out = picker_to_string(&picker_of(vec![]), 60, 10);
        assert!(out.contains("Spawn clanker"), "{out}");
    }

    #[test]
    fn naming_shows_the_mark_the_clanker_will_carry() {
        // The mark is hashed from the id, so this is the one moment it can
        // be seen before it is committed to — and the screen has to actually
        // show it, or rerolling is rolling blind.
        let id = "4f2a91b2-3c1d-4e8a-9f02-7b6c5d4e3a21";
        let out = naming_to_string("Parser work", id, 60, 10);
        assert!(out.contains(&identicon(id).0), "{out}");
        assert!(out.contains("Parser work"), "{out}");
        assert!(out.contains("Tab"), "the reroll key is offered: {out}");

        // A different id is a different mark, which is the whole point.
        let other = "0000ffff-3c1d-4e8a-9f02-7b6c5d4e3a21";
        assert_ne!(identicon(id).0, identicon(other).0);
    }

    /// Renders the launch screen off-screen so the list can be asserted on.
    fn picker_to_string(picker: &Picker, width: u16, height: u16) -> String {
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(width, height)).unwrap();
        terminal
            .draw(|frame| draw(frame, picker, "TITLE", Some(HERE), false, "hint", 0))
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn many_sessions(count: usize) -> Picker {
        let rows: Vec<SessionRow> = (0..count)
            .map(|i| {
                row_in(
                    &format!("{i:08}"),
                    "chat",
                    &format!("session {i:02}"),
                    Some(HERE),
                )
            })
            .collect();
        Picker::launch(rows, Some(HERE))
    }

    #[test]
    fn a_list_taller_than_the_screen_scrolls_to_the_selection() {
        // It used to render every row into a fixed area, so anything past
        // the bottom was simply not drawn — and the cursor could be moved
        // onto a row nobody could see.
        let mut picker = many_sessions(40);
        let shown = |p: &Picker| -> Vec<usize> {
            let out = picker_to_string(p, 80, 12);
            (0..40)
                .filter(|i| out.contains(&format!("session {i:02}")))
                .collect()
        };

        let top = shown(&picker);
        assert!(top.contains(&0), "the first session should start visible");

        // Walk the cursor to the last row.
        for _ in 0..60 {
            picker.move_down();
        }
        let bottom = shown(&picker);
        assert!(
            bottom.contains(&39),
            "the last session should be reachable: {bottom:?}"
        );
        assert!(
            !bottom.contains(&0),
            "the list should have scrolled, not grown: {bottom:?}"
        );
    }

    #[test]
    fn the_rule_says_how_much_is_off_screen() {
        // Without this the extra sessions are invisible *and* unannounced,
        // which is the half of the bug that makes it a trap rather than an
        // inconvenience.
        let picker = many_sessions(40);
        let out = picker_to_string(&picker, 80, 12);
        assert!(out.contains("more below"), "{out}");

        // A list that fits says nothing.
        let small = picker_to_string(&many_sessions(2), 80, 24);
        assert!(!small.contains("more below"), "{small}");
        assert!(!small.contains("more above"), "{small}");
    }

    #[test]
    fn a_session_another_process_holds_is_marked_as_such() {
        // Held but idle looks exactly like free otherwise: the state comes
        // off the messages, so it reads `✓ replied` and invites an open that
        // will be refused.
        let idle = with_last(
            row_in("00000001", "chat", "t", Some(HERE)),
            "assistant",
            false,
            "done",
        );
        assert_eq!(idle.last_state(), LastState::Replied);
        assert!(!idle.is_held());

        let held = running(idle.clone(), Activity::Working);
        // `running` stamps a heartbeat; drop the activity so the state falls
        // back to the messages while the claim stays live.
        let mut held = held;
        held.activity = None;
        assert!(held.is_held());
        assert_eq!(held.last_state(), LastState::Replied);

        assert_ne!(
            state_badge(idle.last_state(), idle.is_held(), 0).0,
            state_badge(held.last_state(), held.is_held(), 0).0,
            "held and free sessions must not draw the same badge"
        );
    }

    #[test]
    fn a_working_session_keeps_its_spinner_rather_than_the_held_mark() {
        // `Working` is already gated on a live heartbeat, so it only happens
        // while held — the spinner says someone is there, and replacing it
        // would lose the more specific fact.
        assert_eq!(
            state_badge(LastState::Working, true, 3).0,
            state_badge(LastState::Working, false, 3).0
        );
    }

    #[test]
    fn every_session_lands_in_one_list_whatever_its_directory() {
        // The list used to be split into "In this directory" and
        // "Elsewhere", which reads well right up until you are looking for a
        // session and have to work out which half it is in.
        let picker = picker_of(vec![
            row_in("00000001", "chat", "here one", Some(HERE)),
            row_in("00000002", "chat", "away", Some("/elsewhere")),
            row_in("00000003", "chat", "here two", Some(HERE)),
            row_in("00000004", "chat", "unrecorded", None),
        ]);

        let shape: Vec<String> = picker
            .items
            .iter()
            .map(|item| match item {
                LaunchItem::NewSession => "new".to_string(),
                LaunchItem::Resume(row) => row.title.clone(),
            })
            .collect();
        assert_eq!(
            shape,
            vec!["new", "here one", "away", "here two", "unrecorded"],
            "the order rows arrived in should survive, ungrouped"
        );
    }

    #[test]
    fn the_directory_column_says_dot_for_the_one_you_are_in() {
        let picker = picker_of(vec![
            row_in("00000001", "chat", "here", Some(HERE)),
            row_in("00000002", "chat", "away", Some("/elsewhere")),
        ]);
        let out = picker_to_string(&picker, 110, 12);

        let row = |title: &str| {
            out.lines()
                .find(|l| l.contains(title))
                .unwrap_or_else(|| panic!("{title} missing from:\n{out}"))
        };
        assert!(
            row("here").contains(" . "),
            "the current directory should read as `.`: {:?}",
            row("here")
        );
        assert!(
            row("away").contains("/elsewhere"),
            "another directory should be spelled out: {:?}",
            row("away")
        );
    }

    #[test]
    fn a_directory_under_home_is_shown_relative_to_it() {
        let Some(home) = home::home_dir() else {
            return;
        };
        let under = format!("{}/projects/thing", home.display());
        let picker = picker_of(vec![row_in("00000001", "chat", "mine", Some(&under))]);
        let out = picker_to_string(&picker, 110, 12);
        let row = out.lines().find(|l| l.contains("mine")).unwrap();

        assert!(row.contains("~/projects"), "{row:?}");
        assert!(
            !row.contains(&home.display().to_string()),
            "home should be abbreviated, not spelled out: {row:?}"
        );
    }

    #[test]
    fn every_row_can_hold_the_cursor_now_that_none_are_labels() {
        let mut picker = picker_of(vec![
            row_in("00000001", "chat", "one", Some(HERE)),
            row_in("00000002", "chat", "two", Some("/elsewhere")),
        ]);
        for _ in 0..picker.items.len() + 2 {
            assert!(
                picker.activate().is_some(),
                "row {} is not selectable",
                picker.selected
            );
            picker.move_down();
        }
    }

    #[test]
    fn a_pending_approval_displaces_the_conversation_preview() {
        // What the session last said matters far less than what it is stuck
        // asking — that's the row someone watching this list has to act on.
        let row = with_last(
            row_in("00000001", "agent_chat", "t", Some(HERE)),
            "user",
            false,
            "please tidy the build",
        );
        assert_eq!(row.preview().as_deref(), Some("please tidy the build"));

        let mut row = running(row, Activity::AwaitingApproval);
        row.activity_detail = Some("run_terminal_command: rm -rf build".to_string());
        assert_eq!(
            row.preview().as_deref(),
            Some("needs approval — run_terminal_command: rm -rf build")
        );
        assert_eq!(row.last_state(), LastState::AwaitingApproval);
    }

    #[test]
    fn an_approval_with_no_detail_still_falls_back_to_the_messages() {
        // Written by an older version, or a tool whose arguments say nothing
        // worth naming.
        let mut row = with_last(
            row_in("00000001", "agent_chat", "t", Some(HERE)),
            "user",
            false,
            "please tidy the build",
        );
        row.activity = Some(Activity::AwaitingApproval);
        assert_eq!(row.preview().as_deref(), Some("please tidy the build"));
    }

    #[test]
    fn every_badge_fills_the_column_and_avoids_ambiguous_glyphs() {
        // What kept the column ragged: a terminal may draw an
        // East-Asian-Ambiguous character two cells wide while drawing the
        // rest one. Measured in cells rather than characters, because the
        // busy badge is deliberately two braille cells and the rest are one
        // padded to match — char count would pass a glyph that draws wide.
        use unicode_width::UnicodeWidthStr;

        const AMBIGUOUS: [char; 7] = ['●', '◐', '○', '•', '…', '→', '⊙'];
        let checked = [
            state_badge(LastState::New, false, 0).0,
            state_badge(LastState::Replied, false, 0).0,
            state_badge(LastState::NoReply, false, 0).0,
            state_badge(LastState::Interrupted, false, 0).0,
            state_badge(LastState::AwaitingApproval, false, 0).0,
            state_badge(LastState::Failed, false, 0).0,
            // Several ticks of the animation, since each is a fresh pair.
            state_badge(LastState::Working, false, 0).0,
            state_badge(LastState::Working, false, 7).0,
            state_badge(LastState::Working, false, 31).0,
            // And the held marker, which is not a state.
            state_badge(LastState::Replied, true, 0).0,
        ];

        for badge in checked {
            assert_eq!(
                UnicodeWidthStr::width(badge.as_str()),
                BADGE_WIDTH,
                "{badge:?} does not fill the badge column"
            );
            for ch in badge.chars() {
                assert!(
                    !AMBIGUOUS.contains(&ch),
                    "{ch:?} is drawn double-width by some terminals"
                );
            }
        }
    }

    #[test]
    fn every_state_has_its_own_glyph() {
        // On a terminal without colour the glyph is all there is, so no two
        // states may share one.
        let glyphs: Vec<String> = [
            LastState::New,
            LastState::Replied,
            LastState::NoReply,
            LastState::Interrupted,
            LastState::Working,
            LastState::AwaitingApproval,
            LastState::Failed,
        ]
        .into_iter()
        .map(|state| state_badge(state, false, 0).0)
        .collect();
        let mut unique = glyphs.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), glyphs.len(), "{glyphs:?}");
    }

    #[test]
    fn refreshing_updates_state_without_disturbing_the_list() {
        let mut picker = picker_of(vec![
            row_in("00000001", "chat", "one", Some(HERE)),
            row_in("00000002", "chat", "two", Some(HERE)),
        ]);
        picker.move_down();
        let selected_before = picker.selected;
        let order_before: Vec<String> = picker
            .items
            .iter()
            .filter_map(|item| match item {
                LaunchItem::Resume(row) => Some(row.id.clone()),
                _ => None,
            })
            .collect();

        // Both rows come back — a refresh rebuilds the list, so anything
        // left out of `latest` is a session that was deleted.
        let moved_on = with_last(
            row_in("00000002", "chat", "two", Some(HERE)),
            "assistant",
            false,
            "all done",
        );
        assert!(picker.refresh(
            vec![row_in("00000001", "chat", "one", Some(HERE)), moved_on],
            Some(HERE)
        ));

        let states: Vec<LastState> = picker
            .items
            .iter()
            .filter_map(|item| match item {
                LaunchItem::Resume(row) => Some(row.last_state()),
                _ => None,
            })
            .collect();
        assert_eq!(states, vec![LastState::New, LastState::Replied]);

        // Nothing moved under the cursor.
        assert_eq!(picker.selected, selected_before);
        let order_after: Vec<String> = picker
            .items
            .iter()
            .filter_map(|item| match item {
                LaunchItem::Resume(row) => Some(row.id.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(order_before, order_after);
    }

    #[test]
    fn refreshing_an_unchanged_list_reports_nothing() {
        // The caller redraws on `true`, so a quiet list must stay quiet
        // rather than repainting every couple of seconds.
        let mut picker = picker_of(vec![row_in("00000001", "chat", "one", Some(HERE))]);
        let same = row_in("00000001", "chat", "one", Some(HERE));
        assert!(!picker.refresh(vec![same], Some(HERE)));
    }

    #[test]
    fn refreshing_picks_up_sessions_added_and_removed_elsewhere() {
        // The list is worth leaving open only if it keeps up with what other
        // processes are doing to it.
        let mut picker = picker_of(vec![
            row_in("00000001", "chat", "one", Some(HERE)),
            row_in("00000002", "chat", "two", Some(HERE)),
        ]);
        picker.move_down();
        picker.move_down();
        let watched = picker.selected_session().unwrap().id.clone();

        // "one" was deleted elsewhere, "three" was started elsewhere.
        assert!(picker.refresh(
            vec![
                row_in("00000002", "chat", "two", Some(HERE)),
                row_in("00000003", "chat", "three", Some(HERE)),
            ],
            Some(HERE)
        ));

        let titles: Vec<String> = picker
            .items
            .iter()
            .filter_map(|item| match item {
                LaunchItem::Resume(row) => Some(row.title.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(titles, vec!["two", "three"]);

        // The cursor followed the session it was on, not the row number it
        // happened to sit at.
        assert_eq!(picker.selected_session().unwrap().id, watched);
    }

    #[test]
    fn losing_the_selected_session_leaves_the_cursor_somewhere_sensible() {
        let mut picker = picker_of(vec![
            row_in("00000001", "chat", "one", Some(HERE)),
            row_in("00000002", "chat", "two", Some(HERE)),
        ]);
        picker.move_down();
        picker.move_down();

        // The one being watched is deleted from under it.
        picker.refresh(
            vec![row_in("00000001", "chat", "one", Some(HERE))],
            Some(HERE),
        );

        assert!(picker.selected < picker.items.len());
        assert!(picker.activate().is_some(), "never left resting on a label");
    }

    #[test]
    fn a_missing_directory_offers_to_repoint() {
        // The row can't open where it says it lives, and resuming here is a
        // real answer — so it's a question, not a refusal.
        let mut picker = picker_of(vec![row_in("00000001", "chat", "t", Some("/gone"))]);
        picker.move_down();
        let row = picker.selected_session().unwrap().clone();

        picker.begin_repoint(row, "/gone".to_string());
        assert!(picker.confirming_repoint.is_some());
        // Declining leaves the session pointed where it was.
        assert!(picker.resolve_repoint(false).is_none());
        assert!(picker.confirming_repoint.is_none());

        let row = picker.selected_session().unwrap().clone();
        picker.begin_repoint(row, "/gone".to_string());
        assert!(matches!(
            picker.resolve_repoint(true),
            Some(Activation::Repoint(_))
        ));
    }

    #[test]
    fn selection_stays_in_bounds() {
        let mut picker = picker_of(vec![row("abcd1234", "chat", "t")]);
        picker.move_up();
        assert_eq!(picker.selected, 0);
        for _ in 0..50 {
            picker.move_down();
        }
        assert_eq!(picker.selected, picker.items.len() - 1);
    }

    #[test]
    fn activating_each_row_type() {
        let mut picker = picker_of(vec![row("abcd1234", "agent_chat", "t")]);
        assert!(matches!(picker.activate(), Some(Activation::NewSession)));
        picker.move_down();
        match picker.activate() {
            Some(Activation::Resume(r)) => assert!(r.is_agentic()),
            other => panic!("expected Resume, got {other:?}"),
        }
    }

    #[test]
    fn with_no_sessions_there_is_still_something_to_start() {
        // A fresh install: one row, and it's the useful one.
        let picker = picker_of(vec![]);
        assert!(matches!(picker.activate(), Some(Activation::NewSession)));
        assert!(picker.selected_session().is_none());
    }

    #[test]
    fn delete_requires_confirmation() {
        let mut picker = picker_of(vec![row("abcd1234", "chat", "t")]);
        picker.move_down();
        picker.begin_delete();
        assert!(picker.confirming_delete.is_some());
        // Declining leaves the session alone.
        assert!(picker.resolve_delete(false).is_none());
        assert!(picker.confirming_delete.is_none());

        picker.begin_delete();
        assert!(matches!(
            picker.resolve_delete(true),
            Some(Activation::Delete(_))
        ));
    }

    #[test]
    fn delete_is_a_no_op_on_a_non_session_row() {
        // "New chat" is selected; there's nothing to delete.
        let mut picker = picker_of(vec![row("abcd1234", "chat", "t")]);
        picker.begin_delete();
        assert!(picker.confirming_delete.is_none());
    }

    #[test]
    fn rename_is_prefilled_and_editable() {
        let mut picker = picker_of(vec![row("abcd1234", "chat", "old title")]);
        picker.move_down();
        picker.begin_rename();
        let (row, input) = picker.renaming.as_ref().unwrap();
        assert_eq!(row.id, picker.selected_session().unwrap().id);
        assert_eq!(input, "old title");

        picker.rename_backspace();
        picker.rename_backspace();
        for c in "le".chars() {
            picker.rename_insert_char(c);
        }
        assert_eq!(picker.renaming.as_ref().unwrap().1, "old title");
    }

    #[test]
    fn confirming_a_rename_updates_the_row_and_clears_the_state() {
        let mut picker = picker_of(vec![row("abcd1234", "chat", "old title")]);
        picker.move_down();
        let id = picker.selected_session().unwrap().id.clone();
        picker.begin_rename();
        for c in " v2".chars() {
            picker.rename_insert_char(c);
        }

        let confirmed = picker.confirm_rename();
        assert_eq!(confirmed, Some((id.clone(), "old title v2".to_string())));
        assert!(picker.renaming.is_none());

        picker.apply_rename(&id, "old title v2".to_string());
        assert_eq!(picker.selected_session().unwrap().title, "old title v2");
    }

    #[test]
    fn a_blank_rename_is_rejected_and_stays_open() {
        let mut picker = picker_of(vec![row("abcd1234", "chat", "t")]);
        picker.move_down();
        picker.begin_rename();
        picker.rename_backspace();
        assert_eq!(picker.confirm_rename(), None);
        // Still editable, not silently dropped.
        assert!(picker.renaming.is_some());
    }

    #[test]
    fn cancelling_a_rename_leaves_the_title_alone() {
        let mut picker = picker_of(vec![row("abcd1234", "chat", "old title")]);
        picker.move_down();
        picker.begin_rename();
        picker.rename_insert_char('!');
        picker.cancel_rename();
        assert!(picker.renaming.is_none());
        assert_eq!(picker.selected_session().unwrap().title, "old title");
    }

    #[test]
    fn rename_is_a_no_op_on_a_non_session_row() {
        let mut picker = picker_of(vec![row("abcd1234", "chat", "t")]);
        picker.begin_rename();
        assert!(picker.renaming.is_none());
    }

    #[test]
    fn removing_a_row_keeps_the_selection_valid() {
        let rows = vec![
            row("aaaaaaaa", "chat", "one"),
            row("bbbbbbbb", "chat", "two"),
        ];
        let mut picker = picker_of(rows);
        picker.move_down();
        let id = picker.selected_session().unwrap().id.clone();
        picker.remove_session(&id);

        let remaining: Vec<&SessionRow> = picker
            .items
            .iter()
            .filter_map(|item| match item {
                LaunchItem::Resume(row) => Some(row),
                _ => None,
            })
            .collect();
        assert_eq!(remaining.len(), 1);
        assert!(picker.selected < picker.items.len());
        // And on a row that can actually be chosen.
        assert!(picker.activate().is_some());
    }

    #[test]
    fn relative_time_buckets() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        assert_eq!(relative_time(now), "just now");
        assert_eq!(relative_time(now - 120), "2m ago");
        assert_eq!(relative_time(now - 7200), "2h ago");
        assert_eq!(relative_time(now - 172_800), "2d ago");
    }
}
