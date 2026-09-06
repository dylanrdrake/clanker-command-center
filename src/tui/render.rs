//! Drawing the TUI. Pure presentation over [`App`] — no state changes here.

use super::app::{App, CommandHint, ModelBrowser, ShellState, ToolStatus, TranscriptItem};
use crate::config::ToolAccess;
use crate::ui::{json_fields, summarize, tool_call_fields, ApprovalRequest};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use unicode_width::UnicodeWidthStr;

/// One frame of the busy animation: two braille cells of scattered dots.
///
/// Replaces the ten-frame dot circle, which read as one thing rotating at a
/// fixed rate — a clock, and a clock that always ticks at the same speed
/// says nothing beyond "still going". Two cells of noise say the same thing
/// with more of the screen and no implied progress.
///
/// Pseudo-random rather than random: the frame is a function of the tick, so
/// every front end drawing the same tick draws the same thing, and a test
/// can assert on it. Hashed rather than indexed in order, because stepping
/// through the patterns in sequence is itself a visible pattern.
///
/// Uses the identicon's alphabet — three to seven dots — for the same reason
/// it does: nothing blank, which reads as a rendering fault, and nothing
/// solid, which reads as stalled.
pub(crate) fn busy_frame(tick: usize) -> String {
    let hash = (tick as u64)
        .wrapping_add(1)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let cell = |bits: u64| {
        let pattern = MARK_PATTERNS[bits as usize % MARK_PATTERNS.len()];
        char::from_u32(0x2800 + u32::from(pattern)).unwrap_or('?')
    };
    format!("{}{}", cell(hash >> 8), cell(hash >> 40))
}

/// Most rows the message box will grow to before it scrolls internally,
/// so a long paste can't squeeze the conversation off the screen.
const MAX_INPUT_ROWS: u16 = 10;

/// Braille patterns of three to seven dots. Nothing emptier (a mark with a
/// blank half reads as a rendering fault) and nothing solid (every solid
/// half looks like every other one).
const MARK_PATTERNS: [u8; 218] = {
    let mut patterns = [0u8; 218];
    let (mut bits, mut found) = (0usize, 0usize);
    while bits < 256 {
        let dots = (bits as u8).count_ones();
        if dots >= 3 && dots <= 7 {
            patterns[found] = bits as u8;
            found += 1;
        }
        bits += 1;
    }
    patterns
};

/// Mid-tone 256-colour indices: saturated enough to tell apart, dark enough
/// to read on a light terminal and light enough to read on a dark one. The
/// mark draws on whatever background the row has, so the palette can't lean
/// on one being behind it. Deliberately clear of the colours that carry
/// meaning here — the cyan and yellow of the mode column, and the badge
/// colours for state.
const IDENTICON_FG: [u8; 12] = [33, 30, 70, 61, 96, 100, 130, 133, 136, 166, 172, 25];

/// A path with the home directory shown as `~`, so the column stays
/// readable on the long paths most projects have.
pub(super) fn home_relative(dir: &str) -> String {
    let Some(home) = home::home_dir() else {
        return dir.to_string();
    };
    let home = home.display().to_string();
    match dir.strip_prefix(&home) {
        Some(rest) => format!("~{rest}"),
        None => dir.to_string(),
    }
}

/// FNV-1a, 64-bit. The mark has to be identical in every process that draws
/// this session, so it can come from nothing but the id, and the wider hash
/// leaves room to slice a half, a half and a colour out of independent bits.
fn identicon_hash(seed: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in seed.bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// The two braille cells of a session's mark, without a colour.
///
/// Split out for the CLI, which paints with `colored` rather than ratatui and
/// so cannot take the `Style` its sibling returns. Braille specifically
/// because every pattern in that block is East Asian Width Neutral — the `●`
/// the CLI drew before is Ambiguous, and some terminals give it two cells,
/// which shifts every wrapped line under it out of line with the gutter.
pub(crate) fn identicon_mark(seed: &str) -> String {
    let hash = identicon_hash(seed);
    let half = |bits: u64| {
        let pattern = MARK_PATTERNS[bits as usize % MARK_PATTERNS.len()];
        char::from_u32(0x2800 + u32::from(pattern)).unwrap_or('?')
    };
    format!("{}{}", half(hash), half(hash >> 12))
}

/// A mark: a square of braille dots derived from `seed`, and the same every
/// time for the same seed.
///
/// It identifies nothing you can type — the id column was removed because
/// nothing in the picker needs it. This is for recognition: a list that
/// refreshes under you, with rows moving as sessions are touched, is easier
/// to keep your place in when the row you were watching carries the same
/// mark it had a moment ago.
pub(super) fn identicon(seed: &str) -> (String, Style) {
    let fg = IDENTICON_FG[(identicon_hash(seed) >> 24) as usize % IDENTICON_FG.len()];
    (identicon_mark(seed), Style::new().fg(Color::Indexed(fg)))
}

pub fn draw(frame: &mut Frame, app: &App, cache: &mut TranscriptCache, tick: usize) {
    // The message box grows with what's been typed into it, and with nothing
    // else. An approval used to take the box over — borrowing the input as
    // its answer buffer — so a decision arriving mid-sentence displaced what
    // you were writing and answering it consumed the draft. It has its own
    // box now.
    let content_rows = input_lines(&app.input, frame.area().width.saturating_sub(2)).len() as u16;
    // Messages waiting to be sent get a box of their own above the prompt,
    // and no space at all when nothing is waiting.
    let pending_rows = pending_height(app.pending.len());
    // Nearest the transcript, above anything to do with what you're typing:
    // it is the thing waiting on you, not the thing you are writing.
    let approval_rows = match &app.pending_approval {
        Some(request) => approval_height(request, frame.area().width),
        None => 0,
    };
    let shell_rows = match &app.pending_shell {
        Some(shell) => shell_height(shell, frame.area().width),
        None => 0,
    };
    let browser_rows = match &app.model_browser {
        Some(browser) => browser_height(browser),
        None => 0,
    };
    // One row, directly above the box, while a command is being typed into
    // it. Nothing is reserved when there is nothing to say, so the prompt
    // does not sit a row higher for the whole session.
    let hint = app.command_hint();
    let hint_rows = u16::from(hint.is_some());

    let input_rows = content_rows
        .clamp(1, MAX_INPUT_ROWS)
        // Never take so much that the conversation has nowhere to go. The
        // reserve covers the title row, the rule under it, the input box's
        // own two border rows, the settings/key-binding rows below it, and
        // whatever the pending and approval boxes are using.
        .min(
            frame
                .area()
                .height
                .saturating_sub(
                    7 + pending_rows + approval_rows + shell_rows + browser_rows + hint_rows,
                )
                .max(1),
        );

    let areas = Layout::vertical([
        Constraint::Length(1),              // session title
        Constraint::Length(1),              // rule
        Constraint::Min(1),                 // chat history
        Constraint::Length(approval_rows),  // a tool waiting on a decision
        Constraint::Length(shell_rows),     // a $ command, running or waiting
        Constraint::Length(browser_rows),   // the /models browser, if open
        Constraint::Length(pending_rows),   // messages waiting, if any
        Constraint::Length(hint_rows),      // what a half-typed command could be
        Constraint::Length(input_rows + 2), // message prompt, bordered, plus its borders
        Constraint::Length(1),              // settings: ask/agent, model, effort, temp, verbose
        Constraint::Length(1),              // key bindings
    ])
    .split(frame.area());

    draw_title(frame, areas[0], app);
    draw_rule(frame, areas[1], None);
    let scrolled = draw_transcript(frame, areas[2], app, cache);
    if let Some(request) = &app.pending_approval {
        draw_approval(frame, areas[3], request);
    }
    if let Some(shell) = &app.pending_shell {
        draw_shell(frame, areas[4], shell, tick);
    }
    if let Some(browser) = &app.model_browser {
        draw_model_browser(frame, areas[5], browser, &app.model, tick);
    }
    if pending_rows > 0 {
        draw_pending(frame, areas[6], app);
    }
    if let Some(hint) = &hint {
        draw_hint(frame, areas[7], hint);
    }
    draw_input(frame, areas[8], app, scrolled);
    draw_settings(frame, areas[9], app, tick);
    // Tab is only offered while there is something for it to complete.
    let completing = matches!(hint, Some(CommandHint::Matches { .. }));
    draw_keybindings(frame, areas[10], app, completing);
}

/// The blank row between the transcript and the approval box, matching the
/// pending box's.
const APPROVAL_GAP: u16 = 1;

/// Past this the box stops growing and scrolls instead, so one enormous
/// argument can't swallow the conversation behind it.
const MAX_APPROVAL_ROWS: u16 = 12;

/// How tall the approval box is, measured against the width it will actually
/// be drawn at.
///
/// Sized from the *wrapped* height rather than the number of lines: a long
/// `content` or `command` value wraps, and sizing by line count alone left
/// the tail of it below the bottom edge, where it silently disappeared —
/// which is the half of the request you most need to read before allowing
/// it.
fn approval_height(request: &ApprovalRequest, width: u16) -> u16 {
    let inner = width.saturating_sub(2).max(1);
    let wrapped = Paragraph::new(Text::from(approval_lines(request)))
        .wrap(Wrap { trim: false })
        .line_count(inner) as u16;
    wrapped.min(MAX_APPROVAL_ROWS) + 2 + APPROVAL_GAP
}

/// At most this many waiting messages are listed; the rest are summarised.
const PENDING_ROWS: usize = 5;

/// How tall the pending box is for `waiting` messages — its rows, its two
/// borders, and a blank row above it so it doesn't sit flush against the last
/// line of the conversation. Zero when nothing is waiting, so the box
/// disappears rather than sitting there empty.
fn pending_height(waiting: usize) -> u16 {
    if waiting == 0 {
        return 0;
    }
    let listed = waiting.min(PENDING_ROWS);
    let overflow = usize::from(waiting > PENDING_ROWS);
    (listed + overflow) as u16 + 2 + PENDING_GAP
}

/// The blank row between the transcript and the box.
const PENDING_GAP: u16 = 1;

/// How tall the `$` box is: the command line, its output once there is any,
/// and the borders. Capped the same way the approval box is.
fn shell_height(shell: &ShellState, width: u16) -> u16 {
    let inner = width.saturating_sub(2).max(1);
    let output_rows = match shell {
        // The command line is all there is until it finishes.
        ShellState::Running { .. } => 0,
        ShellState::Finished { output, .. } => Paragraph::new(output.trim_end().to_string())
            .wrap(Wrap { trim: false })
            .line_count(inner) as u16,
    };
    (1 + output_rows).clamp(1, MAX_APPROVAL_ROWS) + 2 + APPROVAL_GAP
}

/// A command the user ran with `$`: the command itself on the first line,
/// spinning while it runs, then its output beneath — the shape a terminal
/// would show it in. The border carries only the keys that decide whether
/// the model sees it, since that is the part you act on.
///
/// Green against the approval box's yellow. The prompt marker is already
/// green, so green reads as "yours" where yellow reads as "the agent is
/// asking" — which matters because both boxes can be on screen at once.
fn draw_shell(frame: &mut Frame, area: Rect, shell: &ShellState, tick: usize) {
    let green = Style::new().green();
    let (title, mut lines) = match shell {
        ShellState::Running { command } => (
            String::new(),
            vec![Line::from(vec![
                Span::styled("$ ", green.bold()),
                Span::styled(command.clone(), green),
                Span::styled(format!(" {}", busy_frame(tick)), Style::new().green()),
            ])],
        ),
        ShellState::Finished {
            command,
            output,
            exit_code,
        } => {
            // Beside the command, not in the border: the border says what
            // you can do about the output, and how the command ended belongs
            // with the command.
            let mut first = vec![
                Span::styled("$ ", green.bold()),
                Span::styled(command.clone(), green),
            ];
            if *exit_code != 0 {
                first.push(Span::styled(
                    format!("  exit {exit_code}"),
                    Style::new().red(),
                ));
            }
            let mut lines = vec![Line::from(first)];
            match output.trim_end() {
                "" => lines.push(Line::from(Span::styled(
                    "(no output)",
                    Style::new().dark_gray().italic(),
                ))),
                text => lines.extend(text.lines().map(|line| Line::raw(line.to_string()))),
            }
            (
                " Ctrl-S send with next message · Ctrl-D discard ".to_string(),
                lines,
            )
        }
    };
    lines.shrink_to_fit();

    let box_area = Rect {
        y: area.y + APPROVAL_GAP,
        height: area.height.saturating_sub(APPROVAL_GAP),
        ..area
    };
    let mut block = Block::default().borders(Borders::ALL).border_style(green);
    if !title.is_empty() {
        block = block.title(Span::styled(title, green.bold()));
    }
    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .wrap(Wrap { trim: false })
            .block(block),
        box_area,
    );
}

/// The messages typed while a turn is running, in the order they will be
/// taken.
///
/// The title says where they are headed, which differs by whether there are
/// tools: with them the next iteration takes the message into the turn
/// already running, without them it waits for the turn to finish. It reads
/// what the clanker has *now*, so `/tools` mid-turn makes the title disagree
/// with where a message will actually go until that turn ends — rare, and
/// cheaper to live with than threading the turn's own captured answer out to
/// the front end.
fn draw_pending(frame: &mut Frame, area: Rect, app: &App) {
    let title = if app.agentic() {
        " joining this turn "
    } else {
        " next turn "
    };
    let width = area.width.saturating_sub(2).max(1) as usize;

    let mut lines: Vec<Line> = app
        .pending
        .iter()
        .take(PENDING_ROWS)
        .map(|text| Line::from(Span::styled(clip(text, width), Style::new().dark_gray())))
        .collect();
    if app.pending.len() > PENDING_ROWS {
        lines.push(Line::from(Span::styled(
            format!("+{} more", app.pending.len() - PENDING_ROWS),
            Style::new().dark_gray().italic(),
        )));
    }

    // The gap `pending_height` reserved is simply left unpainted.
    let box_area = Rect {
        y: area.y + PENDING_GAP,
        height: area.height.saturating_sub(PENDING_GAP),
        ..area
    };
    frame.render_widget(
        Paragraph::new(Text::from(lines)).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::new().dark_gray())
                .title(Span::styled(title, Style::new().dark_gray().italic())),
        ),
        box_area,
    );
}

/// One row's worth of a message: newlines flattened so a multi-line message
/// stays one entry, and clipped with an ellipsis to fit.
fn clip(text: &str, width: usize) -> String {
    let flat: String = text
        .chars()
        .map(|c| if c == '\n' { ' ' } else { c })
        .collect();
    if flat.chars().count() <= width {
        return flat;
    }
    match width {
        0 => String::new(),
        _ => format!("{}…", flat.chars().take(width - 1).collect::<String>()),
    }
}

/// The session's title, plain — no border, no "clank -" prefix.
fn draw_title(frame: &mut Frame, area: Rect, app: &App) {
    // The clanker's own mark, first: the same one its row carries on the
    // launch screen and its replies carry in the gutter below. Names repeat
    // and directories are shared, so the mark is the thing that says *which*
    // of them you are looking at — and it should say so at the top, not only
    // once a reply has arrived.
    let (mark, mark_style) = identicon(&app.session_id);
    let mut spans = vec![
        Span::styled(mark, mark_style),
        Span::styled(format!(" {}", app.title), Style::new().bold()),
    ];
    // Where this session runs, beside its name: it is the directory the
    // agent's tools act in and the sandbox bounds, so it is worth being able
    // to see without asking for `/status`.
    if let Some(dir) = &app.working_dir {
        spans.push(Span::styled(
            format!("  {}", home_relative(dir)),
            Style::new().dark_gray(),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);

    // Right-aligned on the same row, the way `draw_rule`'s scroll hint
    // overlays its divider — this is the one place a clanker's running
    // total is always in view, not just on `/status`'s one-time printout.
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!("🪙 {} ", crate::ui::format_tokens(app.total_tokens)),
            Style::new().yellow(),
        )))
        .right_aligned(),
        area,
    );
}

/// A subtle full-width divider, standing in for the borders the screen used
/// to have. `hint`, when given, is overlaid right-aligned on top of it —
/// used for the "scrolled" notice, the way a bordered box would have shown
/// it in its own title.
pub(super) fn draw_rule(frame: &mut Frame, area: Rect, hint: Option<&str>) {
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "─".repeat(area.width as usize),
            Style::new().dark_gray(),
        ))),
        area,
    );
    if let Some(hint) = hint {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(hint, Style::new().yellow())).right_aligned()),
            area,
        );
    }
}

/// Draws the transcript and reports whether the view is scrolled away from
/// the newest content, so the top status row can flag it.
fn draw_transcript(frame: &mut Frame, area: Rect, app: &App, cache: &mut TranscriptCache) -> bool {
    // User/Assistant text is wrapped by hand (see `wrap_styled`) rather
    // than left to ratatui's `Wrap`, specifically so a row broken by
    // wrapping — not just one broken by a literal newline — still lines up
    // under the gutter instead of resuming at column 0.
    let content_width = area.width.saturating_sub(2).max(1) as usize;

    // No border, and the title now rides in its own row above `area`
    // (see `draw_title`), so the whole of `area` is free for content.
    let visible = area.height;

    // Every block's rows, rendered once and kept. Building them fresh every
    // frame meant re-parsing the markdown of every reply in the conversation
    // to arrive at rows identical to the ones just thrown away.
    cache.begin();
    let keys: Vec<u64> = app
        .transcript
        .iter()
        .map(|item| cache.ensure(item, app, area.width, content_width))
        .collect();
    // Swept before anything borrows the rows, since the sweep mutates.
    cache.end();

    // Pointers to those rows, in transcript order. This is the only pass
    // that is still proportional to the length of the conversation, and it
    // copies nothing. Two identical messages share a cache entry and so
    // appear here twice, which is right — they draw the same.
    let mut rows: Vec<&Line<'static>> = Vec::new();
    for key in keys {
        rows.extend(cache.get(key));
    }

    // Every block appends a trailing blank; drop it so the newest message
    // sits flush against the bottom rather than floating above a gap.
    while matches!(rows.last(), Some(line) if line.spans.iter().all(|s| s.content.is_empty())) {
        rows.pop();
    }

    // Rows are already wrapped to `content_width` by hand, so one row is one
    // line on screen and the height is just the count — no second pass by
    // ratatui to measure it, and none to draw it. That is also why the
    // paragraph below sets no `Wrap`: anything it had to fold would be a row
    // that was mis-measured on the way in, and `no_rendered_row_is_wider_
    // than_the_pane` is what stops that happening.
    let total = rows.len() as u16;

    // Grow the conversation up from the input box instead of down from the
    // title, the way a chat reads: until there's enough to fill the pane,
    // the newest message still sits at the bottom. Done by standing the
    // paragraph on the pane's bottom edge rather than padding above it.
    let target = if total < visible {
        Rect {
            y: area.y + (visible - total),
            height: total,
            ..area
        }
    } else {
        area
    };

    let max_offset = total.saturating_sub(visible);
    // scroll_back counts up from the bottom; 0 pins to the newest content.
    let offset = max_offset.saturating_sub(app.scroll_back.min(max_offset));

    // Only what will be on screen is copied. Everything above is a pointer
    // that never gets dereferenced, which is what keeps a frame the same
    // price in a long conversation as in a short one.
    let first = offset as usize;
    let last = (first + target.height as usize).min(rows.len());
    let window: Vec<Line<'static>> = rows[first..last].iter().map(|row| (*row).clone()).collect();

    frame.render_widget(Paragraph::new(Text::from(window)), target);

    // Reported so the rule below can carry the "scrolled" notice instead of
    // a border's bottom title, since there's no border to carry it anymore.
    !app.is_pinned_to_bottom() && max_offset > 0
}

/// Rendered rows for each transcript block, so a block that hasn't changed
/// is not built again.
///
/// A finished reply renders to exactly the same rows on every frame, but
/// producing them means parsing its markdown and wrapping it — together
/// about two thirds of the cost of a frame, spent to reproduce rows that
/// were just discarded. Redrawing at ten frames a second made that the
/// dominant cost of a long session.
///
/// Keyed by content rather than by position: a block's index shifts when
/// thinking slots in ahead of the reply it led to, and two identical
/// messages can share one entry because they render identically anyway.
#[derive(Default)]
pub(super) struct TranscriptCache {
    rows: HashMap<u64, Vec<Line<'static>>>,
    /// Keys used by the frame being drawn, so everything else can be
    /// dropped at the end of it. Without this, every streamed delta would
    /// leave its own superseded entry behind for the rest of the session.
    live: HashSet<u64>,
}

impl TranscriptCache {
    fn begin(&mut self) {
        self.live.clear();
    }

    /// Renders `item` if it isn't already held, and returns its key.
    ///
    /// Split from [`Self::get`] so a frame can populate the cache in one
    /// pass and then borrow every block's rows in a second — the rows are
    /// referenced into the visible window rather than copied out of it, and
    /// that can't happen while the cache is still being mutated.
    fn ensure(
        &mut self,
        item: &TranscriptItem,
        app: &App,
        area_width: u16,
        content_width: usize,
    ) -> u64 {
        let key = item_key(item, app, area_width, content_width);
        self.live.insert(key);
        self.rows
            .entry(key)
            .or_insert_with(|| render_item(item, app, area_width, content_width));
        key
    }

    /// The rows held for `key`, which [`Self::ensure`] has already put there.
    fn get(&self, key: u64) -> &[Line<'static>] {
        self.rows.get(&key).map_or(&[], Vec::as_slice)
    }

    fn end(&mut self) {
        // Borrowed out so `retain`'s closure isn't reaching through `self`
        // while `rows` is mutably borrowed.
        let live = &self.live;
        self.rows.retain(|key, _| live.contains(key));
    }
}

/// Everything that decides how `item` draws, hashed into a cache key.
///
/// Deliberately conservative: the layout inputs shared by every block
/// (`area_width`, `content_width`) and the flags that change how some of
/// them draw (`verbose`, `highlight`, `session_id`) go into every key, so a
/// resize or a `/verbose` invalidates the whole cache rather than needing
/// each variant to declare what it happens to read.
///
/// Two different blocks colliding on 64 bits would draw the wrong rows. At
/// a few thousand blocks the odds are on the order of 1e-12, and the cache
/// holds only one session's transcript.
fn item_key(item: &TranscriptItem, app: &App, area_width: u16, content_width: usize) -> u64 {
    let mut hasher = DefaultHasher::new();
    area_width.hash(&mut hasher);
    content_width.hash(&mut hasher);
    app.verbose.hash(&mut hasher);
    app.highlight.hash(&mut hasher);
    app.session_id.hash(&mut hasher);
    // The discriminant is hashed by hand alongside the fields: two variants
    // that happen to hold the same string must not land on the same key.
    match item {
        TranscriptItem::User(text) => (0u8, text).hash(&mut hasher),
        TranscriptItem::Assistant {
            text,
            streaming,
            label,
        } => (1u8, text, streaming, label).hash(&mut hasher),
        TranscriptItem::Thinking(text) => (2u8, text).hash(&mut hasher),
        TranscriptItem::ToolCall {
            name,
            arguments,
            status,
        } => {
            (3u8, name, arguments).hash(&mut hasher);
            match status {
                ToolStatus::AwaitingApproval => 0u8.hash(&mut hasher),
                ToolStatus::Running => 1u8.hash(&mut hasher),
                ToolStatus::Denied => 2u8.hash(&mut hasher),
                ToolStatus::Done { result } => (3u8, result).hash(&mut hasher),
            }
        }
        TranscriptItem::Shell {
            command,
            output,
            exit_code,
            sent,
        } => (4u8, command, output, exit_code, sent).hash(&mut hasher),
        TranscriptItem::Error(message) => (5u8, message).hash(&mut hasher),
        TranscriptItem::Notice(message) => (6u8, message).hash(&mut hasher),
        TranscriptItem::SessionStatus(rows) => (7u8, rows).hash(&mut hasher),
        TranscriptItem::Help(rows) => (9u8, rows).hash(&mut hasher),
        TranscriptItem::ToolStatus { access, changed } => {
            8u8.hash(&mut hasher);
            for (name, _, state) in access.rows() {
                (name, state.label()).hash(&mut hasher);
            }
            changed.hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Rows the `/models` box needs: its borders, a hint line, and up to
/// [`MAX_BROWSER_ROWS`] models.
///
/// Sized to its contents like every other box above the prompt, so a filter
/// that narrows to two models does not leave eight blank rows behind.
fn browser_height(browser: &ModelBrowser) -> u16 {
    let rows = match browser {
        ModelBrowser::Ready { .. } => (browser.matches().len() as u16).clamp(1, MAX_BROWSER_ROWS),
        // "Fetching…" or the reason it failed.
        _ => 1,
    };
    // Two borders and the hint line, plus the margin below.
    rows + 3 + BROWSER_TOP_MARGIN
}

/// Blank rows left above the `/models` box, so it does not butt straight
/// against the last line of the conversation — it is tall and it appears
/// without warning, and flush against a reply it reads as part of the
/// transcript rather than over it.
///
/// A constant because the height reserved and the offset drawn at have to
/// agree: reserving without offsetting squeezes the box, offsetting without
/// reserving clips it, and neither is obvious from looking at one of them.
const BROWSER_TOP_MARGIN: u16 = 1;

/// Most models shown at once. Past this the list scrolls under the cursor —
/// an endpoint can offer four hundred, and the box is sharing the screen
/// with the conversation it was opened from.
const MAX_BROWSER_ROWS: u16 = 8;

fn draw_model_browser(
    frame: &mut Frame,
    area: Rect,
    browser: &ModelBrowser,
    current: &str,
    tick: usize,
) {
    // The margin `browser_height` reserved. Given up here rather than in the
    // layout so the box and its spacing stay one thing to reason about.
    let area = Rect {
        y: area.y + BROWSER_TOP_MARGIN,
        height: area.height.saturating_sub(BROWSER_TOP_MARGIN),
        ..area
    };

    let matches = browser.matches();
    let (title, hint) = match browser {
        ModelBrowser::Loading => (" models ".to_string(), String::new()),
        ModelBrowser::Failed(_) => (" models ".to_string(), String::new()),
        ModelBrowser::Ready { all, .. } => (
            format!(" models  {} of {} ", matches.len(), all.len()),
            " ↑↓ move · Enter set · Esc cancel ".to_string(),
        ),
    };

    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::new().dark_gray())
        .title(Span::styled(title, Style::new().cyan()));
    if !hint.is_empty() {
        block = block
            .title_bottom(Span::styled(hint, Style::new().dark_gray()).into_right_aligned_line());
    }
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let lines: Vec<Line> = match browser {
        ModelBrowser::Loading => vec![Line::from(Span::styled(
            format!("{} Fetching models…", busy_frame(tick)),
            Style::new().yellow(),
        ))],
        ModelBrowser::Failed(why) => vec![Line::from(Span::styled(
            format!("✗ {why}"),
            Style::new().red(),
        ))],
        ModelBrowser::Ready { selected, .. } if matches.is_empty() => {
            let _ = selected;
            vec![Line::from(Span::styled(
                "nothing matches",
                Style::new().dark_gray().italic(),
            ))]
        }
        ModelBrowser::Ready { selected, .. } => {
            // Windowed on the cursor the same way the launch screen's list
            // is, so the selection is always on screen without the box
            // growing to fit four hundred rows.
            let visible = inner.height as usize;
            let offset = selected
                .saturating_sub(visible / 2)
                .min(matches.len().saturating_sub(visible));
            matches
                .iter()
                .enumerate()
                .skip(offset)
                .take(visible)
                .map(|(index, name)| {
                    let picked = index == *selected;
                    let mut spans = vec![
                        Span::styled(if picked { "❯ " } else { "  " }, Style::new().cyan().bold()),
                        Span::styled(
                            (*name).to_string(),
                            if picked {
                                Style::new().bold()
                            } else {
                                Style::new()
                            },
                        ),
                    ];
                    // Marked rather than reordered: the one you are on now
                    // is easier to find if the list does not shuffle.
                    if *name == current {
                        spans.push(Span::styled("  current", Style::new().dark_gray().italic()));
                    }
                    Line::from(spans)
                })
                .collect()
        }
    };

    frame.render_widget(Paragraph::new(Text::from(lines)), inner);
}

/// A titled block of label/value rows, as `/status` and `/help` both show.
///
/// The labels are padded to a common width so the values line up in a
/// column, and each value goes through `push_labeled`, which hangs a long
/// one under its label rather than letting it run off the pane — a working
/// directory for `/status`, a wordy description for `/help`.
fn push_row_block(
    lines: &mut Vec<Line<'static>>,
    heading: &str,
    rows: &[(String, String)],
    content_width: usize,
) {
    lines.push(Line::from(vec![
        Span::styled("— ", Style::new().dark_gray().italic()),
        Span::styled(heading.to_string(), Style::new().dark_gray().italic()),
    ]));
    let width = rows
        .iter()
        .map(|(label, _)| display_width(label))
        .max()
        .unwrap_or(0);
    for (label, value) in rows {
        push_labeled(
            lines,
            format!("      {label:<width$}  "),
            value.clone(),
            content_width,
        );
    }
}

/// One transcript block's rows: everything the cache stores against an item.
///
/// Split out of `draw_transcript` so a block can be rendered on its own and
/// kept — nothing here reads anything but `item` and the parts of `app`
/// that [`item_key`] hashes, which is what makes caching it sound.
fn render_item(
    item: &TranscriptItem,
    app: &App,
    area_width: u16,
    content_width: usize,
) -> Vec<Line<'static>> {
    let mut lines: Vec<Line> = Vec::new();
    match item {
        TranscriptItem::User(text) => {
            let start = lines.len();
            push_block(
                &mut lines,
                Span::styled("❯ ", Style::new().green().bold()),
                text,
                None,
                content_width,
                Style::new(),
            );
            // A band behind what you said, so your own messages are
            // findable while scrolling back through a long turn without
            // having to read them. Padded to the full width first: a
            // background only paints the cells a line actually covers,
            // so an unpadded one would end raggedly at the text.
            highlight_rows(&mut lines[start..], area_width as usize, app.highlight);
            lines.push(Line::raw(""));
        }
        TranscriptItem::Shell {
            command,
            output,
            exit_code,
            sent,
        } => {
            // Green like the user's own prompt marker: this is something
            // you ran, not something the agent did.
            push_rendered(
                &mut lines,
                Span::styled("$ ", Style::new().green().bold()),
                vec![Line::from(vec![
                    Span::styled(command.clone(), Style::new().green()),
                    // Sending adds the output to the conversation without
                    // starting a turn, so that it arrives together with the
                    // question you were going to ask about it. Said outright
                    // because "sent" alone reads as "sent a message", and
                    // then the absence of a reply looks like a fault.
                    Span::styled(
                        match (sent, exit_code) {
                            (true, 0) => "  sent — goes with your next message".to_string(),
                            (true, code) => {
                                format!("  exit {code} · sent — goes with your next message")
                            }
                            (false, 0) => "  not sent".to_string(),
                            (false, code) => format!("  exit {code} · not sent"),
                        },
                        Style::new().dark_gray().italic(),
                    ),
                ])],
                None,
                content_width,
                "  ",
            );
            if !output.trim().is_empty() {
                push_block(
                    &mut lines,
                    Span::raw("  "),
                    output.trim_end(),
                    None,
                    content_width,
                    // Output the model never saw is dimmed to the colour
                    // every other "this is not in play" value wears, so
                    // scrolling back you can tell at a glance which command
                    // results the conversation is actually working from
                    // without reading the label on each one.
                    if *sent {
                        Style::new()
                    } else {
                        Style::new().dark_gray()
                    },
                );
            }
            lines.push(Line::raw(""));
        }
        TranscriptItem::Assistant {
            text, streaming, ..
        } => {
            // The session's own mark, the same one its row carries on the
            // picker — so the conversation you are reading is tied to the
            // one you picked, rather than the gutter saying nothing.
            //
            // Braille is also the only block where every pattern is East
            // Asian Width Neutral. The `●` this replaces is Ambiguous —
            // some terminals draw it two cells wide, which shifted the
            // whole gutter against the wrapped lines beneath it.
            let (glyph, mark_style) = identicon(&app.session_id);
            let prefix = Span::styled(format!("{glyph} "), mark_style);
            let cursor = streaming.then(|| Span::styled("▌", Style::new().cyan()));
            if *streaming {
                // Mid-stream the text is usually mid-construct — an
                // unclosed fence or a half-written list — so render it
                // plainly and let the finished message reformat once.
                push_block(
                    &mut lines,
                    prefix,
                    text,
                    cursor,
                    content_width - 1,
                    Style::new(),
                );
            } else {
                push_rendered(
                    &mut lines,
                    prefix,
                    markdown_lines(text),
                    cursor,
                    // The same treatment as the tool gutter: this one is
                    // 3 columns (two braille cells and a space) where
                    // `content_width` assumes 2, so wrap one narrower or
                    // a full-width row overflows and gets wrapped again,
                    // out from under the gutter.
                    content_width.saturating_sub(1),
                    SQUARE_CONTINUATION,
                );
            }
            lines.push(Line::raw(""));
        }
        TranscriptItem::ToolCall {
            name,
            arguments,
            status,
        } => {
            // No trailing marker while running — the spinner-driven
            // "working" state in the settings row already says so; a
            // static triangle here didn't add anything.
            let marker: Option<(&str, Style)> = match status {
                ToolStatus::AwaitingApproval => Some(("?", Style::new().yellow())),
                ToolStatus::Running => None,
                ToolStatus::Denied => Some(("✗", Style::new().red())),
                ToolStatus::Done { .. } => Some(("✓", Style::new().green())),
            };
            let mut header = vec![Span::styled(name.clone(), Style::new().bold())];
            // The file or command a call is acting on identifies it well
            // enough to show even without -v; the rest of its arguments
            // (and its result) are the detail that gates behind verbose.
            if let Some(detail) = crate::ui::primary_argument(arguments) {
                header.push(Span::styled(
                    format!("  {}", summarize(&detail, 60)),
                    Style::new().dark_gray(),
                ));
            }
            push_rendered(
                &mut lines,
                // The gutter marks the row as a tool call; the status
                // (still color-coded) rides at the end instead, as a
                // `trailing` marker — same mechanism the streaming
                // cursor uses, so it always lands on the last wrapped
                // row rather than getting buried mid-wrap.
                Span::styled("🔨 ", Style::new().magenta()),
                vec![Line::from(header)],
                marker.map(|(m, style)| Span::styled(format!(" {m}"), style)),
                // `content_width` assumes a 2-column prefix, one less
                // than "🔨 "'s actual 3 (🔨 is double-width) — wrap one
                // column narrower so the prefixed row still fits, rather
                // than overflowing the terminal width and getting
                // wrapped a second time, out from under the gutter.
                content_width.saturating_sub(1),
                "   ",
            );
            if app.verbose {
                for (key, shown) in tool_call_fields(name, arguments) {
                    push_labeled(&mut lines, format!("     {key}  "), shown, content_width);
                }
                if let ToolStatus::Done { result } = status {
                    for (key, shown) in json_fields(result) {
                        push_labeled(&mut lines, format!("     {key}  "), shown, content_width);
                    }
                }
            }
            lines.push(Line::raw(""));
        }
        TranscriptItem::Thinking(text) => {
            if app.verbose {
                let thought: Vec<Line<'static>> = text
                    .lines()
                    .map(|line| {
                        Line::from(Span::styled(
                            line.to_string(),
                            Style::new().dark_gray().italic(),
                        ))
                    })
                    .collect();
                push_rendered(
                    &mut lines,
                    Span::styled("💭 ", Style::new().dark_gray()),
                    thought,
                    None,
                    // 💭 is double-width, so wrap a column narrower —
                    // same adjustment the tool-call gutter makes.
                    content_width.saturating_sub(1),
                    "   ",
                );
                lines.push(Line::raw(""));
            }
        }
        TranscriptItem::Error(message) => {
            push_rendered(
                &mut lines,
                Span::styled("✗ ", Style::new().red().bold()),
                vec![Line::from(Span::styled(
                    message.clone(),
                    Style::new().red(),
                ))],
                None,
                content_width,
                "  ",
            );
            lines.push(Line::raw(""));
        }
        TranscriptItem::Notice(message) => {
            push_rendered(
                &mut lines,
                Span::styled("— ", Style::new().dark_gray().italic()),
                vec![Line::from(Span::styled(
                    message.clone(),
                    Style::new().dark_gray().italic(),
                ))],
                None,
                content_width,
                "  ",
            );
            lines.push(Line::raw(""));
        }
        TranscriptItem::SessionStatus(rows) => {
            push_row_block(&mut lines, "Clanker:", rows, content_width);
            lines.push(Line::raw(""));
        }
        TranscriptItem::Help(rows) => {
            push_row_block(&mut lines, "Commands:", rows, content_width);
            lines.push(Line::raw(""));
        }
        TranscriptItem::ToolStatus { access, changed } => {
            lines.push(Line::from(vec![
                Span::styled("— ", Style::new().dark_gray().italic()),
                Span::styled(
                    format!("Tools {}:", if *changed { "set to" } else { "are" }),
                    Style::new().dark_gray().italic(),
                ),
            ]));
            for (name, _, state) in access.rows() {
                // Coloured by how much is being taken on trust: green stops
                // and asks, yellow runs unwatched, grey is not there at all.
                let (mark, style) = match state {
                    ToolAccess::Ask => ("✓", Style::new().green()),
                    ToolAccess::Allow => ("!", Style::new().yellow()),
                    ToolAccess::Never => ("✗", Style::new().dark_gray()),
                };
                lines.push(Line::from(vec![
                    Span::styled(format!("      {name:<22}"), Style::new().dark_gray()),
                    Span::styled(format!("{mark} {}", state.label()), style),
                ]));
            }
            lines.push(Line::raw(""));
        }
    }
    lines
}

/// `scrolled` carries the "scrolled — End to follow" notice onto the box's
/// top border, right-aligned — the same edge the transcript's own border
/// used to show it on, back when it had one.
/// The row above the input box while a command is being typed: the names
/// still matching, or the form of the one already named.
///
/// Indented to sit under the box's first column rather than its border, so
/// a name here lines up with the same name being typed below it.
fn draw_hint(frame: &mut Frame, area: Rect, hint: &CommandHint) {
    let spans = match hint {
        CommandHint::Syntax(syntax) => vec![
            Span::raw(HINT_INDENT),
            Span::styled(*syntax, Style::new().dark_gray()),
        ],
        CommandHint::Matches { names, active } => match_spans(names, *active, area.width as usize),
    };
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// Lines the row up with the text inside the bordered box below it.
const HINT_INDENT: &str = " ";

/// As many matching names as the row holds, the one Tab has landed on picked
/// out in the colour it now wears in the box below.
///
/// A list too long for the row says how many it left off: with nothing
/// there, a bare `/` would look like the session had eight commands. The
/// window slides to keep the stepped-to name visible, since a mark you
/// cannot see is the same as no mark.
fn match_spans(names: &[&'static str], active: Option<usize>, width: usize) -> Vec<Span<'static>> {
    const GAP: &str = "  ";
    // What `count` names starting at `from` take up, gaps included.
    let width_of = |from: usize, count: usize| {
        HINT_INDENT.len()
            + names[from..from + count]
                .iter()
                .map(|n| n.len())
                .sum::<usize>()
            + GAP.len() * count.saturating_sub(1)
    };
    // How many fit if the row starts at `from`. Always at least one, since
    // a row with nothing on it says less than a clipped name does.
    let fits = |from: usize| {
        (1..=names.len() - from)
            .take_while(|count| width_of(from, *count) <= width)
            .last()
            .unwrap_or(1)
    };

    let start = match active {
        Some(i) if i >= fits(0) => i,
        _ => 0,
    };
    let mut shown = fits(start);
    // The "+N" costs a name where there isn't room for both.
    if start + shown < names.len() {
        let left = names.len() - (start + shown);
        let marker = GAP.len() + 1 + left.to_string().len();
        if width_of(start, shown) + marker > width && shown > 1 {
            shown -= 1;
        }
    }

    let mut spans = vec![Span::raw(HINT_INDENT)];
    for (offset, name) in names[start..start + shown].iter().enumerate() {
        if offset > 0 {
            spans.push(Span::raw(GAP));
        }
        let style = match active {
            Some(i) if i == start + offset => COMMAND,
            _ => Style::new().dark_gray(),
        };
        spans.push(Span::styled(*name, style));
    }
    let left = names.len() - (start + shown);
    if left > 0 {
        spans.push(Span::styled(
            format!("{GAP}+{left}"),
            Style::new().dark_gray(),
        ));
    }
    spans
}

fn draw_input(frame: &mut Frame, area: Rect, app: &App, scrolled: bool) {
    // While the browser is open the prompt is the filter. Nothing is lost by
    // borrowing it: `/models` was submitted to get here, so the draft was
    // already taken — which is exactly what is *not* true of an approval,
    // and why that one has a box of its own instead.
    if let Some(ModelBrowser::Ready { filter, .. }) = &app.model_browser {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::new().cyan());
        let inner = block.inner(area);
        frame.render_widget(block, area);
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("filter: ", Style::new().dark_gray()),
                Span::raw(filter.clone()),
            ])),
            inner,
        );
        frame.set_cursor_position((
            inner.x + "filter: ".len() as u16 + display_width(filter) as u16,
            inner.y,
        ));
        return;
    }

    let width = area.width.saturating_sub(2).max(1);
    let rows = styled_input_lines(&app.input, width);
    let (cursor_row, cursor_col) = input_cursor(&app.input, app.cursor, width);

    // Once the text is taller than the box, follow the cursor rather than
    // pinning to the top, so what you're typing stays on screen.
    let visible = area.height.saturating_sub(2).max(1);
    let scroll = (cursor_row + 1).saturating_sub(visible);

    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::new().dark_gray());
    if scrolled {
        block = block.title(
            Line::from(Span::styled(
                " scrolled — End to follow ",
                Style::new().yellow(),
            ))
            .right_aligned(),
        );
    }

    // Wrapped by hand rather than by `Wrap`, so the cursor position below is
    // computed against exactly the rows being drawn.
    let paragraph = Paragraph::new(Text::from(rows))
        .block(block)
        .scroll((scroll, 0));
    frame.render_widget(paragraph, area);

    frame.set_cursor_position((
        area.x + 1 + cursor_col,
        area.y + 1 + cursor_row.saturating_sub(scroll),
    ));
}

/// Splits the input into the rows it occupies: on explicit newlines, and
/// hard-wrapped at `width`. Hard rather than word wrapping so that a cursor
/// position can be computed exactly against what's drawn.
fn input_lines(input: &str, width: u16) -> Vec<String> {
    input_rows(input, width)
        .into_iter()
        .map(|(_, row)| row)
        .collect()
}

/// The same rows, each paired with the `char` offset into `input` it starts
/// at, so a span of the input can be found again once it has been wrapped.
///
/// The offsets count the newlines the split consumes, and are what let
/// [`styled_input_lines`] colour a range without wrapping the text a second
/// time — two wrapping implementations that disagreed would put the colour
/// somewhere the caret isn't.
fn input_rows(input: &str, width: u16) -> Vec<(usize, String)> {
    let width = width.max(1) as usize;
    let mut rows = Vec::new();
    let mut offset = 0usize;
    for (i, segment) in input.split('\n').enumerate() {
        // Every split after the first ate a newline to get here.
        if i > 0 {
            offset += 1;
        }
        let chars: Vec<char> = segment.chars().collect();
        if chars.is_empty() {
            rows.push((offset, String::new()));
            continue;
        }
        for chunk in chars.chunks(width) {
            rows.push((offset, chunk.iter().collect()));
            offset += chunk.len();
        }
        // A segment filling the last row exactly puts the caret on the next.
        if chars.len().is_multiple_of(width) {
            rows.push((offset, String::new()));
        }
    }
    rows
}

/// What a recognized command looks like in the input box, before it is sent.
/// Cyan is the colour the rest of the TUI already spends on "this is live" —
/// the picker's cursor, the model browser's border, the streaming caret.
const COMMAND: Style = Style::new().fg(Color::Cyan);

/// The input box's rows, with the leading `/command` picked out when what is
/// typed names one. Feedback while typing rather than after sending: a
/// message that starts with a slash but isn't a command — a path, a
/// misspelling — simply stays the colour of ordinary text.
fn styled_input_lines(input: &str, width: u16) -> Vec<Line<'static>> {
    let Some(span) = crate::ui::command_span(input) else {
        return input_lines(input, width)
            .into_iter()
            .map(Line::from)
            .collect();
    };
    input_rows(input, width)
        .into_iter()
        .map(|(offset, row)| command_row(offset, row, &span))
        .collect()
}

/// One wrapped row, split at whatever part of `span` falls inside it.
/// Everything is clamped to the row, so a row wholly before or after the
/// span comes back as it went in.
fn command_row(offset: usize, row: String, span: &std::ops::Range<usize>) -> Line<'static> {
    let chars: Vec<char> = row.chars().collect();
    let start = span.start.saturating_sub(offset).min(chars.len());
    let end = span.end.saturating_sub(offset).min(chars.len());
    if start >= end {
        return Line::from(row);
    }
    let take = |range: std::ops::Range<usize>| chars[range].iter().collect::<String>();
    let mut spans = Vec::new();
    if start > 0 {
        spans.push(Span::raw(take(0..start)));
    }
    spans.push(Span::styled(take(start..end), COMMAND));
    if end < chars.len() {
        spans.push(Span::raw(take(end..chars.len())));
    }
    Line::from(spans)
}

/// Where the caret sits, in the same rows [`input_lines`] produces.
fn input_cursor(input: &str, cursor: usize, width: u16) -> (u16, u16) {
    let width = width.max(1) as usize;
    let mut row = 0usize;
    let mut col = 0usize;
    for ch in input[..cursor.min(input.len())].chars() {
        if ch == '\n' {
            row += 1;
            col = 0;
        } else {
            col += 1;
            if col == width {
                row += 1;
                col = 0;
            }
        }
    }
    (row as u16, col as u16)
}

/// A muted-to-intense gradient for `low`/`medium`/`high`; unset (following
/// the configured default) stays the same dark_gray every other "no
/// override" field uses.
fn effort_style(effort_level: Option<&str>) -> Style {
    match effort_level {
        Some("low") => Style::new().cyan(),
        Some("medium") => Style::new().yellow(),
        Some("high") => Style::new().red(),
        _ => Style::new().dark_gray(),
    }
}

/// A cool-to-hot gradient matching the word itself; unset stays the same
/// dark_gray every other "no override" field uses. Red at the top, the same
/// as `effort_style`'s highest band, so the two settings read alike.
fn temperature_style(temperature: Option<f32>) -> Style {
    const ORANGE: Color = Color::Rgb(255, 140, 0);
    match temperature {
        None => Style::new().dark_gray(),
        Some(t) if t < 0.5 => Style::new().cyan(),
        Some(t) if t < 1.0 => Style::new().yellow(),
        Some(t) if t < 1.5 => Style::new().fg(ORANGE),
        Some(_) => Style::new().red(),
    }
}

/// Every controllable setting in one row below the message prompt: ready/
/// busy, whether it has tools, model, effort, temperature and verbose —
/// everything `/model`, `/tools`, `/effort`, `/temperature`, `/verbose` can
/// change. What is waiting to be sent is its own box above the prompt, since
/// a count can't say which message is about to land.
fn draw_settings(frame: &mut Frame, area: Rect, app: &App, tick: usize) {
    let mut spans = Vec::new();

    if app.pending_approval.is_some() {
        // A turn stopped at a gate is still `busy`, and animating it as
        // working says the opposite of what is true: nothing is moving, and
        // the thing that would move it is you. Marked with the same `?` the
        // launch screen puts on the row, so the two agree on what waiting
        // looks like.
        spans.push(Span::styled(" ? waiting ", Style::new().yellow().bold()));
    } else if app.busy {
        spans.push(Span::styled(
            format!(" {} working ", busy_frame(tick)),
            Style::new().yellow(),
        ));
    } else {
        spans.push(Span::styled(" ready ", Style::new().green()));
    }

    spans.push(Span::styled(
        format!("· {} ", app.model),
        Style::new().dark_gray(),
    ));
    spans.push(Span::styled(
        format!("· {} ", crate::store::mode_label(app.agentic())),
        if app.agentic() {
            Style::new().yellow()
        } else {
            Style::new().cyan()
        },
    ));
    let effort_label = app.effort_level.as_deref().unwrap_or("default");
    spans.push(Span::styled(
        format!("· 🧠 {effort_label} "),
        effort_style(app.effort_level.as_deref()),
    ));
    let temp_label = app
        .temperature
        .map(|n| n.to_string())
        .unwrap_or_else(|| "default".to_string());
    spans.push(Span::styled(
        format!("· 🌡 {temp_label} "),
        temperature_style(app.temperature),
    ));
    spans.push(Span::styled(
        format!("· {} ", if app.verbose { "verbose" } else { "quiet" }),
        if app.verbose {
            Style::new().yellow()
        } else {
            Style::new().dark_gray()
        },
    ));

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// The keybinding hints, on their own row at the very bottom.
/// Dimmed a step further than the rest of the muted (dark_gray) text
/// elsewhere, so it recedes into the background rather than competing with
/// the settings row right above it.
fn draw_keybindings(frame: &mut Frame, area: Rect, app: &App, completing: bool) {
    // A shade darker than the plain `dark_gray()` used elsewhere — `.dim()`
    // alone isn't reliable across terminals (some ignore the SGR faint
    // attribute entirely), so the color itself carries the extra dimness.
    const KEYBIND_GRAY: Color = Color::Rgb(90, 90, 90);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            if app.pending_approval.is_some() {
                " Ctrl-Y allow · Ctrl-N deny · Enter send · Esc cancel · Ctrl-B back · Ctrl-C quit"
            } else if matches!(app.pending_shell, Some(ShellState::Finished { .. })) {
                " Ctrl-S send with next message · Ctrl-D discard · Ctrl-B back · Ctrl-C quit"
            } else if completing {
                // Scrolling gives up its place rather than the row wrapping:
                // the keys worth naming are the ones for what is on screen
                // right now, and the list above is what that is.
                " Tab complete · Enter send · Esc cancel · Ctrl-B back · Ctrl-C quit"
            } else {
                " Enter send · Esc cancel · PgUp/PgDn scroll · Ctrl-B back · Ctrl-C quit"
            },
            Style::new().fg(KEYBIND_GRAY).dim(),
        ))),
        area,
    );
}

/// Takes over the input box — rather than floating a modal over the
/// conversation — since typing is already redirected to y/n/esc during
/// approval and the box is otherwise sitting idle.
/// Answered by typing rather than a raw keypress, so the box works like the
/// ordinary input it's replacing: type, edit, Enter to submit. The tool's
/// detail comes first and the prompt sits last, right above where you'd
/// normally be typing a message — its row is computed from the detail's
/// line count, which only stays exact if a field can't reflow onto a
/// second row and silently push the prompt below the box's bottom edge.
/// That's why this deliberately doesn't `.wrap(..)`: a field long enough to
/// overflow the box just gets clipped at the edge instead, which loses
/// characters but never the interactive prompt beneath it.
fn draw_approval(frame: &mut Frame, area: Rect, request: &ApprovalRequest) {
    let category = match request.category {
        "read" => "Read from disk",
        "write" => "Write to disk",
        "terminal" => "Terminal command",
        _ => "Unknown action",
    };
    // The keys live in the title because they are the only place they can be
    // discovered: answering no longer takes over the input box, so there is
    // nothing in the way to suggest that a decision is owed.
    let title = format!(" {category} — Ctrl-Y allow · Ctrl-N deny ");

    // The gap `approval_rows` reserved is left unpainted.
    let box_area = Rect {
        y: area.y + APPROVAL_GAP,
        height: area.height.saturating_sub(APPROVAL_GAP),
        ..area
    };
    frame.render_widget(
        Paragraph::new(Text::from(approval_lines(request)))
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::new().yellow())
                    .title(Span::styled(title, Style::new().yellow().bold())),
            ),
        box_area,
    );
}

/// The tool and its arguments, field by field — matching how the CLI
/// presents an approval prompt and how a verbose tool-call notice presents
/// its arguments.
fn approval_lines(request: &ApprovalRequest) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(vec![
        Span::styled("tool  ", Style::new().dark_gray()),
        Span::styled(request.tool_name.clone(), Style::new().bold()),
    ])];
    for (key, shown) in json_fields(&request.arguments) {
        lines.push(Line::from(vec![
            Span::styled(format!("{key}  "), Style::new().dark_gray()),
            Span::raw(shown),
        ]));
    }
    lines
}

/// Renders markdown to styled rows.
///
/// Applied only to assistant replies: a user's own `*asterisks*` should
/// appear as typed, and tool lines are already formatted. The result is
/// re-homed into owned spans because the transcript outlives the borrow of
/// the message it came from.
fn markdown_lines(text: &str) -> Vec<Line<'static>> {
    let mut inside_code = false;

    tui_markdown::from_str(text)
        .lines
        .into_iter()
        .filter_map(|line| {
            let plain: String = line.spans.iter().map(|s| s.content.as_ref()).collect();

            // The fences come through as literal rows of backticks. The code
            // between them is already syntax-coloured, so they'd just be
            // noise: label the opening one with its language and drop the
            // close.
            if plain.trim_end().starts_with("```") {
                inside_code = !inside_code;
                if !inside_code {
                    return None;
                }
                let language = plain.trim().trim_start_matches('`').trim();
                let tag = if language.is_empty() {
                    "code".to_string()
                } else {
                    language.to_string()
                };
                return Some(Line::from(Span::styled(
                    tag,
                    Style::new().dark_gray().italic(),
                )));
            }

            let spans: Vec<Span<'static>> = line
                .spans
                .into_iter()
                .map(|span| Span::styled(span.content.into_owned(), span.style))
                .collect();
            Some(Line::from(spans))
        })
        .collect()
}

/// Pushes already-rendered rows under a speaker label, keeping the label on
/// the first row and any trailing marker on the last.
/// Every message-start line leads with a 2-column glyph — `❯ `, `✓ `, `— `,
/// or two blank spaces where there's no icon — so replies read as one
/// aligned gutter down the left edge. Continuation lines — a literal
/// embedded newline, or a row `wrap_styled` broke a long one into — get
/// this same blank width instead of the glyph, so their text lines up
/// under the first line's content rather than under the icon.
/// The band behind a user's own messages, and behind the selected row on
/// the launch screen. Deliberately slight — a step off the background rather
/// than a colour, so it separates without competing with the text.
///
/// Which step depends on the terminal: one shade *lighter* than a dark
/// background, one *darker* than a light one. A fixed dark band reads as a
/// heavy bar on a light theme, which is the opposite of subtle.
static BAND: std::sync::OnceLock<Style> = std::sync::OnceLock::new();

/// Asks the terminal what colour it actually is, once, and remembers the
/// band derived from it.
///
/// Must run before the alternate screen is entered: the query writes an
/// escape sequence and reads the reply, and doing that mid-draw would race
/// the renderer for the terminal.
pub(super) fn detect_band() {
    use terminal_colorsaurus::{background_color, QueryOptions};
    if let Ok(background) = background_color(QueryOptions::default()) {
        let _ = BAND.set(band_for(
            background.perceived_lightness(),
            (
                scale(background.r),
                scale(background.g),
                scale(background.b),
            ),
        ));
    }
}

/// The reply gives 16 bits per channel; a terminal colour takes 8.
fn scale(channel: u16) -> u8 {
    (channel >> 8) as u8
}

/// A band a fixed step off the terminal's own background, in the direction
/// that keeps it subtle: lighter on a dark terminal, darker on a light one.
///
/// Derived from the real background rather than named as a palette slot.
/// `Indexed(234)` is only #1c1c1c on a terminal that hasn't remapped its
/// palette, and themes remap it — which is how a light theme ended up
/// showing a near-black band. An RGB value is the colour we asked for.
fn band_for(lightness: f32, (r, g, b): (u8, u8, u8)) -> Style {
    // Small enough to read as a tint of the background rather than a bar
    // drawn over it: 2-4 points of L* across the backgrounds terminals
    // actually use, which is near the floor of what separates two areas at
    // all. The band only has to make your own messages findable when
    // scrolling back, not announce them.
    //
    // A flat step in sRGB rather than a computed one in a perceptual space.
    // sRGB's gamma curve already spends most of its range on the dark end,
    // so one number stays in that narrow band whether the terminal is
    // #1e1e1e, mid-grey or white — the ends drift low (a pure black
    // terminal gets ~2.2) but drift *subtler*, which is the safe direction.
    const STEP: i16 = 8;
    let step = if lightness < 0.5 { STEP } else { -STEP };
    let shift = |channel: u8| (channel as i16 + step).clamp(0, 255) as u8;
    Style::new().bg(Color::Rgb(shift(r), shift(g), shift(b)))
}

/// The band, or no band at all when the terminal never said what colour it
/// is. Guessing is what produced a near-black bar on a light theme, and a
/// missing highlight is a far smaller failure than a wrong one — the
/// selection still has its marker and its bold.
pub(super) fn band() -> Style {
    BAND.get().copied().unwrap_or_default()
}

/// Puts the band behind a block of rows, when the session asks for one.
///
/// Padding and tinting together: a background only paints the cells a line
/// covers, so the two always go with each other. Split out so the `off` case
/// is testable — under test no terminal has answered, and with no band both
/// paths would otherwise look identical.
pub(super) fn highlight_rows(rows: &mut [Line<'static>], width: usize, on: bool) {
    if !on {
        return;
    }
    for line in rows {
        pad_to(line, width);
        line.style = line.style.patch(band());
    }
}

/// Fills a line out to `width` so a background paints the whole row rather
/// than stopping where the text does.
/// How many terminal cells a string occupies.
///
/// Not `chars().count()`: an emoji or a CJK glyph is one character and two
/// cells, so counting characters under-measures and the row overflows the
/// pane. That used to be caught by ratatui's `Wrap` re-wrapping the row —
/// the "wrapped again, out from under the gutter" the gutter widths above
/// still subtract 1 to work around — but the transcript now renders
/// unwrapped, where the same row is silently clipped instead.
fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

pub(super) fn pad_to(line: &mut Line<'static>, width: usize) {
    let used: usize = line
        .spans
        .iter()
        .map(|span| display_width(&span.content))
        .sum();
    if used < width {
        line.spans.push(Span::raw(" ".repeat(width - used)));
    }
}

const GUTTER_CONTINUATION: &str = "  ";

/// The continuation for a gutter three columns wide — the session's braille
/// square and a space, like the tool gutter's double-width hammer.
const SQUARE_CONTINUATION: &str = "   ";

/// Pushes a "label  value" row, one level of indent deeper than the
/// message gutter — a verbose tool-call field or result. The value wraps
/// and, unlike the message gutter, its continuation lines up under the
/// value itself rather than back at the label, since there's no icon here
/// competing for that column.
fn push_labeled(lines: &mut Vec<Line<'static>>, label: String, value: String, width: usize) {
    let label_width = display_width(&label);
    let value_width = width.saturating_sub(label_width).max(1);
    let blank = " ".repeat(label_width);
    for (index, mut row) in wrap_styled(Line::from(Span::raw(value)), value_width)
        .into_iter()
        .enumerate()
    {
        if index == 0 {
            row.spans
                .insert(0, Span::styled(label.clone(), Style::new().dark_gray()));
        } else {
            row.spans.insert(0, Span::raw(blank.clone()));
        }
        lines.push(row);
    }
}

/// `gutter` is the continuation indent for a wrapped row — normally
/// [`GUTTER_CONTINUATION`], sized to match a single-width marker
/// (`❯`/`●`/`—`) plus its trailing space, but callers whose `prefix` is
/// wider (🔨 is double-width, so `"🔨 "` alone fills 3 columns) pass a
/// wider one instead, so a wrapped continuation row still lines up under
/// the first row's actual text rather than the usual 2-column gutter.
fn push_rendered(
    lines: &mut Vec<Line<'static>>,
    prefix: Span<'static>,
    mut rendered: Vec<Line<'static>>,
    trailing: Option<Span<'static>>,
    width: usize,
    gutter: &str,
) {
    if rendered.is_empty() {
        rendered.push(Line::raw(""));
    }
    let last_line = rendered.len() - 1;
    for (line_index, line) in rendered.into_iter().enumerate() {
        let wrapped = wrap_styled(line, width);
        let last_row = wrapped.len() - 1;
        for (row_index, mut row) in wrapped.into_iter().enumerate() {
            if line_index == 0 && row_index == 0 {
                row.spans.insert(0, prefix.clone());
            } else {
                row.spans.insert(0, Span::raw(gutter.to_string()));
            }
            if line_index == last_line && row_index == last_row {
                if let Some(trailing) = trailing.clone() {
                    row.spans.push(trailing);
                }
            }
            lines.push(row);
        }
    }
}

/// Pushes one speaker's text, split into a `Line` per newline (and, within
/// each, further wrapped to `width` — see `wrap_styled`).
///
/// A ratatui `Line` is a single row: it doesn't break on an embedded `\n`,
/// so putting a whole multi-paragraph reply in one Line both renders it as
/// a run-together blob and makes it impossible to measure — the height
/// estimate would count the paragraphs that the render then doesn't make.
/// `prefix` labels the first row; `trailing` (the streaming cursor) goes on
/// the last.
fn push_block(
    lines: &mut Vec<Line<'static>>,
    prefix: Span<'static>,
    text: &str,
    trailing: Option<Span<'static>>,
    width: usize,
    body: Style,
) {
    let segments: Vec<&str> = text.split('\n').collect();
    let last_segment = segments.len() - 1;
    for (seg_index, segment) in segments.into_iter().enumerate() {
        let wrapped = wrap_styled(Line::from(Span::styled(segment.to_string(), body)), width);
        let last_row = wrapped.len() - 1;
        for (row_index, mut row) in wrapped.into_iter().enumerate() {
            if seg_index == 0 && row_index == 0 {
                row.spans.insert(0, prefix.clone());
            } else {
                row.spans.insert(0, Span::raw(GUTTER_CONTINUATION));
            }
            if seg_index == last_segment && row_index == last_row {
                if let Some(trailing) = trailing.clone() {
                    row.spans.push(trailing);
                }
            }
            lines.push(row);
        }
    }
}

/// Word-wraps one styled line to `width` columns, keeping each span's style
/// attached to the text it colors. Breaks preferentially at spaces; a
/// single word longer than `width` is hard-broken so no row ever exceeds
/// it. Doing this ourselves — rather than leaving it to ratatui's `Wrap`,
/// which has no notion of a hanging indent — is what lets a wrapped
/// continuation row share the gutter indent with the row it continues,
/// the same as a row split by a literal newline already does.
fn wrap_styled(line: Line<'static>, width: usize) -> Vec<Line<'static>> {
    let width = width.max(1);

    // Each span's text broken into (word-or-space, is_space) tokens, style
    // still attached, so a word split across a style boundary (rare, but
    // possible around markdown emphasis markers) still wraps sanely.
    let mut tokens: Vec<(String, Style, bool)> = Vec::new();
    for span in line.spans {
        let style = span.style;
        let mut word = String::new();
        for ch in span.content.chars() {
            if ch == ' ' {
                if !word.is_empty() {
                    tokens.push((std::mem::take(&mut word), style, false));
                }
                tokens.push((" ".to_string(), style, true));
            } else {
                word.push(ch);
            }
        }
        if !word.is_empty() {
            tokens.push((word, style, false));
        }
    }

    // The line's own leading whitespace, lifted off the front and put back
    // on every row it wraps onto. Without this a long line inside a code
    // block resumes at column 0, so a wrapped statement reads as though it
    // had jumped out a nesting level. Prose is unaffected — it has no
    // leading whitespace, so the indent is empty and nothing changes.
    let leading: usize = tokens
        .iter()
        .take_while(|(_, _, is_space)| *is_space)
        .count();
    let indent: String = tokens[..leading]
        .iter()
        .map(|(text, ..)| text.as_str())
        .collect();
    let indent_style = tokens
        .first()
        .map(|(_, style, _)| *style)
        .unwrap_or_default();
    let indent_width = display_width(&indent);

    // Only worth hanging while it leaves more room than it takes. A deeply
    // indented line in a narrow pane is better wrapped flush than shredded
    // into a two-column strip down the right-hand side.
    let (indent, body_width) = if indent_width > 0 && indent_width * 2 <= width {
        (indent, width - indent_width)
    } else {
        (String::new(), width)
    };
    let tokens = if indent.is_empty() {
        tokens
    } else {
        tokens.split_off(leading)
    };
    let width = body_width;

    let mut rows: Vec<Vec<Span<'static>>> = vec![Vec::new()];
    let mut col = 0usize;
    for (index, (text, style, is_space)) in tokens.into_iter().enumerate() {
        let token_width = display_width(&text);

        if is_space {
            // A space landing exactly at the start of a row *a wrap broke
            // onto* is just where the break happened to fall — starting
            // that row indented by it would look like a stray extra space.
            // Leading whitespace never reaches here: it was taken off above
            // and is re-applied to every row below.
            if col == 0 && index != 0 {
                continue;
            }
            if col + token_width > width {
                rows.push(Vec::new());
                col = 0;
            } else {
                rows.last_mut().unwrap().push(Span::styled(text, style));
                col += token_width;
            }
            continue;
        }

        if token_width > width {
            // Doesn't fit on a row by itself either way: hard-break it.
            let chars: Vec<char> = text.chars().collect();
            for chunk in chars.chunks(width) {
                if col > 0 {
                    rows.push(Vec::new());
                }
                col = chunk.len();
                rows.last_mut()
                    .unwrap()
                    .push(Span::styled(chunk.iter().collect::<String>(), style));
            }
            continue;
        }

        if col > 0 && col + token_width > width {
            rows.push(Vec::new());
            col = 0;
        }
        rows.last_mut().unwrap().push(Span::styled(text, style));
        col += token_width;
    }

    rows.into_iter()
        .map(|mut spans| {
            if !indent.is_empty() {
                spans.insert(0, Span::styled(indent.clone(), indent_style));
            }
            Line::from(spans)
        })
        .collect()
}

#[cfg(test)]
mod tests {

    #[test]
    fn highlighting_off_leaves_the_rows_alone() {
        // Observable through the padding rather than the colour: with no
        // terminal to answer the query there is no band under test, so the
        // tint alone would make both paths look the same.
        let width =
            |line: &Line| -> usize { line.spans.iter().map(|s| s.content.chars().count()).sum() };

        let mut off = [Line::from(vec![Span::raw("short")])];
        highlight_rows(&mut off, 40, false);
        assert_eq!(width(&off[0]), 5, "untouched");

        let mut on = [Line::from(vec![Span::raw("short")])];
        highlight_rows(&mut on, 40, true);
        assert_eq!(width(&on[0]), 40, "padded so the band covers the row");
    }

    fn rgb(style: Style) -> (u8, u8, u8) {
        match style.bg {
            Some(Color::Rgb(r, g, b)) => (r, g, b),
            other => panic!("expected an RGB background, got {other:?}"),
        }
    }

    #[test]
    fn the_band_steps_off_the_terminals_own_background() {
        // Not a named palette slot: themes remap those, which is how a light
        // theme came to show a near-black band.
        let dark_bg = (0x1e, 0x1e, 0x2e);
        let (r, g, b) = rgb(band_for(0.1, dark_bg));
        assert!(
            r > dark_bg.0 && g > dark_bg.1 && b > dark_bg.2,
            "lift a dark one"
        );
        // A tint of the background, keeping its hue — not neutral grey over
        // a coloured terminal, which reads as a smudge.
        assert!(b > r, "the background's blue cast survives");

        let light_bg = (0xfa, 0xfa, 0xf8);
        let (r, g, b) = rgb(band_for(0.9, light_bg));
        assert!(
            r < light_bg.0 && g < light_bg.1 && b < light_bg.2,
            "darken a light one"
        );
    }

    #[test]
    fn the_step_stays_inside_the_channel_range() {
        // Pure black and pure white are the ends a naive add or subtract
        // would run off.
        let _ = rgb(band_for(0.0, (0, 0, 0)));
        let _ = rgb(band_for(1.0, (255, 255, 255)));
    }

    #[test]
    fn no_band_at_all_when_the_terminal_never_answered() {
        // The default in tests, where nothing queried anything: a missing
        // highlight beats a guessed one that fights the theme.
        assert_eq!(band(), Style::default());
        assert!(band().bg.is_none());
    }

    #[test]
    fn a_highlighted_row_is_padded_to_the_full_width() {
        // The band is a background, and a background only paints the cells a
        // line actually covers — so without this an unpadded row would end
        // raggedly at its text. Tested here rather than against a rendered
        // buffer because no band exists until a terminal answers, which it
        // never does under test.
        let mut line = Line::from(vec![Span::raw("short")]);
        pad_to(&mut line, 40);
        let width: usize = line
            .spans
            .iter()
            .map(|span| span.content.chars().count())
            .sum();
        assert_eq!(width, 40);

        // Already wider than the target: left alone rather than truncated,
        // since cutting a row to fit would lose text.
        let mut wide = Line::from(vec![Span::raw("x".repeat(50))]);
        pad_to(&mut wide, 40);
        assert_eq!(wide.spans.len(), 1);
    }
    #[test]
    fn a_mark_is_two_cells_of_braille_and_the_same_every_time() {
        let (mark, style) = identicon("4f2a91b2-0000-0000-0000-000000000000");
        assert_eq!(mark, identicon("4f2a91b2-0000-0000-0000-000000000000").0);
        assert_eq!(mark.chars().count(), 2);
        for dot in mark.chars() {
            let pattern = dot as u32 - 0x2800;
            assert!(pattern < 256, "{dot:?} is not a braille pattern");
            // Never blank, never solid: a half of either kind reads as a
            // fault rather than as a mark.
            assert!((3..=7).contains(&pattern.count_ones()), "{dot:?}");
        }
        assert!(style.fg.is_some());
        assert!(style.bg.is_none(), "the row's own background shows through");
    }

    #[test]
    fn marks_use_the_whole_palette_and_rarely_repeat() {
        let ids: Vec<String> = (0..500).map(|n| format!("{n:08x}-session")).collect();
        let marks: std::collections::HashSet<_> = ids.iter().map(|id| identicon(id).0).collect();
        // Two of 500 sharing a glyph pair is fine; a hash collapsing onto a
        // handful of patterns is not.
        assert!(marks.len() > 450, "only {} distinct marks", marks.len());

        let fgs: std::collections::HashSet<_> =
            ids.iter().filter_map(|id| identicon(id).1.fg).collect();
        assert_eq!(fgs.len(), IDENTICON_FG.len());
    }

    #[test]
    fn the_gutter_mark_is_the_session_mark() {
        // Seeded with the id the App actually holds, not a literal chosen to
        // make the comparison work: feeding both sides the same string was
        // what let the app hash a truncated id unnoticed.
        let full = "4f2a91b2-3c1d-4e8a-9f02-7b6c5d4e3a21";
        let app = App::new("m".to_string(), None, full.to_string());
        let (mark, style) = identicon(&app.session_id);

        // Two cells, so the gutter is three columns and wraps one narrower —
        // the same treatment the double-width tool hammer gets.
        assert_eq!(mark.chars().count(), 2);
        assert_eq!(SQUARE_CONTINUATION.len(), 3);
        assert!(style.fg.is_some());

        // A truncated id is a different session as far as the hash is
        // concerned, which is how the picker and the gutter came to disagree.
        assert_ne!(mark, identicon(app.short_id()).0);
    }

    #[test]
    fn a_running_command_is_just_a_titled_border() {
        let mut app = sample_app();
        app.pending_shell = Some(ShellState::Running {
            command: "cargo test".into(),
        });
        let out = render_to_string(&app, 74, 14);
        assert!(out.contains("$ cargo test"), "{out}");
        // No decision to offer yet.
        assert!(!out.contains("Ctrl-S"), "{out}");
    }

    #[test]
    fn a_finished_command_shows_its_output_and_its_keys() {
        let mut app = sample_app();
        app.pending_shell = Some(ShellState::Finished {
            command: "cargo test".into(),
            output: "299 passed; 0 failed".into(),
            exit_code: 0,
        });
        let out = render_to_string(&app, 74, 16);
        assert!(out.contains("299 passed"), "{out}");
        assert!(out.contains("Ctrl-S send"), "{out}");
        assert!(out.contains("Ctrl-D discard"), "{out}");
    }

    #[test]
    fn a_failing_command_shows_its_code_beside_the_command() {
        let mut app = sample_app();
        app.pending_shell = Some(ShellState::Finished {
            command: "cargo build".into(),
            output: "error".into(),
            exit_code: 101,
        });
        let out = render_to_string(&app, 74, 16);
        assert!(out.contains("exit 101"), "{out}");
    }

    #[test]
    fn the_two_boxes_name_different_keys() {
        // Both can be open at once, so a shared chord would act on whichever
        // happened to be there. They must not advertise the same keys.
        let mut app = sample_app();
        app.pending_approval = Some(ApprovalRequest {
            tool_name: "write_file".into(),
            category: "write",
            arguments: "{}".into(),
        });
        app.pending_shell = Some(ShellState::Finished {
            command: "ls".into(),
            output: "src".into(),
            exit_code: 0,
        });
        let out = render_to_string(&app, 78, 22);

        assert!(out.contains("Ctrl-Y allow"), "{out}");
        assert!(out.contains("Ctrl-S send"), "{out}");
        let approval_at = out.find("Write to disk").expect("approval shown");
        let shell_at = out.find("$ ls").expect("command shown");
        assert!(approval_at < shell_at, "approval sits above: {out}");
    }

    /// Each span of each row as (text, whether it is coloured as a command),
    /// which is the whole of what these tests care about.
    fn marked(input: &str, width: u16) -> Vec<Vec<(String, bool)>> {
        styled_input_lines(input, width)
            .into_iter()
            .map(|line| {
                line.spans
                    .into_iter()
                    .map(|span| (span.content.to_string(), span.style == COMMAND))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn a_command_name_is_marked_off_from_its_argument() {
        assert_eq!(
            marked("/clanker title Notes", 40),
            vec![vec![
                ("/clanker".to_string(), true),
                (" title Notes".to_string(), false),
            ]]
        );
    }

    #[test]
    fn an_ordinary_message_is_left_alone() {
        // One unstyled span, not a split one: nothing here is a command, so
        // there is nothing to divide.
        assert_eq!(
            marked("what does /etc/hosts do?", 40),
            vec![vec![("what does /etc/hosts do?".to_string(), false)]]
        );
        assert_eq!(marked("/hel", 40), vec![vec![("/hel".to_string(), false)]]);
    }

    #[test]
    fn the_mark_follows_the_name_across_a_wrap() {
        // The box is narrow enough to split the name itself. The colour has
        // to break where the row breaks and pick up again, or it lands on
        // whatever text happens to sit at those columns on the next row.
        assert_eq!(
            marked("/clanker title", 5),
            vec![
                vec![("/clan".to_string(), true)],
                vec![("ker".to_string(), true), (" t".to_string(), false)],
                vec![("itle".to_string(), false)],
            ]
        );
    }

    #[test]
    fn a_newline_before_the_command_does_not_shift_the_mark() {
        // The newline is a character the rows don't show. Uncounted, every
        // offset after it is one short and the colour slides left.
        assert_eq!(
            marked("\n/help me", 40),
            vec![
                // The blank row carries no spans at all to style.
                vec![],
                vec![("/help".to_string(), true), (" me".to_string(), false)],
            ]
        );
    }

    /// The colour the text `needle` is drawn in, found by rendering and
    /// looking at the cells it actually occupies.
    fn colour_of(app: &App, width: u16, height: u16, needle: &str) -> Color {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal
            .draw(|frame| draw(frame, app, &mut TranscriptCache::default(), 0))
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        for y in 0..buffer.area.height {
            let cells: Vec<String> = (0..buffer.area.width)
                .map(|x| buffer[(x, y)].symbol().to_string())
                .collect();
            if let Some(x) = (0..cells.len()).find(|x| cells[*x..].concat().starts_with(needle)) {
                return buffer[(x as u16, y)].fg;
            }
        }
        panic!("{needle} was never drawn");
    }

    fn ran(sent: bool) -> TranscriptItem {
        TranscriptItem::Shell {
            command: "ls".to_string(),
            output: "onlyinthisone".to_string(),
            exit_code: 0,
            sent,
        }
    }

    #[test]
    fn output_the_model_never_saw_is_dimmed() {
        // The label already says "not sent", but reading a label per command
        // is not how you scan a transcript. The colour is the difference you
        // can see without stopping.
        let mut app = sample_app();
        app.transcript.push(ran(false));
        assert_eq!(colour_of(&app, 74, 20, "onlyinthisone"), Color::DarkGray);

        let mut app = sample_app();
        app.transcript.push(ran(true));
        assert_eq!(colour_of(&app, 74, 20, "onlyinthisone"), Color::Reset);
    }

    #[test]
    fn a_turn_stopped_at_a_gate_says_it_is_waiting_not_working() {
        // `busy` stays true through an approval — the turn has not ended, it
        // is standing still. Animating it as working says the opposite.
        let mut app = sample_app();
        app.busy = true;
        let out = render_to_string(&app, 74, 16);
        assert!(out.contains(" working "), "{out}");
        assert!(!out.contains("? waiting"), "{out}");

        app.pending_approval = Some(ApprovalRequest {
            tool_name: "write_file".into(),
            category: "write",
            arguments: "{}".into(),
        });
        let out = render_to_string(&app, 74, 16);
        assert!(out.contains("? waiting"), "{out}");
        assert!(!out.contains(" working "), "{out}");
    }

    /// The symbols on the first rendered row containing `needle` that are
    /// drawn in the command colour, so the wiring into `draw_input` — not
    /// just the row builder — is what's under test.
    fn command_coloured_row(app: &App, width: u16, height: u16, needle: &str) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal
            .draw(|frame| draw(frame, app, &mut TranscriptCache::default(), 0))
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        for y in 0..buffer.area.height {
            let cells: Vec<_> = (0..buffer.area.width).map(|x| &buffer[(x, y)]).collect();
            let text: String = cells.iter().map(|c| c.symbol()).collect();
            if text.contains(needle) {
                return cells
                    .iter()
                    .filter(|c| c.fg == COMMAND.fg.unwrap())
                    .map(|c| c.symbol())
                    .collect();
            }
        }
        panic!("{needle} was never drawn");
    }

    /// The hint row's spans as (text, is-it-the-active-one).
    fn hint_row(
        names: &[&'static str],
        active: Option<usize>,
        width: usize,
    ) -> Vec<(String, bool)> {
        match_spans(names, active, width)
            .into_iter()
            .map(|span| (span.content.to_string(), span.style == COMMAND))
            .collect()
    }

    #[test]
    fn a_list_too_long_for_the_row_says_how_much_it_left_off() {
        // Without the count a bare `/` looks like a session with four
        // commands in it.
        let names = ["help", "models", "model", "agent", "ask", "effort"];
        let text: String = hint_row(&names, None, 22)
            .into_iter()
            .map(|(text, _)| text)
            .collect();
        // "model" would fit on its own, but not alongside the count that
        // says three more are missing, and the count is the load-bearing
        // half.
        assert_eq!(text, " help  models  +4");
    }

    #[test]
    fn the_row_slides_to_keep_the_stepped_to_name_in_view() {
        // A mark you cannot see is the same as no mark at all: Tab has
        // written "effort" into the box, so "effort" has to be on the row.
        let names = ["help", "models", "model", "agent", "ask", "effort"];
        let row = hint_row(&names, Some(5), 22);
        assert!(row.iter().any(|(text, active)| text == "effort" && *active));
        assert!(!row.iter().any(|(text, _)| text == "help"));
    }

    #[test]
    fn the_whole_list_fits_when_the_row_is_wide_enough() {
        let names = ["models", "model"];
        assert_eq!(
            hint_row(&names, Some(1), 40),
            vec![
                (" ".to_string(), false),
                ("models".to_string(), false),
                ("  ".to_string(), false),
                ("model".to_string(), true),
            ]
        );
    }

    #[test]
    fn the_row_above_the_box_shows_what_is_being_typed_into_it() {
        let mut app = sample_app();
        app.input = "/m".to_string();
        app.cursor = app.input.len();
        let out = render_to_string(&app, 74, 16);
        assert!(out.contains("models  model  max-iterations"), "{out}");
        // Tab is worth naming only while there is something to complete.
        assert!(out.contains("Tab complete"), "{out}");

        // Once the name is settled the row turns into the command's form...
        app.input = "/tools ".to_string();
        app.cursor = app.input.len();
        let out = render_to_string(&app, 74, 16);
        assert!(out.contains("/tools [on|off | <ask|allow|never>"), "{out}");
        assert!(!out.contains("Tab complete"), "{out}");

        // ...and an ordinary message gets no row at all.
        app.input = "hello there".to_string();
        app.cursor = app.input.len();
        let out = render_to_string(&app, 74, 16);
        assert!(!out.contains("Tab complete"), "{out}");
        assert!(out.contains("PgUp/PgDn scroll"), "{out}");
    }

    #[test]
    fn the_box_shows_a_command_as_a_command_before_it_is_sent() {
        let mut app = sample_app();
        app.input = "/effort high".to_string();
        app.cursor = app.input.len();
        // Matched on the whole line: the hint row above the box carries the
        // command's name too, and this is about the box itself.
        assert_eq!(
            command_coloured_row(&app, 74, 16, "/effort high"),
            "/effort"
        );

        // ...and says nothing about a message that merely starts with one.
        app.input = "/etc/hosts is the file".to_string();
        app.cursor = app.input.len();
        assert_eq!(command_coloured_row(&app, 74, 16, &app.input.clone()), "");
    }

    #[test]
    fn no_box_when_no_command_has_been_run() {
        let out = render_to_string(&sample_app(), 74, 14);
        assert!(!out.contains("Ctrl-S"), "{out}");
    }
    use super::*;
    use crate::tui::app::App;
    use crate::ui::ApprovalRequest;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    /// Renders to an off-screen buffer and returns it as text, so layout can
    /// be asserted (and panics caught) without a real terminal.
    fn render_to_string(app: &App, width: u16, height: u16) -> String {
        render_with(app, &mut TranscriptCache::default(), width, height)
    }

    /// Renders through a caller-owned cache, so a second call exercises the
    /// warm path that `render_to_string` never reaches.
    fn render_with(app: &App, cache: &mut TranscriptCache, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| draw(frame, app, cache, 0)).unwrap();
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

    fn sample_app() -> App {
        let mut app = App::new("test-model".to_string(), None, "abcd1234".to_string());
        app.transcript.push(TranscriptItem::User("hello".into()));
        app.transcript.push(TranscriptItem::Assistant {
            text: "hi there".into(),
            streaming: false,
            label: Some("test-model".into()),
        });
        app
    }

    #[test]
    fn a_second_frame_off_the_cache_draws_what_the_first_one_did() {
        let app = sample_app();
        let mut cache = TranscriptCache::default();
        let cold = render_with(&app, &mut cache, 60, 20);
        assert!(!cache.rows.is_empty(), "nothing was cached");
        let warm = render_with(&app, &mut cache, 60, 20);
        assert_eq!(cold, warm, "cached rows drew differently from fresh ones");
    }

    #[test]
    fn editing_a_message_retires_the_rows_cached_for_it() {
        let mut app = sample_app();
        let mut cache = TranscriptCache::default();
        let before = render_with(&app, &mut cache, 60, 20);
        assert!(before.contains("hi there"));

        let Some(TranscriptItem::Assistant { text, .. }) = app.transcript.last_mut() else {
            panic!("sample_app should end on a reply");
        };
        *text = "different answer".to_string();

        let after = render_with(&app, &mut cache, 60, 20);
        assert!(after.contains("different answer"), "{after}");
        assert!(
            !after.contains("hi there"),
            "stale rows survived the edit:\n{after}"
        );
    }

    #[test]
    fn a_resize_redraws_rather_than_reusing_rows_wrapped_for_the_old_width() {
        let mut app = App::new("m".to_string(), None, "abcd1234".to_string());
        app.transcript.push(TranscriptItem::Assistant {
            text: "a ".repeat(60),
            streaming: false,
            label: None,
        });
        let mut cache = TranscriptCache::default();

        let narrow = render_with(&app, &mut cache, 40, 20);
        let wide = render_with(&app, &mut cache, 100, 20);
        // Wrapped at 40 the text needs more rows than at 100, so a cache
        // that ignored width would show the narrow shape in a wide pane.
        let rows = |s: &str| s.lines().filter(|l| l.contains('a')).count();
        assert!(
            rows(&narrow) > rows(&wide),
            "narrow={} wide={}",
            rows(&narrow),
            rows(&wide)
        );
    }

    #[test]
    fn toggling_verbose_redraws_the_blocks_it_changes() {
        let mut app = sample_app();
        app.transcript.push(TranscriptItem::ToolCall {
            name: "read_file".into(),
            arguments: r#"{"path":"src/main.rs"}"#.into(),
            status: ToolStatus::Done {
                result: r#"{"bytes":42}"#.into(),
            },
        });
        let mut cache = TranscriptCache::default();

        let quiet = render_with(&app, &mut cache, 80, 24);
        app.verbose = true;
        let loud = render_with(&app, &mut cache, 80, 24);
        assert!(!quiet.contains("bytes"), "{quiet}");
        assert!(loud.contains("bytes"), "{loud}");
    }

    #[test]
    fn a_streamed_reply_does_not_leave_a_cache_entry_per_delta() {
        let mut app = App::new("m".to_string(), None, "abcd1234".to_string());
        app.transcript.push(TranscriptItem::Assistant {
            text: String::new(),
            streaming: true,
            label: None,
        });
        let mut cache = TranscriptCache::default();

        // Each delta gives the block new content and so a new key. Without
        // the end-of-frame sweep every superseded version would be held for
        // the rest of the session.
        for _ in 0..25 {
            let Some(TranscriptItem::Assistant { text, .. }) = app.transcript.last_mut() else {
                unreachable!()
            };
            text.push_str("more ");
            render_with(&app, &mut cache, 60, 20);
        }
        assert_eq!(
            cache.rows.len(),
            app.transcript.len(),
            "cache grew past the transcript it is caching"
        );
    }

    /// Every item type, with content chosen to overflow, at several widths.
    fn overflowing_transcript() -> Vec<TranscriptItem> {
        // Long words, long prose, and double-width glyphs — the last of
        // which `chars().count()` measures as half their real size.
        let prose = "supercalifragilistic ".repeat(12);
        let wide = "絵文字テスト🔥🚀😀 ".repeat(10);
        vec![
            TranscriptItem::User(format!("{prose}{wide}")),
            TranscriptItem::Assistant {
                text: format!("**bold** {prose}\n\n- {wide}\n\n```rust\nlet x = {prose};\n```"),
                streaming: false,
                label: Some("some/very-long-model-name".into()),
            },
            TranscriptItem::Assistant {
                text: format!("{prose}{wide}"),
                streaming: true,
                label: None,
            },
            TranscriptItem::Thinking(format!("{prose}{wide}")),
            TranscriptItem::ToolCall {
                name: "run_terminal_command".into(),
                arguments: format!(r#"{{"command":"{prose}","cwd":"{wide}"}}"#),
                status: ToolStatus::Done {
                    result: format!(r#"{{"stdout":"{prose}"}}"#),
                },
            },
            TranscriptItem::Shell {
                command: format!("echo {prose}"),
                output: format!("{prose}\n{wide}"),
                exit_code: 1,
                sent: true,
            },
            TranscriptItem::Error(format!("{prose}{wide}")),
            TranscriptItem::Notice(format!("{prose}{wide}")),
            TranscriptItem::SessionStatus(vec![
                ("Working dir".into(), format!("/very/long/path/{prose}")),
                ("Model".into(), wide.clone()),
            ]),
            TranscriptItem::ToolStatus {
                access: crate::config::ToolAccessSettings::default(),
                changed: true,
            },
        ]
    }

    #[test]
    fn no_rendered_row_is_wider_than_the_pane() {
        // The transcript is drawn without ratatui's `Wrap`, so a row wider
        // than the pane is silently clipped rather than folded onto the next
        // line. Nothing else catches that, which is what this is for.
        let mut app = App::new("m".to_string(), None, "abcd1234".to_string());
        app.verbose = true;
        for width in [40u16, 60, 80, 100, 200] {
            let content_width = width.saturating_sub(2).max(1) as usize;
            for item in overflowing_transcript() {
                for row in render_item(&item, &app, width, content_width) {
                    // Measured with the unicode-width call directly rather
                    // than through `display_width`, so that this still fails
                    // if `display_width` itself goes back to counting chars.
                    let drawn: usize = row
                        .spans
                        .iter()
                        .map(|s| UnicodeWidthStr::width(s.content.as_ref()))
                        .sum();
                    assert!(
                        drawn <= width as usize,
                        "row of {drawn} cells in a {width}-cell pane: {:?}",
                        row.spans
                            .iter()
                            .map(|s| s.content.as_ref())
                            .collect::<String>()
                    );
                }
            }
        }
    }

    #[test]
    fn scrolling_back_walks_the_window_up_the_transcript() {
        // The pane no longer hands the whole transcript to ratatui and asks
        // it to scroll; it slices the rows itself. That arithmetic is what
        // decides which message you are looking at, so it gets checked
        // directly rather than only through the "scrolled" flag.
        let mut app = App::new("m".to_string(), None, "abcd1234".to_string());
        for i in 0..60 {
            app.transcript
                .push(TranscriptItem::User(format!("message {i:02}")));
        }
        let mut cache = TranscriptCache::default();

        let shown = |app: &App, cache: &mut TranscriptCache| -> Vec<usize> {
            let out = render_with(app, cache, 40, 20);
            (0..60)
                .filter(|i| out.contains(&format!("message {i:02}")))
                .collect()
        };

        let pinned = shown(&app, &mut cache);
        assert!(
            pinned.contains(&59) && !pinned.contains(&0),
            "pinned to the bottom should show the newest: {pinned:?}"
        );

        app.scroll_back = 20;
        let back = shown(&app, &mut cache);
        assert!(
            back.iter().max() < pinned.iter().max(),
            "scrolling back should move the window up: {back:?} vs {pinned:?}"
        );
        assert!(!back.is_empty(), "scrolled window drew nothing");

        // Past the top it clamps rather than running off the front.
        app.scroll_back = 10_000;
        let top = shown(&app, &mut cache);
        assert!(
            top.contains(&0),
            "scrolling past the top should rest on the first message: {top:?}"
        );
        assert!(!top.contains(&59), "{top:?}");
    }

    #[test]
    fn a_transcript_shorter_than_the_pane_still_sits_on_the_bottom() {
        let mut app = App::new("m".to_string(), None, "abcd1234".to_string());
        app.transcript
            .push(TranscriptItem::User("only message".into()));
        let out = render_to_string(&app, 40, 20);
        let rows: Vec<&str> = out.lines().collect();
        let found = rows
            .iter()
            .position(|r| r.contains("only message"))
            .expect("message missing");
        // Above the input box, not pinned under the title.
        assert!(
            found > rows.len() / 2,
            "sat at row {found} of {}",
            rows.len()
        );
    }

    fn flat(line: &Line) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn a_wrapped_line_hangs_under_its_own_indentation() {
        // A statement inside a code block: when it wraps, the rows it wraps
        // onto have to stay at its nesting level. Resuming at column 0 reads
        // as though the code had jumped out a block.
        let code = "        let value = some_function(first_argument, second_argument, third);";
        let rows = wrap_styled(Line::from(Span::raw(code)), 40);
        assert!(rows.len() > 2, "needs a line that actually wraps");
        for (index, row) in rows.iter().enumerate() {
            assert!(
                flat(row).starts_with("        "),
                "row {index} lost the indent: {:?}",
                flat(row)
            );
        }
    }

    #[test]
    fn prose_is_not_given_an_indent_it_never_had() {
        let rows = wrap_styled(Line::from(Span::raw("plain prose ".repeat(12))), 30);
        assert!(rows.len() > 1);
        for row in &rows {
            assert!(!flat(row).starts_with(' '), "{:?}", flat(row));
        }
    }

    #[test]
    fn an_indent_with_no_room_left_to_hang_is_dropped() {
        // 30 columns of indent in a 40-column pane would leave a 10-column
        // strip down the right-hand side, which is worse than wrapping flush.
        let code = format!(
            "{}{}",
            " ".repeat(30),
            "a word list that has to wrap".repeat(2)
        );
        let rows = wrap_styled(Line::from(Span::raw(code)), 40);
        assert!(rows.len() > 1);
        assert!(
            !flat(&rows[1]).starts_with(' '),
            "hung anyway: {:?}",
            flat(&rows[1])
        );
    }

    /// A frame must not cost more as the conversation grows.
    ///
    /// The transcript used to be rebuilt from scratch every frame — every
    /// reply's markdown re-parsed and re-wrapped — and a frame was drawn per
    /// streamed token, so a long session cost the two multiplied together.
    /// Blocks are cached and only the visible rows are assembled now, and
    /// nothing else in the suite would notice if that stopped being true:
    /// the other render tests check what is drawn, never what it cost.
    ///
    /// The budget is deliberately loose. This corpus takes ~0.3ms optimised
    /// and ~5ms in the unoptimised build the test suite actually runs in, so
    /// 50ms leaves an order of magnitude for a loaded or slow machine. What
    /// it catches is a return to per-frame O(transcript): the same corpus
    /// cost tens of milliseconds *optimised* before the cache, so it would
    /// overrun this by a wide margin rather than a marginal one. It is not
    /// here to measure anything finely.
    #[test]
    fn a_frame_does_not_get_more_expensive_as_the_transcript_grows() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        use std::time::{Duration, Instant};

        const BUDGET: Duration = Duration::from_millis(50);

        let body = concat!(
            "Here is **one** paragraph of prose that runs on a while, with ",
            "some `inline code` and a [link](http://example.com) in it.\n\n",
            "- first item\n- second item\n- third item\n\n",
            "```rust\nfn main() {\n    let v = compute(alpha, beta, gamma);\n}\n```\n\n",
            "More trailing prose to bulk the message out, twice over. ",
            "More trailing prose to bulk the message out, twice over.",
        );

        let mut app = App::new("m".to_string(), None, "abcd1234".to_string());
        for i in 0..1200 {
            app.transcript
                .push(TranscriptItem::User(format!("question {i}")));
            app.transcript.push(TranscriptItem::Assistant {
                text: format!("Reply {i}. {body}"),
                streaming: false,
                label: Some("m".into()),
            });
        }
        let bytes: usize = app
            .transcript
            .iter()
            .map(|item| match item {
                TranscriptItem::User(t) => t.len(),
                TranscriptItem::Assistant { text, .. } => text.len(),
                _ => 0,
            })
            .sum();
        assert!(bytes > 300_000, "the corpus should be big enough to matter");

        let mut cache = TranscriptCache::default();
        let mut terminal = Terminal::new(TestBackend::new(100, 40)).unwrap();
        // The first frame renders every block, and is not what is measured:
        // the claim is about the steady state, which is where the ten frames
        // a second are spent.
        terminal.draw(|f| draw(f, &app, &mut cache, 0)).unwrap();

        let frames = 20;
        let start = Instant::now();
        for _ in 0..frames {
            terminal.draw(|f| draw(f, &app, &mut cache, 0)).unwrap();
        }
        let per_frame = start.elapsed() / frames;

        assert!(
            per_frame < BUDGET,
            "a frame over {bytes} bytes of transcript took {per_frame:?}, \
             budget {BUDGET:?} — the transcript is being rebuilt per frame again"
        );
    }

    #[test]
    fn help_draws_as_a_titled_column() {
        // Tall and wide enough for the whole list to fit unwrapped: the pane
        // scrolls to the newest content, so a short one would cut the
        // heading off the top and prove nothing.
        let mut app = App::new("m".to_string(), None, "abcd1234".to_string());
        app.transcript
            .push(TranscriptItem::Help(crate::ui::help_rows()));
        let out = render_to_string(&app, 120, 60);

        assert!(out.contains("Commands:"), "{out}");
        assert!(out.contains("Show this list"), "{out}");

        // The same block shape `/status` uses: labels padded to a common
        // width, so every description starts in the same column.
        let starts: Vec<usize> = crate::ui::help_rows()
            .iter()
            .map(|(_, blurb)| {
                out.lines()
                    .find_map(|line| line.find(blurb.as_str()))
                    .unwrap_or_else(|| panic!("{blurb} missing from:\n{out}"))
            })
            .collect();
        assert!(
            starts.windows(2).all(|w| w[0] == w[1]),
            "descriptions are not in one column: {starts:?}"
        );
    }

    #[test]
    fn the_model_browser_does_not_butt_against_the_transcript() {
        // It is a tall box that appears without warning; flush against the
        // last reply it reads as part of the conversation rather than over
        // it.
        let mut app = App::new("m".to_string(), None, "abcd1234".to_string());
        app.transcript
            .push(TranscriptItem::User("what models?".into()));
        app.model_browser = Some(crate::tui::app::ModelBrowser::Ready {
            all: vec!["one/model".to_string()],
            filter: String::new(),
            selected: 0,
        });
        let out = render_to_string(&app, 60, 16);
        let rows: Vec<&str> = out.lines().collect();

        let top = rows
            .iter()
            .position(|r| r.contains("┌ models"))
            .expect("the box should be drawn");
        assert!(top > 0);
        assert!(
            rows[top - 1].trim().is_empty(),
            "expected a blank row above the box, found {:?}",
            rows[top - 1]
        );
    }

    #[test]
    fn the_busy_frame_is_two_braille_cells() {
        for tick in 0..64 {
            let frame = busy_frame(tick);
            assert_eq!(frame.chars().count(), 2, "{frame:?} at tick {tick}");
            for ch in frame.chars() {
                assert!(
                    ('\u{2800}'..='\u{28FF}').contains(&ch),
                    "{ch:?} is not braille"
                );
            }
        }
    }

    #[test]
    fn the_busy_frame_is_never_blank_and_never_solid() {
        // Blank reads as a rendering fault and solid reads as stalled —
        // which is the opposite of what an indicator that means "still
        // going" should say.
        for tick in 0..256 {
            for ch in busy_frame(tick).chars() {
                let dots = (ch as u32 - 0x2800).count_ones();
                assert!((3..=7).contains(&dots), "{ch:?} has {dots} dots");
            }
        }
    }

    #[test]
    fn the_busy_frame_looks_random_rather_than_cycling() {
        // The old animation was ten frames in a fixed order, which reads as
        // a clock. Consecutive ticks should not repeat, and a short run
        // should not visibly loop.
        let frames: Vec<String> = (0..40).map(busy_frame).collect();
        assert!(
            frames.windows(2).all(|w| w[0] != w[1]),
            "consecutive frames repeat: {frames:?}"
        );
        let mut distinct = frames.clone();
        distinct.sort();
        distinct.dedup();
        assert!(
            distinct.len() > 30,
            "only {} distinct in 40",
            distinct.len()
        );
    }

    #[test]
    fn the_busy_frame_is_the_same_everywhere_for_a_given_tick() {
        // Both front ends and the picker draw from this, and a spinner that
        // disagreed with itself across panes would read as two things
        // happening rather than one.
        assert_eq!(busy_frame(9), busy_frame(9));
        assert_eq!(busy_frame(1_000_000), busy_frame(1_000_000));
    }

    #[test]
    fn renders_conversation_and_status() {
        let out = render_to_string(&sample_app(), 60, 20);
        assert!(out.contains("❯ hello"), "{out}");
        assert!(out.contains("hi there"), "{out}");
        assert!(out.contains("test-model"), "{out}");
        assert!(out.contains("ready"), "{out}");
    }

    #[test]
    fn the_status_bar_says_whether_it_has_tools() {
        // Derived from the tool states, not from a flag beside them: every
        // tool on `never` is a clanker that can only talk.
        let mut app = sample_app();
        app.tool_access = crate::config::ToolAccessSettings::none();
        assert!(!app.agentic());
        let out = render_to_string(&app, 60, 20);
        assert!(out.contains("💬"), "{out}");
        assert!(!out.contains("🔨"), "{out}");

        app.tool_access = crate::config::ToolAccessSettings::defaults();
        assert!(app.agentic());
        let out = render_to_string(&app, 60, 20);
        assert!(out.contains("🔨"), "{out}");
    }

    #[test]
    fn the_title_row_is_the_mark_the_name_and_where_it_runs() {
        let mut app = sample_app();
        app.title = "Write me a snake game".to_string();
        app.working_dir = Some("/tmp/snake".to_string());
        let out = render_to_string(&app, 60, 20);
        let title_row = out.lines().next().unwrap();

        // The same mark the gutter and the launch screen draw for it — not a
        // second one derived from something else. Checked as a prefix, not
        // full equality: the token badge rides right-aligned on the same
        // row — see `the_title_row_carries_the_token_badge_top_right`.
        let (mark, _) = identicon(&app.session_id);
        assert!(
            title_row.starts_with(&format!("{mark} Write me a snake game  /tmp/snake")),
            "{title_row}"
        );
    }

    #[test]
    fn the_title_row_leaves_out_a_directory_it_does_not_have() {
        // Sessions saved before the directory was recorded have none, and an
        // empty gap after the name says nothing.
        let mut app = sample_app();
        app.title = "Older clanker".to_string();
        app.working_dir = None;
        let out = render_to_string(&app, 60, 20);
        let (mark, _) = identicon(&app.session_id);
        assert!(
            out.lines()
                .next()
                .unwrap()
                .starts_with(&format!("{mark} Older clanker")),
            "{out}"
        );
    }

    #[test]
    fn the_title_row_carries_the_token_badge_top_right() {
        let mut app = sample_app();
        app.title = "Older clanker".to_string();
        app.total_tokens = 12345;
        let out = render_to_string(&app, 60, 20);
        let title_row = out.lines().next().unwrap();

        assert!(title_row.contains('🪙'), "{title_row}");
        let title_at = title_row.find("Older clanker").expect("title shown");
        let tokens_at = title_row.find("12,345").expect("token count shown");
        // Right-aligned: the badge sits at the end of the row, well clear
        // of the title, not tucked in right after it like the directory is.
        assert!(tokens_at > title_at, "{title_row}");
    }

    #[test]
    fn top_status_shows_model_and_effort() {
        let mut app = sample_app();
        let out = render_to_string(&app, 80, 20);
        assert!(out.contains("test-model"), "{out}");
        assert!(out.contains("default"), "{out}");

        app.effort_level = Some("high".to_string());
        let out = render_to_string(&app, 80, 20);
        assert!(out.contains("high"), "{out}");
    }

    #[test]
    fn top_status_shows_temperature_at_the_end() {
        let mut app = sample_app();
        let out = render_to_string(&app, 80, 20);
        assert!(out.contains("🌡 default"), "{out}");

        app.temperature = Some(1.2);
        let out = render_to_string(&app, 80, 20);
        assert!(out.contains("🌡 1.2"), "{out}");
    }

    #[test]
    fn temperature_style_follows_a_cool_to_hot_gradient() {
        assert_eq!(temperature_style(None), Style::new().dark_gray());
        assert_eq!(temperature_style(Some(0.0)), Style::new().cyan());
        assert_eq!(temperature_style(Some(0.7)), Style::new().yellow());
        assert_eq!(
            temperature_style(Some(1.2)),
            Style::new().fg(Color::Rgb(255, 140, 0))
        );
        assert_eq!(temperature_style(Some(2.0)), Style::new().red());
    }

    #[test]
    fn effort_style_follows_a_calm_to_intense_gradient() {
        assert_eq!(effort_style(None), Style::new().dark_gray());
        assert_eq!(effort_style(Some("low")), Style::new().cyan());
        assert_eq!(effort_style(Some("medium")), Style::new().yellow());
        assert_eq!(effort_style(Some("high")), Style::new().red());
    }

    #[test]
    fn bottom_status_shows_the_verbose_indicator() {
        let mut app = sample_app();
        assert!(!app.verbose);
        let out = render_to_string(&app, 80, 20);
        assert!(out.contains("quiet"), "{out}");

        app.verbose = true;
        let out = render_to_string(&app, 80, 20);
        assert!(out.contains("verbose"), "{out}");
    }

    #[test]
    fn a_short_conversation_sits_at_the_bottom_of_the_pane() {
        let out = render_to_string(&sample_app(), 60, 20);
        let rows: Vec<&str> = out.lines().collect();
        // Below the chat history: the input box's 3 rows (top border, one
        // content row for the empty input here, bottom border), the
        // settings row, and the key-bindings row — the last content row
        // sits right above all five of those.
        let last_content = rows.len() - 6;
        assert!(
            rows[last_content].contains("hi there"),
            "newest message should be flush with the bottom of the pane, got:\n{out}"
        );
        // ...and the space is above it, not below. Row 0 is the session
        // title and row 1 the rule under it, so content starts at row 2.
        assert!(
            rows[2].trim().is_empty(),
            "expected blank space above the conversation, got:\n{out}"
        );
    }

    #[test]
    fn scrolling_away_from_the_bottom_flags_the_input_box() {
        let mut app = sample_app();
        for i in 0..30 {
            app.transcript
                .push(TranscriptItem::User(format!("message {i}")));
        }
        let pinned = render_to_string(&app, 60, 20);
        assert!(!pinned.contains("scrolled"), "{pinned}");

        app.scroll_back = 3;
        let scrolled = render_to_string(&app, 60, 20);
        assert!(scrolled.contains("scrolled — End to follow"), "{scrolled}");
        // On the input box's own top border, not floating elsewhere.
        let hint_row = scrolled
            .lines()
            .find(|l| l.contains("scrolled"))
            .expect("hint shown");
        assert!(
            hint_row.starts_with('┌') && hint_row.ends_with('┐'),
            "{hint_row}"
        );
    }

    #[test]
    fn wrapped_continuation_rows_align_under_the_gutter() {
        let mut app = App::new("m".to_string(), None, "id".to_string());
        app.transcript.push(TranscriptItem::User(
            "one two three four five six seven eight nine ten".into(),
        ));
        let out = render_to_string(&app, 30, 14);
        let row = out
            .lines()
            .position(|l| l.trim_start().starts_with("❯ one"))
            .expect("first row shown");
        // The row after it is a wrap-induced continuation (no literal
        // newline in the input), and should start 2 columns in — lined up
        // under "one", not back at column 0 under the glyph.
        let continuation = out.lines().nth(row + 1).expect("continuation row");
        assert!(
            continuation.starts_with("  ") && !continuation.trim().is_empty(),
            "{continuation:?}"
        );
    }

    #[test]
    fn code_block_indentation_survives_wrapping() {
        // The wrap-styled "drop a leading space" rule is meant for the
        // stray space a wrap break happens to land on mid-line — not for a
        // line's own leading whitespace, which is real content (nested
        // indentation inside a code block, say) and must be kept.
        let mut app = App::new("m".to_string(), None, "id".to_string());
        app.transcript.push(TranscriptItem::Assistant {
            text: "```python\nif True:\n    return 1\n```".into(),
            streaming: false,
            label: Some("m".into()),
        });
        let out = render_to_string(&app, 50, 20);
        let indented = out
            .lines()
            .find(|l| l.contains("return 1"))
            .expect("indented line shown");
        // Reply gutter (3 cols: the session's braille square and a space)
        // plus the code's own 4-space indent.
        assert!(indented.starts_with("       return 1"), "{indented:?}");
    }

    #[test]
    fn newest_stays_visible_even_with_unbreakable_text() {
        // Long unbroken tokens (paths, URLs, base64) wrap differently than
        // ordinary prose. If the height estimate disagrees with how ratatui
        // actually lays them out, the view scrolls to the wrong place and the
        // newest message falls below the fold.
        let mut app = App::new("m".to_string(), None, "id".to_string());
        let blob = format!("/a/very/long/unbroken/path/{}", "x".repeat(200));
        for i in 0..12 {
            app.transcript.push(TranscriptItem::Assistant {
                text: format!("msg {i} {blob}"),
                streaming: false,
                label: Some("m".into()),
            });
        }
        app.transcript
            .push(TranscriptItem::User("LASTMESSAGE".to_string()));

        let out = render_to_string(&app, 60, 14);
        assert!(
            out.contains("LASTMESSAGE"),
            "newest message scrolled out of view:\n{out}"
        );
    }

    #[test]
    fn a_long_conversation_still_shows_the_newest_at_the_bottom() {
        let mut app = App::new("m".to_string(), None, "id".to_string());
        for i in 0..40 {
            app.transcript
                .push(TranscriptItem::User(format!("message {i}")));
        }
        let out = render_to_string(&app, 60, 12);
        assert!(out.contains("message 39"), "{out}");
        // The oldest has scrolled off the top.
        assert!(!out.contains("message 0 "), "{out}");
    }

    #[test]
    fn streaming_block_shows_a_cursor() {
        let mut app = App::new("m".to_string(), None, "id".to_string());
        app.transcript.push(TranscriptItem::Assistant {
            text: "partial".into(),
            streaming: true,
            label: Some("m".into()),
        });
        let out = render_to_string(&app, 40, 12);
        assert!(out.contains('▌'), "{out}");
    }

    #[test]
    fn busy_state_shows_the_spinner() {
        let mut app = sample_app();
        app.busy = true;
        let out = render_to_string(&app, 80, 15);
        assert!(out.contains("working"), "{out}");
    }

    #[test]
    fn nothing_waiting_means_no_box_at_all() {
        let app = sample_app();
        let out = render_to_string(&app, 80, 15);
        assert!(!out.contains("joining this turn"), "{out}");
        assert!(!out.contains("next turn"), "{out}");
    }

    #[test]
    fn waiting_messages_are_listed_above_the_prompt() {
        let mut app = sample_app();
        app.busy = true;
        app.tool_access = crate::config::ToolAccessSettings::defaults();
        app.pending
            .push_back("check the Windows path too".to_string());
        app.pending.push_back("and skip the slow tests".to_string());

        let out = render_to_string(&app, 80, 18);
        assert!(out.contains("joining this turn"), "{out}");
        assert!(out.contains("check the Windows path too"), "{out}");
        assert!(out.contains("and skip the slow tests"), "{out}");
        // The count moved out of the settings row and into the box.
        assert!(!out.contains("queued"), "{out}");
    }

    #[test]
    fn the_title_says_where_a_waiting_message_is_headed() {
        let mut app = sample_app();
        app.tool_access = crate::config::ToolAccessSettings::none();
        app.pending.push_back("run it again".to_string());
        let out = render_to_string(&app, 80, 18);
        assert!(out.contains("next turn"), "{out}");
        assert!(!out.contains("joining this turn"), "{out}");
    }

    #[test]
    fn a_long_queue_is_summarised_rather_than_filling_the_screen() {
        let mut app = sample_app();
        for n in 0..8 {
            app.pending.push_back(format!("message {n}"));
        }
        let out = render_to_string(&app, 80, 22);
        assert!(out.contains("message 0"), "{out}");
        assert!(out.contains("message 4"), "{out}");
        assert!(!out.contains("message 5"), "{out}");
        assert!(out.contains("+3 more"), "{out}");
    }

    #[test]
    fn the_box_yields_before_the_transcript_does() {
        // Eight waiting messages want more rows than a short terminal has.
        // Whatever gives, the conversation keeps a row and nothing panics.
        let mut app = sample_app();
        for n in 0..8 {
            app.pending.push_back(format!("message {n}"));
        }
        let out = render_to_string(&app, 60, 10);
        assert!(out.contains("hi there"), "{out}");
    }

    #[test]
    fn pending_box_is_only_as_tall_as_it_needs() {
        assert_eq!(pending_height(0), 0, "no box when nothing waits");
        assert_eq!(pending_height(1), 4, "one row, two borders, the gap");
        assert_eq!(pending_height(PENDING_ROWS), PENDING_ROWS as u16 + 3);
        // Past the cap it stops growing except for the "+N more" line.
        assert_eq!(pending_height(PENDING_ROWS + 1), PENDING_ROWS as u16 + 4);
        assert_eq!(pending_height(100), PENDING_ROWS as u16 + 4);
    }

    #[test]
    fn a_waiting_message_is_one_row_however_it_was_typed() {
        assert_eq!(clip("two\nlines", 20), "two lines");
        assert_eq!(clip("abcdefgh", 4), "abc…");
        assert_eq!(clip("abcd", 4), "abcd");
        assert_eq!(clip("abc", 0), "");
    }

    #[test]
    fn an_approval_gets_its_own_box_and_leaves_the_input_alone() {
        let mut app = sample_app();
        for c in "half a thought".chars() {
            app.insert_char(c);
        }
        app.pending_approval = Some(ApprovalRequest {
            tool_name: "write_file".into(),
            category: "write",
            arguments: r#"{"filepath":"/tmp/a.txt"}"#.into(),
        });

        let out = render_to_string(&app, 70, 22);
        assert!(out.contains("Write to disk"), "{out}");
        assert!(out.contains("write_file"), "{out}");
        assert!(out.contains("/tmp/a.txt"), "{out}");
        // The whole point: what was being typed is still there and still
        // editable while the decision waits.
        assert!(out.contains("half a thought"), "{out}");
    }

    #[test]
    fn the_approval_box_names_the_keys_that_answer_it() {
        // Answering no longer takes over the input, so nothing else would
        // tell you a decision is owed or how to give it.
        let mut app = sample_app();
        app.pending_approval = Some(ApprovalRequest {
            tool_name: "run_terminal_command".into(),
            category: "terminal",
            arguments: r#"{"command":"ls"}"#.into(),
        });
        let out = render_to_string(&app, 78, 22);
        assert!(out.contains("Ctrl-Y allow"), "{out}");
        assert!(out.contains("Ctrl-N deny"), "{out}");
        assert!(out.contains("Terminal command"), "{out}");
    }

    #[test]
    fn the_keybinding_row_answers_the_question_the_box_raises() {
        let mut app = sample_app();
        let idle = render_to_string(&app, 78, 22);
        assert!(!idle.contains("Ctrl-Y"), "{idle}");

        app.pending_approval = Some(ApprovalRequest {
            tool_name: "write_file".into(),
            category: "write",
            arguments: "{}".into(),
        });
        let waiting = render_to_string(&app, 78, 22);
        assert!(waiting.contains("Ctrl-Y allow"), "{waiting}");
        assert!(waiting.contains("Ctrl-N deny"), "{waiting}");
    }

    #[test]
    fn the_approval_sits_between_the_transcript_and_the_input() {
        let mut app = sample_app();
        for c in "typing on".chars() {
            app.insert_char(c);
        }
        app.pending_approval = Some(ApprovalRequest {
            tool_name: "write_file".into(),
            category: "write",
            arguments: "{}".into(),
        });
        let out = render_to_string(&app, 70, 22);

        let reply_at = out.find("hi there").expect("transcript shown");
        let approval_at = out.find("Write to disk").expect("approval shown");
        let input_at = out.find("typing on").expect("input shown");
        assert!(reply_at < approval_at, "{out}");
        assert!(approval_at < input_at, "{out}");
    }

    #[test]
    fn approval_prompt_stays_visible_when_a_field_is_too_long_to_fit_one_row() {
        // A `content` value long enough to wrap. The box is sized from the
        // wrapped height, so the tail of the value stays on screen instead
        // of falling below the bottom edge — it is the half of the request
        // you most need to read before allowing it.
        let mut app = sample_app();
        app.pending_approval = Some(ApprovalRequest {
            tool_name: "write_file".into(),
            category: "write",
            arguments: format!(
                r#"{{"filepath":"/tmp/a.txt","content":"{}"}}"#,
                "x".repeat(90)
            ),
        });
        let out = render_to_string(&app, 80, 24);
        // The value wraps onto a second row, and that row is inside the box.
        assert!(out.contains(&"x".repeat(40)), "{out}");
        assert!(out.contains("Write to disk"), "{out}");
    }

    #[test]
    fn tool_calls_render_with_status() {
        let mut app = App::new("m".to_string(), None, "id".to_string());
        app.transcript.push(TranscriptItem::ToolCall {
            name: "read_file".into(),
            arguments: r#"{"filepath":"a.rs"}"#.into(),
            status: ToolStatus::Done {
                result: r#"{"success":true}"#.into(),
            },
        });
        let out = render_to_string(&app, 70, 12);
        assert!(out.contains("read_file"), "{out}");
        assert!(out.contains('✓'), "{out}");
    }

    #[test]
    fn tool_call_gutter_is_generic_and_status_trails_the_line() {
        let mut app = App::new("m".to_string(), None, "id".to_string());
        app.transcript.push(TranscriptItem::ToolCall {
            name: "read_file".into(),
            arguments: r#"{"filepath":"a.rs"}"#.into(),
            status: ToolStatus::Done {
                result: r#"{"success":true}"#.into(),
            },
        });
        let out = render_to_string(&app, 70, 12);
        let row = out
            .lines()
            .find(|l| l.contains("read_file"))
            .expect("header row shown");
        assert!(row.trim_start().starts_with("🔨  read_file"), "{row:?}");
        assert!(row.trim_end().ends_with('✓'), "{row:?}");
    }

    #[test]
    fn a_running_tool_call_has_no_trailing_marker() {
        let mut app = App::new("m".to_string(), None, "id".to_string());
        app.transcript.push(TranscriptItem::ToolCall {
            name: "run_terminal_command".into(),
            arguments: r#"{"command":"cargo build"}"#.into(),
            status: ToolStatus::Running,
        });
        let out = render_to_string(&app, 70, 12);
        let row = out
            .lines()
            .find(|l| l.contains("run_terminal_command"))
            .expect("header row shown");
        assert!(
            row.trim_start().starts_with("🔨  run_terminal_command"),
            "{row:?}"
        );
        assert!(!row.contains('▸'), "{row:?}");
    }

    #[test]
    fn tool_call_header_wraps_under_the_gutter() {
        let mut app = App::new("m".to_string(), None, "id".to_string());
        app.transcript.push(TranscriptItem::ToolCall {
            name: "a_pretty_long_tool_name_that_should_wrap_around".into(),
            arguments: "{}".into(),
            status: ToolStatus::Running,
        });
        let out = render_to_string(&app, 30, 14);
        let row = out
            .lines()
            .position(|l| l.trim_start().starts_with("🔨  a_pretty"))
            .expect("header row shown");
        let continuation = out.lines().nth(row + 1).expect("continuation row");
        // 3 columns, not the usual 2 — "🔨 " is double-width, one column
        // wider than the other markers' gutter.
        assert!(
            continuation.starts_with("   ") && !continuation.trim().is_empty(),
            "{continuation:?}"
        );
    }

    #[test]
    fn verbose_field_values_wrap_under_themselves_not_the_label() {
        let mut app = App::new("m".to_string(), None, "id".to_string());
        app.verbose = true;
        app.transcript.push(TranscriptItem::ToolCall {
            name: "write_file".into(),
            arguments:
                r#"{"content":"a value long enough that it should wrap onto a second row here"}"#
                    .into(),
            status: ToolStatus::Done {
                result: "{}".into(),
            },
        });
        let out = render_to_string(&app, 40, 16);
        let label_row = out
            .lines()
            .position(|l| l.contains("content"))
            .expect("field label shown");
        let continuation = out.lines().nth(label_row + 1).expect("continuation row");
        // Indented past the label's own width, not just the 2-column
        // message gutter, and not empty.
        assert!(
            continuation.starts_with("     ") && !continuation.trim().is_empty(),
            "{continuation:?}"
        );
    }

    #[test]
    fn tool_status_names_every_tool_and_what_it_may_do() {
        let mut app = App::new("m".to_string(), None, "id".to_string());
        app.transcript.push(TranscriptItem::ToolStatus {
            access: crate::config::ToolAccessSettings::default()
                .with("read", crate::config::ToolAccess::Allow)
                .unwrap()
                .with("run_terminal_command", crate::config::ToolAccess::Never)
                .unwrap(),
            changed: false,
        });
        let out = render_to_string(&app, 70, 16);
        assert!(out.contains("Tools are:"), "{out}");
        // Named individually: which tools a category holds is exactly what
        // the old per-category readout could not tell you.
        assert!(out.contains("read_file"), "{out}");
        assert!(out.contains("write_file"), "{out}");
        assert!(out.contains("run_terminal_command"), "{out}");
        assert!(out.contains("! allow"), "{out}");
        assert!(out.contains("✓ ask"), "{out}");
        assert!(out.contains("✗ never"), "{out}");
    }

    #[test]
    fn session_status_lists_every_setting() {
        let mut app = App::new("m".to_string(), None, "id".to_string());
        app.transcript.push(TranscriptItem::SessionStatus(vec![
            ("Model".to_string(), "openrouter/auto".to_string()),
            ("Temperature".to_string(), "none sent".to_string()),
        ]));

        let out = render_to_string(&app, 70, 14);
        assert!(out.contains("Clanker:"), "{out}");
        assert!(out.contains("Model"), "{out}");
        assert!(out.contains("openrouter/auto"), "{out}");
        assert!(out.contains("none sent"), "{out}");
        // Unlike thinking, this is a direct answer to a question the user
        // just asked, so it shows regardless of verbose.
        assert!(!app.verbose);
    }

    #[test]
    fn thinking_only_shows_when_verbose() {
        let mut app = App::new("m".to_string(), None, "id".to_string());
        app.transcript
            .push(TranscriptItem::Thinking("weighing the options".to_string()));

        let out = render_to_string(&app, 70, 12);
        assert!(!out.contains("weighing the options"), "{out}");

        app.verbose = true;
        let out = render_to_string(&app, 70, 12);
        assert!(out.contains("weighing the options"), "{out}");
    }

    #[test]
    fn tool_call_arguments_and_result_only_show_when_verbose() {
        let mut app = App::new("m".to_string(), None, "id".to_string());
        app.transcript.push(TranscriptItem::ToolCall {
            name: "read_file".into(),
            arguments: r#"{"filepath":"a.rs"}"#.into(),
            status: ToolStatus::Done {
                result: r#"{"success":true}"#.into(),
            },
        });

        let out = render_to_string(&app, 70, 12);
        assert!(out.contains("read_file"), "{out}");
        assert!(!out.contains("filepath"), "{out}");
        assert!(!out.contains("success"), "{out}");

        app.verbose = true;
        let out = render_to_string(&app, 70, 12);
        assert!(out.contains("filepath"), "{out}");
        assert!(out.contains("success"), "{out}");
    }

    #[test]
    fn tool_call_shows_its_file_or_command_even_when_not_verbose() {
        let mut app = App::new("m".to_string(), None, "id".to_string());
        app.transcript.push(TranscriptItem::ToolCall {
            name: "write_file".into(),
            arguments: r#"{"filepath":"src/main.rs","content":"fn main() {}"}"#.into(),
            status: ToolStatus::Running,
        });
        app.transcript.push(TranscriptItem::ToolCall {
            name: "run_terminal_command".into(),
            arguments: r#"{"command":"cargo test"}"#.into(),
            status: ToolStatus::Running,
        });

        let out = render_to_string(&app, 70, 16);
        assert!(out.contains("src/main.rs"), "{out}");
        assert!(!out.contains("fn main"), "{out}");
        assert!(out.contains("cargo test"), "{out}");
    }

    #[test]
    fn renders_without_panicking_at_awkward_sizes() {
        // The layout reserves 3 rows for input and 1 for status; a terminal
        // smaller than that must clamp rather than underflow.
        let mut app = sample_app();
        app.pending_approval = Some(ApprovalRequest {
            tool_name: "write_file".into(),
            category: "write",
            arguments: "{}".into(),
        });
        // ...and the hint row has to survive a terminal narrower than the
        // shortest command name.
        app.input = "/".to_string();
        app.cursor = 1;
        for (w, h) in [(1, 1), (3, 2), (10, 4), (20, 5), (200, 60)] {
            let _ = render_to_string(&app, w, h);
        }
    }

    fn plain(lines: &[Line]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    #[test]
    fn markdown_strips_markers_and_styles_the_text() {
        let rendered = markdown_lines("**bold** and `code`");
        let text = plain(&rendered).join("");
        assert!(text.contains("bold"), "{text:?}");
        assert!(!text.contains("**"), "markers should be gone: {text:?}");
        assert!(!text.contains('`'), "markers should be gone: {text:?}");

        let styled = rendered
            .iter()
            .flat_map(|l| l.spans.iter())
            .filter(|s| s.style.add_modifier.contains(Modifier::BOLD))
            .count();
        assert!(styled > 0, "bold should carry a style");
    }

    #[test]
    fn code_fences_become_a_language_tag_and_keep_their_code() {
        let rendered = markdown_lines("before\n\n```rust\nfn main() {}\n```\n\nafter");
        let text = plain(&rendered);
        assert!(
            text.iter().all(|l| !l.contains("```")),
            "backticks should not survive: {text:?}"
        );
        assert!(text.iter().any(|l| l.trim() == "rust"), "{text:?}");
        assert!(text.iter().any(|l| l.contains("fn main()")), "{text:?}");
        assert!(text.iter().any(|l| l.contains("before")), "{text:?}");
        assert!(text.iter().any(|l| l.contains("after")), "{text:?}");
    }

    #[test]
    fn an_unlabelled_fence_still_marks_the_block() {
        let text = plain(&markdown_lines("```\nplain code\n```"));
        assert!(text.iter().any(|l| l.trim() == "code"), "{text:?}");
        assert!(text.iter().any(|l| l.contains("plain code")), "{text:?}");
        assert!(text.iter().all(|l| !l.contains("```")), "{text:?}");
    }

    #[test]
    fn lists_survive_rendering() {
        let text = plain(&markdown_lines("- one\n- two\n\n1. first\n2. second")).join("\n");
        for expected in ["one", "two", "first", "second"] {
            assert!(text.contains(expected), "{text:?}");
        }
    }

    #[test]
    fn only_assistant_text_is_treated_as_markdown() {
        let mut app = App::new("m".to_string(), None, "id".to_string());
        app.transcript
            .push(TranscriptItem::User("literal **stars** here".to_string()));
        app.transcript.push(TranscriptItem::Assistant {
            text: "rendered **stars** here".to_string(),
            streaming: false,
            label: Some("m".into()),
        });
        let out = render_to_string(&app, 60, 14);
        // The user's own asterisks are shown as typed; the reply's are not.
        assert!(out.contains("literal **stars**"), "{out}");
        assert!(!out.contains("rendered **stars**"), "{out}");
        assert!(out.contains("rendered stars"), "{out}");
    }

    #[test]
    fn a_streaming_reply_is_left_unformatted_until_it_finishes() {
        let mut app = App::new("m".to_string(), None, "id".to_string());
        // Mid-stream an unclosed fence would otherwise render as a stray tag.
        app.transcript.push(TranscriptItem::Assistant {
            text: "partial **bold".to_string(),
            streaming: true,
            label: Some("m".into()),
        });
        let out = render_to_string(&app, 60, 14);
        assert!(out.contains("partial **bold"), "{out}");
    }

    #[test]
    fn input_wraps_and_tracks_the_caret_together() {
        // The caret must land in the same rows the box actually draws.
        let rows = input_lines("abcdefghij", 4);
        assert_eq!(rows, vec!["abcd", "efgh", "ij"]);
        assert_eq!(input_cursor("abcdefghij", 10, 4), (2, 2));

        // Explicit newlines start a row, including empty ones.
        assert_eq!(input_lines("a\n\nb", 10), vec!["a", "", "b"]);
        assert_eq!(input_cursor("a\n\nb", 4, 10), (2, 1));

        // A segment that exactly fills a row puts the caret on the next.
        assert_eq!(input_lines("abcd", 4), vec!["abcd", ""]);
        assert_eq!(input_cursor("abcd", 4, 4), (1, 0));
    }

    #[test]
    fn the_message_box_grows_with_multiline_input() {
        let mut app = sample_app();
        let single = render_to_string(&app, 40, 16);
        app.input = "one\ntwo\nthree".to_string();
        app.cursor = app.input.len();
        let multi = render_to_string(&app, 40, 16);

        // All three lines are visible simultaneously — which only happens if
        // the box actually grew to 3 rows; at 1 row, cursor-follow scrolling
        // would show just "three" and hide the rest.
        assert!(
            multi.contains("one") && multi.contains("two") && multi.contains("three"),
            "expected the input box to grow:\n{multi}"
        );
        assert_ne!(multi, single);
    }

    #[test]
    fn a_huge_input_cannot_squeeze_out_the_conversation() {
        let mut app = sample_app();
        app.input = (0..200)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        app.cursor = app.input.len();
        // Must render, and the transcript must still get rows.
        let out = render_to_string(&app, 40, 14);
        assert!(out.lines().count() == 14, "{out}");
    }

    #[test]
    fn summarize_flattens_and_truncates() {
        assert_eq!(summarize("a\nb\tc", 10), "a b c");
        assert_eq!(summarize(&"x".repeat(20), 5), "xxxxx…");
        assert_eq!(summarize("  padded  ", 20), "padded");
    }

    #[test]
    fn summarize_counts_characters_not_bytes() {
        // Truncating by byte index here would panic or corrupt the text.
        let text = "é".repeat(20);
        assert_eq!(summarize(&text, 3).chars().count(), 4); // 3 + ellipsis
    }
}
