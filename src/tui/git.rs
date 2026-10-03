//! The changes pane: what is modified in the clanker's repository, beside
//! the conversation, and the diff of whichever file you pick.
//!
//! Chat screen only. It is there for watching what an agent has done to the
//! tree while it does it, which is a question about one clanker's directory —
//! the launch screen has no directory to ask it about.
//!
//! Holds its own state, does its own I/O (it shells out to `git`) and draws
//! itself, the way the picker does. `App` only carries it, so `/diff` can open
//! it from the same dispatch every other command goes through; the event loop
//! is what calls [`GitPane::refresh`], so nothing that folds events into `App`
//! ever waits on a subprocess.

use super::app::{App, TranscriptItem};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};
use std::cell::Cell;
use std::io::{BufRead, BufReader, Read};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime};
use syntect::easy::HighlightLines;
use syntect::highlighting::{Theme, ThemeSet};
use syntect::parsing::SyntaxSet;
use unicode_width::UnicodeWidthStr;

/// How often an open pane re-reads the tree. The agent edits files with
/// nobody pressing anything, and a list that only changed when asked would
/// be showing the tree as it was before the turn you are watching.
const REFRESH: Duration = Duration::from_secs(2);

/// The tree `git diff` compares against in a repository with no commits yet.
/// Every git knows this hash, so the same diff command covers both cases.
const EMPTY_TREE: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";

/// Context lines asked of `git diff` — the most git will take, so a diff
/// comes back as one hunk spanning the whole file, changes marked in place.
/// A merely large number isn't enough: a file longer than it comes back
/// starting part-way down.
const WHOLE_FILE: &str = "--unified=2147483647";

/// Past this a tracked file is shown as its changes only. Diffing loads
/// both sides into git whatever the context, but a whole-file diff of a big
/// file is a big output, and reading it lazily keeps git alive — and holding
/// several times the file's size — for as long as the file is on screen.
/// Changes alone are small, so git finishes and exits. Measured against both
/// sides, so a huge file cut down to nothing still counts as huge.
const WHOLE_FILE_MAX_BYTES: u64 = 16 * 1024 * 1024;

/// How far past the bottom of the view lines are read ahead, so a page down
/// lands on lines already loaded rather than on a gap that fills a tick
/// later.
const LOOKAHEAD: usize = 500;

/// Lines the reader thread sends at a time. One line per message held End
/// on a big file to a few thousand lines a second; batching is what lets it
/// fill in at the rate the disk reads. A batch goes early when the reader
/// has nothing more buffered, so a slow git is never waited on to fill one.
const READ_BATCH: usize = 1024;

/// Batches the reader may have waiting — how far it runs ahead of what the
/// pane has taken before it, and git behind it, block. With [`READ_BUFFER`]
/// that is a couple of megabytes at most, however big the file.
const READ_AHEAD_BATCHES: usize = 32;

/// The reader's buffer. A batch also goes whenever this runs dry, so its
/// size is what a batch from a file on disk typically carries.
const READ_BUFFER: usize = 64 * 1024;

/// Most lines taken off a loader in one pass of the event loop. End on a
/// large file loads it this many at a time, ten passes a second, rather than
/// in one pass that holds every keypress behind it.
const PUMP_BATCH: usize = 10_000;

/// The most lines the pane holds for one file. Loading is lazy, so this is
/// only reached by scrolling (or End) that far into a very large file; it is
/// the ceiling on memory, not on what is normally read.
const MAX_LINES: usize = 200_000;

/// Longest a line is kept, in bytes; the rest is dropped and marked. A line
/// of minified code or data can be megabytes, all on one row the pane would
/// clip at its edge anyway.
const MAX_LINE_BYTES: usize = 1024;

/// Shown in place of a diff whose file was committed or reverted under you.
const NO_LONGER_CHANGED: &str = "(no longer changed)";

/// Lines of the file kept in view above a change that is jumped to, so it
/// arrives with something to read it against.
const CONTEXT: usize = 3;

/// The narrowest the conversation is squeezed to beside the pane.
const MIN_CHAT_WIDTH: u16 = 30;

/// One line of `git status`: its two-letter code and where it lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangedFile {
    /// Index then worktree, exactly as `git status --short` prints them —
    /// `M `, ` M`, `??`, `R ` and so on — so it reads the way git reads.
    pub status: [char; 2],
    /// Relative to the repository root.
    pub path: String,
    /// Where a rename or copy came from, so its diff can show the move
    /// rather than a whole file appearing from nowhere.
    pub orig: Option<String>,
}

impl ChangedFile {
    fn untracked(&self) -> bool {
        self.status == ['?', '?']
    }
}

/// Why there is no list to show.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unavailable {
    /// The clanker's directory isn't inside a repository — the common case,
    /// and not a fault: plenty of clankers work somewhere git doesn't.
    NoRepo(PathBuf),
    /// There is no `git` to ask.
    NoGit(String),
    /// git ran and refused, with the first line of what it said.
    Failed(String),
}

impl Unavailable {
    /// What the pane says, as lines: what is wrong, then what to do.
    fn lines(&self) -> Vec<Line<'static>> {
        let hint = Style::new().dark_gray();
        match self {
            Unavailable::NoRepo(dir) => vec![
                Line::styled(" No git repository here", Style::new().yellow().bold()),
                Line::from(""),
                Line::styled(
                    format!(
                        " {}",
                        super::render::home_relative(&dir.display().to_string())
                    ),
                    hint,
                ),
                Line::styled(" isn't inside a git repository, so there", hint),
                Line::styled(" are no changes to track.", hint),
                Line::from(""),
                Line::styled(" Run `git init` there, then r to refresh.", hint),
            ],
            Unavailable::NoGit(why) => vec![
                Line::styled(" git isn't available", Style::new().yellow().bold()),
                Line::from(""),
                Line::styled(format!(" {why}"), hint),
                Line::styled(" Install git or put it on PATH, then r to refresh.", hint),
            ],
            Unavailable::Failed(why) => vec![
                Line::styled(
                    " git couldn't read this repository",
                    Style::new().red().bold(),
                ),
                Line::from(""),
                Line::styled(format!(" {why}"), hint),
            ],
        }
    }
}

impl std::fmt::Display for Unavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Unavailable::NoRepo(dir) => write!(f, "Not a git repository: {}", dir.display()),
            Unavailable::NoGit(why) | Unavailable::Failed(why) => f.write_str(why),
        }
    }
}

/// The diff on show, and which file it belongs to.
///
/// Loaded as it is scrolled to rather than all at once: git writes the diff
/// into a pipe, a thread turns it into lines, and the pane takes only as many
/// as the view has reached plus [`LOOKAHEAD`]. When the pane stops taking
/// them the channel fills, the thread blocks, and git blocks on the pipe
/// behind it — so a file of any size costs what has been scrolled through,
/// and the event loop never waits on git at all.
#[derive(Debug)]
pub struct Shown {
    pub file: ChangedFile,
    pub lines: Vec<Row>,
    /// The line number the next added or unchanged line has in the file as
    /// it stands. Set by each `@@`, so changes-only diffs number right too.
    next_number: usize,
    /// First line in view.
    pub scroll: usize,
    /// Holding the view at the end while the rest loads in, after End.
    pin_bottom: bool,
    /// The whole file, rather than its changes only — see
    /// [`WHOLE_FILE_MAX_BYTES`].
    whole: bool,
    /// What git said before the first hunk, kept until it is known whether
    /// there is one. `None` once the file itself has started.
    header: Option<Vec<Row>>,
    /// Still reading. `None` once everything has arrived, or the pane has
    /// taken all it will hold.
    loader: Option<Loader>,
    /// The file as it was when this was read, so a refresh re-reads only a
    /// diff whose file has actually changed.
    stamp: Stamp,
    root: PathBuf,
    /// Each change among the lines held: a run of added and removed lines
    /// with nothing unchanged between them. In order, so the ones above and
    /// below the view are a binary search away.
    runs: Vec<Range<usize>>,
    /// How many changes the whole diff has, held or not — what says there
    /// are more below lines that haven't been read yet. Counted by a git of
    /// its own (see [`spawn_count`]), then by the lines once all are in.
    blocks: Option<usize>,
    counting: Option<Counter>,
    /// Scrolling to the first change at or after this line, as soon as it
    /// has been read: on opening a file, and on a jump past what's held.
    seek: Option<usize>,
}

/// One line of the view, and its number in the file as it stands. `None`
/// for a removed line — it has no place in the file any more — and for the
/// pane's own notes and the `@@` between hunks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub number: Option<usize>,
    pub text: String,
    /// The syntax colours of the code after the marker: where each run
    /// ends, as a byte offset into the text past the marker, and its colour.
    /// `None` is the terminal's own foreground. Empty when the line isn't
    /// code, or the file is in no language the highlighter knows.
    pub colors: Vec<(usize, Option<Color>)>,
}

impl Row {
    fn note(text: impl Into<String>) -> Row {
        Row {
            number: None,
            text: text.into(),
            colors: Vec::new(),
        }
    }
}

/// Where the new side of a hunk starts: `5` from `@@ -3,4 +5,6 @@`.
fn hunk_start(line: &str) -> Option<usize> {
    let new = line
        .split_whitespace()
        .find_map(|word| word.strip_prefix('+'))?;
    new.split(',').next()?.parse().ok()
}

/// A file's entry in the list and its age and size on disk — enough to tell
/// it changed without reading it.
type Stamp = (ChangedFile, Option<(SystemTime, u64)>);

fn stamp(root: &Path, file: &ChangedFile) -> Stamp {
    let disk = std::fs::metadata(root.join(&file.path))
        .ok()
        .and_then(|meta| Some((meta.modified().ok()?, meta.len())));
    (file.clone(), disk)
}

/// The far end of a reader thread: lines arrive here as they are read.
#[derive(Debug)]
struct Loader {
    lines: Receiver<Vec<Row>>,
    /// The git writing them, if it is git. Killed when the loader goes: a
    /// diff closed half-read leaves git blocked on a pipe nobody will drain.
    child: Option<Child>,
}

impl Drop for Loader {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Shown {
    /// Starts reading `file`'s diff. Returns at once; the lines arrive as
    /// [`Shown::pump`] takes them.
    fn load(root: &Path, base: &str, file: &ChangedFile) -> Shown {
        let mut shown = Shown::note(root, file, None);
        // Untracked is all additions, so there is nothing to diff: it is
        // read straight off the disk, which costs nothing however big it is
        // — git would load the whole thing to say the same.
        if file.untracked() {
            match std::fs::File::open(root.join(&file.path)) {
                Ok(disk) if is_binary(&disk) => shown.lines.push(Row::note("(binary file)")),
                Ok(disk) => shown.loader = Some(spawn_reader(disk, "+", None, &file.path)),
                Err(why) => shown
                    .lines
                    .push(Row::note(format!("Couldn't read it: {why}"))),
            }
            return shown;
        }
        let size = file_size(root, file);
        shown.whole = size <= WHOLE_FILE_MAX_BYTES;
        if !shown.whole {
            shown.lines.push(Row::note(format!(
                "({} — too large to show whole, so only its changes)",
                human_size(size)
            )));
        }
        match spawn_diff(root, base, file, shown.whole) {
            Ok(loader) => {
                shown.header = Some(Vec::new());
                shown.loader = Some(loader);
                // A file added or deleted outright is one change, the whole
                // of it, so there is nothing to count — and counting would
                // stream all of it through git a second time.
                if !file.status.iter().any(|s| matches!(s, 'A' | 'D')) {
                    shown.counting = spawn_count(root, base, file);
                }
            }
            Err(why) => shown.lines.push(Row::note(why)),
        }
        shown
    }

    /// A settled view holding `note` and nothing to load.
    fn note(root: &Path, file: &ChangedFile, note: Option<&str>) -> Shown {
        Shown {
            file: file.clone(),
            lines: note.into_iter().map(Row::note).collect(),
            next_number: 1,
            scroll: 0,
            pin_bottom: false,
            whole: true,
            header: None,
            loader: None,
            stamp: stamp(root, file),
            root: root.to_path_buf(),
            runs: Vec::new(),
            blocks: None,
            counting: None,
            seek: None,
        }
    }

    pub fn loading(&self) -> bool {
        self.loader.is_some()
    }

    /// Still looking for the first change, on a file just opened.
    pub fn opening(&self) -> bool {
        self.seek == Some(0)
    }

    /// The first line in view, once a view `rows` tall has been held to the
    /// lines there are.
    fn top(&self, rows: usize) -> usize {
        self.scroll.min(self.lines.len().saturating_sub(rows))
    }

    /// How many changes lie wholly above the view, and wholly below it.
    /// Below counts the lines not read yet too, once git has said how many
    /// changes there are in all.
    pub fn beyond(&self, rows: usize) -> (usize, usize) {
        let top = self.top(rows);
        let above = self.runs.partition_point(|run| run.end <= top);
        let started = self.runs.partition_point(|run| run.start < top + rows);
        let all = self.blocks.unwrap_or(self.runs.len());
        (above, all.saturating_sub(started))
    }

    /// How many lines the view at `scroll` wants held.
    fn wanted(&self, rows: usize) -> usize {
        if self.pin_bottom || self.seek.is_some() {
            MAX_LINES
        } else {
            self.scroll + rows + LOOKAHEAD
        }
    }

    /// Takes what the loader has ready, until `wanted` lines are held. Never
    /// waits — what isn't ready yet is left for the next pass. Returns
    /// whether anything changed.
    fn pump(&mut self, wanted: usize) -> bool {
        let wanted = wanted.min(MAX_LINES);
        let mut changed = false;
        let mut taken = 0;
        while self.lines.len() < wanted && taken < PUMP_BATCH {
            let Some(loader) = &self.loader else {
                break;
            };
            match loader.lines.try_recv() {
                Ok(batch) => {
                    taken += batch.len();
                    for line in batch {
                        self.take(line);
                    }
                    changed = true;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.finish();
                    changed = true;
                }
            }
        }
        if self.loading() && self.lines.len() >= MAX_LINES {
            self.loader = None;
            // A batch can carry it past.
            self.lines.truncate(MAX_LINES);
            self.runs.retain(|run| run.start < MAX_LINES);
            if let Some(run) = self.runs.last_mut() {
                run.end = run.end.min(MAX_LINES);
            }
            self.lines
                .push(Row::note(format!("… stopped at {MAX_LINES} lines")));
            changed = true;
        }
        if let Some(counted) = self.counting.as_mut().and_then(Counter::take) {
            self.counting = None;
            if let Ok(blocks) = counted {
                // The lines' own count, if they are all in already, is the
                // one that matches what's on screen.
                self.blocks.get_or_insert(blocks);
                changed = true;
            }
        }
        if let Some(from) = self.seek {
            let next = self.runs.partition_point(|run| run.start < from);
            if let Some(run) = self.runs.get(next) {
                self.scroll = run.start.saturating_sub(CONTEXT);
                self.seek = None;
                changed = true;
            } else if !self.loading() || self.blocks.is_some_and(|all| self.runs.len() >= all) {
                // There isn't one: the view stays where it was.
                self.seek = None;
                changed = true;
            }
        }
        changed
    }

    /// One line from the reader. The header is dropped — the pane's rule
    /// already names the file, and the list says how it changed. So is the
    /// `@@` after it in a whole-file diff, where it says nothing the file
    /// below it doesn't; between hunks, each one says where it is.
    fn take(&mut self, mut row: Row) {
        match &mut self.header {
            Some(header) if !row.text.starts_with("@@") => header.push(row),
            Some(_) => {
                self.header = None;
                self.next_number = hunk_start(&row.text).unwrap_or(1);
                if !self.whole {
                    self.push(row);
                }
            }
            None if row.text.starts_with("@@") => {
                self.next_number = hunk_start(&row.text).unwrap_or(self.next_number);
                self.push(row);
            }
            // Removed, or git's `\ No newline at end of file`: neither is a
            // line of the file as it stands.
            None if row.text.starts_with(['-', '\\']) => self.push(row),
            None => {
                row.number = Some(self.next_number);
                self.next_number += 1;
                self.push(row);
            }
        }
    }

    /// Holds a line of the diff, keeping count of where the changes are. A
    /// `\ No newline` belongs to the change it follows, as git means it to:
    /// it sits between the removed last line and the added one.
    fn push(&mut self, row: Row) {
        let at = self.lines.len();
        let changed = row.text.starts_with(['+', '-']);
        match self.runs.last_mut() {
            Some(run) if run.end == at && (changed || row.text.starts_with('\\')) => {
                run.end = at + 1
            }
            _ if changed => self.runs.push(at..at + 1),
            _ => {}
        }
        self.lines.push(row);
    }

    /// Everything has arrived. A diff that never reached a hunk — a pure
    /// rename, a mode change — still has a file to show, so the file is read
    /// as it stands, the same lazy way. Failing that (it is binary, or
    /// gone), git's own header says what changed.
    fn finish(&mut self) {
        self.loader = None;
        let Some(header) = self.header.take() else {
            if self.lines.is_empty() {
                self.lines.push(Row::note("(empty file)"));
            }
            self.blocks = Some(self.runs.len());
            return;
        };
        // No hunk, so nothing to find in the file read in its place.
        self.blocks = Some(0);
        let binary = header
            .iter()
            .any(|row| row.text.starts_with("Binary files"));
        // git says "Binary files differ" only when content differs: a
        // binary renamed or re-moded as it was has a header that never says
        // so, and the file itself is the one left to ask.
        match std::fs::File::open(self.root.join(&self.file.path)) {
            Ok(file) if !binary && self.whole && !is_binary(&file) => {
                self.loader = Some(spawn_reader(file, " ", None, &self.file.path))
            }
            _ if header.is_empty() => self.lines.push(Row::note("(no textual changes)")),
            _ => self.lines = header,
        }
    }
}

/// The tools that edit the file their `filepath` names. Reads aren't among
/// them: the gutter marks where the agent is changing things, not looking.
const WRITE_TOOLS: [&str; 2] = ["write_file", "replace_in_file"];

/// How long the gutter's mark stays once the agent has moved on from the
/// file. A write is over in milliseconds and the next step often follows
/// within one, so without this the mark could come and go unseen.
const MARK_HOLD: Duration = Duration::from_secs(1);

/// Where the agent is at work, as the list's gutter shows it: the file it is
/// writing, as the tool was given it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Activity {
    pub filepath: String,
    /// Stopped at an approval, which the chat says with `? waiting` rather
    /// than its animation: nothing is moving until you answer.
    pub waiting: bool,
}

/// The file the agent is writing, if writing is the last thing it did. Any
/// step after the write — another tool, thinking, a reply — means it has
/// moved on, so the mark goes with it rather than lingering on a file it is
/// done with. A tool call only arrives whole, so this is as near to the
/// moment of the edit as the transcript can say.
pub fn activity(app: &App) -> Option<Activity> {
    if !app.busy {
        return None;
    }
    // Only the turn under way counts: a write from an earlier turn is never
    // the one being made now.
    let last = app
        .transcript
        .iter()
        .rev()
        .take_while(|item| !matches!(item, TranscriptItem::User(_)))
        .find(|item| {
            matches!(
                item,
                TranscriptItem::ToolCall { .. }
                    | TranscriptItem::Thinking(_)
                    | TranscriptItem::Assistant { .. }
            )
        })?;
    let filepath = match last {
        TranscriptItem::ToolCall {
            name, arguments, ..
        } if WRITE_TOOLS.contains(&name.as_str()) => {
            // Only the path is taken: a write's arguments carry the whole
            // file, and this runs on every pass of the event loop.
            #[derive(serde::Deserialize)]
            struct Target {
                filepath: String,
            }
            serde_json::from_str::<Target>(arguments)
                .ok()
                .map(|target| target.filepath)
        }
        _ => None,
    }?;
    Some(Activity {
        filepath,
        waiting: app.pending_approval.is_some(),
    })
}

/// `filepath` as a tool resolves it — from `dir`, the clanker's directory,
/// which is the process's own — made relative to the repository `root`, the
/// way `git status` names files. `None` when it is outside the repository.
fn repo_relative(root: &Path, dir: &Path, filepath: &str) -> Option<String> {
    let path = dir.join(filepath);
    // The file may not exist yet — a write about to create it — so it is
    // resolved through its directory, which `..` and symlinks need.
    let resolved = path.canonicalize().ok().or_else(|| {
        let parent = path.parent()?.canonicalize().ok()?;
        Some(parent.join(path.file_name()?))
    })?;
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let relative = resolved.strip_prefix(&root).ok()?;
    Some(
        relative
            .components()
            .map(|part| part.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/"),
    )
}

#[derive(Debug)]
pub struct GitPane {
    /// Whether keys go here rather than to the input box.
    pub focused: bool,
    pub branch: Option<String>,
    pub files: Vec<ChangedFile>,
    /// The row the list cursor is on.
    pub cursor: usize,
    pub shown: Option<Shown>,
    /// A fresh read of the shown file after it changed, kept off screen until
    /// it has caught up with the view — swapping it straight in would flash
    /// the top of the file at you every time the agent saved it.
    reloading: Option<Shown>,
    /// Why there is no list — not a repository, or no `git` to ask.
    pub error: Option<Unavailable>,
    /// Where to ask from: the clanker's own directory.
    dir: PathBuf,
    /// The repository root and what diffs compare against, from the last
    /// read that found a repository.
    repo: Option<(PathBuf, &'static str)>,
    /// When the last read was started. `None` until the first, which is
    /// what makes opening the pane a state change only — the event loop
    /// does the reading.
    refreshed_at: Option<Instant>,
    /// A read under way. `git status` is quick in a small tree and seconds in
    /// a big one, so it runs on a thread of its own, never in the event loop.
    reading: Option<Receiver<Result<Tree, Unavailable>>>,
    /// Whether a read has landed yet, so an empty list before then isn't
    /// mistaken for a clean tree.
    listed: bool,
    /// Where the agent is at work, as last told — see [`activity`] — or
    /// was, while [`MARK_HOLD`] keeps the mark up after it moves on.
    activity: Option<Activity>,
    /// When `activity` was last set, from which the hold is counted.
    marked_at: Instant,
    /// That file's path as the list names it, once resolved against the
    /// repository. Worked out when either changes, not every frame.
    active: Option<String>,
    /// How many diff rows the last frame had room for. Written by [`draw`]
    /// so scrolling can stop at the last screenful rather than run on into
    /// presses that move nothing; a `Cell` because drawing takes the pane
    /// by shared reference, like everything else it draws.
    diff_rows: Cell<usize>,
}

impl GitPane {
    pub fn new(dir: PathBuf) -> Self {
        GitPane {
            focused: true,
            branch: None,
            files: Vec::new(),
            cursor: 0,
            shown: None,
            reloading: None,
            error: None,
            dir,
            repo: None,
            refreshed_at: None,
            reading: None,
            listed: false,
            activity: None,
            marked_at: Instant::now(),
            active: None,
            diff_rows: Cell::new(1),
        }
    }

    /// Whether it is time to re-read the tree.
    pub fn due(&self) -> bool {
        self.refreshed_at.is_none_or(|at| at.elapsed() >= REFRESH)
    }

    /// Tells the pane where the agent is at work. Returns whether the list
    /// changes for it. A mark stays for at least [`MARK_HOLD`] when the
    /// agent moves on, but a write to another file replaces it at once:
    /// holding the old one there would show the agent somewhere it isn't.
    pub fn set_activity(&mut self, activity: Option<Activity>) -> bool {
        if activity == self.activity || activity.is_none() && self.marked_at.elapsed() < MARK_HOLD {
            return false;
        }
        self.marked_at = Instant::now();
        self.activity = activity;
        self.resolve_activity();
        true
    }

    fn resolve_activity(&mut self) {
        self.active = match (&self.activity, &self.repo) {
            (Some(activity), Some((root, _))) => repo_relative(root, &self.dir, &activity.filepath),
            _ => None,
        };
    }

    /// Asks for a re-read on the next pass of the event loop.
    pub fn invalidate(&mut self) {
        self.refreshed_at = None;
    }

    /// Starts a re-read of the list, unless one is already under way; it is
    /// applied by [`GitPane::pump`] when it lands. Returns whether anything on
    /// screen changed, so an idle pane doesn't cost a frame every refresh.
    pub fn refresh(&mut self) -> bool {
        if self.reading.is_none() {
            self.refreshed_at = Some(Instant::now());
            let (sender, reading) = mpsc::sync_channel(1);
            let dir = self.dir.clone();
            let spawned = std::thread::Builder::new()
                .name("git-status".to_string())
                .spawn(move || {
                    let _ = sender.send(read_tree(&dir));
                });
            if spawned.is_ok() {
                self.reading = Some(reading);
            }
        }
        self.pump()
    }

    /// A read of the tree, landed: the list, and the diff on show re-read if
    /// its file changed. Returns whether anything on screen changed.
    fn apply(&mut self, tree: Result<Tree, Unavailable>) -> bool {
        let before = (self.branch.clone(), self.files.clone(), self.error.clone());
        let mut changed = !self.listed;
        self.listed = true;

        match tree {
            Ok(Tree {
                root,
                branch,
                base,
                files,
            }) => {
                // The cursor follows its file, not its row: the list
                // reorders as files change, and a cursor that stayed put
                // would land on something you never chose.
                let held = self.files.get(self.cursor).map(|f| f.path.clone());
                self.files = files;
                self.branch = branch;
                self.error = None;
                self.cursor = held
                    .and_then(|path| self.files.iter().position(|f| f.path == path))
                    .unwrap_or(self.cursor)
                    .min(self.files.len().saturating_sub(1));

                if let Some(shown) = &self.shown {
                    // Measured against a re-read already under way, so one
                    // that hasn't caught up yet isn't started over.
                    let latest = self.reloading.as_ref().unwrap_or(shown);
                    match self.files.iter().find(|f| f.path == shown.file.path) {
                        Some(file) if stamp(&root, file) == latest.stamp => {}
                        Some(file) => self.reloading = Some(Shown::load(&root, base, file)),
                        // Kept on screen with a note rather than closed: it
                        // was committed or reverted under you, and the diff
                        // vanishing outright reads as a fault.
                        None if shown.lines != [Row::note(NO_LONGER_CHANGED)] => {
                            let file = shown.file.clone();
                            self.shown = Some(Shown::note(&root, &file, Some(NO_LONGER_CHANGED)));
                            self.reloading = None;
                            changed = true;
                        }
                        None => {}
                    }
                }
                self.repo = Some((root, base));
            }
            Err(why) => {
                self.files.clear();
                self.branch = None;
                self.shown = None;
                self.reloading = None;
                self.repo = None;
                self.cursor = 0;
                self.error = Some(why);
            }
        }

        let active = self.active.clone();
        self.resolve_activity();
        changed
            || active != self.active
            || before != (self.branch.clone(), self.files.clone(), self.error.clone())
    }

    /// Takes whatever the tree read and the diff's reader have ready, as far
    /// as the view needs. Cheap enough to call on every pass of the event
    /// loop, which is what keeps lines arriving while nothing else is
    /// happening. Returns whether anything on screen changed.
    pub fn pump(&mut self) -> bool {
        let rows = self.diff_rows.get();
        let mut changed = false;
        if let Some(reading) = &self.reading {
            match reading.try_recv() {
                Ok(tree) => {
                    self.reading = None;
                    changed |= self.apply(tree);
                }
                Err(TryRecvError::Empty) => {}
                // Its thread died without answering; the next refresh asks
                // again.
                Err(TryRecvError::Disconnected) => self.reading = None,
            }
        }
        if let Some(shown) = &mut self.shown {
            changed |= shown.pump(shown.wanted(rows));
            if shown.pin_bottom {
                shown.scroll = shown.lines.len().saturating_sub(rows);
            }
        }
        if let (Some(shown), Some(next)) = (&self.shown, &mut self.reloading) {
            next.scroll = shown.scroll;
            next.pin_bottom = shown.pin_bottom;
            next.seek = shown.seek;
            next.pump(next.wanted(rows));
            if !next.loading() || next.lines.len() >= next.scroll + rows {
                self.shown = self.reloading.take();
                changed = true;
            }
        }
        changed
    }
    /// Moves the list cursor, stopping at the ends like the model browser.
    pub fn move_cursor(&mut self, down: bool) {
        let last = self.files.len().saturating_sub(1);
        self.cursor = if down {
            (self.cursor + 1).min(last)
        } else {
            self.cursor.saturating_sub(1)
        };
    }

    /// Shows the diff of the file under the cursor, opened at its first
    /// change rather than its first line.
    pub fn select(&mut self) {
        let Some(file) = self.files.get(self.cursor) else {
            return;
        };
        // Every file listed came from a read that found the repository.
        let Some((root, base)) = &self.repo else {
            return;
        };
        self.reloading = None;
        self.shown = Some(Shown {
            seek: Some(0),
            ..Shown::load(root, base, file)
        });
        self.pump();
    }

    /// Brings the next change below the top of the view, or the one before
    /// it, up to [`CONTEXT`] lines from the top. A next change that hasn't
    /// been read yet is gone to once it has.
    pub fn jump(&mut self, forward: bool) {
        let rows = self.diff_rows.get();
        let Some(shown) = &mut self.shown else {
            return;
        };
        shown.pin_bottom = false;
        shown.seek = None;
        // Where a change already jumped to sits, so it isn't jumped to again.
        let here = shown.top(rows) + CONTEXT;
        let to = if forward {
            let next = shown.runs.partition_point(|run| run.start <= here);
            if next == shown.runs.len() && shown.loading() {
                shown.seek = Some(here + 1);
            }
            shown.runs.get(next)
        } else {
            let before = shown.runs.partition_point(|run| run.start < here);
            before.checked_sub(1).and_then(|i| shown.runs.get(i))
        };
        if let Some(run) = to {
            shown.scroll = run.start.saturating_sub(CONTEXT);
        }
        self.pump();
    }

    /// Scrolls the diff by `delta` lines, reading ahead first so the lines
    /// are there to scroll onto, and holding the last screenful at the
    /// bottom rather than scrolling it away.
    pub fn scroll(&mut self, delta: isize) {
        let rows = self.diff_rows.get();
        if let Some(shown) = &mut self.shown {
            shown.pin_bottom = false;
            shown.seek = None;
            let from = shown.top(rows);
            let target = from.saturating_add_signed(delta);
            shown.pump(target.saturating_add(rows + LOOKAHEAD));
            shown.scroll = target.min(shown.lines.len().saturating_sub(rows));
        }
    }
    /// A screenful, less a line of overlap to keep your place by.
    pub fn page(&self) -> isize {
        self.diff_rows.get().saturating_sub(1).max(1) as isize
    }

    /// The top, or the end — which, for a file still loading, holds the
    /// view at the end while the rest comes in.
    pub fn scroll_to(&mut self, bottom: bool) {
        match &mut self.shown {
            Some(shown) if bottom => {
                shown.pin_bottom = true;
                shown.seek = None;
                self.pump();
            }
            _ => self.scroll(isize::MIN),
        }
    }
}

/// Runs git in `dir`, returning what it printed, or the first line of its
/// complaint when it fails.
fn git(dir: &Path, args: &[&str]) -> Result<String, Unavailable> {
    let output = git_command(dir)
        .args(args)
        .output()
        .map_err(|e| Unavailable::NoGit(format!("Couldn't run git: {e}")))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(Unavailable::Failed(
            stderr
                .trim()
                .lines()
                .next()
                .unwrap_or("git failed")
                .to_string(),
        ))
    }
}

fn git_command(dir: &Path) -> Command {
    let mut command = Command::new("git");
    command
        .args(["-c", "core.quotepath=off", "-c", "color.ui=never"])
        .current_dir(dir)
        // Never a credential prompt or a pager: this runs under a TUI that
        // owns the terminal, and either would wait on a keyboard it can't see.
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_PAGER", "cat")
        // `git status` refreshes the index's stat cache when it can, which
        // means taking `index.lock` — every two seconds, beside an agent
        // running git of its own, whose `git commit` then fails on the lock.
        // This is the switch editors use to look without touching.
        .env("GIT_OPTIONAL_LOCKS", "0")
        .stdin(Stdio::null());
    command
}

fn repo_root(dir: &Path) -> Result<PathBuf, Unavailable> {
    match git(dir, &["rev-parse", "--show-toplevel"]) {
        Ok(out) => Ok(PathBuf::from(out.trim_end())),
        // A missing git is its own answer; any refusal here is git saying
        // this isn't a repository, in one of several wordings.
        Err(Unavailable::Failed(_)) => Err(Unavailable::NoRepo(dir.to_path_buf())),
        Err(other) => Err(other),
    }
}

/// What a read of the repository found.
#[derive(Debug)]
struct Tree {
    root: PathBuf,
    branch: Option<String>,
    /// What diffs compare against — see [`base`].
    base: &'static str,
    files: Vec<ChangedFile>,
}

/// The repository root, the branch, and every changed file — untracked ones
/// included, since a file the agent just created is a change like any other.
fn read_tree(dir: &Path) -> Result<Tree, Unavailable> {
    let root = repo_root(dir)?;
    // `symbolic-ref` rather than `rev-parse --abbrev-ref`, which fails on a
    // branch with no commits yet. A detached HEAD has no branch to name, so
    // it is named by its commit; either way a missing name is not worth
    // failing the whole pane over.
    let branch = git(&root, &["symbolic-ref", "--short", "-q", "HEAD"])
        .or_else(|_| git(&root, &["rev-parse", "--short", "HEAD"]))
        .ok()
        .map(|b| b.trim().to_string())
        .filter(|b| !b.is_empty());
    let status = git(
        &root,
        &["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )?;
    Ok(Tree {
        base: base(&root),
        branch,
        files: parse_status(&status),
        root,
    })
}

/// Parses `git status --porcelain=v1 -z`. NUL-separated, so paths arrive
/// unquoted whatever they contain, and a rename's source follows it as an
/// entry of its own.
fn parse_status(raw: &str) -> Vec<ChangedFile> {
    let mut entries = raw.split('\0').filter(|e| !e.is_empty());
    let mut files = Vec::new();
    while let Some(entry) = entries.next() {
        let mut chars = entry.chars();
        let (Some(x), Some(y)) = (chars.next(), chars.next()) else {
            continue;
        };
        let Some(path) = entry.get(3..) else {
            continue;
        };
        let orig = if matches!(x, 'R' | 'C') || matches!(y, 'R' | 'C') {
            entries.next().map(str::to_string)
        } else {
            None
        };
        files.push(ChangedFile {
            status: [x, y],
            path: path.to_string(),
            orig,
        });
    }
    files
}

/// Starts `git diff` on one file against the last commit — staged and
/// unstaged together, since the question is what the tree holds now, not
/// what has been added to the index — with a reader thread taking its output.
fn spawn_diff(root: &Path, base: &str, file: &ChangedFile, whole: bool) -> Result<Loader, String> {
    let context = if whole { WHOLE_FILE } else { "--unified=3" };
    let mut child = diff_command(root, base, file, context)
        .spawn()
        .map_err(|e| format!("Couldn't run git: {e}"))?;
    let stdout = child.stdout.take().ok_or("git gave no output to read")?;
    Ok(spawn_reader(stdout, "", Some(child), &file.path))
}

/// `git diff` of one file with `context` lines around each change, its
/// output piped back.
fn diff_command(root: &Path, base: &str, file: &ChangedFile, context: &str) -> Command {
    let mut command = git_command(root);
    command
        .args(["diff", "--no-ext-diff", context, "-M", base, "--"])
        .args(&file.orig)
        .arg(&file.path)
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    command
}

/// Counts the changes in `file`'s diff on a thread of its own: the diff
/// with no context around them, where every hunk is one change. Small next
/// to the whole-file diff, which would have to be read to the end to say
/// the same — and is read only as far as it's scrolled.
fn spawn_count(root: &Path, base: &str, file: &ChangedFile) -> Option<Counter> {
    let mut child = diff_command(root, base, file, "--unified=0").spawn().ok()?;
    let stdout = child.stdout.take()?;
    let (sender, count) = mpsc::sync_channel(1);
    let _ = std::thread::Builder::new()
        .name("git-count".to_string())
        .spawn(move || {
            let mut reader = BufReader::with_capacity(READ_BUFFER, stdout);
            let mut hunks = 0;
            while let Some(line) = read_line_capped(&mut reader) {
                hunks += usize::from(line.starts_with("@@"));
            }
            let _ = sender.send(hunks);
        });
    Some(Counter { count, child })
}

/// The far end of [`spawn_count`]. Holds the git, so a file switched away
/// from mid-count stops it rather than leaving it to stream to the end.
#[derive(Debug)]
struct Counter {
    count: Receiver<usize>,
    child: Child,
}

impl Counter {
    /// The count, once git has finished — and only if it finished well: one
    /// that failed printed nothing, and nothing would read as no changes.
    fn take(&mut self) -> Option<Result<usize, ()>> {
        match self.count.try_recv() {
            // Its output has ended, so git has exited or is about to.
            Ok(hunks) => Some(match self.child.wait() {
                Ok(status) if status.success() => Ok(hunks),
                _ => Err(()),
            }),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err(())),
        }
    }
}

impl Drop for Counter {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Reads `source` line by line on a thread of its own, each line led by
/// `prefix` and coloured as the language `path` is in. The channel is
/// bounded, so the thread only ever runs [`READ_AHEAD_BATCHES`] batches ahead
/// of what the pane has taken.
///
/// Highlighting happens here rather than when a line is drawn because it
/// can't happen anywhere else: a line's colours depend on every line before
/// it (it may be inside a comment or a string that opened pages ago), and
/// this is the one place that sees them all in order. It also keeps the
/// cost off the event loop.
fn spawn_reader(
    source: impl Read + Send + 'static,
    prefix: &'static str,
    child: Option<Child>,
    path: &str,
) -> Loader {
    let (sender, lines) = mpsc::sync_channel(READ_AHEAD_BATCHES);
    let path = path.to_string();
    // A thread that can't be started leaves the sender dropped, which the
    // pane reads as a diff that ended at once — not worth a panic.
    let _ = std::thread::Builder::new()
        .name("git-diff".to_string())
        .spawn(move || {
            let mut painter = Painter::for_path(&path);
            // A file read off disk is code from its first line; a diff's
            // code starts after its header, at the first `@@`.
            let mut in_body = !prefix.is_empty();
            let mut reader = BufReader::with_capacity(READ_BUFFER, source);
            let mut batch = Vec::with_capacity(READ_BATCH);
            while let Some(line) = read_line_capped(&mut reader) {
                let text = format!("{prefix}{line}");
                let colors = match &mut painter {
                    Some(painter) if in_body => painter.paint(&text),
                    _ => Vec::new(),
                };
                in_body |= text.starts_with("@@");
                batch.push(Row {
                    number: None,
                    text,
                    colors,
                });
                // Sent when full, or when what's buffered has run out and
                // the next read might wait — lines already read shouldn't.
                if batch.len() >= READ_BATCH || reader.buffer().is_empty() {
                    let full = std::mem::replace(&mut batch, Vec::with_capacity(READ_BATCH));
                    // The pane let go of it: a different file was picked.
                    if sender.send(full).is_err() {
                        return;
                    }
                }
            }
            if !batch.is_empty() {
                let _ = sender.send(batch);
            }
        });
    Loader { lines, child }
}

/// The syntaxes and the theme, loaded once, on the first diff anyone opens.
/// Its own copy of the data `tui-markdown` colours the chat's code blocks
/// with — that crate keeps its loaded copy private — and the same theme, so
/// code reads the same in the pane as it does in the conversation.
struct Highlighting {
    syntaxes: SyntaxSet,
    theme: Theme,
}

fn highlighting() -> &'static Highlighting {
    static HIGHLIGHTING: OnceLock<Highlighting> = OnceLock::new();
    HIGHLIGHTING.get_or_init(|| Highlighting {
        syntaxes: SyntaxSet::load_defaults_newlines(),
        // `tui-markdown`'s default, which is what the chat uses.
        theme: ThemeSet::load_defaults()
            .themes
            .remove("base16-ocean.dark")
            .unwrap_or_default(),
    })
}

/// Colours a diff's lines as the language they're in.
///
/// Two parsers, one per side, as `delta` does it: the old file and the new
/// one are different texts, and feeding both through one parser would let a
/// removed line that opened a string colour the rest of the new file as one.
/// Unchanged lines belong to both, so both read them.
struct Painter {
    old: HighlightLines<'static>,
    new: HighlightLines<'static>,
}

impl Painter {
    /// `None` for a file in no language the highlighter knows, which is then
    /// read without the cost of highlighting at all.
    fn for_path(path: &str) -> Option<Painter> {
        let path = Path::new(path);
        let name = path.file_name()?.to_str()?;
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let highlighting = highlighting();
        let syntaxes = &highlighting.syntaxes;
        // The whole name first, for the files known by it — `Makefile`,
        // `Dockerfile` — and then the extension.
        let syntax = syntaxes
            .find_syntax_by_extension(name)
            .or_else(|| syntaxes.find_syntax_by_extension(extension))?;
        if syntax.name == syntaxes.find_syntax_plain_text().name {
            return None;
        }
        Some(Painter {
            old: HighlightLines::new(syntax, &highlighting.theme),
            new: HighlightLines::new(syntax, &highlighting.theme),
        })
    }

    /// The colours of one line, led by its marker.
    fn paint(&mut self, text: &str) -> Vec<(usize, Option<Color>)> {
        let mut chars = text.chars();
        let (Some(marker), code) = (chars.next(), chars.as_str()) else {
            return Vec::new();
        };
        match marker {
            '+' => colors(&mut self.new, code),
            '-' => colors(&mut self.old, code),
            ' ' => {
                colors(&mut self.old, code);
                colors(&mut self.new, code)
            }
            _ => Vec::new(),
        }
    }
}

/// Runs one line through a parser: where each colour ends, and what it is.
/// The theme's own foreground is left as the terminal's, so plain code reads
/// in the colour the rest of the screen does.
fn colors(parser: &mut HighlightLines, code: &str) -> Vec<(usize, Option<Color>)> {
    // The syntaxes are the newline-terminated kind, which need one.
    let line = format!("{code}\n");
    let Ok(regions) = parser.highlight_line(&line, &highlighting().syntaxes) else {
        return Vec::new();
    };
    let plain = highlighting().theme.settings.foreground;
    let mut runs: Vec<(usize, Option<Color>)> = Vec::new();
    let mut end = 0;
    for (style, piece) in regions {
        end = (end + piece.len()).min(code.len());
        let fg = style.foreground;
        let color = (Some(fg) != plain).then_some(Color::Rgb(fg.r, fg.g, fg.b));
        match runs.last_mut() {
            Some(last) if last.1 == color => last.0 = end,
            _ => runs.push((end, color)),
        }
    }
    runs
}

/// One line, at most [`MAX_LINE_BYTES`] of it — the rest is skipped without
/// ever being held, and marked. `None` at the end.
fn read_line_capped(reader: &mut impl BufRead) -> Option<String> {
    let mut line = Vec::new();
    let (mut any, mut clipped) = (false, false);
    loop {
        let buf = match reader.fill_buf() {
            Ok(buf) => buf,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        };
        if buf.is_empty() {
            break;
        }
        any = true;
        let end = buf.iter().position(|&b| b == b'\n');
        let chunk = &buf[..end.unwrap_or(buf.len())];
        let room = MAX_LINE_BYTES.saturating_sub(line.len());
        clipped |= chunk.len() > room;
        line.extend_from_slice(&chunk[..chunk.len().min(room)]);
        let used = end.map_or(buf.len(), |at| at + 1);
        reader.consume(used);
        if end.is_some() {
            break;
        }
    }
    if !any {
        return None;
    }
    if line.last() == Some(&b'\r') {
        line.pop();
    }
    let mut text = String::from_utf8_lossy(&line).into_owned();
    if clipped {
        text.push_str(" …");
    }
    Some(expand_tabs(&text))
}

/// Whether a file looks binary: a NUL in its first few kilobytes, which is
/// the test git itself uses. Leaves the file read from the start again.
fn is_binary(mut file: &std::fs::File) -> bool {
    use std::io::{Seek, SeekFrom};
    let mut head = [0u8; 8000];
    let read = file.read(&mut head).unwrap_or(0);
    let _ = file.seek(SeekFrom::Start(0));
    head[..read].contains(&0)
}

/// The larger of the file as it stands and as it was committed. Either side
/// can be the large one, and the diff carries both. Zero for a side that
/// doesn't exist — a deleted file's disk, say.
fn file_size(root: &Path, file: &ChangedFile) -> u64 {
    let now = std::fs::metadata(root.join(&file.path)).map_or(0, |m| m.len());
    let committed = format!("HEAD:{}", file.orig.as_ref().unwrap_or(&file.path));
    let was = git(root, &["cat-file", "-s", &committed])
        .ok()
        .and_then(|size| size.trim().parse().ok())
        .unwrap_or(0);
    now.max(was)
}

fn human_size(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes as f64 >= MIB {
        format!("{:.1} MB", bytes as f64 / MIB)
    } else {
        format!("{} KB", bytes.div_ceil(1024))
    }
}

/// What the diff compares against: the last commit, or the empty tree in a
/// repository that has none yet.
fn base(root: &Path) -> &'static str {
    if git(root, &["rev-parse", "--verify", "--quiet", "HEAD"]).is_ok() {
        "HEAD"
    } else {
        EMPTY_TREE
    }
}

/// Tabs expanded here rather than left to the terminal: ratatui counts a
/// tab as one cell, so it would draw the rest of the line out of step with
/// the columns it thinks it used.
fn expand_tabs(line: &str) -> String {
    line.replace('\t', "    ")
}

/// Splits the chat screen into the conversation and the pane beside it.
/// Shared by drawing and by the mouse handler, so a wheel turn lands on
/// whichever side it was turned over.
pub fn split(area: Rect) -> (Rect, Rect) {
    let pane = (area.width / 2).min(area.width.saturating_sub(MIN_CHAT_WIDTH));
    let chat = area.width - pane;
    (
        Rect {
            width: chat,
            ..area
        },
        Rect {
            x: area.x + chat,
            width: pane,
            ..area
        },
    )
}

fn status_style(status: [char; 2]) -> Style {
    // The more telling of the two letters: a conflict, then whatever the
    // worktree did, then the index.
    let letter = if status.contains(&'U') {
        'U'
    } else if status[1] != ' ' {
        status[1]
    } else {
        status[0]
    };
    match letter {
        'M' => Style::new().yellow(),
        'A' | '?' => Style::new().green(),
        'D' => Style::new().red(),
        'R' | 'C' => Style::new().cyan(),
        'U' => Style::new().magenta().bold(),
        _ => Style::new(),
    }
}

/// The backgrounds behind added and removed lines: the terminal's own,
/// pulled a fifth of the way toward green and toward red. Derived from the
/// real background rather than picked, so the tint is a tint on a dark theme
/// and a light one alike. `None` when the terminal never said what colour it
/// is — the lines then say it with green and red text instead, as they did
/// before there were tints.
fn tints() -> Option<(Color, Color)> {
    Some((toward((0, 200, 80), 0.2)?, toward((230, 40, 40), 0.2)?))
}

/// The terminal's background pulled `share` of the way toward `to`, or
/// `None` when the terminal never said what its background is.
fn toward((to_r, to_g, to_b): (u8, u8, u8), share: f32) -> Option<Color> {
    let ((r, g, b), _) = super::render::background()?;
    let mix = |from: u8, to: u8| (from as f32 + (to as f32 - from as f32) * share).round() as u8;
    Some(Color::Rgb(mix(r, to_r), mix(g, to_g), mix(b, to_b)))
}

/// The list's two highlights: the file on show, and the cursor when it has
/// wandered off it. The file on show is the heavy one — a strong tint of the
/// pane's accent — so it stays the landmark while you look around; the
/// cursor is a faint lift of the background, there to say where Enter would
/// go. Without a known background, reversed video and a dark grey ground.
fn list_grounds() -> (Style, Style) {
    match (toward((0, 170, 220), 0.4), toward((128, 128, 128), 0.18)) {
        (Some(open), Some(cursor)) => (Style::new().bg(open).bold(), Style::new().bg(cursor)),
        _ => (
            Style::new().reversed().bold(),
            Style::new().bg(Color::DarkGray),
        ),
    }
}

/// One row of the diff: its number, its marker, and its code in the
/// language's colours — on a green or red ground when it was added or
/// removed, so the change is the background and the code keeps its colours.
fn diff_line(
    row: &Row,
    number_width: usize,
    width: usize,
    tints: Option<(Color, Color)>,
) -> Line<'static> {
    let number = match row.number {
        Some(n) => format!("{n:>number_width$} "),
        None => " ".repeat(number_width + 1),
    };
    let mut spans = vec![Span::styled(number, Style::new().dark_gray())];
    let ground = match (row.text.chars().next(), tints) {
        (Some('+'), Some((added, _))) => Some(added),
        (Some('-'), Some((_, removed))) => Some(removed),
        _ => None,
    };

    // Code is a line led by a marker and coloured by the highlighter. A
    // change with no ground to stand on keeps its green or red text, which
    // is the only thing left saying it changed.
    let changed = matches!(row.text.chars().next(), Some('+' | '-'));
    let painted = !row.colors.is_empty() && (ground.is_some() || !changed);
    if painted || ground.is_some() {
        let (marker, code) = row.text.split_at(1);
        let base = ground.map_or(Style::new(), |ground| Style::new().bg(ground));
        spans.push(Span::styled(
            marker.to_string(),
            diff_style(&row.text).patch(base).bold(),
        ));
        let mut start = 0;
        for &(end, fg) in &row.colors {
            if let Some(piece) = code.get(start..end) {
                let style = match fg {
                    Some(fg) => base.fg(fg),
                    None => base,
                };
                spans.push(Span::styled(piece.to_string(), style));
                start = end;
            }
        }
        if let Some(rest) = code.get(start..).filter(|rest| !rest.is_empty()) {
            spans.push(Span::styled(rest.to_string(), base));
        }
        // Out to the pane's edge, so the ground is a band and not a box
        // around the text — but not under the number, which isn't code.
        if let Some(ground) = ground {
            let used: usize = spans.iter().map(|span| span.content.width()).sum();
            spans.push(Span::styled(
                " ".repeat(width.saturating_sub(used)),
                Style::new().bg(ground),
            ));
        }
    } else {
        spans.push(Span::styled(row.text.clone(), diff_style(&row.text)));
    }
    Line::from(spans)
}

/// The counts of the changes out of view.
const BEYOND: Style = Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD);

/// Read off the marker alone. With the header gone every line is file
/// content, so a removed line that happens to read `-- x` is still a removal
/// and not a `---` header. Anything without a marker is the pane talking —
/// a size note, a truncation, git's word on a binary — and is dimmed.
fn diff_style(line: &str) -> Style {
    match line.chars().next() {
        Some('+') => Style::new().green(),
        Some('-') => Style::new().red(),
        Some('@') => Style::new().cyan(),
        Some(' ') | None => Style::new(),
        _ => Style::new().dark_gray().italic(),
    }
}

pub fn draw(frame: &mut Frame, area: Rect, pane: &GitPane, tick: usize) {
    let accent = if pane.focused {
        Style::new().cyan()
    } else {
        Style::new().dark_gray()
    };
    let block = Block::default().borders(Borders::LEFT).border_style(accent);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height < 4 {
        return;
    }

    // The list takes what it needs up to a third of the height, so a large
    // change set can't push the diff — the reason to be here — off the pane.
    let most = (inner.height / 3).max(3);
    let list_rows = (pane.files.len().max(1) as u16).min(most);
    let hint_rows = u16::from(pane.focused);
    let areas = Layout::vertical([
        Constraint::Length(1),         // header
        Constraint::Length(list_rows), // files
        Constraint::Length(1),         // rule, naming the diff below it
        Constraint::Min(1),            // diff
        Constraint::Length(hint_rows), // keys, while they're live
    ])
    .split(inner);

    let mut header = vec![Span::styled(" Changes", accent.bold())];
    if let Some(branch) = &pane.branch {
        header.push(Span::styled(
            format!(" ⎇ {branch}"),
            Style::new().dark_gray(),
        ));
    }
    if pane.error.is_none() && pane.listed {
        header.push(Span::styled(
            format!(" · {}", pane.files.len()),
            Style::new().dark_gray(),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(header)), areas[0]);

    // No list and no diff to speak of, so the explanation gets the whole
    // body rather than the one row an empty list would leave it.
    if let Some(why) = &pane.error {
        let body = Rect {
            height: areas[4].y - areas[1].y,
            ..areas[1]
        };
        frame.render_widget(
            Paragraph::new(why.lines()).wrap(ratatui::widgets::Wrap { trim: false }),
            body,
        );
        draw_keys(frame, areas[4], pane);
        return;
    }
    draw_list(frame, areas[1], pane, tick);

    let diff_area = areas[3];
    let rows = diff_area.height.max(1) as usize;
    pane.diff_rows.set(rows);
    let opening = pane
        .shown
        .as_ref()
        .is_some_and(|shown| shown.opening() && shown.loading());
    let (above, below) = match &pane.shown {
        Some(shown) if !opening => shown.beyond(rows),
        _ => (0, 0),
    };

    // The rule names the file, and counts the changes out of view at its
    // far end; the keys to go to them are on the row below.
    let mut counts = String::new();
    if above > 0 {
        counts.push_str(&format!(" ▲ {above}"));
    }
    if below > 0 {
        counts.push_str(&format!(" ▼ {below}"));
    }
    if !counts.is_empty() {
        counts.push(' ');
    }
    let tail = if counts.is_empty() { "" } else { "──" };
    // A path too long for the row gives up its start rather than pushing
    // the counts off the end: the file's own name is the part to keep.
    let room = (areas[2].width as usize).saturating_sub(counts.width() + tail.width() + 2);
    let title = match &pane.shown {
        Some(shown) => format!(" {} ", clip_start(&shown.file.path, room)),
        None => String::new(),
    };
    let rule = "─".repeat(
        (areas[2].width as usize).saturating_sub(title.width() + counts.width() + tail.width()),
    );
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(title, Style::new().bold()),
            Span::styled(rule, Style::new().dark_gray()),
            Span::styled(counts, BEYOND),
            Span::styled(tail, Style::new().dark_gray()),
        ])),
        areas[2],
    );

    match &pane.shown {
        Some(shown) if opening => {
            let mut finding = " finding the first change…".to_string();
            if !shown.lines.is_empty() {
                finding.push_str(&format!(" {} lines in", shown.lines.len()));
            }
            frame.render_widget(
                Paragraph::new(Line::styled(finding, Style::new().dark_gray().italic())),
                diff_area,
            );
        }
        Some(shown) if shown.lines.is_empty() && shown.loading() => {
            frame.render_widget(
                Paragraph::new(Line::styled(" loading…", Style::new().dark_gray().italic())),
                diff_area,
            );
        }
        Some(shown) => {
            // Only what's in view is styled and handed over, so a diff of a
            // few thousand lines costs a screenful per frame, not all of it.
            let start = shown.top(rows);
            let visible = &shown.lines[start..(start + rows).min(shown.lines.len())];
            // Wide enough for the biggest number on screen, and never under
            // four, so the column doesn't shift every time a scroll crosses
            // a power of ten in a file of ordinary length.
            let width = visible
                .iter()
                .filter_map(|row| row.number)
                .max()
                .map_or(0, |n| n.to_string().len())
                .max(4);
            let tints = tints();
            let lines: Vec<Line> = visible
                .iter()
                .map(|row| diff_line(row, width, diff_area.width as usize, tints))
                .collect();
            frame.render_widget(Paragraph::new(lines), diff_area);
        }
        None if pane.error.is_none() && !pane.files.is_empty() => {
            frame.render_widget(
                Paragraph::new(Line::styled(
                    " Enter on a file to see its changes",
                    Style::new().dark_gray().italic(),
                )),
                diff_area,
            );
        }
        None => {}
    }

    draw_keys(frame, areas[4], pane);
}

/// `text` cut to `width` columns from the front, marked with `…` where it
/// was cut.
fn clip_start(text: &str, width: usize) -> String {
    if text.width() <= width {
        return text.to_string();
    }
    let mut kept = String::new();
    let mut used = 1; // the `…`
    for c in text.chars().rev() {
        used += c.to_string().width();
        if used > width {
            break;
        }
        kept.insert(0, c);
    }
    format!("…{kept}")
}

fn draw_keys(frame: &mut Frame, area: Rect, pane: &GitPane) {
    if !pane.focused {
        return;
    }
    // Only the keys that do something: with no repository there is nothing
    // to move through, view or scroll.
    let keys = if pane.error.is_some() {
        " r refresh · Tab chat · Esc close"
    } else {
        " ↑/↓ file · Enter view · n/N change · PgUp/PgDn J/K scroll · r refresh · Tab chat · Esc close"
    };
    frame.render_widget(
        Paragraph::new(Line::styled(keys, Style::new().dark_gray())),
        area,
    );
}

/// Columns the list keeps on its left for saying what is happening to a
/// file: two, the width of the chat's animation.
const GUTTER: usize = 2;

/// What a file's gutter shows: the chat's `working` animation, in its
/// yellow, on the file the agent is at — or its `?` while the turn waits on
/// an approval. Blank otherwise.
fn gutter(pane: &GitPane, file: &ChangedFile, tick: usize) -> Span<'static> {
    let at_work = pane.active.as_deref() == Some(file.path.as_str());
    match &pane.activity {
        Some(activity) if at_work && activity.waiting => {
            Span::styled(format!("{:<GUTTER$}", "?"), Style::new().yellow().bold())
        }
        Some(_) if at_work => Span::styled(super::render::busy_frame(tick), Style::new().yellow()),
        _ => Span::raw(" ".repeat(GUTTER)),
    }
}

fn draw_list(frame: &mut Frame, area: Rect, pane: &GitPane, tick: usize) {
    if !pane.listed {
        return;
    }
    if pane.files.is_empty() {
        frame.render_widget(
            Paragraph::new(Line::styled(
                " Nothing changed",
                Style::new().dark_gray().italic(),
            )),
            area,
        );
        return;
    }

    // Scrolled just far enough to keep the cursor on screen.
    let rows = area.height as usize;
    let offset = (pane.cursor + 1).saturating_sub(rows);
    let shown = pane.shown.as_ref().map(|s| s.file.path.as_str());
    let (open_ground, cursor_ground) = list_grounds();
    let lines: Vec<Line> = pane
        .files
        .iter()
        .enumerate()
        .skip(offset)
        .take(rows)
        .map(|(i, file)| {
            let open = shown == Some(file.path.as_str());
            let marker = if open { "▸" } else { " " };
            let mut name = file.path.clone();
            if let Some(orig) = &file.orig {
                name = format!("{orig} → {name}");
            }
            let mut line = Line::from(vec![
                gutter(pane, file, tick),
                Span::styled(marker, Style::new().cyan()),
                Span::styled(
                    file.status.iter().collect::<String>(),
                    status_style(file.status),
                ),
                Span::raw(format!(" {name}")),
            ]);
            // Out to the pane's edge, so a highlight is a band, not a box.
            let pad = (area.width as usize).saturating_sub(line.width());
            line.push_span(" ".repeat(pad));
            // The cursor only shows while the keys move it; on the file on
            // show it is already as marked as a row gets.
            if open {
                line = line.style(open_ground);
            } else if i == pane.cursor && pane.focused {
                line = line.style(cursor_ground);
            }
            line
        })
        .collect();
    frame.render_widget(Paragraph::new(lines), area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_is_read_entry_by_entry_with_renames_taking_their_source() {
        let raw = " M src/main.rs\0R  new.rs\0old.rs\0?? notes dir/a b.txt\0D  gone.rs\0";
        let files = parse_status(raw);
        assert_eq!(
            files,
            vec![
                ChangedFile {
                    status: [' ', 'M'],
                    path: "src/main.rs".into(),
                    orig: None
                },
                ChangedFile {
                    status: ['R', ' '],
                    path: "new.rs".into(),
                    orig: Some("old.rs".into())
                },
                // Spaces survive: -z means nothing is quoted.
                ChangedFile {
                    status: ['?', '?'],
                    path: "notes dir/a b.txt".into(),
                    orig: None
                },
                ChangedFile {
                    status: ['D', ' '],
                    path: "gone.rs".into(),
                    orig: None
                },
            ]
        );
    }

    #[test]
    fn a_directory_outside_any_repository_says_so() {
        // The root of the filesystem is outside every repository a test
        // machine could plausibly have.
        let mut pane = GitPane::new(PathBuf::from("/"));
        pane.refresh();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !pane.listed && Instant::now() < deadline {
            pane.pump();
            std::thread::sleep(Duration::from_millis(5));
        }
        match &pane.error {
            Some(Unavailable::NoRepo(dir)) => assert_eq!(dir, Path::new("/")),
            // No git on this machine is the other honest answer.
            Some(Unavailable::NoGit(_)) => {}
            other => panic!("expected no repository, got {other:?}"),
        }
        assert!(pane.files.is_empty());
    }

    /// A diff loading from `text` as if git were writing it, with no git.
    fn streaming(path: &str, text: &'static str) -> Shown {
        let file = ChangedFile {
            status: [' ', 'M'],
            path: path.to_string(),
            orig: None,
        };
        let mut shown = Shown::note(Path::new("."), &file, None);
        shown.header = Some(Vec::new());
        shown.loader = Some(spawn_reader(text.as_bytes(), "", None, path));
        shown
    }

    fn numbered(shown: &Shown) -> Vec<(Option<usize>, &str)> {
        shown
            .lines
            .iter()
            .map(|row| (row.number, row.text.as_str()))
            .collect()
    }

    #[test]
    fn code_is_coloured_in_its_own_language() {
        let mut painter = Painter::for_path("src/main.rs").expect("Rust is known");
        let colors = painter.paint("+fn main() {}");
        // `fn` is a keyword, so it gets a colour of its own; the runs end on
        // the code's own length, with the marker not counted.
        assert!(
            colors
                .first()
                .is_some_and(|(end, color)| *end == 2 && color.is_some()),
            "{colors:?}"
        );
        assert_eq!(colors.last().unwrap().0, "fn main() {}".len());
    }

    #[test]
    fn plain_text_is_read_without_a_highlighter() {
        assert!(Painter::for_path("notes.txt").is_none());
        assert!(Painter::for_path("huge.log").is_none());
        // Known by its whole name rather than an extension.
        assert!(Painter::for_path("Makefile").is_some());
    }

    #[test]
    fn a_removed_string_does_not_colour_the_new_side() {
        // An unclosed quote on the removed line would, with one parser for
        // both sides, turn every line after it into a string.
        let mut painter = Painter::for_path("a.rs").unwrap();
        painter.paint("-let s = \"open");
        let fresh = Painter::for_path("a.rs").unwrap().paint("+let x = 1;");
        assert_eq!(painter.paint("+let x = 1;"), fresh);
    }

    #[test]
    fn a_change_is_a_band_to_the_edge_and_its_number_stays_plain() {
        let row = Row {
            number: Some(7),
            text: "+let x = 1;".to_string(),
            colors: Painter::for_path("a.rs").unwrap().paint("+let x = 1;"),
        };
        let green = Color::Rgb(0, 80, 0);
        let line = diff_line(&row, 4, 30, Some((green, Color::Rgb(80, 0, 0))));
        assert_eq!(line.spans[0].content, "   7 ");
        assert_eq!(line.spans[0].style.bg, None, "the number isn't code");
        assert!(line.spans[1..]
            .iter()
            .all(|span| span.style.bg == Some(green)));
        let width: usize = line.spans.iter().map(|span| span.content.width()).sum();
        assert_eq!(width, 30, "padded out to the pane's edge");

        // With no background to tint from, the change is green text.
        let line = diff_line(&row, 4, 30, None);
        assert_eq!(line.spans[1].content, "+let x = 1;");
        assert_eq!(line.spans[1].style.fg, Some(Color::Green));
    }

    #[test]
    fn a_hunk_says_where_its_new_side_starts() {
        assert_eq!(hunk_start("@@ -3,4 +5,6 @@ fn main()"), Some(5));
        assert_eq!(hunk_start("@@ -0,0 +1 @@"), Some(1));
        assert_eq!(hunk_start("@@ garbled"), None);
    }

    /// Pumps until `wanted` lines are held or the reader has finished —
    /// the reader is a real thread, so this is the test's event loop.
    fn settle(shown: &mut Shown, wanted: usize) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while shown.loading() && shown.lines.len() < wanted && Instant::now() < deadline {
            shown.pump(wanted);
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn a_whole_file_diff_is_shown_as_the_file_with_its_header_dropped() {
        let mut shown = streaming(
            "m.rs",
            "diff --git a/m.rs b/m.rs\nindex 1..2 100644\n--- a/m.rs\n+++ b/m.rs\n\
             @@ -1,3 +1,3 @@\n fn main() {\n-\told();\n+\tnew();\n }\n",
        );
        settle(&mut shown, MAX_LINES);
        // Numbered as the file stands now: the removed line has no number,
        // and the one that replaced it takes the place it had.
        assert_eq!(
            numbered(&shown),
            vec![
                (Some(1), " fn main() {"),
                (None, "-    old();"),
                (Some(2), "+    new();"),
                (Some(3), " }")
            ]
        );
        assert!(!shown.loading());
    }

    #[test]
    fn only_as_many_lines_as_the_view_wants_are_taken() {
        let text: String = std::iter::once("@@ -0,0 +1,5000 @@\n".to_string())
            .chain((0..5000).map(|n| format!("+{n}\n")))
            .collect();
        let mut shown = streaming("big", Box::leak(text.into_boxed_str()));
        settle(&mut shown, 40);
        // A batch at a time, so a little past what was wanted — but nowhere
        // near the whole of it.
        assert!(
            (40..=READ_BATCH).contains(&shown.lines.len()),
            "{}",
            shown.lines.len()
        );
        assert!(
            shown.loading(),
            "the rest is still waiting to be scrolled to"
        );

        settle(&mut shown, MAX_LINES);
        assert_eq!(shown.lines.len(), 5000);
        assert!(!shown.loading());
    }

    #[test]
    fn a_diff_without_hunks_shows_the_file_as_it_stands() {
        // A pure rename: git has nothing to say about the content.
        let mut shown = streaming(
            "Cargo.toml",
            "diff --git a/old b/Cargo.toml\nsimilarity index 100%\n",
        );
        settle(&mut shown, MAX_LINES);
        assert_eq!(numbered(&shown)[0], (Some(1), " [package]"));
    }

    #[test]
    fn a_binary_says_so_rather_than_being_read() {
        let mut shown = streaming(
            "Cargo.toml",
            "diff --git a/x b/x\nBinary files a/x and b/x differ\n",
        );
        settle(&mut shown, MAX_LINES);
        assert_eq!(
            shown.lines.last().unwrap().text,
            "Binary files a/x and b/x differ"
        );
    }

    #[test]
    fn a_binary_renamed_as_it_was_is_not_read_as_text() {
        // A pure rename's header never says "Binary files", so the file is
        // asked itself.
        let dir = std::env::temp_dir().join(format!("clank-git-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("image.png"), b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR").unwrap();
        let mut shown = streaming(
            "image.png",
            "diff --git a/old.png b/image.png\nsimilarity index 100%\n",
        );
        shown.root = dir.clone();
        settle(&mut shown, MAX_LINES);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(
            shown.lines.last().unwrap().text,
            "similarity index 100%",
            "{:?}",
            numbered(&shown)
        );
    }

    #[test]
    fn changes_only_keeps_the_hunk_lines_that_say_where_each_one_is() {
        let mut shown = streaming(
            "big",
            "diff --git a/b b/b\n--- a/b\n+++ b/b\n@@ -10,1 +10,1 @@\n-a\n+b\n@@ -90,1 +90,1 @@\n-c\n+d\n",
        );
        shown.whole = false;
        settle(&mut shown, MAX_LINES);
        // Each hunk picks the numbering up from where its `@@` says it is.
        assert_eq!(
            numbered(&shown),
            vec![
                (None, "@@ -10,1 +10,1 @@"),
                (None, "-a"),
                (Some(10), "+b"),
                (None, "@@ -90,1 +90,1 @@"),
                (None, "-c"),
                (Some(90), "+d")
            ]
        );
    }

    #[test]
    fn a_huge_line_is_clipped_without_being_held() {
        let long = "x".repeat(MAX_LINE_BYTES * 50);
        let input = format!("short\n{long}\r\nlast");
        let mut reader = std::io::Cursor::new(input.into_bytes());
        assert_eq!(read_line_capped(&mut reader).unwrap(), "short");
        let clipped = read_line_capped(&mut reader).unwrap();
        assert_eq!(clipped.len(), MAX_LINE_BYTES + " …".len());
        assert!(clipped.ends_with(" …"));
        assert_eq!(read_line_capped(&mut reader).unwrap(), "last");
        assert_eq!(read_line_capped(&mut reader), None);
    }

    #[test]
    fn a_clean_tree_has_nothing_to_list() {
        assert!(parse_status("").is_empty());
    }

    fn pane_with(files: &[&str]) -> GitPane {
        let mut pane = GitPane::new(PathBuf::from("."));
        pane.files = files
            .iter()
            .map(|p| ChangedFile {
                status: [' ', 'M'],
                path: p.to_string(),
                orig: None,
            })
            .collect();
        pane
    }

    #[test]
    fn the_cursor_stops_at_both_ends() {
        let mut pane = pane_with(&["a", "b"]);
        pane.move_cursor(false);
        assert_eq!(pane.cursor, 0);
        for _ in 0..5 {
            pane.move_cursor(true);
        }
        assert_eq!(pane.cursor, 1);
    }

    #[test]
    fn scrolling_holds_the_last_screenful() {
        let text: String = std::iter::once("@@ -0,0 +1,50 @@\n".to_string())
            .chain((0..50).map(|n| format!("+{n}\n")))
            .collect();
        let mut pane = pane_with(&["a"]);
        pane.shown = Some(streaming("a", Box::leak(text.into_boxed_str())));
        pane.diff_rows.set(20);

        // End holds the view at the bottom while the rest loads in.
        pane.scroll_to(true);
        let deadline = Instant::now() + Duration::from_secs(5);
        while pane.shown.as_ref().unwrap().loading() && Instant::now() < deadline {
            pane.pump();
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(pane.shown.as_ref().unwrap().scroll, 30);

        pane.scroll(-pane.page());
        assert_eq!(pane.shown.as_ref().unwrap().scroll, 11);
        pane.scroll_to(false);
        assert_eq!(pane.shown.as_ref().unwrap().scroll, 0);
    }

    /// A whole-file diff of `len` lines with a line changed at each of
    /// `changes`, shown in a pane `rows` tall and opened as Enter opens it.
    fn opened(len: usize, changes: &[usize], rows: usize) -> GitPane {
        let mut text = format!("@@ -1,{len} +1,{len} @@\n");
        for n in 0..len {
            if changes.contains(&n) {
                text.push_str(&format!("-{n}\n+changed {n}\n"));
            } else {
                text.push_str(&format!(" {n}\n"));
            }
        }
        let mut pane = pane_with(&["a"]);
        pane.shown = Some(Shown {
            seek: Some(0),
            ..streaming("a", Box::leak(text.into_boxed_str()))
        });
        pane.diff_rows.set(rows);
        settle_pane(&mut pane);
        pane
    }

    fn settle_pane(pane: &mut GitPane) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while pane.shown.as_ref().unwrap().seek.is_some() && Instant::now() < deadline {
            pane.pump();
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn scroll(pane: &GitPane) -> usize {
        pane.shown.as_ref().unwrap().scroll
    }

    #[test]
    fn a_file_opens_at_its_first_change() {
        // Past the lines a first screenful would read, so it's sought.
        let pane = opened(5000, &[3000], 20);
        assert_eq!(scroll(&pane), 3000 - CONTEXT);
        assert_eq!(pane.shown.unwrap().lines[3000].text, "-3000");

        // One near the top leaves the top in view.
        assert_eq!(scroll(&opened(100, &[1], 20)), 0);
    }

    #[test]
    fn changes_out_of_view_are_counted_and_gone_to_in_turn() {
        // Each change is two rows, the removed line and the added one.
        let mut pane = opened(300, &[10, 100, 200], 20);
        let rows = |pane: &GitPane| pane.shown.as_ref().unwrap().beyond(20);
        assert_eq!(scroll(&pane), 7);
        assert_eq!(rows(&pane), (0, 2));

        pane.jump(true);
        assert_eq!(scroll(&pane), 101 - CONTEXT);
        assert_eq!(rows(&pane), (1, 1));
        pane.jump(true);
        settle_pane(&mut pane);
        assert_eq!(scroll(&pane), 202 - CONTEXT);
        assert_eq!(rows(&pane), (2, 0));
        // Nothing past the last: the view stays put.
        pane.jump(true);
        settle_pane(&mut pane);
        assert_eq!(scroll(&pane), 202 - CONTEXT);

        pane.jump(false);
        assert_eq!(scroll(&pane), 101 - CONTEXT);
        pane.jump(false);
        pane.jump(false);
        assert_eq!(scroll(&pane), 7);
    }

    #[test]
    fn a_missing_newline_belongs_to_the_change_around_it() {
        let mut shown = streaming(
            "a",
            "@@ -1,2 +1,2 @@\n a\n-b\n\\ No newline at end of file\n+c\n",
        );
        settle(&mut shown, MAX_LINES);
        assert_eq!(shown.runs, vec![1..4]);
        assert_eq!(shown.blocks, Some(1));
    }

    #[test]
    fn a_long_path_keeps_its_end() {
        assert_eq!(clip_start("src/tui/git.rs", 20), "src/tui/git.rs");
        assert_eq!(clip_start("src/tui/git.rs", 8), "…/git.rs");
        assert_eq!(clip_start("src/tui/git.rs", 0), "…");
    }

    fn tool(name: &str, filepath: &str) -> TranscriptItem {
        TranscriptItem::ToolCall {
            name: name.to_string(),
            arguments: serde_json::json!({ "filepath": filepath, "content": "x" }).to_string(),
            status: super::super::app::ToolStatus::Running,
        }
    }

    #[test]
    fn the_agent_is_at_a_file_only_while_writing_it() {
        let mut app = App::new("m".to_string(), None, "s".to_string());
        app.transcript = vec![
            TranscriptItem::User("earlier".to_string()),
            tool("write_file", "old.rs"),
            TranscriptItem::User("now".to_string()),
            tool("write_file", "src/a.rs"),
            TranscriptItem::Notice("not the agent's doing".to_string()),
        ];
        // Idle, there is nothing being worked on, whatever the transcript.
        assert_eq!(activity(&app), None);

        app.busy = true;
        let at = activity(&app).unwrap();
        assert_eq!((at.filepath.as_str(), at.waiting), ("src/a.rs", false));

        // Whatever it does next, a read included, it has moved on.
        for next in [
            tool("read_file", "src/a.rs"),
            tool("run_terminal_command", "ls"),
            TranscriptItem::Thinking("hm".to_string()),
            TranscriptItem::Assistant {
                text: "done".to_string(),
                streaming: true,
                label: None,
            },
        ] {
            app.transcript.push(next);
            assert_eq!(activity(&app), None);
            app.transcript.pop();
        }

        // A turn that has written nothing yet isn't at the last turn's write.
        app.transcript
            .push(TranscriptItem::User("next".to_string()));
        assert_eq!(activity(&app), None);

        app.transcript.push(tool("replace_in_file", "b.rs"));
        app.pending_approval = Some(crate::ui::ApprovalRequest {
            tool_name: "replace_in_file".to_string(),
            category: "write",
            arguments: String::new(),
        });
        assert!(activity(&app).unwrap().waiting);
    }

    #[test]
    fn a_tools_path_is_named_the_way_git_names_it() {
        let root = std::env::temp_dir().join(format!("clank-at-{}", std::process::id()));
        let dir = root.join("sub");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.rs"), "").unwrap();
        let named = |filepath: &str| repo_relative(&root, &dir, filepath);
        assert_eq!(named("a.rs").as_deref(), Some("sub/a.rs"));
        assert_eq!(named("../sub/./a.rs").as_deref(), Some("sub/a.rs"));
        // Not written yet: resolved through the directory it will land in.
        assert_eq!(named("new.rs").as_deref(), Some("sub/new.rs"));
        let absolute = dir.join("a.rs").display().to_string();
        assert_eq!(named(&absolute).as_deref(), Some("sub/a.rs"));
        assert_eq!(named("/etc/hostname"), None);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_file_at_work_carries_the_animation_in_its_gutter() {
        let mut pane = pane_with(&["src/a.rs", "b.rs"]);
        pane.activity = Some(Activity {
            filepath: "src/a.rs".to_string(),
            waiting: false,
        });
        pane.active = Some("src/a.rs".to_string());
        let frame = super::super::render::busy_frame(7);
        assert_eq!(gutter(&pane, &pane.files[0], 7).content, frame);
        assert_eq!(gutter(&pane, &pane.files[1], 7).content, "  ");

        pane.activity.as_mut().unwrap().waiting = true;
        assert_eq!(gutter(&pane, &pane.files[0], 7).content, "? ");
    }

    #[test]
    fn a_mark_stays_long_enough_to_be_seen() {
        let mut pane = pane_with(&["a.rs", "b.rs"]);
        let at = |filepath: &str| {
            Some(Activity {
                filepath: filepath.to_string(),
                waiting: false,
            })
        };
        assert!(pane.set_activity(at("a.rs")));
        // The agent moving straight on doesn't take the mark down yet...
        assert!(!pane.set_activity(None));
        assert_eq!(pane.activity, at("a.rs"));
        // ...but a write elsewhere moves it at once.
        assert!(pane.set_activity(at("b.rs")));
        assert_eq!(pane.activity, at("b.rs"));
        // Once it has been up long enough, it goes.
        pane.marked_at -= MARK_HOLD;
        assert!(pane.set_activity(None));
        assert_eq!(pane.activity, None);
    }

    #[test]
    fn the_gutter_is_left_of_the_list() {
        let mut pane = pane_with(&["src/a.rs", "b.rs"]);
        pane.listed = true;
        pane.activity = Some(Activity {
            filepath: "b.rs".to_string(),
            waiting: false,
        });
        pane.active = Some("b.rs".to_string());
        let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(30, 12)).unwrap();
        terminal
            .draw(|frame| draw(frame, frame.area(), &pane, 3))
            .unwrap();
        let buffer = terminal.backend().buffer();
        let row = |y: u16| -> String {
            (1..buffer.area.width)
                .map(|x| buffer[(x, y)].symbol().to_string())
                .collect::<String>()
        };
        let frame = super::super::render::busy_frame(3);
        assert_eq!(row(1).trim_end(), "    M src/a.rs");
        assert_eq!(row(2).trim_end(), format!("{frame}  M b.rs"));
    }

    #[test]
    fn the_conversation_keeps_room_beside_the_pane() {
        let (chat, pane) = split(Rect::new(0, 0, 120, 40));
        assert_eq!((chat.width, pane.width, pane.x), (60, 60, 60));
        let (chat, pane) = split(Rect::new(0, 0, 50, 40));
        assert_eq!((chat.width, pane.width), (30, 20));
    }
}
