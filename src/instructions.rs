//! `CLANKERS.md`: what you tell every clanker deployed to a directory.
//!
//! One file, in the clanker's own directory and nowhere else — not its
//! parents, not its subdirectories. Read afresh at the start of every turn
//! and sent inside the agent system prompt, so an edit reaches the next
//! turn while a turn's own requests all share one cached prefix.
//!
//! Never cut short. Like Claude Code with `CLAUDE.md`, a large file is sent
//! whole and only warned about, since a rule dropped off the end to save
//! tokens is a worse failure than the tokens.

use std::hash::{Hash, Hasher};
use std::path::Path;

/// The file's name, looked for in the clanker's directory.
pub const FILE: &str = "CLANKERS.md";

/// Past this many characters a notice says the file is large: it is sent
/// with every request. The threshold Claude Code warns at for `CLAUDE.md`.
pub const LARGE_CHARS: usize = 40_000;

/// The file's contents, if `dir` has one with anything in it. Read lossily,
/// so a stray invalid byte costs a replacement character rather than the
/// whole file.
pub fn load(dir: &Path) -> Option<String> {
    let path = dir.join(FILE);
    if !path.is_file() {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    (!text.trim().is_empty()).then_some(text)
}

/// What a front end is told about the file each turn: enough to say when
/// it appeared, changed or went, without carrying the text itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Seen {
    pub chars: usize,
    /// Tells one version from the next within a process; never stored.
    pub fingerprint: u64,
}

impl Seen {
    pub fn of(text: &str) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        Seen {
            chars: text.chars().count(),
            fingerprint: hasher.finish(),
        }
    }
}

/// The agent system prompt with the file's text after it, introduced so the
/// model knows whose instructions they are and where they stand.
pub fn system_prompt(base: &str, instructions: Option<&str>) -> String {
    match instructions {
        None => base.to_string(),
        Some(text) => format!(
            "{base}\n\nThe user's instructions for this project, from {FILE} in the working \
             directory. Follow them; they take precedence over the guidance above, though \
             not over what the user asks in the conversation.\n\n{}",
            text.trim_end()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_file_with_something_in_it_is_loaded() {
        let dir = std::env::temp_dir().join(format!("clank-instructions-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(load(&dir), None);

        std::fs::write(dir.join(FILE), "  \n").unwrap();
        assert_eq!(load(&dir), None, "blank is as good as missing");

        std::fs::write(dir.join(FILE), "Run `cargo test` before finishing.\n").unwrap();
        assert_eq!(
            load(&dir).as_deref(),
            Some("Run `cargo test` before finishing.\n")
        );

        // Only this directory's: a parent's or a child's is not looked for.
        let child = dir.join("sub");
        std::fs::create_dir_all(&child).unwrap();
        assert_eq!(load(&child), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_file_follows_the_prompt_and_says_where_it_stands() {
        assert_eq!(system_prompt("Base.", None), "Base.");
        let prompt = system_prompt("Base.", Some("Use tabs.\n\n"));
        assert!(prompt.starts_with("Base.\n\n"), "{prompt}");
        assert!(prompt.contains("from CLANKERS.md"), "{prompt}");
        assert!(prompt.ends_with("\n\nUse tabs."), "{prompt}");
    }
}
