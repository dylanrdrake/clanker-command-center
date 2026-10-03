//! Width-safe stand-ins for punctuation terminals disagree about.
//!
//! `·` (U+00B7 MIDDLE DOT) and `—` (U+2014 EM DASH) are East Asian Width
//! *Ambiguous*: `unicode-width` counts each as one column, but a terminal in
//! a CJK locale renders it as two. Anywhere a line is measured or padded by
//! column — the TUI's alignment, and any `{:<n}` padding — the two disagree,
//! and the row drifts one cell per glyph. That is exactly the kind of
//! half-aligned box that looks like a bug and is hard to place.
//!
//! These replacements are East Asian Width *Neutral*, so every terminal
//! agrees with `unicode-width`, and they look close enough to the originals
//! to read as the same punctuation. Use them wherever a line is measured,
//! padded or wrapped by column — the TUI's rows, status bar, notices and key
//! hints. Output that is only printed and never padded (most of the CLI) and
//! ordinary prose in comments and docs are left as they are.

/// Stands in for `·` (U+00B7 MIDDLE DOT): a centred dot between items.
pub const DOT: char = '∙'; // U+2219 BULLET OPERATOR — Neutral

/// Stands in for `—` (U+2014 EM DASH): a dash setting off a clause.
pub const DASH: char = '‒'; // U+2012 FIGURE DASH — Neutral

/// The item separator, `" ∙ "`, for joins and the like.
pub const SEP: &str = " ∙ ";
