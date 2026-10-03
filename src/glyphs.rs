//! Width-safe stand-ins for punctuation terminals disagree about.
//!
//! `·` (U+00B7 MIDDLE DOT), `—` (U+2014 EM DASH), `…`, and the arrows and
//! triangles below are East Asian Width *Ambiguous*: `unicode-width` counts
//! each as one column, but a terminal in a CJK locale renders it as two.
//! Anywhere a line is measured or padded by column — the TUI's alignment,
//! and any `{:<n}` padding — the two disagree, and the row drifts one cell
//! per glyph. That is exactly the kind of half-aligned box that looks like a
//! bug and is hard to place.
//!
//! These replacements are East Asian Width *Neutral*, so every terminal
//! agrees with `unicode-width`, and they look close enough to the originals
//! to read as the same punctuation. Use them wherever a line is measured,
//! padded or wrapped by column — the TUI's rows, status bar, notices and key
//! hints. Output that is only printed and never padded (most of the CLI) and
//! ordinary prose in comments and docs are left as they are.
//!
//! Still Ambiguous and still drawn: the box-drawing rules and borders (`─`,
//! `│`), which have no Neutral glyph that joins up the same way.

/// Stands in for `·` (U+00B7 MIDDLE DOT): a centred dot between items.
pub const DOT: char = '∙'; // U+2219 BULLET OPERATOR — Neutral

/// Stands in for `—` (U+2014 EM DASH): a dash setting off a clause.
pub const DASH: char = '‒'; // U+2012 FIGURE DASH — Neutral

/// The item separator, `" ∙ "`, for joins and the like.
pub const SEP: &str = " ∙ ";

/// Stands in for `…` (U+2026 HORIZONTAL ELLIPSIS): text cut short.
pub const ELLIPSIS: char = '⋯'; // U+22EF MIDLINE HORIZONTAL ELLIPSIS — Neutral

/// Stand in for `↑` `↓` `←` `→` (U+2190–2193) naming the arrow keys, and for
/// `▲` `▼` (U+25B2, U+25BC) pointing up and down.
pub const UP: char = '⏶'; // U+23F6 — Neutral
pub const DOWN: char = '⏷'; // U+23F7 — Neutral
pub const LEFT: char = '⏴'; // U+23F4 — Neutral
pub const RIGHT: char = '⏵'; // U+23F5 — Neutral

/// Stands in for `→` (U+2192) where it means "became": a rename, a count
/// that changed.
pub const ARROW: char = '➔'; // U+2794 HEAVY WIDE-HEADED RIGHTWARDS ARROW — Neutral

#[cfg(test)]
mod tests {
    use super::*;

    /// `width_cjk` is the width a CJK-locale terminal gives a character, so
    /// an Ambiguous one is 2 there: one cell both ways is what Neutral buys.
    #[test]
    fn every_stand_in_is_one_cell_in_every_terminal() {
        use unicode_width::UnicodeWidthChar;
        let stand_ins = SEP
            .chars()
            .chain([DOT, DASH, ELLIPSIS, UP, DOWN, LEFT, RIGHT, ARROW]);
        for glyph in stand_ins {
            assert_eq!(glyph.width(), Some(1), "{glyph:?}");
            assert_eq!(glyph.width_cjk(), Some(1), "{glyph:?} is wide in CJK");
        }
    }
}
