//! Folding the older part of a clanker's history into a summary, so a long
//! conversation stops resending everything it has ever said.
//!
//! Every request carries the whole history, so the cost of a turn grows with
//! the conversation rather than with the question — a file read early on is
//! paid for again on every turn after it. Compaction replaces that older
//! stretch with a summary of it, produced by a model chosen for the job
//! (`clank compactor`) rather than by the clanker's own.
//!
//! **Nothing is deleted.** The messages stay in the database and in the
//! transcript; what changes is only which of them
//! [`crate::session::ChatSession::request_messages`] hands the provider. A
//! compacted clanker still scrolls back to its first word, and a summary
//! that turns out to have dropped something important is a reason to read
//! further up, not a hole.

use anyhow::Result;

use crate::client::{ChatMessage, Client};

/// How many of the most recent user turns are never folded away.
///
/// A ceiling, not a promise: the seam starts here and moves later if keeping
/// this many turns would keep more than [`tail_budget`] allows. What is
/// promised is the other end — the turn in progress is never folded, however
/// large it is.
///
/// A summary is a poor substitute for what the model is in the middle of
/// doing; it is a fine substitute for what it finished an hour ago.
const KEEP_RECENT_TURNS: usize = 2;

/// What fraction of the compaction threshold the kept tail may occupy.
///
/// Turn count alone is the wrong bound, and real clankers show why: one
/// instruction followed by forty tool calls is a single "turn" worth 30k
/// tokens, so keeping two of them left a floor of ~33k against a 60k
/// threshold — compaction crossed the line again within a turn or two and
/// spent a compactor call each time to fold one instruction's work.
///
/// A quarter leaves three quarters of the threshold as headroom, so a
/// clanker compacts once every few turns rather than every other one. Taken
/// as a fraction rather than a fixed number of tokens because a threshold
/// lowered to 10k would otherwise be unreachable: a tail that cannot fit
/// under the line means compacting on every single turn and never getting
/// below it.
const TAIL_FRACTION: u64 = 4;

/// What fraction of the compaction threshold a summary may occupy.
///
/// Nothing used to bound this at all, and a summary is the one part of a
/// compacted request that compounds: each one is fed back in whole to write
/// the next, so a compactor inclined to write at length raises the floor a
/// little on every folding. The tail budget can't see it happen, because the
/// summary is deliberately left out of [`estimated_tokens`] — it isn't
/// written yet when the seam is chosen.
///
/// An eighth, so the summary and the kept tail together take under half the
/// threshold and a compacted clanker has somewhere to grow. Taken as a
/// fraction for the same reason [`TAIL_FRACTION`] is: a fixed size would be
/// most of the budget once the threshold is lowered.
const SUMMARY_FRACTION: u64 = 8;

/// Bytes per token, for judging a span without a tokenizer.
///
/// Deliberately low. Tool-call JSON and file dumps tokenize far worse than
/// prose, and those are what a compacted history is made of — measured
/// against a real provider-reported prompt on a tool-heavy clanker, the
/// whole request came out at ~2.6 bytes per token. Erring small overestimates
/// the tail, which folds slightly more than needed; erring large would let
/// the tail creep over the threshold, which is the failure this exists to
/// prevent.
const BYTES_PER_TOKEN: usize = 3;

/// The longest any single message is rendered at inside the compaction
/// request. A 200KB file dump is exactly what makes a history worth
/// compacting, and also exactly what would blow the compactor's own context
/// if it were sent whole — so the head and tail of one go in and the middle
/// is named rather than included.
const MAX_RENDERED: usize = 6_000;

/// What the compactor is asked to produce. Written as instructions to a
/// model that will see the transcript as a single user message rather than
/// as a conversation to join — hence the explicit "do not answer it".
const COMPACTION_PROMPT: &str = "\
You are compacting the earlier part of a conversation between a user and an \
AI assistant so that it can be carried forward in far fewer tokens.

Write a summary that lets the assistant continue the conversation without \
having seen the original. Preserve, in whatever structure suits the material:

- What the user is trying to accomplish, and any constraints or preferences \
they stated.
- Decisions that were made, and the reasons given for them.
- Files, paths, commands, identifiers, versions and error messages exactly as \
they appeared — these are what the assistant will need to act on, and a \
paraphrased path is a useless one.
- The current state of the work: what is done, what is in progress, what was \
tried and abandoned.
- Anything still open — unanswered questions, known problems, agreed next \
steps.

Leave out pleasantries, restated instructions, and the full text of anything \
already captured by its result. Do not answer the conversation, continue it, \
or address the user: produce only the summary. Do not invent anything that is \
not in the transcript.";

/// How a summary is framed when it goes back out as part of a request.
///
/// Public because [`crate::session::ChatSession::request_messages`] builds
/// the message and this decides what it says — it prefixes this onto the
/// first message past the seam rather than sending it as one of its own, for
/// reasons documented there. The framing matters either way: without it the
/// model reads a summary of its own conversation as something the user just
/// wrote to it.
pub fn summary_message(summary: &str) -> String {
    format!(
        "[The earlier part of this conversation has been compacted to save \
         context. What follows is a summary of it, standing in for the \
         messages themselves; the conversation then continues verbatim from \
         after that point.]\n\n{summary}\n\n[End of summary. The conversation \
         resumes here.]"
    )
}

/// Where the history can be cut, given that `from` messages are already
/// folded away.
///
/// The cut lands on a user message, and only on a user message. Two reasons,
/// both of which produce an invalid request if ignored: a `tool` message
/// whose `tool_calls` parent was folded away references a call the provider
/// can no longer see, and an assistant message that ends in a reasoning
/// block needs the tool result that followed it. Cutting at a user message
/// leaves both pairings intact on the far side, because a user message never
/// sits in the middle of one.
///
/// `None` when there is nothing worth folding: fewer than
/// [`KEEP_RECENT_TURNS`] user turns past the existing seam, or a cut that
/// would not advance it.
pub fn seam(messages: &[ChatMessage], from: usize, compact_at: Option<u64>) -> Option<usize> {
    let user_turns: Vec<usize> = messages
        .iter()
        .enumerate()
        .skip(from)
        .filter(|(_, m)| m.role == "user")
        .map(|(i, _)| i)
        .collect();

    if user_turns.is_empty() {
        return None;
    }

    // Where turn count alone would put it, then later if what that keeps is
    // too big to be worth keeping. Never past the last one: the exchange in
    // progress stays verbatim even when it alone blows the budget, because
    // there is nothing else left to fold and summarizing what the model is
    // in the middle of is how a turn loses the thread.
    let last = user_turns.len() - 1;
    let budget = tail_budget(compact_at);
    let mut at = user_turns.len().saturating_sub(KEEP_RECENT_TURNS);
    while at < last && estimated_tokens(&messages[user_turns[at]..]) > budget {
        at += 1;
    }

    // Not `>=`: a seam that doesn't move folds nothing and would rewrite the
    // same summary for the same span.
    let cut = user_turns[at];
    (cut > from).then_some(cut)
}

/// How many tokens the kept tail may occupy, from the threshold that will be
/// measured against it. Falls back to the default threshold for a clanker
/// with automatic compaction turned off, where `/compact` is the only way in
/// and there is no configured number to derive from.
fn tail_budget(compact_at: Option<u64>) -> u64 {
    compact_at.unwrap_or(crate::config::DEFAULT_COMPACT_AT) / TAIL_FRACTION
}

/// How many tokens a summary may occupy, from the same threshold, falling
/// back the same way.
fn summary_budget(compact_at: Option<u64>) -> u64 {
    compact_at.unwrap_or(crate::config::DEFAULT_COMPACT_AT) / SUMMARY_FRACTION
}

/// Cuts a summary that overran its budget, keeping the start.
///
/// The backstop, not the mechanism: what is supposed to keep a summary short
/// is asking for it short, and this is what happens when that is ignored. The
/// start is kept because a summary is written to be read from the top, and
/// because the alternative — refusing it — throws away the compaction and
/// leaves the clanker to meet the threshold again on its next turn with
/// nothing to show for the call.
///
/// Said out loud in the text, because a summary that stops mid-sentence with
/// no explanation reads as a summary of a conversation that stopped there.
fn truncate_summary(summary: String, budget: u64) -> String {
    let cap = budget as usize * BYTES_PER_TOKEN;
    if summary.len() <= cap {
        return summary;
    }

    let mut end = cap;
    while end > 0 && !summary.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n\n[The summary ran past what a compacted request can carry and \
         was cut here.]",
        &summary[..end]
    )
}

/// Roughly what a span costs, for choosing between candidate seams.
///
/// An estimate is enough here and a tokenizer would not be worth carrying:
/// this only ever decides which user turn the seam lands on, so being a third
/// out moves it by at most one turn. The summary that will sit in front of
/// the tail isn't counted — it isn't written yet when this runs — which is
/// part of why [`BYTES_PER_TOKEN`] errs on the small side.
///
/// Everything counted here is something that goes back on the wire, which is
/// why `reasoning_details` is in and `reasoning` is out: the details carry
/// the signature a provider needs echoed back and are serialized into every
/// follow-up request, while the plain-text reasoning beside them is display
/// only and never sent. Leaving the details out undercounted the tail of
/// exactly the clankers that grow fastest — a reasoning model working
/// through a long run of tool calls.
fn estimated_tokens(messages: &[ChatMessage]) -> u64 {
    let bytes: usize = messages
        .iter()
        .map(|message| {
            message.content.as_deref().map_or(0, str::len)
                + message.tool_calls.as_ref().map_or(0, |calls| {
                    calls
                        .iter()
                        .map(|call| call.function.name.len() + call.function.arguments.len())
                        .sum::<usize>()
                })
                + message.reasoning_details.as_ref().map_or(0, |blocks| {
                    blocks.iter().map(|block| block.to_string().len()).sum()
                })
        })
        .sum();
    (bytes / BYTES_PER_TOKEN) as u64
}

/// Renders one message the way the compactor should read it: who said it,
/// what they said, and what any tool call asked for or returned.
///
/// Deliberately not JSON. The compactor is summarizing a conversation, and a
/// transcript reads as one; the message array's own shape carries rules
/// (`tool_call_id` pairing, reasoning-block signatures) that mean nothing to
/// the job and cost tokens to include.
fn render(message: &ChatMessage) -> String {
    let mut out = String::new();

    let speaker = match message.role.as_str() {
        "user" => "User",
        "assistant" => "Assistant",
        "tool" => "Tool result",
        "system" => "System",
        other => other,
    };
    out.push_str(speaker);
    out.push_str(":\n");

    if let Some(content) = message.content.as_deref() {
        if !content.trim().is_empty() {
            out.push_str(&truncate(content));
            out.push('\n');
        }
    }

    if let Some(calls) = &message.tool_calls {
        for call in calls {
            out.push_str(&format!(
                "[calls {} with {}]\n",
                call.function.name,
                truncate(&call.function.arguments)
            ));
        }
    }

    out
}

/// Keeps the head and tail of an over-long string and says how much of the
/// middle went missing, rather than cutting the end off.
///
/// Which end matters depends on what the message was: a file read wants its
/// start, a command's output usually wants its end (the error is at the
/// bottom), and neither is knowable from here. Keeping both is the answer
/// that is never badly wrong.
fn truncate(text: &str) -> String {
    if text.len() <= MAX_RENDERED {
        return text.to_string();
    }

    // On char boundaries, so this can't panic on a multi-byte character —
    // tool output is arbitrary text and routinely contains them.
    let head = floor_boundary(text, MAX_RENDERED / 2);
    let tail = ceil_boundary(text, text.len() - MAX_RENDERED / 2);
    let omitted = tail - head;
    format!(
        "{}\n[... {omitted} bytes omitted ...]\n{}",
        &text[..head],
        &text[tail..]
    )
}

/// The largest char boundary at or below `at`.
fn floor_boundary(text: &str, at: usize) -> usize {
    let mut at = at.min(text.len());
    while at > 0 && !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

/// The smallest char boundary at or above `at`.
fn ceil_boundary(text: &str, at: usize) -> usize {
    let mut at = at.min(text.len());
    while at < text.len() && !text.is_char_boundary(at) {
        at += 1;
    }
    at
}

/// The single user message the compactor is sent: the previous summary, if
/// there is one, then the span being folded into it.
fn request_text(previous: Option<&str>, span: &[ChatMessage]) -> String {
    let mut text = String::new();

    if let Some(previous) = previous {
        text.push_str(
            "Summary of the conversation before this transcript, which your \
             summary must also cover and supersede:\n\n",
        );
        text.push_str(previous);
        text.push_str("\n\n---\n\n");
    }

    text.push_str("Transcript to summarize:\n\n");
    for message in span {
        text.push_str(&render(message));
        text.push('\n');
    }
    text
}

/// What one compaction produced.
#[derive(Debug)]
pub struct Compacted {
    /// The summary, ready to hand to
    /// [`crate::session::ChatSession::set_compaction`].
    pub summary: String,
    /// Tokens the compaction request itself cost. Real spend on the
    /// clanker's behalf, so it belongs in the clanker's running total — the
    /// point of the feature is to be cheaper overall, which is a claim the
    /// user can only check if this is counted.
    pub tokens: u64,
}

/// Summarizes `messages[from..cut]`, superseding `previous` if the session
/// has been compacted before.
///
/// One non-streaming request, with no tools and no reasoning effort: there
/// is nothing to stream to (the caller is between turns, not rendering), and
/// nothing for a tool to do. Temperature is left to the provider — a summary
/// is not a place to want variety, but naming a number here would override a
/// deliberate choice made for the endpoint as a whole.
pub async fn compact(
    client: &Client,
    model: &str,
    previous: Option<&str>,
    span: &[ChatMessage],
    compact_at: Option<u64>,
) -> Result<Compacted> {
    let budget = summary_budget(compact_at);
    // Asked for, then enforced. The asking is what usually works and what
    // keeps the summary coherent; `truncate_summary` below is only there for
    // a compactor that ignores it. Words rather than tokens because that is
    // the unit a model can actually aim at, at roughly three quarters of one
    // per token.
    let instruction = format!(
        "{COMPACTION_PROMPT}\n\nKeep the summary under about {} words. It is \
         carried in every request from here on, and folded into the next \
         summary after this one, so length here is paid for repeatedly.",
        budget * 3 / 4
    );

    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: Some(instruction),
            ..Default::default()
        },
        ChatMessage {
            role: "user".to_string(),
            content: Some(request_text(previous, span)),
            ..Default::default()
        },
    ];

    let response = client
        .chat(model.to_string(), messages, None, None, None)
        .await?;

    let tokens = response.usage.map(|u| u.total_tokens).unwrap_or(0);
    let summary = response
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message.content)
        .unwrap_or_default();

    // An empty summary is worse than no compaction: the seam would move past
    // messages that nothing stands in for, and the conversation would carry
    // on with a hole where its first half used to be.
    if summary.trim().is_empty() {
        anyhow::bail!("The compactor returned an empty summary, so nothing was compacted");
    }

    Ok(Compacted {
        summary: truncate_summary(summary, budget),
        tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: Some(content.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn the_tail_estimate_counts_reasoning_that_goes_back_on_the_wire() {
        let plain = [message("assistant", "done")];

        let mut with_details = message("assistant", "done");
        with_details.reasoning_details = Some(vec![
            serde_json::json!({"type": "thinking", "text": "x".repeat(3_000)}),
        ]);

        assert!(
            estimated_tokens(&[with_details]) > estimated_tokens(&plain) + 900,
            "reasoning_details is serialized into every follow-up request, so a \
             tail full of it is not the size this used to report"
        );
    }

    #[test]
    fn the_tail_estimate_ignores_reasoning_that_never_leaves_the_process() {
        let plain = [message("assistant", "done")];

        let mut with_prose = message("assistant", "done");
        with_prose.reasoning = Some("x".repeat(3_000));

        assert_eq!(
            estimated_tokens(&[with_prose]),
            estimated_tokens(&plain),
            "the plain-text reasoning is display only and is never sent, so \
             counting it would move the seam for tokens nobody pays for"
        );
    }

    #[test]
    fn a_summary_within_its_budget_is_left_alone() {
        let summary = "They fixed the build.".to_string();
        assert_eq!(truncate_summary(summary.clone(), 1_000), summary);
    }

    #[test]
    fn a_summary_over_its_budget_is_cut_and_says_that_it_was() {
        // A summary that stops mid-sentence with no explanation reads as a
        // summary of a conversation that stopped there.
        let budget = 100;
        let long = "word ".repeat(1_000);
        let cut = truncate_summary(long, budget);

        assert!(cut.starts_with("word word"), "the start survived");
        assert!(cut.contains("was cut here"), "{cut:.200}");
        assert!(
            cut.len() < budget as usize * BYTES_PER_TOKEN + 100,
            "{} bytes",
            cut.len()
        );
    }

    #[test]
    fn cutting_a_summary_never_splits_a_character() {
        let cut = truncate_summary("é".repeat(1_000), 10);
        let body = cut.split("\n\n").next().unwrap();
        assert!(body.chars().all(|c| c == 'é'), "{body:?}");
    }

    #[test]
    fn both_budgets_follow_the_threshold_rather_than_a_fixed_number() {
        // A threshold lowered to 10k has to lower what sits under it, or
        // there is no configuration of this that ever gets below the line.
        assert!(summary_budget(Some(10_000)) < summary_budget(Some(60_000)));
        assert!(summary_budget(Some(60_000)) < tail_budget(Some(60_000)));
        assert_eq!(
            summary_budget(None),
            summary_budget(Some(crate::config::DEFAULT_COMPACT_AT)),
            "a clanker with automatic compaction off still reaches this by /compact"
        );
    }

    #[test]
    fn seam_lands_on_a_user_message() {
        let messages = [
            message("user", "one"),
            message("assistant", "reply"),
            message("user", "two"),
            message("assistant", "reply"),
            message("user", "three"),
        ];
        // Three user turns, two kept: the cut is the second one.
        assert_eq!(seam(&messages, 0, None), Some(2));
        assert_eq!(messages[2].role, "user");
    }

    #[test]
    fn a_tool_result_is_never_left_orphaned_at_the_seam() {
        let messages = [
            message("user", "do it"),
            message("assistant", "calling"),
            message("tool", "done"),
            message("user", "again"),
            message("assistant", "calling"),
            message("tool", "done"),
            message("user", "and again"),
        ];
        let cut = seam(&messages, 0, None).expect("three user turns is enough to fold one");
        assert_eq!(
            messages[cut].role, "user",
            "cutting anywhere else strands a tool result from the call that produced it"
        );
    }

    #[test]
    fn nothing_to_fold_until_there_are_turns_past_the_ones_kept() {
        let messages = [
            message("user", "one"),
            message("assistant", "reply"),
            message("user", "two"),
        ];
        assert_eq!(seam(&messages, 0, None), None);
    }

    #[test]
    fn a_second_compaction_starts_from_the_existing_seam() {
        let messages = [
            message("user", "one"),
            message("user", "two"),
            message("user", "three"),
            message("user", "four"),
            message("user", "five"),
        ];
        // Already folded through 2, so only "three"/"four"/"five" are in
        // play and the cut is the second-to-last of those.
        assert_eq!(seam(&messages, 2, None), Some(3));
    }

    #[test]
    fn the_seam_has_to_advance_to_be_worth_taking() {
        let messages = [
            message("user", "one"),
            message("assistant", "reply"),
            message("user", "two"),
            message("user", "three"),
        ];
        // From 2 there are two user turns left, which is exactly what is
        // kept — folding would move nothing.
        assert_eq!(seam(&messages, 2, None), None);
    }

    /// A message of `tokens` estimated size, as a tool result would be.
    fn bulk(role: &str, tokens: usize) -> ChatMessage {
        message(role, &"x".repeat(tokens * BYTES_PER_TOKEN))
    }

    #[test]
    fn a_tail_too_big_for_the_budget_moves_the_seam_later() {
        // The shape a real clanker had: one instruction, then a long run of
        // tool calls, repeated. Keeping two of those "turns" kept ~40k
        // against a 60k threshold, so the prompt crossed the line again
        // almost immediately.
        let messages = [
            message("user", "first instruction"),
            bulk("tool", 20_000),
            message("user", "second instruction"),
            bulk("tool", 20_000),
            message("user", "third instruction"),
            bulk("tool", 5_000),
        ];

        // Turn count alone keeps the last two: from index 2, ~45k of tail.
        assert_eq!(
            seam(&messages, 0, Some(60_000)),
            Some(4),
            "the budget is 15k, so the seam moves on to the last turn"
        );
    }

    #[test]
    fn the_turn_in_progress_is_never_folded_however_big_it_is() {
        // Nothing else is left to fold, and summarizing what the model is in
        // the middle of is how a turn loses the thread.
        let messages = [
            message("user", "first"),
            bulk("tool", 5_000),
            message("user", "second"),
            bulk("tool", 90_000),
        ];
        assert_eq!(seam(&messages, 0, Some(60_000)), Some(2));
    }

    #[test]
    fn the_budget_follows_the_threshold_rather_than_a_fixed_number() {
        // A threshold lowered to 10k with a fixed 15k tail could never get
        // under the line: it would compact on every turn and never succeed.
        let messages = [
            message("user", "first"),
            bulk("tool", 4_000),
            message("user", "second"),
            bulk("tool", 4_000),
            message("user", "third"),
            bulk("tool", 500),
        ];
        // 60k threshold: a 15k budget, and ~8.5k of tail from index 2 fits.
        assert_eq!(seam(&messages, 0, Some(60_000)), Some(2));
        // 10k threshold: a 2.5k budget, so it has to move on.
        assert_eq!(seam(&messages, 0, Some(10_000)), Some(4));
    }

    #[test]
    fn an_over_long_message_keeps_both_ends() {
        let text = format!("START{}END", "x".repeat(MAX_RENDERED * 2));
        let rendered = truncate(&text);
        assert!(rendered.starts_with("START"));
        assert!(rendered.ends_with("END"));
        assert!(rendered.contains("bytes omitted"));
        assert!(rendered.len() < text.len());
    }

    #[test]
    fn truncating_never_splits_a_character() {
        // Every byte position in the middle of this is inside a multi-byte
        // char, so a naive slice would panic.
        let text = "é".repeat(MAX_RENDERED);
        let rendered = truncate(&text);
        assert!(rendered.contains("bytes omitted"));
    }

    #[test]
    fn a_short_message_is_rendered_whole() {
        assert_eq!(truncate("hello"), "hello");
    }

    #[test]
    fn the_request_carries_the_previous_summary_forward() {
        let span = [message("user", "the new part")];
        let text = request_text(Some("what came before"), &span);
        assert!(text.contains("what came before"));
        assert!(text.contains("the new part"));
        assert!(
            text.find("what came before") < text.find("the new part"),
            "the earlier summary has to read as earlier"
        );
    }

    /// Answers exactly one request with `body`, and hands back what it was
    /// asked. Enough of an HTTP server for one round trip, which is what
    /// this needs: the point is that a real reqwest call reaches a real
    /// socket and the reply parses, not that the server is any good.
    async fn one_shot(body: &'static str) -> (String, tokio::task::JoinHandle<String>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}/v1", listener.local_addr().unwrap());

        let served = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0u8; 4096];
            // Read until the body is complete, which the header says the
            // length of — a chat request is far too big for one read.
            loop {
                let read = socket.read(&mut buffer).await.unwrap();
                request.extend_from_slice(&buffer[..read]);
                let text = String::from_utf8_lossy(&request).to_string();
                let Some((head, so_far)) = text.split_once("\r\n\r\n") else {
                    continue;
                };
                let length: usize = head
                    .lines()
                    .find_map(|line| {
                        line.strip_prefix("content-length: ")
                            .or_else(|| line.strip_prefix("Content-Length: "))
                    })
                    .and_then(|value| value.trim().parse().ok())
                    .unwrap_or(0);
                if so_far.len() >= length || read == 0 {
                    break;
                }
            }

            socket
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                         Content-Length: {}\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
            socket.flush().await.unwrap();

            String::from_utf8_lossy(&request).to_string()
        });

        (base_url, served)
    }

    fn client_for(base_url: String) -> Client {
        Client::for_test(crate::config::Config {
            base_url,
            ..crate::config::Config::default()
        })
    }

    #[tokio::test]
    async fn a_summary_comes_back_with_what_it_cost() {
        let (base_url, served) = one_shot(
            r#"{"choices":[{"message":{"role":"assistant","content":"They fixed the build."}}],
                "usage":{"total_tokens":420,"prompt_tokens":400}}"#,
        )
        .await;

        let span = [message("user", "the build is broken")];
        let compacted = compact(&client_for(base_url), "small/model", None, &span, None)
            .await
            .unwrap();

        assert_eq!(compacted.summary, "They fixed the build.");
        // Compaction is not free, and a clanker's running total has to say so.
        assert_eq!(compacted.tokens, 420);

        let request = served.await.unwrap();
        assert!(request.contains("small/model"), "sent to the compactor");
        assert!(request.contains("the build is broken"), "carries the span");
        // Asking is what keeps a summary short and coherent; the truncation
        // is only the backstop for a compactor that ignores the ask.
        assert!(
            request.contains("words"),
            "the request has to say how long the summary may be"
        );
    }

    #[tokio::test]
    async fn an_empty_summary_is_refused_rather_than_applied() {
        // Applying it would advance the seam past messages that nothing
        // stands in for — the conversation would carry on with a hole where
        // its first half used to be.
        let (base_url, served) = one_shot(
            r#"{"choices":[{"message":{"role":"assistant","content":"   "}}],"usage":{"total_tokens":10}}"#,
        )
        .await;

        let span = [message("user", "something")];
        let error = compact(&client_for(base_url), "small/model", None, &span, None)
            .await
            .expect_err("an empty summary is not a compaction");
        assert!(error.to_string().contains("empty summary"), "{error}");
        let _ = served.await;
    }

    #[test]
    fn a_tool_call_is_rendered_with_what_it_asked_for() {
        let message = ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![crate::client::ToolCall {
                id: "call_1".to_string(),
                call_type: crate::client::function_call_type(),
                function: crate::client::FunctionCall {
                    name: "read_file".to_string(),
                    arguments: r#"{"path":"src/main.rs"}"#.to_string(),
                },
            }]),
            ..Default::default()
        };
        let rendered = render(&message);
        assert!(rendered.contains("read_file"));
        assert!(rendered.contains("src/main.rs"));
    }
}
