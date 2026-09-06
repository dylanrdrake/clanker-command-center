use crate::client::{ChatMessage, Client, StreamEvent};
use crate::config::{SessionGates, ToolAccess, ToolAccessSettings};
use crate::tools::{execute_tool, get_tool_definitions};
use crate::ui::{AgentEvent, AgentUi, ApprovalRequest};
use anyhow::Result;
use futures_util::{pin_mut, StreamExt};
use serde_json::json;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Seeds a continuous agent-chat session so the model treats the growing
/// transcript as history to build on, not a backlog of tasks to redo.
pub const AGENT_CHAT_SYSTEM_PROMPT: &str = "You are a coding agent operating in a continuous \
interactive chat session. The conversation history may contain earlier user requests that you \
already completed, along with your replies and any tool calls/results for them. Treat each new \
user message as the only request currently being asked of you - use earlier turns purely as \
background context. Do not restate, re-summarize, or redo work from earlier turns unless the \
user explicitly asks you to.";

/// The tools this run may offer the model.
///
/// A tool set to `Never` is left out of the request entirely rather than
/// refused when it is called: it costs no tokens, and the model is never
/// invited to try something it cannot have. The refusal at the call site
/// still exists, because this list is built once per turn and a gate can be
/// changed while that turn runs.
fn offered_tools(access: &ToolAccessSettings) -> Vec<serde_json::Value> {
    get_tool_definitions()
        .into_iter()
        .filter(|definition| {
            let name = definition["function"]["name"].as_str().unwrap_or_default();
            access.access(name) != ToolAccess::Never
        })
        .collect()
}

// Every parameter here is a distinct, independently-overridable request
// setting (model, iteration cap, temperature, tool access, effort) —
// bundling them into a struct wouldn't simplify anything, just move the
// same list one level out.
#[allow(clippy::too_many_arguments)]
pub async fn run_agent(
    client: &Client,
    ui: &mut impl AgentUi,
    task: &str,
    model: &str,
    max_iterations: Option<usize>,
    temperature: Option<f32>,
    gates: &SessionGates,
    effort_level: Option<String>,
    stream: bool,
) -> Result<Option<String>> {
    let mut messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(task.to_string()),
        tool_calls: None,
        tool_call_id: None,
        ..Default::default()
    }];

    run_agent_turn(
        client,
        ui,
        &mut messages,
        model,
        max_iterations,
        temperature,
        gates,
        effort_level,
        stream,
        &Steering::default(),
        &UsageTracker::default(),
    )
    .await
}

/// Performs one request to the model and returns the assembled reply,
/// streaming it if the user has streaming on.
///
/// Both paths produce the same `ChatMessage`; streaming additionally emits
/// [`AgentEvent::AssistantDelta`] as text arrives, so a front end that can
/// re-render (a TUI) shows it live while one that can't (the CLI) simply
/// ignores the deltas and renders the finished message.
// Same reasoning as `run_agent`: the parameters are the turn's inputs,
// and a struct would only move the argument list somewhere else.
#[allow(clippy::too_many_arguments)]
async fn request_turn(
    client: &Client,
    ui: &mut impl AgentUi,
    mut messages: Vec<ChatMessage>,
    model: &str,
    temperature: Option<f32>,
    tools: Option<Vec<serde_json::Value>>,
    effort_level: Option<String>,
    stream: bool,
    usage: &UsageTracker,
) -> Result<ChatMessage> {
    normalize_system_prompt(&mut messages, tools.is_some());

    // Passed in rather than read off the client: streaming is a per-session
    // setting now, and one client is shared by every session in a process.
    let mut message = if stream {
        let stream = client.chat_stream(
            model.to_string(),
            messages,
            temperature,
            tools,
            effort_level,
        );
        pin_mut!(stream);

        let mut assembled = None;
        while let Some(event) = stream.next().await {
            match event? {
                StreamEvent::Content(text) => {
                    ui.event(AgentEvent::AssistantDelta { text }).await;
                }
                StreamEvent::Done {
                    message,
                    usage: request_usage,
                } => {
                    if let Some(request_usage) = request_usage {
                        usage.add(request_usage.total_tokens);
                    }
                    assembled = Some(message);
                }
            }
        }

        assembled.ok_or_else(|| anyhow::anyhow!("Response stream ended without a message"))?
    } else {
        let response = client
            .chat(
                model.to_string(),
                messages,
                temperature,
                tools,
                effort_level,
            )
            .await?;
        if let Some(request_usage) = response.usage {
            usage.add(request_usage.total_tokens);
        }
        response
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message)
            .ok_or_else(|| anyhow::anyhow!("Provider returned no choices"))?
    };

    // Before the blocks are pruned: what's shown to the user isn't bound by
    // the rules about what may be sent back to a provider.
    if let Some(text) = message.thinking_text() {
        ui.event(AgentEvent::Thinking { text }).await;
    }

    drop_dangling_reasoning(&mut message);
    Ok(message)
}

/// Some providers (Anthropic among them) reject a `system`-role message
/// that isn't the very first entry — it has to immediately follow a user
/// message or a tool-result-ending assistant message, which in practice
/// means it can only ever validly sit at the start of the conversation.
/// `/tools` can turn tool-calling on at any point mid-conversation, so
/// there's no position session.rs could insert the agent system prompt at
/// that's guaranteed to stay valid as the conversation grows around it.
///
/// So it isn't stored history at all: any stray copy already sitting
/// somewhere in the array — from before this existed, or left over from a
/// clanker whose tools have since been turned off — is dropped, and a
/// fresh one is prepended exactly when this turn actually needs it
/// (`agentic`, i.e. tools are in play). This heals an already-poisoned
/// session automatically, the same way `strip_dangling_reasoning` does for
/// reasoning content, without needing to touch what's actually persisted.
fn normalize_system_prompt(messages: &mut Vec<ChatMessage>, agentic: bool) {
    messages.retain(|m| {
        !(m.role == "system" && m.content.as_deref() == Some(AGENT_CHAT_SYSTEM_PROMPT))
    });
    if agentic {
        messages.insert(
            0,
            ChatMessage {
                role: "system".to_string(),
                content: Some(AGENT_CHAT_SYSTEM_PROMPT.to_string()),
                ..Default::default()
            },
        );
    }
}

/// Anthropic rejects a stored assistant message whose final content block
/// would be `thinking` — exactly what `reasoning_details` becomes once
/// translated, unless a tool_use block follows it. Only keep it when
/// there's a tool call to follow, since that's the only case it's actually
/// needed for (see `ChatMessage::reasoning_details`) — a turn that reasoned
/// but didn't end up calling anything would otherwise poison every later
/// request in the conversation.
fn drop_dangling_reasoning(message: &mut ChatMessage) {
    if !message.has_tool_calls() {
        message.reasoning_details = None;
    }
}

/// Runs one plain (non-agentic) exchange: send the history, report the
/// reply, append it. The `chat` counterpart to [`run_agent_turn`], so both
/// modes reach a front end through the same events instead of `chat` being
/// open-coded by each caller.
// Same reasoning as `run_agent`/`run_agent_turn`: the parameters are the
// turn's inputs, and a struct would only move the argument list somewhere
// else.
#[allow(clippy::too_many_arguments)]
pub async fn run_chat_turn(
    client: &Client,
    ui: &mut impl AgentUi,
    messages: &mut Vec<ChatMessage>,
    model: &str,
    temperature: Option<f32>,
    effort_level: Option<String>,
    stream: bool,
    usage: &UsageTracker,
) -> Result<Option<String>> {
    ui.event(AgentEvent::RequestStarted).await;
    let turn = request_turn(
        client,
        ui,
        messages.clone(),
        model,
        temperature,
        None,
        effort_level.clone(),
        stream,
        usage,
    )
    .await;
    ui.event(AgentEvent::RequestFinished).await;
    let message = turn?;

    let mut final_response = None;
    // Matches the CLI's long-standing behavior: a reply with nothing visible
    // in it is neither shown nor added to the history.
    if message.has_visible_content() {
        let content = message.content.as_deref().unwrap().to_string();
        ui.event(AgentEvent::AssistantMessage {
            model: model.to_string(),
            effort_level,
            text: content.clone(),
        })
        .await;
        final_response = Some(content);
        messages.push(message);
    }

    ui.event(AgentEvent::TurnFinished).await;
    Ok(final_response)
}

/// Runs the tool-calling agent loop against an existing message history,
/// appending the assistant/tool messages produced along the way so the
/// history can be reused for a follow-up turn (e.g. a continuous chat).
///
/// Progress is reported to `ui` rather than printed, and any tool needing
/// permission is put to `ui` as an [`ApprovalRequest`], so the same loop
/// drives the CLI, a GUI, or a test harness unchanged.
// See `run_agent`'s note on why this isn't bundled into a params struct.
#[allow(clippy::too_many_arguments)]
/// Messages typed while a turn is running, waiting to join it.
///
/// A turn is many requests, and the array sent with each one is built fresh,
/// so a message can be added between them without disturbing anything in
/// flight. The loop takes everything waiting at the top of each iteration,
/// which is the first legal place to put it: the previous iteration's tool
/// results have completed their pairing with the calls that produced them,
/// and the next request has not been built yet. Slipping a user message
/// between a `tool_calls` message and its results would make the request
/// invalid.
///
/// Empty for callers with nowhere to type — a one-shot `clank agent` task
/// has no seam to steer through, and passing an idle handle is how they say
/// so.
///
/// Cheap to clone: every clone reads and writes the same queue.
#[derive(Clone, Debug, Default)]
pub struct Steering {
    pending: Arc<Mutex<VecDeque<String>>>,
}

impl Steering {
    /// Adds a message and reports how many are now waiting.
    pub fn push(&self, text: String) -> usize {
        let mut pending = self.pending.lock().expect("steering queue poisoned");
        pending.push_back(text);
        pending.len()
    }

    /// Takes everything waiting, leaving the handle empty. Returns owned
    /// values rather than a guard so the lock is never held across an await.
    pub fn take(&self) -> Vec<String> {
        let mut pending = self.pending.lock().expect("steering queue poisoned");
        pending.drain(..).collect()
    }
}

/// Sums the token usage a turn's requests report, so a caller can add it to
/// a clanker's running total once the turn finishes.
///
/// A turn is one request with no tools, or as many as `max_iterations` with
/// them — and a provider reports usage per request, not per turn — so this
/// is where the pieces get added up. Cheap to clone, like [`Steering`]: the
/// turn runs on its own task, and the caller needs a handle it can still
/// read from after handing the other one in.
#[derive(Clone, Debug, Default)]
pub struct UsageTracker {
    total: Arc<Mutex<u64>>,
}

impl UsageTracker {
    fn add(&self, tokens: u64) {
        *self.total.lock().expect("usage tracker poisoned") += tokens;
    }

    /// Everything accumulated so far, in tokens.
    pub fn total(&self) -> u64 {
        *self.total.lock().expect("usage tracker poisoned")
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn run_agent_turn(
    client: &Client,
    ui: &mut impl AgentUi,
    messages: &mut Vec<ChatMessage>,
    model: &str,
    max_iterations: Option<usize>,
    temperature: Option<f32>,
    gates: &SessionGates,
    effort_level: Option<String>,
    stream: bool,
    steering: &Steering,
    usage: &UsageTracker,
) -> Result<Option<String>> {
    // Unlike `temperature`/`effort_level`, there's no provider to fall back
    // to a default for this — it never leaves the process, so a missing cap
    // can't be sent as "no value" the way an omitted request field can. Fail
    // clearly up front rather than picking a number on the caller's behalf.
    let max_iterations = max_iterations.ok_or_else(|| {
        anyhow::anyhow!(
            "No max-iterations cap is set. Set one with /max-iterations <n> for this session, \
             or clank max-iterations <n> as the persistent default."
        )
    })?;

    let tool_definitions = offered_tools(&gates.access());
    let mut iteration = 0;
    let mut final_response = None;

    while iteration < max_iterations {
        iteration += 1;

        // Anything typed since the last request joins the turn here, so it
        // reaches the model on this iteration's call rather than waiting for
        // the turn to finish and starting one of its own.
        for text in steering.take() {
            ui.event(AgentEvent::Steered { text: text.clone() }).await;
            messages.push(ChatMessage {
                role: "user".to_string(),
                content: Some(text),
                tool_calls: None,
                tool_call_id: None,
                ..Default::default()
            });
        }

        ui.event(AgentEvent::IterationStarted { iteration }).await;

        // Call the LLM with tool definitions
        ui.event(AgentEvent::RequestStarted).await;
        let turn = request_turn(
            client,
            ui,
            messages.clone(),
            model,
            temperature,
            Some(tool_definitions.clone()),
            effort_level.clone(),
            stream,
            usage,
        )
        .await;
        ui.event(AgentEvent::RequestFinished).await;
        let message = turn?;

        let no_tool_calls = !message.has_tool_calls();

        // If the LLM generated text, show it
        if message.has_visible_content() {
            let content = message.content.as_deref().unwrap();
            ui.event(AgentEvent::AssistantMessage {
                model: model.to_string(),
                effort_level: effort_level.clone(),
                text: content.to_string(),
            })
            .await;
            final_response = Some(content.to_string());
        }

        // Record the assistant's turn in history before deciding whether to
        // keep looping, so a plain text answer (no tool calls) is still
        // remembered on the next turn instead of vanishing from context.
        let tool_calls = message.tool_calls.clone();
        messages.push(message);

        if no_tool_calls {
            ui.event(AgentEvent::TurnFinished).await;
            return Ok(final_response);
        }

        // Process each tool call
        if let Some(tool_calls) = &tool_calls {
            for tool_call in tool_calls {
                let tool_name = &tool_call.function.name;

                ui.event(AgentEvent::ToolCallStarted {
                    name: tool_name.clone(),
                    arguments: tool_call.function.arguments.clone(),
                })
                .await;

                // Check if approval is needed
                // Read per tool call rather than once per turn: a gate
                // changed with `/tools` while this turn is running is
                // meant to apply to what the turn does next, not to the
                // turn after it.
                let approved = match gates.access().access(tool_name) {
                    // Denied outright. Reachable even though the tool was
                    // left out of the request: the list was built before the
                    // turn began, and a gate closed part-way through has to
                    // take effect on the call in front of it.
                    ToolAccess::Never => false,
                    ToolAccess::Allow => true,
                    ToolAccess::Ask => {
                        ui.approve(ApprovalRequest {
                            tool_name: tool_name.clone(),
                            category: crate::tools::category_of(tool_name),
                            arguments: tool_call.function.arguments.clone(),
                        })
                        .await?
                    }
                };

                let result = if approved {
                    // Execute the tool
                    let tool_result = execute_tool(
                        tool_name,
                        &tool_call.function.arguments,
                        gates.sandbox(),
                        gates.command_timeout(),
                    )
                    .await;

                    match tool_result {
                        Ok(result) => result,
                        Err(e) => json!({ "error": e.to_string() }),
                    }
                } else {
                    ui.event(AgentEvent::ToolCallDenied {
                        name: tool_name.clone(),
                    })
                    .await;
                    json!({ "error": "User denied permission for this action" })
                };

                ui.event(AgentEvent::ToolCallCompleted {
                    name: tool_name.clone(),
                    result: result.to_string(),
                })
                .await;

                // Add tool result back to messages, threaded to the call that produced it
                messages.push(ChatMessage {
                    role: "tool".to_string(),
                    content: Some(result.to_string()),
                    tool_calls: None,
                    tool_call_id: Some(tool_call.id.clone()),
                    ..Default::default()
                });
            }
        }
    }

    Err(anyhow::anyhow!("Agent exceeded max iterations"))
}

#[cfg(test)]
mod tests {

    /// The drain happens before the request is built, so a request that
    /// fails still proves the message was injected — and where.
    #[tokio::test]
    async fn a_steered_message_joins_the_turn_as_a_user_message() {
        let config = crate::config::Config {
            // Refused immediately, so the loop reaches its request, fails,
            // and returns without this test touching the network.
            base_url: "http://127.0.0.1:1/v1".to_string(),
            ..crate::config::Config::default()
        };
        let client = Client::for_test(config);
        let mut ui = SilentUi;
        let mut messages = vec![user("start the work")];

        let steering = Steering::default();
        steering.push("actually, check Windows too".to_string());

        let _ = run_agent_turn(
            &client,
            &mut ui,
            &mut messages,
            "test-model",
            Some(1),
            None,
            &SessionGates::default(),
            None,
            false,
            &steering,
            &UsageTracker::default(),
        )
        .await;

        assert_eq!(messages.len(), 2, "{messages:?}");
        assert_eq!(messages[1].role, "user");
        assert_eq!(
            messages[1].content.as_deref(),
            Some("actually, check Windows too")
        );
        // Taken, not copied: the next iteration must not send it again.
        assert!(steering.take().is_empty());
    }

    #[test]
    fn steering_is_shared_between_clones() {
        // The whole mechanism rests on this: the turn runs on another task
        // holding a clone, so a push on the worker's handle has to be
        // visible to the loop's.
        let worker = Steering::default();
        let loop_side = worker.clone();

        assert_eq!(worker.push("stop and check the tests".to_string()), 1);
        assert_eq!(worker.push("then keep going".to_string()), 2);

        assert_eq!(
            loop_side.take(),
            vec![
                "stop and check the tests".to_string(),
                "then keep going".to_string()
            ]
        );
    }

    #[test]
    fn taking_drains_so_a_message_joins_exactly_one_iteration() {
        let steering = Steering::default();
        steering.push("one".to_string());
        assert_eq!(steering.take(), vec!["one".to_string()]);
        // A second iteration must not re-inject what the first already sent.
        assert!(steering.take().is_empty());
    }

    #[test]
    fn usage_tracker_sums_across_clones() {
        // The whole point: a turn runs on another task holding a clone, and
        // the caller reads the total back through the handle it kept.
        let caller = UsageTracker::default();
        let worker = caller.clone();
        worker.add(10);
        worker.add(5);
        assert_eq!(caller.total(), 15);
    }

    #[test]
    fn an_idle_handle_yields_nothing() {
        // What `clank agent` and the CLI's blocking loop pass: no way to
        // type mid-turn, so every iteration takes an empty list.
        assert!(Steering::default().take().is_empty());
    }
    use super::*;
    use crate::client::{function_call_type, FunctionCall, ToolCall};
    use crate::config::ApprovalSettings;

    /// An [`AgentUi`] that renders nothing, for exercising the loop's own
    /// decisions without a front end.
    struct SilentUi;

    impl AgentUi for SilentUi {
        async fn event(&mut self, _event: AgentEvent) {}

        async fn approve(&mut self, _request: ApprovalRequest) -> Result<bool> {
            Ok(true)
        }
    }

    #[tokio::test]
    async fn agent_mode_refuses_to_run_without_an_iteration_cap() {
        // `clank max-iterations --clear` leaves no cap anywhere, and agent
        // mode fails loudly rather than picking a number on the user's
        // behalf — the one setting where null is an error instead of
        // "send nothing". Checked before any request goes out, which is why
        // a credential-free client is enough to test it.
        let client = Client::for_test(crate::config::Config::default());
        let mut ui = SilentUi;
        let mut messages = vec![user("do the thing")];

        let error = run_agent_turn(
            &client,
            &mut ui,
            &mut messages,
            "test-model",
            None,
            None,
            &SessionGates::default(),
            None,
            false,
            &Steering::default(),
            &UsageTracker::default(),
        )
        .await
        .expect_err("no cap is an error, not a default");

        let message = error.to_string();
        assert!(
            message.contains("No max-iterations cap is set"),
            "{message}"
        );
        // Says both ways out, since the fix differs by where you are.
        assert!(message.contains("/max-iterations"), "{message}");
        assert!(message.contains("clank max-iterations"), "{message}");
    }

    #[test]
    fn categorizes_known_tools() {
        assert_eq!(crate::tools::category_of("read_file"), "read");
        assert_eq!(crate::tools::category_of("list_files"), "read");
        assert_eq!(crate::tools::category_of("write_file"), "write");
        assert_eq!(crate::tools::category_of("replace_in_file"), "write");
        assert_eq!(
            crate::tools::category_of("run_terminal_command"),
            "terminal"
        );
        assert_eq!(crate::tools::category_of("web_fetch"), "web");
        assert_eq!(crate::tools::category_of("something_else"), "unknown");
    }

    #[test]
    fn the_shell_is_off_by_default_and_the_web_runs_freely() {
        // Three defaults, three reasons. The shell can do anything the user
        // can, so it is not offered until asked for. The web reads a page
        // and changes nothing, and a prompt per page is what would push the
        // model back to curling through the shell. Everything else asks.
        let fresh = ToolAccessSettings::default();
        for tool in crate::tools::TOOLS {
            let expected = match tool.name {
                "run_terminal_command" => ToolAccess::Never,
                "web_fetch" => ToolAccess::Allow,
                _ => ToolAccess::Ask,
            };
            assert_eq!(fresh.access(tool.name), expected, "{}", tool.name);
        }
        // A name we do not know is the last thing that should run unwatched.
        assert_eq!(fresh.access("something_else"), ToolAccess::Ask);
        // And with the shell off, a fresh clanker still has tools.
        assert!(fresh.any_tools());
    }

    #[test]
    fn a_category_sets_every_tool_in_it_and_nothing_else() {
        let settings = ToolAccessSettings::default()
            .with("write", ToolAccess::Allow)
            .unwrap();
        assert_eq!(settings.access("write_file"), ToolAccess::Allow);
        assert_eq!(settings.access("replace_in_file"), ToolAccess::Allow);
        assert_eq!(settings.access("read_file"), ToolAccess::Ask);
        assert_eq!(settings.access("run_terminal_command"), ToolAccess::Never);

        // One tool on its own, and a word that names neither.
        let one = settings.with("read_file", ToolAccess::Never).unwrap();
        assert_eq!(one.access("read_file"), ToolAccess::Never);
        assert_eq!(one.access("list_files"), ToolAccess::Ask);
        assert!(settings.with("nonesuch", ToolAccess::Ask).is_none());
    }

    #[test]
    fn a_tool_set_to_never_is_not_offered_at_all() {
        // Left out of the request rather than refused when called: it costs
        // no tokens, and the model is never invited to try.
        let settings = ToolAccessSettings::default()
            .with("write_file", ToolAccess::Never)
            .unwrap();
        let offered: Vec<String> = offered_tools(&settings)
            .iter()
            .map(|definition| definition["function"]["name"].as_str().unwrap().to_string())
            .collect();

        assert!(!offered.iter().any(|name| name == "write_file"));
        assert!(offered.iter().any(|name| name == "read_file"));
        // The shell is `never` out of the box, so two are missing.
        assert!(!offered.iter().any(|name| name == "run_terminal_command"));
        assert_eq!(offered.len(), crate::tools::TOOLS.len() - 2);

        // And with everything off there is nothing to send, which is what a
        // clanker with no tools now is.
        assert!(offered_tools(&ToolAccessSettings::none()).is_empty());
        assert!(!ToolAccessSettings::none().any_tools());
        assert!(ToolAccessSettings::default().any_tools());
    }

    #[test]
    fn old_category_gates_still_mean_what_they_meant() {
        // Sessions and configs written before tools had their own states
        // hold three booleans. `true` was "ask first"; `false` was "just do
        // it". Nothing is rewritten — they simply read as the same thing.
        let legacy = ApprovalSettings {
            read_disk: false,
            write_disk: true,
            terminal: false,
        };
        let derived = ToolAccessSettings::from_legacy(&legacy);
        assert_eq!(derived.access("read_file"), ToolAccess::Allow);
        assert_eq!(derived.access("list_files"), ToolAccess::Allow);
        assert_eq!(derived.access("write_file"), ToolAccess::Ask);

        // Two exceptions, both keeping their own default. The web was never
        // part of the old model at all. The shell was, but the old model
        // could not say "not offered", so reading its boolean would leave an
        // upgrade with a shell a fresh install does not have.
        assert_eq!(derived.access("web_fetch"), ToolAccess::Allow);
        assert_eq!(
            derived.access("run_terminal_command"),
            ToolAccess::Never,
            "the old terminal gate does not survive the upgrade"
        );
    }

    #[test]
    fn a_tool_we_do_not_know_still_asks() {
        // Fail-safe: a name that is not one of ours belongs to no category,
        // so no bulk setting reaches it and its default stands even when
        // everything else has been waved through.
        let all_allowed = ToolAccessSettings::default()
            .with("all", ToolAccess::Allow)
            .unwrap();
        assert_eq!(all_allowed.access("some_future_tool"), ToolAccess::Ask);
    }

    #[test]
    fn dangling_reasoning_is_dropped_without_a_tool_call() {
        // A turn that reasoned but ended with plain text (or nothing at
        // all) instead of a tool call — resending its reasoning_details
        // would leave `thinking` as the final block, which Anthropic
        // rejects on the next request.
        let mut message = ChatMessage {
            role: "assistant".to_string(),
            content: Some("Here's my answer.".to_string()),
            reasoning: Some("thinking it through".to_string()),
            reasoning_details: Some(vec![serde_json::json!({"type": "reasoning.text"})]),
            ..Default::default()
        };
        drop_dangling_reasoning(&mut message);
        assert_eq!(message.reasoning_details, None);
        // The prose survives: it's never sent to a provider, and `/verbose`
        // still shows the thinking behind a reply that called no tool.
        assert_eq!(message.reasoning, Some("thinking it through".to_string()));
        assert_eq!(message.content, Some("Here's my answer.".to_string()));
    }

    #[test]
    fn reasoning_is_kept_alongside_a_tool_call() {
        let mut message = ChatMessage {
            role: "assistant".to_string(),
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                call_type: function_call_type(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            reasoning_details: Some(vec![serde_json::json!({"type": "reasoning.text"})]),
            ..Default::default()
        };
        drop_dangling_reasoning(&mut message);
        assert!(message.reasoning_details.is_some());
    }

    #[test]
    fn dangling_reasoning_is_dropped_with_an_empty_tool_calls_array_too() {
        // A provider can send `tool_calls: []` rather than omitting the
        // field on a turn that didn't really call anything — that must
        // still count as "no tool call" here, not just a bare `None`.
        let mut message = ChatMessage {
            role: "assistant".to_string(),
            content: Some("no real tool call".to_string()),
            tool_calls: Some(vec![]),
            reasoning_details: Some(vec![serde_json::json!({"type": "reasoning.text"})]),
            ..Default::default()
        };
        drop_dangling_reasoning(&mut message);
        assert_eq!(message.reasoning_details, None);
    }

    fn user(text: &str) -> ChatMessage {
        ChatMessage {
            role: "user".to_string(),
            content: Some(text.to_string()),
            ..Default::default()
        }
    }

    fn assistant(text: &str) -> ChatMessage {
        ChatMessage {
            role: "assistant".to_string(),
            content: Some(text.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn normalize_system_prompt_prepends_it_fresh_for_an_agentic_turn() {
        let mut messages = vec![user("hi"), assistant("hello")];
        normalize_system_prompt(&mut messages, true);
        assert_eq!(messages[0].role, "system");
        assert_eq!(
            messages[0].content.as_deref(),
            Some(AGENT_CHAT_SYSTEM_PROMPT)
        );
        assert_eq!(messages.len(), 3);
    }

    #[test]
    fn normalize_system_prompt_adds_nothing_for_a_plain_turn() {
        let mut messages = vec![user("hi"), assistant("hello")];
        normalize_system_prompt(&mut messages, false);
        assert_eq!(messages.len(), 2);
        assert!(messages.iter().all(|m| m.role != "system"));
    }

    #[test]
    fn normalize_system_prompt_heals_a_stray_copy_left_mid_conversation() {
        // What `/agent` used to do: insert the prompt wherever it happened
        // to be typed, which a provider can reject if that position isn't
        // valid (e.g. right after a plain assistant reply). An
        // already-poisoned session — or one that's since switched back to
        // no tools — must not keep resending that stray copy.
        let stray = ChatMessage {
            role: "system".to_string(),
            content: Some(AGENT_CHAT_SYSTEM_PROMPT.to_string()),
            ..Default::default()
        };
        let mut messages = vec![user("hi"), assistant("hello"), stray, user("again")];

        normalize_system_prompt(&mut messages, false);
        assert!(messages.iter().all(|m| m.role != "system"));
        assert_eq!(messages.len(), 3);

        let mut messages = vec![
            user("hi"),
            assistant("hello"),
            ChatMessage {
                role: "system".to_string(),
                content: Some(AGENT_CHAT_SYSTEM_PROMPT.to_string()),
                ..Default::default()
            },
            user("again"),
        ];
        normalize_system_prompt(&mut messages, true);
        // Exactly one copy, and it's the fresh one at the front — not the
        // stray one left in place.
        assert_eq!(messages.iter().filter(|m| m.role == "system").count(), 1);
        assert_eq!(messages[0].role, "system");
    }
}
