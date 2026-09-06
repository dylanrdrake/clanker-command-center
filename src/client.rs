use crate::config::Config;
use anyhow::{anyhow, Result};
use async_stream::try_stream;
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Duration;

/// Orders model ids the way someone reading a list of them expects.
///
/// The endpoint returns its own order — newest first, or however it feels —
/// which is no help at four hundred entries when you are looking for one you
/// can half remember. Sorted here rather than at each display so `clank
/// models` and the TUI's `/models` cannot disagree about it.
///
/// Case-insensitive, because a stray capital would otherwise sort a model
/// away from its siblings, with the raw comparison breaking ties so the
/// order is total rather than merely consistent.
pub(crate) fn sort_model_ids(ids: &mut [String]) {
    ids.sort_by(|a, b| {
        a.to_lowercase()
            .cmp(&b.to_lowercase())
            .then_with(|| a.cmp(b))
    });
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Plain-text reasoning/thinking summary, when the model returned one.
    /// Display-only: this is what `/verbose` shows the user, and it never
    /// goes back to a provider. [`Self::reasoning_details`] is the channel
    /// that has to round-trip, because it carries the signature
    /// authenticating the same thinking; echoing this unsigned prose
    /// alongside it would offer a second, less trustworthy copy of the same
    /// content, so `skip_serializing` keeps it out of the wire format
    /// rather than relying on it happening to be `None`.
    ///
    /// Being display-only is also why it outlives the rules that govern
    /// `reasoning_details` — it's kept on a turn that called no tool, where
    /// resending a thinking block would be invalid but showing one is
    /// perfectly reasonable.
    #[serde(skip_serializing)]
    pub reasoning: Option<String>,
    /// Provider-specific reasoning blocks (e.g. Anthropic's `thinking`
    /// block plus its signature, as OpenRouter shapes it). Kept as opaque
    /// JSON rather than a typed struct — OpenRouter requires the sequence
    /// to round-trip back to it byte-for-byte unmodified on a follow-up
    /// request that continues past a tool call, so parsing and
    /// re-serializing our own shape risks silently corrupting it. A
    /// reasoning model that pauses to call a tool needs this echoed back on
    /// the next request or the provider can reject the continuation
    /// outright (Anthropic's "prefill"/"must end with a user message"
    /// error) — models that never return it simply never have this set, so
    /// nothing changes for them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_details: Option<Vec<serde_json::Value>>,
}

impl ChatMessage {
    /// True if `content` is `Some` and has non-whitespace text. Some
    /// providers return `content: ""` instead of `null` when a message
    /// carries no visible text (e.g. a tool-calls-only turn).
    pub fn has_visible_content(&self) -> bool {
        self.content
            .as_deref()
            .is_some_and(|c| !c.trim().is_empty())
    }

    /// True if `tool_calls` is `Some` *and* actually has an entry in it.
    /// Some providers send `tool_calls: []` rather than omitting the field
    /// on a turn with no real tool call — treating that the same as `None`
    /// matters wherever "did this turn call a tool" gates something, not
    /// just whether the field happened to be present.
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
    }

    /// This turn's thinking as displayable text, if the model returned any.
    /// Prefers the provider's own prose summary and falls back to the text
    /// carried inside the structured blocks, since a provider may populate
    /// either channel — or, once a reply is reloaded from a session
    /// recorded before the prose was stored, only the blocks.
    pub fn thinking_text(&self) -> Option<String> {
        if let Some(prose) = self.reasoning.as_deref() {
            if !prose.trim().is_empty() {
                return Some(prose.to_string());
            }
        }
        let joined = self
            .reasoning_details
            .as_ref()?
            .iter()
            .filter_map(|detail| detail.get("text").and_then(serde_json::Value::as_str))
            .filter(|text| !text.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        (!joined.is_empty()).then_some(joined)
    }
}

/// The discriminator every tool call carries. Only function calls exist in
/// the OpenAI schema today, so this is always `"function"` — but it is
/// *required*, and a provider that translates the request into another
/// API's shape has to read it to know what kind of call it is.
pub(crate) fn function_call_type() -> String {
    "function".to_string()
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub id: String,
    /// See [`function_call_type`]. Defaulted on the way in, so a provider
    /// that omits it in a response — and, more to the point, a tool call
    /// stored before this field existed — still deserializes; always
    /// written on the way out.
    #[serde(rename = "type", default = "function_call_type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    /// Omitted entirely (not sent as `null`) when there's no temperature to
    /// use — the provider then falls back to its own default, same as an
    /// omitted `reasoning`/`reasoning_effort`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    /// Flat effort field, e.g. OrcaRouter's `reasoning_effort: "high"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Nested effort field, e.g. OpenRouter's `reasoning: { "effort": "high" }`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningEffort>,
    /// Only sent when streaming; omitted entirely for the buffered path so
    /// requests to providers that don't expect it are unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Asks a streaming provider to emit a final usage-only chunk before
    /// `[DONE]` (OpenAI/OpenRouter's `stream_options.include_usage`). Only
    /// meaningful alongside `stream`, so it's built and omitted the same way.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
}

#[derive(Debug, Serialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Serialize)]
pub struct ReasoningEffort {
    pub effort: String,
}

/// Token accounting for one request, as providers report it on
/// `ChatResponse`/the final streamed chunk. Missing entirely on a provider
/// that doesn't report usage, which is why every caller treats it as
/// optional rather than assuming a count is always available.
///
/// Only the total is kept: nothing here breaks it down into prompt versus
/// completion tokens, so there's nothing else worth parsing out of it yet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub total_tokens: u64,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: ChatMessage,
}

/// What a streaming turn produces, in order: any number of `Content` deltas
/// as text arrives, then exactly one `Done` carrying the fully assembled
/// message (text plus any tool calls).
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Content(String),
    Done {
        message: ChatMessage,
        /// Set when the provider sent a final usage chunk — only requested
        /// when `stream_options.include_usage` went out, and not every
        /// provider honors it even then.
        usage: Option<Usage>,
    },
}

// ---------------------------------------------------------------------------
// Streaming wire format
//
// A chunk looks like:
//   {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}
// Tool calls arrive in fragments correlated only by `index`, with
// `function.arguments` split across arbitrarily many chunks:
//   {"choices":[{"delta":{"tool_calls":[
//      {"index":0,"id":"call_1","function":{"name":"write_file","arguments":"{\"pa"}}]}}]}
//   {"choices":[{"delta":{"tool_calls":[
//      {"index":0,"function":{"arguments":"th\":\"a.txt\"}"}}]}}]}
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    /// Present only on the final chunk of a stream that asked for it — see
    /// `StreamOptions::include_usage`.
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: Delta,
}

#[derive(Debug, Default, Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    reasoning_details: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
struct ToolCallDelta {
    #[serde(default)]
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct FunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

/// Pulls complete `data:` payloads out of a rolling byte buffer.
///
/// Works on bytes rather than text because a network chunk can split a
/// multi-byte UTF-8 character; holding the partial line as bytes until its
/// newline arrives keeps such a character intact.
#[derive(Debug, Default)]
struct SseDecoder {
    buf: Vec<u8>,
}

impl SseDecoder {
    fn push_bytes(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Returns every complete `data:` payload now available, leaving any
    /// trailing partial line buffered for the next call.
    fn drain_payloads(&mut self) -> Vec<String> {
        let mut payloads = Vec::new();
        while let Some(newline) = self.buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=newline).collect();
            let line = String::from_utf8_lossy(&line[..line.len() - 1]);
            let line = line.trim_end_matches('\r');
            // Blank separators and any non-data field (`event:`, `id:`, `:`
            // comments) are not payloads.
            if let Some(rest) = line.strip_prefix("data:") {
                payloads.push(rest.trim().to_string());
            }
        }
        payloads
    }
}

/// Reassembles streamed chunks into one [`ChatMessage`].
#[derive(Debug, Default)]
struct StreamAccumulator {
    content: String,
    /// Keyed by the wire's `index` and ordered by it, so tool calls come out
    /// in the order the model asked for them regardless of chunk arrival.
    tool_calls: BTreeMap<u32, PartialToolCall>,
    reasoning: String,
    /// Keyed by each reasoning block's own `index` (OpenRouter streams a
    /// block's `text`/`summary` incrementally across chunks, same shape as
    /// `tool_calls` above), and merged with [`merge_reasoning_detail`].
    reasoning_details: BTreeMap<u64, serde_json::Value>,
    /// The most recent usage chunk seen, if any. Overwritten rather than
    /// summed: providers that report it send one cumulative total for the
    /// whole request, typically on a final chunk with empty `choices`.
    usage: Option<Usage>,
}

/// Folds one incoming reasoning-detail chunk into the block accumulated so
/// far for its index. `text`/`summary` are appended (that's how OpenRouter
/// streams a block's growing content); every other field — `signature`
/// especially, which typically arrives once, later, and is `null` until
/// then — is overwritten only when the new value isn't `null`, so a later
/// chunk can't blank out something an earlier one already captured.
fn merge_reasoning_detail(existing: &mut serde_json::Value, incoming: serde_json::Value) {
    if !existing.is_object() || !incoming.is_object() {
        *existing = incoming;
        return;
    }
    let incoming_obj = match incoming {
        serde_json::Value::Object(obj) => obj,
        _ => unreachable!("checked above"),
    };
    let existing_obj = existing.as_object_mut().expect("checked above");
    for (key, new_value) in incoming_obj {
        let append_text = matches!(key.as_str(), "text" | "summary")
            && new_value.is_string()
            && matches!(existing_obj.get(&key), Some(serde_json::Value::String(_)));
        if append_text {
            if let (Some(serde_json::Value::String(old)), Some(added)) =
                (existing_obj.get_mut(&key), new_value.as_str())
            {
                old.push_str(added);
            }
        } else if !new_value.is_null() {
            existing_obj.insert(key, new_value);
        }
    }
}

impl StreamAccumulator {
    /// Folds one `data:` payload in, returning any new text it carried.
    fn push_payload(&mut self, payload: &str) -> Result<Option<String>> {
        let chunk: StreamChunk =
            serde_json::from_str(payload).map_err(|e| anyhow!("Malformed stream chunk: {e}"))?;

        if let Some(usage) = chunk.usage {
            self.usage = Some(usage);
        }

        let mut new_text = None;
        for choice in chunk.choices {
            if let Some(content) = choice.delta.content {
                if !content.is_empty() {
                    self.content.push_str(&content);
                    new_text.get_or_insert_with(String::new).push_str(&content);
                }
            }

            if let Some(reasoning) = choice.delta.reasoning {
                self.reasoning.push_str(&reasoning);
            }
            for detail in choice.delta.reasoning_details.unwrap_or_default() {
                let index = detail
                    .get("index")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                match self.reasoning_details.entry(index) {
                    std::collections::btree_map::Entry::Vacant(slot) => {
                        slot.insert(detail);
                    }
                    std::collections::btree_map::Entry::Occupied(mut slot) => {
                        merge_reasoning_detail(slot.get_mut(), detail);
                    }
                }
            }

            for delta in choice.delta.tool_calls.unwrap_or_default() {
                let entry = self.tool_calls.entry(delta.index).or_default();
                if let Some(id) = delta.id {
                    entry.id = id;
                }
                if let Some(function) = delta.function {
                    if let Some(name) = function.name {
                        // OpenAI sends the name once, in the opening chunk,
                        // but plenty of compatible providers repeat it whole
                        // in every delta. Appending blindly turns that into
                        // "write_filewrite_file" and the call then fails as
                        // an unknown tool, so only extend on genuinely new
                        // text.
                        if entry.name.is_empty() {
                            entry.name = name;
                        } else if entry.name != name {
                            entry.name.push_str(&name);
                        }
                    }
                    if let Some(arguments) = function.arguments {
                        entry.arguments.push_str(&arguments);
                    }
                }
            }
        }

        Ok(new_text)
    }

    /// The last usage chunk seen so far, if the provider has sent one. Read
    /// before [`Self::finish`] consumes the accumulator.
    fn usage(&self) -> Option<Usage> {
        self.usage
    }

    fn finish(self) -> ChatMessage {
        let tool_calls: Vec<ToolCall> = self
            .tool_calls
            .into_values()
            .map(|partial| ToolCall {
                id: partial.id,
                call_type: function_call_type(),
                function: FunctionCall {
                    name: partial.name,
                    arguments: partial.arguments,
                },
            })
            .collect();

        ChatMessage {
            role: "assistant".to_string(),
            // Empty text is reported as absent, matching how the buffered
            // path's providers return `content: null` on a tool-only turn.
            content: if self.content.is_empty() {
                None
            } else {
                Some(self.content)
            },
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
            reasoning: if self.reasoning.is_empty() {
                None
            } else {
                Some(self.reasoning)
            },
            reasoning_details: if self.reasoning_details.is_empty() {
                None
            } else {
                Some(self.reasoning_details.into_values().collect())
            },
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ModelList {
    pub data: Vec<Model>,
}

#[derive(Debug, Deserialize)]
pub struct Model {
    pub id: String,
}

/// A compact, content-free description of an outgoing request, logged
/// alongside an API error so a structural rejection ("must end with a user
/// message", "the final block ... cannot be `thinking`") can be read
/// against what actually went out. Deliberately carries no message text —
/// just the role sequence and the fields the message-shape rules actually
/// turn on:
///
/// `S U A(c2,r1,-) T U` — system; user; an assistant with 2 tool calls and
/// 1 reasoning block, carrying no text content; a tool result; a user.
///
/// A reasoning count is suffixed with `!` when a block has neither a
/// `signature` nor `data`, since an unsigned thinking block is rejected on
/// its own terms rather than for where it sits — worth telling apart from a
/// well-formed one at the same position.
fn request_skeleton(request: &ChatRequest) -> String {
    let roles: Vec<String> = request
        .messages
        .iter()
        .map(|m| {
            let mut s = match m.role.as_str() {
                "system" => "S",
                "user" => "U",
                "assistant" => "A",
                "tool" => "T",
                _ => "?",
            }
            .to_string();

            let mut marks = Vec::new();
            let calls = m.tool_calls.as_ref().map_or(0, Vec::len);
            if calls > 0 {
                marks.push(format!("c{calls}"));
            }
            if let Some(details) = &m.reasoning_details {
                let unsigned = details.iter().any(|d| {
                    !["signature", "data"]
                        .iter()
                        .any(|key| d.get(key).is_some_and(|v| !v.is_null()))
                });
                marks.push(format!(
                    "r{}{}",
                    details.len(),
                    if unsigned { "!" } else { "" }
                ));
            }
            if !m.has_visible_content() {
                marks.push("-".to_string());
            }
            if !marks.is_empty() {
                s.push_str(&format!("({})", marks.join(",")));
            }
            s
        })
        .collect();

    format!(
        "model={} msgs={} tools={} stream={} [{}]",
        request.model,
        request.messages.len(),
        request.tools.as_ref().map_or(0, Vec::len),
        request.stream.unwrap_or(false),
        roles.join(" ")
    )
}

/// Serializes the outgoing body, but only when raw capture is switched on
/// — the body is the whole conversation, so it is never built speculatively.
fn capture_body(request: &ChatRequest) -> Option<String> {
    if !crate::error_log::request_dumps_enabled() {
        return None;
    }
    serde_json::to_string_pretty(request).ok()
}

/// Writes a captured body out and returns the ` | body: <path>` fragment
/// pointing the log entry at it, or an empty string when capture is off.
fn dump_note(body: Option<&str>) -> String {
    body.and_then(crate::error_log::dump_failed_request)
        .map(|path| format!(" | body: {}", path.display()))
        .unwrap_or_default()
}

/// Drops `reasoning_details` from any message with no `tool_call` for it to
/// lead into. A thinking block that nothing follows is the assistant
/// message's final block, which Anthropic rejects outright — so this is
/// about the shape of one message, not about how far back in the
/// conversation thinking is allowed to live.
///
/// Only the structured blocks go: [`ChatMessage::reasoning`] is never sent
/// to a provider, so nothing about it can make a request invalid, and
/// keeping it is what lets `/verbose` still show the thinking behind a
/// reply that called no tool.
///
/// `agent.rs`'s `drop_dangling_reasoning` already keeps a message from being
/// stored that way, but it can't retroactively clean one written before that
/// existed; sweeping the outgoing array catches those too, whatever the
/// database holds.
///
/// A stricter version of this once scoped thinking to the turn in progress,
/// stripping it from every turn the user had already spoken past. That was
/// written chasing the wrong cause — the real one was tool calls going out
/// without their `type` discriminator (see [`function_call_type`]) — and it
/// asks for more than either Anthropic or OpenRouter do, both of which say
/// to pass reasoning blocks back unmodified. Left at the narrower rule
/// unless a mixed history turns out to need more.
fn strip_dangling_reasoning(messages: &mut [ChatMessage]) {
    for message in messages {
        if !message.has_tool_calls() {
            message.reasoning_details = None;
        }
    }
}

pub struct Client {
    config: Config,
    api_key: String,
    http_client: reqwest::Client,
}

impl Client {
    /// A client with no credentials, for testing request construction.
    /// `new` reads the OS keychain, which a test can't rely on — and none of
    /// what's exercised here (which fields a config does and doesn't put on
    /// the wire) needs a key.
    #[cfg(test)]
    pub fn for_test(config: Config) -> Self {
        Client {
            config,
            api_key: String::new(),
            http_client: reqwest::Client::new(),
        }
    }

    /// The fallback timeout for a terminal command the agent runs, so a
    /// caller holding only a `Client` can build the gates without also
    /// carrying the config it came from.
    /// The gap allowed between streamed chunks before the connection is
    /// treated as stalled.
    fn stream_idle(&self) -> Duration {
        Duration::from_secs(self.config.stream_idle_timeout)
    }

    pub fn command_timeout(&self) -> u64 {
        self.config.command_timeout
    }

    pub fn new(config: Config) -> Result<Self> {
        let api_key = crate::config::get_api_key()?
            .ok_or_else(|| anyhow!("API key not configured. Run: clank login"))?;

        let http_client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(config.connect_timeout))
            .build()?;

        Ok(Client {
            config,
            api_key,
            http_client,
        })
    }

    /// Applies the `Authorization` header plus any user-configured
    /// `extra_headers` (e.g. OpenRouter's optional `HTTP-Referer`/`X-Title`)
    /// to an outgoing request.
    fn apply_headers(&self, mut req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        req = req.header("Authorization", format!("Bearer {}", self.api_key));
        for (key, value) in &self.config.extra_headers {
            req = req.header(key.as_str(), value.as_str());
        }
        req
    }

    /// Builds the request body shared by the buffered and streaming paths,
    /// including translating `effort_level` into whichever shape the
    /// configured provider expects.
    fn build_request(
        &self,
        model: String,
        mut messages: Vec<ChatMessage>,
        temperature: Option<f32>,
        tools: Option<Vec<serde_json::Value>>,
        effort_level: Option<String>,
        stream: bool,
    ) -> ChatRequest {
        strip_dangling_reasoning(&mut messages);

        let effort_style = self
            .config
            .effort_style
            .as_deref()
            .unwrap_or(crate::config::DEFAULT_EFFORT_STYLE);

        let (reasoning_effort, reasoning) = match (&effort_level, effort_style) {
            (Some(effort), "flat") => (Some(effort.clone()), None),
            (Some(effort), "nested") => (
                None,
                Some(ReasoningEffort {
                    effort: effort.clone(),
                }),
            ),
            _ => (None, None),
        };

        ChatRequest {
            model,
            messages,
            tools: tools.clone(),
            temperature,
            tool_choice: if tools.is_some() {
                Some("auto".to_string())
            } else {
                None
            },
            reasoning_effort,
            reasoning,
            stream: if stream { Some(true) } else { None },
            // Only meaningful alongside `stream`; omitted for the buffered
            // path so a provider that rejects unrecognized fields on a
            // non-streaming request is unaffected.
            stream_options: if stream {
                Some(StreamOptions {
                    include_usage: true,
                })
            } else {
                None
            },
        }
    }

    pub async fn chat(
        &self,
        model: String,
        messages: Vec<ChatMessage>,
        temperature: Option<f32>,
        tools: Option<Vec<serde_json::Value>>,
        effort_level: Option<String>,
    ) -> Result<ChatResponse> {
        let request = self.build_request(model, messages, temperature, tools, effort_level, false);
        // Captured before sending, so a rejection can be read back against
        // the shape that actually went out rather than guessed at.
        let skeleton = request_skeleton(&request);
        let body = capture_body(&request);

        let result = self.send_chat(request).await;
        if let Err(e) = &result {
            let note = dump_note(body.as_deref());
            crate::error_log::log_error("llm_api", &format!("{e} | sent: {skeleton}{note}"));
        }
        result
    }

    async fn send_chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let req = self
            .http_client
            .post(format!("{}/chat/completions", self.config.base_url));
        let response = self
            .apply_headers(req)
            .json(&request)
            .timeout(Duration::from_secs(self.config.request_timeout))
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow!("API error: {}", error_text));
        }

        let chat_response: ChatResponse = response.json().await?;
        Ok(chat_response)
    }

    /// The streaming counterpart to [`Client::chat`]: yields text as it
    /// arrives, then one [`StreamEvent::Done`] with the assembled message.
    ///
    /// Callers that need the whole reply before acting (tool calls, saving to
    /// history) use the `Done` message; callers rendering live use the
    /// `Content` deltas. The two never disagree — the deltas concatenate to
    /// the final message's content.
    pub fn chat_stream(
        &self,
        model: String,
        messages: Vec<ChatMessage>,
        temperature: Option<f32>,
        tools: Option<Vec<serde_json::Value>>,
        effort_level: Option<String>,
    ) -> impl Stream<Item = Result<StreamEvent>> + '_ {
        let request = self.build_request(model, messages, temperature, tools, effort_level, true);
        // Captured before sending, so a rejection can be read back against
        // the shape that actually went out rather than guessed at.
        let skeleton = request_skeleton(&request);
        let body = capture_body(&request);

        self.chat_stream_inner(request).map(move |item| {
            if let Err(e) = &item {
                let note = dump_note(body.as_deref());
                crate::error_log::log_error("llm_api", &format!("{e} | sent: {skeleton}{note}"));
            }
            item
        })
    }

    fn chat_stream_inner(
        &self,
        request: ChatRequest,
    ) -> impl Stream<Item = Result<StreamEvent>> + '_ {
        let url = format!("{}/chat/completions", self.config.base_url);

        try_stream! {
            let req = self.http_client.post(url);
            // Not `.timeout()` on the request itself — that would also
            // bound the total time spent reading a long-but-still-arriving
            // stream below, which is exactly what the idle timeout further
            // down is meant to allow. This only bounds how long a first
            // response takes to start showing up at all.
            let response = match tokio::time::timeout(self.stream_idle(), self.apply_headers(req).json(&request).send()).await {
                Ok(response) => response?,
                Err(_) => {
                    Err(anyhow!(
                        "No response from provider within {}s; the connection may have stalled",
                        self.config.stream_idle_timeout
                    ))?;
                    return;
                }
            };

            if !response.status().is_success() {
                let error_text = response.text().await?;
                Err(anyhow!("API error: {}", error_text))?;
                return;
            }

            let mut bytes = response.bytes_stream();
            let mut decoder = SseDecoder::default();
            let mut accumulator = StreamAccumulator::default();

            'outer: loop {
                let chunk = match tokio::time::timeout(self.stream_idle(), bytes.next()).await {
                    Ok(Some(chunk)) => chunk,
                    Ok(None) => break 'outer,
                    Err(_) => {
                        Err(anyhow!(
                            "No response from provider within {}s; the connection may have stalled",
                            self.config.stream_idle_timeout
                        ))?;
                        return;
                    }
                };
                decoder.push_bytes(&chunk?);
                for payload in decoder.drain_payloads() {
                    if payload == "[DONE]" {
                        break 'outer;
                    }
                    if payload.is_empty() {
                        continue;
                    }
                    if let Some(text) = accumulator.push_payload(&payload)? {
                        yield StreamEvent::Content(text);
                    }
                }
            }

            let usage = accumulator.usage();
            yield StreamEvent::Done { message: accumulator.finish(), usage };
        }
    }

    pub async fn list_models(&self) -> Result<Vec<String>> {
        let req = self
            .http_client
            .get(format!("{}/models", self.config.base_url));
        let response = self.apply_headers(req).send().await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow!("API error: {}", error_text));
        }

        let model_list: ModelList = response.json().await?;
        let mut ids: Vec<String> = model_list.data.into_iter().map(|m| m.id).collect();
        sort_model_ids(&mut ids);
        Ok(ids)
    }
}

#[cfg(test)]
mod deser_tests {

    #[test]
    fn model_ids_sort_alphabetically_ignoring_case() {
        // The endpoint returns its own order, which is no help at four
        // hundred entries when you are hunting for one you half remember.
        let mut ids: Vec<String> = [
            "openai/gpt-5",
            "Anthropic/Claude-Opus",
            "anthropic/claude-haiku",
            "~z-ai/glm-latest",
            "google/gemini",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        sort_model_ids(&mut ids);

        assert_eq!(
            ids,
            vec![
                "anthropic/claude-haiku",
                "Anthropic/Claude-Opus",
                "google/gemini",
                "openai/gpt-5",
                "~z-ai/glm-latest",
            ],
            "a stray capital must not sort a model away from its siblings"
        );
    }

    #[test]
    fn ids_differing_only_in_case_still_get_a_stable_order() {
        // Equal under the case-insensitive comparison, so without the
        // tiebreak their order would depend on how they arrived.
        let mut a: Vec<String> = ["Model", "model"].iter().map(|s| s.to_string()).collect();
        let mut b: Vec<String> = ["model", "Model"].iter().map(|s| s.to_string()).collect();
        sort_model_ids(&mut a);
        sort_model_ids(&mut b);
        assert_eq!(a, b);
    }

    use super::*;
    use serde_json::Value;

    #[test]
    fn chat_message_deserializes_without_tool_call_id() {
        let json = r#"{"role":"assistant","content":"hi"}"#;
        let m: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(m.role, "assistant");
        assert_eq!(m.tool_call_id, None);
    }

    // --- SSE framing ---------------------------------------------------

    #[test]
    fn decoder_returns_only_complete_data_lines() {
        let mut d = SseDecoder::default();
        d.push_bytes(b"data: one\ndata: two\ndata: par");
        assert_eq!(d.drain_payloads(), vec!["one", "two"]);
        // The partial third line stays buffered until its newline arrives.
        assert!(d.drain_payloads().is_empty());
        d.push_bytes(b"tial\n");
        assert_eq!(d.drain_payloads(), vec!["partial"]);
    }

    #[test]
    fn decoder_ignores_blank_lines_and_non_data_fields() {
        let mut d = SseDecoder::default();
        d.push_bytes(b"event: message\n: a comment\n\ndata: payload\n\n");
        assert_eq!(d.drain_payloads(), vec!["payload"]);
    }

    #[test]
    fn decoder_handles_crlf() {
        let mut d = SseDecoder::default();
        d.push_bytes(b"data: hello\r\n\r\n");
        assert_eq!(d.drain_payloads(), vec!["hello"]);
    }

    #[test]
    fn decoder_survives_utf8_split_across_chunks() {
        // "café" — the é is two bytes, split across the chunk boundary.
        let mut d = SseDecoder::default();
        d.push_bytes(b"data: caf\xc3");
        assert!(d.drain_payloads().is_empty());
        d.push_bytes(b"\xa9\n");
        assert_eq!(d.drain_payloads(), vec!["café"]);
    }

    // --- Chunk accumulation ---------------------------------------------

    fn content_chunk(text: &str) -> String {
        format!(
            r#"{{"choices":[{{"delta":{{"content":{}}}}}]}}"#,
            serde_json::to_string(text).unwrap()
        )
    }

    #[test]
    fn accumulates_content_deltas_in_order() {
        let mut acc = StreamAccumulator::default();
        assert_eq!(
            acc.push_payload(&content_chunk("Hello")).unwrap(),
            Some("Hello".to_string())
        );
        assert_eq!(
            acc.push_payload(&content_chunk(", world")).unwrap(),
            Some(", world".to_string())
        );
        let message = acc.finish();
        assert_eq!(message.content, Some("Hello, world".to_string()));
        assert!(message.tool_calls.is_none());
        assert_eq!(message.role, "assistant");
    }

    #[test]
    fn empty_content_delta_yields_no_text() {
        let mut acc = StreamAccumulator::default();
        // Providers commonly open a stream with a role-only or empty delta.
        assert_eq!(acc.push_payload(&content_chunk("")).unwrap(), None);
        assert_eq!(
            acc.push_payload(r#"{"choices":[{"delta":{"role":"assistant"}}]}"#)
                .unwrap(),
            None
        );
        // No text at all is reported as absent, not as an empty string.
        assert_eq!(acc.finish().content, None);
    }

    #[test]
    fn reassembles_tool_call_arguments_fragmented_across_chunks() {
        let mut acc = StreamAccumulator::default();
        for payload in [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"write_file","arguments":"{\"filep"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ath\":\"a."}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"txt\"}"}}]}}]}"#,
        ] {
            assert_eq!(acc.push_payload(payload).unwrap(), None);
        }

        let message = acc.finish();
        assert_eq!(message.content, None);
        let calls = message.tool_calls.expect("tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "write_file");
        // The concatenated fragments must be valid JSON, since this string is
        // what gets parsed to actually run the tool.
        assert_eq!(calls[0].function.arguments, r#"{"filepath":"a.txt"}"#);
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["filepath"], "a.txt");
    }

    #[test]
    fn a_repeated_function_name_is_not_concatenated() {
        // Providers that echo the whole name in every delta must not end up
        // with "write_filewrite_file", which would fail as an unknown tool.
        let mut acc = StreamAccumulator::default();
        for payload in [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"write_file","arguments":"{\"a\":"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"write_file","arguments":"1}"}}]}}]}"#,
        ] {
            acc.push_payload(payload).unwrap();
        }
        let calls = acc.finish().tool_calls.expect("tool call");
        assert_eq!(calls[0].function.name, "write_file");
        assert_eq!(calls[0].function.arguments, r#"{"a":1}"#);
    }

    #[test]
    fn a_name_split_across_chunks_still_joins() {
        let mut acc = StreamAccumulator::default();
        for payload in [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"write_"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"file"}}]}}]}"#,
        ] {
            acc.push_payload(payload).unwrap();
        }
        let calls = acc.finish().tool_calls.expect("tool call");
        assert_eq!(calls[0].function.name, "write_file");
    }

    #[test]
    fn orders_tool_calls_by_index_not_arrival() {
        let mut acc = StreamAccumulator::default();
        // Second call's fragment arrives before the first's is finished.
        acc.push_payload(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"b","function":{"name":"second","arguments":"{}"}}]}}]}"#,
        )
        .unwrap();
        acc.push_payload(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"first","arguments":"{}"}}]}}]}"#,
        )
        .unwrap();

        let calls = acc.finish().tool_calls.expect("tool calls");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "first");
        assert_eq!(calls[1].function.name, "second");
    }

    #[test]
    fn reasoning_text_is_appended_across_chunks() {
        let mut acc = StreamAccumulator::default();
        acc.push_payload(r#"{"choices":[{"delta":{"reasoning":"Let me "}}]}"#)
            .unwrap();
        acc.push_payload(r#"{"choices":[{"delta":{"reasoning":"think..."}}]}"#)
            .unwrap();
        assert_eq!(acc.finish().reasoning, Some("Let me think...".to_string()));
    }

    #[test]
    fn reasoning_details_text_is_merged_by_index_and_signature_arrives_late() {
        let mut acc = StreamAccumulator::default();
        // The block's text streams in across two chunks; the signature that
        // authenticates it (Anthropic's thinking blocks carry one) shows up
        // null at first, then populated in a later chunk — a later `null`
        // must never blank out a signature an earlier chunk already set.
        for payload in [
            r#"{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.text","text":"step one, ","index":0,"signature":null}]}}]}"#,
            r#"{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.text","text":"step two","index":0,"signature":null}]}}]}"#,
            r#"{"choices":[{"delta":{"reasoning_details":[{"index":0,"signature":"sig-abc"}]}}]}"#,
        ] {
            acc.push_payload(payload).unwrap();
        }
        let details = acc.finish().reasoning_details.expect("reasoning details");
        assert_eq!(details.len(), 1);
        assert_eq!(details[0]["text"], "step one, step two");
        assert_eq!(details[0]["signature"], "sig-abc");
        assert_eq!(details[0]["type"], "reasoning.text");
    }

    #[test]
    fn reasoning_details_are_ordered_by_index_not_arrival() {
        let mut acc = StreamAccumulator::default();
        acc.push_payload(
            r#"{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.text","text":"second","index":1}]}}]}"#,
        )
        .unwrap();
        acc.push_payload(
            r#"{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.text","text":"first","index":0}]}}]}"#,
        )
        .unwrap();
        let details = acc.finish().reasoning_details.expect("reasoning details");
        assert_eq!(details[0]["text"], "first");
        assert_eq!(details[1]["text"], "second");
    }

    #[test]
    fn no_reasoning_details_yields_none() {
        let mut acc = StreamAccumulator::default();
        acc.push_payload(&content_chunk("hi")).unwrap();
        let message = acc.finish();
        assert_eq!(message.reasoning, None);
        assert_eq!(message.reasoning_details, None);
    }

    fn assistant_with_reasoning(tool_call: bool) -> ChatMessage {
        ChatMessage {
            role: "assistant".to_string(),
            content: Some("some text".to_string()),
            tool_calls: tool_call.then(|| {
                vec![ToolCall {
                    id: "call_1".to_string(),
                    call_type: function_call_type(),
                    function: FunctionCall {
                        name: "read_file".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]
            }),
            reasoning: Some("thinking".to_string()),
            reasoning_details: Some(vec![serde_json::json!({"type": "reasoning.text"})]),
            ..Default::default()
        }
    }

    #[test]
    fn reasoning_is_dropped_without_a_tool_call_to_lead_into() {
        // Thinking with nothing after it would be the message's final
        // block, which Anthropic rejects.
        let mut messages = vec![assistant_with_reasoning(false)];
        strip_dangling_reasoning(&mut messages);
        assert_eq!(messages[0].reasoning_details, None);
        // The prose is display-only and never sent, so it stays — that's
        // what `/verbose` shows for a turn that called no tool.
        assert_eq!(messages[0].reasoning, Some("thinking".to_string()));
        assert_eq!(messages[0].content, Some("some text".to_string()));
    }

    #[test]
    fn thinking_text_prefers_the_provider_prose() {
        let message = ChatMessage {
            reasoning: Some("the prose summary".to_string()),
            reasoning_details: Some(vec![serde_json::json!({"text": "block text"})]),
            ..Default::default()
        };
        assert_eq!(
            message.thinking_text(),
            Some("the prose summary".to_string())
        );
    }

    #[test]
    fn thinking_text_falls_back_to_the_blocks() {
        // What a reply reloaded from a session recorded before the prose
        // was stored looks like — the blocks are all that survived.
        let message = ChatMessage {
            reasoning: None,
            reasoning_details: Some(vec![
                serde_json::json!({"text": "first"}),
                serde_json::json!({"text": "second"}),
                serde_json::json!({"signature": "sig-only"}),
            ]),
            ..Default::default()
        };
        assert_eq!(message.thinking_text(), Some("first\n\nsecond".to_string()));
    }

    #[test]
    fn thinking_text_is_none_when_there_is_nothing_to_show() {
        assert_eq!(ChatMessage::default().thinking_text(), None);
        let blank = ChatMessage {
            reasoning: Some("   ".to_string()),
            reasoning_details: Some(vec![serde_json::json!({"text": ""})]),
            ..Default::default()
        };
        assert_eq!(blank.thinking_text(), None);
    }

    #[test]
    fn reasoning_is_kept_alongside_a_tool_call() {
        // Including in a turn the user has already spoken past — how far
        // back a thinking block sits is not what this rule is about.
        let mut messages = vec![
            assistant_with_reasoning(true),
            ChatMessage {
                role: "user".to_string(),
                content: Some("go on".to_string()),
                ..Default::default()
            },
            assistant_with_reasoning(true),
        ];
        strip_dangling_reasoning(&mut messages);
        assert!(messages[0].reasoning_details.is_some());
        assert!(messages[2].reasoning_details.is_some());
    }

    #[test]
    fn an_empty_tool_calls_array_counts_as_no_tool_call() {
        // The bug this guards against: a provider sending `tool_calls: []`
        // rather than omitting the field entirely on a turn that didn't
        // really call anything. `Some(vec![])` is `is_some()`, so a check
        // that only asked "is this None" would wrongly keep reasoning_details
        // attached to a message with no tool_use block to follow it.
        let mut messages = vec![ChatMessage {
            role: "assistant".to_string(),
            content: Some("no real tool call".to_string()),
            tool_calls: Some(vec![]),
            reasoning_details: Some(vec![serde_json::json!({"type": "reasoning.text"})]),
            ..Default::default()
        }];
        strip_dangling_reasoning(&mut messages);
        assert_eq!(messages[0].reasoning_details, None);
    }

    #[test]
    fn request_skeleton_describes_shape_without_leaking_content() {
        let request = ChatRequest {
            model: "anthropic/claude-sonnet-5".to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: Some("secret system prompt".to_string()),
                    ..Default::default()
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: Some("secret question".to_string()),
                    ..Default::default()
                },
                ChatMessage {
                    role: "assistant".to_string(),
                    tool_calls: Some(vec![ToolCall {
                        id: "call_1".to_string(),
                        call_type: function_call_type(),
                        function: FunctionCall {
                            name: "read_file".to_string(),
                            arguments: "{}".to_string(),
                        },
                    }]),
                    reasoning_details: Some(vec![serde_json::json!({"text": "secret thinking"})]),
                    ..Default::default()
                },
                ChatMessage {
                    role: "tool".to_string(),
                    content: Some("secret tool output".to_string()),
                    ..Default::default()
                },
            ],
            tools: Some(vec![serde_json::json!({})]),
            temperature: None,
            tool_choice: None,
            reasoning_effort: None,
            reasoning: None,
            stream: Some(true),
            stream_options: None,
        };

        let skeleton = request_skeleton(&request);
        // The assistant carries 1 tool call and 1 reasoning block that has
        // no signature (`!`), and no text content of its own (`-`).
        assert_eq!(
            skeleton,
            "model=anthropic/claude-sonnet-5 msgs=4 tools=1 stream=true [S U A(c1,r1!,-) T]"
        );
        // Nothing from any message body may end up in the log.
        for secret in [
            "secret system prompt",
            "secret question",
            "secret thinking",
            "secret tool output",
        ] {
            assert!(!skeleton.contains(secret), "{skeleton}");
        }
    }

    fn request_json(config: Config, temperature: Option<f32>, effort: Option<String>) -> Value {
        let request = Client::for_test(config).build_request(
            "m".to_string(),
            vec![],
            temperature,
            None,
            effort,
            false,
        );
        serde_json::to_value(&request).unwrap()
    }

    #[test]
    fn a_null_temperature_sends_no_temperature_field() {
        // The contract `clank temperature --clear` promises: not a zero, not
        // a default — no field at all, so the provider uses its own.
        let json = request_json(Config::default(), None, None);
        assert!(
            json.get("temperature").is_none(),
            "temperature must be absent, got {json}"
        );

        // Present when set. Compared with a tolerance because the field is
        // an `f32`: 0.7 reaches the wire as 0.699999988079071, which is the
        // same sampling temperature to any provider but not the same JSON
        // number.
        let json = request_json(Config::default(), Some(0.7), None);
        let sent = json["temperature"].as_f64().expect("a number");
        assert!((sent - 0.7).abs() < 1e-6, "{json}");
    }

    #[test]
    fn stream_options_asks_for_usage_only_when_streaming() {
        // Buffered requests get no usage-only final chunk to ask for in the
        // first place, and a provider that rejects unrecognized fields
        // should never see this one on a request it wasn't built for.
        let request = Client::for_test(Config::default()).build_request(
            "m".to_string(),
            vec![],
            None,
            None,
            None,
            false,
        );
        let json = serde_json::to_value(&request).unwrap();
        assert!(json.get("stream_options").is_none(), "{json}");

        let request = Client::for_test(Config::default()).build_request(
            "m".to_string(),
            vec![],
            None,
            None,
            None,
            true,
        );
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["stream_options"]["include_usage"], true, "{json}");
    }

    #[test]
    fn a_null_effort_sends_neither_effort_field() {
        // Both shapes have to stay off the wire, since a provider that
        // rejects unknown fields would fail on either.
        let json = request_json(Config::default(), None, None);
        assert!(json.get("reasoning").is_none(), "{json}");
        assert!(json.get("reasoning_effort").is_none(), "{json}");
    }

    #[test]
    fn effort_style_decides_which_shape_an_effort_takes() {
        let nested = Config {
            effort_style: Some("nested".to_string()),
            ..Config::default()
        };
        let json = request_json(nested, None, Some("high".to_string()));
        assert_eq!(json["reasoning"]["effort"], "high", "{json}");
        assert!(json.get("reasoning_effort").is_none(), "{json}");

        let flat = Config {
            effort_style: Some("flat".to_string()),
            ..Config::default()
        };
        let json = request_json(flat, None, Some("high".to_string()));
        assert_eq!(json["reasoning_effort"], "high", "{json}");
        assert!(json.get("reasoning").is_none(), "{json}");

        // "none" is the escape hatch for providers that reject both.
        let off = Config {
            effort_style: Some("none".to_string()),
            ..Config::default()
        };
        let json = request_json(off, None, Some("high".to_string()));
        assert!(json.get("reasoning").is_none(), "{json}");
        assert!(json.get("reasoning_effort").is_none(), "{json}");
    }

    #[test]
    fn an_unset_effort_style_falls_back_to_the_seeded_shape() {
        // A config written before `effort_style` was seeded holds `null`
        // there; requests still have to take a shape.
        let unset = Config {
            effort_style: None,
            ..Config::default()
        };
        let json = request_json(unset, None, Some("high".to_string()));
        assert_eq!(json["reasoning"]["effort"], "high", "{json}");
    }

    #[test]
    fn a_tool_call_goes_out_with_its_type_discriminator() {
        // The OpenAI schema requires `type` on every tool call. Without it
        // a provider translating the request into another API's shape can
        // fail to recognize the call.
        let call = ToolCall {
            id: "call_1".to_string(),
            call_type: function_call_type(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let json: serde_json::Value = serde_json::to_value(&call).unwrap();
        assert_eq!(json["type"], "function");
    }

    #[test]
    fn a_tool_call_stored_without_a_type_still_loads() {
        // Every tool call already in the database was written before the
        // field existed, so reading one back must not fail — it takes the
        // default and is written out correctly from then on.
        let call: ToolCall = serde_json::from_str(
            r#"{"id":"call_1","function":{"name":"read_file","arguments":"{}"}}"#,
        )
        .expect("a stored tool call with no type still deserializes");
        assert_eq!(call.call_type, "function");
    }

    #[test]
    fn no_captured_body_means_no_dump_note() {
        // Capture is off by default, so nothing is written and the log
        // entry keeps its content-free shape.
        assert_eq!(dump_note(None), "");
    }

    #[test]
    fn reasoning_prose_is_never_serialized_into_a_request() {
        // It comes back on a response and is kept for display, but only
        // `reasoning_details` — which carries the provider's own signature —
        // may be echoed back; unsigned prose must not reach the wire.
        let message = ChatMessage {
            role: "assistant".to_string(),
            reasoning: Some("thinking it through".to_string()),
            reasoning_details: Some(vec![serde_json::json!({"type": "reasoning.text"})]),
            ..Default::default()
        };
        let json = serde_json::to_string(&message).unwrap();
        assert!(!json.contains("thinking it through"), "{json}");
        assert!(!json.contains("\"reasoning\""), "{json}");
        assert!(json.contains("reasoning_details"), "{json}");
    }

    #[test]
    fn has_tool_calls_treats_none_and_empty_alike() {
        let base = ChatMessage {
            role: "assistant".to_string(),
            ..Default::default()
        };
        assert!(!base.has_tool_calls());
        assert!(!ChatMessage {
            tool_calls: Some(vec![]),
            ..base.clone()
        }
        .has_tool_calls());
        assert!(ChatMessage {
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                call_type: function_call_type(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            ..base
        }
        .has_tool_calls());
    }

    #[test]
    fn content_and_tool_calls_can_arrive_in_one_turn() {
        let mut acc = StreamAccumulator::default();
        acc.push_payload(&content_chunk("Let me check.")).unwrap();
        acc.push_payload(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"read_file","arguments":"{}"}}]}}]}"#,
        )
        .unwrap();

        let message = acc.finish();
        assert_eq!(message.content, Some("Let me check.".to_string()));
        assert_eq!(message.tool_calls.unwrap().len(), 1);
    }

    #[test]
    fn malformed_chunk_is_an_error_not_a_panic() {
        let mut acc = StreamAccumulator::default();
        assert!(acc.push_payload("{not json").is_err());
    }

    #[test]
    fn chunk_without_choices_is_tolerated() {
        // Some providers emit keepalive/usage-only frames.
        let mut acc = StreamAccumulator::default();
        assert_eq!(
            acc.push_payload(r#"{"usage":{"total_tokens":5}}"#).unwrap(),
            None
        );
        assert_eq!(acc.finish().content, None);
    }

    #[test]
    fn a_usage_chunk_is_captured_and_readable_before_finish() {
        let mut acc = StreamAccumulator::default();
        acc.push_payload(&content_chunk("hi")).unwrap();
        assert_eq!(acc.usage(), None, "no usage chunk has arrived yet");

        acc.push_payload(r#"{"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}"#)
            .unwrap();
        let usage = acc.usage().expect("usage chunk was sent");
        assert_eq!(usage.total_tokens, 12);
    }

    #[test]
    fn a_later_usage_chunk_replaces_rather_than_sums() {
        // Providers report one cumulative total for the whole request, not a
        // per-chunk delta.
        let mut acc = StreamAccumulator::default();
        acc.push_payload(r#"{"usage":{"total_tokens":5}}"#).unwrap();
        acc.push_payload(r#"{"usage":{"total_tokens":12}}"#).unwrap();
        assert_eq!(acc.usage().unwrap().total_tokens, 12);
    }

    #[test]
    fn has_visible_content_treats_none_and_blank_string_alike() {
        let none = ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: None,
            tool_call_id: None,
            ..Default::default()
        };
        let blank = ChatMessage {
            content: Some("   ".to_string()),
            ..none.clone()
        };
        let real = ChatMessage {
            content: Some("hi".to_string()),
            ..none.clone()
        };
        assert!(!none.has_visible_content());
        assert!(!blank.has_visible_content());
        assert!(real.has_visible_content());
    }
}
