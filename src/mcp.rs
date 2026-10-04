//! Talking to MCP servers, so the agent's tools can come from somewhere
//! other than this binary.
//!
//! An MCP server is a process that speaks JSON-RPC 2.0 over its stdin and
//! stdout, one message per line. It answers `tools/list` with the tools it
//! offers and `tools/call` when one is invoked. What arrives is translated
//! into [`ToolInfo`] and handed to [`crate::tools::set_registered`], after
//! which every gate, listing and bulk target in CCC treats it like any
//! other tool — that part needed no new policy at all.
//!
//! The module is in three layers, kept apart because only the first two are
//! worth testing and the third is where the surprises live:
//!
//! - [`Connection`], the protocol: framing, matching replies to requests,
//!   and giving up on one that never comes. Generic over its streams, so a
//!   test drives it through [`tokio::io::duplex`] with no process anywhere.
//! - [`tool_from_mcp`] and friends, the translation. Pure functions over
//!   JSON, tested against payloads captured from the reference servers.
//! - [`StdioServer`], the process. Spawning, and the part that is not what
//!   it looks like — see [`StdioServer::shutdown`].
//!
//! And a fourth concern that arrived with lazy connection: *when* a server
//! connects. It used to be once, at startup, unconditionally; now a server
//! comes up the first time a turn could offer one of its tools — see
//! [`ensure_connected`] and [`ensure_for_access`]. A clanker whose gates
//! drop a server therefore never pays to start it, and neither does a
//! command that only lists.

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, RwLock};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot};

use crate::tools::ToolInfo;

/// The protocol version CCC asks for.
///
/// A server replies with the version it will actually speak, which may not
/// be this one; nothing here depends on the difference yet, so the reply is
/// recorded rather than negotiated against. Both reference servers agreed
/// to this one.
pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// How long any one request waits before it is given up on.
///
/// Required rather than defensive. A server that stops answering does not
/// close its stdout — see [`StdioServer::shutdown`] — so without this a
/// `tools/call` waits on a pipe nobody will ever write to, and takes the
/// turn with it.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// How long `initialize` gets. Shorter than a call: a server that cannot
/// introduce itself promptly is not going to get better, and this one runs
/// at startup where something is waiting on it.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);

/// What to run, and under what name its tools appear.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerSpec {
    /// The prefix its tools get. No `/`, because that is the character the
    /// namespaced name is split on to route a call back here.
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    /// Extra environment for the child. Values, not names: the names are
    /// what a config file should hold, with the values fetched from the
    /// keychain and put in here on the way past.
    pub env: Vec<(String, String)>,
}

impl ServerSpec {
    pub fn new(name: impl Into<String>, command: impl Into<String>) -> Result<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(anyhow!("a server needs a name"));
        }
        if name.contains(SEPARATOR) {
            return Err(anyhow!(
                "{name:?} cannot be a server name: `{SEPARATOR}` is what separates \
                 a server from its tool in `{name}{SEPARATOR}some_tool`"
            ));
        }
        // The server's name is the front of every tool name it contributes,
        // and a provider rejects the whole request over one bad character —
        // so it is caught here, when it is typed, rather than at the first
        // turn after it.
        if !representable(&name) {
            return Err(anyhow!(
                "{name:?} cannot be a server name: a tool name reaches the model as \
                 `{name}{SEPARATOR}some_tool`, and providers only accept letters, \
                 digits, `_` and `-` there"
            ));
        }
        Ok(Self {
            name,
            command: command.into(),
            args: Vec::new(),
            env: Vec::new(),
        })
    }

    pub fn with_args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }
}

/// What separates a server from its tool in a namespaced name.
///
/// Not `/`, which is what this was and which does not work. A tool name
/// goes to the provider inside the schema, and providers hold it to
/// `^[a-zA-Z0-9_-]{1,128}$` — no slash. The request is rejected whole, so
/// one unrepresentable name costs the entire turn rather than one tool:
///
/// ```text
/// tools.0.custom.name: String should match pattern '^[a-zA-Z0-9_-]{1,128}$'
/// ```
///
/// A doubled underscore is in the permitted set, is the convention other
/// harnesses settled on, and stays unambiguous as long as a server's own
/// name cannot contain one — which [`ServerSpec::new`] enforces.
pub const SEPARATOR: &str = "__";

/// Every character a provider will accept in a tool name.
fn representable(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// The name a tool is known by once it is CCC's: `server__tool`.
///
/// Namespaced because the first server anyone installs collides. The
/// reference filesystem server exports `read_file`, `write_file` and
/// `search_files` — three of the seven built-ins — and
/// `tools::set_registered` refuses a bare collision rather than letting one
/// shadow the other.
pub fn namespaced(server: &str, tool: &str) -> String {
    format!("{server}{SEPARATOR}{tool}")
}

/// Which server a namespaced name belongs to, and what it is called there.
///
/// `None` for a name with no separator, which is a built-in or a model's
/// invention rather than anything to route. Split at the *first* separator,
/// so a tool whose own name contains one still routes to the right server.
pub fn route(name: &str) -> Option<(&str, &str)> {
    name.split_once(SEPARATOR)
}

/// The bucket an MCP tool falls in, from the hints it declares about itself.
///
/// `readOnlyHint` is the whole of it. It split both reference servers'
/// tools correctly, and nothing else in `annotations` maps onto what CCC's
/// categories mean:
///
/// - `web` is a category here because `web_fetch` touches nothing local,
///   which is a different axis from anything a server declares. Mapping
///   `openWorldHint` onto it files `gzip-file-as-resource` — a tool that
///   writes a file — as a web tool, so it is left for built-ins.
/// - `terminal` is CCC's own "this can do anything you can", and a server
///   claiming it would turn a default of `never` into a tool that silently
///   does nothing.
///
/// A tool that declares nothing is `write`, the more restricted of the two
/// that remain: without a `readOnlyHint` there is no basis for believing it
/// only reads. That still lands it on `ask` rather than off, which is the
/// same answer the gates already gave an unrecognised name — "the last
/// thing that should run unattended" — rather than a quieter one.
///
/// These are hints a server asserts about itself, so they set the default a
/// person can see in `clank tools` and change. They are never the
/// permission.
pub fn category_from_annotations(annotations: Option<&Value>) -> &'static str {
    match annotations
        .and_then(|a| a.get("readOnlyHint"))
        .and_then(Value::as_bool)
    {
        Some(true) => "read",
        _ => "write",
    }
}

/// A line for `clank tools`, from whatever the server gave us to say.
///
/// `title` first because every tool on both reference servers had one and
/// it is already written to be short ("Echo Tool", "Read File"). A
/// description is prose and gets cut at a word.
fn summarize(tool: &Value) -> String {
    const WIDTH: usize = 60;
    let text = tool
        .get("title")
        .and_then(Value::as_str)
        .filter(|title| !title.trim().is_empty())
        .or_else(|| tool.get("description").and_then(Value::as_str))
        .unwrap_or("(the server said nothing about this tool)")
        .trim();

    let first_line = text.lines().next().unwrap_or(text).trim();
    if first_line.chars().count() <= WIDTH {
        return first_line.to_string();
    }
    let cut: String = first_line.chars().take(WIDTH).collect();
    let kept = match cut.rsplit_once(' ') {
        Some((head, _)) if head.chars().count() > WIDTH / 2 => head,
        _ => cut.as_str(),
    };
    format!("{}…", kept.trim_end_matches([' ', ',', '.', ';', ':']))
}

/// The schema to send a provider for an MCP tool.
///
/// Named by its *namespaced* name, because that is the name the model will
/// call and the only one that says where to route it.
///
/// `$schema` is dropped from the parameters. Both reference servers put a
/// draft-07 declaration in every `inputSchema`, which is correct JSON
/// Schema and not part of what an OpenAI-compatible `parameters` is
/// expected to carry — and `base_url` can point at anything, including
/// something strict about keys it does not know.
fn schema_for(full_name: &str, tool: &Value) -> Value {
    let mut parameters = tool
        .get("inputSchema")
        .cloned()
        .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
    if let Some(object) = parameters.as_object_mut() {
        object.remove("$schema");
    }
    json!({
        "type": "function",
        "function": {
            "name": full_name,
            "description": tool.get("description").and_then(Value::as_str).unwrap_or(""),
            "parameters": parameters,
        }
    })
}

/// One entry of a `tools/list` reply, as CCC's own [`ToolInfo`].
pub fn tool_from_mcp(server: &str, tool: &Value) -> Result<ToolInfo> {
    let bare = tool
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("{server} offered a tool with no name"))?;
    let full_name = namespaced(server, bare);
    // Refused here so the tool is skipped and named — see `tools_from_list`.
    // A name the provider will not take fails the whole request, not just
    // the call, so one tool with a dot in it would otherwise cost every
    // turn for as long as the server is connected.
    if !representable(&full_name) {
        return Err(anyhow!(
            "{server} offers {bare:?}, which cannot be sent to a provider: \
             a tool name may only hold letters, digits, `_` and `-`, and at \
             most 128 of them"
        ));
    }
    let category = category_from_annotations(tool.get("annotations"));
    ToolInfo::new(
        full_name.clone(),
        category,
        summarize(tool),
        schema_for(&full_name, tool),
    )
}

/// Every tool in a `tools/list` reply.
///
/// A tool that cannot be translated is skipped rather than failing the
/// list: one unnamed entry should cost that entry, not the server.
pub fn tools_from_list(server: &str, reply: &Value) -> (Vec<ToolInfo>, Vec<String>) {
    let mut tools = Vec::new();
    let mut skipped = Vec::new();
    let listed = reply
        .get("tools")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for entry in listed {
        match tool_from_mcp(server, entry) {
            Ok(tool) => tools.push(tool),
            Err(e) => skipped.push(e.to_string()),
        }
    }
    (tools, skipped)
}

/// A JSON-RPC conversation over a pair of streams.
///
/// Owns a reader task and a writer task so a request can be awaited without
/// holding a lock across it, and so the reader can keep draining while
/// several calls are outstanding.
pub struct Connection {
    outgoing: mpsc::UnboundedSender<String>,
    pending: Pending,
    next_id: AtomicI64,
}

type Pending = Arc<Mutex<HashMap<i64, oneshot::Sender<Result<Value>>>>>;

impl Connection {
    /// Starts the reader and writer tasks over streams already connected to
    /// something that speaks the protocol.
    pub fn new<R, W>(reader: R, writer: W) -> Self
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let (outgoing, mut to_write) = mpsc::unbounded_channel::<String>();

        let reading = Arc::clone(&pending);
        tokio::spawn(async move {
            let mut lines = BufReader::new(reader).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(message) = serde_json::from_str::<Value>(&line) else {
                    // Not JSON. Some servers print to stdout despite the
                    // protocol owning it; skipping the line is survivable
                    // where failing the connection is not.
                    continue;
                };
                // A notification has no id and nothing is waiting for it.
                // `notifications/tools/list_changed` arrives unprompted and
                // interleaved with replies, which is why a reply cannot be
                // read by pairing one read to one write.
                let Some(id) = message.get("id").and_then(Value::as_i64) else {
                    continue;
                };
                let waiting = reading
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .remove(&id);
                if let Some(tx) = waiting {
                    let _ = tx.send(reply_to_result(message));
                }
            }
            // Stdout closed. Every outstanding request is now never going
            // to be answered, so each is failed here: the alternative is a
            // caller awaiting a oneshot whose sender has been dropped,
            // which says nothing about why.
            let orphaned: Vec<_> = reading
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .drain()
                .collect();
            for (_, tx) in orphaned {
                let _ = tx.send(Err(anyhow!(
                    "the server closed its output with this call unanswered"
                )));
            }
        });

        tokio::spawn(async move {
            let mut writer = writer;
            while let Some(line) = to_write.recv().await {
                if writer.write_all(line.as_bytes()).await.is_err() {
                    break;
                }
                if writer.flush().await.is_err() {
                    break;
                }
            }
        });

        Self {
            outgoing,
            pending,
            next_id: AtomicI64::new(1),
        }
    }

    /// Sends a request and waits for the reply with that id.
    pub async fn request(&self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, tx);

        let message = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        if self.outgoing.send(format!("{message}\n")).is_err() {
            self.pending
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&id);
            return Err(anyhow!("the server's input is closed"));
        }

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(result)) => result,
            // The reader task went away without answering.
            Ok(Err(_)) => Err(anyhow!("the connection ended before {method} was answered")),
            Err(_) => {
                // Dropped from `pending` so a late reply is discarded rather
                // than handed to whoever asks next.
                self.pending
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .remove(&id);
                Err(anyhow!(
                    "{method} went unanswered for {}s",
                    timeout.as_secs()
                ))
            }
        }
    }

    /// Sends a notification, which by definition has no reply to wait for.
    pub fn notify(&self, method: &str) -> Result<()> {
        let message = json!({"jsonrpc": "2.0", "method": method});
        self.outgoing
            .send(format!("{message}\n"))
            .map_err(|_| anyhow!("the server's input is closed"))
    }

    /// The handshake, then the tools.
    pub async fn initialize(&self) -> Result<Value> {
        let result = self
            .request(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "clanker-command-center", "version": env!("CARGO_PKG_VERSION")},
                }),
                CONNECT_TIMEOUT,
            )
            .await
            .context("the server would not introduce itself")?;
        self.notify("notifications/initialized")?;
        Ok(result)
    }

    pub async fn list_tools(&self) -> Result<Value> {
        self.request("tools/list", json!({}), CONNECT_TIMEOUT).await
    }

    pub async fn call_tool(&self, tool: &str, arguments: Value) -> Result<Value> {
        self.request(
            "tools/call",
            json!({"name": tool, "arguments": arguments}),
            REQUEST_TIMEOUT,
        )
        .await
    }
}

/// A JSON-RPC reply as a `Result`, so an error reply and a dead connection
/// arrive the same way.
fn reply_to_result(message: Value) -> Result<Value> {
    if let Some(error) = message.get("error") {
        let code = error.get("code").and_then(Value::as_i64).unwrap_or(0);
        let text = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("no message");
        return Err(anyhow!("the server refused: {text} (code {code})"));
    }
    Ok(message.get("result").cloned().unwrap_or(Value::Null))
}

/// What a `tools/call` reply says, as text for the model.
///
/// MCP returns a list of content blocks; CCC's tool results are a JSON
/// value, and every existing tool returns something the model reads as
/// text. `isError` is carried through as an error rather than as content,
/// so a failing MCP tool reads like a failing built-in.
pub fn call_result_text(result: &Value) -> Result<String> {
    let blocks = result
        .get("content")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let text = blocks
        .iter()
        .filter_map(|block| match block.get("type").and_then(Value::as_str) {
            Some("text") => block
                .get("text")
                .and_then(Value::as_str)
                .map(str::to_string),
            // A resource link or an image is named rather than inlined: the
            // model cannot be handed bytes here, and a silent omission
            // would read as an empty result.
            Some(kind) => Some(format!("[{kind} content the agent cannot read inline]")),
            None => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    if result.get("isError").and_then(Value::as_bool) == Some(true) {
        return Err(anyhow!(
            "{}",
            if text.is_empty() {
                "the tool failed and said nothing".to_string()
            } else {
                text
            }
        ));
    }
    Ok(text)
}

/// A server running as a child process, talking over its stdio.
///
/// Holds the connection, the tools it offered at connect time, and what is
/// needed to stop it — which is more than it sounds.
pub struct StdioServer {
    connection: Connection,
    tools: Vec<ToolInfo>,
    /// Taken by [`Self::shutdown`], which is why it is behind a lock and an
    /// `Option`: a connected server is shared (every turn may call it) so
    /// stopping it cannot need `self` by value, and stopping it twice has to
    /// be harmless.
    child: Mutex<Option<tokio::process::Child>>,
    /// The child's process group, when it could be put in one of its own.
    /// `None` on platforms where that isn't how processes are grouped.
    #[cfg_attr(not(unix), allow(dead_code))]
    group: Option<u32>,
}

impl StdioServer {
    /// Spawns the server, shakes hands, and asks what it offers.
    ///
    /// The tools are read once here rather than per turn. A server may
    /// announce `notifications/tools/list_changed` later, which is noted
    /// and not yet acted on: re-reading the list means re-registering it,
    /// and the registry is rebuilt wholesale from every connected server
    /// rather than patched per server.
    pub async fn connect(spec: &ServerSpec) -> Result<Self> {
        let mut command = tokio::process::Command::new(&spec.command);
        command
            .args(&spec.args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            // Belt and braces with `shutdown`: this reaps the process we
            // spawned if the handle is ever dropped without one, and does
            // nothing about its children — which is the whole problem.
            .kill_on_drop(true);
        for (key, value) in &spec.env {
            command.env(key, value);
        }

        // Its own process group, so stopping it can mean stopping
        // everything it started. See `shutdown` for why that is necessary
        // rather than tidy.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.as_std_mut().process_group(0);
        }

        let mut child = command
            .spawn()
            .with_context(|| format!("could not start {}: {}", spec.name, spec.command))?;

        // Taken before anything is written: a server that chatters on
        // stderr fills the pipe and then blocks on the write, which looks
        // exactly like a server that has stopped answering. Both reference
        // servers write a startup line, so this is the normal case.
        if let Some(stderr) = child.stderr.take() {
            let name = spec.name.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    // Kept out of the transcript: this is the server
                    // talking to its operator, not to the model.
                    crate::error_log::log_error(&format!("mcp {name}"), &line);
                }
            });
        }

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("{} gave no stdin to write to", spec.name))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("{} gave no stdout to read", spec.name))?;

        #[cfg(unix)]
        let group = child.id();
        #[cfg(not(unix))]
        let group = None;

        let connection = Connection::new(stdout, stdin);
        connection
            .initialize()
            .await
            .with_context(|| format!("{} started but would not speak the protocol", spec.name))?;

        let listed = connection
            .list_tools()
            .await
            .with_context(|| format!("{} would not say what tools it has", spec.name))?;
        let (tools, skipped) = tools_from_list(&spec.name, &listed);
        for complaint in skipped {
            crate::error_log::log_error(&format!("mcp {}", spec.name), &complaint);
        }

        Ok(Self {
            connection,
            tools,
            child: Mutex::new(Some(child)),
            group,
        })
    }

    /// What this server offered, already namespaced and categorised.
    pub fn tools(&self) -> &[ToolInfo] {
        &self.tools
    }

    /// Calls one of this server's tools, by its bare name.
    pub async fn call(&self, tool: &str, arguments: Value) -> Result<String> {
        let result = self.connection.call_tool(tool, arguments).await?;
        call_result_text(&result)
    }

    /// Stops the server and everything it started.
    ///
    /// The part that is not what it looks like. A server is usually
    /// launched through a runner — `npx`, `uvx`, a wrapper script — and the
    /// pid we were handed belongs to the runner, not the server. Killing it
    /// leaves the real process alive, reparented to init, **still holding
    /// the stdout pipe open**: a call in flight gets no reply and no EOF
    /// either, so a reader waits forever on a pipe with no writer that will
    /// ever write. Measured with the reference everything server, where
    /// `npm exec` forks the server as a grandchild.
    ///
    /// So the signal goes to the process *group*, which is why `connect`
    /// puts the child in one of its own. `SIGTERM` first, because a server
    /// may have state worth flushing; `SIGKILL` after a grace period,
    /// because one that ignores the first is exactly the kind that needed
    /// stopping. Closing stdin would be gentler still and both reference
    /// servers exit on it, but nothing in the protocol promises that.
    ///
    /// On a platform with no process groups this degrades to killing the
    /// process we spawned, and a server launched through a runner there can
    /// outlive CCC. Noted rather than solved: it wants a job object, which
    /// is a different mechanism rather than a different signal.
    pub async fn shutdown(&self) {
        // Taken, so a second call is a no-op rather than a second round of
        // signals at a pgid that may have been reused by then. Nothing is
        // awaited while the lock is held.
        let Some(mut child) = self.child.lock().unwrap_or_else(|e| e.into_inner()).take() else {
            return;
        };

        #[cfg(unix)]
        if let Some(group) = self.group {
            let pgid = group as libc::pid_t;
            // SAFETY: a kill to a pgid we created, with a signal number
            // from libc. The group is this server's own, so nothing else is
            // in it to be hit by mistake.
            unsafe {
                libc::killpg(pgid, libc::SIGTERM);
            }

            // Waited on the *group*, not on the child. An earlier version
            // of this returned as soon as the child exited, which a live
            // test caught: `npm exec` honours SIGTERM and goes immediately
            // while the server it forked does not, so the pid being gone
            // proved nothing and left the orphan this whole mechanism
            // exists to prevent. The child is reaped each time round only
            // so it stops counting as a member of its own group.
            for _ in 0..20 {
                let _ = child.try_wait();
                if !group_alive(pgid) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }

            // SAFETY: as above.
            unsafe {
                libc::killpg(pgid, libc::SIGKILL);
            }
            let _ = child.wait().await;
            // Reparented children are reaped by init rather than by us, so
            // the group empties a moment after the signal rather than with
            // it. Confirmed here so a caller that quits immediately after
            // this does not race it.
            for _ in 0..20 {
                if !group_alive(pgid) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            return;
        }
        let _ = child.kill().await;
    }
}

/// The servers that are up, by name.
///
/// Process-wide for the same reason the tool registry is: one server per
/// config entry, shared by every clanker, outliving any one session. A
/// server is behind an `Arc` so a call can take a handle and drop the lock
/// before awaiting — the alternative serialises every tool call in the
/// process behind one mutex.
static CONNECTED: RwLock<Option<HashMap<String, Arc<StdioServer>>>> = RwLock::new(None);

fn connected() -> std::sync::RwLockReadGuard<'static, Option<HashMap<String, Arc<StdioServer>>>> {
    CONNECTED.read().unwrap_or_else(|e| e.into_inner())
}

/// What happened to one server as it was brought up.
pub struct Started {
    pub name: String,
    /// How many tools it offered, or why it did not get that far.
    pub outcome: Result<usize>,
}

/// Serialises every connect.
///
/// Two clankers whose first turns land together would otherwise each see
/// the same server missing and spawn a copy of it. A `tokio` mutex rather
/// than a `std` one because connecting awaits, and it is always taken
/// first, before anything touches [`CONNECTED`], so there is no order to
/// get wrong.
static CONNECTING: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Servers that would not come up, by name, for the life of the process.
///
/// Without this a server that will not start is retried at every turn that
/// wants it — and the failure is not always the fast spawn error of a
/// missing binary, but a handshake that waits out [`CONNECT_TIMEOUT`],
/// twenty seconds a turn. One attempt per process; `/mcp reconnect` is how
/// to try again, and clears this.
static FAILED: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

/// What has already been said about a server that would not start, so the
/// per-turn path says it once rather than once a turn.
///
/// [`ensure_for_access`] runs before every turn, and a server whose token is
/// not set would otherwise write the same line every time — into a log that
/// keeps a hundred entries, so a handful of turns would push everything else
/// out of it. Keyed by what was wrong rather than by the server, so a second
/// missing variable is still reported. `/mcp reconnect` clears it along with
/// [`FAILED`], which is the point at which the reason to complain may have
/// gone away.
static REPORTED: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

/// [`crate::error_log::log_error`] for the first time `key` is seen this
/// process — see [`REPORTED`].
fn report_once(key: &str, context: &str, message: &str) {
    let first = REPORTED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key.to_string());
    if first {
        crate::error_log::log_error(context, message);
    }
}

/// Connects every configured server and registers everything they offer.
///
/// One server failing costs that server. The rest connect, their tools are
/// registered, and the failure is reported for whoever is listening to
/// print — a missing `npx` or a server that will not start is a reason to
/// carry on without it, not a reason to refuse to open a clanker.
///
/// This is the *reconnect* path — `/mcp reconnect`, and the CLI's version
/// of it — so it starts from nothing: everything up is stopped and every
/// failure forgotten, because the reason to run it is a server that was
/// rebuilt. Opening a clanker goes through [`ensure_connected`] instead,
/// which starts only what a turn needs.
pub async fn connect_all(specs: &[ServerSpec]) -> Vec<Started> {
    let _connecting = CONNECTING.lock().await;

    // Whatever was up before this call. Publishing a new map would only
    // *drop* the old servers, and a drop reaps the process we spawned
    // without touching the group it started — which for anything behind a
    // runner leaves the real server alive holding its pipes.
    shutdown_all().await;
    FAILED.lock().unwrap_or_else(|e| e.into_inner()).clear();
    REPORTED.lock().unwrap_or_else(|e| e.into_inner()).clear();
    // The tools go with the servers. [`connect_needed`] rebuilds the
    // registry from what is up, but only when something came up — so
    // without this a reconnect where every server failed would leave the
    // model offered tools whose server is gone. An empty set cannot be
    // refused, since there is nothing in it to collide, so there is no
    // failure here to report.
    let _ = crate::tools::set_registered(Vec::new());

    connect_needed(specs, |_| true).await
}

/// Brings up the servers `wanted` asks for, and leaves the rest alone.
///
/// The lazy half of connecting, and the reason a clanker that never calls a
/// server never starts it: a server already up is left as it is, one that
/// failed earlier this process is not retried, and one `wanted` says no to
/// is not touched at all. What comes up is shared with every other clanker,
/// because the servers are the process's — see [`CONNECTED`].
///
/// The registry is rebuilt wholesale from the servers that are up, because
/// `set_registered` is total: that is what makes a first connection, a
/// reconnect and a removal the same code path with no stale entry to clean
/// up.
pub async fn ensure_connected(specs: &[ServerSpec], wanted: impl Fn(&str) -> bool) -> Vec<Started> {
    let _connecting = CONNECTING.lock().await;
    connect_needed(specs, wanted).await
}

/// [`ensure_connected`] with the lock already held — see [`CONNECTING`].
async fn connect_needed(specs: &[ServerSpec], wanted: impl Fn(&str) -> bool) -> Vec<Started> {
    let up: HashSet<String> = connected()
        .as_ref()
        .map(|live| live.keys().cloned().collect())
        .unwrap_or_default();
    let failed = FAILED.lock().unwrap_or_else(|e| e.into_inner()).clone();

    let mut started = Vec::new();
    let mut fresh: Vec<(String, Arc<StdioServer>)> = Vec::new();

    for spec in specs {
        if !wants_connecting(&spec.name, &wanted, &up, &failed) {
            continue;
        }
        match StdioServer::connect(spec).await {
            Ok(server) => {
                let count = server.tools().len();
                started.push(Started {
                    name: spec.name.clone(),
                    outcome: Ok(count),
                });
                fresh.push((spec.name.clone(), Arc::new(server)));
            }
            Err(e) => {
                FAILED
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(spec.name.clone());
                started.push(Started {
                    name: spec.name.clone(),
                    outcome: Err(e),
                });
            }
        }
    }

    if fresh.is_empty() {
        // Nothing new, so nothing to publish and nothing to re-register:
        // the tools of the servers already up are already in the registry.
        return started;
    }

    // The servers that were up, plus the ones that just came. Cloned out of
    // the lock before anything is awaited — the map is a handful of `Arc`s.
    let mut live: HashMap<String, Arc<StdioServer>> =
        connected().as_ref().cloned().unwrap_or_default();
    let mut offered: Vec<ToolInfo> = Vec::new();
    for server in live.values() {
        offered.extend(server.tools().iter().cloned());
    }
    for (_, server) in &fresh {
        offered.extend(server.tools().iter().cloned());
    }

    // Registered before the servers are published, so a tool can never be
    // callable before it is governable.
    if let Err(e) = crate::tools::set_registered(offered) {
        // The tools are refused as a set — a collision or a duplicate — so
        // nothing is registered and nothing should be reachable either.
        // Remembered as failed like a server that would not start: the
        // collision is in what they offer, so the next turn would spawn
        // them only to refuse them again.
        FAILED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend(fresh.iter().map(|(name, _)| name.clone()));
        for (_, server) in &fresh {
            server.shutdown().await;
        }
        started.push(Started {
            name: "(registry)".to_string(),
            outcome: Err(e),
        });
        return started;
    }

    for (name, server) in fresh {
        live.insert(name, server);
    }
    *CONNECTED.write().unwrap_or_else(|e| e.into_inner()) = Some(live);
    started
}

/// Whether `spec` is a server this call should try to start.
///
/// Split out from the spawning so the decision is testable on its own: the
/// async path around it mutates process-wide state, which a test running
/// beside others has no business doing.
fn wants_connecting(
    name: &str,
    wanted: &impl Fn(&str) -> bool,
    up: &HashSet<String>,
    failed: &HashSet<String>,
) -> bool {
    wanted(name) && !up.contains(name) && !failed.contains(name)
}

/// Whether a configured server is worth resolving at all.
///
/// [`ensure_for_access`] runs before every turn, and resolving a server
/// reads the OS keychain once per environment variable it declares — so one
/// that is already up, one that already failed this process, and one the
/// gates say no to are all skipped before that cost is paid. Only the
/// resolution is saved: the same three questions are asked again, under the
/// lock, by [`connect_needed`], so nothing here is load-bearing.
fn worth_resolving(name: &str, access: &crate::config::ToolAccessSettings) -> bool {
    let up: HashSet<String> = connected()
        .as_ref()
        .map(|servers| servers.keys().cloned().collect())
        .unwrap_or_default();
    let failed = FAILED.lock().unwrap_or_else(|e| e.into_inner()).clone();
    wants_connecting(name, &|name: &str| access.server_needed(name), &up, &failed)
}

/// Connects the configured servers this clanker's gates still leave room
/// for.
///
/// The glue between the gates (config's business) and the connection (this
/// module's), called by a turn before it builds its request: a server comes
/// up only if the gates could offer one of its tools, which is what lets
/// `tools never gj` keep gj's process from ever being spawned, and lets a
/// clanker that is not agentic start nothing at all.
///
/// Returns what happened, for the caller to report through its own channel.
/// A success is silent, because the evidence is the tools arriving in the
/// next request.
pub async fn ensure_for_access(access: &crate::config::ToolAccessSettings) -> Vec<Started> {
    let Ok(config) = crate::config::load_config() else {
        // A config that will not parse is reported by every other path
        // already; failing here as well would say it twice.
        return Vec::new();
    };
    if config.mcp_servers.is_empty() {
        return Vec::new();
    }

    let mut specs = Vec::new();
    for server in &config.mcp_servers {
        if !worth_resolving(&server.name, access) {
            continue;
        }
        match crate::config::resolve_server(server) {
            Ok((spec, missing)) => {
                for variable in missing {
                    // The log rather than the screen: this runs inside a
                    // turn, where a TUI owns the display. It is where a
                    // server that will not start explains itself.
                    report_once(
                        &format!("env:{}:{variable}", server.name),
                        &format!("mcp {}", server.name),
                        &format!(
                            "{variable} is not set — clank mcp env {} {variable}",
                            server.name
                        ),
                    );
                }
                specs.push(spec);
            }
            Err(e) => report_once(
                &format!("spec:{}", server.name),
                &format!("mcp {}", server.name),
                &e.to_string(),
            ),
        }
    }

    ensure_connected(&specs, |name| access.server_needed(name)).await
}

/// Routes a call to the server that owns the tool.
///
/// The namespaced name is the whole routing table: `fs/read_text_file` is
/// the server `fs` and its own `read_text_file`, which is the name it has
/// to be called by.
pub async fn call(full_name: &str, arguments: Value) -> Result<String> {
    let Some((server_name, tool)) = route(full_name) else {
        return Err(anyhow!(
            "{full_name} is not a tool from a server; there is no `{SEPARATOR}` in it to route on"
        ));
    };
    let server = connected()
        .as_ref()
        .and_then(|live| live.get(server_name).cloned())
        .ok_or_else(|| {
            anyhow!("no server named {server_name} is connected, so {full_name} cannot be called")
        })?;
    server.call(tool, arguments).await
}

/// How many tools each connected server is offering, in name order.
///
/// What `/mcp` reports and what a reconnect is measured against: the
/// difference between a server being configured and a server being up.
pub fn connected_counts() -> Vec<(String, usize)> {
    let mut counts: Vec<(String, usize)> = connected()
        .as_ref()
        .map(|live| {
            live.iter()
                .map(|(name, server)| (name.clone(), server.tools().len()))
                .collect()
        })
        .unwrap_or_default();
    counts.sort();
    counts
}

/// Stops every server, and everything they started.
///
/// Called on the way out. Nothing depends on it running — a dropped
/// `StdioServer` still kills the process it spawned — but a drop cannot
/// reach that process's own children, which is the whole point of
/// [`StdioServer::shutdown`].
pub async fn shutdown_all() {
    let live = CONNECTED
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .unwrap_or_default();
    for (_, server) in live {
        server.shutdown().await;
    }
}

/// Whether any process is still in this group.
///
/// Signal 0 is delivered to nothing and checks for existence, so this
/// answers "is the group empty" without touching whatever is in it. The
/// question the direct child's exit status cannot answer: a server launched
/// through a runner outlives the pid we were handed.
#[cfg(unix)]
fn group_alive(pgid: libc::pid_t) -> bool {
    // SAFETY: signal 0 delivers nothing; this is a liveness query.
    unsafe { libc::killpg(pgid, 0) == 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::duplex;

    /// Whether the process has no server up — an unpublished map and an
    /// empty one say the same thing, so both count.
    fn nothing_is_up() -> bool {
        connected().as_ref().is_none_or(HashMap::is_empty)
    }

    /// One entry from the reference filesystem server's `tools/list`,
    /// copied verbatim from a captured reply rather than written by hand.
    fn real_read_text_file() -> Value {
        json!({
            "name": "read_text_file",
            "title": "Read Text File",
            "description": "Read the complete contents of a file from the file system as text.",
            "inputSchema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "tail": {"description": "the last N lines", "type": "number"},
                    "head": {"description": "the first N lines", "type": "number"}
                },
                "required": ["path"]
            },
            "annotations": {"readOnlyHint": true, "openWorldHint": false},
            "execution": {"taskSupport": "forbidden"},
            "outputSchema": {"type": "object", "properties": {"content": {"type": "string"}}}
        })
    }

    fn real_write_file() -> Value {
        json!({
            "name": "write_file",
            "title": "Write File",
            "description": "Create a new file or completely overwrite an existing file.",
            "inputSchema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"]
            },
            "annotations": {
                "readOnlyHint": false,
                "idempotentHint": true,
                "destructiveHint": true,
                "openWorldHint": false
            }
        })
    }

    #[test]
    fn a_tool_is_namespaced_so_it_cannot_shadow_a_built_in() {
        // `write_file` is a built-in. The filesystem server exports one
        // too, and `set_registered` refuses a bare collision — so the name
        // has to arrive already namespaced, not get fixed up later.
        let tool = tool_from_mcp("fs", &real_write_file()).unwrap();
        assert_eq!(tool.name, "fs__write_file");
        assert_eq!(route(&tool.name), Some(("fs", "write_file")));
        // And the model is told the namespaced name, which is the only one
        // that says where to send the call.
        assert_eq!(tool.schema["function"]["name"], "fs__write_file");
        crate::tools::validate(&[tool]).expect("namespaced, so it no longer collides");
    }

    #[test]
    fn read_only_tools_land_in_read_and_everything_else_in_write() {
        assert_eq!(
            tool_from_mcp("fs", &real_read_text_file())
                .unwrap()
                .category,
            "read"
        );
        assert_eq!(
            tool_from_mcp("fs", &real_write_file()).unwrap().category,
            "write"
        );
    }

    #[test]
    fn a_tool_that_declares_nothing_is_treated_as_a_writer() {
        // No basis for believing it only reads, so it gets the more
        // restricted of the two buckets — which still means `ask`, not off.
        assert_eq!(category_from_annotations(None), "write");
        assert_eq!(category_from_annotations(Some(&json!({}))), "write");
        assert_eq!(
            category_from_annotations(Some(&json!({"readOnlyHint": false}))),
            "write"
        );
        let tool = tool_from_mcp("x", &json!({"name": "mystery"})).unwrap();
        assert_eq!(tool.category, "write");
        assert_eq!(
            crate::config::default_access(&tool.name),
            crate::config::ToolAccess::Ask
        );
    }

    #[test]
    fn open_world_does_not_mean_web() {
        // Measured: `gzip-file-as-resource` on the everything server is
        // openWorldHint true and writes a file. `web` is CCC's bucket for
        // "touches nothing local", which is a different axis.
        let gzip = json!({
            "name": "gzip-file-as-resource",
            "annotations": {"readOnlyHint": false, "openWorldHint": true}
        });
        assert_eq!(tool_from_mcp("ev", &gzip).unwrap().category, "write");
    }

    #[test]
    fn the_draft_declaration_is_stripped_from_the_parameters() {
        // Correct JSON Schema, not part of what an OpenAI-compatible
        // `parameters` carries, and `base_url` can point at something
        // strict about keys it does not recognise.
        let tool = tool_from_mcp("fs", &real_read_text_file()).unwrap();
        let parameters = &tool.schema["function"]["parameters"];
        assert!(parameters.get("$schema").is_none(), "{parameters}");
        // The rest of the schema survives intact.
        assert_eq!(parameters["type"], "object");
        assert_eq!(parameters["required"], json!(["path"]));
        assert!(parameters["properties"]["head"].is_object());
    }

    #[test]
    fn a_tool_with_no_schema_still_gets_a_usable_one() {
        let tool = tool_from_mcp("x", &json!({"name": "bare"})).unwrap();
        assert_eq!(tool.schema["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn a_summary_is_short_enough_for_a_terminal_row() {
        assert_eq!(
            tool_from_mcp("fs", &real_read_text_file()).unwrap().summary,
            "Read Text File"
        );
        // No title: the description, cut at a word.
        let wordy = json!({
            "name": "x",
            "description": "Reads the complete contents of a file from the file \
                            system as text and returns every line of it"
        });
        let summary = tool_from_mcp("fs", &wordy).unwrap().summary;
        assert!(summary.chars().count() <= 61, "{summary:?}");
        assert!(summary.ends_with('…'), "{summary:?}");
        assert!(!summary.contains("  "), "{summary:?}");
    }

    #[test]
    fn an_unnamed_tool_costs_its_own_entry_and_not_the_server() {
        let reply = json!({"tools": [
            real_read_text_file(),
            {"title": "no name at all"},
            real_write_file(),
        ]});
        let (tools, skipped) = tools_from_list("fs", &reply);
        assert_eq!(tools.len(), 2);
        assert_eq!(skipped.len(), 1);
        assert!(skipped[0].contains("no name"), "{skipped:?}");
    }

    #[test]
    fn a_server_name_cannot_contain_the_separator() {
        let err = ServerSpec::new("a__b", "x").expect_err("that would split wrong");
        assert!(err.to_string().contains("separates"), "{err}");
        assert!(ServerSpec::new("fs", "npx").is_ok());
    }

    #[test]
    fn every_name_sent_to_a_provider_matches_what_a_provider_accepts() {
        // The bug this pins, verbatim from the 400 that found it:
        //   tools.0.custom.name: String should match pattern
        //   '^[a-zA-Z0-9_-]{1,128}$'
        // The name was namespaced with a `/`, which is not in that set, and
        // a provider rejects the entire request over it — so every turn
        // failed for as long as a server was connected, before the model
        // saw anything. Asserted against the pattern itself rather than
        // against the separator, so changing the separator to something
        // else invalid fails here instead of in front of a user.
        let pattern = regex::Regex::new("^[a-zA-Z0-9_-]{1,128}$").unwrap();

        for tool in [real_read_text_file(), real_write_file()] {
            let translated = tool_from_mcp("fs", &tool).unwrap();
            let sent = translated.schema["function"]["name"].as_str().unwrap();
            assert!(pattern.is_match(sent), "{sent:?} would be refused");
            assert_eq!(sent, translated.name, "the gate keys off the same name");
        }

        // And the real server that found it: hyphens are fine, which is
        // what the everything server uses.
        let hyphenated = json!({"name": "get-annotated-message"});
        let translated = tool_from_mcp("ev", &hyphenated).unwrap();
        assert!(pattern.is_match(&translated.name), "{}", translated.name);
        assert_eq!(
            route(&translated.name),
            Some(("ev", "get-annotated-message"))
        );
    }

    #[test]
    fn a_name_a_provider_would_refuse_costs_its_tool_and_not_the_turn() {
        // One unrepresentable name used to fail the whole request, so a
        // single tool with a dot in it would have cost every turn. Skipped
        // and named instead, like an unnamed tool.
        let reply = json!({"tools": [
            real_read_text_file(),
            {"name": "has.a.dot"},
            {"name": "has a space"},
            {"name": "x".repeat(130)},
            real_write_file(),
        ]});
        let (tools, skipped) = tools_from_list("fs", &reply);

        assert_eq!(tools.len(), 2, "the two good ones survive");
        assert_eq!(skipped.len(), 3);
        assert!(
            skipped.iter().all(|s| s.contains("cannot be sent")),
            "{skipped:?}"
        );
    }

    #[test]
    fn a_server_name_a_provider_would_refuse_is_caught_when_it_is_typed() {
        // Rather than at the first turn after it: the server name is the
        // front of every tool name it contributes.
        for bad in ["has space", "has.dot", "has/slash", "héllo"] {
            let err = ServerSpec::new(bad, "x")
                .expect_err("a provider would refuse every tool this server named");
            assert!(
                err.to_string().contains("providers only accept"),
                "{bad:?}: {err}"
            );
        }
        for fine in ["fs", "gj", "my-server", "server_2"] {
            assert!(ServerSpec::new(fine, "x").is_ok(), "{fine:?}");
        }
    }

    #[test]
    fn a_call_result_becomes_text_the_model_can_read() {
        let result = json!({"content": [
            {"type": "text", "text": "one"},
            {"type": "text", "text": "two"},
        ]});
        assert_eq!(call_result_text(&result).unwrap(), "one\ntwo");

        // Content the agent cannot pass on is named rather than dropped: an
        // empty result would read as "the tool did nothing".
        let image = json!({"content": [{"type": "image", "data": "…"}]});
        assert!(call_result_text(&image).unwrap().contains("image content"));

        // `isError` is an error, so a failing MCP tool reads like a failing
        // built-in rather than like a successful call returning a complaint.
        let failed = json!({"isError": true, "content": [{"type": "text", "text": "nope"}]});
        assert_eq!(call_result_text(&failed).unwrap_err().to_string(), "nope");
    }

    #[tokio::test]
    async fn connecting_again_stops_what_was_connected_before() {
        // Not a spawn test — `connect_all` with no specs still has to clear
        // the previous set, because a map left behind would hold servers
        // that are no longer running, and a drop cannot reach the processes
        // they started.
        let _ = connect_all(&[]).await;
        assert!(nothing_is_up(), "the previous set is gone, not replaced");
        // Idempotent: a second pass over an already-empty set is a no-op
        // rather than a second round of signals at a reused pgid.
        let _ = connect_all(&[]).await;
        assert!(nothing_is_up());
    }

    #[tokio::test]
    async fn a_call_to_a_tool_with_no_server_says_which_part_is_missing() {
        // Two different failures that both arrive as "this didn't work".
        // A name with no `/` was never a server's tool, so there is
        // nothing to look up; a name with one belongs to a server that
        // isn't there, which is a different thing to go and fix.
        let unroutable = call("read_file", json!({}))
            .await
            .expect_err("no separator");
        assert!(unroutable.to_string().contains("no `__`"), "{unroutable}");

        let absent = call("nope__anything", json!({}))
            .await
            .expect_err("no such server");
        assert!(
            absent.to_string().contains("no server named nope"),
            "{absent}"
        );
    }

    /// Drives a `Connection` from the other end of a duplex, so the
    /// protocol is tested with no process involved.
    struct FakeServer {
        lines: tokio::io::Lines<BufReader<tokio::io::DuplexStream>>,
        out: tokio::io::DuplexStream,
    }

    impl FakeServer {
        fn start() -> (Connection, Self) {
            // `ours` is what the Connection reads/writes; `theirs` is the
            // server end this struct speaks on.
            let (client_rx, server_tx) = duplex(8192);
            let (server_rx, client_tx) = duplex(8192);
            let connection = Connection::new(client_rx, client_tx);
            (
                connection,
                Self {
                    lines: BufReader::new(server_rx).lines(),
                    out: server_tx,
                },
            )
        }

        async fn next_request(&mut self) -> Value {
            let line = self.lines.next_line().await.unwrap().expect("a request");
            serde_json::from_str(&line).expect("valid JSON-RPC")
        }

        async fn send(&mut self, message: Value) {
            self.out
                .write_all(format!("{message}\n").as_bytes())
                .await
                .unwrap();
        }
    }

    #[tokio::test]
    async fn a_reply_is_matched_to_its_request_by_id() {
        let (connection, mut server) = FakeServer::start();

        let asked = tokio::spawn(async move {
            connection
                .request("tools/list", json!({}), Duration::from_secs(5))
                .await
        });

        let request = server.next_request().await;
        assert_eq!(request["method"], "tools/list");
        assert_eq!(request["jsonrpc"], "2.0");
        let id = request["id"].clone();
        server
            .send(json!({"jsonrpc": "2.0", "id": id, "result": {"tools": []}}))
            .await;

        let result = asked.await.unwrap().unwrap();
        assert_eq!(result["tools"], json!([]));
    }

    #[tokio::test]
    async fn a_notification_in_the_middle_does_not_answer_a_request() {
        // Measured against the everything server, which pushes
        // `notifications/tools/list_changed` unprompted. A client that
        // paired one read to one write would hand this to the caller as a
        // reply, and then read the real reply as an answer to whatever was
        // asked next.
        let (connection, mut server) = FakeServer::start();
        let asked = tokio::spawn(async move {
            connection
                .request("tools/call", json!({}), Duration::from_secs(5))
                .await
        });

        let id = server.next_request().await["id"].clone();
        server
            .send(json!({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}))
            .await;
        server.send(json!({"not even json-rpc": true})).await;
        server
            .send(json!({"jsonrpc": "2.0", "id": 9999, "result": "for nobody"}))
            .await;
        server
            .send(json!({"jsonrpc": "2.0", "id": id, "result": "the real one"}))
            .await;

        assert_eq!(asked.await.unwrap().unwrap(), "the real one");
    }

    #[tokio::test]
    async fn two_calls_in_flight_get_their_own_answers() {
        let (connection, mut server) = FakeServer::start();
        let connection = Arc::new(connection);

        let first = {
            let connection = Arc::clone(&connection);
            tokio::spawn(async move {
                connection
                    .request("a", json!({}), Duration::from_secs(5))
                    .await
            })
        };
        let second = {
            let connection = Arc::clone(&connection);
            tokio::spawn(async move {
                connection
                    .request("b", json!({}), Duration::from_secs(5))
                    .await
            })
        };

        let mut ids = HashMap::new();
        for _ in 0..2 {
            let request = server.next_request().await;
            ids.insert(
                request["method"].as_str().unwrap().to_string(),
                request["id"].clone(),
            );
        }
        // Answered out of order on purpose.
        server
            .send(json!({"jsonrpc": "2.0", "id": ids["b"], "result": "B"}))
            .await;
        server
            .send(json!({"jsonrpc": "2.0", "id": ids["a"], "result": "A"}))
            .await;

        assert_eq!(first.await.unwrap().unwrap(), "A");
        assert_eq!(second.await.unwrap().unwrap(), "B");
    }

    #[tokio::test]
    async fn a_call_that_is_never_answered_gives_up() {
        // The reason the timeout is required rather than defensive: a
        // wedged server does not close its stdout, so there is nothing to
        // notice except time passing.
        let (connection, _server) = FakeServer::start();
        let error = connection
            .request("tools/call", json!({}), Duration::from_millis(50))
            .await
            .expect_err("nothing answered");
        assert!(error.to_string().contains("went unanswered"), "{error}");
    }

    #[tokio::test]
    async fn an_error_reply_is_an_error_rather_than_a_result() {
        let (connection, mut server) = FakeServer::start();
        let asked = tokio::spawn(async move {
            connection
                .request("resources/list", json!({}), Duration::from_secs(5))
                .await
        });
        let id = server.next_request().await["id"].clone();
        server
            .send(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32601, "message": "Method not found"}
            }))
            .await;

        let error = asked.await.unwrap().expect_err("an error reply");
        assert!(error.to_string().contains("Method not found"), "{error}");
        assert!(error.to_string().contains("-32601"), "{error}");
    }

    #[tokio::test]
    async fn a_server_that_closes_fails_the_calls_it_never_answered() {
        // What killing a server mid-call actually looks like from here.
        let (connection, server) = FakeServer::start();
        let asked = tokio::spawn(async move {
            connection
                .request("tools/call", json!({}), Duration::from_secs(5))
                .await
        });
        // Give the request time to be written and registered, then hang up.
        tokio::time::sleep(Duration::from_millis(50)).await;
        drop(server);

        let error = asked.await.unwrap().expect_err("the server hung up");
        assert!(error.to_string().contains("closed its output"), "{error}");
    }

    #[tokio::test]
    async fn the_handshake_sends_a_version_and_then_says_it_is_ready() {
        let (connection, mut server) = FakeServer::start();
        let handshake = tokio::spawn(async move { connection.initialize().await });

        let request = server.next_request().await;
        assert_eq!(request["method"], "initialize");
        assert_eq!(request["params"]["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(
            request["params"]["clientInfo"]["name"],
            "clanker-command-center"
        );
        server
            .send(json!({
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"protocolVersion": PROTOCOL_VERSION, "serverInfo": {"name": "fake"}}
            }))
            .await;

        let result = handshake.await.unwrap().unwrap();
        assert_eq!(result["serverInfo"]["name"], "fake");
        // The notification that follows it, which a server may wait for
        // before it will answer anything else.
        let ready = server.next_request().await;
        assert_eq!(ready["method"], "notifications/initialized");
        assert!(ready.get("id").is_none(), "a notification has no id");
    }

    /// The decision a lazy connection turns on, tested without the
    /// process-wide state the async path around it mutates.
    #[test]
    fn only_a_server_that_is_wanted_up_and_unfailed_is_started() {
        let up: HashSet<String> = ["live".to_string()].into_iter().collect();
        let failed: HashSet<String> = ["broken".to_string()].into_iter().collect();
        let wanted = |name: &str| name != "unwanted";

        assert!(wants_connecting("fresh", &wanted, &up, &failed));
        // Already up: connecting again would spawn a second copy of a
        // server the process already has, which is the whole reason the
        // lazy path checks first.
        assert!(!wants_connecting("live", &wanted, &up, &failed));
        // Failed earlier in this process, so the turn does not pay the
        // connect timeout again — `/mcp reconnect` is how to retry.
        assert!(!wants_connecting("broken", &wanted, &up, &failed));
        // The gates said no, which is what keeps a server set to `never`
        // from ever being spawned.
        assert!(!wants_connecting("unwanted", &wanted, &up, &failed));
    }
}
