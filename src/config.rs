use anyhow::{anyhow, Result};
use keyring::Entry;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

const KEYRING_SERVICE: &str = "clanker-command-center";
const KEYRING_USERNAME: &str = "api_key";

/// The three category gates, as configs and session rows written before
/// tools had states of their own hold them. Read to work out what those
/// meant — see `ToolAccessSettings::from_legacy` — and never written again.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ApprovalSettings {
    #[serde(default = "default_true")]
    pub read_disk: bool,
    #[serde(default = "default_true")]
    pub write_disk: bool,
    #[serde(default = "default_true")]
    pub terminal: bool,
}

fn default_true() -> bool {
    true
}

impl Default for ApprovalSettings {
    fn default() -> Self {
        ApprovalSettings {
            read_disk: true,
            write_disk: true,
            terminal: true,
        }
    }
}

/// What a tool may do without being asked about.
///
/// Three states rather than a yes/no, because "may this run unattended" and
/// "may this run at all" are different questions and only the second one can
/// be answered by not offering the tool. A clanker with everything on
/// `Never` is a plain chat with no tools, which is what "ask mode" used to
/// be — so the mode is not a separate thing to store any more.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolAccess {
    /// Stops and asks before every call.
    Ask,
    /// Runs without asking.
    Allow,
    /// Not offered to the model at all, and refused if it somehow asks.
    Never,
}

impl ToolAccess {
    /// The word used to set it and the word shown when listing it — one
    /// spelling, so what you read back is what you would type.
    pub fn label(&self) -> &'static str {
        match self {
            ToolAccess::Ask => "ask",
            ToolAccess::Allow => "allow",
            ToolAccess::Never => "never",
        }
    }

    pub fn parse(word: &str) -> Option<Self> {
        match word {
            "ask" => Some(ToolAccess::Ask),
            "allow" => Some(ToolAccess::Allow),
            "never" => Some(ToolAccess::Never),
            _ => None,
        }
    }
}

/// What every tool may do, held as only the tools that differ from their
/// default.
///
/// Storing the exceptions rather than the whole set is what lets a tool
/// added later arrive with its own default already in force, in sessions
/// that were created before it existed — no migration, no row that has to be
/// rewritten to learn about it.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolAccessSettings {
    overrides: std::collections::BTreeMap<String, ToolAccess>,
}

/// The key under which "and anything else" is stored.
///
/// `tools off` used to mean *every tool that existed when you typed it*: it
/// wrote an entry per tool, so a tool that turned up later had no entry,
/// fell back to its own default of `ask`, and a clanker deliberately
/// disarmed quietly had a tool again. That was reachable by upgrading the
/// binary and is now reachable by rebuilding an MCP server, so `off` had to
/// become a standing policy rather than a snapshot.
///
/// Stored as a reserved key rather than a new field so every `config.json`
/// and every session row written so far still reads as itself — the struct
/// stays `transparent` over the same map. The key is safe from collision by
/// construction, not by luck: a tool name can only hold
/// `[a-zA-Z0-9_-]` — see [`crate::tools::ToolInfo::new`], which is where a
/// provider's own rule is enforced — so nothing can ever be called `*`.
const EVERYTHING_ELSE: &str = "*";

/// The key under which "and anything else from this server" is stored:
/// `gj__*`, which is also what you can type to name it.
///
/// The same idea as [`EVERYTHING_ELSE`], one level down, and for the same
/// reason. A server is a program somebody rebuilds, so "none of gj's tools"
/// that held only for the tools gj had when you typed it would stop holding
/// the next time it grew one. Safe from collision for the same reason too:
/// no tool name can hold a `*`.
fn server_key(server: &str) -> String {
    format!("{server}{}*", crate::mcp::SEPARATOR)
}

/// The server a tool name belongs to, if it is a namespaced one.
///
/// Read off the name rather than carried beside it: `ServerSpec::new`
/// refuses a server name holding the separator, which is what makes the
/// first `__` an unambiguous place to cut, and no built-in has one.
fn server_of(tool_name: &str) -> Option<&str> {
    crate::mcp::route(tool_name).map(|(server, _)| server)
}

/// Which configured server a `clank tools` target names, and whether it
/// names the whole of it.
///
/// `gj` and `gj__*` are the whole server; `gj__read_file` is one of its
/// tools, which is a different thing to store and has to exist to be
/// validated. `None` for anything that is not a configured server's name —
/// a built-in, a category, `all`, or something unknown, which the gate
/// itself refuses in its own words.
///
/// The order mirrors [`ToolAccessSettings::with_in`]'s: a category or a
/// built-in wins over a server that happens to share its name, so `tools
/// never read` stays about the read tools and `tools never read_file` stays
/// about the built-in.
pub fn server_target(target: &str, servers: &[McpServerConfig]) -> Option<(String, bool)> {
    if target == "all" || crate::tools::CATEGORIES.contains(&target) {
        return None;
    }
    if crate::tools::BUILTIN.iter().any(|tool| tool.name == target) {
        return None;
    }
    for server in servers {
        if target == server.name {
            return Some((server.name.clone(), true));
        }
        let prefix = format!("{}{}", server.name, crate::mcp::SEPARATOR);
        if target == format!("{prefix}*") {
            return Some((server.name.clone(), true));
        }
        // A prefix with nothing after it names no tool of the server, which
        // is the same refusal [`ToolAccessSettings::with_in`] makes.
        if target.len() > prefix.len() && target.starts_with(&prefix) {
            return Some((server.name.clone(), false));
        }
    }
    None
}

/// What a tool does when nothing has been said about it.
///
/// The shell is off. It is the one tool whose blast radius is everything the
/// user can do — every other tool is bounded by what it is *for*, and the
/// sandbox bounds the writes on top of that — so it starts not offered at
/// all rather than merely gated. `clank tools ask run_terminal_command`
/// turns it on for anyone who wants it, per clanker or globally.
///
/// The web is the opposite case: it reads a page and changes nothing, and a
/// prompt per page is the friction that would send the model back to
/// `curl`ing through the shell, which is the call worth being careful about.
/// That used to be a name checked at the top of the gate; as a default it is
/// a row you can see in `clank tools`, and change.
///
/// Everything else asks.
pub fn default_access(tool_name: &str) -> ToolAccess {
    match crate::tools::category_of(tool_name) {
        "web" => ToolAccess::Allow,
        "terminal" => ToolAccess::Never,
        // Including "unknown": a name we do not recognise is the last thing
        // that should run unattended.
        _ => ToolAccess::Ask,
    }
}

impl ToolAccessSettings {
    pub fn access(&self, tool_name: &str) -> ToolAccess {
        self.overrides
            .get(tool_name)
            .copied()
            .unwrap_or_else(|| self.fallback(tool_name))
    }

    /// What a tool with nothing said about it gets: the standing policy if
    /// one is set, otherwise the tool's own default.
    ///
    /// The narrower policy wins: a standing policy for the tool's own
    /// server, then the one for everything, then the default.
    fn fallback(&self, tool_name: &str) -> ToolAccess {
        server_of(tool_name)
            .and_then(|server| self.overrides.get(&server_key(server)))
            .or_else(|| self.overrides.get(EVERYTHING_ELSE))
            .copied()
            .unwrap_or_else(|| default_access(tool_name))
    }

    /// Whether any tool from `server` could still be offered — the one
    /// question a lazy connection can answer without connecting.
    ///
    /// Only the two standing policies can answer it in the negative,
    /// because they are the only things that drop a tool sight unseen:
    /// `tools off` ([`EVERYTHING_ELSE`]) and `tools never <server>` (the
    /// `server__*` key). Anything else — a `never` on one of the server's
    /// tools, a standing `ask` — leaves the rest of them offered, and which
    /// tools those are is exactly what connecting is for. So this is
    /// deliberately optimistic: a server comes up unless a policy covering
    /// all of it says no.
    pub fn server_needed(&self, server: &str) -> bool {
        self.overrides.get(EVERYTHING_ELSE) != Some(&ToolAccess::Never)
            && self.overrides.get(&server_key(server)) != Some(&ToolAccess::Never)
    }

    /// Whether this clanker has any tools at all — the thing that used to be
    /// stored as "agent mode".
    pub fn any_tools(&self) -> bool {
        self.any_tools_in(&crate::tools::tools())
    }

    /// Every tool with its access, in listing order.
    pub fn rows(&self) -> Vec<(Cow<'static, str>, &'static str, ToolAccess)> {
        self.rows_in(&crate::tools::tools())
    }

    /// The two above against a list handed in rather than the live one.
    ///
    /// The live list is process-wide and now writable at runtime — see
    /// `tools::set_registered` — so these exist to keep the rules testable
    /// without a test having to mutate state every other test can see.
    pub fn any_tools_in(&self, tools: &[crate::tools::ToolInfo]) -> bool {
        tools
            .iter()
            .any(|tool| self.access(&tool.name) != ToolAccess::Never)
    }

    /// [`Self::rows`] against a list handed in — see
    /// [`Self::any_tools_in`].
    pub fn rows_in(
        &self,
        tools: &[crate::tools::ToolInfo],
    ) -> Vec<(Cow<'static, str>, &'static str, ToolAccess)> {
        tools
            .iter()
            .map(|tool| (tool.name.clone(), tool.category, self.access(&tool.name)))
            .collect()
    }

    /// A copy with `target` set to `access`. `target` is a tool's name, a
    /// category (`read`/`write`/`terminal`/`web`), `all`, or an MCP server —
    /// by its name or as `server__*`. `None` for a
    /// word that names none of those, so a caller can report the typo rather
    /// than silently changing nothing.
    pub fn with(&self, target: &str, access: ToolAccess) -> Option<Self> {
        self.with_in(&crate::tools::tools(), target, access)
    }

    /// [`Self::with`] against a list handed in rather than the live one —
    /// see [`Self::any_tools_in`].
    pub fn with_in(
        &self,
        tools: &[crate::tools::ToolInfo],
        target: &str,
        access: ToolAccess,
    ) -> Option<Self> {
        // `tools off` is a policy, not a list. It used to write an entry
        // per tool that existed at the time, which made it a snapshot —
        // see `EVERYTHING_ELSE`.
        //
        // Only `never` is stored this way, and the asymmetry is the whole
        // point. "Everything off" holding for a tool that turns up later is
        // the safe direction; "everything allowed" holding for one would
        // wave through a tool nobody has ever seen, which is exactly what
        // `default_access` refuses to do for an unrecognised name. So
        // `allow all` and `ask all` stay claims about the tools in front of
        // you, and a test pins that.
        //
        // A category stays a list for the same reason: "every read tool" is
        // a claim about what you can see.
        if target == "all" && access == ToolAccess::Never {
            let mut updated = self.clone();
            updated.overrides.clear();
            updated
                .overrides
                .insert(EVERYTHING_ELSE.to_string(), access);
            return Some(updated);
        }

        let by_name_or_category: Vec<String> = tools
            .iter()
            .filter(|tool| target == "all" || tool.name == target || tool.category == target)
            .map(|tool| tool.name.to_string())
            .collect();

        // A server is the last thing a target can mean, so a server that
        // happens to be called `read` or `all` is still reachable — as
        // `read__*`, which no tool or category can be.
        if by_name_or_category.is_empty() {
            let server = target
                .strip_suffix(&format!("{}*", crate::mcp::SEPARATOR))
                .unwrap_or(target);
            return self.with_server_in(tools, server, access);
        }

        let mut updated = self.clone();
        if target == "all" {
            // Otherwise a standing `never` would outlive the `allow all`
            // that was meant to lift it, and every tool added afterwards
            // would still arrive off. That goes for a server's as much as
            // for the global one.
            updated.overrides.retain(|key, _| !key.ends_with('*'));
        }
        for name in by_name_or_category {
            updated.set_one(name, access);
        }
        Some(updated)
    }

    /// `self` with every tool of `server` set to `access`, or `None` if no
    /// connected server has that name.
    ///
    /// `never` is a standing policy and the others are a list, for the
    /// reason given in [`Self::with_in`]: a tool the server grows later
    /// should arrive off, and must not arrive unattended.
    ///
    /// `None` also for a server that is configured but did not connect,
    /// because this only sees tools. That is a limit rather than a choice —
    /// a server that is down has nothing to name — and it is why a typo is
    /// still reported instead of quietly stored. A caller that can tell a
    /// configured server from a typo resolves it with [`Self::never_server`]
    /// for the one case that needs no tool list.
    fn with_server_in(
        &self,
        tools: &[crate::tools::ToolInfo],
        server: &str,
        access: ToolAccess,
    ) -> Option<Self> {
        let matched: Vec<String> = tools
            .iter()
            .filter(|tool| server_of(&tool.name) == Some(server))
            .map(|tool| tool.name.to_string())
            .collect();
        if matched.is_empty() {
            return None;
        }
        let mut updated = self.clone();
        // Whatever was said about this server before is replaced rather than
        // layered on: the specific entries are what a more targeted earlier
        // command left, and the one just typed is about the whole server. The
        // prefix is cleared by key rather than by the tools matched, so a
        // tool the server has since dropped does not leave an entry behind.
        let prefix = format!("{server}{}", crate::mcp::SEPARATOR);
        updated.overrides.retain(|key, _| !key.starts_with(&prefix));

        if access == ToolAccess::Never {
            // Redundant under a global `never`, and left out for the same
            // reason a tool set to its default is.
            if updated.overrides.get(EVERYTHING_ELSE) != Some(&ToolAccess::Never) {
                updated.overrides.insert(server_key(server), access);
            }
        } else {
            for name in matched {
                updated.set_one(name, access);
            }
        }
        Some(updated)
    }

    /// `self` with a standing `never` for every tool `server` has or ever
    /// grows — what `tools never <server>` stores when the server is not
    /// connected and its tools cannot be enumerated.
    ///
    /// [`Self::with_server_in`] refuses that case rather than guess,
    /// because from a list of tools it cannot tell a server that is down
    /// from one that was mistyped. The caller here has already checked the
    /// name against the configured servers, which is the difference — and
    /// storing the policy is what lets a server be switched off without
    /// first being started, which is the pair the lazy connection turns on.
    ///
    /// The same one key [`Self::with_server_in`] writes, so a connection
    /// later changes nothing about what it means.
    pub fn never_server(&self, server: &str) -> Self {
        let mut updated = self.clone();
        let prefix = format!("{server}{}", crate::mcp::SEPARATOR);
        updated.overrides.retain(|key, _| !key.starts_with(&prefix));
        if updated.overrides.get(EVERYTHING_ELSE) != Some(&ToolAccess::Never) {
            updated
                .overrides
                .insert(server_key(server), ToolAccess::Never);
        }
        updated
    }

    /// Records `access` for one tool, held only while it differs from what
    /// the tool would get anyway.
    ///
    /// So "set it back to what it would have been" and "never mentioned it"
    /// store the same thing — and a later change of default reaches both.
    /// Compared against the effective fallback rather than the tool's own
    /// default, so setting a tool to what a standing policy already says
    /// stores nothing instead of a redundant row.
    fn set_one(&mut self, name: String, access: ToolAccess) {
        if access == self.fallback(&name) {
            self.overrides.remove(&name);
        } else {
            self.overrides.insert(name, access);
        }
    }

    /// Every tool back to its default: what `tools on` means.
    pub fn defaults() -> Self {
        Self::default()
    }

    /// Every tool off: what `tools off` means, and what a clanker with no
    /// tools is.
    pub fn none() -> Self {
        Self::none_in(&[])
    }

    /// [`Self::none`] over a list handed in, which it no longer needs: the
    /// list is irrelevant now that "every tool off" is one stored policy
    /// rather than an entry per tool. Kept so the call sites that pass a
    /// list read the same as the ones around them.
    pub fn none_in(tools: &[crate::tools::ToolInfo]) -> Self {
        Self::default()
            .with_in(tools, "all", ToolAccess::Never)
            .expect("\"all\" always matches")
    }

    /// What the three old category booleans meant, for a session or a config
    /// written before tools had their own states. `true` was "ask first".
    ///
    /// The terminal is deliberately not among them: the old model had no way
    /// to say a tool is not offered at all, so there is nothing there worth
    /// preserving, and reading `terminal: true` as "ask" would leave anyone
    /// who upgrades with a shell a fresh install does not have. It keeps its
    /// new default, which is `never`.
    pub fn from_legacy(legacy: &ApprovalSettings) -> Self {
        let mut settings = Self::default();
        for (category, asks) in [("read", legacy.read_disk), ("write", legacy.write_disk)] {
            let access = if asks {
                ToolAccess::Ask
            } else {
                ToolAccess::Allow
            };
            settings = settings.with(category, access).unwrap_or(settings);
        }
        settings
    }
}

impl Config {
    /// What each tool may do in a clanker created now.
    ///
    /// Derived from the old category gates when this config predates tools
    /// having states of their own, so an upgrade keeps whatever was
    /// configured rather than silently resetting to the defaults.
    pub fn tool_access(&self) -> ToolAccessSettings {
        self.tools
            .clone()
            .unwrap_or_else(|| ToolAccessSettings::from_legacy(&self.approval))
    }
}

/// How `effort_level` is sent to the provider:
/// - `flat`: top-level `reasoning_effort: "<level>"` (OrcaRouter's shape)
/// - `nested`: `reasoning: { "effort": "<level>" }` (OpenRouter's shape)
/// - `none`: don't send an effort field at all (providers that reject unknown fields)
pub const VALID_EFFORT_STYLES: [&str; 3] = ["flat", "nested", "none"];
pub const DEFAULT_EFFORT_STYLE: &str = "nested";

/// The model a request falls back to when neither a `--model` flag nor the
/// config names one. Still consulted at the point of use as well as seeded
/// into the config, because `clank model --clear` deliberately writes `null`
/// and that has to keep meaning "use this".
pub const DEFAULT_MODEL: &str = "openrouter/auto";

/// The prompt size, in tokens, a clanker's history has to reach before the
/// compactor folds the older part of it into a summary.
///
/// A default rather than an off switch, because compaction is what the
/// setting is for — but a generous one. Most conversations never reach it;
/// the ones that do are the long agentic runs where a single file read sits
/// in every request from then on, which is exactly the spend worth cutting.
/// `clank compact-at --clear` turns automatic compaction off entirely and
/// leaves `/compact` as the only way in.
pub const DEFAULT_COMPACT_AT: u64 = 60_000;

/// The lowest `compact-at` accepted — a fixed guard rail, not the real floor.
///
/// Two things a compaction cannot shrink set the real one. The system prompt
/// and the tool schemas ride every request whatever is folded away, and the
/// three compaction budgets together (`TAIL_FRACTION`, `SUMMARY_FRACTION` and
/// `EXEMPT_FRACTION`: a quarter, an eighth and a sixth) claim roughly half
/// the threshold by construction. A threshold the fixed part plus that half
/// can't fit under means every turn compacts again.
///
/// The fixed part is not a constant. It was about a thousand tokens with only
/// the built-in tools; each connected MCP server adds its tool schemas, so
/// one ordinary server makes it around 3k and several make it more. 4k is
/// what suits a clanker with no servers, and nothing here knows how many are
/// up. With servers connected, raise `compact-at` accordingly — and again
/// when adding more of them. Neither this check nor a hand-edited
/// `config.json` accounts for it; the compaction guard only stops the
/// resulting loop from running on every turn.
pub const MIN_COMPACT_AT: u64 = 4_000;

/// One MCP server to start, as `config.json` holds it.
///
/// `env` is variable *names*, never values. A server is usually configured
/// with an API token, and `config.json` holds no secrets — the one key CCC
/// has lives in the OS keychain, and anything written here would reverse
/// that quietly. The value for each name is looked up at connect time: the
/// keychain first (`clank mcp env`), then this process's own environment,
/// so a token already exported for other tools works without being copied
/// anywhere.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct McpServerConfig {
    /// The prefix its tools get: `name/tool`.
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub env: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    /// Legacy field: API keys used to be stored here in plaintext. Only
    /// populated when reading an old config.json during migration; new
    /// keys are stored in the OS keychain via `get_api_key`/`set_api_key`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default = "default_base_url")]
    pub base_url: String,
    #[serde(default = "default_model")]
    pub default_model: Option<String>,
    /// The model that compacts a clanker's history — see [`crate::compact`].
    /// Deliberately its own setting rather than the clanker's own model: the
    /// job is summarizing a transcript, which a small cheap model does well,
    /// and paying reasoning-model rates to save tokens would defeat the
    /// point. `None` falls back to [`DEFAULT_MODEL`], the same way
    /// `default_model` does.
    ///
    /// Global only for now. A per-clanker override belongs here eventually,
    /// alongside the model and temperature ones, but a clanker has to be
    /// able to say "the configured one" before that means anything.
    #[serde(default = "default_compactor")]
    pub compactor: Option<String>,
    /// How large a request's prompt has to get before the next turn compacts
    /// first, in tokens as the provider reported them. `None` means never
    /// automatically — `/compact` still works, and is then the only thing
    /// that compacts.
    #[serde(default = "default_compact_at")]
    pub compact_at: Option<u64>,
    /// Whether requests carry prompt-caching breakpoints, so a provider
    /// reuses its work on the part of the history that hasn't changed
    /// instead of reprocessing it. `None` falls back to [`DEFAULT_CACHE`].
    ///
    /// Worth a switch because it changes the shape of what goes on the wire:
    /// a marked message's `content` becomes a block array instead of a
    /// string, and while that is the documented form for OpenRouter and the
    /// providers behind it, `base_url` can point anywhere. Turning this off
    /// restores the byte-for-byte request CCC sent before caching existed.
    #[serde(default = "default_cache")]
    pub cache: Option<bool>,
    /// The MCP servers to start, and whose tools to offer — see
    /// [`McpServerConfig`]. Empty by default: a tool that can do anything
    /// the server's author wanted is not something to acquire by upgrading.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mcp_servers: Vec<McpServerConfig>,
    /// Legacy: the three category gates, as configs written before tools had
    /// their own states hold them. Read so those keep meaning what they
    /// meant, never written again — it disappears from the file the next
    /// time anything saves. `tool_access()` is what to read.
    #[serde(default, skip_serializing)]
    pub approval: ApprovalSettings,
    /// What each tool may do by default in a new clanker. `None` in a config
    /// written before this existed, which then derives from `approval`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolAccessSettings>,
    /// `None` means no persistent default is configured at all — `ask`/
    /// `agent`/a new `session` then run with no iteration cap unless
    /// `--max-iterations` is passed for that call, which errors immediately
    /// with tools rather than guessing a number.
    #[serde(default = "default_max_iterations")]
    pub max_iterations: Option<usize>,
    /// `None` means no persistent default is configured at all — a request
    /// is then sent with no `temperature` field, and the provider uses its
    /// own default.
    #[serde(default = "default_temperature")]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub effort_level: Option<String>,
    /// How to serialize `effort_level` for the current `base_url`'s provider.
    /// `None` falls back to `DEFAULT_EFFORT_STYLE` ("nested").
    #[serde(default = "default_effort_style")]
    pub effort_style: Option<String>,
    /// Extra HTTP headers sent with every API request, for providers that
    /// need something beyond `Authorization: Bearer <key>` (e.g. OpenRouter's
    /// optional `HTTP-Referer`/`X-Title` attribution headers).
    #[serde(default)]
    pub extra_headers: HashMap<String, String>,
    /// Whether new sessions start showing full tool-call detail. Off by
    /// default; `/verbose` toggles it for the session you're in, and that
    /// choice is remembered per session rather than changing this.
    #[serde(default)]
    pub verbose: bool,
    /// Whether a session shows a band behind your own messages. A display
    /// preference rather than a behaviour, so it changes nothing a turn
    /// does — but it is per-session like `verbose`, because a session you
    /// read back through and one you are working in want different amounts
    /// of decoration.
    #[serde(default = "default_true")]
    pub highlight: bool,
    /// Whether the launch screen bands its selected row. Global only: the
    /// launch screen belongs to no session.
    #[serde(default = "default_true")]
    pub selection: bool,
    /// Whether the agent's file writes are confined to the working
    /// directory. On by default; turning it off lets its write tools touch
    /// any path the process can. Reads are never bounded either way — they
    /// mutate nothing, and confining them would break ordinary work like
    /// reading a file under `/etc`.
    ///
    /// This gates the agent's tools only. The app's own state —
    /// `config.json`, `chats.db`, `errors.log` — is written directly and is
    /// unaffected at any setting.
    #[serde(default = "default_true")]
    pub sandbox: bool,
    /// Whether to stream responses token-by-token. On by default; turn it off
    /// for providers that handle streaming (especially streaming alongside
    /// tool calls) badly, which falls back to waiting for the whole reply.
    #[serde(default = "default_true")]
    pub stream: bool,

    /// How long connecting (DNS/TCP/TLS) may take before giving up —
    /// independent of how long a slow-to-answer provider may then take once
    /// connected, which the two below cover instead.
    #[serde(default = "default_connect_timeout")]
    pub connect_timeout: u64,

    /// The ceiling on a whole non-streaming round trip. It has no partial
    /// progress to show, so it gets one generous bound: long enough for a
    /// slow reasoning model, short enough that a stalled connection
    /// eventually surfaces as an error instead of waiting forever.
    #[serde(default = "default_request_timeout")]
    pub request_timeout: u64,

    /// The gap allowed *between* chunks of a streaming reply, which has no
    /// meaningful total ceiling — a long answer legitimately keeps sending.
    /// No new bytes within this window means the connection stalled, not
    /// that the model is still thinking.
    ///
    /// The one most worth changing: 90s has cut real turns short more than
    /// once behind a slow provider.
    #[serde(default = "default_stream_idle_timeout")]
    pub stream_idle_timeout: u64,

    /// How long a terminal command the agent runs may take, when the model
    /// does not name a timeout of its own in the call.
    #[serde(default = "default_command_timeout")]
    pub command_timeout: u64,
}

pub fn default_connect_timeout() -> u64 {
    20
}

pub fn default_request_timeout() -> u64 {
    300
}

pub fn default_stream_idle_timeout() -> u64 {
    90
}

pub fn default_command_timeout() -> u64 {
    30
}

pub fn default_base_url() -> String {
    "https://openrouter.ai/api/v1".to_string()
}

/// The model used when nothing else names one. A seed rather than a bare
/// `None`, so a config written from defaults says which model it will
/// actually use instead of leaving `null` next to a literal buried in
/// `resolve_model`.
pub fn default_model() -> Option<String> {
    Some(DEFAULT_MODEL.to_string())
}

/// Same deal as [`default_model`]: the compactor named in a config written
/// from defaults, so the file says which model will do the summarizing
/// rather than leaving `null` beside a literal.
pub fn default_compactor() -> Option<String> {
    Some(DEFAULT_MODEL.to_string())
}

/// The threshold seeded into a new config, and into one written before this
/// existed. `Some` rather than `None`, so compaction is on out of the box —
/// see [`DEFAULT_COMPACT_AT`] for why the number is as high as it is. Once
/// cleared it stays cleared: this is never consulted again after that.
pub fn default_compact_at() -> Option<u64> {
    Some(DEFAULT_COMPACT_AT)
}

/// Whether prompt caching is on when the config doesn't say.
///
/// On. Every request a clanker makes carries the whole history it has
/// already sent, so the second request in any conversation is where caching
/// starts paying — and a provider that stores a prefix charges a premium
/// once to store it and a fraction of the rate to read it back, which breaks
/// even at two requests. A clanker that sends one request and stops is the
/// only case this costs anything, and it costs the write premium on one
/// prefix.
pub const DEFAULT_CACHE: bool = true;

/// The switch seeded into a new config, so the file says caching is on
/// rather than leaving `null` beside a literal.
pub fn default_cache() -> Option<bool> {
    Some(DEFAULT_CACHE)
}

/// Same deal: the shape effort is serialized in when the config doesn't say.
/// See [`DEFAULT_EFFORT_STYLE`].
pub fn default_effort_style() -> Option<String> {
    Some(DEFAULT_EFFORT_STYLE.to_string())
}

/// The factory default for a fresh install (no `config.json` yet) and for
/// migrating an old `config.json` written before this field existed. Once a
/// user explicitly clears it with `clank max-iterations --clear`, it stays
/// `None` — this is never consulted again after that.
pub fn default_max_iterations() -> Option<usize> {
    Some(20)
}

/// Same deal as [`default_max_iterations`].
pub fn default_temperature() -> Option<f32> {
    Some(0.7)
}

impl Default for Config {
    fn default() -> Self {
        Config {
            api_key: None,
            base_url: default_base_url(),
            default_model: default_model(),
            compactor: default_compactor(),
            compact_at: default_compact_at(),
            mcp_servers: Vec::new(),
            cache: default_cache(),
            approval: ApprovalSettings::default(),
            // Explicitly the defaults, not `None`: `None` means "this config
            // predates tools having states" and derives from the three old
            // booleans, which have no way to say the shell is off. A config
            // seeded here is a new one, and a new one starts with
            // `run_terminal_command` never offered.
            tools: Some(ToolAccessSettings::default()),
            max_iterations: default_max_iterations(),
            temperature: default_temperature(),
            sandbox: true,
            verbose: false,
            highlight: true,
            selection: true,
            effort_level: None,
            effort_style: default_effort_style(),
            extra_headers: HashMap::new(),
            stream: true,
            connect_timeout: default_connect_timeout(),
            request_timeout: default_request_timeout(),
            stream_idle_timeout: default_stream_idle_timeout(),
            command_timeout: default_command_timeout(),
        }
    }
}

pub fn get_config_dir() -> Result<PathBuf> {
    let config_dir = home::home_dir()
        .ok_or(anyhow!("Could not determine home directory"))?
        .join(".clank");

    fs::create_dir_all(&config_dir)?;
    Ok(config_dir)
}

pub fn get_config_path() -> Result<PathBuf> {
    Ok(get_config_dir()?.join("config.json"))
}

/// Parses `config.json`, naming the file and the position when it can't be.
///
/// Split from [`load_config`] so it's testable without moving `HOME` around,
/// and separate from the file-missing path, which is not an error: an absent
/// config means "use the defaults", a malformed one means "this says
/// something I can't read".
fn parse_config(content: &str, path: &Path) -> Result<Config> {
    serde_json::from_str(content).map_err(|e| {
        anyhow!(
            "Could not parse {}: {e}\n\n\
             Fix the file, or delete it to start from defaults.",
            path.display()
        )
    })
}

pub fn load_config() -> Result<Config> {
    let config_path = get_config_path()?;

    let mut config = if config_path.exists() {
        let content = fs::read_to_string(&config_path)?;
        // Refused rather than defaulted. Carrying on would mean sending the
        // API key to whatever `base_url` defaults to instead of the provider
        // that was configured — and worse, the next setting command would
        // save defaults-plus-one-change over the file, destroying everything
        // else in it. Nothing is written here: the file stays exactly as it
        // was typed so it can be fixed.
        parse_config(&content, &config_path)?
    } else {
        Config::default()
    };

    // Migrate a plaintext key from an older config.json into the OS keychain.
    if let Some(legacy_key) = config.api_key.take() {
        set_api_key(&legacy_key)?;
        save_config(&config)?;
    }

    Ok(config)
}

pub fn save_config(config: &Config) -> Result<()> {
    let json = serde_json::to_string_pretty(config)?;
    write_atomic(&get_config_path()?, &json)
}

/// Writes beside the target and renames over it, so an interrupted write
/// leaves the old file intact rather than a truncated one. The temp name
/// includes the PID so concurrent processes don't clobber each other's.
///
/// `config.json` is the one file the app refuses to start without parsing —
/// a half-written one turns "fix the file, or delete it to start from
/// defaults" from advice about a typo into the recovery path for a bug this
/// function exists to prevent. So the rename is not the whole of it:
///
/// The contents are **fsynced before** the rename. A rename is atomic in
/// ordering, not in durability, and without the sync a power cut can commit
/// the rename while the temp file's blocks are still in the page cache —
/// which lands a zero-length `config.json`, the exact outcome being
/// defended against. ext4's heuristics usually cover this; usually is not
/// the guarantee worth having here.
///
/// The directory is deliberately *not* synced afterwards. That would make
/// the rename itself durable, and losing it leaves the previous config in
/// place — the last setting change is gone, nothing is corrupt — so it buys
/// a safe outcome at the price of a platform split, since a directory
/// cannot be opened as a file on Windows.
///
/// The target's permissions are carried onto the replacement. A rename
/// swaps the inode, so without this a `chmod 600 config.json` is silently
/// undone by the next setting change — no secrets live in the file today
/// (see [`Config::api_key`]), but reverting a mode the user chose is not
/// this function's call to make.
fn write_atomic(path: &Path, contents: &str) -> Result<()> {
    let mut tmp_name = path.as_os_str().to_owned();
    tmp_name.push(format!(".{}.tmp", std::process::id()));
    let tmp_path = PathBuf::from(tmp_name);

    // Every failure past this point leaves a temp file behind, so they all
    // land in one place and clean up before returning.
    let staged = (|| -> Result<()> {
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(contents.as_bytes())?;
        // Before the sync, so the mode is part of what gets flushed. A
        // missing target is the first-ever save, which keeps the default.
        if let Ok(existing) = fs::metadata(path) {
            fs::set_permissions(&tmp_path, existing.permissions())?;
        }
        file.sync_all()?;
        Ok(())
    })();

    if let Err(e) = staged.and_then(|()| fs::rename(&tmp_path, path).map_err(Into::into)) {
        let _ = fs::remove_file(&tmp_path);
        return Err(e);
    }
    Ok(())
}

fn keyring_entry() -> Result<Entry> {
    Ok(Entry::new(KEYRING_SERVICE, KEYRING_USERNAME)?)
}

/// Reads the API key from the OS keychain (macOS Keychain, Windows
/// Credential Manager, or the Linux Secret Service). Returns `Ok(None)`
/// if no key has been stored yet.
/// The keychain entry holding one server's one environment value.
///
/// Namespaced by server so two servers can both want `API_TOKEN`, and
/// prefixed so none of them can collide with [`KEYRING_USERNAME`].
fn mcp_env_entry(server: &str, variable: &str) -> Result<Entry> {
    Ok(Entry::new(
        KEYRING_SERVICE,
        &format!("mcp:{server}:{variable}"),
    )?)
}

/// The stored value for one of a server's environment variables.
pub fn get_mcp_env(server: &str, variable: &str) -> Result<Option<String>> {
    match mcp_env_entry(server, variable)?.get_password() {
        Ok(value) => Ok(Some(value)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

pub fn set_mcp_env(server: &str, variable: &str, value: &str) -> Result<()> {
    mcp_env_entry(server, variable)?.set_password(value)?;
    Ok(())
}

pub fn clear_mcp_env(server: &str, variable: &str) -> Result<()> {
    match mcp_env_entry(server, variable)?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// What to start a server with, with every environment value resolved.
///
/// A name with no value anywhere is left out rather than passed as empty:
/// a server that checks whether its token is set should find it missing,
/// not find it blank. What is missing is named, so `clank mcp` can say so
/// instead of the server failing obscurely.
pub fn resolve_server(config: &McpServerConfig) -> Result<(crate::mcp::ServerSpec, Vec<String>)> {
    let mut spec =
        crate::mcp::ServerSpec::new(&config.name, &config.command)?.with_args(&config.args);
    let mut missing = Vec::new();
    for variable in &config.env {
        let value = get_mcp_env(&config.name, variable)
            .ok()
            .flatten()
            .or_else(|| std::env::var(variable).ok());
        match value {
            Some(value) => spec.env.push((variable.clone(), value)),
            None => missing.push(variable.clone()),
        }
    }
    Ok((spec, missing))
}

pub fn get_api_key() -> Result<Option<String>> {
    match keyring_entry()?.get_password() {
        Ok(key) => Ok(Some(key)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(anyhow!("Failed to read API key from OS keychain: {e}")),
    }
}

/// Stores the API key in the OS keychain.
pub fn set_api_key(key: &str) -> Result<()> {
    keyring_entry()?
        .set_password(key)
        .map_err(|e| anyhow!("Failed to save API key to OS keychain: {e}"))
}

/// Removes the API key from the OS keychain, if present.
pub fn clear_api_key() -> Result<()> {
    match keyring_entry()?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(anyhow!("Failed to remove API key from OS keychain: {e}")),
    }
}

/// A live view of a session's safety controls, rather than a copy of them.
///
/// The agent loop runs on its own task, so it used to be handed a snapshot
/// taken when the turn was spawned — which meant a `/tools allow write`
/// typed while a turn was running had no effect until the *next* turn, even
/// though the settings row updated immediately and said otherwise. Sharing
/// the settings instead lets each tool call read what they say right now,
/// which is what someone flipping a gate mid-turn is asking for.
///
/// Both controls live here for the same reason: they decide what a tool is
/// allowed to do, so a turn in progress is exactly when a change to one
/// matters most. Settings that only shape the *next* request — model,
/// effort, temperature — are deliberately not here, and still apply from the
/// next turn.
///
/// Cheap to clone: every clone reads and writes the same state.
#[derive(Clone, Debug, Default)]
pub struct SessionGates {
    access: Arc<Mutex<ToolAccessSettings>>,
    sandbox: Arc<AtomicBool>,
    /// The fallback timeout for a terminal command, for calls where the
    /// model names none. Fixed for the run, unlike the two above, which
    /// `/tools` and `/sandbox` can change partway through a turn.
    command_timeout: u64,
    /// The files this session has read, which an overwrite or a replace
    /// needs first. Kept with the gates because it outlives a turn, as they
    /// do, and decides what a write tool may do, as they do.
    reads: crate::tools::FileReads,
}

impl SessionGates {
    pub fn new(access: ToolAccessSettings, sandbox: bool, command_timeout: u64) -> Self {
        Self {
            access: Arc::new(Mutex::new(access)),
            sandbox: Arc::new(AtomicBool::new(sandbox)),
            command_timeout,
            reads: crate::tools::FileReads::default(),
        }
    }

    /// The same gates sharing an existing record of reads, for a caller that
    /// builds them afresh each turn but runs one session across them.
    pub fn with_reads(mut self, reads: crate::tools::FileReads) -> Self {
        self.reads = reads;
        self
    }

    /// What this session has read, shared with every clone.
    pub fn reads(&self) -> &crate::tools::FileReads {
        &self.reads
    }

    /// How long a terminal command may run when the call does not say.
    pub fn command_timeout(&self) -> u64 {
        self.command_timeout
    }

    /// What each tool may do as things stand. Cloned out rather than handing
    /// back a guard, so a caller can't hold the lock across an await.
    pub fn access(&self) -> ToolAccessSettings {
        self.lock().clone()
    }

    pub fn set_access(&self, access: ToolAccessSettings) {
        *self.lock() = access;
    }

    /// Whether the agent's file writes are confined to the working directory.
    pub fn sandbox(&self) -> bool {
        self.sandbox.load(Ordering::Relaxed)
    }

    pub fn set_sandbox(&self, sandbox: bool) {
        self.sandbox.store(sandbox, Ordering::Relaxed);
    }

    /// A poisoned lock still holds perfectly good settings — the panic that
    /// poisoned it happened elsewhere — and refusing to read them would turn
    /// an unrelated panic into a dead gate.
    fn lock(&self) -> MutexGuard<'_, ToolAccessSettings> {
        self.access.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory of this test module's own, named so two runs and two
    /// test binaries can't collide in it.
    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "clank-config-test-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn write_atomic_replaces_and_leaves_no_temp() {
        let dir = scratch_dir("atomic");
        let path = dir.join("config.json");
        write_atomic(&path, "old").unwrap();
        write_atomic(&path, "new").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "new");
        assert_eq!(
            fs::read_dir(&dir).unwrap().count(),
            1,
            "the temp file has to be gone, not merely unused"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn write_atomic_keeps_the_mode_the_file_already_had() {
        // A rename swaps the inode, so the replacement is born with the
        // umask default unless the old mode is carried over. Tightening
        // config.json and then changing a setting must not loosen it again.
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("mode");
        let path = dir.join("config.json");
        write_atomic(&path, "first").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();

        write_atomic(&path, "second").unwrap();

        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "mode was reset to {mode:o}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn tools_off_stays_off_when_a_tool_turns_up_later() {
        // The bug. `off` used to write an entry per tool that existed at
        // the time, so a tool added afterwards — by upgrading the binary, or
        // now by rebuilding an MCP server — had no entry, fell back to its
        // own default of `ask`, and a clanker deliberately disarmed had a
        // tool again.
        let off = ToolAccessSettings::none();
        assert_eq!(off.access("gj__delete_everything"), ToolAccess::Never);
        assert_eq!(
            off.access("a_built_in_added_next_version"),
            ToolAccess::Never
        );
        assert_eq!(off.access("read_file"), ToolAccess::Never);
    }

    #[test]
    fn allow_all_does_not_wave_through_a_tool_nobody_has_seen() {
        // The asymmetry, and the reason only `never` is stored as a
        // standing policy. "Everything off" holding for a future tool is
        // safe; "everything allowed" holding for one would hand `allow` to
        // a tool that did not exist when it was typed.
        let allowed = ToolAccessSettings::default()
            .with("all", ToolAccess::Allow)
            .expect("\"all\" always matches");
        assert_eq!(allowed.access("read_file"), ToolAccess::Allow);
        assert_eq!(allowed.access("some_future_tool"), ToolAccess::Ask);

        let asked = ToolAccessSettings::default()
            .with("all", ToolAccess::Ask)
            .expect("\"all\" always matches");
        assert_eq!(asked.access("some_future_tool"), ToolAccess::Ask);
    }

    #[test]
    fn allow_all_lifts_a_standing_never() {
        // Otherwise `off` would outlive the `on` meant to undo it, and
        // every tool added afterwards would still arrive off.
        let back_on = ToolAccessSettings::none()
            .with("all", ToolAccess::Allow)
            .expect("\"all\" always matches");
        assert_eq!(back_on.access("some_future_tool"), ToolAccess::Ask);
        assert_eq!(back_on.access("read_file"), ToolAccess::Allow);

        // And `tools on` is the defaults, which is not a policy at all.
        let defaults = ToolAccessSettings::defaults();
        assert_eq!(defaults.access("run_terminal_command"), ToolAccess::Never);
        assert_eq!(defaults.access("web_fetch"), ToolAccess::Allow);
        assert_eq!(defaults.access("some_future_tool"), ToolAccess::Ask);
    }

    #[test]
    fn one_tool_can_be_switched_back_on_under_a_standing_never() {
        let off_except_one = ToolAccessSettings::none()
            .with("read_file", ToolAccess::Allow)
            .expect("a tool by its own name");
        assert_eq!(off_except_one.access("read_file"), ToolAccess::Allow);
        assert_eq!(off_except_one.access("list_files"), ToolAccess::Never);
        assert_eq!(off_except_one.access("anything_new"), ToolAccess::Never);
    }

    /// Two servers and the built-ins, as `tools::merge` would build them
    /// without writing to the process-wide registry.
    fn two_servers() -> Vec<crate::tools::ToolInfo> {
        let tool = |name: &str, category: &str| {
            crate::tools::ToolInfo::new(
                name,
                category,
                "x",
                serde_json::json!({
                    "type": "function",
                    "function": {"name": name, "parameters": {"type": "object"}}
                }),
            )
            .unwrap()
        };
        crate::tools::merge(&[
            tool("gj__find", "read"),
            tool("gj__delete", "write"),
            tool("fs__read_text_file", "read"),
            tool("fs__write_file", "write"),
        ])
    }

    #[test]
    fn a_server_can_be_switched_off_by_name_or_as_a_glob() {
        let all = two_servers();
        for target in ["gj", "gj__*"] {
            let off = ToolAccessSettings::default()
                .with_in(&all, target, ToolAccess::Never)
                .unwrap_or_else(|| panic!("{target} names a connected server"));
            assert_eq!(off.access("gj__find"), ToolAccess::Never, "{target}");
            assert_eq!(off.access("gj__delete"), ToolAccess::Never, "{target}");
            // Only that server: not the other one, and not the built-ins —
            // which `never write` could not manage, since it takes
            // `write_file` along with every server's writers.
            assert_eq!(off.access("fs__write_file"), ToolAccess::Ask, "{target}");
            assert_eq!(off.access("write_file"), ToolAccess::Ask, "{target}");
        }
    }

    #[test]
    fn a_server_switched_off_stays_off_when_it_grows_a_tool() {
        // The reason it is a standing policy: a server is a program that
        // gets rebuilt, and "none of gj" has to survive that.
        let off = ToolAccessSettings::default()
            .with_in(&two_servers(), "gj", ToolAccess::Never)
            .unwrap();
        assert_eq!(off.access("gj__added_in_the_next_build"), ToolAccess::Never);
        assert_eq!(off.access("fs__added_in_the_next_build"), ToolAccess::Ask);
        // And it is one entry, not one per tool.
        assert_eq!(serde_json::to_string(&off).unwrap(), r#"{"gj__*":"never"}"#);
    }

    #[test]
    fn a_server_allowed_is_a_claim_about_the_tools_in_front_of_you() {
        // The asymmetry again: only `never` holds for a tool nobody has
        // seen. `allow gj` must not wave through whatever gj grows next.
        let all = two_servers();
        let allowed = ToolAccessSettings::default()
            .with_in(&all, "gj", ToolAccess::Allow)
            .unwrap();
        assert_eq!(allowed.access("gj__find"), ToolAccess::Allow);
        assert_eq!(allowed.access("gj__delete"), ToolAccess::Allow);
        assert_eq!(
            allowed.access("gj__added_in_the_next_build"),
            ToolAccess::Ask
        );
        assert_eq!(allowed.access("fs__read_text_file"), ToolAccess::Ask);
    }

    #[test]
    fn one_tool_can_be_switched_back_on_under_a_server_never() {
        let all = two_servers();
        let settings = ToolAccessSettings::default()
            .with_in(&all, "gj", ToolAccess::Never)
            .unwrap()
            .with_in(&all, "gj__find", ToolAccess::Allow)
            .unwrap();
        assert_eq!(settings.access("gj__find"), ToolAccess::Allow);
        assert_eq!(settings.access("gj__delete"), ToolAccess::Never);
        assert_eq!(settings.access("gj__new"), ToolAccess::Never);
    }

    #[test]
    fn a_server_command_replaces_what_was_said_about_that_server_before() {
        let all = two_servers();
        let settings = ToolAccessSettings::default()
            .with_in(&all, "gj__find", ToolAccess::Allow)
            .unwrap()
            .with_in(&all, "fs__write_file", ToolAccess::Allow)
            .unwrap()
            .with_in(&all, "gj", ToolAccess::Never)
            .unwrap();
        // The earlier, narrower `allow` on gj is gone rather than punching
        // a hole in "all of gj off" — and the other server's is untouched.
        assert_eq!(settings.access("gj__find"), ToolAccess::Never);
        assert_eq!(settings.access("fs__write_file"), ToolAccess::Allow);

        // Lifting it: back to the ordinary default, standing entry gone.
        let lifted = settings.with_in(&all, "gj", ToolAccess::Ask).unwrap();
        assert_eq!(lifted.access("gj__find"), ToolAccess::Ask);
        assert_eq!(lifted.access("gj__new"), ToolAccess::Ask);
        assert_eq!(lifted.access("fs__write_file"), ToolAccess::Allow);
    }

    #[test]
    fn allow_all_lifts_a_server_never_too() {
        let all = two_servers();
        let off = ToolAccessSettings::default()
            .with_in(&all, "gj", ToolAccess::Never)
            .unwrap();
        let back = off.with_in(&all, "all", ToolAccess::Allow).unwrap();
        assert_eq!(back.access("gj__find"), ToolAccess::Allow);
        assert_eq!(back.access("gj__new"), ToolAccess::Ask);

        // And `off` swallows it: one policy, nothing left over.
        let none = off.with_in(&all, "all", ToolAccess::Never).unwrap();
        assert_eq!(serde_json::to_string(&none).unwrap(), r#"{"*":"never"}"#);
    }

    #[test]
    fn a_server_never_under_a_global_never_stores_nothing_extra() {
        let all = two_servers();
        let settings = ToolAccessSettings::none()
            .with_in(&all, "gj", ToolAccess::Never)
            .unwrap();
        assert_eq!(
            serde_json::to_string(&settings).unwrap(),
            r#"{"*":"never"}"#
        );
    }

    /// A configured server, for the two things that have to read a name
    /// without a tool list: [`server_target`] and [`never_server`].
    fn configured(name: &str) -> McpServerConfig {
        McpServerConfig {
            name: name.to_string(),
            command: "run-it".to_string(),
            args: Vec::new(),
            env: Vec::new(),
        }
    }

    #[test]
    fn a_whole_server_is_recognised_from_the_config_alone() {
        // What a lazy connection needs: the name of a server nothing is
        // connected for, told apart from a typo and from a tool of it.
        let servers = [configured("gj"), configured("fs")];
        assert_eq!(server_target("gj", &servers), Some(("gj".into(), true)));
        assert_eq!(server_target("gj__*", &servers), Some(("gj".into(), true)));
        assert_eq!(
            server_target("gj__find", &servers),
            Some(("gj".into(), false))
        );
        // Everything that is not a configured server, in the order the gate
        // itself resolves them: a category and a built-in win over a server
        // that happens to share the name.
        for target in ["all", "read", "write_file", "nonesuch", "gj__", ""] {
            assert_eq!(server_target(target, &servers), None, "{target:?}");
        }
    }

    #[test]
    fn a_server_can_be_switched_off_before_it_has_ever_connected() {
        // The pair the lazy connection turns on: `never` for a whole server
        // is one standing key, so it needs no tool list — and a server set
        // to `never` is then never started at all.
        let off = ToolAccessSettings::default().never_server("gj");
        assert_eq!(off.access("gj__find"), ToolAccess::Never);
        assert_eq!(off.access("gj__grown_a_tool_since"), ToolAccess::Never);
        assert_eq!(off.access("fs__read_text_file"), ToolAccess::Ask);
        // The same one key a connected server would have written, so the
        // connection later changes nothing about what it means.
        assert_eq!(serde_json::to_string(&off).unwrap(), r#"{"gj__*":"never"}"#);
        assert_eq!(
            serde_json::to_string(&off).unwrap(),
            serde_json::to_string(
                &ToolAccessSettings::default()
                    .with_in(&two_servers(), "gj", ToolAccess::Never)
                    .unwrap()
            )
            .unwrap()
        );
        // Redundant under a global `never`, and left out for the same
        // reason a tool set to its default is.
        let none = ToolAccessSettings::none().never_server("gj");
        assert_eq!(serde_json::to_string(&none).unwrap(), r#"{"*":"never"}"#);
    }

    #[test]
    fn a_server_is_needed_unless_a_policy_covers_all_of_it() {
        // The question a lazy connection answers without connecting. Only
        // the two standing policies can answer it in the negative, because
        // they are the only things that drop a tool sight unseen — so a
        // `never` on one of a server's tools still leaves the server worth
        // starting, which is what makes this optimistic rather than exact.
        assert!(ToolAccessSettings::default().server_needed("gj"));
        assert!(ToolAccessSettings::default()
            .with_in(&two_servers(), "gj__find", ToolAccess::Never)
            .unwrap()
            .server_needed("gj"));
        assert!(ToolAccessSettings::default()
            .with_in(&two_servers(), "gj", ToolAccess::Ask)
            .unwrap()
            .server_needed("gj"));

        assert!(!ToolAccessSettings::default()
            .never_server("gj")
            .server_needed("gj"));
        // Only the server named: switching one off is not switching them
        // all off.
        assert!(ToolAccessSettings::default()
            .never_server("gj")
            .server_needed("fs"));
        assert!(!ToolAccessSettings::none().server_needed("gj"));
        assert!(!ToolAccessSettings::none().server_needed("fs"));
    }

    #[test]
    fn a_name_that_is_a_tool_or_category_is_not_read_as_a_server() {
        // A server called `read` is reachable, but only spelled out — the
        // bare word keeps meaning the category it always meant.
        let mut all = two_servers();
        all.push(
            crate::tools::ToolInfo::new(
                "read__thing",
                "write",
                "x",
                serde_json::json!({
                    "type": "function",
                    "function": {"name": "read__thing", "parameters": {"type": "object"}}
                }),
            )
            .unwrap(),
        );
        let by_category = ToolAccessSettings::default()
            .with_in(&all, "read", ToolAccess::Never)
            .unwrap();
        assert_eq!(by_category.access("read_file"), ToolAccess::Never);
        assert_eq!(by_category.access("read__thing"), ToolAccess::Ask);

        let by_server = ToolAccessSettings::default()
            .with_in(&all, "read__*", ToolAccess::Never)
            .unwrap();
        assert_eq!(by_server.access("read__thing"), ToolAccess::Never);
        assert_eq!(by_server.access("read_file"), ToolAccess::Ask);
    }

    #[test]
    fn a_server_that_is_not_there_is_reported_not_stored() {
        let all = two_servers();
        for target in ["nonesuch", "nonesuch__*", "gj__", "__*", ""] {
            assert!(
                ToolAccessSettings::default()
                    .with_in(&all, target, ToolAccess::Never)
                    .is_none(),
                "{target:?}"
            );
        }
    }

    #[test]
    fn the_standing_policy_survives_the_config_file() {
        // Stored as a reserved key rather than a new field, so the struct
        // stays `transparent` and every config written so far still reads
        // as itself. A tool can never be called `*` — the name charset in
        // `ToolInfo::new` is what guarantees that — so it cannot collide.
        let off = ToolAccessSettings::none();
        let written = serde_json::to_string(&off).unwrap();
        assert_eq!(written, r#"{"*":"never"}"#);
        let read: ToolAccessSettings = serde_json::from_str(&written).unwrap();
        assert_eq!(read.access("anything_new"), ToolAccess::Never);

        // An older row, written as a bare enumeration, still means exactly
        // what it meant — including that it says nothing about a new tool.
        let legacy: ToolAccessSettings =
            serde_json::from_str(r#"{"read_file":"allow","write_file":"never"}"#).unwrap();
        assert_eq!(legacy.access("read_file"), ToolAccess::Allow);
        assert_eq!(legacy.access("write_file"), ToolAccess::Never);
        assert_eq!(legacy.access("list_files"), ToolAccess::Ask);
    }

    #[test]
    fn a_config_written_before_mcp_existed_loads_with_no_servers() {
        // Every config.json on disk predates this field. A missing one has
        // to mean "no servers" rather than a parse error, the same way
        // every other added setting has.
        let config: Config = serde_json::from_str("{}").expect("an empty object is a config");
        assert!(config.mcp_servers.is_empty());
        // And it stays out of the file while it is empty, so adding the
        // feature does not rewrite everyone's config.
        let written = serde_json::to_string(&config).unwrap();
        assert!(!written.contains("mcp_servers"), "{written}");
    }

    #[test]
    fn a_server_round_trips_through_the_config_file() {
        let mut config = Config::default();
        config.mcp_servers.push(McpServerConfig {
            name: "fs".to_string(),
            command: "npx".to_string(),
            args: vec![
                "-y".to_string(),
                "@modelcontextprotocol/server-filesystem".to_string(),
            ],
            env: vec!["FS_TOKEN".to_string()],
        });
        let written = serde_json::to_string(&config).unwrap();
        let read: Config = serde_json::from_str(&written).unwrap();
        assert_eq!(read.mcp_servers, config.mcp_servers);

        // The variable's *name* is in the file. Its value is not, and must
        // never be: config.json holds no secrets.
        assert!(written.contains("FS_TOKEN"), "the name belongs in the file");
    }

    #[test]
    fn an_environment_value_is_found_without_the_keychain() {
        // The fallback that makes a token already exported for other tools
        // work without being copied anywhere. Named per-test because the
        // process environment is shared with every other test.
        let variable = format!("CLANK_TEST_TOKEN_{}", std::process::id());
        std::env::set_var(&variable, "from-the-environment");

        let config = McpServerConfig {
            name: "probe".to_string(),
            command: "true".to_string(),
            args: vec!["--flag".to_string()],
            env: vec![variable.clone()],
        };
        let (spec, missing) = resolve_server(&config).unwrap();

        assert!(missing.is_empty(), "{missing:?}");
        assert_eq!(
            spec.env,
            vec![(variable.clone(), "from-the-environment".to_string())]
        );
        assert_eq!(spec.args, vec!["--flag".to_string()]);
        std::env::remove_var(&variable);
    }

    #[test]
    fn a_value_that_is_nowhere_is_named_rather_than_passed_as_empty() {
        // A server that checks whether its token is set should find it
        // missing, not find it blank — and whoever ran the command should
        // be told which one, because the alternative is a server that
        // starts and then fails every call.
        let config = McpServerConfig {
            name: "probe".to_string(),
            command: "true".to_string(),
            args: Vec::new(),
            env: vec![format!("CLANK_TEST_UNSET_{}", std::process::id())],
        };
        let (spec, missing) = resolve_server(&config).unwrap();
        assert_eq!(missing, config.env);
        assert!(spec.env.is_empty(), "{:?}", spec.env);
    }

    #[test]
    fn a_server_name_that_cannot_be_routed_is_refused_before_it_is_stored() {
        let config = McpServerConfig {
            name: "has/slash".to_string(),
            command: "true".to_string(),
            args: Vec::new(),
            env: Vec::new(),
        };
        assert!(resolve_server(&config).is_err());
    }

    #[test]
    fn timeouts_seed_themselves_in_a_config_that_predates_them() {
        // Every existing config.json was written before these fields, so
        // they have to read back as the values that were compiled in rather
        // than as zero — which would fail every call instantly.
        let old = r#"{"base_url":"https://example.test/v1"}"#;
        let config: Config = serde_json::from_str(old).unwrap();
        assert_eq!(config.connect_timeout, 20);
        assert_eq!(config.request_timeout, 300);
        assert_eq!(config.stream_idle_timeout, 90);
        assert_eq!(config.command_timeout, 30);
    }

    #[test]
    fn a_configured_timeout_survives_a_round_trip() {
        let config = Config {
            stream_idle_timeout: 240,
            ..Config::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(
            serde_json::from_str::<Config>(&json)
                .unwrap()
                .stream_idle_timeout,
            240
        );
    }

    #[test]
    fn the_gates_carry_the_command_timeout_to_the_tool() {
        // `execute_tool` has no config; the gates are how the run's fallback
        // reaches it.
        let gates = SessionGates::new(ToolAccessSettings::default(), true, 45);
        assert_eq!(gates.command_timeout(), 45);
    }

    #[test]
    fn a_fresh_config_has_the_shell_off_and_says_so() {
        // Seeded explicitly rather than left `None`, which would fall back
        // to the three old booleans — and those have no way to say a tool is
        // not offered at all, so a new config would come out with the shell
        // merely gated.
        let fresh = Config::default();
        assert_eq!(
            fresh.tool_access().access("run_terminal_command"),
            ToolAccess::Never
        );
        assert_eq!(fresh.tool_access().access("write_file"), ToolAccess::Ask);
        assert!(fresh.tools.is_some(), "the seed is explicit");
    }

    #[test]
    fn a_config_from_before_tools_had_states_keeps_what_it_configured() {
        // The old shape: three category booleans under `approval`, and no
        // `tools` key at all. Upgrading must not quietly re-arm gates the
        // user had turned off.
        let old = r#"{"base_url":"https://x","approval":{"read_disk":false,"write_disk":true,"terminal":false}}"#;
        let config: Config = serde_json::from_str(old).unwrap();
        let access = config.tool_access();
        assert_eq!(access.access("read_file"), ToolAccess::Allow);
        assert_eq!(access.access("write_file"), ToolAccess::Ask);
        // Not `allow`, whatever the old boolean said: the shell starts off.
        assert_eq!(access.access("run_terminal_command"), ToolAccess::Never);

        // And once saved, it is written in the new shape and the old key is
        // gone — read for one upgrade, then never again.
        let saved = serde_json::to_string(&config).unwrap();
        assert!(!saved.contains("approval"), "{saved}");

        // A config that already has the new key ignores the old one.
        let both = r#"{"base_url":"https://x","approval":{"read_disk":false,"write_disk":false,"terminal":false},"tools":{"write_file":"never"}}"#;
        let config: Config = serde_json::from_str(both).unwrap();
        assert_eq!(config.tool_access().access("write_file"), ToolAccess::Never);
        assert_eq!(config.tool_access().access("read_file"), ToolAccess::Ask);
    }

    #[test]
    fn a_partial_config_keeps_its_values_and_seeds_the_rest() {
        // Hand-writing one key is a supported way to configure this, so the
        // keys that are there must survive and the rest must come from
        // their seeds — not from `Config::default()` wholesale.
        let config = parse_config(
            r#"{"temperature": 1.5, "base_url": "https://example.test/v1"}"#,
            Path::new("config.json"),
        )
        .expect("a partial config is valid");

        assert_eq!(config.temperature, Some(1.5));
        assert_eq!(config.base_url, "https://example.test/v1");
        // Untouched keys take their seeds, including the two that used to
        // sit at `null` while a literal supplied the real value.
        assert_eq!(config.default_model.as_deref(), Some(DEFAULT_MODEL));
        assert_eq!(config.effort_style.as_deref(), Some(DEFAULT_EFFORT_STYLE));
        assert_eq!(config.max_iterations, Some(20));
        assert!(config.sandbox);
    }

    #[test]
    fn an_explicit_null_is_not_the_seed() {
        // serde only defaults an *absent* key. `clank model --clear` writes
        // null deliberately, and that has to keep meaning "cleared" rather
        // than being quietly refilled.
        let config = parse_config(r#"{"default_model": null}"#, Path::new("config.json"))
            .expect("null is valid");
        assert_eq!(config.default_model, None);
    }

    #[test]
    fn a_malformed_config_is_refused_and_says_where() {
        let error = parse_config(
            "{\n  \"temperature\": 1.9,\n}",
            Path::new("/tmp/config.json"),
        )
        .expect_err("a trailing comma is not valid json");
        let message = error.to_string();

        // Names the file, so it's obvious which one to open...
        assert!(message.contains("/tmp/config.json"), "{message}");
        // ...where the problem is...
        assert!(message.contains("line"), "{message}");
        // ...and how to get out of it.
        assert!(message.contains("delete it"), "{message}");
    }

    #[test]
    fn a_gate_flipped_on_one_handle_is_seen_through_another() {
        // The whole point: the running turn holds a clone, and the worker
        // that answers `/tools` or `/sandbox` holds the original. A write
        // through one has to be visible through the other, or the turn keeps
        // running on the gates it started with.
        let worker = SessionGates::new(ToolAccessSettings::default(), true, 30);
        let running_turn = worker.clone();
        assert_eq!(running_turn.access().access("write_file"), ToolAccess::Ask);
        assert!(running_turn.sandbox());

        worker.set_access(worker.access().with("write", ToolAccess::Allow).unwrap());
        worker.set_sandbox(false);

        assert_eq!(
            running_turn.access().access("write_file"),
            ToolAccess::Allow
        );
        assert!(!running_turn.sandbox());
        // Only the category asked for moves; the shell keeps the default it
        // starts with, which is off.
        assert_eq!(running_turn.access().access("read_file"), ToolAccess::Ask);
        assert_eq!(
            running_turn.access().access("run_terminal_command"),
            ToolAccess::Never
        );
    }

    #[test]
    fn gates_survive_a_poisoned_lock() {
        // A panic somewhere else must not leave the gates unreadable — the
        // settings behind the lock are still perfectly good, and failing
        // here would turn an unrelated panic into a dead gate.
        let gates = SessionGates::new(ToolAccessSettings::default(), true, 30);
        let poisoner = gates.clone();
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.lock();
            panic!("poison the lock");
        })
        .join();

        assert_eq!(gates.access().access("read_file"), ToolAccess::Ask);
        gates.set_access(gates.access().with("read", ToolAccess::Allow).unwrap());
        assert_eq!(gates.access().access("read_file"), ToolAccess::Allow);
    }
}
