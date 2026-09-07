use anyhow::{anyhow, Result};
use keyring::Entry;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
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
            .unwrap_or_else(|| default_access(tool_name))
    }

    /// Whether this clanker has any tools at all — the thing that used to be
    /// stored as "agent mode".
    pub fn any_tools(&self) -> bool {
        crate::tools::TOOLS
            .iter()
            .any(|tool| self.access(tool.name) != ToolAccess::Never)
    }

    /// Every tool with its access, in listing order.
    pub fn rows(&self) -> Vec<(&'static str, &'static str, ToolAccess)> {
        crate::tools::TOOLS
            .iter()
            .map(|tool| (tool.name, tool.category, self.access(tool.name)))
            .collect()
    }

    /// A copy with `target` set to `access`. `target` is a tool's name, a
    /// category (`read`/`write`/`terminal`/`web`), or `all`. `None` for a
    /// word that names none of those, so a caller can report the typo rather
    /// than silently changing nothing.
    pub fn with(&self, target: &str, access: ToolAccess) -> Option<Self> {
        let matched: Vec<&'static str> = crate::tools::TOOLS
            .iter()
            .filter(|tool| target == "all" || tool.name == target || tool.category == target)
            .map(|tool| tool.name)
            .collect();
        if matched.is_empty() {
            return None;
        }
        let mut updated = self.clone();
        for name in matched {
            // Held only while it differs from the default, so "set it back
            // to what it would have been" and "never mentioned it" store the
            // same thing — and a later change of default reaches both.
            if access == default_access(name) {
                updated.overrides.remove(name);
            } else {
                updated.overrides.insert(name.to_string(), access);
            }
        }
        Some(updated)
    }

    /// Every tool back to its default: what `tools on` means.
    pub fn defaults() -> Self {
        Self::default()
    }

    /// Every tool off: what `tools off` means, and what a clanker with no
    /// tools is.
    pub fn none() -> Self {
        Self::default()
            .with("all", ToolAccess::Never)
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
    let config_path = get_config_path()?;
    let json = serde_json::to_string_pretty(config)?;
    fs::write(&config_path, json)?;
    Ok(())
}

fn keyring_entry() -> Result<Entry> {
    Ok(Entry::new(KEYRING_SERVICE, KEYRING_USERNAME)?)
}

/// Reads the API key from the OS keychain (macOS Keychain, Windows
/// Credential Manager, or the Linux Secret Service). Returns `Ok(None)`
/// if no key has been stored yet.
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
}

impl SessionGates {
    pub fn new(access: ToolAccessSettings, sandbox: bool, command_timeout: u64) -> Self {
        Self {
            access: Arc::new(Mutex::new(access)),
            sandbox: Arc::new(AtomicBool::new(sandbox)),
            command_timeout,
        }
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
