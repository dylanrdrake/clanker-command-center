use clanker_command_center::conversation::{self, Conversation};
use clanker_command_center::{
    agent, client, config, mcp, session, spinner, store, terminal_ui, tui, ui, wrap,
};

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;
use futures_util::FutureExt;
use rustyline::DefaultEditor;
use std::io::{self, Write};
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use client::{ChatMessage, Client};
use config::{
    clear_api_key, get_api_key, get_config_path, load_config, save_config, set_api_key,
    SessionGates, ToolAccess, ToolAccessSettings, VALID_EFFORT_STYLES,
};
use session::ChatSession;
use spinner::Spinner;
use store::{KIND_AGENT_CHAT, KIND_CHAT};
use terminal_ui::TerminalAgentUi;
use ui::{parse_bool, response_label};

#[derive(Parser)]
#[command(name = "clank")]
#[command(about = "Clanker Command Center - An OpenAI-compatible frontend for any LLM provider", long_about = None)]
#[command(version = "0.1.0")]
struct Cli {
    /// A one-off question or task. With no prompt and no subcommand at all,
    /// `clank` launches the full-screen TUI on its launch screen — the only
    /// way in; there are no flags to skip straight into a clanker.
    ///
    /// A subcommand wins over a prompt, so a one-word prompt that happens to
    /// be a subcommand name (`clank status`) runs the subcommand. Quote it
    /// after `--` to force the prompt: `clank -- status`.
    prompt: Option<String>,

    /// Let this run use tools — read and write files, run commands — under
    /// whatever `clank tools` allows. Without it a prompt is answered with
    /// no tools at all.
    #[arg(long)]
    tools: bool,

    /// Model to use (overrides the persistent default for this call)
    #[arg(short, long)]
    model: Option<String>,

    /// Sampling temperature (overrides the persistent default for this call)
    #[arg(long)]
    temperature: Option<f32>,

    /// Reasoning effort (overrides the persistent default for this call).
    /// Not checked against a fixed list — pass whatever your model accepts.
    #[arg(long)]
    effort_level: Option<String>,

    /// Maximum tool-calling iterations for this run (with `--tools`)
    #[arg(long)]
    max_iterations: Option<usize>,

    /// Show each tool call and its result as it happens
    #[arg(short, long)]
    verbose: bool,

    /// Keep this run as a clanker, so it shows up in the picker and can be
    /// resumed. Without it a one-off leaves nothing behind.
    #[arg(long)]
    save: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Set up your API key
    Login,

    /// Remove stored API key
    Logout,

    /// Check configuration status
    Status,

    /// Add, list or remove MCP servers, whose tools the agent can then use
    Mcp {
        #[command(subcommand)]
        action: Option<McpAction>,
    },

    /// List available models
    Models,

    /// View or set the persistent default model
    Model {
        /// Model to set as the default (omit to show the current default)
        name: Option<String>,

        /// Clear the stored default model (falls back to openrouter/auto)
        #[arg(long)]
        clear: bool,
    },

    /// View or set the API base URL, to point at any OpenAI-compatible service
    Endpoint {
        /// Base URL to use, e.g. https://openrouter.ai/api/v1 (omit to show the current value)
        url: Option<String>,

        /// Clear the stored endpoint (falls back to the OpenRouter default)
        #[arg(long)]
        clear: bool,
    },

    /// Show or set the model that compacts a clanker's history
    Compactor {
        /// Model to compact with (omit to show the current one)
        name: Option<String>,

        /// Clear the stored compactor (falls back to openrouter/auto)
        #[arg(long)]
        clear: bool,
    },

    /// Show or set the prompt size that sets compaction going
    CompactAt {
        /// Prompt tokens a request has to reach before the next turn
        /// compacts first (omit to show the current threshold)
        value: Option<u64>,

        /// Clear it — nothing compacts on its own, and `/compact` inside a
        /// clanker becomes the only way to compact at all
        #[arg(long)]
        clear: bool,
    },

    /// View or set how the reasoning effort level is sent to the provider
    EffortStyle {
        /// Style to set: flat, nested, or none (omit to show the current value)
        value: Option<String>,

        /// Clear the stored effort style (falls back to "nested")
        #[arg(long)]
        clear: bool,
    },

    /// Manage extra HTTP headers sent with every API request
    Headers {
        #[command(subcommand)]
        action: Option<HeaderCommands>,
    },

    /// Show what each tool may do, or change it
    Tools {
        /// `ask`, `allow`, `never` — or `on`/`off` for every tool at once.
        /// Omit to list them.
        state: Option<String>,
        /// A tool's name, a category (read/write/terminal/web), an MCP server's name (or `server__*`), or `all`.
        target: Option<String>,
    },

    /// View or set the persistent default max agent iterations
    MaxIterations {
        /// Value to set as the default (omit to show the current default)
        value: Option<usize>,

        /// Clear the stored default — `ask`/`agent`/a new `session` then run
        /// with no cap unless `--max-iterations` is passed for that call
        #[arg(long)]
        clear: bool,
    },

    /// View or set the persistent default sampling temperature
    #[command(visible_alias = "temp")]
    Temperature {
        /// Value to set as the default (omit to show the current default)
        value: Option<f32>,

        /// Clear the stored default — requests are then sent with no
        /// temperature field, and the provider uses its own default
        #[arg(long)]
        clear: bool,
    },

    /// View or set whether responses stream in as they're generated
    Stream {
        /// on/off (also accepts true/false, yes/no, 1/0). Omit to show the
        /// current setting.
        #[arg(value_parser = parse_bool)]
        value: Option<bool>,
    },

    /// View or set whether new sessions band your own messages in the
    /// transcript
    Highlight {
        /// on/off (also accepts true/false, yes/no, 1/0). Omit to show the
        /// current setting.
        #[arg(value_parser = parse_bool)]
        value: Option<bool>,
    },

    /// View or set whether the launch screen bands its selected row
    Selection {
        /// on/off (also accepts true/false, yes/no, 1/0). Omit to show the
        /// current setting.
        #[arg(value_parser = parse_bool)]
        value: Option<bool>,
    },

    /// View or set whether new sessions start showing full tool-call detail
    Verbose {
        /// on/off (also accepts true/false, yes/no, 1/0). Omit to show the
        /// current setting.
        #[arg(value_parser = parse_bool)]
        value: Option<bool>,
    },

    /// View or set whether the agent's file writes are confined to the
    /// working directory
    Sandbox {
        /// on/off (also accepts true/false, yes/no, 1/0). Omit to show the
        /// current setting.
        #[arg(value_parser = parse_bool)]
        value: Option<bool>,
    },

    /// View or set the persistent default reasoning effort level (low, medium, high)
    #[command(name = "effort", visible_alias = "effort-level")]
    EffortLevel {
        /// Effort level to set as the default (omit to show the current default)
        value: Option<String>,

        /// Clear the stored effort level (falls back to provider default)
        #[arg(long)]
        clear: bool,
    },

    /// View or set how long the client waits — connecting, on a whole
    /// reply, between streamed chunks, and on a command the agent runs
    Timeout {
        /// Which one: connect, request, stream-idle, or command. Omit to
        /// show them all.
        name: Option<String>,
        /// Seconds. Omit to show just this one.
        secs: Option<u64>,
    },

    /// Start or resume a clanker in a line-based conversation — the
    /// counterpart to the full-screen UI
    Clanker {
        /// Model to use for a new clanker (overrides the persistent
        /// default; ignored when resuming, which keeps its saved model)
        #[arg(short, long)]
        model: Option<String>,

        /// Maximum number of tool-calling iterations per turn while in
        /// a clanker with tools (overrides the persistent default for this call)
        #[arg(long)]
        max_iterations: Option<usize>,

        /// Sampling temperature for this clanker (overrides the persistent
        /// default for this call)
        #[arg(long)]
        temperature: Option<f32>,

        /// Reasoning effort for a new clanker (overrides the persistent
        /// default; ignored when resuming, which keeps its saved value).
        /// Not checked against a fixed list — pass whatever your model
        /// accepts.
        #[arg(long)]
        effort_level: Option<String>,

        /// Resume a saved clanker by id (or unique id prefix); pass with no
        /// value to pick from a list of all your saved clankers
        #[arg(long, num_args = 0..=1, default_missing_value = PICK_SESSION_SENTINEL)]
        resume: Option<String>,

        /// Resume in the current directory instead of the one the clanker
        /// was started in, and remember it — for a project that has moved
        #[arg(long)]
        here: bool,

        /// Name the clanker. Prompted for if omitted; ignored when resuming,
        /// since a resumed clanker keeps the name it has
        #[arg(long)]
        title: Option<String>,
    },

    /// List, show or delete your saved clankers
    Clankers {
        #[command(subcommand)]
        action: Option<ClankerCommands>,
    },
}

#[derive(Subcommand, Clone)]
enum McpAction {
    /// Add a server. Everything after `--` is the command to run it.
    ///
    /// e.g. clank mcp add fs -- npx -y @modelcontextprotocol/server-filesystem .
    Add {
        /// What its tools are prefixed with: `name/tool`
        name: String,

        /// Environment variable this server needs. Repeatable. The *name*
        /// only — set its value with `clank mcp env`, which puts it in the
        /// OS keychain rather than in config.json.
        #[arg(long = "env", value_name = "VARIABLE")]
        env: Vec<String>,

        /// The command and its arguments
        #[arg(last = true, required = true)]
        command: Vec<String>,
    },

    /// List the configured servers
    List,

    /// Remove a server, and forget any values stored for it
    Remove { name: String },

    /// Store the value of one of a server's environment variables
    Env {
        name: String,
        /// The variable to set
        variable: String,
        /// Forget the stored value instead of setting one
        #[arg(long)]
        clear: bool,
    },
}

#[derive(Subcommand)]
enum HeaderCommands {
    /// Show current extra headers
    Show,
    /// Set (or overwrite) a header
    Set {
        /// Header name, e.g. HTTP-Referer
        name: String,
        /// Header value
        value: String,
    },
    /// Remove a header
    Unset {
        /// Header name to remove
        name: String,
    },
}

#[derive(Subcommand)]
enum ClankerCommands {
    /// List saved clankers
    List,
    /// Show a clanker's full message history
    Show {
        /// Clanker id (or unique id prefix)
        id: String,
    },
    /// Delete a saved clanker
    Delete {
        /// Clanker id (or unique id prefix)
        id: String,
    },
}

/// Sentinel value for `--resume` passed with no id, meaning "show a picker".
/// Never collides with a real session id since those are lowercase-hex UUIDs
/// (no 'p', 'i', 'c', 'k' in hex).
const PICK_SESSION_SENTINEL: &str = "pick";

/// Resolves a `--resume` value (an id/prefix, or the "pick" sentinel) to the
/// session it refers to, prompting the user to choose from a numbered list
/// of their saved clankers of the given kind when no id was given.
fn resolve_resume_target(
    conn: &rusqlite::Connection,
    id_or_prefix: &str,
) -> Result<store::SessionSummary> {
    if id_or_prefix != PICK_SESSION_SENTINEL {
        return store::find_session(conn, id_or_prefix)?
            .ok_or_else(|| anyhow::anyhow!("No clanker found matching '{}'", id_or_prefix));
    }

    let sessions = store::list_sessions(conn)?;
    if sessions.is_empty() {
        anyhow::bail!("No saved clankers to resume");
    }

    println!("{}\n", "Select a clanker to resume:".blue());
    for (i, s) in sessions.iter().enumerate() {
        let mode = store::mode_label(s.kind == KIND_AGENT_CHAT);
        println!(
            "  {}. {}  {} {}",
            i + 1,
            (&s.id[..8]).bright_black(),
            mode,
            s.title
        );
    }

    print!("\n{} ", "Clanker number:".blue());
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let choice: usize = input
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid selection: '{}'", input.trim()))?;

    choice
        .checked_sub(1)
        .and_then(|i| sessions.into_iter().nth(i))
        .ok_or_else(|| anyhow::anyhow!("Invalid selection: {}", choice))
}

fn resolve_model(config: &config::Config, cli_model: Option<String>) -> String {
    cli_model
        .or_else(|| config.default_model.clone())
        .unwrap_or_else(|| config::DEFAULT_MODEL.to_string())
}

/// The model that compacts a clanker's history. No flag to override it: it
/// is a global setting, and a per-clanker one is the next thing to build
/// here rather than a per-invocation one — see [`config::Config::compactor`].
fn resolve_compactor(config: &config::Config) -> String {
    config
        .compactor
        .clone()
        .unwrap_or_else(|| config::DEFAULT_MODEL.to_string())
}

/// `None` if neither the flag nor the config default is set — genuinely no
/// value, not a hardcoded floor. `ask`/`agent`/a new `session` pass this
/// straight through: a request goes out with no `temperature` field, and an
/// agent-mode turn with no cap errors immediately rather than guessing one.
fn resolve_max_iterations(config: &config::Config, cli_value: Option<usize>) -> Option<usize> {
    cli_value.or(config.max_iterations)
}

/// Same deal as [`resolve_max_iterations`].
fn resolve_temperature(config: &config::Config, cli_value: Option<f32>) -> Option<f32> {
    cli_value.or(config.temperature)
}

/// Same deal as [`resolve_max_iterations`] — `None` is itself a meaningful
/// value (no effort field sent), not just "unset".
fn resolve_effort_level(config: &config::Config, cli_value: Option<String>) -> Option<String> {
    cli_value.or_else(|| config.effort_level.clone())
}

#[tokio::main]
async fn main() -> Result<()> {
    quiet_on_closed_output();
    let cli = Cli::parse();
    // Caught so a panic still reaches the shutdown below rather than
    // unwinding straight past it.
    let outcome = AssertUnwindSafe(dispatch(cli)).catch_unwind().await;
    // Whatever happened. An MCP server is a child process of this one, and
    // a process that exits without stopping them is the orphan problem by
    // another route — see `mcp::StdioServer::shutdown`. A no-op when
    // nothing connected, which is every command that does not run a tool.
    mcp::shutdown_all().await;
    match outcome {
        Ok(result) => result,
        // What a shell reports for a tool `SIGPIPE` killed, so a pipeline
        // under `set -o pipefail` sees the same as it would for `cat`.
        Err(payload) if panic_message(&*payload).is_some_and(is_closed_output) => {
            std::process::exit(141)
        }
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

/// Keeps `clank tools | head` from ending in a panic message.
///
/// Rust ignores `SIGPIPE`, so writing to a pipe whose reader has gone is an
/// error, and `println!` turns that error into a panic. Putting the signal
/// back to fatal would end such a write quietly, but it would also end the
/// whole process the moment an MCP server exits with its stdin still open
/// to us. So the panic happens, this hook says nothing about it, and `main`
/// turns it into an exit. Any other panic is reported as before.
fn quiet_on_closed_output() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if !panic_message(info.payload()).is_some_and(is_closed_output) {
            previous(info);
        }
    }));
}

/// A panic's message, when it has one: `panic!` with arguments carries a
/// `String`, and with a bare literal a `&str`.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> Option<&str> {
    payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
}

/// Whether this is std's "failed printing to stdout" (or stderr) panic for a
/// reader that went away. Matched against the error std itself formats for
/// `EPIPE` rather than a hard-coded "Broken pipe", so the wording is
/// whatever this platform's is.
fn is_closed_output(message: &str) -> bool {
    let epipe = io::Error::from_raw_os_error(libc::EPIPE).to_string();
    message.starts_with("failed printing to std") && message.ends_with(&epipe)
}

/// These servers, resolved for starting, with what could not be resolved
/// said out loud.
///
/// For the `clank tools` commands that have to name a server's tools — the
/// one place outside the worker that connects on purpose rather than
/// lazily. Takes the servers rather than the config so a caller that
/// needs one of them resolves one of them: a broken server it is not about
/// has no business complaining here.
fn resolve_configured(servers: &[config::McpServerConfig]) -> Vec<mcp::ServerSpec> {
    let mut specs = Vec::new();
    for server in servers {
        match config::resolve_server(server) {
            Ok((spec, missing)) => {
                for variable in missing {
                    eprintln!(
                        "{} {}: {variable} is not set — clank mcp env {} {variable}",
                        "!".yellow(),
                        server.name,
                        server.name
                    );
                }
                specs.push(spec);
            }
            Err(e) => eprintln!("{} {}: {e}", "✗".red(), server.name),
        }
    }
    specs
}

/// Brings up the servers a `tools` command has to see before it can set
/// `target`, reporting any that will not start — see
/// `config::servers_to_name` — the CLI's counterpart to the worker's
/// `set_tool_access_connecting`, so `clank tools` and `/tools` cover the
/// same tools.
async fn connect_to_name(
    target: &str,
    access: ToolAccess,
    current: &config::ToolAccessSettings,
    servers: &[config::McpServerConfig],
) {
    let specs = resolve_configured(&config::servers_to_name(target, access, current, servers));
    for started in mcp::ensure_connected(&specs, |_| true).await {
        if let Err(e) = started.outcome {
            eprintln!("{} {}: {e}", "✗".red(), started.name);
        }
    }
}

async fn dispatch(cli: Cli) -> Result<()> {
    // Nothing here connects MCP servers. They come up where they are
    // needed — a turn that could offer one of their tools, or the one
    // `clank tools` command that has to name one — so a command that only
    // reads settings, or a clanker that never calls a server, doesn't pay
    // to start one. See `mcp::ensure_for_access`.
    match cli.command {
        // A prompt on its own is a one-off run; nothing at all opens the TUI.
        None => match cli.prompt {
            Some(prompt) => {
                cmd_run(
                    &prompt,
                    cli.tools,
                    cli.model,
                    cli.verbose,
                    cli.max_iterations,
                    cli.temperature,
                    cli.effort_level,
                    cli.save,
                )
                .await?
            }
            None => cmd_tui().await?,
        },
        Some(Commands::Login) => cmd_login()?,
        Some(Commands::Logout) => cmd_logout()?,
        Some(Commands::Status) => cmd_status()?,
        Some(Commands::Mcp { action }) => cmd_mcp(action)?,
        Some(Commands::Models) => cmd_models().await?,
        Some(Commands::Model { name, clear }) => cmd_model(name, clear)?,
        Some(Commands::Endpoint { url, clear }) => cmd_endpoint(url, clear)?,
        Some(Commands::Compactor { name, clear }) => cmd_compactor(name, clear)?,
        Some(Commands::CompactAt { value, clear }) => cmd_compact_at(value, clear)?,
        Some(Commands::EffortStyle { value, clear }) => cmd_effort_style(value, clear)?,
        Some(Commands::Headers { action }) => cmd_headers(action)?,
        Some(Commands::Tools { state, target }) => cmd_tools(state, target).await?,
        Some(Commands::MaxIterations { value, clear }) => cmd_max_iterations(value, clear)?,
        Some(Commands::Temperature { value, clear }) => cmd_temperature(value, clear)?,
        Some(Commands::Stream { value }) => cmd_stream(value)?,
        Some(Commands::Sandbox { value }) => cmd_sandbox(value)?,
        Some(Commands::Highlight { value }) => cmd_highlight(value)?,
        Some(Commands::Selection { value }) => cmd_selection(value)?,
        Some(Commands::Verbose { value }) => cmd_verbose(value)?,
        Some(Commands::EffortLevel { value, clear }) => cmd_effort_level(value, clear)?,
        Some(Commands::Timeout { name, secs }) => cmd_timeout(name, secs)?,
        Some(Commands::Clanker {
            model,
            max_iterations,
            temperature,
            effort_level,
            resume,
            here,
            title,
        }) => {
            cmd_clanker(
                model,
                max_iterations,
                temperature,
                effort_level,
                resume,
                here,
                title,
            )
            .await?
        }
        Some(Commands::Clankers { action }) => cmd_clankers(action)?,
    }

    Ok(())
}

fn cmd_login() -> Result<()> {
    let mut config = load_config()?;

    // Pre-filled with the current endpoint (which is itself the configured
    // default until `clank endpoint` changes it), so accepting it is just
    // pressing Enter — only typing something else actually changes it.
    let mut rl = DefaultEditor::new()?;
    let endpoint = match rl.readline_with_initial("Endpoint URL: ", (&config.base_url, "")) {
        Ok(line) => line,
        Err(rustyline::error::ReadlineError::Interrupted)
        | Err(rustyline::error::ReadlineError::Eof) => {
            println!("{} Login cancelled", "✗".red());
            return Ok(());
        }
        Err(e) => return Err(e.into()),
    };
    let endpoint = endpoint.trim().trim_end_matches('/').to_string();
    if !endpoint.is_empty() && endpoint != config.base_url {
        config.base_url = endpoint;
        save_config(&config)?;
        println!("{} Endpoint set to {}\n", "✓".green(), config.base_url);
    }

    print!("{} ", "Enter your API key:".blue());
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let api_key = input.trim();

    if api_key.is_empty() {
        anyhow::bail!("API key cannot be empty");
    }

    set_api_key(api_key)?;

    println!("{} API key saved to OS keychain", "✓".green());
    println!(
        "{} {}",
        "Config location:".bright_black(),
        get_config_path()?.display()
    );

    Ok(())
}

fn cmd_logout() -> Result<()> {
    clear_api_key()?;
    println!("{} API key removed", "✓".green());
    Ok(())
}

/// `clank mcp`, and its four actions.
///
/// Bare `clank mcp` lists, like `clank model` with no argument shows the
/// current one rather than complaining.
fn cmd_mcp(action: Option<McpAction>) -> Result<()> {
    match action.unwrap_or(McpAction::List) {
        McpAction::List => cmd_mcp_list(),
        McpAction::Add { name, env, command } => cmd_mcp_add(name, env, command),
        McpAction::Remove { name } => cmd_mcp_remove(&name),
        McpAction::Env {
            name,
            variable,
            clear,
        } => cmd_mcp_env(&name, &variable, clear),
    }
}

fn cmd_mcp_add(name: String, env: Vec<String>, command: Vec<String>) -> Result<()> {
    // Rejected here rather than at connect time: a name with a `/` in it
    // cannot be routed, and finding that out when a tool is called is far
    // too late.
    mcp::ServerSpec::new(&name, command.first().cloned().unwrap_or_default())?;

    let mut config = load_config()?;
    if config.mcp_servers.iter().any(|server| server.name == name) {
        anyhow::bail!(
            "There is already a server called {name}. Remove it first, or pick another name."
        );
    }

    let (program, args) = command
        .split_first()
        .ok_or_else(|| anyhow::anyhow!("Give the command to run the server after `--`"))?;
    config.mcp_servers.push(config::McpServerConfig {
        name: name.clone(),
        command: program.clone(),
        args: args.to_vec(),
        env: env.clone(),
    });
    save_config(&config)?;

    println!("{} Added {name}", "✓".green());
    println!("  {} {}", program, args.join(" "));
    for variable in &env {
        let stored = config::get_mcp_env(&name, variable)
            .ok()
            .flatten()
            .is_some();
        let where_from = if stored {
            format!("{} in the keychain", "✓".green())
        } else if std::env::var(variable).is_ok() {
            format!("{} from this environment", "✓".green())
        } else {
            format!("{} unset — clank mcp env {name} {variable}", "✗".red())
        };
        println!("  {variable}: {where_from}");
    }
    println!(
        "\n{}",
        "Its tools are off until you allow them — see clank tools.".bright_black()
    );
    Ok(())
}

fn cmd_mcp_list() -> Result<()> {
    let config = load_config()?;
    if config.mcp_servers.is_empty() {
        println!(
            "\n{}\n  {}",
            "No MCP servers configured.".blue(),
            "clank mcp add <name> -- <command to run it>".bright_black()
        );
        return Ok(());
    }

    println!("\n{}", "MCP servers:".blue());
    for server in &config.mcp_servers {
        println!(
            "  {} · {} {}",
            server.name.bold(),
            server.command,
            server.args.join(" ")
        );
        // Reported per variable because a server missing its token starts
        // and then fails every call, which reads as the server being broken.
        for variable in &server.env {
            let state = if config::get_mcp_env(&server.name, variable)
                .ok()
                .flatten()
                .is_some()
            {
                format!("{} keychain", "✓".green())
            } else if std::env::var(variable).is_ok() {
                format!("{} environment", "✓".green())
            } else {
                format!("{} unset", "✗".red())
            };
            println!("      {variable}: {state}");
        }
    }
    println!(
        "\n{}",
        "Tools appear as <server>/<tool> in clank tools once the server is reached.".bright_black()
    );
    Ok(())
}

fn cmd_mcp_remove(name: &str) -> Result<()> {
    let mut config = load_config()?;
    let before = config.mcp_servers.len();
    let removed: Vec<config::McpServerConfig> = config
        .mcp_servers
        .iter()
        .filter(|server| server.name == name)
        .cloned()
        .collect();
    config.mcp_servers.retain(|server| server.name != name);
    if config.mcp_servers.len() == before {
        anyhow::bail!("There is no server called {name}. `clank mcp` lists them.");
    }
    save_config(&config)?;

    // Its stored values go with it. Leaving a token in the keychain for a
    // server nobody has any more is the kind of thing that is never found
    // again to be cleaned up.
    for server in &removed {
        for variable in &server.env {
            if let Err(e) = config::clear_mcp_env(name, variable) {
                eprintln!("{} Could not forget {variable}: {e}", "✗".yellow());
            }
        }
    }
    println!("{} Removed {name}", "✓".green());
    Ok(())
}

fn cmd_mcp_env(name: &str, variable: &str, clear: bool) -> Result<()> {
    let config = load_config()?;
    if !config.mcp_servers.iter().any(|server| server.name == name) {
        anyhow::bail!("There is no server called {name}. `clank mcp` lists them.");
    }

    if clear {
        config::clear_mcp_env(name, variable)?;
        println!("{} Forgot {variable} for {name}", "✓".green());
        return Ok(());
    }

    // Read the same way `clank login` reads a key, which is to say with
    // the echo still on — see the note about that in TODO.md. Worth fixing
    // in both places at once rather than differently in each.
    print!("{} ", format!("Value for {variable}:").blue());
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let value = input.trim();
    if value.is_empty() {
        anyhow::bail!("Nothing entered, so nothing was stored.");
    }
    config::set_mcp_env(name, variable, value)?;
    println!(
        "{} Stored {variable} for {name} in the OS keychain",
        "✓".green()
    );
    Ok(())
}

fn cmd_status() -> Result<()> {
    let config = load_config()?;
    println!("\n{}", "Clanker Command Center Configuration:".blue());
    println!("  Base URL: {}", config.base_url);
    println!(
        "  API Key: {}",
        if get_api_key()?.is_some() {
            format!("{} Set (OS keychain)", "✓".green())
        } else {
            format!("{} Not set", "✗".red())
        }
    );
    println!(
        "  Default model: {}",
        config
            .default_model
            .as_deref()
            .unwrap_or(config::DEFAULT_MODEL)
    );
    println!(
        "  Max iterations: {}",
        config
            .max_iterations
            .map(|n| n.to_string())
            .unwrap_or_else(|| "(not set)".to_string())
    );
    println!(
        "  Temperature: {}",
        config
            .temperature
            .map(|n| n.to_string())
            .unwrap_or_else(|| "(not set)".to_string())
    );
    println!(
        "  Effort level: {}",
        config
            .effort_level
            .as_deref()
            .unwrap_or("(not set, provider default)")
    );
    println!(
        "  Effort style: {}",
        config
            .effort_style
            .as_deref()
            .unwrap_or(config::DEFAULT_EFFORT_STYLE)
    );
    println!("  Streaming: {}", if config.stream { "on" } else { "off" });
    println!("  Compactor: {}", resolve_compactor(&config));
    println!(
        "  Compact at: {}",
        match config.compact_at {
            Some(threshold) => format!("{} prompt tokens", ui::format_tokens(threshold as i64)),
            None => "off — /compact only".to_string(),
        }
    );
    println!("  Sandbox: {}", if config.sandbox { "on" } else { "off" });
    println!("  Verbose: {}", if config.verbose { "on" } else { "off" });
    println!(
        "  Message highlighting: {}",
        if config.highlight { "on" } else { "off" }
    );
    println!(
        "  Launch screen selection: {}",
        if config.selection { "on" } else { "off" }
    );
    if config.extra_headers.is_empty() {
        println!("  Extra headers: none");
    } else {
        println!("  Extra headers: {}", config.extra_headers.len());
    }
    println!("  Config file: {}", get_config_path()?.display());
    println!("\n{}", "Tools:".blue());
    print_tools(&config.tool_access());
    println!();
    Ok(())
}

/// A session's state as one padded, coloured word, matching the launch
/// screen's badges: yellow for anything wanting attention, red for a
/// failure, green for a turn that finished.
fn format_state(state: store::LastState) -> colored::ColoredString {
    use store::LastState;
    let (word, colour): (&str, fn(&str) -> colored::ColoredString) = match state {
        LastState::Working => ("working", |s| s.yellow()),
        LastState::AwaitingApproval => ("approval", |s| s.yellow()),
        LastState::Failed => ("failed", |s| s.red()),
        LastState::Interrupted => ("stopped", |s| s.yellow()),
        LastState::Replied => ("replied", |s| s.green()),
        LastState::NoReply => ("no reply", |s| s.cyan()),
        LastState::New => ("new", |s| s.bright_black()),
    };
    colour(&format!("{word:<8}"))
}

/// Every configurable wait, with the field each one sets.
///
/// A table rather than a match per arm so that showing them all, naming
/// them in an error, and setting one all read the same list — a timeout
/// added to the config and forgotten here would be invisible to `clank
/// timeout`, which is the only way to discover it exists.
/// Name, the field it sets, and what it bounds.
type TimeoutField = for<'a> fn(&'a mut config::Config) -> &'a mut u64;
type TimeoutEntry = (&'static str, TimeoutField, &'static str);

const TIMEOUTS: [TimeoutEntry; 4] = [
    (
        "connect",
        |c| &mut c.connect_timeout,
        "connecting: DNS, TCP and TLS",
    ),
    (
        "request",
        |c| &mut c.request_timeout,
        "a whole non-streaming reply",
    ),
    (
        "stream-idle",
        |c| &mut c.stream_idle_timeout,
        "the gap between streamed chunks",
    ),
    (
        "command",
        |c| &mut c.command_timeout,
        "a terminal command, when the model names no timeout",
    ),
];

fn cmd_timeout(name: Option<String>, secs: Option<u64>) -> Result<()> {
    let mut config = load_config()?;

    let Some(name) = name else {
        println!("\n{}", "Timeouts:".blue());
        let width = TIMEOUTS.iter().map(|(n, ..)| n.len()).max().unwrap_or(0);
        for (name, field, blurb) in TIMEOUTS {
            let value = *field(&mut config);
            println!(
                "  {}  {:>4}s  {}",
                format!("{name:<width$}").bright_black(),
                value,
                blurb.bright_black()
            );
        }
        println!("\n{}", "  clank timeout <name> <seconds>".bright_black());
        return Ok(());
    };

    let Some((_, field, blurb)) = TIMEOUTS.iter().find(|(n, ..)| *n == name) else {
        let names: Vec<&str> = TIMEOUTS.iter().map(|(n, ..)| *n).collect();
        anyhow::bail!("Unknown timeout '{name}'. One of: {}", names.join(", "));
    };

    let Some(secs) = secs else {
        println!("{name}: {}s — {blurb}", field(&mut config));
        return Ok(());
    };

    // Zero would mean "time out immediately", which is never what anyone
    // wants and reads as a way to disable the bound rather than to make it
    // absolute.
    if secs == 0 {
        anyhow::bail!("A timeout of 0 would fail every call before it started");
    }

    *field(&mut config) = secs;
    save_config(&config)?;
    println!("{} {name} timeout set to {secs}s", "✓".green());
    Ok(())
}

/// `clank tools` — the listing, and the one command that changes it.
///
/// One verb for both the global default and a clanker's own `/tools`, and
/// one listing for both, so what you read in one place is what you would
/// type in the other.
async fn cmd_tools(state: Option<String>, target: Option<String>) -> Result<()> {
    let mut config = load_config()?;

    let updated = match (state.as_deref(), target.as_deref()) {
        (None, _) => {
            print_tools(&config.tool_access());
            // What a server would add, which is not in the list above: a
            // fresh process has nothing connected, so this says what is
            // configured and whether it is up rather than leaving the
            // servers out of the picture entirely.
            if !config.mcp_servers.is_empty() {
                println!();
                for (name, state) in ui::mcp_rows(&config.mcp_servers, &mcp::connected_counts()) {
                    println!("  {name}: {}", state.bright_black());
                }
            }
            println!("\n{}", "Usage:".bright_black());
            println!("  clank tools <ask|allow|never> <tool|category|server|all>");
            println!("  clank tools on                 Every tool back to its default");
            println!("  clank tools off                Every tool off");
            return Ok(());
        }
        // Back to the defaults, which is not "ask for everything": the web
        // tool reads a page and changes nothing, and its default is to get
        // on with it.
        (Some("on"), None) => ToolAccessSettings::defaults(),
        (Some("off"), None) => ToolAccessSettings::none(),
        (Some(state), Some(target)) => {
            let access = ToolAccess::parse(state).ok_or_else(|| {
                anyhow::anyhow!("Unknown state '{state}'. Use ask, allow or never.")
            })?;

            // Only what naming `target` actually needs comes up — see
            // `config::servers_to_name`.
            let servers = config.mcp_servers.clone();
            let access_settings = config.tool_access();
            connect_to_name(target, access, &access_settings, &servers).await;

            match access_settings.with(target, access) {
                Some(updated) => updated,
                // A configured server that is not connected has no tools to
                // be named with, and `with` cannot tell that from a typo —
                // the same case `/tools` handles. `never` for the whole
                // server is a standing key that needs no tool list, which is
                // what lets it be switched off without being started.
                None => match config::server_target(target, &servers) {
                    Some((server, true)) if access == ToolAccess::Never => {
                        access_settings.never_server(&server)
                    }
                    _ => {
                        return Err(anyhow::anyhow!(
                            "No tool, category or server called '{target}'."
                        ))
                    }
                },
            }
        }
        (Some(state), None) => {
            anyhow::bail!(
                "'{state}' needs something to act on: a tool, a category, a server, or all."
            )
        }
    };

    config.tools = Some(updated);
    save_config(&config)?;
    println!("{} Tools:", "✓".green());
    print_tools(&config.tool_access());
    Ok(())
}

/// The tool listing, shared by `clank tools` and `clank status`.
fn print_tools(access: &ToolAccessSettings) {
    for line in terminal_ui::tool_lines(access) {
        println!("{line}");
    }
}

fn cmd_model(name: Option<String>, clear: bool) -> Result<()> {
    let mut config = load_config()?;

    if clear {
        config.default_model = None;
        save_config(&config)?;
        println!(
            "{} Default model cleared, falling back to {}",
            "✓".green(),
            config::DEFAULT_MODEL
        );
        return Ok(());
    }

    match name {
        Some(name) => {
            config.default_model = Some(name.clone());
            save_config(&config)?;
            println!("{} Default model set to {}", "✓".green(), name);
        }
        None => {
            println!(
                "Current default model: {}",
                config
                    .default_model
                    .as_deref()
                    .unwrap_or(config::DEFAULT_MODEL)
            );
        }
    }

    Ok(())
}

fn cmd_endpoint(url: Option<String>, clear: bool) -> Result<()> {
    let mut config = load_config()?;

    if clear {
        config.base_url = config::default_base_url();
        save_config(&config)?;
        println!(
            "{} Endpoint cleared, falling back to {}",
            "✓".green(),
            config.base_url
        );
        return Ok(());
    }

    match url {
        Some(url) => {
            let trimmed = url.trim_end_matches('/').to_string();
            config.base_url = trimmed.clone();
            save_config(&config)?;
            println!("{} Endpoint set to {}", "✓".green(), trimmed);
            println!(
                "{} Remember to run `clank login` if this provider uses a different API key",
                "i".bright_black()
            );
        }
        None => {
            println!("Current endpoint: {}", config.base_url);
        }
    }

    Ok(())
}

fn cmd_effort_style(value: Option<String>, clear: bool) -> Result<()> {
    let mut config = load_config()?;

    if clear {
        config.effort_style = None;
        save_config(&config)?;
        println!(
            "{} Effort style cleared, falling back to {}",
            "✓".green(),
            config::DEFAULT_EFFORT_STYLE
        );
        return Ok(());
    }

    match value {
        Some(value) => {
            let normalized = value.to_lowercase();
            if !VALID_EFFORT_STYLES.contains(&normalized.as_str()) {
                anyhow::bail!(
                    "Invalid effort style '{value}'. Valid values: {}",
                    VALID_EFFORT_STYLES.join(", ")
                );
            }
            config.effort_style = Some(normalized.clone());
            save_config(&config)?;
            println!("{} Effort style set to {}", "✓".green(), normalized);
        }
        None => {
            println!(
                "Current effort style: {}",
                config
                    .effort_style
                    .as_deref()
                    .unwrap_or(config::DEFAULT_EFFORT_STYLE)
            );
        }
    }

    Ok(())
}

fn cmd_headers(action: Option<HeaderCommands>) -> Result<()> {
    let mut config = load_config()?;

    match action.unwrap_or(HeaderCommands::Show) {
        HeaderCommands::Show => {
            if config.extra_headers.is_empty() {
                println!("No extra headers set.");
            } else {
                println!("{}\n", "Extra headers:".blue());
                for (key, value) in &config.extra_headers {
                    println!("  {}: {}", key, value);
                }
            }
        }
        HeaderCommands::Set { name, value } => {
            config.extra_headers.insert(name.clone(), value.clone());
            save_config(&config)?;
            println!("{} Header set: {}: {}", "✓".green(), name, value);
        }
        HeaderCommands::Unset { name } => {
            if config.extra_headers.remove(&name).is_some() {
                save_config(&config)?;
                println!("{} Header removed: {}", "✓".green(), name);
            } else {
                println!("No header named '{}' was set.", name);
            }
        }
    }

    Ok(())
}

/// The persistent default for showing full tool-call detail. A session
/// snapshots this when it's created, so changing it here affects new
/// sessions; `/verbose` toggles the one you're in.
/// The band behind your own messages. Per-session like `verbose`, so this
/// only sets what a *new* session starts with; `/highlight` changes one that
/// already exists.
fn cmd_highlight(value: Option<bool>) -> Result<()> {
    let mut config = load_config()?;
    let state = |on: bool| if on { "highlighted" } else { "plain" };

    match value {
        Some(enabled) => {
            config.highlight = enabled;
            save_config(&config)?;
            println!(
                "{} New sessions start with your messages {}",
                "✓".green(),
                state(enabled)
            );
        }
        None => {
            println!(
                "New sessions start with your messages: {}",
                state(config.highlight)
            );
        }
    }

    Ok(())
}

/// The band on the launch screen's selected row. Global only — that screen
/// belongs to no session, so there is nothing to override it with.
fn cmd_selection(value: Option<bool>) -> Result<()> {
    let mut config = load_config()?;
    let state = |on: bool| if on { "highlighted" } else { "plain" };

    match value {
        Some(enabled) => {
            config.selection = enabled;
            save_config(&config)?;
            println!(
                "{} The launch screen's selected row is {}",
                "✓".green(),
                state(enabled)
            );
        }
        None => {
            println!(
                "The launch screen's selected row is: {}",
                state(config.selection)
            );
        }
    }

    Ok(())
}

fn cmd_verbose(value: Option<bool>) -> Result<()> {
    let mut config = load_config()?;

    match value {
        Some(enabled) => {
            config.verbose = enabled;
            save_config(&config)?;
            println!(
                "{} New sessions start {}",
                "✓".green(),
                if enabled { "verbose" } else { "quiet" }
            );
        }
        None => {
            println!(
                "New sessions start: {}",
                if config.verbose { "verbose" } else { "quiet" }
            );
        }
    }

    Ok(())
}

/// The persistent default for confining the agent's file writes. A session
/// snapshots this when it's created, so changing it here affects new
/// sessions; `/sandbox` changes the one you're in.
fn cmd_sandbox(value: Option<bool>) -> Result<()> {
    let mut config = load_config()?;

    match value {
        Some(enabled) => {
            config.sandbox = enabled;
            save_config(&config)?;
            println!("{} {}", "✓".green(), ui::sandbox_notice(enabled, true));
        }
        None => {
            println!("{}", ui::sandbox_notice(config.sandbox, false));
        }
    }

    Ok(())
}

fn cmd_stream(value: Option<bool>) -> Result<()> {
    let mut config = load_config()?;

    match value {
        Some(enabled) => {
            config.stream = enabled;
            save_config(&config)?;
            println!(
                "{} Streaming responses {}",
                "✓".green(),
                if enabled { "enabled" } else { "disabled" }
            );
        }
        None => {
            println!(
                "Streaming responses: {}",
                if config.stream { "on" } else { "off" }
            );
        }
    }

    Ok(())
}

fn cmd_max_iterations(value: Option<usize>, clear: bool) -> Result<()> {
    let mut config = load_config()?;

    if clear {
        config.max_iterations = None;
        save_config(&config)?;
        println!(
            "{} Default max iterations cleared — a run with tools now needs one set per call \
             (--max-iterations) or per session (/max-iterations) to run at all",
            "✓".green()
        );
        return Ok(());
    }

    match value {
        Some(0) => anyhow::bail!("max-iterations must be greater than 0"),
        Some(value) => {
            config.max_iterations = Some(value);
            save_config(&config)?;
            println!("{} Default max iterations set to {}", "✓".green(), value);
        }
        None => {
            println!(
                "Current default max iterations: {}",
                config
                    .max_iterations
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "(not set)".to_string())
            );
        }
    }

    Ok(())
}

fn cmd_temperature(value: Option<f32>, clear: bool) -> Result<()> {
    let mut config = load_config()?;

    if clear {
        config.temperature = None;
        save_config(&config)?;
        println!(
            "{} Default temperature cleared — requests now have no temperature field \
             unless set per call (--temperature) or per session (/temperature)",
            "✓".green()
        );
        return Ok(());
    }

    match value {
        Some(value) if !(0.0..=2.0).contains(&value) => {
            anyhow::bail!("temperature must be between 0 and 2")
        }
        Some(value) => {
            config.temperature = Some(value);
            save_config(&config)?;
            println!("{} Default temperature set to {}", "✓".green(), value);
        }
        None => {
            println!(
                "Current default temperature: {}",
                config
                    .temperature
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "(not set)".to_string())
            );
        }
    }

    Ok(())
}

fn cmd_effort_level(value: Option<String>, clear: bool) -> Result<()> {
    let mut config = load_config()?;

    if clear {
        config.effort_level = None;
        save_config(&config)?;
        println!(
            "{} Effort level cleared, falling back to provider default",
            "✓".green()
        );
        return Ok(());
    }

    match value {
        Some(value) => {
            // Not checked against a fixed low/medium/high list — models
            // vary in what reasoning-effort values they actually accept,
            // and this is easy to correct with another `clank effort-level`
            // if it turns out wrong for whatever you're pointed at.
            config.effort_level = Some(value.clone());
            save_config(&config)?;
            println!("{} Effort level set to {}", "✓".green(), value);
        }
        None => {
            println!(
                "Current effort level: {}",
                config
                    .effort_level
                    .as_deref()
                    .unwrap_or("(not set, provider default)")
            );
        }
    }

    Ok(())
}

/// `clank compactor [name] [--clear]` — the model that does the summarizing.
/// Clearing falls back to the same default an unset model does, rather than
/// switching compaction off; `clank compact-at --clear` is the off switch.
fn cmd_compactor(name: Option<String>, clear: bool) -> Result<()> {
    let mut config = load_config()?;

    if clear {
        config.compactor = None;
        save_config(&config)?;
        println!(
            "{} Compactor cleared, falling back to {}",
            "✓".green(),
            config::DEFAULT_MODEL
        );
        return Ok(());
    }

    match name {
        Some(name) => {
            config.compactor = Some(name.clone());
            save_config(&config)?;
            println!("{} Compactor set to {}", "✓".green(), name);
        }
        None => {
            println!("Current compactor: {}", resolve_compactor(&config));
        }
    }

    Ok(())
}

/// `clank compact-at [n] [--clear]` — how large a request's prompt has to get
/// before the next turn summarizes the older part of the conversation first.
fn cmd_compact_at(value: Option<u64>, clear: bool) -> Result<()> {
    let mut config = load_config()?;

    if clear {
        config.compact_at = None;
        save_config(&config)?;
        println!(
            "{} Automatic compaction turned off — /compact still works inside a clanker",
            "✓".green()
        );
        return Ok(());
    }

    match value {
        Some(0) => anyhow::bail!(
            "A threshold of 0 would compact before every turn. Give a token count, \
             or --clear to turn automatic compaction off."
        ),
        Some(value) if value < config::MIN_COMPACT_AT => anyhow::bail!(
            "A threshold of {value} is below the {} minimum. The system prompt and tool \
             schemas ride every request whatever is folded away, and the summary and \
             kept tail take about half the threshold on top, so a compacted request \
             could not get back under it. Give a larger token count, or --clear to turn \
             automatic compaction off. (With MCP servers connected the real minimum is \
             higher, since their tool schemas ride every request too.)",
            config::MIN_COMPACT_AT
        ),
        Some(value) => {
            config.compact_at = Some(value);
            save_config(&config)?;
            println!(
                "{} Compacting once a request's prompt reaches {} tokens",
                "✓".green(),
                ui::format_tokens(value as i64)
            );
        }
        None => match config.compact_at {
            Some(threshold) => println!(
                "Compacting once a request's prompt reaches {} tokens",
                ui::format_tokens(threshold as i64)
            ),
            None => println!("Automatic compaction is off — /compact only"),
        },
    }

    Ok(())
}

async fn cmd_models() -> Result<()> {
    let config = load_config()?;
    let client = Client::new(config)?;

    let spinner = Spinner::start("Fetching models...");
    let models = client.list_models().await;
    spinner.stop().await;
    let models = models?;

    println!("{} ", "✓".green());
    println!(
        "\n{}\n",
        format!("Available models ({}):", models.len()).blue()
    );

    for (i, model) in models.iter().take(20).enumerate() {
        println!("  {}. {}", i + 1, model);
    }

    if models.len() > 20 {
        println!("  ... and {} more", models.len() - 20);
    }

    Ok(())
}

/// The `kind` a clanker is created with, derived from what its tools add up
/// to. Only a cache — see `ChatSession::is_agentic` — but it has to start
/// right, since the launch screen and `clank clankers list` read the row
/// without opening it.
fn kind_for(access: &ToolAccessSettings) -> &'static str {
    if access.any_tools() {
        KIND_AGENT_CHAT
    } else {
        KIND_CHAT
    }
}

/// A one-off run: `clank "..."`, with or without tools.
///
/// The two used to be separate commands — `ask` and `agent` — which made the
/// difference between them a thing to learn rather than a flag. It is the
/// same question either way: what may this run use? Without `--tools` the
/// answer is nothing, and a run with nothing to call is a plain reply.
#[allow(clippy::too_many_arguments)]
async fn cmd_run(
    prompt: &str,
    tools: bool,
    model: Option<String>,
    verbose: bool,
    max_iterations: Option<usize>,
    temperature: Option<f32>,
    effort_level: Option<String>,
    save: bool,
) -> Result<()> {
    // A run that is kept goes through the path that has a clanker to keep it
    // in, tools or not — the only thing tools change is what it may call.
    // A one-off with neither takes the shorter path: one request, printed.
    if !tools && !save {
        return cmd_ask(prompt, model, temperature, effort_level).await;
    }
    cmd_agent(
        prompt,
        tools,
        model,
        verbose,
        max_iterations,
        temperature,
        effort_level,
        save,
    )
    .await
}

async fn cmd_ask(
    prompt: &str,
    model: Option<String>,
    temperature: Option<f32>,
    effort_level: Option<String>,
) -> Result<()> {
    let config = load_config()?;
    let settings = agent::TurnSettings {
        model: resolve_model(&config, model),
        temperature: resolve_temperature(&config, temperature),
        effort_level: resolve_effort_level(&config, effort_level),
        stream: config.stream,
        // One request, whatever the configured cap.
        max_iterations: None,
    };
    let client = Client::new(config)?;

    let mut messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(prompt.to_string()),
        ..Default::default()
    }];

    // The same exchange every other path runs, so a one-off reply is shown
    // the way a `--tools` run's is. Labelled, like `cmd_agent`'s: a one-off
    // has no other way to show which model answered.
    let mut ui = TerminalAgentUi::new(false, true);
    agent::run_chat_turn(
        &client,
        &mut ui,
        &mut messages,
        &settings,
        &agent::UsageTracker::default(),
    )
    .await?;
    Ok(())
}

/// Prints a saved transcript's user/assistant turns (tool and system messages
/// are omitted since they're internal bookkeeping, not conversation content).
/// No model label on replies, matching the TUI transcript — current model
/// is `/model`'s job, not every reply's.
fn print_transcript(messages: &[store::StoredMessage]) {
    for sm in messages {
        let m = &sm.message;
        match m.role.as_str() {
            "user" => {
                if let Some(content) = &m.content {
                    println!(
                        "{} {}",
                        "❯".green().bold(),
                        wrap::wrap_indented(content, "  ")
                    );
                }
            }
            "assistant" => {
                if let Some(content) = &m.content {
                    println!("\n{} {}\n", "●".cyan(), wrap::wrap_indented(content, "  "));
                }
            }
            _ => {}
        }
    }
}

/// Pulls the user turns out of a resumed session's history, in order, so
/// they can seed the readline history and stay recallable with Up/Down.
fn user_prompts(messages: &[store::StoredMessage]) -> Vec<String> {
    messages
        .iter()
        .filter(|sm| sm.message.role == "user")
        .filter_map(|sm| sm.message.content.clone())
        .collect()
}

/// Asks for a session title, for `clank session` without `--title`.
///
/// Refuses a blank one rather than falling back to naming the session from
/// its first message: creating a session should be deliberate, and a name is
/// what makes it worth keeping whether or not anything is said in it.
fn prompt_for_title() -> Result<String> {
    let mut rl = DefaultEditor::new()?;
    loop {
        match rl.readline(&format!("{} ", "Session title:".blue())) {
            Ok(line) if !line.trim().is_empty() => return Ok(line.trim().to_string()),
            Ok(_) => println!("{} A title is required.", "✗".red()),
            Err(rustyline::error::ReadlineError::Interrupted)
            | Err(rustyline::error::ReadlineError::Eof) => {
                anyhow::bail!("Cancelled")
            }
            Err(e) => return Err(e.into()),
        }
    }
}

async fn cmd_clanker(
    model: Option<String>,
    max_iterations: Option<usize>,
    temperature: Option<f32>,
    effort_level: Option<String>,
    resume: Option<String>,
    here: bool,
    title: Option<String>,
) -> Result<()> {
    let config = load_config()?;
    let conn = store::open_db()?;

    // The merge of the global config default with any `--flag` override,
    // used as the concrete snapshot for a brand new session below, and as
    // the target `/max-iterations default`/`/temperature default`/
    // `/effort default` resolve to for the rest of this run.
    let default_max_iterations = resolve_max_iterations(&config, max_iterations);
    let default_temperature = resolve_temperature(&config, temperature);
    let default_effort_level = resolve_effort_level(&config, effort_level);
    // Read before `config` moves into the client below. Global settings, so
    // unlike the three above there is nothing per-clanker to resolve them
    // against — see `Config::compactor`.
    let compactor = resolve_compactor(&config);
    let compact_at = config.compact_at;

    let mut prior_prompts: Vec<String> = Vec::new();
    let session = match resume {
        Some(id_or_prefix) => {
            let summary = resolve_resume_target(&conn, &id_or_prefix)?;
            // A resumed session keeps its own saved settings; `-m` (like any
            // other override flag) only ever applies to a brand new one.
            if model.is_some() {
                println!(
                    "{} Ignoring --model: resumed sessions keep their saved model",
                    "note:".bright_black()
                );
            }
            if title.is_some() {
                println!(
                    "{} Ignoring --title: resumed sessions keep their saved name",
                    "note:".bright_black()
                );
            }
            println!(
                "{} Resuming session {} ({})\n",
                "✓".green(),
                summary.id,
                summary.title
            );
            let (mut session, history) =
                ChatSession::resume(conn, &summary, summary.model.clone())?;

            // The session's directory is its sandbox boundary and what its
            // relative paths mean, so resuming somewhere else would silently
            // rebind both to wherever the shell happens to be.
            if here {
                if let Some(cwd) = std::env::current_dir()
                    .ok()
                    .map(|d| d.display().to_string())
                {
                    session.set_working_dir(cwd.clone())?;
                    println!("{} Session repointed at {}", "✓".green(), cwd);
                }
            } else {
                match session::enter_working_dir(&session)? {
                    session::EnteredDir::Moved(dir) => {
                        println!("{} Working directory: {}", "↳".blue(), dir);
                    }
                    session::EnteredDir::Unchanged => {}
                    session::EnteredDir::Missing(dir) => {
                        anyhow::bail!(
                            "This session was started in {dir}, which no longer exists.\n\n\
                             Its sandbox and relative paths are anchored there, so resuming \
                             elsewhere would quietly rebind them to the current directory. \
                             Re-run with --here to resume where you are and repoint the session."
                        );
                    }
                }
            }

            print_transcript(&history);
            prior_prompts = user_prompts(&history);
            session
        }
        // A new clanker starts with no tools, same as the TUI's "Spawn
        // clanker" — `/tools on` gives it the ones `clank tools` allows.
        None => {
            let model = resolve_model(&config, model);
            // Naming it is the deliberate act of starting one, so there's no
            // untitled path — an omitted `--title` is asked for rather than
            // defaulted.
            let title = match title {
                Some(title) => title,
                None => prompt_for_title()?,
            };
            let mut session = ChatSession::create(
                conn,
                session::new_id(),
                model,
                KIND_CHAT,
                default_effort_level.clone(),
                default_max_iterations,
                default_temperature,
                ToolAccessSettings::none(),
                config.sandbox,
                config.verbose,
                config.highlight,
                config.stream,
                std::env::current_dir()
                    .ok()
                    .map(|dir| dir.display().to_string()),
            )?;
            session.set_title(title)?;
            session
        }
    };

    // The claim, not merely the reporting: two processes appending turns to
    // one history write colliding `seq` values, and the result reloads as a
    // conversation with its turns shuffled and its tool results detached
    // from the calls they answer. Nothing detects that and nothing repairs
    // it, so a session that cannot be claimed is not run.
    let claim = match session::Heartbeat::claim(session.id().to_string()) {
        Ok(Some(claim)) => claim,
        Ok(None) => anyhow::bail!(
            "Session {} is already being run by another process.\n\n\
             Wait for it to finish. A claim left behind by a process that \
             died expires on its own within half a minute.",
            session.short_id()
        ),
        Err(e) => anyhow::bail!("Could not claim clanker {}: {e}", session.short_id()),
    };

    // What `/tools on` turns on — read before `config` moves into the client.
    let tool_access_default = config.tool_access();
    let client = Arc::new(Client::new(config)?);

    println!("{}\n", "Starting clanker (type 'exit' to quit)".blue());

    let mut rl = DefaultEditor::new()?;
    // So Up/Down can recall prompts from before this resume, not just what's
    // typed in the current sitting.
    for prompt in prior_prompts {
        let _ = rl.add_history_entry(prompt);
    }
    // No `-v` here (unlike a one-off run) — matching the TUI, a clanker
    // always starts quiet; `/verbose` is the only way to turn it on. No model
    // label either, again matching the TUI transcript. Seeded from the
    // session rather than hardcoded, so a resumed session that had
    // `/verbose` on comes back showing detail.
    // The same worker the TUI drives, so a clanker means the same thing
    // whichever front end it is opened in: this loop only reads lines,
    // sends what they mean, and prints what comes back.
    let mut conversation = Conversation::spawn(
        client,
        session,
        default_max_iterations,
        default_temperature,
        default_effort_level,
        tool_access_default,
        Some(compactor.clone()),
        compact_at,
        claim,
    );
    // Everything below knows the session only as the worker reports it,
    // starting from the snapshot it always sends first.
    let Some(conversation::Event::Snapshot(state)) = conversation.next_event().await else {
        anyhow::bail!("The clanker stopped before it started");
    };
    let mut ui = TerminalAgentUi::new(state.verbose, false);
    ui.mark_as(&state.id);
    let mut view = terminal_ui::SessionView::of(state);

    loop {
        // Blocks this thread until a line arrives, so the runtime is told to
        // move its other work — the worker, the claim's heartbeat — off it.
        let readline =
            tokio::task::block_in_place(|| rl.readline(&format!("{} ", "❯".green().bold())));

        let line = match readline {
            Ok(line) => line,
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("{} Session ended", "✓".green());
                break;
            }
            Err(e) => {
                eprintln!("{} Error: {}", "✗".red(), e);
                break;
            }
        };

        if line.trim().is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(line.as_str());

        if line.to_lowercase() == "exit" {
            println!("{} Session ended", "✓".green());
            break;
        }

        let submission = ui::classify(&line);
        // `$` and the model browser go to the worker in the TUI, but each
        // needs a screen this loop doesn't have — see `answer_locally`.
        let tui_only = matches!(
            submission,
            ui::Submission::Shell(_) | ui::Submission::BrowseModels
        );
        match conversation::command_for(&submission).filter(|_| !tui_only) {
            Some(command) => {
                let message = matches!(command, conversation::Command::Send(_));
                if message {
                    println!();
                }
                conversation.send(command);
                if !follow(&mut conversation, &mut ui, &mut view).await {
                    eprintln!("{} The clanker stopped unexpectedly", "✗".red());
                    break;
                }
                // A reply ends with its own blank line; a notice doesn't.
                if !message {
                    println!();
                }
            }
            None => {
                answer_locally(submission, &view, &compactor, compact_at);
                // One blank line after every transcript unit, matching a
                // message reply and the TUI's own Notice spacing.
                println!();
            }
        }
    }

    // So the worker's last writes land before the process exits.
    conversation.shutdown().await;
    println!(
        "{} Clanker saved. Resume with: clank clanker --resume {}",
        "✓".green(),
        view.state.short_id()
    );

    Ok(())
}

/// Prints what the worker reports until it has finished what it was just
/// sent — [`conversation::Event::Ready`] — answering any approval it asks
/// for on the way. `false` if the worker has gone.
async fn follow(
    conversation: &mut Conversation,
    ui: &mut TerminalAgentUi,
    view: &mut terminal_ui::SessionView,
) -> bool {
    use conversation::Event;
    use ui::AgentUi;
    // A compaction is a full round trip with the whole folded span in it,
    // so on a long history it is the longest silence `clank` produces. The
    // same spinner a request gets, until whatever the worker says next.
    let mut compacting: Option<Spinner> = None;
    loop {
        let event = conversation.next_event().await;
        if let Some(spinner) = compacting.take() {
            spinner.stop().await;
        }
        let Some(event) = event else {
            return false;
        };
        match event {
            Event::Ready => return true,
            Event::Agent(event) => ui.event(event).await,
            Event::ApprovalRequested(request) => {
                // Unanswerable — stdin closed — reads as a no, the safe
                // answer to a question nobody can hear.
                let allowed = ui.approve(request).await.unwrap_or(false);
                conversation.send(conversation::Command::Approve(allowed));
            }
            event => {
                if let Event::VerboseChanged { verbose } = event {
                    ui.set_verbose(verbose);
                }
                for line in view.apply(&event) {
                    println!("{line}");
                }
                if matches!(event, Event::Compacting { .. }) {
                    compacting = Some(Spinner::start("Compacting..."));
                }
            }
        }
    }
}

/// Answers a submission that changes nothing, from what the view already
/// holds — the read-only half of [`conversation::command_for`]'s split —
/// or says why a command can't run here.
fn answer_locally(
    submission: ui::Submission,
    view: &terminal_ui::SessionView,
    compactor: &str,
    compact_at: Option<u64>,
) {
    match submission {
        // TUI-only for now. The box that shows a command's output and asks
        // whether to send it has no equivalent in a blocking prompt loop, so
        // `$` here would have to mean something different — see TODO.
        ui::Submission::Shell(_)
        | ui::Submission::SendShell
        | ui::Submission::DiscardShell
        // The CLI answers an approval at its own blocking prompt, and has no
        // launch screen to go back to.
        | ui::Submission::AllowTool
        | ui::Submission::DenyTool
        // A cursor moving through a list of 400 names, which a blocking
        // prompt has nowhere to draw. `clank models` lists them here.
        | ui::Submission::BrowseModels
        // A pane beside the conversation; `git diff` is right there instead.
        | ui::Submission::ToggleDiff
        | ui::Submission::Back => {
            println!(
                "{} that's a TUI command (`clank tui`), not available here",
                "✗".red()
            );
        }
        ui::Submission::ShowModel => {
            println!("Model: {}", response_label(&view.state.model, &view.state.effort_level));
        }
        ui::Submission::ShowEffort => {
            println!(
                "{}",
                ui::effort_notice(view.state.effort_level.as_deref(), false).blue()
            );
        }
        ui::Submission::ShowVerbose => {
            println!("{}", ui::verbose_notice(view.state.verbose, false).blue());
        }
        ui::Submission::ShowHighlight => {
            println!("{}", ui::highlight_notice(view.state.highlight, false).blue());
        }
        ui::Submission::ShowStream => {
            println!("{}", ui::stream_notice(view.state.stream, false).blue());
        }
        ui::Submission::ShowTemperature => {
            println!(
                "{}",
                ui::temperature_notice(view.state.temperature, false).blue()
            );
        }
        ui::Submission::ShowSandbox => {
            println!("{}", ui::sandbox_notice(view.state.sandbox, false).blue());
        }
        ui::Submission::ShowTitle => {
            println!("{}", ui::title_notice(&view.state.title, false).blue());
        }
        ui::Submission::ShowTools => {
            println!("{}", "Tools:".blue());
            print_tools(&view.state.tool_access);
        }
        ui::Submission::ShowMcp => {
            let servers = load_config().map(|c| c.mcp_servers).unwrap_or_default();
            println!("{}", "MCP servers:".blue());
            for (name, state) in ui::mcp_rows(&servers, &mcp::connected_counts()) {
                println!("  {name}: {}", state.bright_black());
            }
        }
        ui::Submission::ShowHelp => {
            let rows = ui::help_rows();
            let width = rows.iter().map(|(name, _)| name.len()).max().unwrap_or(0);
            println!("\n{}", "Commands:".blue());
            for (name, blurb) in rows {
                // Padded before colouring, since the escape codes count
                // toward a format width and would misalign the column.
                println!("  {}  {blurb}", format!("{name:<width$}").bright_black());
            }
            println!();
        }
        ui::Submission::ShowStatus => {
            let rows = ui::session_settings_rows(&ui::SessionSettings {
                id: view.state.short_id(),
                title: &view.state.title,
                model: &view.state.model,
                effort_level: view.state.effort_level.as_deref(),
                temperature: view.state.temperature,
                max_iterations: view.state.max_iterations,
                verbose: view.state.verbose,
                highlight: view.state.highlight,
                sandbox: view.state.sandbox,
                stream: view.state.stream,
                working_dir: view.state.working_dir.as_deref(),
                tool_access: &view.state.tool_access,
                total_tokens: view.state.total_tokens,
                compactor,
                compact_at,
            });
            let width = rows.iter().map(|(label, _)| label.len()).max().unwrap_or(0);
            println!("\n{}", "Clanker:".blue());
            for (label, value) in rows {
                // Padded before colouring: the escape codes count toward a
                // format width, so colouring first misaligns the column.
                println!("  {}  {value}", format!("{label:<width$}").bright_black());
            }
            println!();
        }
        ui::Submission::UnknownCommand(message) => {
            println!("{} {}", "✗".red(), message);
        }
        // Everything else changes the session, which the worker owns —
        // `command_for` sends those there before this is reached.
        other => unreachable!("{other:?} goes to the worker"),
    }
}

/// Opens the session a `--session` agent run writes to, or `None` for the
/// default one-shot that leaves nothing behind.
///
/// Split from [`cmd_agent`] because it does a job of its own: snapshotting
/// the config-plus-flags defaults onto a new row the same way `clank
/// session` does, and taking the claim before anything else touches the
/// session.
fn open_agent_session(
    config: &config::Config,
    model: Option<String>,
    max_iterations: Option<usize>,
    temperature: Option<f32>,
    effort_level: Option<String>,
    session: bool,
    access: &ToolAccessSettings,
) -> Result<Option<(ChatSession, terminal_ui::ActivityWriter)>> {
    if !session {
        return Ok(None);
    }

    let conn = store::open_db()?;

    let session = ChatSession::create(
        conn,
        session::new_id(),
        resolve_model(config, model),
        kind_for(access),
        resolve_effort_level(config, effort_level),
        resolve_max_iterations(config, max_iterations),
        resolve_temperature(config, temperature),
        access.clone(),
        config.sandbox,
        config.verbose,
        config.highlight,
        config.stream,
        std::env::current_dir()
            .ok()
            .map(|dir| dir.display().to_string()),
    )?;
    // Nothing else can be holding a session created a line ago, but it goes
    // through the same claim so that every path out of here owns one.
    let Some(activity) = terminal_ui::ActivityWriter::claim(session.id().to_string())? else {
        anyhow::bail!("Could not claim the clanker just created");
    };
    // No title is set, which leaves it eligible for the usual
    // derive-from-first-message step — and the first message is the task, so
    // the picker names the session after the work rather than "Untitled".
    println!(
        "{} Clanker {} — resume it with `clank clanker --resume {}`\n",
        "✓".green(),
        session.short_id(),
        session.short_id()
    );
    Ok(Some((session, activity)))
}

#[allow(clippy::too_many_arguments)]
async fn cmd_agent(
    task: &str,
    tools: bool,
    model: Option<String>,
    verbose: bool,
    max_iterations: Option<usize>,
    temperature: Option<f32>,
    effort_level: Option<String>,
    session: bool,
) -> Result<()> {
    let config = load_config()?;

    let access = if tools {
        config.tool_access()
    } else {
        ToolAccessSettings::none()
    };
    let stored = open_agent_session(
        &config,
        model.clone(),
        max_iterations,
        temperature,
        effort_level.clone(),
        session,
        &access,
    )?;

    // A session's own settings are the ones it runs with; the flag-plus-config
    // merge only applies to a run that has no session to remember anything.
    let (mut settings, tool_access, sandbox) = match &stored {
        Some((session, _)) => (
            session.turn_settings(),
            session.tool_access().clone(),
            session.sandbox(),
        ),
        None => (
            agent::TurnSettings {
                model: resolve_model(&config, model),
                temperature: resolve_temperature(&config, temperature),
                effort_level: resolve_effort_level(&config, effort_level),
                stream: config.stream,
                max_iterations: resolve_max_iterations(&config, max_iterations),
            },
            access,
            config.sandbox,
        ),
    };

    // Lazy connection, before the agentic checks just below: a clanker
    // whose built-ins are all off but whose servers are not is still
    // agentic, and they need the servers' tools to see it.
    for started in mcp::ensure_for_access(&tool_access).await {
        if let Err(e) = started.outcome {
            eprintln!("{} {}: {e}", "✗".red(), started.name);
        }
    }

    // Nothing to call means nothing to loop over: one iteration is the whole
    // of the turn, so a nullified cap is no reason to refuse the run.
    if !tool_access.any_tools() {
        settings.max_iterations = settings.max_iterations.or(Some(1));
    }

    let client = Client::new(config)?;

    if tool_access.any_tools() {
        println!("{}\n", "Starting task...".blue());
    }

    // Unlike `session`, a one-shot task has no other way to show which
    // model answered, so it keeps the label `session`/`tui` dropped.
    let mut ui = TerminalAgentUi::new(verbose, true);

    let gates = SessionGates::new(tool_access, sandbox, client.command_timeout());

    let Some((mut session, activity)) = stored else {
        agent::run_agent(&client, &mut ui, task, &settings, &gates).await?;
        return Ok(());
    };

    // Reports Working/Failed to the picker, which is the whole point of
    // running with a session: a detached task is otherwise invisible until
    // it finishes.
    // Bound before the first write, so a turn that outlives its claim is
    // refused rather than interleaved into a session someone else now holds.
    session.writes_under_claim(activity.claim_owner().to_string());
    ui.watch(activity);

    session.push_user(task.to_string());
    if let Err(e) = session.persist_pending() {
        eprintln!("{} Failed to save message: {}", "✗".red(), e);
    }

    session.set_activity(Some(store::Activity::Working), None);
    let usage = agent::UsageTracker::default();
    let turn = agent::run_agent_turn(
        &client,
        &mut ui,
        session.messages_mut(),
        &settings,
        &gates,
        // Nothing can join a turn that has no input to type into.
        &agent::Steering::default(),
        &usage,
    )
    .await;

    let failed = turn.is_err();
    session.set_activity(failed.then_some(store::Activity::Failed), None);
    // The prompt size among them, which this path used to leave out — so a
    // saved run resumed into a clanker never compacted on its first turn.
    for message in session.record_turn(&usage) {
        eprintln!("{} {message}", "✗".red());
    }

    // Persisted before the error is returned: the turn's messages are worth
    // keeping either way, and a failed run that saved nothing would be
    // indistinguishable in the picker from one that never started.
    if let Err(e) = session.persist_pending() {
        eprintln!("{} Failed to save message: {}", "✗".red(), e);
    }

    turn?;
    Ok(())
}

async fn cmd_tui() -> Result<()> {
    let config = load_config()?;

    let context = tui::Context {
        default_model: resolve_model(&config, None),
        effort_level: config.effort_level.clone(),
        max_iterations: config.max_iterations,
        temperature: config.temperature,
        tool_access: config.tool_access(),
        sandbox: config.sandbox,
        verbose: config.verbose,
        highlight: config.highlight,
        selection: config.selection,
        stream: config.stream,
        compactor: config.compactor.clone(),
        compact_at: config.compact_at,
        client: Arc::new(Client::new(config)?),
    };

    tui::run(context).await
}

fn cmd_clankers(action: Option<ClankerCommands>) -> Result<()> {
    let conn = store::open_db()?;

    match action.unwrap_or(ClankerCommands::List) {
        ClankerCommands::List => {
            let sessions = store::list_sessions(&conn)?;
            if sessions.is_empty() {
                println!("No saved clankers.");
                return Ok(());
            }

            // The same state the launch screen shows, from the same
            // derivation — without a picker this is the only way to see a
            // session that is running in another terminal.
            let last = store::last_messages(&conn).unwrap_or_default();

            println!("{}\n", "Saved clankers:".blue());
            for s in &sessions {
                let state = store::last_state(s.activity, s.heartbeat, last.get(&s.id));
                println!(
                    "  {}  {}  {}  {}  {}",
                    (&s.id[..8]).bright_black(),
                    // Padded: "[ask]" and "[agent]" differ in width, so
                    // everything after them was landing raggedly.
                    format!(
                        "{:<7}",
                        format!("[{}]", store::mode_label(s.kind == KIND_AGENT_CHAT))
                    )
                    .bright_black(),
                    format_state(state),
                    s.model,
                    s.title
                );
                // What it is waiting on, when it is waiting on you.
                if let Some(detail) = &s.activity_detail {
                    println!("            {}", detail.bright_black());
                }
            }
        }
        ClankerCommands::Show { id } => {
            let summary = store::find_session(&conn, &id)?
                .ok_or_else(|| anyhow::anyhow!("No clanker found matching '{}'", id))?;
            let messages = store::load_messages(&conn, &summary.id)?;

            println!(
                "{} {} ({}, {})\n",
                "Session:".blue(),
                summary.id,
                store::mode_label(summary.kind == KIND_AGENT_CHAT),
                summary.model
            );
            print_transcript(&messages);
        }
        ClankerCommands::Delete { id } => {
            let summary = store::find_session(&conn, &id)?
                .ok_or_else(|| anyhow::anyhow!("No clanker found matching '{}'", id))?;
            store::delete_session(&conn, &summary.id)?;
            println!(
                "{} Deleted session {} ({})",
                "✓".green(),
                summary.id,
                summary.title
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_print_to_a_closed_pipe_counts_as_closed_output() {
        let epipe = io::Error::from_raw_os_error(libc::EPIPE);
        assert!(is_closed_output(&format!(
            "failed printing to stdout: {epipe}"
        )));
        assert!(is_closed_output(&format!(
            "failed printing to stderr: {epipe}"
        )));
        // A different write failure is still a panic worth seeing.
        let full = io::Error::from_raw_os_error(libc::ENOSPC);
        assert!(!is_closed_output(&format!(
            "failed printing to stdout: {full}"
        )));
        // And so is a broken pipe that something else panicked over.
        assert!(!is_closed_output(&format!(
            "writing to the server: {epipe}"
        )));
    }

    #[test]
    fn a_panic_message_is_read_from_either_payload_type() {
        let owned: Box<dyn std::any::Any + Send> = Box::new(String::from("formatted"));
        let literal: Box<dyn std::any::Any + Send> = Box::new("literal");
        let other: Box<dyn std::any::Any + Send> = Box::new(7);
        assert_eq!(panic_message(&*owned), Some("formatted"));
        assert_eq!(panic_message(&*literal), Some("literal"));
        assert_eq!(panic_message(&*other), None);
    }

    fn config_with(
        default_model: Option<&str>,
        max_iterations: Option<usize>,
        temperature: Option<f32>,
        effort_level: Option<&str>,
    ) -> config::Config {
        config::Config {
            default_model: default_model.map(str::to_string),
            max_iterations,
            temperature,
            effort_level: effort_level.map(str::to_string),
            ..config::Config::default()
        }
    }

    #[test]
    fn a_flag_beats_the_config_which_beats_the_fallback() {
        let config = config_with(Some("config-model"), None, None, None);
        assert_eq!(
            resolve_model(&config, Some("flag-model".to_string())),
            "flag-model"
        );
        assert_eq!(resolve_model(&config, None), "config-model");

        // Cleared with `clank model --clear`, which writes null: the
        // fallback is what a request is actually made with.
        let cleared = config_with(None, None, None, None);
        assert_eq!(resolve_model(&cleared, None), config::DEFAULT_MODEL);
    }

    #[test]
    fn a_cleared_temperature_stays_cleared() {
        // The documented contract: null means "send no temperature field",
        // not "fall back to 0.7". Resolving it to a number would silently
        // undo `clank temperature --clear`.
        let cleared = config_with(None, None, None, None);
        assert_eq!(resolve_temperature(&cleared, None), None);
        // A flag still overrides it for that one call.
        assert_eq!(resolve_temperature(&cleared, Some(1.5)), Some(1.5));

        let set = config_with(None, None, Some(0.7), None);
        assert_eq!(resolve_temperature(&set, None), Some(0.7));
        assert_eq!(resolve_temperature(&set, Some(1.5)), Some(1.5));
    }

    #[test]
    fn a_cleared_max_iterations_stays_cleared() {
        // Same rule, and the one with teeth: a run with tools refuses to start
        // without a cap rather than inventing one — see
        // `agent::run_agent_turn`.
        let cleared = config_with(None, None, None, None);
        assert_eq!(resolve_max_iterations(&cleared, None), None);
        assert_eq!(resolve_max_iterations(&cleared, Some(5)), Some(5));

        let set = config_with(None, Some(20), None, None);
        assert_eq!(resolve_max_iterations(&set, None), Some(20));
        assert_eq!(resolve_max_iterations(&set, Some(5)), Some(5));
    }

    #[test]
    fn a_cleared_effort_level_stays_cleared() {
        let cleared = config_with(None, None, None, None);
        assert_eq!(resolve_effort_level(&cleared, None), None);
        assert_eq!(
            resolve_effort_level(&cleared, Some("high".to_string())),
            Some("high".to_string())
        );

        let set = config_with(None, None, None, Some("low"));
        assert_eq!(resolve_effort_level(&set, None), Some("low".to_string()));
        assert_eq!(
            resolve_effort_level(&set, Some("high".to_string())),
            Some("high".to_string())
        );
    }

    #[test]
    fn an_entirely_empty_config_resolves_to_something_usable() {
        // What a fresh install with no config.json resolves to: a model to
        // call, and nothing else asserted — every other field being absent
        // is itself the correct answer.
        let empty = config_with(None, None, None, None);
        assert_eq!(resolve_model(&empty, None), config::DEFAULT_MODEL);
        assert_eq!(resolve_temperature(&empty, None), None);
        assert_eq!(resolve_effort_level(&empty, None), None);
        assert_eq!(resolve_max_iterations(&empty, None), None);
    }
}
