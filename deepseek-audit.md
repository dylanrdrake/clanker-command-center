# DeepSeek Audit — clanker-command-center

Review of the source (main.rs, config.rs, client.rs, tools.rs, the TUI, Cargo.toml, TODO.md, README.md) and the existing CI setup.

The codebase is genuinely well-engineered — several initial guesses turned out to already be done (configurable timeouts in `config.rs`, `run_terminal_command` stdin freezes fixed, picker state derivation solid). The gaps that are actually there, prioritized:

## Top gaps to act on first

### 1. No CI — only a manual release job

`.github/workflows/release.yml` is `workflow_dispatch`-only and runs `cargo test` but no `cargo fmt --check` or `clippy -D warnings`. There is no workflow on push/PR at all. Given how much deliberate formatting/clippy discipline is baked into the code (and how much of it is async/`Result` plumbing), a PR sitting on a compile error or an unused import until release day is a real risk.

- Add a `ci.yml` on `push` + `pull_request`: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`.
- Add `cargo clippy` to the release job too.

Cheapest, highest-leverage change.

### 2. The agent has no search tool — the single biggest capability gap

The tool surface (`src/tools.rs`) is read/`list_files`/write/replace/fetch/shell. For an *agentic codebase* tool there's no way to `grep`. A model narrowing a bug must read whole files and reconstruct the workspace from a flat `list_files` of names.

Add a bounded, read-only `search_files` tool: substring/regex match with line numbers + a few lines of context, capped at N results. It's read-only, so it fits the existing sandbox model trivially (reads are "never bounded"), and it slots into the same `TOOLS`/schema/`category_of` wiring with the existing drift-catching tests.

### 3. `read_file` returns an entire file with no bound

In `tools.rs`, `read_file` reads the whole file into `content` and hands it over — a multi-MB file (a lockfile, a generated bundle, `node_modules` output) goes to the model in one turn and sits in every request after via the message array. Nothing stops this except compaction downstream.

Add an optional `lines`/`start`+`end` range param and a hard cap, so the *agent* can page through a big file instead of eating context.

### 4. `save_config` writes non-atomically

`src/config.rs::save_config` does `fs::write(&config_path, json)`. A crash/power-loss mid-write corrupts `config.json`. That's the *one* file the codebase deliberately refuses to run unless it parses cleanly — and the refusal path ("delete it to start over") is now the recovery from a bug the app itself could create.

Write to `config.json.tmp` + `rename` so the "nothing corrupts my config" guarantee actually holds. Small, self-contained.

## Worth doing, and already on the roadmap (so mostly: prioritize)

- **Provider profiles** — the README correctly flags that switching providers means re-running `endpoint`→`login`→`model`→`effort-style`. Multi-provider is the horizontal that makes everything else (per-provider keys, effort styles, headers) coherent. Build *before* `--json`/headless, because detached runs only make sense per-provider.
- **`--json` output / stdout-stderr split** — the TODO analysis is right that this is a precondition for usable headless runs. Not gated on anything else; good parallel work with #4.
- **Picker scrolling + decryption caching** — both already written up in TODO with accurate cost estimates. Don't touch them early; they scale with stored-session count, not with the product.

## Looser threads worth tightening

- **Windows freezes** — the README calls Linux "most stable" and TODO owns the Windows terminal-freeze report. Turn it into a tracked issue with a repro rather than a vague TODO; it's the kind of thing that stays "known" forever. (Needs a Windows machine to fix blind.)
- **`cmd_login` reads the key with echo visible** — minor; move to a masked prompt once that command is touched.

## Recommended starting point

Quickest concrete win that changes the product's actual usefulness: **#1 (CI) and #2 (search tool)** together — one is safety, one is the capability the design is genuinely missing. Then the atomic config write (#4).
