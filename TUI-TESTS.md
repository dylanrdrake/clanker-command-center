# Manual test plan

Started on `ea6c987` · `204b418` · `8c6a317` · `8adb1a6` · `144e511` (TODO
only, nothing to test); later sections are appended as changes land, so the
list is no longer five commits long.

Rebuild first — `cargo install --path .`, or run `./target/release/clank`.
Ordered by how likely each is to be broken, not by how new it is.

## 1. Every in-clanker command — highest risk

`204b418` moved the TUI's submission dispatch out of the key handler: 416
lines of `src/tui/mod.rs`, touching the wiring of *every* slash command.
Five paths have tests. The rest are held up by the compiler and nothing
else, so this is where a silent breakage would be.

In a TUI clanker (`clank`, open or start one), run each and check it does
what it claims:

```
/help                 lists every command
/status               shows the clanker's settings
/model                reports the current model
/model <name>         switches it; settings bar updates
/effort               reports the level
/effort high          sets it
/temperature 0.5      sets it
/temp                 reports it
/verbose on           tool arguments and results start showing
/highlight off        the band behind your messages goes
/sandbox              reports whether writes are confined
/stream               reports streaming
/tools                lists every tool and what it may do
/tools allow write    changes both write tools; takes effect immediately
/tools never run_terminal_command   refused on the spot mid-turn
/tools on             every tool back to its default
/clanker title Foo    renames; the header updates
/max-iterations 5     sets the cap
/back                 returns to the launch screen
```

Then the non-slash paths through the same code:

```bash
hello                 # a plain message still sends
$ echo hi             # output box appears
                      # Ctrl-S sends it, Ctrl-D discards
                      # /send and /discard do the same
                      # Ctrl-Y / Ctrl-N answer a tool approval
/mdoel gpt-5          # a typo is reported, NOT sent to the model
/effor                # near-miss suggestion, not prose
```

## 2. `/models` — new, and shipped broken once

It was routed to the worker while the code opening the box sat on the
branch for things the worker *doesn't* handle, so it displayed nothing at
all. Fixed in `204b418`; worth confirming for real.

```
/models                     box opens at once with a spinner, then the list
type "claude"               filters as you type
↑ ↓                         moves; stops at both ends rather than wrapping
Enter                       sets that model — check the settings bar
Esc                         closes, changes nothing
```

Edge cases:

- A filter matching nothing → "nothing matches", not an empty box.
- `/models` then `Esc` before the list arrives → stays closed, does not
  reopen when the fetch lands.
- `/models` in the line-based `clank clanker` → says it is a TUI command.
- `/models` with an argument (`/models claude`) → a usage error, not prose.

## 3. Clanker list, now flat

`8adb1a6` removed the "In this directory" / "Elsewhere" split.

- `clank` → one list, newest first, no section headings.
- Columns, left to right: mark, name, state badge, 🔨/💬, 🪙 tokens,
  directory, when, last message. Check they stay lined up with each other
  down the whole list, including rows whose state badge is animating.
- Directory column: `.` for the directory you are in, `~/…` under home, a
  full path above home, `dir not recorded` for clankers saved before it was
  tracked.
- Arrow top to bottom — the cursor should never skip a row or land
  somewhere that cannot be opened.
- Delete one with `d` → the list closes up, the cursor stays sensible.

## 4. Sorting and spacing

- `clank models` → alphabetical. The truncated twenty are now the first
  twenty *alphabetically*, not whatever the endpoint led with.
- `/models` → same order.
- With `/models` open → exactly one blank row between the box and the last
  line of the transcript.

## 5. The busy animation

The rotating dot circle is now two braille cells of scattered dots,
regenerated each tick. It should look like noise, not a clock.

- A running turn in the TUI — the `working` indicator in the settings bar.
- The same clanker watched from the launch screen in another terminal — the
  badge should animate identically, and the badge column is one cell wider
  than it was, so check nothing in the list sits ragged.
- `clank "..."` — the CLI spinner uses the same frames.
- `/models` while it fetches.

## 6. A clanker surviving being reopened

Fixed after this plan was written; the exact sequence that lost one:

1. Launch `clank`, choose **Deploy clanker**, give it a name.
2. `Ctrl-B` straight back out without typing anything.
3. Resume it from the picker.
4. `Ctrl-B` back out again.

It should still be listed. Before the fix it was deleted here, because
reopening rebuilt "was this named?" from whether anyone had spoken in it.

The same root caused a second one worth checking: name a clanker, back out,
resume it, and *then* type something. The name you gave it should survive —
it used to be replaced by one derived from that first message.

## 7. Commands, as you type them

New: the leading `/command` in the input box turns cyan once it names a real
command, and a row above the box says what it could still be. Type these
without pressing Enter and watch the box.

```
/hel                  plain — not a command yet (the row above the box
                      does list "help"; that part is the next block)
/help                 the whole word turns cyan
/helpful              plain again — a longer word is a different word
/effort               cyan
/effort high          the name stays cyan, "high" does not
/etc/hosts            plain, the case this must never claim
/mdoel gpt-5          plain — send it and it is still caught as a typo
```

- Narrow the terminal until `/max-iterations` wraps mid-name. The colour
  should break where the row breaks and pick up on the next row, under the
  right characters.
- Backspace through a lit command — it should go plain the moment the name
  stops being one, not a keystroke later.
- `↑` to recall a command from history — it should come back lit.

Then the row above the box, and `Tab`:

```
/m                    lists: models  model  max-iterations
Tab                   /models, and "models" is marked on the row
Tab Tab               /model, then /max-iterations
Tab                   back round to /models
/hel then Tab         /help — one match, so it just fills it in
/te then Tab          /temp — as far as temperature and temp agree,
                      and no further
/                     lists as many as fit, then a "+N" count
Tab × 25              steps through all 21 and round again; the row
                      should slide so the marked one is always visible
/tools                the row becomes the command's form
/tools allow read     the form stays up while you type the arguments
hello                 no row at all
/etc/hosts            no row at all
```

- The bottom row should read `Tab complete …` only while there is a list
  above the box — not while the form is showing, where Tab does nothing.
- Tab in the middle of an ordinary message must do nothing at all.
- Tab with the `/models` browser open must do nothing (the browser owns the
  keyboard; the box is its filter).
- The line-based `clank clanker` does none of this, by design.

## 8. Backing out of a working clanker, and going back in

New, and the part with no automated coverage — it needs a live model call,
and there is no fake client to run a worker against, so this section is the
test.

Start a clanker with tools (the default) and give it a task that takes a while and
writes something, then:

1. `Ctrl-B` while it is working.
2. The launch screen should show that clanker `working`, animating, with the
   tool it is running on the line beneath.
3. Watch it change on its own — the detail line should follow what it is
   doing, without you touching anything.
4. When it wants to write or run something, the badge should become `?` with
   what it is asking about. It must **wait** there, not deny itself.
5. Press Enter on that row. You should land back in the clanker — the whole
   transcript, including everything it did while you were away — with the
   approval box up and answerable (`Ctrl-Y` / `Ctrl-N`).
6. Answer it. The turn should carry on from where it paused.
7. Back out again mid-turn, let it finish this time. The row should go to
   `✓ replied` on its own, and opening it then should work the ordinary way.

Then the things that must not have broken:

- **Two at once.** Back out of a working clanker, start a second one, set it
  working, back out of that too. Both should show as working and both should
  be resumable. A third, opened and left idle, should not disturb them.
- **Another terminal still cannot touch them.** With a clanker parked here,
  open `clank` in a second terminal: that clanker must be refused, saying it
  is in use. This is the point — parking does not make a claimed clanker
  shareable, it makes *this* clank able to hand its own screen back.
- **Renaming a parked clanker** from the picker (`r`) should stick. Go back
  into it afterwards and the header should show the new name, not the old.
- **Deleting a parked clanker** (`d`) should be refused while it works, and
  allowed once it has finished.
- **Type at it after resuming.** Steering a resumed turn should work exactly
  as it does in one you never left.
- `Esc` mid-turn still cancels. `Ctrl-C` still quits promptly — it must not
  hang waiting for parked clankers, and a tool subprocess (`sleep 60`) must
  not be left behind. Check with `ps` after.
- Back out of an *idle* clanker — instant, and an empty unnamed one is still
  discarded rather than left in the list.

## 9. Two smaller ones

- **`? waiting` in the settings row.** In a clanker, get an approval to come
  up (ask a clanker with tools to write a file). The row above the key hints should
  stop animating `working` and read `? waiting` for as long as the box is up,
  then go back to `working` when you answer. The launch screen's badge for
  the same clanker should agree.
- **Discarded output is dimmed.** Run `$ ls`, press `Ctrl-D` to discard. The
  output should stay in the transcript, dimmed to the same grey as a `default`
  value in `/status`. Then run `$ ls` again and press `Ctrl-S` — that one
  should stay at full brightness. Scroll back and forth: the two should be
  distinguishable at a glance, without reading `sent` / `not sent`.

## 10. Deploying a clanker

- `clank` → the first row reads **Deploy clanker**, not "New clanker".
- Enter on it → the screen's title bar reads **Clanker Deployment**, and it
  shows a mark above the name field with `Tab for another` beside it, a
  **Settings** section, and an **Initial Orders** one.
- `Tab` repeatedly → the mark should change every press, shape and colour,
  and whatever you have typed must stay put.
- `↑`/`↓` → the `❯` marker walks every row, wrapping at both ends. Typing
  reaches the text fields only: on `Tools`, `Effort` or `Sandbox` a letter
  should do nothing, and `←`/`→` (or Space) should change the value.
- Enter with a blank name → nothing is created, and `A name is required`
  appears where the key hints were. Type a character → it disappears again.
- Empty the `Temperature` field → the row reads `none sent` once you move
  off it. Type `warm` into it and press Enter → refused, with the reason
  under the form.
- Type a name, `Tab` some more, then Enter → the clanker that opens should
  carry **the mark that was on screen when you pressed Enter**, both in the
  reply gutter and on its row back in the list. A different one means the id
  being shown is not the id being created.
- Deploy one with `Tools` on → `/status` inside it says tools are on, and its
  row in the list carries 🔨 rather than 💬. Deploy one with a `Model` or
  `Effort` you changed on the form → `/status` says the same values back.
- Deploy one with **Initial Orders** filled in → it opens with that message
  already in the transcript and a turn already running, exactly as if you had
  typed it. Backing out with `Ctrl-B` should show it `working` in the list.
- `Esc` from the deployment screen → nothing is created; the list is
  unchanged.
- `clank clankers list` → the same mark again, drawn by the CLI.
- Inside the clanker, the top row reads `<mark> <name>  <directory>`, with
  the same mark as its row and its reply gutter. Deploy two clankers with the
  same name and check the top rows still tell them apart.

## 11. Tools, three ways

New: approvals are per tool, with three states, and `approval` is gone.

```
clank tools                          six listed: web_fetch "allow",
                                     run_terminal_command "never",
                                     the rest "ask"
clank tools never run_terminal_command
clank tools                          it now reads "never"
clank tools on                       back to defaults, web_fetch "allow" again
clank tools allow read               both read tools, nothing else
clank tools bogus off                refused, naming the word it did not know
clank status                         the same listing at the bottom
```

In a clanker (`/tools` takes the same arguments):

1. `/tools` → the listing, with `✓ ask` / `! allow` / `✗ never` marks.
2. Ask a fresh clanker to run a shell command *without* touching `/tools`.
   It should say it cannot — the shell is `never` out of the box, so it is
   not in the request at all and a verbose turn shows the model never
   trying. Then `/tools ask run_terminal_command` and try again: now it
   prompts.
3. Start a turn that will write a file, and while it is running set
   `/tools never write_file` from the box. The call in flight should be
   **refused**, not prompted — the list was sent before you changed it, so
   the refusal has to happen at the call.
4. `/tools off` → the clanker becomes a plain chat: ask it to read a file
   and it should say it has no way to.
5. `/tools on` → tools return, and `web_fetch` is back to `allow` rather
   than `ask`.
6. `/status` → a `Tools` row saying on/off, and an `Each tool` row naming
   every one.

**The upgrade path, worth checking once against your real database:** open a
clanker created before this change. Its read and write tools should read as
whatever its approval gates were — a category you had turned off comes back
as `allow`, not `ask`. The shell is the exception: it reads `never` whatever
the old gate said. Same for `clank tools` against your existing
`~/.clank/config.json`, which has no `tools` key in it yet.

## 12. One noun, and no modes

New: `ask`, `agent`, `session` and `sessions` are gone from the CLI, and
`/agent` and `/ask` are gone from inside a clanker.

```bash
clank "what is a monad"                    a plain answer, no tools
clank "create a test.txt file" --tools     it writes the file
clank "create a test.txt file"             it says it cannot
clank -- status                            the prompt, not the subcommand
clank status                               the subcommand
clank "..." --save                         kept, with no tools
clank clanker --title "Parser work"        the line-based conversation
clank clankers list                        the saved ones
clank ask "..."                            unknown command now
```

- The list column shows 🔨 or 💬 where it used to say `agent` / `ask`, in
  both `clank clankers list` and the launch screen. Check the columns after
  it still line up — those glyphs are two cells wide.
- **A new clanker starts with no tools** (💬), same as ask mode used to.
  Ask it to read a file and it should say it cannot. `/tools on` → 🔨, and
  the same request works.
- `/tools on` gives it what `clank tools` allows, not the built-in
  defaults: set `clank tools allow read`, then `/tools on` in a fresh
  clanker should show `read_file allow`.
- `/agent` and `/ask` are ordinary messages now — type one and it goes to
  the model rather than being caught as a command.
- `/clanker title Foo` renames; `/clanker` shows the name.
- `/status` has no "Mode" row; it has `Tools` (on/off) and `Each tool`.
- **The old `kind` column is still written**, so check that a clanker shows
  💬 in the list without being opened, then 🔨 after `/tools on` and backing
  out.
- `clank "explain this repo" --save` — saved with no tools, resumable, and
  it reads 💬 in the list.

## 13. From the round before, worth a glance

- **Picker scrolls.** Accumulate more clankers than fit, or shrink the
  terminal. The list should follow the cursor, and the rule above it should
  say `N more below` / `N above · N below`.
- **`clank timeout`** — shows four values; `clank timeout stream-idle 180`
  sets one; `clank timeout bogus` and `clank timeout connect 0` are refused.
- **Held clankers** show `⎚` in the list. Open a clanker in one terminal,
  look at the launch screen in another; opening it there should be refused
  with a one-line notice, and the notice should clear on the next keypress.
- **Bare `/effort`** reports the level instead of erroring.

## 14. Token counting

New: every clanker keeps a running total of what it has spent.

- A fresh clanker reads `🪙 0` in its title row (top right) and on its
  launch-screen row.
- Send one message → both go up, and to the same number. `/status` prints
  the same total again with every digit, comma-grouped.
- Send another → it *accumulates* rather than being replaced by the last
  turn's cost.
- `Ctrl-B` out and back in → the number survives. Quit `clank` entirely and
  reopen it → still there, since it is stored on the clanker.
- A turn with tools (several requests in one turn) should add the whole
  loop's cost, not just the last call's — run something that makes the model
  call two or three tools and check the jump is proportionate.
- Cancel a turn with `Esc` part-way through → whatever had already come back
  is still counted. A request that was paid for should not vanish because
  the turn it belonged to was abandoned.
- `/stream off`, then send a message → still counted. Streaming and buffered
  requests report usage by different routes, so both need checking.
- Against a provider that reports no usage at all, the total should simply
  stay where it was rather than resetting or showing something odd.

## Known, not worth reporting

- `clank model --clear extra-arg` exits 0 and silently ignores the name.
  Pre-existing; `--clear` just wins.
