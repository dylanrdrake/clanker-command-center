//! End-to-end checks against a real MCP server, ignored by default.
//!
//! `cargo test --test mcp_live -- --ignored --nocapture` runs them. Not part
//! of the normal suite: they spawn `npx`, which wants a network on a cold
//! cache and is a package manager rather than a test fixture.
use clanker_command_center::mcp::{route, ServerSpec, StdioServer};
use serde_json::json;

fn filesystem_server(dir: &str) -> ServerSpec {
    ServerSpec::new("fs", "npx").unwrap().with_args([
        "-y",
        "@modelcontextprotocol/server-filesystem",
        dir,
    ])
}

#[tokio::test]
#[ignore = "spawns npx"]
async fn connects_lists_calls_and_cleans_up_after_itself() {
    let dir = std::env::temp_dir().join(format!("clank-mcp-live-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("hello.txt"), "the file says this\n").unwrap();

    let server = StdioServer::connect(&filesystem_server(dir.to_str().unwrap()))
        .await
        .expect("the reference filesystem server should connect");

    let tools = server.tools();
    println!("{} tools offered:", tools.len());
    for tool in tools {
        println!("  {:<28} {:<6} {}", tool.name, tool.category, tool.summary);
    }
    assert!(tools.len() > 5, "it exports more than a handful");

    // Namespaced, so the three that collide with built-ins no longer do.
    assert!(tools.iter().all(|t| t.name.starts_with("fs/")));
    clanker_command_center::tools::validate(tools).expect("no collisions after namespacing");

    // Annotations actually bucketed them.
    let reads = tools.iter().filter(|t| t.category == "read").count();
    let writes = tools.iter().filter(|t| t.category == "write").count();
    println!("categorised: {reads} read, {writes} write");
    assert!(reads > 0 && writes > 0);

    // A real call, routed by the namespaced name the model would use.
    let full = "fs/read_text_file";
    let (_, bare) = route(full).unwrap();
    let text = server
        .call(
            bare,
            json!({"path": dir.join("hello.txt").to_str().unwrap()}),
        )
        .await
        .expect("the file is inside the server's allowed directory");
    println!("call result: {text:?}");
    assert!(text.contains("the file says this"));

    // An error from the server is an error here, not an empty success.
    let refused = server
        .call(bare, json!({"path": "/etc/shadow"}))
        .await
        .expect_err("outside the allowed directory");
    println!("refusal: {refused}");

    // The part worth having a live test for: the pid we spawned is `npm
    // exec`, and the server is its child. Shutdown has to reap both.
    let before = ps_matching(dir.to_str().unwrap());
    println!("before shutdown, {} process(es):", before.len());
    for line in &before {
        println!("    {line}");
    }
    assert!(!before.is_empty(), "the server should be running");

    // No sleep afterwards on purpose: `shutdown` is supposed to have
    // waited for the group itself, so a crutch here would hide the bug a
    // caller that quits immediately would hit.
    server.shutdown().await;
    let after = ps_matching(dir.to_str().unwrap());
    println!("after shutdown, {} process(es):", after.len());
    for line in &after {
        println!("    {line}");
    }
    assert!(
        after.is_empty(),
        "shutdown left {} orphan(s) holding the pipe: {after:?}",
        after.len()
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The server processes matching `pattern`, named rather than counted —
/// which of them survived is the useful half.
///
/// Pass something unique to one server, which is why both tests pass their
/// own temp directory: it is on the server's command line and nothing
/// else's. Two earlier versions of this got it wrong in the same way, by
/// matching too widely — first the shell running the test, then the *other*
/// test's server, since these share a process and run at the same time.
///
/// Filtered on the executable too: a server is launched by a node runner,
/// so anything else matching is a bystander.
fn ps_matching(pattern: &str) -> Vec<String> {
    let out = std::process::Command::new("ps")
        .args(["-eo", "pid,ppid,pgid,comm,args"])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|line| line.contains(pattern))
        .filter(|line| {
            let command = line.split_whitespace().nth(3).unwrap_or_default();
            command.starts_with("node") || command.starts_with("npm")
        })
        .map(|line| line.trim().chars().take(110).collect())
        .collect()
}

/// The dispatch chain, which is the one link a fake server cannot test:
/// `tools::execute_tool` is given a namespaced name by the model, finds it
/// in the registry, and routes it to the server that owns it.
#[tokio::test]
#[ignore = "spawns npx"]
async fn a_registered_tool_is_callable_by_the_name_the_model_sees() {
    use clanker_command_center::{mcp, tools};

    let dir = std::env::temp_dir().join(format!("clank-mcp-dispatch-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("note.txt"), "routed all the way through\n").unwrap();

    let started = mcp::connect_all(&[filesystem_server(dir.to_str().unwrap())]).await;
    assert!(
        started.iter().all(|s| s.outcome.is_ok()),
        "the server should have connected"
    );

    // Registered as a side effect of connecting, so the model is offered it.
    let offered: Vec<String> = tools::get_tool_definitions()
        .iter()
        .map(|d| d["function"]["name"].as_str().unwrap().to_string())
        .collect();
    assert!(
        offered.iter().any(|name| name == "fs/read_text_file"),
        "{offered:?}"
    );

    // The actual dispatch: the name the model would emit, straight into
    // the function the agent loop calls.
    let result = tools::execute_tool(
        "fs/read_text_file",
        &json!({"path": dir.join("note.txt").to_str().unwrap()}).to_string(),
        true,
        30,
    )
    .await
    .expect("a registered tool routes to its server");
    println!("execute_tool result: {result}");
    assert_eq!(result["success"], true);
    assert!(result["content"]
        .as_str()
        .unwrap()
        .contains("routed all the way"));

    // A server error comes back as an error, not as a successful call
    // carrying a complaint.
    let refused = tools::execute_tool(
        "fs/read_text_file",
        &json!({"path": "/etc/shadow"}).to_string(),
        true,
        30,
    )
    .await
    .expect_err("outside the allowed directory");
    println!("execute_tool refusal: {refused}");

    // And a name that looks namespaced but belongs to nobody.
    let nobody = tools::execute_tool("ghost/tool", "{}", true, 30)
        .await
        .expect_err("no such tool");
    assert!(nobody.to_string().contains("Unknown tool"), "{nobody}");

    mcp::shutdown_all().await;
    assert!(
        ps_matching(dir.to_str().unwrap()).is_empty(),
        "shutdown_all left something running"
    );
    std::fs::remove_dir_all(&dir).ok();
}
