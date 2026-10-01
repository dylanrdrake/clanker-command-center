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
    let before = ps_matching("server-filesystem");
    println!("before shutdown, {} process(es):", before.len());
    for line in &before {
        println!("    {line}");
    }
    assert!(!before.is_empty(), "the server should be running");

    // No sleep afterwards on purpose: `shutdown` is supposed to have
    // waited for the group itself, so a crutch here would hide the bug a
    // caller that quits immediately would hit.
    server.shutdown().await;
    let after = ps_matching("server-filesystem");
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

/// Every process whose command line mentions `pattern`, named rather than
/// counted — which of them survived is the useful half.
fn ps_matching(pattern: &str) -> Vec<String> {
    let out = std::process::Command::new("ps")
        .args(["-eo", "pid,ppid,pgid,args"])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|line| line.contains(pattern) && !line.contains("ps -eo"))
        .map(|line| line.trim().chars().take(110).collect())
        .collect()
}
