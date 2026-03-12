use anyhow::Result;
use std::io::Write;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        return Ok(());
    }
    if args.iter().any(|a| a == "-V" || a == "--version") {
        print_version();
        return Ok(());
    }

    cursor_acp::run_main().await?;
    Ok(())
}

fn print_version() {
    let mut out = std::io::stdout().lock();
    drop(writeln!(out, "cursor-acp {}", env!("CARGO_PKG_VERSION")));
}

fn print_help() {
    let mut out = std::io::stdout().lock();
    let text = format!(
        "\
cursor-acp {version}

ACP proxy for Cursor CLI.

Reads newline-delimited JSON-RPC from stdin and writes JSON-RPC responses/notifications to stdout.

Usage:
  cursor-acp
  cursor-acp --help
  cursor-acp --version

Environment:
  CURSOR_AGENT_BIN / CURSOR_AGENT_PATH   Path to `cursor-agent` binary (optional)
",
        version = env!("CARGO_PKG_VERSION")
    );
    drop(out.write_all(text.as_bytes()));
}
