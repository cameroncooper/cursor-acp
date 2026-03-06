#![deny(clippy::print_stdout, clippy::print_stderr)]

use agent_client_protocol::AgentSideConnection;
use std::io::Result as IoResult;
use std::sync::Arc;
use tokio::task::LocalSet;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use tracing_subscriber::EnvFilter;

pub mod acp;
pub mod auth;
pub mod cli;
pub mod error;
pub mod prompt;
pub mod session;

pub async fn run_main() -> IoResult<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let agent = Arc::new(acp::CursorAcpAgent::new());
    let stdin = tokio::io::stdin().compat();
    let stdout = tokio::io::stdout().compat_write();

    LocalSet::new()
        .run_until(async move {
            let (connection, io_task) =
                AgentSideConnection::new(agent.clone(), stdout, stdin, |fut| {
                    tokio::task::spawn_local(fut);
                });
            agent.attach_connection(Arc::new(connection));

            io_task
                .await
                .map_err(|error| std::io::Error::other(format!("ACP I/O error: {error}")))
        })
        .await?;

    Ok(())
}
