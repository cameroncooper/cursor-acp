use anyhow::Result;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    cursor_acp::run_main().await?;
    Ok(())
}
