//! Prepare and render a bounded, resumable catalogue manifest.

use std::{fs, path::PathBuf};

use anyhow::{Result, ensure};
use clap::Parser;
use ghostline_renderer::{BatchJob, BatchLimits, Cancellation, GameSource, Renderer};

#[derive(Parser)]
#[command(about = "Render a native mesh/entity catalogue manifest with validated resume")]
struct Arguments {
    /// Game installation containing native archives.
    #[arg(long)]
    game: PathBuf,
    /// Reflected RED schema.
    #[arg(long)]
    schema: Option<PathBuf>,
    /// JSON array of BatchJob requests.
    #[arg(long)]
    manifest: PathBuf,
    /// Output directory outside the game installation.
    #[arg(long)]
    out: PathBuf,
    /// Stop between preparation or frame work units when this file exists.
    #[arg(long)]
    cancel_file: Option<PathBuf>,
    /// Maximum decoded vertices resident in a job scene.
    #[arg(long, default_value_t = 2_000_000)]
    max_vertices: usize,
    /// Maximum number of jobs in the manifest.
    #[arg(long, default_value_t = 10_000)]
    max_jobs: usize,
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    ensure!(
        fs::metadata(&args.manifest)?.len() <= 16 * 1024 * 1024,
        "manifest exceeds 16 MiB"
    );
    let jobs: Vec<BatchJob> = serde_json::from_slice(&fs::read(&args.manifest)?)?;
    let source = GameSource::open(&args.game, args.schema.as_deref())?;
    let renderer = pollster::block_on(Renderer::new())?;
    let cancel = args
        .cancel_file
        .map(Cancellation::with_file)
        .unwrap_or_default();
    let limits = BatchLimits {
        jobs: args.max_jobs,
        vertices_per_scene: args.max_vertices,
        ..BatchLimits::default()
    };
    let report = renderer.run_batch(&source, &jobs, &args.out, limits, &cancel, |result| {
        match serde_json::to_string(result) {
            Ok(json) => println!("{json}"),
            Err(error) => eprintln!("progress serialization failed: {error}"),
        }
    })?;
    ensure!(
        !report.jobs.iter().any(|job| job.error.is_some()),
        "one or more jobs failed; see batch.json"
    );
    if report.cancelled {
        eprintln!("Cancelled; committed frames can be resumed with the same manifest.");
    }
    Ok(())
}
