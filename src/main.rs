//! Render a mesh appearance from installed game archives.

use std::{fs, path::PathBuf, sync::Arc, time::Instant};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use ghostline_renderer::{Camera, GameSource, Node, RenderOptions, Renderer, Scene, View};
use serde_json::json;
use sha2::{Digest, Sha256};

#[derive(Parser)]
#[command(about = "Render a native Cyberpunk mesh appearance to PNG")]
struct Arguments {
    /// Game installation containing archive/pc/content and optional archive/pc/ep1.
    #[arg(long)]
    game: PathBuf,
    /// Reflected RED schema JSON.
    #[arg(long)]
    schema: Option<PathBuf>,
    /// Exact native mesh depot path.
    #[arg(long)]
    mesh: String,
    /// Exact authored mesh appearance.
    #[arg(long)]
    appearance: String,
    /// New preparation directory outside the game installation.
    #[arg(long)]
    work: PathBuf,
    /// PNG destination; must not already exist.
    #[arg(long)]
    out: PathBuf,
    /// Square output dimensions in pixels.
    #[arg(long, default_value_t = 512, value_parser = clap::value_parser!(u32).range(64..=4096))]
    size: u32,
    /// Camera view.
    #[arg(long, value_enum, default_value_t = View::Hero)]
    view: View,
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    ensure!(
        args.out
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("png")),
        "output must be a PNG path"
    );
    ensure!(
        !args.out.exists(),
        "output already exists: {}",
        args.out.display()
    );
    let receipt_path = args.out.with_extension("render.json");
    ensure!(!receipt_path.exists(), "render receipt already exists");
    let started = Instant::now();
    eprintln!("Reading native geometry and materials...");
    let source = GameSource::open(&args.game, args.schema.as_deref())
        .context("game source initialization failed")?;
    let asset = Arc::new(
        source
            .load_mesh(&args.mesh, &args.appearance, &args.work)
            .context("mesh preparation failed")?,
    );
    let preparation_seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "Rendering {} chunks / {} triangles...",
        asset.chunk_count(),
        asset.triangle_count()
    );
    let renderer = pollster::block_on(Renderer::new())?;
    let mut scene = Scene::new();
    let mesh = scene.add_shared_asset(Arc::clone(&asset));
    scene.add_node(Node::new("mesh", Some(mesh)))?;
    let mut prepared = renderer.prepare_scene(&scene)?;
    let result = renderer.render(
        &mut prepared,
        RenderOptions {
            size: args.size,
            camera: Camera::Frame(args.view),
        },
    )?;
    let visible_pixels = result.image.pixels().filter(|p| p.0[3] > 0).count();
    ensure!(visible_pixels > 0, "render contains no visible pixels");
    if let Some(parent) = args.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    result.image.save(&args.out)?;
    let receipt = json!({
        "mesh": asset.mesh, "appearance": asset.appearance_name, "mesh_sha256": asset.source_sha256,
        "schema_sha256": source.schema_sha256(),
        "chunks": asset.chunk_count(), "triangles": asset.triangle_count(),
        "size": args.size, "view": format!("{:?}", args.view).to_lowercase(),
        "visible_pixels": visible_pixels,
        "pixel_sha256": format!("{:x}", Sha256::digest(result.image.as_raw())),
        "gpu": result.adapter.name, "backend": format!("{:?}", result.adapter.backend),
        "driver": result.adapter.driver_info,
        "preparation_seconds": preparation_seconds, "total_seconds": started.elapsed().as_secs_f64(),
        "material_notes": asset.material_notes().collect::<Vec<_>>(),
    });
    fs::write(receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
