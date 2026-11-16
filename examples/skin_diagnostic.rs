//! Export native mesh skin inputs for independent deformation diagnostics.
use anyhow::Result;
use clap::Parser;
use ghostline_red::assets::ResourceKey;
use ghostline_renderer::GameSource;
use serde_json::json;
use std::{fs, path::PathBuf};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    game: PathBuf,
    #[arg(long)]
    schema: Option<PathBuf>,
    #[arg(long)]
    resource: String,
    #[arg(long)]
    overlay: Vec<PathBuf>,
    #[arg(long)]
    out: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let source = GameSource::open_with_overlays(&a.game, a.schema.as_deref(), &a.overlay)?;
    let key = ResourceKey::parse(&a.resource);
    fs::create_dir_all(&a.out)?;
    fs::write(
        a.out.join("resource.json"),
        serde_json::to_vec(&*source.resources().document(&key)?)?,
    )?;
    if a.resource.ends_with(".rig") {
        return Ok(());
    }
    let geometry = source.load_geometry(&key)?;
    let skin = geometry.skin();
    let chunks:Vec<_>=geometry.chunks().iter().map(|c|json!({"index":c.chunk_index,"positions":c.positions,"indices":c.indices,"joints":c.skin.iter().map(|v|v.joints).collect::<Vec<_>>(),"weights":c.skin.iter().map(|v|v.weights).collect::<Vec<_>>() })).collect();
    fs::write(
        a.out.join("skin.json"),
        serde_json::to_vec(
            &json!({"names":skin.map(|s|&s.names),"inverse_bind":skin.map(|s|s.inverse_bind.iter().map(|m|m.to_cols_array()).collect::<Vec<_>>()),"chunks":chunks}),
        )?,
    )?;
    Ok(())
}
