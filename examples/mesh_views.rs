//! Prepare appearances with one source and render views with one GPU device.

use std::{fs, path::PathBuf, sync::Arc};

use anyhow::{Result, ensure};
use clap::Parser;
use ghostline_renderer::{Camera, GameSource, Node, RenderOptions, Renderer, Scene, View};
use serde_json::json;
use sha2::{Digest, Sha256};

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    game: PathBuf,
    #[arg(long)]
    schema: Option<PathBuf>,
    #[arg(long)]
    mesh: String,
    #[arg(long, required = true)]
    appearance: Vec<String>,
    #[arg(long)]
    out: PathBuf,
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    ensure!(!args.out.exists(), "output directory already exists");
    let source = GameSource::open(&args.game, args.schema.as_deref())?;
    let assets = args
        .appearance
        .iter()
        .enumerate()
        .map(|(index, appearance)| {
            source
                .load_mesh(
                    &args.mesh,
                    appearance,
                    &args.out.join(index.to_string()).join("prepared"),
                )
                .map(Arc::new)
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Geometry and prepared maps belong to the asset, independently of the index.
    drop(source);
    let renderer = pollster::block_on(Renderer::new())?;
    let mut reports = Vec::new();
    for (index, asset) in assets.iter().enumerate() {
        let directory = args.out.join(index.to_string());
        let mut scene = Scene::new();
        let mesh = scene.add_shared_asset(Arc::clone(asset));
        scene.add_node(Node::new("mesh", Some(mesh)))?;
        let mut prepared = renderer.prepare_scene(&scene)?;
        let mut receipts = Vec::new();
        for (name, view) in [
            ("hero", View::Hero),
            ("front", View::Front),
            ("back", View::Back),
        ] {
            let result = renderer.render(
                &mut prepared,
                RenderOptions {
                    size: 512,
                    camera: Camera::Frame(view),
                },
            )?;
            ensure!(
                result.image.pixels().any(|p| p.0[3] > 0),
                "empty {name} render"
            );
            result.image.save(directory.join(format!("{name}.png")))?;
            let repeated = renderer.render(
                &mut prepared,
                RenderOptions {
                    size: 512,
                    camera: Camera::Frame(view),
                },
            )?;
            ensure!(
                result.image == repeated.image,
                "repeated {name} render changed pixels"
            );
            receipts.push(json!({"view":name,"pixel_sha256":format!("{:x}",Sha256::digest(result.image.as_raw())),
            "repeated_pixels_match":true,"gpu":result.adapter.name,"backend":format!("{:?}",result.adapter.backend)}));
        }
        reports.push(json!({"mesh":asset.mesh,"appearance":asset.appearance_name,"source_sha256":asset.source_sha256,"views":receipts}));
    }
    let report = json!({"appearances":reports});
    fs::write(
        args.out.join("views.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
