//! Inspect native morph shapes and animation samples without preparing materials.

use std::{fs, path::PathBuf};

use anyhow::{Result, ensure};
use clap::Parser;
use ghostline_red::assets::ResourceKey;
use ghostline_renderer::GameSource;
use serde_json::{Value, json};

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    game: PathBuf,
    #[arg(long)]
    schema: Option<PathBuf>,
    #[arg(long)]
    morph: Option<String>,
    #[arg(long)]
    geometry: Option<String>,
    #[arg(long)]
    animset: Option<String>,
    #[arg(long, requires = "animset")]
    animation: Option<String>,
    #[arg(long)]
    out: PathBuf,
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    ensure!(!args.out.exists(), "output already exists");
    let source = GameSource::open(&args.game, args.schema.as_deref())?;
    let mut resources = source.resources();
    let mut report = serde_json::Map::new();
    if let Some(path) = &args.geometry {
        let geometry = source.load_geometry(&ResourceKey::parse(path))?;
        report.insert("geometry".into(),json!({"resource":path,"appearances":geometry.appearance_names(),"chunks":geometry.chunks().len(),"morph_targets":geometry.morph_targets().len(),"shadow_only":geometry.is_shadow_only()}));
    }
    if let Some(path) = &args.morph {
        let mesh = resources.morph_mesh(&ResourceKey::parse(path))?;
        let names: Vec<_> = mesh
            .targets
            .iter()
            .map(|t| json!({"name":t.name,"region":t.region}))
            .collect();
        report.insert("morph".into(), json!({"resource":path,"base_mesh":mesh.base_mesh.to_string(),"chunks":mesh.chunks.len(),"targets":names}));
    }
    if let Some(path) = &args.animset {
        let requested = args
            .animation
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--animation required"))?;
        let clip = resources.animation(&ResourceKey::parse(path), requested)?;
        let rig_key = clip
            .rig
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("animation set has no rig identity"))?;
        let rig = resources.rig(rig_key)?;
        let reference: Vec<_> = rig.bones.iter().map(|b| b.rest).collect();
        let samples = [0., 0.5, 1.]
            .map(|fraction| clip.sample(clip.duration * fraction, &reference, None))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        report.insert("animation".into(), json!({"resource":path,"name":clip.name,"kind":clip.kind,"buffer_type":clip.buffer_type,"rig":rig_key.to_string(),"duration":clip.duration,"frames":clip.frame_count,"joints":rig.bones.len(),"extra_joints":clip.extra_joints,"tracks":clip.tracks.len(),"samples":samples.iter().map(|s| s.local.len()).collect::<Vec<_>>(),"motion_extraction":clip.motion.is_some()}));
    }
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    let report = Value::Object(report);
    fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
    println!("Wrote {}", args.out.display());
    Ok(())
}
