//! Prepare animated native NPC scenes for browser playback without allocating a GPU.
use std::{
    fs,
    io::{self, BufRead, Write},
    path::PathBuf,
};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use ghostline_red::assets::ResourceKey;
use ghostline_renderer::{AssemblyOptions, GameSource, RootMotion};
use glam::{Mat4, Vec3};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    game: PathBuf,
    #[arg(long)]
    schema: Option<PathBuf>,
    #[arg(long)]
    resource: String,
    #[arg(long)]
    appearance: Option<String>,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 512)]
    size: u32,
    #[arg(long, default_value_t = 12)]
    fps: u32,
    #[arg(long, default_value_t = 5.0)]
    seconds: f32,
    #[arg(long)]
    origin_anchor: Vec<String>,
    #[arg(long)]
    overlay: Vec<PathBuf>,
    #[arg(long)]
    rigid_hair: Option<String>,
}

#[derive(Deserialize)]
struct ClipRequest {
    label: String,
    animset: String,
    animation: String,
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    ensure!(
        (64..=2048).contains(&args.size)
            && (1..=60).contains(&args.fps)
            && args.seconds.is_finite()
            && args.seconds > 0.
            && args.seconds <= 30.,
        "Invalid render limits"
    );
    let source = GameSource::open_with_overlays(&args.game, args.schema.as_deref(), &args.overlay)?;
    let key = ResourceKey::parse(&args.resource);
    let mut reader = source.resources();
    let Some(appearance) = args.appearance.as_deref() else {
        let document = reader.document(&key)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&document["Data"]["RootChunk"]["appearances"])?
        );
        return Ok(());
    };
    fs::create_dir_all(&args.out)?;
    let mut options = AssemblyOptions {
        origin_anchors: args.origin_anchor.iter().cloned().collect(),
        ..AssemblyOptions::default()
    };
    if let Some(name) = &args.rigid_hair {
        options.excluded_components.insert(name.clone());
    }
    let mut assembly =
        source.load_assembly(&key, Some(appearance), &options, &args.out.join("prepared"))?;
    if let Some(name) = &args.rigid_hair {
        let component = assembly
            .native
            .components
            .iter()
            .find(|c| c.name == *name)
            .context("Hair component absent")?;
        let asset = source.load_shared_mesh_resource(
            component.mesh.as_ref().context("Hair mesh absent")?,
            &component.mesh_appearance,
        )?;
        let rig = *assembly.rigs.get("root").context("Body rig absent")?;
        let skeleton = assembly.scene.skeleton(rig)?;
        let head_index = skeleton
            .bone_names()
            .iter()
            .position(|n| n == "Head")
            .context("Head bone absent")?;
        let head = assembly
            .scene
            .bone_node(rig, "Head")
            .context("Head anchor absent")?;
        let inspection = assembly.inspection(&[])?;
        let nodes = inspection["nodes"]
            .as_array()
            .context("No inspection nodes")?;
        let world = |node_name: &str| -> Result<Mat4> {
            let row = nodes
                .iter()
                .find(|n| n["name"] == node_name)
                .context("Placement absent")?;
            let values: [f32; 16] = serde_json::from_value(row["world"].clone())?;
            Ok(Mat4::from_cols_array(&values))
        };
        let head_world =
            world("root")? * skeleton.model_matrices(skeleton.rest_pose())?[head_index];
        let basis = Mat4::from_cols_array(&[
            1., 0., 0., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
        ]);
        let hair_world =
            world(&component.parent.component)? * basis * component.local * basis.transpose();
        let local = head_world.inverse() * hair_world;
        let mesh = assembly.scene.add_shared_asset(asset);
        assembly.scene.add_node(
            ghostline_renderer::Node::new("preview-rigid-hair", Some(mesh))
                .with_parent(head)
                .with_transform(local),
        )?;
        assembly.preview_notes.push(format!("Explicit preview approximation: {name} follows Head rigidly; hair simulation is omitted"));
    }
    fs::write(
        args.out.join("assembly.json"),
        serde_json::to_vec_pretty(&assembly.inspection(&[])?)?,
    )?;
    println!(
        "{}",
        json!({"ready":true,"limitations":assembly.preview_notes})
    );
    io::stdout().flush()?;
    for line in io::stdin().lock().lines() {
        let request: ClipRequest = serde_json::from_str(&line?)?;
        ensure!(
            !request.label.is_empty()
                && request
                    .label
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_'),
            "Invalid clip label"
        );
        let result = (|| -> Result<serde_json::Value> {
            let clip =
                reader.animation(&ResourceKey::parse(&request.animset), &request.animation)?;
            let native_rig = clip.rig.as_ref().context("Animation lacks source rig")?;
            let rig = assembly
                .native
                .components
                .iter()
                .find_map(|c| {
                    (c.rig
                        .as_ref()
                        .is_some_and(|r| r.native_hash() == native_rig.native_hash()))
                    .then(|| assembly.rigs.get(&c.name).copied())
                    .flatten()
                })
                .context("NPC assembly has no matching animation rig")?;
            let duration = clip.duration.min(args.seconds);
            let count = (duration * args.fps as f32).ceil() as u32;
            let times: Vec<_> = (0..=count)
                .map(|i| (i as f32 / args.fps as f32).min(duration))
                .collect();
            let mut posed = assembly.scene.clone();
            let mut minimum = Vec3::splat(f32::INFINITY);
            let mut maximum = Vec3::splat(f32::NEG_INFINITY);
            for i in 0..=8 {
                let pose = posed
                    .skeleton(rig)?
                    .sample_animation(&clip, duration * i as f32 / 8., None, RootMotion::Preserve)?
                    .pose;
                posed.set_pose(rig, &pose)?;
                for chunk in posed.evaluated_geometry()? {
                    for point in chunk.positions {
                        minimum = minimum.min(Vec3::from_array(point));
                        maximum = maximum.max(Vec3::from_array(point));
                    }
                }
            }
            let center = (minimum + maximum) * 0.5;
            let radius = (maximum - minimum).length() * 0.5;
            ensure!(
                center.is_finite() && radius > 0.1 && radius < 10.,
                "Invalid animated browser bounds"
            );
            let bytes = assembly.scene.browser_animation_glb(
                &request.animation,
                &times,
                |scene, time| {
                    let sample = scene.skeleton(rig)?.sample_animation(
                        &clip,
                        time,
                        None,
                        RootMotion::Preserve,
                    )?;
                    scene.set_pose(rig, &sample.pose)
                },
            )?;
            let output = args.out.join(&request.label);
            fs::create_dir_all(&output)?;
            fs::write(output.join("scene.glb"), &bytes)?;
            Ok(
                json!({"label":request.label,"animation":request.animation,"duration":duration,"fps":args.fps,"bounds":{"center":center.to_array(),"radius":radius},"samples":times.len(),"bytes":bytes.len(),"glb_sha256":format!("{:x}",Sha256::digest(&bytes)),"source_identity":source.content_identity(),"build":ghostline_renderer::BUILD_ID,"limitations":assembly.preview_notes}),
            )
        })();
        match result {
            Ok(value) => println!("{}", json!({"ok":true,"result":value})),
            Err(error) => println!(
                "{}",
                json!({"ok":false,"label":request.label,"error":format!("{error:#}")})
            ),
        }
        io::stdout().flush()?;
    }
    Ok(())
}
