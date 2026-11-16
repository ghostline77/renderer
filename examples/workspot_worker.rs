//! Render exact native clips on complete entity appearances with retained GPU assets.
use std::{
    fs,
    io::{self, BufRead, Write},
    path::PathBuf,
};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use ghostline_red::assets::ResourceKey;
use ghostline_renderer::{
    AssemblyOptions, Camera, Cancellation, FrameRequest, GameSource, RenderOptions, Renderer,
    RootMotion,
};
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
    let renderer = pollster::block_on(Renderer::new())?;
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
    let mut prepared = renderer.prepare_scene(&assembly.scene)?;
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
            let mut minimum = Vec3::splat(f32::INFINITY);
            let mut maximum = Vec3::splat(f32::NEG_INFINITY);
            for i in 0..=8 {
                let pose = assembly
                    .scene
                    .skeleton(rig)?
                    .sample_animation(&clip, duration * i as f32 / 8., None, RootMotion::Preserve)?
                    .pose;
                assembly.scene.set_pose(rig, &pose)?;
                for chunk in assembly.scene.evaluated_geometry()? {
                    for p in chunk.positions {
                        let p = Vec3::from_array(p);
                        minimum = minimum.min(p);
                        maximum = maximum.max(p);
                    }
                }
            }
            ensure!(
                minimum.is_finite() && maximum.is_finite(),
                "Invalid posed bounds"
            );
            let center = (minimum + maximum) * 0.5;
            let radius = (maximum - minimum).length() * 0.5;
            ensure!(
                radius > 0.1 && radius < 10.,
                "Implausible NPC bounds: {radius}"
            );
            let camera = Camera::Orthographic {
                eye: (center + Vec3::new(-0.65, 0.25, -1.).normalize() * radius * 4.).to_array(),
                target: center.to_array(),
                up: [0., 1., 0.],
                half_height: radius * 1.08,
                near: 0.01,
                far: radius * 12.,
            };
            let frames: Vec<_> = (0..(duration * args.fps as f32).ceil() as u32)
                .map(|i| FrameRequest {
                    time: i as f32 / args.fps as f32,
                    render: RenderOptions {
                        size: args.size,
                        camera,
                    },
                })
                .collect();
            let identity = serde_json::to_string(
                &json!({"source":source.content_identity(),"entity":args.resource,"appearance":appearance,"animset":request.animset,"animation":request.animation,"root_motion":"preserve","example_sha256":format!("{:x}", Sha256::digest(include_bytes!("workspot_worker.rs")))}),
            )?;
            let report = renderer.render_sequence(
                &mut prepared,
                &frames,
                &args.out.join(&request.label),
                &identity,
                &Cancellation::default(),
                |scene, time| {
                    scene.apply_animation(rig, &clip, time, None, RootMotion::Preserve)?;
                    Ok(())
                },
            )?;
            Ok(
                json!({"label":request.label,"animation":request.animation,"animset":request.animset,"source_duration":clip.duration,"camera":camera,"report":report,"limitations":assembly.preview_notes,"fps":args.fps,"source_identity":source.content_identity()}),
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
