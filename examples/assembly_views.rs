//! Resolve a native entity/appearance and optionally render the assembled scene.

use std::{fs, path::PathBuf};

use anyhow::{Result, ensure};
use clap::Parser;
use ghostline_red::assets::ResourceKey;
use ghostline_renderer::{AssemblyOptions, Camera, GameSource, RenderOptions, Renderer, View};
use serde_json::json;

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
    #[arg(long)]
    native_only: bool,
    #[arg(long)]
    component_appearances: Option<PathBuf>,
    #[arg(long)]
    bind_pose: bool,
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    ensure!(!args.out.exists(), "output directory already exists");
    let source = GameSource::open(&args.game, args.schema.as_deref())?;
    let key = ResourceKey::parse(&args.resource);
    if args.native_only {
        let native = source.resolve_assembly(&key, args.appearance.as_deref())?;
        fs::create_dir_all(&args.out)?;
        let rigs: Vec<_> = native
            .rigs
            .iter()
            .map(|(key, rig)| json!({"resource":key.to_string(),"bones":rig.bones.len()}))
            .collect();
        let report = json!({"resource":key.to_string(),"components":native.components.len(),"meshes":native.components.iter().filter(|c| c.mesh.is_some()).count(),"rigs":rigs});
        fs::write(
            args.out.join("assembly.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        println!("{report}");
    } else {
        let options = AssemblyOptions {
            mesh_lods: Default::default(),
            origin_anchors: Default::default(),
            slot_rig_overrides: Default::default(),
            duplicate_components: Default::default(),
            resource_overrides: Default::default(),
            excluded_components: Default::default(),
            slot_bone_overrides: Default::default(),
            parent_component_overrides: Default::default(),
            omit_unbound_duplicates: false,
            visibility_parameters: Default::default(),
            bind_pose: args.bind_pose,
            standalone_bind_pose_fallback: false,
            references: Vec::new(),
            binding_contexts: Vec::new(),
            neutral_references: Vec::new(),
            neutral_assemblies: Vec::new(),
            mesh_appearances: args
                .component_appearances
                .as_ref()
                .map(|p| -> Result<_> { Ok(serde_json::from_slice(&fs::read(p)?)?) })
                .transpose()?
                .unwrap_or_default(),
        };
        let assembly = source.load_assembly(
            &key,
            args.appearance.as_deref(),
            &options,
            &args.out.join("prepared"),
        )?;
        let renderer = pollster::block_on(Renderer::new())?;
        let mut scene = renderer.prepare_scene(&assembly.scene)?;
        for (name, view) in [
            ("hero", View::Hero),
            ("front", View::Front),
            ("back", View::Back),
        ] {
            let result = renderer.render(
                &mut scene,
                RenderOptions {
                    size: 512,
                    camera: Camera::Frame(view),
                },
            )?;
            ensure!(
                result.image.pixels().any(|p| p.0[3] != 0),
                "empty assembly render"
            );
            result.image.save(args.out.join(format!("{name}.png")))?;
        }
        println!("{:?}", scene.statistics());
    }
    Ok(())
}
