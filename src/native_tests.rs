//! Game-fixture checks for native morphs, animation and assembled vehicles.

use std::{collections::HashMap, fs, path::PathBuf, sync::Arc};

use ghostline_red::{
    assets::ResourceKey,
    pbr::{BakeSummary, BakedAppearance, bake_appearance},
};
use glam::{Mat4, Vec3};

use crate::*;

fn source() -> GameSource {
    GameSource::open(
        &PathBuf::from(std::env::var_os("CP2077_GAME_ROOT").expect("CP2077_GAME_ROOT")),
        std::env::var_os("RED_SCHEMA_PATH")
            .map(PathBuf::from)
            .as_deref(),
    )
    .unwrap()
}

fn output(name: &str) -> PathBuf {
    let output = PathBuf::from("work/native-validation")
        .join(&BUILD_ID[..12])
        .join(name);
    fs::create_dir_all(&output).unwrap();
    output
}

fn fixed_camera(points: impl Iterator<Item = Vec3>) -> Camera {
    let (center, radius) = crate::scene::transformed_bounds(points).unwrap();
    let eye = center + Vec3::new(-0.1, 0.08, -1.).normalize() * radius * 4.;
    Camera::Orthographic {
        eye: eye.to_array(),
        target: center.to_array(),
        up: [0., 1., 0.],
        half_height: radius * 1.12,
        near: radius * 0.01,
        far: radius * 10.,
    }
}

#[test]
#[ignore = "requires native game archive fixtures and a GPU"]
fn native_head_morphs_render_with_explicit_clay_material() {
    let source = source();
    let geometry = source.load_geometry(&ResourceKey::parse("base/characters/head/player_base_heads/player_man_average/h0_000_pma__morphs.morphtarget")).unwrap();
    let camera = fixed_camera(
        geometry
            .chunks()
            .iter()
            .flat_map(|c| c.positions.iter().copied().map(Vec3::from)),
    );
    let targets = geometry.morph_targets().len();
    let nose = geometry
        .morph_targets()
        .iter()
        .rposition(|t| t.region == "nose")
        .unwrap();
    let material = crate::gpu::tests::triangle().appearance.materials[0].clone();
    let mut material = material;
    material.base_color_factor = [0.6, 0.6, 0.6, 1.];
    let asset = geometry
        .with_materials(
            "diagnostic-clay",
            BakedAppearance {
                chunk_materials: vec![material.name.clone()],
                materials: vec![material],
                summary: BakeSummary::default(),
            },
        )
        .unwrap();
    let mut scene = Scene::new();
    let mesh = scene.add_asset(asset);
    let node = scene.add_node(Node::new("head", Some(mesh))).unwrap();
    let renderer = pollster::block_on(Renderer::new()).unwrap();
    let mut prepared = renderer.prepare_scene(&scene).unwrap();
    let options = RenderOptions { size: 512, camera };
    let rest = renderer.render(&mut prepared, options).unwrap().image;
    let mut weights = vec![0.; targets];
    weights[nose] = 1.;
    prepared.set_morph_weights(node, &weights).unwrap();
    let changed = renderer.render(&mut prepared, options).unwrap().image;
    assert_ne!(rest, changed);
    assert!(rest.pixels().any(|p| p.0[3] > 0));
    let output = output("head-morph");
    rest.save(output.join("base.png")).unwrap();
    changed.save(output.join("nose.png")).unwrap();
    println!(
        "native morph targets={targets}; selected={nose}; images={}",
        output.display()
    );
}

#[test]
#[ignore = "requires native game archive fixtures and a GPU"]
fn native_clip_drives_retained_rig_frames() {
    let source = source();
    let mut reader = source.resources();
    let clip = reader.animation(&ResourceKey::parse("base/animations/npc/generic_characters/male_average/open_world/generic_male_average_work__stand__rh_phone__03.anims"),"stand__rh_phone__03__talk__01").unwrap();
    let native = reader.rig(clip.rig.as_ref().unwrap()).unwrap();
    let skeleton = Arc::new(Skeleton::from_rig(&native).unwrap());
    let times = [0., clip.duration * 0.5, clip.duration];
    let points: Vec<_> = times
        .iter()
        .flat_map(|&time| {
            let pose = skeleton
                .sample_animation(&clip, time, None, RootMotion::Preserve)
                .unwrap()
                .pose;
            skeleton
                .model_matrices(&pose)
                .unwrap()
                .into_iter()
                .flat_map(|m| {
                    [
                        m.w_axis.truncate() - Vec3::splat(0.05),
                        m.w_axis.truncate() + Vec3::splat(0.05),
                    ]
                })
        })
        .collect();
    let camera = fixed_camera(points.into_iter());
    let mut scene = Scene::new();
    let root = scene.add_node(Node::new("actor", None)).unwrap();
    let rig = scene.add_rig(root, Arc::clone(&skeleton)).unwrap();
    let marker = scene.add_asset(crate::gpu::tests::triangle());
    for name in skeleton.bone_names() {
        let parent = scene.bone_node(rig, name).unwrap();
        scene
            .add_node(
                Node::new(name, Some(marker))
                    .with_parent(parent)
                    .with_transform(Mat4::from_scale(Vec3::splat(0.025))),
            )
            .unwrap();
    }
    let renderer = pollster::block_on(Renderer::new()).unwrap();
    let mut prepared = renderer.prepare_scene(&scene).unwrap();
    let frames: Vec<_> = times
        .into_iter()
        .map(|time| FrameRequest {
            time,
            render: RenderOptions { size: 512, camera },
        })
        .collect();
    let output = output("native-animation").join("frames");
    let report = renderer
        .render_sequence(
            &mut prepared,
            &frames,
            &output,
            &source.content_identity(),
            &Cancellation::default(),
            |scene, time| {
                scene.apply_animation(rig, &clip, time, None, RootMotion::Preserve)?;
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(report.written + report.reused, 3);
    assert_ne!(
        fs::read(output.join("000000.png")).unwrap(),
        fs::read(output.join("000001.png")).unwrap()
    );
    println!(
        "native clip format={} bones={} duration={} images={}",
        clip.buffer_type,
        native.bones.len(),
        clip.duration,
        output.display()
    );
}

#[test]
#[ignore = "requires recorded native material fixtures, game archives and a GPU"]
fn native_vehicle_assembly_uses_recorded_material_fixtures() {
    let source = source();
    let fixtures = PathBuf::from(
        std::env::var_os("GHOSTLINE_PREPARED_FIXTURES").expect("GHOSTLINE_PREPARED_FIXTURES"),
    );
    let options: AssemblyOptions = AssemblyOptions {
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
        bind_pose: false,
        standalone_bind_pose_fallback: false,
        references: Vec::new(),
        binding_contexts: Vec::new(),
        neutral_references: Vec::new(),
        neutral_assemblies: Vec::new(),
        mesh_appearances: serde_json::from_slice(
            &fs::read(
                std::env::var_os("GHOSTLINE_COMPONENT_APPEARANCES")
                    .expect("GHOSTLINE_COMPONENT_APPEARANCES"),
            )
            .unwrap(),
        )
        .unwrap(),
    };
    let mut recorded = HashMap::new();
    for directory in fs::read_dir(fixtures).unwrap() {
        let directory = directory.unwrap().path();
        let receipt = directory.join("source.json");
        if !receipt.exists() {
            continue;
        }
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(receipt).unwrap()).unwrap();
        recorded.insert(
            (
                receipt["mesh"].as_str().unwrap().to_owned(),
                receipt["appearance"].as_str().unwrap().to_owned(),
            ),
            (
                directory,
                receipt["source_sha256"].as_str().unwrap().to_owned(),
            ),
        );
    }
    let native = source
        .resolve_assembly(
            &ResourceKey::parse("base/vehicles/sport/v_sport2_porsche_911turbo__basic_01.ent"),
            Some("porsche_911turbo__basic_johnny"),
        )
        .unwrap();
    let assembly = SceneAssembly::from_native(native, &options, |resource, appearance| {
        let geometry = source.load_geometry(resource)?;
        let (directory, hash) = recorded
            .get(&(geometry.mesh.clone(), appearance.to_owned()))
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "missing recorded material fixture {resource}/{appearance}"
                ))
            })?;
        if geometry.source_sha256 != *hash {
            return Err(Error::InvalidInput(
                "recorded geometry source hash changed".into(),
            ));
        }
        let indices: Vec<_> = geometry.chunks().iter().map(|c| c.chunk_index).collect();
        let material = bake_appearance(
            &directory.join("appearance.Material.json"),
            appearance,
            &indices,
            Some(512),
        )?;
        geometry.with_materials(appearance, material).map(Arc::new)
    })
    .unwrap();
    let renderer = pollster::block_on(Renderer::new()).unwrap();
    let mut prepared = renderer.prepare_scene(&assembly.scene).unwrap();
    let output = output("assembled-porsche");
    for (name, view) in [
        ("hero", View::Hero),
        ("front", View::Front),
        ("back", View::Back),
    ] {
        let image = renderer
            .render(
                &mut prepared,
                RenderOptions {
                    size: 512,
                    camera: Camera::Frame(view),
                },
            )
            .unwrap()
            .image;
        assert!(image.pixels().any(|p| p.0[3] > 0));
        image.save(output.join(format!("{name}.png"))).unwrap();
    }
    println!(
        "native vehicle {:?}; rigs={}; images={}",
        prepared.statistics(),
        assembly.rigs.len(),
        output.display()
    );
}
