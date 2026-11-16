# Renderer API and command reference

Rust library and command-line tools for native Cyberpunk 2077 asset previews.
It assembles shared GPU scenes, samples skeletal animation and morphs, and writes
transparent PNGs and timestamped frame sequences.

`Scene::browser_glb` exports visible native chunks with their evaluated skeletal
placement and baked PBR materials. `PreparedScene::browser_glb` additionally applies
current morph weights and material overrides. Both embed all texture bytes in a
single GLB, preserve UV transforms and material-slot wrapping, and leave source
resources untouched. Browser shading approximates the native shader: strand flow
lighting and receiving-surface decal blending are not reproduced by standard PBR.

## Build and runtime inputs

Use Rust 1.92 or newer, with `renderer` and `ghostline-red` checkouts beside each
other. Run `cargo build --release --locked` from the renderer directory.

Windows executables are `target/release/ghostline-renderer.exe` and
`target/release/ghostline-render-batch.exe`. Runtime inputs are a local game
installation, a DirectX 12 or Vulkan
GPU driver, and writable preparation/output directories outside the game.

Reflection comes from `ghostline-red`'s bundled game-derived schema and the
selected installation. `--schema` is an optional advanced override. Library
callers use `GameSource::open(game_root, None)` for automatic reflection.

## Asset flow

| Stage | Owner | Result |
| --- | --- | --- |
| Index selected archives and resolve depot paths or hashes | `ghostline-red` | Native resource bytes and payload identities |
| Decode CR2W, entity/appearance parts, rigs and animation | `ghostline-red` | Typed components, bones, clips and resource references |
| Decode mesh streams and embedded morph geometry | `ghostline-red` | Render chunks, skin weights, inverse binds and sparse targets |
| Resolve materials and extract/bake texture dependencies | `ghostline-red` | Exact appearance bindings, PBR maps and conversion notes |
| Assemble visual components, attachments and rig drivers | `renderer` | CPU scenes with shared assets and editable instances |
| Prepare GPU assets and render/update frames | `renderer` | Retained scenes, RGBA images and output receipts |

`GameSource::open` selects archives in `archive/pc/content` and `archive/pc/ep1`
and retains their index and reflection schema. Reuse it across loads; reopen it
after installing a game patch. `GameSource::resources` creates a cached native
resource reader over that index for advanced asset inspection.

`GameSource::load_geometry` decodes geometry without preparing material maps.
For a `.morphtarget`, embedded base geometry determines the shape, while its
referenced `baseMesh` supplies material definitions and the bone palette.
`GeometryAsset::appearance_names` and `morph_targets` expose authored choices.
`with_materials` attaches explicitly prepared maps, including caller-managed
caches or diagnostic materials.

`load_mesh_resource` prepares geometry and maps for an exact appearance, accepting
`ResourceKey::Path` or `ResourceKey::Hash`. `load_mesh` accepts a depot spelling.
Preparation directories must be new; they contain extracted dependencies, baked
maps and source receipts. Loaded assets remain usable after the source is dropped.

## Render one appearance

```powershell
./target/release/ghostline-renderer.exe `
  --game "path/to/Cyberpunk 2077" `
  --mesh "ep1/characters/garment/player_equipment/torso/t2_157_jacket__gambeson/t2_157_pwa_jacket__gambeson.mesh" `
  --appearance gold_2 --work work/gambeson --out work/gambeson.png `
  --view hero --size 512
```

Views are `hero`, `front` and `back`; sizes range from 64 to 4096 pixels. The CLI
renders bind-pose geometry with 2× supersampling. Its `.render.json` records
source/schema hashes, geometry counts, camera, GPU/backend/driver, pixel hash,
timings and material notes. Existing output images and receipts are rejected.

Materials use base color, metallic/roughness, tangent-space normals, emission and
OPAQUE/MASK/BLEND coverage. Template coverage is defined by
`ghostline_red::pbr::SUPPORTED_TEMPLATES`. Unsupported templates return errors;
conversion approximations are recorded in material notes. Morph geometry deltas
modify positions, normals and tangents. Native morph texture-difference layers
are not composited into the PBR maps.

Character templates include skin albedo/tint/normal/roughness, eye iris gradients,
eye-shadow coverage, hair profile gradients and strand alpha. These are static PBR
previews: subsurface scattering, refraction and animated wrinkle layers remain conversion
notes. Hair uses the approximate strand-lighting model described below.

Static conversion also supports blendable/Blackwall skin, eyes and glass, skin-morph
endpoints, hair proxies and cyberspace hair, animated decals, transparent emission
and holographic masks. Native coverage, texture bindings and surface inputs are
retained; runtime transitions and exposure approximations are recorded in each
material's notes. Holographic masks use their complete diffuse texture and alpha
silhouette rather than treating the glitch divisions as atlas dimensions.

Nonfinite primary UVs in otherwise usable native geometry receive an explicit
preview repair from a finite duplicate-position or adjacent vertex. The source
bytes, vertices and triangles are preserved, and material notes report each
repair. Missing neighbours, invalid geometry and missing surface textures still
return errors.

## Scenes and live updates

```rust
use ghostline_renderer::{Camera, MeshAsset, Node, RenderOptions, Renderer, Scene, View};

fn preview(asset: MeshAsset) -> Result<(), ghostline_renderer::Error> {
    let mut scene = Scene::new();
    let mesh = scene.add_asset(asset);
    let node = scene.add_node(Node::new("jacket", Some(mesh)))?;
    let renderer = pollster::block_on(Renderer::new())?;
    let mut prepared = renderer.prepare_scene(&scene)?;
    let _hero = renderer.render(&mut prepared, RenderOptions::default())?;
    prepared.set_chunk_mask(node, 0b11)?;
    let _front = renderer.render(
        &mut prepared,
        RenderOptions { size: 512, camera: Camera::Frame(View::Front) },
    )?;
    Ok(())
}
```

Nodes have optional shared assets, parents, local transforms, visibility and
64-bit native chunk masks. Parent transforms and visibility propagate to children.
IDs belong to their originating scene. Transforms must be finite, affine and
invertible; reflected instances use matching winding and normal orientation.

`prepare_scene` uploads each shared asset once. Geometry, material bindings,
instance buffers and same-sized output/readback targets stay allocated between
renders. Identical texture pixels share GPU storage through a bounded cache.
Skinned/morphed instances have separate reusable deformation buffers.

Update prepared scenes with `set_transform`, `set_parent`, `set_visible`,
`set_chunk_mask`, `set_pose` and `set_morph_weights`. Morph weights follow authored
target order and combine before skinning. `Renderer::set_material` changes one
instance's maps/parameters; `reset_material` restores its asset material.
`statistics` reports assets, geometry chunks, material bindings and mesh instances.

`Camera::Frame` fits visible bounds using a studio direction. Explicit
`Camera::Orthographic` and `Camera::Perspective` retain a fixed camera across
scene edits and animation frames.

`Camera::FrameSubject` uses the same studio directions but frames the selected
geometry independently of reference mannequins. References remain visible and
participate in depth occlusion. `AssemblyOptions::neutral_assemblies` adds native
body/reference selections with neutral materials, preserving authored local
hierarchy and masks. Chunks already supplied by the subject are removed from
neutral reference geometry to avoid coincident surfaces.

## Assemblies and skeletons

`resolve_assembly` reads an exact entity/appearance, expands part entities and
component overrides, and resolves rigs. `load_assembly` constructs its visual
scene and prepares distinct mesh/appearance pairs once. Nonvisual FX/physics
branches and distant proxies are excluded. Shadow-only meshes retain attachment
transforms without visible geometry. Missing visual dependencies, slots or
palette bones return named errors.

Use `AssemblyOptions::mesh_appearances` for explicit component choices. Requests
are exact; absent appearances are not replaced by unrelated ones.
Set `AssemblyOptions::bind_pose` for isolated equipment and hair previews. This
uses authored geometry without evaluating skin deformation or external/bone
attachments. Known local component parents and chunk masks are retained. It
does not establish fit on a complete character. The `assembly_views` example
exposes this mode as `--bind-pose`.
`AssemblyOptions::neutral_references` adds exact mesh or morph geometry with a
texture-free matte grey material, independently of native subject components.
It requires bind-pose mode. Callers choose the reference resource; the renderer
does not infer head/body compatibility. `SceneAssembly::inspection` reports native
bindings, effective placements, masks, materials and camera matrices; assembly
batch jobs retain this report as `assembly-inspection.json`.
`SceneAssembly::from_native` accepts a mesh-loading closure for applications that
manage prepared assets themselves. The closure owns its cache validation policy.

Hair previews use authored profile gradients, opacity, flow maps and vertex-color
strength with an approximate directional strand-lighting model. Intersecting hair
cards and companion caps use depth-writing stochastic coverage rather than whole-
chunk alpha sorting. Coverage uses 4× supersampling up to 2048 pixels, otherwise
2×; PNG output resolves premultiplied coverage in linear space and stores straight
alpha. Native card meshes can encode both windings of the same vertex triplet.
GPU preparation draws one surface when the material is two-sided, with winding
aligned to authored normals; native mesh indices and chunk masks remain intact.
Cast shadows, multiple scattering and the native hair BRDF are not reproduced.

`Skeleton::from_rig` converts the authored local A-pose into Y-up coordinates for
geometry binding, falling back to reference transforms when no A-pose is present.
Animation sampling retains the separate native reference transforms for unkeyed
channels. This avoids calibrating an A-pose mesh against a different reference
pose. `add_rig`
creates bone nodes; `bone_node` exposes rigid attachment anchors. `bind_skin`
matches palette names to skeleton bones and retargets motion onto the mesh's
bind pose. Native four/eight-influence weights are normalized.

`link_rigs` transfers named control-rig motion to deformation rigs. Unanimated
helper bones retain local rest transforms under posed parents. Native control
bindings establish these links during assembly. Update the driver rig;
`unlink_rig` preserves the current deformation pose for independent control.

## Animation and frame output

`AssetReader::animation` selects an exact native clip, resolves streamed payloads,
and decodes compressed-key or SIMD buffers. Clips retain rig identity, duration,
frame count, scalar tracks, additive mode and motion-extraction channels. Sampling
uses linear translation/scale and quaternion slerp; synchronization-only joints
are excluded from the body pose.

`Skeleton::sample_animation` and `PreparedScene::apply_animation` validate native
rig identity and return a pose, tracks and root motion. Additive clips require an
explicit base pose. `RootMotion::Preserve` returns extraction separately;
`ApplyExtraction` applies it to skeleton roots; `InPlace` removes a selected root
bone's translation/rotation and returns the extracted motion.

`render_sequence` accepts timestamped `FrameRequest`s and an update closure, such
as a call to `apply_animation`. The closure establishes complete state for each
time that needs rendering. Use a fixed camera for stable animation framing. The
caller supplies an identity covering its source selections and authoring inputs.

Sequences have manifests, numbered PNGs and per-frame receipts with timestamps,
dimensions, encoded-file hashes and decoded-pixel hashes. Resume verifies outputs
and rerenders incomplete or corrupt frames; conflicting sequence identities are
rejected. `Cancellation` supports thread-safe requests and an optional control
file, checked between preparation and frame work units.

## Catalogue batches

The batch executable reads a JSON array of `BatchJob`s:

```json
[
  {
    "name": "gambeson-gold",
    "source": {
      "kind": "mesh",
      "resource": { "Path": "ep1/characters/garment/player_equipment/torso/t2_157_jacket__gambeson/t2_157_pwa_jacket__gambeson.mesh" },
      "appearance": "gold_2"
    },
    "views": [
      { "size": 512, "camera": { "frame": "hero" } },
      { "size": 512, "camera": { "frame": "back" } }
    ]
  }
]
```

```powershell
./target/release/ghostline-render-batch.exe `
  --game "path/to/Cyberpunk 2077" `
  --manifest catalogue-jobs.json --out work/catalogue --cancel-file work/stop-rendering
```

Unnamed resources use a decimal string, for example
`{ "Hash": "18446744073709551614" }`, preserving all 64 bits in browser clients.

`BatchSource::Assembly` supports entity/appearance selections. `run_batch` prepares
one scene at a time and bounds jobs, vertices, nodes, texture bytes and views.
Its directories account for effective archive payload hashes, schema, build
identity, GPU/driver, selections and output settings. Valid entries resume without
extracting assets. Per-job errors are recorded while other jobs continue.
`batch.json` and job receipts contain results; the CLI emits progress JSON.

For isolated rigid character attachments, `BatchJob.native_frame` accepts a native
rig resource and bone name. Its model-rest rotation supplies the preview axes;
translation and scale are omitted. Skinned scenes retain their authored bind
frame. This orientation does not establish body compatibility or refit an asset.
Camera bounds use indexed, nondegenerate surface triangles rather than unused
vertices. Explicit subject-height crops can omit native off-screen helpers.
`Camera::FrameSubjectSide` fits indexed surface vertices on one model-X side of
a paired subject. The retained renderer caches those bounds alongside the full
bounds and refreshes them when deformation changes. Geometry remains intact.
Flat mesh decals retain the receiving surface normal; normal-map perturbations
are blended without exposing the overlay mesh's polygon edges.
Retained assets are bounded by estimated geometry and texture residency, including
textures held through material bind groups. Oversized assets remain usable by the
active scene without being retained in the reuse cache. Each renderer also bounds
its texture lookup cache independently.
`AssemblyReference.frame` includes a selected context assembly in subject-camera
bounds. Other references remain visible without widening those bounds, allowing
a mounted detail and its attachment anatomy to be framed together.

## Examples and validation

`animated_assembly` is a diagnostic example for exact clips on an entity appearance.
Pass `--game`, `--resource`, `--appearance`, `--clips` and a new `--out`
directory. The clip manifest is an array of `{ "label", "animset", "animation" }`
objects. Optional `--overlay` archives load authored characters without installing
them. It uses fixed per-clip cameras and writes source-bound frame sequences.
Component placement anchors, rigid hair and body-palette experiments require
explicit flags and are recorded as preview approximations. Successful frame output
does not establish correct character deformation; inspect the resulting motion.

`mesh_views` renders repeated views through one device. `assembly_views` resolves
or renders an entity/appearance. `native_resources` inspects morph targets and
samples native clips. Their `--help` output lists inputs.

`workspot_browser` prepares exact entity appearances and exports sampled native
skeletal animation as a self-contained GLB without creating a GPU device. It
accepts newline-delimited `{ "label", "animset", "animation" }` requests on stdin.
Calibrated inverse binds and all eight influences are preserved. Clients must
support both `JOINTS/WEIGHTS_0` and `_1`; the workspot viewer provides this adapter
for Three.js. `Scene::browser_animation_glb` retains component hierarchy, rigid
attachments and sampled linked-rig transforms. Browser materials approximate
native shading. Animated local transforms must be representable as TRS; invalid
samples and topology changes return explicit errors.

```powershell
cargo fmt --check
cargo test --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
```

GPU tests require a compatible adapter. Game-fixture tests are ignored by default
and take `CP2077_GAME_ROOT` and `RED_SCHEMA_PATH`. The vehicle fixture test also
uses `GHOSTLINE_PREPARED_FIXTURES` and `GHOSTLINE_COMPONENT_APPEARANCES` for recorded
material preparation and component choices. Keep local game data and generated
images in ignored `work/`.
