# Ghostline Renderer

Rust library and command-line tools for native Cyberpunk 2077 asset previews.
Assemble entity appearances, render transparent images, sample animation, or
export self-contained GLBs for browser previews.

## Build

Use Rust 1.92 or newer and keep the sibling Ghostline Red checkout available.
Run from this repository:

```powershell
cargo build --release --locked
./target/release/ghostline-renderer.exe --help
./target/release/ghostline-render-batch.exe --help
```

Native previews need an installed game and a DirectX 12 or Vulkan GPU driver.
Keep preparation and output directories outside the game. Reflection metadata is
automatic; `--schema` is an optional advanced override.

## Choose a workflow

| Task | Reference |
| --- | --- |
| Render a mesh with an exact appearance | [Single appearance](docs/rendering.md#render-one-appearance) |
| Build and update a retained GPU scene | [Scenes and live updates](docs/rendering.md#scenes-and-live-updates) |
| Resolve native entity components and rigs | [Assemblies and skeletons](docs/rendering.md#assemblies-and-skeletons) |
| Sample animation and render resumable sequences | [Animation and frame output](docs/rendering.md#animation-and-frame-output) |
| Render a catalogue job manifest | [Catalogue batches](docs/rendering.md#catalogue-batches) |
| Export browser scenes and inspect native clips | [Examples and validation](docs/rendering.md#examples-and-validation) |

The library import is `ghostline_renderer`. Reuse `GameSource` across asset
loads and prepared scenes across frames. Ghostline Red owns native decoding and
material inputs; Renderer owns scene assembly, deformation and GPU rendering.

Browser GLBs use baked PBR materials. Native previews also approximate parts of
the game shader. Inspect material notes and rendered output; successful decoding
does not prove a game-equivalent appearance.
