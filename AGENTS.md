# Renderer

Work from the repository root. This is a Rust library and CLI. Native game-resource
decoding uses the sibling `ghostline-red` crate directly. Geometry/material/rig
conversion and GPU rendering belong here; catalogue and authoring rules belong
to callers.

Use typed errors at library boundaries and document public APIs. Validate inputs
before GPU allocation. Keep generated game assets and diagnostic reports in ignored
`work/`. Source game files are read-only.

Run `cargo fmt --check`, `cargo test --locked`, and
`cargo clippy --all-targets --all-features --locked -- -D warnings`.
Verify rendering changes on the GPU with representative assets and inspect images.

Documentation describes supported behavior, architecture and usage. Keep discussion
history, development logs and machine-specific configuration out of public docs.
