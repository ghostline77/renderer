//! Fingerprint rendering and native decoding sources for output cache validation.

use std::{
    fs,
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let mut digest = Sha256::new();
    for variable in ["TARGET", "PROFILE", "CARGO_CFG_TARGET_FEATURE"] {
        digest.update(variable.as_bytes());
        digest.update(std::env::var(variable).unwrap_or_default().as_bytes());
    }
    let compiler = std::process::Command::new(std::env::var("RUSTC")?)
        .arg("-vV")
        .output()?;
    if !compiler.status.success() {
        return Err("could not identify Rust compiler".into());
    }
    digest.update(compiler.stdout);
    for (label, directory) in [
        ("renderer", root.clone()),
        ("ghostline-red", root.join("../ghostline-red")),
    ] {
        let mut files = Vec::new();
        collect(&directory.join("src"), &mut files)?;
        for name in [
            "Cargo.toml",
            "Cargo.lock",
            "build.rs",
            "data/schema.json",
            "data/serialization.json",
        ] {
            let file = directory.join(name);
            if file.is_file() {
                files.push(file);
            }
        }
        files.sort();
        for file in files {
            println!("cargo:rerun-if-changed={}", file.display());
            let relative = file
                .strip_prefix(&directory)?
                .to_string_lossy()
                .replace('\\', "/");
            digest.update(label.as_bytes());
            digest.update([0]);
            digest.update(relative.as_bytes());
            digest.update([0]);
            digest.update(fs::read(&file)?);
        }
        println!("cargo:rerun-if-changed={}", directory.join("src").display());
    }
    println!(
        "cargo:rustc-env=GHOSTLINE_RENDERER_BUILD_SHA256={:x}",
        digest.finalize()
    );
    Ok(())
}

fn collect(directory: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            collect(&path, files)?;
        } else if path.is_file() {
            files.push(path);
        }
    }
    Ok(())
}
