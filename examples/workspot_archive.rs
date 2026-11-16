//! Inspect character entities and exact authored appearances in an explicit archive.
use anyhow::{Result, ensure};
use clap::Parser;
use ghostline_red::{
    archive,
    assets::{AssetReader, ResourceKey},
    cr2w,
    material::ArchiveSet,
    schema::RedSchema,
};
use ghostline_renderer::GameSource;
use serde_json::json;
use std::{collections::HashMap, fs, path::PathBuf};

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    game: PathBuf,
    #[arg(long)]
    schema: Option<PathBuf>,
    #[arg(long)]
    archive: PathBuf,
    #[arg(long)]
    entity: Option<String>,
    #[arg(long, requires = "entity")]
    appearance: Option<String>,
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    ensure!(args.archive.is_file(), "Archive does not exist");
    if let Some(entity) = args.entity {
        let source =
            GameSource::open_with_overlays(&args.game, args.schema.as_deref(), &[args.archive])?;
        let assembly =
            source.resolve_assembly(&ResourceKey::parse(&entity), args.appearance.as_deref())?;
        let rigs: Vec<_> = assembly.rigs.keys().map(ToString::to_string).collect();
        let matches = |path: &str| {
            assembly
                .rigs
                .keys()
                .any(|key| key.native_hash() == ResourceKey::parse(path).native_hash())
        };
        let male = matches("base/characters/base_entities/man_base/man_base.rig");
        let female = matches("base/characters/base_entities/woman_base/woman_base.rig");
        let rig = match (male, female) {
            (true, false) => Some("male"),
            (false, true) => Some("female"),
            _ => None,
        };
        println!(
            "{}",
            json!({"rig":rig,"rig_resources":rigs,"source_identity":source.content_identity(),"components":assembly.components.len(),"meshes":assembly.components.iter().filter(|c|c.mesh.is_some()).count()})
        );
        return Ok(());
    }
    let index = archive::read_archive(&args.archive)?;
    ensure!(
        index.entries.len() <= 20_000,
        "Archive exceeds the 20,000-resource inspection limit"
    );
    let mut warnings = Vec::new();
    let paths = match archive::read_archive_paths(&args.archive) {
        Ok(paths) => paths,
        Err(error) => {
            warnings.push(format!(
                "Depot names unavailable; inspecting native resource classes: {error}"
            ));
            Vec::new()
        }
    };
    let names: HashMap<_, _> = paths
        .into_iter()
        .map(|path| (archive::depot_path_hash(&path), path))
        .collect();
    let schema = RedSchema::load(Some(&args.game), args.schema.as_deref())?;
    let archives = ArchiveSet::from_paths(vec![args.archive.clone()])?;
    let mut reader = AssetReader::new(&archives, &schema);
    let mut entities = Vec::new();
    let mut inspected = 0;
    let mut skipped = 0;
    let mut bytes_scanned = 0u64;
    for (i, entry) in index.entries.iter().enumerate() {
        let path = names.get(&entry.name_hash);
        if path.is_some_and(|p| !p.to_ascii_lowercase().ends_with(".ent")) {
            continue;
        }
        if entry.size > 16 * 1024 * 1024
            || bytes_scanned.saturating_add(entry.size) > 256 * 1024 * 1024
        {
            skipped += 1;
            continue;
        }
        bytes_scanned += entry.size;
        let bytes = archive::extract_entry_bytes(&args.archive, &index, i)?;
        if !bytes.starts_with(b"CR2W") {
            continue;
        }
        inspected += 1;
        let info = cr2w::inspect_bytes(&bytes)?;
        if info
            .exports
            .first()
            .is_none_or(|e| e.class_name != "entEntityTemplate")
        {
            continue;
        }
        let resource = path.cloned().unwrap_or_else(|| entry.name_hash.to_string());
        let key = ResourceKey::parse(&resource);
        match reader.appearances(&key){
            Ok(appearances)=>entities.push(json!({"resource":resource,"depot_path":path,"hash":entry.name_hash.to_string(),"appearances":appearances})),
            Err(error)=>entities.push(json!({"resource":resource,"depot_path":path,"hash":entry.name_hash.to_string(),"appearances":[],"error":error.to_string()})),
        }
    }
    if skipped > 0 {
        warnings.push(format!("{skipped} unnamed/large resources exceeded the bounded inspection budget; discovery may be incomplete"));
    }
    entities.sort_by_key(|e| e["resource"].as_str().unwrap_or_default().to_owned());
    println!(
        "{}",
        json!({"archive":args.archive,"archive_bytes":fs::metadata(&args.archive)?.len(),"resources":index.entries.len(),"inspected":inspected,"skipped":skipped,"entities":entities,"warnings":warnings})
    );
    Ok(())
}
