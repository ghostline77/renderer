use std::{collections::HashMap, path::Path, sync::Arc};

use ghostline_red::assets::{Assembly, ResourceKey};

use crate::{Error, GameSource, Node, NodeId, RigId, Scene, Skeleton, pose::basis};

/// Native bone axes for an isolated rigid mesh authored in bone-local space.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NativeBoneFrame {
    /// Rig supplying the authored rest axes; no body fit is inferred.
    pub rig: ResourceKey,
    /// Exact native bone name.
    pub bone: String,
}

/// Explicit mesh selections used when preparing an assembled native entity.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct AssemblyOptions {
    /// Explicit native LOD masks keyed by mesh resource hash. Omitted meshes use mask 1.
    #[serde(default)]
    pub mesh_lods: std::collections::BTreeMap<String, u64>,
    /// External world-placement inputs frozen at the entity origin for a
    /// standalone preview. Native mesh-local transforms and rig poses are retained.
    #[serde(default)]
    pub origin_anchors: std::collections::BTreeSet<String>,
    /// Slot-table component to its explicit native rig context.
    #[serde(default)]
    pub slot_rig_overrides: std::collections::BTreeMap<String, String>,
    /// Redundant mesh component to its canonical, placeable preview instance.
    #[serde(default)]
    pub duplicate_components: std::collections::BTreeMap<String, String>,
    /// Explicit corrections keyed by requested native resource hash.
    #[serde(default)]
    pub resource_overrides: std::collections::BTreeMap<String, ResourceKey>,
    /// Mesh components omitted by the caller's explicit preview recipe. Native
    /// transform nodes remain available for attachments and diagnostics.
    #[serde(default)]
    pub excluded_components: std::collections::BTreeSet<String>,
    /// Explicit preview bone corrections keyed by native `component/slot`.
    #[serde(default)]
    pub slot_bone_overrides: std::collections::BTreeMap<String, String>,
    /// Explicit preview attachment corrections, keyed by child component name.
    /// Native bindings remain unchanged in the catalogue; corrections are noted.
    #[serde(default)]
    pub parent_component_overrides: std::collections::BTreeMap<String, String>,
    /// Explicit visibility animation parameters for this preview state.
    #[serde(default)]
    pub visibility_parameters: std::collections::BTreeMap<String, bool>,
    /// Omit unplaceable duplicate decoration props when a placeable copy of
    /// the same mesh and local transform is present. Omissions are reported.
    #[serde(default)]
    pub omit_unbound_duplicates: bool,
    /// Component name to exact mesh appearance, applied before material preparation.
    pub mesh_appearances: std::collections::BTreeMap<String, String>,
    /// Render isolated geometry in its authored bind pose, without runtime rig or
    /// external attachment evaluation. Known local component parents are retained;
    /// external/bone attachments remain at the component's authored local origin.
    #[serde(default)]
    pub bind_pose: bool,
    /// Permit an explicitly standalone prop to retain authored mesh bind geometry
    /// when its runtime skinning rig cannot be supplied. The approximation is noted.
    #[serde(default)]
    pub standalone_bind_pose_fallback: bool,
    /// Explicit reference assemblies displayed alongside an isolated bind-pose
    /// subject. They do not become part of the subject's native selection.
    #[serde(default)]
    pub references: Vec<AssemblyReference>,
    /// Native owners supplying transform/rig dependencies for standalone parts;
    /// owner meshes are not added to the preview.
    #[serde(default)]
    pub binding_contexts: Vec<AssemblyReference>,
    /// Native mesh/morph geometry shown with a matte grey diagnostic material.
    /// References remain separate from native subject components and use bind pose.
    #[serde(default)]
    pub neutral_references: Vec<ResourceKey>,
    /// Native assemblies drawn as matte neutral context with their authored masks.
    #[serde(default)]
    pub neutral_assemblies: Vec<AssemblyReference>,
}

/// Native reference geometry for an isolated preview context.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AssemblyReference {
    /// Include this context geometry in subject camera bounds.
    #[serde(default)]
    pub frame: bool,
    /// Exact native entity or appearance resource.
    pub resource: ResourceKey,
    /// Exact authored appearance, absent only for a simple entity.
    pub appearance: Option<String>,
}

impl AssemblyOptions {
    /// Establishes an isolated prop context. External world attachments are fixed
    /// at the preview origin; internal component transforms and rigs stay native.
    #[must_use]
    pub fn standalone(native: &Assembly) -> Self {
        let mut options = Self {
            standalone_bind_pose_fallback: true,
            ..Self::default()
        };
        for component in &native.components {
            if component.kind == "entExternalComponent" {
                options.origin_anchors.insert(component.name.clone());
            }
            for dependency in [&component.parent.component, &component.control.component] {
                if !dependency.is_empty()
                    && !dependency.eq_ignore_ascii_case("none")
                    && !native
                        .components
                        .iter()
                        .any(|peer| peer.name == *dependency)
                {
                    options.origin_anchors.insert(dependency.clone());
                }
            }
        }
        options
    }
}

/// Loaded native entity with addressable components and skeleton instances.
#[derive(Debug)]
pub struct SceneAssembly {
    /// Explicit preview constraints and omitted redundant native components.
    pub preview_notes: Vec<String>,
    /// Shared meshes, hierarchy, skeletons and skin bindings.
    pub scene: Scene,
    /// Component names mapped to editable scene nodes.
    pub components: HashMap<String, NodeId>,
    /// Rig-bearing or rig-inheriting component names mapped to skeleton instances.
    pub rigs: HashMap<String, RigId>,
    /// Native selection and component details used to construct the scene.
    pub native: Assembly,
}

impl GameSource {
    pub(crate) fn apply_native_bone_frame(
        &self,
        scene: &mut Scene,
        frame: &NativeBoneFrame,
    ) -> Result<(), Error> {
        // Skinned meshes are already authored in their model bind frame.
        if scene.assets.iter().any(|asset| asset.skin.is_some()) {
            return Ok(());
        }
        let mut reader = self.resources();
        let rig = reader.rig(&frame.rig)?;
        let index = rig
            .bones
            .iter()
            .position(|bone| bone.name == frame.bone)
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "native preview bone {} absent from {}",
                    frame.bone, frame.rig
                ))
            })?;
        let (_, rotation, _) = rig.model_rest()?[index].to_scale_rotation_translation();
        let native_rotation = glam::Mat4::from_quat(rotation.normalize());
        let transform = basis() * native_rotation * basis().transpose();
        let roots = scene
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.parent.is_none())
            .map(|(i, _)| NodeId(i))
            .collect::<Vec<_>>();
        let root = scene.add_node(
            Node::new(format!("preview-frame/{}/{}", frame.rig, frame.bone), None)
                .with_transform(transform),
        )?;
        for node in roots {
            scene.set_parent(node, Some(root))?;
        }
        Ok(())
    }
    /// Supplies non-mesh native owner components and rigs for a standalone part.
    ///
    /// # Errors
    /// Returns native errors when an explicit owner context cannot be resolved.
    pub fn add_binding_context(
        &self,
        native: &mut Assembly,
        options: &AssemblyOptions,
    ) -> Result<(), Error> {
        for reference in &options.binding_contexts {
            let context =
                self.resolve_assembly(&reference.resource, reference.appearance.as_deref())?;
            for component in context
                .components
                .into_iter()
                .filter(|component| component.mesh.is_none())
            {
                if !native
                    .components
                    .iter()
                    .any(|existing| existing.name == component.name)
                {
                    native.components.push(component);
                }
            }
            native.rigs.extend(context.rigs);
        }
        Ok(())
    }
    /// Loads an exact entity/appearance selection directly from native archives.
    ///
    /// Shared mesh/appearance pairs are prepared once. Disabled components and
    /// distant appearance proxy meshes retain transform nodes without geometry.
    /// Bindings to bones and named rigid slots become ordinary scene hierarchy.
    ///
    /// # Errors
    /// Returns [`Error`] for invalid native selections, unresolved attachments,
    /// missing palette bones, unsupported materials or existing preparation paths.
    pub fn load_assembly(
        &self,
        resource: &ResourceKey,
        appearance: Option<&str>,
        options: &AssemblyOptions,
        work: &Path,
    ) -> Result<SceneAssembly, Error> {
        if work.exists() {
            return Err(Error::InvalidInput(
                "assembly preparation directory already exists".into(),
            ));
        }
        let native = self.preview_assembly(resource, appearance, options)?;
        self.load_resolved_assembly(native, options, work)
    }

    /// Prepares a caller-composed native component graph with the normal rig,
    /// attachment, material and geometry pipeline.
    ///
    /// # Errors
    /// Returns errors for invalid hierarchy, unresolved bindings or mesh preparation.
    pub fn load_resolved_assembly(
        &self,
        native: Assembly,
        options: &AssemblyOptions,
        work: &Path,
    ) -> Result<SceneAssembly, Error> {
        let work = self.validate_output(work)?;
        std::fs::create_dir_all(&work)?;
        let mut assembly = SceneAssembly::from_native(native, options, |resource, appearance| {
            self.load_assembly_mesh_resource(resource, appearance, options)
        })?;
        self.add_neutral_references(&mut assembly.scene, options)?;
        Ok(assembly)
    }

    pub(crate) fn load_assembly_mesh_resource(
        &self,
        resource: &ResourceKey,
        appearance: &str,
        options: &AssemblyOptions,
    ) -> Result<Arc<crate::MeshAsset>, Error> {
        if let Some(mask) = options.mesh_lods.get(&resource.native_hash().to_string()) {
            self.load_shared_mesh_resource_lod(resource, appearance, *mask)
        } else {
            self.load_shared_mesh_resource(resource, appearance)
        }
    }

    pub(crate) fn add_neutral_references(
        &self,
        scene: &mut Scene,
        options: &AssemblyOptions,
    ) -> Result<(), Error> {
        let mut subject_masks = HashMap::<u64, u64>::new();
        for node in &mut scene.nodes {
            if let Some((_, reference)) = options
                .references
                .iter()
                .enumerate()
                .find(|(i, _)| node.name.starts_with(&format!("reference-{i}/")))
            {
                node.reference = !reference.frame;
            }
            if !node.reference
                && let Some(asset) = node.asset
            {
                *subject_masks
                    .entry(ResourceKey::parse(&scene.assets[asset.0].mesh).native_hash())
                    .or_default() |= node.chunk_mask;
            }
        }
        for node in &mut scene.nodes {
            if node.reference
                && let Some(asset) = node.asset
            {
                let hash = ResourceKey::parse(&scene.assets[asset.0].mesh).native_hash();
                node.chunk_mask &= !subject_masks.get(&hash).copied().unwrap_or(0);
            }
        }
        for (index, resource) in options.neutral_references.iter().enumerate() {
            let mask = !subject_masks
                .get(&resource.native_hash())
                .copied()
                .unwrap_or(0);
            if mask == 0 {
                continue;
            }
            let asset = self.load_shared_neutral(resource)?;
            let asset = scene.add_shared_asset(asset);
            let mut node =
                Node::new(format!("neutral-reference-{index}"), Some(asset)).with_chunk_mask(mask);
            node.reference = true;
            scene.add_node(node)?;
        }
        for (index, reference) in options.neutral_assemblies.iter().enumerate() {
            let native =
                self.resolve_assembly(&reference.resource, reference.appearance.as_deref())?;
            let reference = SceneAssembly::from_native(
                native,
                &AssemblyOptions {
                    bind_pose: true,
                    ..Default::default()
                },
                |resource, _| self.load_shared_neutral(resource),
            )?;
            let assets: Vec<_> = reference
                .scene
                .assets
                .iter()
                .map(|a| scene.add_shared_asset(Arc::clone(a)))
                .collect();
            let mut nodes = Vec::new();
            for node in &reference.scene.nodes {
                let mut copy = node.clone();
                copy.name = format!("neutral-assembly-{index}/{}", node.name);
                copy.parent = None;
                copy.reference = true;
                if let Some(asset) = node.asset {
                    let hash =
                        ResourceKey::parse(&reference.scene.assets[asset.0].mesh).native_hash();
                    copy.chunk_mask &= !subject_masks.get(&hash).copied().unwrap_or(0);
                    copy.asset = Some(assets[asset.0]);
                }
                nodes.push(scene.add_node(copy)?);
            }
            for (i, node) in reference.scene.nodes.iter().enumerate() {
                if let Some(parent) = node.parent {
                    scene.set_parent(nodes[i], Some(nodes[parent.0]))?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn preview_assembly(
        &self,
        resource: &ResourceKey,
        appearance: Option<&str>,
        options: &AssemblyOptions,
    ) -> Result<Assembly, Error> {
        if !options.bind_pose
            && (!options.references.is_empty()
                || !options.neutral_references.is_empty()
                || !options.neutral_assemblies.is_empty())
        {
            return Err(Error::InvalidInput(
                "reference assemblies require explicit bind-pose mode".into(),
            ));
        }
        let mut native = self.resources().assembly_with_resource_overrides(
            resource,
            appearance,
            &options.excluded_components,
            &options.resource_overrides,
        )?;
        self.add_binding_context(&mut native, options)?;
        for component in &mut native.components {
            if options.excluded_components.contains(&component.name) || component.is_collapsed() {
                component.enabled = false;
                continue;
            }
            if component.enabled
                && component.chunk_mask != 0
                && let Some(mesh) = &component.mesh
                && self.load_geometry(mesh)?.is_shadow_only()
            {
                component.enabled = false;
            }
        }
        for (index, reference) in options.references.iter().enumerate() {
            let mut assembly =
                self.resolve_assembly(&reference.resource, reference.appearance.as_deref())?;
            let names: std::collections::HashSet<_> =
                assembly.components.iter().map(|c| c.name.clone()).collect();
            let prefix = format!("reference-{index}/");
            for component in &mut assembly.components {
                component.name = format!("{prefix}{}", component.name);
                for binding in [&mut component.parent, &mut component.control] {
                    binding.component = if names.contains(&binding.component) {
                        format!("{prefix}{}", binding.component)
                    } else {
                        String::new()
                    };
                }
            }
            native.components.extend(assembly.components);
            native.rigs.extend(assembly.rigs);
        }
        Ok(native)
    }
}

impl SceneAssembly {
    /// Returns component placements relative to the assembled entity, in native
    /// RED coordinates (metres, Z up). Rig and slot transforms are evaluated.
    ///
    /// # Errors
    /// Returns an error for an invalid component hierarchy or transform.
    pub fn native_component_placements(
        &self,
    ) -> Result<std::collections::BTreeMap<String, [f32; 16]>, Error> {
        let worlds = self.scene.world_transforms()?;
        let basis = basis();
        Ok(self
            .components
            .iter()
            .map(|(name, node)| {
                (
                    name.clone(),
                    (basis.transpose() * worlds[node.0].0 * basis).to_cols_array(),
                )
            })
            .collect())
    }
    /// Builds a visual scene using caller-provided prepared mesh assets.
    ///
    /// The loader runs once per distinct resource/appearance pair, after native
    /// hierarchy validation. This supports shared preparation caches and explicit
    /// diagnostic materials while retaining the same assembly and binding rules.
    ///
    /// # Errors
    /// Returns [`Error`] for invalid visual dependencies, attachments or loaded assets.
    pub fn from_native(
        mut native: Assembly,
        options: &AssemblyOptions,
        mut load_mesh: impl FnMut(&ResourceKey, &str) -> Result<Arc<crate::MeshAsset>, Error>,
    ) -> Result<Self, Error> {
        let mut preview_notes = Vec::new();
        for component in &mut native.components {
            if component.kind == "entStaticOccluderMeshComponent" {
                component.enabled = false;
                preview_notes.push(format!("Preview omits occlusion-only component {}; native occlusion remains part of the entity",component.name));
            }
        }
        for (resource, mask) in &options.mesh_lods {
            preview_notes.push(format!(
                "Explicit native mesh LOD: resource {resource}, mask {mask}"
            ));
        }
        for name in &options.origin_anchors {
            let slots: Vec<_> = native.components.iter()
                .filter(|component| component.parent.component == *name)
                .map(|component| component.parent.slot.clone())
                .filter(|slot| !slot.is_empty() && !slot.eq_ignore_ascii_case("none"))
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter().map(|slot| {
                    preview_notes.push(format!("Standalone preview freezes external attachment {name}/{slot} at its declared preview origin"));
                    ghostline_red::assets::Slot {
                        name:slot,bone:String::new(),transform:ghostline_red::assets::Transform {
                            translation:[0.;3],rotation:[0.,0.,0.,1.],scale:[1.;3]
                        }
                    }
                }).collect();
            if let Some(component) = native
                .components
                .iter_mut()
                .find(|component| component.name == *name)
            {
                if component.kind == "entExternalComponent" {
                    for slot in slots {
                        if !component
                            .slots
                            .iter()
                            .any(|existing| existing.name == slot.name)
                        {
                            component.slots.push(slot);
                        }
                    }
                }
                continue;
            }
            native.components.push(ghostline_red::assets::AssemblyComponent {
                template:Arc::new(serde_json::json!({"$type":"GhostlinePreviewOrigin","preview_context":true})),
                name:name.clone(),kind:"GhostlinePreviewOrigin".into(),local:glam::Mat4::IDENTITY,
                parent:Default::default(),control:Default::default(),rig:None,mesh:None,
                mesh_appearance:String::new(),mesh_appearance_requested:None,chunk_mask:u64::MAX,enabled:true,slots
            });
            preview_notes.push(format!("Standalone preview freezes external world-placement input {name} at the entity origin; native local transforms and rig bindings are retained"));
        }
        for (duplicate, canonical) in &options.duplicate_components {
            let original = native
                .components
                .iter()
                .find(|component| component.name == *duplicate)
                .ok_or_else(|| {
                    Error::InvalidInput(format!("duplicate component {duplicate} is absent"))
                })?;
            let peer = native
                .components
                .iter()
                .find(|component| component.name == *canonical)
                .ok_or_else(|| {
                    Error::InvalidInput(format!("canonical component {canonical} is absent"))
                })?;
            if original.mesh.is_none()
                || original.mesh != peer.mesh
                || original.mesh_appearance != peer.mesh_appearance
                || !peer.enabled
                || !attachment_known(&native, peer)
            {
                return Err(Error::InvalidInput(format!(
                    "duplicate {duplicate} does not match a placeable {canonical}"
                )));
            }
            native
                .components
                .iter_mut()
                .find(|component| component.name == *duplicate)
                .expect("validated duplicate")
                .enabled = false;
            preview_notes.push(format!("Preview uses canonical component {canonical} for duplicate {duplicate}; native mesh and appearance identities match"));
        }
        for (owner, rig) in &options.slot_rig_overrides {
            preview_notes.push(format!(
                "Preview slot table {owner} uses the native rig component {rig}"
            ));
        }
        for component in &mut native.components {
            if options.excluded_components.contains(&component.name) || component.is_collapsed() {
                preview_notes.push(format!(
                    "Preview omits mesh {}: {}",
                    component.name,
                    if component.is_collapsed() {
                        "native visual scale collapses its geometry"
                    } else {
                        "excluded by the explicit catalogue recipe"
                    }
                ));
                component.enabled = false;
            }
        }
        for (key, bone) in &options.slot_bone_overrides {
            let (owner, name) = key.rsplit_once('/').ok_or_else(|| {
                Error::InvalidInput("slot override requires component/slot".into())
            })?;
            let owner_component = native
                .components
                .iter_mut()
                .find(|component| component.name == owner)
                .ok_or_else(|| Error::InvalidInput(format!("preview slot {key} is absent")))?;
            if !owner_component.slots.iter().any(|slot| slot.name == name) {
                let candidates: Vec<_> = owner_component
                    .slots
                    .iter()
                    .filter(|slot| slot.bone == *bone)
                    .collect();
                if candidates.len() != 1 {
                    return Err(Error::InvalidInput(format!(
                        "preview slot alias {key} requires one native slot on bone {bone}"
                    )));
                }
                let mut alias = candidates[0].clone();
                preview_notes.push(format!(
                    "Preview slot {key} uses the authored {} frame on bone {bone}",
                    alias.name
                ));
                alias.name = name.into();
                owner_component.slots.push(alias);
            }
            let slot = owner_component
                .slots
                .iter_mut()
                .find(|slot| slot.name == name)
                .expect("validated preview slot");
            let prior = std::mem::replace(&mut slot.bone, bone.clone());
            preview_notes.push(format!(
                "Preview slot correction for {key}: native bone {prior} is replaced by {bone}"
            ));
        }
        for (child, parent) in &options.parent_component_overrides {
            if !native
                .components
                .iter()
                .any(|component| component.name == *parent)
            {
                return Err(Error::InvalidInput(format!(
                    "preview attachment parent {parent} is absent"
                )));
            }
            let component = native
                .components
                .iter_mut()
                .find(|component| component.name == *child)
                .ok_or_else(|| {
                    Error::InvalidInput(format!("preview attachment child {child} is absent"))
                })?;
            let prior = std::mem::replace(&mut component.parent.component, parent.clone());
            preview_notes.push(format!("Preview attachment correction for {child}: native parent {prior} is replaced by existing component {parent}"));
        }
        if options.omit_unbound_duplicates {
            let omitted: Vec<_> = native
                .components
                .iter()
                .enumerate()
                .filter(|(_, component)| {
                    component.enabled
                        && component.mesh.is_some()
                        && component.rig.is_none()
                        && component.kind == "entMeshComponent"
                        && component
                            .template
                            .get("forcedLodDistance")
                            .and_then(serde_json::Value::as_str)
                            == Some("VehicleDecoration")
                        && component.control.component.is_empty()
                        && !attachment_known(&native, component)
                        && native.components.iter().any(|peer| {
                            peer.name != component.name
                                && peer.enabled
                                && peer.chunk_mask != 0
                                && peer.mesh == component.mesh
                                && peer.local == component.local
                                && peer.parent.component == component.parent.component
                                && attachment_known(&native, peer)
                        })
                })
                .map(|(i, _)| i)
                .collect();
            for index in omitted {
                let component = &mut native.components[index];
                preview_notes.push(format!("Omitted redundant native component {}: attachment {}/{} is absent; the same mesh and local transform have an enabled, placeable instance",component.name,component.parent.component,component.parent.slot));
                component.enabled = false;
            }
        }
        for component in &mut native.components {
            let parameter = component
                .template
                .get("visibilityAnimationParam")
                .and_then(|value| {
                    value
                        .as_str()
                        .or_else(|| value.get("$value").and_then(serde_json::Value::as_str))
                });
            if let Some(false) = parameter
                .and_then(|name| options.visibility_parameters.get(name))
                .copied()
            {
                component.enabled = false;
            }
        }
        for (name, appearance) in &options.mesh_appearances {
            let component = native
                .components
                .iter_mut()
                .find(|c| c.name == *name)
                .ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "appearance override names absent component {name}"
                    ))
                })?;
            component.mesh_appearance.clone_from(appearance);
        }
        if options.bind_pose {
            let mut assembly = Self::isolated_bind_pose(native, load_mesh)?;
            assembly.preview_notes = preview_notes;
            return Ok(assembly);
        }
        let needed = visual_dependencies(&native, &options.slot_rig_overrides)?;
        let mut scene = Scene::new();
        let mut components = HashMap::new();
        let mut assets = HashMap::new();
        let basis = basis();
        for component in native
            .components
            .iter()
            .filter(|c| needed.contains(&c.name))
        {
            let node = scene.add_node(
                Node::new(&component.name, None)
                    .with_transform(basis * component.local * basis.transpose())
                    .with_chunk_mask(component.chunk_mask),
            )?;
            components.insert(component.name.clone(), node);
        }
        let mut rigs = HashMap::new();
        for component in native
            .components
            .iter()
            .filter(|c| needed.contains(&c.name))
        {
            if let Some(key) = &component.rig {
                let skeleton = Arc::new(Skeleton::from_rig(&native.rigs[key])?);
                rigs.insert(
                    component.name.clone(),
                    scene.add_rig(components[&component.name], skeleton)?,
                );
            }
        }
        // Control bindings can inherit a skeleton without an authored rig resource.
        for _ in 0..native.components.len() {
            let mut changed = false;
            for component in native
                .components
                .iter()
                .filter(|c| needed.contains(&c.name))
            {
                let inherited = rigs.get(&component.control.component).copied();
                if !rigs.contains_key(&component.name)
                    && let Some(id) = inherited
                {
                    rigs.insert(component.name.clone(), id);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        for component in native
            .components
            .iter()
            .filter(|c| needed.contains(&c.name))
        {
            let node = components[&component.name];
            let binding = &component.parent;
            if binding.component.is_empty() || binding.component == "None" {
                continue;
            }
            let parent = components.get(&binding.component).copied().ok_or_else(|| {
                Error::InvalidInput(format!(
                    "unresolved parent {} on {}",
                    binding.component, component.name
                ))
            })?;
            let parent = if binding.slot.is_empty() || binding.slot.eq_ignore_ascii_case("none") {
                parent
            } else if let Some(bone) = rigs
                .get(&binding.component)
                .and_then(|&rig| scene.bone_node(rig, &binding.slot))
                .filter(|_| {
                    !native.components.iter().any(|owner| {
                        owner.name == binding.component
                            && owner.slots.iter().any(|slot| slot.name == binding.slot)
                    })
                })
            {
                bone
            } else {
                let owner = native
                    .components
                    .iter()
                    .find(|c| c.name == binding.component)
                    .ok_or_else(|| Error::InvalidInput("missing slot component".into()))?;
                let matches: Vec<_> = owner
                    .slots
                    .iter()
                    .filter(|s| s.name == binding.slot)
                    .collect();
                if matches.is_empty()
                    && owner.kind != "entSlotComponent"
                    && owner.rig.is_none()
                    && !rigs.contains_key(&owner.name)
                {
                    // These targets expose their component transform, not a slot
                    // table. Inherited slotName values do not define another frame.
                    preview_notes.push(format!(
                        "Attachment {}/{} uses the native {} component transform",
                        owner.name, binding.slot, owner.kind
                    ));
                    scene.set_parent(node, Some(parent))?;
                    continue;
                }
                if matches.is_empty() {
                    return Err(Error::InvalidInput(format!(
                        "unresolved slot {}/{}",
                        binding.component, binding.slot
                    )));
                }
                let slot = matches[0];
                let slot_matrix = slot.transform.matrix()?;
                for duplicate in matches.iter().skip(1) {
                    if duplicate.bone != slot.bone || duplicate.transform.matrix()? != slot_matrix {
                        return Err(Error::InvalidInput(format!(
                            "ambiguous native slot {}/{}",
                            owner.name, slot.name
                        )));
                    }
                }
                let anchor = if slot.bone.is_empty() {
                    parent
                } else {
                    let anchor = slot_bone_node(
                        &native,
                        &rigs,
                        &scene,
                        options
                            .slot_rig_overrides
                            .get(&owner.name)
                            .unwrap_or(&owner.parent.component),
                        &slot.bone,
                    );
                    (if anchor.is_some() {
                        anchor
                    } else {
                        mesh_bind_bone_node(
                            &native,
                            &mut scene,
                            &components,
                            &owner.parent.component,
                            &slot.bone,
                            &mut load_mesh,
                        )?
                    })
                    .ok_or_else(|| {
                        Error::InvalidInput(format!("unresolved slot bone {}", slot.bone))
                    })?
                };
                scene.add_node(
                    Node::new(format!("{}/{}", owner.name, slot.name), None)
                        .with_parent(anchor)
                        .with_transform(basis * slot_matrix * basis.transpose()),
                )?
            };
            scene.set_parent(node, Some(parent))?;
        }
        scene.world_transforms()?;
        for component in native
            .components
            .iter()
            .filter(|c| needed.contains(&c.name))
        {
            if let Some((&destination, &source)) = rigs
                .get(&component.name)
                .zip(rigs.get(&component.control.component))
                && source != destination
            {
                scene.link_rigs(source, destination)?;
            }
        }
        for component in native
            .components
            .iter()
            .filter(|c| needed.contains(&c.name))
        {
            let asset = if component.enabled
                && component.chunk_mask != 0
                && component.kind != "entAppearanceProxyMeshComponent"
            {
                if let Some(mesh) = &component.mesh {
                    let key = (mesh.clone(), component.mesh_appearance.clone());
                    if let Some(&id) = assets.get(&key) {
                        id
                    } else {
                        let asset =
                            load_mesh(mesh, &component.mesh_appearance).map_err(|source| {
                                Error::Component {
                                    name: component.name.clone(),
                                    resource: mesh.to_string(),
                                    source: Box::new(source),
                                }
                            })?;
                        let id = (!asset.is_shadow_only()).then(|| scene.add_shared_asset(asset));
                        assets.insert(key, id);
                        id
                    }
                } else {
                    None
                }
            } else {
                None
            };
            scene.nodes[components[&component.name].0].asset = asset;
        }
        for component in native
            .components
            .iter()
            .filter(|c| needed.contains(&c.name))
        {
            let node = components[&component.name];
            let Some(asset) = scene.nodes[node.0].asset else {
                continue;
            };
            let asset = &scene.assets[asset.0];
            if (component.kind.contains("Skinned") || !component.control.component.is_empty())
                && asset.chunks.iter().any(|chunk| {
                    chunk
                        .skin
                        .iter()
                        .any(|vertex| vertex.weights.iter().any(|weight| *weight > 0.))
                })
            {
                if component.control.component.is_empty()
                    && !rigs.contains_key(&component.parent.component)
                {
                    preview_notes.push(format!("Component {} has weighted mesh data but no native skinning/control or parent rig binding; retained its authored local bind pose", component.name));
                    continue;
                }
                let rig = rigs
                    .get(&component.control.component)
                    .or_else(|| rigs.get(&component.parent.component))
                    .copied();
                if rig.is_none() && options.standalone_bind_pose_fallback {
                    preview_notes.push(format!("Standalone prop retains authored bind geometry for {}: runtime skinning rig {}/{} is unavailable", component.name, component.parent.component, component.control.component));
                    continue;
                }
                let rig = rig.ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "skinned component {} has no bound rig",
                        component.name
                    ))
                })?;
                let fallbacks: Vec<_> = [
                    component.parent.component.as_str(),
                    component.control.component.as_str(),
                    "deformations",
                    "root",
                    // Neckline garments can share weighted neck bones with
                    // the actor's linked face rig rather than its body rig.
                    "face_rig",
                ]
                .into_iter()
                .filter_map(|name| rigs.get(name).copied())
                .filter(|candidate| *candidate != rig)
                .collect();
                if let Err(source) = scene.bind_skin_with_fallbacks(node, rig, &fallbacks) {
                    if options.standalone_bind_pose_fallback
                        && matches!(&source, Error::InvalidInput(message) if message == "mesh has no bone palette" || message.starts_with("skin palette bone ") && message.ends_with(" is absent from rig"))
                    {
                        preview_notes.push(format!(
                            "Standalone prop retains authored bind geometry for {}: {source}",
                            component.name
                        ));
                        continue;
                    }
                    return Err(Error::Component {
                        name: component.name.clone(),
                        resource: format!(
                            "skin binding {} / {}",
                            component.parent.component, component.control.component
                        ),
                        source: Box::new(source),
                    });
                }
            }
        }
        Ok(SceneAssembly {
            preview_notes,
            scene,
            components,
            rigs,
            native,
        })
    }

    fn isolated_bind_pose(
        native: Assembly,
        mut load_mesh: impl FnMut(&ResourceKey, &str) -> Result<Arc<crate::MeshAsset>, Error>,
    ) -> Result<Self, Error> {
        let mut scene = Scene::new();
        let mut components = HashMap::new();
        let mut assets = HashMap::new();
        let basis = basis();
        for component in &native.components {
            let node = scene.add_node(
                Node::new(&component.name, None)
                    .with_transform(basis * component.local * basis.transpose())
                    .with_chunk_mask(component.chunk_mask),
            )?;
            if components.insert(component.name.clone(), node).is_some() {
                return Err(Error::InvalidInput(format!(
                    "duplicate component {}",
                    component.name
                )));
            }
        }
        for component in &native.components {
            // Bone/slot frames require a complete character. Bind-pose mode only
            // composes directly authored local component transforms.
            if (component.parent.slot.is_empty()
                || component.parent.slot.eq_ignore_ascii_case("none"))
                && let Some(parent) = components.get(&component.parent.component)
            {
                scene.set_parent(components[&component.name], Some(*parent))?;
            }
            if !component.enabled
                || component.chunk_mask == 0
                || component.kind == "entAppearanceProxyMeshComponent"
            {
                continue;
            }
            let Some(mesh) = &component.mesh else {
                continue;
            };
            let key = (mesh.clone(), component.mesh_appearance.clone());
            let asset = if let Some(asset) = assets.get(&key) {
                *asset
            } else {
                let asset = load_mesh(mesh, &component.mesh_appearance).map_err(|source| {
                    Error::Component {
                        name: component.name.clone(),
                        resource: mesh.to_string(),
                        source: Box::new(source),
                    }
                })?;
                let id = (!asset.is_shadow_only()).then(|| scene.add_shared_asset(asset));
                assets.insert(key, id);
                id
            };
            scene.nodes[components[&component.name].0].asset = asset;
        }
        scene.world_transforms()?;
        if !scene.nodes.iter().any(|node| {
            node.asset.is_some_and(|id| {
                scene.assets[id.0].chunks.iter().any(|chunk| {
                    crate::scene::chunk_visible(node.chunk_mask, chunk.chunk_index)
                        && !chunk.indices.is_empty()
                })
            })
        }) {
            return Err(Error::NonVisual);
        }
        Ok(Self {
            preview_notes: Vec::new(),
            scene,
            components,
            rigs: HashMap::new(),
            native,
        })
    }
}

fn attachment_known(
    native: &Assembly,
    component: &ghostline_red::assets::AssemblyComponent,
) -> bool {
    let binding = &component.parent;
    if binding.component.is_empty() || binding.component.eq_ignore_ascii_case("none") {
        return true;
    }
    let Some(owner) = native
        .components
        .iter()
        .find(|c| c.name == binding.component)
    else {
        return false;
    };
    if binding.slot.is_empty() || binding.slot.eq_ignore_ascii_case("none") {
        return true;
    }
    if owner.slots.iter().any(|slot| slot.name == binding.slot) {
        return true;
    }
    let mut pending = vec![owner];
    let mut visited = std::collections::HashSet::new();
    while let Some(owner) = pending.pop() {
        if !visited.insert(owner.name.as_str()) {
            continue;
        }
        if let Some(rig) = owner.rig.as_ref().and_then(|key| native.rigs.get(key))
            && rig.bones.iter().any(|bone| bone.name == binding.slot)
        {
            return true;
        }
        for parent in [&owner.parent.component, &owner.control.component] {
            if let Some(parent) = native.components.iter().find(|c| c.name == *parent) {
                pending.push(parent);
            }
        }
    }
    false
}

fn slot_bone_node(
    native: &Assembly,
    rigs: &HashMap<String, RigId>,
    scene: &Scene,
    start: &str,
    bone: &str,
) -> Option<NodeId> {
    let mut pending = vec![start];
    let mut seen = std::collections::HashSet::new();
    while let Some(name) = pending.pop() {
        if name.is_empty() || !seen.insert(name) {
            continue;
        }
        if let Some(node) = rigs.get(name).and_then(|&rig| scene.bone_node(rig, bone)) {
            return Some(node);
        }
        if let Some(component) = native
            .components
            .iter()
            .find(|component| component.name == name)
        {
            pending.push(&component.parent.component);
            pending.push(&component.control.component);
        }
    }
    None
}

fn mesh_bind_bone_node(
    native: &Assembly,
    scene: &mut Scene,
    components: &HashMap<String, NodeId>,
    start: &str,
    bone: &str,
    load_mesh: &mut impl FnMut(&ResourceKey, &str) -> Result<Arc<crate::MeshAsset>, Error>,
) -> Result<Option<NodeId>, Error> {
    let mut pending = vec![start];
    let mut visited = std::collections::HashSet::new();
    while let Some(name) = pending.pop() {
        if name.is_empty() || !visited.insert(name) {
            continue;
        }
        let Some(component) = native
            .components
            .iter()
            .find(|component| component.name == name)
        else {
            continue;
        };
        if let Some(mesh) = &component.mesh {
            let asset = load_mesh(mesh, &component.mesh_appearance)?;
            if let Some(skin) = &asset.skin
                && let Some(index) = skin.names.iter().position(|name| name == bone)
                && let Some(&parent) = components.get(name)
            {
                let node = scene.add_node(
                    Node::new(format!("{name}/physics-bind/{bone}"), None)
                        .with_parent(parent)
                        .with_transform(skin.inverse_bind[index].inverse()),
                )?;
                return Ok(Some(node));
            }
        }
        pending.push(&component.parent.component);
        pending.push(&component.control.component);
    }
    Ok(None)
}

fn visual_dependencies(
    native: &Assembly,
    rig_contexts: &std::collections::BTreeMap<String, String>,
) -> Result<std::collections::HashSet<String>, Error> {
    let mut pending: Vec<_> = native
        .components
        .iter()
        .filter(|c| {
            c.mesh.is_some()
                && c.enabled
                && c.chunk_mask != 0
                && c.kind != "entAppearanceProxyMeshComponent"
        })
        .map(|c| c.name.clone())
        .collect();
    let mut required = std::collections::HashSet::new();
    pending.extend(rig_contexts.values().cloned());
    while let Some(name) = pending.pop() {
        if !required.insert(name.clone()) {
            continue;
        }
        let component = native
            .components
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| Error::InvalidInput(format!("unresolved visual component {name}")))?;
        for dependency in [&component.parent.component, &component.control.component] {
            if !dependency.is_empty() && !dependency.eq_ignore_ascii_case("none") {
                if !native.components.iter().any(|c| c.name == *dependency) {
                    return Err(Error::InvalidInput(format!(
                        "unresolved visual dependency {dependency} on {name}"
                    )));
                }
                pending.push(dependency.clone());
            }
        }
    }
    if required.is_empty() {
        return Err(Error::InvalidInput(
            "assembly has no enabled visual components".into(),
        ));
    }
    Ok(required)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ghostline_red::assets::{AssemblyComponent, Binding};
    use glam::Mat4;

    #[test]
    fn explicit_aim_state_hides_the_native_aiming_mask_before_preparation() {
        let mut mask = native();
        mask.components[0].template = Arc::new(
            serde_json::json!({"visibilityAnimationParam":{"$value":"visibleWhenAiming"}}),
        );
        let mut options = AssemblyOptions {
            bind_pose: true,
            ..Default::default()
        };
        assert!(
            SceneAssembly::from_native(mask.clone(), &options, |_, _| Ok(Arc::new(
                crate::gpu::tests::triangle()
            )))
            .is_ok()
        );
        options
            .visibility_parameters
            .insert("visibleWhenAiming".into(), false);
        let result = SceneAssembly::from_native(mask, &options, |_, _| {
            panic!("hidden aiming mask must not prepare geometry or materials")
        });
        assert!(matches!(result, Err(crate::Error::NonVisual)));
    }

    #[test]
    fn explicit_preview_parent_corrections_apply_to_shared_batch_assembly() {
        let mut assembled = native();
        assembled.components[0].kind = "entMeshComponent".into();
        assembled.components[0].parent = Binding {
            component: "obsolete_window".into(),
            slot: String::new(),
        };
        let mut host = assembled.components[0].clone();
        host.name = "window".into();
        host.mesh = None;
        host.parent = Binding::default();
        host.local = Mat4::from_translation(glam::Vec3::new(2., 0., 0.));
        assembled.components.push(host);
        let options = AssemblyOptions {
            parent_component_overrides: std::collections::BTreeMap::from([(
                "garment".into(),
                "window".into(),
            )]),
            ..Default::default()
        };
        let scene = SceneAssembly::from_native(assembled, &options, |_, _| {
            Ok(Arc::new(crate::gpu::tests::triangle()))
        })
        .unwrap();
        let node = scene.components["garment"];
        assert_eq!(
            scene.scene.world_transforms().unwrap()[node.0].0.w_axis.x,
            2.
        );
        assert!(scene.preview_notes[0].contains("obsolete_window"));
    }

    #[test]
    fn nested_slot_tables_find_the_owning_rig_and_keep_relative_offsets() {
        use ghostline_red::assets::{Bone, Rig, Slot, Transform};
        let mut assembled = native();
        assembled.components[0].kind = "entMeshComponent".into();
        assembled.components[0].parent = Binding {
            component: "nested_slots".into(),
            slot: "mount".into(),
        };
        let key = ResourceKey::Hash(3);
        let mut rig = assembled.components[0].clone();
        rig.name = "rig".into();
        rig.mesh = None;
        rig.parent = Binding::default();
        rig.rig = Some(key.clone());
        let mut group = rig.clone();
        group.name = "group".into();
        group.rig = None;
        group.parent.component = "rig".into();
        let mut slots = group.clone();
        slots.name = "nested_slots".into();
        slots.parent.component = "group".into();
        slots.slots.push(Slot {
            name: "mount".into(),
            bone: "Base".into(),
            transform: Transform {
                translation: [3., 0., 0.],
                ..Default::default()
            },
        });
        assembled.components.extend([rig, group, slots]);
        assembled.rigs.insert(
            key,
            Rig {
                a_pose: None,
                source: None,
                bones: vec![Bone {
                    name: "Base".into(),
                    parent: None,
                    rest: Transform {
                        translation: [2., 0., 0.],
                        ..Default::default()
                    },
                }],
            },
        );
        let scene = SceneAssembly::from_native(assembled, &AssemblyOptions::default(), |_, _| {
            Ok(Arc::new(crate::gpu::tests::triangle()))
        })
        .unwrap();
        let node = scene.components["garment"];
        assert_eq!(
            scene.scene.world_transforms().unwrap()[node.0].0.w_axis.x,
            5.
        );
    }

    #[test]
    fn identical_native_slots_resolve_but_conflicting_slots_are_rejected() {
        let mut assembled = native();
        assembled.components[0].kind = "entMeshComponent".into();
        assembled.components[0].parent = Binding {
            component: "slots".into(),
            slot: "mount".into(),
        };
        let mut owner = assembled.components[0].clone();
        owner.name = "slots".into();
        owner.mesh = None;
        owner.parent = Binding::default();
        let slot = ghostline_red::assets::Slot {
            name: "mount".into(),
            bone: String::new(),
            transform: ghostline_red::assets::Transform::default(),
        };
        owner.slots = vec![slot.clone(), slot];
        assembled.components.push(owner);
        assert!(
            SceneAssembly::from_native(assembled.clone(), &AssemblyOptions::default(), |_, _| Ok(
                Arc::new(crate::gpu::tests::triangle())
            ))
            .is_ok()
        );
        assembled.components[1].slots[1].bone = "conflicting".into();
        assert!(
            SceneAssembly::from_native(assembled, &AssemblyOptions::default(), |_, _| Ok(
                Arc::new(crate::gpu::tests::triangle())
            ))
            .is_err()
        );
    }
    #[test]
    fn standalone_origin_context_preserves_mesh_local_placement() {
        let mut assembled = native();
        assembled.components[0].kind = "entMeshComponent".into();
        assembled.components[0].local = Mat4::from_translation(glam::Vec3::new(2., 3., 4.));
        assert!(
            SceneAssembly::from_native(assembled.clone(), &AssemblyOptions::default(), |_, _| {
                Ok(Arc::new(crate::gpu::tests::triangle()))
            })
            .is_err()
        );
        let options = AssemblyOptions {
            origin_anchors: ["external_character".into()].into_iter().collect(),
            ..Default::default()
        };
        let scene = SceneAssembly::from_native(assembled, &options, |_, _| {
            Ok(Arc::new(crate::gpu::tests::triangle()))
        })
        .unwrap();
        let node = scene.components["garment"];
        let world = scene.scene.world_transforms().unwrap()[node.0].0;
        let expected =
            basis() * Mat4::from_translation(glam::Vec3::new(2., 3., 4.)) * basis().transpose();
        assert_eq!(world, expected);
        assert!(
            scene
                .preview_notes
                .iter()
                .any(|note| note.contains("external_character"))
        );
    }

    fn native() -> Assembly {
        Assembly {
            source: ResourceKey::Hash(1),
            appearance: "default".into(),
            rigs: HashMap::new(),
            components: vec![AssemblyComponent {
                template: Arc::new(serde_json::Value::Null),
                name: "garment".into(),
                kind: "entSkinnedMeshComponent".into(),
                local: Mat4::IDENTITY,
                parent: Binding {
                    component: "external_character".into(),
                    slot: "spine".into(),
                },
                control: Binding::default(),
                rig: None,
                mesh: Some(ResourceKey::Hash(2)),
                mesh_appearance: "default".into(),
                mesh_appearance_requested: None,
                chunk_mask: u64::MAX,
                enabled: true,
                slots: Vec::new(),
            }],
        }
    }

    #[test]
    fn isolated_bind_pose_is_explicit_and_preserves_native_bindings() {
        assert!(
            SceneAssembly::from_native(native(), &AssemblyOptions::default(), |_, _| Ok(Arc::new(
                crate::gpu::tests::triangle()
            )))
            .is_err()
        );
        let scene = SceneAssembly::from_native(
            native(),
            &AssemblyOptions {
                bind_pose: true,
                ..Default::default()
            },
            |_, _| Ok(Arc::new(crate::gpu::tests::triangle())),
        )
        .unwrap();
        assert_eq!(scene.scene.assets.len(), 1);
        assert!(scene.rigs.is_empty());
        assert_eq!(
            scene.native.components[0].parent.component,
            "external_character"
        );
        let mut cyclic = native();
        cyclic.components[0].parent = Binding {
            component: "garment".into(),
            slot: String::new(),
        };
        assert!(
            SceneAssembly::from_native(
                cyclic,
                &AssemblyOptions {
                    bind_pose: true,
                    ..Default::default()
                },
                |_, _| Ok(Arc::new(crate::gpu::tests::triangle()))
            )
            .is_err()
        );
    }
}
