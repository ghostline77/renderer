//! Mesh instances and their component hierarchy.

use std::sync::Arc;

use glam::{Affine3A, Mat4, Vec3};

use crate::{
    Error, MeshAsset, Pose, RigId, Skeleton,
    pose::{RigInstance, SkinBinding},
};

/// Index of an asset stored in a scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AssetId(pub(crate) usize);

/// Index of a node stored in a scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) usize);

/// A component transform with optional geometry and a parent component.
#[derive(Clone, Debug)]
pub struct Node {
    /// Component name used by callers to identify instances.
    pub name: String,
    pub(crate) asset: Option<AssetId>,
    pub(crate) parent: Option<NodeId>,
    pub(crate) transform: Mat4,
    pub(crate) chunk_mask: u64,
    pub(crate) visible: bool,
    pub(crate) reference: bool,
}

impl Node {
    /// Creates a visible root node with identity transform and all chunks enabled.
    pub fn new(name: impl Into<String>, asset: Option<AssetId>) -> Self {
        Self {
            name: name.into(),
            asset,
            parent: None,
            transform: Mat4::IDENTITY,
            chunk_mask: u64::MAX,
            visible: true,
            reference: false,
        }
    }

    /// Sets the local-to-parent transform.
    pub fn with_transform(mut self, transform: Mat4) -> Self {
        self.transform = transform;
        self
    }

    /// Sets the parent component.
    pub fn with_parent(mut self, parent: NodeId) -> Self {
        self.parent = Some(parent);
        self
    }

    /// Sets the native render-chunk visibility bits.
    pub fn with_chunk_mask(mut self, mask: u64) -> Self {
        self.chunk_mask = mask;
        self
    }
}

/// CPU-side scene containing shared immutable assets and editable instances.
#[derive(Clone, Debug, Default)]
pub struct Scene {
    pub(crate) assets: Vec<Arc<MeshAsset>>,
    pub(crate) nodes: Vec<Node>,
    pub(crate) rigs: Vec<RigInstance>,
    pub(crate) skins: std::collections::HashMap<NodeId, SkinBinding>,
    pub(crate) rig_links: Vec<crate::pose::RigLink>,
}

impl Scene {
    /// Retargets named control-rig motion onto a deformation rig's authored rest pose.
    ///
    /// Shared bones receive model-space motion deltas. Unanimated helper bones
    /// retain their local rest transforms under their posed parents.
    ///
    /// # Errors
    /// Returns [`Error`] for absent rigs, no shared bones, competing drivers or cycles.
    pub fn link_rigs(&mut self, source: RigId, destination: RigId) -> Result<(), Error> {
        let from = self
            .rigs
            .get(source.0)
            .ok_or_else(|| Error::InvalidInput("absent control rig".into()))?;
        let to = self
            .rigs
            .get(destination.0)
            .ok_or_else(|| Error::InvalidInput("absent deformation rig".into()))?;
        if source == destination
            || self
                .rig_links
                .iter()
                .any(|link| link.destination == destination)
        {
            return Err(Error::InvalidInput("cyclic or competing rig driver".into()));
        }
        let mut pending = vec![destination];
        while let Some(rig) = pending.pop() {
            for link in self.rig_links.iter().filter(|link| link.source == rig) {
                if link.destination == source {
                    return Err(Error::InvalidInput("cyclic rig control bindings".into()));
                }
                pending.push(link.destination);
            }
        }
        let mapping: Vec<_> = to
            .skeleton
            .names
            .iter()
            .map(|name| {
                from.skeleton
                    .names
                    .iter()
                    .position(|candidate| name == candidate)
            })
            .collect();
        if mapping.iter().all(Option::is_none) {
            return Err(Error::InvalidInput(
                "control and deformation rigs have no shared bone names".into(),
            ));
        }
        self.rig_links.push(crate::pose::RigLink {
            source,
            destination,
            mapping,
        });
        self.update_rig_links()
    }

    pub(crate) fn update_rig_links(&mut self) -> Result<(), Error> {
        let mut indegree = vec![0_usize; self.rigs.len()];
        for link in &self.rig_links {
            indegree[link.destination.0] += 1;
        }
        let mut pending: std::collections::VecDeque<_> = indegree
            .iter()
            .enumerate()
            .filter(|(_, count)| **count == 0)
            .map(|(index, _)| RigId(index))
            .collect();
        while let Some(source) = pending.pop_front() {
            let outgoing: Vec<_> = self
                .rig_links
                .iter()
                .filter(|link| link.source == source)
                .cloned()
                .collect();
            for link in outgoing {
                let pose = self.retarget_pose(&link)?;
                for (&node, &local) in self.rigs[link.destination.0].bones.iter().zip(&pose.local) {
                    self.nodes[node.0].transform = local;
                }
                indegree[link.destination.0] -= 1;
                if indegree[link.destination.0] == 0 {
                    pending.push_back(link.destination);
                }
            }
        }
        Ok(())
    }

    fn retarget_pose(&self, link: &crate::pose::RigLink) -> Result<Pose, Error> {
        let source = &self.rigs[link.source.0];
        let destination = &self.rigs[link.destination.0];
        let source_pose = Pose {
            local: source
                .bones
                .iter()
                .map(|node| self.nodes[node.0].transform)
                .collect(),
        };
        let source_model = source.skeleton.model_matrices(&source_pose)?;
        let source_rest = source
            .skeleton
            .model_matrices(source.skeleton.rest_pose())?;
        let destination_rest = destination
            .skeleton
            .model_matrices(destination.skeleton.rest_pose())?;
        let worlds = self.world_transforms()?;
        // These transforms are affine by construction. Keeping the type affine
        // avoids accumulating roundoff in the homogeneous row when several
        // linked rigs contain small or nonuniform authored scales.
        let destination_from_source = Affine3A::from_mat4(worlds[destination.root.0].0).inverse()
            * Affine3A::from_mat4(worlds[source.root.0].0);
        let source_from_destination = destination_from_source.inverse();
        let mut models = vec![Affine3A::IDENTITY; destination.bones.len()];
        let mut local = vec![Mat4::IDENTITY; destination.bones.len()];
        for bone in crate::pose::evaluation_order(&destination.skeleton.parents) {
            let parent =
                destination.skeleton.parents[bone].map_or(Affine3A::IDENTITY, |p| models[p]);
            models[bone] = if let Some(mapped) = link.mapping[bone] {
                destination_from_source
                    * Affine3A::from_mat4(source_model[mapped])
                    * Affine3A::from_mat4(source_rest[mapped]).inverse()
                    * source_from_destination
                    * Affine3A::from_mat4(destination_rest[bone])
            } else {
                parent * Affine3A::from_mat4(destination.skeleton.rest.local[bone])
            };
            local[bone] = Mat4::from(parent.inverse() * models[bone]);
            validate_transform(local[bone])?;
        }
        Ok(Pose { local })
    }
    /// Adds a named skeleton beneath a component root, including attachment nodes.
    ///
    /// # Errors
    /// Returns [`Error`] for an absent root or invalid skeleton.
    pub fn add_rig(&mut self, root: NodeId, skeleton: Arc<Skeleton>) -> Result<RigId, Error> {
        if root.0 >= self.nodes.len() {
            return Err(Error::InvalidInput("absent rig root".into()));
        }
        skeleton.model_matrices(skeleton.rest_pose())?;
        let base = self.nodes.len();
        let bones: Vec<_> = (0..skeleton.names.len())
            .map(|i| NodeId(base + i))
            .collect();
        for (index, name) in skeleton.names.iter().enumerate() {
            let mut node = Node::new(name, None).with_transform(skeleton.rest.local[index]);
            node.parent = Some(skeleton.parents[index].map_or(root, |p| bones[p]));
            self.nodes.push(node);
        }
        let id = RigId(self.rigs.len());
        self.rigs.push(RigInstance {
            skeleton,
            root,
            bones,
        });
        Ok(id)
    }

    /// Looks up a skeleton bone's transform node for rigid attachments.
    pub fn bone_node(&self, rig: RigId, name: &str) -> Option<NodeId> {
        let rig = self.rigs.get(rig.0)?;
        rig.skeleton
            .names
            .iter()
            .position(|n| n == name)
            .map(|i| rig.bones[i])
    }

    /// Updates a complete skeleton pose and all rigid attachments under its bones.
    ///
    /// # Errors
    /// Returns [`Error`] for invalid rig identities, counts or bone transforms.
    pub fn set_pose(&mut self, rig: RigId, pose: &Pose) -> Result<(), Error> {
        if self.rig_links.iter().any(|link| link.destination == rig) {
            return Err(Error::InvalidInput(
                "rig is driven by a control rig; update the driver or unlink it first".into(),
            ));
        }
        let rig = self
            .rigs
            .get(rig.0)
            .ok_or_else(|| Error::InvalidInput("absent scene rig".into()))?;
        rig.skeleton.model_matrices(pose)?;
        for (&node, &local) in rig.bones.iter().zip(&pose.local) {
            self.nodes[node.0].transform = local;
        }
        self.update_rig_links()
    }

    /// Removes a control binding, preserving the deformation rig's current pose.
    pub fn unlink_rig(&mut self, destination: RigId) -> bool {
        let before = self.rig_links.len();
        self.rig_links
            .retain(|link| link.destination != destination);
        self.rig_links.len() != before
    }

    /// Binds a skinned instance to a skeleton using names and its current rest frame.
    ///
    /// Bone motion is retargeted onto the mesh's authored bind pose. Instances
    /// sharing the same mesh can bind to different skeletons and poses.
    ///
    /// # Errors
    /// Returns [`Error`] for absent nodes/rigs, missing palette bones or invalid skin data.
    pub fn bind_skin(&mut self, node: NodeId, rig: RigId) -> Result<(), Error> {
        self.bind_skin_with_fallbacks(node, rig, &[])
    }

    /// Binds a mesh palette across its native primary and linked support rigs.
    /// Primary-rig bones win; fallbacks provide body bones missing from cloth rigs.
    ///
    /// # Errors
    /// Returns an error for invalid nodes, rigs, skin data or unresolved weighted bones.
    pub fn bind_skin_with_fallbacks(
        &mut self,
        node: NodeId,
        rig: RigId,
        fallbacks: &[RigId],
    ) -> Result<(), Error> {
        let component = self
            .nodes
            .get(node.0)
            .ok_or_else(|| Error::InvalidInput("absent skin node".into()))?;
        let asset = component
            .asset
            .map(|a| &self.assets[a.0])
            .ok_or_else(|| Error::InvalidInput("skin node has no mesh".into()))?;
        let skin = asset
            .skin
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("mesh has no bone palette".into()))?;
        self.rigs
            .get(rig.0)
            .ok_or_else(|| Error::InvalidInput("absent skin rig".into()))?;
        let mut models = std::collections::HashMap::new();
        let worlds = self.world_transforms()?;
        let mut palette = Vec::with_capacity(skin.names.len());
        let mut bind_from_mesh = Vec::with_capacity(skin.names.len());
        let mut used_palette = vec![false; skin.names.len()];
        for vertex in asset.chunks.iter().flat_map(|chunk| &chunk.skin) {
            for (&joint, weight) in vertex.joints.iter().zip(vertex.weights) {
                if weight > 0. {
                    *used_palette.get_mut(usize::from(joint)).ok_or_else(|| {
                        Error::InvalidInput("skin index exceeds bone palette".into())
                    })? = true;
                }
            }
        }
        for (index, (name, inverse_bind)) in skin.names.iter().zip(&skin.inverse_bind).enumerate() {
            if !used_palette[index] {
                // Native palettes may retain unused sentinel entries, including
                // `None`. They never participate in deformation.
                palette.push(node);
                bind_from_mesh.push(Mat4::IDENTITY);
                continue;
            }
            let (source, bone) = std::iter::once(rig)
                .chain(fallbacks.iter().copied())
                .find_map(|source| {
                    self.rigs
                        .get(source.0)?
                        .skeleton
                        .names
                        .iter()
                        .position(|n| n == name)
                        .map(|bone| (source, bone))
                })
                .ok_or_else(|| {
                    Error::InvalidInput(format!("skin palette bone {name} is absent from rig"))
                })?;
            let instance = &self.rigs[source.0];
            if let std::collections::hash_map::Entry::Vacant(entry) = models.entry(source.0) {
                entry.insert(
                    instance
                        .skeleton
                        .model_matrices(instance.skeleton.rest_pose())?,
                );
            }
            let mesh_to_rig = worlds[instance.root.0].0.inverse() * worlds[node.0].0;
            palette.push(instance.bones[bone]);
            let authored_bind = inverse_bind.inverse();
            let calibration = models[&source.0][bone].inverse() * mesh_to_rig * authored_bind;
            bind_from_mesh.push(calibration * *inverse_bind);
        }
        for chunk in asset.chunks.iter() {
            if !chunk.skin.is_empty() && chunk.skin.len() != chunk.positions.len() {
                return Err(Error::InvalidInput(
                    "skin vertex count differs from geometry".into(),
                ));
            }
            if chunk.skin.iter().any(|v| {
                v.joints
                    .iter()
                    .zip(v.weights)
                    .any(|(&joint, weight)| weight > 0.0 && usize::from(joint) >= palette.len())
            }) {
                return Err(Error::InvalidInput(
                    "skin index exceeds bone palette".into(),
                ));
            }
        }
        self.skins.insert(
            node,
            SkinBinding {
                palette,
                bind_from_mesh,
            },
        );
        Ok(())
    }
    /// Creates an empty scene.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores geometry once for use by any number of nodes.
    pub fn add_asset(&mut self, asset: MeshAsset) -> AssetId {
        self.add_shared_asset(Arc::new(asset))
    }

    /// Shares an asset across scenes without copying its geometry or materials.
    pub fn add_shared_asset(&mut self, asset: Arc<MeshAsset>) -> AssetId {
        if let Some(index) = self
            .assets
            .iter()
            .position(|existing| Arc::ptr_eq(existing, &asset))
        {
            return AssetId(index);
        }
        let id = AssetId(self.assets.len());
        self.assets.push(asset);
        id
    }

    /// Adds a component after validating its asset, parent and transform.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] for absent references or invalid transforms.
    pub fn add_node(&mut self, node: Node) -> Result<NodeId, Error> {
        validate_transform(node.transform)?;
        if node.asset.is_some_and(|asset| asset.0 >= self.assets.len())
            || node
                .parent
                .is_some_and(|parent| parent.0 >= self.nodes.len())
        {
            return Err(Error::InvalidInput(
                "scene node references an absent asset or parent".into(),
            ));
        }
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        Ok(id)
    }

    /// Reads a component by its scene-local identity.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.0)
    }

    /// Returns the shared mesh asset of one scene instance, when it has geometry.
    #[must_use]
    pub fn asset_for_node(&self, node: NodeId) -> Option<&MeshAsset> {
        let asset = self.nodes.get(node.0)?.asset?;
        self.assets.get(asset.0).map(Arc::as_ref)
    }

    /// Number of stored assets.
    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }

    /// Number of components, including transform-only components.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Updates a component's local transform.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] for an absent node or invalid transform.
    pub fn set_transform(&mut self, node: NodeId, transform: Mat4) -> Result<(), Error> {
        validate_transform(transform)?;
        self.node_mut(node)?.transform = transform;
        Ok(())
    }

    /// Updates native chunk visibility without changing shared geometry.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if the node is absent.
    pub fn set_chunk_mask(&mut self, node: NodeId, mask: u64) -> Result<(), Error> {
        self.node_mut(node)?.chunk_mask = mask;
        Ok(())
    }

    /// Hides or shows a component and its descendants.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if the node is absent.
    pub fn set_visible(&mut self, node: NodeId, visible: bool) -> Result<(), Error> {
        self.node_mut(node)?.visible = visible;
        Ok(())
    }

    /// Changes a parent while rejecting cyclic hierarchies.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] for absent nodes or a cycle.
    pub fn set_parent(&mut self, node: NodeId, parent: Option<NodeId>) -> Result<(), Error> {
        if self.rigs.iter().any(|rig| rig.bones.contains(&node)) {
            return Err(Error::InvalidInput(
                "skeleton bone parents are fixed by the rig".into(),
            ));
        }
        if node.0 >= self.nodes.len() {
            return Err(Error::InvalidInput("absent scene node".into()));
        }
        let mut current = parent;
        while let Some(id) = current {
            if id == node {
                return Err(Error::InvalidInput("cyclic scene hierarchy".into()));
            }
            current = self
                .nodes
                .get(id.0)
                .ok_or_else(|| Error::InvalidInput("absent scene parent".into()))?
                .parent;
        }
        self.nodes[node.0].parent = parent;
        Ok(())
    }

    pub(crate) fn world_transforms(&self) -> Result<Vec<(Mat4, bool)>, Error> {
        resolve_transforms(&self.nodes)
    }

    fn node_mut(&mut self, node: NodeId) -> Result<&mut Node, Error> {
        self.nodes
            .get_mut(node.0)
            .ok_or_else(|| Error::InvalidInput("absent scene node".into()))
    }
}

pub(crate) fn validate_transform(transform: Mat4) -> Result<(), Error> {
    let row = transform.row(3);
    if !transform.is_finite()
        || !transform.inverse().is_finite()
        || row.truncate().length_squared() > 1e-12
        || (row.w - 1.0).abs() > 1e-6
    {
        return Err(Error::InvalidInput(
            "scene transform must be finite, affine and invertible".into(),
        ));
    }
    Ok(())
}

pub(crate) fn resolve_transforms(nodes: &[Node]) -> Result<Vec<(Mat4, bool)>, Error> {
    let mut result = vec![(Mat4::IDENTITY, true); nodes.len()];
    let mut state = vec![0_u8; nodes.len()];
    for index in 0..nodes.len() {
        let mut path = Vec::new();
        let mut current = Some(index);
        while let Some(id) = current {
            if id >= nodes.len() {
                return Err(Error::InvalidInput("absent scene parent".into()));
            }
            if state[id] == 2 {
                break;
            }
            if state[id] == 1 {
                return Err(Error::InvalidInput("cyclic scene hierarchy".into()));
            }
            state[id] = 1;
            path.push(id);
            current = nodes[id].parent.map(|p| p.0);
        }
        for id in path.into_iter().rev() {
            let parent = nodes[id]
                .parent
                .map_or((Mat4::IDENTITY, true), |p| result[p.0]);
            result[id] = (
                parent.0 * nodes[id].transform,
                parent.1 && nodes[id].visible,
            );
            validate_transform(result[id].0)?;
            state[id] = 2;
        }
    }
    Ok(result)
}

pub(crate) fn chunk_visible(mask: u64, index: usize) -> bool {
    index < 64 && mask & (1_u64 << index) != 0
}

pub(crate) fn transformed_bounds(points: impl Iterator<Item = Vec3>) -> Result<(Vec3, f32), Error> {
    let mut lower = Vec3::splat(f32::INFINITY);
    let mut upper = Vec3::splat(f32::NEG_INFINITY);
    for point in points {
        if !point.is_finite() {
            return Err(Error::InvalidInput("nonfinite scene position".into()));
        }
        lower = lower.min(point);
        upper = upper.max(point);
    }
    let radius = (upper - lower).length() * 0.5;
    if !radius.is_finite() || radius <= 0.0 {
        return Err(Error::InvalidInput(
            "scene has empty or degenerate bounds".into(),
        ));
    }
    Ok(((lower + upper) * 0.5, radius))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skin_motion_is_relative_to_geometry_a_pose_not_animation_reference() {
        use ghostline_red::{
            assets::{Bone, Rig, Transform},
            mesh::{MeshSkin, VertexSkin},
        };
        let skeleton = Arc::new(
            Skeleton::from_rig(&Rig {
                source: None,
                bones: vec![Bone {
                    name: "Root".into(),
                    parent: None,
                    rest: Transform::default(),
                }],
                a_pose: Some(vec![Transform {
                    translation: [2., 0., 0.],
                    ..Transform::default()
                }]),
            })
            .unwrap(),
        );
        let mut mesh = crate::gpu::tests::triangle();
        mesh.skin = Some(MeshSkin {
            names: vec!["Root".into()],
            inverse_bind: vec![Mat4::from_translation(-Vec3::X * 2.)],
        });
        let chunk = &mut Arc::make_mut(&mut mesh.chunks)[0];
        let original = chunk.positions.clone();
        chunk.skin = vec![
            VertexSkin {
                joints: [0; 8],
                weights: [1., 0., 0., 0., 0., 0., 0., 0.]
            };
            chunk.positions.len()
        ];
        let mut scene = Scene::new();
        let asset = scene.add_asset(mesh);
        let root = scene.add_node(Node::new("actor", None)).unwrap();
        let node = scene
            .add_node(Node::new("mesh", Some(asset)).with_parent(root))
            .unwrap();
        let rig = scene.add_rig(root, skeleton).unwrap();
        scene.bind_skin(node, rig).unwrap();
        assert_eq!(scene.evaluated_geometry().unwrap()[0].positions, original);
        // The clip moves the bone from its A-pose x=2 to its reference x=0.
        scene
            .set_pose(
                rig,
                &Pose {
                    local: vec![Mat4::IDENTITY],
                },
            )
            .unwrap();
        for (point, original) in scene.evaluated_geometry().unwrap()[0]
            .positions
            .iter()
            .zip(original)
        {
            assert!((Vec3::from(*point) - (Vec3::from(original) - Vec3::X * 2.)).length() < 1e-6);
        }
    }

    #[test]
    fn control_rig_motion_preserves_unanimated_deformation_helpers() {
        use ghostline_red::assets::{Bone, Rig, Transform};
        let bone = |name: &str, parent, y| Bone {
            name: name.into(),
            parent,
            rest: Transform {
                translation: [0., y, 0.],
                ..Transform::default()
            },
        };
        let control = Arc::new(
            Skeleton::from_rig(&Rig {
                a_pose: None,
                source: None,
                bones: vec![bone("root", None, 0.), bone("hand", Some(0), 1.)],
            })
            .unwrap(),
        );
        let deform = Arc::new(
            Skeleton::from_rig(&Rig {
                a_pose: None,
                source: None,
                bones: vec![
                    bone("root", None, 0.),
                    bone("helper", Some(0), 0.5),
                    bone("hand", Some(1), 0.5),
                ],
            })
            .unwrap(),
        );
        let mut scene = Scene::new();
        let from = scene.add_node(Node::new("control", None)).unwrap();
        let to = scene
            .add_node(
                Node::new("deform", None).with_transform(Mat4::from_translation(Vec3::X * 0.5)),
            )
            .unwrap();
        let source = scene.add_rig(from, Arc::clone(&control)).unwrap();
        let destination = scene.add_rig(to, Arc::clone(&deform)).unwrap();
        scene.link_rigs(source, destination).unwrap();
        let mut pose = control.rest_pose().clone();
        pose.local[0] = Mat4::from_translation(Vec3::X * 0.2) * pose.local[0];
        pose.local[1] = Mat4::from_translation(Vec3::X * 0.5) * pose.local[1];
        scene.set_pose(source, &pose).unwrap();
        let helper = scene.bone_node(destination, "helper").unwrap();
        let hand = scene.bone_node(destination, "hand").unwrap();
        let worlds = scene.world_transforms().unwrap();
        assert!((worlds[helper.0].0.w_axis.x - 0.7).abs() < 1e-6);
        assert!((worlds[hand.0].0.w_axis.x - 1.2).abs() < 1e-6);
        assert!(scene.link_rigs(destination, source).is_err());
        assert!(scene.set_pose(destination, deform.rest_pose()).is_err());
        assert!(scene.unlink_rig(destination));
        scene.set_pose(destination, deform.rest_pose()).unwrap();
    }

    #[test]
    fn hierarchy_updates_preserve_parent_transforms_and_reject_cycles() {
        let mut scene = Scene::new();
        let root = scene
            .add_node(Node::new("root", None).with_transform(Mat4::from_translation(Vec3::X)))
            .unwrap();
        let child = scene
            .add_node(
                Node::new("child", None)
                    .with_parent(root)
                    .with_transform(Mat4::from_translation(Vec3::Y)),
            )
            .unwrap();
        let transforms = scene.world_transforms().unwrap();
        assert!(
            (transforms[child.0].0.transform_point3(Vec3::ZERO) - Vec3::new(1.0, 1.0, 0.0))
                .length()
                < 1e-6
        );
        assert!(scene.set_parent(root, Some(child)).is_err());
        assert!(
            scene
                .set_transform(child, Mat4::from_scale(Vec3::ZERO))
                .is_err()
        );
        scene.set_visible(root, false).unwrap();
        assert!(!scene.world_transforms().unwrap()[child.0].1);
        scene.set_parent(child, None).unwrap();
        assert!(scene.world_transforms().unwrap()[child.0].1);
    }
}
