use glam::Mat4;

use crate::{Error, Node, NodeId, RigId, Scene};

impl Scene {
    /// Appends an independent scene beneath a new placement node.
    ///
    /// Shared immutable assets remain shared. Node, bone, skin and rig-link
    /// identities are remapped together so repeated native entities can retain
    /// their own hierarchies and skeleton instances in a larger world preview.
    /// The returned node controls the whole appended instance.
    ///
    /// # Errors
    /// Returns [`Error`] if the source hierarchy or placement transform is invalid.
    pub fn append_scene(
        &mut self,
        source: &Scene,
        name: impl Into<String>,
        transform: Mat4,
    ) -> Result<NodeId, Error> {
        source.world_transforms()?;
        let root = self.add_node(Node::new(name, None).with_transform(transform))?;
        let asset_ids: Vec<_> = source
            .assets
            .iter()
            .map(|asset| self.add_shared_asset(std::sync::Arc::clone(asset)))
            .collect();
        let offset = self.nodes.len();
        let node_id = |id: NodeId| NodeId(id.0 + offset);
        self.nodes
            .extend(source.nodes.iter().cloned().map(|mut node| {
                node.asset = node.asset.map(|asset| asset_ids[asset.0]);
                node.parent = Some(node.parent.map_or(root, node_id));
                node
            }));
        let rig_offset = self.rigs.len();
        self.rigs.extend(source.rigs.iter().cloned().map(|mut rig| {
            rig.root = node_id(rig.root);
            rig.bones = rig.bones.into_iter().map(node_id).collect();
            rig
        }));
        self.skins.extend(source.skins.iter().map(|(&node, skin)| {
            let mut skin = skin.clone();
            skin.palette = skin.palette.into_iter().map(node_id).collect();
            (node_id(node), skin)
        }));
        self.rig_links
            .extend(source.rig_links.iter().cloned().map(|mut link| {
                link.source = RigId(link.source.0 + rig_offset);
                link.destination = RigId(link.destination.0 + rig_offset);
                link
            }));
        Ok(root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn repeated_scenes_have_independent_placement_and_parent_identity() {
        let mut source = Scene::new();
        let parent = source.add_node(Node::new("parent", None)).unwrap();
        source
            .add_node(
                Node::new("child", None)
                    .with_parent(parent)
                    .with_transform(Mat4::from_translation(Vec3::Y)),
            )
            .unwrap();
        let mut scene = Scene::new();
        scene
            .append_scene(&source, "first", Mat4::from_translation(Vec3::X))
            .unwrap();
        scene
            .append_scene(&source, "second", Mat4::from_translation(Vec3::Z))
            .unwrap();
        let transforms = scene.world_transforms().unwrap();
        assert_eq!(
            transforms[2].0.transform_point3(Vec3::ZERO),
            Vec3::new(1., 1., 0.)
        );
        assert_eq!(
            transforms[5].0.transform_point3(Vec3::ZERO),
            Vec3::new(0., 1., 1.)
        );
    }
}
