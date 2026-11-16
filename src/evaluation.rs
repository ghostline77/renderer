//! CPU evaluation of the same calibrated skin palettes used for GPU rendering.
use crate::{Error, NodeId, RigId, Scene, Skeleton};
use glam::{Mat4, Vec3};

/// Visible native chunk evaluated in scene world coordinates.
#[derive(Clone, Debug)]
pub struct EvaluatedChunk {
    /// Scene node retaining this mesh instance.
    pub node: NodeId,
    /// Native component name.
    pub name: String,
    /// Source native mesh identity.
    pub mesh: String,
    /// Original render-chunk index.
    pub chunk: usize,
    /// Skeletally deformed world positions, in metres with Y up.
    pub positions: Vec<[f32; 3]>,
    /// Native smooth normals transformed by the calibrated inverse skin matrix.
    pub normals: Vec<[f32; 3]>,
    /// Source triangle winding and topology.
    pub indices: Vec<u16>,
    /// Per-vertex bind-space to world-space deformation, for inverse corrections.
    pub transforms: Vec<Mat4>,
}

impl Scene {
    /// Skeleton for a scene rig, including native bone names and authored rest.
    ///
    /// # Errors
    /// Returns an error for an absent rig.
    pub fn skeleton(&self, rig: RigId) -> Result<&Skeleton, Error> {
        self.rigs
            .get(rig.0)
            .map(|r| r.skeleton.as_ref())
            .ok_or_else(|| Error::InvalidInput("absent scene rig".into()))
    }

    /// Evaluates visible geometry using the GPU's calibrated skin bindings.
    /// Morph targets are not applied; use this for body/garment skeletal checks.
    ///
    /// # Errors
    /// Returns an error for invalid transforms, skin indices or singular blends.
    pub fn evaluated_geometry(&self) -> Result<Vec<EvaluatedChunk>, Error> {
        let worlds = self.world_transforms()?;
        let mut result = Vec::new();
        for (index, node) in self.nodes.iter().enumerate().filter(|(i, _)| worlds[*i].1) {
            let Some(asset) = node.asset.map(|id| &self.assets[id.0]) else {
                continue;
            };
            let palette: Vec<_> = self
                .skins
                .get(&NodeId(index))
                .into_iter()
                .flat_map(|binding| {
                    binding
                        .palette
                        .iter()
                        .enumerate()
                        .map(|(i, bone)| worlds[bone.0].0 * binding.bind_from_mesh[i])
                })
                .collect();
            for chunk in asset
                .chunks
                .iter()
                .filter(|c| crate::scene::chunk_visible(node.chunk_mask, c.chunk_index))
            {
                let mut positions = Vec::with_capacity(chunk.positions.len());
                let mut normals = Vec::with_capacity(chunk.positions.len());
                let mut transforms = Vec::with_capacity(chunk.positions.len());
                for (i, point) in chunk.positions.iter().enumerate() {
                    let transform =
                        if let Some(skin) = chunk.skin.get(i).filter(|_| !palette.is_empty()) {
                            let mut matrix = Mat4::ZERO;
                            for (&joint, &weight) in skin
                                .joints
                                .iter()
                                .zip(&skin.weights)
                                .filter(|(_, w)| **w > 0.)
                            {
                                matrix += *palette.get(usize::from(joint)).ok_or_else(|| {
                                    Error::InvalidInput("skin index exceeds palette".into())
                                })? * weight;
                            }
                            matrix
                        } else {
                            worlds[index].0
                        };
                    if !transform.is_finite() || transform.determinant().abs() < 1e-12 {
                        return Err(Error::InvalidInput("singular evaluated skin blend".into()));
                    }
                    positions.push(
                        transform
                            .transform_point3(Vec3::from_array(*point))
                            .to_array(),
                    );
                    normals.push(
                        chunk
                            .normals
                            .get(i)
                            .map_or([0.; 3], |normal| transformed_normal(transform, *normal)),
                    );
                    transforms.push(transform);
                }
                result.push(EvaluatedChunk {
                    node: NodeId(index),
                    name: node.name.clone(),
                    mesh: asset.mesh.clone(),
                    chunk: chunk.chunk_index,
                    positions,
                    normals,
                    indices: chunk.indices.clone(),
                    transforms,
                });
            }
        }
        Ok(result)
    }
}

fn transformed_normal(transform: Mat4, normal: [f32; 3]) -> [f32; 3] {
    transform
        .inverse()
        .transpose()
        .transform_vector3(Vec3::from_array(normal))
        .normalize_or_zero()
        .to_array()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluated_normals_stay_perpendicular_under_scaled_skinning() {
        let transform = Mat4::from_scale_rotation_translation(
            Vec3::new(2., 1., 0.5),
            glam::Quat::from_rotation_y(0.6),
            Vec3::new(3., 4., 5.),
        );
        let normal = Vec3::new(1., 1., 0.).normalize();
        let tangent = Vec3::new(1., -1., 0.).normalize();
        let evaluated = Vec3::from_array(transformed_normal(transform, normal.to_array()));
        assert!((evaluated.length() - 1.).abs() < 1e-6);
        assert!(evaluated.dot(transform.transform_vector3(tangent)).abs() < 1e-6);
    }
}
