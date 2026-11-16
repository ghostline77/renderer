//! Native calibrated skin palettes and sampled local transforms for browser playback.
use super::{Glb, invalid};
use crate::{Error, NodeId, Scene};
use glam::{Mat4, Quat};
use serde_json::json;
use std::collections::HashMap;

type LocalTrs = ([f32; 3], [f32; 4], [f32; 3]);

fn trs(matrix: Mat4) -> Result<LocalTrs, Error> {
    let (scale, rotation, translation) = matrix.to_scale_rotation_translation();
    let rebuilt = Mat4::from_scale_rotation_translation(scale, rotation, translation);
    let error = matrix
        .to_cols_array()
        .iter()
        .zip(rebuilt.to_cols_array())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max);
    if !scale.is_finite() || !rotation.is_finite() || !translation.is_finite() || error > 0.002 {
        return Err(invalid(
            "browser animation requires decomposable local transforms",
        ));
    }
    Ok((
        translation.to_array(),
        rotation.normalize().to_array(),
        scale.to_array(),
    ))
}

impl Scene {
    /// Exports a self-contained animated GLB using calibrated native skin bindings.
    /// The callback updates the cloned scene at each strictly increasing sample time.
    /// All eight native influences are retained as JOINTS/WEIGHTS_0 and _1.
    /// Browser clients must support both influence sets; native shader effects are approximated.
    ///
    /// # Errors
    /// Returns binding, texture, non-TRS transform, sampling or GLB size errors.
    /// At most 901 samples and three million sampled node transforms are accepted.
    pub fn browser_animation_glb(
        &self,
        name: &str,
        times: &[f32],
        mut update: impl FnMut(&mut Scene, f32) -> Result<(), Error>,
    ) -> Result<Vec<u8>, Error> {
        if times.is_empty()
            || times.len() > 901
            || times.iter().any(|v| !v.is_finite() || *v < 0.)
            || times.windows(2).any(|v| v[0] >= v[1])
            || times.len().saturating_mul(self.nodes.len()) > 3_000_000
        {
            return Err(invalid(
                "invalid or excessive browser animation sample times",
            ));
        }
        let mut scene = self.clone();
        update(&mut scene, times[0])?;
        let worlds = scene.world_transforms()?;
        let mut glb = Glb::default();
        let mut nodes = Vec::new();
        for node in &scene.nodes {
            let (translation, rotation, scale) = trs(node.transform)?;
            nodes.push(json!({"name":node.name,"translation":translation,"rotation":rotation,"scale":scale}));
        }
        let mut roots = Vec::new();
        for (i, node) in scene.nodes.iter().enumerate() {
            if let Some(parent) = node.parent {
                if nodes[parent.0].get("children").is_none() {
                    nodes[parent.0]["children"] = json!([]);
                }
                nodes[parent.0]["children"]
                    .as_array_mut()
                    .expect("initialized children")
                    .push(json!(i));
            } else {
                roots.push(i);
            }
        }
        let mut meshes = Vec::new();
        let mut skins = Vec::new();
        let mut materials = Vec::new();
        let mut material_ids = HashMap::new();
        for (index, node) in scene.nodes.iter().enumerate().filter(|(i, _)| worlds[*i].1) {
            let Some(asset_id) = node.asset else { continue };
            let asset = &scene.assets[asset_id.0];
            let binding = scene.skins.get(&NodeId(index));
            let skin = if let Some(binding) = binding {
                let matrices: Vec<_> = binding
                    .bind_from_mesh
                    .iter()
                    .map(Mat4::to_cols_array)
                    .collect();
                let accessor = glb.floats(&matrices, "MAT4", false)?;
                let skin = skins.len();
                skins.push(json!({"inverseBindMatrices":accessor,"joints":binding.palette.iter().map(|n|n.0).collect::<Vec<_>>()}));
                Some(skin)
            } else {
                None
            };
            let mut primitives = Vec::new();
            for chunk in asset
                .chunks
                .iter()
                .filter(|c| crate::scene::chunk_visible(node.chunk_mask, c.chunk_index))
            {
                if chunk.positions.is_empty() || chunk.indices.is_empty() {
                    continue;
                }
                let material_name = asset
                    .appearance
                    .chunk_materials
                    .get(
                        chunk
                            .chunk_index
                            .checked_rem(asset.appearance.chunk_materials.len())
                            .ok_or_else(|| invalid("browser material table empty"))?,
                    )
                    .ok_or_else(|| invalid("browser material absent"))?;
                let slot = asset
                    .appearance
                    .materials
                    .iter()
                    .position(|m| &m.name == material_name)
                    .ok_or_else(|| invalid("browser material binding absent"))?;
                let material = if let Some(id) = material_ids.get(&(asset_id, slot)) {
                    *id
                } else {
                    let id = materials.len();
                    materials.push(glb.material(&asset.appearance.materials[slot])?);
                    material_ids.insert((asset_id, slot), id);
                    id
                };
                let mut attributes = json!({"POSITION":glb.floats(&chunk.positions,"VEC3",true)?});
                if chunk.normals.len() == chunk.positions.len() {
                    attributes["NORMAL"] = json!(glb.floats(&chunk.normals, "VEC3", false)?);
                }
                if chunk.tex_coords.len() == chunk.positions.len() {
                    attributes["TEXCOORD_0"] =
                        json!(glb.floats(&chunk.tex_coords, "VEC2", false)?);
                }
                if chunk.tex_coords_1.len() == chunk.positions.len() {
                    attributes["TEXCOORD_1"] =
                        json!(glb.floats(&chunk.tex_coords_1, "VEC2", false)?);
                }
                if let Some(binding) = binding {
                    if chunk.skin.len() != chunk.positions.len() {
                        return Err(invalid("browser skin vertex count mismatch"));
                    }
                    for offset in [0, 4] {
                        let mut joints = Vec::new();
                        let mut weights = Vec::new();
                        for vertex in &chunk.skin {
                            for (&joint, &weight) in vertex.joints.iter().zip(&vertex.weights) {
                                if weight > 0. && usize::from(joint) >= binding.palette.len() {
                                    return Err(invalid("browser skin palette index out of range"));
                                }
                            }
                            let mut row = [0.; 4];
                            for (i, component) in row.iter_mut().enumerate() {
                                let joint = if vertex.weights[offset + i] > 0. {
                                    vertex.joints[offset + i]
                                } else {
                                    0
                                };
                                joints.extend_from_slice(&joint.to_le_bytes());
                                *component = vertex.weights[offset + i];
                            }
                            weights.push(row);
                        }
                        let view = glb.view(&joints);
                        let accessor = glb.accessors.len();
                        glb.accessors.push(json!({"bufferView":view,"componentType":5123,"count":chunk.skin.len(),"type":"VEC4"}));
                        attributes[format!("JOINTS_{}", offset / 4)] = json!(accessor);
                        attributes[format!("WEIGHTS_{}", offset / 4)] =
                            json!(glb.floats(&weights, "VEC4", false)?);
                    }
                }
                let bytes: Vec<_> = chunk.indices.iter().flat_map(|v| v.to_le_bytes()).collect();
                let view = glb.view(&bytes);
                let indices = glb.accessors.len();
                glb.accessors.push(json!({"bufferView":view,"componentType":5123,"count":chunk.indices.len(),"type":"SCALAR"}));
                primitives
                    .push(json!({"attributes":attributes,"indices":indices,"material":material}));
            }
            if !primitives.is_empty() {
                let mesh = meshes.len();
                meshes.push(json!({"name":node.name,"primitives":primitives}));
                nodes[index]["mesh"] = json!(mesh);
                nodes[index]["extras"] = json!({"assetId":node.name,"native_influences":8});
                if let Some(skin) = skin {
                    nodes[index]["skin"] = json!(skin);
                }
            }
        }
        if meshes.is_empty() {
            return Err(Error::NonVisual);
        }
        let input = glb.floats(
            &times.iter().map(|t| [*t]).collect::<Vec<_>>(),
            "SCALAR",
            true,
        )?;
        let mut translations = vec![Vec::new(); scene.nodes.len()];
        let mut rotations = vec![Vec::new(); scene.nodes.len()];
        let mut scales = vec![Vec::new(); scene.nodes.len()];
        for &time in times {
            update(&mut scene, time)?;
            if scene.nodes.len() != nodes.len() {
                return Err(invalid("animation callback changed scene topology"));
            }
            for (i, node) in scene.nodes.iter().enumerate() {
                let (t, mut r, s) = trs(node.transform)?;
                if rotations[i].last().is_some_and(|previous: &[f32; 4]| {
                    Quat::from_array(*previous).dot(Quat::from_array(r)) < 0.
                }) {
                    r = r.map(|v| -v);
                }
                translations[i].push(t);
                rotations[i].push(r);
                scales[i].push(s);
            }
        }
        let mut samplers = Vec::new();
        let mut channels = Vec::new();
        for i in 0..nodes.len() {
            for (path, output) in [
                ("translation", glb.floats(&translations[i], "VEC3", false)?),
                ("rotation", glb.floats(&rotations[i], "VEC4", false)?),
                ("scale", glb.floats(&scales[i], "VEC3", false)?),
            ] {
                let sampler = samplers.len();
                samplers.push(json!({"input":input,"output":output,"interpolation":"LINEAR"}));
                channels.push(json!({"sampler":sampler,"target":{"node":i,"path":path}}));
            }
        }
        let document = json!({"asset":{"version":"2.0","generator":"Ghostline native skeletal browser preview"},"scene":0,"scenes":[{"nodes":roots}],"nodes":nodes,"meshes":meshes,"skins":skins,"materials":materials,"animations":[{"name":name,"channels":channels,"samplers":samplers}],"accessors":glb.accessors,"bufferViews":glb.views,"buffers":[{"byteLength":glb.binary.len()}],"images":glb.images,"textures":glb.textures,"samplers":[{"magFilter":9729,"minFilter":9987,"wrapS":10497,"wrapT":10497}],"extensionsUsed":["KHR_texture_transform","KHR_materials_transmission","KHR_materials_emissive_strength"]});
        glb.finish(document)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Node, Pose, Skeleton};
    use ghostline_red::{
        assets::{Bone, Rig, Transform},
        mesh::{MeshSkin, VertexSkin},
    };
    use std::sync::Arc;

    #[test]
    fn animation_retains_native_bindings_eight_weights_and_sampled_motion() {
        let skeleton = Arc::new(
            Skeleton::from_rig(&Rig {
                source: None,
                a_pose: None,
                bones: vec![
                    Bone {
                        name: "Root".into(),
                        parent: None,
                        rest: Transform::default(),
                    },
                    Bone {
                        name: "Tip".into(),
                        parent: None,
                        rest: Transform::default(),
                    },
                ],
            })
            .unwrap(),
        );
        let mut mesh = crate::gpu::tests::triangle();
        mesh.skin = Some(MeshSkin {
            names: vec!["Root".into(), "Tip".into()],
            inverse_bind: vec![Mat4::IDENTITY; 2],
        });
        let chunk = &mut Arc::make_mut(&mut mesh.chunks)[0];
        chunk.skin = vec![
            VertexSkin {
                joints: [0, 0, 0, 0, 1, 1, 1, 1],
                weights: [0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15]
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
        let bytes = scene
            .browser_animation_glb("test", &[0., 1.], |scene, time| {
                scene.set_pose(
                    rig,
                    &Pose {
                        local: vec![
                            Mat4::IDENTITY,
                            Mat4::from_translation(glam::Vec3::X * 2. * time),
                        ],
                    },
                )
            })
            .unwrap();
        let length = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let document: serde_json::Value = serde_json::from_slice(&bytes[20..20 + length]).unwrap();
        let values = |accessor: usize| -> Vec<f32> {
            let a = &document["accessors"][accessor];
            let view = &document["bufferViews"][a["bufferView"].as_u64().unwrap() as usize];
            let start = 28 + length + view["byteOffset"].as_u64().unwrap() as usize;
            bytes[start..start + view["byteLength"].as_u64().unwrap() as usize]
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect()
        };
        let weights = document["meshes"][0]["primitives"][0]["attributes"]["WEIGHTS_1"]
            .as_u64()
            .unwrap() as usize;
        assert!(values(weights).iter().all(|w| (*w - 0.15).abs() < 1e-6));
        assert_eq!(document["skins"][0]["joints"], json!([2, 3]));
        let channel = document["animations"][0]["channels"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["target"]["node"] == 3 && c["target"]["path"] == "translation")
            .unwrap();
        let sampler = channel["sampler"].as_u64().unwrap() as usize;
        let output = document["animations"][0]["samplers"][sampler]["output"]
            .as_u64()
            .unwrap() as usize;
        assert_eq!(values(output), vec![0., 0., 0., 2., 0., 0.]);
        assert!(
            scene
                .browser_animation_glb("bad", &[0., 0.], |_, _| Ok(()))
                .is_err()
        );
        // Exporting never mutates the caller's pose.
        assert!(scene.nodes[3].transform.abs_diff_eq(Mat4::IDENTITY, 1e-6));
    }
}
