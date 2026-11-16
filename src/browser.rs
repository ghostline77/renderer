//! Self-contained browser previews of the renderer's evaluated native assemblies.
mod animation;

use crate::{Error, NodeId, Scene};
use ghostline_red::pbr::PbrMaterial;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

#[derive(Default)]
struct Glb {
    binary: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
    images: Vec<Value>,
    textures: Vec<Value>,
    paths: HashMap<PathBuf, usize>,
}
fn invalid(message: &str) -> Error {
    Error::InvalidInput(message.into())
}
impl Glb {
    fn view(&mut self, bytes: &[u8]) -> usize {
        while !self.binary.len().is_multiple_of(4) {
            self.binary.push(0);
        }
        let index = self.views.len();
        self.views
            .push(json!({"buffer":0,"byteOffset":self.binary.len(),"byteLength":bytes.len()}));
        self.binary.extend_from_slice(bytes);
        index
    }
    fn floats<const N: usize>(
        &mut self,
        data: &[[f32; N]],
        kind: &str,
        bounds: bool,
    ) -> Result<usize, Error> {
        if data.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite browser geometry"));
        }
        let bytes: Vec<u8> = data
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let view = self.view(&bytes);
        let index = self.accessors.len();
        let mut value =
            json!({"bufferView":view,"componentType":5126,"count":data.len(),"type":kind});
        if bounds {
            value["min"] = json!(
                (0..N)
                    .map(|i| data.iter().map(|p| p[i]).fold(f32::INFINITY, f32::min))
                    .collect::<Vec<_>>()
            );
            value["max"] = json!(
                (0..N)
                    .map(|i| data.iter().map(|p| p[i]).fold(f32::NEG_INFINITY, f32::max))
                    .collect::<Vec<_>>()
            );
        }
        self.accessors.push(value);
        Ok(index)
    }
    fn texture(&mut self, path: &Path, uv: [f32; 4]) -> Result<Value, Error> {
        let index = if let Some(index) = self.paths.get(path) {
            *index
        } else {
            let bytes = std::fs::read(path)?;
            let mime = if bytes.starts_with(b"\x89PNG") {
                "image/png"
            } else if bytes.starts_with(&[255, 216, 255]) {
                "image/jpeg"
            } else {
                return Err(invalid("browser texture must be PNG or JPEG"));
            };
            let view = self.view(&bytes);
            let image = self.images.len();
            self.images.push(json!({"bufferView":view,"mimeType":mime}));
            let index = self.textures.len();
            self.textures.push(json!({"source":image,"sampler":0}));
            self.paths.insert(path.to_owned(), index);
            index
        };
        Ok(
            json!({"index":index,"extensions":{"KHR_texture_transform":{"scale":[uv[0],uv[1]],"offset":[uv[2],uv[3]]}}}),
        )
    }
    fn finish(mut self, document: Value) -> Result<Vec<u8>, Error> {
        let mut json =
            serde_json::to_vec(&document).map_err(|e| Error::InvalidInput(e.to_string()))?;
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        while !self.binary.len().is_multiple_of(4) {
            self.binary.push(0);
        }
        let total = u32::try_from(28 + json.len() + self.binary.len())
            .map_err(|_| invalid("browser scene exceeds GLB size limit"))?;
        let mut result = Vec::with_capacity(total as usize);
        result.extend_from_slice(b"glTF");
        result.extend_from_slice(&2_u32.to_le_bytes());
        result.extend_from_slice(&total.to_le_bytes());
        result.extend_from_slice(&(json.len() as u32).to_le_bytes());
        result.extend_from_slice(b"JSON");
        result.extend_from_slice(&json);
        result.extend_from_slice(&(self.binary.len() as u32).to_le_bytes());
        result.extend_from_slice(b"BIN\0");
        result.extend_from_slice(&self.binary);
        Ok(result)
    }
    fn material(&mut self, m: &PbrMaterial) -> Result<Value, Error> {
        let mut pbr = json!({"baseColorFactor":m.base_color_factor,"metallicFactor":m.metallic_factor,"roughnessFactor":m.roughness_factor});
        if let Some(path) = &m.base_color_texture {
            pbr["baseColorTexture"] = self.texture(path, m.uv_transform)?;
        }
        if let Some(path) = &m.metallic_roughness_texture {
            pbr["metallicRoughnessTexture"] = self.texture(path, m.uv_transform)?;
        }
        let mut value = json!({"name":m.name,"pbrMetallicRoughness":pbr,"doubleSided":m.double_sided,"alphaMode":m.alpha_mode,
            "extras":{"ghostline_hair":m.hair.is_some(),"ghostline_decal":m.decal.is_some(),"ghostline_preview_notes":m.preview_notes}});
        if let Some(cutoff) = m.alpha_cutoff {
            value["alphaCutoff"] = json!(cutoff);
        }
        if let Some(path) = &m.normal_texture {
            value["normalTexture"] = self.texture(path, m.uv_transform)?;
        }
        if let Some(path) = &m.emissive_texture {
            value["emissiveTexture"] = self.texture(path, m.uv_transform)?;
        }
        value["emissiveFactor"] = json!(m.emissive_factor);
        value["extensions"]["KHR_materials_emissive_strength"] =
            json!({"emissiveStrength":m.emissive_strength});
        if m.transmission > 0. {
            value["extensions"]["KHR_materials_transmission"] =
                json!({"transmissionFactor":m.transmission});
        }
        Ok(value)
    }
}

pub(crate) fn encode(
    scene: &Scene,
    weights: &HashMap<NodeId, Vec<f32>>,
    overrides: &HashMap<(NodeId, usize), &PbrMaterial>,
) -> Result<Vec<u8>, Error> {
    let mut glb = Glb::default();
    let mut meshes = Vec::new();
    let mut nodes = Vec::new();
    let mut materials = Vec::new();
    let mut material_ids = HashMap::new();
    for mut evaluated in scene.evaluated_geometry()? {
        if evaluated.positions.is_empty() || evaluated.indices.is_empty() {
            continue;
        }
        if evaluated
            .transforms
            .first()
            .is_some_and(|m| m.determinant() < 0.)
        {
            for triangle in evaluated.indices.chunks_exact_mut(3) {
                triangle.swap(1, 2);
            }
        }
        let node = &scene.nodes[evaluated.node.0];
        let asset_id = node
            .asset
            .ok_or_else(|| invalid("browser node has no asset"))?;
        let asset = &scene.assets[asset_id.0];
        let chunk = asset
            .chunks
            .iter()
            .find(|c| c.chunk_index == evaluated.chunk)
            .ok_or_else(|| invalid("browser chunk absent"))?;
        let name = asset
            .appearance
            .chunk_materials
            .get(
                chunk
                    .chunk_index
                    .checked_rem(asset.appearance.chunk_materials.len())
                    .ok_or_else(|| invalid("browser chunk material table empty"))?,
            )
            .ok_or_else(|| invalid("browser chunk material absent"))?;
        let slot = asset
            .appearance
            .materials
            .iter()
            .position(|m| &m.name == name)
            .ok_or_else(|| invalid("browser material binding absent"))?;
        let material = overrides
            .get(&(evaluated.node, slot))
            .copied()
            .unwrap_or(&asset.appearance.materials[slot]);
        let material_key = (
            asset_id.0,
            slot,
            overrides
                .contains_key(&(evaluated.node, slot))
                .then_some(evaluated.node),
        );
        let material_id = if let Some(id) = material_ids.get(&material_key) {
            *id
        } else {
            let id = materials.len();
            materials.push(glb.material(material)?);
            material_ids.insert(material_key, id);
            id
        };
        for (target, &weight) in asset
            .morphs
            .iter()
            .zip(weights.get(&evaluated.node).into_iter().flatten())
            .filter(|(_, w)| **w != 0.)
        {
            if let Some(change) = target.chunks.iter().find(|c| c.chunk == chunk.chunk_index) {
                for (i, &vertex) in change.vertices.iter().enumerate() {
                    let v = usize::from(vertex);
                    let transform = *evaluated
                        .transforms
                        .get(v)
                        .ok_or_else(|| invalid("browser morph vertex out of range"))?;
                    let position = glam::Vec3::from_array(evaluated.positions[v])
                        + transform.transform_vector3(glam::Vec3::from_array(change.positions[i]))
                            * weight;
                    evaluated.positions[v] = position.to_array();
                    let normal = glam::Vec3::from_array(evaluated.normals[v])
                        + transform
                            .inverse()
                            .transpose()
                            .transform_vector3(glam::Vec3::from_array(change.normals[i]))
                            * weight;
                    evaluated.normals[v] = normal.to_array();
                }
            }
        }
        for n in &mut evaluated.normals {
            *n = glam::Vec3::from_array(*n).normalize_or_zero().to_array();
        }
        let mut attributes = json!({"POSITION":glb.floats(&evaluated.positions,"VEC3",true)?,"NORMAL":glb.floats(&evaluated.normals,"VEC3",false)?});
        if chunk.tex_coords.len() == evaluated.positions.len() {
            attributes["TEXCOORD_0"] = json!(glb.floats(&chunk.tex_coords, "VEC2", false)?);
        }
        if chunk.tex_coords_1.len() == evaluated.positions.len() {
            attributes["TEXCOORD_1"] = json!(glb.floats(&chunk.tex_coords_1, "VEC2", false)?);
        }
        if let Some(hair) = &material.hair {
            let colors: Vec<_> = (0..evaluated.positions.len())
                .map(|i| {
                    let red = chunk.colors.get(i).map_or(0., |c| c[0]);
                    let ao = (1. - red * hair.vertex_color_strength * 0.7).clamp(0., 1.);
                    [ao, ao, ao, 1.]
                })
                .collect();
            attributes["COLOR_0"] = json!(glb.floats(&colors, "VEC4", false)?);
        }
        let bytes: Vec<_> = evaluated
            .indices
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let view = glb.view(&bytes);
        let indices = glb.accessors.len();
        glb.accessors.push(json!({"bufferView":view,"componentType":5123,"count":evaluated.indices.len(),"type":"SCALAR"}));
        let mesh = meshes.len();
        meshes.push(json!({"name":evaluated.name,"primitives":[{"attributes":attributes,"indices":indices,"material":material_id}]}));
        nodes.push(json!({"name":evaluated.name,"mesh":mesh,"extras":{"assetId":node.name,"source_mesh":evaluated.mesh,"chunk":evaluated.chunk}}));
    }
    if meshes.is_empty() {
        return Err(Error::NonVisual);
    }
    let document = json!({"asset":{"version":"2.0","generator":"Ghostline native browser preview"},"scene":0,"scenes":[{"nodes":(0..nodes.len()).collect::<Vec<_>>()}],"nodes":nodes,"meshes":meshes,"materials":materials,"accessors":glb.accessors,"bufferViews":glb.views,"buffers":[{"byteLength":glb.binary.len()}],"images":glb.images,"textures":glb.textures,"samplers":[{"magFilter":9729,"minFilter":9987,"wrapS":10497,"wrapT":10497}],"extensionsUsed":["KHR_texture_transform","KHR_materials_transmission","KHR_materials_emissive_strength"]});
    glb.finish(document)
}

impl Scene {
    /// Encodes visible native geometry, current skeletal placement and baked PBR maps
    /// as a self-contained GLB for browser orbit/pan/zoom. Native shader effects are approximated.
    /// # Errors
    /// Returns geometry, binding, texture I/O or GLB size errors. Source resources are never modified.
    pub fn browser_glb(&self) -> Result<Vec<u8>, Error> {
        encode(self, &HashMap::new(), &HashMap::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ghostline_red::{
        mesh::MeshChunk,
        morph::{MorphChunk, MorphTarget},
        pbr::BakedAppearance,
    };
    use std::sync::Arc;
    #[test]
    fn browser_scene_retains_morph_placement_visibility_materials_and_embedded_maps() {
        let temporary = tempfile::tempdir().unwrap();
        let texture = temporary.path().join("color.png");
        image::RgbaImage::from_pixel(1, 1, image::Rgba([120, 40, 20, 255]))
            .save(&texture)
            .unwrap();
        let material = PbrMaterial {
            name: "skin".into(),
            base_color_texture: Some(texture),
            base_color_factor: [0.4, 0.2, 0.1, 1.],
            uv_transform: [1., 1., 0., 0.],
            metallic_roughness_texture: None,
            metallic_factor: 0.,
            roughness_factor: 0.7,
            normal_texture: None,
            emissive_texture: None,
            emissive_factor: [0.; 3],
            emissive_strength: 1.,
            alpha_mode: "OPAQUE",
            alpha_cutoff: None,
            stochastic_coverage: false,
            hair: None,
            decal: None,
            transmission: 0.,
            double_sided: false,
            preview_notes: vec![],
        };
        let asset = crate::MeshAsset {
            _preparation: None,
            _material_repository: None,
            shadow_only: false,
            chunks: Arc::new(vec![MeshChunk {
                chunk_index: 0,
                name: "triangle".into(),
                positions: vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                normals: vec![[0., 0., 1.]; 3],
                tangents: vec![],
                colors: vec![[1., 0., 0., 1.]; 3],
                tex_coords: vec![[0., 0.]; 3],
                tex_coords_1: vec![],
                indices: vec![0, 1, 2],
                material_names: vec!["skin".into()],
                skin: vec![],
            }]),
            appearance: BakedAppearance {
                chunk_materials: vec!["skin".into()],
                materials: vec![material.clone()],
                summary: Default::default(),
            },
            skin: None,
            morphs: Arc::new(vec![MorphTarget {
                name: "shape".into(),
                region: "face".into(),
                chunks: vec![MorphChunk {
                    chunk: 0,
                    vertices: vec![0],
                    positions: vec![[1., 0., 0.]],
                    normals: vec![[0.; 3]],
                    tangents: vec![[0.; 3]],
                }],
            }]),
            source_sha256: "fixture".into(),
            mesh: "test.mesh".into(),
            appearance_name: "skin".into(),
        };
        let mut scene = Scene::new();
        let asset = scene.add_asset(asset);
        scene
            .add_node(crate::Node::new("hidden", Some(asset)).with_chunk_mask(0))
            .unwrap();
        let node = scene
            .add_node(
                crate::Node::new("face", Some(asset))
                    .with_transform(glam::Mat4::from_translation(glam::Vec3::X * 2.)),
            )
            .unwrap();
        let mut changed = material;
        changed.base_color_factor = [0.1, 0.5, 0.3, 1.];
        let bytes = encode(
            &scene,
            &HashMap::from([(node, vec![1.])]),
            &HashMap::from([((node, 0), &changed)]),
        )
        .unwrap();
        drop(temporary);
        assert_eq!(&bytes[..4], b"glTF");
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            bytes.len()
        );
        let length = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let document: Value = serde_json::from_slice(&bytes[20..20 + length]).unwrap();
        assert_eq!(document["nodes"].as_array().unwrap().len(), 1);
        assert_eq!(document["nodes"][0]["name"], "face");
        assert_eq!(
            document["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"],
            json!(changed.base_color_factor)
        );
        assert!(
            document["meshes"][0]["primitives"][0]["attributes"]
                .get("COLOR_0")
                .is_none()
        );
        let position = document["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
            .as_u64()
            .unwrap() as usize;
        let view = document["accessors"][position]["bufferView"]
            .as_u64()
            .unwrap() as usize;
        let offset = 28
            + length
            + document["bufferViews"][view]["byteOffset"]
                .as_u64()
                .unwrap() as usize;
        assert_eq!(
            f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()),
            3.
        );
        assert!(document["images"][0].get("uri").is_none());
        let view = document["images"][0]["bufferView"].as_u64().unwrap() as usize;
        let offset = 28
            + length
            + document["bufferViews"][view]["byteOffset"]
                .as_u64()
                .unwrap() as usize;
        assert_eq!(&bytes[offset..offset + 4], b"\x89PNG");
    }
}
