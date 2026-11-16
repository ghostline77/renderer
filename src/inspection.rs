//! Source-backed assembly diagnostics before GPU upload.
use glam::Vec3;
use serde_json::{Value, json};

use crate::{Error, SceneAssembly, scene::chunk_visible};

impl SceneAssembly {
    /// Collects original material inputs for auditing a composed preview.
    ///
    /// # Errors
    /// Returns an error if retained material manifests cannot be read.
    pub fn material_provenance(&self) -> Result<Value, Error> {
        Ok(Value::Array(self.scene.assets.iter().map(|asset|Ok(json!({"resource":asset.mesh,"appearance":asset.appearance_name,"manifest":asset.material_provenance()?}))).collect::<Result<Vec<_>,Error>>()?))
    }
    /// Reports native selection, placements, visible chunks and prepared materials.
    /// Material diagnostics include texture paths and the actual render parameters.
    ///
    /// # Errors
    /// Returns [`Error`] for invalid scene transforms.
    pub fn inspection(&self, views: &[crate::RenderOptions]) -> Result<Value, Error> {
        let worlds = self.scene.world_transforms()?;
        let nodes: Vec<_> = self
            .scene
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let chunks: Vec<_> = node
                    .asset
                    .into_iter()
                    .flat_map(|id| {
                        let asset = &self.scene.assets[id.0];
                        let world = worlds[i];
                        asset.chunks.iter().map(move |chunk| {
                            let visible =
                                world.1 && chunk_visible(node.chunk_mask, chunk.chunk_index);
                            let mut lower = Vec3::splat(f32::INFINITY);
                            let mut upper = Vec3::splat(f32::NEG_INFINITY);
                            for &position in &chunk.positions {
                                let position = world.0.transform_point3(position.into());
                                lower = lower.min(position);
                                upper = upper.max(position);
                            }
                            let color_min: [f32;4]=std::array::from_fn(|i|chunk.colors.iter().map(|c|c[i]).fold(1.,f32::min));
                            let color_max: [f32;4]=std::array::from_fn(|i|chunk.colors.iter().map(|c|c[i]).fold(0.,f32::max));
                            json!({"index":chunk.chunk_index,"visible":visible,
                        "vertices":chunk.positions.len(),"native_triangles":chunk.indices.len()/3,
                        "gpu_triangles":asset.surface_indices(chunk).len()/3,
                        "world_bounds":{"min":lower.to_array(),"max":upper.to_array()},
                        "normals":chunk.normals.len(),"tangents":chunk.tangents.len(),
                        "vertex_colors":chunk.colors.len(),"vertex_color_range":{"min":color_min,"max":color_max},"uv0":chunk.tex_coords.len(),
                        "uv1":chunk.tex_coords_1.len(),"skin_vertices":chunk.skin.len()})
                        })
                    })
                    .collect();
                json!({"name":node.name,"asset":node.asset.map(|id|id.0),
                "parent":node.parent.map(|id|&self.scene.nodes[id.0].name),
                "local":node.transform.to_cols_array(),"world":worlds[i].0.to_cols_array(),
                "visible":worlds[i].1,"reference":node.reference,"chunk_mask":node.chunk_mask.to_string(),"chunks":chunks})
            })
            .collect();
        let assets: Vec<_> = self.scene.assets.iter().map(|asset| {
            let materials: Vec<_> = asset.materials().iter().map(|m| json!({
                "name":m.name,"base_color_texture":m.base_color_texture,
                "base_color_factor":m.base_color_factor,"uv_transform":m.uv_transform,"surface_texture":m.metallic_roughness_texture,
                "roughness":m.roughness_factor,"metallic":m.metallic_factor,
                "normal_texture":m.normal_texture,"emissive_texture":m.emissive_texture,
                "alpha_mode":m.alpha_mode,"alpha_cutoff":m.alpha_cutoff,
                "stochastic_coverage":m.stochastic_coverage,"double_sided":m.double_sided,
                "decal":m.decal.as_ref().map(|d|json!({"normal_alpha":d.normal_alpha,
                    "gradient_texture":d.gradient_texture,"gradient_intensity":d.gradient_intensity,"gradient_uv":d.gradient_uv,
                    "secondary_texture":d.secondary_texture,"secondary_color":d.secondary_color,"secondary_alpha":d.secondary_alpha,
                    "native_depth_threshold_evaluated":false,
                    "roughness_metalness_alpha":d.roughness_metalness_alpha,"depth_threshold":d.depth_threshold,
                    "normal_blending_mode":d.normal_blending_mode,"normal_alpha_texture":d.normal_alpha_texture,
                    "composition":"receiving surface before studio lighting"})),
                "hair":m.hair.as_ref().map(|h|json!({"flow_texture":h.flow_texture,
                    "flow_strength":h.flow_strength,"vertex_color_strength":h.vertex_color_strength})),
                "notes":m.preview_notes
            })).collect();
            json!({"resource":asset.mesh,"source_sha256":asset.source_sha256,
                "appearance":asset.appearance_name,"chunk_materials":asset.appearance.chunk_materials,
                "shadow_only":asset.is_shadow_only(),"morph_targets":asset.morphs.len(),
                "bone_palette":asset.skin.as_ref().map(|s|&s.names),"materials":materials})
        }).collect();
        let mut points = Vec::new();
        let mut subject_points = Vec::new();
        for (i, node) in self.scene.nodes.iter().enumerate() {
            if let Some(asset) = node.asset.filter(|_| worlds[i].1) {
                for chunk in self.scene.assets[asset.0]
                    .chunks
                    .iter()
                    .filter(|c| chunk_visible(node.chunk_mask, c.chunk_index))
                {
                    let lower = chunk
                        .positions
                        .iter()
                        .copied()
                        .map(Vec3::from)
                        .fold(Vec3::splat(f32::INFINITY), Vec3::min);
                    let upper = chunk
                        .positions
                        .iter()
                        .copied()
                        .map(Vec3::from)
                        .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
                    for corner in 0..8 {
                        let point = worlds[i].0.transform_point3(Vec3::new(
                            if corner & 1 == 0 { lower.x } else { upper.x },
                            if corner & 2 == 0 { lower.y } else { upper.y },
                            if corner & 4 == 0 { lower.z } else { upper.z },
                        ));
                        points.push(point);
                        if !node.reference {
                            subject_points.push(point);
                        }
                    }
                }
            }
        }
        let (center, radius) = crate::scene::transformed_bounds(points.iter().copied())?;
        let subject_bounds = if subject_points.is_empty() {
            (center, radius)
        } else {
            crate::scene::transformed_bounds(subject_points.into_iter())?
        };
        let cameras=views.iter().map(|view| {
            let (center,radius)=if matches!(view.camera,crate::Camera::FrameSubject(_) | crate::Camera::FrameSubjectRegion { .. } | crate::Camera::FrameSubjectCrop { .. }){subject_bounds}else{(center,radius)};
            let part = match view.camera { crate::Camera::FrameSubjectPart { part, .. } | crate::Camera::FrameSubjectCrop { part: Some(part), .. } => Some(part), _ => None };
            let (center, radius) = if let Some(part) = part {
                let node = nodes.iter().filter(|n| n["visible"] == true && n["reference"] == false && !n["asset"].is_null()
                    && n["chunks"].as_array().is_some_and(|chunks| chunks.iter().any(|c| c["visible"] == true && c["vertices"].as_u64().is_some_and(|count| count > 0))))
                    .nth(part).ok_or_else(|| Error::InvalidInput("camera subject part is missing".into()))?;
                let mut points = Vec::new();
                for chunk in node["chunks"].as_array().into_iter().flatten().filter(|c| c["visible"] == true) {
                    for key in ["min", "max"] {
                        let point: [f32; 3] = serde_json::from_value(chunk["world_bounds"][key].clone())
                            .map_err(|e| Error::InvalidInput(e.to_string()))?;
                        points.push(Vec3::from(point));
                    }
                }
                view.camera.subject_bounds(&mut points)?
            } else { (center, radius) };
            let radius = if let crate::Camera::FrameSubjectRegion { min_radius, .. } = view.camera { radius.max(min_radius) } else { radius };
            let (matrix,eye)=view.camera.matrices(center,radius)?;
            let matrix=view.camera.fit_projection(matrix,&points)?;
            Ok(json!({"request":view,"eye":eye.to_array(),"projection_view":matrix.to_cols_array(),
                "frame_center":center.to_array(),"frame_radius":radius}))
        }).collect::<Result<Vec<_>,Error>>()?;
        let native: Vec<_> = self
            .native
            .components
            .iter()
            .map(|c| {
                json!({
            "name":c.name,"class":c.kind,"mesh":c.mesh,"appearance":c.mesh_appearance,"requested_appearance":c.mesh_appearance_requested,
                    "enabled":c.enabled,"chunk_mask":c.chunk_mask.to_string(),
                    "local":c.local.to_cols_array(),"rig":c.rig,
                    "parent":{"component":c.parent.component,"slot":c.parent.slot},
                    "control":{"component":c.control.component,"slot":c.control.slot}
                })
            })
            .collect();
        Ok(
            json!({"renderer_build":crate::BUILD_ID,"source":self.native.source,
            "appearance":self.native.appearance,"native_components":native,"preview_notes":self.preview_notes,
            "native_rigs":self.native.rigs.iter().map(|(key,rig)|json!({
                "resource":key,"bones":rig.bones.iter().map(|b|json!({
                    "name":b.name,"parent":b.parent,"rest":b.rest.matrix().map(|m|m.to_cols_array()).ok()
                })).collect::<Vec<_>>()
            })).collect::<Vec<_>>(),"evaluated_rigs":self.rigs.len(),
            "nodes":nodes,"assets":assets,"cameras":cameras,
            "lighting":{"shader":"studio.wgsl","shadows":false,"environment":false},
            "coordinate_system":"metres, Y up, character front -Z"}),
        )
    }
}
