use std::sync::Arc;

use ghostline_red::{
    animation::AnimationClip,
    assets::{ResourceKey, Rig, Transform},
};
use glam::Mat4;

use crate::{Error, NodeId, scene::validate_transform};

/// Scene-local skeleton instance identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RigId(pub(crate) usize);

/// Named skeleton with local rest transforms in the renderer's Y-up coordinates.
#[derive(Clone, Debug)]
pub struct Skeleton {
    pub(crate) source: Option<ResourceKey>,
    native_rest: Vec<Transform>,
    pub(crate) names: Vec<String>,
    pub(crate) parents: Vec<Option<usize>>,
    pub(crate) rest: Pose,
}

/// Local bone transforms in skeleton index order.
#[derive(Clone, Debug)]
pub struct Pose {
    /// Local-to-parent matrices, in metres with Y up.
    pub local: Vec<Mat4>,
}

/// Root-motion treatment when sampling animation.
#[derive(Clone, Copy, Debug, Default)]
pub enum RootMotion {
    /// Keep bone channels and return the external extraction transform separately.
    #[default]
    Preserve,
    /// Apply authored external extraction to all skeleton root bones.
    ApplyExtraction,
    /// Remove translation/rotation from a selected root bone and return its motion.
    InPlace {
        /// Root bone index whose motion is extracted relative to rest.
        bone: usize,
    },
}

/// Sampled pose, scalar tracks and root motion at one timestamp.
#[derive(Clone, Debug)]
pub struct PoseSample {
    /// Rendering-space local bone transforms.
    pub pose: Pose,
    /// Rendering-space root-motion transform.
    pub root_motion: Mat4,
    /// Scalar native animation channels.
    pub tracks: Vec<f32>,
}

impl Skeleton {
    /// Converts a decoded native rig into rendering coordinates.
    ///
    /// # Errors
    /// Returns [`Error`] for a malformed hierarchy or transform.
    pub fn from_rig(rig: &Rig) -> Result<Self, Error> {
        rig.model_rest()?;
        let basis = basis();
        let native_rest: Vec<_> = rig.bones.iter().map(|b| b.rest).collect();
        let bind_transforms = rig.a_pose.as_deref().unwrap_or(&native_rest);
        if bind_transforms.len() != native_rest.len() {
            return Err(Error::InvalidInput(
                "rig bind pose differs from bone count".into(),
            ));
        }
        let rest = Pose {
            local: bind_transforms
                .iter()
                .map(|transform| Ok(basis * transform.matrix()? * basis.transpose()))
                .collect::<Result<_, Error>>()?,
        };
        Ok(Self {
            source: rig.source.clone(),
            native_rest,
            names: rig.bones.iter().map(|b| b.name.clone()).collect(),
            parents: rig.bones.iter().map(|b| b.parent).collect(),
            rest,
        })
    }

    /// Samples a native clip on its source skeleton with explicit additive/root handling.
    ///
    /// # Errors
    /// Returns [`Error`] for a different native rig, invalid base pose or sample data.
    pub fn sample_animation(
        &self,
        clip: &AnimationClip,
        time: f32,
        additive_base: Option<&Pose>,
        root_motion: RootMotion,
    ) -> Result<PoseSample, Error> {
        if self
            .source
            .as_ref()
            .zip(clip.rig.as_ref())
            .is_some_and(|(left, right)| left.native_hash() != right.native_hash())
        {
            return Err(Error::InvalidInput(
                "animation and skeleton have different native rig identities".into(),
            ));
        }
        let basis = basis();
        let base = additive_base
            .map(|pose| -> Result<Vec<Transform>, Error> {
                self.model_matrices(pose)?;
                pose.local
                    .iter()
                    .map(|matrix| {
                        let (scale, rotation, translation) =
                            (basis.transpose() * *matrix * basis).to_scale_rotation_translation();
                        let transform = Transform {
                            translation: translation.to_array(),
                            rotation: rotation.to_array(),
                            scale: scale.to_array(),
                        };
                        transform.matrix()?;
                        Ok(transform)
                    })
                    .collect()
            })
            .transpose()?;
        let sample = clip.sample(time, &self.native_rest, base.as_deref())?;
        let mut pose = Pose {
            local: sample
                .local
                .iter()
                .map(|t| Ok(basis * t.matrix()? * basis.transpose()))
                .collect::<Result<_, Error>>()?,
        };
        let mut motion = basis * sample.motion.matrix()? * basis.transpose();
        match root_motion {
            RootMotion::Preserve => {}
            RootMotion::ApplyExtraction => {
                for (index, parent) in self.parents.iter().enumerate() {
                    if parent.is_none() {
                        pose.local[index] = motion * pose.local[index];
                    }
                }
            }
            RootMotion::InPlace { bone } => {
                if self.parents.get(bone).is_none_or(Option::is_some) {
                    return Err(Error::InvalidInput(
                        "in-place motion requires a skeleton root bone".into(),
                    ));
                }
                let (scale, animated_rotation, animated_translation) =
                    pose.local[bone].to_scale_rotation_translation();
                let reference = basis * self.native_rest[bone].matrix()? * basis.transpose();
                let (_, rotation, translation) = reference.to_scale_rotation_translation();
                motion *= Mat4::from_rotation_translation(animated_rotation, animated_translation)
                    * Mat4::from_rotation_translation(rotation, translation).inverse();
                pose.local[bone] =
                    Mat4::from_scale_rotation_translation(scale, rotation, translation);
            }
        }
        Ok(PoseSample {
            pose,
            root_motion: motion,
            tracks: sample.tracks,
        })
    }

    /// Native names in bone index order.
    pub fn bone_names(&self) -> &[String] {
        &self.names
    }

    /// Local authored rest pose.
    pub fn rest_pose(&self) -> &Pose {
        &self.rest
    }

    /// Computes bone model matrices for a complete pose.
    ///
    /// # Errors
    /// Returns [`Error`] for mismatched or invalid pose transforms.
    pub fn model_matrices(&self, pose: &Pose) -> Result<Vec<Mat4>, Error> {
        if pose.local.len() != self.names.len() {
            return Err(Error::InvalidInput(
                "pose bone count differs from skeleton".into(),
            ));
        }
        let mut nodes = Vec::with_capacity(self.names.len());
        for (index, local) in pose.local.iter().enumerate() {
            validate_transform(*local)?;
            nodes.push(crate::Node::new(&self.names[index], None).with_transform(*local));
            nodes[index].parent = self.parents[index].map(NodeId);
        }
        Ok(crate::scene::resolve_transforms(&nodes)?
            .into_iter()
            .map(|v| v.0)
            .collect())
    }
}

pub(crate) fn basis() -> Mat4 {
    Mat4::from_cols_array(&[
        1., 0., 0., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
    ])
}

#[derive(Clone, Debug)]
pub(crate) struct RigInstance {
    pub skeleton: Arc<Skeleton>,
    pub root: NodeId,
    pub bones: Vec<NodeId>,
}

#[derive(Clone, Debug)]
pub(crate) struct SkinBinding {
    pub palette: Vec<NodeId>,
    pub bind_from_mesh: Vec<Mat4>,
}

#[derive(Clone, Debug)]
pub(crate) struct RigLink {
    pub source: RigId,
    pub destination: RigId,
    pub mapping: Vec<Option<usize>>,
}

pub(crate) fn evaluation_order(parents: &[Option<usize>]) -> Vec<usize> {
    let mut seen = vec![false; parents.len()];
    let mut order = Vec::with_capacity(parents.len());
    for index in 0..parents.len() {
        let mut path = Vec::new();
        let mut current = Some(index);
        while let Some(bone) = current {
            if seen[bone] {
                break;
            }
            path.push(bone);
            current = parents[bone];
        }
        for bone in path.into_iter().rev() {
            seen[bone] = true;
            order.push(bone);
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use ghostline_red::{
        animation::{JointChannels, Key},
        assets::Bone,
    };
    use glam::Vec3;

    #[test]
    fn bind_a_pose_does_not_replace_animation_reference_defaults() {
        let reference = Transform {
            translation: [1., 0., 0.],
            ..Transform::default()
        };
        let binding = Transform {
            translation: [2., 0., 0.],
            ..Transform::default()
        };
        let skeleton = Skeleton::from_rig(&Rig {
            source: None,
            bones: vec![Bone {
                name: "Root".into(),
                parent: None,
                rest: reference,
            }],
            a_pose: Some(vec![binding]),
        })
        .unwrap();
        assert_eq!(skeleton.rest_pose().local[0].w_axis.x, 2.);
        let clip = AnimationClip {
            buffer_type: "fixture".into(),
            name: "unkeyed".into(),
            kind: "Normal".into(),
            rig: None,
            duration: 1.,
            frame_count: 1,
            extra_joints: 0,
            joints: vec![JointChannels::default()],
            tracks: vec![],
            motion: None,
        };
        let sampled = skeleton
            .sample_animation(&clip, 0., None, RootMotion::Preserve)
            .unwrap();
        assert_eq!(sampled.pose.local[0].w_axis.x, 1.);
        let in_place = skeleton
            .sample_animation(&clip, 0., None, RootMotion::InPlace { bone: 0 })
            .unwrap();
        assert_eq!(in_place.pose.local[0].w_axis.x, 1.);
        assert_eq!(in_place.root_motion.w_axis.x, 0.);
    }

    #[test]
    fn root_motion_extraction_preserves_animated_scale_without_applying_it_twice() {
        let skeleton = Skeleton::from_rig(&Rig {
            a_pose: None,
            source: None,
            bones: vec![Bone {
                name: "root".into(),
                parent: None,
                rest: Transform::default(),
            }],
        })
        .unwrap();
        let clip = AnimationClip {
            buffer_type: "fixture".into(),
            name: "move".into(),
            kind: "Normal".into(),
            rig: None,
            duration: 1.,
            frame_count: 2,
            extra_joints: 0,
            joints: vec![JointChannels {
                translation: vec![
                    Key {
                        time: 0.,
                        value: Vec3::ZERO,
                    },
                    Key {
                        time: 1.,
                        value: Vec3::X,
                    },
                ],
                scale: vec![Key {
                    time: 0.,
                    value: Vec3::splat(2.),
                }],
                ..JointChannels::default()
            }],
            tracks: vec![],
            motion: None,
        };
        let kept = skeleton
            .sample_animation(&clip, 1., None, RootMotion::Preserve)
            .unwrap();
        let extracted = skeleton
            .sample_animation(&clip, 1., None, RootMotion::InPlace { bone: 0 })
            .unwrap();
        let (scale, _, translation) = extracted.root_motion.to_scale_rotation_translation();
        assert!(scale.abs_diff_eq(Vec3::ONE, 1e-6));
        assert!(translation.abs_diff_eq(Vec3::X, 1e-6));
        assert!(
            (extracted.root_motion * extracted.pose.local[0]).abs_diff_eq(kept.pose.local[0], 1e-6)
        );
    }
}
