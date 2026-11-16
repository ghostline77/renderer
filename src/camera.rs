use glam::{Mat4, Vec3};

use crate::{Error, View};

/// Framed studio view or an explicit camera retained across scene edits and frames.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Camera {
    /// Fit the visible scene bounds using a studio direction.
    Frame(View),
    /// Fit each view's projected visible bounds with a ten-percent border.
    FrameProjected(View),
    /// Frame the selected subject while drawing reference geometry for context.
    FrameSubject(View),
    /// Fit one native X half of a paired subject without moving its geometry.
    /// A subject present only on the opposite side is framed intact.
    FrameSubjectSide {
        /// Studio viewing direction.
        view: View,
        /// Select vertices on the positive native/model X side when true.
        positive: bool,
    },
    /// Fit an anatomical height range of a subject, retaining the whole scene.
    FrameSubjectCrop {
        /// Studio viewing direction.
        view: View,
        /// Optional native mesh instance to isolate for framing.
        part: Option<usize>,
        /// Lower and upper fractions of the subject's vertical extent.
        vertical_range: [f32; 2],
    },
    /// Frame a small mounted subject with enough surrounding anatomical context.
    FrameSubjectRegion {
        /// Studio viewing direction; hero/side views follow the subject's side.
        view: View,
        /// Minimum framing radius in metres.
        min_radius: f32,
    },
    /// Fit one visible subject mesh instance while retaining surrounding context.
    FrameSubjectPart {
        /// Studio viewing direction.
        view: View,
        /// Zero-based visible subject mesh instance in native assembly order.
        part: usize,
    },
    /// Fixed orthographic camera, suitable for comparisons and animation sequences.
    Orthographic {
        /// Camera position in Y-up metres.
        eye: [f32; 3],
        /// World-space look-at point.
        target: [f32; 3],
        /// Camera up direction.
        up: [f32; 3],
        /// Half the vertical coverage in metres.
        half_height: f32,
        /// Positive near clip distance.
        near: f32,
        /// Far clip distance, greater than `near`.
        far: f32,
    },
    /// Fixed perspective camera.
    Perspective {
        /// Camera position in Y-up metres.
        eye: [f32; 3],
        /// World-space look-at point.
        target: [f32; 3],
        /// Camera up direction.
        up: [f32; 3],
        /// Vertical field of view in radians, strictly between zero and pi.
        vertical_fov: f32,
        /// Positive near clip distance.
        near: f32,
        /// Far clip distance, greater than `near`.
        far: f32,
    },
}

impl Default for Camera {
    fn default() -> Self {
        Self::Frame(View::Hero)
    }
}

impl Camera {
    pub(crate) fn subject_bounds(self, points: &mut [Vec3]) -> Result<(Vec3, f32), Error> {
        if let Self::FrameSubjectCrop {
            vertical_range: [low, high],
            ..
        } = self
        {
            if !low.is_finite() || !high.is_finite() || low < 0. || high > 1. || low >= high {
                return Err(Error::InvalidInput("invalid subject height range".into()));
            }
            let minimum = points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
            let maximum = points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
            let height = maximum - minimum;
            for point in points.iter_mut() {
                point.y = point
                    .y
                    .clamp(minimum + low * height, minimum + high * height);
            }
        }
        crate::scene::transformed_bounds(points.iter().copied())
    }

    pub(crate) fn fit_projection(self, projection: Mat4, points: &[Vec3]) -> Result<Mat4, Error> {
        if !matches!(
            self,
            Self::FrameProjected(_) | Self::FrameSubjectSide { .. }
        ) {
            return Ok(projection);
        }
        let mut low = glam::Vec2::splat(f32::INFINITY);
        let mut high = glam::Vec2::splat(f32::NEG_INFINITY);
        for point in points {
            let projected = projection.project_point3(*point).truncate();
            low = low.min(projected);
            high = high.max(projected);
        }
        let extent = (high - low).max_element();
        if !extent.is_finite() || extent <= f32::EPSILON {
            return Err(Error::InvalidInput(
                "degenerate projected view bounds".into(),
            ));
        }
        let center = (low + high) * 0.5;
        let scale = 1.8 / extent;
        Ok(Mat4::from_scale(Vec3::new(scale, scale, 1.))
            * Mat4::from_translation(Vec3::new(-center.x, -center.y, 0.))
            * projection)
    }
    pub(crate) fn matrices(self, center: Vec3, radius: f32) -> Result<(Mat4, Vec3), Error> {
        let (eye, target, up, projection) = match self {
            Self::Frame(view)
            | Self::FrameProjected(view)
            | Self::FrameSubject(view)
            | Self::FrameSubjectSide { view, .. }
            | Self::FrameSubjectCrop { view, .. }
            | Self::FrameSubjectPart { view, .. }
            | Self::FrameSubjectRegion { view, .. } => {
                let radius = if let Self::FrameSubjectRegion { min_radius, .. } = self {
                    if !min_radius.is_finite() || min_radius <= 0. {
                        return Err(Error::InvalidInput("invalid subject context radius".into()));
                    }
                    radius.max(min_radius)
                } else {
                    radius
                };
                let direction = match view {
                    View::Hero
                        if matches!(self, Self::FrameSubjectRegion { .. }) && center.x > 0. =>
                    {
                        Vec3::new(0.65, 0.25, -1.)
                    }
                    View::Hero => Vec3::new(-0.65, 0.25, -1.),
                    View::Front => Vec3::new(0., 0.08, -1.),
                    View::Back => Vec3::new(0., 0.08, 1.),
                    View::Left => Vec3::new(-1., 0.08, 0.),
                    View::Right => Vec3::new(1., 0.08, 0.),
                    View::Side => Vec3::new(if center.x < 0. { -1. } else { 1. }, 0.08, 0.),
                }
                .normalize();
                let eye = center + direction * radius * 4.;
                let half = radius * 1.08;
                (
                    eye,
                    center,
                    Vec3::Y,
                    Mat4::orthographic_rh(-half, half, -half, half, radius * 0.01, radius * 10.),
                )
            }
            Self::Orthographic {
                eye,
                target,
                up,
                half_height,
                near,
                far,
            } => {
                planes(near, far)?;
                if !half_height.is_finite() || half_height <= 0. {
                    return Err(Error::InvalidInput("invalid orthographic coverage".into()));
                }
                (
                    eye.into(),
                    target.into(),
                    up.into(),
                    Mat4::orthographic_rh(
                        -half_height,
                        half_height,
                        -half_height,
                        half_height,
                        near,
                        far,
                    ),
                )
            }
            Self::Perspective {
                eye,
                target,
                up,
                vertical_fov,
                near,
                far,
            } => {
                planes(near, far)?;
                if !vertical_fov.is_finite()
                    || !(0. ..std::f32::consts::PI).contains(&vertical_fov)
                    || vertical_fov == 0.
                {
                    return Err(Error::InvalidInput(
                        "invalid perspective field of view".into(),
                    ));
                }
                (
                    eye.into(),
                    target.into(),
                    up.into(),
                    Mat4::perspective_rh(vertical_fov, 1., near, far),
                )
            }
        };
        if !eye.is_finite()
            || !target.is_finite()
            || !up.is_finite()
            || (target - eye).cross(up).length_squared() < 1e-12
        {
            return Err(Error::InvalidInput(
                "invalid camera position or up direction".into(),
            ));
        }
        let matrix = projection * Mat4::look_at_rh(eye, target, up.normalize());
        if !matrix.is_finite() {
            return Err(Error::InvalidInput("nonfinite camera projection".into()));
        }
        Ok((matrix, eye))
    }
}

fn planes(near: f32, far: f32) -> Result<(), Error> {
    if !near.is_finite() || !far.is_finite() || near <= 0. || far <= near {
        return Err(Error::InvalidInput("invalid camera clipping planes".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projected_views_fill_the_frame_without_clipping_long_subjects() {
        for view in [View::Front, View::Left, View::Hero] {
            let mut points = Vec::new();
            for x in [-1., 1.] {
                for y in [-0.5, 0.5] {
                    for z in [-5., 5.] {
                        points.push(Vec3::new(x, y, z));
                    }
                }
            }
            let camera = Camera::FrameProjected(view);
            let (center, radius) = camera.subject_bounds(&mut points).unwrap();
            let (base, _) = camera.matrices(center, radius).unwrap();
            let fitted = camera.fit_projection(base, &points).unwrap();
            let projected: Vec<_> = points.iter().map(|p| fitted.project_point3(*p)).collect();
            assert!(
                projected
                    .iter()
                    .all(|p| p.x.abs() <= 0.901 && p.y.abs() <= 0.901 && p.z >= 0. && p.z <= 1.)
            );
            assert!(
                projected
                    .iter()
                    .any(|p| p.x.abs() > 0.899 || p.y.abs() > 0.899)
            );
        }
    }
}
