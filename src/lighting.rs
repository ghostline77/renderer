use serde::{Deserialize, Serialize};

use crate::Error;

/// Explicit preview illumination, independent of material preparation and cameras.
/// Positions and directions use the same Y-up space as the scene geometry.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Lighting {
    /// Uniform diffuse ambient irradiance in linear RGB.
    pub ambient: [f32; 3],
    /// Additional upward-facing diffuse ambient contribution.
    pub sky: f32,
    /// Up to 64 directional or local point lights.
    pub lights: Vec<Light>,
}

/// A directional key or bounded local point light used by native previews.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Light {
    /// Direction toward the light, and its linear RGB irradiance.
    Directional {
        /// Nonzero vector from the surface toward the source.
        direction: [f32; 3],
        /// Linear RGB irradiance.
        color: [f32; 3],
    },
    /// Local position, linear RGB radiant intensity and influence radius in metres.
    Point {
        /// Source position in scene Y-up metres.
        position: [f32; 3],
        /// Linear RGB radiant intensity before distance attenuation.
        color: [f32; 3],
        /// Positive influence radius with a smooth cutoff.
        radius: f32,
    },
}

impl Default for Lighting {
    fn default() -> Self {
        Self {
            ambient: [0.12; 3],
            sky: 0.1,
            lights: vec![
                Light::Directional {
                    direction: [-0.6, 0.8, -0.7],
                    color: [3.5, 3.3, 3.],
                },
                Light::Directional {
                    direction: [0.8, 0.3, -0.2],
                    color: [1.2, 1.4, 1.7],
                },
                Light::Directional {
                    direction: [0.2, 0.7, 0.8],
                    color: [1.8; 3],
                },
            ],
        }
    }
}

impl Lighting {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        let nonnegative = |values: &[f32]| values.iter().all(|v| v.is_finite() && *v >= 0.);
        if self.lights.len() > 64
            || !nonnegative(&self.ambient)
            || !self.sky.is_finite()
            || self.sky < 0.
        {
            return Err(Error::InvalidInput("invalid preview illumination".into()));
        }
        for light in &self.lights {
            let valid = match light {
                Light::Directional { direction, color } => {
                    direction.iter().all(|v| v.is_finite())
                        && glam::Vec3::from_array(*direction).length_squared() > 0.000001
                        && nonnegative(color)
                }
                Light::Point {
                    position,
                    color,
                    radius,
                } => {
                    position.iter().all(|v| v.is_finite())
                        && nonnegative(color)
                        && radius.is_finite()
                        && *radius > 0.
                }
            };
            if !valid {
                return Err(Error::InvalidInput("invalid preview light".into()));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lights_reject_invalid_energy_and_undefined_direction() {
        let mut lighting = Lighting {
            lights: vec![Light::Directional {
                direction: [0.; 3],
                color: [1.; 3],
            }],
            ..Default::default()
        };
        assert!(lighting.validate().is_err());
        lighting.lights = vec![Light::Point {
            position: [0.; 3],
            color: [1.; 3],
            radius: 4.,
        }];
        assert!(lighting.validate().is_ok());
        lighting.ambient[0] = f32::NAN;
        assert!(lighting.validate().is_err());
    }
}
