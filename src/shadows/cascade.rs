//! Cascaded Shadow Maps (CSM) for directional lights.

use crate::math::{Matrix4, Vector3};

use super::{CascadeConfig, MAX_CASCADES};

/// Cascaded shadow map manager for directional lights.
#[derive(Debug, Clone)]
pub struct CascadedShadowMap {
    /// Split distances for each cascade (near, split1, split2, split3, far).
    splits: [f32; MAX_CASCADES + 1],
    /// Light-space matrices for each cascade.
    matrices: [[[f32; 4]; 4]; MAX_CASCADES],
    /// Configuration.
    config: CascadeConfig,
}

impl Default for CascadedShadowMap {
    fn default() -> Self {
        Self::new(CascadeConfig::default())
    }
}

impl CascadedShadowMap {
    /// Create a new cascaded shadow map.
    pub fn new(config: CascadeConfig) -> Self {
        Self {
            splits: [0.0; MAX_CASCADES + 1],
            matrices: [[[0.0; 4]; 4]; MAX_CASCADES],
            config,
        }
    }

    /// Calculate cascade splits using practical split scheme.
    ///
    /// Uses a blend between logarithmic and uniform distributions:
    /// - Logarithmic: Better distribution for near cascades
    /// - Uniform: Better distribution for far cascades
    pub fn calculate_splits(&mut self, near: f32, far: f32) {
        let n = self.config.num_cascades as usize;
        let lambda = self.config.split_lambda;
        let max_dist = self.config.max_distance.min(far);

        self.splits[0] = near;

        for i in 1..=n {
            let p = i as f32 / n as f32;
            // Logarithmic split
            let log_split = near * (max_dist / near).powf(p);
            // Uniform split
            let uniform_split = near + (max_dist - near) * p;
            // Blend between logarithmic and uniform
            self.splits[i] = lambda * log_split + (1.0 - lambda) * uniform_split;
        }
    }

    /// Get the split distances as a vec4 for shader use.
    /// Returns [split0, split1, split2, split3] where each is the far distance of that cascade.
    pub fn get_split_distances(&self) -> [f32; 4] {
        [
            self.splits[1],
            self.splits[2],
            self.splits[3],
            self.splits[4],
        ]
    }

    /// Calculate cascade matrices for a directional light.
    ///
    /// # Arguments
    /// * `light_direction` - Normalized direction the light is pointing
    /// * `camera_view` - Camera view matrix (world to camera space)
    /// * `camera_proj` - Camera projection matrix
    /// * `camera_position` - Camera world position
    /// * `fov` - Camera field of view in radians
    /// * `aspect` - Camera aspect ratio
    pub fn calculate_matrices(
        &mut self,
        light_direction: &Vector3,
        camera_position: &Vector3,
        camera_forward: &Vector3,
        camera_up: &Vector3,
        camera_right: &Vector3,
        fov: f32,
        aspect: f32,
    ) {
        let tan_half_fov = (fov / 2.0).tan();

        for i in 0..self.config.num_cascades as usize {
            let near = self.splits[i];
            let far = self.splits[i + 1];

            // Calculate frustum corners in world space
            let corners = self.calculate_frustum_corners(
                camera_position,
                camera_forward,
                camera_up,
                camera_right,
                near,
                far,
                tan_half_fov,
                aspect,
            );

            // Calculate center and radius of the frustum
            let center = corners.iter().fold(Vector3::ZERO, |acc, c| acc + *c) / 8.0;
            let radius = corners
                .iter()
                .map(|c| (*c - center).length())
                .fold(0.0f32, |a, b| a.max(b));

            // Snap to texel grid to reduce shadow swimming
            let texels_per_unit = self.config.max_distance / 2048.0; // Approximate
            let snapped_center = Vector3::new(
                (center.x / texels_per_unit).floor() * texels_per_unit,
                (center.y / texels_per_unit).floor() * texels_per_unit,
                (center.z / texels_per_unit).floor() * texels_per_unit,
            );

            // Position light far from center, looking at center
            let light_pos = snapped_center - *light_direction * radius * 2.0;

            // Create light view matrix
            let light_view = Matrix4::look_at(&light_pos, &snapped_center, &Vector3::UP);

            // Create orthographic projection that encompasses the frustum
            let light_proj =
                Matrix4::orthographic(-radius, radius, -radius, radius, 0.1, radius * 4.0);

            // Combine into light-space matrix
            let light_matrix = light_proj.multiply(&light_view);
            self.matrices[i] = light_matrix.to_cols_array_2d();
        }
    }

    /// Calculate the 8 corners of a frustum slice in world space.
    fn calculate_frustum_corners(
        &self,
        camera_position: &Vector3,
        camera_forward: &Vector3,
        camera_up: &Vector3,
        camera_right: &Vector3,
        near: f32,
        far: f32,
        tan_half_fov: f32,
        aspect: f32,
    ) -> [Vector3; 8] {
        let near_height = near * tan_half_fov;
        let near_width = near_height * aspect;
        let far_height = far * tan_half_fov;
        let far_width = far_height * aspect;

        let near_center = *camera_position + *camera_forward * near;
        let far_center = *camera_position + *camera_forward * far;

        [
            // Near plane corners
            near_center - *camera_up * near_height - *camera_right * near_width,
            near_center - *camera_up * near_height + *camera_right * near_width,
            near_center + *camera_up * near_height + *camera_right * near_width,
            near_center + *camera_up * near_height - *camera_right * near_width,
            // Far plane corners
            far_center - *camera_up * far_height - *camera_right * far_width,
            far_center - *camera_up * far_height + *camera_right * far_width,
            far_center + *camera_up * far_height + *camera_right * far_width,
            far_center + *camera_up * far_height - *camera_right * far_width,
        ]
    }

    /// Get the cascade matrices.
    #[inline]
    pub fn matrices(&self) -> &[[[f32; 4]; 4]; MAX_CASCADES] {
        &self.matrices
    }

    /// Get the matrix for a specific cascade.
    #[inline]
    pub fn matrix(&self, cascade: usize) -> Option<&[[f32; 4]; 4]> {
        self.matrices.get(cascade)
    }

    /// Get the configuration.
    #[inline]
    pub fn config(&self) -> &CascadeConfig {
        &self.config
    }

    /// Set the configuration.
    pub fn set_config(&mut self, config: CascadeConfig) {
        self.config = config;
    }

    /// Get the number of cascades.
    #[inline]
    pub fn num_cascades(&self) -> u32 {
        self.config.num_cascades
    }
}
