//! Raycaster utility for screen-to-world ray casting.

use super::{Matrix4, Ray, Vector3};

/// Utility for creating rays from screen coordinates.
pub struct Raycaster;

impl Raycaster {
    /// Create a ray from screen coordinates.
    ///
    /// # Arguments
    /// * `screen_x` - X coordinate in pixels (0 = left)
    /// * `screen_y` - Y coordinate in pixels (0 = top)
    /// * `width` - Screen width in pixels
    /// * `height` - Screen height in pixels
    /// * `view_proj_inverse` - Inverse of (projection * view) matrix
    ///
    /// # Returns
    /// A ray in world space from the camera through the screen point.
    pub fn ray_from_screen(
        screen_x: f32,
        screen_y: f32,
        width: f32,
        height: f32,
        view_proj_inverse: &Matrix4,
    ) -> Ray {
        // Convert screen coordinates to NDC (-1 to 1)
        let ndc_x = (screen_x / width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_y / height) * 2.0; // Y is flipped

        // Near point in NDC (z = -1 for WebGPU/Vulkan clip space)
        let near_ndc = Vector3::new(ndc_x, ndc_y, 0.0);
        // Far point in NDC (z = 1)
        let far_ndc = Vector3::new(ndc_x, ndc_y, 1.0);

        // Transform to world space
        let near_world = view_proj_inverse.transform_point(&near_ndc);
        let far_world = view_proj_inverse.transform_point(&far_ndc);

        // Create ray
        let direction = (far_world - near_world).normalized();
        Ray::new(near_world, direction)
    }

    /// Create a ray from normalized device coordinates directly.
    ///
    /// # Arguments
    /// * `ndc_x` - X coordinate in NDC (-1 to 1)
    /// * `ndc_y` - Y coordinate in NDC (-1 to 1)
    /// * `view_proj_inverse` - Inverse of (projection * view) matrix
    pub fn ray_from_ndc(ndc_x: f32, ndc_y: f32, view_proj_inverse: &Matrix4) -> Ray {
        let near_ndc = Vector3::new(ndc_x, ndc_y, 0.0);
        let far_ndc = Vector3::new(ndc_x, ndc_y, 1.0);

        let near_world = view_proj_inverse.transform_point(&near_ndc);
        let far_world = view_proj_inverse.transform_point(&far_ndc);

        let direction = (far_world - near_world).normalized();
        Ray::new(near_world, direction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_screen_center_ray() {
        // Identity view-proj inverse should give a ray along -Z
        let identity = Matrix4::IDENTITY;
        let ray = Raycaster::ray_from_screen(400.0, 300.0, 800.0, 600.0, &identity);

        // Center of screen should give NDC (0, 0)
        assert!(ray.origin.x.abs() < 0.01);
        assert!(ray.origin.y.abs() < 0.01);
    }
}
