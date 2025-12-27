//! Orbit controls for rotating camera around a target.

use crate::camera::PerspectiveCamera;
use crate::math::Vector3;

/// Orbit controls allowing camera rotation around a target point.
pub struct OrbitControls {
    /// Target point to orbit around.
    pub target: Vector3,
    /// Minimum distance from target.
    pub min_distance: f32,
    /// Maximum distance from target.
    pub max_distance: f32,
    /// Minimum polar angle (radians, 0 = top).
    pub min_polar_angle: f32,
    /// Maximum polar angle (radians, PI = bottom).
    pub max_polar_angle: f32,
    /// Enable rotation.
    pub enable_rotate: bool,
    /// Enable panning.
    pub enable_pan: bool,
    /// Enable zooming.
    pub enable_zoom: bool,
    /// Rotation speed multiplier.
    pub rotate_speed: f32,
    /// Pan speed multiplier.
    pub pan_speed: f32,
    /// Zoom speed multiplier.
    pub zoom_speed: f32,
    /// Enable damping (smooth movement).
    pub enable_damping: bool,
    /// Damping factor (0-1, lower = more damping).
    pub damping_factor: f32,
    // Internal state
    spherical_delta: SphericalDelta,
    pan_offset: Vector3,
    scale: f32,
    // For damping
    spherical_target: SphericalCoords,
}

/// Spherical coordinates (radius, theta, phi).
#[derive(Debug, Clone, Copy, Default)]
struct SphericalCoords {
    radius: f32,
    /// Polar angle (up/down, 0 = top, PI = bottom).
    phi: f32,
    /// Azimuthal angle (left/right).
    theta: f32,
}

/// Delta for spherical movement.
#[derive(Debug, Clone, Copy, Default)]
struct SphericalDelta {
    theta: f32,
    phi: f32,
}

impl Default for OrbitControls {
    fn default() -> Self {
        Self {
            target: Vector3::ZERO,
            min_distance: 0.1,
            max_distance: 1000.0,
            min_polar_angle: 0.0,
            max_polar_angle: std::f32::consts::PI,
            enable_rotate: true,
            enable_pan: true,
            enable_zoom: true,
            rotate_speed: 1.0,
            pan_speed: 1.0,
            zoom_speed: 1.0,
            enable_damping: true,
            damping_factor: 0.05,
            spherical_delta: SphericalDelta::default(),
            pan_offset: Vector3::ZERO,
            scale: 1.0,
            spherical_target: SphericalCoords::default(),
        }
    }
}

impl OrbitControls {
    /// Create new orbit controls.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create orbit controls with a target.
    pub fn with_target(target: Vector3) -> Self {
        Self {
            target,
            ..Self::default()
        }
    }

    /// Rotate the camera by delta angles (in radians).
    pub fn rotate(&mut self, delta_theta: f32, delta_phi: f32) {
        if self.enable_rotate {
            // Inverted for natural "grab and drag" feel
            self.spherical_delta.theta -= delta_theta * self.rotate_speed;
            self.spherical_delta.phi += delta_phi * self.rotate_speed;
        }
    }

    /// Rotate based on mouse movement (pixels).
    pub fn rotate_by_pixels(&mut self, delta_x: f32, delta_y: f32, _screen_height: f32) {
        // Very small fixed scale - about 0.2 degrees per pixel
        let rotate_scale = 0.004;
        self.rotate(delta_x * rotate_scale, delta_y * rotate_scale);
    }

    /// Pan the camera.
    pub fn pan(&mut self, delta_x: f32, delta_y: f32, camera: &PerspectiveCamera) {
        if !self.enable_pan {
            return;
        }

        let position = camera.position;
        let offset = position - self.target;
        let distance = offset.length();

        // Half of the fov is center to top of screen
        let fov_rad = camera.fov.to_radians();
        let target_distance = distance * (fov_rad / 2.0).tan();

        // Pan relative to distance from target
        let pan_x = delta_x * target_distance * self.pan_speed * 0.002;
        let pan_y = delta_y * target_distance * self.pan_speed * 0.002;

        // Get camera's right and up vectors
        let forward = (self.target - position).normalized();
        let right = forward.cross(&Vector3::UP).normalized();
        let up = right.cross(&forward);

        self.pan_offset = self.pan_offset + right * (-pan_x) + up * pan_y;
    }

    /// Zoom in/out.
    pub fn zoom(&mut self, delta: f32) {
        if self.enable_zoom {
            if delta > 0.0 {
                self.scale /= 1.0 + delta * self.zoom_speed * 0.1;
            } else {
                self.scale *= 1.0 - delta * self.zoom_speed * 0.1;
            }
        }
    }

    /// Zoom by mouse wheel delta.
    pub fn zoom_by_wheel(&mut self, delta: f32) {
        // Invert: scroll up (negative delta) = zoom in
        self.zoom(-delta * 0.01);
    }

    /// Update the camera based on accumulated input.
    pub fn update(&mut self, camera: &mut PerspectiveCamera) {
        let position = camera.position;
        let offset = position - self.target;

        // Convert to spherical coordinates
        let radius = offset.length();
        let mut theta = offset.x.atan2(offset.z);
        let mut phi = (offset.y / radius).asin();

        // Apply rotation deltas and immediately clear (no momentum)
        theta += self.spherical_delta.theta;
        phi += self.spherical_delta.phi;
        self.spherical_delta.theta = 0.0;
        self.spherical_delta.phi = 0.0;

        // Clamp phi to avoid flipping over poles
        phi = phi.clamp(-std::f32::consts::FRAC_PI_2 + 0.1, std::f32::consts::FRAC_PI_2 - 0.1);

        // Apply scale (zoom) and immediately clear
        let new_radius = (radius * self.scale).clamp(self.min_distance, self.max_distance);
        self.scale = 1.0;

        // Apply pan and immediately clear
        self.target = self.target + self.pan_offset;
        self.pan_offset = Vector3::ZERO;

        // Convert back to Cartesian
        let new_offset = Vector3::new(
            new_radius * phi.cos() * theta.sin(),
            new_radius * phi.sin(),
            new_radius * phi.cos() * theta.cos(),
        );

        let new_position = self.target + new_offset;
        camera.set_position(new_position);
        camera.set_target(self.target);
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.spherical_delta = SphericalDelta::default();
        self.pan_offset = Vector3::ZERO;
        self.scale = 1.0;
    }
}
