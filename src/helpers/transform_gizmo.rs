//! Transform gizmo for interactive object manipulation.

use crate::math::{Plane, Quaternion, Ray, Vector3};
use crate::objects::{Line, LineVertex};

/// Gizmo operation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GizmoMode {
    /// Move objects along axes.
    #[default]
    Translate,
    /// Rotate objects around axes.
    Rotate,
    /// Scale objects along axes.
    Scale,
}

/// Gizmo axis selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GizmoAxis {
    /// No axis selected.
    #[default]
    None,
    /// X axis (red).
    X,
    /// Y axis (green).
    Y,
    /// Z axis (blue).
    Z,
    /// XY plane.
    XY,
    /// XZ plane.
    XZ,
    /// YZ plane.
    YZ,
    /// Uniform scale from center.
    Center,
}

impl GizmoAxis {
    /// Get the axis color (unhighlighted).
    pub fn color(&self) -> [f32; 4] {
        match self {
            GizmoAxis::X => [1.0, 0.2, 0.2, 1.0],
            GizmoAxis::Y => [0.2, 1.0, 0.2, 1.0],
            GizmoAxis::Z => [0.2, 0.5, 1.0, 1.0],
            GizmoAxis::XY => [1.0, 1.0, 0.2, 0.6],
            GizmoAxis::XZ => [1.0, 0.2, 1.0, 0.6],
            GizmoAxis::YZ => [0.2, 1.0, 1.0, 0.6],
            GizmoAxis::Center => [1.0, 1.0, 1.0, 1.0],
            GizmoAxis::None => [0.5, 0.5, 0.5, 1.0],
        }
    }

    /// Get highlight color.
    pub fn highlight_color() -> [f32; 4] {
        [1.0, 1.0, 0.0, 1.0]
    }
}

/// Gizmo configuration.
#[derive(Debug, Clone)]
pub struct GizmoConfig {
    /// Base size of the gizmo.
    pub size: f32,
    /// Pick radius for hit testing (in world units at unit distance).
    pub pick_radius: f32,
    /// Size of arrowheads relative to size.
    pub arrowhead_size: f32,
    /// Size of plane handles relative to size.
    pub plane_handle_size: f32,
    /// Number of segments for rotation circles.
    pub circle_segments: u32,
}

impl Default for GizmoConfig {
    fn default() -> Self {
        Self {
            size: 1.0,
            pick_radius: 0.08,
            arrowhead_size: 0.15,
            plane_handle_size: 0.25,
            circle_segments: 64,
        }
    }
}

/// Result of a drag operation.
#[derive(Debug, Clone, Copy)]
pub enum GizmoDragResult {
    /// No change.
    None,
    /// Translation delta.
    Translate(Vector3),
    /// Rotation delta as quaternion.
    Rotate(Quaternion),
    /// Scale delta (multiply with current scale).
    Scale(Vector3),
}

/// Internal drag state.
#[derive(Debug, Clone)]
struct DragState {
    axis: GizmoAxis,
    start_point: Vector3,
    start_position: Vector3,
    start_rotation: Quaternion,
    start_scale: Vector3,
    plane: Plane,
    accumulated_angle: f32,
}

/// Transform gizmo for translate, rotate, and scale operations.
pub struct TransformGizmo {
    line: Line,
    config: GizmoConfig,
    mode: GizmoMode,
    position: Vector3,
    rotation: Quaternion,
    scale: Vector3,
    hovered_axis: GizmoAxis,
    drag_state: Option<DragState>,
    /// Scale factor based on camera distance.
    screen_scale: f32,
    /// Cache for rebuilding geometry.
    needs_rebuild: bool,
}

impl TransformGizmo {
    /// Create a new transform gizmo.
    pub fn new() -> Self {
        let mut gizmo = Self {
            line: Line::new(),
            config: GizmoConfig::default(),
            mode: GizmoMode::Translate,
            position: Vector3::ZERO,
            rotation: Quaternion::IDENTITY,
            scale: Vector3::ONE,
            hovered_axis: GizmoAxis::None,
            drag_state: None,
            screen_scale: 1.0,
            needs_rebuild: true,
        };
        gizmo.rebuild_geometry();
        gizmo
    }

    /// Create with custom configuration.
    pub fn with_config(config: GizmoConfig) -> Self {
        let mut gizmo = Self {
            line: Line::new(),
            config,
            mode: GizmoMode::Translate,
            position: Vector3::ZERO,
            rotation: Quaternion::IDENTITY,
            scale: Vector3::ONE,
            hovered_axis: GizmoAxis::None,
            drag_state: None,
            screen_scale: 1.0,
            needs_rebuild: true,
        };
        gizmo.rebuild_geometry();
        gizmo
    }

    /// Set the gizmo mode.
    pub fn set_mode(&mut self, mode: GizmoMode) {
        if self.mode != mode {
            self.mode = mode;
            self.needs_rebuild = true;
        }
    }

    /// Get current mode.
    pub fn mode(&self) -> GizmoMode {
        self.mode
    }

    /// Set position.
    pub fn set_position(&mut self, position: Vector3) {
        self.position = position;
        self.line.position = position;
    }

    /// Get position.
    pub fn position(&self) -> Vector3 {
        self.position
    }

    /// Set rotation (for local space gizmo).
    pub fn set_rotation(&mut self, rotation: Quaternion) {
        self.rotation = rotation;
        self.needs_rebuild = true;
    }

    /// Set scale (for reference, not for gizmo visual scale).
    pub fn set_scale(&mut self, scale: Vector3) {
        self.scale = scale;
    }

    /// Update scale factor for constant screen size.
    /// Call this each frame with the camera position.
    pub fn update_screen_scale(&mut self, camera_position: Vector3) {
        let distance = (camera_position - self.position).length();
        // Scale factor to maintain constant screen size
        self.screen_scale = distance * 0.15;
        self.needs_rebuild = true;
    }

    /// Check if currently dragging.
    pub fn is_dragging(&self) -> bool {
        self.drag_state.is_some()
    }

    /// Get hovered axis.
    pub fn hovered_axis(&self) -> GizmoAxis {
        self.hovered_axis
    }

    /// Hit test the gizmo with a ray.
    /// Returns the axis under the ray, if any.
    pub fn hit_test(&mut self, ray: &Ray) -> Option<GizmoAxis> {
        let size = self.config.size * self.screen_scale;
        let pick_radius = self.config.pick_radius * self.screen_scale;

        let result = match self.mode {
            GizmoMode::Translate => self.hit_test_translate(ray, size, pick_radius),
            GizmoMode::Rotate => self.hit_test_rotate(ray, size, pick_radius),
            GizmoMode::Scale => self.hit_test_scale(ray, size, pick_radius),
        };

        let new_hovered = result.unwrap_or(GizmoAxis::None);
        if new_hovered != self.hovered_axis {
            self.hovered_axis = new_hovered;
            self.needs_rebuild = true;
        }

        result
    }

    fn hit_test_translate(&self, ray: &Ray, size: f32, pick_radius: f32) -> Option<GizmoAxis> {
        let origin = self.position;

        // Test plane handles first (they're smaller, so prioritize)
        let plane_size = self.config.plane_handle_size * size;
        let plane_offset = size * 0.3;

        // XY plane handle
        if self.hit_test_plane_handle(ray, origin, Vector3::UNIT_Z, plane_offset, plane_size) {
            return Some(GizmoAxis::XY);
        }
        // XZ plane handle
        if self.hit_test_plane_handle(ray, origin, Vector3::UNIT_Y, plane_offset, plane_size) {
            return Some(GizmoAxis::XZ);
        }
        // YZ plane handle
        if self.hit_test_plane_handle(ray, origin, Vector3::UNIT_X, plane_offset, plane_size) {
            return Some(GizmoAxis::YZ);
        }

        // Test axes
        if let Some(t) = self.ray_line_distance(ray, origin, origin + Vector3::UNIT_X * size) {
            if t < pick_radius {
                return Some(GizmoAxis::X);
            }
        }
        if let Some(t) = self.ray_line_distance(ray, origin, origin + Vector3::UNIT_Y * size) {
            if t < pick_radius {
                return Some(GizmoAxis::Y);
            }
        }
        if let Some(t) = self.ray_line_distance(ray, origin, origin + Vector3::UNIT_Z * size) {
            if t < pick_radius {
                return Some(GizmoAxis::Z);
            }
        }

        None
    }

    fn hit_test_rotate(&self, ray: &Ray, size: f32, pick_radius: f32) -> Option<GizmoAxis> {
        let origin = self.position;
        let pick_radius_sq = pick_radius * pick_radius;

        // Check distance to each rotation circle
        let mut best_axis = None;
        let mut best_dist = f32::MAX;

        // X rotation (YZ plane)
        if let Some(dist) = self.ray_circle_distance(ray, origin, Vector3::UNIT_X, size) {
            if dist < pick_radius && dist < best_dist {
                best_dist = dist;
                best_axis = Some(GizmoAxis::X);
            }
        }
        // Y rotation (XZ plane)
        if let Some(dist) = self.ray_circle_distance(ray, origin, Vector3::UNIT_Y, size) {
            if dist < pick_radius && dist < best_dist {
                best_dist = dist;
                best_axis = Some(GizmoAxis::Y);
            }
        }
        // Z rotation (XY plane)
        if let Some(dist) = self.ray_circle_distance(ray, origin, Vector3::UNIT_Z, size) {
            if dist < pick_radius && dist < best_dist {
                best_dist = dist;
                best_axis = Some(GizmoAxis::Z);
            }
        }

        best_axis
    }

    fn hit_test_scale(&self, ray: &Ray, size: f32, pick_radius: f32) -> Option<GizmoAxis> {
        let origin = self.position;

        // Test center handle for uniform scale
        let center_size = size * 0.15;
        let to_origin = origin - ray.origin;
        let t = to_origin.dot(&ray.direction);
        if t > 0.0 {
            let closest = ray.at(t);
            if closest.distance_to(&origin) < center_size {
                return Some(GizmoAxis::Center);
            }
        }

        // Test axes (same as translate)
        if let Some(t) = self.ray_line_distance(ray, origin, origin + Vector3::UNIT_X * size) {
            if t < pick_radius {
                return Some(GizmoAxis::X);
            }
        }
        if let Some(t) = self.ray_line_distance(ray, origin, origin + Vector3::UNIT_Y * size) {
            if t < pick_radius {
                return Some(GizmoAxis::Y);
            }
        }
        if let Some(t) = self.ray_line_distance(ray, origin, origin + Vector3::UNIT_Z * size) {
            if t < pick_radius {
                return Some(GizmoAxis::Z);
            }
        }

        None
    }

    fn hit_test_plane_handle(
        &self,
        ray: &Ray,
        origin: Vector3,
        normal: Vector3,
        offset: f32,
        size: f32,
    ) -> bool {
        let plane = Plane::from_normal_and_point(normal, &origin);
        if let Some(point) = ray.intersect_plane(&plane) {
            let local = point - origin;
            // Check if within the plane handle quad
            let (u, v) = match normal {
                n if n.approx_eq(&Vector3::UNIT_X, 0.01) => (local.y, local.z),
                n if n.approx_eq(&Vector3::UNIT_Y, 0.01) => (local.x, local.z),
                _ => (local.x, local.y),
            };
            u >= offset && u <= offset + size && v >= offset && v <= offset + size
        } else {
            false
        }
    }

    /// Compute distance from ray to line segment.
    fn ray_line_distance(&self, ray: &Ray, p0: Vector3, p1: Vector3) -> Option<f32> {
        let u = ray.direction;
        let v = p1 - p0;
        let w = ray.origin - p0;

        let a = u.dot(&u);
        let b = u.dot(&v);
        let c = v.dot(&v);
        let d = u.dot(&w);
        let e = v.dot(&w);

        let denom = a * c - b * b;
        if denom.abs() < 1e-8 {
            return None;
        }

        let mut s = (b * e - c * d) / denom;
        let mut t = (a * e - b * d) / denom;

        // Clamp to segment
        t = t.clamp(0.0, 1.0);
        s = s.max(0.0);

        let closest_ray = ray.at(s);
        let closest_line = p0 + v * t;

        Some(closest_ray.distance_to(&closest_line))
    }

    /// Compute distance from ray to circle.
    fn ray_circle_distance(&self, ray: &Ray, center: Vector3, normal: Vector3, radius: f32) -> Option<f32> {
        let plane = Plane::from_normal_and_point(normal, &center);
        if let Some(point) = ray.intersect_plane(&plane) {
            let to_point = point - center;
            let dist_from_center = to_point.length();
            // Distance to the circle
            Some((dist_from_center - radius).abs())
        } else {
            None
        }
    }

    /// Begin a drag operation.
    pub fn begin_drag(
        &mut self,
        axis: GizmoAxis,
        ray: &Ray,
        current_position: Vector3,
        current_rotation: Quaternion,
        current_scale: Vector3,
    ) -> bool {
        if axis == GizmoAxis::None {
            return false;
        }

        let plane = self.compute_drag_plane(axis, ray);
        let start_point = if let Some(p) = ray.intersect_plane(&plane) {
            p
        } else {
            return false;
        };

        self.drag_state = Some(DragState {
            axis,
            start_point,
            start_position: current_position,
            start_rotation: current_rotation,
            start_scale: current_scale,
            plane,
            accumulated_angle: 0.0,
        });

        self.hovered_axis = axis;
        self.needs_rebuild = true;
        true
    }

    fn compute_drag_plane(&self, axis: GizmoAxis, ray: &Ray) -> Plane {
        let origin = self.position;
        let view_dir = ray.direction;

        match axis {
            GizmoAxis::X => {
                // Choose plane that's most perpendicular to view
                let dot_y = view_dir.dot(&Vector3::UNIT_Y).abs();
                let dot_z = view_dir.dot(&Vector3::UNIT_Z).abs();
                if dot_y > dot_z {
                    Plane::from_normal_and_point(Vector3::UNIT_Y, &origin)
                } else {
                    Plane::from_normal_and_point(Vector3::UNIT_Z, &origin)
                }
            }
            GizmoAxis::Y => {
                let dot_x = view_dir.dot(&Vector3::UNIT_X).abs();
                let dot_z = view_dir.dot(&Vector3::UNIT_Z).abs();
                if dot_x > dot_z {
                    Plane::from_normal_and_point(Vector3::UNIT_X, &origin)
                } else {
                    Plane::from_normal_and_point(Vector3::UNIT_Z, &origin)
                }
            }
            GizmoAxis::Z => {
                let dot_x = view_dir.dot(&Vector3::UNIT_X).abs();
                let dot_y = view_dir.dot(&Vector3::UNIT_Y).abs();
                if dot_x > dot_y {
                    Plane::from_normal_and_point(Vector3::UNIT_X, &origin)
                } else {
                    Plane::from_normal_and_point(Vector3::UNIT_Y, &origin)
                }
            }
            GizmoAxis::XY => Plane::from_normal_and_point(Vector3::UNIT_Z, &origin),
            GizmoAxis::XZ => Plane::from_normal_and_point(Vector3::UNIT_Y, &origin),
            GizmoAxis::YZ => Plane::from_normal_and_point(Vector3::UNIT_X, &origin),
            GizmoAxis::Center | GizmoAxis::None => {
                // Use view-aligned plane
                Plane::from_normal_and_point(-view_dir, &origin)
            }
        }
    }

    /// Update drag operation and return the transform delta.
    pub fn update_drag(&mut self, ray: &Ray) -> GizmoDragResult {
        let state = match &mut self.drag_state {
            Some(s) => s,
            None => return GizmoDragResult::None,
        };

        let current_point = match ray.intersect_plane(&state.plane) {
            Some(p) => p,
            None => return GizmoDragResult::None,
        };

        match self.mode {
            GizmoMode::Translate => {
                let delta = current_point - state.start_point;
                let constrained_delta = match state.axis {
                    GizmoAxis::X => Vector3::new(delta.x, 0.0, 0.0),
                    GizmoAxis::Y => Vector3::new(0.0, delta.y, 0.0),
                    GizmoAxis::Z => Vector3::new(0.0, 0.0, delta.z),
                    GizmoAxis::XY => Vector3::new(delta.x, delta.y, 0.0),
                    GizmoAxis::XZ => Vector3::new(delta.x, 0.0, delta.z),
                    GizmoAxis::YZ => Vector3::new(0.0, delta.y, delta.z),
                    _ => delta,
                };
                GizmoDragResult::Translate(constrained_delta)
            }
            GizmoMode::Rotate => {
                let origin = self.position;
                let start_dir = (state.start_point - origin).normalized();
                let current_dir = (current_point - origin).normalized();

                let rotation_axis = match state.axis {
                    GizmoAxis::X => Vector3::UNIT_X,
                    GizmoAxis::Y => Vector3::UNIT_Y,
                    GizmoAxis::Z => Vector3::UNIT_Z,
                    _ => return GizmoDragResult::None,
                };

                // Project vectors onto the rotation plane
                let start_proj = (start_dir - rotation_axis * start_dir.dot(&rotation_axis)).normalized();
                let current_proj = (current_dir - rotation_axis * current_dir.dot(&rotation_axis)).normalized();

                if start_proj.length() < 0.001 || current_proj.length() < 0.001 {
                    return GizmoDragResult::None;
                }

                let mut angle = start_proj.dot(&current_proj).clamp(-1.0, 1.0).acos();
                let cross = start_proj.cross(&current_proj);
                if cross.dot(&rotation_axis) < 0.0 {
                    angle = -angle;
                }

                let rotation = Quaternion::from_axis_angle(&rotation_axis, angle);
                GizmoDragResult::Rotate(rotation)
            }
            GizmoMode::Scale => {
                let delta = current_point - state.start_point;
                let origin = self.position;

                match state.axis {
                    GizmoAxis::X => {
                        let factor = 1.0 + delta.x / self.config.size / self.screen_scale;
                        GizmoDragResult::Scale(Vector3::new(factor, 1.0, 1.0))
                    }
                    GizmoAxis::Y => {
                        let factor = 1.0 + delta.y / self.config.size / self.screen_scale;
                        GizmoDragResult::Scale(Vector3::new(1.0, factor, 1.0))
                    }
                    GizmoAxis::Z => {
                        let factor = 1.0 + delta.z / self.config.size / self.screen_scale;
                        GizmoDragResult::Scale(Vector3::new(1.0, 1.0, factor))
                    }
                    GizmoAxis::Center => {
                        let start_dist = state.start_point.distance_to(&origin);
                        let current_dist = current_point.distance_to(&origin);
                        let factor = if start_dist > 0.001 {
                            current_dist / start_dist
                        } else {
                            1.0
                        };
                        GizmoDragResult::Scale(Vector3::new(factor, factor, factor))
                    }
                    _ => GizmoDragResult::None,
                }
            }
        }
    }

    /// End drag operation.
    pub fn end_drag(&mut self) {
        self.drag_state = None;
        self.hovered_axis = GizmoAxis::None;
        self.needs_rebuild = true;
    }

    /// Get the underlying line for rendering.
    pub fn line(&self) -> &Line {
        &self.line
    }

    /// Get mutable line.
    pub fn line_mut(&mut self) -> &mut Line {
        &mut self.line
    }

    /// Rebuild geometry if needed and update buffers.
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.needs_rebuild {
            self.rebuild_geometry();
            self.needs_rebuild = false;
        }
        self.line.update_buffer(device, queue);
    }

    fn rebuild_geometry(&mut self) {
        let size = self.config.size * self.screen_scale;
        let vertices = match self.mode {
            GizmoMode::Translate => self.build_translate_geometry(size),
            GizmoMode::Rotate => self.build_rotate_geometry(size),
            GizmoMode::Scale => self.build_scale_geometry(size),
        };
        self.line.set_vertices(vertices);
        self.line.position = self.position;
    }

    fn get_axis_color(&self, axis: GizmoAxis) -> [f32; 4] {
        if self.hovered_axis == axis || self.drag_state.as_ref().map(|s| s.axis) == Some(axis) {
            GizmoAxis::highlight_color()
        } else {
            axis.color()
        }
    }

    fn build_translate_geometry(&self, size: f32) -> Vec<LineVertex> {
        let mut vertices = Vec::with_capacity(64);
        let arrowhead = self.config.arrowhead_size * size;
        let plane_offset = size * 0.3;
        let plane_size = self.config.plane_handle_size * size;

        // X axis with arrowhead
        let x_color = self.get_axis_color(GizmoAxis::X);
        vertices.push(LineVertex::new([0.0, 0.0, 0.0], x_color));
        vertices.push(LineVertex::new([size, 0.0, 0.0], x_color));
        // Arrowhead
        vertices.push(LineVertex::new([size, 0.0, 0.0], x_color));
        vertices.push(LineVertex::new([size - arrowhead, arrowhead * 0.5, 0.0], x_color));
        vertices.push(LineVertex::new([size, 0.0, 0.0], x_color));
        vertices.push(LineVertex::new([size - arrowhead, -arrowhead * 0.5, 0.0], x_color));
        vertices.push(LineVertex::new([size, 0.0, 0.0], x_color));
        vertices.push(LineVertex::new([size - arrowhead, 0.0, arrowhead * 0.5], x_color));
        vertices.push(LineVertex::new([size, 0.0, 0.0], x_color));
        vertices.push(LineVertex::new([size - arrowhead, 0.0, -arrowhead * 0.5], x_color));

        // Y axis with arrowhead
        let y_color = self.get_axis_color(GizmoAxis::Y);
        vertices.push(LineVertex::new([0.0, 0.0, 0.0], y_color));
        vertices.push(LineVertex::new([0.0, size, 0.0], y_color));
        vertices.push(LineVertex::new([0.0, size, 0.0], y_color));
        vertices.push(LineVertex::new([arrowhead * 0.5, size - arrowhead, 0.0], y_color));
        vertices.push(LineVertex::new([0.0, size, 0.0], y_color));
        vertices.push(LineVertex::new([-arrowhead * 0.5, size - arrowhead, 0.0], y_color));
        vertices.push(LineVertex::new([0.0, size, 0.0], y_color));
        vertices.push(LineVertex::new([0.0, size - arrowhead, arrowhead * 0.5], y_color));
        vertices.push(LineVertex::new([0.0, size, 0.0], y_color));
        vertices.push(LineVertex::new([0.0, size - arrowhead, -arrowhead * 0.5], y_color));

        // Z axis with arrowhead
        let z_color = self.get_axis_color(GizmoAxis::Z);
        vertices.push(LineVertex::new([0.0, 0.0, 0.0], z_color));
        vertices.push(LineVertex::new([0.0, 0.0, size], z_color));
        vertices.push(LineVertex::new([0.0, 0.0, size], z_color));
        vertices.push(LineVertex::new([arrowhead * 0.5, 0.0, size - arrowhead], z_color));
        vertices.push(LineVertex::new([0.0, 0.0, size], z_color));
        vertices.push(LineVertex::new([-arrowhead * 0.5, 0.0, size - arrowhead], z_color));
        vertices.push(LineVertex::new([0.0, 0.0, size], z_color));
        vertices.push(LineVertex::new([0.0, arrowhead * 0.5, size - arrowhead], z_color));
        vertices.push(LineVertex::new([0.0, 0.0, size], z_color));
        vertices.push(LineVertex::new([0.0, -arrowhead * 0.5, size - arrowhead], z_color));

        // XY plane handle
        let xy_color = self.get_axis_color(GizmoAxis::XY);
        vertices.push(LineVertex::new([plane_offset, plane_offset, 0.0], xy_color));
        vertices.push(LineVertex::new([plane_offset + plane_size, plane_offset, 0.0], xy_color));
        vertices.push(LineVertex::new([plane_offset + plane_size, plane_offset, 0.0], xy_color));
        vertices.push(LineVertex::new([plane_offset + plane_size, plane_offset + plane_size, 0.0], xy_color));
        vertices.push(LineVertex::new([plane_offset + plane_size, plane_offset + plane_size, 0.0], xy_color));
        vertices.push(LineVertex::new([plane_offset, plane_offset + plane_size, 0.0], xy_color));
        vertices.push(LineVertex::new([plane_offset, plane_offset + plane_size, 0.0], xy_color));
        vertices.push(LineVertex::new([plane_offset, plane_offset, 0.0], xy_color));

        // XZ plane handle
        let xz_color = self.get_axis_color(GizmoAxis::XZ);
        vertices.push(LineVertex::new([plane_offset, 0.0, plane_offset], xz_color));
        vertices.push(LineVertex::new([plane_offset + plane_size, 0.0, plane_offset], xz_color));
        vertices.push(LineVertex::new([plane_offset + plane_size, 0.0, plane_offset], xz_color));
        vertices.push(LineVertex::new([plane_offset + plane_size, 0.0, plane_offset + plane_size], xz_color));
        vertices.push(LineVertex::new([plane_offset + plane_size, 0.0, plane_offset + plane_size], xz_color));
        vertices.push(LineVertex::new([plane_offset, 0.0, plane_offset + plane_size], xz_color));
        vertices.push(LineVertex::new([plane_offset, 0.0, plane_offset + plane_size], xz_color));
        vertices.push(LineVertex::new([plane_offset, 0.0, plane_offset], xz_color));

        // YZ plane handle
        let yz_color = self.get_axis_color(GizmoAxis::YZ);
        vertices.push(LineVertex::new([0.0, plane_offset, plane_offset], yz_color));
        vertices.push(LineVertex::new([0.0, plane_offset + plane_size, plane_offset], yz_color));
        vertices.push(LineVertex::new([0.0, plane_offset + plane_size, plane_offset], yz_color));
        vertices.push(LineVertex::new([0.0, plane_offset + plane_size, plane_offset + plane_size], yz_color));
        vertices.push(LineVertex::new([0.0, plane_offset + plane_size, plane_offset + plane_size], yz_color));
        vertices.push(LineVertex::new([0.0, plane_offset, plane_offset + plane_size], yz_color));
        vertices.push(LineVertex::new([0.0, plane_offset, plane_offset + plane_size], yz_color));
        vertices.push(LineVertex::new([0.0, plane_offset, plane_offset], yz_color));

        vertices
    }

    fn build_rotate_geometry(&self, size: f32) -> Vec<LineVertex> {
        let segments = self.config.circle_segments as usize;
        let mut vertices = Vec::with_capacity(segments * 6);

        // X rotation circle (in YZ plane)
        let x_color = self.get_axis_color(GizmoAxis::X);
        for i in 0..segments {
            let a1 = (i as f32 / segments as f32) * std::f32::consts::TAU;
            let a2 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;
            vertices.push(LineVertex::new([0.0, a1.cos() * size, a1.sin() * size], x_color));
            vertices.push(LineVertex::new([0.0, a2.cos() * size, a2.sin() * size], x_color));
        }

        // Y rotation circle (in XZ plane)
        let y_color = self.get_axis_color(GizmoAxis::Y);
        for i in 0..segments {
            let a1 = (i as f32 / segments as f32) * std::f32::consts::TAU;
            let a2 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;
            vertices.push(LineVertex::new([a1.cos() * size, 0.0, a1.sin() * size], y_color));
            vertices.push(LineVertex::new([a2.cos() * size, 0.0, a2.sin() * size], y_color));
        }

        // Z rotation circle (in XY plane)
        let z_color = self.get_axis_color(GizmoAxis::Z);
        for i in 0..segments {
            let a1 = (i as f32 / segments as f32) * std::f32::consts::TAU;
            let a2 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;
            vertices.push(LineVertex::new([a1.cos() * size, a1.sin() * size, 0.0], z_color));
            vertices.push(LineVertex::new([a2.cos() * size, a2.sin() * size, 0.0], z_color));
        }

        vertices
    }

    fn build_scale_geometry(&self, size: f32) -> Vec<LineVertex> {
        let mut vertices = Vec::with_capacity(48);
        let box_size = size * 0.1;
        let center_size = size * 0.15;

        // X axis with box end
        let x_color = self.get_axis_color(GizmoAxis::X);
        vertices.push(LineVertex::new([0.0, 0.0, 0.0], x_color));
        vertices.push(LineVertex::new([size, 0.0, 0.0], x_color));
        self.add_box_at(&mut vertices, [size, 0.0, 0.0], box_size, x_color);

        // Y axis with box end
        let y_color = self.get_axis_color(GizmoAxis::Y);
        vertices.push(LineVertex::new([0.0, 0.0, 0.0], y_color));
        vertices.push(LineVertex::new([0.0, size, 0.0], y_color));
        self.add_box_at(&mut vertices, [0.0, size, 0.0], box_size, y_color);

        // Z axis with box end
        let z_color = self.get_axis_color(GizmoAxis::Z);
        vertices.push(LineVertex::new([0.0, 0.0, 0.0], z_color));
        vertices.push(LineVertex::new([0.0, 0.0, size], z_color));
        self.add_box_at(&mut vertices, [0.0, 0.0, size], box_size, z_color);

        // Center box for uniform scale
        let center_color = self.get_axis_color(GizmoAxis::Center);
        self.add_box_at(&mut vertices, [0.0, 0.0, 0.0], center_size, center_color);

        vertices
    }

    fn add_box_at(&self, vertices: &mut Vec<LineVertex>, center: [f32; 3], half_size: f32, color: [f32; 4]) {
        let [cx, cy, cz] = center;
        let h = half_size;

        // Bottom face
        vertices.push(LineVertex::new([cx - h, cy - h, cz - h], color));
        vertices.push(LineVertex::new([cx + h, cy - h, cz - h], color));
        vertices.push(LineVertex::new([cx + h, cy - h, cz - h], color));
        vertices.push(LineVertex::new([cx + h, cy - h, cz + h], color));
        vertices.push(LineVertex::new([cx + h, cy - h, cz + h], color));
        vertices.push(LineVertex::new([cx - h, cy - h, cz + h], color));
        vertices.push(LineVertex::new([cx - h, cy - h, cz + h], color));
        vertices.push(LineVertex::new([cx - h, cy - h, cz - h], color));

        // Top face
        vertices.push(LineVertex::new([cx - h, cy + h, cz - h], color));
        vertices.push(LineVertex::new([cx + h, cy + h, cz - h], color));
        vertices.push(LineVertex::new([cx + h, cy + h, cz - h], color));
        vertices.push(LineVertex::new([cx + h, cy + h, cz + h], color));
        vertices.push(LineVertex::new([cx + h, cy + h, cz + h], color));
        vertices.push(LineVertex::new([cx - h, cy + h, cz + h], color));
        vertices.push(LineVertex::new([cx - h, cy + h, cz + h], color));
        vertices.push(LineVertex::new([cx - h, cy + h, cz - h], color));

        // Vertical edges
        vertices.push(LineVertex::new([cx - h, cy - h, cz - h], color));
        vertices.push(LineVertex::new([cx - h, cy + h, cz - h], color));
        vertices.push(LineVertex::new([cx + h, cy - h, cz - h], color));
        vertices.push(LineVertex::new([cx + h, cy + h, cz - h], color));
        vertices.push(LineVertex::new([cx + h, cy - h, cz + h], color));
        vertices.push(LineVertex::new([cx + h, cy + h, cz + h], color));
        vertices.push(LineVertex::new([cx - h, cy - h, cz + h], color));
        vertices.push(LineVertex::new([cx - h, cy + h, cz + h], color));
    }
}

impl Default for TransformGizmo {
    fn default() -> Self {
        Self::new()
    }
}
