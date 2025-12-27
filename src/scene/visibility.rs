//! Visibility settings for scene objects.

use super::Layers;

/// Visibility settings for a scene object.
#[derive(Debug, Clone)]
pub struct Visibility {
    /// Whether the object is visible.
    visible: bool,
    /// Whether the object casts shadows.
    cast_shadow: bool,
    /// Whether the object receives shadows.
    receive_shadow: bool,
    /// Whether to frustum cull this object.
    frustum_culled: bool,
    /// Render order (lower = rendered first).
    render_order: i32,
    /// Layer mask for camera visibility.
    layers: Layers,
}

impl Default for Visibility {
    fn default() -> Self {
        Self::new()
    }
}

impl Visibility {
    /// Create new visibility settings with defaults.
    pub fn new() -> Self {
        Self {
            visible: true,
            cast_shadow: false,
            receive_shadow: false,
            frustum_culled: true,
            render_order: 0,
            layers: Layers::DEFAULT,
        }
    }

    /// Check if visible.
    #[inline]
    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// Set visibility.
    #[inline]
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Check if casts shadows.
    #[inline]
    pub fn casts_shadow(&self) -> bool {
        self.cast_shadow
    }

    /// Set cast shadow.
    #[inline]
    pub fn set_cast_shadow(&mut self, cast: bool) {
        self.cast_shadow = cast;
    }

    /// Check if receives shadows.
    #[inline]
    pub fn receives_shadow(&self) -> bool {
        self.receive_shadow
    }

    /// Set receive shadow.
    #[inline]
    pub fn set_receive_shadow(&mut self, receive: bool) {
        self.receive_shadow = receive;
    }

    /// Check if frustum culled.
    #[inline]
    pub fn is_frustum_culled(&self) -> bool {
        self.frustum_culled
    }

    /// Set frustum culling.
    #[inline]
    pub fn set_frustum_culled(&mut self, culled: bool) {
        self.frustum_culled = culled;
    }

    /// Get render order.
    #[inline]
    pub fn render_order(&self) -> i32 {
        self.render_order
    }

    /// Set render order.
    #[inline]
    pub fn set_render_order(&mut self, order: i32) {
        self.render_order = order;
    }

    /// Get layers.
    #[inline]
    pub fn layers(&self) -> &Layers {
        &self.layers
    }

    /// Get mutable layers.
    #[inline]
    pub fn layers_mut(&mut self) -> &mut Layers {
        &mut self.layers
    }

    /// Check if visible to a camera with the given layers.
    #[inline]
    pub fn visible_to_camera(&self, camera_layers: &Layers) -> bool {
        self.visible && self.layers.intersects(camera_layers)
    }
}
