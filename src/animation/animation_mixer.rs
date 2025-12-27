//! Animation mixer for managing multiple animations.

use super::{AnimationAction, AnimationClip, Track, ActionState};
use crate::core::Id;
use std::collections::HashMap;
use std::sync::Arc;

/// Result of sampling an animation - the animated values.
#[derive(Debug, Clone, Default)]
pub struct AnimationOutput {
    /// Position values by property name.
    pub positions: HashMap<String, [f32; 3]>,
    /// Rotation values by property name (quaternion).
    pub rotations: HashMap<String, [f32; 4]>,
    /// Scale values by property name.
    pub scales: HashMap<String, [f32; 3]>,
    /// Number values by property name.
    pub numbers: HashMap<String, f32>,
    /// Color values by property name.
    pub colors: HashMap<String, [f32; 4]>,
}

impl AnimationOutput {
    /// Create a new empty output.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear all values.
    pub fn clear(&mut self) {
        self.positions.clear();
        self.rotations.clear();
        self.scales.clear();
        self.numbers.clear();
        self.colors.clear();
    }
}

/// Animation mixer manages multiple animation actions and blends them.
pub struct AnimationMixer {
    /// Unique identifier.
    id: Id,
    /// Active animation actions.
    actions: Vec<AnimationAction>,
    /// Global time scale.
    pub time_scale: f32,
    /// Cached output.
    output: AnimationOutput,
}

impl AnimationMixer {
    /// Create a new animation mixer.
    pub fn new() -> Self {
        Self {
            id: Id::new(),
            actions: Vec::new(),
            time_scale: 1.0,
            output: AnimationOutput::new(),
        }
    }

    /// Get the unique ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.id
    }

    /// Create and add an action for a clip.
    pub fn clip_action(&mut self, clip: Arc<AnimationClip>) -> &mut AnimationAction {
        let action = AnimationAction::new(clip);
        self.actions.push(action);
        self.actions.last_mut().unwrap()
    }

    /// Add an existing action.
    pub fn add_action(&mut self, action: AnimationAction) {
        self.actions.push(action);
    }

    /// Get all actions.
    #[inline]
    pub fn actions(&self) -> &[AnimationAction] {
        &self.actions
    }

    /// Get mutable actions.
    #[inline]
    pub fn actions_mut(&mut self) -> &mut Vec<AnimationAction> {
        &mut self.actions
    }

    /// Get action by index.
    pub fn get_action(&self, index: usize) -> Option<&AnimationAction> {
        self.actions.get(index)
    }

    /// Get mutable action by index.
    pub fn get_action_mut(&mut self, index: usize) -> Option<&mut AnimationAction> {
        self.actions.get_mut(index)
    }

    /// Stop all actions.
    pub fn stop_all(&mut self) {
        for action in &mut self.actions {
            action.stop();
        }
    }

    /// Remove stopped actions.
    pub fn remove_stopped(&mut self) {
        self.actions.retain(|a| a.state() != ActionState::Stopped);
    }

    /// Update all animations by delta time.
    pub fn update(&mut self, delta_time: f32) {
        let scaled_delta = delta_time * self.time_scale;

        for action in &mut self.actions {
            action.update(scaled_delta);
        }
    }

    /// Sample all active animations and blend the results.
    pub fn sample(&mut self) -> &AnimationOutput {
        self.output.clear();

        // Temporary storage for weighted blending
        let mut position_weights: HashMap<String, (f32, [f32; 3])> = HashMap::new();
        let mut rotation_weights: HashMap<String, (f32, [f32; 4])> = HashMap::new();
        let scale_weights: HashMap<String, (f32, [f32; 3])> = HashMap::new();
        let mut number_weights: HashMap<String, (f32, f32)> = HashMap::new();
        let mut color_weights: HashMap<String, (f32, [f32; 4])> = HashMap::new();

        for action in &self.actions {
            if !action.is_playing() && !action.is_paused() {
                continue;
            }

            let weight = action.effective_weight();
            if weight <= 0.0 {
                continue;
            }

            let time = action.time();
            let clip = action.clip();

            for track in clip.tracks() {
                match track {
                    Track::Vector(t) => {
                        let value = t.sample(time);
                        let name = t.name.clone();

                        if let Some((w, v)) = position_weights.get_mut(&name) {
                            // Blend with existing
                            let total_weight = *w + weight;
                            let blend_factor = weight / total_weight;
                            for i in 0..3 {
                                v[i] = v[i] * (1.0 - blend_factor) + value[i] * blend_factor;
                            }
                            *w = total_weight;
                        } else {
                            position_weights.insert(name, (weight, value));
                        }
                    }
                    Track::Quaternion(t) => {
                        let value = t.sample(time);
                        let name = t.name.clone();

                        if let Some((w, v)) = rotation_weights.get_mut(&name) {
                            // Simple blend (proper slerp blending would be better)
                            let total_weight = *w + weight;
                            let blend_factor = weight / total_weight;
                            for i in 0..4 {
                                v[i] = v[i] * (1.0 - blend_factor) + value[i] * blend_factor;
                            }
                            // Normalize
                            let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3]).sqrt();
                            if len > 0.0 {
                                for i in 0..4 { v[i] /= len; }
                            }
                            *w = total_weight;
                        } else {
                            rotation_weights.insert(name, (weight, value));
                        }
                    }
                    Track::Number(t) => {
                        let value = t.sample(time);
                        let name = t.name.clone();

                        if let Some((w, v)) = number_weights.get_mut(&name) {
                            let total_weight = *w + weight;
                            let blend_factor = weight / total_weight;
                            *v = *v * (1.0 - blend_factor) + value * blend_factor;
                            *w = total_weight;
                        } else {
                            number_weights.insert(name, (weight, value));
                        }
                    }
                    Track::Color(t) => {
                        let value = t.sample(time);
                        let name = t.name.clone();

                        if let Some((w, v)) = color_weights.get_mut(&name) {
                            let total_weight = *w + weight;
                            let blend_factor = weight / total_weight;
                            for i in 0..4 {
                                v[i] = v[i] * (1.0 - blend_factor) + value[i] * blend_factor;
                            }
                            *w = total_weight;
                        } else {
                            color_weights.insert(name, (weight, value));
                        }
                    }
                }
            }
        }

        // Extract final values
        for (name, (_, value)) in position_weights {
            // Check if this is a scale track by name convention
            if name.contains("scale") {
                self.output.scales.insert(name, value);
            } else {
                self.output.positions.insert(name, value);
            }
        }

        for (name, (_, value)) in rotation_weights {
            self.output.rotations.insert(name, value);
        }

        for (name, (_, value)) in scale_weights {
            self.output.scales.insert(name, value);
        }

        for (name, (_, value)) in number_weights {
            self.output.numbers.insert(name, value);
        }

        for (name, (_, value)) in color_weights {
            self.output.colors.insert(name, value);
        }

        &self.output
    }

    /// Get the last sampled output.
    #[inline]
    pub fn output(&self) -> &AnimationOutput {
        &self.output
    }

    /// Check if any animations are playing.
    pub fn is_playing(&self) -> bool {
        self.actions.iter().any(|a| a.is_playing())
    }

    /// Get the number of active actions.
    pub fn active_action_count(&self) -> usize {
        self.actions.iter().filter(|a| a.is_playing()).count()
    }
}

impl Default for AnimationMixer {
    fn default() -> Self {
        Self::new()
    }
}
