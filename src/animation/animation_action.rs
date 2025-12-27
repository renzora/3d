//! Animation action - a single animation instance.

use super::AnimationClip;
use std::sync::Arc;

/// Loop mode for animations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoopMode {
    /// Play once and stop.
    #[default]
    Once,
    /// Loop continuously.
    Loop,
    /// Ping-pong (forward then backward).
    PingPong,
}

/// State of an animation action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActionState {
    /// Not playing.
    #[default]
    Stopped,
    /// Currently playing.
    Playing,
    /// Paused.
    Paused,
}

/// An animation action controls playback of an animation clip.
#[derive(Debug, Clone)]
pub struct AnimationAction {
    /// The animation clip being played.
    clip: Arc<AnimationClip>,
    /// Current playback time in seconds.
    time: f32,
    /// Playback speed multiplier (1.0 = normal, 2.0 = double speed).
    pub time_scale: f32,
    /// Weight for blending (0.0 - 1.0).
    pub weight: f32,
    /// Loop mode.
    pub loop_mode: LoopMode,
    /// Number of repetitions (for Loop mode, 0 = infinite).
    pub repetitions: u32,
    /// Current repetition count.
    current_repetition: u32,
    /// Current playback state.
    state: ActionState,
    /// Whether playing in reverse (for PingPong).
    is_reversed: bool,
    /// Blend in duration (seconds).
    pub fade_in_duration: f32,
    /// Blend out duration (seconds).
    pub fade_out_duration: f32,
    /// Time when fade started.
    fade_start_time: f32,
    /// Whether fading in.
    is_fading_in: bool,
    /// Whether fading out.
    is_fading_out: bool,
}

impl AnimationAction {
    /// Create a new animation action for a clip.
    pub fn new(clip: Arc<AnimationClip>) -> Self {
        Self {
            clip,
            time: 0.0,
            time_scale: 1.0,
            weight: 1.0,
            loop_mode: LoopMode::Once,
            repetitions: 0,
            current_repetition: 0,
            state: ActionState::Stopped,
            is_reversed: false,
            fade_in_duration: 0.0,
            fade_out_duration: 0.0,
            fade_start_time: 0.0,
            is_fading_in: false,
            is_fading_out: false,
        }
    }

    /// Get the animation clip.
    #[inline]
    pub fn clip(&self) -> &AnimationClip {
        &self.clip
    }

    /// Get the current time.
    #[inline]
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Set the current time.
    pub fn set_time(&mut self, time: f32) {
        self.time = time.clamp(0.0, self.clip.duration());
    }

    /// Get the current state.
    #[inline]
    pub fn state(&self) -> ActionState {
        self.state
    }

    /// Check if playing.
    #[inline]
    pub fn is_playing(&self) -> bool {
        self.state == ActionState::Playing
    }

    /// Check if stopped.
    #[inline]
    pub fn is_stopped(&self) -> bool {
        self.state == ActionState::Stopped
    }

    /// Check if paused.
    #[inline]
    pub fn is_paused(&self) -> bool {
        self.state == ActionState::Paused
    }

    /// Start playing the animation.
    pub fn play(&mut self) {
        self.state = ActionState::Playing;
        if self.fade_in_duration > 0.0 {
            self.is_fading_in = true;
            self.fade_start_time = self.time;
        }
    }

    /// Stop the animation and reset to start.
    pub fn stop(&mut self) {
        self.state = ActionState::Stopped;
        self.time = 0.0;
        self.current_repetition = 0;
        self.is_reversed = false;
        self.is_fading_in = false;
        self.is_fading_out = false;
    }

    /// Pause the animation.
    pub fn pause(&mut self) {
        if self.state == ActionState::Playing {
            self.state = ActionState::Paused;
        }
    }

    /// Resume from pause.
    pub fn resume(&mut self) {
        if self.state == ActionState::Paused {
            self.state = ActionState::Playing;
        }
    }

    /// Reset to the beginning.
    pub fn reset(&mut self) {
        self.time = 0.0;
        self.current_repetition = 0;
        self.is_reversed = false;
    }

    /// Start fading out.
    pub fn fade_out(&mut self, duration: f32) {
        self.fade_out_duration = duration;
        self.is_fading_out = true;
        self.fade_start_time = self.time;
    }

    /// Get the effective weight (including fade).
    pub fn effective_weight(&self) -> f32 {
        let mut weight = self.weight;

        if self.is_fading_in && self.fade_in_duration > 0.0 {
            let fade_progress = (self.time - self.fade_start_time) / self.fade_in_duration;
            weight *= fade_progress.clamp(0.0, 1.0);
        }

        if self.is_fading_out && self.fade_out_duration > 0.0 {
            let fade_progress = (self.time - self.fade_start_time) / self.fade_out_duration;
            weight *= (1.0 - fade_progress).clamp(0.0, 1.0);
        }

        weight
    }

    /// Update the animation by delta time.
    /// Returns true if the animation is still active.
    pub fn update(&mut self, delta_time: f32) -> bool {
        if self.state != ActionState::Playing {
            return self.state != ActionState::Stopped;
        }

        let duration = self.clip.duration();
        if duration <= 0.0 {
            return false;
        }

        // Apply time scale and direction
        let time_delta = delta_time * self.time_scale * if self.is_reversed { -1.0 } else { 1.0 };
        self.time += time_delta;

        // Handle fade out completion
        if self.is_fading_out {
            let fade_progress = (self.time - self.fade_start_time) / self.fade_out_duration;
            if fade_progress >= 1.0 {
                self.stop();
                return false;
            }
        }

        // Handle fade in completion
        if self.is_fading_in {
            let fade_progress = (self.time - self.fade_start_time) / self.fade_in_duration;
            if fade_progress >= 1.0 {
                self.is_fading_in = false;
            }
        }

        // Handle looping
        match self.loop_mode {
            LoopMode::Once => {
                if self.time >= duration {
                    self.time = duration;
                    self.state = ActionState::Stopped;
                    return false;
                }
            }
            LoopMode::Loop => {
                if self.time >= duration {
                    self.time %= duration;
                    self.current_repetition += 1;
                    if self.repetitions > 0 && self.current_repetition >= self.repetitions {
                        self.stop();
                        return false;
                    }
                }
            }
            LoopMode::PingPong => {
                if !self.is_reversed && self.time >= duration {
                    self.time = duration;
                    self.is_reversed = true;
                } else if self.is_reversed && self.time <= 0.0 {
                    self.time = 0.0;
                    self.is_reversed = false;
                    self.current_repetition += 1;
                    if self.repetitions > 0 && self.current_repetition >= self.repetitions {
                        self.stop();
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Get normalized time (0-1).
    pub fn normalized_time(&self) -> f32 {
        let duration = self.clip.duration();
        if duration <= 0.0 {
            0.0
        } else {
            self.time / duration
        }
    }
}
