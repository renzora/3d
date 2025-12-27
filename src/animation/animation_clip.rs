//! Animation clip containing keyframe tracks.

use super::Track;
use crate::core::Id;

/// An animation clip contains multiple tracks that animate properties over time.
#[derive(Debug, Clone)]
pub struct AnimationClip {
    /// Unique identifier.
    id: Id,
    /// Name of the animation.
    pub name: String,
    /// Duration in seconds (computed from tracks).
    duration: f32,
    /// Tracks in this clip.
    tracks: Vec<Track>,
}

impl AnimationClip {
    /// Create a new empty animation clip.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Id::new(),
            name: name.into(),
            duration: 0.0,
            tracks: Vec::new(),
        }
    }

    /// Get the unique ID.
    #[inline]
    pub fn id(&self) -> Id {
        self.id
    }

    /// Get the duration in seconds.
    #[inline]
    pub fn duration(&self) -> f32 {
        self.duration
    }

    /// Get the tracks.
    #[inline]
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Get mutable tracks.
    #[inline]
    pub fn tracks_mut(&mut self) -> &mut Vec<Track> {
        &mut self.tracks
    }

    /// Add a track to the clip.
    pub fn add_track(&mut self, track: Track) {
        let track_duration = track.duration();
        if track_duration > self.duration {
            self.duration = track_duration;
        }
        self.tracks.push(track);
    }

    /// Recalculate duration from all tracks.
    pub fn update_duration(&mut self) {
        self.duration = self.tracks.iter()
            .map(|t| t.duration())
            .fold(0.0, f32::max);
    }

    /// Find a track by name.
    pub fn find_track(&self, name: &str) -> Option<&Track> {
        self.tracks.iter().find(|t| t.name() == name)
    }

    /// Find a mutable track by name.
    pub fn find_track_mut(&mut self, name: &str) -> Option<&mut Track> {
        self.tracks.iter_mut().find(|t| t.name() == name)
    }

    /// Get the number of tracks.
    #[inline]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Check if the clip is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Clear all tracks.
    pub fn clear(&mut self) {
        self.tracks.clear();
        self.duration = 0.0;
    }
}

impl Default for AnimationClip {
    fn default() -> Self {
        Self::new("Unnamed")
    }
}
