//! Loading manager for tracking multiple asset loads.

use super::{LoadProgress, LoadState};
use std::collections::HashMap;

/// Manages multiple loading operations and tracks overall progress.
pub struct LoadingManager {
    /// Items currently being loaded.
    items: HashMap<String, LoadState>,
    /// Number of items loaded.
    loaded_count: usize,
    /// Total number of items to load.
    total_count: usize,
    /// Current overall state.
    state: LoadState,
    /// Errors encountered.
    errors: Vec<String>,
}

impl LoadingManager {
    /// Create a new loading manager.
    pub fn new() -> Self {
        Self {
            items: HashMap::new(),
            loaded_count: 0,
            total_count: 0,
            state: LoadState::Idle,
            errors: Vec::new(),
        }
    }

    /// Start tracking a new item.
    pub fn item_start(&mut self, url: impl Into<String>) {
        let url = url.into();
        self.items.insert(url, LoadState::Loading);
        self.total_count += 1;
        self.state = LoadState::Loading;
    }

    /// Mark an item as successfully loaded.
    pub fn item_end(&mut self, url: &str) {
        if let Some(state) = self.items.get_mut(url) {
            *state = LoadState::Loaded;
            self.loaded_count += 1;
            self.update_state();
        }
    }

    /// Mark an item as failed.
    pub fn item_error(&mut self, url: &str, error: impl Into<String>) {
        if let Some(state) = self.items.get_mut(url) {
            *state = LoadState::Failed;
            self.errors.push(format!("{}: {}", url, error.into()));
            self.loaded_count += 1;
            self.update_state();
        }
    }

    /// Update overall state based on item states.
    fn update_state(&mut self) {
        if self.loaded_count >= self.total_count {
            if self.errors.is_empty() {
                self.state = LoadState::Loaded;
            } else {
                self.state = LoadState::Failed;
            }
        }
    }

    /// Get current overall state.
    #[inline]
    pub fn state(&self) -> LoadState {
        self.state
    }

    /// Check if all items are loaded.
    #[inline]
    pub fn is_loaded(&self) -> bool {
        self.state == LoadState::Loaded
    }

    /// Check if loading is in progress.
    #[inline]
    pub fn is_loading(&self) -> bool {
        self.state == LoadState::Loading
    }

    /// Check if any errors occurred.
    #[inline]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Get all errors.
    #[inline]
    pub fn errors(&self) -> &[String] {
        &self.errors
    }

    /// Get progress information.
    pub fn progress(&self) -> LoadProgress {
        LoadProgress {
            loaded: self.loaded_count,
            total: self.total_count,
            current_item: self.items.iter()
                .find(|(_, state)| **state == LoadState::Loading)
                .map(|(url, _)| url.clone()),
            error: self.errors.first().cloned(),
        }
    }

    /// Get progress as a fraction (0.0 - 1.0).
    pub fn fraction(&self) -> f32 {
        if self.total_count == 0 {
            1.0
        } else {
            self.loaded_count as f32 / self.total_count as f32
        }
    }

    /// Reset the manager.
    pub fn reset(&mut self) {
        self.items.clear();
        self.loaded_count = 0;
        self.total_count = 0;
        self.state = LoadState::Idle;
        self.errors.clear();
    }
}

impl Default for LoadingManager {
    fn default() -> Self {
        Self::new()
    }
}
