//! High-resolution clock for timing and animation.

#[cfg(feature = "web")]
use web_sys::window;

#[cfg(not(feature = "web"))]
use std::time::Instant;

/// A clock for measuring elapsed time and delta time.
pub struct Clock {
    /// Whether the clock is running.
    running: bool,
    /// Start time in seconds.
    start_time: f64,
    /// Time of the last update in seconds.
    old_time: f64,
    /// Total elapsed time while running.
    elapsed_time: f64,

    #[cfg(not(feature = "web"))]
    instant: Option<Instant>,
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock {
    /// Create a new clock (not started).
    pub fn new() -> Self {
        Self {
            running: false,
            start_time: 0.0,
            old_time: 0.0,
            elapsed_time: 0.0,
            #[cfg(not(feature = "web"))]
            instant: None,
        }
    }

    /// Create and start a new clock.
    pub fn start_new() -> Self {
        let mut clock = Self::new();
        clock.start();
        clock
    }

    /// Get the current time in seconds.
    fn now(&self) -> f64 {
        #[cfg(feature = "web")]
        {
            window()
                .and_then(|w| w.performance())
                .map(|p| p.now() / 1000.0)
                .unwrap_or(0.0)
        }

        #[cfg(not(feature = "web"))]
        {
            self.instant
                .map(|i| i.elapsed().as_secs_f64())
                .unwrap_or(0.0)
        }
    }

    /// Start the clock.
    pub fn start(&mut self) {
        #[cfg(not(feature = "web"))]
        {
            self.instant = Some(Instant::now());
        }

        self.start_time = self.now();
        self.old_time = self.start_time;
        self.elapsed_time = 0.0;
        self.running = true;
    }

    /// Stop the clock.
    pub fn stop(&mut self) {
        self.get_elapsed_time();
        self.running = false;
    }

    /// Get the elapsed time since the clock started (in seconds).
    pub fn get_elapsed_time(&mut self) -> f64 {
        self.get_delta();
        self.elapsed_time
    }

    /// Get the time since the last call to get_delta (in seconds).
    pub fn get_delta(&mut self) -> f64 {
        if !self.running {
            self.start();
            return 0.0;
        }

        let new_time = self.now();
        let diff = new_time - self.old_time;
        self.old_time = new_time;
        self.elapsed_time += diff;

        diff
    }

    /// Check if the clock is running.
    #[inline]
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Reset the clock.
    pub fn reset(&mut self) {
        self.start_time = self.now();
        self.old_time = self.start_time;
        self.elapsed_time = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_starts_stopped() {
        let clock = Clock::new();
        assert!(!clock.is_running());
    }

    #[test]
    fn test_clock_start() {
        let mut clock = Clock::new();
        clock.start();
        assert!(clock.is_running());
    }
}
