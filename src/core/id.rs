//! Unique ID generation for engine objects.

use std::sync::atomic::{AtomicU64, Ordering};

/// Global ID counter.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// A unique identifier for engine objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(u64);

impl Id {
    /// Generate a new unique ID.
    #[inline]
    pub fn new() -> Self {
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Get the raw ID value.
    #[inline]
    pub fn value(&self) -> u64 {
        self.0
    }
}

impl Default for Id {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Generator for sequential IDs with a specific prefix.
pub struct IdGenerator {
    prefix: String,
    counter: AtomicU64,
}

impl IdGenerator {
    /// Create a new ID generator with a prefix.
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            counter: AtomicU64::new(0),
        }
    }

    /// Generate the next ID.
    pub fn next(&self) -> String {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        format!("{}_{}", self.prefix, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unique_ids() {
        let id1 = Id::new();
        let id2 = Id::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_id_generator() {
        let gen = IdGenerator::new("mesh");
        assert_eq!(gen.next(), "mesh_0");
        assert_eq!(gen.next(), "mesh_1");
        assert_eq!(gen.next(), "mesh_2");
    }
}
