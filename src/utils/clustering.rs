// src/utils/clustering.rs
use std::collections::HashMap;

/// Highly optimized Disjoint Set Union (Union-Find) using integer indices
pub struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    pub fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
        }
    }

    pub fn find(&mut self, i: usize) -> usize {
        if self.parent[i] == i {
            return i;
        }
        // Path compression for O(1) subsequent lookups
        self.parent[i] = self.find(self.parent[i]);
        self.parent[i]
    }

    pub fn union(&mut self, i: usize, j: usize) {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i != root_j {
            self.parent[root_i] = root_j; // Attach one root to another
        }
    }

    /// Consolidates the flat sets into a grouped HashMap of cluster indices
    pub fn get_groups(mut self) -> HashMap<usize, Vec<usize>> {
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..self.parent.len() {
            let root = self.find(i);
            groups.entry(root).or_default().push(i);
        }
        groups
    }
}
