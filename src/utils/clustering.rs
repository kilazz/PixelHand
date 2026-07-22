// src/utils/clustering.rs

use std::collections::HashMap;

/// Disjoint Set Union (Union-Find) data structure implementing Path Compression
/// and Union by Rank to provide near-constant amortized time complexity.
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    pub fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    /// Finds the representative root of the set containing element `i` recursively.
    /// Performs Path Compression along the traversal to flatten the tree structure.
    pub fn find(&mut self, i: usize) -> usize {
        if self.parent[i] == i {
            return i;
        }
        // Path compression step
        self.parent[i] = self.find(self.parent[i]);
        self.parent[i]
    }

    /// Merges the set containing element `i` with the set containing element `j`.
    /// Utilizes Union by Rank to attach the shallower tree under the root of the deeper tree.
    pub fn union(&mut self, i: usize, j: usize) {
        let root_i = self.find(i);
        let root_j = self.find(j);

        if root_i != root_j {
            // Attach the smaller rank tree under the root of the larger rank tree
            match self.rank[root_i].cmp(&self.rank[root_j]) {
                std::cmp::Ordering::Less => {
                    self.parent[root_i] = root_j;
                }
                std::cmp::Ordering::Greater => {
                    self.parent[root_j] = root_i;
                }
                std::cmp::Ordering::Equal => {
                    self.parent[root_i] = root_j;
                    self.rank[root_j] += 1;
                }
            }
        }
    }

    /// Consolidates disjoint sets into a mapped collection of root indices to their member indices.
    pub fn get_groups(mut self) -> HashMap<usize, Vec<usize>> {
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..self.parent.len() {
            let root = self.find(i);
            groups.entry(root).or_default().push(i);
        }
        groups
    }
}
