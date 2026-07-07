// src/state/mod.rs
pub mod models;
pub mod store;

// Re-export everything so other files can just do `use crate::state::AppState;`
pub use models::*;
pub use store::*;
