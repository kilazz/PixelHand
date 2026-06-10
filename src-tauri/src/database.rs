// src-tauri/src/database.rs
use std::error::Error;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::connection::Connection;
use lancedb::index::Index;
use lancedb::query::ExecutableQuery;
use lancedb::query::QueryBase;
use lancedb::table::Table;

/// Represents a single vector record structured for LanceDB storage.
#[derive(Debug, Clone)]
pub struct DbRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub path: String,
    pub channel: String,
}

/// Represents the normalized result of a vector similarity query.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub path: String,
    pub distance: f32,
}

/// Service manager handling all embedded database interactions using LanceDB.
pub struct DatabaseService {
    pub db: Option<Connection>,
    pub table: Option<Table>,
    pub dim: usize,
}

impl DatabaseService {
    /// Creates a new, uninitialized DatabaseService instance.
    pub fn new() -> Self {
        Self {
            db: None,
            table: None,
            dim: 0,
        }
    }

    /// Connects to (or creates) the local LanceDB database and ensures the target table exists.
    ///
    /// Fixes compilation issues with LanceDB v0.30 by providing the required namespace
    /// argument for `drop_table`.
    pub async fn initialize(
        &mut self,
        db_directory: &Path,
        table_name: &str,
        dim: usize,
    ) -> Result<(), Box<dyn Error>> {
        let db_path_str = db_directory.to_string_lossy().to_string();
        let db = lancedb::connect(&db_path_str).execute().await?;

        // Define the rigid Apache Arrow schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
            Field::new("path", DataType::Utf8, false),
            Field::new("channel", DataType::Utf8, false),
        ]));

        let table_names = db.table_names().execute().await?;

        // FIX E0061: Provide the required empty namespace path argument for drop_table in LanceDB v0.30
        if table_names.contains(&table_name.to_string()) {
            let empty_ns: Vec<String> = Vec::new();
            db.drop_table(table_name, &empty_ns).await?;
        }

        // Create a fresh table with the defined schema to start a clean scanning session
        let empty_batch = RecordBatch::new_empty(schema.clone());
        let table = db
            .create_table(table_name, vec![empty_batch])
            .execute()
            .await?;

        self.db = Some(db);
        self.table = Some(table);
        self.dim = dim;

        Ok(())
    }

    /// Converts records into Apache Arrow RecordBatches and appends them to the LanceDB table.
    pub async fn add_batch(&self, records: &[DbRecord]) -> Result<(), Box<dyn Error>> {
        let table = self
            .table
            .as_ref()
            .ok_or("Database table is not initialized")?;
        if records.is_empty() {
            return Ok(());
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.dim as i32,
                ),
                false,
            ),
            Field::new("path", DataType::Utf8, false),
            Field::new("channel", DataType::Utf8, false),
        ]));

        // Build Arrow columnar arrays from the flat DbRecords
        let ids: Vec<&str> = records.iter().map(|r| r.id.as_str()).collect();
        let id_array = Arc::new(StringArray::from(ids)) as ArrayRef;

        let paths: Vec<&str> = records.iter().map(|r| r.path.as_str()).collect();
        let path_array = Arc::new(StringArray::from(paths)) as ArrayRef;

        let channels: Vec<&str> = records.iter().map(|r| r.channel.as_str()).collect();
        let channel_array = Arc::new(StringArray::from(channels)) as ArrayRef;

        // Flatten vectors to load them into the FixedSizeList array layout
        let mut flat_vectors = Vec::with_capacity(records.len() * self.dim);
        for r in records {
            flat_vectors.extend_from_slice(&r.vector);
        }
        let values_array = Float32Array::from(flat_vectors);

        let value_field = Arc::new(Field::new("item", DataType::Float32, true));
        let vector_array = Arc::new(FixedSizeListArray::try_new(
            value_field,
            self.dim as i32,
            Arc::new(values_array),
            None,
        )?) as ArrayRef;

        let batch = RecordBatch::try_new(
            schema,
            vec![id_array, vector_array, path_array, channel_array],
        )?;

        table.add(vec![batch]).execute().await?;
        Ok(())
    }

    /// Generates an IVF-PQ vector index on the `vector` column to accelerate nearest-neighbor lookups.
    /// Safely skips index creation if there are fewer than 256 rows to avoid training errors.
    pub async fn create_vector_index(&self) -> Result<(), Box<dyn Error>> {
        let table = self
            .table
            .as_ref()
            .ok_or("Database table is not initialized")?;
        let row_count = table.count_rows(None).await?;
        if row_count < 256 {
            return Ok(());
        }

        table
            .create_index(&["vector"], Index::Auto)
            .execute()
            .await?;
        Ok(())
    }

    /// Performs a high-speed vector similarity search using cosine metrics.
    pub async fn search_similarity(
        &self,
        query_vector: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchResult>, Box<dyn Error>> {
        let table = self
            .table
            .as_ref()
            .ok_or("Database table is not initialized")?;
        let query_result = table
            .query()
            .nearest_to(query_vector)?
            .limit(limit)
            .execute()
            .await?;
        let record_batches = query_result.try_collect::<Vec<RecordBatch>>().await?;
        let mut results = Vec::new();

        for batch in record_batches {
            if batch.num_rows() == 0 {
                continue;
            }

            // Downcast columns back to Arrow native array representations
            let paths_array = batch
                .column(2)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or("Failed to downcast path")?;

            // Retrieve the dynamic `_distance` column appended by LanceDB
            let distance_col_idx = batch
                .schema()
                .index_of("_distance")
                .unwrap_or(batch.num_columns() - 1);
            let distances_array = batch
                .column(distance_col_idx)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or("Failed to downcast distance")?;

            for i in 0..batch.num_rows() {
                results.push(SearchResult {
                    path: paths_array.value(i).to_string(),
                    distance: distances_array.value(i),
                });
            }
        }
        Ok(results)
    }
}
