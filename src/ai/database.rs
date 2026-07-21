// src/ai/database.rs

use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;

use lancedb::arrow::arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray, UInt64Array,
};
use lancedb::arrow::arrow_schema::{DataType, Field, Schema};

use futures::TryStreamExt;
use lancedb::connection::Connection;
use lancedb::index::Index;
use lancedb::query::ExecutableQuery;
use lancedb::query::QueryBase;
use lancedb::table::Table;

/// Helper function to construct the unified Apache Arrow schema for LanceDB image metadata caching.
fn get_schema(dim: usize) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
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
        Field::new("file_size", DataType::UInt64, false),
        Field::new("mtime", DataType::Int64, false),
    ]))
}

/// Represents a single vector record structured for LanceDB storage.
#[derive(Debug, Clone)]
pub struct DbRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub path: String,
    pub channel: String,
    pub file_size: u64,
    pub mtime: i64,
}

/// Represents the normalized result of a vector similarity query.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub path: String,
    pub channel: String,
    pub distance: f32,
}

/// Service manager handling all embedded database interactions using LanceDB.
#[derive(Clone)]
pub struct DatabaseService {
    pub db: Option<Connection>,
    pub table: Option<Table>,
    pub dim: usize,
}

impl DatabaseService {
    pub fn new() -> Self {
        Self {
            db: None,
            table: None,
            dim: 0,
        }
    }

    /// Connects to (or creates) the local LanceDB workspace and ensures the target table exists.
    pub async fn initialize(
        &mut self,
        db_directory: &Path,
        table_name: &str,
        dim: usize,
    ) -> Result<()> {
        let db_path_str = db_directory.to_string_lossy().to_string();
        let db = lancedb::connect(&db_path_str)
            .execute()
            .await
            .context("Failed to connect to LanceDB")?;

        let schema = get_schema(dim);
        let table_names = db.table_names().execute().await?;

        // Open existing table instead of dropping to preserve previous vector cache
        let table = if table_names.contains(&table_name.to_string()) {
            db.open_table(table_name).execute().await?
        } else {
            let empty_batch = RecordBatch::new_empty(schema.clone());
            db.create_table(table_name, vec![empty_batch])
                .execute()
                .await?
        };

        self.db = Some(db);
        self.table = Some(table);
        self.dim = dim;

        Ok(())
    }

    /// Converts database cache entries into Apache Arrow RecordBatches and appends them to the table.
    pub async fn add_batch(&self, records: &[DbRecord]) -> Result<()> {
        let table = self
            .table
            .as_ref()
            .context("Database table is not initialized")?;
        if records.is_empty() {
            return Ok(());
        }

        let schema = get_schema(self.dim);

        let ids: Vec<&str> = records.iter().map(|r| r.id.as_str()).collect();
        let id_array = Arc::new(StringArray::from(ids)) as ArrayRef;

        let paths: Vec<&str> = records.iter().map(|r| r.path.as_str()).collect();
        let path_array = Arc::new(StringArray::from(paths)) as ArrayRef;

        let channels: Vec<&str> = records.iter().map(|r| r.channel.as_str()).collect();
        let channel_array = Arc::new(StringArray::from(channels)) as ArrayRef;

        let file_sizes: Vec<u64> = records.iter().map(|r| r.file_size).collect();
        let file_size_array = Arc::new(UInt64Array::from(file_sizes)) as ArrayRef;

        let mtimes: Vec<i64> = records.iter().map(|r| r.mtime).collect();
        let mtime_array = Arc::new(Int64Array::from(mtimes)) as ArrayRef;

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
            vec![
                id_array,
                vector_array,
                path_array,
                channel_array,
                file_size_array,
                mtime_array,
            ],
        )?;
        table.add(vec![batch]).execute().await?;
        Ok(())
    }

    /// Fetches all cached metadata from the database dynamically mapping columns to prevent schema drift errors.
    pub async fn load_cache_metadata(&self) -> Result<Vec<(String, String, u64, i64, Vec<f32>)>> {
        let table = self.table.as_ref().context("Table not initialized")?;
        let query_result = table
            .query()
            .select(lancedb::query::Select::Columns(vec![
                "path".to_string(),
                "channel".to_string(),
                "file_size".to_string(),
                "mtime".to_string(),
                "vector".to_string(),
            ]))
            .execute()
            .await?;

        let record_batches = query_result.try_collect::<Vec<RecordBatch>>().await?;
        let mut cache_entries = Vec::new();

        for batch in record_batches {
            if batch.num_rows() == 0 {
                continue;
            }

            // Resolve column indices dynamically to guarantee stability against future schema refactoring
            let schema = batch.schema();
            let path_idx = schema.index_of("path").context("Missing path column")?;
            let channel_idx = schema
                .index_of("channel")
                .context("Missing channel column")?;
            let size_idx = schema
                .index_of("file_size")
                .context("Missing file_size column")?;
            let mtime_idx = schema.index_of("mtime").context("Missing mtime column")?;
            let vector_idx = schema.index_of("vector").context("Missing vector column")?;

            let paths_array = batch
                .column(path_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Failed to downcast path column")?;
            let channels_array = batch
                .column(channel_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Failed to downcast channel column")?;
            let sizes_array = batch
                .column(size_idx)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .context("Failed to downcast file_size column")?;
            let mtimes_array = batch
                .column(mtime_idx)
                .as_any()
                .downcast_ref::<Int64Array>()
                .context("Failed to downcast mtime column")?;
            let vectors_array = batch
                .column(vector_idx)
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .context("Failed to downcast vector column")?;

            for i in 0..batch.num_rows() {
                let path = paths_array.value(i).to_string();
                let channel = channels_array.value(i).to_string();
                let size = sizes_array.value(i);
                let mtime = mtimes_array.value(i);

                let list_val = vectors_array.value(i);
                let float_arr = list_val
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .context("Failed to extract floats from list cell")?;
                let vec: Vec<f32> = (0..float_arr.len())
                    .map(|idx| float_arr.value(idx))
                    .collect();

                cache_entries.push((path, channel, size, mtime, vec));
            }
        }
        Ok(cache_entries)
    }

    /// Deletes records from LanceDB matching a custom SQL-like filter string.
    pub async fn delete(&self, filter: &str) -> Result<()> {
        let table = self.table.as_ref().context("Table not initialized")?;
        table.delete(filter).await?;
        Ok(())
    }

    /// Generates an IVF-PQ vector index to accelerate nearest-neighbor lookups.
    pub async fn create_vector_index(&self) -> Result<()> {
        let table = self
            .table
            .as_ref()
            .context("Database table is not initialized")?;
        let row_count = table.count_rows(None).await?;

        // Avoid building indices on tiny datasets since exhaustive scan is faster under 5000 records
        if row_count < 5000 {
            return Ok(());
        }

        table
            .create_index(&["vector"], Index::Auto)
            .execute()
            .await?;
        Ok(())
    }

    /// Performs a high-speed vector similarity search using Cosine metrics.
    pub async fn search_similarity(
        &self,
        query_vector: &[f32],
        limit: usize,
        nprobes: Option<usize>,
        refine_factor: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        let table = self
            .table
            .as_ref()
            .context("Database table is not initialized")?;

        let mut query = table.query().nearest_to(query_vector)?;

        if let Some(np) = nprobes {
            query = query.nprobes(np);
        }
        if let Some(rf) = refine_factor {
            query = query.refine_factor(rf as u32);
        }

        let query_result = query.limit(limit).execute().await?;
        let record_batches = query_result.try_collect::<Vec<RecordBatch>>().await?;
        let mut results = Vec::new();

        for batch in record_batches {
            if batch.num_rows() == 0 {
                continue;
            }

            let schema = batch.schema();
            let path_idx = schema
                .index_of("path")
                .context("Missing path column in search results")?;
            let channel_idx = schema
                .index_of("channel")
                .context("Missing channel column in search results")?;

            let paths_array = batch
                .column(path_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Failed to downcast path column")?;

            let channels_array = batch
                .column(channel_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Failed to downcast channel column")?;

            let distance_col_idx = batch
                .schema()
                .index_of("_distance")
                .unwrap_or_else(|_| batch.num_columns() - 1);
            let distances_array = batch
                .column(distance_col_idx)
                .as_any()
                .downcast_ref::<Float32Array>()
                .context("Failed to downcast distance column")?;

            for i in 0..batch.num_rows() {
                results.push(SearchResult {
                    path: paths_array.value(i).to_string(),
                    channel: channels_array.value(i).to_string(),
                    distance: distances_array.value(i),
                });
            }
        }
        Ok(results)
    }
}
