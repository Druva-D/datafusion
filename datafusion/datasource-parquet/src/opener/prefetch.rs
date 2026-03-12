// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Row group prefetching for parquet scans.
//!
//! Wraps a [`ParquetRecordBatchStream`] to overlap I/O for the next row group
//! with CPU decoding of the current one. Uses the [`next_row_group`] API from
//! arrow-rs which separates async I/O (fetching column chunks) from synchronous
//! decoding (producing `RecordBatch`es).
//!
//! [`next_row_group`]: parquet::arrow::async_reader::ParquetRecordBatchStream::next_row_group

use std::pin::Pin;
use std::task::{Context, Poll};

use arrow::array::RecordBatch;
use datafusion_common::DataFusionError;
use datafusion_common_runtime::SpawnedTask;
use futures::Stream;
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use parquet::arrow::async_reader::{AsyncFileReader, ParquetRecordBatchStream};
use tokio::sync::mpsc;

/// A stream that prefetches row groups from a [`ParquetRecordBatchStream`] in a
/// background task, overlapping I/O for upcoming row groups with CPU decoding of
/// the current row group.
///
/// The background task calls [`ParquetRecordBatchStream::next_row_group`] in a
/// loop, sending each [`ParquetRecordBatchReader`] through a bounded channel.
/// The channel capacity controls how many row groups can be buffered ahead.
///
/// The foreground (stream consumer) reads batches synchronously from the current
/// [`ParquetRecordBatchReader`], and when exhausted, receives the next
/// prefetched reader from the channel.
pub(crate) struct EagerRowGroupPrefetchStream {
    /// Channel receiver for prefetched row group readers.
    receiver: mpsc::Receiver<
        Result<Option<ParquetRecordBatchReader>, parquet::errors::ParquetError>,
    >,
    /// Background task handle. Kept alive to ensure the task runs and is aborted
    /// on drop (SpawnedTask aborts on drop).
    _prefetch_task: SpawnedTask<Result<(), parquet::errors::ParquetError>>,
    /// The currently active reader producing RecordBatches.
    active_reader: Option<ParquetRecordBatchReader>,
}

impl EagerRowGroupPrefetchStream {
    /// Create a new prefetch stream wrapping the given parquet stream.
    ///
    /// `prefetch_depth` controls how many row groups can be buffered ahead of
    /// what the consumer is currently reading. A value of 1 means the next row
    /// group's I/O starts while the current one is being decoded.
    pub fn new<T>(stream: ParquetRecordBatchStream<T>, prefetch_depth: usize) -> Self
    where
        T: AsyncFileReader + Unpin + Send + 'static,
    {
        let (tx, rx) = mpsc::channel(prefetch_depth);
        let task = SpawnedTask::spawn(Self::prefetch_loop(stream, tx));
        Self {
            receiver: rx,
            _prefetch_task: task,
            active_reader: None,
        }
    }

    /// Background task: repeatedly calls `next_row_group()` and sends the
    /// resulting readers through the channel. Stops when the stream is
    /// exhausted, on error, or when the receiver is dropped.
    async fn prefetch_loop<T>(
        mut stream: ParquetRecordBatchStream<T>,
        tx: mpsc::Sender<
            Result<Option<ParquetRecordBatchReader>, parquet::errors::ParquetError>,
        >,
    ) -> Result<(), parquet::errors::ParquetError>
    where
        T: AsyncFileReader + Unpin + Send + 'static,
    {
        loop {
            match stream.next_row_group().await {
                Ok(Some(reader)) => {
                    // Send the prefetched reader. If the receiver is dropped
                    // (consumer stopped reading), stop prefetching.
                    if tx.send(Ok(Some(reader))).await.is_err() {
                        break;
                    }
                }
                Ok(None) => {
                    // End of stream — signal the consumer.
                    let _ = tx.send(Ok(None)).await;
                    break;
                }
                Err(e) => {
                    // Propagate the error to the consumer.
                    let _ = tx.send(Err(e)).await;
                    break;
                }
            }
        }
        Ok(())
    }
}

impl Stream for EagerRowGroupPrefetchStream {
    type Item = datafusion_common::Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        loop {
            // If we have an active reader, produce batches from it.
            if let Some(reader) = &mut self.active_reader {
                match reader.next() {
                    Some(Ok(batch)) => return Poll::Ready(Some(Ok(batch))),
                    Some(Err(e)) => {
                        self.active_reader = None;
                        return Poll::Ready(Some(Err(DataFusionError::from(e))));
                    }
                    None => {
                        // Current row group exhausted — fetch next from channel.
                        self.active_reader = None;
                    }
                }
            }

            // No active reader — poll the channel for the next prefetched one.
            match self.receiver.poll_recv(cx) {
                Poll::Ready(Some(Ok(Some(reader)))) => {
                    self.active_reader = Some(reader);
                    continue; // loop back to read batches from it
                }
                Poll::Ready(Some(Ok(None))) => {
                    // End of stream signal from background task.
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(DataFusionError::from(e))));
                }
                Poll::Ready(None) => {
                    // Channel closed — background task finished or panicked.
                    return Poll::Ready(None);
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use bytes::Bytes;
    use futures::FutureExt;
    use futures::StreamExt;
    use futures::future::BoxFuture;
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
    use parquet::basic::Compression;
    use parquet::errors::Result;
    use parquet::file::metadata::ParquetMetaData;
    use parquet::file::properties::WriterProperties;
    use std::ops::Range;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    const NUM_COLUMNS: usize = 10;

    /// Generate an in-memory parquet file with the specified number of row
    /// groups. Each row group has `rows_per_rg` rows across multiple Int32
    /// columns to make decode work non-trivial.
    fn generate_multi_rg_parquet(num_row_groups: usize, rows_per_rg: usize) -> Bytes {
        let fields: Vec<Field> = (0..NUM_COLUMNS)
            .map(|i| Field::new(format!("c{i}"), DataType::Int32, false))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let props = WriterProperties::builder()
            .set_max_row_group_size(rows_per_rg)
            .set_compression(Compression::SNAPPY)
            .build();
        let mut buf = Vec::new();
        let mut writer =
            ArrowWriter::try_new(&mut buf, Arc::clone(&schema), Some(props)).unwrap();
        for rg in 0..num_row_groups {
            let columns: Vec<Arc<dyn arrow::array::Array>> = (0..NUM_COLUMNS)
                .map(|col| {
                    let values: Vec<i32> = (0..rows_per_rg)
                        .map(|i| {
                            (rg * rows_per_rg * NUM_COLUMNS + col * rows_per_rg + i)
                                as i32
                        })
                        .collect();
                    Arc::new(Int32Array::from(values)) as _
                })
                .collect();
            let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
            writer.write(&batch).unwrap();
        }
        writer.close().unwrap();
        Bytes::from(buf)
    }

    /// An `AsyncFileReader` that wraps in-memory bytes and injects a
    /// configurable delay on every `get_bytes` call, simulating cloud storage
    /// I/O latency.
    struct SlowReader {
        data: Bytes,
        delay: Duration,
        metadata: Option<Arc<ParquetMetaData>>,
    }

    impl SlowReader {
        fn new(data: Bytes, delay: Duration) -> Self {
            Self {
                data,
                delay,
                metadata: None,
            }
        }
    }

    impl AsyncFileReader for SlowReader {
        fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, Result<Bytes>> {
            let data = self.data.slice(range.start as usize..range.end as usize);
            futures::future::ready(Ok(data)).boxed()
        }

        /// Adds a single delay per batch of byte ranges, simulating one
        /// network round-trip to fetch all column chunks for a row group.
        fn get_byte_ranges(
            &mut self,
            ranges: Vec<Range<u64>>,
        ) -> BoxFuture<'_, Result<Vec<Bytes>>> {
            let delay = self.delay;
            let data = self.data.clone();
            async move {
                tokio::time::sleep(delay).await;
                Ok(ranges
                    .into_iter()
                    .map(|r| data.slice(r.start as usize..r.end as usize))
                    .collect())
            }
            .boxed()
        }

        fn get_metadata<'a>(
            &'a mut self,
            _options: Option<&'a parquet::arrow::arrow_reader::ArrowReaderOptions>,
        ) -> BoxFuture<'a, Result<Arc<ParquetMetaData>>> {
            let reader = parquet::file::metadata::ParquetMetaDataReader::new();
            self.metadata = Some(Arc::new(reader.parse_and_finish(&self.data).unwrap()));
            futures::future::ready(Ok(self.metadata.clone().unwrap())).boxed()
        }
    }

    /// Helper: read all batches from a parquet file using the standard
    /// `ParquetRecordBatchStream` (no prefetching — sequential I/O then
    /// decode).
    async fn read_sequential(data: Bytes, delay: Duration) -> (usize, Duration) {
        let reader = SlowReader::new(data, delay);
        let builder = ParquetRecordBatchStreamBuilder::new(reader).await.unwrap();
        let mut stream = builder.build().unwrap();
        let start = Instant::now();
        let mut total_rows = 0usize;
        while let Some(batch) = stream.next().await {
            total_rows += batch.unwrap().num_rows();
        }
        (total_rows, start.elapsed())
    }

    /// Helper: read all batches through `EagerRowGroupPrefetchStream`.
    async fn read_prefetched(
        data: Bytes,
        delay: Duration,
        depth: usize,
    ) -> (usize, Duration) {
        let reader = SlowReader::new(data, delay);
        let builder = ParquetRecordBatchStreamBuilder::new(reader).await.unwrap();
        let stream = builder.build().unwrap();
        let mut prefetch = EagerRowGroupPrefetchStream::new(stream, depth);
        let start = Instant::now();
        let mut total_rows = 0usize;
        while let Some(batch) = prefetch.next().await {
            total_rows += batch.unwrap().num_rows();
        }
        (total_rows, start.elapsed())
    }

    /// Correctness: prefetch stream produces the same number of rows as
    /// sequential reading.
    #[tokio::test]
    async fn prefetch_produces_same_rows() {
        let data = generate_multi_rg_parquet(5, 1000);

        let (seq_rows, _) = read_sequential(data.clone(), Duration::ZERO).await;
        let (pre_rows, _) = read_prefetched(data, Duration::ZERO, 1).await;

        assert_eq!(seq_rows, pre_rows);
        assert_eq!(seq_rows, 5 * 1000);
    }

    /// Correctness: works with a single row group (no overlap possible, but
    /// should not hang or panic).
    #[tokio::test]
    async fn prefetch_single_row_group() {
        let data = generate_multi_rg_parquet(1, 500);
        let (rows, _) = read_prefetched(data, Duration::ZERO, 1).await;
        assert_eq!(rows, 500);
    }

    /// Performance: with simulated I/O latency, prefetch should be measurably
    /// faster than sequential because I/O for the next row group overlaps with
    /// decode of the current one.
    ///
    /// With N row groups, D ms delay per row group, and T ms decode per RG:
    /// - Sequential: N * (D + T) — I/O and decode are serial
    /// - Prefetch(1): D + N*max(D,T) — first RG waits, rest overlap
    ///   When D ≈ T, prefetch ≈ (N+1)/2N * sequential ≈ 50% faster
    ///
    /// We use 100K rows * 10 columns per RG (debug mode decode is slow enough
    /// to overlap with 50ms I/O delay).
    #[tokio::test]
    async fn prefetch_overlaps_io_with_latency() {
        let num_rgs = 6;
        let rows_per_rg = 100_000;
        let io_delay = Duration::from_millis(50);
        let data = generate_multi_rg_parquet(num_rgs, rows_per_rg);

        let (seq_rows, seq_time) = read_sequential(data.clone(), io_delay).await;
        let (pre_rows, pre_time) = read_prefetched(data, io_delay, 1).await;

        assert_eq!(seq_rows, pre_rows);

        let speedup = seq_time.as_secs_f64() / pre_time.as_secs_f64();
        eprintln!(
            "Sequential: {:.1}ms, Prefetch: {:.1}ms, Speedup: {:.2}x",
            seq_time.as_secs_f64() * 1000.0,
            pre_time.as_secs_f64() * 1000.0,
            speedup
        );

        // With 6 RGs * 50ms = 300ms I/O + decode time, prefetch should save
        // most of the I/O wait. Require at least 15% speedup (conservative to
        // avoid flakiness in CI).
        assert!(
            speedup > 1.15,
            "Expected prefetch to be at least 1.15x faster, got {speedup:.2}x \
             (seq={seq_time:?}, pre={pre_time:?})"
        );
    }
}
