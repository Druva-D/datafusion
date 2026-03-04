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
