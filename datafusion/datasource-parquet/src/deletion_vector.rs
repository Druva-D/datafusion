use arrow::array::{BooleanArray, RecordBatch};
use arrow::compute::filter_record_batch;
use arrow::datatypes::SchemaRef;
use datafusion_common::{DataFusionError, Result, ScalarValue};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{Column, Literal, in_list};
use futures::{Stream, StreamExt};

use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll, ready};

#[derive(Debug, Clone)]
pub struct DeletionVectorHolder {
    pub offsets: Vec<u64>,
    filter_expr: OnceLock<Arc<dyn PhysicalExpr>>,
}

impl DeletionVectorHolder {
    pub fn try_new(offsets: Vec<u64>) -> Result<Self> {
        Ok(Self {
            offsets,
            filter_expr: OnceLock::new(),
        })
    }

    pub fn filter_expr(&self, schema: SchemaRef) -> Result<Arc<dyn PhysicalExpr>> {
        match self.filter_expr.get() {
            Some(filter_expr) => Ok(Arc::clone(&filter_expr)),
            None => {
                let expr: Arc<dyn PhysicalExpr> =
                    Arc::new(Column::new_with_schema("row_number", &schema)?);
                let dv_offsets: Vec<Arc<dyn PhysicalExpr>> = self
                    .offsets
                    .iter()
                    .map(|offset| {
                        Arc::new(Literal::new(ScalarValue::Int64(Some(*offset as i64))))
                            as _
                    })
                    .collect();

                let filter_expr = in_list(expr, dv_offsets, &true, &schema)?;
                self.filter_expr
                    .set(Arc::clone(&filter_expr))
                    .map_err(|_| {
                        DataFusionError::Internal(
                            "Unable to set DV filter expr".to_string(),
                        )
                    })?;
                Ok(filter_expr)
            }
        }
    }
}

/// Wraps an RecordBatchStream with DV
pub struct DVWrappedStream<S> {
    done: bool,
    deletion_vector_holder: Arc<DeletionVectorHolder>,
    inner: S,
}

impl<S> DVWrappedStream<S> {
    pub fn new(stream: S, deletion_vector_holder: Arc<DeletionVectorHolder>) -> Self {
        Self {
            done: false,
            inner: stream,
            deletion_vector_holder,
        }
    }
}

impl<S> DVWrappedStream<S>
where
    S: Stream<Item = Result<RecordBatch>> + Unpin,
{
    fn apply_dv(&mut self, input: Result<RecordBatch>) -> Result<Option<RecordBatch>> {
        let batch = input?;
        let schema = batch.schema();

        let predicate_result = self
            .deletion_vector_holder
            .filter_expr(schema)?
            .evaluate(&batch)?;
        let boolean_array = predicate_result.into_array(batch.num_rows())?;

        let mask = boolean_array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();

        Some(
            filter_record_batch(&batch, mask)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None)),
        )
        .transpose()
    }

    pub fn schema(&self) -> &SchemaRef {
        todo!("")
    }
}

impl<S> Stream for DVWrappedStream<S>
where
    S: Stream<Item = Result<RecordBatch>> + Unpin,
{
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        match ready!(self.inner.poll_next_unpin(cx)) {
            None => {
                self.done = true;
                Poll::Ready(None)
            }
            Some(input_batch) => {
                let output = self.apply_dv(input_batch);
                Poll::Ready(output.transpose())
            }
        }
    }
}
