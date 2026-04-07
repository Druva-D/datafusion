use arrow::array::{Array, AsArray, BooleanArray, PrimitiveArray, RecordBatch};
use arrow::buffer::BooleanBuffer;
use arrow::compute::{filter_record_batch, take};
use arrow::datatypes::{Int64Type, Schema, SchemaRef};
use arrow::downcast_dictionary_array;
use datafusion_common::{DataFusionError, Result, exec_datafusion_err};
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::Column;
use futures::{Stream, StreamExt};
use itertools::Itertools;

use std::any::Any;
use std::fmt::Display;
use std::hash::Hash;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll, ready};

#[derive(Debug, Clone)]
pub struct DeletionVectorHolder {
    pub offsets: Vec<u64>,
    filter_expr: OnceLock<Result<Arc<dyn PhysicalExpr>, Arc<DataFusionError>>>,
}

impl DeletionVectorHolder {
    pub fn try_new(offsets: Vec<u64>) -> Result<Self> {
        Ok(Self {
            offsets,
            filter_expr: OnceLock::new(),
        })
    }

    pub fn filter_expr(&self, schema: SchemaRef) -> Result<Arc<dyn PhysicalExpr>> {
        self.filter_expr
            .get_or_init(|| {
                DeletionVectorFilter::try_new(self.offsets.clone(), &schema)
                    .map(|f| Arc::new(f) as Arc<dyn PhysicalExpr>)
                    .map_err(|e| Arc::new(e))
            })
            .clone()
            .map_err(|e| DataFusionError::Shared(e))
    }
}

#[derive(Debug, Eq)]
pub struct DeletionVectorFilter {
    deleted_rows_sorted: Vec<i64>,
    column_expr: Arc<dyn PhysicalExpr>,
}

impl DeletionVectorFilter {
    pub fn try_new(offsets: Vec<u64>, schema: &Schema) -> Result<Self> {
        Ok(Self {
            deleted_rows_sorted: offsets.into_iter().map(|o| o as i64).sorted().collect(),
            column_expr: Arc::new(Column::new_with_schema("row_number", schema)?),
        })
    }

    /// Optimized search for deleted rows.
    ///
    /// Makes use of the fact that we have sorted `row_number`s in `deleted_rows_sorted` and that
    /// `row_number` is contiguous and non-decreasing within a record batch.
    ///
    /// Uses binary search to first do a fast-path check if the array can even contain deleted rows.
    /// This operator also gives a subset of the deleted rows in which the array can potentially
    /// have matches. Then we do a binary search for each element of the array.
    fn filter_contiguous_batch(
        &self,
        array: &PrimitiveArray<Int64Type>,
    ) -> BooleanBuffer {
        let array_values = array.values();

        let row_number_start = *array_values
            .first()
            .expect("row_number column did not have first value");
        let row_number_end = *array_values
            .last()
            .expect("row_number column did not have last value");

        // Binary search to find the slice of deleted rows in [batch_start, batch_end]
        let lo = self
            .deleted_rows_sorted
            .partition_point(|&x| x < row_number_start);
        let hi = self
            .deleted_rows_sorted
            .partition_point(|&x| x <= row_number_end);
        let relevant = &self.deleted_rows_sorted[lo..hi];

        let batch_len = array.len();

        // Fast path: no rows in this batch deleted, so return all true.
        if relevant.is_empty() {
            return BooleanBuffer::new_set(batch_len);
        }

        // Linear cursor merge: both array_values and relevant are sorted,
        // so we walk a single cursor forward through relevant. O(n + d).
        let mut del_idx = 0;
        BooleanBuffer::collect_bool(batch_len, |i| {
            let row = array_values[i];
            while del_idx < relevant.len() && relevant[del_idx] < row {
                del_idx += 1;
            }
            del_idx >= relevant.len() || relevant[del_idx] != row
        })
    }

    fn contains(&self, array: &dyn Array) -> Result<BooleanArray> {
        let array = array.as_primitive_opt::<Int64Type>().ok_or_else(|| {
            exec_datafusion_err!(
                "Failed to downcast deletion vector array to a 'int64' array. Actual type is {}", array.data_type()
            )
        })?;

        let contains_buffer = self.filter_contiguous_batch(array);

        Ok(BooleanArray::new(contains_buffer, None))
    }
}

impl Hash for DeletionVectorFilter {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.deleted_rows_sorted.hash(state);
        self.column_expr.hash(state);
    }
}

impl PartialEq for DeletionVectorFilter {
    fn eq(&self, other: &Self) -> bool {
        self.deleted_rows_sorted == other.deleted_rows_sorted
            && &self.column_expr == &other.column_expr
    }
}

impl Display for DeletionVectorFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const VALUES_TO_SHOW: usize = 10;

        let mut values = self
            .deleted_rows_sorted
            .iter()
            .take(VALUES_TO_SHOW)
            .map(|num| num.to_string())
            .join(", ");

        if values.len() > VALUES_TO_SHOW {
            values.push_str(&format!("...+{}", values.len() - VALUES_TO_SHOW));
        }

        write!(f, "DeletionVector({}) IN ({values})", self.column_expr)
    }
}

impl PhysicalExpr for DeletionVectorFilter {
    fn as_any(&self) -> &dyn Any {
        self as &dyn Any
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let value = self.column_expr.evaluate(batch)?;
        match value {
            ColumnarValue::Array(array) => {
                let array = array.as_ref();
                // Handle dictionary arrays by recursing on the values
                downcast_dictionary_array! {
                    array => {
                        let values_contains = self.contains(array.values().as_ref())?;
                        let result = take(&values_contains, array.keys(), None)?;
                        return Ok(ColumnarValue::Array(result));
                    }
                    _ => {}
                }

                Ok(ColumnarValue::Array(Arc::new(self.contains(array)?)))
            }
            ColumnarValue::Scalar(scalar_value) => {
                let array = scalar_value.to_array()?;
                Ok(ColumnarValue::Array(Arc::new(self.contains(&array)?)))
            }
        }
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.column_expr]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(Self {
            column_expr: Arc::clone(&children[0]),
            deleted_rows_sorted: self.deleted_rows_sorted.clone(),
        }))
    }

    fn fmt_sql(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.column_expr.fmt_sql(f)?;

        write!(f, " IN (")?;
        for (i, value) in self.deleted_rows_sorted.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", value)?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;

    fn make_filter(offsets: Vec<u64>) -> DeletionVectorFilter {
        let schema = Schema::new(vec![arrow::datatypes::Field::new(
            "row_number",
            arrow::datatypes::DataType::Int64,
            false,
        )]);
        DeletionVectorFilter::try_new(offsets, &schema).unwrap()
    }

    fn kept_rows(filter: &DeletionVectorFilter, start: i64, len: usize) -> Vec<i64> {
        let values: Vec<i64> = (start..start + len as i64).collect();
        let array = Int64Array::from(values.clone());
        let mask = filter.filter_contiguous_batch(&array);
        values
            .into_iter()
            .enumerate()
            .filter(|(i, _)| mask.value(*i))
            .map(|(_, v)| v)
            .collect()
    }

    #[test]
    fn test_no_deletions_in_range() {
        let filter = make_filter(vec![0, 1, 2, 100, 200]);
        // Batch rows 10..15, no deletions overlap
        assert_eq!(kept_rows(&filter, 10, 5), vec![10, 11, 12, 13, 14]);
    }

    #[test]
    fn test_partial_deletions() {
        let filter = make_filter(vec![5, 11, 13, 20]);
        assert_eq!(kept_rows(&filter, 10, 5), vec![10, 12, 14]);
    }

    #[test]
    fn test_all_rows_deleted() {
        let filter = make_filter(vec![10, 11, 12, 13, 14]);
        assert!(kept_rows(&filter, 10, 5).is_empty());
    }

    #[test]
    fn test_last_row_deleted() {
        let filter = make_filter(vec![14]);
        assert_eq!(kept_rows(&filter, 10, 5), vec![10, 11, 12, 13]);
    }

    #[test]
    fn test_unsorted_offsets() {
        let filter = make_filter(vec![14, 10, 12]);
        assert_eq!(kept_rows(&filter, 10, 5), vec![11, 13]);
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
