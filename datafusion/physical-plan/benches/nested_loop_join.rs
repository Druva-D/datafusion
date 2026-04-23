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

//! Benchmark for Nested Loop Join with filter.
//!
//! Measures end-to-end NLJ execution with non-equi join filters.
//! The scalar-aware filter optimization is active by default and
//! avoids broadcasting build-side scalars into full arrays.

use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use datafusion_common::JoinSide;
use datafusion_execution::TaskContext;
use datafusion_expr::{JoinType, Operator};
use datafusion_physical_expr::expressions::{BinaryExpr, Column};
use datafusion_physical_plan::collect;
use datafusion_physical_plan::joins::NestedLoopJoinExec;
use datafusion_physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use datafusion_physical_plan::test::TestMemoryExec;
use tokio::runtime::Runtime;

/// Build a batch with a string payload column to emphasize broadcast cost.
fn build_string_batch(num_rows: usize, string_len: usize, prefix: &str) -> RecordBatch {
    let ids: Vec<i32> = (0..num_rows as i32).collect();
    let strings: Vec<String> = (0..num_rows)
        .map(|i| format!("{prefix}_{i:0>string_len$}"))
        .collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(strings)),
        ],
    )
    .unwrap()
}

fn build_int_batch(num_rows: usize, offset: i32) -> RecordBatch {
    let a: Vec<i32> = (0..num_rows as i32).map(|i| i + offset).collect();
    let b: Vec<i32> = (0..num_rows as i32).map(|i| i * 2 + offset).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(a)), Arc::new(Int32Array::from(b))],
    )
    .unwrap()
}

/// Build NLJ with string filter: left.val < right.val
fn build_string_nlj(
    left_batch: RecordBatch,
    right_batch: RecordBatch,
) -> Arc<NestedLoopJoinExec> {
    let left_schema = left_batch.schema();
    let right_schema = right_batch.schema();

    let left = Arc::new(
        TestMemoryExec::try_new(&[vec![left_batch]], left_schema, None).unwrap(),
    );
    let right = Arc::new(
        TestMemoryExec::try_new(&[vec![right_batch]], right_schema, None).unwrap(),
    );

    // Filter: left.val < right.val
    let column_indices = vec![
        ColumnIndex {
            index: 1,
            side: JoinSide::Left,
        },
        ColumnIndex {
            index: 1,
            side: JoinSide::Right,
        },
    ];
    let filter_schema = Schema::new(vec![
        Field::new("l_val", DataType::Utf8, false),
        Field::new("r_val", DataType::Utf8, false),
    ]);
    let expression = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("l_val", 0)),
        Operator::Lt,
        Arc::new(Column::new("r_val", 1)),
    ));
    let filter = JoinFilter::new(expression, column_indices, Arc::new(filter_schema));

    Arc::new(
        NestedLoopJoinExec::try_new(left, right, Some(filter), &JoinType::Inner, None)
            .unwrap(),
    )
}

/// Build NLJ with int filter: left.a > right.a AND left.b < right.b
fn build_int_nlj(
    left_batch: RecordBatch,
    right_batch: RecordBatch,
) -> Arc<NestedLoopJoinExec> {
    let left_schema = left_batch.schema();
    let right_schema = right_batch.schema();

    let left = Arc::new(
        TestMemoryExec::try_new(&[vec![left_batch]], left_schema, None).unwrap(),
    );
    let right = Arc::new(
        TestMemoryExec::try_new(&[vec![right_batch]], right_schema, None).unwrap(),
    );

    let column_indices = vec![
        ColumnIndex {
            index: 0,
            side: JoinSide::Left,
        },
        ColumnIndex {
            index: 0,
            side: JoinSide::Right,
        },
        ColumnIndex {
            index: 1,
            side: JoinSide::Left,
        },
        ColumnIndex {
            index: 1,
            side: JoinSide::Right,
        },
    ];
    let filter_schema = Schema::new(vec![
        Field::new("l_a", DataType::Int32, false),
        Field::new("r_a", DataType::Int32, false),
        Field::new("l_b", DataType::Int32, false),
        Field::new("r_b", DataType::Int32, false),
    ]);
    let cond1 = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("l_a", 0)),
        Operator::Gt,
        Arc::new(Column::new("r_a", 1)),
    ));
    let cond2 = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("l_b", 2)),
        Operator::Lt,
        Arc::new(Column::new("r_b", 3)),
    ));
    let expression = Arc::new(BinaryExpr::new(cond1, Operator::And, cond2));
    let filter = JoinFilter::new(expression, column_indices, Arc::new(filter_schema));

    Arc::new(
        NestedLoopJoinExec::try_new(left, right, Some(filter), &JoinType::Inner, None)
            .unwrap(),
    )
}

fn bench_nlj(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let ctx = Arc::new(TaskContext::default());
    let mut group = c.benchmark_group("nested_loop_join");

    // String comparison: 50 left rows x N right rows
    // This is where scalar-aware filter has the largest impact
    for probe_rows in [1000, 4096] {
        let left = build_string_batch(50, 128, "left");
        let right = build_string_batch(probe_rows, 128, "right");

        group.bench_with_input(
            BenchmarkId::new("string_filter", probe_rows),
            &probe_rows,
            |b, _| {
                b.iter(|| {
                    let nlj = build_string_nlj(left.clone(), right.clone());
                    rt.block_on(async { collect(nlj, Arc::clone(&ctx)).await.unwrap() });
                });
            },
        );
    }

    // Integer comparison: 100 left rows x N right rows
    for probe_rows in [1000, 4096] {
        let left = build_int_batch(100, 0);
        let right = build_int_batch(probe_rows, 50);

        group.bench_with_input(
            BenchmarkId::new("int_filter", probe_rows),
            &probe_rows,
            |b, _| {
                b.iter(|| {
                    let nlj = build_int_nlj(left.clone(), right.clone());
                    rt.block_on(async { collect(nlj, Arc::clone(&ctx)).await.unwrap() });
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_nlj);
criterion_main!(benches);
