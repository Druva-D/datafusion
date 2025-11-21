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

use datafusion_physical_plan::{DisplayAs, DisplayFormatType};
use std::collections::HashSet;

use crate::file_groups::FileGroup;
use std::fmt::{Debug, Formatter, Result as FmtResult};

/// A wrapper to customize partitioned file display
///
/// Prints in the format:
/// ```text
/// {NUM_GROUPS groups: [[file1, file2,...], [fileN, fileM, ...], ...]}
/// ```
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct FileGroupsDisplay<'a> {
    pub(crate) groups: &'a [FileGroup],
    pub(crate) show_summary: bool,
}

impl<'a> FileGroupsDisplay<'a> {
    pub fn new(groups: &'a [FileGroup], show_summary: bool) -> Self {
        Self {
            groups,
            show_summary,
        }
    }
}

impl DisplayAs for FileGroupsDisplay<'_> {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> FmtResult {
        let n_groups = self.groups.len();
        let groups = if n_groups == 1 { "group" } else { "groups" };
        write!(f, "{{{n_groups} {groups}: [")?;
        match t {
            DisplayFormatType::Default => {
                if self.show_summary {
                    // Summary mode: show counts only
                    let mut file_set = HashSet::new();
                    let mut num_partitioned_files: usize = 0;
                    for fg in self.groups {
                        num_partitioned_files = num_partitioned_files + fg.len();
                        for pf in fg.files() {
                            file_set.insert(&pf.object_meta.location);
                        }
                    }
                    write!(
                        f,
                        "num_files:{}, num_partitioned_files:{}",
                        file_set.len(),
                        num_partitioned_files
                    )?;
                } else {
                    // Detailed mode: list files (up to max_groups)
                    let max_groups = 5;
                    fmt_up_to_n_elements(self.groups, max_groups, f, |group, f| {
                        FileGroupDisplay(group).fmt_as(t, f)
                    })?;
                }
            }
            DisplayFormatType::TreeRender => {
                // To avoid showing too many partitions
                let max_groups = 5;
                fmt_up_to_n_elements(self.groups, max_groups, f, |group, f| {
                    FileGroupDisplay(group).fmt_as(t, f)
                })?;
            }
            DisplayFormatType::Verbose => {
                fmt_elements_split_by_commas(self.groups.iter(), f, |group, f| {
                    FileGroupDisplay(group).fmt_as(t, f)
                })?
            }
        }
        write!(f, "]}}")
    }
}

/// A wrapper to customize partitioned group of files display
///
/// Prints in the format:
/// ```text
/// [file1, file2,...]
/// ```
#[derive(Debug)]
pub struct FileGroupDisplay<'a>(pub &'a FileGroup);

impl DisplayAs for FileGroupDisplay<'_> {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> FmtResult {
        write!(f, "[")?;
        match t {
            DisplayFormatType::Default | DisplayFormatType::TreeRender => {
                // To avoid showing too many files
                let max_files = 5;
                fmt_up_to_n_elements(self.0.files(), max_files, f, |pf, f| {
                    write!(f, "{}", pf.object_meta.location.as_ref())?;
                    if let Some(range) = pf.range.as_ref() {
                        write!(f, ":{}..{}", range.start, range.end)?;
                    }
                    Ok(())
                })?
            }
            DisplayFormatType::Verbose => {
                fmt_elements_split_by_commas(self.0.iter(), f, |pf, f| {
                    write!(f, "{}", pf.object_meta.location.as_ref())?;
                    if let Some(range) = pf.range.as_ref() {
                        write!(f, ":{}..{}", range.start, range.end)?;
                    }
                    Ok(())
                })?
            }
        }
        write!(f, "]")
    }
}

/// helper to format an array of up to N elements
fn fmt_up_to_n_elements<E, F>(
    elements: &[E],
    n: usize,
    f: &mut Formatter,
    format_element: F,
) -> FmtResult
where
    F: Fn(&E, &mut Formatter) -> FmtResult,
{
    let len = elements.len();
    fmt_elements_split_by_commas(elements.iter().take(n), f, |element, f| {
        format_element(element, f)
    })?;
    // Remaining elements are showed as `...` (to indicate there is more)
    if len > n {
        write!(f, ", ...")?;
    }
    Ok(())
}

/// helper formatting array elements with a comma and a space between them
fn fmt_elements_split_by_commas<E, I, F>(
    iter: I,
    f: &mut Formatter,
    format_element: F,
) -> FmtResult
where
    I: Iterator<Item = E>,
    F: Fn(E, &mut Formatter) -> FmtResult,
{
    for (idx, element) in iter.enumerate() {
        if idx > 0 {
            write!(f, ", ")?;
        }
        format_element(element, f)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion_physical_plan::{DefaultDisplay, VerboseDisplay};
    use object_store::{ObjectMeta, path::Path};

    use crate::PartitionedFile;
    use chrono::Utc;

    #[test]
    fn file_groups_display_empty_summary() {
        // Summary mode (show_summary: true) - shows counts
        let expected = "{0 groups: [num_files:0, num_partitioned_files:0]}";
        assert_eq!(
            DefaultDisplay(FileGroupsDisplay::new(&[], true)).to_string(),
            expected
        );
    }

    #[test]
    fn file_groups_display_empty_detailed() {
        // Detailed mode (show_summary: false) - shows file list
        let expected = "{0 groups: []}";
        assert_eq!(
            DefaultDisplay(FileGroupsDisplay::new(&[], false)).to_string(),
            expected
        );
    }

    #[test]
    fn file_groups_display_one_summary() {
        let files = [FileGroup::new(vec![
            partitioned_file("foo"),
            partitioned_file("bar"),
        ])];

        // Summary mode shows counts
        let expected = "{1 group: [num_files:2, num_partitioned_files:2]}";
        assert_eq!(
            DefaultDisplay(FileGroupsDisplay::new(&files, true)).to_string(),
            expected
        );
    }

    #[test]
    fn file_groups_display_one_detailed() {
        let files = [FileGroup::new(vec![
            partitioned_file("foo"),
            partitioned_file("bar"),
        ])];

        // Detailed mode shows file list
        let expected = "{1 group: [[foo, bar]]}";
        assert_eq!(
            DefaultDisplay(FileGroupsDisplay::new(&files, false)).to_string(),
            expected
        );
    }

    #[test]
    fn file_groups_display_many_summary() {
        let files = [
            FileGroup::new(vec![partitioned_file("foo"), partitioned_file("bar")]),
            FileGroup::new(vec![partitioned_file("baz")]),
            FileGroup::default(),
        ];

        // Summary mode shows counts (3 unique files, 3 partitioned files)
        let expected = "{3 groups: [num_files:3, num_partitioned_files:3]}";
        assert_eq!(
            DefaultDisplay(FileGroupsDisplay::new(&files, true)).to_string(),
            expected
        );
    }

    #[test]
    fn file_groups_display_many_detailed() {
        let files = [
            FileGroup::new(vec![partitioned_file("foo"), partitioned_file("bar")]),
            FileGroup::new(vec![partitioned_file("baz")]),
            FileGroup::default(),
        ];

        // Detailed mode shows file list
        let expected = "{3 groups: [[foo, bar], [baz], []]}";
        assert_eq!(
            DefaultDisplay(FileGroupsDisplay::new(&files, false)).to_string(),
            expected
        );
    }

    #[test]
    fn file_groups_display_many_verbose() {
        let files = [
            FileGroup::new(vec![partitioned_file("foo"), partitioned_file("bar")]),
            FileGroup::new(vec![partitioned_file("baz")]),
            FileGroup::default(),
        ];

        // Verbose mode always shows all files (show_summary ignored)
        let expected = "{3 groups: [[foo, bar], [baz], []]}";
        assert_eq!(
            VerboseDisplay(FileGroupsDisplay::new(&files, true)).to_string(),
            expected
        );
    }

    #[test]
    fn file_groups_display_too_many_summary() {
        let files = [
            FileGroup::new(vec![partitioned_file("foo"), partitioned_file("bar")]),
            FileGroup::new(vec![partitioned_file("baz")]),
            FileGroup::new(vec![partitioned_file("qux")]),
            FileGroup::new(vec![partitioned_file("quux")]),
            FileGroup::new(vec![partitioned_file("quuux")]),
            FileGroup::new(vec![partitioned_file("quuuux")]),
            FileGroup::default(),
        ];

        // Summary mode shows counts (7 unique files: foo,bar,baz,qux,quux,quuux,quuuux)
        let expected = "{7 groups: [num_files:7, num_partitioned_files:7]}";
        assert_eq!(
            DefaultDisplay(FileGroupsDisplay::new(&files, true)).to_string(),
            expected
        );
    }

    #[test]
    fn file_groups_display_too_many_detailed() {
        let files = [
            FileGroup::new(vec![partitioned_file("foo"), partitioned_file("bar")]),
            FileGroup::new(vec![partitioned_file("baz")]),
            FileGroup::new(vec![partitioned_file("qux")]),
            FileGroup::new(vec![partitioned_file("quux")]),
            FileGroup::new(vec![partitioned_file("quuux")]),
            FileGroup::new(vec![partitioned_file("quuuux")]),
            FileGroup::default(),
        ];

        // Detailed mode shows truncated list (max 5 groups)
        let expected = "{7 groups: [[foo, bar], [baz], [qux], [quux], [quuux], ...]}";
        assert_eq!(
            DefaultDisplay(FileGroupsDisplay::new(&files, false)).to_string(),
            expected
        );
    }

    #[test]
    fn file_groups_display_too_many_verbose() {
        let files = [
            FileGroup::new(vec![partitioned_file("foo"), partitioned_file("bar")]),
            FileGroup::new(vec![partitioned_file("baz")]),
            FileGroup::new(vec![partitioned_file("qux")]),
            FileGroup::new(vec![partitioned_file("quux")]),
            FileGroup::new(vec![partitioned_file("quuux")]),
            FileGroup::new(vec![partitioned_file("quuuux")]),
            FileGroup::default(),
        ];

        // Verbose mode shows all files
        let expected =
            "{7 groups: [[foo, bar], [baz], [qux], [quux], [quuux], [quuuux], []]}";
        assert_eq!(
            VerboseDisplay(FileGroupsDisplay::new(&files, true)).to_string(),
            expected
        );
    }

    #[test]
    fn file_group_display_many_default() {
        let files =
            FileGroup::new(vec![partitioned_file("foo"), partitioned_file("bar")]);

        let expected = "[foo, bar]";
        assert_eq!(
            DefaultDisplay(FileGroupDisplay(&files)).to_string(),
            expected
        );
    }

    #[test]
    fn file_group_display_too_many_default() {
        let files = FileGroup::new(vec![
            partitioned_file("foo"),
            partitioned_file("bar"),
            partitioned_file("baz"),
            partitioned_file("qux"),
            partitioned_file("quux"),
            partitioned_file("quuux"),
        ]);

        let expected = "[foo, bar, baz, qux, quux, ...]";
        assert_eq!(
            DefaultDisplay(FileGroupDisplay(&files)).to_string(),
            expected
        );
    }

    #[test]
    fn file_group_display_too_many_verbose() {
        let files = FileGroup::new(vec![
            partitioned_file("foo"),
            partitioned_file("bar"),
            partitioned_file("baz"),
            partitioned_file("qux"),
            partitioned_file("quux"),
            partitioned_file("quuux"),
        ]);

        let expected = "[foo, bar, baz, qux, quux, quuux]";
        assert_eq!(
            VerboseDisplay(FileGroupDisplay(&files)).to_string(),
            expected
        );
    }

    /// create a PartitionedFile for testing
    fn partitioned_file(path: &str) -> PartitionedFile {
        let object_meta = ObjectMeta {
            location: Path::parse(path).unwrap(),
            last_modified: Utc::now(),
            size: 42,
            e_tag: None,
            version: None,
        };

        PartitionedFile {
            object_meta,
            partition_values: vec![],
            range: None,
            statistics: None,
            extensions: None,
            metadata_size_hint: None,
        }
    }
}
