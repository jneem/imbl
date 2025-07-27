// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Every codebase needs a `util` module.

use std::ops::{Bound, Range, RangeBounds};

use archery::{SharedPointer, SharedPointerKind};

pub(crate) fn clone_ref<A, P>(r: SharedPointer<A, P>) -> A
where
    A: Clone,
    P: SharedPointerKind,
{
    SharedPointer::try_unwrap(r).unwrap_or_else(|r| (*r).clone())
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Side {
    Left,
    Right,
}

pub(crate) fn to_range<R>(range: &R, right_unbounded: usize) -> Range<usize>
where
    R: RangeBounds<usize>,
{
    let start_index = match range.start_bound() {
        Bound::Included(i) => *i,
        Bound::Excluded(i) => *i + 1,
        Bound::Unbounded => 0,
    };
    let end_index = match range.end_bound() {
        Bound::Included(i) => *i + 1,
        Bound::Excluded(i) => *i,
        Bound::Unbounded => right_unbounded,
    };
    start_index..end_index
}

#[cfg(test)]
macro_rules! assert_covariant {
    ($name:ident<$($gen:tt),*> in $param:ident) => {
        #[allow(dead_code, unused_assignments, unused_variables)]
        const _: () = {
            type Tmp<$param> = $name<$($gen),*>;
            fn assign<'a, 'b: 'a>(src: Tmp<&'b i32>, mut dst: Tmp<&'a i32>) {
                dst = src;
            }
        };
    }
}
