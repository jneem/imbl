// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Every codebase needs a `util` module.

use std::cmp::Ordering;
use std::ops::{Bound, Range, RangeBounds};

#[cfg(feature = "pool")]
pub(crate) use refpool::{PoolClone, PoolDefault};

// The `Ref` type is an alias for either `Rc` or `Arc`, user's choice.
// FIXME: we have temporarily disabled `Rc`, so this is always `Arc`.
// `Arc` without refpool
pub(crate) use crate::fakepool::{Arc as PoolRef, Pool, PoolClone, PoolDefault};

// `Ref` == `Arc` when threadsafe
pub(crate) type Ref<A> = std::sync::Arc<A>;

pub(crate) fn clone_ref<A>(r: Ref<A>) -> A
where
    A: Clone,
{
    Ref::try_unwrap(r).unwrap_or_else(|r| (*r).clone())
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Side {
    Left,
    Right,
}

#[allow(dead_code)]
pub(crate) fn linear_search_by<'a, A, I, F>(iterable: I, mut cmp: F) -> Result<usize, usize>
where
    A: 'a,
    I: IntoIterator<Item = &'a A>,
    F: FnMut(&A) -> Ordering,
{
    let mut pos = 0;
    for value in iterable {
        match cmp(value) {
            Ordering::Equal => return Ok(pos),
            Ordering::Greater => return Err(pos),
            Ordering::Less => {}
        }
        pos += 1;
    }
    Err(pos)
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

macro_rules! def_pool {
    ($name:ident<$($arg:tt),*>, $pooltype:ty) => {
        /// A memory pool for the appropriate node type.
        pub struct $name<$($arg,)*>(Pool<$pooltype>);

        impl<$($arg,)*> $name<$($arg,)*> {
            /// Create a new pool with the given size.
            pub fn new(size: usize) -> Self {
                Self(Pool::new(size))
            }

            /// Fill the pool with preallocated chunks.
            pub fn fill(&self) {
                self.0.fill();
            }

            ///Get the current size of the pool.
            pub fn pool_size(&self) -> usize {
                self.0.get_pool_size()
            }
        }

        impl<$($arg,)*> Default for $name<$($arg,)*> {
            fn default() -> Self {
                Self::new($crate::config::POOL_SIZE)
            }
        }

        impl<$($arg,)*> Clone for $name<$($arg,)*> {
            fn clone(&self) -> Self {
                Self(self.0.clone())
            }
        }
    };
}

#[cfg(test)]
macro_rules! assert_covariant {
    ($name:ident<$($gen:tt),*> in $param:ident) => {
        #[allow(unused_assignments, unused_variables)]
        const _: () = {
            type Tmp<$param> = $name<$($gen),*>;
            fn assign<'a, 'b: 'a>(src: Tmp<&'b i32>, mut dst: Tmp<&'a i32>) {
                dst = src;
            }
        };
    }
}
