// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#![allow(dead_code)]

use crate::nodes::chunk::Chunk;
use std::marker::PhantomData;

pub(crate) trait PoolDefault: Default {}
pub(crate) trait PoolClone: Clone {}

impl<A> PoolDefault for Chunk<A> {}
impl<A> PoolClone for Chunk<A> where A: Clone {}

pub(crate) struct Pool<A>(PhantomData<A>);

impl<A> Pool<A> {
    pub(crate) fn new(_size: usize) -> Self {
        Pool(PhantomData)
    }

    pub(crate) fn get_pool_size(&self) -> usize {
        0
    }

    pub(crate) fn fill(&self) {}
}

impl<A> Clone for Pool<A> {
    fn clone(&self) -> Self {
        Self::new(0)
    }
}
