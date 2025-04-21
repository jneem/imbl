// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use arbitrary::{size_hint, Arbitrary, Result, Unstructured};
use std::hash::{BuildHasher, Hash};

use crate::{
    shared_ptr::SharedPointerKind, GenericHashMap, GenericHashSet, GenericOrdMap, GenericOrdSet,
    GenericVector,
};

impl<'a, A: Arbitrary<'a> + Clone, P: SharedPointerKind + 'static> Arbitrary<'a>
    for GenericVector<A, P>
{
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        u.arbitrary_iter()?.collect()
    }

    fn arbitrary_take_rest(u: Unstructured<'a>) -> Result<Self> {
        u.arbitrary_take_rest_iter()?.collect()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        size_hint::recursion_guard(depth, |depth| {
            size_hint::and(<usize as Arbitrary>::size_hint(depth), (0, None))
        })
    }
}

impl<
        'a,
        K: Arbitrary<'a> + Ord + Clone,
        V: Arbitrary<'a> + Clone,
        P: SharedPointerKind + 'static,
    > Arbitrary<'a> for GenericOrdMap<K, V, P>
{
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        u.arbitrary_iter()?.collect()
    }

    fn arbitrary_take_rest(u: Unstructured<'a>) -> Result<Self> {
        u.arbitrary_take_rest_iter()?.collect()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        size_hint::recursion_guard(depth, |depth| {
            size_hint::and(<usize as Arbitrary>::size_hint(depth), (0, None))
        })
    }
}

impl<'a, A: Arbitrary<'a> + Ord + Clone, P: SharedPointerKind + 'static> Arbitrary<'a>
    for GenericOrdSet<A, P>
{
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        u.arbitrary_iter()?.collect()
    }

    fn arbitrary_take_rest(u: Unstructured<'a>) -> Result<Self> {
        u.arbitrary_take_rest_iter()?.collect()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        size_hint::recursion_guard(depth, |depth| {
            size_hint::and(<usize as Arbitrary>::size_hint(depth), (0, None))
        })
    }
}

impl<'a, K, V, S, P> Arbitrary<'a> for GenericHashMap<K, V, S, P>
where
    K: Arbitrary<'a> + Hash + Eq + Clone,
    V: Arbitrary<'a> + Clone,
    S: BuildHasher + Clone + Default + 'static,
    P: SharedPointerKind + 'static,
{
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        u.arbitrary_iter()?.collect()
    }

    fn arbitrary_take_rest(u: Unstructured<'a>) -> Result<Self> {
        u.arbitrary_take_rest_iter()?.collect()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        size_hint::recursion_guard(depth, |depth| {
            size_hint::and(<usize as Arbitrary>::size_hint(depth), (0, None))
        })
    }
}

impl<'a, A, S, P> Arbitrary<'a> for GenericHashSet<A, S, P>
where
    A: Arbitrary<'a> + Hash + Eq + Clone,
    S: BuildHasher + Clone + Default + 'static,
    P: SharedPointerKind + 'static,
{
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        u.arbitrary_iter()?.collect()
    }

    fn arbitrary_take_rest(u: Unstructured<'a>) -> Result<Self> {
        u.arbitrary_take_rest_iter()?.collect()
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        size_hint::recursion_guard(depth, |depth| {
            size_hint::and(<usize as Arbitrary>::size_hint(depth), (0, None))
        })
    }
}
