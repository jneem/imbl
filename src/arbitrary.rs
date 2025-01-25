// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use ::arbitrary::{size_hint, Arbitrary, Result, Unstructured};
use std::hash::{BuildHasher, Hash};

use crate::{HashMap, HashSet, OrdMap, OrdSet, Vector, shared_ptr::SharedPointerKind};

impl<'a, A: Arbitrary<'a> + Clone, P: SharedPointerKind + 'static> Arbitrary<'a> for Vector<A, P> {
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

impl<'a, K: Arbitrary<'a> + Ord + Clone, V: Arbitrary<'a> + Clone, P: SharedPointerKind + 'static> Arbitrary<'a> for OrdMap<K, V, P> {
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

impl<'a, A: Arbitrary<'a> + Ord + Clone, P: SharedPointerKind + 'static> Arbitrary<'a> for OrdSet<A, P> {
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

impl<'a, K, V, S, P> Arbitrary<'a> for HashMap<K, V, S, P>
where
    K: Arbitrary<'a> + Hash + Eq + Clone,
    V: Arbitrary<'a> + Clone,
    S: BuildHasher + Default + 'static,
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

impl<'a, A, S, P> Arbitrary<'a> for HashSet<A, S, P>
where
    A: Arbitrary<'a> + Hash + Eq + Clone,
    S: BuildHasher + Default + 'static,
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
