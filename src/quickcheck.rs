use crate::{
    shared_ptr::SharedPointerKind, GenericHashMap, GenericHashSet, GenericOrdMap, GenericOrdSet,
    GenericVector,
};
use ::quickcheck::{Arbitrary, Gen};
use std::hash::{BuildHasher, Hash};
use std::iter::FromIterator;

impl<A: Arbitrary + Sync + Clone, P: SharedPointerKind + 'static> Arbitrary
    for GenericVector<A, P>
{
    fn arbitrary(g: &mut Gen) -> Self {
        GenericVector::from_iter(Vec::<A>::arbitrary(g))
    }
}

impl<
        K: Ord + Clone + Arbitrary + Sync,
        V: Clone + Arbitrary + Sync,
        P: SharedPointerKind + 'static,
    > Arbitrary for GenericOrdMap<K, V, P>
{
    fn arbitrary(g: &mut Gen) -> Self {
        GenericOrdMap::from_iter(Vec::<(K, V)>::arbitrary(g))
    }
}

impl<A: Ord + Clone + Arbitrary + Sync, P: SharedPointerKind + 'static> Arbitrary
    for GenericOrdSet<A, P>
{
    fn arbitrary(g: &mut Gen) -> Self {
        GenericOrdSet::from_iter(Vec::<A>::arbitrary(g))
    }
}

impl<A, S, P> Arbitrary for GenericHashSet<A, S, P>
where
    A: Hash + Eq + Arbitrary + Sync,
    S: BuildHasher + Default + Send + Sync + 'static,
    P: SharedPointerKind + 'static,
{
    fn arbitrary(g: &mut Gen) -> Self {
        GenericHashSet::from_iter(Vec::<A>::arbitrary(g))
    }
}

impl<K, V, S, P> Arbitrary for GenericHashMap<K, V, S, P>
where
    K: Hash + Eq + Arbitrary + Sync,
    V: Arbitrary + Sync,
    S: BuildHasher + Default + Send + Sync + 'static,
    P: SharedPointerKind + 'static,
{
    fn arbitrary(g: &mut Gen) -> Self {
        GenericHashMap::from(Vec::<(K, V)>::arbitrary(g))
    }
}
