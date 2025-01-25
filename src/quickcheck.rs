use crate::{HashMap, HashSet, OrdMap, OrdSet, Vector, shared_ptr::SharedPointerKind};
use ::quickcheck::{Arbitrary, Gen};
use std::hash::{BuildHasher, Hash};
use std::iter::FromIterator;

impl<A: Arbitrary + Sync + Clone, P: SharedPointerKind + 'static> Arbitrary for Vector<A, P> {
    fn arbitrary(g: &mut Gen) -> Self {
        Vector::from_iter(Vec::<A>::arbitrary(g))
    }
}

impl<K: Ord + Clone + Arbitrary + Sync, V: Clone + Arbitrary + Sync, P: SharedPointerKind + 'static> Arbitrary for OrdMap<K, V, P> {
    fn arbitrary(g: &mut Gen) -> Self {
        OrdMap::from_iter(Vec::<(K, V)>::arbitrary(g))
    }
}

impl<A: Ord + Clone + Arbitrary + Sync, P: SharedPointerKind + 'static> Arbitrary for OrdSet<A, P> {
    fn arbitrary(g: &mut Gen) -> Self {
        OrdSet::from_iter(Vec::<A>::arbitrary(g))
    }
}

impl<A, S, P> Arbitrary for HashSet<A, S, P>
where
    A: Hash + Eq + Arbitrary + Sync,
    S: BuildHasher + Default + Send + Sync + 'static,
    P: SharedPointerKind + 'static,
{
    fn arbitrary(g: &mut Gen) -> Self {
        HashSet::from_iter(Vec::<A>::arbitrary(g))
    }
}

impl<K, V, S, P> Arbitrary for HashMap<K, V, S, P>
where
    K: Hash + Eq + Arbitrary + Sync,
    V: Arbitrary + Sync,
    S: BuildHasher + Default + Send + Sync + 'static,
    P: SharedPointerKind + 'static,
{
    fn arbitrary(g: &mut Gen) -> Self {
        HashMap::from(Vec::<(K, V)>::arbitrary(g))
    }
}
