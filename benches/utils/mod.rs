#![allow(dead_code)]
use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::collections::BTreeSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

// Trait for generating test data
pub trait TestData: Clone + Debug + Ord + Eq + Hash {
    fn generate(size: usize) -> Vec<Self>;
}

impl TestData for i64 {
    fn generate(size: usize) -> Vec<Self> {
        let mut gen = SmallRng::seed_from_u64(1);
        let mut set = BTreeSet::new();
        while set.len() < size {
            let next = gen.random::<i64>();
            set.insert(next);
        }
        set.into_iter().collect()
    }
}

impl TestData for String {
    fn generate(size: usize) -> Vec<Self> {
        let mut gen = SmallRng::seed_from_u64(1);
        let mut set = BTreeSet::new();
        while set.len() < size {
            let len = gen.random_range(5..20);
            let s: String = (0..len)
                .map(|_| gen.random_range(b'a'..=b'z') as char)
                .collect();
            set.insert(s);
        }
        set.into_iter().collect()
    }
}

impl<T> TestData for Arc<T>
where
    T: TestData + 'static,
{
    fn generate(size: usize) -> Vec<Self> {
        T::generate(size).into_iter().map(Arc::new).collect()
    }
}

pub fn reorder<A: Clone>(vec: &[A]) -> Vec<A> {
    let mut gen = SmallRng::seed_from_u64(1);
    let mut out = vec.to_vec();
    out.shuffle(&mut gen);
    out
}
