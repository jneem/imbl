use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use imbl::ordmap::OrdMap;
use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::iter::FromIterator;
use std::sync::Arc;

use archery::ArcTK;
use rpds::RedBlackTreeMapSync;

mod utils;
use utils::*;

// Trait to abstract over different map implementations
trait BenchMap<K, V>: Clone + FromIterator<(K, V)>
where
    K: Clone + Ord,
    V: Clone,
{
    const IMMUTABLE: bool = true;
    type Iter<'a>: Iterator<Item = (&'a K, &'a V)>
    where
        Self: 'a,
        K: 'a,
        V: 'a;
    type RangeIter<'a>: Iterator<Item = (&'a K, &'a V)>
    where
        Self: 'a,
        K: 'a,
        V: 'a;

    fn new() -> Self;
    fn insert(&mut self, k: K, v: V) -> Option<V>;
    fn insert_clone(&self, k: K, v: V) -> Self;
    fn remove(&mut self, k: &K) -> Option<V>;
    fn remove_clone(&self, k: &K) -> Self;
    fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized;
    fn iter(&self) -> Self::Iter<'_>;
    fn range<'a>(&'a self, range: std::ops::RangeFrom<&'a K>) -> Self::RangeIter<'a>;
    fn is_empty(&self) -> bool;
    fn without_min(&self) -> (Option<(K, V)>, Self);
    fn without_max(&self) -> (Option<(K, V)>, Self);
}

// Implementation for OrdMap
impl<K, V> BenchMap<K, V> for OrdMap<K, V>
where
    K: Clone + Ord,
    V: Clone,
{
    type Iter<'a>
        = imbl::ordmap::Iter<'a, K, V, imbl::shared_ptr::DefaultSharedPtr>
    where
        K: 'a,
        V: 'a;
    type RangeIter<'a>
        = imbl::ordmap::RangedIter<'a, K, V, imbl::shared_ptr::DefaultSharedPtr>
    where
        K: 'a,
        V: 'a;

    fn new() -> Self {
        OrdMap::new()
    }

    fn insert(&mut self, k: K, v: V) -> Option<V> {
        self.insert(k, v)
    }

    fn insert_clone(&self, k: K, v: V) -> Self {
        self.update(k, v)
    }

    fn remove(&mut self, k: &K) -> Option<V> {
        self.remove(k)
    }

    fn remove_clone(&self, k: &K) -> Self {
        self.without(k)
    }

    fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(k)
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }

    fn range<'a>(&'a self, range: std::ops::RangeFrom<&'a K>) -> Self::RangeIter<'a> {
        self.range(range)
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn without_min(&self) -> (Option<(K, V)>, Self) {
        self.without_min_with_key()
    }

    fn without_max(&self) -> (Option<(K, V)>, Self) {
        self.without_max_with_key()
    }
}

// Implementation for RedBlackTreeMapSync
impl<K, V> BenchMap<K, V> for RedBlackTreeMapSync<K, V>
where
    K: Clone + Ord,
    V: Clone,
{
    type Iter<'a>
        = rpds::map::red_black_tree_map::Iter<'a, K, V, ArcTK>
    where
        K: 'a,
        V: 'a;
    type RangeIter<'a>
        = std::iter::Map<
        rpds::map::red_black_tree_map::RangeIter<'a, K, V, std::ops::RangeFrom<&'a K>, K, ArcTK>,
        fn((&'a K, &'a V)) -> (&'a K, &'a V),
    >
    where
        K: 'a,
        V: 'a;

    fn new() -> Self {
        RedBlackTreeMapSync::new_sync()
    }

    fn insert(&mut self, k: K, v: V) -> Option<V> {
        self.insert_mut(k, v);
        None
    }

    fn insert_clone(&self, k: K, v: V) -> Self {
        self.insert(k, v)
    }

    fn remove(&mut self, k: &K) -> Option<V> {
        if self.remove_mut(k) {
            None // rpds doesn't return the removed value
        } else {
            None
        }
    }

    fn remove_clone(&self, k: &K) -> Self {
        self.remove(k)
    }

    fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(k)
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }

    fn range<'a>(&'a self, range: std::ops::RangeFrom<&'a K>) -> Self::RangeIter<'a> {
        self.range::<K, _>(range).map(|(k, v)| (k, v))
    }

    fn is_empty(&self) -> bool {
        self.size() == 0
    }

    fn without_min(&self) -> (Option<(K, V)>, Self) {
        match self.first() {
            Some((k, _)) => {
                let k = k.clone();
                let new_map = self.remove(&k);
                (self.get(&k).map(|v| (k, v.clone())), new_map)
            }
            None => (None, self.clone()),
        }
    }

    fn without_max(&self) -> (Option<(K, V)>, Self) {
        match self.last() {
            Some((k, _)) => {
                let k = k.clone();
                let new_map = self.remove(&k);
                (self.get(&k).map(|v| (k, v.clone())), new_map)
            }
            None => (None, self.clone()),
        }
    }
}

// Implementation for BTreeMap
impl<K, V> BenchMap<K, V> for BTreeMap<K, V>
where
    K: Clone + Ord,
    V: Clone,
{
    const IMMUTABLE: bool = false;
    type Iter<'a>
        = std::collections::btree_map::Iter<'a, K, V>
    where
        K: 'a,
        V: 'a;
    type RangeIter<'a>
        = std::collections::btree_map::Range<'a, K, V>
    where
        K: 'a,
        V: 'a;

    fn new() -> Self {
        BTreeMap::new()
    }

    fn insert(&mut self, k: K, v: V) -> Option<V> {
        self.insert(k, v)
    }

    fn insert_clone(&self, k: K, v: V) -> Self {
        let mut ret = self.clone();
        ret.insert(k, v);
        ret
    }

    fn remove(&mut self, k: &K) -> Option<V> {
        self.remove(k)
    }

    fn remove_clone(&self, k: &K) -> Self {
        let mut ret = self.clone();
        ret.remove(k);
        ret
    }

    fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(k)
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }

    fn range<'a>(&'a self, range: std::ops::RangeFrom<&'a K>) -> Self::RangeIter<'a> {
        self.range(range)
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn without_min(&self) -> (Option<(K, V)>, Self) {
        let mut ret = self.clone();
        if let Some(o) = ret.first_entry() {
            let (k, v) = o.remove_entry();
            (Some((k, v)), ret)
        } else {
            (None, ret)
        }
    }

    fn without_max(&self) -> (Option<(K, V)>, Self) {
        let mut ret = self.clone();
        if let Some(o) = ret.last_entry() {
            let (k, v) = o.remove_entry();
            (Some((k, v)), ret)
        } else {
            (None, ret)
        }
    }
}

// Generic benchmark functions
fn bench_lookup<M, K, V>(b: &mut Bencher, size: usize)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let keys = K::generate(size);
    let values = V::generate(size);
    let order = reorder(&keys);
    let m: M = keys.into_iter().zip(values).collect();
    b.iter(|| {
        for k in &order {
            black_box(m.get(k));
        }
    })
}

fn bench_lookup_ne<M, K, V>(b: &mut Bencher, size: usize)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let keys = K::generate(size * 2);
    let values = V::generate(size);
    let order = reorder(&keys[size..]);
    let m: M = keys.into_iter().zip(values).collect();
    b.iter(|| {
        for k in &order {
            black_box(m.get(k));
        }
    })
}

fn bench_insert<M, K, V>(b: &mut Bencher, size: usize)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let keys = K::generate(size);
    let values = V::generate(size);
    b.iter(|| {
        let mut m = M::new();
        for (k, v) in keys.clone().into_iter().zip(values.clone()) {
            m = m.insert_clone(k, v);
        }
        m
    })
}

fn bench_insert_mut<M, K, V>(b: &mut Bencher, size: usize)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let keys = K::generate(size);
    let values = V::generate(size);
    b.iter(|| {
        let mut m = M::new();
        for (k, v) in keys.clone().into_iter().zip(values.clone()) {
            m.insert(k, v);
        }
        m
    })
}

fn bench_remove<M, K, V>(b: &mut Bencher, size: usize)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let keys = K::generate(size);
    let values = V::generate(size);
    let order = reorder(&keys);
    let map: M = keys.into_iter().zip(values).collect();
    b.iter(|| {
        let mut m = map.clone();
        for k in &order {
            m = m.remove_clone(k);
        }
        m
    })
}

fn bench_remove_mut<M, K, V>(b: &mut Bencher, size: usize)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let keys = K::generate(size);
    let values = V::generate(size);
    let order = reorder(&keys);
    let map: M = keys.into_iter().zip(values).collect();
    b.iter(|| {
        let mut m = map.clone();
        for k in &order {
            m.remove(k);
        }
        m
    })
}

fn bench_remove_min<M, K, V>(b: &mut Bencher, size: usize)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let keys = K::generate(size);
    let values = V::generate(size);
    let map: M = keys.into_iter().zip(values).collect();
    b.iter(|| {
        let mut m = map.clone();
        assert!(!m.is_empty());
        for _ in 0..size {
            m = m.without_min().1;
        }
        assert!(m.is_empty());
        m
    })
}

fn bench_remove_max<M, K, V>(b: &mut Bencher, size: usize)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    if !M::IMMUTABLE {
        return; // Skip for non-immutable maps
    }
    let keys = K::generate(size);
    let values = V::generate(size);
    let map: M = keys.into_iter().zip(values).collect();
    b.iter(|| {
        let mut m = map.clone();
        assert!(!m.is_empty());
        for _ in 0..size {
            m = m.without_max().1;
        }
        assert!(m.is_empty());
        m
    })
}

fn bench_insert_once<M, K, V>(b: &mut Bencher, size: usize)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let keys = K::generate(size);
    let values = V::generate(size);
    let korder = reorder(&keys);
    let vorder = reorder(&values);
    let m: M = keys.clone().into_iter().zip(values).collect();
    b.iter(|| {
        for (k, v) in korder.iter().zip(vorder.iter()).take(100) {
            black_box(m.insert_clone(k.clone(), v.clone()));
        }
    })
}

fn bench_remove_once<M, K, V>(b: &mut Bencher, size: usize)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let keys = K::generate(size);
    let values = V::generate(size);
    let order = reorder(&keys);
    let map: M = keys.clone().into_iter().zip(values).collect();
    b.iter(|| {
        for k in order.iter().take(100) {
            black_box(map.remove_clone(k));
        }
    })
}

fn bench_iter<M, K, V>(b: &mut Bencher, size: usize)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let keys = K::generate(size);
    let values = V::generate(size);
    let m: M = keys.into_iter().zip(values).collect();
    b.iter(|| {
        for p in m.iter() {
            black_box(p);
        }
    })
}

fn bench_range_iter<M, K, V>(b: &mut Bencher, size: usize)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let keys = K::generate(size);
    let values = V::generate(size);
    let order = reorder(&keys);
    let m: M = keys.into_iter().zip(values).collect();
    b.iter(|| {
        for k in order.iter().take(10) {
            for p in m.range(k..).take(100) {
                black_box(p);
            }
        }
    })
}

// Benchmark functions for each map type
fn bench_ordmap(c: &mut Criterion) {
    bench_group::<OrdMap<i64, i64>, i64, i64>(c, "ordmap_i64");
    bench_group::<OrdMap<Arc<String>, Arc<String>>, Arc<String>, Arc<String>>(c, "ordmap_str");
}

fn bench_rpds(c: &mut Criterion) {
    bench_group::<RedBlackTreeMapSync<i64, i64>, i64, i64>(c, "rpds_i64");
    bench_group::<RedBlackTreeMapSync<Arc<String>, Arc<String>>, Arc<String>, Arc<String>>(
        c, "rpds_str",
    );
}

fn bench_btreemap(c: &mut Criterion) {
    bench_group::<BTreeMap<i64, i64>, i64, i64>(c, "btreemap_i64");
    bench_group::<BTreeMap<Arc<String>, Arc<String>>, Arc<String>, Arc<String>>(c, "btreemap_str");
}

// Helper function to run all benchmarks for a specific map/key/value type
fn bench_group<M, K, V>(c: &mut Criterion, group_name: &str)
where
    M: BenchMap<K, V>,
    K: TestData,
    V: TestData,
{
    let mut group = c.benchmark_group(group_name);

    for size in &[100, 1000, 10000, 100000] {
        group.bench_function(format!("lookup_{}", size), |b| {
            bench_lookup::<M, K, V>(b, *size)
        });
    }

    for size in &[10000, 100000] {
        group.bench_function(format!("lookup_ne_{}", size), |b| {
            bench_lookup_ne::<M, K, V>(b, *size)
        });
    }

    for size in &[100, 1000, 10000, 100000] {
        group.bench_function(format!("insert_mut_{}", size), |b| {
            bench_insert_mut::<M, K, V>(b, *size)
        });
    }

    for size in &[100, 1000, 10000] {
        group.bench_function(format!("remove_mut_{}", size), |b| {
            bench_remove_mut::<M, K, V>(b, *size)
        });
    }

    for size in &[1000, 10000] {
        group.bench_function(format!("iter_{}", size), |b| {
            bench_iter::<M, K, V>(b, *size)
        });
    }

    for size in &[100, 1000, 10000, 100000] {
        group.bench_function(format!("range_iter_{}", size), |b| {
            bench_range_iter::<M, K, V>(b, *size)
        });
    }

    if M::IMMUTABLE {
        for size in &[100, 1000, 10000] {
            group.bench_function(format!("insert_{}", size), |b| {
                bench_insert::<M, K, V>(b, *size)
            });

            group.bench_function(format!("remove_{}", size), |b| {
                bench_remove::<M, K, V>(b, *size)
            });

            group.bench_function(format!("insert_once_{}", size), |b| {
                bench_insert_once::<M, K, V>(b, *size)
            });

            group.bench_function(format!("remove_once_{}", size), |b| {
                bench_remove_once::<M, K, V>(b, *size)
            });
        }

        for size in &[1000] {
            group.bench_function(format!("remove_min_{}", size), |b| {
                bench_remove_min::<M, K, V>(b, *size)
            });

            group.bench_function(format!("remove_max_{}", size), |b| {
                bench_remove_max::<M, K, V>(b, *size)
            });
        }
    }

    group.finish();
}

// Main benchmark entry point
fn ordmap_benches(c: &mut Criterion) {
    bench_ordmap(c);

    if std::env::var("BENCH_STD").is_ok() {
        bench_btreemap(c);
    }

    if std::env::var("BENCH_RPDS").is_ok() {
        bench_rpds(c);
    }
}

criterion_group!(benches, ordmap_benches);
criterion_main!(benches);
