use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use imbl::vector::Vector;
use rand::seq::SliceRandom;
use std::collections::VecDeque;
use std::hint::black_box;
use std::iter::FromIterator;

mod utils;

// Trait to abstract over different vector-like implementations
trait BenchVector<T>: Clone + FromIterator<T>
where
    T: Clone,
{
    type Iter<'a>: Iterator<Item = &'a T>
    where
        Self: 'a,
        T: 'a;

    fn new() -> Self;
    fn push_front(&mut self, value: T);
    fn push_back(&mut self, value: T);
    fn pop_front(&mut self) -> Option<T>;
    fn pop_back(&mut self) -> Option<T>;
    fn get(&self, index: usize) -> Option<&T>;
    fn iter(&self) -> Self::Iter<'_>;

    // Only some implementations support these
    fn split_off(&mut self, at: usize) -> Self;
    fn append(&mut self, other: Self);
    fn sort(&mut self)
    where
        T: Ord;

    // Vector-specific features
    fn supports_focus() -> bool {
        false
    }
    fn focus(&self) -> Option<VectorFocus<'_, T>> {
        None
    }
    fn focus_mut(&mut self) -> Option<VectorFocusMut<'_, T>> {
        None
    }
}

// Wrapper types for Vector's focus feature
struct VectorFocus<'a, T> {
    focus: imbl::vector::Focus<'a, T, imbl::shared_ptr::DefaultSharedPtr>,
}

impl<'a, T> VectorFocus<'a, T> {
    fn get(&mut self, index: usize) -> Option<&T> {
        self.focus.get(index)
    }
}

struct VectorFocusMut<'a, T> {
    focus: imbl::vector::FocusMut<'a, T, imbl::shared_ptr::DefaultSharedPtr>,
}

impl<'a, T: Clone> VectorFocusMut<'a, T> {
    fn get(&mut self, index: usize) -> Option<&T> {
        self.focus.get(index)
    }
}

// Implementation for imbl::Vector
impl<T: Clone> BenchVector<T> for Vector<T> {
    type Iter<'a>
        = imbl::vector::Iter<'a, T, imbl::shared_ptr::DefaultSharedPtr>
    where
        T: 'a;

    fn new() -> Self {
        Vector::new()
    }

    fn push_front(&mut self, value: T) {
        self.push_front(value);
    }

    fn push_back(&mut self, value: T) {
        self.push_back(value);
    }

    fn pop_front(&mut self) -> Option<T> {
        self.pop_front()
    }

    fn pop_back(&mut self) -> Option<T> {
        self.pop_back()
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.get(index)
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }

    fn split_off(&mut self, at: usize) -> Self {
        self.split_off(at)
    }

    fn append(&mut self, other: Self) {
        self.append(other);
    }

    fn sort(&mut self)
    where
        T: Ord,
    {
        self.sort();
    }

    fn supports_focus() -> bool {
        true
    }

    fn focus(&self) -> Option<VectorFocus<'_, T>> {
        Some(VectorFocus {
            focus: self.focus(),
        })
    }

    fn focus_mut(&mut self) -> Option<VectorFocusMut<'_, T>> {
        Some(VectorFocusMut {
            focus: self.focus_mut(),
        })
    }
}

// Implementation for std::collections::VecDeque
impl<T: Clone> BenchVector<T> for VecDeque<T> {
    type Iter<'a>
        = std::collections::vec_deque::Iter<'a, T>
    where
        T: 'a;

    fn new() -> Self {
        VecDeque::new()
    }

    fn push_front(&mut self, value: T) {
        self.push_front(value);
    }

    fn push_back(&mut self, value: T) {
        self.push_back(value);
    }

    fn pop_front(&mut self) -> Option<T> {
        self.pop_front()
    }

    fn pop_back(&mut self) -> Option<T> {
        self.pop_back()
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.get(index)
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }

    fn split_off(&mut self, at: usize) -> Self {
        self.split_off(at)
    }

    fn append(&mut self, mut other: Self) {
        self.append(&mut other);
    }

    fn sort(&mut self)
    where
        T: Ord,
    {
        self.make_contiguous().sort();
    }
}

// Generic benchmark functions
fn bench_sort_sorted<V: BenchVector<usize>>(b: &mut Bencher, size: usize) {
    b.iter(|| {
        let mut v: V = (0..size).collect();
        v.sort();
        black_box(v);
    });
}

fn bench_sort_reverse<V: BenchVector<usize>>(b: &mut Bencher, size: usize) {
    b.iter(|| {
        let mut v: V = (0..size).rev().collect();
        v.sort();
        black_box(v);
    });
}

fn bench_sort_shuffled<V: BenchVector<usize>>(b: &mut Bencher, size: usize) {
    let mut rng = rand::rng();
    b.iter(|| {
        let mut v: Vec<_> = (0..size).collect();
        v.shuffle(&mut rng);
        let mut v: V = v.into_iter().collect();
        v.sort();
        black_box(v);
    });
}

fn bench_push_front<V: BenchVector<usize>>(b: &mut Bencher, size: usize) {
    b.iter(|| {
        let mut v = V::new();
        for i in 0..size {
            v.push_front(i);
        }
        black_box(v);
    });
}

fn bench_push_back<V: BenchVector<usize>>(b: &mut Bencher, size: usize) {
    b.iter(|| {
        let mut v = V::new();
        for i in 0..size {
            v.push_back(i);
        }
        black_box(v);
    });
}

fn bench_pop_front<V: BenchVector<usize>>(b: &mut Bencher, size: usize) {
    let v: V = (0..size).collect();
    b.iter(|| {
        let mut v = v.clone();
        for _ in 0..size {
            v.pop_front();
        }
        black_box(v);
    });
}

fn bench_pop_back<V: BenchVector<usize>>(b: &mut Bencher, size: usize) {
    let v: V = (0..size).collect();
    b.iter(|| {
        let mut v = v.clone();
        for _ in 0..size {
            v.pop_back();
        }
        black_box(v);
    });
}

fn bench_split<V: BenchVector<usize>>(b: &mut Bencher, size: usize) {
    let v: V = (0..size).collect();
    b.iter(|| {
        let mut v = v.clone();
        black_box(v.split_off(size / 2));
    });
}

fn bench_append<V: BenchVector<usize>>(b: &mut Bencher, size: usize) {
    let size = size;
    let v1: V = (0..size / 2).collect();
    let v2: V = (size / 2..size).collect();
    b.iter(|| {
        let mut v = v1.clone();
        v.append(v2.clone());
        black_box(v);
    });
}

fn bench_iter<V: BenchVector<usize>>(b: &mut Bencher, size: usize) {
    let v: V = (0..size).collect();
    b.iter(|| {
        for item in v.iter() {
            black_box(item);
        }
    });
}

fn bench_get_seq<V: BenchVector<usize>>(b: &mut Bencher, size: usize) {
    let v: V = (0..size).collect();
    b.iter(|| {
        for i in 0..size {
            black_box(v.get(i));
        }
    });
}

fn bench_get_seq_focus<V: BenchVector<usize>>(b: &mut Bencher, size: usize) {
    if !V::supports_focus() {
        return;
    }
    let v: V = (0..size).collect();
    if let Some(mut focus) = v.focus() {
        b.iter(|| {
            for i in 0..size {
                black_box(focus.get(i));
            }
        });
    }
}

fn bench_get_seq_focus_mut<V: BenchVector<usize>>(b: &mut Bencher, size: usize) {
    if !V::supports_focus() {
        return;
    }
    let v: V = (0..size).collect();
    b.iter(|| {
        let mut v = v.clone();
        if let Some(mut focus) = v.focus_mut() {
            for i in 0..size {
                black_box(focus.get(i));
            }
        }
    });
}

fn bench_iter_max<V: BenchVector<usize>>(b: &mut Bencher, size: usize) {
    let v: V = (0..size).collect();
    b.iter(|| black_box(v.iter().max()));
}

// Helper function to run sort benchmarks
fn bench_sort_group<V: BenchVector<usize>>(c: &mut Criterion, group_name: &str) {
    let mut group = c.benchmark_group(format!("{}_sort", group_name));

    for size in &[500, 1000, 1500, 2000, 2500] {
        group.bench_function(format!("sorted_{}", size), |b| {
            bench_sort_sorted::<V>(b, *size)
        });

        group.bench_function(format!("reverse_{}", size), |b| {
            bench_sort_reverse::<V>(b, *size)
        });

        group.bench_function(format!("shuffled_{}", size), |b| {
            bench_sort_shuffled::<V>(b, *size)
        });
    }

    group.finish();
}

// Helper function to run vector operation benchmarks
fn bench_ops_group<V: BenchVector<usize>>(c: &mut Criterion, group_name: &str) {
    let mut group = c.benchmark_group(format!("{}_ops", group_name));

    for size in &[100, 1000, 100000] {
        group.bench_function(format!("push_front_{}", size), |b| {
            bench_push_front::<V>(b, *size)
        });

        group.bench_function(format!("push_back_{}", size), |b| {
            bench_push_back::<V>(b, *size)
        });

        group.bench_function(format!("pop_front_{}", size), |b| {
            bench_pop_front::<V>(b, *size)
        });

        group.bench_function(format!("pop_back_{}", size), |b| {
            bench_pop_back::<V>(b, *size)
        });

        group.bench_function(format!("split_{}", size), |b| bench_split::<V>(b, *size));

        group.bench_function(format!("iter_{}", size), |b| bench_iter::<V>(b, *size));

        group.bench_function(format!("get_seq_{}", size), |b| {
            bench_get_seq::<V>(b, *size)
        });

        if <V as BenchVector<usize>>::supports_focus() {
            group.bench_function(format!("get_seq_focus_{}", size), |b| {
                bench_get_seq_focus::<V>(b, *size)
            });

            group.bench_function(format!("get_seq_focus_mut_{}", size), |b| {
                bench_get_seq_focus_mut::<V>(b, *size)
            });
        }
    }

    // Append has different sizes
    for size in &[10, 100, 1000, 10000, 100000] {
        group.bench_function(format!("append_{}", size), |b| bench_append::<V>(b, *size));
    }

    // Iterator max benchmarks
    for size in &[1000, 100000, 10000000] {
        group.bench_function(format!("iter_max_{}", size), |b| {
            bench_iter_max::<V>(b, *size)
        });
    }

    group.finish();
}

// Benchmark functions for each vector type
fn bench_vector(c: &mut Criterion) {
    bench_sort_group::<Vector<usize>>(c, "vector");
    bench_ops_group::<Vector<usize>>(c, "vector");
}

fn bench_vecdeque(c: &mut Criterion) {
    bench_sort_group::<VecDeque<usize>>(c, "vecdeque");
    bench_ops_group::<VecDeque<usize>>(c, "vecdeque");
}

// Main benchmark entry point
fn vector_benches(c: &mut Criterion) {
    bench_vector(c);

    if std::env::var("BENCH_STD").is_ok() {
        bench_vecdeque(c);
    }
}

criterion_group!(benches, vector_benches);
criterion_main!(benches);
