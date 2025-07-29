//! Parallel iterators.
//!
//! These are only available when using the `rayon` feature flag.

use super::*;
use ::rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use ::rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

impl<'a, A, P: SharedPointerKind> IntoParallelRefIterator<'a> for GenericVector<A, P>
where
    A: Clone + Send + Sync + 'a,
    P: SharedPointerKind + Send + 'a,
{
    type Item = &'a A;
    type Iter = ParIter<'a, A, P>;

    fn par_iter(&'a self) -> Self::Iter {
        ParIter {
            focus: self.focus(),
        }
    }
}

impl<'a, A, P> IntoParallelRefMutIterator<'a> for GenericVector<A, P>
where
    A: Clone + Send + Sync + 'a,
    P: SharedPointerKind + Send + Sync + 'a,
{
    type Item = &'a mut A;
    type Iter = ParIterMut<'a, A, P>;

    fn par_iter_mut(&'a mut self) -> Self::Iter {
        ParIterMut {
            focus: self.focus_mut(),
        }
    }
}

/// A parallel iterator for [`Vector`][Vector].
///
/// [Vector]: ../type.Vector.html
pub struct ParIter<'a, A, P: SharedPointerKind>
where
    A: Clone + Send + Sync,
{
    focus: Focus<'a, A, P>,
}

impl<'a, A, P> ParallelIterator for ParIter<'a, A, P>
where
    A: Clone + Send + Sync + 'a,
    P: SharedPointerKind + Send + 'a,
{
    type Item = &'a A;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }
}

impl<'a, A, P> IndexedParallelIterator for ParIter<'a, A, P>
where
    A: Clone + Send + Sync + 'a,
    P: SharedPointerKind + Send + 'a,
{
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.focus.len()
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(VectorProducer { focus: self.focus })
    }
}

/// A mutable parallel iterator for [`Vector`][Vector].
///
/// [Vector]: ../type.Vector.html
pub struct ParIterMut<'a, A, P>
where
    A: Clone + Send + Sync,
    P: SharedPointerKind,
{
    focus: FocusMut<'a, A, P>,
}

impl<'a, A, P> ParallelIterator for ParIterMut<'a, A, P>
where
    A: Clone + Send + Sync + 'a,
    P: SharedPointerKind + Send + Sync,
{
    type Item = &'a mut A;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }
}

impl<'a, A, P> IndexedParallelIterator for ParIterMut<'a, A, P>
where
    A: Clone + Send + Sync + 'a,
    P: SharedPointerKind + Send + Sync,
{
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.focus.len()
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(VectorMutProducer { focus: self.focus })
    }
}

struct VectorProducer<'a, A, P>
where
    A: Clone + Send + Sync,
    P: SharedPointerKind,
{
    focus: Focus<'a, A, P>,
}

impl<'a, A, P> Producer for VectorProducer<'a, A, P>
where
    A: Clone + Send + Sync + 'a,
    P: SharedPointerKind + Send + 'a,
{
    type Item = &'a A;
    type IntoIter = Iter<'a, A, P>;

    fn into_iter(self) -> Self::IntoIter {
        self.focus.into_iter()
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.focus.split_at(index);
        (
            VectorProducer { focus: left },
            VectorProducer { focus: right },
        )
    }
}

struct VectorMutProducer<'a, A, P>
where
    A: Clone + Send + Sync,
    P: SharedPointerKind,
{
    focus: FocusMut<'a, A, P>,
}

impl<'a, A, P> Producer for VectorMutProducer<'a, A, P>
where
    A: Clone + Send + Sync + 'a,
    P: SharedPointerKind + Send + Sync,
{
    type Item = &'a mut A;
    type IntoIter = IterMut<'a, A, P>;

    fn into_iter(self) -> Self::IntoIter {
        self.focus.into_iter()
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.focus.split_at(index);
        (
            VectorMutProducer { focus: left },
            VectorMutProducer { focus: right },
        )
    }
}

#[cfg(test)]
mod test {
    use super::super::*;
    use super::proptest::vector;
    use ::proptest::num::i32;
    use ::proptest::proptest;
    use ::rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};

    proptest! {
        #[cfg_attr(miri, ignore)]
        #[test]
        fn par_iter(ref mut input in vector(i32::ANY, 0..10000)) {
            assert_eq!(input.iter().max(), input.par_iter().max())
        }

        #[cfg_attr(miri, ignore)]
        #[test]
        fn par_mut_iter(ref mut input in vector(i32::ANY, 0..10000)) {
            let mut vec = input.clone();
            vec.par_iter_mut().for_each(|i| *i = i.overflowing_add(1).0);
            let expected: Vector<i32> = input.clone().into_iter().map(|i| i.overflowing_add(1).0).collect();
            assert_eq!(expected, vec);
        }
    }
}
