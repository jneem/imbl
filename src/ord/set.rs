// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! An ordered set.
//!
//! An immutable ordered set implemented as a [B-tree] [1].
//!
//! Most operations on this type of set are O(log n). A
//! [`GenericHashSet`] is usually a better choice for
//! performance, but the `OrdSet` has the advantage of only requiring
//! an [`Ord`][std::cmp::Ord] constraint on its values, and of being
//! ordered, so values always come out from lowest to highest, where a
//! [`GenericHashSet`] has no guaranteed ordering.
//!
//! [1]: https://en.wikipedia.org/wiki/B-tree

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections;
use std::fmt::{Debug, Error, Formatter};
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::{FromIterator, Sum};
use std::ops::{Add, Deref, Mul, RangeBounds};

use archery::{SharedPointer, SharedPointerKind};

use crate::hashset::GenericHashSet;
use crate::nodes::btree::{
    BTreeValue, ConsumingIter as ConsumingNodeIter, DiffIter as NodeDiffIter, Insert,
    Iter as NodeIter, Node, Remove,
};
use crate::shared_ptr::DefaultSharedPtr;
#[cfg(has_specialisation)]
use crate::util::linear_search_by;
use crate::util::Pool;

pub use crate::nodes::btree::DiffItem;

/// Construct a set from a sequence of values.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate imbl;
/// # use imbl::ordset::OrdSet;
/// # fn main() {
/// assert_eq!(
///   ordset![1, 2, 3],
///   OrdSet::from(vec![1, 2, 3])
/// );
/// # }
/// ```
#[macro_export]
macro_rules! ordset {
    () => { $crate::ordset::OrdSet::new() };

    ( $($x:expr),* ) => {{
        let mut l = $crate::ordset::OrdSet::new();
        $(
            l.insert($x);
        )*
            l
    }};
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Value<A>(A);

impl<A> Deref for Value<A> {
    type Target = A;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// FIXME lacking specialisation, we can't simply implement `BTreeValue`
// for `A`, we have to use the `Value<A>` indirection.
#[cfg(not(has_specialisation))]
impl<A: Ord> BTreeValue for Value<A> {
    type Key = A;

    fn ptr_eq(&self, _other: &Self) -> bool {
        false
    }

    fn search_key<BK>(slice: &[Self], key: &BK) -> Result<usize, usize>
    where
        BK: Ord + ?Sized,
        Self::Key: Borrow<BK>,
    {
        slice.binary_search_by(|value| Self::Key::borrow(value).cmp(key))
    }

    fn search_value(slice: &[Self], key: &Self) -> Result<usize, usize> {
        slice.binary_search_by(|value| value.cmp(key))
    }

    fn cmp_keys<BK>(&self, other: &BK) -> Ordering
    where
        BK: Ord + ?Sized,
        Self::Key: Borrow<BK>,
    {
        Self::Key::borrow(self).cmp(other)
    }

    fn cmp_values(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

#[cfg(has_specialisation)]
impl<A: Ord> BTreeValue for Value<A> {
    type Key = A;

    fn ptr_eq(&self, _other: &Self) -> bool {
        false
    }

    default fn search_key<BK>(slice: &[Self], key: &BK) -> Result<usize, usize>
    where
        BK: Ord + ?Sized,
        Self::Key: Borrow<BK>,
    {
        slice.binary_search_by(|value| Self::Key::borrow(value).cmp(key))
    }

    default fn search_value(slice: &[Self], key: &Self) -> Result<usize, usize> {
        slice.binary_search_by(|value| value.cmp(key))
    }

    fn cmp_keys<BK>(&self, other: &BK) -> Ordering
    where
        BK: Ord + ?Sized,
        Self::Key: Borrow<BK>,
    {
        Self::Key::borrow(self).cmp(other)
    }

    fn cmp_values(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

#[cfg(has_specialisation)]
impl<A: Ord + Copy> BTreeValue for Value<A> {
    fn search_key<BK>(slice: &[Self], key: &BK) -> Result<usize, usize>
    where
        BK: Ord + ?Sized,
        Self::Key: Borrow<BK>,
    {
        linear_search_by(slice, |value| Self::Key::borrow(value).cmp(key))
    }

    fn search_value(slice: &[Self], key: &Self) -> Result<usize, usize> {
        linear_search_by(slice, |value| value.cmp(key))
    }
}
def_pool!(OrdSetPool<A>, Node<Value<A>, P>);

/// Type alias for [`GenericOrdSet`] that uses [`DefaultSharedPtr`] as the pointer type.
///
/// [GenericOrdSet]: ./struct.GenericOrdSet.html
/// [DefaultSharedPtr]: ../shared_ptr/type.DefaultSharedPtr.html
pub type OrdSet<A> = GenericOrdSet<A, DefaultSharedPtr>;

/// An ordered set.
///
/// An immutable ordered set implemented as a [B-tree] [1].
///
/// Most operations on this type of set are O(log n). A
/// [`GenericHashSet`] is usually a better choice for
/// performance, but the `OrdSet` has the advantage of only requiring
/// an [`Ord`][std::cmp::Ord] constraint on its values, and of being
/// ordered, so values always come out from lowest to highest, where a
/// [`GenericHashSet`] has no guaranteed ordering.
///
/// [1]: https://en.wikipedia.org/wiki/B-tree
pub struct GenericOrdSet<A, P: SharedPointerKind> {
    size: usize,
    pool: OrdSetPool<A, P>,
    root: SharedPointer<Node<Value<A>, P>, P>,
}

impl<A, P: SharedPointerKind> GenericOrdSet<A, P> {
    /// Construct an empty set.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        let pool = OrdSetPool::default();
        let root = SharedPointer::default();
        GenericOrdSet {
            size: 0,
            pool,
            root,
        }
    }

    /// Construct an empty set using a specific memory pool.
    #[cfg(feature = "pool")]
    #[must_use]
    pub fn with_pool(pool: &OrdSetPool<A>) -> Self {
        let root = SharedPointer::default();
        GenericOrdSet {
            size: 0,
            pool: pool.clone(),
            root,
        }
    }

    /// Construct a set with a single value.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # type OrdSet<T> = imbl::ordset::OrdSet<T>;
    /// let set = OrdSet::unit(123);
    /// assert!(set.contains(&123));
    /// ```
    #[inline]
    #[must_use]
    pub fn unit(a: A) -> Self {
        let pool = OrdSetPool::default();
        let root = SharedPointer::new(Node::unit(Value(a)));
        GenericOrdSet {
            size: 1,
            pool,
            root,
        }
    }

    /// Test whether a set is empty.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordset::OrdSet;
    /// assert!(
    ///   !ordset![1, 2, 3].is_empty()
    /// );
    /// assert!(
    ///   OrdSet::<i32>::new().is_empty()
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the size of a set.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordset::OrdSet;
    /// assert_eq!(3, ordset![1, 2, 3].len());
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Test whether two sets refer to the same content in memory.
    ///
    /// This is true if the two sides are references to the same set,
    /// or if the two sets refer to the same root node.
    ///
    /// This would return true if you're comparing a set to itself, or
    /// if you're comparing a set to a fresh clone of itself.
    ///
    /// Time: O(1)
    pub fn ptr_eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other) || SharedPointer::ptr_eq(&self.root, &other.root)
    }

    /// Get a reference to the memory pool used by this set.
    ///
    /// Note that if you didn't specifically construct it with a pool, you'll
    /// get back a reference to a pool of size 0.
    #[cfg(feature = "pool")]
    pub fn pool(&self) -> &OrdSetPool<A> {
        &self.pool
    }

    /// Discard all elements from the set.
    ///
    /// This leaves you with an empty set, and all elements that
    /// were previously inside it are dropped.
    ///
    /// Time: O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::OrdSet;
    /// let mut set = ordset![1, 2, 3];
    /// set.clear();
    /// assert!(set.is_empty());
    /// ```
    pub fn clear(&mut self) {
        if !self.is_empty() {
            self.root = SharedPointer::default();
            self.size = 0;
        }
    }
}

impl<A, P> GenericOrdSet<A, P>
where
    A: Ord,
    P: SharedPointerKind,
{
    /// Get the smallest value in a set.
    ///
    /// If the set is empty, returns `None`.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn get_min(&self) -> Option<&A> {
        self.root.min().map(Deref::deref)
    }

    /// Get the largest value in a set.
    ///
    /// If the set is empty, returns `None`.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn get_max(&self) -> Option<&A> {
        self.root.max().map(Deref::deref)
    }

    /// Create an iterator over the contents of the set.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, A, P> {
        Iter {
            it: NodeIter::new(&self.root, self.size, ..),
        }
    }

    /// Create an iterator over a range inside the set.
    #[must_use]
    pub fn range<R, BA>(&self, range: R) -> RangedIter<'_, A, P>
    where
        R: RangeBounds<BA>,
        A: Borrow<BA>,
        BA: Ord + ?Sized,
    {
        RangedIter {
            it: NodeIter::new(&self.root, self.size, range),
        }
    }

    /// Get an iterator over the differences between this set and
    /// another, i.e. the set of entries to add or remove to this set
    /// in order to make it equal to the other set.
    ///
    /// This function will avoid visiting nodes which are shared
    /// between the two sets, meaning that even very large sets can be
    /// compared quickly if most of their structure is shared.
    ///
    /// Time: O(n) (where n is the number of unique elements across
    /// the two sets, minus the number of elements belonging to nodes
    /// shared between them)
    #[must_use]
    pub fn diff<'a, 'b>(&'a self, other: &'b Self) -> DiffIter<'a, 'b, A, P> {
        DiffIter {
            it: NodeDiffIter::new(&self.root, &other.root),
        }
    }

    /// Test if a value is part of a set.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordset::OrdSet;
    /// let mut set = ordset!{1, 2, 3};
    /// assert!(set.contains(&1));
    /// assert!(!set.contains(&4));
    /// ```
    #[inline]
    #[must_use]
    pub fn contains<BA>(&self, a: &BA) -> bool
    where
        BA: Ord + ?Sized,
        A: Borrow<BA>,
    {
        self.root.lookup(a).is_some()
    }

    /// Returns a reference to the element in the set, if any, that is equal to the value.
    /// The value may be any borrowed form of the set’s element type, but the ordering on
    /// the borrowed form must match the ordering on the element type.
    ///
    /// This is useful when the elements in the set are unique by for example an id,
    /// and you want to get the element out of the set by using the id.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use std::borrow::Borrow;
    /// # use std::cmp::Ordering;
    /// # use imbl::ordset::OrdSet;
    /// # #[derive(Clone)]
    /// // Implements Eq and ord by delegating to id
    /// struct FancyItem {
    ///     id: u32,
    ///     data: String,
    /// }
    /// # impl Eq for FancyItem {}
    /// # impl PartialEq<Self> for FancyItem {fn eq(&self, other: &Self) -> bool { self.id.eq(&other.id)}}
    /// # impl PartialOrd<Self> for FancyItem {fn partial_cmp(&self, other: &Self) -> Option<Ordering> {self.id.partial_cmp(&other.id)}}
    /// # impl Ord for FancyItem {fn cmp(&self, other: &Self) -> Ordering {self.id.cmp(&other.id)}}
    /// # impl Borrow<u32> for FancyItem {fn borrow(&self) -> &u32 {&self.id}}
    /// let mut set = ordset!{
    ///     FancyItem {id: 0, data: String::from("Hello")},
    ///     FancyItem {id: 1, data: String::from("Test")}
    /// };
    /// assert_eq!(set.get(&1).unwrap().data, "Test");
    /// assert_eq!(set.get(&0).unwrap().data, "Hello");
    ///
    /// ```
    pub fn get<BK>(&self, k: &BK) -> Option<&A>
    where
        BK: Ord + ?Sized,
        A: Borrow<BK>,
    {
        self.root.lookup(k).map(|v| &v.0)
    }

    /// Get the closest smaller value in a set to a given value.
    ///
    /// If the set contains the given value, this is returned.
    /// Otherwise, the closest value in the set smaller than the
    /// given value is returned. If the smallest value in the set
    /// is larger than the given value, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::OrdSet;
    /// let set = ordset![1, 3, 5, 7, 9];
    /// assert_eq!(Some(&5), set.get_prev(&6));
    /// ```
    #[must_use]
    pub fn get_prev<BK>(&self, k: &BK) -> Option<&A>
    where
        BK: Ord + ?Sized,
        A: Borrow<BK>,
    {
        self.root.lookup_prev(k).map(|v| &v.0)
    }

    /// Get the closest larger value in a set to a given value.
    ///
    /// If the set contains the given value, this is returned.
    /// Otherwise, the closest value in the set larger than the
    /// given value is returned. If the largest value in the set
    /// is smaller than the given value, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::OrdSet;
    /// let set = ordset![1, 3, 5, 7, 9];
    /// assert_eq!(Some(&5), set.get_next(&4));
    /// ```
    #[must_use]
    pub fn get_next<BK>(&self, k: &BK) -> Option<&A>
    where
        BK: Ord + ?Sized,
        A: Borrow<BK>,
    {
        self.root.lookup_next(k).map(|v| &v.0)
    }

    /// Test whether a set is a subset of another set, meaning that
    /// all values in our set must also be in the other set.
    ///
    /// Time: O(n log m) where m is the size of the other set
    #[must_use]
    pub fn is_subset<RS>(&self, other: RS) -> bool
    where
        RS: Borrow<Self>,
    {
        let other = other.borrow();
        if other.len() < self.len() {
            return false;
        }
        self.iter().all(|a| other.contains(a))
    }

    /// Test whether a set is a proper subset of another set, meaning
    /// that all values in our set must also be in the other set. A
    /// proper subset must also be smaller than the other set.
    ///
    /// Time: O(n log m) where m is the size of the other set
    #[must_use]
    pub fn is_proper_subset<RS>(&self, other: RS) -> bool
    where
        RS: Borrow<Self>,
    {
        self.len() != other.borrow().len() && self.is_subset(other)
    }
}

impl<A, P> GenericOrdSet<A, P>
where
    A: Ord + Clone,
    P: SharedPointerKind,
{
    /// Insert a value into a set.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordset::OrdSet;
    /// let mut set = ordset!{};
    /// set.insert(123);
    /// set.insert(456);
    /// assert_eq!(
    ///   set,
    ///   ordset![123, 456]
    /// );
    /// ```
    #[inline]
    pub fn insert(&mut self, a: A) -> Option<A> {
        let new_root = {
            let root = SharedPointer::make_mut(&mut self.root);
            match root.insert(&self.pool.0, Value(a)) {
                Insert::Replaced(Value(old_value)) => return Some(old_value),
                Insert::Added => {
                    self.size += 1;
                    return None;
                }
                Insert::Split(left, median, right) => {
                    SharedPointer::new(Node::new_from_split(&self.pool.0, left, median, right))
                }
            }
        };
        self.size += 1;
        self.root = new_root;
        None
    }

    /// Remove a value from a set.
    ///
    /// Time: O(log n)
    #[inline]
    pub fn remove<BA>(&mut self, a: &BA) -> Option<A>
    where
        BA: Ord + ?Sized,
        A: Borrow<BA>,
    {
        let (new_root, removed_value) = {
            let root = SharedPointer::make_mut(&mut self.root);
            match root.remove(&self.pool.0, a) {
                Remove::Update(value, root) => (SharedPointer::new(root), Some(value.0)),
                Remove::Removed(value) => {
                    self.size -= 1;
                    return Some(value.0);
                }
                Remove::NoChange => return None,
            }
        };
        self.size -= 1;
        self.root = new_root;
        removed_value
    }

    /// Remove the smallest value from a set.
    ///
    /// Time: O(log n)
    pub fn remove_min(&mut self) -> Option<A> {
        // FIXME implement this at the node level for better efficiency
        let key = match self.get_min() {
            None => return None,
            Some(v) => v,
        }
        .clone();
        self.remove(&key)
    }

    /// Remove the largest value from a set.
    ///
    /// Time: O(log n)
    pub fn remove_max(&mut self) -> Option<A> {
        // FIXME implement this at the node level for better efficiency
        let key = match self.get_max() {
            None => return None,
            Some(v) => v,
        }
        .clone();
        self.remove(&key)
    }

    /// Construct a new set from the current set with the given value
    /// added.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordset::OrdSet;
    /// let set = ordset![456];
    /// assert_eq!(
    ///   set.update(123),
    ///   ordset![123, 456]
    /// );
    /// ```
    #[must_use]
    pub fn update(&self, a: A) -> Self {
        let mut out = self.clone();
        out.insert(a);
        out
    }

    /// Construct a new set with the given value removed if it's in
    /// the set.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn without<BA>(&self, a: &BA) -> Self
    where
        BA: Ord + ?Sized,
        A: Borrow<BA>,
    {
        let mut out = self.clone();
        out.remove(a);
        out
    }

    /// Remove the smallest value from a set, and return that value as
    /// well as the updated set.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn without_min(&self) -> (Option<A>, Self) {
        match self.get_min() {
            Some(v) => (Some(v.clone()), self.without(v)),
            None => (None, self.clone()),
        }
    }

    /// Remove the largest value from a set, and return that value as
    /// well as the updated set.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn without_max(&self) -> (Option<A>, Self) {
        match self.get_max() {
            Some(v) => (Some(v.clone()), self.without(v)),
            None => (None, self.clone()),
        }
    }

    /// Construct the union of two sets.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordset::OrdSet;
    /// let set1 = ordset!{1, 2};
    /// let set2 = ordset!{2, 3};
    /// let expected = ordset!{1, 2, 3};
    /// assert_eq!(expected, set1.union(set2));
    /// ```
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        let (mut to_mutate, to_consume) = if self.len() >= other.len() {
            (self, other)
        } else {
            (other, self)
        };
        for value in to_consume {
            to_mutate.insert(value);
        }
        to_mutate
    }

    /// Construct the union of multiple sets.
    ///
    /// Time: O(n log n)
    #[must_use]
    pub fn unions<I>(i: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        i.into_iter().fold(Self::default(), Self::union)
    }

    /// Construct the symmetric difference between two sets.
    ///
    /// This is an alias for the
    /// [`symmetric_difference`][symmetric_difference] method.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordset::OrdSet;
    /// let set1 = ordset!{1, 2};
    /// let set2 = ordset!{2, 3};
    /// let expected = ordset!{1, 3};
    /// assert_eq!(expected, set1.difference(set2));
    /// ```
    ///
    /// [symmetric_difference]: #method.symmetric_difference
    #[deprecated(
        since = "2.0.1",
        note = "to avoid conflicting behaviors between std and imbl, the `difference` alias for `symmetric_difference` will be removed."
    )]
    #[must_use]
    pub fn difference(self, other: Self) -> Self {
        self.symmetric_difference(other)
    }

    /// Construct the symmetric difference between two sets.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordset::OrdSet;
    /// let set1 = ordset!{1, 2};
    /// let set2 = ordset!{2, 3};
    /// let expected = ordset!{1, 3};
    /// assert_eq!(expected, set1.symmetric_difference(set2));
    /// ```
    #[must_use]
    pub fn symmetric_difference(mut self, other: Self) -> Self {
        for value in other {
            if self.remove(&value).is_none() {
                self.insert(value);
            }
        }
        self
    }

    /// Construct the relative complement between two sets, that is the set
    /// of values in `self` that do not occur in `other`.
    ///
    /// Time: O(m log n) where m is the size of the other set
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordset::OrdSet;
    /// let set1 = ordset!{1, 2};
    /// let set2 = ordset!{2, 3};
    /// let expected = ordset!{1};
    /// assert_eq!(expected, set1.relative_complement(set2));
    /// ```
    #[must_use]
    pub fn relative_complement(mut self, other: Self) -> Self {
        for value in other {
            let _ = self.remove(&value);
        }
        self
    }

    /// Construct the intersection of two sets.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordset::OrdSet;
    /// let set1 = ordset!{1, 2};
    /// let set2 = ordset!{2, 3};
    /// let expected = ordset!{2};
    /// assert_eq!(expected, set1.intersection(set2));
    /// ```
    #[must_use]
    pub fn intersection(self, other: Self) -> Self {
        let mut out = Self::default();
        for value in other {
            if self.contains(&value) {
                out.insert(value);
            }
        }
        out
    }

    /// Split a set into two, with the left hand set containing values
    /// which are smaller than `split`, and the right hand set
    /// containing values which are larger than `split`.
    ///
    /// The `split` value itself is discarded.
    ///
    /// Time: O(n)
    #[must_use]
    pub fn split<BA>(self, split: &BA) -> (Self, Self)
    where
        BA: Ord + ?Sized,
        A: Borrow<BA>,
    {
        let (left, _, right) = self.split_member(split);
        (left, right)
    }

    /// Split a set into two, with the left hand set containing values
    /// which are smaller than `split`, and the right hand set
    /// containing values which are larger than `split`.
    ///
    /// Returns a tuple of the two sets and a boolean which is true if
    /// the `split` value existed in the original set, and false
    /// otherwise.
    ///
    /// Time: O(n)
    #[must_use]
    pub fn split_member<BA>(self, split: &BA) -> (Self, bool, Self)
    where
        BA: Ord + ?Sized,
        A: Borrow<BA>,
    {
        let mut left = Self::default();
        let mut right = Self::default();
        let mut present = false;
        for value in self {
            match value.borrow().cmp(&split) {
                Ordering::Less => {
                    left.insert(value);
                }
                Ordering::Equal => {
                    present = true;
                }
                Ordering::Greater => {
                    right.insert(value);
                }
            }
        }
        (left, present, right)
    }

    /// Construct a set with only the `n` smallest values from a given
    /// set.
    ///
    /// Time: O(n)
    #[must_use]
    pub fn take(&self, n: usize) -> Self {
        self.iter().take(n).cloned().collect()
    }

    /// Construct a set with the `n` smallest values removed from a
    /// given set.
    ///
    /// Time: O(n)
    #[must_use]
    pub fn skip(&self, n: usize) -> Self {
        self.iter().skip(n).cloned().collect()
    }
}

// Core traits

impl<A, P: SharedPointerKind> Clone for GenericOrdSet<A, P> {
    /// Clone a set.
    ///
    /// Time: O(1)
    #[inline]
    fn clone(&self) -> Self {
        GenericOrdSet {
            size: self.size,
            pool: self.pool.clone(),
            root: self.root.clone(),
        }
    }
}

// TODO: Support PartialEq for OrdSet that have different P
impl<A: Ord, P: SharedPointerKind> PartialEq for GenericOrdSet<A, P> {
    fn eq(&self, other: &Self) -> bool {
        SharedPointer::ptr_eq(&self.root, &other.root)
            || (self.len() == other.len() && self.diff(other).next().is_none())
    }
}

impl<A: Ord, P: SharedPointerKind> Eq for GenericOrdSet<A, P> {}

impl<A: Ord, P: SharedPointerKind> PartialOrd for GenericOrdSet<A, P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<A: Ord, P: SharedPointerKind> Ord for GenericOrdSet<A, P> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<A: Ord + Hash, P: SharedPointerKind> Hash for GenericOrdSet<A, P> {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        for i in self.iter() {
            i.hash(state);
        }
    }
}

impl<A, P: SharedPointerKind> Default for GenericOrdSet<A, P> {
    fn default() -> Self {
        GenericOrdSet::new()
    }
}

impl<A: Ord + Clone, P: SharedPointerKind> Add for GenericOrdSet<A, P> {
    type Output = GenericOrdSet<A, P>;

    fn add(self, other: Self) -> Self::Output {
        self.union(other)
    }
}

impl<'a, A: Ord + Clone, P: SharedPointerKind> Add for &'a GenericOrdSet<A, P> {
    type Output = GenericOrdSet<A, P>;

    fn add(self, other: Self) -> Self::Output {
        self.clone().union(other.clone())
    }
}

impl<A: Ord + Clone, P: SharedPointerKind> Mul for GenericOrdSet<A, P> {
    type Output = GenericOrdSet<A, P>;

    fn mul(self, other: Self) -> Self::Output {
        self.intersection(other)
    }
}

impl<'a, A: Ord + Clone, P: SharedPointerKind> Mul for &'a GenericOrdSet<A, P> {
    type Output = GenericOrdSet<A, P>;

    fn mul(self, other: Self) -> Self::Output {
        self.clone().intersection(other.clone())
    }
}

impl<A: Ord + Clone, P: SharedPointerKind> Sum for GenericOrdSet<A, P> {
    fn sum<I>(it: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        it.fold(Self::new(), |a, b| a + b)
    }
}

impl<A, R, P> Extend<R> for GenericOrdSet<A, P>
where
    A: Ord + Clone + From<R>,
    P: SharedPointerKind,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = R>,
    {
        for value in iter {
            self.insert(From::from(value));
        }
    }
}

impl<A: Ord + Debug, P: SharedPointerKind> Debug for GenericOrdSet<A, P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.debug_set().entries(self.iter()).finish()
    }
}

// Iterators

/// An iterator over the elements of a set.
pub struct Iter<'a, A, P: SharedPointerKind> {
    it: NodeIter<'a, Value<A>, P>,
}

// We impl Clone instead of deriving it, because we want Clone even if K and V aren't.
impl<'a, A, P: SharedPointerKind> Clone for Iter<'a, A, P> {
    fn clone(&self) -> Self {
        Iter {
            it: self.it.clone(),
        }
    }
}

impl<'a, A, P: SharedPointerKind> Iterator for Iter<'a, A, P>
where
    A: 'a + Ord,
{
    type Item = &'a A;

    /// Advance the iterator and return the next value.
    ///
    /// Time: O(1)*
    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(Deref::deref)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.it.remaining, Some(self.it.remaining))
    }
}

impl<'a, A, P> DoubleEndedIterator for Iter<'a, A, P>
where
    A: 'a + Ord,
    P: SharedPointerKind,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.it.next_back().map(Deref::deref)
    }
}

impl<'a, A, P> ExactSizeIterator for Iter<'a, A, P>
where
    A: 'a + Ord,
    P: SharedPointerKind,
{
}

/// A ranged iterator over the elements of a set.
///
/// The only difference from `Iter` is that this one doesn't implement
/// `ExactSizeIterator` because we can't know the size of the range without first
/// iterating over it to count.
pub struct RangedIter<'a, A, P: SharedPointerKind> {
    it: NodeIter<'a, Value<A>, P>,
}

impl<'a, A, P> Iterator for RangedIter<'a, A, P>
where
    A: 'a + Ord,
    P: SharedPointerKind,
{
    type Item = &'a A;

    /// Advance the iterator and return the next value.
    ///
    /// Time: O(1)*
    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(Deref::deref)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, A, P> DoubleEndedIterator for RangedIter<'a, A, P>
where
    A: 'a + Ord,
    P: SharedPointerKind,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.it.next_back().map(Deref::deref)
    }
}

/// A consuming iterator over the elements of a set.
pub struct ConsumingIter<A, P: SharedPointerKind> {
    it: ConsumingNodeIter<Value<A>, P>,
}

impl<A, P> Iterator for ConsumingIter<A, P>
where
    A: Ord + Clone,
    P: SharedPointerKind,
{
    type Item = A;

    /// Advance the iterator and return the next value.
    ///
    /// Time: O(1)*
    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|v| v.0)
    }
}

/// An iterator over the difference between two sets.
pub struct DiffIter<'a, 'b, A, P: SharedPointerKind> {
    it: NodeDiffIter<'a, 'b, Value<A>, P>,
}

impl<'a, 'b, A, P> Iterator for DiffIter<'a, 'b, A, P>
where
    A: Ord + PartialEq,
    P: SharedPointerKind,
{
    type Item = DiffItem<'a, 'b, A>;

    /// Advance the iterator and return the next value.
    ///
    /// Time: O(1)*
    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|item| match item {
            DiffItem::Add(v) => DiffItem::Add(v.deref()),
            DiffItem::Update { old, new } => DiffItem::Update {
                old: old.deref(),
                new: new.deref(),
            },
            DiffItem::Remove(v) => DiffItem::Remove(v.deref()),
        })
    }
}

impl<A, R, P> FromIterator<R> for GenericOrdSet<A, P>
where
    A: Ord + Clone + From<R>,
    P: SharedPointerKind,
{
    fn from_iter<T>(i: T) -> Self
    where
        T: IntoIterator<Item = R>,
    {
        let mut out = Self::new();
        for item in i {
            out.insert(From::from(item));
        }
        out
    }
}

impl<'a, A, P> IntoIterator for &'a GenericOrdSet<A, P>
where
    A: 'a + Ord,
    P: SharedPointerKind,
{
    type Item = &'a A;
    type IntoIter = Iter<'a, A, P>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<A, P> IntoIterator for GenericOrdSet<A, P>
where
    A: Ord + Clone,
    P: SharedPointerKind,
{
    type Item = A;
    type IntoIter = ConsumingIter<A, P>;

    fn into_iter(self) -> Self::IntoIter {
        ConsumingIter {
            it: ConsumingNodeIter::new(&self.root, self.size),
        }
    }
}

// Conversions

impl<'s, 'a, A, OA, P1, P2> From<&'s GenericOrdSet<&'a A, P2>> for GenericOrdSet<OA, P1>
where
    A: ToOwned<Owned = OA> + Ord + ?Sized,
    OA: Borrow<A> + Ord + Clone,
    P1: SharedPointerKind,
    P2: SharedPointerKind,
{
    fn from(set: &GenericOrdSet<&A, P2>) -> Self {
        set.iter().map(|a| (*a).to_owned()).collect()
    }
}

impl<'a, A, P> From<&'a [A]> for GenericOrdSet<A, P>
where
    A: Ord + Clone,
    P: SharedPointerKind,
{
    fn from(slice: &'a [A]) -> Self {
        slice.iter().cloned().collect()
    }
}

impl<A: Ord + Clone, P: SharedPointerKind> From<Vec<A>> for GenericOrdSet<A, P> {
    fn from(vec: Vec<A>) -> Self {
        vec.into_iter().collect()
    }
}

impl<'a, A: Ord + Clone, P: SharedPointerKind> From<&'a Vec<A>> for GenericOrdSet<A, P> {
    fn from(vec: &Vec<A>) -> Self {
        vec.iter().cloned().collect()
    }
}

impl<A: Eq + Hash + Ord + Clone, P: SharedPointerKind> From<collections::HashSet<A>>
    for GenericOrdSet<A, P>
{
    fn from(hash_set: collections::HashSet<A>) -> Self {
        hash_set.into_iter().collect()
    }
}

impl<'a, A: Eq + Hash + Ord + Clone, P: SharedPointerKind> From<&'a collections::HashSet<A>>
    for GenericOrdSet<A, P>
{
    fn from(hash_set: &collections::HashSet<A>) -> Self {
        hash_set.iter().cloned().collect()
    }
}

impl<A: Ord + Clone, P: SharedPointerKind> From<collections::BTreeSet<A>> for GenericOrdSet<A, P> {
    fn from(btree_set: collections::BTreeSet<A>) -> Self {
        btree_set.into_iter().collect()
    }
}

impl<'a, A: Ord + Clone, P: SharedPointerKind> From<&'a collections::BTreeSet<A>>
    for GenericOrdSet<A, P>
{
    fn from(btree_set: &collections::BTreeSet<A>) -> Self {
        btree_set.iter().cloned().collect()
    }
}

impl<A: Hash + Eq + Ord + Clone, S: BuildHasher, P1: SharedPointerKind, P2: SharedPointerKind>
    From<GenericHashSet<A, S, P2>> for GenericOrdSet<A, P1>
{
    fn from(hashset: GenericHashSet<A, S, P2>) -> Self {
        hashset.into_iter().collect()
    }
}

impl<
        'a,
        A: Hash + Eq + Ord + Clone,
        S: BuildHasher,
        P1: SharedPointerKind,
        P2: SharedPointerKind,
    > From<&'a GenericHashSet<A, S, P2>> for GenericOrdSet<A, P1>
{
    fn from(hashset: &GenericHashSet<A, S, P2>) -> Self {
        hashset.into_iter().cloned().collect()
    }
}

// Proptest
#[cfg(any(test, feature = "proptest"))]
#[doc(hidden)]
pub mod proptest {
    #[deprecated(
        since = "14.3.0",
        note = "proptest strategies have moved to imbl::proptest"
    )]
    pub use crate::proptest::ord_set;
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::proptest::*;
    use ::proptest::proptest;
    use static_assertions::{assert_impl_all, assert_not_impl_any};

    assert_impl_all!(OrdSet<i32>: Send, Sync);
    assert_not_impl_any!(OrdSet<*const i32>: Send, Sync);
    assert_covariant!(OrdSet<T> in T);

    #[test]
    fn match_strings_with_string_slices() {
        let mut set: OrdSet<String> = From::from(&ordset!["foo", "bar"]);
        set = set.without("bar");
        assert!(!set.contains("bar"));
        set.remove("foo");
        assert!(!set.contains("foo"));
    }

    #[test]
    fn ranged_iter() {
        let set = ordset![1, 2, 3, 4, 5];
        let range: Vec<i32> = set.range(..).cloned().collect();
        assert_eq!(vec![1, 2, 3, 4, 5], range);
        let range: Vec<i32> = set.range(..).rev().cloned().collect();
        assert_eq!(vec![5, 4, 3, 2, 1], range);
        let range: Vec<i32> = set.range(2..5).cloned().collect();
        assert_eq!(vec![2, 3, 4], range);
        let range: Vec<i32> = set.range(2..5).rev().cloned().collect();
        assert_eq!(vec![4, 3, 2], range);
        let range: Vec<i32> = set.range(3..).cloned().collect();
        assert_eq!(vec![3, 4, 5], range);
        let range: Vec<i32> = set.range(3..).rev().cloned().collect();
        assert_eq!(vec![5, 4, 3], range);
        let range: Vec<i32> = set.range(..4).cloned().collect();
        assert_eq!(vec![1, 2, 3], range);
        let range: Vec<i32> = set.range(..4).rev().cloned().collect();
        assert_eq!(vec![3, 2, 1], range);
        let range: Vec<i32> = set.range(..=3).cloned().collect();
        assert_eq!(vec![1, 2, 3], range);
        let range: Vec<i32> = set.range(..=3).rev().cloned().collect();
        assert_eq!(vec![3, 2, 1], range);
    }

    proptest! {
        #[test]
        fn proptest_a_set(ref s in ord_set(".*", 10..100)) {
            assert!(s.len() < 100);
            assert!(s.len() >= 10);
            s.root.check_sane();
        }

        #[test]
        fn long_ranged_iter(max in 1..1000) {
            let range = 0..max;
            let expected: Vec<i32> = range.clone().collect();
            let set: OrdSet<i32> = OrdSet::from_iter(range.clone());
            set.root.check_sane();
            let result: Vec<i32> = set.range(..).cloned().collect();
            assert_eq!(expected, result);

            let expected: Vec<i32> = range.clone().rev().collect();
            let set: OrdSet<i32> = OrdSet::from_iter(range);
            set.root.check_sane();
            let result: Vec<i32> = set.range(..).rev().cloned().collect();
            assert_eq!(expected, result);
        }
    }
}
