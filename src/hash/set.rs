// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! An unordered set.
//!
//! An immutable hash set using [hash array mapped tries] [1].
//!
//! Most operations on this set are O(log<sub>x</sub> n) for a
//! suitably high *x* that it should be nearly O(1) for most sets.
//! Because of this, it's a great choice for a generic set as long as
//! you don't mind that values will need to implement
//! [`Hash`][std::hash::Hash] and [`Eq`][std::cmp::Eq].
//!
//! Values will have a predictable order based on the hasher
//! being used. Unless otherwise specified, this will be the standard
//! [`RandomState`][std::collections::hash_map::RandomState] hasher.
//!
//! [1]: https://en.wikipedia.org/wiki/Hash_array_mapped_trie
//! [std::cmp::Eq]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
//! [std::hash::Hash]: https://doc.rust-lang.org/std/hash/trait.Hash.html
//! [std::collections::hash_map::RandomState]: https://doc.rust-lang.org/std/collections/hash_map/struct.RandomState.html

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::collections::{self, BTreeSet};
use std::fmt::{Debug, Error, Formatter};
use std::hash::{BuildHasher, Hash};
use std::iter::{FromIterator, FusedIterator, Sum};
use std::ops::{Add, Deref, Mul};

use archery::{SharedPointer, SharedPointerKind};

use crate::nodes::hamt::{hash_key, Drain as NodeDrain, HashValue, Iter as NodeIter, Node};
use crate::ordset::GenericOrdSet;
use crate::shared_ptr::DefaultSharedPtr;
use crate::GenericVector;

/// Construct a set from a sequence of values.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate imbl;
/// # use imbl::HashSet;
/// # fn main() {
/// assert_eq!(
///   hashset![1, 2, 3],
///   HashSet::from(vec![1, 2, 3])
/// );
/// # }
/// ```
#[macro_export]
macro_rules! hashset {
    () => { $crate::hashset::HashSet::new() };

    ( $($x:expr),* ) => {{
        let mut l = $crate::hashset::HashSet::new();
        $(
            l.insert($x);
        )*
            l
    }};

    ( $($x:expr ,)* ) => {{
        let mut l = $crate::hashset::HashSet::new();
        $(
            l.insert($x);
        )*
            l
    }};
}

/// Type alias for [`GenericHashSet`] that uses [`std::hash::RandomState`] as the default hasher and [`DefaultSharedPtr`] as the pointer type.
///
/// [GenericHashSet]: ./struct.GenericHashSet.html
/// [`std::hash::RandomState`]: https://doc.rust-lang.org/stable/std/collections/hash_map/struct.RandomState.html
/// [DefaultSharedPtr]: ../shared_ptr/type.DefaultSharedPtr.html
pub type HashSet<A> = GenericHashSet<A, RandomState, DefaultSharedPtr>;

/// An unordered set.
///
/// An immutable hash set using [hash array mapped tries] [1].
///
/// Most operations on this set are O(log<sub>x</sub> n) for a
/// suitably high *x* that it should be nearly O(1) for most sets.
/// Because of this, it's a great choice for a generic set as long as
/// you don't mind that values will need to implement
/// [`Hash`][std::hash::Hash] and [`Eq`][std::cmp::Eq].
///
/// Values will have a predictable order based on the hasher
/// being used. Unless otherwise specified, this will be the standard
/// [`RandomState`][std::collections::hash_map::RandomState] hasher.
///
/// [1]: https://en.wikipedia.org/wiki/Hash_array_mapped_trie
/// [std::cmp::Eq]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
/// [std::hash::Hash]: https://doc.rust-lang.org/std/hash/trait.Hash.html
/// [std::collections::hash_map::RandomState]: https://doc.rust-lang.org/std/collections/hash_map/struct.RandomState.html
pub struct GenericHashSet<A, S, P: SharedPointerKind> {
    hasher: S,
    root: Option<SharedPointer<Node<Value<A>, P>, P>>,
    size: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Value<A>(A);

impl<A> Deref for Value<A> {
    type Target = A;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// FIXME lacking specialisation, we can't simply implement `HashValue`
// for `A`, we have to use the `Value<A>` indirection.
impl<A> HashValue for Value<A>
where
    A: Hash + Eq,
{
    type Key = A;

    fn extract_key(&self) -> &Self::Key {
        &self.0
    }

    fn ptr_eq(&self, _other: &Self) -> bool {
        false
    }
}

impl<A, S, P> GenericHashSet<A, S, P>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    /// Construct a set with a single value.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::hashset::HashSet;
    /// # use std::sync::Arc;
    /// let set = HashSet::unit(123);
    /// assert!(set.contains(&123));
    /// ```
    #[inline]
    #[must_use]
    pub fn unit(a: A) -> Self {
        GenericHashSet::new().update(a)
    }
}

impl<A, S, P: SharedPointerKind> GenericHashSet<A, S, P> {
    /// Construct an empty set.
    #[must_use]
    pub fn new() -> Self
    where
        S: Default,
    {
        Self::default()
    }

    /// Test whether a set is empty.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::hashset::HashSet;
    /// assert!(
    ///   !hashset![1, 2, 3].is_empty()
    /// );
    /// assert!(
    ///   HashSet::<i32>::new().is_empty()
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
    /// # use imbl::hashset::HashSet;
    /// assert_eq!(3, hashset![1, 2, 3].len());
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
        match (&self.root, &other.root) {
            (Some(a), Some(b)) => SharedPointer::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        }
    }

    /// Construct an empty hash set using the provided hasher.
    #[inline]
    #[must_use]
    pub fn with_hasher(hasher: S) -> Self {
        GenericHashSet {
            size: 0,
            root: None,
            hasher,
        }
    }

    /// Get a reference to the set's [`BuildHasher`][BuildHasher].
    ///
    /// [BuildHasher]: https://doc.rust-lang.org/std/hash/trait.BuildHasher.html
    #[must_use]
    pub fn hasher(&self) -> &S {
        &self.hasher
    }

    /// Construct an empty hash set using the same hasher as the current hash set.
    #[inline]
    #[must_use]
    pub fn new_from<A2>(&self) -> GenericHashSet<A2, S, P>
    where
        A2: Hash + Eq + Clone,
        S: Clone,
    {
        GenericHashSet {
            size: 0,
            root: None,
            hasher: self.hasher.clone(),
        }
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
    /// # use imbl::HashSet;
    /// let mut set = hashset![1, 2, 3];
    /// set.clear();
    /// assert!(set.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.root = None;
        self.size = 0;
    }

    /// Get an iterator over the values in a hash set.
    ///
    /// Please note that the order is consistent between sets using
    /// the same hasher, but no other ordering guarantee is offered.
    /// Items will not come out in insertion order or sort order.
    /// They will, however, come out in the same order every time for
    /// the same set.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, A, P> {
        Iter {
            it: NodeIter::new(self.root.as_deref(), self.size),
        }
    }
}

impl<A, S, P> GenericHashSet<A, S, P>
where
    A: Hash + Eq,
    S: BuildHasher,
    P: SharedPointerKind,
{
    fn test_eq<S2: BuildHasher, P2: SharedPointerKind>(
        &self,
        other: &GenericHashSet<A, S2, P2>,
    ) -> bool {
        if self.len() != other.len() {
            return false;
        }
        let mut seen = collections::HashSet::new();
        for value in self.iter() {
            if !other.contains(value) {
                return false;
            }
            seen.insert(value);
        }
        for value in other.iter() {
            if !seen.contains(&value) {
                return false;
            }
        }
        true
    }

    /// Test if a value is part of a set.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn contains<BA>(&self, a: &BA) -> bool
    where
        BA: Hash + Eq + ?Sized,
        A: Borrow<BA>,
    {
        if let Some(root) = &self.root {
            root.get(hash_key(&self.hasher, a), 0, a).is_some()
        } else {
            false
        }
    }

    /// Test whether a set is a subset of another set, meaning that
    /// all values in our set must also be in the other set.
    ///
    /// Time: O(n log n)
    #[must_use]
    pub fn is_subset<RS>(&self, other: RS) -> bool
    where
        RS: Borrow<Self>,
    {
        let o = other.borrow();
        self.iter().all(|a| o.contains(a))
    }

    /// Test whether a set is a proper subset of another set, meaning
    /// that all values in our set must also be in the other set. A
    /// proper subset must also be smaller than the other set.
    ///
    /// Time: O(n log n)
    #[must_use]
    pub fn is_proper_subset<RS>(&self, other: RS) -> bool
    where
        RS: Borrow<Self>,
    {
        self.len() != other.borrow().len() && self.is_subset(other)
    }
}

impl<A, S, P> GenericHashSet<A, S, P>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Clone,
    P: SharedPointerKind,
{
    /// Insert a value into a set.
    ///
    /// Time: O(log n)
    #[inline]
    pub fn insert(&mut self, a: A) -> Option<A> {
        let hash = hash_key(&self.hasher, &a);
        let root = SharedPointer::make_mut(self.root.get_or_insert_with(Default::default));
        match root.insert(hash, 0, Value(a)) {
            None => {
                self.size += 1;
                None
            }
            Some(Value(old_value)) => Some(old_value),
        }
    }

    /// Remove a value from a set if it exists.
    ///
    /// Time: O(log n)
    pub fn remove<BA>(&mut self, a: &BA) -> Option<A>
    where
        BA: Hash + Eq + ?Sized,
        A: Borrow<BA>,
    {
        let root = SharedPointer::make_mut(self.root.get_or_insert_with(Default::default));
        let result = root.remove(hash_key(&self.hasher, a), 0, a);
        if result.is_some() {
            self.size -= 1;
        }
        result.map(|v| v.0)
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
    /// # use imbl::hashset::HashSet;
    /// # use std::sync::Arc;
    /// let set = hashset![123];
    /// assert_eq!(
    ///   set.update(456),
    ///   hashset![123, 456]
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
        BA: Hash + Eq + ?Sized,
        A: Borrow<BA>,
    {
        let mut out = self.clone();
        out.remove(a);
        out
    }

    /// Filter out values from a set which don't satisfy a predicate.
    ///
    /// This is slightly more efficient than filtering using an
    /// iterator, in that it doesn't need to rehash the retained
    /// values, but it still needs to reconstruct the entire tree
    /// structure of the set.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::HashSet;
    /// let mut set = hashset![1, 2, 3];
    /// set.retain(|v| *v > 1);
    /// let expected = hashset![2, 3];
    /// assert_eq!(expected, set);
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&A) -> bool,
    {
        let Some(root) = &mut self.root else {
            return;
        };
        let old_root = root.clone();
        let root = SharedPointer::make_mut(root);
        for (value, hash) in NodeIter::new(Some(&old_root), self.size) {
            if !f(value) && root.remove(hash, 0, value).is_some() {
                self.size -= 1;
            }
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
    /// # use imbl::hashset::HashSet;
    /// let set1 = hashset!{1, 2};
    /// let set2 = hashset!{2, 3};
    /// let expected = hashset!{1, 2, 3};
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
        S: Default,
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
    /// # use imbl::hashset::HashSet;
    /// let set1 = hashset!{1, 2};
    /// let set2 = hashset!{2, 3};
    /// let expected = hashset!{1, 3};
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
    /// # use imbl::hashset::HashSet;
    /// let set1 = hashset!{1, 2};
    /// let set2 = hashset!{2, 3};
    /// let expected = hashset!{1, 3};
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
    /// # use imbl::hashset::HashSet;
    /// let set1 = hashset!{1, 2};
    /// let set2 = hashset!{2, 3};
    /// let expected = hashset!{1};
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
    /// # use imbl::hashset::HashSet;
    /// let set1 = hashset!{1, 2};
    /// let set2 = hashset!{2, 3};
    /// let expected = hashset!{2};
    /// assert_eq!(expected, set1.intersection(set2));
    /// ```
    #[must_use]
    pub fn intersection(self, other: Self) -> Self {
        let mut out = self.new_from();
        for value in other {
            if self.contains(&value) {
                out.insert(value);
            }
        }
        out
    }
}

// Core traits

impl<A, S, P: SharedPointerKind> Clone for GenericHashSet<A, S, P>
where
    A: Clone,
    S: Clone,
    P: SharedPointerKind,
{
    /// Clone a set.
    ///
    /// Time: O(1)
    #[inline]
    fn clone(&self) -> Self {
        GenericHashSet {
            hasher: self.hasher.clone(),
            root: self.root.clone(),
            size: self.size,
        }
    }
}

impl<A, S1, P1, S2, P2> PartialEq<GenericHashSet<A, S2, P2>> for GenericHashSet<A, S1, P1>
where
    A: Hash + Eq,
    S1: BuildHasher,
    S2: BuildHasher,
    P1: SharedPointerKind,
    P2: SharedPointerKind,
{
    fn eq(&self, other: &GenericHashSet<A, S2, P2>) -> bool {
        self.test_eq(other)
    }
}

impl<A, S, P> Eq for GenericHashSet<A, S, P>
where
    A: Hash + Eq,
    S: BuildHasher,
    P: SharedPointerKind,
{
}

impl<A, S, P> Default for GenericHashSet<A, S, P>
where
    S: Default,
    P: SharedPointerKind,
{
    fn default() -> Self {
        GenericHashSet {
            hasher: Default::default(),
            root: None,
            size: 0,
        }
    }
}

impl<A, S, P> Add for GenericHashSet<A, S, P>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Clone,
    P: SharedPointerKind,
{
    type Output = GenericHashSet<A, S, P>;

    fn add(self, other: Self) -> Self::Output {
        self.union(other)
    }
}

impl<A, S, P> Mul for GenericHashSet<A, S, P>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Clone,
    P: SharedPointerKind,
{
    type Output = GenericHashSet<A, S, P>;

    fn mul(self, other: Self) -> Self::Output {
        self.intersection(other)
    }
}

impl<A, S, P> Add for &GenericHashSet<A, S, P>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Clone,
    P: SharedPointerKind,
{
    type Output = GenericHashSet<A, S, P>;

    fn add(self, other: Self) -> Self::Output {
        self.clone().union(other.clone())
    }
}

impl<A, S, P> Mul for &GenericHashSet<A, S, P>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Clone,
    P: SharedPointerKind,
{
    type Output = GenericHashSet<A, S, P>;

    fn mul(self, other: Self) -> Self::Output {
        self.clone().intersection(other.clone())
    }
}

impl<A, S, P: SharedPointerKind> Sum for GenericHashSet<A, S, P>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    fn sum<I>(it: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        it.fold(Self::default(), |a, b| a + b)
    }
}

impl<A, S, R, P: SharedPointerKind> Extend<R> for GenericHashSet<A, S, P>
where
    A: Hash + Eq + Clone + From<R>,
    S: BuildHasher + Clone,
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

impl<A, S, P> Debug for GenericHashSet<A, S, P>
where
    A: Hash + Eq + Debug,
    S: BuildHasher,
    P: SharedPointerKind,
{
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

impl<'a, A, P> Iterator for Iter<'a, A, P>
where
    A: 'a,
    P: SharedPointerKind,
{
    type Item = &'a A;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(v, _)| &v.0)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, A, P: SharedPointerKind> ExactSizeIterator for Iter<'a, A, P> {}

impl<'a, A, P: SharedPointerKind> FusedIterator for Iter<'a, A, P> {}

/// A consuming iterator over the elements of a set.
pub struct ConsumingIter<A, P>
where
    A: Hash + Eq + Clone,
    P: SharedPointerKind,
{
    it: NodeDrain<Value<A>, P>,
}

impl<A, P> Iterator for ConsumingIter<A, P>
where
    A: Hash + Eq + Clone,
    P: SharedPointerKind,
{
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(v, _)| v.0)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<A, P> ExactSizeIterator for ConsumingIter<A, P>
where
    A: Hash + Eq + Clone,
    P: SharedPointerKind,
{
}

impl<A, P> FusedIterator for ConsumingIter<A, P>
where
    A: Hash + Eq + Clone,
    P: SharedPointerKind,
{
}

// Iterator conversions

impl<A, RA, S, P> FromIterator<RA> for GenericHashSet<A, S, P>
where
    A: Hash + Eq + Clone + From<RA>,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    fn from_iter<T>(i: T) -> Self
    where
        T: IntoIterator<Item = RA>,
    {
        let mut set = Self::default();
        for value in i {
            set.insert(From::from(value));
        }
        set
    }
}

impl<'a, A, S, P> IntoIterator for &'a GenericHashSet<A, S, P>
where
    A: Hash + Eq,
    S: BuildHasher,
    P: SharedPointerKind,
{
    type Item = &'a A;
    type IntoIter = Iter<'a, A, P>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<A, S, P> IntoIterator for GenericHashSet<A, S, P>
where
    A: Hash + Eq + Clone,
    S: BuildHasher,
    P: SharedPointerKind,
{
    type Item = A;
    type IntoIter = ConsumingIter<Self::Item, P>;

    fn into_iter(self) -> Self::IntoIter {
        ConsumingIter {
            it: NodeDrain::new(self.root, self.size),
        }
    }
}

// Conversions

impl<A, OA, SA, SB, P1, P2> From<&GenericHashSet<&A, SA, P1>> for GenericHashSet<OA, SB, P2>
where
    A: ToOwned<Owned = OA> + Hash + Eq + ?Sized,
    OA: Borrow<A> + Hash + Eq + Clone,
    SA: BuildHasher,
    SB: BuildHasher + Default + Clone,
    P1: SharedPointerKind,
    P2: SharedPointerKind,
{
    fn from(set: &GenericHashSet<&A, SA, P1>) -> Self {
        set.iter().map(|a| (*a).to_owned()).collect()
    }
}

impl<A, S, const N: usize, P> From<[A; N]> for GenericHashSet<A, S, P>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    fn from(arr: [A; N]) -> Self {
        IntoIterator::into_iter(arr).collect()
    }
}

impl<'a, A, S, P> From<&'a [A]> for GenericHashSet<A, S, P>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    fn from(slice: &'a [A]) -> Self {
        slice.iter().cloned().collect()
    }
}

impl<A, S, P> From<Vec<A>> for GenericHashSet<A, S, P>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    fn from(vec: Vec<A>) -> Self {
        vec.into_iter().collect()
    }
}

impl<A, S, P> From<&Vec<A>> for GenericHashSet<A, S, P>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    fn from(vec: &Vec<A>) -> Self {
        vec.iter().cloned().collect()
    }
}

impl<A, S, P1, P2> From<GenericVector<A, P2>> for GenericHashSet<A, S, P1>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Default + Clone,
    P1: SharedPointerKind,
    P2: SharedPointerKind,
{
    fn from(vector: GenericVector<A, P2>) -> Self {
        vector.into_iter().collect()
    }
}

impl<A, S, P1, P2> From<&GenericVector<A, P2>> for GenericHashSet<A, S, P1>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Default + Clone,
    P1: SharedPointerKind,
    P2: SharedPointerKind,
{
    fn from(vector: &GenericVector<A, P2>) -> Self {
        vector.iter().cloned().collect()
    }
}

impl<A, S, P> From<collections::HashSet<A>> for GenericHashSet<A, S, P>
where
    A: Eq + Hash + Clone,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    fn from(hash_set: collections::HashSet<A>) -> Self {
        hash_set.into_iter().collect()
    }
}

impl<A, S, P> From<&collections::HashSet<A>> for GenericHashSet<A, S, P>
where
    A: Eq + Hash + Clone,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    fn from(hash_set: &collections::HashSet<A>) -> Self {
        hash_set.iter().cloned().collect()
    }
}

impl<A, S, P> From<&BTreeSet<A>> for GenericHashSet<A, S, P>
where
    A: Hash + Eq + Clone,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    fn from(btree_set: &BTreeSet<A>) -> Self {
        btree_set.iter().cloned().collect()
    }
}

impl<A, S, P1, P2> From<GenericOrdSet<A, P2>> for GenericHashSet<A, S, P1>
where
    A: Ord + Hash + Eq + Clone,
    S: BuildHasher + Default + Clone,
    P1: SharedPointerKind,
    P2: SharedPointerKind,
{
    fn from(ordset: GenericOrdSet<A, P2>) -> Self {
        ordset.into_iter().collect()
    }
}

impl<A, S, P1, P2> From<&GenericOrdSet<A, P2>> for GenericHashSet<A, S, P1>
where
    A: Ord + Hash + Eq + Clone,
    S: BuildHasher + Default + Clone,
    P1: SharedPointerKind,
    P2: SharedPointerKind,
{
    fn from(ordset: &GenericOrdSet<A, P2>) -> Self {
        ordset.into_iter().cloned().collect()
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
    pub use crate::proptest::hash_set;
}

#[cfg(test)]
mod test {
    use super::proptest::*;
    use super::*;
    use crate::test::LolHasher;
    use ::proptest::num::i16;
    use ::proptest::proptest;
    use static_assertions::{assert_impl_all, assert_not_impl_any};
    use std::hash::BuildHasherDefault;

    assert_impl_all!(HashSet<i32>: Send, Sync);
    assert_not_impl_any!(HashSet<*const i32>: Send, Sync);
    assert_covariant!(HashSet<T> in T);

    #[test]
    fn insert_failing() {
        let mut set: GenericHashSet<i16, BuildHasherDefault<LolHasher>, DefaultSharedPtr> =
            Default::default();
        set.insert(14658);
        assert_eq!(1, set.len());
        set.insert(-19198);
        assert_eq!(2, set.len());
    }

    #[test]
    fn match_strings_with_string_slices() {
        let mut set: HashSet<String> = From::from(&hashset!["foo", "bar"]);
        set = set.without("bar");
        assert!(!set.contains("bar"));
        set.remove("foo");
        assert!(!set.contains("foo"));
    }

    #[test]
    fn macro_allows_trailing_comma() {
        let set1 = hashset! {"foo", "bar"};
        let set2 = hashset! {
            "foo",
            "bar",
        };
        assert_eq!(set1, set2);
    }

    #[test]
    fn issue_60_drain_iterator_memory_corruption() {
        use crate::test::MetroHashBuilder;
        for i in 0..1000 {
            let mut lhs = vec![0, 1, 2];
            lhs.sort_unstable();

            let hasher = MetroHashBuilder::new(i);
            let mut iset: GenericHashSet<_, MetroHashBuilder, DefaultSharedPtr> =
                GenericHashSet::with_hasher(hasher);
            for &i in &lhs {
                iset.insert(i);
            }

            let mut rhs: Vec<_> = iset.clone().into_iter().collect();
            rhs.sort_unstable();

            if lhs != rhs {
                println!("iteration: {}", i);
                println!("seed: {}", hasher.seed());
                println!("lhs: {}: {:?}", lhs.len(), &lhs);
                println!("rhs: {}: {:?}", rhs.len(), &rhs);
                panic!();
            }
        }
    }

    proptest! {
        #[test]
        fn proptest_a_set(ref s in hash_set(".*", 10..100)) {
            assert!(s.len() < 100);
            assert!(s.len() >= 10);
        }
    }
}
