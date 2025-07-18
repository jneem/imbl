// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! An ordered map.
//!
//! An immutable ordered map implemented as a [B+tree] [1].
//!
//! Most operations on this type of map are O(log n). A
//! [`HashMap`][hashmap::HashMap] is usually a better choice for
//! performance, but the `OrdMap` has the advantage of only requiring
//! an [`Ord`][std::cmp::Ord] constraint on the key, and of being
//! ordered, so that keys always come out from lowest to highest,
//! where a [`HashMap`][hashmap::HashMap] has no guaranteed ordering.
//!
//! [1]: https://en.wikipedia.org/wiki/B%2B_tree
//! [hashmap::HashMap]: ../hashmap/struct.HashMap.html
//! [std::cmp::Ord]: https://doc.rust-lang.org/std/cmp/trait.Ord.html

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections;
use std::fmt::{Debug, Error, Formatter};
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::{FromIterator, FusedIterator, Sum};
use std::mem;
use std::ops::{Add, Bound, Index, IndexMut, RangeBounds};

use archery::{SharedPointer, SharedPointerKind};

use crate::hashmap::GenericHashMap;
use crate::nodes::btree::{
    ConsumingIter as NodeConsumingIter, Cursor, InsertAction, Iter as NodeIter, Node,
};
use crate::shared_ptr::DefaultSharedPtr;

/// Construct a map from a sequence of key/value pairs.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate imbl;
/// # use imbl::ordmap::OrdMap;
/// # fn main() {
/// assert_eq!(
///   ordmap!{
///     1 => 11,
///     2 => 22,
///     3 => 33
///   },
///   OrdMap::from(vec![(1, 11), (2, 22), (3, 33)])
/// );
/// # }
/// ```
#[macro_export]
macro_rules! ordmap {
    () => { $crate::ordmap::OrdMap::new() };

    ( $( $key:expr => $value:expr ),* ) => {{
        let mut map = $crate::ordmap::OrdMap::new();
        $({
            map.insert($key, $value);
        })*;
        map
    }};
}

/// Type alias for [`GenericOrdMap`] that uses [`DefaultSharedPtr`] as the pointer type.
///
/// [GenericOrdMap]: ./struct.GenericOrdMap.html
/// [DefaultSharedPtr]: ../shared_ptr/type.DefaultSharedPtr.html
pub type OrdMap<K, V> = GenericOrdMap<K, V, DefaultSharedPtr>;

/// An ordered map.
///
/// An immutable ordered map implemented as a B+tree [1].
///
/// Most operations on this type of map are O(log n). A
/// [`HashMap`][hashmap::HashMap] is usually a better choice for
/// performance, but the `OrdMap` has the advantage of only requiring
/// an [`Ord`][std::cmp::Ord] constraint on the key, and of being
/// ordered, so that keys always come out from lowest to highest,
/// where a [`HashMap`][hashmap::HashMap] has no guaranteed ordering.
///
/// [1]: https://en.wikipedia.org/wiki/B%2B_tree
/// [hashmap::HashMap]: ../hashmap/struct.HashMap.html
/// [std::cmp::Ord]: https://doc.rust-lang.org/std/cmp/trait.Ord.html
pub struct GenericOrdMap<K, V, P: SharedPointerKind> {
    size: usize,
    root: Option<Node<K, V, P>>,
}

impl<K, V, P: SharedPointerKind> GenericOrdMap<K, V, P> {
    /// Construct an empty map.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        GenericOrdMap {
            size: 0,
            root: None,
        }
    }

    /// Construct a map with a single mapping.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # type OrdMap<K, V> = imbl::ordmap::OrdMap<K, V>;
    /// let map = OrdMap::unit(123, "onetwothree");
    /// assert_eq!(
    ///   map.get(&123),
    ///   Some(&"onetwothree")
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn unit(key: K, value: V) -> Self {
        Self {
            size: 1,
            root: Some(Node::unit(key, value)),
        }
    }

    /// Test whether a map is empty.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// assert!(
    ///   !ordmap!{1 => 2}.is_empty()
    /// );
    /// assert!(
    ///   OrdMap::<i32, i32>::new().is_empty()
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Test whether two maps refer to the same content in memory.
    ///
    /// This is true if the two sides are references to the same map,
    /// or if the two maps refer to the same root node.
    ///
    /// This would return true if you're comparing a map to itself, or
    /// if you're comparing a map to a fresh clone of itself.
    ///
    /// Time: O(1)
    pub fn ptr_eq(&self, other: &Self) -> bool {
        match (&self.root, &other.root) {
            (Some(a), Some(b)) => a.ptr_eq(&b),
            (None, None) => true,
            _ => false,
        }
    }

    /// Get the size of a map.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// assert_eq!(3, ordmap!{
    ///   1 => 11,
    ///   2 => 22,
    ///   3 => 33
    /// }.len());
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Discard all elements from the map.
    ///
    /// This leaves you with an empty map, and all elements that
    /// were previously inside it are dropped.
    ///
    /// Time: O(n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::OrdMap;
    /// let mut map = ordmap![1=>1, 2=>2, 3=>3];
    /// map.clear();
    /// assert!(map.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.root = None;
        self.size = 0;
    }
}

impl<K, V, P> GenericOrdMap<K, V, P>
where
    K: Ord,
    P: SharedPointerKind,
{
    /// Get the largest key in a map, along with its value. If the map
    /// is empty, return `None`.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// assert_eq!(Some(&(3, 33)), ordmap!{
    ///   1 => 11,
    ///   2 => 22,
    ///   3 => 33
    /// }.get_max());
    /// ```
    #[must_use]
    pub fn get_max(&self) -> Option<&(K, V)> {
        self.root.as_ref().and_then(|root| root.max())
    }

    /// Get the smallest key in a map, along with its value. If the
    /// map is empty, return `None`.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// assert_eq!(Some(&(1, 11)), ordmap!{
    ///   1 => 11,
    ///   2 => 22,
    ///   3 => 33
    /// }.get_min());
    /// ```
    #[must_use]
    pub fn get_min(&self) -> Option<&(K, V)> {
        self.root.as_ref().and_then(|root| root.min())
    }

    /// Get an iterator over the key/value pairs of a map.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, K, V, P> {
        Iter {
            it: NodeIter::new(self.root.as_ref(), self.size, ..),
        }
    }

    /// Create an iterator over a range of key/value pairs.
    #[must_use]
    pub fn range<R, BK>(&self, range: R) -> RangedIter<'_, K, V, P>
    where
        R: RangeBounds<BK>,
        K: Borrow<BK>,
        BK: Ord + ?Sized,
    {
        RangedIter {
            it: NodeIter::new(self.root.as_ref(), self.size, range),
        }
    }

    /// Get an iterator over a map's keys.
    #[must_use]
    pub fn keys(&self) -> Keys<'_, K, V, P> {
        Keys { it: self.iter() }
    }

    /// Get an iterator over a map's values.
    #[must_use]
    pub fn values(&self) -> Values<'_, K, V, P> {
        Values { it: self.iter() }
    }

    /// Get an iterator over the differences between this map and
    /// another, i.e. the set of entries to add, update, or remove to
    /// this map in order to make it equal to the other map.
    ///
    /// This function will avoid visiting nodes which are shared
    /// between the two sets, meaning that even very large sets can be
    /// compared quickly if most of their structure is shared.
    ///
    /// Time: O(n) where n is the size of the larger map.
    #[must_use]
    pub fn diff<'a, 'b>(&'a self, other: &'b Self) -> DiffIter<'a, 'b, K, V, P> {
        let mut diff = DiffIter {
            it1: Cursor::empty(),
            it2: Cursor::empty(),
        };
        // If the two maps are the same, don't even initialize the cursors
        if self.ptr_eq(other) {
            return diff;
        }
        diff.it1.init(self.root.as_ref());
        diff.it2.init(other.root.as_ref());
        diff.it1.seek_to_first();
        diff.it2.seek_to_first();
        diff
    }

    /// Get the value for a key from a map.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let map = ordmap!{123 => "lol"};
    /// assert_eq!(
    ///   map.get(&123),
    ///   Some(&"lol")
    /// );
    /// ```
    #[must_use]
    pub fn get<BK>(&self, key: &BK) -> Option<&V>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        self.root
            .as_ref()
            .and_then(|r| r.lookup(key).map(|(_, v)| v))
    }

    /// Get the key/value pair for a key from a map.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let map = ordmap!{123 => "lol"};
    /// assert_eq!(
    ///   map.get_key_value(&123),
    ///   Some((&123, &"lol"))
    /// );
    /// ```
    #[must_use]
    pub fn get_key_value<BK>(&self, key: &BK) -> Option<(&K, &V)>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        self.root
            .as_ref()
            .and_then(|r| r.lookup(key).map(|(k, v)| (k, v)))
    }

    /// Get a reference to the closest smaller entry in a map
    /// to a given key.
    ///
    /// If the map contains the given key, this is returned.
    /// Otherwise, the closest key in the map smaller than the
    /// given value is returned. If the smallest key in the map
    /// is larger than the given key, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::OrdMap;
    /// let map = ordmap![1 => 1, 3 => 3, 5 => 5];
    /// assert_eq!(Some((&3, &3)), map.get_prev(&4));
    /// ```
    #[must_use]
    pub fn get_prev<BK>(&self, key: &BK) -> Option<(&K, &V)>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        self.range((Bound::Unbounded, Bound::Included(key)))
            .next_back()
    }

    /// Get a reference to the closest larger entry in a map
    /// to a given key.
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
    /// # use imbl::OrdMap;
    /// let map = ordmap![1 => 1, 3 => 3, 5 => 5];
    /// assert_eq!(Some((&5, &5)), map.get_next(&4));
    /// ```
    #[must_use]
    pub fn get_next<BK>(&self, key: &BK) -> Option<(&K, &V)>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        self.range((Bound::Included(key), Bound::Unbounded)).next()
    }

    /// Test for the presence of a key in a map.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let map = ordmap!{123 => "lol"};
    /// assert!(
    ///   map.contains_key(&123)
    /// );
    /// assert!(
    ///   !map.contains_key(&321)
    /// );
    /// ```
    #[must_use]
    pub fn contains_key<BK>(&self, k: &BK) -> bool
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        self.get(k).is_some()
    }

    /// Test whether a map is a submap of another map, meaning that
    /// all keys in our map must also be in the other map, with the
    /// same values.
    ///
    /// Use the provided function to decide whether values are equal.
    ///
    /// Time: O(n log n)
    #[must_use]
    pub fn is_submap_by<B, RM, F, P2>(&self, other: RM, mut cmp: F) -> bool
    where
        F: FnMut(&V, &B) -> bool,
        RM: Borrow<GenericOrdMap<K, B, P2>>,
        P2: SharedPointerKind,
    {
        self.iter()
            .all(|(k, v)| other.borrow().get(k).map(|ov| cmp(v, ov)).unwrap_or(false))
    }

    /// Test whether a map is a proper submap of another map, meaning
    /// that all keys in our map must also be in the other map, with
    /// the same values. To be a proper submap, ours must also contain
    /// fewer keys than the other map.
    ///
    /// Use the provided function to decide whether values are equal.
    ///
    /// Time: O(n log n)
    #[must_use]
    pub fn is_proper_submap_by<B, RM, F, P2>(&self, other: RM, cmp: F) -> bool
    where
        F: FnMut(&V, &B) -> bool,
        RM: Borrow<GenericOrdMap<K, B, P2>>,
        P2: SharedPointerKind,
    {
        self.len() != other.borrow().len() && self.is_submap_by(other, cmp)
    }

    /// Test whether a map is a submap of another map, meaning that
    /// all keys in our map must also be in the other map, with the
    /// same values.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let map1 = ordmap!{1 => 1, 2 => 2};
    /// let map2 = ordmap!{1 => 1, 2 => 2, 3 => 3};
    /// assert!(map1.is_submap(map2));
    /// ```
    #[must_use]
    pub fn is_submap<RM>(&self, other: RM) -> bool
    where
        V: PartialEq,
        RM: Borrow<Self>,
    {
        self.is_submap_by(other.borrow(), PartialEq::eq)
    }

    /// Test whether a map is a proper submap of another map, meaning
    /// that all keys in our map must also be in the other map, with
    /// the same values. To be a proper submap, ours must also contain
    /// fewer keys than the other map.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let map1 = ordmap!{1 => 1, 2 => 2};
    /// let map2 = ordmap!{1 => 1, 2 => 2, 3 => 3};
    /// assert!(map1.is_proper_submap(map2));
    ///
    /// let map3 = ordmap!{1 => 1, 2 => 2};
    /// let map4 = ordmap!{1 => 1, 2 => 2};
    /// assert!(!map3.is_proper_submap(map4));
    /// ```
    #[must_use]
    pub fn is_proper_submap<RM>(&self, other: RM) -> bool
    where
        V: PartialEq,
        RM: Borrow<Self>,
    {
        self.is_proper_submap_by(other.borrow(), PartialEq::eq)
    }

    /// Check invariants
    #[cfg(any(test, fuzzing))]
    #[allow(unreachable_pub)]
    pub fn check_sane(&self)
    where
        K: std::fmt::Debug,
        V: std::fmt::Debug,
    {
        let size = self
            .root
            .as_ref()
            .map(|root| root.check_sane(true))
            .unwrap_or(0);
        assert_eq!(size, self.size);
    }
}

impl<K, V, P> GenericOrdMap<K, V, P>
where
    K: Ord + Clone,
    V: Clone,
    P: SharedPointerKind,
{
    /// Get a mutable reference to the value for a key from a map.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let mut map = ordmap!{123 => "lol"};
    /// if let Some(value) = map.get_mut(&123) {
    ///     *value = "omg";
    /// }
    /// assert_eq!(
    ///   map.get(&123),
    ///   Some(&"omg")
    /// );
    /// ```
    #[must_use]
    pub fn get_mut<BK>(&mut self, key: &BK) -> Option<&mut V>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        let root = self.root.as_mut()?;
        root.lookup_mut(key).map(|(_, v)| v)
    }

    /// Get the key/value pair for a key from a map.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let mut map = ordmap!{123 => "lol"};
    /// assert_eq!(
    ///   map.get_key_value_mut(&123),
    ///   Some((&123, &mut "lol"))
    /// );
    /// ```
    #[must_use]
    pub fn get_key_value_mut<BK>(&mut self, key: &BK) -> Option<(&K, &mut V)>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        self.root.as_mut()?.lookup_mut(key)
    }

    /// Get the closest smaller entry in a map to a given key
    /// as a mutable reference.
    ///
    /// If the map contains the given key, this is returned.
    /// Otherwise, the closest key in the map smaller than the
    /// given value is returned. If the smallest key in the map
    /// is larger than the given key, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::OrdMap;
    /// let mut map = ordmap![1 => 1, 3 => 3, 5 => 5];
    /// if let Some((key, value)) = map.get_prev_mut(&4) {
    ///     *value = 4;
    /// }
    /// assert_eq!(ordmap![1 => 1, 3 => 4, 5 => 5], map);
    /// ```
    #[must_use]
    pub fn get_prev_mut<BK>(&mut self, key: &BK) -> Option<(&K, &mut V)>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        let prev = self.get_prev(key)?.0.clone();
        let root = self.root.as_mut()?;
        root.lookup_mut(prev.borrow())
    }

    /// Get the closest larger entry in a map to a given key
    /// as a mutable reference.
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
    /// # use imbl::OrdMap;
    /// let mut map = ordmap![1 => 1, 3 => 3, 5 => 5];
    /// if let Some((key, value)) = map.get_next_mut(&4) {
    ///     *value = 4;
    /// }
    /// assert_eq!(ordmap![1 => 1, 3 => 3, 5 => 4], map);
    /// ```
    #[must_use]
    pub fn get_next_mut<BK>(&mut self, key: &BK) -> Option<(&K, &mut V)>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        let next = self.get_next(key)?.0.clone();
        let root = self.root.as_mut()?;
        root.lookup_mut(next.borrow())
    }

    /// Insert a key/value mapping into a map.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// map's structure which are shared with other maps will be
    /// safely copied before mutating.
    ///
    /// If the map already has a mapping for the given key, the
    /// previous value is overwritten.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let mut map = ordmap!{};
    /// map.insert(123, "123");
    /// map.insert(456, "456");
    /// assert_eq!(
    ///   map,
    ///   ordmap!{123 => "123", 456 => "456"}
    /// );
    /// ```
    ///
    /// [insert]: #method.insert
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.insert_key_value(key, value).map(|(_, v)| v)
    }

    /// Insert a key/value mapping into a map.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// map's structure which are shared with other maps will be
    /// safely copied before mutating.
    ///
    /// If the map already has a mapping for the given key, the
    /// previous key and value are overwritten and returned.
    #[inline]
    pub(crate) fn insert_key_value(&mut self, key: K, value: V) -> Option<(K, V)> {
        let root = self.root.get_or_insert_with(Node::default);
        match root.insert(key, value) {
            InsertAction::Replaced(old_key, old_value) => return Some((old_key, old_value)),
            InsertAction::Inserted => (),
            InsertAction::Split(separator, right) => {
                let left = mem::take(root);
                *root = Node::new_from_split(left, separator, right);
            }
        }
        self.size += 1;
        None
    }

    /// Remove a key/value mapping from a map if it exists.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let mut map = ordmap!{123 => "123", 456 => "456"};
    /// map.remove(&123);
    /// map.remove(&456);
    /// assert!(map.is_empty());
    /// ```
    ///
    /// [remove]: #method.remove
    #[inline]
    pub fn remove<BK>(&mut self, k: &BK) -> Option<V>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        self.remove_with_key(k).map(|(_, v)| v)
    }

    /// Remove a key/value pair from a map, if it exists, and return
    /// the removed key and value.
    ///
    /// Time: O(log n)
    pub fn remove_with_key<BK>(&mut self, k: &BK) -> Option<(K, V)>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        let root = self.root.as_mut()?;
        let mut removed = None;
        if root.remove(k, &mut removed) {
            if let Node::Branch(branch) = root {
                if let Some(child) = SharedPointer::make_mut(branch).pop_single_child() {
                    self.root = Some(child);
                }
            }
            // Note that even if the root leaf is empty, we don't
            // drop it, but retain the allocation for future use.
        }
        self.size -= removed.is_some() as usize;
        removed
    }

    /// Construct a new map by inserting a key/value mapping into a
    /// map.
    ///
    /// If the map already has a mapping for the given key, the
    /// previous value is overwritten.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let map = ordmap!{};
    /// assert_eq!(
    ///   map.update(123, "123"),
    ///   ordmap!{123 => "123"}
    /// );
    /// ```
    #[must_use]
    pub fn update(&self, key: K, value: V) -> Self {
        let mut out = self.clone();
        out.insert(key, value);
        out
    }

    /// Construct a new map by inserting a key/value mapping into a
    /// map.
    ///
    /// If the map already has a mapping for the given key, we call
    /// the provided function with the old value and the new value,
    /// and insert the result as the new value.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn update_with<F>(self, k: K, v: V, f: F) -> Self
    where
        F: FnOnce(V, V) -> V,
    {
        self.update_with_key(k, v, |_, v1, v2| f(v1, v2))
    }

    /// Construct a new map by inserting a key/value mapping into a
    /// map.
    ///
    /// If the map already has a mapping for the given key, we call
    /// the provided function with the key, the old value and the new
    /// value, and insert the result as the new value.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn update_with_key<F>(self, k: K, v: V, f: F) -> Self
    where
        F: FnOnce(&K, V, V) -> V,
    {
        match self.extract_with_key(&k) {
            None => self.update(k, v),
            Some((_, v2, m)) => {
                let out_v = f(&k, v2, v);
                m.update(k, out_v)
            }
        }
    }

    /// Construct a new map by inserting a key/value mapping into a
    /// map, returning the old value for the key as well as the new
    /// map.
    ///
    /// If the map already has a mapping for the given key, we call
    /// the provided function with the key, the old value and the new
    /// value, and insert the result as the new value.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn update_lookup_with_key<F>(self, k: K, v: V, f: F) -> (Option<V>, Self)
    where
        F: FnOnce(&K, &V, V) -> V,
    {
        match self.extract_with_key(&k) {
            None => (None, self.update(k, v)),
            Some((_, v2, m)) => {
                let out_v = f(&k, &v2, v);
                (Some(v2), m.update(k, out_v))
            }
        }
    }

    /// Update the value for a given key by calling a function with
    /// the current value and overwriting it with the function's
    /// return value.
    ///
    /// The function gets an [`Option<V>`][std::option::Option] and
    /// returns the same, so that it can decide to delete a mapping
    /// instead of updating the value, and decide what to do if the
    /// key isn't in the map.
    ///
    /// Time: O(log n)
    ///
    /// [std::option::Option]: https://doc.rust-lang.org/std/option/enum.Option.html
    #[must_use]
    pub fn alter<F>(&self, f: F, k: K) -> Self
    where
        F: FnOnce(Option<V>) -> Option<V>,
    {
        let pop = self.extract_with_key(&k);
        match (f(pop.as_ref().map(|(_, v, _)| v.clone())), pop) {
            (None, None) => self.clone(),
            (Some(v), None) => self.update(k, v),
            (None, Some((_, _, m))) => m,
            (Some(v), Some((_, _, m))) => m.update(k, v),
        }
    }

    /// Remove a key/value pair from a map, if it exists.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn without<BK>(&self, k: &BK) -> Self
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        self.extract(k)
            .map(|(_, m)| m)
            .unwrap_or_else(|| self.clone())
    }

    /// Remove a key/value pair from a map, if it exists, and return
    /// the removed value as well as the updated list.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn extract<BK>(&self, k: &BK) -> Option<(V, Self)>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        self.extract_with_key(k).map(|(_, v, m)| (v, m))
    }

    /// Remove a key/value pair from a map, if it exists, and return
    /// the removed key and value as well as the updated list.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn extract_with_key<BK>(&self, k: &BK) -> Option<(K, V, Self)>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        let mut out = self.clone();
        let result = out.remove_with_key(k);
        result.map(|(k, v)| (k, v, out))
    }

    /// Construct the union of two maps, keeping the values in the
    /// current map when keys exist in both maps.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let map1 = ordmap!{1 => 1, 3 => 3};
    /// let map2 = ordmap!{2 => 2, 3 => 4};
    /// let expected = ordmap!{1 => 1, 2 => 2, 3 => 3};
    /// assert_eq!(expected, map1.union(map2));
    /// ```
    #[inline]
    #[must_use]
    pub fn union(mut self, mut other: Self) -> Self {
        // We get better performance by consuming the small one and growing the big one. But the
        // code isn't quite symmetric, because we need to keep values that are present in `self`.
        if self.len() >= other.len() {
            for (k, v) in other {
                self.entry(k).or_insert(v);
            }
            self
        } else {
            for (k, v) in self {
                other.insert(k, v);
            }
            other
        }
    }

    /// Construct the union of two maps, using a function to decide
    /// what to do with the value when a key is in both maps.
    ///
    /// The function is called when a value exists in both maps, and
    /// receives the value from the current map as its first argument,
    /// and the value from the other map as the second. It should
    /// return the value to be inserted in the resulting map.
    ///
    /// Time: O(n log n)
    #[inline]
    #[must_use]
    pub fn union_with<F>(self, other: Self, mut f: F) -> Self
    where
        F: FnMut(V, V) -> V,
    {
        self.union_with_key(other, |_, v1, v2| f(v1, v2))
    }

    /// Construct the union of two maps, using a function to decide
    /// what to do with the value when a key is in both maps.
    ///
    /// The function is called when a value exists in both maps, and
    /// receives a reference to the key as its first argument, the
    /// value from the current map as the second argument, and the
    /// value from the other map as the third argument. It should
    /// return the value to be inserted in the resulting map.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let map1 = ordmap!{1 => 1, 3 => 4};
    /// let map2 = ordmap!{2 => 2, 3 => 5};
    /// let expected = ordmap!{1 => 1, 2 => 2, 3 => 9};
    /// assert_eq!(expected, map1.union_with_key(
    ///     map2,
    ///     |key, left, right| left + right
    /// ));
    /// ```
    #[must_use]
    pub fn union_with_key<F>(self, other: Self, mut f: F) -> Self
    where
        F: FnMut(&K, V, V) -> V,
    {
        if self.len() >= other.len() {
            self.union_with_key_inner(other, f)
        } else {
            other.union_with_key_inner(self, |key, other_value, self_value| {
                f(key, self_value, other_value)
            })
        }
    }

    fn union_with_key_inner<F>(mut self, other: Self, mut f: F) -> Self
    where
        F: FnMut(&K, V, V) -> V,
    {
        for (key, right_value) in other {
            match self.remove(&key) {
                None => {
                    self.insert(key, right_value);
                }
                Some(left_value) => {
                    let final_value = f(&key, left_value, right_value);
                    self.insert(key, final_value);
                }
            }
        }
        self
    }

    /// Construct the union of a sequence of maps, selecting the value
    /// of the leftmost when a key appears in more than one map.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let map1 = ordmap!{1 => 1, 3 => 3};
    /// let map2 = ordmap!{2 => 2};
    /// let expected = ordmap!{1 => 1, 2 => 2, 3 => 3};
    /// assert_eq!(expected, OrdMap::unions(vec![map1, map2]));
    /// ```
    #[must_use]
    pub fn unions<I>(i: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        i.into_iter().fold(Self::default(), Self::union)
    }

    /// Construct the union of a sequence of maps, using a function to
    /// decide what to do with the value when a key is in more than
    /// one map.
    ///
    /// The function is called when a value exists in multiple maps,
    /// and receives the value from the current map as its first
    /// argument, and the value from the next map as the second. It
    /// should return the value to be inserted in the resulting map.
    ///
    /// Time: O(n log n)
    #[must_use]
    pub fn unions_with<I, F>(i: I, f: F) -> Self
    where
        I: IntoIterator<Item = Self>,
        F: Fn(V, V) -> V,
    {
        i.into_iter()
            .fold(Self::default(), |a, b| a.union_with(b, &f))
    }

    /// Construct the union of a sequence of maps, using a function to
    /// decide what to do with the value when a key is in more than
    /// one map.
    ///
    /// The function is called when a value exists in multiple maps,
    /// and receives a reference to the key as its first argument, the
    /// value from the current map as the second argument, and the
    /// value from the next map as the third argument. It should
    /// return the value to be inserted in the resulting map.
    ///
    /// Time: O(n log n)
    #[must_use]
    pub fn unions_with_key<I, F>(i: I, f: F) -> Self
    where
        I: IntoIterator<Item = Self>,
        F: Fn(&K, V, V) -> V,
    {
        i.into_iter()
            .fold(Self::default(), |a, b| a.union_with_key(b, &f))
    }

    /// Construct the symmetric difference between two maps by discarding keys
    /// which occur in both maps.
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
    /// # use imbl::ordmap::OrdMap;
    /// let map1 = ordmap!{1 => 1, 3 => 4};
    /// let map2 = ordmap!{2 => 2, 3 => 5};
    /// let expected = ordmap!{1 => 1, 2 => 2};
    /// assert_eq!(expected, map1.difference(map2));
    /// ```
    ///
    /// [symmetric_difference]: #method.symmetric_difference
    #[deprecated(
        since = "2.0.1",
        note = "to avoid conflicting behaviors between std and imbl, the `difference` alias for `symmetric_difference` will be removed."
    )]
    #[inline]
    #[must_use]
    pub fn difference(self, other: Self) -> Self {
        self.symmetric_difference(other)
    }

    /// Construct the symmetric difference between two maps by discarding keys
    /// which occur in both maps.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let map1 = ordmap!{1 => 1, 3 => 4};
    /// let map2 = ordmap!{2 => 2, 3 => 5};
    /// let expected = ordmap!{1 => 1, 2 => 2};
    /// assert_eq!(expected, map1.symmetric_difference(map2));
    /// ```
    #[inline]
    #[must_use]
    pub fn symmetric_difference(self, other: Self) -> Self {
        self.symmetric_difference_with_key(other, |_, _, _| None)
    }

    /// Construct the symmetric difference between two maps by using a function
    /// to decide what to do if a key occurs in both.
    ///
    /// This is an alias for the
    /// [`symmetric_difference_with`][symmetric_difference_with] method.
    ///
    /// Time: O(n log n)
    ///
    /// [symmetric_difference_with]: #method.symmetric_difference_with
    #[deprecated(
        since = "2.0.1",
        note = "to avoid conflicting behaviors between std and imbl, the `difference_with` alias for `symmetric_difference_with` will be removed."
    )]
    #[inline]
    #[must_use]
    pub fn difference_with<F>(self, other: Self, f: F) -> Self
    where
        F: FnMut(V, V) -> Option<V>,
    {
        self.symmetric_difference_with(other, f)
    }

    /// Construct the symmetric difference between two maps by using a function
    /// to decide what to do if a key occurs in both.
    ///
    /// Time: O(n log n)
    #[inline]
    #[must_use]
    pub fn symmetric_difference_with<F>(self, other: Self, mut f: F) -> Self
    where
        F: FnMut(V, V) -> Option<V>,
    {
        self.symmetric_difference_with_key(other, |_, a, b| f(a, b))
    }

    /// Construct the symmetric difference between two maps by using a function
    /// to decide what to do if a key occurs in both. The function
    /// receives the key as well as both values.
    ///
    /// This is an alias for the
    /// [`symmetric_difference_with_key`][symmetric_difference_with_key]
    /// method.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let map1 = ordmap!{1 => 1, 3 => 4};
    /// let map2 = ordmap!{2 => 2, 3 => 5};
    /// let expected = ordmap!{1 => 1, 2 => 2, 3 => 9};
    /// assert_eq!(expected, map1.difference_with_key(
    ///     map2,
    ///     |key, left, right| Some(left + right)
    /// ));
    /// ```
    /// [symmetric_difference_with_key]: #method.symmetric_difference_with_key
    #[deprecated(
        since = "2.0.1",
        note = "to avoid conflicting behaviors between std and imbl, the `difference_with_key` alias for `symmetric_difference_with_key` will be removed."
    )]
    #[must_use]
    pub fn difference_with_key<F>(self, other: Self, f: F) -> Self
    where
        F: FnMut(&K, V, V) -> Option<V>,
    {
        self.symmetric_difference_with_key(other, f)
    }

    /// Construct the symmetric difference between two maps by using a function
    /// to decide what to do if a key occurs in both. The function
    /// receives the key as well as both values.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let map1 = ordmap!{1 => 1, 3 => 4};
    /// let map2 = ordmap!{2 => 2, 3 => 5};
    /// let expected = ordmap!{1 => 1, 2 => 2, 3 => 9};
    /// assert_eq!(expected, map1.symmetric_difference_with_key(
    ///     map2,
    ///     |key, left, right| Some(left + right)
    /// ));
    /// ```
    #[must_use]
    pub fn symmetric_difference_with_key<F>(mut self, other: Self, mut f: F) -> Self
    where
        F: FnMut(&K, V, V) -> Option<V>,
    {
        let mut out = Self::default();
        for (key, right_value) in other {
            match self.remove(&key) {
                None => {
                    out.insert(key, right_value);
                }
                Some(left_value) => {
                    if let Some(final_value) = f(&key, left_value, right_value) {
                        out.insert(key, final_value);
                    }
                }
            }
        }
        out.union(self)
    }

    /// Construct the relative complement between two maps by discarding keys
    /// which occur in `other`.
    ///
    /// Time: O(m log n) where m is the size of the other map
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let map1 = ordmap!{1 => 1, 3 => 4};
    /// let map2 = ordmap!{2 => 2, 3 => 5};
    /// let expected = ordmap!{1 => 1};
    /// assert_eq!(expected, map1.relative_complement(map2));
    /// ```
    #[inline]
    #[must_use]
    pub fn relative_complement(mut self, other: Self) -> Self {
        for (key, _) in other {
            let _ = self.remove(&key);
        }
        self
    }

    /// Construct the intersection of two maps, keeping the values
    /// from the current map.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let map1 = ordmap!{1 => 1, 2 => 2};
    /// let map2 = ordmap!{2 => 3, 3 => 4};
    /// let expected = ordmap!{2 => 2};
    /// assert_eq!(expected, map1.intersection(map2));
    /// ```
    #[inline]
    #[must_use]
    pub fn intersection(self, other: Self) -> Self {
        self.intersection_with_key(other, |_, v, _| v)
    }

    /// Construct the intersection of two maps, calling a function
    /// with both values for each key and using the result as the
    /// value for the key.
    ///
    /// Time: O(n log n)
    #[inline]
    #[must_use]
    pub fn intersection_with<B, C, F, P2, P3>(
        self,
        other: GenericOrdMap<K, B, P2>,
        mut f: F,
    ) -> GenericOrdMap<K, C, P3>
    where
        B: Clone,
        C: Clone,
        F: FnMut(V, B) -> C,
        P2: SharedPointerKind,
        P3: SharedPointerKind,
    {
        self.intersection_with_key(other, |_, v1, v2| f(v1, v2))
    }

    /// Construct the intersection of two maps, calling a function
    /// with the key and both values for each key and using the result
    /// as the value for the key.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::ordmap::OrdMap;
    /// let map1 = ordmap!{1 => 1, 2 => 2};
    /// let map2 = ordmap!{2 => 3, 3 => 4};
    /// let expected = ordmap!{2 => 5};
    /// assert_eq!(expected, map1.intersection_with_key(
    ///     map2,
    ///     |key, left, right| left + right
    /// ));
    /// ```
    #[must_use]
    pub fn intersection_with_key<B, C, F, P2, P3>(
        mut self,
        other: GenericOrdMap<K, B, P2>,
        mut f: F,
    ) -> GenericOrdMap<K, C, P3>
    where
        B: Clone,
        C: Clone,
        F: FnMut(&K, V, B) -> C,
        P2: SharedPointerKind,
        P3: SharedPointerKind,
    {
        let mut out = GenericOrdMap::<K, C, P3>::default();
        for (key, right_value) in other {
            match self.remove(&key) {
                None => (),
                Some(left_value) => {
                    let result = f(&key, left_value, right_value);
                    out.insert(key, result);
                }
            }
        }
        out
    }

    /// Split a map into two, with the left hand map containing keys
    /// which are smaller than `split`, and the right hand map
    /// containing keys which are larger than `split`.
    ///
    /// The `split` mapping is discarded.
    #[must_use]
    pub fn split<BK>(&self, split: &BK) -> (Self, Self)
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        let (l, _, r) = self.split_lookup(split);
        (l, r)
    }

    /// Split a map into two, with the left hand map containing keys
    /// which are smaller than `split`, and the right hand map
    /// containing keys which are larger than `split`.
    ///
    /// Returns both the two maps and the value of `split`.
    #[must_use]
    pub fn split_lookup<BK>(&self, split: &BK) -> (Self, Option<V>, Self)
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        // TODO this is atrociously slow, got to be a better way
        self.iter().fold(
            (GenericOrdMap::new(), None, GenericOrdMap::new()),
            |(l, m, r), (k, v)| match k.borrow().cmp(split) {
                Ordering::Less => (l.update(k.clone(), v.clone()), m, r),
                Ordering::Equal => (l, Some(v.clone()), r),
                Ordering::Greater => (l, m, r.update(k.clone(), v.clone())),
            },
        )
    }

    /// Construct a map with only the `n` smallest keys from a given
    /// map.
    #[must_use]
    pub fn take(&self, n: usize) -> Self {
        self.iter()
            .take(n)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Construct a map with the `n` smallest keys removed from a
    /// given map.
    #[must_use]
    pub fn skip(&self, n: usize) -> Self {
        self.iter()
            .skip(n)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Remove the smallest key from a map, and return its value as
    /// well as the updated map.
    #[must_use]
    pub fn without_min(&self) -> (Option<V>, Self) {
        let (pop, next) = self.without_min_with_key();
        (pop.map(|(_, v)| v), next)
    }

    /// Remove the smallest key from a map, and return that key, its
    /// value as well as the updated map.
    #[must_use]
    pub fn without_min_with_key(&self) -> (Option<(K, V)>, Self) {
        match self.get_min() {
            None => (None, self.clone()),
            Some((k, _)) => {
                let (key, value, next) = self.extract_with_key(k).unwrap();
                (Some((key, value)), next)
            }
        }
    }

    /// Remove the largest key from a map, and return its value as
    /// well as the updated map.
    #[must_use]
    pub fn without_max(&self) -> (Option<V>, Self) {
        let (pop, next) = self.without_max_with_key();
        (pop.map(|(_, v)| v), next)
    }

    /// Remove the largest key from a map, and return that key, its
    /// value as well as the updated map.
    #[must_use]
    pub fn without_max_with_key(&self) -> (Option<(K, V)>, Self) {
        match self.get_max() {
            None => (None, self.clone()),
            Some((k, _)) => {
                let (key, value, next) = self.extract_with_key(k).unwrap();
                (Some((key, value)), next)
            }
        }
    }

    /// Get the [`Entry`][Entry] for a key in the map for in-place manipulation.
    ///
    /// Time: O(log n)
    ///
    /// [Entry]: enum.Entry.html
    #[must_use]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, P> {
        if self.contains_key(&key) {
            Entry::Occupied(OccupiedEntry { map: self, key })
        } else {
            Entry::Vacant(VacantEntry { map: self, key })
        }
    }
}

// Entries

/// A handle for a key and its associated value.
pub enum Entry<'a, K, V, P>
where
    K: Ord + Clone,
    V: Clone,
    P: SharedPointerKind,
{
    /// An entry which exists in the map.
    Occupied(OccupiedEntry<'a, K, V, P>),
    /// An entry which doesn't exist in the map.
    Vacant(VacantEntry<'a, K, V, P>),
}

impl<'a, K, V, P> Entry<'a, K, V, P>
where
    K: Ord + Clone,
    V: Clone,
    P: SharedPointerKind,
{
    /// Insert the default value provided if there was no value
    /// already, and return a mutable reference to the value.
    pub fn or_insert(self, default: V) -> &'a mut V {
        self.or_insert_with(|| default)
    }

    /// Insert the default value from the provided function if there
    /// was no value already, and return a mutable reference to the
    /// value.
    pub fn or_insert_with<F>(self, default: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default()),
        }
    }

    /// Insert a default value if there was no value already, and
    /// return a mutable reference to the value.
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        #[allow(clippy::unwrap_or_default)]
        self.or_insert_with(Default::default)
    }

    /// Get the key for this entry.
    #[must_use]
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(entry) => entry.key(),
            Entry::Vacant(entry) => entry.key(),
        }
    }

    /// Call the provided function to modify the value if the value
    /// exists.
    #[must_use]
    pub fn and_modify<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        match &mut self {
            Entry::Occupied(ref mut entry) => f(entry.get_mut()),
            Entry::Vacant(_) => (),
        }
        self
    }
}

/// An entry for a mapping that already exists in the map.
pub struct OccupiedEntry<'a, K, V, P>
where
    K: Ord + Clone,
    V: Clone,
    P: SharedPointerKind,
{
    map: &'a mut GenericOrdMap<K, V, P>,
    key: K,
}

impl<'a, K, V, P> OccupiedEntry<'a, K, V, P>
where
    K: 'a + Ord + Clone,
    V: 'a + Clone,
    P: SharedPointerKind,
{
    /// Get the key for this entry.
    #[must_use]
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Remove this entry from the map and return the removed mapping.
    pub fn remove_entry(self) -> (K, V) {
        self.map
            .remove_with_key(&self.key)
            .expect("ordmap::OccupiedEntry::remove_entry: key has vanished!")
    }

    /// Get the current value.
    #[must_use]
    pub fn get(&self) -> &V {
        self.map.get(&self.key).unwrap()
    }

    /// Get a mutable reference to the current value.
    #[must_use]
    pub fn get_mut(&mut self) -> &mut V {
        self.map.get_mut(&self.key).unwrap()
    }

    /// Convert this entry into a mutable reference.
    #[must_use]
    pub fn into_mut(self) -> &'a mut V {
        self.map.get_mut(&self.key).unwrap()
    }

    /// Overwrite the current value.
    pub fn insert(&mut self, value: V) -> V {
        mem::replace(self.get_mut(), value)
    }

    /// Remove this entry from the map and return the removed value.
    pub fn remove(self) -> V {
        self.remove_entry().1
    }
}

/// An entry for a mapping that does not already exist in the map.
pub struct VacantEntry<'a, K, V, P>
where
    K: Ord + Clone,
    V: Clone,
    P: SharedPointerKind,
{
    map: &'a mut GenericOrdMap<K, V, P>,
    key: K,
}

impl<'a, K, V, P> VacantEntry<'a, K, V, P>
where
    K: 'a + Ord + Clone,
    V: 'a + Clone,
    P: SharedPointerKind,
{
    /// Get the key for this entry.
    #[must_use]
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Convert this entry into its key.
    #[must_use]
    pub fn into_key(self) -> K {
        self.key
    }

    /// Insert a value into this entry.
    pub fn insert(self, value: V) -> &'a mut V {
        self.map.insert(self.key.clone(), value);
        // TODO insert_mut ought to return this reference
        self.map.get_mut(&self.key).unwrap()
    }
}

// Core traits

impl<K, V, P: SharedPointerKind> Clone for GenericOrdMap<K, V, P> {
    /// Clone a map.
    ///
    /// Time: O(1)
    #[inline]
    fn clone(&self) -> Self {
        GenericOrdMap {
            size: self.size,
            root: self.root.clone(),
        }
    }
}

// TODO: Support PartialEq for OrdMap that have different P
impl<K, V, P> PartialEq for GenericOrdMap<K, V, P>
where
    K: Ord + PartialEq,
    V: PartialEq,
    P: SharedPointerKind,
{
    fn eq(&self, other: &GenericOrdMap<K, V, P>) -> bool {
        self.len() == other.len() && self.diff(other).next().is_none()
    }
}

impl<K: Ord + Eq, V: Eq, P: SharedPointerKind> Eq for GenericOrdMap<K, V, P> {}

// TODO: Support PartialOrd for OrdMap that have different P
impl<K, V, P> PartialOrd for GenericOrdMap<K, V, P>
where
    K: Ord,
    V: PartialOrd,
    P: SharedPointerKind,
{
    fn partial_cmp(&self, other: &GenericOrdMap<K, V, P>) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<K, V, P> Ord for GenericOrdMap<K, V, P>
where
    K: Ord,
    V: Ord,
    P: SharedPointerKind,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<K, V, P> Hash for GenericOrdMap<K, V, P>
where
    K: Ord + Hash,
    V: Hash,
    P: SharedPointerKind,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        for i in self.iter() {
            i.hash(state);
        }
    }
}

impl<K, V, P: SharedPointerKind> Default for GenericOrdMap<K, V, P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, P> Add for &GenericOrdMap<K, V, P>
where
    K: Ord + Clone,
    V: Clone,
    P: SharedPointerKind,
{
    type Output = GenericOrdMap<K, V, P>;

    fn add(self, other: Self) -> Self::Output {
        self.clone().union(other.clone())
    }
}

impl<K, V, P> Add for GenericOrdMap<K, V, P>
where
    K: Ord + Clone,
    V: Clone,
    P: SharedPointerKind,
{
    type Output = GenericOrdMap<K, V, P>;

    fn add(self, other: Self) -> Self::Output {
        self.union(other)
    }
}

impl<K, V, P> Sum for GenericOrdMap<K, V, P>
where
    K: Ord + Clone,
    V: Clone,
    P: SharedPointerKind,
{
    fn sum<I>(it: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        it.fold(Self::default(), |a, b| a + b)
    }
}

impl<K, V, RK, RV, P> Extend<(RK, RV)> for GenericOrdMap<K, V, P>
where
    K: Ord + Clone + From<RK>,
    V: Clone + From<RV>,
    P: SharedPointerKind,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (RK, RV)>,
    {
        for (key, value) in iter {
            self.insert(From::from(key), From::from(value));
        }
    }
}

impl<BK, K, V, P: SharedPointerKind> Index<&BK> for GenericOrdMap<K, V, P>
where
    BK: Ord + ?Sized,
    K: Ord + Borrow<BK>,
{
    type Output = V;

    fn index(&self, key: &BK) -> &Self::Output {
        match self.get(key) {
            None => panic!("OrdMap::index: invalid key"),
            Some(value) => value,
        }
    }
}

impl<BK, K, V, P> IndexMut<&BK> for GenericOrdMap<K, V, P>
where
    BK: Ord + ?Sized,
    K: Ord + Clone + Borrow<BK>,
    V: Clone,
    P: SharedPointerKind,
{
    fn index_mut(&mut self, key: &BK) -> &mut Self::Output {
        match self.get_mut(key) {
            None => panic!("OrdMap::index: invalid key"),
            Some(value) => value,
        }
    }
}

impl<K, V, P> Debug for GenericOrdMap<K, V, P>
where
    K: Ord + Debug,
    V: Debug,
    P: SharedPointerKind,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        let mut d = f.debug_map();
        for (k, v) in self.iter() {
            d.entry(k, v);
        }
        d.finish()
    }
}

// Iterators

/// An iterator over the key/value pairs of a map.
pub struct Iter<'a, K, V, P: SharedPointerKind> {
    it: NodeIter<'a, K, V, P>,
}

// We impl Clone instead of deriving it, because we want Clone even if K and V aren't.
impl<'a, K, V, P: SharedPointerKind> Clone for Iter<'a, K, V, P> {
    fn clone(&self) -> Self {
        Iter {
            it: self.it.clone(),
        }
    }
}

impl<'a, K, V, P> Iterator for Iter<'a, K, V, P>
where
    P: SharedPointerKind,
{
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next()
    }

    // We only construct an `Iter` when the range is full, meaning that we can
    // override `size_hint` and implement `ExactSizeIterator`.
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, K, V, P> DoubleEndedIterator for Iter<'a, K, V, P>
where
    P: SharedPointerKind,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.it.next_back()
    }
}

impl<'a, K, V, P> ExactSizeIterator for Iter<'a, K, V, P> where P: SharedPointerKind {}
impl<'a, K, V, P> FusedIterator for Iter<'a, K, V, P> where P: SharedPointerKind {}

/// An iterator over a range of key/value pairs in a map.
#[derive(Debug)]
pub struct RangedIter<'a, K, V, P: SharedPointerKind> {
    it: NodeIter<'a, K, V, P>,
}

// We impl Clone instead of deriving it, because we want Clone even if K and V aren't.
impl<'a, K, V, P: SharedPointerKind> Clone for RangedIter<'a, K, V, P> {
    fn clone(&self) -> Self {
        RangedIter {
            it: self.it.clone(),
        }
    }
}

impl<'a, K, V, P> Iterator for RangedIter<'a, K, V, P>
where
    P: SharedPointerKind,
{
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, K, V, P> DoubleEndedIterator for RangedIter<'a, K, V, P>
where
    P: SharedPointerKind,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.it.next_back()
    }
}
impl<'a, K, V, P> FusedIterator for RangedIter<'a, K, V, P> where P: SharedPointerKind {}

/// An iterator over the differences between two maps.
pub struct DiffIter<'a, 'b, K, V, P: SharedPointerKind> {
    it1: Cursor<'a, K, V, P>,
    it2: Cursor<'b, K, V, P>,
}

/// A description of a difference between two ordered maps.
#[derive(PartialEq, Eq, Debug)]
pub enum DiffItem<'a, 'b, K, V> {
    /// This value has been added to the new map.
    Add(&'b K, &'b V),
    /// This value has been changed between the two maps.
    Update {
        /// The old value.
        old: (&'a K, &'a V),
        /// The new value.
        new: (&'b K, &'b V),
    },
    /// This value has been removed from the new map.
    Remove(&'a K, &'a V),
}

impl<'a, 'b, K, V, P> Iterator for DiffIter<'a, 'b, K, V, P>
where
    K: Ord,
    V: PartialEq,
    P: SharedPointerKind,
{
    type Item = DiffItem<'a, 'b, K, V>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match (self.it1.peek(), self.it2.peek()) {
                (Some((k1, v1)), Some((k2, v2))) => match k1.cmp(k2) {
                    Ordering::Less => {
                        self.it1.next();
                        break Some(DiffItem::Remove(k1, v1));
                    }
                    Ordering::Equal => {
                        // Advance both iterator while trying to skip over the shared nodes.
                        self.it1.advance_skipping_shared(&mut self.it2);
                        if v1 != v2 {
                            break Some(DiffItem::Update {
                                old: (k1, v1),
                                new: (k2, v2),
                            });
                        }
                    }
                    Ordering::Greater => {
                        self.it2.next();
                        break Some(DiffItem::Add(k2, v2));
                    }
                },
                (Some((k1, v1)), None) => {
                    self.it1.next();
                    break Some(DiffItem::Remove(k1, v1));
                }
                (None, Some((k2, v2))) => {
                    self.it2.next();
                    break Some(DiffItem::Add(k2, v2));
                }
                (None, None) => break None,
            }
        }
    }
}

impl<'a, 'b, K, V, P> FusedIterator for DiffIter<'a, 'b, K, V, P>
where
    K: Ord,
    V: PartialEq,
    P: SharedPointerKind,
{
}

/// An iterator ove the keys of a map.
pub struct Keys<'a, K, V, P: SharedPointerKind> {
    it: Iter<'a, K, V, P>,
}

impl<'a, K, V, P> Iterator for Keys<'a, K, V, P>
where
    K: 'a + Ord,
    V: 'a,
    P: SharedPointerKind,
{
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, K, V, P> DoubleEndedIterator for Keys<'a, K, V, P>
where
    K: 'a + Ord,
    V: 'a,
    P: SharedPointerKind,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.it.next_back() {
            None => None,
            Some((k, _)) => Some(k),
        }
    }
}

impl<'a, K, V, P> ExactSizeIterator for Keys<'a, K, V, P>
where
    K: 'a + Ord,
    V: 'a,
    P: SharedPointerKind,
{
}

impl<'a, K, V, P> FusedIterator for Keys<'a, K, V, P>
where
    K: 'a + Ord,
    V: 'a,
    P: SharedPointerKind,
{
}

/// An iterator over the values of a map.
pub struct Values<'a, K, V, P: SharedPointerKind> {
    it: Iter<'a, K, V, P>,
}

impl<'a, K, V, P> Iterator for Values<'a, K, V, P>
where
    K: 'a + Ord,
    V: 'a,
    P: SharedPointerKind,
{
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, K, V, P> DoubleEndedIterator for Values<'a, K, V, P>
where
    K: 'a + Ord,
    V: 'a,
    P: SharedPointerKind,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.it.next_back() {
            None => None,
            Some((_, v)) => Some(v),
        }
    }
}

impl<'a, K, V, P> FusedIterator for Values<'a, K, V, P>
where
    K: 'a + Ord,
    V: 'a,
    P: SharedPointerKind,
{
}

impl<'a, K, V, P> ExactSizeIterator for Values<'a, K, V, P>
where
    K: 'a + Ord,
    V: 'a,
    P: SharedPointerKind,
{
}

impl<K, V, RK, RV, P> FromIterator<(RK, RV)> for GenericOrdMap<K, V, P>
where
    K: Ord + Clone + From<RK>,
    V: Clone + From<RV>,
    P: SharedPointerKind,
{
    fn from_iter<T>(i: T) -> Self
    where
        T: IntoIterator<Item = (RK, RV)>,
    {
        let mut m = GenericOrdMap::default();
        for (k, v) in i {
            m.insert(From::from(k), From::from(v));
        }
        m
    }
}

impl<'a, K, V, P> IntoIterator for &'a GenericOrdMap<K, V, P>
where
    K: Ord,
    P: SharedPointerKind,
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V, P>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K, V, P> IntoIterator for GenericOrdMap<K, V, P>
where
    K: Clone,
    V: Clone,
    P: SharedPointerKind,
{
    type Item = (K, V);
    type IntoIter = ConsumingIter<K, V, P>;

    fn into_iter(self) -> Self::IntoIter {
        ConsumingIter {
            it: NodeConsumingIter::new(self.root, self.size),
        }
    }
}

/// A consuming iterator over the elements of a map.
pub struct ConsumingIter<K, V, P: SharedPointerKind> {
    it: NodeConsumingIter<K, V, P>,
}

impl<K, V, P> Iterator for ConsumingIter<K, V, P>
where
    K: Clone,
    V: Clone,
    P: SharedPointerKind,
{
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        self.it.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<K: Clone, V: Clone, P: SharedPointerKind> DoubleEndedIterator for ConsumingIter<K, V, P> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.it.next_back()
    }
}

impl<K, V, P> ExactSizeIterator for ConsumingIter<K, V, P>
where
    K: Clone,
    V: Clone,
    P: SharedPointerKind,
{
}
impl<K, V, P> FusedIterator for ConsumingIter<K, V, P>
where
    K: Clone,
    V: Clone,
    P: SharedPointerKind,
{
}

// Conversions

impl<K, V, P: SharedPointerKind> AsRef<GenericOrdMap<K, V, P>> for GenericOrdMap<K, V, P> {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<K, V, OK, OV, P1, P2> From<&GenericOrdMap<&K, &V, P2>> for GenericOrdMap<OK, OV, P1>
where
    K: Ord + ToOwned<Owned = OK> + ?Sized,
    V: ToOwned<Owned = OV> + ?Sized,
    OK: Ord + Clone + Borrow<K>,
    OV: Clone + Borrow<V>,
    P1: SharedPointerKind,
    P2: SharedPointerKind,
{
    fn from(m: &GenericOrdMap<&K, &V, P2>) -> Self {
        m.iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }
}

impl<'a, K, V, RK, RV, OK, OV, P> From<&'a [(RK, RV)]> for GenericOrdMap<K, V, P>
where
    K: Ord + Clone + From<OK>,
    V: Clone + From<OV>,
    OK: Borrow<RK>,
    OV: Borrow<RV>,
    RK: ToOwned<Owned = OK>,
    RV: ToOwned<Owned = OV>,
    P: SharedPointerKind,
{
    fn from(m: &'a [(RK, RV)]) -> GenericOrdMap<K, V, P> {
        m.iter()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect()
    }
}

impl<K, V, RK, RV, P> From<Vec<(RK, RV)>> for GenericOrdMap<K, V, P>
where
    K: Ord + Clone + From<RK>,
    V: Clone + From<RV>,
    P: SharedPointerKind,
{
    fn from(m: Vec<(RK, RV)>) -> GenericOrdMap<K, V, P> {
        m.into_iter().collect()
    }
}

impl<'a, K: Ord, V, RK, RV, OK, OV, P> From<&'a Vec<(RK, RV)>> for GenericOrdMap<K, V, P>
where
    K: Ord + Clone + From<OK>,
    V: Clone + From<OV>,
    OK: Borrow<RK>,
    OV: Borrow<RV>,
    RK: ToOwned<Owned = OK>,
    RV: ToOwned<Owned = OV>,
    P: SharedPointerKind,
{
    fn from(m: &'a Vec<(RK, RV)>) -> GenericOrdMap<K, V, P> {
        m.iter()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect()
    }
}

impl<K: Ord, V, RK: Eq + Hash, RV, P> From<collections::HashMap<RK, RV>> for GenericOrdMap<K, V, P>
where
    K: Ord + Clone + From<RK>,
    V: Clone + From<RV>,
    P: SharedPointerKind,
{
    fn from(m: collections::HashMap<RK, RV>) -> GenericOrdMap<K, V, P> {
        m.into_iter().collect()
    }
}

impl<'a, K, V, OK, OV, RK, RV, P> From<&'a collections::HashMap<RK, RV>> for GenericOrdMap<K, V, P>
where
    K: Ord + Clone + From<OK>,
    V: Clone + From<OV>,
    OK: Borrow<RK>,
    OV: Borrow<RV>,
    RK: Hash + Eq + ToOwned<Owned = OK>,
    RV: ToOwned<Owned = OV>,
    P: SharedPointerKind,
{
    fn from(m: &'a collections::HashMap<RK, RV>) -> GenericOrdMap<K, V, P> {
        m.iter()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect()
    }
}

impl<K: Ord, V, RK, RV, P> From<collections::BTreeMap<RK, RV>> for GenericOrdMap<K, V, P>
where
    K: Ord + Clone + From<RK>,
    V: Clone + From<RV>,
    P: SharedPointerKind,
{
    fn from(m: collections::BTreeMap<RK, RV>) -> GenericOrdMap<K, V, P> {
        m.into_iter().collect()
    }
}

impl<'a, K: Ord, V, RK, RV, OK, OV, P> From<&'a collections::BTreeMap<RK, RV>>
    for GenericOrdMap<K, V, P>
where
    K: Ord + Clone + From<OK>,
    V: Clone + From<OV>,
    OK: Borrow<RK>,
    OV: Borrow<RV>,
    RK: Ord + ToOwned<Owned = OK>,
    RV: ToOwned<Owned = OV>,
    P: SharedPointerKind,
{
    fn from(m: &'a collections::BTreeMap<RK, RV>) -> GenericOrdMap<K, V, P> {
        m.iter()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect()
    }
}

impl<
        K: Ord + Hash + Eq + Clone,
        V: Clone,
        S: BuildHasher + Clone,
        P1: SharedPointerKind,
        P2: SharedPointerKind,
    > From<GenericHashMap<K, V, S, P2>> for GenericOrdMap<K, V, P1>
{
    fn from(m: GenericHashMap<K, V, S, P2>) -> Self {
        m.into_iter().collect()
    }
}

impl<
        'a,
        K: Ord + Hash + Eq + Clone,
        V: Clone,
        S: BuildHasher + Clone,
        P1: SharedPointerKind,
        P2: SharedPointerKind,
    > From<&'a GenericHashMap<K, V, S, P2>> for GenericOrdMap<K, V, P1>
{
    fn from(m: &'a GenericHashMap<K, V, S, P2>) -> Self {
        m.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
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
    pub use crate::proptest::ord_map;
}

// Tests

#[cfg(test)]
mod test {
    use std::collections::BTreeMap;

    use super::*;
    use crate::proptest::*;
    #[rustfmt::skip]
    use ::proptest::num::{i16, usize};
    #[rustfmt::skip]
    use ::proptest::{bool, collection, proptest};
    use static_assertions::{assert_impl_all, assert_not_impl_any};

    assert_impl_all!(OrdMap<i32, i32>: Send, Sync);
    assert_not_impl_any!(OrdMap<i32, *const i32>: Send, Sync);
    assert_not_impl_any!(OrdMap<*const i32, i32>: Send, Sync);
    assert_covariant!(OrdMap<T, i32> in T);
    assert_covariant!(OrdMap<i32, T> in T);

    #[test]
    fn iterates_in_order() {
        let map = ordmap! {
            2 => 22,
            1 => 11,
            3 => 33,
            8 => 88,
            9 => 99,
            4 => 44,
            5 => 55,
            7 => 77,
            6 => 66
        };
        let mut it = map.iter();
        assert_eq!(it.next(), Some((&1, &11)));
        assert_eq!(it.next(), Some((&2, &22)));
        assert_eq!(it.next(), Some((&3, &33)));
        assert_eq!(it.next(), Some((&4, &44)));
        assert_eq!(it.next(), Some((&5, &55)));
        assert_eq!(it.next(), Some((&6, &66)));
        assert_eq!(it.next(), Some((&7, &77)));
        assert_eq!(it.next(), Some((&8, &88)));
        assert_eq!(it.next(), Some((&9, &99)));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn into_iter() {
        let map = ordmap! {
            2 => 22,
            1 => 11,
            3 => 33,
            8 => 88,
            9 => 99,
            4 => 44,
            5 => 55,
            7 => 77,
            6 => 66
        };
        let mut vec = vec![];
        for (k, v) in map {
            assert_eq!(k * 11, v);
            vec.push(k)
        }
        assert_eq!(vec, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    struct PanicOnClone;

    impl Clone for PanicOnClone {
        fn clone(&self) -> Self {
            panic!("PanicOnClone::clone called")
        }
    }

    #[test]
    fn into_iter_no_clone() {
        let mut map = OrdMap::new();
        let mut map_rev = OrdMap::new();
        for i in 0..10_000 {
            map.insert(i, PanicOnClone);
            map_rev.insert(i, PanicOnClone);
        }
        let _ = map.into_iter().collect::<Vec<_>>();
        let _ = map_rev.into_iter().rev().collect::<Vec<_>>();
    }

    #[test]
    fn iter_no_clone() {
        let mut map = OrdMap::new();
        for i in 0..10_000 {
            map.insert(i, PanicOnClone);
        }
        let _ = map.iter().collect::<Vec<_>>();
        let _ = map.iter().rev().collect::<Vec<_>>();
    }

    #[test]
    fn deletes_correctly() {
        let map = ordmap! {
            2 => 22,
            1 => 11,
            3 => 33,
            8 => 88,
            9 => 99,
            4 => 44,
            5 => 55,
            7 => 77,
            6 => 66
        };
        assert_eq!(map.extract(&11), None);
        let (popped, less) = map.extract(&5).unwrap();
        assert_eq!(popped, 55);
        let mut it = less.iter();
        assert_eq!(it.next(), Some((&1, &11)));
        assert_eq!(it.next(), Some((&2, &22)));
        assert_eq!(it.next(), Some((&3, &33)));
        assert_eq!(it.next(), Some((&4, &44)));
        assert_eq!(it.next(), Some((&6, &66)));
        assert_eq!(it.next(), Some((&7, &77)));
        assert_eq!(it.next(), Some((&8, &88)));
        assert_eq!(it.next(), Some((&9, &99)));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn debug_output() {
        assert_eq!(
            format!("{:?}", ordmap! { 3 => 4, 5 => 6, 1 => 2 }),
            "{1: 2, 3: 4, 5: 6}"
        );
    }

    #[test]
    fn equality2() {
        let v1 = "1".to_string();
        let v2 = "1".to_string();
        assert_eq!(v1, v2);
        let p1 = Vec::<String>::new();
        let p2 = Vec::<String>::new();
        assert_eq!(p1, p2);
        let c1: OrdMap<_, _> = OrdMap::unit(v1, p1);
        let c2: OrdMap<_, _> = OrdMap::unit(v2, p2);
        assert_eq!(c1, c2);
    }

    #[test]
    fn insert_remove_single_mut() {
        let mut m = OrdMap::new();
        m.insert(0, 0);
        assert_eq!(OrdMap::<_, _>::unit(0, 0), m);
        m.remove(&0);
        assert_eq!(OrdMap::new(), m);
    }

    #[test]
    fn double_ended_iterator_1() {
        let m = ordmap! {1 => 1, 2 => 2, 3 => 3, 4 => 4};
        let mut it = m.iter();
        assert_eq!(Some((&1, &1)), it.next());
        assert_eq!(Some((&4, &4)), it.next_back());
        assert_eq!(Some((&2, &2)), it.next());
        assert_eq!(Some((&3, &3)), it.next_back());
        assert_eq!(None, it.next());
    }

    #[test]
    fn double_ended_iterator_2() {
        let m = ordmap! {1 => 1, 2 => 2, 3 => 3, 4 => 4};
        let mut it = m.iter();
        assert_eq!(Some((&1, &1)), it.next());
        assert_eq!(Some((&4, &4)), it.next_back());
        assert_eq!(Some((&2, &2)), it.next());
        assert_eq!(Some((&3, &3)), it.next_back());
        assert_eq!(None, it.next_back());
    }

    #[test]
    fn safe_mutation() {
        let v1 = OrdMap::<_, _>::from_iter((0..131_072).map(|i| (i, i)));
        let mut v2 = v1.clone();
        v2.insert(131_000, 23);
        assert_eq!(Some(&23), v2.get(&131_000));
        assert_eq!(Some(&131_000), v1.get(&131_000));
    }

    #[test]
    fn index_operator() {
        let mut map = ordmap! {1 => 2, 3 => 4, 5 => 6};
        assert_eq!(4, map[&3]);
        map[&3] = 8;
        assert_eq!(ordmap! {1 => 2, 3 => 8, 5 => 6}, map);
    }

    #[test]
    fn entry_api() {
        let mut map = ordmap! {"bar" => 5};
        map.entry("foo").and_modify(|v| *v += 5).or_insert(1);
        assert_eq!(1, map[&"foo"]);
        map.entry("foo").and_modify(|v| *v += 5).or_insert(1);
        assert_eq!(6, map[&"foo"]);
        map.entry("bar").and_modify(|v| *v += 5).or_insert(1);
        assert_eq!(10, map[&"bar"]);
        assert_eq!(
            10,
            match map.entry("bar") {
                Entry::Occupied(entry) => entry.remove(),
                _ => panic!(),
            }
        );
        assert!(!map.contains_key(&"bar"));
    }

    #[test]
    fn match_string_keys_with_string_slices() {
        let mut map: OrdMap<String, i32> =
            From::from(&ordmap! { "foo" => &1, "bar" => &2, "baz" => &3 });
        assert_eq!(Some(&1), map.get("foo"));
        map = map.without("foo");
        assert_eq!(Some(3), map.remove("baz"));
        map["bar"] = 8;
        assert_eq!(8, map["bar"]);
    }

    #[test]
    fn ranged_iter() {
        let map: OrdMap<i32, i32> = ordmap![1=>2, 2=>3, 3=>4, 4=>5, 5=>6, 7=>8];
        let range: Vec<(i32, i32)> = map.range(..).map(|(k, v)| (*k, *v)).collect();
        assert_eq!(vec![(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8)], range);
        let range: Vec<(i32, i32)> = map.range(..).rev().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(vec![(7, 8), (5, 6), (4, 5), (3, 4), (2, 3), (1, 2)], range);
        let range: Vec<(i32, i32)> = map.range(&2..&5).map(|(k, v)| (*k, *v)).collect();
        assert_eq!(vec![(2, 3), (3, 4), (4, 5)], range);
        let range: Vec<(i32, i32)> = map.range(&2..&5).rev().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(vec![(4, 5), (3, 4), (2, 3)], range);
        let range: Vec<(i32, i32)> = map.range(&3..).map(|(k, v)| (*k, *v)).collect();
        assert_eq!(vec![(3, 4), (4, 5), (5, 6), (7, 8)], range);
        let range: Vec<(i32, i32)> = map.range(&3..).rev().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(vec![(7, 8), (5, 6), (4, 5), (3, 4)], range);
        let range: Vec<(i32, i32)> = map.range(..4).map(|(k, v)| (*k, *v)).collect();
        assert_eq!(vec![(1, 2), (2, 3), (3, 4)], range);
        let range: Vec<(i32, i32)> = map.range(..&4).rev().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(vec![(3, 4), (2, 3), (1, 2)], range);
        let range: Vec<(i32, i32)> = map.range(..=&3).map(|(k, v)| (*k, *v)).collect();
        assert_eq!(vec![(1, 2), (2, 3), (3, 4)], range);
        let range: Vec<(i32, i32)> = map.range(..=&3).rev().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(vec![(3, 4), (2, 3), (1, 2)], range);
        let range: Vec<(i32, i32)> = map.range(..&6).map(|(k, v)| (*k, *v)).collect();
        assert_eq!(vec![(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)], range);
        let range: Vec<(i32, i32)> = map.range(..=&6).map(|(k, v)| (*k, *v)).collect();
        assert_eq!(vec![(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)], range);

        assert_eq!(map.range(&2..&5).size_hint(), (0, Some(6)));
        let mut iter = map.range(&2..&5);
        iter.next();
        assert_eq!(iter.size_hint(), (0, Some(5)));
    }

    #[test]
    fn range_iter_big() {
        use crate::nodes::btree::NODE_SIZE;
        use std::ops::Bound::Included;
        const N: usize = NODE_SIZE * NODE_SIZE * NODE_SIZE / 2; // enough for a sizeable 3 level tree

        let data = (1usize..N).filter(|i| i % 2 == 0).map(|i| (i, ()));
        let bmap = data
            .clone()
            .collect::<std::collections::BTreeMap<usize, ()>>();
        let omap = data.collect::<OrdMap<usize, ()>>();
        assert_eq!(bmap.len(), omap.len());

        for i in (0..NODE_SIZE * 5).chain(N - NODE_SIZE * 5..=N + 1) {
            assert_eq!(omap.range(i..).count(), bmap.range(i..).count());
            assert_eq!(omap.range(..i).count(), bmap.range(..i).count());
            assert_eq!(
                omap.range(i..(i + 7)).count(),
                bmap.range(i..(i + 7)).count()
            );
            assert_eq!(
                omap.range(i..=(i + 7)).count(),
                bmap.range(i..=(i + 7)).count()
            );
            assert_eq!(
                omap.range((Included(i), Included(i + 7))).count(),
                bmap.range((Included(i), Included(i + 7))).count(),
            );
            assert_eq!(omap.range(..=i).next_back(), omap.get_prev(&i));
            assert_eq!(omap.range(i..).next(), omap.get_next(&i));
        }
    }

    #[test]
    fn issue_124() {
        let mut map = OrdMap::new();
        let contents = include_str!("test-fixtures/issue_124.txt");
        for line in contents.lines() {
            if let Some(tail) = line.strip_prefix("insert ") {
                map.insert(tail.parse::<u32>().unwrap(), 0);
            } else if let Some(tail) = line.strip_prefix("remove ") {
                map.remove(&tail.parse::<u32>().unwrap());
            }
        }
    }

    fn expected_diff<'a, K, V, P>(
        a: &'a GenericOrdMap<K, V, P>,
        b: &'a GenericOrdMap<K, V, P>,
    ) -> Vec<DiffItem<'a, 'a, K, V>>
    where
        K: Ord + Clone,
        V: PartialEq + Clone,
        P: SharedPointerKind,
    {
        let mut diff = Vec::new();
        for (k, v) in a.iter() {
            if let Some(v2) = b.get(k) {
                if v != v2 {
                    diff.push(DiffItem::Update {
                        old: (k, v),
                        new: (k, v2),
                    });
                }
            } else {
                diff.push(DiffItem::Remove(k, v));
            }
        }
        for (k, v) in b.iter() {
            if a.get(k).is_none() {
                diff.push(DiffItem::Add(k, v));
            }
        }
        fn diff_item_key<'a, 'b, K, V>(di: &'a DiffItem<'b, 'b, K, V>) -> &'b K {
            match di {
                DiffItem::Add(k, _) => k,
                DiffItem::Remove(k, _) => k,
                DiffItem::Update { old: (k, _), .. } => k,
            }
        }
        diff.sort_unstable_by(|a, b| diff_item_key(a).cmp(diff_item_key(b)));
        diff
    }

    proptest! {
        #[test]
        fn length(ref input in collection::btree_map(i16::ANY, i16::ANY, 0..1000)) {
            let map: OrdMap<i16, i16> = OrdMap::from(input.clone());
            assert_eq!(input.len(), map.len());
        }

        #[test]
        fn order(ref input in collection::hash_map(i16::ANY, i16::ANY, 0..1000)) {
            let map: OrdMap<i16, i16> = OrdMap::from(input.clone());
            let keys = map.keys().cloned().collect::<Vec<_>>();
            let mut expected_keys = input.keys().cloned().collect::<Vec<_>>();
            expected_keys.sort();
            assert_eq!(keys, expected_keys);
        }

        #[test]
        fn overwrite_values(ref vec in collection::vec((i16::ANY, i16::ANY), 1..1000), index_rand in usize::ANY, new_val in i16::ANY) {
            let index = vec[index_rand % vec.len()].0;
            let map1 = OrdMap::<_, _>::from_iter(vec.clone());
            let map2 = map1.update(index, new_val);
            for (k, v) in map2 {
                if k == index {
                    assert_eq!(v, new_val);
                } else {
                    match map1.get(&k) {
                        None => panic!("map1 didn't have key {:?}", k),
                        Some(other_v) => {
                            assert_eq!(v, *other_v);
                        }
                    }
                }
            }
        }

        #[test]
        fn delete_values(ref vec in collection::vec((usize::ANY, usize::ANY), 1..1000), index_rand in usize::ANY) {
            let index = vec[index_rand % vec.len()].0;
            let map1: OrdMap<usize, usize> = OrdMap::from_iter(vec.clone());
            let map2 = map1.without(&index);
            assert_eq!(map1.len(), map2.len() + 1);
            for k in map2.keys() {
                assert_ne!(*k, index);
            }
        }

        #[test]
        fn insert_and_delete_values(
            ref input in ord_map(0usize..64, 0usize..64, 1..1000),
            ref ops in collection::vec((bool::ANY, usize::ANY, usize::ANY), 1..1000)
        ) {
            let mut map = input.clone();
            let mut tree: collections::BTreeMap<usize, usize> = input.iter().map(|(k, v)| (*k, *v)).collect();
            for (ins, key, val) in ops {
                if *ins {
                    tree.insert(*key, *val);
                    map = map.update(*key, *val)
                } else {
                    tree.remove(key);
                    map = map.without(key)
                }
            }
            assert!(map.iter().map(|(k, v)| (*k, *v)).eq(tree.iter().map(|(k, v)| (*k, *v))));
        }

        #[test]
        fn proptest_works(ref m in ord_map(0..9999, ".*", 10..100)) {
            assert!(m.len() < 100);
            assert!(m.len() >= 10);
        }

        #[test]
        fn insert_and_length(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..1000)) {
            let mut map = OrdMap::new();
            for (k, v) in m.iter() {
                map = map.update(*k, *v)
            }
            assert_eq!(m.len(), map.len());
        }

        #[test]
        fn from_iterator(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..1000)) {
            let map: OrdMap<i16, i16> =
                FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            assert_eq!(m.len(), map.len());
        }

        #[test]
        fn iterate_over(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..1000)) {
            let map: OrdMap<i16, i16> =
                OrdMap::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            let expected = m.iter().map(|(k, v)| (*k, *v)).collect::<BTreeMap<_, _>>();
            assert!(map.iter().eq(expected.iter()));
        }

        #[test]
        fn iterate_over_rev(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..1000)) {
            let map: OrdMap<i16, i16> =
                OrdMap::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            let expected = m.iter().map(|(k, v)| (*k, *v)).collect::<BTreeMap<_, _>>();
            assert!(map.iter().rev().eq(expected.iter().rev()));
        }

        #[test]
        fn equality(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..1000)) {
            let map1: OrdMap<i16, i16> =
                FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            let map2: OrdMap<i16, i16> =
                FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            assert_eq!(map1, map2);
        }

        #[test]
        fn lookup(ref m in ord_map(i16::ANY, i16::ANY, 0..1000)) {
            let map: OrdMap<i16, i16> =
                FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            for (k, v) in m.iter() {
                assert_eq!(Some(*v), map.get(k).cloned());
            }
        }

        #[test]
        fn remove(ref m in ord_map(i16::ANY, i16::ANY, 0..1000)) {
            let mut map: OrdMap<i16, i16> =
                OrdMap::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            for k in m.keys() {
                let l = map.len();
                assert_eq!(m.get(k).cloned(), map.get(k).cloned());
                map = map.without(k);
                assert_eq!(None, map.get(k));
                assert_eq!(l - 1, map.len());
            }
        }

        #[test]
        fn insert_mut(ref m in ord_map(i16::ANY, i16::ANY, 0..1000)) {
            let mut mut_map = OrdMap::new();
            let mut map = OrdMap::new();
            for (k, v) in m.iter() {
                map = map.update(*k, *v);
                mut_map.insert(*k, *v);
            }
            assert_eq!(map, mut_map);
        }

        #[test]
        fn remove_mut(ref orig in ord_map(i16::ANY, i16::ANY, 0..1000)) {
            let mut map = orig.clone();
            for key in orig.keys() {
                let len = map.len();
                assert_eq!(orig.get(key), map.get(key));
                assert_eq!(orig.get(key).cloned(), map.remove(key));
                assert_eq!(None, map.get(key));
                assert_eq!(len - 1, map.len());
            }
        }

        #[test]
        fn remove_alien(ref orig in collection::hash_map(i16::ANY, i16::ANY, 0..1000)) {
            let mut map: OrdMap<i16, i16> = OrdMap::from(orig.clone());
            for key in orig.keys() {
                let len = map.len();
                assert_eq!(orig.get(key), map.get(key));
                assert_eq!(orig.get(key).cloned(), map.remove(key));
                assert_eq!(None, map.get(key));
                assert_eq!(len - 1, map.len());
            }
        }

        #[test]
        fn delete_and_reinsert(
            ref input in collection::hash_map(i16::ANY, i16::ANY, 1..1000),
            index_rand in usize::ANY
        ) {
            let index = *input.keys().nth(index_rand % input.len()).unwrap();
            let map1 = OrdMap::from_iter(input.clone());
            let (val, map2): (i16, _) = map1.extract(&index).unwrap();
            let map3 = map2.update(index, val);
            for key in map2.keys() {
                assert!(*key != index);
            }
            assert_eq!(map1.len(), map2.len() + 1);
            assert_eq!(map1, map3);
        }

        #[test]
        fn exact_size_iterator(ref m in ord_map(i16::ANY, i16::ANY, 1..1000)) {
            let mut should_be = m.len();
            let mut it = m.iter();
            loop {
                assert_eq!(should_be, it.len());
                match it.next() {
                    None => break,
                    Some(_) => should_be -= 1,
                }
            }
            assert_eq!(0, it.len());
        }

        #[test]
        fn diff_all_values(a in collection::vec((usize::ANY, usize::ANY), 1..1000), b in collection::vec((usize::ANY, usize::ANY), 1..1000)) {
            let a: OrdMap<usize, usize> = OrdMap::from(a);
            let b: OrdMap<usize, usize> = OrdMap::from(b);

            let diff: Vec<_> = a.diff(&b).collect();
            let expected = expected_diff(&a, &b);
            assert_eq!(expected, diff);
        }

        #[test]
        fn diff_all_values_shared(a in collection::vec((usize::ANY, usize::ANY), 1..1000), ops in collection::vec((usize::ANY, usize::ANY), 1..1000)) {
            let a: OrdMap<usize, usize> = OrdMap::from(a);
            let mut b = a.clone();
            for (k, v) in ops {
                b.insert(k, v);
            }

            let diff: Vec<_> = a.diff(&b).collect();
            let expected = expected_diff(&a, &b);
            assert_eq!(expected, diff);
        }

        #[test]
        fn union(ref map1 in ord_map(i16::ANY, i16::ANY, 0..100),
                 ref map2 in ord_map(i16::ANY, i16::ANY, 0..100)) {
            let union_map = map1.clone().union(map2.clone());

            for k in map1.keys() {
                assert!(union_map.contains_key(k));
            }

            for k in map2.keys() {
                assert!(union_map.contains_key(k));
            }

            for (k, v) in union_map.iter() {
                assert_eq!(v, map1.get(k).or_else(|| map2.get(k)).unwrap());
            }
        }
    }
}
