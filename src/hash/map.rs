// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! An unordered map.
//!
//! An immutable hash map using [hash array mapped tries][1].
//!
//! Most operations on this map are O(log<sub>x</sub> n) for a
//! suitably high *x* that it should be nearly O(1) for most maps.
//! Because of this, it's a great choice for a generic map as long as
//! you don't mind that keys will need to implement
//! [`Hash`][std::hash::Hash] and [`Eq`][std::cmp::Eq].
//!
//! Map entries will have a predictable order based on the hasher
//! being used. Unless otherwise specified, this will be the standard
//! [`RandomState`][std::collections::hash_map::RandomState] hasher.
//!
//! [1]: https://en.wikipedia.org/wiki/Hash_array_mapped_trie
//! [std::cmp::Eq]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
//! [std::hash::Hash]: https://doc.rust-lang.org/std/hash/trait.Hash.html
//! [std::collections::hash_map::RandomState]: https://doc.rust-lang.org/std/collections/hash_map/struct.RandomState.html

use std::borrow::Borrow;
use std::collections;
use std::collections::hash_map::RandomState;
use std::fmt::{Debug, Error, Formatter};
use std::hash::{BuildHasher, Hash};
use std::iter::{FromIterator, FusedIterator, Sum};
use std::mem;
use std::ops::{Add, Index, IndexMut};

use archery::{SharedPointer, SharedPointerKind};

use crate::nodes::hamt::{
    hash_key, Drain as NodeDrain, HashBits, HashValue, Iter as NodeIter, IterMut as NodeIterMut,
    Node,
};
use crate::shared_ptr::DefaultSharedPtr;

/// Construct a hash map from a sequence of key/value pairs.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate imbl;
/// # use imbl::HashMap;
/// # fn main() {
/// assert_eq!(
///   hashmap!{
///     1 => 11,
///     2 => 22,
///     3 => 33
///   },
///   HashMap::from(vec![(1, 11), (2, 22), (3, 33)])
/// );
/// # }
/// ```
#[macro_export]
macro_rules! hashmap {
    () => { $crate::hashmap::HashMap::new() };

    ( $( $key:expr => $value:expr ),* ) => {{
        let mut map = $crate::hashmap::HashMap::new();
        $({
            map.insert($key, $value);
        })*;
        map
    }};

    ( $( $key:expr => $value:expr ,)* ) => {{
        let mut map = $crate::hashmap::HashMap::new();
        $({
            map.insert($key, $value);
        })*;
        map
    }};
}

/// Type alias for [`GenericHashMap`] that uses [`std::hash::RandomState`] as the default hasher and [`DefaultSharedPtr`] as the pointer type.
///
/// [GenericHashMap]: ./struct.GenericHashMap.html
/// [`std::hash::RandomState`]: https://doc.rust-lang.org/stable/std/collections/hash_map/struct.RandomState.html
/// [DefaultSharedPtr]: ../shared_ptr/type.DefaultSharedPtr.html
pub type HashMap<K, V> = GenericHashMap<K, V, RandomState, DefaultSharedPtr>;

/// An unordered map.
///
/// An immutable hash map using [hash array mapped tries] [1].
///
/// Most operations on this map are O(log<sub>x</sub> n) for a
/// suitably high *x* that it should be nearly O(1) for most maps.
/// Because of this, it's a great choice for a generic map as long as
/// you don't mind that keys will need to implement
/// [`Hash`][std::hash::Hash] and [`Eq`][std::cmp::Eq].
///
/// Map entries will have a predictable order based on the hasher
/// being used. Unless otherwise specified, this will be the standard
/// [`RandomState`][std::collections::hash_map::RandomState] hasher.
///
/// [1]: https://en.wikipedia.org/wiki/Hash_array_mapped_trie
/// [std::cmp::Eq]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
/// [std::hash::Hash]: https://doc.rust-lang.org/std/hash/trait.Hash.html
/// [std::collections::hash_map::RandomState]: https://doc.rust-lang.org/std/collections/hash_map/struct.RandomState.html
pub struct GenericHashMap<K, V, S, P: SharedPointerKind> {
    size: usize,
    root: Option<SharedPointer<Node<(K, V), P>, P>>,
    hasher: S,
}

impl<K, V> HashValue for (K, V)
where
    K: Eq,
{
    type Key = K;

    fn extract_key(&self) -> &Self::Key {
        &self.0
    }

    fn ptr_eq(&self, _other: &Self) -> bool {
        false
    }
}

impl<K, V, P> GenericHashMap<K, V, RandomState, P>
where
    K: Hash + Eq + Clone,
    V: Clone,
    P: SharedPointerKind,
{
    /// Construct a hash map with a single mapping.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::HashMap;
    /// let map = HashMap::unit(123, "onetwothree");
    /// assert_eq!(
    ///   map.get(&123),
    ///   Some(&"onetwothree")
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn unit(k: K, v: V) -> GenericHashMap<K, V, RandomState, P> {
        GenericHashMap::new().update(k, v)
    }
}

impl<K, V, S, P: SharedPointerKind> GenericHashMap<K, V, S, P> {
    /// Construct an empty hash map.
    #[inline]
    #[must_use]
    pub fn new() -> Self
    where
        S: Default,
    {
        Self::default()
    }

    /// Test whether a hash map is empty.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::hashmap::HashMap;
    /// assert!(
    ///   !hashmap!{1 => 2}.is_empty()
    /// );
    /// assert!(
    ///   HashMap::<i32, i32>::new().is_empty()
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the size of a hash map.
    ///
    /// Time: O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::hashmap::HashMap;
    /// assert_eq!(3, hashmap!{
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
            (Some(a), Some(b)) => SharedPointer::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        }
    }

    /// Construct an empty hash map using the provided hasher.
    #[inline]
    #[must_use]
    pub fn with_hasher(hasher: S) -> Self {
        GenericHashMap {
            size: 0,
            hasher,
            root: None,
        }
    }

    /// Get a reference to the map's [`BuildHasher`][BuildHasher].
    ///
    /// [BuildHasher]: https://doc.rust-lang.org/std/hash/trait.BuildHasher.html
    #[must_use]
    pub fn hasher(&self) -> &S {
        &self.hasher
    }

    /// Construct an empty hash map using the same hasher as the
    /// current hash map.
    #[inline]
    #[must_use]
    pub fn new_from<K1, V1>(&self) -> GenericHashMap<K1, V1, S, P>
    where
        K1: Hash + Eq + Clone,
        V1: Clone,
        S: Clone,
    {
        GenericHashMap {
            size: 0,
            root: None,
            hasher: self.hasher.clone(),
        }
    }

    /// Get an iterator over the key/value pairs of a hash map.
    ///
    /// Please note that the order is consistent between maps using
    /// the same hasher, but no other ordering guarantee is offered.
    /// Items will not come out in insertion order or sort order.
    /// They will, however, come out in the same order every time for
    /// the same map.
    #[inline]
    #[must_use]
    pub fn iter(&self) -> Iter<'_, K, V, P> {
        Iter {
            it: NodeIter::new(self.root.as_deref(), self.size),
        }
    }

    /// Get an iterator over a hash map's keys.
    ///
    /// Please note that the order is consistent between maps using
    /// the same hasher, but no other ordering guarantee is offered.
    /// Items will not come out in insertion order or sort order.
    /// They will, however, come out in the same order every time for
    /// the same map.
    #[inline]
    #[must_use]
    pub fn keys(&self) -> Keys<'_, K, V, P> {
        Keys {
            it: NodeIter::new(self.root.as_deref(), self.size),
        }
    }

    /// Get an iterator over a hash map's values.
    ///
    /// Please note that the order is consistent between maps using
    /// the same hasher, but no other ordering guarantee is offered.
    /// Items will not come out in insertion order or sort order.
    /// They will, however, come out in the same order every time for
    /// the same map.
    #[inline]
    #[must_use]
    pub fn values(&self) -> Values<'_, K, V, P> {
        Values {
            it: NodeIter::new(self.root.as_deref(), self.size),
        }
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
    /// # use imbl::HashMap;
    /// let mut map = hashmap![1=>1, 2=>2, 3=>3];
    /// map.clear();
    /// assert!(map.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.root = None;
        self.size = 0;
    }

    /// Print a summary of the HashMap structure showing per-level statistics.
    /// This includes the number of nodes at each level and the distribution of child types.
    #[cfg(test)]
    pub fn print_structure_summary(&self) {
        use crate::nodes::hamt::Entry as NodeEntry;
        use std::collections::VecDeque;

        println!("HashMap Structure Summary:");

        #[derive(Default, Debug)]
        struct LevelStats {
            node_count: usize,
            value_count: usize,
            collision_count: usize,
            collision_entry_sum: usize,
            child_node_count: usize,
            small_simd_node_count: usize,
            large_simd_node_count: usize,
            small_simd_entry_sum: usize,
            large_simd_entry_sum: usize,
            total_entries: usize,
        }

        if self.root.is_none() {
            println!("  Empty HashMap (no root node)");
            println!("  Total entries: 0");
            return;
        }

        let mut level_stats: Vec<LevelStats> = Vec::new();
        let mut queue: VecDeque<(usize, SharedPointer<Node<(K, V), P>, P>)> = VecDeque::new();
        let mut max_depth = 0;

        // Start with root node at level 0
        if let Some(ref root) = self.root {
            queue.push_back((0, root.clone()));
        }

        // BFS traversal to collect statistics
        while let Some((level, node)) = queue.pop_front() {
            // Ensure we have stats for this level
            while level_stats.len() <= level {
                level_stats.push(LevelStats::default());
            }

            let stats = &mut level_stats[level];
            stats.node_count += 1;

            // Analyze this node's entries
            node.analyze_structure(|entry| {
                stats.total_entries += 1;
                match entry {
                    NodeEntry::Value(_, _) => {
                        stats.value_count += 1;
                        max_depth = max_depth.max(level);
                    }
                    NodeEntry::Collision(_coll) => {
                        stats.collision_count += 1;
                        // stats.collision_entry_sum += coll.len();
                        max_depth = max_depth.max(level);
                    }
                    NodeEntry::HamtNode(child_node) => {
                        stats.child_node_count += 1;
                        queue.push_back((level + 1, child_node.clone()));
                    }
                    NodeEntry::SmallSimdNode(small_node) => {
                        stats.small_simd_node_count += 1;
                        stats.small_simd_entry_sum += small_node.len();
                        max_depth = max_depth.max(level + 1);
                    }
                    NodeEntry::LargeSimdNode(large_node) => {
                        stats.large_simd_node_count += 1;
                        stats.large_simd_entry_sum += large_node.len();
                        max_depth = max_depth.max(level + 1);
                    }
                }
            })
        }

        // Print the summary
        println!(
            "  Hash level size (bits): {}",
            crate::config::HASH_LEVEL_SIZE
        );
        println!(
            "  Branching factor: {}",
            2_usize.pow(crate::config::HASH_LEVEL_SIZE as u32)
        );
        println!("  Total entries: {}", self.size);
        println!("  Tree depth: {} levels", max_depth + 1);
        println!();

        for (level, stats) in level_stats.iter().enumerate() {
            println!("  Level {}:", level);
            println!("    Nodes: {}", stats.node_count);

            if stats.total_entries > 0 {
                let avg_entries = stats.total_entries as f64 / stats.node_count as f64;
                println!("    Average entries per node: {:.2}", avg_entries);

                println!("    Entry types:");
                println!(
                    "      Values: {} ({:.1}%)",
                    stats.value_count,
                    (stats.value_count as f64 / stats.total_entries as f64) * 100.0
                );
                println!(
                    "      Collisions: {} (avg len: {:.1}) ({:.1}%)",
                    stats.collision_count,
                    if stats.collision_count > 0 {
                        stats.collision_entry_sum as f64 / stats.collision_count as f64
                    } else {
                        0.0
                    },
                    (stats.collision_count as f64 / stats.total_entries as f64) * 100.0
                );
                println!(
                    "      Child HAMT nodes: {} ({:.1}%)",
                    stats.child_node_count,
                    (stats.child_node_count as f64 / stats.total_entries as f64) * 100.0
                );
                if stats.small_simd_node_count > 0 {
                    println!(
                        "      Small SIMD leaf nodes: {} ({:.1}%) [total values: {}]",
                        stats.small_simd_node_count,
                        (stats.small_simd_node_count as f64 / stats.total_entries as f64) * 100.0,
                        stats.small_simd_entry_sum
                    );
                    println!(
                        "        → Avg values per small SIMD node: {:.1}",
                        stats.small_simd_entry_sum as f64 / stats.small_simd_node_count as f64
                    );
                }
                if stats.large_simd_node_count > 0 {
                    println!(
                        "      Large SIMD leaf nodes: {} ({:.1}%) [total values: {}]",
                        stats.large_simd_node_count,
                        (stats.large_simd_node_count as f64 / stats.total_entries as f64) * 100.0,
                        stats.large_simd_entry_sum
                    );
                    println!(
                        "        → Avg values per large SIMD node: {:.1}",
                        stats.large_simd_entry_sum as f64 / stats.large_simd_node_count as f64
                    );
                }
            }
            println!();
        }
    }
}

impl<K, V, S, P> GenericHashMap<K, V, S, P>
where
    K: Hash + Eq,
    S: BuildHasher + Clone,
    P: SharedPointerKind,
{
    fn test_eq<S2: BuildHasher + Clone, P2: SharedPointerKind>(
        &self,
        other: &GenericHashMap<K, V, S2, P2>,
    ) -> bool
    where
        V: PartialEq,
    {
        if self.len() != other.len() {
            return false;
        }
        let mut seen = collections::HashSet::new();
        for (key, value) in self.iter() {
            if Some(value) != other.get(key) {
                return false;
            }
            seen.insert(key);
        }
        for key in other.keys() {
            if !seen.contains(&key) {
                return false;
            }
        }
        true
    }

    /// Get the value for a key from a hash map.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::hashmap::HashMap;
    /// let map = hashmap!{123 => "lol"};
    /// assert_eq!(
    ///   map.get(&123),
    ///   Some(&"lol")
    /// );
    /// ```
    #[must_use]
    pub fn get<BK>(&self, key: &BK) -> Option<&V>
    where
        BK: Hash + Eq + ?Sized,
        K: Borrow<BK>,
    {
        if let Some(root) = &self.root {
            root.get(hash_key(&self.hasher, key), 0, key)
                .map(|(_, v)| v)
        } else {
            None
        }
    }

    /// Get the key/value pair for a key from a hash map.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::hashmap::HashMap;
    /// let map = hashmap!{123 => "lol"};
    /// assert_eq!(
    ///   map.get_key_value(&123),
    ///   Some((&123, &"lol"))
    /// );
    /// ```
    #[must_use]
    pub fn get_key_value<BK>(&self, key: &BK) -> Option<(&K, &V)>
    where
        BK: Hash + Eq + ?Sized,
        K: Borrow<BK>,
    {
        if let Some(root) = &self.root {
            root.get(hash_key(&self.hasher, key), 0, key)
                .map(|(k, v)| (k, v))
        } else {
            None
        }
    }

    /// Test for the presence of a key in a hash map.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::hashmap::HashMap;
    /// let map = hashmap!{123 => "lol"};
    /// assert!(
    ///   map.contains_key(&123)
    /// );
    /// assert!(
    ///   !map.contains_key(&321)
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn contains_key<BK>(&self, k: &BK) -> bool
    where
        BK: Hash + Eq + ?Sized,
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
    pub fn is_submap_by<B, RM, F, P2: SharedPointerKind>(&self, other: RM, mut cmp: F) -> bool
    where
        F: FnMut(&V, &B) -> bool,
        RM: Borrow<GenericHashMap<K, B, S, P2>>,
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
    pub fn is_proper_submap_by<B, RM, F, P2: SharedPointerKind>(&self, other: RM, cmp: F) -> bool
    where
        F: FnMut(&V, &B) -> bool,
        RM: Borrow<GenericHashMap<K, B, S, P2>>,
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
    /// # use imbl::hashmap::HashMap;
    /// let map1 = hashmap!{1 => 1, 2 => 2};
    /// let map2 = hashmap!{1 => 1, 2 => 2, 3 => 3};
    /// assert!(map1.is_submap(map2));
    /// ```
    #[inline]
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
    /// # use imbl::hashmap::HashMap;
    /// let map1 = hashmap!{1 => 1, 2 => 2};
    /// let map2 = hashmap!{1 => 1, 2 => 2, 3 => 3};
    /// assert!(map1.is_proper_submap(map2));
    ///
    /// let map3 = hashmap!{1 => 1, 2 => 2};
    /// let map4 = hashmap!{1 => 1, 2 => 2};
    /// assert!(!map3.is_proper_submap(map4));
    /// ```
    #[inline]
    #[must_use]
    pub fn is_proper_submap<RM>(&self, other: RM) -> bool
    where
        V: PartialEq,
        RM: Borrow<Self>,
    {
        self.is_proper_submap_by(other.borrow(), PartialEq::eq)
    }
}

impl<K, V, S, P> GenericHashMap<K, V, S, P>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Clone,
    P: SharedPointerKind,
{
    /// Get a mutable iterator over the values of a hash map.
    ///
    /// Please note that the order is consistent between maps using
    /// the same hasher, but no other ordering guarantee is offered.
    /// Items will not come out in insertion order or sort order.
    /// They will, however, come out in the same order every time for
    /// the same map.
    #[inline]
    #[must_use]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V, P> {
        let root = self.root.as_mut().map(|r| SharedPointer::make_mut(r));
        IterMut {
            it: NodeIterMut::new(root, self.size),
        }
    }

    /// Get a mutable reference to the value for a key from a hash
    /// map.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::hashmap::HashMap;
    /// let mut map = hashmap!{123 => "lol"};
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
        BK: Hash + Eq + ?Sized,
        K: Borrow<BK>,
    {
        self.get_key_value_mut(key).map(|(_, v)| v)
    }

    /// Get the key/value pair for a key from a hash map, returning a mutable reference to the value.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::hashmap::HashMap;
    /// let mut map = hashmap!{123 => "lol"};
    /// assert_eq!(
    ///   map.get_key_value_mut(&123),
    ///   Some((&123, &mut "lol"))
    /// );
    /// ```
    #[must_use]
    pub fn get_key_value_mut<BK>(&mut self, key: &BK) -> Option<(&K, &mut V)>
    where
        BK: Hash + Eq + ?Sized,
        K: Borrow<BK>,
    {
        let Some(root) = self.root.as_mut() else {
            return None;
        };
        match SharedPointer::make_mut(root).get_mut(hash_key(&self.hasher, key), 0, key) {
            None => None,
            Some((key, value)) => Some((key, value)),
        }
    }

    /// Insert a key/value mapping into a map.
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
    /// # use imbl::hashmap::HashMap;
    /// let mut map = hashmap!{};
    /// map.insert(123, "123");
    /// map.insert(456, "456");
    /// assert_eq!(
    ///   map,
    ///   hashmap!{123 => "123", 456 => "456"}
    /// );
    /// ```
    #[inline]
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        let hash = hash_key(&self.hasher, &k);
        let root = SharedPointer::make_mut(self.root.get_or_insert_with(SharedPointer::default));
        let result = root.insert(hash, 0, (k, v));
        if result.is_none() {
            self.size += 1;
        }
        result.map(|(_, v)| v)
    }

    /// Remove a key/value pair from a map, if it exists, and return
    /// the removed value.
    ///
    /// This is a copy-on-write operation, so that the parts of the
    /// set's structure which are shared with other sets will be
    /// safely copied before mutating.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::hashmap::HashMap;
    /// let mut map = hashmap!{123 => "123", 456 => "456"};
    /// assert_eq!(Some("123"), map.remove(&123));
    /// assert_eq!(Some("456"), map.remove(&456));
    /// assert_eq!(None, map.remove(&789));
    /// assert!(map.is_empty());
    /// ```
    pub fn remove<BK>(&mut self, k: &BK) -> Option<V>
    where
        BK: Hash + Eq + ?Sized,
        K: Borrow<BK>,
    {
        self.remove_with_key(k).map(|(_, v)| v)
    }

    /// Remove a key/value pair from a map, if it exists, and return
    /// the removed key and value.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::hashmap::HashMap;
    /// let mut map = hashmap!{123 => "123", 456 => "456"};
    /// assert_eq!(Some((123, "123")), map.remove_with_key(&123));
    /// assert_eq!(Some((456, "456")), map.remove_with_key(&456));
    /// assert_eq!(None, map.remove_with_key(&789));
    /// assert!(map.is_empty());
    /// ```
    pub fn remove_with_key<BK>(&mut self, k: &BK) -> Option<(K, V)>
    where
        BK: Hash + Eq + ?Sized,
        K: Borrow<BK>,
    {
        let Some(root) = &mut self.root else {
            return None;
        };
        let result = SharedPointer::make_mut(root).remove(hash_key(&self.hasher, k), 0, k);
        if result.is_some() {
            self.size -= 1;
        }
        result
    }

    /// Get the [`Entry`][Entry] for a key in the map for in-place manipulation.
    ///
    /// Time: O(log n)
    ///
    /// [Entry]: enum.Entry.html
    #[must_use]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, S, P> {
        let hash = hash_key(&self.hasher, &key);
        if self
            .root
            .as_ref()
            .and_then(|r| r.get(hash, 0, &key))
            .is_some()
        {
            Entry::Occupied(OccupiedEntry {
                map: self,
                hash,
                key,
            })
        } else {
            Entry::Vacant(VacantEntry {
                map: self,
                hash,
                key,
            })
        }
    }

    /// Construct a new hash map by inserting a key/value mapping into a map.
    ///
    /// If the map already has a mapping for the given key, the previous value
    /// is overwritten.
    ///
    /// Time: O(log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::hashmap::HashMap;
    /// let map = hashmap!{};
    /// assert_eq!(
    ///   map.update(123, "123"),
    ///   hashmap!{123 => "123"}
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn update(&self, k: K, v: V) -> Self {
        let mut out = self.clone();
        out.insert(k, v);
        out
    }

    /// Construct a new hash map by inserting a key/value mapping into
    /// a map.
    ///
    /// If the map already has a mapping for the given key, we call
    /// the provided function with the old value and the new value,
    /// and insert the result as the new value.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn update_with<F>(&self, k: K, v: V, f: F) -> Self
    where
        F: FnOnce(V, V) -> V,
    {
        match self.extract_with_key(&k) {
            None => self.update(k, v),
            Some((_, v2, m)) => m.update(k, f(v2, v)),
        }
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
    pub fn update_with_key<F>(&self, k: K, v: V, f: F) -> Self
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
    pub fn update_lookup_with_key<F>(&self, k: K, v: V, f: F) -> (Option<V>, Self)
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

    /// Construct a new map without the given key.
    ///
    /// Construct a map that's a copy of the current map, absent the
    /// mapping for `key` if it's present.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn without<BK>(&self, k: &BK) -> Self
    where
        BK: Hash + Eq + ?Sized,
        K: Borrow<BK>,
    {
        match self.extract_with_key(k) {
            None => self.clone(),
            Some((_, _, map)) => map,
        }
    }

    /// Filter out values from a map which don't satisfy a predicate.
    ///
    /// This is slightly more efficient than filtering using an
    /// iterator, in that it doesn't need to rehash the retained
    /// values, but it still needs to reconstruct the entire tree
    /// structure of the map.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::HashMap;
    /// let mut map = hashmap!{1 => 1, 2 => 2, 3 => 3};
    /// map.retain(|k, v| *k > 1);
    /// let expected = hashmap!{2 => 2, 3 => 3};
    /// assert_eq!(expected, map);
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &V) -> bool,
    {
        let Some(root) = &mut self.root else {
            return;
        };
        let old_root = root.clone();
        let root = SharedPointer::make_mut(root);
        for ((key, value), hash) in NodeIter::new(Some(&old_root), self.size) {
            if !f(key, value) && root.remove(hash, 0, key).is_some() {
                self.size -= 1;
            }
        }
    }

    /// Remove a key/value pair from a map, if it exists, and return
    /// the removed value as well as the updated map.
    ///
    /// Time: O(log n)
    #[must_use]
    pub fn extract<BK>(&self, k: &BK) -> Option<(V, Self)>
    where
        BK: Hash + Eq + ?Sized,
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
        BK: Hash + Eq + ?Sized,
        K: Borrow<BK>,
    {
        let mut out = self.clone();
        out.remove_with_key(k).map(|(k, v)| (k, v, out))
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
    /// # use imbl::hashmap::HashMap;
    /// let map1 = hashmap!{1 => 1, 3 => 3};
    /// let map2 = hashmap!{2 => 2, 3 => 4};
    /// let expected = hashmap!{1 => 1, 2 => 2, 3 => 3};
    /// assert_eq!(expected, map1.union(map2));
    /// ```
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        let (mut to_mutate, to_consume, use_to_consume) = if self.len() >= other.len() {
            (self, other, false)
        } else {
            (other, self, true)
        };
        for (k, v) in to_consume {
            match to_mutate.entry(k) {
                Entry::Occupied(mut e) if use_to_consume => {
                    e.insert(v);
                }
                Entry::Vacant(e) => {
                    e.insert(v);
                }
                _ => {}
            }
        }
        to_mutate
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
    /// # use imbl::hashmap::HashMap;
    /// let map1 = hashmap!{1 => 1, 3 => 4};
    /// let map2 = hashmap!{2 => 2, 3 => 5};
    /// let expected = hashmap!{1 => 1, 2 => 2, 3 => 9};
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
    /// # use imbl::hashmap::HashMap;
    /// let map1 = hashmap!{1 => 1, 3 => 3};
    /// let map2 = hashmap!{2 => 2};
    /// let expected = hashmap!{1 => 1, 2 => 2, 3 => 3};
    /// assert_eq!(expected, HashMap::unions(vec![map1, map2]));
    /// ```
    #[must_use]
    pub fn unions<I>(i: I) -> Self
    where
        S: Default,
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
        S: Default,
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
        S: Default,
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
    /// # use imbl::hashmap::HashMap;
    /// let map1 = hashmap!{1 => 1, 3 => 4};
    /// let map2 = hashmap!{2 => 2, 3 => 5};
    /// let expected = hashmap!{1 => 1, 2 => 2};
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
    /// # use imbl::hashmap::HashMap;
    /// let map1 = hashmap!{1 => 1, 3 => 4};
    /// let map2 = hashmap!{2 => 2, 3 => 5};
    /// let expected = hashmap!{1 => 1, 2 => 2};
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
    /// [`symmetric_difference_with`_key][symmetric_difference_with_key]
    /// method.
    ///
    /// Time: O(n log n)
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate imbl;
    /// # use imbl::hashmap::HashMap;
    /// let map1 = hashmap!{1 => 1, 3 => 4};
    /// let map2 = hashmap!{2 => 2, 3 => 5};
    /// let expected = hashmap!{1 => 1, 2 => 2, 3 => 9};
    /// assert_eq!(expected, map1.difference_with_key(
    ///     map2,
    ///     |key, left, right| Some(left + right)
    /// ));
    /// ```
    ///
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
    /// # use imbl::hashmap::HashMap;
    /// let map1 = hashmap!{1 => 1, 3 => 4};
    /// let map2 = hashmap!{2 => 2, 3 => 5};
    /// let expected = hashmap!{1 => 1, 2 => 2, 3 => 9};
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
        let mut out = self.new_from();
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
    /// # use imbl::hashmap::HashMap;
    /// let map1 = hashmap!{1 => 1, 3 => 4};
    /// let map2 = hashmap!{2 => 2, 3 => 5};
    /// let expected = hashmap!{1 => 1};
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
    /// # use imbl::hashmap::HashMap;
    /// let map1 = hashmap!{1 => 1, 2 => 2};
    /// let map2 = hashmap!{2 => 3, 3 => 4};
    /// let expected = hashmap!{2 => 2};
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
    pub fn intersection_with<B, C, F>(
        self,
        other: GenericHashMap<K, B, S, P>,
        mut f: F,
    ) -> GenericHashMap<K, C, S, P>
    where
        B: Clone,
        C: Clone,
        F: FnMut(V, B) -> C,
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
    /// # use imbl::hashmap::HashMap;
    /// let map1 = hashmap!{1 => 1, 2 => 2};
    /// let map2 = hashmap!{2 => 3, 3 => 4};
    /// let expected = hashmap!{2 => 5};
    /// assert_eq!(expected, map1.intersection_with_key(
    ///     map2,
    ///     |key, left, right| left + right
    /// ));
    /// ```
    #[must_use]
    pub fn intersection_with_key<B, C, F>(
        mut self,
        other: GenericHashMap<K, B, S, P>,
        mut f: F,
    ) -> GenericHashMap<K, C, S, P>
    where
        B: Clone,
        C: Clone,
        F: FnMut(&K, V, B) -> C,
    {
        let mut out = self.new_from();
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
}

// Entries

/// A handle for a key and its associated value.
///
/// ## Performance Note
///
/// When using an `Entry`, the key is only ever hashed once, when you
/// create the `Entry`. Operations on an `Entry` will never trigger a
/// rehash, where eg. a `contains_key(key)` followed by an
/// `insert(key, default_value)` (the equivalent of
/// `Entry::or_insert()`) would need to hash the key once for the
/// `contains_key` and again for the `insert`. The operations
/// generally perform similarly otherwise.
pub enum Entry<'a, K, V, S, P>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Clone,
    P: SharedPointerKind,
{
    /// An entry which exists in the map.
    Occupied(OccupiedEntry<'a, K, V, S, P>),
    /// An entry which doesn't exist in the map.
    Vacant(VacantEntry<'a, K, V, S, P>),
}

impl<'a, K, V, S, P> Entry<'a, K, V, S, P>
where
    K: 'a + Hash + Eq + Clone,
    V: 'a + Clone,
    S: 'a + BuildHasher + Clone,
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
pub struct OccupiedEntry<'a, K, V, S, P>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Clone,
    P: SharedPointerKind,
{
    map: &'a mut GenericHashMap<K, V, S, P>,
    hash: HashBits,
    key: K,
}

impl<'a, K, V, S, P> OccupiedEntry<'a, K, V, S, P>
where
    K: 'a + Hash + Eq + Clone,
    V: 'a + Clone,
    S: 'a + BuildHasher + Clone,
    P: SharedPointerKind,
{
    /// Get the key for this entry.
    #[must_use]
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Remove this entry from the map and return the removed mapping.
    pub fn remove_entry(self) -> (K, V) {
        // unwrap: occupied entries can only be created for non-empty maps
        let root = SharedPointer::make_mut(self.map.root.as_mut().unwrap());
        let result = root.remove(self.hash, 0, &self.key);
        self.map.size -= 1;
        result.unwrap()
    }

    /// Get the current value.
    #[must_use]
    pub fn get(&self) -> &V {
        // unwrap: occupied entries can only be created for non-empty maps
        &self
            .map
            .root
            .as_ref()
            .unwrap()
            .get(self.hash, 0, &self.key)
            .unwrap()
            .1
    }

    /// Get a mutable reference to the current value.
    #[must_use]
    pub fn get_mut(&mut self) -> &mut V {
        // unwrap: occupied entries can only be created for non-empty maps
        let root = SharedPointer::make_mut(self.map.root.as_mut().unwrap());
        &mut root.get_mut(self.hash, 0, &self.key).unwrap().1
    }

    /// Convert this entry into a mutable reference.
    #[must_use]
    pub fn into_mut(self) -> &'a mut V {
        // unwrap: occupied entries can only be created for non-empty maps
        let root = SharedPointer::make_mut(self.map.root.as_mut().unwrap());
        &mut root.get_mut(self.hash, 0, &self.key).unwrap().1
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
pub struct VacantEntry<'a, K, V, S, P>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Clone,
    P: SharedPointerKind,
{
    map: &'a mut GenericHashMap<K, V, S, P>,
    hash: HashBits,
    key: K,
}

impl<'a, K, V, S, P> VacantEntry<'a, K, V, S, P>
where
    K: 'a + Hash + Eq + Clone,
    V: 'a + Clone,
    S: 'a + BuildHasher + Clone,
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
        let root =
            SharedPointer::make_mut(self.map.root.get_or_insert_with(SharedPointer::default));
        if root
            .insert(self.hash, 0, (self.key.clone(), value))
            .is_none()
        {
            self.map.size += 1;
        }
        // TODO it's unfortunate that we need to look up the key again
        // here to get the mut ref.
        &mut root.get_mut(self.hash, 0, &self.key).unwrap().1
    }
}

// Core traits

impl<K, V, S, P> Clone for GenericHashMap<K, V, S, P>
where
    K: Clone,
    V: Clone,
    S: Clone,
    P: SharedPointerKind,
{
    /// Clone a map.
    ///
    /// Time: O(1)
    #[inline]
    fn clone(&self) -> Self {
        GenericHashMap {
            root: self.root.clone(),
            size: self.size,
            hasher: self.hasher.clone(),
        }
    }
}

impl<K, V, S1, S2, P1, P2> PartialEq<GenericHashMap<K, V, S2, P2>> for GenericHashMap<K, V, S1, P1>
where
    K: Hash + Eq,
    V: PartialEq,
    S1: BuildHasher + Clone,
    S2: BuildHasher + Clone,
    P1: SharedPointerKind,
    P2: SharedPointerKind,
{
    fn eq(&self, other: &GenericHashMap<K, V, S2, P2>) -> bool {
        self.test_eq(other)
    }
}

impl<K, V, S, P> Eq for GenericHashMap<K, V, S, P>
where
    K: Hash + Eq,
    V: Eq,
    S: BuildHasher + Clone,
    P: SharedPointerKind,
{
}

impl<K, V, S, P> Default for GenericHashMap<K, V, S, P>
where
    S: Default,
    P: SharedPointerKind,
{
    #[inline]
    fn default() -> Self {
        GenericHashMap {
            size: 0,
            root: None,
            hasher: Default::default(),
        }
    }
}

impl<K, V, S, P> Add for GenericHashMap<K, V, S, P>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Clone,
    P: SharedPointerKind,
{
    type Output = GenericHashMap<K, V, S, P>;

    fn add(self, other: Self) -> Self::Output {
        self.union(other)
    }
}

impl<K, V, S, P> Add for &GenericHashMap<K, V, S, P>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Clone,
    P: SharedPointerKind,
{
    type Output = GenericHashMap<K, V, S, P>;

    fn add(self, other: Self) -> Self::Output {
        self.clone().union(other.clone())
    }
}

impl<K, V, S, P> Sum for GenericHashMap<K, V, S, P>
where
    K: Hash + Eq + Clone,
    V: Clone,
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

impl<K, V, S, RK, RV, P> Extend<(RK, RV)> for GenericHashMap<K, V, S, P>
where
    K: Hash + Eq + Clone + From<RK>,
    V: Clone + From<RV>,
    S: BuildHasher + Clone,
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

impl<BK, K, V, S, P> Index<&BK> for GenericHashMap<K, V, S, P>
where
    BK: Hash + Eq + ?Sized,
    K: Hash + Eq + Borrow<BK>,
    S: BuildHasher + Clone,
    P: SharedPointerKind,
{
    type Output = V;

    fn index(&self, key: &BK) -> &Self::Output {
        match self.get(key) {
            None => panic!("HashMap::index: invalid key"),
            Some(value) => value,
        }
    }
}

impl<BK, K, V, S, P> IndexMut<&BK> for GenericHashMap<K, V, S, P>
where
    BK: Hash + Eq + ?Sized,
    K: Hash + Eq + Clone + Borrow<BK>,
    V: Clone,
    S: BuildHasher + Clone,
    P: SharedPointerKind,
{
    fn index_mut(&mut self, key: &BK) -> &mut Self::Output {
        match self.get_mut(key) {
            None => panic!("HashMap::index_mut: invalid key"),
            Some(value) => value,
        }
    }
}

impl<K, V, S, P> Debug for GenericHashMap<K, V, S, P>
where
    K: Debug,
    V: Debug,
    P: SharedPointerKind,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        let mut d = f.debug_map();
        for (k, v) in self {
            d.entry(k, v);
        }
        d.finish()
    }
}

// // Iterators

/// An iterator over the elements of a map.
pub struct Iter<'a, K, V, P: SharedPointerKind> {
    it: NodeIter<'a, (K, V), P>,
}

// We impl Clone instead of deriving it, because we want Clone even if K and V aren't.
impl<'a, K, V, P: SharedPointerKind> Clone for Iter<'a, K, V, P> {
    fn clone(&self) -> Self {
        Iter {
            it: self.it.clone(),
        }
    }
}

impl<'a, K, V, P: SharedPointerKind> Iterator for Iter<'a, K, V, P> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|((k, v), _)| (k, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, K, V, P: SharedPointerKind> ExactSizeIterator for Iter<'a, K, V, P> {}

impl<'a, K, V, P: SharedPointerKind> FusedIterator for Iter<'a, K, V, P> {}

/// A mutable iterator over the elements of a map.
pub struct IterMut<'a, K, V, P>
where
    K: Clone,
    V: Clone,
    P: SharedPointerKind,
{
    it: NodeIterMut<'a, (K, V), P>,
}

impl<'a, K, V, P> Iterator for IterMut<'a, K, V, P>
where
    K: Clone,
    V: Clone,
    P: SharedPointerKind,
{
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|((k, v), _)| (&*k, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, K, V, P> ExactSizeIterator for IterMut<'a, K, V, P>
where
    K: Clone,
    V: Clone,
    P: SharedPointerKind,
{
}

impl<'a, K, V, P> FusedIterator for IterMut<'a, K, V, P>
where
    K: Clone,
    V: Clone,
    P: SharedPointerKind,
{
}

/// A consuming iterator over the elements of a map.
pub struct ConsumingIter<A: HashValue, P: SharedPointerKind> {
    it: NodeDrain<A, P>,
}

impl<A, P: SharedPointerKind> Iterator for ConsumingIter<A, P>
where
    A: HashValue + Clone,
{
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(p, _)| p)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<A, P> ExactSizeIterator for ConsumingIter<A, P>
where
    A: HashValue + Clone,
    P: SharedPointerKind,
{
}

impl<A, P> FusedIterator for ConsumingIter<A, P>
where
    A: HashValue + Clone,
    P: SharedPointerKind,
{
}

/// An iterator over the keys of a map.
pub struct Keys<'a, K, V, P: SharedPointerKind> {
    it: NodeIter<'a, (K, V), P>,
}

impl<'a, K, V, P: SharedPointerKind> Iterator for Keys<'a, K, V, P> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|((k, _), _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, K, V, P: SharedPointerKind> ExactSizeIterator for Keys<'a, K, V, P> {}

impl<'a, K, V, P: SharedPointerKind> FusedIterator for Keys<'a, K, V, P> {}

/// An iterator over the values of a map.
pub struct Values<'a, K, V, P: SharedPointerKind> {
    it: NodeIter<'a, (K, V), P>,
}

impl<'a, K, V, P: SharedPointerKind> Iterator for Values<'a, K, V, P> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|((_, v), _)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, K, V, P: SharedPointerKind> ExactSizeIterator for Values<'a, K, V, P> {}

impl<'a, K, V, P: SharedPointerKind> FusedIterator for Values<'a, K, V, P> {}

impl<'a, K, V, S, P: SharedPointerKind> IntoIterator for &'a GenericHashMap<K, V, S, P> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V, P>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K, V, S, P> IntoIterator for GenericHashMap<K, V, S, P>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Clone,
    P: SharedPointerKind,
{
    type Item = (K, V);
    type IntoIter = ConsumingIter<(K, V), P>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        ConsumingIter {
            it: NodeDrain::new(self.root, self.size),
        }
    }
}

// Conversions

impl<K, V, S, P> FromIterator<(K, V)> for GenericHashMap<K, V, S, P>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    fn from_iter<T>(i: T) -> Self
    where
        T: IntoIterator<Item = (K, V)>,
    {
        let mut map = Self::default();
        for (k, v) in i {
            map.insert(k, v);
        }
        map
    }
}

impl<K, V, S, P: SharedPointerKind> AsRef<GenericHashMap<K, V, S, P>>
    for GenericHashMap<K, V, S, P>
{
    #[inline]
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<K, V, OK, OV, SA, SB, P1, P2> From<&GenericHashMap<&K, &V, SA, P1>>
    for GenericHashMap<OK, OV, SB, P2>
where
    K: Hash + Eq + ToOwned<Owned = OK> + ?Sized,
    V: ToOwned<Owned = OV> + ?Sized,
    OK: Hash + Eq + Clone + Borrow<K>,
    OV: Borrow<V> + Clone,
    SA: BuildHasher + Clone,
    SB: BuildHasher + Default + Clone,
    P1: SharedPointerKind,
    P2: SharedPointerKind,
{
    fn from(m: &GenericHashMap<&K, &V, SA, P1>) -> Self {
        m.iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }
}

impl<'a, K, V, S, P> From<&'a [(K, V)]> for GenericHashMap<K, V, S, P>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    fn from(m: &'a [(K, V)]) -> Self {
        m.iter().cloned().collect()
    }
}

impl<K, V, S, P> From<Vec<(K, V)>> for GenericHashMap<K, V, S, P>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    fn from(m: Vec<(K, V)>) -> Self {
        m.into_iter().collect()
    }
}

impl<'a, K, V, S, P> From<&'a Vec<(K, V)>> for GenericHashMap<K, V, S, P>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    fn from(m: &'a Vec<(K, V)>) -> Self {
        m.iter().cloned().collect()
    }
}

impl<K, V, S1, S2, P> From<collections::HashMap<K, V, S2>> for GenericHashMap<K, V, S1, P>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S1: BuildHasher + Default + Clone,
    S2: BuildHasher,
    P: SharedPointerKind,
{
    fn from(m: collections::HashMap<K, V, S2>) -> Self {
        m.into_iter().collect()
    }
}

impl<'a, K, V, S1, S2, P> From<&'a collections::HashMap<K, V, S2>> for GenericHashMap<K, V, S1, P>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S1: BuildHasher + Default + Clone,
    S2: BuildHasher,
    P: SharedPointerKind,
{
    fn from(m: &'a collections::HashMap<K, V, S2>) -> Self {
        m.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
}

impl<K, V, S, P> From<collections::BTreeMap<K, V>> for GenericHashMap<K, V, S, P>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    fn from(m: collections::BTreeMap<K, V>) -> Self {
        m.into_iter().collect()
    }
}

impl<'a, K, V, S, P> From<&'a collections::BTreeMap<K, V>> for GenericHashMap<K, V, S, P>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    fn from(m: &'a collections::BTreeMap<K, V>) -> Self {
        m.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
}

// impl<K: Ord + Hash + Eq, V, S> From<OrdMap<K, V>> for HashMap<K, V, S>
// where
//     S: BuildHasher + Default,
// {
//     fn from(m: OrdMap<K, V>) -> Self {
//         m.into_iter().collect()
//     }
// }

// impl<'a, K: Ord + Hash + Eq, V, S> From<&'a OrdMap<K, V>> for HashMap<K, V, S>
// where
//     S: BuildHasher + Default,
// {
//     fn from(m: &'a OrdMap<K, V>) -> Self {
//         m.into_iter().collect()
//     }
// }

// Proptest
#[cfg(any(test, feature = "proptest"))]
#[doc(hidden)]
pub mod proptest {
    #[deprecated(
        since = "14.3.0",
        note = "proptest strategies have moved to imbl::proptest"
    )]
    pub use crate::proptest::hash_map;
}

// Tests

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::LolHasher;
    #[rustfmt::skip]
    use ::proptest::{collection, num::{i16, usize}, proptest};
    use static_assertions::{assert_impl_all, assert_not_impl_any};
    use std::hash::BuildHasherDefault;

    assert_impl_all!(HashMap<i32, i32>: Send, Sync);
    assert_not_impl_any!(HashMap<i32, *const i32>: Send, Sync);
    assert_not_impl_any!(HashMap<*const i32, i32>: Send, Sync);
    assert_covariant!(HashMap<T, i32> in T);
    assert_covariant!(HashMap<i32, T> in T);

    #[test]
    fn safe_mutation() {
        let v1: HashMap<usize, usize> = GenericHashMap::from_iter((0..131_072).map(|i| (i, i)));
        let mut v2 = v1.clone();
        v2.insert(131_000, 23);
        assert_eq!(Some(&23), v2.get(&131_000));
        assert_eq!(Some(&131_000), v1.get(&131_000));
    }

    #[test]
    fn index_operator() {
        let mut map: HashMap<usize, usize> = hashmap![1 => 2, 3 => 4, 5 => 6];
        assert_eq!(4, map[&3]);
        map[&3] = 8;
        let target_map: HashMap<usize, usize> = hashmap![1 => 2, 3 => 8, 5 => 6];
        assert_eq!(target_map, map);
    }

    #[test]
    fn proper_formatting() {
        let map: HashMap<usize, usize> = hashmap![1 => 2];
        assert_eq!("{1: 2}", format!("{:?}", map));

        assert_eq!("{}", format!("{:?}", HashMap::<(), ()>::new()));
    }

    #[test]
    fn remove_failing() {
        let pairs = [(1469, 0), (-67, 0)];
        let mut m: collections::HashMap<i16, i16, _> =
            collections::HashMap::with_hasher(BuildHasherDefault::<LolHasher>::default());
        for (k, v) in &pairs {
            m.insert(*k, *v);
        }
        let mut map: GenericHashMap<i16, i16, _, DefaultSharedPtr> =
            GenericHashMap::with_hasher(BuildHasherDefault::<LolHasher>::default());
        for (k, v) in &m {
            map = map.update(*k, *v);
        }
        for k in m.keys() {
            let l = map.len();
            assert_eq!(m.get(k).cloned(), map.get(k).cloned());
            map = map.without(k);
            assert_eq!(None, map.get(k));
            assert_eq!(l - 1, map.len());
        }
    }

    #[test]
    fn match_string_keys_with_string_slices() {
        let tmp_map: HashMap<&str, &i32> = hashmap! { "foo" => &1, "bar" => &2, "baz" => &3 };
        let mut map: HashMap<String, i32> = From::from(&tmp_map);
        assert_eq!(Some(&1), map.get("foo"));
        map = map.without("foo");
        assert_eq!(Some(3), map.remove("baz"));
        map["bar"] = 8;
        assert_eq!(8, map["bar"]);
    }

    #[test]
    fn macro_allows_trailing_comma() {
        let map1: HashMap<&str, i32> = hashmap! {"x" => 1, "y" => 2};
        let map2: HashMap<&str, i32> = hashmap! {
            "x" => 1,
            "y" => 2,
        };
        assert_eq!(map1, map2);
    }

    #[test]
    fn remove_top_level_collisions() {
        let pairs = vec![9, 2569, 27145];
        let mut map: GenericHashMap<i16, i16, BuildHasherDefault<LolHasher>, DefaultSharedPtr> =
            Default::default();
        for k in pairs.clone() {
            map.insert(k, k);
        }
        assert_eq!(pairs.len(), map.len());
        let keys: Vec<_> = map.keys().cloned().collect();
        for k in keys {
            let l = map.len();
            assert_eq!(Some(&k), map.get(&k));
            map.remove(&k);
            assert_eq!(None, map.get(&k));
            assert_eq!(l - 1, map.len());
        }
    }

    #[test]
    fn entry_api() {
        let mut map: HashMap<&str, i32> = hashmap! {"bar" => 5};
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
    fn refpool_crash() {
        let _map = HashMap::<u128, usize>::new();
    }

    #[test]
    fn large_map() {
        let mut map = HashMap::<_, _>::new();
        let size = 32769;
        for i in 0..size {
            map.insert(i, i);
        }
        assert_eq!(size, map.len());
        for i in 0..size {
            assert_eq!(Some(&i), map.get(&i));
        }
    }

    struct PanicOnClone;

    impl Clone for PanicOnClone {
        fn clone(&self) -> Self {
            panic!("PanicOnClone::clone called")
        }
    }

    #[test]
    fn into_iter_no_clone() {
        let mut map = HashMap::new();
        for i in 0..10_000 {
            map.insert(i, PanicOnClone);
        }
        let _ = map.into_iter().collect::<Vec<_>>();
    }

    #[test]
    fn iter_mut_no_clone() {
        let mut map = HashMap::new();
        for i in 0..10_000 {
            map.insert(i, PanicOnClone);
        }
        let _ = map.iter_mut().collect::<Vec<_>>();
    }

    #[test]
    fn iter_no_clone() {
        let mut map = HashMap::new();
        for i in 0..10_000 {
            map.insert(i, PanicOnClone);
        }
        let _ = map.iter().collect::<Vec<_>>();
    }

    proptest! {
        #[test]
        fn update_and_length(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..1000)) {
            let mut map: GenericHashMap<i16, i16, BuildHasherDefault<LolHasher>, DefaultSharedPtr> = Default::default();
            for (index, (k, v)) in m.iter().enumerate() {
                map = map.update(*k, *v);
                assert_eq!(Some(v), map.get(k));
                assert_eq!(index + 1, map.len());
            }
        }

        #[test]
        fn from_iterator(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..1000)) {
            let map: HashMap<i16, i16> =
                FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            assert_eq!(m.len(), map.len());
        }

        #[test]
        fn iterate_over(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..1000)) {
            let map: HashMap<i16, i16> = FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            assert_eq!(m.len(), map.iter().count());
        }

        #[test]
        fn equality(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..1000)) {
            let map1: HashMap<i16, i16> = FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            let map2: HashMap<i16, i16> = FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            assert_eq!(map1, map2);
        }

        #[test]
        fn lookup(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..1000)) {
            let map: HashMap<i16, i16> = FromIterator::from_iter(m.iter().map(|(k, v)| (*k, *v)));
            for (k, v) in m {
                assert_eq!(Some(*v), map.get(k).cloned(), "{k} not found in map {map:?}");
            }
        }

        #[test]
        fn without(ref pairs in collection::vec((i16::ANY, i16::ANY), 0..100)) {
            let mut m: collections::HashMap<i16, i16, _> =
                collections::HashMap::with_hasher(BuildHasherDefault::<LolHasher>::default());
            for (k, v) in pairs {
                m.insert(*k, *v);
            }
            let mut map: GenericHashMap<i16, i16, _, DefaultSharedPtr> = GenericHashMap::with_hasher(BuildHasherDefault::<LolHasher>::default());
            for (k, v) in &m {
                map = map.update(*k, *v);
            }
            for k in m.keys() {
                let l = map.len();
                assert_eq!(m.get(k).cloned(), map.get(k).cloned());
                map = map.without(k);
                assert_eq!(None, map.get(k));
                assert_eq!(l - 1, map.len());
            }
        }

        #[test]
        fn insert(ref m in collection::hash_map(i16::ANY, i16::ANY, 0..1000)) {
            let mut mut_map: GenericHashMap<i16, i16, BuildHasherDefault<LolHasher>, DefaultSharedPtr> = Default::default();
            let mut map: GenericHashMap<i16, i16, BuildHasherDefault<LolHasher>, DefaultSharedPtr> = Default::default();
            for (count, (k, v)) in m.iter().enumerate() {
                map = map.update(*k, *v);
                mut_map.insert(*k, *v);
                assert_eq!(count + 1, map.len());
                assert_eq!(count + 1, mut_map.len());
            }
            for (k, v) in m {
                assert_eq!(Some(v), map.get(k));
                assert_eq!(Some(v), mut_map.get(k));
            }
            assert_eq!(map, mut_map);
        }

        #[test]
        fn remove(ref pairs in collection::vec((i16::ANY, i16::ANY), 0..100)) {
            let mut m: collections::HashMap<i16, i16, _> =
                collections::HashMap::with_hasher(BuildHasherDefault::<LolHasher>::default());
            for (k, v) in pairs {
                m.insert(*k, *v);
            }
            let mut map: GenericHashMap<i16, i16, _, DefaultSharedPtr> = GenericHashMap::with_hasher(BuildHasherDefault::<LolHasher>::default());
            for (k, v) in &m {
                map.insert(*k, *v);
            }
            for k in m.keys() {
                let l = map.len();
                assert_eq!(m.get(k).cloned(), map.get(k).cloned());
                map.remove(k);
                assert_eq!(None, map.get(k));
                assert_eq!(l - 1, map.len());
            }
        }

        #[test]
        fn delete_and_reinsert(
            ref input in collection::hash_map(i16::ANY, i16::ANY, 1..1000),
            index_rand in usize::ANY
        ) {
            let index = *input.keys().nth(index_rand % input.len()).unwrap();
            let map1: HashMap<_, _> = HashMap::from_iter(input.clone());
            let (val, map2) = map1.extract(&index).unwrap();
            let map3 = map2.update(index, val);
            for key in map2.keys() {
                assert!(*key != index);
            }
            assert_eq!(map1.len(), map2.len() + 1);
            assert_eq!(map1, map3);
        }

        #[test]
        fn proptest_works(ref m in proptest::hash_map(0..9999, ".*", 10..100)) {
            assert!(m.len() < 100);
            assert!(m.len() >= 10);
        }

        #[test]
        fn exact_size_iterator(ref m in proptest::hash_map(i16::ANY, i16::ANY, 0..100)) {
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
        fn union(ref m1 in collection::hash_map(i16::ANY, i16::ANY, 0..100),
                 ref m2 in collection::hash_map(i16::ANY, i16::ANY, 0..100)) {
            let map1: HashMap<i16, i16> = FromIterator::from_iter(m1.iter().map(|(k, v)| (*k, *v)));
            let map2: HashMap<i16, i16> = FromIterator::from_iter(m2.iter().map(|(k, v)| (*k, *v)));
            let union_map = map1.union(map2);

            for k in m1.keys() {
                assert!(union_map.contains_key(k));
            }

            for k in m2.keys() {
                assert!(union_map.contains_key(k));
            }

            for (k, v) in union_map.iter() {
                assert_eq!(v, m1.get(k).or_else(|| m2.get(k)).unwrap());
            }
        }
    }

    #[test]
    fn test_structure_summary() {
        // Test with different sizes of HashMaps
        let sizes = vec![10, 100, 1_000, 10_000, 100_000];

        for size in sizes {
            println!("\n=== Testing with {} entries ===", size);

            let mut map = HashMap::new();

            // Insert entries
            for i in 0..size {
                // dbg!(i);
                map.insert(i, i * 2);
            }

            // Print structure summary
            map.print_structure_summary();
        }
    }
}
