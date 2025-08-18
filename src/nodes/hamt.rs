// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::borrow::Borrow;
use std::cell::UnsafeCell;
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::FusedIterator;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::slice::{Iter as SliceIter, IterMut as SliceIterMut};
use std::{fmt, mem, ptr};

use archery::{SharedPointer, SharedPointerKind};
use bitmaps::{Bits, BitsImpl};
use imbl_sized_chunks::inline_array::InlineArray;
use imbl_sized_chunks::sparse_chunk::{Iter as ChunkIter, IterMut as ChunkIterMut, SparseChunk};

use crate::config::HASH_LEVEL_SIZE as HASH_SHIFT;
pub(crate) type HashBits = <BitsImpl<HASH_WIDTH> as Bits>::Store; // a uint of HASH_WIDTH bits

const HASH_WIDTH: usize = 2_usize.pow(HASH_SHIFT as u32);
const ITER_STACK_CAPACITY: usize = HASH_WIDTH.div_ceil(HASH_SHIFT) + 1;
const SMALL_NODE_WIDTH: usize = HASH_WIDTH / 2;

pub(crate) fn hash_key<K: Hash + ?Sized, S: BuildHasher>(bh: &S, key: &K) -> HashBits {
    let mut hasher = bh.build_hasher();
    key.hash(&mut hasher);
    hasher.finish() as HashBits
}

pub trait HashValue {
    type Key: Eq;

    fn extract_key(&self) -> &Self::Key;
    fn ptr_eq(&self, other: &Self) -> bool;
}

pub(crate) struct GenericNode<A, P: SharedPointerKind, const WIDTH: usize>
where
    BitsImpl<WIDTH>: Bits,
{
    /// Nodes with  `WIDTH` < `HASH_WIDTH` are used for small nodes.
    /// Small only contains `Value`s.
    ///
    /// Note that using SparseChunk<(A, HashBits), WIDTH> for small nodes wouldn't yield
    /// memory savings in most cases the padding space allows the enum
    /// Entry to be stored in the same space. So we use
    /// SparseChunk<Entry<A, P>, WIDTH> instead to increase code reuse.
    data: SparseChunk<Entry<A, P>, WIDTH>,

    /// SIMD control bytes for fast parallel lookup.
    /// Each byte corresponds to a hash prefix (hash >> (HashBits::BITS - 8)).
    /// 0 indicates an empty slot, 1-255 are valid hash prefixes.
    /// For SmallNode (WIDTH=16): single group covers all slots
    /// For Node (WIDTH=32): two groups, each item belongs to one group based on hash
    /// When in PureHamt mode, these are unused (all zeros)
    control: [wide::u8x16; 2],

    /// Current mode of operation
    /// SmallNode always uses SimdGroups, Node can switch to PureHamt when groups are full
    mode: NodeMode,
}

/// Mode of operation for a Node
#[derive(Clone, Copy, Debug, PartialEq)]
enum NodeMode {
    /// Using SIMD groups for fast lookup
    SimdGroups,
    /// Fallback to classic HAMT when groups are full
    PureHamt,
}

impl<A: Clone, P: SharedPointerKind, const WIDTH: usize> Clone for GenericNode<A, P, WIDTH>
where
    BitsImpl<WIDTH>: Bits,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            control: self.control,
            mode: self.mode,
        }
    }
}

pub(crate) type Node<A, P> = GenericNode<A, P, HASH_WIDTH>;
pub(crate) type SmallNode<A, P> = GenericNode<A, P, SMALL_NODE_WIDTH>;

impl<A, P: SharedPointerKind, const WIDTH: usize> Default for GenericNode<A, P, WIDTH>
where
    BitsImpl<WIDTH>: Bits,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<A, P: SharedPointerKind, const WIDTH: usize> GenericNode<A, P, WIDTH>
where
    BitsImpl<WIDTH>: Bits,
{
    #[inline(always)]
    pub(crate) fn new() -> Self {
        GenericNode {
            data: SparseChunk::new(),
            control: [wide::u8x16::default(); 2],
            mode: NodeMode::SimdGroups,
        }
    }

    /// Special constructor to allow initializing Nodes w/o incurring multiple memory copies.
    /// These copies really slow things down once Node crosses a certain size threshold and copies become calls to memcopy.
    #[inline]
    fn with(with: impl FnOnce(&mut Self)) -> SharedPointer<Self, P> {
        let result: SharedPointer<UnsafeCell<mem::MaybeUninit<Self>>, P> =
            SharedPointer::new(UnsafeCell::new(mem::MaybeUninit::uninit()));
        #[allow(unsafe_code)]
        unsafe {
            (&mut *result.get()).write(Self::new());
            let mut_ptr = &mut *UnsafeCell::raw_get(&*result);
            let mut_ptr = MaybeUninit::as_mut_ptr(mut_ptr);
            with(&mut *mut_ptr);
            let result = ManuallyDrop::new(result);
            mem::transmute_copy(&result)
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn mask(hash: HashBits, shift: usize) -> HashBits {
        let mask = (WIDTH - 1) as HashBits;
        hash >> shift & mask
    }

    fn pop(&mut self) -> Entry<A, P> {
        self.data.pop().unwrap()
    }
}

impl<A: HashValue, P: SharedPointerKind> SmallNode<A, P> {
    pub(crate) fn get<BK>(&self, hash: HashBits, _shift: usize, key: &BK) -> Option<&A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let search = (hash >> (HashBits::BITS - 8)).max(1) as u8;
        let mut mask = bitmaps::Bitmap::<16>::from_value(
            self.control[0]
                .cmp_eq(wide::u8x16::splat(search))
                .move_mask() as u16,
        );

        while let Some(index) = mask.first_index() {
            match self.data.get(index).unwrap() {
                Entry::Value(ref value, value_hash) => {
                    if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                        return Some(value);
                    } else {
                        mask.set(index, false);
                    }
                }
                _ => unreachable!("Control byte doesn't point to a Value"),
            }
        }
        None
    }

    pub(crate) fn get_mut<BK>(&mut self, hash: HashBits, _shift: usize, key: &BK) -> Option<&mut A>
    where
        A: Clone,
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let search = (hash >> (HashBits::BITS - 8)).max(1) as u8;
        let mut mask = bitmaps::Bitmap::<16>::from_value(
            self.control[0]
                .cmp_eq(wide::u8x16::splat(search))
                .move_mask() as u16,
        );
        let this = self as *mut Self;
        #[allow(dropping_references)]
        drop(self); // prevent self from being used or moved, so it's is safe to dereference `this` later
        while let Some(index) = mask.first_index() {
            // Restore a mutable reference to self to avoid hitting the borrow checker
            // limitation that prevents us from returning mutable references from the original
            // `self` inside a loop. This is safe because we only restore the mutable reference
            // once per iteration and the references goes out of scope at the end of the loop.
            #[allow(unsafe_code)]
            let this = unsafe { &mut *this };
            match this.data.get_mut(index).unwrap() {
                Entry::Value(ref mut value, value_hash) => {
                    if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                        return Some(value);
                    } else {
                        mask.set(index, false);
                    }
                }
                _ => unreachable!("Control byte doesn't point to a Value"),
            }
        }
        None
    }

    pub(crate) fn insert(&mut self, hash: HashBits, _shift: usize, value: A) -> Result<Option<A>, A>
    where
        A: Clone,
    {
        let search = (hash >> (HashBits::BITS - 8)).max(1) as u8;
        let mut bitmap = bitmaps::Bitmap::<16>::from_value(
            self.control[0]
                .cmp_eq(wide::u8x16::splat(search))
                .move_mask() as u16,
        );

        while let Some(index) = bitmap.first_index() {
            if let Some(Entry::Value(current, current_hash)) = self.data.get_mut(index) {
                if hash_may_eq::<A>(hash, *current_hash)
                    && current.extract_key() == value.extract_key()
                {
                    return Ok(Some(mem::replace(current, value)));
                }
                bitmap.set(index, false);
            } else {
                unreachable!("Control byte doesn't point to a Value")
            }
        }

        let bitmap = bitmaps::Bitmap::<16>::from_value(
            self.control[0].cmp_eq(wide::u8x16::default()).move_mask() as u16,
        );

        if let Some(index) = bitmap.first_index() {
            // If we found a free slot, we can insert the value
            self.control[0].as_array_mut()[index] = search;
            self.data.insert(index, Entry::Value(value, hash));
            return Ok(None);
        }

        // If no free slot was found, we need to handle the collision
        Err(value)
    }

    pub(crate) fn remove<BK>(&mut self, hash: HashBits, _shift: usize, key: &BK) -> Option<A>
    where
        A: Clone,
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let search = (hash >> (HashBits::BITS - 8)).max(1) as u8;
        let mut bitmap = bitmaps::Bitmap::<16>::from_value(
            self.control[0]
                .cmp_eq(wide::u8x16::splat(search))
                .move_mask() as u16,
        );

        // First find the entry to remove
        while let Some(index) = bitmap.first_index() {
            if let Some(Entry::Value(value, value_hash)) = self.data.get(index) {
                if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                    self.control[0].as_array_mut()[index] = 0;
                    return self.data.remove(index).map(Entry::unwrap_value);
                }
                bitmap.set(index, false);
            } else {
                unreachable!("Control byte doesn't point to a Value")
            }
        }
        None
    }
}

impl<A: HashValue, P: SharedPointerKind> Node<A, P> {
    pub(crate) fn get<BK>(&self, hash: HashBits, shift: usize, key: &BK) -> Option<&A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        if self.mode == NodeMode::SimdGroups {
            // Try SIMD search first for Value lookups
            let search = (hash >> (HashBits::BITS - 8)).max(1) as u8;
            let group = ((hash >> (HashBits::BITS - 9)) & 1) as usize;

            // Search in the designated group
            let mask = self.control[group]
                .cmp_eq(wide::u8x16::splat(search))
                .move_mask() as u16;
            let mut bitmap = bitmaps::Bitmap::<16>::from_value(mask);

            while let Some(offset) = bitmap.first_index() {
                let index = group * 16 + offset;
                if let Some(Entry::Value(value, value_hash)) = self.data.get(index) {
                    if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                        return Some(value);
                    }
                    bitmap.set(offset, false);
                } else {
                    unreachable!("Control byte doesn't point to a Value")
                }
            }
            return None; // No match found in SIMD groups
        }

        // Classic HAMT lookup
        let index = Self::mask(hash, shift) as usize;
        match self.data.get(index)? {
            Entry::Value(ref value, value_hash) => {
                if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                    Some(value)
                } else {
                    None
                }
            }
            Entry::Node(ref child) => child.get(hash, shift + HASH_SHIFT, key),
            Entry::SmallNode(ref small) => small.get(hash, shift + HASH_SHIFT, key),
            Entry::Collision(ref coll) => coll.get(key),
        }
    }

    pub(crate) fn get_mut<BK>(&mut self, hash: HashBits, shift: usize, key: &BK) -> Option<&mut A>
    where
        A: Clone,
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let mode = self.mode;
        let this = self as *mut Self;
        #[allow(dropping_references)]
        drop(self); // prevent self from being used or moved, so it's is safe to dereference `this` later

        if mode == NodeMode::SimdGroups {
            // Try SIMD search first
            let search = (hash >> (HashBits::BITS - 8)).max(1) as u8;
            let group = ((hash >> (HashBits::BITS - 9)) & 1) as usize;

            // Search in the designated group
            #[allow(unsafe_code)]
            let control = unsafe { &(*this).control };
            let mask = control[group]
                .cmp_eq(wide::u8x16::splat(search))
                .move_mask() as u16;
            let mut bitmap = bitmaps::Bitmap::<16>::from_value(mask);

            while let Some(offset) = bitmap.first_index() {
                let index = group * 16 + offset;
                #[allow(unsafe_code)]
                let this = unsafe { &mut *this };
                if let Some(Entry::Value(value, value_hash)) = this.data.get_mut(index) {
                    if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                        return Some(value);
                    }
                    bitmap.set(offset, false);
                } else {
                    unreachable!("Control byte doesn't point to a Value")
                }
            }
            return None; // No match found in SIMD groups
        }

        // Classic HAMT lookup
        let index = Self::mask(hash, shift) as usize;
        #[allow(unsafe_code)]
        let this = unsafe { &mut *this };
        match this.data.get_mut(index) {
            Some(Entry::Value(ref mut value, value_hash)) => {
                if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                    Some(value)
                } else {
                    None
                }
            }
            Some(Entry::Node(ref mut child_ref)) => {
                SharedPointer::make_mut(child_ref).get_mut(hash, shift + HASH_SHIFT, key)
            }
            Some(Entry::SmallNode(ref mut small_ref)) => {
                SharedPointer::make_mut(small_ref).get_mut(hash, shift + HASH_SHIFT, key)
            }
            Some(Entry::Collision(ref mut coll_ref)) => {
                SharedPointer::make_mut(coll_ref).get_mut(key)
            }
            None => None,
        }
    }

    #[inline]
    fn merge_values(
        value1: A,
        hash1: HashBits,
        value2: A,
        hash2: HashBits,
        _shift: usize,
    ) -> Entry<A, P> {
        let small_node = SmallNode::with(|node| {
            node.data.insert(0, Entry::Value(value1, hash1));
            node.data.insert(1, Entry::Value(value2, hash2));
            node.control[0].as_array_mut()[0] = ((hash1 >> (HashBits::BITS - 8)) as u8).max(1);
            node.control[0].as_array_mut()[1] = ((hash2 >> (HashBits::BITS - 8)) as u8).max(1);
        });
        Entry::SmallNode(small_node)
    }

    #[allow(unsafe_code)]
    pub(crate) fn insert(&mut self, hash: HashBits, shift: usize, value: A) -> Option<A>
    where
        A: Clone,
    {
        if self.mode == NodeMode::SimdGroups {
            // Try to place in the designated SIMD group
            let search = (hash >> (HashBits::BITS - 8)).max(1) as u8;
            let group = ((hash >> (HashBits::BITS - 9)) & 1) as usize;

            // First check if we're updating an existing value in the group
            let mask = self.control[group]
                .cmp_eq(wide::u8x16::splat(search))
                .move_mask() as u16;
            let mut bitmap = bitmaps::Bitmap::<16>::from_value(mask);

            while let Some(offset) = bitmap.first_index() {
                let index = group * 16 + offset;
                if let Some(Entry::Value(current, current_hash)) = self.data.get_mut(index) {
                    if hash_may_eq::<A>(hash, *current_hash)
                        && current.extract_key() == value.extract_key()
                    {
                        return Some(mem::replace(current, value));
                    }
                    bitmap.set(offset, false);
                } else {
                    unreachable!("Control byte doesn't point to a Value");
                }
            }

            // Try to find a free slot in the group
            let mask = self.control[group]
                .cmp_eq(wide::u8x16::default())
                .move_mask() as u16;
            let bitmap = bitmaps::Bitmap::<16>::from_value(mask);

            if let Some(offset) = bitmap.first_index() {
                // Found a free slot in the group
                let index = group * 16 + offset;
                self.data.insert(index, Entry::Value(value, hash));
                self.control[group].as_array_mut()[offset] = search;
                return None;
            }

            // Group is full, upgrade to PureHamt mode
            // We need to relocate all existing values to their correct HAMT positions
            self.mode = NodeMode::PureHamt;
            for entry in mem::take(&mut self.data) {
                let Entry::Value(value, hash) = entry else {
                    unreachable!("Expected Value entry in SimdGroups mode");
                };
                self.insert(hash, shift, value);
            }

            // All entries have been relocated, we can now insert the new value below
        }

        // Classic HAMT insertion (either we're in PureHamt mode or just upgraded to it)
        let index = Self::mask(hash, shift) as usize;
        if let Some(entry) = self.data.get_mut(index) {
            // Value is here
            match entry {
                // Update value or create a subtree
                Entry::Value(ref mut current, current_hash) => {
                    if hash_may_eq::<A>(hash, *current_hash)
                        && current.extract_key() == value.extract_key()
                    {
                        return Some(mem::replace(current, value));
                    }
                }
                Entry::Node(ref mut child_ref) => {
                    let child = SharedPointer::make_mut(child_ref);
                    return child.insert(hash, shift + HASH_SHIFT, value);
                }
                Entry::SmallNode(ref mut small_ref) => {
                    let small = SharedPointer::make_mut(small_ref);
                    match small.insert(hash, shift + HASH_SHIFT, value) {
                        Ok(result) => return result,
                        Err(value) => {
                            // It's a collision, need to upgrade to Node
                            let node = Node::with(|node| {
                                let mut group_offsets = [0; 2];
                                // Move small node entries into new node groups. These are guaranteed to fit
                                while let Some(entry) = small.data.pop() {
                                    let Entry::Value(_, hash) = entry else {
                                        unreachable!("Expected Value entry in SmallNode data");
                                    };
                                    let search = (hash >> (HashBits::BITS - 8)).max(1) as u8;
                                    let group = ((hash >> (HashBits::BITS - 9)) & 1) as usize;
                                    let group_offset = group_offsets[group];
                                    let data_offset = group * 16 + group_offset;
                                    node.control[group].as_array_mut()[group_offset] = search;
                                    node.data.insert(data_offset, entry);
                                    group_offsets[group] += 1;
                                }
                                // Insert the new value by recursing as it may overflow a group
                                node.insert(hash, shift + HASH_SHIFT, value);
                            });

                            *entry = Entry::Node(node);
                            return None;
                        }
                    }
                }
                // There's already a collision here.
                Entry::Collision(ref mut collision) => {
                    let coll = SharedPointer::make_mut(collision);
                    return coll.insert(value);
                }
            }

            // If we get here, we're inserting a value over an exiting value (collision).
            // We're going to be unsafe and pry it out of the reference, trusting
            // that we overwrite it with the merged node.
            let Entry::Value(old_value, old_hash) = (unsafe { ptr::read(entry) }) else {
                unreachable!()
            };
            let new_entry = if shift + HASH_SHIFT >= HASH_WIDTH {
                // We're at the lowest level, need to set up a collision node.
                Entry::from(CollisionNode::new(hash, old_value, value))
            } else {
                Node::merge_values(old_value, old_hash, value, hash, shift + HASH_SHIFT)
            };
            unsafe { ptr::write(entry, new_entry) };
            return None;
        }

        // Insert at the classic HAMT empty position
        self.data.insert(index, Entry::Value(value, hash));
        None
    }

    pub(crate) fn remove<BK>(&mut self, hash: HashBits, shift: usize, key: &BK) -> Option<A>
    where
        A: Clone,
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        if self.mode == NodeMode::SimdGroups {
            // Try SIMD search first
            let search = (hash >> (HashBits::BITS - 8)).max(1) as u8;
            let group = ((hash >> (HashBits::BITS - 9)) & 1) as usize;

            // Search in the designated group
            let mask = self.control[group]
                .cmp_eq(wide::u8x16::splat(search))
                .move_mask() as u16;
            let mut bitmap = bitmaps::Bitmap::<16>::from_value(mask);

            while let Some(offset) = bitmap.first_index() {
                let index = group * 16 + offset;
                if let Some(Entry::Value(value, value_hash)) = self.data.get(index) {
                    if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                        // Found it in the group
                        let removed = self.data.remove(index).map(Entry::unwrap_value);
                        self.control[group].as_array_mut()[offset] = 0;
                        return removed;
                    }
                    bitmap.set(offset, false);
                } else {
                    unreachable!("Node should only contain Values in SimdGroups mode")
                }
            }
            return None; // No match found in SIMD groups
        }

        // Classic HAMT lookup
        let index = Self::mask(hash, shift) as usize;
        match self.data.get(index) {
            None => return None,
            Some(Entry::Value(value, value_hash)) => {
                if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                    return self.data.remove(index).map(Entry::unwrap_value);
                } else {
                    return None;
                }
            }
            Some(Entry::Collision(_) | Entry::Node(_) | Entry::SmallNode(_)) => {
                // Continue with nested structure handling
            }
        }

        let new_node;
        let removed;
        match self.data.get_mut(index).unwrap() {
            Entry::Node(ref mut child_ref) => {
                let child = SharedPointer::make_mut(child_ref);
                match child.remove(hash, shift + HASH_SHIFT, key) {
                    None => return None,
                    Some(value) => {
                        if child.len() == 1
                            && child.data.iter().next().is_some_and(|e| e.is_value())
                        {
                            removed = Some(value);
                            new_node = Some(child.pop());
                        } else {
                            return Some(value);
                        }
                    }
                }
            }
            Entry::SmallNode(ref mut small_ref) => {
                let small = SharedPointer::make_mut(small_ref);
                match small.remove(hash, shift + HASH_SHIFT, key) {
                    None => return None,
                    Some(value) => {
                        if small.len() == 1 {
                            removed = Some(value);
                            new_node = Some(small.pop());
                        } else {
                            return Some(value);
                        }
                    }
                }
            }
            Entry::Value(..) => {
                new_node = None;
                removed = self.data.remove(index).map(Entry::unwrap_value);
            }
            Entry::Collision(ref mut coll_ref) => {
                let coll = SharedPointer::make_mut(coll_ref);
                removed = coll.remove(key);
                if coll.len() == 1 {
                    new_node = Some(coll.pop());
                } else {
                    return removed;
                }
            }
        }

        if let Some(node) = new_node {
            self.data.insert(index, node);
        } else {
            // if there's a single value child, restore SIMD mode
            // if self.data.len() == 1 && self.data.iter().next().is_some_and(|e| e.is_value()) {
            //     self.mode = NodeMode::SimdGroups;
            // }
        }

        removed
    }
}

#[derive(Clone)]
pub(crate) struct CollisionNode<A> {
    hash: HashBits,
    data: Vec<A>,
}

pub(crate) enum Entry<A, P: SharedPointerKind> {
    Value(A, HashBits),
    Collision(SharedPointer<CollisionNode<A>, P>),
    Node(SharedPointer<Node<A, P>, P>),
    SmallNode(SharedPointer<SmallNode<A, P>, P>),
}

impl<A: Clone, P: SharedPointerKind> Clone for Entry<A, P> {
    fn clone(&self) -> Self {
        match self {
            Entry::Value(value, hash) => Entry::Value(value.clone(), *hash),
            Entry::Collision(coll) => Entry::Collision(coll.clone()),
            Entry::Node(node) => Entry::Node(node.clone()),
            Entry::SmallNode(node) => Entry::SmallNode(node.clone()),
        }
    }
}

impl<A, P: SharedPointerKind> Entry<A, P> {
    fn is_value(&self) -> bool {
        matches!(self, Entry::Value(_, _))
    }

    fn unwrap_value(self) -> A {
        match self {
            Entry::Value(a, _) => a,
            _ => panic!("nodes::hamt::Entry::unwrap_value: unwrapped a non-value"),
        }
    }
}

impl<A, P: SharedPointerKind> From<CollisionNode<A>> for Entry<A, P> {
    fn from(node: CollisionNode<A>) -> Self {
        Entry::Collision(SharedPointer::new(node))
    }
}

/// Compare two hashes, returning true if the keys may be equal.
/// This function will always return true if it thinks keys may be cheap to compare.
#[inline]
fn hash_may_eq<A: HashValue>(hash: HashBits, other_hash: HashBits) -> bool {
    (!mem::needs_drop::<A::Key>() && mem::size_of::<A::Key>() <= 16) || hash == other_hash
}

impl<A: HashValue> CollisionNode<A> {
    fn new(hash: HashBits, value1: A, value2: A) -> Self {
        CollisionNode {
            hash,
            data: vec![value1, value2],
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    fn get<BK>(&self, key: &BK) -> Option<&A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        self.data
            .iter()
            .find(|&entry| key == entry.extract_key().borrow())
    }

    fn get_mut<BK>(&mut self, key: &BK) -> Option<&mut A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        self.data
            .iter_mut()
            .find(|entry| key == entry.extract_key().borrow())
    }

    fn insert(&mut self, value: A) -> Option<A> {
        for item in &mut self.data {
            if value.extract_key() == item.extract_key() {
                return Some(mem::replace(item, value));
            }
        }
        self.data.push(value);
        None
    }

    fn remove<BK>(&mut self, key: &BK) -> Option<A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        for (index, item) in self.data.iter().enumerate() {
            if key == item.extract_key().borrow() {
                return Some(self.data.swap_remove(index));
            }
        }
        None
    }

    fn pop<P: SharedPointerKind>(&mut self) -> Entry<A, P> {
        Entry::Value(self.data.pop().unwrap(), self.hash)
    }
}

#[cfg(test)]
impl<A, P: SharedPointerKind> Node<A, P> {
    /// Analyze the node structure for debugging/statistics
    pub(crate) fn analyze_structure<F>(&self, mut visitor: F)
    where
        F: FnMut(&Entry<A, P>),
    {
        for i in self.data.indices() {
            visitor(&self.data[i]);
        }
    }
}

/// An allocation-free stack for iterators.
type InlineStack<T> = InlineArray<T, (usize, [T; ITER_STACK_CAPACITY])>;

enum IterItem<'a, A, P: SharedPointerKind> {
    Node(ChunkIter<'a, Entry<A, P>, HASH_WIDTH>),
    SmallNode(ChunkIter<'a, Entry<A, P>, SMALL_NODE_WIDTH>),
}

// We manually impl Clone for IterItem to allow cloning even when A isn't Clone
// This works because the iterators hold references, not owned values
impl<'a, A, P: SharedPointerKind> Clone for IterItem<'a, A, P> {
    fn clone(&self) -> Self {
        match self {
            IterItem::Node(iter) => IterItem::Node(iter.clone()),
            IterItem::SmallNode(iter) => IterItem::SmallNode(iter.clone()),
        }
    }
}

// Ref iterator

pub(crate) struct Iter<'a, A, P: SharedPointerKind> {
    count: usize,
    stack: InlineStack<IterItem<'a, A, P>>,
    collision: Option<(HashBits, SliceIter<'a, A>)>,
}

// We impl Clone instead of deriving it, because we want Clone even if K and V aren't.
impl<'a, A, P: SharedPointerKind> Clone for Iter<'a, A, P> {
    fn clone(&self) -> Self {
        Self {
            count: self.count,
            stack: self.stack.clone(),
            collision: self.collision.clone(),
        }
    }
}

impl<'a, A, P> Iter<'a, A, P>
where
    A: 'a,
    P: SharedPointerKind,
{
    pub(crate) fn new(root: Option<&'a Node<A, P>>, size: usize) -> Self {
        let mut result = Iter {
            count: size,
            stack: InlineStack::new(),
            collision: None,
        };
        if let Some(node) = root {
            result.stack.push(IterItem::Node(node.data.iter()));
        }
        result
    }
}

impl<'a, A, P> Iterator for Iter<'a, A, P>
where
    A: 'a,
    P: SharedPointerKind,
{
    type Item = (&'a A, HashBits);

    fn next(&mut self) -> Option<Self::Item> {
        'outer: loop {
            if let Some((hash, ref mut coll)) = self.collision {
                match coll.next() {
                    None => self.collision = None,
                    Some(value) => {
                        self.count -= 1;
                        return Some((value, hash));
                    }
                };
            }

            while let Some(current) = self.stack.last_mut() {
                let next_entry = match current {
                    IterItem::Node(iter) => iter.next(),
                    IterItem::SmallNode(iter) => iter.next(),
                };
                match next_entry {
                    Some(Entry::Value(value, hash)) => {
                        self.count -= 1;
                        return Some((value, *hash));
                    }
                    Some(Entry::Node(child)) => {
                        self.stack.push(IterItem::Node(child.data.iter()));
                    }
                    Some(Entry::SmallNode(small)) => {
                        self.stack.push(IterItem::SmallNode(small.data.iter()));
                    }
                    Some(Entry::Collision(coll)) => {
                        self.collision = Some((coll.hash, coll.data.iter()));
                        continue 'outer;
                    }
                    None => {
                        self.stack.pop();
                    }
                }
            }
            return None;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

impl<'a, A, P: SharedPointerKind> ExactSizeIterator for Iter<'a, A, P> where A: 'a {}

impl<'a, A, P: SharedPointerKind> FusedIterator for Iter<'a, A, P> where A: 'a {}

// Mut ref iterator

enum IterMutItem<'a, A, P: SharedPointerKind> {
    Node(ChunkIterMut<'a, Entry<A, P>, HASH_WIDTH>),
    SmallNode(ChunkIterMut<'a, Entry<A, P>, SMALL_NODE_WIDTH>),
}

pub(crate) struct IterMut<'a, A, P: SharedPointerKind> {
    count: usize,
    stack: InlineStack<IterMutItem<'a, A, P>>,
    collision: Option<(HashBits, SliceIterMut<'a, A>)>,
}

impl<'a, A, P> IterMut<'a, A, P>
where
    A: 'a,
    P: SharedPointerKind,
{
    pub(crate) fn new(root: Option<&'a mut Node<A, P>>, size: usize) -> Self {
        let mut result = IterMut {
            count: size,
            stack: InlineStack::new(),
            collision: None,
        };
        if let Some(node) = root {
            result.stack.push(IterMutItem::Node(node.data.iter_mut()));
        }
        result
    }
}

impl<'a, A, P> Iterator for IterMut<'a, A, P>
where
    A: Clone + 'a,
    P: SharedPointerKind,
{
    type Item = (&'a mut A, HashBits);

    fn next(&mut self) -> Option<Self::Item> {
        'outer: loop {
            if let Some((hash, ref mut coll)) = self.collision {
                match coll.next() {
                    None => self.collision = None,
                    Some(value) => {
                        self.count -= 1;
                        return Some((value, hash));
                    }
                };
            }

            while let Some(current) = self.stack.last_mut() {
                let next_entry = match current {
                    IterMutItem::Node(iter) => iter.next(),
                    IterMutItem::SmallNode(iter) => iter.next(),
                };

                match next_entry {
                    Some(Entry::Value(value, hash)) => {
                        self.count -= 1;
                        return Some((value, *hash));
                    }
                    Some(Entry::Node(child_ref)) => {
                        let child = SharedPointer::make_mut(child_ref);
                        self.stack.push(IterMutItem::Node(child.data.iter_mut()));
                    }
                    Some(Entry::SmallNode(small_ref)) => {
                        let small = SharedPointer::make_mut(small_ref);
                        self.stack
                            .push(IterMutItem::SmallNode(small.data.iter_mut()));
                    }
                    Some(Entry::Collision(coll_ref)) => {
                        let coll = SharedPointer::make_mut(coll_ref);
                        self.collision = Some((coll.hash, coll.data.iter_mut()));
                        continue 'outer;
                    }
                    None => {
                        self.stack.pop();
                    }
                }
            }
            return None;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

impl<'a, A, P: SharedPointerKind> ExactSizeIterator for IterMut<'a, A, P> where A: Clone + 'a {}

impl<'a, A, P: SharedPointerKind> FusedIterator for IterMut<'a, A, P> where A: Clone + 'a {}

// Consuming iterator

enum DrainItem<A, P: SharedPointerKind> {
    Node(SharedPointer<Node<A, P>, P>),
    SmallNode(SharedPointer<SmallNode<A, P>, P>),
    Collision(SharedPointer<CollisionNode<A>, P>),
}

pub(crate) struct Drain<A, P: SharedPointerKind> {
    count: usize,
    stack: InlineStack<DrainItem<A, P>>,
}

impl<A, P: SharedPointerKind> Drain<A, P> {
    pub(crate) fn new(root: Option<SharedPointer<Node<A, P>, P>>, size: usize) -> Self {
        let mut result = Drain {
            count: size,
            stack: InlineStack::new(),
        };
        if let Some(root) = root {
            result.stack.push(DrainItem::Node(root));
        }
        result
    }
}

impl<A, P: SharedPointerKind> Iterator for Drain<A, P>
where
    A: Clone,
{
    type Item = (A, HashBits);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(current) = self.stack.last_mut() {
            match current {
                DrainItem::Node(node_ref) => match SharedPointer::make_mut(node_ref).data.pop() {
                    Some(Entry::Value(value, hash)) => {
                        self.count -= 1;
                        return Some((value, hash));
                    }
                    Some(Entry::Node(child)) => {
                        self.stack.push(DrainItem::Node(child));
                    }
                    Some(Entry::SmallNode(small)) => {
                        self.stack.push(DrainItem::SmallNode(small));
                    }
                    Some(Entry::Collision(coll)) => {
                        self.stack.push(DrainItem::Collision(coll));
                    }
                    None => {
                        self.stack.pop();
                    }
                },
                DrainItem::SmallNode(small_ref) => {
                    let small = SharedPointer::make_mut(small_ref);
                    if let Some(Entry::Value(value, hash)) = small.data.pop() {
                        self.count -= 1;
                        return Some((value, hash));
                    }
                    self.stack.pop();
                }
                DrainItem::Collision(coll_ref) => {
                    let coll = SharedPointer::make_mut(coll_ref);
                    if let Some(value) = coll.data.pop() {
                        self.count -= 1;
                        return Some((value, coll.hash));
                    }
                    self.stack.pop();
                }
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

impl<A, P: SharedPointerKind> ExactSizeIterator for Drain<A, P> where A: Clone {}

impl<A, P: SharedPointerKind> FusedIterator for Drain<A, P> where A: Clone {}

impl<A: fmt::Debug, P: SharedPointerKind> fmt::Debug for Node<A, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "Node[ ")?;
        for i in self.data.indices() {
            write!(f, "{}: ", i)?;
            match &self.data[i] {
                Entry::Value(v, h) => write!(f, "{:?} :: {}, ", v, h)?,
                Entry::Collision(c) => write!(f, "Coll{:?} :: {}", c.data, c.hash)?,
                Entry::Node(n) => write!(f, "{:?}, ", n)?,
                Entry::SmallNode(s) => write!(f, "{:?}, ", s)?,
            }
        }
        write!(f, " ]")
    }
}

impl<A: fmt::Debug, P: SharedPointerKind> fmt::Debug for SmallNode<A, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "SmallNode[ ")?;
        for i in self.data.indices() {
            write!(f, "{}: ", i)?;
            match &self.data[i] {
                Entry::Value(v, h) => write!(f, "{:?} :: {}, ", v, h)?,
                _ => unreachable!("Control byte doesn't point to a Value"),
            }
        }
        write!(f, " ]")
    }
}
