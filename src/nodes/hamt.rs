// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::borrow::Borrow;
use std::cell::UnsafeCell;
use std::hash::{BuildHasher, Hash};
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
const GROUP_WIDTH: usize = HASH_WIDTH / 2;

type SimdGroup = wide::u8x16;
type GroupBitmap = bitmaps::Bitmap<GROUP_WIDTH>;

const _: () = {
    // Limitations of the current implementation, can only handle up to 2 groups,
    // but can be lifted with further code changes
    assert!(HASH_SHIFT <= 5, "HASH_LEVEL_SIZE must be at most 5");
    assert!(HASH_SHIFT >= 3, "HASH_LEVEL_SIZE must be at least 3");
};

#[inline]
pub(crate) fn hash_key<K: Hash + ?Sized, S: BuildHasher>(bh: &S, key: &K) -> HashBits {
    bh.hash_one(key) as HashBits
}

#[inline]
fn group_find_empty(control: &SimdGroup) -> Option<usize> {
    let idx = group_find(control, 0).first_index();
    // if the GROUP_WIDTH != SimdGroup lanes, we need to handle finding
    // a zero in an index outside the valid range
    if GROUP_WIDTH != size_of::<SimdGroup>() {
        idx.filter(|&i| i < GROUP_WIDTH)
    } else {
        idx
    }
}

#[inline]
fn group_find(control: &SimdGroup, value: u8) -> GroupBitmap {
    let mask = control.cmp_eq(SimdGroup::splat(value)).move_mask();
    GroupBitmap::from_value(mask as _)
}

/// Special constructor to allow initializing Nodes w/o incurring multiple memory copies.
/// These copies really slow things down once Node crosses a certain size threshold and copies become calls to memcopy.
#[inline]
fn node_with<T, P: SharedPointerKind>(with: impl FnOnce(&mut T)) -> SharedPointer<T, P>
where
    T: Default,
{
    let result: SharedPointer<UnsafeCell<mem::MaybeUninit<T>>, P> =
        SharedPointer::new(UnsafeCell::new(mem::MaybeUninit::uninit()));
    #[allow(unsafe_code)]
    unsafe {
        (&mut *result.get()).write(T::default());
        let mut_ptr = &mut *UnsafeCell::raw_get(&*result);
        let mut_ptr = MaybeUninit::as_mut_ptr(mut_ptr);
        with(&mut *mut_ptr);
        let result = ManuallyDrop::new(result);
        mem::transmute_copy(&result)
    }
}

pub trait HashValue {
    type Key: Eq;

    fn extract_key(&self) -> &Self::Key;
    fn ptr_eq(&self, other: &Self) -> bool;
}

/// Generic SIMD node that stores leaf values only (no child nodes).
/// Uses SIMD control bytes for fast parallel lookup.
pub(crate) struct GenericSimdNode<A, const WIDTH: usize, const GROUPS: usize>
where
    BitsImpl<WIDTH>: Bits,
{
    /// Stores value-hash pairs directly (leaf-only)
    data: SparseChunk<(A, HashBits), WIDTH>,

    /// SIMD control bytes for fast parallel lookup.
    /// Each byte corresponds to the u8 suffix of the hash.
    /// 0 indicates an empty slot, 1-255 are valid hash prefixes.
    control: [SimdGroup; GROUPS],
}

/// HAMT node that stores Entry enum (can contain values or child nodes).
/// Uses classic HAMT bitmap-indexed structure without SIMD.
pub(crate) struct HamtNode<A, P: SharedPointerKind>
where
    BitsImpl<HASH_WIDTH>: Bits,
{
    /// Stores Entry enum which can contain values, collision nodes, or child nodes
    data: SparseChunk<Entry<A, P>, HASH_WIDTH>,
}

impl<A: Clone, const WIDTH: usize, const GROUPS: usize> Clone for GenericSimdNode<A, WIDTH, GROUPS>
where
    BitsImpl<WIDTH>: Bits,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            control: self.control,
        }
    }
}

impl<A: Clone, P: SharedPointerKind> Clone for HamtNode<A, P>
where
    BitsImpl<HASH_WIDTH>: Bits,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

pub(crate) type SmallSimdNode<A> = GenericSimdNode<A, SMALL_NODE_WIDTH, 1>;
pub(crate) type LargeSimdNode<A> = GenericSimdNode<A, HASH_WIDTH, 2>;

// Legacy type alias for compatibility
pub(crate) type Node<A, P> = HamtNode<A, P>;

impl<A, const WIDTH: usize, const GROUPS: usize> Default for GenericSimdNode<A, WIDTH, GROUPS>
where
    BitsImpl<WIDTH>: Bits,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<A, P: SharedPointerKind> Default for HamtNode<A, P>
where
    BitsImpl<HASH_WIDTH>: Bits,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<A, const WIDTH: usize, const GROUPS: usize> GenericSimdNode<A, WIDTH, GROUPS>
where
    BitsImpl<WIDTH>: Bits,
{
    #[inline(always)]
    pub(crate) fn new() -> Self {
        GenericSimdNode {
            data: SparseChunk::new(),
            control: [SimdGroup::default(); GROUPS],
        }
    }

    #[inline]
    fn with<P: SharedPointerKind>(with: impl FnOnce(&mut Self)) -> SharedPointer<Self, P> {
        node_with(with)
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn ctrl_hash_and_group(hash: HashBits) -> (u8, usize) {
        let ctrl_hash = Self::ctrl_hash(hash);
        if GROUPS == 1 {
            return (ctrl_hash, 0);
        }
        let group = (hash >> (HashBits::BITS.saturating_sub(9))) as usize % GROUPS;
        (ctrl_hash, group)
    }

    #[inline]
    fn ctrl_hash(hash: HashBits) -> u8 {
        ((hash >> (HashBits::BITS - 8)) as u8).max(1)
    }

    #[inline]
    fn pop_value<P: SharedPointerKind>(&mut self) -> Entry<A, P> {
        let (val, hash) = self.data.pop().unwrap();
        Entry::Value(val, hash)
    }
}

impl<A: HashValue, const WIDTH: usize, const GROUPS: usize> GenericSimdNode<A, WIDTH, GROUPS>
where
    BitsImpl<WIDTH>: Bits,
{
    #[inline]
    pub(crate) fn get<BK>(&self, hash: HashBits, key: &BK) -> Option<&A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let (search, group) = Self::ctrl_hash_and_group(hash);
        let mut bitmap = group_find(&self.control[group], search);

        while let Some(offset) = bitmap.first_index() {
            let index = group * GROUP_WIDTH + offset;
            let (ref value, value_hash) = self.data.get(index).unwrap();
            if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                return Some(value);
            }
            bitmap.set(offset, false);
        }
        None
    }

    pub(crate) fn get_mut<BK>(&mut self, hash: HashBits, key: &BK) -> Option<&mut A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let (search, group) = Self::ctrl_hash_and_group(hash);
        let mut bitmap = group_find(&self.control[group], search);
        let this = self as *mut Self;
        #[allow(dropping_references)]
        drop(self);

        while let Some(offset) = bitmap.first_index() {
            let index = group * GROUP_WIDTH + offset;
            #[allow(unsafe_code)]
            let this = unsafe { &mut *this };
            let (ref mut value, value_hash) = this.data.get_mut(index).unwrap();
            if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                return Some(value);
            }
            bitmap.set(offset, false);
        }
        None
    }

    pub(crate) fn remove<BK>(&mut self, hash: HashBits, key: &BK) -> Option<A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let (search, group) = Self::ctrl_hash_and_group(hash);
        let mut bitmap = group_find(&self.control[group], search);

        while let Some(offset) = bitmap.first_index() {
            let index = group * GROUP_WIDTH + offset;
            let (ref value, value_hash) = self.data.get(index).unwrap();
            if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                let mut ctrl_array = self.control[group].to_array();
                ctrl_array[offset] = 0;
                self.control[group] = SimdGroup::from(ctrl_array);
                return self.data.remove(index).map(|(v, _)| v);
            }
            bitmap.set(offset, false);
        }
        None
    }

    pub(crate) fn insert(&mut self, hash: HashBits, value: A) -> Result<Option<A>, A> {
        let (search, group) = Self::ctrl_hash_and_group(hash);
        // First check if we're updating an existing value in the group
        let mut bitmap = group_find(&self.control[group], search);
        while let Some(offset) = bitmap.first_index() {
            let index = group * GROUP_WIDTH + offset;
            let (current, current_hash) = self.data.get_mut(index).unwrap();
            if hash_may_eq::<A>(hash, *current_hash) && current.extract_key() == value.extract_key()
            {
                return Ok(Some(mem::replace(current, value)));
            }
            bitmap.set(offset, false);
        }

        // Try to insert into the designated group
        if let Some(offset) = group_find_empty(&self.control[group]) {
            let index = group * GROUP_WIDTH + offset;
            self.data.insert(index, (value, hash));
            let mut ctrl_array = self.control[group].to_array();
            ctrl_array[offset] = search;
            self.control[group] = SimdGroup::from(ctrl_array);
            return Ok(None);
        }

        // Group is full, need to upgrade
        Err(value)
    }
}

impl<A, P: SharedPointerKind> HamtNode<A, P>
where
    BitsImpl<HASH_WIDTH>: Bits,
{
    #[inline(always)]
    pub(crate) fn new() -> Self {
        HamtNode {
            data: SparseChunk::new(),
        }
    }

    #[inline]
    fn with(with: impl FnOnce(&mut Self)) -> SharedPointer<Self, P> {
        node_with(with)
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn mask(hash: HashBits, shift: usize) -> HashBits {
        let mask = (HASH_WIDTH - 1) as HashBits;
        (hash >> shift) & mask
    }

    fn pop(&mut self) -> Entry<A, P> {
        self.data.pop().unwrap()
    }
}

impl<A: HashValue> SmallSimdNode<A> {
    #[cold]
    fn upgrade_to_large<P: SharedPointerKind>(
        &mut self,
        hash: HashBits,
        shift: usize,
        value: A,
    ) -> Entry<A, P>
    where
        A: Clone,
    {
        // Move all small node entries into a LargeSimdNode and try to insert the new value
        // Existing entries are guaranteed to fit since SmallNode has 16 entries max
        // and LargeSimdNode has 2 groups of 16 (32 total)
        let mut remaining_value = None;
        let mut large_node = LargeSimdNode::with(|node| {
            let mut group_offsets = [0; 2];
            while let Some((val, entry_hash)) = self.data.pop() {
                let (search, group) = LargeSimdNode::<A>::ctrl_hash_and_group(entry_hash);
                let group_offset = group_offsets[group];
                group_offsets[group] += 1;
                let data_offset = group * GROUP_WIDTH + group_offset;
                let mut ctrl_array = node.control[group].to_array();
                ctrl_array[group_offset] = search;
                node.control[group] = SimdGroup::from(ctrl_array);
                node.data.insert(data_offset, (val, entry_hash));
            }
            if let Err(val) = node.insert(hash, value) {
                // Put it back if insert failed
                remaining_value = Some(val);
            }
        });

        // Check if insertion succeeded
        if let Some(value) = remaining_value {
            // LargeSimdNode group is full, upgrade to HamtNode
            let large_mut = SharedPointer::make_mut(&mut large_node);
            large_mut.upgrade_to_hamt(hash, shift, value)
        } else {
            // Successfully inserted into LargeSimdNode
            Entry::LargeSimdNode(large_node)
        }
    }
}

impl<A: HashValue> LargeSimdNode<A> {
    #[cold]
    fn upgrade_to_hamt<P: SharedPointerKind>(
        &mut self,
        hash: HashBits,
        shift: usize,
        value: A,
    ) -> Entry<A, P>
    where
        A: Clone,
    {
        let hamt_node = HamtNode::with(|node| {
            // Relocate all existing values to their correct HAMT positions
            while let Some((value, hash)) = self.data.pop() {
                node.insert(hash, shift, value);
            }
            // Insert the new value
            node.insert(hash, shift, value);
        });
        Entry::HamtNode(hamt_node)
    }
}

impl<A: HashValue, P: SharedPointerKind> HamtNode<A, P> {
    pub(crate) fn get<BK>(&self, hash: HashBits, shift: usize, key: &BK) -> Option<&A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let mut node = self;
        let mut shift = shift;

        loop {
            let index = Self::mask(hash, shift) as usize;
            let entry = node.data.get(index)?;

            // Check HamtNode and Value first and check the others
            // in a cold function that's also inlined.
            // This prevents the compiler from putting all node types
            // in a jump table, which makes things slower.
            // This is less relevant in other code paths that may include
            // atomics, memory allocation (e.g. insert, remove) etc..
            match entry {
                Entry::HamtNode(ref child) => {
                    node = child;
                    shift += HASH_SHIFT;
                    continue;
                }
                Entry::Value(ref value, value_hash) => {
                    return if hash_may_eq::<A>(hash, *value_hash)
                        && key == value.extract_key().borrow()
                    {
                        Some(value)
                    } else {
                        None
                    };
                }
                // Note: tried a bunch of things here, like (un)likely intrinsics,
                // but none of them worked as reliably as the cold function
                // that is also inlined.
                _ => return Self::get_terminal(entry, hash, key),
            }
        }
    }

    #[cold]
    #[inline(always)]
    fn get_terminal<'a, BK>(entry: &'a Entry<A, P>, hash: HashBits, key: &BK) -> Option<&'a A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        match entry {
            Entry::SmallSimdNode(ref small) => small.get(hash, key),
            Entry::LargeSimdNode(ref large) => large.get(hash, key),
            Entry::Collision(ref coll) => coll.get(key),
            _ => unreachable!(),
        }
    }

    pub(crate) fn get_mut<BK>(&mut self, hash: HashBits, shift: usize, key: &BK) -> Option<&mut A>
    where
        A: Clone,
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let index = Self::mask(hash, shift) as usize;
        match self.data.get_mut(index) {
            Some(Entry::HamtNode(ref mut child_ref)) => {
                SharedPointer::make_mut(child_ref).get_mut(hash, shift + HASH_SHIFT, key)
            }
            Some(Entry::SmallSimdNode(ref mut small_ref)) => {
                SharedPointer::make_mut(small_ref).get_mut(hash, key)
            }
            Some(Entry::LargeSimdNode(ref mut large_ref)) => {
                SharedPointer::make_mut(large_ref).get_mut(hash, key)
            }
            Some(Entry::Value(ref mut value, value_hash)) => {
                if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                    Some(value)
                } else {
                    None
                }
            }
            Some(Entry::Collision(ref mut coll_ref)) => {
                SharedPointer::make_mut(coll_ref).get_mut(key)
            }
            None => None,
        }
    }

    fn merge_values(value1: A, hash1: HashBits, value2: A, hash2: HashBits) -> Entry<A, P> {
        let small_node = SmallSimdNode::with(|node| {
            node.data.insert(0, (value1, hash1));
            node.data.insert(1, (value2, hash2));
            let mut ctrl_array = node.control[0].to_array();
            ctrl_array[0] = SmallSimdNode::<A>::ctrl_hash(hash1);
            ctrl_array[1] = SmallSimdNode::<A>::ctrl_hash(hash2);
            node.control[0] = SimdGroup::from(ctrl_array);
        });
        Entry::SmallSimdNode(small_node)
    }

    #[allow(unsafe_code)]
    pub(crate) fn insert(&mut self, hash: HashBits, shift: usize, value: A) -> Option<A>
    where
        A: Clone,
    {
        let index = Self::mask(hash, shift) as usize;
        let Some(entry) = self.data.get_mut(index) else {
            // Insert at empty HAMT position
            self.data.insert(index, Entry::Value(value, hash));
            return None;
        };

        match entry {
            Entry::HamtNode(child_ref) => {
                let child = SharedPointer::make_mut(child_ref);
                return child.insert(hash, shift + HASH_SHIFT, value);
            }
            Entry::SmallSimdNode(small_ref) => {
                let small = SharedPointer::make_mut(small_ref);
                match small.insert(hash, value) {
                    Ok(result) => return result,
                    Err(value) => {
                        // Small SIMD node is full, upgrade to LargeSimdNode
                        *entry = small.upgrade_to_large(hash, shift + HASH_SHIFT, value);
                        return None;
                    }
                }
            }
            Entry::LargeSimdNode(large_ref) => {
                let large = SharedPointer::make_mut(large_ref);
                match large.insert(hash, value) {
                    Ok(result) => return result,
                    Err(value) => {
                        // Large SIMD node is full, upgrade to HamtNode
                        *entry = large.upgrade_to_hamt(hash, shift + HASH_SHIFT, value);
                        return None;
                    }
                }
            }
            // Update value or create a subtree
            Entry::Value(current, current_hash) => {
                if hash_may_eq::<A>(hash, *current_hash)
                    && current.extract_key() == value.extract_key()
                {
                    return Some(mem::replace(current, value));
                }
            }
            Entry::Collision(collision) => {
                let coll = SharedPointer::make_mut(collision);
                return coll.insert(value);
            }
        }

        // If we get here, we're inserting a value over an existing value (collision).
        // We're going to be unsafe and pry it out of the reference, trusting
        // that we overwrite it with the merged node.
        let Entry::Value(old_value, old_hash) = (unsafe { ptr::read(entry) }) else {
            unreachable!()
        };
        let new_entry = if shift + HASH_SHIFT >= HASH_WIDTH {
            // We're at the lowest level, need to set up a collision node.
            Entry::from(CollisionNode::new(hash, old_value, value))
        } else {
            Self::merge_values(old_value, old_hash, value, hash)
        };
        unsafe { ptr::write(entry, new_entry) };
        None
    }

    pub(crate) fn remove<BK>(&mut self, hash: HashBits, shift: usize, key: &BK) -> Option<A>
    where
        A: Clone,
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let index = Self::mask(hash, shift) as usize;
        let removed;
        let new_node = match self.data.get_mut(index)? {
            Entry::HamtNode(child_ref) => {
                let child = SharedPointer::make_mut(child_ref);
                removed = child.remove(hash, shift + HASH_SHIFT, key);
                if child.len() == 1 && child.data.iter().next().is_some_and(|e| e.is_value()) {
                    Some(child.pop())
                } else {
                    None
                }
            }
            Entry::SmallSimdNode(small_ref) => {
                let small = SharedPointer::make_mut(small_ref);
                removed = small.remove(hash, key);
                (small.len() == 1).then(|| small.pop_value())
            }
            Entry::LargeSimdNode(large_ref) => {
                let large = SharedPointer::make_mut(large_ref);
                removed = large.remove(hash, key);
                (large.len() == 1).then(|| large.pop_value())
            }
            Entry::Value(value, value_hash) => {
                if hash_may_eq::<A>(hash, *value_hash) && key == value.extract_key().borrow() {
                    return self.data.remove(index).map(Entry::unwrap_value);
                } else {
                    return None;
                }
            }
            Entry::Collision(coll_ref) => {
                let coll = SharedPointer::make_mut(coll_ref);
                removed = coll.remove(key);
                (coll.len() == 1).then(|| coll.pop_value())
            }
        };
        if let Some(new_node) = new_node {
            self.data.insert(index, new_node);
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
    HamtNode(SharedPointer<HamtNode<A, P>, P>),
    SmallSimdNode(SharedPointer<SmallSimdNode<A>, P>),
    LargeSimdNode(SharedPointer<LargeSimdNode<A>, P>),
    Value(A, HashBits),
    Collision(SharedPointer<CollisionNode<A>, P>),
}

impl<A: Clone, P: SharedPointerKind> Clone for Entry<A, P> {
    fn clone(&self) -> Self {
        match self {
            Entry::HamtNode(node) => Entry::HamtNode(node.clone()),
            Entry::SmallSimdNode(node) => Entry::SmallSimdNode(node.clone()),
            Entry::LargeSimdNode(node) => Entry::LargeSimdNode(node.clone()),
            Entry::Value(value, hash) => Entry::Value(value.clone(), *hash),
            Entry::Collision(coll) => Entry::Collision(coll.clone()),
        }
    }
}

impl<A, P: SharedPointerKind> Entry<A, P> {
    fn is_value(&self) -> bool {
        matches!(self, Entry::Value(_, _))
    }

    #[inline(always)]
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
    #[cold]
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

    #[cold]
    fn get<BK>(&self, key: &BK) -> Option<&A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        self.data
            .iter()
            .find(|&entry| key == entry.extract_key().borrow())
    }

    #[cold]
    fn get_mut<BK>(&mut self, key: &BK) -> Option<&mut A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        self.data
            .iter_mut()
            .find(|entry| key == entry.extract_key().borrow())
    }

    #[cold]
    fn insert(&mut self, value: A) -> Option<A> {
        for item in &mut self.data {
            if value.extract_key() == item.extract_key() {
                return Some(mem::replace(item, value));
            }
        }
        self.data.push(value);
        None
    }

    #[cold]
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

    #[inline]
    fn pop_value<P: SharedPointerKind>(&mut self) -> Entry<A, P> {
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
    SmallSimdNode(ChunkIter<'a, (A, HashBits), SMALL_NODE_WIDTH>),
    LargeSimdNode(ChunkIter<'a, (A, HashBits), HASH_WIDTH>),
    HamtNode(ChunkIter<'a, Entry<A, P>, HASH_WIDTH>),
    CollisionNode(HashBits, SliceIter<'a, A>),
}

// We manually impl Clone for IterItem to allow cloning even when A isn't Clone
// This works because the iterators hold references, not owned values
impl<'a, A, P: SharedPointerKind> Clone for IterItem<'a, A, P> {
    fn clone(&self) -> Self {
        match self {
            IterItem::SmallSimdNode(iter) => IterItem::SmallSimdNode(iter.clone()),
            IterItem::LargeSimdNode(iter) => IterItem::LargeSimdNode(iter.clone()),
            IterItem::HamtNode(iter) => IterItem::HamtNode(iter.clone()),
            IterItem::CollisionNode(hash, iter) => IterItem::CollisionNode(*hash, iter.clone()),
        }
    }
}

// Ref iterator

pub(crate) struct Iter<'a, A, P: SharedPointerKind> {
    count: usize,
    stack: InlineStack<IterItem<'a, A, P>>,
}

// We impl Clone instead of deriving it, because we want Clone even if K and V aren't.
impl<'a, A, P: SharedPointerKind> Clone for Iter<'a, A, P> {
    fn clone(&self) -> Self {
        Self {
            count: self.count,
            stack: self.stack.clone(),
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
        };
        if let Some(node) = root {
            result.stack.push(IterItem::HamtNode(node.data.iter()));
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
        while let Some(current) = self.stack.last_mut() {
            match current {
                IterItem::SmallSimdNode(iter) => {
                    if let Some((value, hash)) = iter.next() {
                        self.count -= 1;
                        return Some((value, *hash));
                    }
                }
                IterItem::LargeSimdNode(iter) => {
                    if let Some((value, hash)) = iter.next() {
                        self.count -= 1;
                        return Some((value, *hash));
                    }
                }
                IterItem::HamtNode(iter) => {
                    if let Some(entry) = iter.next() {
                        let iter_item = match entry {
                            Entry::Value(value, hash) => {
                                self.count -= 1;
                                return Some((value, *hash));
                            }
                            Entry::HamtNode(child) => IterItem::HamtNode(child.data.iter()),
                            Entry::SmallSimdNode(small) => {
                                IterItem::SmallSimdNode(small.data.iter())
                            }
                            Entry::LargeSimdNode(large) => {
                                IterItem::LargeSimdNode(large.data.iter())
                            }
                            Entry::Collision(coll) => {
                                IterItem::CollisionNode(coll.hash, coll.data.iter())
                            }
                        };
                        self.stack.push(iter_item);
                        continue;
                    }
                }
                IterItem::CollisionNode(hash, iter) => {
                    if let Some(value) = iter.next() {
                        self.count -= 1;
                        return Some((value, *hash));
                    }
                }
            }
            self.stack.pop();
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

impl<'a, A, P: SharedPointerKind> ExactSizeIterator for Iter<'a, A, P> where A: 'a {}

impl<'a, A, P: SharedPointerKind> FusedIterator for Iter<'a, A, P> where A: 'a {}

// Mut ref iterator

enum IterMutItem<'a, A, P: SharedPointerKind> {
    SmallSimdNode(ChunkIterMut<'a, (A, HashBits), SMALL_NODE_WIDTH>),
    LargeSimdNode(ChunkIterMut<'a, (A, HashBits), HASH_WIDTH>),
    HamtNode(ChunkIterMut<'a, Entry<A, P>, HASH_WIDTH>),
    CollisionNode(HashBits, SliceIterMut<'a, A>),
}

pub(crate) struct IterMut<'a, A, P: SharedPointerKind> {
    count: usize,
    stack: InlineStack<IterMutItem<'a, A, P>>,
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
        };
        if let Some(node) = root {
            result
                .stack
                .push(IterMutItem::HamtNode(node.data.iter_mut()));
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
        while let Some(current) = self.stack.last_mut() {
            match current {
                IterMutItem::SmallSimdNode(iter) => {
                    if let Some((value, hash)) = iter.next() {
                        self.count -= 1;
                        return Some((value, *hash));
                    }
                }
                IterMutItem::LargeSimdNode(iter) => {
                    if let Some((value, hash)) = iter.next() {
                        self.count -= 1;
                        return Some((value, *hash));
                    }
                }
                IterMutItem::HamtNode(iter) => {
                    if let Some(entry) = iter.next() {
                        let iter_item = match entry {
                            Entry::Value(value, hash) => {
                                self.count -= 1;
                                return Some((value, *hash));
                            }
                            Entry::HamtNode(child_ref) => {
                                let child = SharedPointer::make_mut(child_ref);
                                IterMutItem::HamtNode(child.data.iter_mut())
                            }
                            Entry::SmallSimdNode(small_ref) => {
                                let small = SharedPointer::make_mut(small_ref);
                                IterMutItem::SmallSimdNode(small.data.iter_mut())
                            }
                            Entry::LargeSimdNode(large_ref) => {
                                let large = SharedPointer::make_mut(large_ref);
                                IterMutItem::LargeSimdNode(large.data.iter_mut())
                            }
                            Entry::Collision(coll_ref) => {
                                let coll = SharedPointer::make_mut(coll_ref);
                                IterMutItem::CollisionNode(coll.hash, coll.data.iter_mut())
                            }
                        };
                        self.stack.push(iter_item);
                        continue;
                    }
                }
                IterMutItem::CollisionNode(hash, iter) => {
                    if let Some(value) = iter.next() {
                        self.count -= 1;
                        return Some((value, *hash));
                    }
                }
            }
            self.stack.pop();
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

impl<'a, A, P: SharedPointerKind> ExactSizeIterator for IterMut<'a, A, P> where A: Clone + 'a {}

impl<'a, A, P: SharedPointerKind> FusedIterator for IterMut<'a, A, P> where A: Clone + 'a {}

// Consuming iterator

enum DrainItem<A, P: SharedPointerKind> {
    SmallSimdNode(SharedPointer<SmallSimdNode<A>, P>),
    LargeSimdNode(SharedPointer<LargeSimdNode<A>, P>),
    HamtNode(SharedPointer<HamtNode<A, P>, P>),
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
            result.stack.push(DrainItem::HamtNode(root));
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
                DrainItem::SmallSimdNode(small_ref) => {
                    if let Some((value, hash)) = SharedPointer::make_mut(small_ref).data.pop() {
                        self.count -= 1;
                        return Some((value, hash));
                    }
                }
                DrainItem::LargeSimdNode(large_ref) => {
                    if let Some((value, hash)) = SharedPointer::make_mut(large_ref).data.pop() {
                        self.count -= 1;
                        return Some((value, hash));
                    }
                }
                DrainItem::HamtNode(node_ref) => {
                    if let Some(entry) = SharedPointer::make_mut(node_ref).data.pop() {
                        let drain_item = match entry {
                            Entry::Value(value, hash) => {
                                self.count -= 1;
                                return Some((value, hash));
                            }
                            Entry::HamtNode(child) => DrainItem::HamtNode(child),
                            Entry::SmallSimdNode(small) => DrainItem::SmallSimdNode(small),
                            Entry::LargeSimdNode(large) => DrainItem::LargeSimdNode(large),
                            Entry::Collision(coll) => DrainItem::Collision(coll),
                        };
                        self.stack.push(drain_item);
                        continue;
                    }
                }
                DrainItem::Collision(coll_ref) => {
                    let coll = SharedPointer::make_mut(coll_ref);
                    if let Some(value) = coll.data.pop() {
                        self.count -= 1;
                        return Some((value, coll.hash));
                    }
                }
            }
            self.stack.pop();
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

impl<A, P: SharedPointerKind> ExactSizeIterator for Drain<A, P> where A: Clone {}

impl<A, P: SharedPointerKind> FusedIterator for Drain<A, P> where A: Clone {}

impl<A: fmt::Debug, P: SharedPointerKind> fmt::Debug for HamtNode<A, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "HamtNode[ ")?;
        for i in self.data.indices() {
            write!(f, "{}: ", i)?;
            match &self.data[i] {
                Entry::Value(v, h) => write!(f, "{:?} :: {}, ", v, h)?,
                Entry::Collision(c) => write!(f, "Coll{:?} :: {}", c.data, c.hash)?,
                Entry::HamtNode(n) => write!(f, "{:?}, ", n)?,
                Entry::SmallSimdNode(s) => write!(f, "{:?}, ", s)?,
                Entry::LargeSimdNode(l) => write!(f, "{:?}, ", l)?,
            }
        }
        write!(f, " ]")
    }
}

impl<A: fmt::Debug, const WIDTH: usize, const GROUPS: usize> fmt::Debug
    for GenericSimdNode<A, WIDTH, GROUPS>
where
    BitsImpl<WIDTH>: Bits,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "SimdNode<{}, {}>[ ", WIDTH, GROUPS)?;
        for i in self.data.indices() {
            write!(f, "{}: ", i)?;
            let (v, h) = &self.data[i];
            write!(f, "{:?} :: {}, ", v, h)?;
        }
        write!(f, " ]")
    }
}
