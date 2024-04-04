// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::borrow::Borrow;
use std::cell::UnsafeCell;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::FusedIterator;
use std::slice::{Iter as SliceIter, IterMut as SliceIterMut};
use std::{mem, ptr};

use bitmaps::{Bits, BitsImpl};
use imbl_sized_chunks::sparse_chunk::{Iter as ChunkIter, IterMut as ChunkIterMut, SparseChunk};

use crate::util::{clone_ref, Pool, PoolClone, PoolDefault, PoolRef, Ref};

pub(crate) use crate::config::HASH_LEVEL_SIZE as HASH_SHIFT;
pub(crate) const HASH_WIDTH: usize = 2_usize.pow(HASH_SHIFT as u32);
pub(crate) type HashBits = <BitsImpl<HASH_WIDTH> as Bits>::Store; // a uint of HASH_WIDTH bits
pub(crate) const HASH_MASK: HashBits = (HASH_WIDTH - 1) as HashBits;

pub(crate) fn hash_key<K: Hash + ?Sized, S: BuildHasher>(bh: &S, key: &K) -> HashBits {
    let mut hasher = bh.build_hasher();
    key.hash(&mut hasher);
    hasher.finish() as HashBits
}

#[inline]
fn mask(hash: HashBits, shift: usize) -> HashBits {
    hash >> shift & HASH_MASK
}

pub trait HashValue {
    type Key: Eq;

    fn extract_key(&self) -> &Self::Key;
    fn ptr_eq(&self, other: &Self) -> bool;
}

#[derive(Clone)]
pub(crate) struct Node<A> {
    data: SparseChunk<Entry<A>, HASH_WIDTH>,
}

impl<A> PoolDefault for Node<A> {
    #[cfg(feature = "pool")]
    unsafe fn default_uninit(target: &mut mem::MaybeUninit<Self>) {
        SparseChunk::default_uninit(
            target
                .as_mut_ptr()
                .cast::<mem::MaybeUninit<SparseChunk<Entry<A>, HASH_WIDTH>>>()
                .as_mut()
                .unwrap(),
        )
    }
}

impl<A> PoolClone for Node<A>
where
    A: Clone,
{
    #[cfg(feature = "pool")]
    unsafe fn clone_uninit(&self, target: &mut mem::MaybeUninit<Self>) {
        self.data.clone_uninit(
            target
                .as_mut_ptr()
                .cast::<mem::MaybeUninit<SparseChunk<Entry<A>, HASH_WIDTH>>>()
                .as_mut()
                .unwrap(),
        )
    }
}

#[derive(Clone)]
pub(crate) struct CollisionNode<A> {
    hash: HashBits,
    data: Vec<A>,
}

pub(crate) enum Entry<A> {
    Value(A, HashBits),
    Collision(Ref<CollisionNode<A>>),
    Node(PoolRef<Node<A>>),
}

impl<A: Clone> Clone for Entry<A> {
    fn clone(&self) -> Self {
        match self {
            Entry::Value(value, hash) => Entry::Value(value.clone(), *hash),
            Entry::Collision(coll) => Entry::Collision(coll.clone()),
            Entry::Node(node) => Entry::Node(node.clone()),
        }
    }
}

impl<A> Entry<A> {
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

impl<A> From<CollisionNode<A>> for Entry<A> {
    fn from(node: CollisionNode<A>) -> Self {
        Entry::Collision(Ref::new(node))
    }
}

impl<A> Default for Node<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A> Node<A> {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Node {
            data: SparseChunk::new(),
        }
    }

    /// Special constructor to allow initializing Nodes w/o incurring multiple memory copies.
    /// These copies really slow things down once Node crosses a certain size threshold and copies become calls to memcopy.
    #[inline]
    fn with(pool: &Pool<Self>, with: impl FnOnce(&mut Self)) -> PoolRef<Self> {
        let result: PoolRef<UnsafeCell<mem::MaybeUninit<Node<A>>>> = PoolRef::new_uninit(pool);
        #[allow(unsafe_code)]
        unsafe {
            // Initialize the MaybeUninit node
            (&mut *result.get()).write(Node::new());
            // Safety: UnsafeCell<Self> and UnsafeCell<MaybeUninit<Self>> have the same memory representation
            let result: PoolRef<UnsafeCell<Self>> = mem::transmute(result);
            let mut_ptr = UnsafeCell::raw_get(&*result);
            with(&mut *mut_ptr);
            // Safety UnsafeCell<Self> and Self have the same memory representation
            mem::transmute(result)
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn unit(pool: &Pool<Node<A>>, index: usize, value: Entry<A>) -> PoolRef<Self> {
        Self::with(pool, |this| {
            this.data.insert(index, value);
        })
    }

    #[inline]
    fn pair(
        pool: &Pool<Node<A>>,
        index1: usize,
        value1: Entry<A>,
        index2: usize,
        value2: Entry<A>,
    ) -> PoolRef<Self> {
        Self::with(pool, |this| {
            this.data.insert(index1, value1);
            this.data.insert(index2, value2);
        })
    }

    #[inline]
    pub(crate) fn single_child(
        pool: &Pool<Node<A>>,
        index: usize,
        node: PoolRef<Self>,
    ) -> PoolRef<Self> {
        Self::unit(pool, index, Entry::Node(node))
    }

    fn pop(&mut self) -> Entry<A> {
        self.data.pop().unwrap()
    }
}

impl<A: HashValue> Node<A> {
    fn merge_values(
        pool: &Pool<Node<A>>,
        value1: A,
        hash1: HashBits,
        value2: A,
        hash2: HashBits,
        shift: usize,
    ) -> PoolRef<Self> {
        let index1 = mask(hash1, shift) as usize;
        let index2 = mask(hash2, shift) as usize;
        if index1 != index2 {
            // Both values fit on the same level.
            Node::pair(
                pool,
                index1,
                Entry::Value(value1, hash1),
                index2,
                Entry::Value(value2, hash2),
            )
        } else if shift + HASH_SHIFT >= HASH_WIDTH {
            // If we're at the bottom, we've got a collision.
            Node::unit(
                pool,
                index1,
                Entry::from(CollisionNode::new(hash1, value1, value2)),
            )
        } else {
            // Pass the values down a level.
            let node = Node::merge_values(pool, value1, hash1, value2, hash2, shift + HASH_SHIFT);
            Node::single_child(pool, index1, node)
        }
    }

    pub(crate) fn get<BK>(&self, hash: HashBits, shift: usize, key: &BK) -> Option<&A>
    where
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let index = mask(hash, shift) as usize;
        if let Some(entry) = self.data.get(index) {
            match entry {
                Entry::Value(ref value, _) => {
                    if key == value.extract_key().borrow() {
                        Some(value)
                    } else {
                        None
                    }
                }
                Entry::Collision(ref coll) => coll.get(key),
                Entry::Node(ref child) => child.get(hash, shift + HASH_SHIFT, key),
            }
        } else {
            None
        }
    }

    pub(crate) fn get_mut<BK>(
        &mut self,
        pool: &Pool<Node<A>>,
        hash: HashBits,
        shift: usize,
        key: &BK,
    ) -> Option<&mut A>
    where
        A: Clone,
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let index = mask(hash, shift) as usize;
        if let Some(entry) = self.data.get_mut(index) {
            match entry {
                Entry::Value(ref mut value, _) => {
                    if key == value.extract_key().borrow() {
                        Some(value)
                    } else {
                        None
                    }
                }
                Entry::Collision(ref mut coll_ref) => {
                    let coll = Ref::make_mut(coll_ref);
                    coll.get_mut(key)
                }
                Entry::Node(ref mut child_ref) => {
                    let child = PoolRef::make_mut(pool, child_ref);
                    child.get_mut(pool, hash, shift + HASH_SHIFT, key)
                }
            }
        } else {
            None
        }
    }

    pub(crate) fn insert(
        &mut self,
        pool: &Pool<Node<A>>,
        hash: HashBits,
        shift: usize,
        value: A,
    ) -> Option<A>
    where
        A: Clone,
    {
        let index = mask(hash, shift) as usize;
        if let Some(entry) = self.data.get_mut(index) {
            let mut fallthrough = false;
            // Value is here
            match entry {
                // Update value or create a subtree
                Entry::Value(ref current, _) => {
                    if current.extract_key() == value.extract_key() {
                        // If we have a key match, fall through to the outer
                        // level where we replace the current value. If we
                        // don't, fall through to the inner level where we merge
                        // some nodes.
                        fallthrough = true;
                    }
                }
                // There's already a collision here.
                Entry::Collision(ref mut collision) => {
                    let coll = Ref::make_mut(collision);
                    return coll.insert(value);
                }
                Entry::Node(ref mut child_ref) => {
                    // Child node
                    let child = PoolRef::make_mut(pool, child_ref);
                    return child.insert(pool, hash, shift + HASH_SHIFT, value);
                }
            }
            #[allow(unsafe_code)]
            if !fallthrough {
                // If we get here, we're looking at a value entry that needs a merge.
                // We're going to be unsafe and pry it out of the reference, trusting
                // that we overwrite it with the merged node.
                let old_entry = unsafe { ptr::read(entry) };
                if shift + HASH_SHIFT >= HASH_WIDTH {
                    // We're at the lowest level, need to set up a collision node.
                    let coll = CollisionNode::new(hash, old_entry.unwrap_value(), value);
                    unsafe { ptr::write(entry, Entry::from(coll)) };
                } else if let Entry::Value(old_value, old_hash) = old_entry {
                    let node = Node::merge_values(
                        pool,
                        old_value,
                        old_hash,
                        value,
                        hash,
                        shift + HASH_SHIFT,
                    );
                    unsafe { ptr::write(entry, Entry::Node(node)) };
                } else {
                    unreachable!()
                }
                return None;
            }
        }
        // If we get here, either we found nothing at this index, in which case
        // we insert a new entry, or we hit a value entry with the same key, in
        // which case we replace it.
        self.data
            .insert(index, Entry::Value(value, hash))
            .map(Entry::unwrap_value)
    }

    pub(crate) fn remove<BK>(
        &mut self,
        pool: &Pool<Node<A>>,
        hash: HashBits,
        shift: usize,
        key: &BK,
    ) -> Option<A>
    where
        A: Clone,
        BK: Eq + ?Sized,
        A::Key: Borrow<BK>,
    {
        let index = mask(hash, shift) as usize;
        let mut new_node = None;
        let mut removed = None;
        if let Some(entry) = self.data.get_mut(index) {
            match entry {
                Entry::Value(ref value, _) => {
                    if key != value.extract_key().borrow() {
                        // Key wasn't in the map.
                        return None;
                    } // Otherwise, fall through to the removal.
                }
                Entry::Collision(ref mut coll_ref) => {
                    let coll = Ref::make_mut(coll_ref);
                    removed = coll.remove(key);
                    if coll.len() == 1 {
                        new_node = Some(coll.pop());
                    } else {
                        return removed;
                    }
                }
                Entry::Node(ref mut child_ref) => {
                    let child = PoolRef::make_mut(pool, child_ref);
                    match child.remove(pool, hash, shift + HASH_SHIFT, key) {
                        None => {
                            return None;
                        }
                        Some(value) => {
                            if child.len() == 1
                                && child.data[child.data.first_index().unwrap()].is_value()
                            {
                                // If the child now contains only a single value node,
                                // pull it up one level and discard the child.
                                removed = Some(value);
                                new_node = Some(child.pop());
                            } else {
                                return Some(value);
                            }
                        }
                    }
                }
            }
        }
        if let Some(node) = new_node {
            self.data.insert(index, node);
            return removed;
        }
        self.data.remove(index).map(Entry::unwrap_value)
    }
}

impl<A: HashValue> CollisionNode<A> {
    fn new(hash: HashBits, value1: A, value2: A) -> Self {
        CollisionNode {
            hash,
            data: vec![value1, value2],
        }
    }

    #[inline]
    fn len(&self) -> usize {
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
        let mut loc = None;
        for (index, item) in self.data.iter().enumerate() {
            if key == item.extract_key().borrow() {
                loc = Some(index);
            }
        }
        if let Some(index) = loc {
            Some(self.data.remove(index))
        } else {
            None
        }
    }

    fn pop(&mut self) -> Entry<A> {
        Entry::Value(self.data.pop().unwrap(), self.hash)
    }
}

// Ref iterator

pub(crate) struct Iter<'a, A> {
    count: usize,
    stack: Vec<ChunkIter<'a, Entry<A>, HASH_WIDTH>>,
    collision: Option<(HashBits, SliceIter<'a, A>)>,
}

// We impl Clone instead of deriving it, because we want Clone even if K and V aren't.
impl<'a, A> Clone for Iter<'a, A> {
    fn clone(&self) -> Self {
        Self {
            count: self.count,
            stack: self.stack.clone(),
            collision: self.collision.clone(),
        }
    }
}

impl<'a, A> Iter<'a, A>
where
    A: 'a,
{
    pub(crate) fn new(root: &'a Node<A>, size: usize) -> Self {
        let mut result = Iter {
            count: size,
            stack: Vec::with_capacity((HASH_WIDTH / HASH_SHIFT) + 1),
            collision: None,
        };
        result.stack.push(root.data.iter());
        result
    }
}

impl<'a, A> Iterator for Iter<'a, A>
where
    A: 'a,
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
                match current.next() {
                    Some(Entry::Value(value, hash)) => {
                        self.count -= 1;
                        return Some((value, *hash));
                    }
                    Some(Entry::Node(child)) => {
                        self.stack.push(child.data.iter());
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

impl<'a, A> ExactSizeIterator for Iter<'a, A> where A: 'a {}

impl<'a, A> FusedIterator for Iter<'a, A> where A: 'a {}

// Mut ref iterator

pub(crate) struct IterMut<'a, A> {
    count: usize,
    pool: Pool<Node<A>>,
    stack: Vec<ChunkIterMut<'a, Entry<A>, HASH_WIDTH>>,
    collision: Option<(HashBits, SliceIterMut<'a, A>)>,
}

impl<'a, A> IterMut<'a, A>
where
    A: 'a,
{
    pub(crate) fn new(pool: &Pool<Node<A>>, root: &'a mut Node<A>, size: usize) -> Self {
        let mut result = IterMut {
            count: size,
            pool: pool.clone(),
            stack: Vec::with_capacity((HASH_WIDTH / HASH_SHIFT) + 1),
            collision: None,
        };
        result.stack.push(root.data.iter_mut());
        result
    }
}

impl<'a, A> Iterator for IterMut<'a, A>
where
    A: Clone + 'a,
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
                match current.next() {
                    Some(Entry::Value(value, hash)) => {
                        self.count -= 1;
                        return Some((value, *hash));
                    }
                    Some(Entry::Node(child_ref)) => {
                        let child = PoolRef::make_mut(&self.pool, child_ref);
                        self.stack.push(child.data.iter_mut());
                    }
                    Some(Entry::Collision(coll_ref)) => {
                        let coll = Ref::make_mut(coll_ref);
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

impl<'a, A> ExactSizeIterator for IterMut<'a, A> where A: Clone + 'a {}

impl<'a, A> FusedIterator for IterMut<'a, A> where A: Clone + 'a {}

// Consuming iterator

pub(crate) struct Drain<A> {
    count: usize,
    pool: Pool<Node<A>>,
    stack: Vec<PoolRef<Node<A>>>,
    collision: Option<CollisionNode<A>>,
}

impl<A> Drain<A> {
    pub(crate) fn new(pool: &Pool<Node<A>>, root: PoolRef<Node<A>>, size: usize) -> Self {
        let mut result = Drain {
            count: size,
            pool: pool.clone(),
            stack: Vec::with_capacity((HASH_WIDTH / HASH_SHIFT) + 1),
            collision: None,
        };
        result.stack.push(root);
        result
    }
}

impl<A> Iterator for Drain<A>
where
    A: Clone,
{
    type Item = (A, HashBits);

    fn next(&mut self) -> Option<Self::Item> {
        'outer: loop {
            if let Some(coll) = &mut self.collision {
                match coll.data.pop() {
                    None => self.collision = None,
                    Some(value) => {
                        self.count -= 1;
                        return Some((value, coll.hash));
                    }
                };
            }

            while let Some(current) = self.stack.last_mut() {
                match PoolRef::make_mut(&self.pool, current).data.pop() {
                    Some(Entry::Value(value, hash)) => {
                        self.count -= 1;
                        return Some((value, hash));
                    }
                    Some(Entry::Node(child)) => {
                        self.stack.push(child);
                    }
                    Some(Entry::Collision(coll)) => {
                        self.collision = Some(clone_ref(coll));
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

impl<A> ExactSizeIterator for Drain<A> where A: Clone {}

impl<A> FusedIterator for Drain<A> where A: Clone {}

impl<A: fmt::Debug> fmt::Debug for Node<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "Node[ ")?;
        for i in self.data.indices() {
            write!(f, "{}: ", i)?;
            match &self.data[i] {
                Entry::Value(v, h) => write!(f, "{:?} :: {}, ", v, h)?,
                Entry::Collision(c) => write!(f, "Coll{:?} :: {}", c.data, c.hash)?,
                Entry::Node(n) => write!(f, "{:?}, ", n)?,
            }
        }
        write!(f, " ]")
    }
}
