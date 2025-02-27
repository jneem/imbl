// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::mem;
use std::ops::{Bound, RangeBounds};

use archery::{SharedPointer, SharedPointerKind};
use imbl_sized_chunks::Chunk;

pub(crate) use crate::config::ORD_CHUNK_SIZE as NODE_SIZE;
use crate::util::clone_ref;

use self::Insert::*;
use self::InsertAction::*;

const MEDIAN: usize = (NODE_SIZE + 1) >> 1;
const NUM_CHILDREN: usize = NODE_SIZE + 1;

pub trait BTreeValue {
    type Key;
    fn ptr_eq(&self, other: &Self) -> bool;
    fn search_key<BK>(slice: &[Self], key: &BK) -> Result<usize, usize>
    where
        BK: Ord + ?Sized,
        Self: Sized,
        Self::Key: Borrow<BK>;
    fn search_value(slice: &[Self], value: &Self) -> Result<usize, usize>
    where
        Self: Sized;
    fn cmp_keys<BK>(&self, other: &BK) -> Ordering
    where
        BK: Ord + ?Sized,
        Self::Key: Borrow<BK>;
    fn cmp_values(&self, other: &Self) -> Ordering;
}

/// A node in a `BTree`.
///
/// A node is either internal, or a leaf. Leaf nodes have `None` for every child, and internal
/// nodes have `Some(_)` for every child. There will never be a mixture of `None`s and `Some`s.
///
/// The `children` array is never empty, and always has exactly one more element than `keys`. The
/// empty tree has no keys, and a single `None` child.
pub(crate) struct Node<A, P: SharedPointerKind> {
    keys: Chunk<A, NODE_SIZE>,
    children: Chunk<Option<SharedPointer<Node<A, P>, P>>, NUM_CHILDREN>,
}

pub(crate) enum Insert<A, P: SharedPointerKind> {
    Added,
    Replaced(A),
    Split(Node<A, P>, A, Node<A, P>),
}

enum InsertAction<A, P: SharedPointerKind> {
    AddedAction,
    ReplacedAction(A),
    InsertAt,
    InsertSplit(Node<A, P>, A, Node<A, P>),
}

/// The result of a remove operation.
pub(crate) enum Remove<A, P: SharedPointerKind> {
    /// The key to remove was not found in the tree; nothing changed.
    NoChange,
    /// The key was found and removed: here it is.
    Removed(A),
    /// The key was found, and the root node of the tree was modified: here is the found key, and
    /// the new root node.
    Update(A, Node<A, P>),
}

enum Boundary {
    Lowest,
    Highest,
}

enum RemoveAction {
    DeleteAt(usize),
    PullUp(Boundary, usize, usize),
    Merge(usize),
    StealFromLeft(usize),
    StealFromRight(usize),
    MergeFirst(usize),
    ContinueDown(usize),
}

impl<A, P> Clone for Node<A, P>
where
    A: Clone,
    P: SharedPointerKind,
{
    fn clone(&self) -> Self {
        Node {
            keys: self.keys.clone(),
            children: self.children.clone(),
        }
    }
}

impl<A, P: SharedPointerKind> Default for Node<A, P> {
    fn default() -> Self {
        Node {
            keys: Chunk::new(),
            children: Chunk::unit(None),
        }
    }
}

impl<A, P: SharedPointerKind> Node<A, P> {
    #[inline]
    fn has_room(&self) -> bool {
        self.keys.len() < NODE_SIZE
    }

    /// This name is slightly misleading, because we actually check whether this node is the
    /// minimum allowed size (for a non-root node).
    #[inline]
    fn too_small(&self) -> bool {
        self.keys.len() < MEDIAN
    }

    #[inline]
    pub(crate) fn unit(value: A) -> Self {
        Node {
            keys: Chunk::unit(value),
            children: Chunk::pair(None, None),
        }
    }

    #[inline]
    pub(crate) fn new_from_split(left: Node<A, P>, median: A, right: Node<A, P>) -> Self {
        Node {
            keys: Chunk::unit(median),
            children: Chunk::pair(
                Some(SharedPointer::new(left)),
                Some(SharedPointer::new(right)),
            ),
        }
    }

    pub(crate) fn min(&self) -> Option<&A> {
        match self.children.first().unwrap() {
            None => self.keys.first(),
            Some(ref child) => child.min(),
        }
    }

    pub(crate) fn max(&self) -> Option<&A> {
        match self.children.last().unwrap() {
            None => self.keys.last(),
            Some(ref child) => child.max(),
        }
    }
}

impl<A: BTreeValue, P: SharedPointerKind> Node<A, P> {
    // Checks that this tree is balanced, and returns its depth.
    #[cfg(test)]
    pub(crate) fn check_depth(&self) -> usize {
        if self.children.is_empty() {
            // This is an empty tree.
            0
        } else if self.children[0].is_none() {
            // This is a leaf node.
            1
        } else {
            let mut depth = None;
            for c in self.children.iter() {
                let d = c.as_ref().unwrap().check_depth();
                assert!(depth.is_none() || depth == Some(d));
                depth = Some(d);
            }
            depth.unwrap()
        }
    }

    // Checks that the keys are in the right order.
    #[cfg(test)]
    pub(crate) fn check_order(&self) {
        fn recurse<A: BTreeValue, P: SharedPointerKind>(node: &Node<A, P>) -> (&A, &A) {
            for window in node.keys.windows(2) {
                assert!(window[0].cmp_values(&window[1]) == Ordering::Less);
            }
            if node.is_leaf() {
                (node.keys.first().unwrap(), node.keys.last().unwrap())
            } else {
                for i in 0..node.keys.len() {
                    let left_max = recurse(node.children[i].as_ref().unwrap()).1;
                    let right_min = recurse(node.children[i + 1].as_ref().unwrap()).0;
                    assert!(node.keys[i].cmp_values(left_max) == Ordering::Greater);
                    assert!(node.keys[i].cmp_values(right_min) == Ordering::Less);
                }
                (
                    recurse(node.children.first().unwrap().as_ref().unwrap()).0,
                    recurse(node.children.last().unwrap().as_ref().unwrap()).1,
                )
            }
        }
        if !self.keys.is_empty() {
            recurse(self);
        }
    }

    #[cfg(test)]
    pub(crate) fn check_size(&self) {
        fn recurse<A: BTreeValue, P: SharedPointerKind>(node: &Node<A, P>) {
            assert!(node.keys.len() + 1 == node.children.len());
            assert!(node.keys.len() + 1 >= MEDIAN);
            if !node.is_leaf() {
                for c in &node.children {
                    recurse(c.as_ref().unwrap());
                }
            }
        }
        if !self.is_leaf() {
            for c in &self.children {
                recurse(c.as_ref().unwrap());
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn check_sane(&self) {
        self.check_depth();
        self.check_order();
        self.check_size();
    }

    fn child_contains<BK>(&self, index: usize, key: &BK) -> bool
    where
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        if let Some(Some(ref child)) = self.children.get(index) {
            child.lookup(key).is_some()
        } else {
            false
        }
    }

    pub(crate) fn lookup<BK>(&self, key: &BK) -> Option<&A>
    where
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        if self.keys.is_empty() {
            return None;
        }
        // Perform a binary search, resulting in either a match or
        // the index of the first higher key, meaning we search the
        // child to the left of it.
        match A::search_key(&self.keys, key) {
            Ok(index) => Some(&self.keys[index]),
            Err(index) => match self.children[index] {
                None => None,
                Some(ref node) => node.lookup(key),
            },
        }
    }

    pub(crate) fn lookup_mut<BK>(&mut self, key: &BK) -> Option<&mut A>
    where
        A: Clone,
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        if self.keys.is_empty() {
            return None;
        }
        // Perform a binary search, resulting in either a match or
        // the index of the first higher key, meaning we search the
        // child to the left of it.
        match A::search_key(&self.keys, key) {
            Ok(index) => Some(&mut self.keys[index]),
            Err(index) => match self.children[index] {
                None => None,
                Some(ref mut child_ref) => {
                    let child = SharedPointer::make_mut(child_ref);
                    child.lookup_mut(key)
                }
            },
        }
    }

    pub(crate) fn lookup_prev<BK>(&self, key: &BK) -> Option<&A>
    where
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        if self.keys.is_empty() {
            return None;
        }
        match A::search_key(&self.keys, key) {
            Ok(index) => Some(&self.keys[index]),
            Err(index) => self.children[index]
                .as_ref()
                .and_then(|node| node.lookup_prev(key))
                // If we haven't found our search key yet, it isn't in any child subtree of ours.
                // That means that if index == 0 then we have no predecessor for the search key,
                // and if index > 0 then the predecessor is our key at index - 1.
                .or_else(|| index.checked_sub(1).and_then(|i| self.keys.get(i))),
        }
    }

    pub(crate) fn lookup_next<BK>(&self, key: &BK) -> Option<&A>
    where
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        if self.keys.is_empty() {
            return None;
        }
        match A::search_key(&self.keys, key) {
            Ok(index) => Some(&self.keys[index]),
            Err(index) => self.children[index]
                .as_ref()
                .and_then(|node| node.lookup_next(key))
                // If we don't find the search key in the child subtree, then either our next key
                // is the search key's successor, or else we don't have a successor in our subtree.
                .or_else(|| self.keys.get(index)),
        }
    }

    pub(crate) fn lookup_prev_mut<BK>(&mut self, key: &BK) -> Option<&mut A>
    where
        A: Clone,
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        if self.keys.is_empty() {
            return None;
        }
        let keys = &mut self.keys;
        match A::search_key(keys, key) {
            Ok(index) => Some(&mut keys[index]),
            Err(index) => self.children[index]
                .as_mut()
                .and_then(|node| SharedPointer::make_mut(node).lookup_prev_mut(key))
                .or_else(|| index.checked_sub(1).and_then(move |i| keys.get_mut(i))),
        }
    }

    pub(crate) fn lookup_next_mut<BK>(&mut self, key: &BK) -> Option<&mut A>
    where
        A: Clone,
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        if self.keys.is_empty() {
            return None;
        }
        let keys = &mut self.keys;
        match A::search_key(keys, key) {
            Ok(index) => Some(&mut keys[index]),
            Err(index) => self.children[index]
                .as_mut()
                .and_then(|node| SharedPointer::make_mut(node).lookup_next_mut(key))
                .or_else(move || keys.get_mut(index)),
        }
    }

    pub(crate) fn path_first<'a, BK>(
        &'a self,
        mut path: Vec<(&'a Node<A, P>, usize)>,
    ) -> Vec<(&'a Node<A, P>, usize)>
    where
        A: 'a,
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        if self.keys.is_empty() {
            return Vec::new();
        }
        match self.children[0] {
            None => {
                path.push((self, 0));
                path
            }
            Some(ref node) => {
                path.push((self, 0));
                node.path_first(path)
            }
        }
    }

    pub(crate) fn path_last<'a, BK>(
        &'a self,
        mut path: Vec<(&'a Node<A, P>, usize)>,
    ) -> Vec<(&'a Node<A, P>, usize)>
    where
        A: 'a,
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        if self.keys.is_empty() {
            return Vec::new();
        }
        let end = self.children.len() - 1;
        match self.children[end] {
            None => {
                path.push((self, end - 1));
                path
            }
            Some(ref node) => {
                path.push((self, end));
                node.path_last(path)
            }
        }
    }

    pub(crate) fn path_next<'a, BK>(
        &'a self,
        key: &BK,
        mut path: Vec<(&'a Node<A, P>, usize)>,
    ) -> Vec<(&'a Node<A, P>, usize)>
    where
        A: 'a,
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        if self.keys.is_empty() {
            return Vec::new();
        }
        match A::search_key(&self.keys, key) {
            Ok(index) => {
                path.push((self, index));
                path
            }
            Err(index) => match self.children[index] {
                None => match self.keys.get(index) {
                    Some(_) => {
                        path.push((self, index));
                        path
                    }
                    None => {
                        // go back up to find next
                        while let Some((node, idx)) = path.last() {
                            if node.keys.len() == *idx {
                                path.pop();
                            } else {
                                break;
                            }
                        }
                        path
                    }
                },
                Some(ref node) => {
                    path.push((self, index));
                    node.path_next(key, path)
                }
            },
        }
    }

    pub(crate) fn path_prev<'a, BK>(
        &'a self,
        key: &BK,
        mut path: Vec<(&'a Node<A, P>, usize)>,
    ) -> Vec<(&'a Node<A, P>, usize)>
    where
        A: 'a,
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        if self.keys.is_empty() {
            return Vec::new();
        }
        match A::search_key(&self.keys, key) {
            Ok(index) => {
                path.push((self, index));
                path
            }
            Err(index) => match self.children[index] {
                None if index == 0 => {
                    // go back up to find prev
                    while let Some((_, idx)) = path.last_mut() {
                        if *idx == 0 {
                            path.pop();
                        } else {
                            *idx -= 1;
                            break;
                        }
                    }
                    path
                }
                None => {
                    path.push((self, index - 1));
                    path
                }
                Some(ref node) => {
                    path.push((self, index));
                    node.path_prev(key, path)
                }
            },
        }
    }

    fn split(
        &mut self,
        value: A,
        ins_left: Option<Node<A, P>>,
        ins_right: Option<Node<A, P>>,
    ) -> Insert<A, P> {
        let left_child = ins_left.map(SharedPointer::new);
        let right_child = ins_right.map(SharedPointer::new);
        let index = A::search_value(&self.keys, &value).unwrap_err();
        let mut left_keys;
        let mut left_children;
        let mut right_keys;
        let mut right_children;
        let median;
        match index.cmp(&MEDIAN) {
            Ordering::Less => {
                self.children[index] = left_child;

                left_keys = Chunk::from_front(&mut self.keys, index);
                left_keys.push_back(value);
                left_keys.drain_from_front(&mut self.keys, MEDIAN - index - 1);

                left_children = Chunk::from_front(&mut self.children, index + 1);
                left_children.push_back(right_child);
                left_children.drain_from_front(&mut self.children, MEDIAN - index - 1);

                median = self.keys.pop_front();

                right_keys = Chunk::drain_from(&mut self.keys);
                right_children = Chunk::drain_from(&mut self.children);
            }
            Ordering::Greater => {
                self.children[index] = left_child;

                left_keys = Chunk::from_front(&mut self.keys, MEDIAN);
                left_children = Chunk::from_front(&mut self.children, MEDIAN + 1);

                median = self.keys.pop_front();

                right_keys = Chunk::from_front(&mut self.keys, index - MEDIAN - 1);
                right_keys.push_back(value);
                right_keys.append(&mut self.keys);

                right_children = Chunk::from_front(&mut self.children, index - MEDIAN);
                right_children.push_back(right_child);
                right_children.append(&mut self.children);
            }
            Ordering::Equal => {
                left_keys = Chunk::from_front(&mut self.keys, MEDIAN);
                left_children = Chunk::from_front(&mut self.children, MEDIAN);
                left_children.push_back(left_child);

                median = value;

                right_keys = Chunk::drain_from(&mut self.keys);
                right_children = Chunk::drain_from(&mut self.children);
                right_children[0] = right_child;
            }
        }

        debug_assert!(left_keys.len() == MEDIAN);
        debug_assert!(left_children.len() == MEDIAN + 1);
        debug_assert!(right_keys.len() == MEDIAN);
        debug_assert!(right_children.len() == MEDIAN + 1);

        Split(
            Node {
                keys: left_keys,
                children: left_children,
            },
            median,
            Node {
                keys: right_keys,
                children: right_children,
            },
        )
    }

    fn merge(middle: A, left: Node<A, P>, mut right: Node<A, P>) -> Node<A, P> {
        let mut keys = left.keys;
        keys.push_back(middle);
        keys.append(&mut right.keys);
        let mut children = left.children;
        children.append(&mut right.children);
        Node { keys, children }
    }

    fn pop_min(&mut self) -> (A, Option<SharedPointer<Node<A, P>, P>>) {
        let value = self.keys.pop_front();
        let child = self.children.pop_front();
        (value, child)
    }

    fn pop_max(&mut self) -> (A, Option<SharedPointer<Node<A, P>, P>>) {
        let value = self.keys.pop_back();
        let child = self.children.pop_back();
        (value, child)
    }

    fn push_min(&mut self, child: Option<SharedPointer<Node<A, P>, P>>, value: A) {
        self.keys.push_front(value);
        self.children.push_front(child);
    }

    fn push_max(&mut self, child: Option<SharedPointer<Node<A, P>, P>>, value: A) {
        self.keys.push_back(value);
        self.children.push_back(child);
    }

    fn is_leaf(&self) -> bool {
        // `children` is never empty, so we can index it.
        self.children[0].is_none()
    }

    pub(crate) fn insert(&mut self, value: A) -> Insert<A, P>
    where
        A: Clone,
    {
        if self.keys.is_empty() {
            self.keys.push_back(value);
            self.children.push_back(None);
            return Insert::Added;
        }
        let (median, left, right) = match A::search_value(&self.keys, &value) {
            // Key exists in node
            Ok(index) => {
                return Insert::Replaced(mem::replace(&mut self.keys[index], value));
            }
            // Key is adjacent to some key in node
            Err(index) => {
                let has_room = self.has_room();
                let action = match self.children[index] {
                    // No child at location, this is the target node.
                    None => InsertAt,
                    // Child at location, pass it on.
                    Some(ref mut child_ref) => {
                        let child = SharedPointer::make_mut(child_ref);
                        match child.insert(value.clone()) {
                            Insert::Added => AddedAction,
                            Insert::Replaced(value) => ReplacedAction(value),
                            Insert::Split(left, median, right) => InsertSplit(left, median, right),
                        }
                    }
                };
                match action {
                    ReplacedAction(value) => return Insert::Replaced(value),
                    AddedAction => {
                        return Insert::Added;
                    }
                    InsertAt => {
                        if has_room {
                            self.keys.insert(index, value);
                            self.children.insert(index + 1, None);
                            return Insert::Added;
                        } else {
                            (value, None, None)
                        }
                    }
                    InsertSplit(left, median, right) => {
                        if has_room {
                            self.children[index] = Some(SharedPointer::new(left));
                            self.keys.insert(index, median);
                            self.children
                                .insert(index + 1, Some(SharedPointer::new(right)));
                            return Insert::Added;
                        } else {
                            (median, Some(left), Some(right))
                        }
                    }
                }
            }
        };
        self.split(median, left, right)
    }

    pub(crate) fn remove<BK>(&mut self, key: &BK) -> Remove<A, P>
    where
        A: Clone,
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        let index = A::search_key(&self.keys, key);
        self.remove_index(index, Ok(key))
    }

    fn remove_target<BK>(&mut self, target: Result<&BK, Boundary>) -> Remove<A, P>
    where
        A: Clone,
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        let index = match target {
            Ok(key) => A::search_key(&self.keys, key),
            Err(Boundary::Lowest) => Err(0),
            Err(Boundary::Highest) => Err(self.keys.len()),
        };
        self.remove_index(index, target)
    }

    fn remove_index<BK>(
        &mut self,
        index: Result<usize, usize>,
        target: Result<&BK, Boundary>,
    ) -> Remove<A, P>
    where
        A: Clone,
        BK: Ord + ?Sized,
        A::Key: Borrow<BK>,
    {
        let action = match index {
            // Key exists in node, remove it.
            Ok(index) => {
                match (&self.children[index], &self.children[index + 1]) {
                    // If we're a leaf, just delete the entry.
                    (&None, &None) => RemoveAction::DeleteAt(index),
                    // First consider pulling either predecessor (from left) or successor (from right).
                    // otherwise just merge the two small children.
                    (Some(left), Some(right)) => {
                        if !left.too_small() {
                            RemoveAction::PullUp(Boundary::Highest, index, index)
                        } else if !right.too_small() {
                            RemoveAction::PullUp(Boundary::Lowest, index, index + 1)
                        } else {
                            RemoveAction::Merge(index)
                        }
                    }
                    _ => unreachable!("Branch missing children"),
                }
            }
            // Target is adjacent to some key in node
            Err(index) => match self.children[index] {
                // We're deading with a leaf node
                None => match target {
                    // No child at location means key isn't in map.
                    Ok(_key) => return Remove::NoChange,
                    // Looking for the lowest or highest key
                    Err(Boundary::Lowest) => RemoveAction::DeleteAt(0),
                    Err(Boundary::Highest) => RemoveAction::DeleteAt(self.keys.len() - 1),
                },
                // Child at location, but it's at minimum capacity.
                Some(ref child) if child.too_small() => {
                    let left = if index > 0 {
                        self.children.get(index - 1)
                    } else {
                        None
                    }; // index is usize and can't be negative, best make sure it never is.
                    match (left, self.children.get(index + 1)) {
                        // If it has a left sibling with capacity, steal a key from it.
                        (Some(Some(old_left)), _) if !old_left.too_small() => {
                            RemoveAction::StealFromLeft(index)
                        }
                        // If it has a right sibling with capacity, same as above.
                        (_, Some(Some(old_right))) if !old_right.too_small() => {
                            RemoveAction::StealFromRight(index)
                        }
                        // If it has neither, we'll have to merge it with a sibling.
                        // If we have a right sibling, we'll merge with that.
                        (_, Some(&Some(_))) => RemoveAction::MergeFirst(index),
                        // If we have a left sibling, we'll merge with that.
                        (Some(&Some(_)), _) => RemoveAction::MergeFirst(index - 1),
                        // If none of the above, we're in a bad state.
                        _ => unreachable!(),
                    }
                }
                // Child at location, and it's big enough, we can recurse down.
                Some(_) => RemoveAction::ContinueDown(index),
            },
        };
        match action {
            RemoveAction::DeleteAt(index) => {
                let pair = self.keys.remove(index);
                self.children.remove(index);
                Remove::Removed(pair)
            }
            RemoveAction::PullUp(boundary, pull_to, child_index) => {
                let children = &mut self.children;
                let mut update = None;
                let value;
                if let Some(&mut Some(ref mut child_ref)) = children.get_mut(child_index) {
                    let child = SharedPointer::make_mut(child_ref);
                    match child.remove_target(Err(boundary)) {
                        Remove::NoChange => unreachable!(),
                        Remove::Removed(pulled_value) => {
                            value = self.keys.set(pull_to, pulled_value);
                        }
                        Remove::Update(pulled_value, new_child) => {
                            value = self.keys.set(pull_to, pulled_value);
                            update = Some(new_child);
                        }
                    }
                } else {
                    unreachable!()
                }
                if let Some(new_child) = update {
                    children[child_index] = Some(SharedPointer::new(new_child));
                }
                Remove::Removed(value)
            }
            RemoveAction::Merge(index) => {
                let left = self.children.remove(index).unwrap();
                let right = self.children[index].take().unwrap();
                let value = self.keys.remove(index);
                let mut merged_child = Node::merge(value, clone_ref(left), clone_ref(right));
                let (removed, new_child) = match merged_child.remove_target(target) {
                    Remove::NoChange => unreachable!(),
                    Remove::Removed(removed) => (removed, merged_child),
                    Remove::Update(removed, updated_child) => (removed, updated_child),
                };
                if self.keys.is_empty() {
                    // If we've depleted the root node, the merged child becomes the root.
                    Remove::Update(removed, new_child)
                } else {
                    self.children[index] = Some(SharedPointer::new(new_child));
                    Remove::Removed(removed)
                }
            }
            RemoveAction::StealFromLeft(index) => {
                let mut update = None;
                let out_value;
                {
                    let mut children = self.children.as_mut_slice()[index - 1..=index]
                        .iter_mut()
                        .map(|n| n.as_mut().unwrap());
                    let left = SharedPointer::make_mut(children.next().unwrap());
                    let child = SharedPointer::make_mut(children.next().unwrap());
                    // Prepare the rebalanced node.
                    child.push_min(
                        left.children.last().unwrap().clone(),
                        self.keys[index - 1].clone(),
                    );
                    match child.remove_target(target) {
                        Remove::NoChange => {
                            // Key wasn't there, we need to revert the steal.
                            child.pop_min();
                            return Remove::NoChange;
                        }
                        Remove::Removed(value) => {
                            // If we did remove something, we complete the rebalancing.
                            let (left_value, _) = left.pop_max();
                            self.keys[index - 1] = left_value;
                            out_value = value;
                        }
                        Remove::Update(value, new_child) => {
                            // If we did remove something, we complete the rebalancing.
                            let (left_value, _) = left.pop_max();
                            self.keys[index - 1] = left_value;
                            update = Some(new_child);
                            out_value = value;
                        }
                    }
                }
                if let Some(new_child) = update {
                    self.children[index] = Some(SharedPointer::new(new_child));
                }
                Remove::Removed(out_value)
            }
            RemoveAction::StealFromRight(index) => {
                let mut update = None;
                let out_value;
                {
                    let mut children = self.children.as_mut_slice()[index..index + 2]
                        .iter_mut()
                        .map(|n| n.as_mut().unwrap());
                    let child = SharedPointer::make_mut(children.next().unwrap());
                    let right = SharedPointer::make_mut(children.next().unwrap());
                    // Prepare the rebalanced node.
                    child.push_max(right.children[0].clone(), self.keys[index].clone());
                    match child.remove_target(target) {
                        Remove::NoChange => {
                            // Key wasn't there, we need to revert the steal.
                            child.pop_max();
                            return Remove::NoChange;
                        }
                        Remove::Removed(value) => {
                            // If we did remove something, we complete the rebalancing.
                            let (right_value, _) = right.pop_min();
                            self.keys[index] = right_value;
                            out_value = value;
                        }
                        Remove::Update(value, new_child) => {
                            // If we did remove something, we complete the rebalancing.
                            let (right_value, _) = right.pop_min();
                            self.keys[index] = right_value;
                            update = Some(new_child);
                            out_value = value;
                        }
                    }
                }
                if let Some(new_child) = update {
                    self.children[index] = Some(SharedPointer::new(new_child));
                }
                Remove::Removed(out_value)
            }
            RemoveAction::MergeFirst(index) => {
                if let Ok(key) = target {
                    // Bail early if we're looking for a not existing key
                    match self.keys[index].cmp_keys(key) {
                        Ordering::Less if !self.child_contains(index + 1, key) => {
                            return Remove::NoChange
                        }
                        Ordering::Greater if !self.child_contains(index, key) => {
                            return Remove::NoChange
                        }
                        _ => (),
                    }
                }
                let left = self.children.remove(index).unwrap();
                let right = self.children[index].take().unwrap();
                let middle = self.keys.remove(index);
                let mut merged = Node::merge(middle, clone_ref(left), clone_ref(right));
                let update;
                let out_value;
                match merged.remove_target(target) {
                    Remove::NoChange => {
                        panic!("nodes::btree::Node::remove: caught an absent key too late while merging");
                    }
                    Remove::Removed(value) => {
                        if self.keys.is_empty() {
                            return Remove::Update(value, merged);
                        }
                        update = merged;
                        out_value = value;
                    }
                    Remove::Update(value, new_child) => {
                        if self.keys.is_empty() {
                            return Remove::Update(value, new_child);
                        }
                        update = new_child;
                        out_value = value;
                    }
                }
                self.children[index] = Some(SharedPointer::new(update));
                Remove::Removed(out_value)
            }
            RemoveAction::ContinueDown(index) => {
                let mut update = None;
                let out_value;
                if let Some(&mut Some(ref mut child_ref)) = self.children.get_mut(index) {
                    let child = SharedPointer::make_mut(child_ref);
                    match child.remove_target(target) {
                        Remove::NoChange => return Remove::NoChange,
                        Remove::Removed(value) => {
                            out_value = value;
                        }
                        Remove::Update(value, new_child) => {
                            update = Some(new_child);
                            out_value = value;
                        }
                    }
                } else {
                    unreachable!()
                }
                if let Some(new_child) = update {
                    self.children[index] = Some(SharedPointer::new(new_child));
                }
                Remove::Removed(out_value)
            }
        }
    }
}

// Iterator

/// An iterator over an ordered set.
pub(crate) struct Iter<'a, A, P: SharedPointerKind> {
    /// Path to the next element that we'll yield if we take a forward step.  Each element here is
    /// of the form `(node, index)`. For the last path element, `index` points to the next key to
    /// yield. For every other path element, `index` is the child index of the next node in the
    /// path.
    fwd_path: Vec<(&'a Node<A, P>, usize)>,
    /// Path to the next element that we'll yield if we take a backward step. This has the same
    /// format as `fwd_path`.
    back_path: Vec<(&'a Node<A, P>, usize)>,
    pub(crate) remaining: usize,
}

// We impl Clone instead of deriving it, because we want Clone even if K and V aren't.
impl<'a, A, P: SharedPointerKind> Clone for Iter<'a, A, P> {
    fn clone(&self) -> Self {
        Iter {
            fwd_path: self.fwd_path.clone(),
            back_path: self.back_path.clone(),
            remaining: self.remaining,
        }
    }
}

impl<'a, A: BTreeValue, P: SharedPointerKind> Iter<'a, A, P> {
    pub(crate) fn new<R, BK>(root: &'a Node<A, P>, size: usize, range: R) -> Self
    where
        R: RangeBounds<BK>,
        A::Key: Borrow<BK>,
        BK: Ord + ?Sized,
    {
        let fwd_path = match range.start_bound() {
            Bound::Included(key) => root.path_next(key, Vec::new()),
            Bound::Excluded(key) => {
                let mut path = root.path_next(key, Vec::new());
                if let Some(value) = Self::get(&path) {
                    if value.cmp_keys(key) == Ordering::Equal {
                        Self::step_forward(&mut path);
                    }
                }
                path
            }
            Bound::Unbounded => root.path_first(Vec::new()),
        };
        let back_path = match range.end_bound() {
            Bound::Included(key) => root.path_prev(key, Vec::new()),
            Bound::Excluded(key) => {
                let mut path = root.path_prev(key, Vec::new());
                if let Some(value) = Self::get(&path) {
                    if value.cmp_keys(key) == Ordering::Equal {
                        Self::step_back(&mut path);
                    }
                }
                path
            }
            Bound::Unbounded => root.path_last(Vec::new()),
        };
        Iter {
            fwd_path,
            back_path,
            remaining: size,
        }
    }

    fn get(path: &[(&'a Node<A, P>, usize)]) -> Option<&'a A> {
        match path.last() {
            Some((node, index)) => Some(&node.keys[*index]),
            None => None,
        }
    }

    fn step_forward(path: &mut Vec<(&'a Node<A, P>, usize)>) -> Option<&'a A> {
        match path.pop() {
            Some((node, index)) => {
                let index = index + 1;
                match node.children[index] {
                    // Child between current and next key -> step down
                    Some(ref child) => {
                        path.push((node, index));
                        path.push((child, 0));
                        let mut node = child;
                        while let Some(ref left_child) = node.children[0] {
                            path.push((left_child, 0));
                            node = left_child;
                        }
                        Some(&node.keys[0])
                    }
                    None => match node.keys.get(index) {
                        // Yield next key
                        value @ Some(_) => {
                            path.push((node, index));
                            value
                        }
                        // No more keys -> exhausted level, step up and yield
                        None => loop {
                            match path.pop() {
                                None => {
                                    return None;
                                }
                                Some((node, index)) => {
                                    if let value @ Some(_) = node.keys.get(index) {
                                        path.push((node, index));
                                        return value;
                                    }
                                }
                            }
                        },
                    },
                }
            }
            None => None,
        }
    }

    fn step_back(path: &mut Vec<(&'a Node<A, P>, usize)>) -> Option<&'a A> {
        // TODO: we're doing some repetitive leaf-vs-internal checking.
        match path.pop() {
            Some((node, index)) => match node.children[index] {
                Some(ref child) => {
                    path.push((node, index));
                    let mut end = if child.is_leaf() {
                        child.keys.len() - 1
                    } else {
                        child.children.len() - 1
                    };
                    path.push((child, end));
                    let mut node = child;
                    while let Some(ref right_child) = node.children[end] {
                        end = if right_child.is_leaf() {
                            right_child.keys.len() - 1
                        } else {
                            right_child.children.len() - 1
                        };
                        path.push((right_child, end));
                        node = right_child;
                    }
                    Some(&node.keys[end])
                }
                None => {
                    if index == 0 {
                        loop {
                            match path.pop() {
                                None => {
                                    return None;
                                }
                                Some((node, index)) => {
                                    if index > 0 {
                                        let index = index - 1;
                                        path.push((node, index));
                                        return Some(&node.keys[index]);
                                    }
                                }
                            }
                        }
                    } else {
                        let index = index - 1;
                        path.push((node, index));
                        Some(&node.keys[index])
                    }
                }
            },
            None => None,
        }
    }
}

impl<'a, A: 'a + BTreeValue, P: SharedPointerKind> Iterator for Iter<'a, A, P> {
    type Item = &'a A;

    fn next(&mut self) -> Option<Self::Item> {
        match Iter::get(&self.fwd_path) {
            None => None,
            Some(value) => match Iter::get(&self.back_path) {
                Some(last_value) if value.cmp_values(last_value) == Ordering::Greater => None,
                None => None,
                Some(_) => {
                    Iter::step_forward(&mut self.fwd_path);
                    self.remaining -= 1;
                    Some(value)
                }
            },
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.remaining))
    }
}

impl<'a, A: 'a + BTreeValue, P: SharedPointerKind> DoubleEndedIterator for Iter<'a, A, P> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match Iter::get(&self.back_path) {
            None => None,
            Some(value) => match Iter::get(&self.fwd_path) {
                Some(last_value) if value.cmp_values(last_value) == Ordering::Less => None,
                None => None,
                Some(_) => {
                    Iter::step_back(&mut self.back_path);
                    self.remaining -= 1;
                    Some(value)
                }
            },
        }
    }
}

// Consuming iterator

enum ConsumingIterItem<A, P: SharedPointerKind> {
    Consider(Node<A, P>),
    Yield(A),
}

/// A consuming iterator over an ordered set.
pub struct ConsumingIter<A, P: SharedPointerKind> {
    fwd_last: Option<A>,
    fwd_stack: Vec<ConsumingIterItem<A, P>>,
    back_last: Option<A>,
    back_stack: Vec<ConsumingIterItem<A, P>>,
    remaining: usize,
}

impl<A: Clone, P: SharedPointerKind> ConsumingIter<A, P> {
    pub(crate) fn new(root: &Node<A, P>, total: usize) -> Self {
        ConsumingIter {
            fwd_last: None,
            fwd_stack: vec![ConsumingIterItem::Consider(root.clone())],
            back_last: None,
            back_stack: vec![ConsumingIterItem::Consider(root.clone())],
            remaining: total,
        }
    }

    fn push_node(
        stack: &mut Vec<ConsumingIterItem<A, P>>,
        maybe_node: Option<SharedPointer<Node<A, P>, P>>,
    ) {
        if let Some(node) = maybe_node {
            stack.push(ConsumingIterItem::Consider(clone_ref(node)))
        }
    }

    fn push(stack: &mut Vec<ConsumingIterItem<A, P>>, mut node: Node<A, P>) {
        for _n in 0..node.keys.len() {
            ConsumingIter::push_node(stack, node.children.pop_back());
            stack.push(ConsumingIterItem::Yield(node.keys.pop_back()));
        }
        ConsumingIter::push_node(stack, node.children.pop_back());
    }

    fn push_fwd(&mut self, node: Node<A, P>) {
        ConsumingIter::push(&mut self.fwd_stack, node)
    }

    fn push_node_back(&mut self, maybe_node: Option<SharedPointer<Node<A, P>, P>>) {
        if let Some(node) = maybe_node {
            self.back_stack
                .push(ConsumingIterItem::Consider(clone_ref(node)))
        }
    }

    fn push_back(&mut self, mut node: Node<A, P>) {
        for _i in 0..node.keys.len() {
            self.push_node_back(node.children.pop_front());
            self.back_stack
                .push(ConsumingIterItem::Yield(node.keys.pop_front()));
        }
        self.push_node_back(node.children.pop_back());
    }
}

impl<A, P> Iterator for ConsumingIter<A, P>
where
    A: BTreeValue + Clone,
    P: SharedPointerKind,
{
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.fwd_stack.pop() {
                None => {
                    self.remaining = 0;
                    return None;
                }
                Some(ConsumingIterItem::Consider(node)) => self.push_fwd(node),
                Some(ConsumingIterItem::Yield(value)) => {
                    if let Some(ref last) = self.back_last {
                        if value.cmp_values(last) != Ordering::Less {
                            self.fwd_stack.clear();
                            self.back_stack.clear();
                            self.remaining = 0;
                            return None;
                        }
                    }
                    self.remaining -= 1;
                    self.fwd_last = Some(value.clone());
                    return Some(value);
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<A, P> DoubleEndedIterator for ConsumingIter<A, P>
where
    A: BTreeValue + Clone,
    P: SharedPointerKind,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            match self.back_stack.pop() {
                None => {
                    self.remaining = 0;
                    return None;
                }
                Some(ConsumingIterItem::Consider(node)) => self.push_back(node),
                Some(ConsumingIterItem::Yield(value)) => {
                    if let Some(ref last) = self.fwd_last {
                        if value.cmp_values(last) != Ordering::Greater {
                            self.fwd_stack.clear();
                            self.back_stack.clear();
                            self.remaining = 0;
                            return None;
                        }
                    }
                    self.remaining -= 1;
                    self.back_last = Some(value.clone());
                    return Some(value);
                }
            }
        }
    }
}

impl<A: BTreeValue + Clone, P: SharedPointerKind> ExactSizeIterator for ConsumingIter<A, P> {}

// DiffIter

/// An iterator over the differences between two ordered sets.
pub struct DiffIter<'a, 'b, A, P: SharedPointerKind> {
    old_stack: Vec<IterItem<'a, A, P>>,
    new_stack: Vec<IterItem<'b, A, P>>,
}

/// A description of a difference between two ordered sets.
#[derive(PartialEq, Eq, Debug)]
pub enum DiffItem<'a, 'b, A> {
    /// This value has been added to the new set.
    Add(&'b A),
    /// This value has been changed between the two sets.
    Update {
        /// The old value.
        old: &'a A,
        /// The new value.
        new: &'b A,
    },
    /// This value has been removed from the new set.
    Remove(&'a A),
}

enum IterItem<'a, A, P: SharedPointerKind> {
    Consider(&'a Node<A, P>),
    Yield(&'a A),
}

impl<'a, 'b, A: 'a + 'b, P: SharedPointerKind> DiffIter<'a, 'b, A, P> {
    pub(crate) fn new(old: &'a Node<A, P>, new: &'b Node<A, P>) -> Self {
        DiffIter {
            old_stack: if old.keys.is_empty() {
                Vec::new()
            } else {
                vec![IterItem::Consider(old)]
            },
            new_stack: if new.keys.is_empty() {
                Vec::new()
            } else {
                vec![IterItem::Consider(new)]
            },
        }
    }

    fn push_node<'either>(
        stack: &mut Vec<IterItem<'either, A, P>>,
        maybe_node: &'either Option<SharedPointer<Node<A, P>, P>>,
    ) {
        if let Some(node) = maybe_node {
            stack.push(IterItem::Consider(node))
        }
    }

    fn push<'either>(stack: &mut Vec<IterItem<'either, A, P>>, node: &'either Node<A, P>) {
        for n in 0..node.keys.len() {
            let i = node.keys.len() - n;
            Self::push_node(stack, &node.children[i]);
            stack.push(IterItem::Yield(&node.keys[i - 1]));
        }
        Self::push_node(stack, &node.children[0]);
    }
}

impl<'a, 'b, A, P> Iterator for DiffIter<'a, 'b, A, P>
where
    A: 'a + 'b + BTreeValue + PartialEq,
    P: SharedPointerKind,
{
    type Item = DiffItem<'a, 'b, A>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match (self.old_stack.pop(), self.new_stack.pop()) {
                (None, None) => return None,
                (None, Some(new)) => match new {
                    IterItem::Consider(new) => Self::push(&mut self.new_stack, new),
                    IterItem::Yield(new) => return Some(DiffItem::Add(new)),
                },
                (Some(old), None) => match old {
                    IterItem::Consider(old) => Self::push(&mut self.old_stack, old),
                    IterItem::Yield(old) => return Some(DiffItem::Remove(old)),
                },
                (Some(old), Some(new)) => match (old, new) {
                    (IterItem::Consider(old), IterItem::Consider(new)) => {
                        if !std::ptr::eq(old, new) {
                            match old.keys[0].cmp_values(&new.keys[0]) {
                                Ordering::Less => {
                                    Self::push(&mut self.old_stack, old);
                                    self.new_stack.push(IterItem::Consider(new));
                                }
                                Ordering::Greater => {
                                    self.old_stack.push(IterItem::Consider(old));
                                    Self::push(&mut self.new_stack, new);
                                }
                                Ordering::Equal => {
                                    Self::push(&mut self.old_stack, old);
                                    Self::push(&mut self.new_stack, new);
                                }
                            }
                        }
                    }
                    (IterItem::Consider(old), IterItem::Yield(new)) => {
                        Self::push(&mut self.old_stack, old);
                        self.new_stack.push(IterItem::Yield(new));
                    }
                    (IterItem::Yield(old), IterItem::Consider(new)) => {
                        self.old_stack.push(IterItem::Yield(old));
                        Self::push(&mut self.new_stack, new);
                    }
                    (IterItem::Yield(old), IterItem::Yield(new)) => match old.cmp_values(new) {
                        Ordering::Less => {
                            self.new_stack.push(IterItem::Yield(new));
                            return Some(DiffItem::Remove(old));
                        }
                        Ordering::Equal => {
                            if old != new {
                                return Some(DiffItem::Update { old, new });
                            }
                        }
                        Ordering::Greater => {
                            self.old_stack.push(IterItem::Yield(old));
                            return Some(DiffItem::Add(new));
                        }
                    },
                },
            }
        }
    }
}
