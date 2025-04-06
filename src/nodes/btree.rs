// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::borrow::Borrow;
use std::collections::VecDeque;
use std::iter::FromIterator;
use std::mem;
use std::ops::{Bound, RangeBounds};

use archery::{SharedPointer, SharedPointerKind};
use imbl_sized_chunks::Chunk;

pub(crate) use crate::config::ORD_CHUNK_SIZE as NODE_SIZE;

const MEDIAN: usize = NODE_SIZE / 2;
const THIRD: usize = NODE_SIZE / 3;
const NUM_CHILDREN: usize = NODE_SIZE + 1;

/// A node in a `B+Tree`.
#[derive(Debug)]
pub(crate) enum Node<K, V, P: SharedPointerKind> {
    Branch(Branch<K, V, P>),
    Leaf(Leaf<K, V>),
}

impl<K, V, P: SharedPointerKind> Node<K, V, P> {
    pub(crate) fn unit(key: K, value: V) -> Self {
        Node::Leaf(Leaf {
            keys: Chunk::unit((key, value)),
        })
    }

    fn level(&self) -> usize {
        match self {
            Node::Branch(branch) => branch.level,
            Node::Leaf(_) => 0,
        }
    }
}

/// A branch node in a `B+Tree`.
/// Invariants:
/// * keys are ordered and unique
/// * keys.len() + 1 == children.len()
/// * all children have level = level - 1 (or level is 1 and all children are leaves)
/// * all keys in the subtree at children[i] are between keys[i - 1] (if i > 0) and keys[i] (if i < keys.len()).
#[derive(Debug)]
pub(crate) struct Branch<K, V, P: SharedPointerKind> {
    keys: Chunk<K, NODE_SIZE>,
    children: Chunk<SharedPointer<Node<K, V, P>, P>, NUM_CHILDREN>,
    /// The level of the node in the tree, leaves are implicitly the level 0.
    level: usize,
}

impl<K, V, P: SharedPointerKind> Branch<K, V, P> {
    pub(crate) fn pop_single_child(&mut self) -> Option<SharedPointer<Node<K, V, P>, P>> {
        if self.children.len() == 1 {
            debug_assert_eq!(self.keys.len(), 0);
            return Some(self.children.pop_back());
        }
        None
    }
}

/// A leaf node in a `B+Tree`.
///
/// Invariants:
/// * keys are ordered and unique
/// * leaf is the lowest level in the tree (level 0)
#[derive(Debug)]
pub(crate) struct Leaf<K, V> {
    keys: Chunk<(K, V), NODE_SIZE>,
}

impl<K: Ord + Clone, V: Clone, P: SharedPointerKind> Node<K, V, P> {
    /// Removes a key from the node or its children.
    /// Returns `true` if the node is underflowed and should be rebalanced.
    pub(crate) fn remove<BK>(&mut self, key: &BK, removed: &mut Option<(K, V)>) -> bool
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        match self {
            Node::Branch(branch) => {
                let i = branch
                    .keys
                    .binary_search_by(|k| k.borrow().cmp(key))
                    .map(|x| x + 1)
                    .unwrap_or_else(|x| x);
                let child = &mut branch.children[i];
                if SharedPointer::make_mut(child).remove(key, removed) {
                    Self::branch_rebalance_children(branch, i);
                }
                // Underflow if the branch is < 1/2 full. Since the branches are relatively
                // rarely rebalanced (given relaxed leaf underflow), we can afford to be
                // a bit more conservative here.
                branch.keys.len() < MEDIAN
            }
            Node::Leaf(leaf) => {
                if let Ok(i) = leaf.keys.binary_search_by(|(k, _)| k.borrow().cmp(key)) {
                    *removed = Some(leaf.keys.remove(i));
                }
                // Underflow if the leaf is < 1/3 full. This relaxed underflow (vs. 1/2 full) is
                // useful to prevent degenerate cases where a random insert/remove workload will
                // constantly merge/split a leaf.
                leaf.keys.len() < THIRD
            }
        }
    }

    #[cold]
    pub(crate) fn branch_rebalance_children(branch: &mut Branch<K, V, P>, underflow_idx: usize) {
        let left_idx = underflow_idx.saturating_sub(1);
        let (left, mid, right) = match &branch.children[left_idx..] {
            [left, mid, right, ..] => (&**left, &**mid, Some(&**right)),
            [left, mid, ..] => (&**left, &**mid, None),
            _ => return,
        };
        // Prefer merging two sibling children if we can fit them into a single node.
        // But also try to rebalance is the smallest child is small (< 1/3), to amortize the cost of rebalancing.
        // Since we prefer merging, for rebalancing to apply the the largest child will be least 2/3 full,
        // which results in two at least half full nodes after rebalancing.
        match (left, mid, right) {
            (Node::Leaf(left), Node::Leaf(mid), _)
                if left.keys.len() + mid.keys.len() <= NODE_SIZE =>
            {
                Self::merge_leaves(branch, left_idx, false);
            }
            (_, Node::Leaf(mid), Some(Node::Leaf(right)))
                if mid.keys.len() + right.keys.len() <= NODE_SIZE =>
            {
                Self::merge_leaves(branch, left_idx + 1, true);
            }
            (Node::Leaf(left), Node::Leaf(mid), _)
                if mid.keys.len().min(left.keys.len()) < THIRD =>
            {
                Self::rebalance_leaves(branch, left_idx);
            }
            (_, Node::Leaf(mid), Some(Node::Leaf(right)))
                if mid.keys.len().min(right.keys.len()) < THIRD =>
            {
                Self::rebalance_leaves(branch, left_idx + 1);
            }
            (Node::Branch(left), Node::Branch(mid), _)
                if left.keys.len() + mid.keys.len() < NODE_SIZE =>
            {
                Self::merge_branches(branch, left_idx, false);
            }
            (_, Node::Branch(mid), Some(Node::Branch(right)))
                if mid.keys.len() + right.keys.len() < NODE_SIZE =>
            {
                Self::merge_branches(branch, left_idx + 1, true);
            }
            (Node::Branch(left), Node::Branch(mid), _)
                if mid.keys.len().min(left.keys.len()) < THIRD =>
            {
                Self::rebalance_branches(branch, left_idx);
            }
            (_, Node::Branch(mid), Some(Node::Branch(right)))
                if mid.keys.len().min(right.keys.len()) < THIRD =>
            {
                Self::rebalance_branches(branch, left_idx + 1);
            }
            _ => (),
        }
    }

    /// Merges two children leaves of this branch.
    ///
    /// Assumes that the two children can fit in a single leaf, panicking if not.
    fn merge_leaves(branch: &mut Branch<K, V, P>, left_idx: usize, keep_left: bool) {
        debug_assert_eq!(branch.level, 1);
        let [left, right, ..] = &mut branch.children[left_idx..] else {
            unreachable!()
        };
        if keep_left {
            let left = SharedPointer::make_mut(left);
            let (Node::Leaf(left), Node::Leaf(right)) = (left, &**right) else {
                unreachable!()
            };
            left.keys.extend(right.keys.iter().cloned());
        } else {
            let right = SharedPointer::make_mut(right);
            let (Node::Leaf(left), Node::Leaf(right)) = (&**left, right) else {
                unreachable!()
            };
            right.keys.insert_from(0, left.keys.iter().cloned());
        }
        branch.keys.remove(left_idx);
        branch.children.remove(left_idx + (keep_left as usize));
        debug_assert_eq!(branch.keys.len() + 1, branch.children.len());
    }

    /// Assuming `branch` is at level 1, rebalances two adjacent leaves so that they have the same
    /// number of keys (or differ by at most 1).
    fn rebalance_leaves(branch: &mut Branch<K, V, P>, left_idx: usize) {
        debug_assert_eq!(branch.level, 1);
        let [left, right, ..] = &mut branch.children[left_idx..] else {
            unreachable!()
        };
        let (Node::Leaf(left), Node::Leaf(right)) = (
            SharedPointer::make_mut(left),
            SharedPointer::make_mut(right),
        ) else {
            unreachable!()
        };
        let num_to_move = left.keys.len().abs_diff(right.keys.len()) / 2;
        if num_to_move == 0 {
            return;
        }
        if left.keys.len() > right.keys.len() {
            right.keys.drain_from_back(&mut left.keys, num_to_move);
        } else {
            left.keys.drain_from_front(&mut right.keys, num_to_move);
        }
        branch.keys[left_idx] = right.keys.first().unwrap().0.clone();
        debug_assert_ne!(left.keys.len(), 0);
        debug_assert_ne!(right.keys.len(), 0);
    }

    /// Rebalances two adjacent child branches so that they have the same number of keys
    /// (or differ by at most 1). The separator key is rotated between the two branches.
    /// to keep the invariants of the parent branch.
    fn rebalance_branches(branch: &mut Branch<K, V, P>, left_idx: usize) {
        let [left, right, ..] = &mut branch.children[left_idx..] else {
            unreachable!()
        };
        let (Node::Branch(left), Node::Branch(right)) = (
            SharedPointer::make_mut(left),
            SharedPointer::make_mut(right),
        ) else {
            unreachable!()
        };
        let num_to_move = left.keys.len().abs_diff(right.keys.len()) / 2;
        if num_to_move == 0 {
            return;
        }
        let separator = &mut branch.keys[left_idx];
        if left.keys.len() > right.keys.len() {
            right.keys.push_front(separator.clone());
            right.keys.drain_from_back(&mut left.keys, num_to_move - 1);
            *separator = left.keys.pop_back();
            right
                .children
                .drain_from_back(&mut left.children, num_to_move);
        } else {
            left.keys.push_back(separator.clone());
            left.keys.drain_from_front(&mut right.keys, num_to_move - 1);
            *separator = right.keys.pop_front();
            left.children
                .drain_from_front(&mut right.children, num_to_move);
        }
        debug_assert_ne!(left.keys.len(), 0);
        debug_assert_eq!(left.children.len(), left.keys.len() + 1);
        debug_assert_ne!(right.keys.len(), 0);
        debug_assert_eq!(right.children.len(), right.keys.len() + 1);
    }

    /// Merges two children of this branch.
    ///
    /// Assumes that the two children can fit in a single branch, panicking if not.
    fn merge_branches(branch: &mut Branch<K, V, P>, left_idx: usize, keep_left: bool) {
        debug_assert!(branch.level >= 2);
        let [left, right, ..] = &mut branch.children[left_idx..] else {
            unreachable!()
        };
        let separator = branch.keys.remove(left_idx);
        if keep_left {
            let left = SharedPointer::make_mut(left);
            let (Node::Branch(left), Node::Branch(right)) = (left, &**right) else {
                unreachable!()
            };
            left.keys.push_back(separator);
            left.keys.extend(right.keys.iter().cloned());
            left.children.extend(right.children.iter().cloned());
        } else {
            let right = SharedPointer::make_mut(right);
            let (Node::Branch(left), Node::Branch(right)) = (&**left, right) else {
                unreachable!()
            };
            right.keys.push_front(separator);
            right.keys.insert_from(0, left.keys.iter().cloned());
            right.children.insert_from(0, left.children.iter().cloned());
        }
        branch.children.remove(left_idx + (keep_left as usize));
        debug_assert_eq!(branch.keys.len() + 1, branch.children.len());
    }

    pub(crate) fn insert(&mut self, key: K, value: V) -> InsertAction<K, V, P> {
        match self {
            Node::Branch(branch) => {
                let i = branch
                    .keys
                    .binary_search(&key)
                    .map(|x| x + 1)
                    .unwrap_or_else(|x| x);
                match SharedPointer::make_mut(&mut branch.children[i]).insert(key, value) {
                    InsertAction::Split(new_key, new_node) if branch.keys.len() >= NODE_SIZE => {
                        Self::split_branch_insert(branch, i, new_key, new_node)
                    }
                    InsertAction::Split(separator, new_node) => {
                        branch.keys.insert(i, separator);
                        branch.children.insert(i + 1, new_node);
                        InsertAction::Inserted
                    }
                    action => action,
                }
            }
            Node::Leaf(leaf) => match leaf.keys.binary_search_by(|(k, _)| k.cmp(&key)) {
                Ok(i) => {
                    let (k, v) = mem::replace(&mut leaf.keys[i], (key, value));
                    InsertAction::Replaced(k, v)
                }
                Err(i) if leaf.keys.len() >= NODE_SIZE => {
                    Self::split_leaf_insert(leaf, i, key, value)
                }
                Err(i) => {
                    leaf.keys.insert(i, (key, value));
                    InsertAction::Inserted
                }
            },
        }
    }

    #[cold]
    fn split_branch_insert(
        branch: &mut Branch<K, V, P>,
        i: usize,
        new_key: K,
        new_node: SharedPointer<Node<K, V, P>, P>,
    ) -> InsertAction<K, V, P> {
        let split_idx = MEDIAN + (i > MEDIAN) as usize;
        let mut right_keys = branch.keys.split_off(split_idx);
        let split_idx = MEDIAN + (i >= MEDIAN) as usize;
        let mut right_children = branch.children.split_off(split_idx);
        let separator = if i == MEDIAN {
            right_children.push_front(new_node.clone());
            new_key
        } else {
            if i < MEDIAN {
                branch.keys.insert(i, new_key);
                branch.children.insert(i + 1, new_node);
            } else {
                right_keys.insert(i - (MEDIAN + 1), new_key);
                right_children.insert(i - (MEDIAN + 1) + 1, new_node);
            }
            branch.keys.pop_back()
        };
        debug_assert_eq!(branch.keys.len(), right_keys.len());
        debug_assert_eq!(branch.keys.len() + 1, branch.children.len());
        debug_assert_eq!(right_keys.len() + 1, right_children.len());
        InsertAction::Split(
            separator,
            SharedPointer::new(Node::Branch(Branch {
                level: branch.level,
                keys: right_keys,
                children: right_children,
            })),
        )
    }

    #[inline]
    fn split_leaf_insert(
        leaf: &mut Leaf<K, V>,
        i: usize,
        key: K,
        value: V,
    ) -> InsertAction<K, V, P> {
        let mut right_keys = leaf.keys.split_off(MEDIAN);
        if i < MEDIAN {
            leaf.keys.insert(i, (key, value));
        } else {
            right_keys.insert(i - MEDIAN, (key, value));
        }
        InsertAction::Split(
            right_keys.first().unwrap().0.clone(),
            SharedPointer::new(Node::Leaf(Leaf { keys: right_keys })),
        )
    }

    pub(crate) fn lookup_mut<BK>(&mut self, key: &BK) -> Option<(&K, &mut V)>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        match self {
            Node::Branch(branch) => {
                let i = branch
                    .keys
                    .binary_search_by(|k| k.borrow().cmp(key))
                    .map(|x| x + 1)
                    .unwrap_or_else(|x| x);
                SharedPointer::make_mut(&mut branch.children[i]).lookup_mut(key)
            }
            Node::Leaf(leaf) => {
                let keys = &mut leaf.keys;
                let i = keys.binary_search_by(|(k, _)| k.borrow().cmp(key)).ok()?;
                keys.get_mut(i).map(|(k, v)| (&*k, v))
            }
        }
    }

    pub(crate) fn new_from_split(
        left: SharedPointer<Self, P>,
        separator: K,
        right: SharedPointer<Self, P>,
    ) -> Self {
        Node::Branch(Branch {
            level: left.level() + 1,
            keys: Chunk::unit(separator),
            children: Chunk::from_iter([left, right]),
        })
    }
}

impl<K: Ord, V, P: SharedPointerKind> Node<K, V, P> {
    pub(crate) fn min(&self) -> Option<&(K, V)> {
        let mut node = self;
        loop {
            node = match node {
                Node::Branch(branch) => &branch.children[0],
                Node::Leaf(leaf) => return leaf.keys.first(),
            };
        }
    }

    pub(crate) fn max(&self) -> Option<&(K, V)> {
        let mut node = self;
        loop {
            node = match node {
                Node::Branch(branch) => &branch.children[branch.children.len() - 1],
                Node::Leaf(leaf) => return leaf.keys.last(),
            };
        }
    }

    pub(crate) fn lookup<BK>(&self, key: &BK) -> Option<&(K, V)>
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        let mut node = self;
        loop {
            match node {
                Node::Branch(branch) => {
                    let i = branch
                        .keys
                        .binary_search_by(|k| k.borrow().cmp(key))
                        .map(|x| x + 1)
                        .unwrap_or_else(|x| x);
                    node = &branch.children[i];
                }
                Node::Leaf(leaf) => {
                    let keys = &leaf.keys;
                    let i = keys.binary_search_by(|(k, _)| k.borrow().cmp(key)).ok()?;
                    return keys.get(i);
                }
            }
        }
    }
}

impl<K: Clone, V: Clone> Clone for Leaf<K, V> {
    fn clone(&self) -> Self {
        Self {
            keys: self.keys.clone(),
        }
    }
}

impl<K: Clone, V: Clone, P: SharedPointerKind> Clone for Branch<K, V, P> {
    fn clone(&self) -> Self {
        Self {
            keys: self.keys.clone(),
            children: self.children.clone(),
            level: self.level,
        }
    }
}

impl<K: Clone, V: Clone, P: SharedPointerKind> Clone for Node<K, V, P> {
    fn clone(&self) -> Self {
        match self {
            Node::Branch(branch) => Node::Branch(branch.clone()),
            Node::Leaf(leaf) => Node::Leaf(leaf.clone()),
        }
    }
}

pub(crate) enum InsertAction<K, V, P: SharedPointerKind> {
    Inserted,
    Replaced(K, V),
    Split(K, SharedPointer<Node<K, V, P>, P>),
}

impl<K, V, P: SharedPointerKind> Default for Node<K, V, P> {
    fn default() -> Self {
        Node::Leaf(Leaf { keys: Chunk::new() })
    }
}

#[derive(Debug)]
pub(crate) struct ConsumingIter<K, V, P: SharedPointerKind> {
    /// The leaves of the tree, in order, note that this will remain the shared ptr
    /// as it will allows us to have a smaller VecDeque allocation and avoid eagerly
    /// cloning the leaves, which defeats the purpose of this iterator.
    /// Leaves present in the VecDeque are guaranteed to be non-empty.
    leaves: VecDeque<SharedPointer<Node<K, V, P>, P>>,
    remaining: usize,
}

impl<K, V, P: SharedPointerKind> ConsumingIter<K, V, P> {
    pub(crate) fn new(node: Option<SharedPointer<Node<K, V, P>, P>>, size: usize) -> Self {
        fn push<K, V, P: SharedPointerKind>(
            leaves: &mut VecDeque<SharedPointer<Node<K, V, P>, P>>,
            node: SharedPointer<Node<K, V, P>, P>,
        ) {
            match &*node {
                Node::Branch(branch) => {
                    if branch.level == 1 {
                        leaves.extend(branch.children.iter().cloned());
                    } else {
                        for child in branch.children.iter() {
                            push(leaves, child.clone());
                        }
                    }
                }
                Node::Leaf(leaf) if !leaf.keys.is_empty() => leaves.push_back(node),
                Node::Leaf(_) => (),
            }
        }
        // preallocate the VecDeque assuming each leaf is half full
        let mut leaves = VecDeque::with_capacity(size.div_ceil(NODE_SIZE / 2));
        if let Some(node) = node {
            push(&mut leaves, node);
        }
        Self {
            leaves,
            remaining: size,
        }
    }
}

impl<K: Clone, V: Clone, P: SharedPointerKind> Iterator for ConsumingIter<K, V, P> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.leaves.front_mut()?;
        let Node::Leaf(leaf) = SharedPointer::make_mut(node) else {
            unreachable!()
        };
        self.remaining -= 1;
        let item = leaf.keys.pop_front();
        if leaf.keys.is_empty() {
            self.leaves.pop_front();
        }
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K: Clone, V: Clone, P: SharedPointerKind> DoubleEndedIterator for ConsumingIter<K, V, P> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let node = self.leaves.back_mut()?;
        let Node::Leaf(leaf) = SharedPointer::make_mut(node) else {
            unreachable!()
        };
        self.remaining -= 1;
        let item = leaf.keys.pop_back();
        if leaf.keys.is_empty() {
            self.leaves.pop_back();
        }
        Some(item)
    }
}

#[derive(Debug)]
pub(crate) struct Iter<'a, K, V, P: SharedPointerKind> {
    /// The forward and backward cursors
    /// The cursors are lazily initialized if their corresponding bound is unbounded
    fwd: Cursor<'a, K, V, P>,
    bwd: Cursor<'a, K, V, P>,
    fwd_yielded: bool,
    bwd_yielded: bool,
    exhausted: bool,
    exact: bool,
    remaining: usize,
    root: Option<&'a Node<K, V, P>>,
}

impl<'a, K, V, P: SharedPointerKind> Iter<'a, K, V, P> {
    pub(crate) fn new<R, BK>(root: Option<&'a Node<K, V, P>>, len: usize, range: R) -> Self
    where
        R: RangeBounds<BK>,
        K: Borrow<BK>,
        BK: Ord + ?Sized,
    {
        let mut fwd = Cursor::empty();
        let mut bwd = Cursor::empty();
        let mut exhausted = match range.start_bound() {
            Bound::Included(key) | Bound::Excluded(key) => {
                fwd.init(root);
                if fwd.seek_to_key(key, false) && matches!(range.start_bound(), Bound::Excluded(_))
                {
                    fwd.next().is_none()
                } else {
                    fwd.stack.is_empty()
                }
            }
            Bound::Unbounded => false,
        };

        exhausted = match (exhausted, range.end_bound()) {
            (false, Bound::Included(key) | Bound::Excluded(key)) => {
                bwd.init(root);
                if bwd.seek_to_key(key, true) && matches!(range.end_bound(), Bound::Excluded(_)) {
                    bwd.prev().is_none()
                } else {
                    bwd.stack.is_empty()
                }
            }
            (exhausted, _) => exhausted,
        };

        // Check if forward is > backward cursor to determine if we are exhausted
        // Due to the usage of zip this is correct even if the cursors are already or not initialized yet
        for (&(fi, f), &(bi, b)) in fwd.stack.iter().zip(bwd.stack.iter()) {
            if !std::ptr::addr_eq(f, b) {
                break;
            }
            if fi > bi {
                exhausted = true;
                break;
            }
        }

        let exact = matches!(range.start_bound(), Bound::Unbounded)
            && matches!(range.end_bound(), Bound::Unbounded);

        Self {
            fwd,
            bwd,
            remaining: len,
            exact,
            exhausted,
            fwd_yielded: false,
            bwd_yielded: false,
            root,
        }
    }

    /// Updates the exhausted state of the iterator.
    /// Returns true if the iterator is immaterially exhausted, which implies ignoring the
    /// current next candidate, if any.
    #[inline]
    fn update_exhausted(&mut self, has_next: bool, other_side_yielded: bool) -> bool {
        debug_assert!(!self.exhausted);
        if !has_next {
            self.exhausted = true;
            return true;
        }
        // Check if the cursors are exhausted by checking their leaves
        // This is valid even if the cursors are empty due to not being initialized yet.
        // If they were empty because exhaustion we would not be in this function.
        if let (Some(&(fi, f)), Some(&(bi, b))) = (self.fwd.stack.last(), self.bwd.stack.last()) {
            if std::ptr::addr_eq(f, b) && fi >= bi {
                self.exhausted = true;
                return fi == bi && other_side_yielded;
            }
        }
        false
    }

    #[cold]
    fn peek_initial(&mut self, fwd: bool) -> Option<&'a (K, V)> {
        debug_assert!(!self.exhausted);
        let cursor = if fwd {
            self.fwd_yielded = true;
            &mut self.fwd
        } else {
            self.bwd_yielded = true;
            &mut self.bwd
        };
        // If the cursor is empty we need to initialize it and seek to the first/last element.
        // If they were empty because exhaustion we would not be in this function.
        if cursor.stack.is_empty() {
            cursor.init(self.root);
            if fwd {
                cursor.seek_to_first();
            } else {
                cursor.seek_to_last();
            }
        }
        cursor.peek()
    }
}

impl<'a, K, V, P: SharedPointerKind> Iterator for Iter<'a, K, V, P> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        let next = if self.fwd_yielded {
            self.fwd.next()
        } else {
            self.peek_initial(true)
        }
        .map(|(k, v)| (k, v));
        if self.update_exhausted(next.is_some(), self.bwd_yielded) {
            return None;
        }
        self.remaining -= 1;
        next
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.exhausted {
            return (0, Some(0));
        }
        let lb = if self.exact { self.remaining } else { 0 };
        (lb, Some(self.remaining))
    }
}

impl<'a, K, V, P: SharedPointerKind> DoubleEndedIterator for Iter<'a, K, V, P> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        let next = if self.bwd_yielded {
            self.bwd.prev()
        } else {
            self.peek_initial(false)
        }
        .map(|(k, v)| (k, v));
        if self.update_exhausted(next.is_some(), self.fwd_yielded) {
            return None;
        }
        self.remaining -= 1;
        next
    }
}

impl<'a, K, V, P: SharedPointerKind> Clone for Iter<'a, K, V, P> {
    fn clone(&self) -> Self {
        Self {
            fwd: self.fwd.clone(),
            bwd: self.bwd.clone(),
            exact: self.exact,
            fwd_yielded: self.fwd_yielded,
            bwd_yielded: self.bwd_yielded,
            exhausted: self.exhausted,
            remaining: self.remaining,
            root: self.root,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Cursor<'a, K, V, P: SharedPointerKind> {
    stack: Vec<(usize, &'a Node<K, V, P>)>,
}

impl<'a, K, V, P: SharedPointerKind> Clone for Cursor<'a, K, V, P> {
    fn clone(&self) -> Self {
        Self {
            stack: self.stack.clone(),
        }
    }
}

impl<'a, K, V, P: SharedPointerKind> Cursor<'a, K, V, P> {
    /// Creates a new empty cursor.
    /// The variety of methods is to allow for a more efficient initialization
    /// in all cases.
    pub(crate) fn empty() -> Self {
        Self { stack: Vec::new() }
    }

    pub(crate) fn init(&mut self, node: Option<&'a Node<K, V, P>>) {
        if let Some(node) = node {
            self.stack.reserve_exact(node.level() + 1);
            self.stack.push((0, node));
        }
    }

    pub(crate) fn seek_to_first(&mut self) -> Option<&'a (K, V)> {
        while let Some((i, node)) = self.stack.last_mut() {
            debug_assert_eq!(i, &0);
            match node {
                Node::Branch(branch) => {
                    self.stack.push((0, &branch.children[0]));
                }
                Node::Leaf(leaf) => return leaf.keys.first(),
            }
        }
        None
    }

    fn seek_to_last(&mut self) -> Option<&'a (K, V)> {
        while let Some((i, node)) = self.stack.last_mut() {
            debug_assert_eq!(i, &0);
            match node {
                Node::Branch(branch) => {
                    *i = branch.children.len() - 1;
                    let child = &branch.children[*i];
                    self.stack.push((0, child));
                }
                Node::Leaf(leaf) => {
                    *i = leaf.keys.len().saturating_sub(1);
                    return leaf.keys.last();
                }
            }
        }
        None
    }

    fn seek_to_key<BK>(&mut self, key: &BK, for_prev: bool) -> bool
    where
        BK: Ord + ?Sized,
        K: Borrow<BK>,
    {
        while let Some((i, node)) = self.stack.last_mut() {
            match node {
                Node::Branch(branch) => {
                    *i = branch
                        .keys
                        .binary_search_by(|k| k.borrow().cmp(key))
                        .map(|x| x + 1)
                        .unwrap_or_else(|x| x);
                    let child = &branch.children[*i];
                    self.stack.push((0, child));
                }
                Node::Leaf(leaf) => {
                    let search = leaf.keys.binary_search_by(|(k, _)| k.borrow().cmp(key));
                    *i = search.unwrap_or_else(|x| x);
                    if for_prev {
                        if search.is_err() {
                            self.prev();
                        }
                    } else if search == Err(leaf.keys.len()) {
                        self.next();
                    }
                    return search.is_ok();
                }
            }
        }
        false
    }

    /// Advances this and another cursor to their next position.
    /// While doing so skip all shared nodes between them.
    pub(crate) fn advance_skipping_shared<'b>(&mut self, other: &mut Cursor<'b, K, V, P>) {
        // The current implementation is not optimal as it will still visit many nodes unnecessarily
        // before skipping them. But it requires very little additional code.
        // Nevertheless it will still improve performance when there are shared nodes.
        loop {
            let shared_levels = self
                .stack
                .iter()
                .rev()
                .zip(other.stack.iter().rev())
                .take_while(|(this, that)| std::ptr::addr_eq(this.1, that.1))
                .count();
            if shared_levels != 0 {
                self.stack.drain(self.stack.len() - shared_levels..);
                other.stack.drain(other.stack.len() - shared_levels..);
            }
            self.next();
            other.next();
            if shared_levels == 0 {
                break;
            }
        }
    }

    pub(crate) fn next(&mut self) -> Option<&'a (K, V)> {
        while let Some((i, node)) = self.stack.last_mut() {
            match node {
                Node::Branch(branch) => {
                    if *i + 1 < branch.children.len() {
                        *i += 1;
                        let child = &branch.children[*i];
                        self.stack.push((0, child));
                        break;
                    }
                }
                Node::Leaf(leaf) => {
                    if *i + 1 < leaf.keys.len() {
                        *i += 1;
                        return leaf.keys.get(*i);
                    }
                }
            }
            self.stack.pop();
        }
        self.seek_to_first()
    }

    fn prev(&mut self) -> Option<&'a (K, V)> {
        while let Some((i, node)) = self.stack.last_mut() {
            match node {
                Node::Branch(branch) => {
                    if *i > 0 {
                        *i -= 1;
                        let child = &branch.children[*i];
                        self.stack.push((0, child));
                        break;
                    }
                }
                Node::Leaf(leaf) => {
                    if *i > 0 {
                        *i -= 1;
                        return leaf.keys.get(*i);
                    }
                }
            }
            self.stack.pop();
        }
        self.seek_to_last()
    }

    pub(crate) fn peek(&self) -> Option<&'a (K, V)> {
        if let Some((i, Node::Leaf(leaf))) = self.stack.last() {
            leaf.keys.get(*i)
        } else {
            None
        }
    }
}
