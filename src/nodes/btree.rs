// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::collections::VecDeque;
use std::iter::FromIterator;
use std::mem;
use std::num::NonZeroUsize;
use std::ops::{Bound, RangeBounds};

use archery::{SharedPointer, SharedPointerKind};
use equivalent::Comparable;
use imbl_sized_chunks::Chunk;

pub(crate) use crate::config::ORD_CHUNK_SIZE as NODE_SIZE;

const MEDIAN: usize = NODE_SIZE / 2;
const THIRD: usize = NODE_SIZE / 3;
const NUM_CHILDREN: usize = NODE_SIZE + 1;

/// A node in a `B+Tree`.
///
/// The main tree representation uses [`Branch`] and [`Leaf`]; this is only used
/// in places that want to handle either a branch or a leaf.
#[derive(Debug)]
pub(crate) enum Node<K, V, P: SharedPointerKind> {
    Branch(SharedPointer<Branch<K, V, P>, P>),
    Leaf(SharedPointer<Leaf<K, V>, P>),
}

impl<K: Ord + std::fmt::Debug, V: std::fmt::Debug, P: SharedPointerKind> Branch<K, V, P> {
    #[cfg(any(test, fuzzing))]
    pub(crate) fn check_sane(&self, is_root: bool) -> usize {
        assert!(self.keys.len() >= if is_root { 1 } else { MEDIAN - 1 });
        assert_eq!(self.keys.len() + 1, self.children.len());
        assert!(self.keys.windows(2).all(|w| w[0] < w[1]));
        match &self.children {
            Children::Leaves { leaves } => {
                for i in 0..self.keys.len() {
                    let left = &leaves[i];
                    let right = &leaves[i + 1];
                    assert!(left.keys.last().unwrap().0 < right.keys.first().unwrap().0);
                }
                leaves.iter().map(|child| child.check_sane(false)).sum()
            }
            Children::Branches { branches, level } => {
                for i in 0..self.keys.len() {
                    let left = &branches[i];
                    let right = &branches[i + 1];
                    assert!(left.level() == level.get() - 1);
                    assert!(right.level() == level.get() - 1);
                }
                branches.iter().map(|child| child.check_sane(false)).sum()
            }
        }
    }
}
impl<K: Ord + std::fmt::Debug, V: std::fmt::Debug> Leaf<K, V> {
    #[cfg(any(test, fuzzing))]
    pub(crate) fn check_sane(&self, is_root: bool) -> usize {
        assert!(self.keys.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(self.keys.len() >= if is_root { 0 } else { THIRD });
        self.keys.len()
    }
}
impl<K: Ord + std::fmt::Debug, V: std::fmt::Debug, P: SharedPointerKind> Node<K, V, P> {
    /// Check invariants
    #[cfg(any(test, fuzzing))]
    pub(crate) fn check_sane(&self, is_root: bool) -> usize {
        match self {
            Node::Branch(branch) => branch.check_sane(is_root),
            Node::Leaf(leaf) => leaf.check_sane(is_root),
        }
    }
}

impl<K, V, P: SharedPointerKind> Node<K, V, P> {
    pub(crate) fn unit(key: K, value: V) -> Self {
        Node::Leaf(SharedPointer::new(Leaf {
            keys: Chunk::unit((key, value)),
        }))
    }

    pub(crate) fn level(&self) -> usize {
        match self {
            Node::Branch(branch) => branch.level(),
            Node::Leaf(_) => 0,
        }
    }

    /// The address of each node of this tree, this one included, and
    /// whether it is a leaf (for the tests of the work of a diff).
    #[cfg(test)]
    pub(crate) fn addresses(&self, out: &mut Vec<(*const (), bool)>) {
        fn branch<K, V, P: SharedPointerKind>(
            b: &Branch<K, V, P>,
            out: &mut Vec<(*const (), bool)>,
        ) {
            out.push((std::ptr::from_ref(b).cast(), false));
            match &b.children {
                Children::Leaves { leaves } => out.extend(
                    leaves
                        .iter()
                        .map(|l| (std::ptr::from_ref(&**l).cast(), true)),
                ),
                Children::Branches { branches, .. } => {
                    for c in branches.iter() {
                        branch(c, out);
                    }
                }
            }
        }
        match self {
            Node::Branch(b) => branch(b, out),
            Node::Leaf(l) => out.push((std::ptr::from_ref(&**l).cast(), true)),
        }
    }

    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Node::Branch(a), Node::Branch(b)) => SharedPointer::ptr_eq(a, b),
            (Node::Leaf(a), Node::Leaf(b)) => SharedPointer::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// A branch node in a `B+Tree`.
/// Invariants:
/// * keys are ordered and unique
/// * keys.len() + 1 == children.len()
/// * all children have level = level - 1 (or level is 1 and all children are leaves)
/// * all keys in the subtree at children[i] are between keys[i - 1] (if i > 0) and keys[i] (if i < keys.len()).
/// * root branch must have at least 1 key, whereas non-root branches must have at least MEDIAN - 1 keys
#[derive(Debug)]
pub(crate) struct Branch<K, V, P: SharedPointerKind> {
    keys: Chunk<K, NODE_SIZE>,
    children: Children<K, V, P>,
}

#[derive(Debug)]
pub(crate) enum Children<K, V, P: SharedPointerKind> {
    /// implicitly level 1
    Leaves {
        leaves: Chunk<SharedPointer<Leaf<K, V>, P>, NUM_CHILDREN>,
    },
    /// level >= 2
    Branches {
        branches: Chunk<SharedPointer<Branch<K, V, P>, P>, NUM_CHILDREN>,
        /// The level of the tree node that contains these children.
        ///
        /// Leaves have level zero, so branches have level at least one. Since this is the
        /// level of something containing branches, it is at least two.
        level: NonZeroUsize,
    },
}

impl<K, V, P: SharedPointerKind> Children<K, V, P> {
    fn len(&self) -> usize {
        match self {
            Children::Leaves { leaves } => leaves.len(),
            Children::Branches { branches, .. } => branches.len(),
        }
    }
    fn drain_from_front(&mut self, other: &mut Self, count: usize) {
        match (self, other) {
            (
                Children::Leaves { leaves },
                Children::Leaves {
                    leaves: other_leaves,
                },
            ) => leaves.drain_from_front(other_leaves, count),
            (
                Children::Branches { branches, .. },
                Children::Branches {
                    branches: other_branches,
                    ..
                },
            ) => branches.drain_from_front(other_branches, count),
            _ => panic!("mismatched drain_from_front"),
        }
    }
    fn drain_from_back(&mut self, other: &mut Self, count: usize) {
        match (self, other) {
            (
                Children::Leaves { leaves },
                Children::Leaves {
                    leaves: other_leaves,
                },
            ) => leaves.drain_from_back(other_leaves, count),
            (
                Children::Branches { branches, .. },
                Children::Branches {
                    branches: other_branches,
                    ..
                },
            ) => branches.drain_from_back(other_branches, count),
            _ => panic!("mismatched drain_from_back"),
        }
    }
    fn extend(&mut self, other: &Self) {
        match (self, other) {
            (
                Children::Leaves { leaves },
                Children::Leaves {
                    leaves: other_leaves,
                },
            ) => leaves.extend(other_leaves.iter().cloned()),
            (
                Children::Branches { branches, .. },
                Children::Branches {
                    branches: other_branches,
                    ..
                },
            ) => branches.extend(other_branches.iter().cloned()),
            _ => panic!("mismatched extend"),
        }
    }
    fn insert_front(&mut self, other: &Self) {
        match (self, other) {
            (
                Children::Leaves { leaves },
                Children::Leaves {
                    leaves: other_leaves,
                },
            ) => leaves.insert_from(0, other_leaves.iter().cloned()),
            (
                Children::Branches { branches, .. },
                Children::Branches {
                    branches: other_branches,
                    ..
                },
            ) => branches.insert_from(0, other_branches.iter().cloned()),
            _ => panic!("mismatched insert_front"),
        }
    }
    fn insert(&mut self, index: usize, node: Node<K, V, P>) {
        match (self, node) {
            (Children::Leaves { leaves }, Node::Leaf(node)) => leaves.insert(index, node),
            (Children::Branches { branches, .. }, Node::Branch(node)) => {
                branches.insert(index, node)
            }
            _ => panic!("mismatched insert"),
        }
    }
    fn split_off(&mut self, at: usize) -> Self {
        match self {
            Children::Leaves { leaves } => Children::Leaves {
                leaves: leaves.split_off(at),
            },
            Children::Branches { branches, level } => Children::Branches {
                branches: branches.split_off(at),
                level: *level,
            },
        }
    }
}

impl<K, V, P: SharedPointerKind> Branch<K, V, P> {
    pub(crate) fn pop_single_child(&mut self) -> Option<Node<K, V, P>> {
        if self.children.len() == 1 {
            debug_assert_eq!(self.keys.len(), 0);
            Some(match &mut self.children {
                Children::Leaves { leaves } => Node::Leaf(leaves.pop_back()),
                Children::Branches { branches, .. } => Node::Branch(branches.pop_back()),
            })
        } else {
            None
        }
    }

    fn level(&self) -> usize {
        match &self.children {
            Children::Leaves { .. } => 1,
            Children::Branches { level, .. } => level.get(),
        }
    }
}

/// A leaf node in a `B+Tree`.
///
/// Invariants:
/// * keys are ordered and unique
/// * leaf is the lowest level in the tree (level 0)
/// * non-root leaves must have at least THIRD keys
#[derive(Debug)]
pub(crate) struct Leaf<K, V> {
    keys: Chunk<(K, V), NODE_SIZE>,
}

impl<K: Ord + Clone, V: Clone, P: SharedPointerKind> Node<K, V, P> {
    /// Removes a key from the node or its children.
    /// Returns `true` if the node is underflowed and should be rebalanced.
    pub(crate) fn remove<Q>(&mut self, key: &Q, removed: &mut Option<(K, V)>) -> bool
    where
        Q: Comparable<K> + ?Sized,
    {
        match self {
            Node::Branch(branch) => SharedPointer::make_mut(branch).remove(key, removed),
            Node::Leaf(leaf) => SharedPointer::make_mut(leaf).remove(key, removed),
        }
    }
}

impl<K: Ord + Clone, V: Clone, P: SharedPointerKind> Branch<K, V, P> {
    pub(crate) fn remove<Q>(&mut self, key: &Q, removed: &mut Option<(K, V)>) -> bool
    where
        Q: Comparable<K> + ?Sized,
    {
        let i = slice_ext::binary_search_by(&self.keys, |k| key.compare(k).reverse())
            .map(|x| x + 1)
            .unwrap_or_else(|x| x);
        let rebalance = match &mut self.children {
            Children::Leaves { leaves } => {
                SharedPointer::make_mut(&mut leaves[i]).remove(key, removed)
            }
            Children::Branches { branches, .. } => {
                SharedPointer::make_mut(&mut branches[i]).remove(key, removed)
            }
        };
        if rebalance {
            self.branch_rebalance_children(i);
        }
        // Underflow if the branch is < 1/2 full. Since the branches are relatively
        // rarely rebalanced (given relaxed leaf underflow), we can afford to be
        // a bit more conservative here.
        self.keys.len() < MEDIAN
    }
}

impl<K: Ord + Clone, V: Clone> Leaf<K, V> {
    pub(crate) fn remove<Q>(&mut self, key: &Q, removed: &mut Option<(K, V)>) -> bool
    where
        Q: Comparable<K> + ?Sized,
    {
        if let Ok(i) = slice_ext::binary_search_by(&self.keys, |(k, _)| key.compare(k).reverse()) {
            *removed = Some(self.keys.remove(i));
        }
        // Underflow if the leaf is < 1/3 full. This relaxed underflow (vs. 1/2 full) is
        // useful to prevent degenerate cases where a random insert/remove workload will
        // constantly merge/split a leaf.
        self.keys.len() < THIRD
    }
}

impl<K: Ord + Clone, V: Clone, P: SharedPointerKind> Branch<K, V, P> {
    #[cold]
    pub(crate) fn branch_rebalance_children(&mut self, underflow_idx: usize) {
        let left_idx = underflow_idx.saturating_sub(1);
        match &mut self.children {
            Children::Leaves { leaves } => {
                let (left, mid, right) = match &leaves[left_idx..] {
                    [left, mid, right, ..] => (&**left, &**mid, Some(&**right)),
                    [left, mid, ..] => (&**left, &**mid, None),
                    _ => return,
                };
                // Prefer merging two sibling children if we can fit them into a single node.
                // But also try to rebalance if the smallest child is small (< 1/3), to amortize the cost of rebalancing.
                // Since we prefer merging, for rebalancing to apply the the largest child will be least 2/3 full,
                // which results in two at least half full nodes after rebalancing.
                match (left, mid, right) {
                    (left, mid, _) if left.keys.len() + mid.keys.len() <= NODE_SIZE => {
                        Self::merge_leaves(leaves, &mut self.keys, left_idx, false);
                    }
                    (_, mid, Some(right)) if mid.keys.len() + right.keys.len() <= NODE_SIZE => {
                        Self::merge_leaves(leaves, &mut self.keys, left_idx + 1, true);
                    }
                    (left, mid, _) if mid.keys.len().min(left.keys.len()) < THIRD => {
                        Self::rebalance_leaves(leaves, &mut self.keys, left_idx);
                    }
                    (_, mid, Some(right)) if mid.keys.len().min(right.keys.len()) < THIRD => {
                        Self::rebalance_leaves(leaves, &mut self.keys, left_idx + 1);
                    }
                    _ => (),
                }
            }
            Children::Branches { branches, .. } => {
                let (left, mid, right) = match &branches[left_idx..] {
                    [left, mid, right, ..] => (&**left, &**mid, Some(&**right)),
                    [left, mid, ..] => (&**left, &**mid, None),
                    _ => return,
                };
                match (left, mid, right) {
                    (left, mid, _) if left.keys.len() + mid.keys.len() < NODE_SIZE => {
                        Self::merge_branches(branches, &mut self.keys, left_idx, false);
                    }
                    (_, mid, Some(right)) if mid.keys.len() + right.keys.len() < NODE_SIZE => {
                        Self::merge_branches(branches, &mut self.keys, left_idx + 1, true);
                    }
                    (left, mid, _) if mid.keys.len().min(left.keys.len()) < THIRD => {
                        Self::rebalance_branches(branches, &mut self.keys, left_idx);
                    }
                    (_, mid, Some(right)) if mid.keys.len().min(right.keys.len()) < THIRD => {
                        Self::rebalance_branches(branches, &mut self.keys, left_idx + 1);
                    }
                    _ => (),
                }
            }
        }
    }

    /// Merges two children leaves of this branch.
    ///
    /// Assumes that the two children can fit in a single leaf, panicking if not.
    fn merge_leaves(
        children: &mut Chunk<SharedPointer<Leaf<K, V>, P>, NUM_CHILDREN>,
        keys: &mut Chunk<K, NODE_SIZE>,
        left_idx: usize,
        keep_left: bool,
    ) {
        let [left, right, ..] = &mut children[left_idx..] else {
            unreachable!()
        };
        if keep_left {
            let left = SharedPointer::make_mut(left);
            let (left, right) = (left, &**right);
            left.keys.extend(right.keys.iter().cloned());
        } else {
            let right = SharedPointer::make_mut(right);
            let (left, right) = (&**left, right);
            right.keys.insert_from(0, left.keys.iter().cloned());
        }
        keys.remove(left_idx);
        children.remove(left_idx + (keep_left as usize));
        debug_assert_eq!(keys.len() + 1, children.len());
    }

    /// Rebalances two adjacent leaves so that they have the same
    /// number of keys (or differ by at most 1).
    fn rebalance_leaves(
        children: &mut Chunk<SharedPointer<Leaf<K, V>, P>, NUM_CHILDREN>,
        keys: &mut Chunk<K, NODE_SIZE>,
        left_idx: usize,
    ) {
        let [left, right, ..] = &mut children[left_idx..] else {
            unreachable!()
        };
        let (left, right) = (
            SharedPointer::make_mut(left),
            SharedPointer::make_mut(right),
        );
        let num_to_move = left.keys.len().abs_diff(right.keys.len()) / 2;
        if num_to_move == 0 {
            return;
        }
        if left.keys.len() > right.keys.len() {
            right.keys.drain_from_back(&mut left.keys, num_to_move);
        } else {
            left.keys.drain_from_front(&mut right.keys, num_to_move);
        }
        keys[left_idx] = right.keys.first().unwrap().0.clone();
        debug_assert_ne!(left.keys.len(), 0);
        debug_assert_ne!(right.keys.len(), 0);
    }

    /// Rebalances two adjacent child branches so that they have the same number of keys
    /// (or differ by at most 1). The separator key is rotated between the two branches.
    /// to keep the invariants of the parent branch.
    fn rebalance_branches(
        children: &mut Chunk<SharedPointer<Branch<K, V, P>, P>, NUM_CHILDREN>,
        keys: &mut Chunk<K, NODE_SIZE>,
        left_idx: usize,
    ) {
        let [left, right, ..] = &mut children[left_idx..] else {
            unreachable!()
        };
        let (left, right) = (
            SharedPointer::make_mut(left),
            SharedPointer::make_mut(right),
        );
        let num_to_move = left.keys.len().abs_diff(right.keys.len()) / 2;
        if num_to_move == 0 {
            return;
        }
        let separator = &mut keys[left_idx];
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
    fn merge_branches(
        children: &mut Chunk<SharedPointer<Branch<K, V, P>, P>, NUM_CHILDREN>,
        keys: &mut Chunk<K, NODE_SIZE>,
        left_idx: usize,
        keep_left: bool,
    ) {
        let [left, right, ..] = &mut children[left_idx..] else {
            unreachable!()
        };
        let separator = keys.remove(left_idx);
        if keep_left {
            let left = SharedPointer::make_mut(left);
            let (left, right) = (left, &**right);
            left.keys.push_back(separator);
            left.keys.extend(right.keys.iter().cloned());
            left.children.extend(&right.children);
        } else {
            let right = SharedPointer::make_mut(right);
            let (left, right) = (&**left, right);
            right.keys.push_front(separator);
            right.keys.insert_from(0, left.keys.iter().cloned());
            right.children.insert_front(&left.children);
        }
        children.remove(left_idx + (keep_left as usize));
        debug_assert_eq!(keys.len() + 1, children.len());
    }
}

impl<K: Ord + Clone, V: Clone, P: SharedPointerKind> Branch<K, V, P> {
    pub(crate) fn insert(&mut self, key: K, value: V) -> InsertAction<K, V, P> {
        let i = slice_ext::binary_search_by(&self.keys, |k| k.cmp(&key))
            .map(|x| x + 1)
            .unwrap_or_else(|x| x);
        let insert_action = match &mut self.children {
            Children::Leaves { leaves } => {
                SharedPointer::make_mut(&mut leaves[i]).insert(key, value)
            }
            Children::Branches { branches, .. } => {
                SharedPointer::make_mut(&mut branches[i]).insert(key, value)
            }
        };
        match insert_action {
            InsertAction::Split(new_key, new_node) if self.keys.len() >= NODE_SIZE => {
                self.split_branch_insert(i, new_key, new_node)
            }
            InsertAction::Split(separator, new_node) => {
                self.keys.insert(i, separator);
                self.children.insert(i + 1, new_node);
                InsertAction::Inserted
            }
            action => action,
        }
    }
}
impl<K: Ord + Clone, V: Clone> Leaf<K, V> {
    pub(crate) fn insert<P: SharedPointerKind>(
        &mut self,
        key: K,
        value: V,
    ) -> InsertAction<K, V, P> {
        match slice_ext::binary_search_by(&self.keys, |(k, _)| k.cmp(&key)) {
            Ok(i) => {
                let (k, v) = mem::replace(&mut self.keys[i], (key, value));
                InsertAction::Replaced(k, v)
            }
            Err(i) if self.keys.len() >= NODE_SIZE => self.split_leaf_insert(i, key, value),
            Err(i) => {
                self.keys.insert(i, (key, value));
                InsertAction::Inserted
            }
        }
    }
}
impl<K: Ord + Clone, V: Clone, P: SharedPointerKind> Node<K, V, P> {
    pub(crate) fn insert(&mut self, key: K, value: V) -> InsertAction<K, V, P> {
        match self {
            Node::Branch(branch) => SharedPointer::make_mut(branch).insert(key, value),
            Node::Leaf(leaf) => SharedPointer::make_mut(leaf).insert(key, value),
        }
    }
}
impl<K: Ord + Clone, V: Clone, P: SharedPointerKind> Branch<K, V, P> {
    #[cold]
    fn split_branch_insert(
        &mut self,
        i: usize,
        new_key: K,
        new_node: Node<K, V, P>,
    ) -> InsertAction<K, V, P> {
        let split_idx = MEDIAN + (i > MEDIAN) as usize;
        let mut right_keys = self.keys.split_off(split_idx);
        let split_idx = MEDIAN + (i >= MEDIAN) as usize;
        let mut right_children = self.children.split_off(split_idx);
        let separator = if i == MEDIAN {
            right_children.insert(0, new_node.clone());
            new_key
        } else {
            if i < MEDIAN {
                self.keys.insert(i, new_key);
                self.children.insert(i + 1, new_node);
            } else {
                right_keys.insert(i - (MEDIAN + 1), new_key);
                right_children.insert(i - (MEDIAN + 1) + 1, new_node);
            }
            self.keys.pop_back()
        };
        debug_assert_eq!(self.keys.len(), right_keys.len());
        debug_assert_eq!(self.keys.len() + 1, self.children.len());
        debug_assert_eq!(right_keys.len() + 1, right_children.len());
        InsertAction::Split(
            separator,
            Node::Branch(SharedPointer::new(Branch {
                keys: right_keys,
                children: right_children,
            })),
        )
    }
}

impl<K: Ord + Clone, V: Clone> Leaf<K, V> {
    #[inline]
    fn split_leaf_insert<P: SharedPointerKind>(
        &mut self,
        i: usize,
        key: K,
        value: V,
    ) -> InsertAction<K, V, P> {
        let mut right_keys = self.keys.split_off(MEDIAN);
        if i < MEDIAN {
            self.keys.insert(i, (key, value));
        } else {
            right_keys.insert(i - MEDIAN, (key, value));
        }
        InsertAction::Split(
            right_keys.first().unwrap().0.clone(),
            Node::Leaf(SharedPointer::new(Leaf { keys: right_keys })),
        )
    }
}

impl<K: Ord + Clone, V: Clone, P: SharedPointerKind> Branch<K, V, P> {
    pub(crate) fn lookup_mut<Q>(&mut self, key: &Q) -> Option<(&K, &mut V)>
    where
        Q: Comparable<K> + ?Sized,
    {
        let i = slice_ext::binary_search_by(&self.keys, |k| key.compare(k).reverse())
            .map(|x| x + 1)
            .unwrap_or_else(|x| x);
        match &mut self.children {
            Children::Leaves { leaves } => SharedPointer::make_mut(&mut leaves[i]).lookup_mut(key),
            Children::Branches { branches, .. } => {
                SharedPointer::make_mut(&mut branches[i]).lookup_mut(key)
            }
        }
    }
}

impl<K: Ord + Clone, V: Clone> Leaf<K, V> {
    pub(crate) fn lookup_mut<Q>(&mut self, key: &Q) -> Option<(&K, &mut V)>
    where
        Q: Comparable<K> + ?Sized,
    {
        let keys = &mut self.keys;
        let i = slice_ext::binary_search_by(keys, |(k, _)| key.compare(k).reverse()).ok()?;
        keys.get_mut(i).map(|(k, v)| (&*k, v))
    }
}

impl<K: Ord + Clone, V: Clone, P: SharedPointerKind> Node<K, V, P> {
    pub(crate) fn lookup_mut<Q>(&mut self, key: &Q) -> Option<(&K, &mut V)>
    where
        Q: Comparable<K> + ?Sized,
    {
        match self {
            Node::Branch(branch) => SharedPointer::make_mut(branch).lookup_mut(key),
            Node::Leaf(leaf) => SharedPointer::make_mut(leaf).lookup_mut(key),
        }
    }

    pub(crate) fn new_from_split(left: Self, separator: K, right: Self) -> Self {
        Node::Branch(SharedPointer::new(Branch {
            keys: Chunk::unit(separator),
            children: match (left, right) {
                (Node::Branch(left), Node::Branch(right)) => Children::Branches {
                    level: NonZeroUsize::new(left.level() + 1).unwrap(),
                    branches: Chunk::from_iter([left, right]),
                },
                (Node::Leaf(left), Node::Leaf(right)) => Children::Leaves {
                    leaves: Chunk::from_iter([left, right]),
                },
                _ => panic!("mismatched split"),
            },
        }))
    }
}

impl<K: Ord, V, P: SharedPointerKind> Branch<K, V, P> {
    fn min(&self) -> Option<&(K, V)> {
        let mut node = self;
        loop {
            match &node.children {
                Children::Leaves { leaves } => return leaves.first()?.min(),
                Children::Branches { branches, .. } => node = branches.first()?,
            }
        }
    }
    fn max(&self) -> Option<&(K, V)> {
        let mut node = self;
        loop {
            match &node.children {
                Children::Leaves { leaves } => return leaves.last()?.max(),
                Children::Branches { branches, .. } => node = branches.last()?,
            }
        }
    }
    pub(crate) fn lookup<Q>(&self, key: &Q) -> Option<&(K, V)>
    where
        Q: Comparable<K> + ?Sized,
    {
        let mut node = self;
        loop {
            let i = slice_ext::binary_search_by(&node.keys, |k| key.compare(k).reverse())
                .map(|x| x + 1)
                .unwrap_or_else(|x| x);
            match &node.children {
                Children::Leaves { leaves } => return leaves[i].lookup(key),
                Children::Branches { branches, .. } => node = &branches[i],
            }
        }
    }
}

impl<K: Ord, V> Leaf<K, V> {
    fn min(&self) -> Option<&(K, V)> {
        self.keys.first()
    }
    fn max(&self) -> Option<&(K, V)> {
        self.keys.last()
    }
    fn lookup<Q>(&self, key: &Q) -> Option<&(K, V)>
    where
        Q: Comparable<K> + ?Sized,
    {
        let keys = &self.keys;
        let i = slice_ext::binary_search_by(keys, |(k, _)| key.compare(k).reverse()).ok()?;
        keys.get(i)
    }
}

impl<K: Ord, V, P: SharedPointerKind> Node<K, V, P> {
    pub(crate) fn min(&self) -> Option<&(K, V)> {
        match self {
            Node::Branch(branch) => branch.min(),
            Node::Leaf(leaf) => leaf.min(),
        }
    }

    pub(crate) fn max(&self) -> Option<&(K, V)> {
        match self {
            Node::Branch(branch) => branch.max(),
            Node::Leaf(leaf) => leaf.max(),
        }
    }

    pub(crate) fn lookup<Q>(&self, key: &Q) -> Option<&(K, V)>
    where
        Q: Comparable<K> + ?Sized,
    {
        match self {
            Node::Branch(branch) => branch.lookup(key),
            Node::Leaf(leaf) => leaf.lookup(key),
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
        }
    }
}

impl<K: Clone, V: Clone, P: SharedPointerKind> Clone for Children<K, V, P> {
    fn clone(&self) -> Self {
        match self {
            Children::Leaves { leaves } => Children::Leaves {
                leaves: leaves.clone(),
            },
            Children::Branches { branches, level } => Children::Branches {
                branches: branches.clone(),
                level: *level,
            },
        }
    }
}

impl<K, V, P: SharedPointerKind> Clone for Node<K, V, P> {
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
    Split(K, Node<K, V, P>),
}

impl<K, V, P: SharedPointerKind> Default for Node<K, V, P> {
    fn default() -> Self {
        Node::Leaf(SharedPointer::new(Leaf { keys: Chunk::new() }))
    }
}

#[derive(Debug)]
pub(crate) struct ConsumingIter<K, V, P: SharedPointerKind> {
    /// The leaves of the tree, in order, note that this will remain the shared ptr
    /// as it will allows us to have a smaller VecDeque allocation and avoid eagerly
    /// cloning the leaves, which defeats the purpose of this iterator.
    /// Leaves present in the VecDeque are guaranteed to be non-empty.
    leaves: VecDeque<SharedPointer<Leaf<K, V>, P>>,
    remaining: usize,
}

impl<K, V, P: SharedPointerKind> Clone for ConsumingIter<K, V, P> {
    fn clone(&self) -> Self {
        Self {
            leaves: self.leaves.clone(),
            remaining: self.remaining,
        }
    }
}

impl<K, V, P: SharedPointerKind> ConsumingIter<K, V, P> {
    pub(crate) fn new(node: Option<Node<K, V, P>>, size: usize) -> Self {
        fn push<K, V, P: SharedPointerKind>(
            out: &mut VecDeque<SharedPointer<Leaf<K, V>, P>>,
            node: SharedPointer<Branch<K, V, P>, P>,
        ) {
            match &node.children {
                Children::Leaves { leaves } => {
                    out.extend(leaves.iter().filter(|leaf| !leaf.keys.is_empty()).cloned())
                }
                Children::Branches { branches, .. } => {
                    for child in branches.iter() {
                        push(out, child.clone());
                    }
                }
            }
        }
        // preallocate the VecDeque assuming each leaf is half full
        let mut leaves = VecDeque::with_capacity(size.div_ceil(NODE_SIZE / 2));
        match node {
            Some(Node::Branch(b)) => push(&mut leaves, b),
            Some(Node::Leaf(l)) => {
                if !l.keys.is_empty() {
                    leaves.push_back(l)
                }
            }
            None => (),
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
        let leaf = SharedPointer::make_mut(node);
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
        let leaf = SharedPointer::make_mut(node);
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
    pub(crate) fn new<R, Q>(root: Option<&'a Node<K, V, P>>, len: usize, range: R) -> Self
    where
        R: RangeBounds<Q>,
        Q: Comparable<K> + ?Sized,
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
                    fwd.is_empty()
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
                    bwd.is_empty()
                }
            }
            (exhausted, _) => exhausted,
        };

        // Check if forward is > backward cursor to determine if we are exhausted
        // Due to the usage of zip this is correct even if the cursors are already or not initialized yet
        fn cursors_exhausted<K, V, P: SharedPointerKind>(
            fwd: &Cursor<'_, K, V, P>,
            bwd: &Cursor<'_, K, V, P>,
        ) -> bool {
            for (&(fi, f), &(bi, b)) in fwd.stack.iter().zip(bwd.stack.iter()) {
                if !std::ptr::eq(f, b) {
                    return false;
                }
                if fi > bi {
                    return true;
                }
            }
            if let (Some((fi, f)), Some((bi, b))) = (fwd.leaf, bwd.leaf) {
                if !std::ptr::eq(f, b) {
                    return false;
                }
                if fi > bi {
                    return true;
                }
            }
            false
        }
        exhausted = exhausted || cursors_exhausted(&fwd, &bwd);

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
        if let (Some((fi, f)), Some((bi, b))) = (self.fwd.leaf, self.bwd.leaf) {
            if std::ptr::eq(f, b) && fi >= bi {
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
        if cursor.is_empty() {
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

#[cfg(test)]
thread_local! {
    /// How many nodes the cursors of this thread have entered
    /// (`Cursor::push_child`), for the tests of the work of a diff.
    pub(crate) static PUSHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[derive(Debug)]
pub(crate) struct Cursor<'a, K, V, P: SharedPointerKind> {
    // a sequence of nodes starting at the root
    stack: Vec<(usize, &'a Branch<K, V, P>)>,
    leaf: Option<(usize, &'a Leaf<K, V>)>,
}

impl<'a, K, V, P: SharedPointerKind> Clone for Cursor<'a, K, V, P> {
    fn clone(&self) -> Self {
        Self {
            stack: self.stack.clone(),
            leaf: self.leaf,
        }
    }
}

impl<'a, K, V, P: SharedPointerKind> Cursor<'a, K, V, P> {
    /// Creates a new empty cursor.
    /// The variety of methods is to allow for a more efficient initialization
    /// in all cases.
    pub(crate) fn empty() -> Self {
        Self {
            stack: Vec::new(),
            leaf: None,
        }
    }

    fn is_empty(&self) -> bool {
        self.stack.is_empty() && self.leaf.is_none()
    }

    pub(crate) fn init(&mut self, node: Option<&'a Node<K, V, P>>) {
        if let Some(node) = node {
            self.stack.reserve_exact(node.level());
            match node {
                Node::Branch(branch) => self.stack.push((0, branch)),
                Node::Leaf(leaf) => {
                    debug_assert!(self.leaf.is_none());
                    self.leaf = Some((0, leaf))
                }
            }
        }
    }

    // pushes the `ix`th child of `branch` onto the stack, whether it's a leaf
    // or a branch
    fn push_child(&mut self, branch: &'a Branch<K, V, P>, ix: usize) {
        #[cfg(test)]
        PUSHES.with(|n| n.set(n.get() + 1));
        debug_assert!(
            self.leaf.is_none(),
            "it doesn't make sense to push when we're already at a leaf"
        );
        match &branch.children {
            Children::Leaves { leaves } => self.leaf = Some((0, &leaves[ix])),
            Children::Branches { branches, .. } => self.stack.push((0, &branches[ix])),
        }
    }

    pub(crate) fn seek_to_first(&mut self) -> Option<&'a (K, V)> {
        loop {
            if let Some((i, leaf)) = &self.leaf {
                debug_assert_eq!(i, &0);
                return leaf.keys.first();
            }
            let (i, branch) = self.stack.last()?;
            debug_assert_eq!(i, &0);
            self.push_child(branch, 0);
        }
    }

    fn seek_to_last(&mut self) -> Option<&'a (K, V)> {
        loop {
            if let Some((i, leaf)) = &mut self.leaf {
                debug_assert_eq!(i, &0);
                *i = leaf.keys.len().saturating_sub(1);
                return leaf.keys.last();
            }
            let (i, branch) = self.stack.last_mut()?;
            debug_assert_eq!(i, &0);
            *i = branch.children.len() - 1;
            let (i, branch) = (*i, *branch);
            self.push_child(branch, i);
        }
    }

    fn seek_to_key<Q>(&mut self, key: &Q, for_prev: bool) -> bool
    where
        Q: Comparable<K> + ?Sized,
    {
        loop {
            if let Some((i, leaf)) = &mut self.leaf {
                let search =
                    slice_ext::binary_search_by(&leaf.keys, |(k, _)| key.compare(k).reverse());
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
            let Some((i, branch)) = self.stack.last_mut() else {
                return false;
            };
            *i = slice_ext::binary_search_by(&branch.keys, |k| key.compare(k).reverse())
                .map(|x| x + 1)
                .unwrap_or_else(|x| x);
            let (i, branch) = (*i, *branch);
            self.push_child(branch, i);
        }
    }

    /// Leaves the current leaf, or the pending child (see `pending`), without
    /// entering anything: the next child of the innermost branch that has
    /// one becomes pending, and branches with no children left are popped.
    /// If no branch has one, the cursor is at the end.
    fn step_over(&mut self) {
        self.leaf = None;
        while let Some((i, branch)) = self.stack.last_mut() {
            if *i + 1 < branch.children.len() {
                *i += 1;
                return;
            }
            self.stack.pop();
        }
    }

    /// Steps past the current entry: to the next entry of the same leaf, or
    /// out of the leaf as `step_over` does, entering nothing.
    fn step_entry(&mut self) {
        if let Some((i, leaf)) = &mut self.leaf {
            if *i + 1 < leaf.keys.len() {
                *i += 1;
                return;
            }
        }
        self.step_over();
    }

    /// The node the cursor stands before but has not entered, and its
    /// level: child `i` of the innermost branch `(i, branch)` when the cursor
    /// is in no leaf. `None` when the cursor is in a leaf or at the end.
    /// Only `step_over` and `enter` leave a cursor in this state, and only
    /// within `advance_skipping_shared`.
    fn pending(&self) -> Option<(*const (), usize)> {
        if self.leaf.is_some() {
            return None;
        }
        let &(i, branch) = self.stack.last()?;
        Some(match &branch.children {
            Children::Leaves { leaves } => (std::ptr::from_ref(&*leaves[i]).cast(), 0),
            Children::Branches { branches, level } => {
                (std::ptr::from_ref(&*branches[i]).cast(), level.get() - 1)
            }
        })
    }

    /// Enters the pending node: a branch becomes the innermost branch with
    /// its first child pending; a leaf becomes the current leaf at its first
    /// entry.
    fn enter(&mut self) {
        let (i, branch) = *self.stack.last().expect("a cursor with a pending node");
        self.push_child(branch, i);
    }

    /// Brings both cursors into a leaf (or to the end), stepping over each
    /// node that is pending in both at once. Pointers are compared only
    /// between nodes of one level, so equal pointers name one node.
    fn align<'b>(&mut self, other: &mut Cursor<'b, K, V, P>) {
        loop {
            match (self.pending(), other.pending()) {
                (Some((p, l)), Some((q, m))) => {
                    if l == m && p == q {
                        self.step_over();
                        other.step_over();
                    } else {
                        if l >= m {
                            self.enter();
                        }
                        if m >= l {
                            other.enter();
                        }
                    }
                }
                (Some(_), None) => {
                    self.enter();
                    self.seek_to_first();
                }
                (None, Some(_)) => {
                    other.enter();
                    other.seek_to_first();
                }
                (None, None) => return,
            }
        }
    }

    /// Advances this cursor and `other` past their current entries, whose
    /// keys the caller has verified to be equal, skipping in bulk over any
    /// run of nodes shared between the two trees.
    ///
    /// Skipping is sound only for whole shared subtrees: pointer equality
    /// implies equal contents, so a shared subtree contributes nothing to a
    /// diff. When both cursors sit at the same position of the same shared
    /// leaf, the rest of that leaf can be dropped wholesale. When both stand
    /// before the same node, at any level, that node is stepped over without
    /// entering it. But as soon as a cursor lands on an entry of an unshared
    /// leaf, that entry has not been compared yet, and the cursors must stop
    /// there so the caller can compare it.
    ///
    /// Pointers are compared at every level, not only at leaves: a cursor
    /// leaves a node without entering the next one (`step_over`), and `align`
    /// compares the nodes the two cursors stand before. Moving each cursor
    /// with `next` instead, which enters every node down to a leaf, would
    /// enter the first leaf of each shared sibling of a copied path, `B`
    /// nodes at each of `h` levels, `h` deep: O(B h²) nodes for a change of
    /// one key (178 nodes for 20,000 keys in the test
    /// `diff_skips_a_shared_subtree_at_the_root`, against 16 here).
    ///
    /// The proof below gives O(B (u + h (ℓ + 1))) steps for a whole diff,
    /// where `u` is the number of nodes of either tree not in the other, `ℓ`
    /// the leaves among them, `h` the larger height and `B` the node size:
    /// O(B h) for maps that differ by one insert, remove or update
    /// (u ≤ 2 h + 3, ℓ ≤ 3). The cost is two pointer comparisons per node
    /// pair stepped over, and no extra memory.
    ///
    /// Proof. Let the cursors be A (`self`) and B (`other`); a key is
    /// *consumed* by a cursor when the cursor has moved past it.
    ///
    /// Soundness. The caller (`DiffIter`) is a merge: a key consumed by one
    /// cursor and present in the other tree has been consumed by the other
    /// too. At entry both cursors are at one key `k`, so after it each has
    /// consumed exactly the keys of its tree up to `k`. (1) Two cursors at
    /// the same position of the same leaf: the rest of the leaf is the same
    /// entries in both trees, so dropping it yields no diff item and leaves
    /// both having consumed exactly their keys up to the leaf's last key.
    /// (2) In `align`, both pending at one level with one pointer: the node
    /// holds the keys of its subtree, which in each tree are the next keys
    /// after those consumed, since a cursor that stands before a node has
    /// consumed every key before the node and none in it. Stepping over it
    /// in both yields no diff item and keeps both having consumed exactly
    /// their keys up to the node's last key. Pointers are compared only at
    /// one level, and a level-`l` node of one tree is a `Leaf` when `l = 0`
    /// and a `Branch` otherwise, so equal pointers name one node of one
    /// type. (3) Entering a node consumes nothing. So when `align` returns,
    /// both cursors are at the least key not yet consumed in their trees, or
    /// at the end, and the loop below returns only with two entries the
    /// caller has not compared, both cursors in a leaf, or a cursor at the
    /// end, as the caller expects.
    ///
    /// Cost. Count the nodes the cursors enter (`push_child`): a node is
    /// entered at most once by each cursor, as cursors only move forward.
    /// Every other step either moves a stack index forward (at most 17 per
    /// entered node, plus the two roots), pops an entered node, or moves
    /// within an entered leaf. Within an unshared leaf that is at most `B`
    /// per leaf; within a shared leaf, a cursor moves only in (1), once, as
    /// a shared leaf holds no key missing in the other tree and so is never
    /// moved through by `DiffIter`'s Less or Greater arms. So the steps are
    /// O(B) times the entered nodes plus `B ℓ`. Unshared nodes are entered
    /// at most `u` times. Shared nodes are entered:
    /// (a) at `DiffIter::new`'s `seek_to_first`: at most the two heights;
    /// (b) by `next` in `DiffIter`'s other arms, which leave a leaf
    ///     holding a key missing in the other tree, so an unshared leaf: at
    ///     most `h` per unshared leaf;
    /// (c) in an `align` in which the two cursors' next keys differ, say A's
    ///     key `a` below B's key `b`. Then `a` is not in B's tree (both have
    ///     consumed exactly their keys up to some key below `a`, and B's next
    ///     key is `b`), so no node holding `a` is shared. If both are pending
    ///     when the keys come to differ (after a step over), no node is
    ///     stepped over from then on (a shared node's first key is the next
    ///     key in both trees), A enters only unshared nodes and ends in an
    ///     unshared leaf, and B enters at most `h` nodes: charge them to A's
    ///     leaf, which A enters once. If instead one cursor is in a leaf when
    ///     the other descends alone (`seek_to_first` in `align`), the one in a
    ///     leaf got there by `step_entry` within the leaf at the start of this
    ///     call (descents by one level reach leaves in both cursors at once,
    ///     as the levels stay equal or the higher one moves, and a step out of
    ///     a shared leaf leaves both cursors pending), so it stands at an
    ///     index of at least 1. If the one descending has the smaller key, it
    ///     enters only unshared nodes. If it has the larger key, the other's
    ///     leaf L holds the smaller key, which is missing in the descending
    ///     cursor's tree, so L is unshared; charge the at most `h` nodes
    ///     entered to L. All later keys of L are below the descending cursor's
    ///     new key (that key is the first key of a leaf of its tree, and L
    ///     holds the smaller key at an index of at least 1, so the new key is
    ///     not in L), so `DiffIter` then moves the other cursor out of L by
    ///     its Greater or Less arm, with no call of this function between:
    ///     L takes this charge once. A cursor that descends alone while the
    ///     other is at the end has a key missing in the other tree, and
    ///     enters only unshared nodes. So these entries number at most `2 h`
    ///     per unshared leaf;
    /// (d) in an `align` with equal next keys, by one-level descents. When
    ///     the two levels are equal and A's node X is shared, X is the node of
    ///     that level holding B's next key in B's tree too (all leaves of a
    ///     B-tree are at level 0), so X is B's pending node and is stepped
    ///     over: entered nodes at equal levels are unshared. When A's level is
    ///     higher, X holds B's next key in B's tree and is above B's pending
    ///     node, so X is on B's stack: B entered X before, by a step of kind
    ///     (a) to (c) (it cannot be of kind (d), which would need X on A's
    ///     stack, entered by A before), or X is B's root. A enters X once, so
    ///     these entries number at most the entries of kinds (a) to (c), plus
    ///     the two roots.
    /// With `H` the sum of the heights, the entries number at most
    /// `u + 2 (H + 3 h ℓ) + 2` (the bound `check_diff_work` tests in
    /// `ord::map`'s tests), so the diff takes O(B (u + h (ℓ + 1))) steps. ∎
    pub(crate) fn advance_skipping_shared<'b>(&mut self, other: &mut Cursor<'b, K, V, P>) {
        debug_assert!(self.leaf.is_some());
        debug_assert!(other.leaf.is_some());
        // Whether the entries at the cursors' current positions have already
        // been compared by the caller; only such entries may be stepped past
        // without looking at them.
        let mut current_pair_compared = true;
        while let (Some(this), Some(that)) = (self.leaf, other.leaf) {
            if std::ptr::eq(this.1, that.1) && this.0 == that.0 {
                // Same leaf at the same position: the rest of this leaf is
                // identical in both trees (case (1) of the proof).
                self.step_over();
                other.step_over();
            } else if !current_pair_compared {
                // Unshared leaves holding entries the caller has not seen:
                // stop here so the caller can compare them.
                return;
            } else {
                self.step_entry();
                other.step_entry();
            }
            self.align(other);
            current_pair_compared = false;
        }
    }

    pub(crate) fn next(&mut self) -> Option<&'a (K, V)> {
        loop {
            if let Some((i, leaf)) = &mut self.leaf {
                if *i + 1 < leaf.keys.len() {
                    *i += 1;
                    return leaf.keys.get(*i);
                }
                self.leaf = None;
            }
            let Some((i, branch)) = self.stack.last_mut() else {
                break;
            };
            if *i + 1 < branch.children.len() {
                *i += 1;
                let (i, branch) = (*i, *branch);
                self.push_child(branch, i);
                break;
            }
            self.stack.pop();
        }
        self.seek_to_first()
    }

    fn prev(&mut self) -> Option<&'a (K, V)> {
        loop {
            if let Some((i, leaf)) = &mut self.leaf {
                if *i > 0 {
                    *i -= 1;
                    return leaf.keys.get(*i);
                }
                self.leaf = None;
            }
            let Some((i, branch)) = self.stack.last_mut() else {
                break;
            };
            if *i > 0 {
                *i -= 1;
                let (i, branch) = (*i, *branch);
                self.push_child(branch, i);
                break;
            }
            self.stack.pop();
        }
        self.seek_to_last()
    }

    pub(crate) fn peek(&self) -> Option<&'a (K, V)> {
        if let Some((i, leaf)) = &self.leaf {
            leaf.keys.get(*i)
        } else {
            None
        }
    }
}

mod slice_ext {
    #[inline]
    #[allow(unsafe_code)]
    pub(super) fn binary_search_by<T, F>(slice: &[T], mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> std::cmp::Ordering,
    {
        // Optimization: defer to std-lib if we think we're comparing integers, in which case
        // the stdlib implementation optimizes better using a fully branchless approach.
        // This branch is fully resolved at compile-time and will not incur any space or runtime overhead.
        // There is a mild assumption that the std-lib implementation will remain optimized for primitive types.
        if !std::mem::needs_drop::<T>() && std::mem::size_of::<T>() <= 16 {
            return slice.binary_search_by(f);
        }

        // This binary search implementation will always perform the minimum number of
        // comparisons and also allows for early return from the search loop when the comparison
        // function returns `Equal`, which is best when the comparison function isn't trivial
        // (e.g. `memcmp` vs. integer comparison).

        use std::cmp::Ordering::*;
        let mut low = 0;
        let mut high = slice.len();
        // Compared to the stdlib this implementation perform early return when the comparison
        // function returns Equal and will perform the optimal number of comparisons.
        // This is a tradeoff when the comparisons aren't cheap, as is the case
        // when the comparison is a memcmp of the field name and CRDT type.
        while low < high {
            // the midpoint is biased (truncated) towards low so it will always be less than high
            let mid = low + (high - low) / 2;
            // Safety: mid is always in bounds as low < high <= slice.len(); thus mid < slice.len()
            let cmp = f(unsafe { slice.get_unchecked(mid) });
            // TODO: Use select_unpredictable when min rustc_version >= 1.88
            // to guarantee conditional move optimization.
            // low can only get up to slice.len() as mid < slice.len()
            low = if cmp == Less { mid + 1 } else { low };
            high = if cmp == Greater { mid } else { high };
            if cmp == Equal {
                // Safety: same as above
                unsafe {
                    std::hint::assert_unchecked(mid < slice.len());
                }
                return Ok(mid);
            }
        }
        // Safety: see low assignment above
        unsafe {
            std::hint::assert_unchecked(low <= slice.len());
        }
        Err(low)
    }
}
