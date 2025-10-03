// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/// The branching factor of RRB-trees
#[cfg(feature = "small-chunks")]
pub(crate) const VECTOR_CHUNK_SIZE: usize = 4;
#[cfg(not(feature = "small-chunks"))]
pub(crate) const VECTOR_CHUNK_SIZE: usize = 64;

/// The branching factor of B-trees
// Value of 6 chosen improve test coverage, specifically
// so that both deletion node merging and rebalancing are tested.
// Must be an even number!
#[cfg(feature = "small-chunks")]
pub(crate) const ORD_CHUNK_SIZE: usize = 6;
// Value of 16 chosen based on performance analysis. Larger nodes might improve lookup slightly
// and bulk mutable operations more so, but suffers severe copy overhead for small mutations.
// Under typical workloads (e.g. 70% lookup, 25% small mutation, 5% bulk mutation), 16 arguably
// provides the best balance.
#[cfg(not(feature = "small-chunks"))]
pub(crate) const ORD_CHUNK_SIZE: usize = 16;

/// The level size of HAMTs, in bits
/// Branching factor is 2 ^ HashLevelSize.
// The smallest supported value is 3 currently, as the small node
// (half the size of a full node) requires at least 4 slots.
#[cfg(feature = "small-chunks")]
pub(crate) const HASH_LEVEL_SIZE: usize = 3;
// Value of 5 (branching factor 32) chosen based on performance analysis. Smaller value 4
// (branching factor 16) improves immutable inserts by 16-25% but suffers severe lookup
// regressions. Under typical workloads (e.g. 70% lookup, 25% small mutation, 5% bulk mutation),
// 5 is arguably better overall.
#[cfg(not(feature = "small-chunks"))]
pub(crate) const HASH_LEVEL_SIZE: usize = 5;
