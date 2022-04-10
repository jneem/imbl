// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/// The branching factor of RRB-trees
#[cfg(feature = "small-chunks")]
pub(crate) const VECTOR_CHUNK_SIZE: usize = 4;
#[cfg(not(feature = "small-chunks"))]
pub(crate) const VECTOR_CHUNK_SIZE: usize = 64;

/// The branching factor of B-trees
// Must be an even number!
#[cfg(feature = "small-chunks")]
pub(crate) const ORD_CHUNK_SIZE: usize = 4;
#[cfg(not(feature = "small-chunks"))]
pub(crate) const ORD_CHUNK_SIZE: usize = 64;

/// The level size of HAMTs, in bits
/// Branching factor is 2 ^ HashLevelSize.
#[cfg(feature = "small-chunks")]
pub(crate) const HASH_LEVEL_SIZE: usize = 2;
#[cfg(not(feature = "small-chunks"))]
pub(crate) const HASH_LEVEL_SIZE: usize = 5;

/// The size of per-instance memory pools if the `pool` feature is enabled.
/// This is set to 0, meaning you have to opt in to using a pool by constructing
/// with eg. `Vector::with_pool(pool)` even if the `pool` feature is enabled.
pub(crate) const POOL_SIZE: usize = 0;
