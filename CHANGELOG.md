# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project
adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Lower HashMap and HashSet memory usage (#129)
- Optimize HashMap and HashSet iteration speed (#129)

## [6.0.0] - 2025-07-15

### Fixed

- Fix `vector::Iter` is not `Clone` when `triomphe` is enabled (#107)

### Changed

- HashMap and HashSet iterators improved to not perform allocations whenever possible (#112)
- HashMap and HashSet new/default functions improved to not perform allocations for the empty map/set (#112)
- BuildHasher is now required to be Clone (technically a breaking change, but unlikely in practice) (#112)
- Restore Performance improvements for mutation of HashMap and HashSet (#108)
- OrdMap and OrdSet implementation rewritten as a B+Tree with significant performance improvements (#109)
- OrdMap and OrdSet new/default functions improved to not perform allocations for the empty map/set (#109)

## [5.0.0] - 2025-02-12

### Added

- Support different pointer types (thanks @inker0!) (#90). This is technically a breaking
   change since some types (like iterators) grew additional generic parameters.

### Fixed

- Make `Vector::skip` better in corner cases (#104)

### Changed

- Bump MSRV to 1.77

## [4.0.1] - 2025-01-22

### Fixed

- Fix bounds checks on `Focus::narrow` and `Focus::split_at` (#89)

## [4.0.0] - 2025-01-10

### Changed
- Remove the implementation of `ExactSizedIterator` for `OrdMap::range`, since it was
  incorrect. This is a breaking change.
- Fix stacked-borrows violations in `Vector`.
- Fix docs for `OrdMap::get_prev` and `OrdMap::get_next`.

### Added
- Implement `IntoIterator` for `&mut Vector`, as the standard library does for `&mut Vec`.

## [3.0.0] - 2024-04-28

### Changed

- Make `OrdSet::get_prev` and `OrdSet::get_next` more generic, using the `Borrow` trait.
- Improve the performance of HashMap iterators (#73)
- Speedup hashmap by speeding up node creation (#76)

## [2.0.3] - 2023-10-25

### Changed
- `Vector::truncate` no longer panics if the `len` argument is larger than the
  vector's length (instead it does nothing)
- Added `OrdSet::get` to align with `std::collections::BTreeSet` and make it possible
  to get values in the set by using a borrowed form of the element.

## [2.0.2] - 2023-08-19

### Changed
- Some unnecessary trait bounds on `HashMap` were removed.

## [2.0.1] - 2022-04-19

### Deprecated
- The `difference` alias for `symmetric_difference` has been deprecated.

    To avoid conflicting with the *std* library where `difference` is equivalent
    to *imbl*'s `relative_complement`.


## [2.0.0] - 2022-04-12

### Fixed
-   Fixed several critical bugs in `Vector` (see PRs #34 and #37).
-   Removed `Hash` and `PartialOrd` impls for `HashMap` and `HashSet`.
-   Made all container types covariant in their type parameters.

### Added
-   Added `Vector::insert_ord_by` and `Vector::insert_ord_by_key`
-   Added `From` impls from arrays.

## [1.0.1] - 2021-08-12

### Fixed

-   Fixed #18, a critical bug that prevented everything from being `Send` and `Sync`.
-   Fixed value priority of unions in `OrdMap` and `HashMap`: the values in `self` should always win.

## [1.0.0] - 2021-08-11

This is the initial release of `imbl`, our fork/continuation of `im`. It is
fully compatible with version `15.0.0` of `im`, and this changelog only lists
those things which have changed since the fork.

### Fixed

-   Fixed bugs when deleting elements from large `OrdMap`s and `OrdSet`s
-   Fixed bugs where iterating over `OrdMap`s and `OrdSet`s could skip some elements.

[2.0.0] - 2022-04-12: https://github.com/jneem/imbl/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/jneem/imbl/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/jneem/imbl/compare/releases/tag/v1.0.0
