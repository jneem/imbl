# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project
adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed
- `Vector::truncate` no longer panics if the `len` argument is larger than the
  vector's length (instead it does nothing)

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
