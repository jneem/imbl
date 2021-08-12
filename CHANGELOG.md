# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project
adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

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

