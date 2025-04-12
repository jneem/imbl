# imbl

[![crates.io](https://img.shields.io/crates/v/imbl)](https://crates.io/crates/imbl)
![tests](https://github.com/jneem/imbl/actions/workflows/ci.yml/badge.svg)
[![docs.rs](https://docs.rs/imbl/badge.svg)](https://docs.rs/imbl/)
[![coverage](https://coveralls.io/repos/github/jneem/imbl/badge.svg)](https://coveralls.io/github/jneem/imbl)

Blazing fast immutable collection datatypes for Rust.

This is originally a fork of the [`im`](https://github.com/bodil/im-rs) crate, which is unmaintained. The `1.0` release of `imbl` is compatible with the
`15.0.0` release of `im`, but with some fixes to `OrdMap` and `OrdSet`.

Changes from `im` include:

* Bug fixes on OrdMap/OrdSet
* Bug fixes on Vector
* Significant performance improvements for HashMap/HashSet
* Significant performance improvements for OrdMap/OrdSet
* Supports using [`trimphe::Arc`](https://docs.rs/triomphe/latest/triomphe/struct.Arc.html) for the shared pointer implementation

## Documentation

* [API docs](https://docs.rs/imbl/)

## Minimum supported rust version

This crate supports rust 1.77 and later. As const generics become more useful,
the minimum supported rust version will increase.

## Licence

Copyright 2017--2021 Bodil Stokke

Copyright 2021 Joe Neeman

This software is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at <http://mozilla.org/MPL/2.0/>.

## Code of Conduct

Please note that this project is released with a [Contributor Code of
Conduct][coc]. By participating in this project you agree to abide by its
terms.

[coc]: https://github.com/jneem/imbl/blob/master/CODE_OF_CONDUCT.md
