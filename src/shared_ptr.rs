//! About shared pointers. Re-export the [`archery`] crate.
//!
//! [`archery`]: https://docs.rs/archery/latest/

pub use archery::{ArcK, RcK, SharedPointer, SharedPointerKind};

#[cfg(feature = "triomphe")]
pub use archery::ArcTK;

#[cfg(not(feature = "triomphe"))]
/// Default shared pointer used in data structures like [`vector::Vector`] in this crate. This alias points to [`ArcK`] if `triomphe` is disabled, [`ArcTK`] otherwise.
///
/// [`vector::Vector`]: ./type.Vector.html
/// [`ArcK`]: https://docs.rs/archery/latest/archery/shared_pointer/kind/struct.ArcK.html
/// [`ArcTK`]: https://docs.rs/archery/latest/archery/shared_pointer/kind/struct.ArcTK.html
pub type DefaultSharedPtr = ArcK;

#[cfg(feature = "triomphe")]
/// Default shared pointer used in data structures like [`vector::Vector`] in this crate. This alias points to [`ArcK`] if `triomphe` is disabled, [`ArcTK`] otherwise.
///
/// [`vector::Vector`]: ./type.Vector.html
/// [`ArcK`]: https://docs.rs/archery/latest/archery/shared_pointer/kind/struct.ArcK.html
/// [`ArcTK`]: https://docs.rs/archery/latest/archery/shared_pointer/kind/struct.ArcTK.html
pub type DefaultSharedPtr = ArcTK;
