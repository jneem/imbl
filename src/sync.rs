// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

pub(crate) use self::lock::Lock;

// FIXME: This is intended to be replaced by something else when using the non-threadsafe, `Rc`
// version of this crate. We have temporarily disabled that.
mod lock {
    use std::sync::{Arc, Mutex, MutexGuard};

    /// Thread safe lock: just wraps a `Mutex`.
    pub(crate) struct Lock<A> {
        lock: Arc<Mutex<A>>,
    }

    impl<A> Lock<A> {
        pub(crate) fn new(value: A) -> Self {
            Lock {
                lock: Arc::new(Mutex::new(value)),
            }
        }

        #[inline]
        pub(crate) fn lock(&mut self) -> Option<MutexGuard<'_, A>> {
            self.lock.lock().ok()
        }
    }

    impl<A> Clone for Lock<A> {
        fn clone(&self) -> Self {
            Lock {
                lock: self.lock.clone(),
            }
        }
    }
}
