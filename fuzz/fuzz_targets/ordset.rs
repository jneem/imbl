#![no_main]

use std::collections::BTreeSet as NatSet;
use std::fmt::Debug;
use std::ops::Bound;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use imbl::OrdSet;

#[derive(Arbitrary, Debug)]
enum NextAction {
    Fwd,
    Bwd,
    BwdFwd,
    FwdBwd,
}

#[derive(Arbitrary, Debug)]
enum Action<A: Clone + PartialOrd> {
    Insert(A),
    Remove(A),
    Range((Bound<A>, Bound<A>), NextAction),
}

fuzz_target!(|actions: Vec<Action<u32>>| {
    let mut set = OrdSet::new();
    let mut nat = NatSet::new();
    for action in actions {
        match action {
            Action::Insert(value) => {
                nat.insert(value);
                set.insert(value);
            }
            Action::Remove(value) => {
                nat.remove(&value);
                set.remove(&value);
            }
            Action::Range(range, na) => {
                assert_eq!(set.get_min(), nat.first());
                assert_eq!(set.get_max(), nat.last());
                match (range.0, range.1) {
                    (Bound::Included(v) | Bound::Excluded(v), ..)
                    | (.., Bound::Included(v) | Bound::Excluded(v)) => {
                        assert_eq!(set.get_next(&v), nat.range(v..).next());
                        assert_eq!(set.get_prev(&v), nat.range(..=v).last());
                        assert_eq!(set.get(&v), nat.get(&v));
                    }
                    _ => {}
                }
                // std Btree panics if the range end isn't >= range start
                // but OrdSet returns an empty iterator
                let valid_std = match (range.0, range.1) {
                    (Bound::Included(v), Bound::Included(w)) => v <= w,
                    (Bound::Excluded(v), Bound::Excluded(w))
                    | (Bound::Included(v), Bound::Excluded(w))
                    | (Bound::Excluded(v), Bound::Included(w)) => v < w,
                    _ => true,
                };
                if !valid_std {
                    assert_eq!(set.range(range).count(), 0);
                    assert_eq!(set.range(range).rev().count(), 0);
                    continue;
                }

                let mut set_it = set.range(range.clone());
                let mut nat_it = nat.range(range);
                loop {
                    let (a, b) = match na {
                        NextAction::Fwd => (set_it.next(), nat_it.next()),
                        NextAction::Bwd => (set_it.next_back(), nat_it.next_back()),
                        NextAction::BwdFwd => {
                            assert_eq!(set_it.next_back(), nat_it.next_back());
                            (set_it.next(), nat_it.next())
                        }
                        NextAction::FwdBwd => {
                            assert_eq!(set_it.next(), nat_it.next());
                            (set_it.next_back(), nat_it.next_back())
                        }
                    };
                    assert_eq!(a, b);
                    if a.is_none() {
                        assert_eq!(set_it.next(), None);
                        assert_eq!(set_it.next_back(), None);
                        break;
                    }
                }
            }
        }
        assert_eq!(nat.len(), set.len());
    }
    set.check_sane();
    assert_eq!(OrdSet::<_>::from(nat.clone()), set);
    assert_eq!(OrdSet::<_>::from_iter(nat.iter().cloned()), set);
    for (a, b) in set.iter().zip(&nat) {
        assert_eq!(a, b);
    }
    for (a, b) in set.iter().rev().zip(nat.iter().rev()) {
        assert_eq!(a, b);
    }
    for (a, b) in set.into_iter().zip(nat) {
        assert_eq!(a, b);
    }
});
