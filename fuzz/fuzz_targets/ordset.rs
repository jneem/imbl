#![no_main]

use std::collections::BTreeSet as NatSet;
use std::fmt::Debug;
use std::ops::Range;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use imbl::OrdSet;

#[derive(Arbitrary, Debug)]
enum Action<A: Clone + PartialOrd> {
    Insert(A),
    Remove(A),
    Range(Range<A>),
}

fuzz_target!(|actions: Vec<Action<u32>>| {
    let mut set = OrdSet::new();
    let mut nat = NatSet::new();
    for action in actions {
        match action {
            Action::Insert(value) => {
                let len = nat.len() + if nat.contains(&value) { 0 } else { 1 };
                nat.insert(value);
                set.insert(value);
                assert_eq!(len, set.len());
            }
            Action::Remove(value) => {
                let len = nat.len() - if nat.contains(&value) { 1 } else { 0 };
                nat.remove(&value);
                set.remove(&value);
                assert_eq!(len, set.len());
            }
            Action::Range(range) => {
                assert_eq!(set.get_min(), nat.first());
                assert_eq!(set.get_max(), nat.last());
                assert_eq!(set.get_next(&range.start), nat.range(range.start..).next());
                assert_eq!(set.get_prev(&range.start), nat.range(..=range.start).last());

                let mut set_it = set.range(range.clone());
                let mut nat_it = nat.range(range.clone());
                loop {
                    let (a, b) = (set_it.next(), nat_it.next());
                    assert_eq!(a, b);
                    if a.is_none() {
                        break;
                    }
                }
                let range = range.start..=range.end;
                let mut set_it = set.range(range.clone());
                let mut nat_it = nat.range(range);
                loop {
                    let (a, b) = (set_it.next(), nat_it.next());
                    assert_eq!(a, b);
                    if a.is_none() {
                        break;
                    }
                }
            }
        }
        assert_eq!(nat.len(), set.len());
    }
    assert_eq!(OrdSet::from(nat.clone()), set);
    assert_eq!(OrdSet::from_iter(nat.iter().cloned()), set);
    for (a, b) in set.range(..).zip(nat.range(..)) {
        assert_eq!(a, b);
    }
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
