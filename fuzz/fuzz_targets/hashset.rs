#![no_main]

use std::collections::HashSet as NatSet;
use std::fmt::Debug;
use std::iter::FromIterator;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use imbl::HashSet;

#[derive(Arbitrary, Debug)]
enum Action<A> {
    Insert(A),
    Remove(A),
}

fuzz_target!(|actions: Vec<Action<u32>>| {
    let mut set = HashSet::new();
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
        }
        assert_eq!(nat.len(), set.len());
    }
    assert_eq!(HashSet::from(nat.clone()), set);
    assert_eq!(NatSet::from_iter(set.iter().cloned()), nat);
    assert_eq!(set.iter().count(), nat.len());
    assert_eq!(set.into_iter().count(), nat.len());
});
