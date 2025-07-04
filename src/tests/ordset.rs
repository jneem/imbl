#![allow(clippy::unit_arg)]

use std::collections::BTreeSet;
use std::fmt::{Debug, Error, Formatter, Write};

use crate::OrdSet;

use proptest::proptest;
use proptest_derive::Arbitrary;

#[derive(Arbitrary, Debug)]
enum Action<A> {
    Insert(A),
    Remove(A),
}

#[derive(Arbitrary)]
struct Actions<A>(Vec<Action<A>>)
where
    A: Ord + Clone;

impl<A> Debug for Actions<A>
where
    A: Ord + Debug + Clone,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        let mut out = String::new();
        let mut expected = BTreeSet::new();
        writeln!(out, "let mut set = OrdSet::new();")?;
        for action in &self.0 {
            match action {
                Action::Insert(ref value) => {
                    expected.insert(value.clone());
                    writeln!(out, "set.insert({:?});", value)?;
                }
                Action::Remove(ref value) => {
                    expected.remove(value);
                    writeln!(out, "set.remove({:?});", value)?;
                }
            }
        }
        writeln!(
            out,
            "let expected = vec!{:?};",
            expected.into_iter().collect::<Vec<_>>()
        )?;
        writeln!(out, "assert_eq!(OrdSet::from(expected), set);")?;
        write!(f, "{}", super::code_fmt(&out))
    }
}

proptest! {
    #[test]
    fn comprehensive(actions: Actions<u8>) {
        let mut set = OrdSet::new();
        let mut nat = BTreeSet::new();
        for action in actions.0 {
            match action {
                Action::Insert(value) => {
                    let len = nat.len() + if nat.contains(&value) {
                        0
                    } else {
                        1
                    };
                    nat.insert(value);
                    set.insert(value);
                    assert_eq!(len, set.len());
                }
                Action::Remove(value) => {
                    let len = nat.len() - if nat.contains(&value) {
                        1
                    } else {
                        0
                    };
                    nat.remove(&value);
                    set.remove(&value);
                    assert_eq!(len, set.len());
                }
            }
            assert_eq!(nat.len(), set.len());
            assert_eq!(OrdSet::from(nat.clone()), set);
            assert!(nat.iter().eq(set.iter()));
        }
    }
}

#[test]
fn test_empty_leaf() {
    let mut set = OrdSet::new();
    set.insert(1);
    set.remove(&1);
    // This leaves the tree with just an empty leaf node; check that iterators
    // still work
    assert_eq!(set.range(1..2).count(), 0);
    assert_eq!(set.iter().count(), 0);
    assert_eq!(set.into_iter().count(), 0);
}
