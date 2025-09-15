// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use archery::SharedPointerKind;
use serde_core::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_core::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

use crate::hashmap::GenericHashMap;
use crate::hashset::GenericHashSet;
use crate::ordmap::GenericOrdMap;
use crate::ordset::GenericOrdSet;
use crate::vector::GenericVector;

struct SeqVisitor<'de, S, A> {
    phantom_s: PhantomData<S>,
    phantom_a: PhantomData<A>,
    phantom_lifetime: PhantomData<&'de ()>,
}

impl<'de, S, A> SeqVisitor<'de, S, A> {
    pub(crate) fn new() -> SeqVisitor<'de, S, A> {
        SeqVisitor {
            phantom_s: PhantomData,
            phantom_a: PhantomData,
            phantom_lifetime: PhantomData,
        }
    }
}

impl<'de, S, A> Visitor<'de> for SeqVisitor<'de, S, A>
where
    S: From<Vec<A>>,
    A: Deserialize<'de>,
{
    type Value = S;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a sequence")
    }

    fn visit_seq<Access>(self, mut access: Access) -> Result<Self::Value, Access::Error>
    where
        Access: SeqAccess<'de>,
    {
        let mut v: Vec<A> = match access.size_hint() {
            None => Vec::new(),
            Some(l) => Vec::with_capacity(l),
        };
        while let Some(i) = access.next_element()? {
            v.push(i)
        }
        Ok(From::from(v))
    }
}

struct MapVisitor<'de, S, K, V> {
    phantom_s: PhantomData<S>,
    phantom_k: PhantomData<K>,
    phantom_v: PhantomData<V>,
    phantom_lifetime: PhantomData<&'de ()>,
}

impl<'de, S, K, V> MapVisitor<'de, S, K, V> {
    pub(crate) fn new() -> MapVisitor<'de, S, K, V> {
        MapVisitor {
            phantom_s: PhantomData,
            phantom_k: PhantomData,
            phantom_v: PhantomData,
            phantom_lifetime: PhantomData,
        }
    }
}

impl<'de, S, K, V> Visitor<'de> for MapVisitor<'de, S, K, V>
where
    S: From<Vec<(K, V)>>,
    K: Deserialize<'de>,
    V: Deserialize<'de>,
{
    type Value = S;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a sequence")
    }

    fn visit_map<Access>(self, mut access: Access) -> Result<Self::Value, Access::Error>
    where
        Access: MapAccess<'de>,
    {
        let mut v: Vec<(K, V)> = match access.size_hint() {
            None => Vec::new(),
            Some(l) => Vec::with_capacity(l),
        };
        while let Some(i) = access.next_entry()? {
            v.push(i)
        }
        Ok(From::from(v))
    }
}

// Set

impl<'de, A: Deserialize<'de> + Ord + Clone, P: SharedPointerKind> Deserialize<'de>
    for GenericOrdSet<A, P>
{
    fn deserialize<D>(des: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        des.deserialize_seq(SeqVisitor::new())
    }
}

impl<A: Ord + Serialize, P: SharedPointerKind> Serialize for GenericOrdSet<A, P> {
    fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = ser.serialize_seq(Some(self.len()))?;
        for i in self.iter() {
            s.serialize_element(i)?;
        }
        s.end()
    }
}

// Map

impl<'de, K: Deserialize<'de> + Ord + Clone, V: Deserialize<'de> + Clone, P: SharedPointerKind>
    Deserialize<'de> for GenericOrdMap<K, V, P>
{
    fn deserialize<D>(des: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        des.deserialize_map(MapVisitor::<'de, GenericOrdMap<K, V, P>, K, V>::new())
    }
}

impl<K: Serialize + Ord, V: Serialize, P: SharedPointerKind> Serialize for GenericOrdMap<K, V, P> {
    fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = ser.serialize_map(Some(self.len()))?;
        for (k, v) in self.iter() {
            s.serialize_entry(k, v)?;
        }
        s.end()
    }
}

// HashMap

impl<'de, K, V, S, P: SharedPointerKind> Deserialize<'de> for GenericHashMap<K, V, S, P>
where
    K: Deserialize<'de> + Hash + Eq + Clone,
    V: Deserialize<'de> + Clone,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    fn deserialize<D>(des: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        des.deserialize_map(MapVisitor::<'de, GenericHashMap<K, V, S, P>, K, V>::new())
    }
}

impl<K, V, S, P> Serialize for GenericHashMap<K, V, S, P>
where
    K: Serialize + Hash + Eq,
    V: Serialize,
    S: BuildHasher + Default,
    P: SharedPointerKind,
{
    fn serialize<Ser>(&self, ser: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        let mut s = ser.serialize_map(Some(self.len()))?;
        for (k, v) in self.iter() {
            s.serialize_entry(k, v)?;
        }
        s.end()
    }
}

// HashSet

impl<
        'de,
        A: Deserialize<'de> + Hash + Eq + Clone,
        S: BuildHasher + Default + Clone,
        P: SharedPointerKind,
    > Deserialize<'de> for GenericHashSet<A, S, P>
{
    fn deserialize<D>(des: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        des.deserialize_seq(SeqVisitor::new())
    }
}

impl<A: Serialize + Hash + Eq, S: BuildHasher + Default, P: SharedPointerKind> Serialize
    for GenericHashSet<A, S, P>
{
    fn serialize<Ser>(&self, ser: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        let mut s = ser.serialize_seq(Some(self.len()))?;
        for i in self.iter() {
            s.serialize_element(i)?;
        }
        s.end()
    }
}

// Vector

impl<'de, A: Clone + Deserialize<'de>, P: SharedPointerKind> Deserialize<'de>
    for GenericVector<A, P>
{
    fn deserialize<D>(des: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        des.deserialize_seq(SeqVisitor::<'de, GenericVector<A, P>, A>::new())
    }
}

impl<A: Serialize, P: SharedPointerKind> Serialize for GenericVector<A, P> {
    fn serialize<S>(&self, ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = ser.serialize_seq(Some(self.len()))?;
        for i in self.iter() {
            s.serialize_element(i)?;
        }
        s.end()
    }
}

// Tests

#[cfg(test)]
mod test {
    use crate::{
        proptest::{hash_map, hash_set, ord_map, ord_set, vector},
        HashMap, HashSet, OrdMap, OrdSet, Vector,
    };
    use proptest::num::i32;
    use proptest::proptest;
    use serde_json::{from_str, to_string};

    proptest! {
        #[cfg_attr(miri, ignore)]
        #[test]
        fn ser_ordset(ref v in ord_set(i32::ANY, 0..100)) {
            assert_eq!(v, &from_str::<OrdSet<i32>>(&to_string(&v).unwrap()).unwrap());
        }

        #[cfg_attr(miri, ignore)]
        #[test]
        fn ser_ordmap(ref v in ord_map(i32::ANY, i32::ANY, 0..100)) {
            assert_eq!(v, &from_str::<OrdMap<i32, i32>>(&to_string(&v).unwrap()).unwrap());
        }

        #[cfg_attr(miri, ignore)]
        #[test]
        fn ser_hashmap(ref v in hash_map(i32::ANY, i32::ANY, 0..100)) {
            assert_eq!(v, &from_str::<HashMap<i32, i32>>(&to_string(&v).unwrap()).unwrap());
        }

        #[cfg_attr(miri, ignore)]
        #[test]
        fn ser_hashset(ref v in hash_set(i32::ANY, 0..100)) {
            assert_eq!(v, &from_str::<HashSet<i32>>(&to_string(&v).unwrap()).unwrap());
        }

        #[cfg_attr(miri, ignore)]
        #[test]
        fn ser_vector(ref v in vector(i32::ANY, 0..100)) {
            assert_eq!(v, &from_str::<Vector<i32>>(&to_string(&v).unwrap()).unwrap());
        }
    }
}
