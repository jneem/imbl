// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::hash::{BuildHasher, Hash};

use archery::SharedPointerKind;
use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::{DecodeError, EncodeError};
use bincode::{Decode, Encode};

use crate::hashmap::GenericHashMap;
use crate::hashset::GenericHashSet;
use crate::ordmap::GenericOrdMap;
use crate::ordset::GenericOrdSet;
use crate::vector::GenericVector;

// Set

impl<C, A: Decode<C> + Ord + Clone, P: SharedPointerKind> Decode<C> for GenericOrdSet<A, P> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let mut output = Self::new();
        let length: usize = Decode::decode(decoder)?;
        for _ in 0..length {
            let item: A = Decode::decode(decoder)?;
            // Duplicates are silently ignored.
            output.insert(item);
        }
        Ok(output)
    }
}

impl<A: Ord + Encode, P: SharedPointerKind> Encode for GenericOrdSet<A, P> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.len(), encoder)?;
        for item in self.iter() {
            Encode::encode(item, encoder)?;
        }
        Ok(())
    }
}

// Map

impl<C, K: Decode<C> + Ord + Clone, V: Decode<C> + Clone, P: SharedPointerKind> Decode<C>
    for GenericOrdMap<K, V, P>
{
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let len: usize = Decode::decode(decoder)?;
        let mut output = Self::new();
        for _ in 0..len {
            let (k, v): (K, V) = Decode::decode(decoder)?;
            // Duplicates are silently ignored
            output.insert(k, v);
        }
        Ok(output)
    }
}

impl<K: Encode + Ord + Clone, V: Encode + Clone, P: SharedPointerKind> Encode
    for GenericOrdMap<K, V, P>
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.len(), encoder)?;
        for (k, v) in self.iter() {
            Encode::encode(&(k, v), encoder)?;
        }
        Ok(())
    }
}

// HashMap

impl<C, K, V, S, P: SharedPointerKind> Decode<C> for GenericHashMap<K, V, S, P>
where
    K: Decode<C> + Hash + Eq + Clone,
    V: Decode<C> + Clone,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let len: usize = Decode::decode(decoder)?;
        let mut output = Self::new();
        for _ in 0..len {
            let (k, v): (K, V) = Decode::decode(decoder)?;
            // Duplicates are silently ignored
            output.insert(k, v);
        }
        Ok(output)
    }
}

impl<K, V, S, P> Encode for GenericHashMap<K, V, S, P>
where
    K: Encode + Hash + Eq,
    V: Encode,
    S: BuildHasher + Default,
    P: SharedPointerKind,
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.len(), encoder)?;
        for (k, v) in self.iter() {
            Encode::encode(&(k, v), encoder)?;
        }
        Ok(())
    }
}

// HashSet

impl<C, A, S, P> Decode<C> for GenericHashSet<A, S, P>
where
    A: Decode<C> + Hash + Eq + Clone,
    S: BuildHasher + Default + Clone,
    P: SharedPointerKind,
{
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let mut output = Self::new();
        let length: usize = Decode::decode(decoder)?;
        for _ in 0..length {
            let item: A = Decode::decode(decoder)?;
            // Duplicates are silently ignored.
            output.insert(item);
        }
        Ok(output)
    }
}

impl<A: Encode + Hash + Eq, S: BuildHasher + Default, P: SharedPointerKind> Encode
    for GenericHashSet<A, S, P>
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.len(), encoder)?;
        for item in self.iter() {
            Encode::encode(item, encoder)?;
        }
        Ok(())
    }
}

// Vector

impl<C, A: Clone + Decode<C>, P: SharedPointerKind> Decode<C> for GenericVector<A, P> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let mut output = Self::new();
        let length: usize = Decode::decode(decoder)?;
        for _ in 0..length {
            let item: A = Decode::decode(decoder)?;
            output.push_back(item);
        }
        Ok(output)
    }
}

impl<A: Encode, P: SharedPointerKind> Encode for GenericVector<A, P> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.len(), encoder)?;
        for item in self.iter() {
            Encode::encode(item, encoder)?;
        }
        Ok(())
    }
}

// Tests

#[cfg(test)]
mod test {
    use crate::{
        proptest::{hash_map, hash_set, ord_map, ord_set, vector},
        HashMap, HashSet, OrdMap, OrdSet, Vector,
    };
    use bincode::{decode_from_slice, encode_to_vec};
    use proptest::num::i32;
    use proptest::proptest;

    proptest! {
        #[cfg_attr(miri, ignore)]
        #[test]
        fn encode_and_decode_ordset(ref v in ord_set(i32::ANY, 0..100)) {
            let config =  bincode::config::standard();
            assert_eq!(v,
                &decode_from_slice::<OrdSet::<i32>, _>(&encode_to_vec(v, config).unwrap(), config).unwrap().0
            )
        }

        #[cfg_attr(miri, ignore)]
        #[test]
        fn encode_and_decode_ordmap(ref v in ord_map(i32::ANY, i32::ANY, 0..100)) {
            let config =  bincode::config::standard();
            assert_eq!(v,
                &decode_from_slice::<OrdMap::<i32, i32>, _>(&encode_to_vec(v, config).unwrap(), config).unwrap().0
            )
        }

        #[cfg_attr(miri, ignore)]
        #[test]
        fn encode_and_decode_hashmap(ref v in hash_map(i32::ANY, i32::ANY, 0..100)) {
            let config =  bincode::config::standard();
            assert_eq!(v,
                &decode_from_slice::<HashMap::<i32, i32>, _>(&encode_to_vec(v, config).unwrap(), config).unwrap().0
            )
        }

        #[cfg_attr(miri, ignore)]
        #[test]
        fn encode_and_decode_hashset(ref v in hash_set(i32::ANY, 0..100)) {
            let config =  bincode::config::standard();
            assert_eq!(v,
                &decode_from_slice::<HashSet::<i32>, _>(&encode_to_vec(v, config).unwrap(), config).unwrap().0
            )
        }

        #[cfg_attr(miri, ignore)]
        #[test]
        fn encode_and_decode_vector(ref v in vector(i32::ANY, 0..100)) {
            let config =  bincode::config::standard();
            assert_eq!(v,
                &decode_from_slice::<Vector::<i32>, _>(&encode_to_vec(v, config).unwrap(), config).unwrap().0
            )
        }
    }
}
