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
        fn ordset_and_btree_set_encoded_the_same_way(ref bts in ::proptest::collection::btree_set(i32::ANY, 0..100)) {
            let s = OrdSet::from(bts);
            let config =  bincode::config::standard();
            assert_eq!(encode_to_vec(bts, config).unwrap(), encode_to_vec(s, config).unwrap());
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
        fn ordmap_and_btree_map_encoded_the_same_way(ref btm in ::proptest::collection::btree_map(i32::ANY, i32::ANY, 0..100)) {
            let s: OrdMap<i32, i32>  = OrdMap::from(btm);
            let config =  bincode::config::standard();
            assert_eq!(encode_to_vec(btm, config).unwrap(), encode_to_vec(s, config).unwrap());
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
        fn encoding_std_hashmap_and_decoding_as_imbl_hashmap_is_same_as_converting(
            ref std_hm in ::proptest::collection::hash_map(i32::ANY, i32::ANY, 0..100)
        ) {
            // In fact, std's HashMap and imbl's HashMap are _not_ encoded the same since the order of the items can differ.
            let config =  bincode::config::standard();
            let encoded = encode_to_vec(std_hm, config).unwrap();
            let converted = HashMap::from(std_hm);
            let decoded : HashMap<i32, i32> = decode_from_slice(&encoded, config).unwrap().0;
            assert_eq!(decoded, converted);
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
        #[cfg_attr(miri, ignore)]
        #[test]
        fn encoding_std_hashset_and_decoding_as_imbl_hashset_is_same_as_converting(
            ref std_hs in ::proptest::collection::hash_set(i32::ANY, 0..100)
        ) {
            // In fact, std's HashSet and imbl's HashSet are _not_ encoded the same since the order of the items can differ.
            let config =  bincode::config::standard();
            let encoded = encode_to_vec(std_hs, config).unwrap();
            let converted = HashSet::from(std_hs);
            let decoded : HashSet<i32> = decode_from_slice(&encoded, config).unwrap().0;
            assert_eq!(decoded, converted);
        }

        #[test]
        fn encode_and_decode_vector(ref v in vector(i32::ANY, 0..100)) {
            let config =  bincode::config::standard();
            assert_eq!(v,
                &decode_from_slice::<Vector::<i32>, _>(&encode_to_vec(v, config).unwrap(), config).unwrap().0
            )
        }

        #[cfg_attr(miri, ignore)]
        #[test]
        fn vector_and_vec_encoded_the_same_way(ref vec in ::proptest::collection::vec(i32::ANY, 0..100)) {
            let vector = Vector::from(vec);
            let config =  bincode::config::standard();
            assert_eq!(encode_to_vec(vec, config).unwrap(), encode_to_vec(vector, config).unwrap());
        }
    }
}
