use std::rc::Rc;

use crate::{
    errors::{ParserError, SimpleParserError},
    lex::{LexIterState, LexIterTrait},
    parsec::{Parsec, any},
};

pub type BasicByteParser = Parsec<ByteLexIter, SimpleParserError<[u8]>, u8>;

#[derive(Clone, Debug)]
pub struct ByteLexIter {
    state: LexIterState<[u8]>,
}

impl ByteLexIter {
    pub fn new(context: &[u8]) -> Self {
        ByteLexIter {
            state: LexIterState {
                context: Rc::from(context),
                current_line: 1,
                current_column: 0,
                current_pos: 0,
                current_indent: 0,
                indent_flag: false,
                position: 0,
            },
        }
    }
}

impl LexIterTrait for ByteLexIter {
    type Context = [u8];

    type Item = u8;

    fn get_state(&self) -> LexIterState<Self::Context> {
        self.state.clone()
    }

    fn get_state_mut(&mut self) -> &mut LexIterState<Self::Context> {
        &mut self.state
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.state.position >= self.state.context.len() {
            return None;
        }
        let item = self.state.context[self.state.position];
        self.state.position += 1;
        self.state.current_pos += 1;
        Some(item)
    }
}

impl<E, A> Parsec<ByteLexIter, E, A>
where
    E: ParserError<Context = [u8]> + 'static,
    A: 'static,
{
    pub fn parse(&self, input: &[u8]) -> Result<A, E> {
        let lexer = ByteLexIter::new(input);
        self.run(lexer)
    }

    pub fn parse_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<A, E> {
        use std::fs::File;
        use std::io::{BufReader, Read};
        let file = File::open(path).unwrap();
        let mut reader = BufReader::new(file);
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).unwrap();
        let lexer = ByteLexIter::new(&buffer);
        self.run(lexer)
    }
}

pub fn byte<S, E>() -> Parsec<S, E, u8>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    any()
}

pub fn n_bytes<S, E>(n: usize) -> Parsec<S, E, Vec<u8>>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    any().take(n)
}

pub fn u8<S, E>() -> Parsec<S, E, u8>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    byte()
}

pub fn u16<S, E>() -> Parsec<S, E, u16>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + Clone + 'static,
{
    n_bytes(2)
        .map(|bytes| bytes.try_into().map(u16::from_be_bytes))
        .internalize(|err| format!("Failed to parse u16 from bytes: {:?}", err))
}

pub fn u32<S, E>() -> Parsec<S, E, u32>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + Clone + 'static,
{
    n_bytes(4)
        .map(|bytes| bytes.try_into().map(u32::from_be_bytes))
        .internalize(|err| format!("Failed to parse u32 from bytes: {:?}", err))
}

pub fn u64<S, E>() -> Parsec<S, E, u64>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + Clone + 'static,
{
    n_bytes(8)
        .map(|bytes| bytes.try_into().map(u64::from_be_bytes))
        .internalize(|err| format!("Failed to parse u64 from bytes: {:?}", err))
}

pub fn u128<S, E>() -> Parsec<S, E, u128>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + Clone + 'static,
{
    n_bytes(16)
        .map(|bytes| bytes.try_into().map(u128::from_be_bytes))
        .internalize(|err| format!("Failed to parse u128 from bytes: {:?}", err))
}
pub fn i8<S, E>() -> Parsec<S, E, i8>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    byte().map(|b| i8::from_le_bytes([b]))
}

pub fn i16<S, E>() -> Parsec<S, E, i16>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    n_bytes(2)
        .map(|bytes| bytes.try_into().map(i16::from_le_bytes))
        .internalize(|err| format!("Failed to parse i16 from bytes: {:?}", err))
}

pub fn i32<S, E>() -> Parsec<S, E, i32>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    n_bytes(4)
        .map(|bytes| bytes.try_into().map(i32::from_le_bytes))
        .internalize(|err| format!("Failed to parse i32 from bytes: {:?}", err))
}

pub fn i64<S, E>() -> Parsec<S, E, i64>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    n_bytes(8)
        .map(|bytes| bytes.try_into().map(i64::from_le_bytes))
        .internalize(|err| format!("Failed to parse i64 from bytes: {:?}", err))
}

pub fn i128<S, E>() -> Parsec<S, E, i128>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    n_bytes(16)
        .map(|bytes| bytes.try_into().map(i128::from_le_bytes))
        .internalize(|err| format!("Failed to parse i128 from bytes: {:?}", err))
}

pub fn f32<S, E>() -> Parsec<S, E, f32>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + Clone + 'static,
{
    u32().map(|bits| f32::from_bits(bits))
}

pub fn f64<S, E>() -> Parsec<S, E, f64>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + Clone + 'static,
{
    u64().map(|bits| f64::from_bits(bits))
}

pub unsafe fn align_to<S, E, T>() -> Parsec<S, E, *const T>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + 'static,
    T: 'static,
{
    let size = std::mem::size_of::<T>();
    n_bytes(size)
        .map(move |bytes| {
            let (head, body, _tail) = unsafe { bytes.align_to::<T>() };
            if head.is_empty() {
                return Err("Failed to align bytes");
            }
            Ok(body.first().unwrap() as *const T)
        })
        .internalize(|err| err)
}
