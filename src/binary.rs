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
                text: Rc::from(context),
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

    fn next(&mut self) -> Option<Self::Item> {
        if self.state.position >= self.state.text.len() {
            return None;
        }
        let item = self.state.text[self.state.position];
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
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    any()
}

pub fn n_bytes<S, E>(n: usize) -> Parsec<S, E, Vec<u8>>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    any().take(n - 1..=n - 1)
}

pub fn u8<S, E>() -> Parsec<S, E, u8>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    byte()
}
