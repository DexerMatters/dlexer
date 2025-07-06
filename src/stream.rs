use std::{collections::VecDeque, io::Read, rc::Rc};

use crate::lex::{LexIterState, LexIterTrait};

pub const DEFAULT_BUFFER_SIZE: usize = 128;

pub struct StreamContext<I, C: ?Sized + Read, const BUFFER_SIZE: usize = DEFAULT_BUFFER_SIZE> {
    buffer: VecDeque<I>,
    current_reader_position: usize,
    context: C,
}

impl<I, C, const BUFFER_SIZE: usize> StreamContext<I, C, BUFFER_SIZE>
where
    C: Read + 'static,
{
    pub fn new(context: C) -> Self {
        StreamContext {
            buffer: VecDeque::with_capacity(BUFFER_SIZE),
            current_reader_position: 0,
            context,
        }
    }
}

pub struct StreamLexIter<I: LexIterTrait, const BUFFER_SIZE: usize = DEFAULT_BUFFER_SIZE> {
    state: LexIterState<StreamContext<I::Item, dyn Read, BUFFER_SIZE>>,
    _marker: std::marker::PhantomData<I>,
}

impl<I: LexIterTrait + Clone> Clone for StreamLexIter<I> {
    fn clone(&self) -> Self {
        StreamLexIter {
            state: LexIterState::clone(&self.state),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<I: LexIterTrait> StreamLexIter<I> {
    pub fn new(context: impl Read + 'static) -> Self {
        StreamLexIter {
            state: LexIterState {
                context: Rc::new(StreamContext::new(context)),
                current_line: 1,
                current_column: 0,
                current_pos: 0,
                current_indent: 0,
                indent_flag: false,
                position: 0,
            },
            _marker: std::marker::PhantomData,
        }
    }
}
