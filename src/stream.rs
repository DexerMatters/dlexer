use std::{io::Read, rc::Rc};

use crate::lex::{LexIterState, LexIterTrait};

pub struct StreamLexIter<I: LexIterTrait> {
    state: LexIterState<dyn Read>,
    sub_iter: I,
}

impl<I: LexIterTrait + Clone> Clone for StreamLexIter<I> {
    fn clone(&self) -> Self {
        StreamLexIter {
            state: self.state.clone(),
            sub_iter: self.sub_iter.clone(),
        }
    }
}

impl<I: LexIterTrait> StreamLexIter<I> {
    pub fn new(context: impl Read + 'static, sub_iter: I) -> Self {
        StreamLexIter {
            state: LexIterState {
                text: Rc::new(context),
                current_line: 1,
                current_column: 0,
                current_pos: 0,
                current_indent: 0,
                indent_flag: false,
                position: 0,
            },
            sub_iter,
        }
    }
}
