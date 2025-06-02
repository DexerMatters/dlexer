use std::rc::Rc;
use std::str::Chars;

use crate::errors::ParserError;
use crate::parsec::{Parsec, digit};

pub trait Skipper {
    fn next(&self, state: &mut LexIterState) -> Option<char>;

    fn skip(&self, state: &mut LexIterState) -> usize {
        let original = state.current_pos;
        if let Some(c) = self.next(state) {
            // Push the character back to the buffer
            state.unread(c);
            if c == '\n' {
                state.current_line = state.current_line.saturating_sub(1);
                state.current_column = 1;
            } else {
                state.current_column = state.current_column.saturating_sub(1);
            }
        };

        state.current_pos.saturating_sub(original)
    }

    fn clone_box(&self) -> Box<dyn Skipper>;
}

pub trait AsSkipper {
    fn as_skipper(self) -> Box<dyn Skipper>;
}

impl<T: Skipper + 'static> AsSkipper for T {
    fn as_skipper(self) -> Box<dyn Skipper> {
        Box::new(self)
    }
}

impl AsSkipper for Box<dyn Skipper> {
    fn as_skipper(self) -> Box<dyn Skipper> {
        self
    }
}

impl<const N: usize> AsSkipper for [Box<dyn Skipper>; N] {
    fn as_skipper(self) -> Box<dyn Skipper> {
        let mut result = Vec::with_capacity(N);
        for skipper in self {
            result.push(skipper);
        }
        Box::new(Skippers::new(result))
    }
}

impl<T: Skipper + 'static> From<T> for Box<dyn Skipper> {
    fn from(skipper: T) -> Self {
        Box::new(skipper)
    }
}

#[derive(Clone)]
pub struct NoSkipper;

impl Skipper for NoSkipper {
    // No skipping so always return 0
    fn skip(&self, _state: &mut LexIterState) -> usize {
        0
    }
    fn next(&self, state: &mut LexIterState) -> Option<char> {
        state.next_one_char()
    }
    fn clone_box(&self) -> Box<dyn Skipper> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct WhitespaceSkipper;

impl Skipper for WhitespaceSkipper {
    // Skip consecutive whitespace (except newline)
    fn skip(&self, state: &mut LexIterState) -> usize {
        let mut count = 0;
        while let Some(c) = state.next_one_char() {
            if c.is_whitespace() && c != '\n' {
                count += 1;
            } else {
                state.unread(c);
                break;
            }
        }
        count
    }
    fn next(&self, state: &mut LexIterState) -> Option<char> {
        let _ = Self::skip(self, state);
        state.next_one_char()
    }
    fn clone_box(&self) -> Box<dyn Skipper> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct CharSkipper<const N: usize>(pub [char; N]);

impl<const N: usize> Skipper for CharSkipper<N> {
    // Skip characters contained in self.0
    fn skip(&self, state: &mut LexIterState) -> usize {
        let mut count = 0;
        while let Some(c) = state.next_one_char() {
            if self.0.contains(&c) {
                count += 1;
            } else {
                state.unread(c);
                break;
            }
        }
        count
    }
    fn next(&self, state: &mut LexIterState) -> Option<char> {
        let _ = Self::skip(self, state);
        state.next_one_char()
    }
    fn clone_box(&self) -> Box<dyn Skipper> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct LineSkipper(pub &'static str);

impl Skipper for LineSkipper {
    // Skip a line marker (substring) and then consume all chars until newline is reached;
    // stops before newline so it can be processed later.
    fn skip(&self, state: &mut LexIterState) -> usize {
        let mut count = 0;
        loop {
            let checkpoint = state.clone();
            if state.try_next_substring(self.0).is_some() {
                while let Some(c) = state.next_one_char() {
                    count += 1;
                    if c == '\n' {
                        state.unread(c); // Stop before newline
                        break;
                    }
                }
                continue;
            }
            *state = checkpoint;
            break;
        }
        count
    }
    fn next(&self, state: &mut LexIterState) -> Option<char> {
        let _ = Self::skip(self, state);
        state.next_one_char()
    }
    fn clone_box(&self) -> Box<dyn Skipper> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct BlockSkipper(pub &'static str, pub &'static str);

impl Skipper for BlockSkipper {
    // Skip block delimiters (self.0 and self.1) with nested block handling.
    fn skip(&self, state: &mut LexIterState) -> usize {
        let mut count = 0;
        let mut block_depth = 0;
        loop {
            let checkpoint = state.clone();
            if state.try_next_substring(self.0).is_some() {
                block_depth += 1;
                count += self.0.len();
                continue;
            }
            *state = checkpoint.clone();
            if state.try_next_substring(self.1).is_some() {
                if block_depth > 0 {
                    block_depth -= 1;
                    count += self.1.len();
                    continue;
                } else {
                    *state = checkpoint;
                    break;
                }
            }
            *state = checkpoint;
            if block_depth == 0 {
                break;
            }
            if let Some(_) = state.next_one_char() {
                count += 1;
            } else {
                break;
            }
        }
        count
    }
    fn next(&self, state: &mut LexIterState) -> Option<char> {
        let _ = Self::skip(self, state);
        state.next_one_char()
    }
    fn clone_box(&self) -> Box<dyn Skipper> {
        Box::new(self.clone())
    }
}

pub struct Skippers {
    pub skippers: Vec<Box<dyn Skipper>>,
}

impl Skippers {
    pub fn new(skippers: Vec<Box<dyn Skipper>>) -> Self {
        Skippers { skippers }
    }
}

impl Skipper for Skippers {
    fn skip(&self, state: &mut LexIterState) -> usize {
        let original = state.current_pos;
        loop {
            let pos_before = state.current_pos;
            for skipper in &self.skippers {
                let _ = skipper.skip(state);
            }
            if state.current_pos == pos_before {
                break;
            }
        }
        state.current_pos.saturating_sub(original)
    }
    fn next(&self, state: &mut LexIterState) -> Option<char> {
        let _ = Self::skip(self, state);
        state.next_one_char()
    }
    fn clone_box(&self) -> Box<dyn Skipper> {
        Box::new(Skippers {
            skippers: self.skippers.iter().map(|s| s.clone_box()).collect(),
        })
    }
}

pub trait LexIterTrait: Iterator<Item = char> + Sized {
    fn get_state(&self) -> LexIterState;
}

#[derive(Clone, Debug)]
pub struct LexIterState {
    pub text: Rc<str>,
    pub current_line: usize,
    pub current_column: usize,
    pub current_pos: usize,
    pub current_indent: usize,
    indent_flag: bool, // True for indent
    position: usize,   // Current position in the string
}

impl LexIterState {
    fn next_one_char(&mut self) -> Option<char> {
        if self.position >= self.text.len() {
            return None;
        }

        // Get the character at the current position
        let c = self.text[self.position..].chars().next().unwrap();
        self.position += c.len_utf8();

        if c == ' ' {
            if self.indent_flag {
                self.current_indent += 1; // Increment indent on space
            }
        } else if c == '\t' {
            if self.indent_flag {
                self.current_indent += 4; // Increment indent by 4 on tab
            }
        } else {
            self.indent_flag = false; // Reset indent flag for non-whitespace characters
        }

        if c == '\n' {
            self.current_line += 1;
            self.current_column = 0;
            self.current_indent = 0; // Reset indent on new line
            self.indent_flag = true; // Set indent flag for new line
        } else {
            self.current_column += 1;
        }
        self.current_pos += 1;

        Some(c)
    }

    fn unread(&mut self, c: char) {
        self.position -= c.len_utf8();

        if c == '\n' {
            self.current_line -= 1;
            self.current_column = 0;
        } else {
            self.current_column -= 1;
        }
        self.current_pos -= 1;
    }

    fn try_next_substring(&mut self, s: &str) -> Option<String> {
        let original = self.clone();
        let mut result = String::with_capacity(s.len());

        for expected_char in s.chars() {
            if let Some(actual_char) = self.next_one_char() {
                result.push(actual_char);
                if actual_char != expected_char {
                    *self = original; // Restore original state
                    return None;
                }
            } else {
                *self = original; // Restore original state
                return None;
            }
        }
        Some(result)
    }
}

pub struct LexIter {
    pub(crate) skipper: Box<dyn Skipper>,
    pub(crate) state: LexIterState,
}

impl Clone for LexIter {
    fn clone(&self) -> Self {
        LexIter {
            skipper: self.skipper.clone_box(),
            state: self.state.clone(),
        }
    }
}

impl LexIter {
    pub fn new<'a>(text: &'a str, skipper: impl Into<Box<dyn Skipper>>) -> Self {
        LexIter {
            skipper: skipper.into(),
            state: LexIterState {
                text: Rc::from(text),
                position: 0,
                indent_flag: true, // Start with indent flag set
                current_line: 1,
                current_column: 0,
                current_pos: 0,
                current_indent: 0,
            },
        }
    }
}

impl LexIterTrait for LexIter {
    fn get_state(&self) -> LexIterState {
        self.state.clone()
    }
}

impl Iterator for LexIter {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        self.skipper.next(&mut self.state)
    }
}

impl<'a> From<Chars<'a>> for LexIter {
    fn from(iter: Chars<'a>) -> Self {
        // Convert Chars to a String and then to Rc<str>
        let text = iter.collect::<String>();
        LexIter::new(text.as_str().into(), NoSkipper)
    }
}

impl From<String> for LexIter {
    fn from(text: String) -> Self {
        LexIter::new(text.as_str(), NoSkipper)
    }
}

pub fn token<E: ParserError + 'static, A: 'static>(
    p: Parsec<LexIter, E, A>,
) -> Parsec<LexIter, E, A> {
    Parsec::new(move |mut input: LexIter| {
        input.skipper.skip(&mut input.state);
        let original_skipper = input.skipper.clone_box();
        input.skipper = NoSkipper.into();
        let (mut result, rest) = p.eval(input)?;
        result.skipper = original_skipper;
        result.skipper.skip(&mut result.state);
        Ok((result, rest))
    })
}

impl<E: ParserError + 'static, A: 'static> Parsec<LexIter, E, A> {
    pub fn test<'a>(&self, input: impl Into<String>) -> Result<A, E> {
        let input = LexIter::new(&input.into(), NoSkipper);
        self.run(input)
    }
    pub fn parse<'a>(&self, input: impl Into<&'a str>, skipper: impl AsSkipper) -> Result<A, E> {
        let input = LexIter::new(input.into(), skipper.as_skipper());
        self.run(input)
    }
    pub fn parse_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
        skipper: impl AsSkipper,
    ) -> Result<A, E> {
        use std::fs::read_to_string;
        let content = read_to_string(path).unwrap();
        let input = LexIter::new(&content, skipper.as_skipper());
        self.run(input)
    }
    pub fn integer(radix: u32) -> Parsec<LexIter, E, i64> {
        token(
            digit(radix)
                .many1()
                .collect::<String>()
                .map(move |s| i64::from_str_radix(&s, radix).unwrap())
                .expected("integer"),
        )
    }

    pub fn symbol(expected: &'static str) -> Parsec<LexIter, E, String> {
        let expected_str = expected.to_string();
        token(
            Parsec::new(move |input: LexIter| {
                let original_state = input.get_state();
                let mut current_input = input;
                let mut matched = String::new();

                for expected_char in expected_str.chars() {
                    match current_input.next() {
                        Some(c) if c == expected_char => {
                            matched.push(c);
                        }
                        Some(c) => {
                            return Err(E::unexpected(
                                (original_state, current_input.get_state()),
                                &c,
                            )
                            .with_expected(&expected_str));
                        }
                        None => {
                            return Err(E::eof((original_state, current_input.get_state())));
                        }
                    }
                }

                Ok((current_input, matched))
            })
            .expected(expected),
        )
    }
}
