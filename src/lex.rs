use std::fmt::Debug;
use std::rc::Rc;
use std::str::Chars;

use crate::errors::ParserError;
use crate::parsec::{Parsec, char, digit, fail, pure};

pub trait Skipper {
    fn next(&self, state: &mut LexIterState<str>) -> Option<char>;

    fn skip(&self, state: &mut LexIterState<str>) -> usize {
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
    fn skip(&self, _state: &mut LexIterState<str>) -> usize {
        0
    }
    fn next(&self, state: &mut LexIterState<str>) -> Option<char> {
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
    fn skip(&self, state: &mut LexIterState<str>) -> usize {
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
    fn next(&self, state: &mut LexIterState<str>) -> Option<char> {
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
    fn skip(&self, state: &mut LexIterState<str>) -> usize {
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
    fn next(&self, state: &mut LexIterState<str>) -> Option<char> {
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
    fn skip(&self, state: &mut LexIterState<str>) -> usize {
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
    fn next(&self, state: &mut LexIterState<str>) -> Option<char> {
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
    fn skip(&self, state: &mut LexIterState<str>) -> usize {
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
    fn next(&self, state: &mut LexIterState<str>) -> Option<char> {
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
    fn skip(&self, state: &mut LexIterState<str>) -> usize {
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
    fn next(&self, state: &mut LexIterState<str>) -> Option<char> {
        let _ = Self::skip(self, state);
        state.next_one_char()
    }
    fn clone_box(&self) -> Box<dyn Skipper> {
        Box::new(Skippers {
            skippers: self.skippers.iter().map(|s| s.clone_box()).collect(),
        })
    }
}

pub trait HasSkipper {
    fn get_skipper(&self) -> &dyn Skipper;
    fn set_skipper(&mut self, skipper: Box<dyn Skipper>);
}

pub trait LexIterTrait {
    type Context: ?Sized;
    type Item;
    fn get_state(&self) -> LexIterState<Self::Context>;
    fn get_state_mut(&mut self) -> &mut LexIterState<Self::Context>;
    fn next(&mut self) -> Option<Self::Item>;
}

pub trait RcDefault {
    fn default() -> Self;
}

impl RcDefault for Rc<str> {
    fn default() -> Self {
        Rc::from("")
    }
}

impl<T> RcDefault for Rc<T>
where
    T: ?Sized + Default,
{
    fn default() -> Self {
        Rc::new(T::default())
    }
}

#[derive(Debug)]
pub struct LexIterState<T: ?Sized> {
    pub context: Rc<T>,
    pub current_line: usize,
    pub current_column: usize,
    pub current_pos: usize,
    pub current_indent: usize,
    pub indent_flag: bool,
    pub position: usize,
}

impl<T: ?Sized> Clone for LexIterState<T> {
    fn clone(&self) -> Self {
        LexIterState {
            context: Rc::clone(&self.context),
            current_line: self.current_line,
            current_column: self.current_column,
            current_pos: self.current_pos,
            current_indent: self.current_indent,
            indent_flag: self.indent_flag,
            position: self.position,
        }
    }
}

impl<T: ?Sized> Default for LexIterState<T>
where
    Rc<T>: RcDefault,
{
    fn default() -> Self {
        LexIterState {
            context: RcDefault::default(),
            current_line: 1,
            current_column: 0,
            current_pos: 0,
            current_indent: 0,
            indent_flag: true, // Start with indent flag set
            position: 0,
        }
    }
}

impl LexIterState<str> {
    fn next_one_char(&mut self) -> Option<char> {
        if self.position >= self.context.len() {
            return None;
        }

        // Get the character at the current position
        let c = self.context[self.position..].chars().next().unwrap();
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
    pub(crate) state: LexIterState<str>,
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
    pub fn new<'a>(context: &'a str, skipper: impl Into<Box<dyn Skipper>>) -> Self {
        LexIter {
            skipper: skipper.into(),
            state: LexIterState {
                context: Rc::from(context),
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
    type Context = str;
    type Item = char;
    fn get_state(&self) -> LexIterState<Self::Context> {
        self.state.clone()
    }
    fn get_state_mut(&mut self) -> &mut LexIterState<Self::Context> {
        &mut self.state
    }
    fn next(&mut self) -> Option<Self::Item> {
        let skipper = &*self.skipper;
        skipper.next(&mut self.state)
    }
}

impl HasSkipper for LexIter {
    fn get_skipper(&self) -> &dyn Skipper {
        &*self.skipper
    }
    fn set_skipper(&mut self, skipper: Box<dyn Skipper>) {
        self.skipper = skipper;
    }
}

impl<'a> From<Chars<'a>> for LexIter {
    fn from(iter: Chars<'a>) -> Self {
        // Convert Chars to a String and then to Rc<str>
        let context = iter.collect::<String>();
        LexIter::new(context.as_str().into(), NoSkipper)
    }
}

impl From<String> for LexIter {
    fn from(context: String) -> Self {
        LexIter::new(context.as_str(), NoSkipper)
    }
}

pub fn token<S, E, A>(p: Parsec<S, E, A>) -> Parsec<S, E, A>
where
    S: LexIterTrait<Context = str> + HasSkipper + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
{
    Parsec::new(move |mut input: S| {
        let skipper = input.get_skipper().clone_box();
        skipper.skip(input.get_state_mut());
        let original_skipper = input.get_skipper().clone_box();
        input.set_skipper(NoSkipper.into());
        let (mut result, rest) = p.eval(input)?;
        result.set_skipper(original_skipper);
        let skipper = result.get_skipper().clone_box();
        skipper.skip(result.get_state_mut());
        Ok((result, rest))
    })
}

impl<E, A> Parsec<LexIter, E, A>
where
    E: ParserError<Context = str> + 'static,
    A: 'static,
{
    pub fn test(&self, input: impl Into<String>) -> Result<A, E> {
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

    pub fn dbg(self) -> Parsec<LexIter, E, A>
    where
        A: Debug,
    {
        Parsec::new(move |input: LexIter| {
            let original = input.get_state();
            println!("Input:");
            println!(
                "  Position:\t{}:{}",
                original.current_line, original.current_column
            );
            let mut rest = original
                .context
                .get(original.current_pos..)
                .unwrap_or("End of input");
            if rest.len() > 8 {
                rest = &rest[..8];
                println!("  Parsing:\t{:?}...", rest);
            } else {
                println!("  Parsing:\t{:?}", rest);
            }

            let result = self.eval(input.clone());
            match result {
                Ok((next_input, value)) => {
                    let next = next_input.get_state();
                    println!("Output:");
                    println!("  Value:\t{:?}", value);
                    println!("  Position:\t{}:{}", next.current_line, next.current_column);
                    println!("  Indentation:\t{}", next.current_indent);
                    let mut next_rest = next
                        .context
                        .get(next.current_pos..)
                        .unwrap_or("End of input");
                    if next_rest.len() > 8 {
                        next_rest = &next_rest[..8];
                        println!("  Remaining:\t{:?}...", next_rest);
                    } else {
                        println!("  Remaining:\t{:?}", next_rest);
                    }
                    Ok((next_input, value))
                }
                Err(error) => {
                    println!("Error:");
                    println!("  Message:\t{}", error);
                    Err(error)
                }
            }
        })
    }
}

pub fn integer<S, E>(radix: u32) -> Parsec<S, E, i64>
where
    S: LexIterTrait<Context = str, Item = char> + Clone + HasSkipper + 'static,
    E: ParserError<Context = S::Context> + Clone + 'static,
    Rc<<E as ParserError>::Context>: RcDefault,
{
    token(
        digit(radix)
            .many1()
            .collect::<String>()
            .bind(move |s| {
                if let Ok(num) = i64::from_str_radix(&s, radix) {
                    pure(num)
                } else {
                    fail(E::unexpected(
                        (LexIterState::default(), LexIterState::default()),
                        &s,
                    ))
                }
            })
            .expected("integer"),
    )
}

pub fn float<S, E>() -> Parsec<S, E, f64>
where
    S: LexIterTrait<Context = str, Item = char> + Clone + HasSkipper + 'static,
    E: ParserError<Context = S::Context> + Clone + 'static,
    Rc<<E as ParserError>::Context>: RcDefault,
{
    let integral = digit(10).many();
    let fractional = char('.') >> digit(10).many1();
    token(
        integral
            .concat(fractional)
            .collect::<String>()
            .bind(|s| {
                if let Ok(num) = s.parse::<f64>() {
                    pure(num)
                } else {
                    fail(E::unexpected(
                        (LexIterState::default(), LexIterState::default()),
                        &s,
                    ))
                }
            })
            .expected("float"),
    )
}

pub fn number<S, E>() -> Parsec<S, E, f64>
where
    S: LexIterTrait<Context = str, Item = char> + Clone + HasSkipper + 'static,
    E: ParserError<Context = S::Context> + Clone + 'static,
    Rc<<E as ParserError>::Context>: RcDefault,
{
    let integral = digit(10).many();
    let fractional = || char('.') >> digit(10).many1();
    token(
        (fractional()
            | integral.concat(
                fractional() //
                    .try_()
                    .unwrap_or_default(),
            ))
        .collect::<String>()
        .bind(|s| {
            if let Ok(num) = s.parse::<f64>() {
                pure(num)
            } else {
                fail(E::unexpected(
                    (LexIterState::default(), LexIterState::default()),
                    &s,
                ))
            }
        })
        .expected("number"),
    )
}

pub fn symbol<S, E>(expected: &str) -> Parsec<S, E, String>
where
    S: LexIterTrait<Context = str, Item = char> + Clone + HasSkipper + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    let expected_ = expected.to_string();
    let expected_clone = expected_.clone();
    token(
        Parsec::new(move |input: S| {
            let original_state = input.get_state();
            let mut current_input = input;
            let mut matched = String::new();

            for expected_char in expected_.chars() {
                match current_input.next() {
                    Some(c) if c == expected_char => {
                        matched.push(c);
                    }
                    Some(c) => {
                        return Err(
                            E::unexpected((original_state, current_input.get_state()), &c)
                                .with_expected(&expected_),
                        );
                    }
                    None => {
                        return Err(E::eof((original_state, current_input.get_state())));
                    }
                }
            }

            Ok((current_input, matched))
        })
        .expected(expected_clone),
    )
}
