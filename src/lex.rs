//! Lexer utilities for tokenization and input stream handling.
//!
//! This module provides tools for lexical analysis, primarily focused on handling
//! whitespace, comments, and defining token-level parsers. It integrates with the
//! `parsec` module to create robust parsers that can ignore irrelevant input.
//!
//! # Key Components
//!
//! - **`Skipper` Trait**: Defines how to skip parts of the input. Implementations
//!   are provided for common cases like whitespace (`WhitespaceSkipper`), line
//!   comments (`LineSkipper`), and block comments (`BlockSkipper`).
//!
//! - **`LexIter`**: A stateful iterator over string input that tracks position
//!   (line, column) and uses a `Skipper` to ignore characters.
//!
//! - **`token` function**: A higher-order parser that wraps another parser to
//!   handle skipping before and after the token is parsed. This is crucial for
//!   building parsers for languages with free-form whitespace.
//!
//! # Basic Usage
//!
//! The main way to use this module is by creating a `Skipper` and passing it to
//! the `parse` method, or by using token parsers like `symbol`, `integer`, etc.,
//! which use `token` internally.
//!
//! ```
//! use dlexer::lex::{symbol, WhitespaceSkipper};
//! use dlexer::parsec::*;
//!
//! // A parser for the keyword "let", automatically skipping whitespace.
//! let let_keyword = symbol("let");
//!
//! // The `symbol` parser uses `token`, which uses a skipper.
//! // We can provide a skipper to the `parse` method.
//! let result = let_keyword.parse("  let  ", WhitespaceSkipper);
//!
//! assert_eq!(result.unwrap(), "let");
//! ```
use std::fmt::Debug;
use std::rc::Rc;
use std::str::Chars;

use crate::errors::ParserError;
use crate::parsec::{char, digit, fail, pure, Parsec};

/// A trait for defining how to skip over parts of the input stream.
///
/// Skippers are used to ignore characters like whitespace or comments during parsing,
/// allowing parsers to focus on meaningful tokens.
pub trait Skipper {
    /// Returns the next character from the input after skipping.
    fn next(&self, state: &mut LexIterState<str>) -> Option<char>;

    /// Skips characters from the current position in the input.
    ///
    /// Returns the number of characters skipped.
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

    /// Clones the skipper into a `Box<dyn Skipper>`.
    fn clone_box(&self) -> Box<dyn Skipper>;
}

/// A trait for converting a type into a boxed `Skipper`.
///
/// This allows for flexible composition of different skipper types.
pub trait AsSkipper {
    /// Converts `self` into a `Box<dyn Skipper>`.
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

/// A skipper that does nothing.
///
/// This is the default skipper, which does not skip any characters.
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

/// A skipper that skips whitespace characters, excluding newlines.
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

/// A skipper that skips a specific set of characters.
///
/// # Example
///
/// ```
/// use dlexer::lex::{CharSkipper, LexIter, LexIterTrait};
///
/// let skipper = CharSkipper([' ', '\t']);
/// let mut iter = LexIter::new("  \t  hello", skipper);
/// assert_eq!(iter.next(), Some('h'));
/// ```
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

/// A skipper that skips line comments.
///
/// It skips from a given line comment marker to the end of the line.
///
/// # Example
///
/// ```
/// use dlexer::lex::{LexIter, LexIterTrait, LineSkipper};
///
/// let skipper = LineSkipper("//");
/// let mut iter = LexIter::new("// comment\nhello", skipper);
/// assert_eq!(iter.next(), Some('\n')); // The newline is not consumed
/// ```
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

/// A skipper that skips nested blocks of comments.
///
/// It skips from a given start delimiter to an end delimiter, handling nested blocks correctly.
///
/// # Example
///
/// ```
/// use dlexer::lex::{BlockSkipper, LexIter, LexIterTrait};
///
/// let skipper = BlockSkipper("/*", "*/");
/// let mut iter = LexIter::new("/* comment /* nested */ */hello", skipper);
/// assert_eq!(iter.next(), Some('h'));
/// ```
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

/// A collection of skippers that are applied in sequence.
///
/// The `Skippers` struct allows combining multiple skipper behaviors. It repeatedly
/// applies each skipper in its collection until no more characters can be skipped.
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

/// A trait for types that contain a `Skipper`.
pub trait HasSkipper {
    /// Returns a reference to the skipper.
    fn get_skipper(&self) -> &dyn Skipper;
    /// Sets the skipper.
    fn set_skipper(&mut self, skipper: Box<dyn Skipper>);
}

/// A trait for lexer iterators.
///
/// This defines the core interface for stateful iterators used by the parser combinators.
pub trait LexIterTrait {
    /// The type of the context being parsed (e.g., `str` or `[u8]`).
    type Context: ?Sized;
    /// The type of item produced by the iterator.
    type Item;
    /// Returns a clone of the current iterator state.
    fn get_state(&self) -> LexIterState<Self::Context>;
    /// Returns a mutable reference to the iterator state.
    fn get_state_mut(&mut self) -> &mut LexIterState<Self::Context>;
    /// Advances the iterator and returns the next item.
    fn next(&mut self) -> Option<Self::Item>;
}

/// A helper trait for creating default `Rc` values.
pub trait RcDefault {
    /// Creates a default `Rc` value.
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

/// Represents the state of a `LexIter`.
#[derive(Debug)]
pub struct LexIterState<T: ?Sized> {
    /// A reference-counted pointer to the input context (e.g., the string being parsed).
    pub context: Rc<T>,
    /// The current line number (1-based).
    pub current_line: usize,
    /// The current column number (0-based).
    pub current_column: usize,
    /// The current character position (0-based).
    pub current_pos: usize,
    /// The current indentation level.
    pub current_indent: usize,
    /// A flag indicating if the iterator is at the start of a line, used for indentation tracking.
    pub indent_flag: bool,
    /// The byte position in the context string.
    pub position: usize,
}

impl<T: ?Sized> LexIterState<T> {
    pub fn new(context: impl Into<Rc<T>>) -> Self {
        LexIterState {
            context: context.into(),
            current_line: 1,
            current_column: 0,
            current_pos: 0,
            current_indent: 0,
            indent_flag: true,
            position: 0,
        }
    }
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

impl<T> LexIterState<[T]> {
    pub fn context_len(&self) -> usize {
        self.context.len()
    }
}

impl LexIterState<str> {
    pub fn context_len(&self) -> usize {
        self.context.len()
    }
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

/// A lexer iterator over a string slice.
///
/// `LexIter` is the default iterator for parsing text. It tracks position,
/// handles indentation, and uses a `Skipper` to ignore irrelevant characters.
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
    /// Creates a new `LexIter` with a given context and skipper.
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

/// A parser combinator that treats a parser as a "token" parser.
///
/// It applies the configured skipper before and after running the given parser,
/// but ensures that no skipping occurs *within* the parser itself. This is
/// essential for parsing atomic tokens like identifiers or numbers where internal
/// whitespace is not allowed.
///
/// # Example
///
/// ```
/// use dlexer::lex::{symbol, token, WhitespaceSkipper};
/// use dlexer::parsec::*;
///
/// let parser = token(char('a') >> char('b'));
/// assert_eq!(parser.parse("ab", WhitespaceSkipper).unwrap(), 'b');
/// assert!(parser.parse("a b", WhitespaceSkipper).is_err());
/// 
/// ```
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
    /// Runs the parser on a string with no skipping.
    ///
    /// This is a convenience method for testing parsers on simple inputs.
    pub fn test(&self, input: impl Into<String>) -> Result<A, E> {
        let input = LexIter::new(&input.into(), NoSkipper);
        self.run(input)
    }
    /// Runs the parser on a string with a specified skipper.
    pub fn parse<'a>(&self, input: impl Into<&'a str>, skipper: impl AsSkipper) -> Result<A, E> {
        let input = LexIter::new(input.into(), skipper.as_skipper());
        self.run(input)
    }
    /// Reads a file into a string and runs the parser on its contents.
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

    /// A utility for debugging a parser.
    ///
    /// Wraps a parser to print its input, output, and state at each step.
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

/// Parses an integer from the input.
///
/// The parser is configured with a radix and handles token-level skipping.
///
/// # Example
///
/// ```
/// use dlexer::lex::{integer, WhitespaceSkipper};
/// use dlexer::parsec::*;
///
/// let hex_parser = integer(16);
/// let result = hex_parser.parse("  FF  ", WhitespaceSkipper);
/// assert_eq!(result.unwrap(), 255);
/// ```
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

/// Parses a floating-point number.
///
/// This parser expects a number with a decimal point (e.g., "123.45").
///
/// # Example
///
/// ```
/// use dlexer::lex::{float, WhitespaceSkipper};
/// use dlexer::parsec::*;
///
/// let parser = float();
/// let result = parser.parse("  3.14  ", WhitespaceSkipper);
/// assert_eq!(result.unwrap(), 3.14);
/// ```
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

/// Parses a number, which can be an integer or a float.
///
/// # Example
///
/// ```
/// use dlexer::lex::{number, WhitespaceSkipper};
/// use dlexer::parsec::*;
///
/// let parser = number();
/// assert_eq!(parser.parse("123", WhitespaceSkipper).unwrap(), 123.0);
/// assert_eq!(parser.parse("123.45", WhitespaceSkipper).unwrap(), 123.45);
/// ```
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

/// Parses a specific string symbol.
///
/// This is useful for parsing keywords or operators. It is wrapped with `token`
/// to handle surrounding whitespace automatically.
///
/// # Example
///
/// ```
/// use dlexer::lex::{symbol, WhitespaceSkipper};
/// use dlexer::parsec::*;
///
/// let parser = symbol("if");
/// let result = parser.parse("  if  ", WhitespaceSkipper);
/// assert_eq!(result.unwrap(), "if");
/// ```
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
