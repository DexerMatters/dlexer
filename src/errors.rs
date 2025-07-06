//! Error handling types for the parser combinator library.
//!
//! This module defines the core traits and structs for representing and managing
//! parsing errors. The central component is the [`ParserError`] trait, which
//! provides a standard interface for creating and manipulating errors throughout
//! the parsing process.
//!
//! # Key Components
//!
//! - **[`ParserError`]**: A trait that defines the contract for parser errors,
//!   including methods for creating errors for unexpected input, end-of-file,
//!   and for adding contextual information like expected tokens.
//!
//! - **[`SimpleParserError`]**: A concrete implementation of `ParserError` that
//!   provides a basic error structure with "expected" and "unexpected" messages,
//!   along with location information (line and column).

use std::fmt::Display;

use crate::lex::{LexIterState, Skipper};

/// A trait for defining custom parser errors.
///
/// This trait provides a flexible way to create and manage errors during parsing.
/// Implementors can define how errors are constructed and displayed.
pub trait ParserError: Display {
    /// The context type associated with the error (e.g., `str` or `[u8]`).
    type Context: ?Sized;
    /// Creates an error for an unexpected end of input.
    fn eof(state: (LexIterState<Self::Context>, LexIterState<Self::Context>)) -> Self
    where
        Self: Sized,
    {
        Self::unexpected(state, EOFError)
    }
    /// Creates an error for an unexpected item.
    ///
    /// # Parameters
    /// - `state`: A tuple containing the start and end states of the input where the error occurred.
    /// - `item`: The unexpected item that was found.
    fn unexpected<T: Display>(
        state: (LexIterState<Self::Context>, LexIterState<Self::Context>),
        item: T,
    ) -> Self;

    /// Adds an "expected" description to the error.
    ///
    /// This is used to provide more helpful error messages, such as "Expected 'identifier', found '123'".
    fn with_expected<T: Display>(self, expected: T) -> Self
    where
        Self: Sized;

    /// Sets the state (location) of the error.
    fn with_state(
        self,
        from: LexIterState<Self::Context>,
        to: LexIterState<Self::Context>,
    ) -> Self;

    /// Returns the starting state of the input where the error occurred.
    fn from(&self) -> LexIterState<Self::Context>;

    /// Returns the ending state of the input where the error occurred.
    fn to(&self) -> LexIterState<Self::Context> {
        self.from()
    }

    /// Returns the skipper associated with the error, if any.
    fn skipper(&self) -> Option<&dyn Skipper> {
        None
    }
}

#[derive(Debug, Clone)]
struct EOFError;

impl Display for EOFError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "end of input")
    }
}

/// A simple, concrete implementation of the `ParserError` trait.
///
/// This error type includes information about what was expected, what was found,
/// and the location of the error in the input.
#[derive(Debug)]
pub struct SimpleParserError<T: ?Sized> {
    expected: String,
    unexpected: String,
    from: LexIterState<T>,
    to: LexIterState<T>,
}

impl<T: ?Sized> Clone for SimpleParserError<T> {
    fn clone(&self) -> Self {
        Self {
            expected: self.expected.clone(),
            unexpected: self.unexpected.clone(),
            from: LexIterState::clone(&self.from),
            to: LexIterState::clone(&self.to),
        }
    }
}

impl<T: ?Sized> SimpleParserError<T> {
    /// Creates a new `SimpleParserError`.
    pub fn new(
        expected: String,
        unexpected: String,
        state: (LexIterState<T>, LexIterState<T>),
    ) -> Self {
        Self {
            expected,
            unexpected,
            from: state.0,
            to: state.1,
        }
    }
}

impl<U: ?Sized> ParserError for SimpleParserError<U> {
    type Context = U;
    fn unexpected<T>(
        state: (LexIterState<Self::Context>, LexIterState<Self::Context>),
        item: T,
    ) -> Self
    where
        T: Display,
    {
        Self::new(String::new(), item.to_string(), state)
    }

    fn with_expected<T>(mut self, expected: T) -> Self
    where
        T: Display,
    {
        self.expected = expected.to_string();
        self
    }
    fn with_state(
        mut self,
        from: LexIterState<Self::Context>,
        to: LexIterState<Self::Context>,
    ) -> Self {
        self.from = from;
        self.to = to;
        self
    }
    fn from(&self) -> LexIterState<Self::Context> {
        self.from.clone()
    }
    fn to(&self) -> LexIterState<Self::Context> {
        self.to.clone()
    }
}

impl<T: ?Sized> Display for SimpleParserError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let from = format!("{}:{}", self.from.current_line, self.from.current_column);
        let to = format!("{}:{}", self.to.current_line, self.to.current_column);
        let expected = replace_escapes(&self.expected);
        let unexpected = replace_escapes(&self.unexpected);
        if self.expected.is_empty() {
            write!(f, "Unexpected '{}'", unexpected)?;
        } else {
            write!(f, "Expected '{}', found '{}'", expected, unexpected)?;
        }
        write!(f, " at {} to {}", from, to)
    }
}

/// Replaces special characters with their escaped versions for display.
fn replace_escapes(s: impl Display) -> String {
    s.to_string()
        .replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\t', "\\t")
        .replace('\r', "\\r")
        .replace('\0', "\\0")
}
