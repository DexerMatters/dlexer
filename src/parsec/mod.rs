//! Parser combinator library for building composable parsers.
//!
//! This module provides a functional approach to parsing using combinators.
//! Combinators can be composed together to build complex parsers from simple
//! building blocks.
//!
//! The core type is [`Parsec`], which represents a parser that:
//! - Consumes input of type `S` (which implements [`LexIterTrait`])
//! - May produce errors of type `E` (which implements [`ParserError`])
//! - On success, produces values of type `A`
//!
//! # Basic Usage
//!
//! ```
//! use dlexer::parsec::*;
//!
//! // Parse a single character
//! let parser = char('a');
//!
//! // Combine parsers
//! let combined = char('a').then(char('b'));
//!
//! // Use operator syntax
//! let using_operators = char('a') >> char('b');
//!
//! // Transform results
//! let mapped = char('a').map(|c| c.to_uppercase().next().unwrap());
//! ```

pub mod extra;

use std::{
    fmt::{Debug, Display},
    ops::{
        Add, BitAnd, BitOr, Range, RangeBounds, RangeFrom, RangeInclusive, RangeTo,
        RangeToInclusive, Shl, Shr,
    },
    rc::Rc,
};

use crate::{
    errors::{ParserError, SimpleParserError},
    lex::{LexIter, LexIterState, LexIterTrait},
};

pub type BuildParser<S, E> = Parsec<S, E, <S as Iterator>::Item>;

/// A basic parser type alias using the default lexer iterator and error types.
pub type BasicParser = Parsec<LexIter, SimpleParserError<str>, <LexIter as LexIterTrait>::Item>;

/// Trait for types that can be converted to ranges for use with parser combinators.
///
/// This enables methods like `take` to accept different kinds of range specifications.
pub trait Take<T> {
    /// The range type this converts to.
    type Output: RangeBounds<T>;

    /// Converts this type into a range bounds.
    fn as_range(self) -> Self::Output;
}

impl Take<usize> for Range<usize> {
    type Output = Range<usize>;
    fn as_range(self) -> Self::Output {
        self
    }
}

impl Take<usize> for RangeInclusive<usize> {
    type Output = RangeInclusive<usize>;
    fn as_range(self) -> Self::Output {
        self
    }
}

impl Take<usize> for RangeToInclusive<usize> {
    type Output = RangeToInclusive<usize>;
    fn as_range(self) -> Self::Output {
        self
    }
}

impl Take<usize> for RangeTo<usize> {
    type Output = RangeTo<usize>;
    fn as_range(self) -> Self::Output {
        self
    }
}

impl Take<usize> for RangeFrom<usize> {
    type Output = RangeFrom<usize>;
    fn as_range(self) -> Self::Output {
        self
    }
}

impl Take<usize> for usize {
    type Output = RangeInclusive<usize>;
    fn as_range(self) -> Self::Output {
        self - 1..=self - 1
    }
}

/// A parser combinator that consumes input and produces results.
///
/// `Parsec<S, E, A>` represents a parser that:
/// - Consumes input of type `S` (which implements [`LexIterTrait`])
/// - May produce errors of type `E` (which implements [`ParserError`])
/// - Produces values of type `A` on success
#[derive(Clone)]
pub struct Parsec<S: LexIterTrait, E: ParserError, A> {
    run: Rc<dyn Fn(S) -> Result<(S, A), E>>,
}

impl<S, E, A> Parsec<S, E, A>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
{
    /// Creates a new parser from a function.
    ///
    /// The function should take input state and return either:
    /// - `Ok((remaining_input, value))` on success
    /// - `Err(error)` on failure
    pub fn new(run: impl Fn(S) -> Result<(S, A), E> + 'static) -> Self {
        Parsec { run: Rc::new(run) }
    }

    /// Evaluates the parser on the given input.
    ///
    /// Returns both the remaining input and the parsed value on success.
    pub fn eval(&self, input: S) -> Result<(S, A), E> {
        (self.run)(input)
    }

    /// Runs the parser on the given input, returning only the parsed value.
    pub fn run(&self, input: S) -> Result<A, E> {
        (self.run)(input).map(|(_, value)| value)
    }

    /// Transforms the parsed value using the given function.
    ///
    /// This is the functor operation for parsers.
    ///
    /// # Example
    ///
    /// ```
    /// use dlexer::parsec::*;
    ///
    /// let parser = decimal_digit().map(|c| c.to_digit(10).unwrap());
    /// assert_eq!(parser.test("7").unwrap(), 7);
    /// ```
    pub fn map<B: 'static, F>(self, f: F) -> Parsec<S, E, B>
    where
        F: Fn(A) -> B + 'static,
    {
        Parsec::new(move |input| {
            let (next_input, value) = self.eval(input)?;
            Ok((next_input, f(value)))
        })
    }

    /// Sets the expected description for error reporting.
    ///
    /// When this parser fails, the error will include the provided expectation.
    pub fn expected<T>(self, expected: T) -> Parsec<S, E, A>
    where
        T: Display + Clone + 'static,
        E: ParserError<Context = S::Context> + 'static,
    {
        Parsec::new(move |input: S| {
            let original_state = input.get_state();
            let result = self.eval(input);
            match result {
                Ok((next_input, value)) => Ok((next_input, value)),
                Err(error) => {
                    let to = error.to();
                    Err(error
                        .with_expected(expected.clone())
                        .with_state(original_state, to))
                }
            }
        })
    }

    /// Chains this parser with another parser-producing function.
    ///
    /// This is the monadic bind operation for parsers.
    ///
    /// # Example
    ///
    /// ```
    /// use dlexer::parsec::*;
    ///
    /// // Parse a digit, then parse that many 'x' characters.
    /// let parser = decimal_digit()
    ///     .map(|c| c.to_digit(10).unwrap() as usize)
    ///     .bind(|n| char('x').take(n));
    ///
    /// assert_eq!(parser.test("3xxx").unwrap().len(), 3);
    /// ```
    pub fn bind<B: 'static, F>(self, f: F) -> Parsec<S, E, B>
    where
        F: Fn(A) -> Parsec<S, E, B> + 'static,
    {
        Parsec::new(move |input| {
            let (next_input, value) = self.eval(input)?;
            f(value).eval(next_input)
        })
    }

    /// Applies a function parser to an argument parser.
    ///
    /// This is the applicative apply operation.
    pub fn apply<B: 'static, R: 'static>(self, arg: Parsec<S, E, B>) -> Parsec<S, E, R>
    where
        A: FnOnce(B) -> R + 'static,
    {
        Parsec::new(move |input| {
            let (next_input, func) = self.eval(input)?;
            let (final_input, arg_value) = arg.eval(next_input)?;
            Ok((final_input, func(arg_value)))
        })
    }

    /// Sequences this parser with another, keeping only the second result.
    ///
    /// Runs this parser followed by the other parser, returning the result of the other parser.
    ///
    /// # Example
    ///
    /// ```
    /// use dlexer::parsec::*;
    ///
    /// let parser = char('a').then(char('b')); // equivalent to char('a') >> char('b')
    /// assert_eq!(parser.test("ab").unwrap(), 'b');
    /// ```
    pub fn then<B: 'static>(self, other: Parsec<S, E, B>) -> Parsec<S, E, B> {
        Parsec::new(move |input| {
            let (next_input, _) = self.eval(input)?;
            other.eval(next_input)
        })
    }

    /// Sequences this parser with another, keeping only the first result.
    ///
    /// Runs this parser followed by the other parser, returning the result of this parser.
    ///
    /// # Example
    ///
    /// ```
    /// use dlexer::parsec::*;
    ///
    /// let parser = char('a').with(char('b')); // equivalent to char('a') << char('b')
    /// assert_eq!(parser.test("ab").unwrap(), 'a');
    /// ```
    pub fn with<B: 'static>(self, other: Parsec<S, E, B>) -> Parsec<S, E, A> {
        Parsec::new(move |input| {
            let (next_input, value) = self.eval(input)?;
            let (final_input, _) = other.eval(next_input)?;
            Ok((final_input, value))
        })
    }

    /// Parses this parser between two other parsers.
    ///
    /// Equivalent to `left.then(self).with(right)`.
    ///
    /// # Example
    ///
    /// ```
    /// use dlexer::parsec::*;
    ///
    /// let content = alpha().many1().collect::<String>();
    /// let parser = content.between(char('('), char(')'));
    /// assert_eq!(parser.test("(hello)").unwrap(), "hello");
    /// ```
    pub fn between<B: 'static, C: 'static>(
        self,
        left: Parsec<S, E, B>,
        right: Parsec<S, E, C>,
    ) -> Parsec<S, E, A> {
        Parsec::new(move |input: S| {
            let (next_input, _) = left.eval(input)?;
            let (final_input, value) = self.eval(next_input)?;
            let (end_input, _) = right.eval(final_input)?;
            Ok((end_input, value))
        })
    }

    /// Tries this parser, falling back to another on failure.
    ///
    /// If this parser fails, tries the other parser on the original input.
    ///
    /// # Example
    ///
    /// ```
    /// use dlexer::parsec::*;
    ///
    /// let parser = char('a').or(char('b')); // equivalent to char('a') | char('b')
    /// assert_eq!(parser.test("a").unwrap(), 'a');
    /// assert_eq!(parser.test("b").unwrap(), 'b');
    /// assert!(parser.test("c").is_err());
    /// ```
    pub fn or(self, other: Parsec<S, E, A>) -> Parsec<S, E, A>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| match self.eval(input.clone()) {
            Ok((next_input, value)) => Ok((next_input, value)),
            Err(err) => match other.eval(input) {
                Ok((next_input, value)) => Ok((next_input, value)),
                Err(_) => Err(err),
            },
        })
    }

    /// Makes this parser optional, returning `None` on failure.
    ///
    /// If the parser succeeds, returns `Some(value)`, otherwise returns `None`.
    ///
    /// # Example
    ///
    /// ```
    /// use dlexer::parsec::*;
    ///
    /// let parser = char('a').opt();
    /// assert_eq!(parser.test("a").unwrap(), Some('a'));
    /// assert_eq!(parser.test("b").unwrap(), None);
    /// ```
    pub fn opt(self) -> Parsec<S, E, Option<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| match self.eval(input.clone()) {
            Ok((next_input, value)) => Ok((next_input, Some(value))),
            Err(_) => Ok((input, None)),
        })
    }

    /// Wraps the result in a single-element vector.
    ///
    /// Useful as a building block for parsers that return vectors.
    pub fn one(self) -> Parsec<S, E, Vec<A>> {
        Parsec::new(move |input: S| {
            let (next_input, value) = self.eval(input)?;
            Ok((next_input, vec![value]))
        })
    }

    /// Converts parse errors into `Result` values.
    ///
    /// Instead of failing, returns `Err(error)` in the success value.
    pub fn try_(self) -> Parsec<S, E, Result<A, E>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| match self.eval(input.clone()) {
            Ok((next_input, value)) => Ok((next_input, Ok(value))),
            Err(error) => Ok((input, Err(error))),
        })
    }

    /// Applies this parser zero or more times.
    ///
    /// Collects all successful parses into a vector.
    ///
    /// # Example
    ///
    /// ```
    /// use dlexer::parsec::*;
    ///
    /// let parser = char('a').many();
    /// assert_eq!(parser.test("aaab").unwrap(), vec!['a', 'a', 'a']);
    /// assert_eq!(parser.test("b").unwrap(), vec![]);
    /// ```
    pub fn many(self) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let mut results = Vec::new();
            let mut current_input = input;
            while let Ok((new_input, value)) = self.eval(current_input.clone()) {
                results.push(value);
                current_input = new_input;
            }
            Ok((current_input, results))
        })
    }

    /// Applies this parser one or more times.
    ///
    /// Must succeed at least once, then collects all successful parses into a vector.
    ///
    /// # Example
    ///
    /// ```
    /// use dlexer::parsec::*;
    ///
    /// let parser = char('a').many1();
    /// assert_eq!(parser.test("aaab").unwrap(), vec!['a', 'a', 'a']);
    /// assert!(parser.test("b").is_err());
    /// ```
    pub fn many1(self) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
        A: Debug,
    {
        Parsec::new(move |input: S| {
            let (mut current_input, first_value) = self.eval(input)?;
            let mut results = vec![first_value];

            while let Ok((new_input, value)) = self.eval(current_input.clone()) {
                results.push(value);
                current_input = new_input;
            }

            Ok((current_input, results))
        })
    }

    /// Applies this parser a specific number of times based on a range.
    ///
    /// The range specifies minimum and maximum number of repetitions.
    ///
    /// # Example
    ///
    /// ```
    /// use dlexer::parsec::*;
    ///
    /// let parser = char('a').take(2..=3).collect::<String>();
    /// assert_eq!(parser.test("aa").unwrap(), "aa");
    /// assert_eq!(parser.test("aaa").unwrap(), "aaa");
    /// assert!(parser.test("a").is_err());
    /// ```
    pub fn take<R>(self, range: R) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
        R: Take<usize> + 'static,
    {
        let range = range.as_range();
        Parsec::new(move |input: S| {
            let mut results = Vec::new();
            let mut current_input = input;
            let mut count = 0;

            let start = match range.start_bound() {
                std::ops::Bound::Included(&n) => n,
                std::ops::Bound::Excluded(&n) => n + 1,
                std::ops::Bound::Unbounded => 0,
            };

            let end = match range.end_bound() {
                std::ops::Bound::Included(&n) => n + 1,
                std::ops::Bound::Excluded(&n) => n,
                std::ops::Bound::Unbounded => usize::MAX,
            };

            while count < end {
                match self.eval(current_input.clone()) {
                    Ok((new_input, value)) => {
                        results.push(value);
                        current_input = new_input;
                        count += 1;
                    }
                    Err(err) => {
                        if count < start {
                            return Err(err);
                        }
                        break;
                    }
                }
            }

            if count < start {
                return Err(E::eof((
                    current_input.get_state(),
                    current_input.get_state(),
                )));
            }

            Ok((current_input, results))
        })
    }

    /// Applies this parser repeatedly until another parser succeeds.
    ///
    /// Collects the results into a vector, not including the end parser's result.
    ///
    /// # Example
    ///
    /// ```
    /// use dlexer::parsec::*;
    ///
    /// let parser = any().many_till(char(';')).collect::<String>();
    /// let result = parser.test("hello;world").unwrap();
    /// assert_eq!(result, "hello");
    /// ```
    pub fn many_till<B: 'static>(self, end: Parsec<S, E, B>) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let mut results = Vec::new();
            let mut current_input = input;

            // Already check if the end condition is met
            if end.eval(current_input.clone()).is_ok() {
                return Ok((current_input, results));
            }

            loop {
                match self.eval(current_input.clone()) {
                    Ok((new_input, value)) => {
                        results.push(value);
                        current_input = new_input;
                    }
                    Err(err) => return Err(err),
                }

                if end.eval(current_input.clone()).is_ok() {
                    return Ok((current_input, results));
                }
            }
        })
    }

    /// Applies this parser one or more times until another parser succeeds.
    ///
    /// Must succeed at least once, then continues until the end parser succeeds.
    pub fn many1_till<B: 'static>(self, end: Parsec<S, E, B>) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let (mut current_input, first_value) = self.eval(input)?;
            let mut results = vec![first_value];

            loop {
                if end.eval(current_input.clone()).is_ok() {
                    return Ok((current_input, results));
                }

                match self.eval(current_input.clone()) {
                    Ok((new_input, value)) => {
                        results.push(value);
                        current_input = new_input;
                    }
                    Err(_) => return Ok((current_input, results)),
                }
            }
        })
    }

    /// Parses zero or more occurrences of this parser separated by another parser.
    ///
    /// Returns the results of this parser in a vector, discarding the separator results.
    ///
    /// # Example
    ///
    /// ```
    /// use dlexer::parsec::*;
    ///
    /// let parser = decimal_digit().sep(char(','));
    /// assert_eq!(parser.test("1,2,3").unwrap(), vec!['1', '2', '3']);
    /// assert_eq!(parser.test("").unwrap(), vec![]);
    /// ```
    pub fn sep<T: 'static>(self, sep: Parsec<S, E, T>) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let mut results = Vec::new();
            let mut current_input = input;

            if let Ok((new_input, first_value)) = self.eval(current_input.clone()) {
                results.push(first_value);
                current_input = new_input;
            } else {
                return Ok((current_input, results));
            }

            while let Ok((sep_input, _)) = sep.eval(current_input.clone()) {
                if let Ok((new_input, value)) = self.eval(sep_input) {
                    results.push(value);
                    current_input = new_input;
                } else {
                    break;
                }
            }
            Ok((current_input, results))
        })
    }

    /// Parses one or more occurrences of this parser separated by another parser.
    ///
    /// Must succeed at least once, then parses separator-item pairs as long as possible.
    ///
    /// # Example
    ///
    /// ```
    /// use dlexer::parsec::*;
    ///
    /// let parser = decimal_digit().sep1(char(','));
    /// assert_eq!(parser.test("1,2,3").unwrap(), vec!['1', '2', '3']);
    /// assert!(parser.test("").is_err());
    /// ```
    pub fn sep1(self, sep: Parsec<S, E, S::Item>) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let (mut current_input, first_value) = self.eval(input)?;
            let mut results = vec![first_value];

            while let Ok((sep_input, _)) = sep.eval(current_input.clone()) {
                if let Ok((new_input, value)) = self.eval(sep_input) {
                    results.push(value);
                    current_input = new_input;
                } else {
                    break;
                }
            }
            Ok((current_input, results))
        })
    }

    /// Parses separated values until an end condition is met.
    ///
    /// Stops when the end parser succeeds, returning all parsed values.
    pub fn sep_till<T: 'static, U: 'static>(
        self,
        sep: Parsec<S, E, T>,
        end: Parsec<S, E, U>,
    ) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let mut results = Vec::new();
            let mut current_input = input;

            if end.eval(current_input.clone()).is_ok() {
                return Ok((current_input, results));
            }

            match self.eval(current_input.clone()) {
                Ok((new_input, value)) => {
                    results.push(value);
                    current_input = new_input;
                }
                Err(err) => return Err(err),
            }

            loop {
                if end.eval(current_input.clone()).is_ok() {
                    return Ok((current_input, results));
                }

                match sep.eval(current_input.clone()) {
                    Ok((sep_input, _)) => match self.eval(sep_input) {
                        Ok((new_input, value)) => {
                            results.push(value);
                            current_input = new_input;
                        }
                        Err(err) => return Err(err),
                    },
                    Err(err) => {
                        if end.eval(current_input.clone()).is_ok() {
                            return Ok((current_input, results));
                        } else {
                            return Err(err);
                        }
                    }
                }
            }
        })
    }

    /// Parses one or more separated values until an end condition is met.
    ///
    /// Must succeed at least once, then parses separator-item pairs until the end condition.
    pub fn sep1_till<T: 'static, U: 'static>(
        self,
        sep: Parsec<S, E, T>,
        end: Parsec<S, E, U>,
    ) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let (mut current_input, first_value) = self.eval(input)?;
            let mut results = vec![first_value];

            loop {
                if end.eval(current_input.clone()).is_ok() {
                    return Ok((current_input, results));
                }
                match sep.eval(current_input.clone()) {
                    Ok((sep_input, _)) => match self.eval(sep_input) {
                        Ok((new_input, value)) => {
                            results.push(value);
                            current_input = new_input;
                        }
                        Err(err) => return Err(err),
                    },
                    Err(err) => {
                        if end.eval(current_input.clone()).is_ok() {
                            return Ok((current_input, results));
                        } else {
                            return Err(err);
                        }
                    }
                }
            }
        })
    }

    /// Parses a specific number of separated values.
    ///
    /// The range specifies minimum and maximum number of items to parse.
    pub fn sep_take<T: 'static, R>(
        self,
        sep: Parsec<S, E, T>,
        range: impl Take<usize> + 'static,
    ) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        let range = range.as_range();
        Parsec::new(move |input: S| {
            let mut results = Vec::new();
            let mut current_input = input;
            let mut count = 0;

            let start = match range.start_bound() {
                std::ops::Bound::Included(&n) => n,
                std::ops::Bound::Excluded(&n) => n + 1,
                std::ops::Bound::Unbounded => 0,
            };

            let end = match range.end_bound() {
                std::ops::Bound::Included(&n) => n + 1,
                std::ops::Bound::Excluded(&n) => n,
                std::ops::Bound::Unbounded => usize::MAX,
            };

            if count < end {
                match self.eval(current_input.clone()) {
                    Ok((new_input, first_value)) => {
                        results.push(first_value);
                        current_input = new_input;
                        count += 1;
                    }
                    Err(err) => {
                        if start == 0 {
                            return Ok((current_input, results));
                        } else {
                            return Err(err);
                        }
                    }
                }
            }

            while count < end {
                match sep.eval(current_input.clone()) {
                    Ok((sep_input, _)) => match self.eval(sep_input) {
                        Ok((new_input, value)) => {
                            results.push(value);
                            current_input = new_input;
                            count += 1;
                        }
                        Err(err) => {
                            if count < start {
                                return Err(err);
                            }
                            break;
                        }
                    },
                    Err(err) => {
                        if count < start {
                            return Err(err);
                        }
                        break;
                    }
                }
            }

            if count < start {
                return Err(E::eof((
                    current_input.get_state(),
                    current_input.get_state(),
                )));
            }

            Ok((current_input, results))
        })
    }

    /// Chains parser applications with a binary operator.
    ///
    /// Repeatedly applies the operator and parser, combining results left-associatively.
    pub fn chain<F, B>(self, op: Parsec<S, E, F>, init: A) -> Parsec<S, E, A>
    where
        S: Clone,
        F: Fn(A, A) -> A + Clone + 'static,
        A: Clone,
    {
        Parsec::new(move |input: S| {
            let mut acc = init.clone();
            let mut inp = input.clone();
            loop {
                let op_res = op.eval(inp.clone());
                let val_res = self.eval(inp.clone());
                match (op_res, val_res) {
                    (Ok((_, f)), Ok((val_input, val))) => {
                        acc = f(acc, val);
                        inp = val_input;
                    }
                    _ => break,
                }
            }
            Ok((inp, acc))
        })
    }

    /// Chains parser applications with right associativity.
    ///
    /// Repeatedly applies the operator and parser, combining results right-associatively.
    pub fn chain_right<F>(self, op: Parsec<S, E, F>, init: A) -> Parsec<S, E, A>
    where
        S: Clone,
        F: Fn(A, A) -> A + Clone + 'static,
        A: Clone,
    {
        fn parse_rec<S: LexIterTrait, E, A, F>(
            parser: &Parsec<S, E, A>,
            op: &Parsec<S, E, F>,
            input: S,
            init: &A,
        ) -> Result<(S, A), E>
        where
            S: Clone + 'static,
            E: ParserError<Context = S::Context> + 'static,
            F: Fn(A, A) -> A + Clone + 'static,
            A: Clone + 'static,
        {
            match parser.eval(input.clone()) {
                Ok((next_input, x)) => match op.eval(next_input.clone()) {
                    Ok((op_input, f)) => {
                        let (rest_input, y) = parse_rec(parser, op, op_input, init)?;
                        Ok((rest_input, f(x, y)))
                    }
                    Err(_) => Ok((next_input, x)),
                },
                Err(_) => Ok((input, init.clone())),
            }
        }

        let parser = self;
        let op = op;
        let init = init;
        Parsec::new(move |input: S| parse_rec(&parser, &op, input, &init))
    }

    /// Combines this parser with another into a pair.
    ///
    /// Runs both parsers in sequence and returns both results as a tuple.
    pub fn pair<B: 'static>(self, other: Parsec<S, E, B>) -> Parsec<S, E, (A, B)> {
        Parsec::new(move |input: S| {
            let (next_input, value_a) = self.eval(input)?;
            let (final_input, value_b) = other.eval(next_input)?;
            Ok((final_input, (value_a, value_b)))
        })
    }

    /// Prepends a single value to a vector result.
    ///
    /// Runs this parser to get a single value, then runs the other parser to get a vector,
    /// and inserts the first value at the beginning of the vector.
    pub fn extend(self, other: Parsec<S, E, Vec<A>>) -> Parsec<S, E, Vec<A>> {
        Parsec::new(move |input: S| {
            let (next_input, value) = self.eval(input)?;
            let (final_input, mut other_values) = other.eval(next_input)?;
            other_values.insert(0, value);
            Ok((final_input, other_values))
        })
    }

    /// Filters parsed values based on a predicate.
    ///
    /// If the predicate returns false, the parser fails.
    pub fn hold<F>(self, f: F) -> Parsec<S, E, A>
    where
        F: Fn(&A) -> bool + 'static,
        A: Display,
        E: Clone + 'static,
    {
        Parsec::new(move |input: S| {
            let original_state = input.get_state();
            let (next_input, item) = self.eval(input)?;
            if f(&item) {
                Ok((next_input, item))
            } else {
                Err(E::unexpected(
                    (original_state, next_input.get_state()),
                    &item,
                ))
            }
        })
    }

    /// Succeeds only if the parsed value is not equal to the given value.
    ///
    /// If the parsed value equals the specified value, the parser fails.
    pub fn not(self, value: A) -> Parsec<S, E, A>
    where
        A: PartialEq + Display,
        E: Clone + 'static,
    {
        Parsec::new(move |input: S| {
            let original_state = input.get_state();
            let (next_input, item) = self.eval(input)?;
            if item == value {
                Err(
                    E::unexpected((original_state, next_input.get_state()), &item)
                        .with_expected(&value),
                )
            } else {
                Ok((next_input, item))
            }
        })
    }

    /// Succeeds only if the parsed value equals the given value.
    ///
    /// If the parsed value doesn't equal the specified value, the parser fails.
    pub fn is(self, value: A) -> Parsec<S, E, A>
    where
        A: PartialEq + Display,
        E: Clone + 'static,
    {
        Parsec::new(move |input: S| {
            let original_state = input.get_state();
            let (next_input, item) = self.eval(input)?;
            if item == value {
                Ok((next_input, item))
            } else {
                Err(
                    E::unexpected((original_state, next_input.get_state()), &item)
                        .with_expected(&value),
                )
            }
        })
    }

    /// Succeeds only if the parsed value is one of the given values.
    ///
    /// If the parsed value isn't in the provided iterator, the parser fails.
    pub fn one_of(self, values: impl Iterator<Item = A> + Clone + 'static) -> Parsec<S, E, A>
    where
        A: PartialEq + Display,
        E: Clone + 'static,
    {
        Parsec::new(move |input: S| {
            let original_state = input.get_state();
            let (next_input, item) = self.eval(input)?;
            let mut values = values.clone();
            if values.any(|v| v == item) {
                Ok((next_input, item))
            } else {
                Err(
                    E::unexpected((original_state, next_input.get_state()), &item).with_expected(
                        &values.map(|v| v.to_string()).collect::<Vec<_>>().join(", "),
                    ),
                )
            }
        })
    }

    /// Succeeds only if the parsed value is none of the given values.
    ///
    /// If the parsed value is in the provided iterator, the parser fails.
    pub fn none_of(self, values: impl Iterator<Item = A> + Clone + 'static) -> Parsec<S, E, A>
    where
        A: PartialEq + Display,
        E: 'static,
    {
        Parsec::new(move |input: S| {
            let original_state = input.get_state();
            let (next_input, item) = self.eval(input)?;
            if values.clone().any(|v| v == item) {
                Err(E::unexpected(
                    (original_state, next_input.get_state()),
                    &item,
                ))
            } else {
                Ok((next_input, item))
            }
        })
    }

    /// Converts the parsed value using the `Into` trait.
    ///
    /// Transforms the result type using the standard Rust conversion trait.
    pub fn into<B: 'static>(self) -> Parsec<S, E, B>
    where
        A: Into<B>,
    {
        self.map(|a| a.into())
    }

    /// Captures the parser state before and after parsing.
    ///
    /// Returns a tuple of ((start_state, end_state), value).
    pub fn states(self) -> Parsec<S, E, ((LexIterState<S::Context>, LexIterState<S::Context>), A)>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let original_state = input.get_state();
            let (next_input, value) = self.eval(input)?;
            let next_state = next_input.get_state();
            Ok((next_input, ((original_state, next_state), value)))
        })
    }
}

impl<S, E> Parsec<S, E, String>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    /// Converts the string to a static string slice by leaking memory.
    ///
    /// This is useful when you need a `&'static str`, but use with caution as it creates a memory leak.
    pub fn leak(self) -> Parsec<S, E, &'static str> {
        Parsec::new(move |input: S| {
            let (next_input, value) = self.eval(input)?;
            let leaked: &'static str = Box::leak(value.into_boxed_str());
            Ok((next_input, leaked))
        })
    }

    /// Trims whitespace from the parsed string.
    ///
    /// Removes leading and trailing whitespace from the string.
    pub fn trim(self) -> Parsec<S, E, String> {
        Parsec::new(move |input: S| {
            let (next_input, value) = self.eval(input)?;
            let trimmed = value.trim().to_string();
            Ok((next_input, trimmed))
        })
    }
}

impl<S, E, A> Parsec<S, E, Parsec<S, E, A>>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
{
    /// Flattens a nested parser by executing the inner parser.
    ///
    /// This is the monadic join operation.
    pub fn join(self) -> Parsec<S, E, A> {
        Parsec::new(move |input: S| {
            let (next_input, parser) = self.eval(input)?;
            parser.eval(next_input)
        })
    }
}

impl<S, E, A> Parsec<S, E, Vec<Parsec<S, E, A>>>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
{
    /// Sequences a vector of parsers, collecting their results.
    ///
    /// Runs each parser in the vector in sequence and collects the results.
    pub fn sequence(self) -> Parsec<S, E, Vec<A>> {
        Parsec::new(move |input: S| {
            let (mut current_input, parsers) = self.eval(input)?;
            let mut results = Vec::with_capacity(parsers.len());

            for parser in parsers {
                let (next_input, value) = parser.eval(current_input)?;
                results.push(value);
                current_input = next_input;
            }

            Ok((current_input, results))
        })
    }
}

impl<const N: usize, S, E, A> Parsec<S, E, [Parsec<S, E, A>; N]>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: Debug + 'static,
{
    /// Sequences an array of parsers, collecting their results into an array.
    ///
    /// Runs each parser in the array in sequence and collects the results.
    pub fn sequence(self) -> Parsec<S, E, [A; N]> {
        Parsec::new(move |input: S| {
            let (mut current_input, parsers) = self.eval(input)?;
            let mut results = Vec::with_capacity(N);

            for parser in parsers.into_iter() {
                let (next_input, value) = parser.eval(current_input)?;
                results.push(value);
                current_input = next_input;
            }

            // Convert Vec to array
            let array = results.try_into().unwrap();

            Ok((current_input, array))
        })
    }
}

impl<S, E, A> Parsec<S, E, Vec<A>>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
{
    /// Collects the vector into another collection type.
    ///
    /// Converts the vector of results into any type that implements `FromIterator<A>`.
    pub fn collect<B: 'static>(self) -> Parsec<S, E, B>
    where
        A: Into<B>,
        B: FromIterator<A>,
    {
        self.map(|vec| vec.into_iter().collect())
    }

    /// Appends a single value to the vector.
    ///
    /// Parses the vector, then parses a single value and adds it to the end.
    pub fn append(self, other: Parsec<S, E, A>) -> Parsec<S, E, Vec<A>> {
        Parsec::new(move |input: S| {
            let (next_input, mut values) = self.eval(input)?;
            let (final_input, value) = other.eval(next_input)?;
            values.push(value);
            Ok((final_input, values))
        })
    }

    /// Concatenates two vectors.
    ///
    /// Parses both vectors and concatenates them.
    pub fn concat(self, other: Parsec<S, E, Vec<A>>) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let (next_input, mut values) = self.eval(input)?;
            let (final_input, other_values) = other.eval(next_input)?;
            values.extend(other_values);
            Ok((final_input, values))
        })
    }
}

impl<S, E> Parsec<S, E, Vec<String>>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    /// Removes empty strings from the vector.
    ///
    /// Filters out any empty strings in the vector.
    pub fn trim(self) -> Parsec<S, E, Vec<String>> {
        Parsec::new(move |input: S| {
            let (next_input, mut values) = self.eval(input)?;
            values.retain_mut(|s| !s.is_empty());
            Ok((next_input, values))
        })
    }
}

// === Utility Functions ===

/// Creates a parser that always succeeds with the given value.
///
/// This is the monadic `return` or applicative `pure` operation.
pub fn pure<S, E, T>(value: T) -> Parsec<S, E, T>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    T: Clone + 'static,
{
    Parsec::new(move |input| Ok((input, value.clone())))
}

/// Creates a parser that always fails with the given error.
///
/// This is useful for handling error conditions explicitly.
pub fn fail<S, E, A>(error: E) -> Parsec<S, E, A>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static + Clone,
    A: 'static,
{
    Parsec::new(move |_| Err(error.clone()))
}

/// Parses any single input item.
///
/// Consumes and returns the next item from the input, or fails if at the end.
pub fn any<S, E>() -> Parsec<S, E, S::Item>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    S::Item: 'static,
{
    Parsec::new(move |mut input: S| {
        let original_state = input.get_state();
        if let Some(item) = input.next() {
            Ok((input, item))
        } else {
            Err(E::eof((original_state, input.get_state())))
        }
    })
}

/// Parses a single item that satisfies the given predicate.
///
/// Consumes the next item if it satisfies the predicate, otherwise fails.
///
/// # Example
///
/// ```
/// use dlexer::parsec::*;
///
/// let parser = satisfy(|c| c.is_alphabetic());
/// assert_eq!(parser.test("a").unwrap(), 'a');
/// assert!(parser.test("1").is_err());
/// ```
pub fn satisfy<S, E, F>(f: F) -> Parsec<S, E, S::Item>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    S::Item: Display + 'static,
    F: Fn(&S::Item) -> bool + 'static,
{
    Parsec::new(move |mut input: S| {
        let original_state = input.get_state();
        if let Some(item) = input.next() {
            if f(&item) {
                Ok((input, item))
            } else {
                Err(E::unexpected((original_state, input.get_state()), item))
            }
        } else {
            Err(E::eof((original_state, input.get_state())))
        }
    })
}

/// Parses a specific input item.
///
/// Succeeds only if the next item equals the expected item.
///
/// # Example
///
/// ```
/// use dlexer::parsec::*;
///
/// let parser = item('a');
/// assert_eq!(parser.test("a").unwrap(), 'a');
/// assert!(parser.test("b").is_err());
/// ```
pub fn item<S, E>(expected: S::Item) -> Parsec<S, E, S::Item>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    S::Item: PartialEq + Display + Clone + 'static,
{
    let expected_ = expected.clone();
    satisfy::<S, E, _>(move |item| *item == expected_).expected(expected)
}

/// Parses a decimal digit character (0-9).
pub fn decimal_digit<S, E>() -> Parsec<S, E, char>
where
    S: LexIterTrait<Item = char> + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    satisfy::<S, E, _>(|c: &char| c.is_digit(10)).expected("digit")
}

/// Parses a hexadecimal digit character (0-9, a-f, A-F).
pub fn hex_digit<S, E>() -> Parsec<S, E, char>
where
    S: LexIterTrait<Item = char> + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    satisfy::<S, E, _>(|c: &char| c.is_digit(16)).expected("hex digit")
}

/// Parses an octal digit character (0-7).
pub fn octal_digit<S, E>() -> Parsec<S, E, char>
where
    S: LexIterTrait<Item = char> + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    satisfy::<S, E, _>(|c: &char| c.is_digit(8)).expected("octal digit")
}

/// Parses a digit character in the specified radix.
pub fn digit<S, E>(radix: u32) -> Parsec<S, E, char>
where
    S: LexIterTrait<Item = char> + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    satisfy::<S, E, _>(move |c: &char| c.is_digit(radix)).expected("digit")
}

/// Parses an alphabetic character.
pub fn alpha<S, E>() -> Parsec<S, E, char>
where
    S: LexIterTrait<Item = char> + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    satisfy::<S, E, _>(|c: &char| c.is_alphabetic()).expected("alphabetic character")
}

/// Parses an alphanumeric character.
pub fn alphanumeric<S, E>() -> Parsec<S, E, char>
where
    S: LexIterTrait<Item = char> + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    satisfy::<S, E, _>(|c: &char| c.is_alphanumeric()).expected("alphanumeric character")
}

/// Parses a whitespace character.
pub fn whitespace<S, E>() -> Parsec<S, E, char>
where
    S: LexIterTrait<Item = char> + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    satisfy::<S, E, _>(|c: &char| c.is_whitespace()).expected("whitespace character")
}

/// Parses a newline character.
pub fn newline<S, E>() -> Parsec<S, E, char>
where
    S: LexIterTrait<Item = char> + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    satisfy::<S, E, _>(|c: &char| *c == '\n').expected("newline character")
}

/// Parses the end of input.
///
/// Succeeds only if there are no more items in the input.
pub fn eof<S, E>() -> Parsec<S, E, ()>
where
    S: LexIterTrait + Clone + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    Parsec::new(move |mut input: S| {
        let original_state = input.get_state();
        if input.next().is_none() {
            Ok((input, ()))
        } else {
            Err(E::eof((original_state, input.get_state())))
        }
    })
}

/// Parses a specific character.
///
/// # Example
///
/// ```
/// use dlexer::parsec::*;
///
/// let parser = char('a');
/// assert_eq!(parser.test("a").unwrap(), 'a');
/// assert!(parser.test("b").is_err());
/// ```
pub fn char<S, E>(expected: char) -> Parsec<S, E, char>
where
    S: LexIterTrait<Item = char> + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    satisfy::<S, E, _>(move |c: &char| *c == expected).expected(expected)
}

/// Returns the current parser state without consuming input.
pub fn state<S, E>() -> Parsec<S, E, LexIterState<S::Context>>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    Parsec::new(move |input: S| {
        let state = input.get_state();
        Ok((input, state))
    })
}

impl<S, E, A> Parsec<S, E, Result<A, E>>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
{
    /// Unwraps the result, propagating errors.
    ///
    /// If the result is `Ok(value)`, returns `value`, otherwise propagates the error.
    pub fn unwrap(self) -> Parsec<S, E, A>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let (next_input, result) = self.eval(input)?;
            match result {
                Ok(value) => Ok((next_input, value)),
                Err(error) => Err(error),
            }
        })
    }

    /// Unwraps the result, using a default value on error.
    ///
    /// If the result is `Ok(value)`, returns `value`, otherwise returns the default.
    pub fn unwrap_or(self, default: A) -> Parsec<S, E, A>
    where
        S: Clone,
        A: Clone,
    {
        Parsec::new(move |input: S| {
            let (next_input, result) = self.eval(input)?;
            match result {
                Ok(value) => Ok((next_input, value)),
                Err(_) => Ok((next_input, default.clone())),
            }
        })
    }

    /// Unwraps the result, using the default value on error.
    ///
    /// If the result is `Ok(value)`, returns `value`, otherwise returns `A::default()`.
    pub fn unwrap_or_default(self) -> Parsec<S, E, A>
    where
        S: Clone,
        A: Default,
    {
        Parsec::new(move |input: S| {
            let (next_input, result) = self.eval(input)?;
            match result {
                Ok(value) => Ok((next_input, value)),
                Err(_) => Ok((next_input, A::default())),
            }
        })
    }
}

impl<S, E, A, B> Parsec<S, E, Result<A, B>>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
    B: 'static,
{
    /// Converts inner errors to parser errors using a function.
    ///
    /// Maps the inner error type to a displayable value for error reporting.
    pub fn lift_err<F, D: Display>(self, f: F) -> Parsec<S, E, A>
    where
        F: Fn(B) -> D + 'static,
    {
        Parsec::new(move |input: S| {
            let original_state = input.get_state();
            let (next_input, result) = self.eval(input)?;
            match result {
                Ok(value) => Ok((next_input, value)),
                Err(error) => {
                    let msg = f(error);
                    Err(E::unexpected(
                        (original_state, next_input.get_state()),
                        &msg,
                    ))
                }
            }
        })
    }
}

impl<S, E, A> BitOr for Parsec<S, E, A>
where
    S: LexIterTrait + Clone + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
{
    type Output = Parsec<S, E, A>;

    /// Alternative operator. Equivalent to `or`.
    ///
    /// Tries the left parser, falling back to the right parser on failure.
    fn bitor(self, other: Self) -> Self::Output {
        self.or(other)
    }
}

impl<S, E, A, B> Add<Parsec<S, E, B>> for Parsec<S, E, A>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
    B: 'static,
{
    type Output = Parsec<S, E, (A, B)>;

    /// Pair operator. Equivalent to `pair`.
    ///
    /// Runs both parsers in sequence and returns the results as a tuple.
    fn add(self, other: Parsec<S, E, B>) -> Self::Output {
        self.pair(other)
    }
}

impl<S, E, A, B> Shr<Parsec<S, E, B>> for Parsec<S, E, A>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
    B: 'static,
{
    type Output = Parsec<S, E, B>;

    /// Sequence operator. Equivalent to `then`.
    ///
    /// Runs both parsers in sequence and returns the second result.
    fn shr(self, other: Parsec<S, E, B>) -> Self::Output {
        self.then(other)
    }
}

impl<S, E, A, B> Shl<Parsec<S, E, B>> for Parsec<S, E, A>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
    B: 'static,
{
    type Output = Parsec<S, E, A>;

    /// Left sequence operator. Equivalent to `with`.
    ///
    /// Runs both parsers in sequence and returns the first result.
    fn shl(self, other: Parsec<S, E, B>) -> Self::Output {
        self.with(other)
    }
}

impl<S, F, E, A, B> BitAnd<F> for Parsec<S, E, A>
where
    S: LexIterTrait + 'static,
    F: Fn(A) -> B + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
    B: 'static,
{
    type Output = Parsec<S, E, B>;

    /// Map operator. Equivalent to `map`.
    ///
    /// Transforms the parsed value using the given function.
    fn bitand(self, other: F) -> Self::Output {
        self.map(other)
    }
}

/// Creates a recursive parser using a function that returns a parser.
///
/// This is useful for creating parsers that reference themselves, such as
/// in recursive grammar definitions.
pub fn rec<F, S, E, A>(f: F) -> Parsec<S, E, A>
where
    F: Fn() -> Parsec<S, E, A> + 'static,
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
{
    Parsec::new(move |input: S| {
        let parser = f();
        parser.eval(input)
    })
}

/// Creates a parser that tries each parser in sequence until one succeeds.
///
/// This is similar to a series of `or` operations but more efficient for
/// many alternatives.
pub fn branch<S, E, A>(parsers: Vec<Parsec<S, E, A>>) -> Parsec<S, E, A>
where
    S: LexIterTrait + Clone + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
{
    Parsec::new(move |input: S| {
        if parsers.is_empty() {
            return Err(E::eof((input.get_state(), input.get_state())));
        }
        for parser in &parsers[..parsers.len() - 1] {
            if let Ok((next_input, value)) = parser.eval(input.clone()) {
                return Ok((next_input, value));
            }
        }
        let last_parser = &parsers[parsers.len() - 1];
        last_parser.eval(input)
    })
}

/// Trait for mapping parser output and error types.
pub trait ParserF {
    /// The type constructor for mapping over the output type.
    type MapOutput<U>;
    /// The type constructor for mapping over the error type.
    type MapError<E: ParserError>;
}

// Implement this trait for ParserBuilder
impl<S, E, U> ParserF for Parsec<S, E, U>
where
    S: LexIterTrait + 'static,
    E: ParserError + 'static,
    U: 'static,
{
    type MapOutput<T> = Parsec<S, E, T>;
    type MapError<T: ParserError> = Parsec<S, T, U>;
}
pub type With<P, T> = <P as ParserF>::MapOutput<T>;
pub type WithError<P, T> = <P as ParserF>::MapError<T>;
