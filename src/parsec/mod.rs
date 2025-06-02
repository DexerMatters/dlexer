//! # Parser Combinator Core
//!
//! This module implements the core parser combinator functionality, providing
//! both monadic and applicative interfaces for building complex parsers from
//! simple components.
//!
//! ## Key Concepts
//!
//! - **Parser**: A function that consumes input and produces a result
//! - **Combinator**: A function that combines parsers to create new parsers
//! - **Monadic interface**: Sequential composition with `bind` and `do_parse!`
//! - **Applicative interface**: Pure functional composition with `apply`
//!
//! ## Common Patterns
//!
//! ```rust
//! use dlexer::parsec::*;
//!
//! // Monadic style - sequential parsing
//! let parser1 = do_parse!(
//!     let% name = alpha().many().collect::<String>();
//!     let% _ = char('=');
//!     let% value = digit().many().collect::<String>();
//!     pure((name, value))
//! );
//!
//! // Applicative style - pure functional
//! let parser2 = pure(|name| move |value| (name, value))
//!     .apply(alpha().many().collect::<String>())
//!     .apply(digit().many().collect::<String>());
//!
//! // Choice and repetition
//! let items = (alpha() | digit()).sep_by(char(','));
//! ```
//!
//! ## Error Handling
//!
//! All parsers can fail with position information:
//! ```rust
//! let parser = alpha().expected("letter");
//! match parser.test("123") {
//!     Ok(result) => println!("Parsed: {}", result),
//!     Err(error) => println!("Error: {}", error), // Shows position and expectation
//! }
//! ```

pub mod extra;

use std::{
    fmt::{Debug, Display},
    ops::BitOr,
    rc::Rc,
};

use crate::{
    errors::{ParserError, SimpleParserError},
    lex::{LexIter, LexIterTrait},
};

/// Type alias for building parsers with custom state and error types.
pub type BuildParser<S, E> = Parsec<S, E, <S as Iterator>::Item>;

/// Type alias for the standard parser using LexIter and SimpleParserError.
///
/// This is the most commonly used parser type in the library, providing
/// a good balance of functionality and ease of use.
pub type BasicParser = Parsec<LexIter, SimpleParserError, <LexIter as Iterator>::Item>;

/// Core parser combinator type.
///
/// A `Parsec` represents a parser that consumes input of type `S`, can fail
/// with errors of type `E`, and produces values of type `A` on success.
///
/// # Type Parameters
/// - `S`: The input stream type (must implement `LexIterTrait`)
/// - `E`: The error type (must implement `ParserError`)
/// - `A`: The output type produced by successful parsing
///
/// # Example
/// ```rust
/// use dlexer::parsec::{Parsec, pure};
/// use dlexer::errors::SimpleParserError;
/// use dlexer::lex::LexIter;
///
/// // Create a parser that always succeeds with value 42
/// let parser: Parsec<LexIter, SimpleParserError, i32> = pure(42);
///
/// // Create a parser that parses a specific character
/// let char_parser = char('a');
/// ```
#[derive(Clone)]
pub struct Parsec<S: LexIterTrait, E: ParserError, A> {
    run: Rc<dyn Fn(S) -> Result<(S, A), E>>,
}

impl<S: LexIterTrait + 'static, E: ParserError + 'static, A: 'static> Parsec<S, E, A> {
    /// Create a new parser from a parsing function.
    ///
    /// # Parameters
    /// - `run`: A function that takes input and returns either success with
    ///   remaining input and parsed value, or an error
    pub fn new(run: impl Fn(S) -> Result<(S, A), E> + 'static) -> Self {
        Parsec { run: Rc::new(run) }
    }

    /// Run the parser and return both remaining input and parsed value.
    ///
    /// This is the low-level interface that preserves the remaining input
    /// for further parsing.
    pub fn eval(&self, input: S) -> Result<(S, A), E> {
        (self.run)(input)
    }

    /// Run the parser and return only the parsed value.
    ///
    /// This is a convenience method for when you don't need the remaining input.
    pub fn run(&self, input: S) -> Result<A, E> {
        (self.run)(input).map(|(_, value)| value)
    }

    /// Transform the output of a parser using a function (Functor).
    ///
    /// This is the fundamental operation for transforming parser results.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let digit_parser = digit();
    /// let number_parser = digit_parser.map(|c| c.to_digit(10).unwrap());
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

    /// Add an expectation message to a parser for better error reporting.
    ///
    /// When this parser fails, the error will include information about
    /// what was expected, improving debugging experience.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let identifier = alpha().many().collect::<String>()
    ///     .expected("identifier");
    /// ```
    pub fn expected<T>(self, expected: T) -> Parsec<S, E, A>
    where
        T: Display + Clone + 'static,
        E: ParserError + 'static,
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

    /// Monadic bind operation for sequential parser composition.
    ///
    /// This allows you to use the result of one parser to determine
    /// what parser to run next.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let dynamic_parser = digit().bind(|d| {
    ///     let count = d.to_digit(10).unwrap() as usize;
    ///     alpha().times(count) // Parse 'count' letters
    /// });
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

    /// Applicative apply operation for pure functional composition.
    ///
    /// This allows you to apply a function parser to an argument parser,
    /// useful for building up complex data structures.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let pair_parser = pure(|x| move |y| (x, y))
    ///     .apply(digit())
    ///     .apply(alpha());
    /// ```
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

    /// Sequential composition - run this parser then another, keeping the second result.
    ///
    /// This is useful when you need to parse and discard a delimiter or keyword.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let after_arrow = symbol("->").then(alpha().many().collect::<String>());
    /// ```
    pub fn then<B: 'static>(self, other: Parsec<S, E, B>) -> Parsec<S, E, B> {
        Parsec::new(move |input| {
            let (next_input, _) = self.eval(input)?;
            other.eval(next_input)
        })
    }

    /// Sequential composition - run this parser then another, keeping the first result.
    ///
    /// This is useful when you need to parse something followed by a delimiter.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let terminated = alpha().many().collect::<String>().with(char(';'));
    /// ```
    pub fn with<B: 'static>(self, other: Parsec<S, E, B>) -> Parsec<S, E, A> {
        Parsec::new(move |input| {
            let (next_input, value) = self.eval(input)?;
            let (final_input, _) = other.eval(next_input)?;
            Ok((final_input, value))
        })
    }

    /// Parse between two delimiters.
    ///
    /// This parser requires that the input starts with the `left` parser
    /// and ends with the `right` parser. The result is the value of the
    /// inner parser (usually `self`).
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().between(char('('), char(')'));
    /// ```
    pub fn between<B: 'static>(
        self,
        left: Parsec<S, E, B>,
        right: Parsec<S, E, B>,
    ) -> Parsec<S, E, A>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let (next_input, _) = left.eval(input.clone())?;
            let (final_input, value) = self.eval(next_input)?;
            let (end_input, _) = right.eval(final_input)?;
            Ok((end_input, value))
        })
    }

    /// Try to parse with this parser, and if it fails, try the other parser.
    ///
    /// This implements a logical OR between two parsers.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().or(digit());
    /// ```
    pub fn or(self, other: Parsec<S, E, A>) -> Parsec<S, E, A>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| self.eval(input.clone()).or_else(|_| other.eval(input)))
    }

    /// Optional parser - parses this parser or nothing.
    ///
    /// The result is `Some(value)` if this parser succeeds, or `None` if it fails.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().opt();
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

    /// Parse one occurrence of this parser.
    ///
    /// This is useful for parsers that should match exactly once.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().one();
    /// ```
    pub fn one(self) -> Parsec<S, E, Vec<A>> {
        Parsec::new(move |input: S| {
            let (next_input, value) = self.eval(input)?;
            Ok((next_input, vec![value]))
        })
    }

    /// Attempt to parse with this parser, returning the result wrapped in `Ok`,
    /// or the error wrapped in `Err` if it fails.
    ///
    /// This is useful for parsers that you want to ensure always succeed
    /// or fail in a controlled manner.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().try_();
    /// ```
    pub fn try_(self) -> Parsec<S, E, Result<A, E>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| match self.eval(input.clone()) {
            Ok((next_input, value)) => Ok((next_input, Ok(value))),
            Err(error) => Ok((input, Err(error))),
        })
    }

    /// Parse zero or more occurrences of this parser.
    ///
    /// This will continue parsing until the parser fails, collecting
    /// all results in a vector.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().many();
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

    /// Parse one or more occurrences of this parser.
    ///
    /// This is similar to `many1`, but it will fail if the parser does not
    /// match at least once.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().many1();
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

    /// Parse until the end parser succeeds, collecting results in a vector.
    ///
    /// This will parse as many occurrences of this parser as possible until
    /// the `end` parser succeeds.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().many_till(char(';'));
    /// ```
    pub fn many_till(self, end: Parsec<S, E, S::Item>) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let mut results = Vec::new();
            let mut current_input = input;

            // Already check if the end condition is met
            if let Ok((end_input, _)) = end.eval(current_input.clone()) {
                return Ok((end_input, results));
            }

            loop {
                match self.eval(current_input.clone()) {
                    Ok((new_input, value)) => {
                        results.push(value);
                        current_input = new_input;
                    }
                    Err(err) => return Err(err),
                }

                if let Ok((end_input, _)) = end.eval(current_input.clone()) {
                    return Ok((end_input, results));
                }
            }
        })
    }

    /// Parse one or more occurrences of this parser until the end parser succeeds.
    ///
    /// This is similar to `many_till`, but it requires at least one successful
    /// parse of this parser.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().many1_till(char(';'));
    /// ```
    pub fn many1_till(self, end: Parsec<S, E, S::Item>) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let (mut current_input, first_value) = self.eval(input)?;
            let mut results = vec![first_value];

            loop {
                match self.eval(current_input.clone()) {
                    Ok((new_input, value)) => {
                        results.push(value);
                        current_input = new_input;
                    }
                    Err(err) => return Err(err),
                }

                if let Ok((end_input, _)) = end.eval(current_input.clone()) {
                    return Ok((end_input, results));
                }
            }
        })
    }

    /// Parse separated by a separator parser, collecting results in a vector.
    ///
    /// This will parse occurrences of this parser separated by the `sep` parser.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().sep(char(','));
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
    /// Parse one or more occurrences of this parser separated by a separator parser.
    ///
    /// This is similar to `sep`, but it requires at least one successful parse
    /// of this parser.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().sep1(char(','));
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

    /// Parse separated by a separator parser until an end parser succeeds.
    ///
    /// This will parse occurrences of this parser separated by the `sep` parser,
    /// and stop when the `end` parser succeeds.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().sep_till(char(','), char(';'));
    /// ```
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

            // Check if the end condition is met immediately
            if let Ok((end_input, _)) = end.eval(current_input.clone()) {
                return Ok((end_input, results));
            }

            // Parse the first item if possible
            match self.eval(current_input.clone()) {
                Ok((new_input, value)) => {
                    results.push(value);
                    current_input = new_input;
                }
                Err(err) => return Err(err),
            }

            loop {
                // Check if the end condition is met
                if let Ok((end_input, _)) = end.eval(current_input.clone()) {
                    return Ok((end_input, results));
                }

                // Try to parse separator followed by an item
                match sep.eval(current_input.clone()) {
                    Ok((sep_input, _)) => match self.eval(sep_input) {
                        Ok((new_input, value)) => {
                            results.push(value);
                            current_input = new_input;
                        }
                        Err(err) => return Err(err),
                    },
                    Err(err) => {
                        // If we can't parse a separator, check if end condition is met
                        if let Ok((end_input, _)) = end.eval(current_input.clone()) {
                            return Ok((end_input, results));
                        } else {
                            // Neither separator nor end condition could be parsed
                            return Err(err);
                        }
                    }
                }
            }
        })
    }

    /// Parse one or more occurrences of this parser separated by a separator parser
    /// until an end parser succeeds.
    ///
    /// This is similar to `sep_till`, but it requires at least one successful parse
    /// of this parser.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().sep1_till(char(','), char(';'));
    /// ```
    pub fn sep1_till<T: 'static, U: 'static>(
        self,
        sep: Parsec<S, E, T>,
        end: Parsec<S, E, U>,
    ) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            // Must parse at least one item
            let (mut current_input, first_value) = self.eval(input)?;
            let mut results = vec![first_value];

            loop {
                // Check if the end condition is met
                if let Ok((end_input, _)) = end.eval(current_input.clone()) {
                    return Ok((end_input, results));
                }

                // Try to parse separator followed by an item
                match sep.eval(current_input.clone()) {
                    Ok((sep_input, _)) => match self.eval(sep_input) {
                        Ok((new_input, value)) => {
                            results.push(value);
                            current_input = new_input;
                        }
                        Err(err) => return Err(err),
                    },
                    Err(err) => {
                        // If we can't parse a separator, check if end condition is met
                        if let Ok((end_input, _)) = end.eval(current_input.clone()) {
                            return Ok((end_input, results));
                        } else {
                            // Neither separator nor end condition could be parsed
                            return Err(err);
                        }
                    }
                }
            }
        })
    }

    /// Left-associative parsing with a binary operator.
    ///
    /// This will apply the operator between the results of this parser,
    /// accumulating the result.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let sum_parser = digit().chain(op('+'), 0);
    /// ```
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

    /// Right-associative parsing with a binary operator.
    ///
    /// This will apply the operator between the results of this parser,
    /// accumulating the result, but in a right-associative manner.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let sum_parser = digit().chain_right(op('+'), 0);
    /// ```
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
            E: ParserError + 'static,
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

    /// Parse a fixed pair of values with two parsers.
    ///
    /// This runs both parsers sequentially and collects their results in a tuple.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().pair(digit());
    /// ```
    pub fn pair<B: 'static>(self, other: Parsec<S, E, B>) -> Parsec<S, E, (A, B)>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let (next_input, value_a) = self.eval(input.clone())?;
            let (final_input, value_b) = other.eval(next_input)?;
            Ok((final_input, (value_a, value_b)))
        })
    }

    /// Extend the result of this parser with the result of another parser.
    ///
    /// This is useful for combining parsers that produce parts of a data structure.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let extended_parser = alpha().extend(digit().many());
    /// ```
    pub fn extend(self, other: Parsec<S, E, Vec<A>>) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let (next_input, value) = self.eval(input.clone())?;
            let (final_input, mut other_values) = other.eval(next_input)?;
            other_values.insert(0, value);
            Ok((final_input, other_values))
        })
    }

    /// Hold the result of this parser and apply a predicate function.
    ///
    /// This parser will only succeed if the predicate returns true for the
    /// parsed value. If it fails, the error will include the expected predicate.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let even_parser = digit().hold(|&d| d % 2 == 0);
    /// ```
    pub fn hold<F>(self, f: F) -> Parsec<S, E, A>
    where
        F: Fn(&A) -> bool + 'static,
        A: Display,
        S: Clone,
        E: Clone + 'static,
    {
        Parsec::new(move |input: S| {
            let (next_input, item) = self.eval(input.clone())?;
            if f(&item) {
                Ok((next_input, item))
            } else {
                Err(E::unexpected(
                    (input.get_state(), next_input.get_state()),
                    &item,
                ))
            }
        })
    }

    /// Negate the result of this parser, expecting a different value.
    ///
    /// This parser will succeed if the parsed value is not equal to the given
    /// value. If it fails, the error will include the expected value.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let not_a_parser = alpha().not('a');
    /// ```
    pub fn not(self, value: A) -> Parsec<S, E, A>
    where
        A: PartialEq + Display,
        S: Clone,
        E: Clone + 'static,
    {
        Parsec::new(move |input: S| {
            let (next_input, item) = self.eval(input.clone())?;
            if item == value {
                Err(
                    E::unexpected((input.get_state(), next_input.get_state()), &item)
                        .with_expected(&value),
                )
            } else {
                Ok((next_input, item))
            }
        })
    }

    /// Ensure the result of this parser matches a specific value.
    ///
    /// This parser will succeed if the parsed value is equal to the given
    /// value. If it fails, the error will include the expected value.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let a_parser = alpha().is('a');
    /// ```
    pub fn is(self, value: A) -> Parsec<S, E, A>
    where
        A: PartialEq + Display,
        S: Clone,
        E: Clone + 'static,
    {
        Parsec::new(move |input: S| {
            let (next_input, item) = self.eval(input.clone())?;
            if item == value {
                Ok((next_input, item))
            } else {
                Err(
                    E::unexpected((input.get_state(), next_input.get_state()), &item)
                        .with_expected(&value),
                )
            }
        })
    }

    /// Succeed if the parsed value is one of the given set of values.
    ///
    /// This parser will succeed if the parsed value matches any value
    /// from the `values` iterator. If it fails, the error will include
    /// the expected values.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().one_of("abc".chars());
    /// ```
    pub fn one_of<'a>(self, values: impl Iterator<Item = A> + Clone + 'static) -> Parsec<S, E, A>
    where
        A: PartialEq + Display,
        S: Clone,
        E: Clone + 'static,
    {
        Parsec::new(move |input: S| {
            let (next_input, item) = self.eval(input.clone())?;
            let mut values = values.clone();
            if values.any(|v| v == item) {
                Ok((next_input, item))
            } else {
                Err(
                    E::unexpected((input.get_state(), next_input.get_state()), &item)
                        .with_expected(
                            &values.map(|v| v.to_string()).collect::<Vec<_>>().join(", "),
                        ),
                )
            }
        })
    }

    /// Succeed if the parsed value is none of the given set of values.
    ///
    /// This parser will succeed if the parsed value does not match any value
    /// from the `values` iterator. If it fails, the error will include
    /// the unexpected value.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().none_of("abc".chars());
    /// ```
    pub fn none_of<'a>(self, values: impl Iterator<Item = A> + Clone + 'static) -> Parsec<S, E, A>
    where
        A: PartialEq + Display,
        S: Clone,
        E: 'static,
    {
        Parsec::new(move |input: S| {
            let (next_input, item) = self.eval(input.clone())?;
            if values.clone().any(|v| v == item) {
                Err(E::unexpected(
                    (input.get_state(), next_input.get_state()),
                    &item,
                ))
            } else {
                Ok((next_input, item))
            }
        })
    }

    /// Debugging parser that prints input and output details.
    ///
    /// This parser is useful for development and debugging purposes,
    /// as it prints the state of the input before and after parsing,
    /// as well as the parsed value.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().dbg();
    /// ```
    pub fn dbg(self) -> Parsec<S, E, A>
    where
        S: Clone,
        A: Debug,
    {
        Parsec::new(move |input: S| {
            let original = input.get_state();
            println!("Input:");
            println!(
                "  Position:\t{}:{}",
                original.current_line, original.current_column
            );
            println!("  Indentation:\t{}", original.current_indent);
            let mut rest = original
                .text
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
                    let mut next_rest = next.text.get(next.current_pos..).unwrap_or("End of input");
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

    /// Convert the output of this parser to a different type.
    ///
    /// This is a convenience method that uses `map` under the hood.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = digit().into::<i32>();
    /// ```
    pub fn into<B: 'static>(self) -> Parsec<S, E, B>
    where
        A: Into<B>,
    {
        self.map(|a| a.into())
    }
}

impl<S: LexIterTrait + 'static, E: ParserError + 'static, A: 'static> Parsec<S, E, Parsec<S, E, A>>
where
    S: Clone,
{
    /// Join nested parsers into a single parser.
    ///
    /// This is useful for flattening the result of parser combinators
    /// that produce nested `Parsec` values.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().many().collect::<Parsec<_, _, String>>().join();
    /// ```
    pub fn join(self) -> Parsec<S, E, A> {
        Parsec::new(move |input: S| {
            let (next_input, parser) = self.eval(input)?;
            parser.eval(next_input)
        })
    }
}

impl<S: LexIterTrait + 'static, E: ParserError + 'static, A: 'static>
    Parsec<S, E, Vec<Parsec<S, E, A>>>
where
    S: Clone,
{
    /// Sequence a list of parsers, collecting the results in a vector.
    ///
    /// This is useful for applying a series of parsers in order and
    /// collecting their results.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = (alpha() | digit()).many().sequence();
    /// ```
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

impl<S: LexIterTrait + 'static, E: ParserError + 'static, A: 'static> Parsec<S, E, Vec<A>> {
    /// Collect the results of this parser into a different container type.
    ///
    /// This uses the `Into` and `FromIterator` traits to convert the
    /// collected results into the desired container type.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let parser = alpha().many().collect::<Vec<_>>();
    /// ```
    pub fn collect<B: 'static>(self) -> Parsec<S, E, B>
    where
        A: Into<B>,
        B: FromIterator<A>,
    {
        self.map(|vec| vec.into_iter().collect())
    }

    /// Append the result of another parser to this parser's result.
    ///
    /// This is useful for combining the results of two parsers into a single
    /// vector.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let combined_parser = alpha().many().append(digit().many());
    /// ```
    pub fn append(self, other: Parsec<S, E, A>) -> Parsec<S, E, Vec<A>> {
        Parsec::new(move |input: S| {
            let (next_input, mut values) = self.eval(input)?;
            let (final_input, value) = other.eval(next_input)?;
            values.push(value);
            Ok((final_input, values))
        })
    }

    /// Concatenate the results of another parser to this parser's results.
    ///
    /// This is useful for combining the results of two parsers into a single
    /// vector, extending the existing results.
    ///
    /// # Example
    /// ```rust
    /// use dlexer::parsec::*;
    ///
    /// let concatenated_parser = alpha().many().concat(digit().many());
    /// ```
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

// === Utility Functions ===

/// Create a parser that always succeeds with the given value.
///
/// This parser consumes no input and always returns the provided value.
/// It's useful for creating default values or starting points for parser
/// composition.
///
/// # Example
/// ```rust
/// use dlexer::parsec::*;
///
/// let parser = pure(42);
/// assert_eq!(parser.test(""), Ok(42));
/// ```
pub fn pure<S: LexIterTrait + 'static, E: ParserError + 'static, T>(value: T) -> Parsec<S, E, T>
where
    T: Clone + 'static,
{
    Parsec::new(move |input| Ok((input, value.clone())))
}

/// Create a parser that always fails with the given error.
///
/// This parser never consumes input and always returns the provided error.
/// It's useful for creating conditional failures or error scenarios.
///
/// # Example
/// ```rust
/// use dlexer::parsec::*;
/// use dlexer::errors::SimpleParserError;
///
/// let error = SimpleParserError::new("test".into(), "fail".into(), Default::default());
/// let parser = fail::<_, _, i32>(error);
/// ```
pub fn fail<S: LexIterTrait + 'static, E: ParserError + 'static, A: 'static>(
    error: E,
) -> Parsec<S, E, A>
where
    E: Clone,
{
    Parsec::new(move |_| Err(error.clone()))
}

/// Parse any single item from the input.
///
/// This parser succeeds if there is at least one item left in the input,
/// returning that item. It fails with an EOF error if the input is empty.
///
/// # Example
/// ```rust
/// use dlexer::parsec::*;
///
/// let parser = any();
/// // Will parse any character
/// ```
pub fn any<S: LexIterTrait + 'static, E: ParserError + 'static>() -> Parsec<S, E, S::Item>
where
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

/// Create a parser that succeeds only if the predicate returns true for the next item.
///
/// This parser consumes one item from the input if the predicate function
/// returns true for that item. Otherwise, it fails with an unexpected error.
///
/// # Example
/// ```rust
/// use dlexer::parsec::*;
///
/// let digit_parser = satisfy(|c: &char| c.is_ascii_digit());
/// let vowel_parser = satisfy(|c: &char| "aeiou".contains(*c));
/// ```
pub fn satisfy<S: LexIterTrait + 'static, E: ParserError + 'static, F>(
    f: F,
) -> Parsec<S, E, S::Item>
where
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

pub fn item<S: LexIterTrait + 'static, E: ParserError + 'static>(
    expected: S::Item,
) -> Parsec<S, E, S::Item>
where
    S::Item: PartialEq + Display + Clone + 'static,
{
    let expected_ = expected.clone();
    satisfy::<S, E, _>(move |item| *item == expected_).expected(expected)
}

pub fn decimal_digit<S: LexIterTrait + 'static, E: ParserError + 'static>() -> Parsec<S, E, char> {
    satisfy::<S, E, _>(|c: &char| c.is_digit(10)).expected("digit")
}

pub fn hex_digit<S: LexIterTrait + 'static, E: ParserError + 'static>() -> Parsec<S, E, char> {
    satisfy::<S, E, _>(|c: &char| c.is_digit(16)).expected("hex digit")
}

pub fn octal_digit<S: LexIterTrait + 'static, E: ParserError + 'static>() -> Parsec<S, E, char> {
    satisfy::<S, E, _>(|c: &char| c.is_digit(8)).expected("octal digit")
}

pub fn digit<S: LexIterTrait + 'static, E: ParserError + 'static>(
    radix: u32,
) -> Parsec<S, E, char> {
    satisfy::<S, E, _>(move |c: &char| c.is_digit(radix)).expected("digit")
}

pub fn alpha<S: LexIterTrait + 'static, E: ParserError + 'static>() -> Parsec<S, E, char> {
    satisfy::<S, E, _>(|c: &char| c.is_alphabetic()).expected("alphabetic character")
}

pub fn alphanumeric<S: LexIterTrait + 'static, E: ParserError + 'static>() -> Parsec<S, E, char> {
    satisfy::<S, E, _>(|c: &char| c.is_alphanumeric()).expected("alphanumeric character")
}

pub fn whitespace<S: LexIterTrait + 'static, E: ParserError + 'static>() -> Parsec<S, E, char> {
    satisfy::<S, E, _>(|c: &char| c.is_whitespace()).expected("whitespace character")
}

pub fn newline<S: LexIterTrait + 'static, E: ParserError + 'static>() -> Parsec<S, E, char> {
    satisfy::<S, E, _>(|c: &char| *c == '\n').expected("newline character")
}

pub fn eof<S: LexIterTrait + Clone + 'static, E: ParserError + 'static>() -> Parsec<S, E, ()> {
    Parsec::new(move |mut input: S| {
        let original_state = input.get_state();
        if input.next().is_none() {
            Ok((input, ()))
        } else {
            Err(E::eof((original_state, input.get_state())))
        }
    })
}

pub fn char<S: LexIterTrait + 'static, E: ParserError + 'static>(
    expected: char,
) -> Parsec<S, E, char> {
    satisfy::<S, E, _>(move |c: &char| *c == expected).expected(expected)
}

impl<S: LexIterTrait + 'static, E: ParserError + 'static, A: 'static> Parsec<S, E, Result<A, E>> {
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
}

impl<S: LexIterTrait + Clone + 'static, E: ParserError + 'static, A: 'static> BitOr
    for Parsec<S, E, A>
{
    type Output = Parsec<S, E, A>;

    fn bitor(self, other: Self) -> Self::Output {
        self.or(other)
    }
}

pub fn rec<F, S: LexIterTrait + 'static, E: ParserError + 'static, A: 'static>(
    f: F,
) -> Parsec<S, E, A>
where
    F: Fn() -> Parsec<S, E, A> + 'static,
{
    Parsec::new(move |input: S| {
        let parser = f();
        parser.eval(input)
    })
}

pub fn branch<S: LexIterTrait + Clone + 'static, E: ParserError + 'static, A: 'static>(
    parsers: Vec<Parsec<S, E, A>>,
) -> Parsec<S, E, A> {
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

// Add this new trait to provide a clear path to the M associated type
pub trait ParserF {
    type MapOutput<U>;
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
