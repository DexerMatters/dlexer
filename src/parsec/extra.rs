//! Extra, higher-level parser combinators.
//!
//! This module provides additional, more specialized parser combinators built
//! upon the core components in the `parsec` module. These are useful for common
//! but more complex parsing tasks.
//!
//! # Contents
//!
//! - **`build_ident`**: A convenient parser for identifiers that handles reserved keywords.
//! - **`indent_block`**: An experimental parser for handling indentation-sensitive blocks,
//!   useful for languages like Python. (Available with the `experimental` feature).

use crate::parsec::{alpha, alphanumeric, char, LexIterTrait};
use crate::parsec::{Parsec, ParserError};

/// Creates a parser for an identifier, excluding a list of reserved words.
///
/// An identifier is defined as starting with an alphabetic character or an underscore,
/// followed by zero or more alphanumeric characters or underscores. The parser will
/// fail if the parsed identifier is present in the `reserved` list.
///
/// # Example
///
/// ```
/// use dlexer::parsec::extra::build_ident;
/// use dlexer::parsec::*;
///
/// // A parser for identifiers, reserving "if" and "else".
/// let ident_parser = build_ident(["if", "else"]);
///
/// assert_eq!(ident_parser.test("my_var").unwrap(), "my_var");
/// // Fails because "if" is a reserved keyword.
/// assert!(ident_parser.test("if").is_err());
/// // Fails because it does not start with a letter or underscore.
/// assert!(ident_parser.test("123").is_err());
/// ```
#[inline]
pub fn build_ident<const N: usize, S, E>(reserved: [&'static str; N]) -> Parsec<S, E, String>
where
    S: LexIterTrait<Item = char> + Clone + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    let rsv = reserved.map(|s| s.to_string());
    (alpha() | char('_'))
        .extend((alphanumeric() | char('_')).many())
        .collect::<String>()
        .none_of(rsv.into_iter())
        .expected("identifier")
}

/// (Experimental) Parses a block of indented lines.
///
/// This parser is designed for indentation-sensitive languages. It consumes a newline
/// and then applies the given `parser` to each subsequent line that has a greater
/// indentation level than the starting line. It collects the results into a vector.
///
/// The block ends when a line is encountered with an indentation level less than or
/// equal to the original indentation level.
///
/// **Note:** This parser is experimental and its API may change. It requires the
/// `experimental` feature to be enabled.
#[cfg(feature = "experimental")]
pub fn indent_block<S, E, A>(parser: Parsec<S, E, A>) -> Parsec<S, E, Vec<A>>
where
    S: LexIterTrait<Item = char> + Clone + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
{
    Parsec::new(move |mut input: S| {
        let original = input.get_state();
        let original_indent = original.current_indent;

        // Skip first newline
        let next = input.next();
        if let Some(next) = next {
            if next != '\n' {
                return Err(E::unexpected((original, input.get_state()), next)
                    .with_expected("indent block"));
            }
        } else {
            return Err(E::eof((original, input.get_state())).with_expected("indent block"));
        }

        let mut result = vec![];

        loop {
            let input_o = input.clone();
            if let Some(next) = input.next() {
                if next == '\n' {
                    continue; // Skip newlines
                } else {
                    input = input_o; // Reset input if not newline
                    let (next_input, value) = parser.eval(input.clone())?;
                    if next_input.get_state().current_indent <= original_indent {
                        if result.is_empty() {
                            return Err(E::unexpected(
                                (original, input.get_state()),
                                next_input.get_state().current_indent,
                            )
                            .with_expected("indent block"));
                        }
                        return Ok((input, result));
                    }
                    result.push(value);
                    input = next_input;
                    continue;
                }
            } else {
                return Ok((input, result));
            }
        }
    })
}
