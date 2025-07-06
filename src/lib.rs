//! # DLexer: A Parser Combinator Library
//!
//! `dlexer` is a flexible and composable parser combinator library for Rust, inspired by
//! libraries like Parsec in Haskell. It provides a rich set of tools for building
//! robust parsers for both text-based and binary formats.
//!
//! ## Core Concepts
//!
//! The library is built around the [`parsec::Parsec`] type, which represents a parser.
//! These parsers can be combined and transformed using a variety of functions and
//! operators to build up complex parsing logic from simple, reusable components.
//!
//! ## Key Modules
//!
//! - [`parsec`]: The core parser combinator library. Contains the `Parsec` struct
//!   and fundamental combinators like `map`, `bind`, `or`, `many`, etc.
//! - [`lex`]: Provides tools for lexical analysis, including "skippers" for handling
//!   whitespace and comments, and token-level parsers for common data types like
//!   integers and symbols.
//! - [`binary`]: A specialized module for parsing binary data, with parsers for
//!   various integer and float types with controlled endianness.
//! - [`errors`]: Defines the error handling system, including the `ParserError` trait.
//!
//! ## Getting Started
//!
//! Here is a simple example of parsing a comma-separated list of numbers:
//!
//! ```
//! use dlexer::lex::{integer, WhitespaceSkipper};
//! use dlexer::parsec::*;
//!
//! // A parser for a single decimal integer, handling surrounding whitespace.
//! let number_parser = integer(10);
//!
//! // A parser for a list of numbers separated by commas.
//! let list_parser = number_parser.sep(char(','));
//!
//! // Run the parser on some input.
//! let result = list_parser.parse("1, 2, 3", WhitespaceSkipper);
//!
//! assert_eq!(result.unwrap(), vec![1, 2, 3]);
//! ```
//!
//! ## Macros
//!
//! The library also provides helpful macros like [`do_parse!`] for monadic chaining
//! and [`map!`] for mapping multiple parsers to values.
pub mod binary;
pub mod errors;
pub mod lex;
pub mod parsec;

//mod stream;

mod examples;

/// A convenience macro for mapping multiple parsers to specific values.
///
/// This macro creates a new parser that tries each provided parser in sequence.
/// If a parser succeeds, it returns the corresponding value. This is a concise
/// alternative to chaining multiple `or` calls with `map`.
///
/// # Syntax
///
/// `map!(parser1 => value1, parser2 => value2, ...)`
///
/// # Example
///
/// ```
/// use dlexer::lex::symbol;
/// use dlexer::map;
/// use dlexer::parsec::*;
///
/// #[derive(Debug, PartialEq)]
/// enum Keyword {
///     Let,
///     If,
///     Else,
/// }
///
/// let keyword_parser = map!(
///     symbol("let") => Keyword::Let,
///     symbol("if") => Keyword::If,
///     symbol("else") => Keyword::Else
/// );
///
/// assert_eq!(keyword_parser.test("if").unwrap(), Keyword::If);
/// assert!(keyword_parser.test("other").is_err());
/// ```
#[macro_export]
macro_rules! map {
    ($($parser:expr => $value:expr),*) => {
        $(
            $parser.map(|_| $value)
        )|*
    };
}

/// A macro for writing parsers in a sequential, do-notation style.
///
/// This macro provides a more imperative-looking syntax for chaining parsers,
/// which can be more readable than deeply nested calls to `bind` and `then`.
///
/// # Syntax
///
/// - `let% <var> = <parser>;` : Binds the result of `<parser>` to `<var>`. This is equivalent to `bind`.
/// - `<parser>;` : Runs `<parser>` and discards its result. This is equivalent to `then`.
/// - `let <var> = <expr>;` : Binds the result of a standard Rust expression to `<var>`.
/// - The last expression in the block is the final parser, which determines the return value.
///
/// # Example
///
/// ```
/// use dlexer::do_parse;
/// use dlexer::parsec::*;
///
/// // A parser for a pair of numbers in parentheses, like "(1, 2)".
/// let pair_parser: BasicParser = do_parse!(
///     char('(');
///     let% x = decimal_digit();
///     char(',');
///     let% y = decimal_digit();
///     char(')');
///     pure((x, y)) // The return value
/// );
///
/// let result = pair_parser.test("(3,4)").unwrap();
/// assert_eq!(result, ('3', '4'));
/// ```
#[macro_export]
macro_rules! do_parse {
    ($e:expr) => {
        $e
    };

    (let% $v:ident = $m:expr; $($rest:tt)*) => {
        $m.bind(move |$v| do_parse!($($rest)*))
    };

    (let $v:ident $(:$t: ty)? = $m:expr; $($rest:tt)*) => {
        {let $v $(:$t)? = $m; do_parse!($($rest)*)}
    };

    ($m:expr; $($rest:tt)*) => {
        $m.then(do_parse!($($rest)*))
    };
}

#[cfg(test)]
mod tests {
    #![allow(dead_code)]

    use crate::{
        binary::{n_bytes, u32, BasicByteParser},
        lex::{symbol, token, WhitespaceSkipper},
        parsec::*,
    };

    type P = BasicParser;

    #[test]
    fn it_works() {
        #[derive(Debug, Clone)]
        enum AST {
            Identifier(String),
            Boolean(bool),
        }

        // Identifier parsing example
        let ident: With<P, AST> = token(do_parse!(
            let% initial = (alpha() | char('_'));
            let% rest    = alphanumeric().many().collect::<String>();
            let  result  = format!("{}{}", initial, rest);
            pure(AST::Identifier(result))
        ))
        .expected("identifier");

        // Applicative style identifier parsing
        let _ident: With<P, AST> = token(
            pure(AST::Identifier).apply(
                (alpha() | char('_'))
                    .extend(alphanumeric().many())
                    .collect::<String>(),
            ),
        );

        // Boolean parsing example
        let boolean = token(map!(
            symbol("true") => AST::Boolean(true),
            symbol("false") => AST::Boolean(false)
        ));

        let p = (boolean | ident).sep_till(char(','), eof());
        let input = "foo, bar, a12, \ntrue, false";
        let result = p.test(input);
        match result {
            Ok(a) => {
                println!("Parsed successfully: {:?}", a);
            }
            Err(e) => println!("{}", e),
        }
    }

    #[test]
    fn util_test() {
        let p: With<P, _> = token(any().many1_till(char('<')).collect::<String>());
        let input = "fo o < bar";
        match p.parse(input, WhitespaceSkipper) {
            Ok(a) => {
                println!("Parsed successfully: {:?}", a);
            }
            Err(e) => println!("{}", e),
        }
    }

    #[test]
    fn escape_test() {
        let escapes: With<P, _> = map!(
            symbol("\\n") => "\n",
            symbol("\\t") => "\t",
            symbol("\\r") => "\r",
            symbol("\\\\") => "\\",
            symbol("\\\"") => "\""
        );

        let string = token(
            (escapes.into() | any().not('\"').into())
                .many()
                .between(char('"'), char('"'))
                & |s: Vec<String>| s.join(""),
        );

        let result = string
            .test(r#""This is a string with an escape: \n and a quote: \" and a backslash: \\""#);
        match result {
            Ok(a) => {
                println!("Parsed successfully:\n {}", a);
            }
            Err(e) => println!("{}", e),
        }
    }

    #[test]
    fn take_test() {
        let p: With<P, _> = char('a').take(1..2).collect::<String>();
        let input = "";
        match p.dbg().test(input) {
            Ok(a) => {
                println!("Parsed successfully: {:?}", a);
            }
            Err(e) => println!("{}", e),
        }
    }

    type BP = BasicByteParser;
    #[test]
    fn binary_test() {
        let p: With<BP, _> = do_parse!(
            let% length = u32().dbg_();
            let% bytes = n_bytes(length as usize);
            eof();
            pure(bytes)
        );

        let input = [0x00, 0x00, 0x00, 0x04, 0xAA, 0xBB, 0xCC, 0xDD]; // Represents "abcd" with length 4
        match p.parse(&input) {
            Ok(a) => println!("Parsed successfully: {:?}", a),
            Err(e) => println!("{}", e),
        }
    }
}
