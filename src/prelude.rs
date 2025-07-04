//! # DLexer Prelude
//!
//! This module provides convenient imports for common DLexer functionality.
//! Import this module to get access to the most frequently used parser
//! combinators and types without having to import them individually.
//!
//! ## Usage
//!
//! ```rust
//! use dlexer::prelude::*;
//!
//! // Now you have access to:
//! // - BasicParser type alias
//! // - Common parser functions (alpha, digit, char, etc.)
//! // - Parser combinator methods (many, sep_by, etc.)
//! // - Error types and utilities
//! ```
//!
//! ## What's Included
//!
//! When you import the prelude, you get access to:
//! - Core parser types and aliases
//! - Character parsing functions
//! - Parser combinator methods
//! - Error handling types
//! - Lexical analysis utilities
//!
//! This provides the most commonly used items for building parsers quickly.

// Core parser types and type aliases
pub use crate::parsec::{BasicParser, BuildParser, Parsec};

// Essential parsing functions
pub use crate::parsec::{
    alpha, alphanumeric, any, char, decimal_digit, digit, eof, fail, hex_digit, newline,
    octal_digit, pure, satisfy,
};

// Error handling
pub use crate::errors::{ParserError, SimpleParserError};

// Lexical analysis
pub use crate::lex::{
    BlockSkipper, CharSkipper, LexIter, LexIterTrait, LineSkipper, NoSkipper, Skippers,
    WhitespaceSkipper, token,
};

// Convenience macros
pub use crate::{do_parse, map};

// Extra combinators for common patterns
pub use crate::parsec::extra::build_ident;
