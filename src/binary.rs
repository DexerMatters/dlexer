//! Tools for parsing and processing binary data.
//!
//! This module provides parser combinators for working with binary formats
//! by treating byte sequences as the input stream instead of character streams.
//!
//! # Components
//!
//! - [`ByteLexIter`] - Lexer iterator for reading byte arrays
//! - Parser functions for various numeric types (u8, u16, i32, etc.)
//! - Extension methods on [`Parsec<ByteLexIter, E, A>`](crate::parsec::Parsec)
//!
//! # Endianness Conventions
//!
//! This module follows these endianness conventions:
//!
//! - Unsigned types (`u16`, `u32`, etc.): **Big-endian** byte order
//! - Signed types (`i8`, `i16`, etc.): **Little-endian** byte order
//! - Floating-point types: Use bit patterns from their respective unsigned integer types
//!
//! # Usage Example
//!
//! ```rust
//! use dlexer::binary::{n_bytes, u32, BasicByteParser};
//! use dlexer::parsec::*;
//!
//! // Parse a binary structure with a length prefix followed by that many bytes
//! let parser: BasicByteParser = do_parse!(
//!     let% length = u32();        // Read a 32-bit unsigned integer (big-endian)
//!     let% bytes = n_bytes(length as usize); // Read the specified number of bytes
//!     eof();                      // Ensure we've consumed all input
//!     pure(bytes)                 // Return the bytes as the result
//! );
//!
//! // Input representing a 4-byte payload with length prefix
//! let input = [0x00, 0x00, 0x00, 0x04, 0xAA, 0xBB, 0xCC, 0xDD];
//! let result = parser.parse(&input);
//! ```
//!
//! # Safety
//!
//! The `align_to` function is marked as unsafe because it performs unsafe memory operations.
//! It should be used with caution and only when the caller can guarantee proper memory alignment.

use std::fmt::Debug;
use std::rc::Rc;

use crate::{
    errors::{ParserError, SimpleParserError},
    lex::{LexIterState, LexIterTrait},
    parsec::{any, Parsec},
};

/// A type alias for a basic parser that works with binary data.
///
/// This is a convenient shorthand for a binary parser that produces `u8` values
/// with a default error type.
pub type BasicByteParser = Parsec<ByteLexIter, SimpleParserError<[u8]>, u8>;

/// A lexer iterator for processing binary data.
///
/// Reads and processes raw bytes from a byte array, providing the foundation
/// for binary parsers in the DLexer framework.
#[derive(Clone, Debug)]
pub struct ByteLexIter {
    state: LexIterState<[u8]>,
}

impl ByteLexIter {
    /// Creates a new `ByteLexIter` from a byte slice.
    pub fn new(context: &[u8]) -> Self {
        ByteLexIter {
            state: LexIterState {
                context: Rc::from(context),
                current_line: 1,
                current_column: 0,
                current_pos: 0,
                current_indent: 0,
                indent_flag: false,
                position: 0,
            },
        }
    }
}

/// Implementation of `LexIterTrait` for `ByteLexIter`.
impl LexIterTrait for ByteLexIter {
    /// The source context for this lexer is a byte array.
    type Context = [u8];

    /// Each item produced by this lexer is a single byte.
    type Item = u8;

    fn get_state(&self) -> LexIterState<Self::Context> {
        self.state.clone()
    }

    fn get_state_mut(&mut self) -> &mut LexIterState<Self::Context> {
        &mut self.state
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.state.position >= self.state.context.len() {
            return None;
        }
        let item = self.state.context[self.state.position];
        self.state.position += 1;
        self.state.current_pos += 1;
        Some(item)
    }
}

/// Extension methods for binary parsers.
impl<E, A> Parsec<ByteLexIter, E, A>
where
    E: ParserError<Context = [u8]> + 'static,
    A: 'static,
{
    /// Parses a byte slice with this parser.
    ///
    /// Takes a byte slice as input and returns the parsing result or an error.
    pub fn parse(&self, input: &[u8]) -> Result<A, E> {
        let lexer = ByteLexIter::new(input);
        self.run(lexer)
    }

    /// Parses a file containing binary data with this parser.
    ///
    /// The file at the specified path will be read completely into memory
    /// before parsing.
    pub fn parse_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<A, E> {
        use std::fs::File;
        use std::io::{BufReader, Read};
        let file = File::open(path).unwrap();
        let mut reader = BufReader::new(file);
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).unwrap();
        let lexer = ByteLexIter::new(&buffer);
        self.run(lexer)
    }

    /// Creates a parser that adds debug output when parsing binary data.
    ///
    /// This is useful for debugging binary parsers as it will show:
    /// - The input bytes in hexadecimal format
    /// - The parsed value
    /// - The position after parsing
    /// - The remaining bytes
    /// - How many bytes were consumed
    pub fn dbg_(self) -> Parsec<ByteLexIter, E, A>
    where
        A: Debug,
    {
        Parsec::new(move |input: ByteLexIter| {
            let original = input.get_state();
            println!("Binary Input:");
            println!("  Position:\t{}", original.current_pos);

            // Show a hex dump of the next few bytes
            let remaining = &original.context[original.position..];
            let preview_len = std::cmp::min(16, remaining.len());
            if preview_len > 0 {
                let preview = &remaining[..preview_len];
                let hex_preview: Vec<String> =
                    preview.iter().map(|b| format!("{:02x}", b)).collect();
                println!(
                    "  Parsing:\t[{}]{}",
                    hex_preview.join(" "),
                    if remaining.len() > preview_len {
                        "..."
                    } else {
                        ""
                    }
                );
            } else {
                println!("  Parsing:\tEnd of input");
            }

            let result = self.eval(input.clone());
            match result {
                Ok((next_input, value)) => {
                    let next = next_input.get_state();
                    println!("Binary Output:");
                    println!("  Value:\t{:?}", value);
                    println!("  Position:\t{}", next.current_pos);

                    let next_remaining = &next.context[next.position..];
                    let next_preview_len = std::cmp::min(16, next_remaining.len());
                    if next_preview_len > 0 {
                        let next_preview = &next_remaining[..next_preview_len];
                        let next_hex_preview: Vec<String> =
                            next_preview.iter().map(|b| format!("{:02x}", b)).collect();
                        println!(
                            "  Remaining:\t[{}]{}",
                            next_hex_preview.join(" "),
                            if next_remaining.len() > next_preview_len {
                                "..."
                            } else {
                                ""
                            }
                        );
                    } else {
                        println!("  Remaining:\tEnd of input");
                    }

                    // Calculate and display how many bytes were consumed
                    let consumed = next.current_pos - original.current_pos;
                    println!("  Consumed:\t{} bytes", consumed);

                    Ok((next_input, value))
                }
                Err(error) => {
                    println!("Binary Error:");
                    println!("  Message:\t{}", error);
                    Err(error)
                }
            }
        })
    }
}

/// Creates a parser that consumes a single byte from the input.
///
/// Returns the byte as a `u8` value.
#[inline]
pub fn byte<S, E>() -> Parsec<S, E, u8>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    any()
}

/// Creates a parser for a specified number of bytes.
///
/// Consumes exactly `n` bytes from the input stream and returns them as a `Vec<u8>`.
#[inline]
pub fn n_bytes<S, E>(n: usize) -> Parsec<S, E, Vec<u8>>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    any().take(n)
}

/// Creates a parser for an unsigned 8-bit integer.
///
/// This function is an alias for [`byte()`], provided for consistency with other numeric parsers.
#[inline]
pub fn u8<S, E>() -> Parsec<S, E, u8>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    byte()
}

/// Creates a parser for an unsigned 16-bit integer (u16) in big-endian byte order.
#[inline]
pub fn u16<S, E>() -> Parsec<S, E, u16>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + Clone + 'static,
{
    n_bytes(2)
        .map(|bytes| bytes.try_into().map(u16::from_be_bytes))
        .lift_err(|err| format!("invalid u16 bytes: {:?}", err))
}

/// Creates a parser for an unsigned 32-bit integer (u32) in big-endian byte order.
#[inline]
pub fn u32<S, E>() -> Parsec<S, E, u32>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + Clone + 'static,
{
    n_bytes(4)
        .map(|bytes| bytes.try_into().map(u32::from_be_bytes))
        .lift_err(|err| format!("invalid u32 bytes: {:?}", err))
}

/// Creates a parser for an unsigned 64-bit integer (u64) in big-endian byte order.
#[inline]
pub fn u64<S, E>() -> Parsec<S, E, u64>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + Clone + 'static,
{
    n_bytes(8)
        .map(|bytes| bytes.try_into().map(u64::from_be_bytes))
        .lift_err(|err| format!("invalid u64 bytes: {:?}", err))
}

/// Creates a parser for an unsigned 128-bit integer (u128) in big-endian byte order.
#[inline]
pub fn u128<S, E>() -> Parsec<S, E, u128>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + Clone + 'static,
{
    n_bytes(16)
        .map(|bytes| bytes.try_into().map(u128::from_be_bytes))
        .lift_err(|err| format!("invalid u128 bytes: {:?}", err))
}

/// Creates a parser for a signed 8-bit integer (i8) in little-endian byte order.
#[inline]
pub fn i8<S, E>() -> Parsec<S, E, i8>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    byte().map(|b| i8::from_le_bytes([b]))
}

/// Creates a parser for a signed 16-bit integer (i16) in little-endian byte order.
#[inline]
pub fn i16<S, E>() -> Parsec<S, E, i16>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    n_bytes(2)
        .map(|bytes| bytes.try_into().map(i16::from_le_bytes))
        .lift_err(|err| format!("invalid i16 bytes: {:?}", err))
}

/// Creates a parser for a signed 32-bit integer (i32) in little-endian byte order.
#[inline]
pub fn i32<S, E>() -> Parsec<S, E, i32>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    n_bytes(4)
        .map(|bytes| bytes.try_into().map(i32::from_le_bytes))
        .lift_err(|err| format!("invalid i32 bytes: {:?}", err))
}

/// Creates a parser for a signed 64-bit integer (i64) in little-endian byte order.
#[inline]
pub fn i64<S, E>() -> Parsec<S, E, i64>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    n_bytes(8)
        .map(|bytes| bytes.try_into().map(i64::from_le_bytes))
        .lift_err(|err| format!("invalid i64 bytes: {:?}", err))
}

/// Creates a parser for a signed 128-bit integer (i128) in little-endian byte order.
#[inline]
pub fn i128<S, E>() -> Parsec<S, E, i128>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + 'static,
{
    n_bytes(16)
        .map(|bytes| bytes.try_into().map(i128::from_le_bytes))
        .lift_err(|err| format!("invalid i128 bytes: {:?}", err))
}

/// Creates a parser for a 32-bit floating-point number.
///
/// Reads 4 bytes in big-endian order and interprets the bit pattern as an `f32`
/// using [`f32::from_bits`].
#[inline]
pub fn f32<S, E>() -> Parsec<S, E, f32>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + Clone + 'static,
{
    u32().map(|bits| f32::from_bits(bits))
}

/// Creates a parser for a 64-bit floating-point number (f64).
/// This converts the bit pattern of a u64 to an f64.
#[inline]
pub fn f64<S, E>() -> Parsec<S, E, f64>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + Clone + 'static,
{
    u64().map(|bits| f64::from_bits(bits))
}

/// Creates a parser that aligns the input bytes to a specific type `T` and returns a pointer to it.
///
/// This function is useful for working with binary data formats where you need to interpret
/// a sequence of bytes as a specific structured type.
///
/// # Safety
///
/// This function is unsafe because:
/// - It uses the unsafe [`slice::align_to`] method internally
/// - It returns a raw pointer that may be invalidated if the input data is moved or dropped
/// - The caller must ensure that the alignment requirements of type `T` are satisfied
/// - The caller must ensure that the memory layout of `T` matches the expected binary format
///
/// Only use this when you have complete control over the memory layout and alignment.
#[inline]
pub unsafe fn align_to<S, E, T>() -> Parsec<S, E, *const T>
where
    S: LexIterTrait<Item = u8, Context = [u8]> + Clone + 'static,
    E: ParserError<Context = [u8]> + 'static,
    T: 'static,
{
    let size = std::mem::size_of::<T>();
    n_bytes(size)
        .map(move |bytes| {
            let (head, body, _tail) = unsafe { bytes.align_to::<T>() };
            if head.is_empty() {
                return Err("bad alignment");
            }
            Ok(body.first().unwrap() as *const T)
        })
        .lift_err(|err| err)
}
