# DLexer: A Functional Parser Combinator Library for Rust

[![Crates.io](https://img.shields.io/crates/v/dlexer.svg)](https://crates.io/crates/dlexer)
[![Docs.rs](https://docs.rs/dlexer/badge.svg)](https://docs.rs/dlexer)

DLexer is a high-performance, functional parser combinator library for Rust, designed for building elegant and efficient parsers. It provides a monadic interface for composing simple parsers into complex ones, supporting both text and binary formats with robust error handling.

## Key Features

- **✨ Elegant & Concise**: Write powerful parsers with minimal code. A complete JSON parser, for instance, is implemented in about 40 lines.
- **💪 Functional Core**: Built for functional programmers. Enjoy a monadic interface with familiar operators (`>>`, `|`, `+`) and methods (`map`, `bind`, `sep`). For OCaml and Haskell lovers, there are macros with do-notation style for sequential monadic operations.
- **📝 Extremely Versatile**: A unified API for parsing anything from simple text formats to complex binary data structures.
- **⚡ High-Performance**: Operates on input slices (`&str`, `&[u8]`) to minimize allocations and overhead.
- **⚙️ Advanced Control Flow**: Use the `do_parse!` macro for Haskell-style do-notation, simplifying sequential parsing logic.
- **🌪️ Whitespace & Comment Handling**: Built-in "skippers" automatically handle whitespace and comments (line and block), keeping your parsing logic clean.
- **🚨 Rich Error Reporting**: Provides detailed error messages with context, position, and expected inputs.

## Installation

Add DLexer to your `Cargo.toml`:

```toml
[dependencies]
dlexer = "0.1.1"
```

## Quick Start

Here’s a simple example of parsing a parenthesized, comma-separated list of numbers like `(1, 2, 3)`.

```rust
fn main() {
    // Define a parser for a decimal integer, surrounded by optional whitespace.
    let number: With<BasicParser, _> = integer(10);

    // Define a parser for a list of numbers, separated by commas.
    let number_list = number.sep(char(','));

    // Define a parser that expects the list to be enclosed in parentheses.
    let parser = number_list.between(char('('), char(')'));

    // Run the parser.
    // The `parse` method takes the input string and a "skipper" for whitespace.
    let result = parser.parse("( 1, 2, 3 )", WhitespaceSkipper);

    assert_eq!(result.unwrap(), vec![1, 2, 3]);
}
```

## Example: Hex Color Parser

DLexer is well-suited for parsing real-world formats. Here is a complete parser for CSS-style hex color codes (e.g., `#FF5733` or `#80FF5733` with an alpha channel).

```rust
pub fn hex_color() -> With<BasicParser, Color> {
    let u8_hex = hex_digit()
        .pair(hex_digit())
        .map(|(h1, h2)| u8::from_str_radix(&format!("{}{}", h1, h2), 16))
        .lift_err(|err| format!("Failed to parse u8 from hex digits: {:?}", err));

    u8_hex
        .many_till(eof())
        .between(char('#'), eof())
        .map(|v| match v.as_slice() {
            [a, r, g, b] => Ok(Color { a: *a, r: *r, g: *g, b: *b }),
            [r, g, b] => Ok(Color { a: 255, r: *r, g: *g, b: *b }),
            _ => Err("invalid hex color format"),
        })
        .lift_err(|e| e)
}
// Usage:
let color = hex_color().test("#FF5733").unwrap();
assert_eq!(color, Color { r: 255, g: 87, b: 51, a: 255 });

let color_with_alpha = hex_color().test("#80FF5733").unwrap();
assert_eq!(color_with_alpha, Color { r: 255, g: 87, b: 51, a: 128 });
```

## More Examples

You can find more complete examples in the `src/examples` directory:

- **[JSON Parser](./src/examples/json_parser.rs)**: A parser for the JSON data format, including support for comments and flexible whitespace.
- **[XML Parser](./src/examples/xml_parser.rs)**: A basic parser for a subset of the XML format.

## Macros

### `do_parse!`

A macro for writing parsers in a sequential, do-notation style. This macro provides a more imperative-looking syntax for chaining parsers, which can be more readable than deeply nested calls to `bind` and `then`.

**Syntax**

- `let% <var> = <parser>;` : Binds the result of `<parser>` to `<var>`. This is equivalent to `bind`.
- `<parser>;` : Runs `<parser>` and discards its result. This is equivalent to `then`.
- `let <var> = <expr>;` : Binds the result of a standard Rust expression to `<var>`.
- The last expression in the block is the final parser, which determines the return value.

**Example**

```rust
let p: With<BasicByteParser, _> = do_parse!(
    let% length = u32();
    let% bytes = n_bytes(length as usize);
    eof();
    pure(bytes)
);

let input = [0x00, 0x00, 0x00, 0x04, 0xAA, 0xBB, 0xCC, 0xDD]; // Represents "abcd" with length 4
match p.parse(&input) {
    Ok(a) => println!("Parsed successfully: {:?}", a),
    Err(e) => println!("{}", e),
}
```

### `map!`

A macro to map multiple parsers to a single value. This is useful when you want to create a parser that recognizes a set of keywords and maps them to an enum or other value.

**Example**

```rust
#[derive(Debug, PartialEq, Clone)]
enum Keyword {
    Let,
    In,
    If,
}

let keyword_parser: With<BasicParser, _> = map!(
    string("let") => Keyword::Let,
    string("in") => Keyword::In,
    string("if") => Keyword::If
);

assert_eq!(keyword_parser.test("let").unwrap(), Keyword::Let);
assert_eq!(keyword_parser.test("if").unwrap(), Keyword::If);
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
