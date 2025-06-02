# DLexer - A Rust Parser Combinator Library

DLexer is a functional parser combinator library for Rust that provides a flexible and composable approach to building parsers. It combines lexical analysis with parser combinators, offering both monadic and applicative interfaces for building complex parsers from simple components.

## Features

- **Parser Combinators**: Build complex parsers by combining simple ones
- **Lexical Analysis**: Built-in tokenization and whitespace handling
- **Error Handling**: Comprehensive error reporting with position tracking
- **Monadic Interface**: Use `do_parse!` macro for sequential parsing
- **Applicative Interface**: Pure functional style with `apply` methods
- **Flexible Skipping**: Configurable whitespace and comment skipping

## Quick Start

```rust
use dlexer::prelude::*;

// Parse identifiers and booleans
let ident = token(do_parse!(
    let% initial = (alpha() | char('_'));
    let% rest    = alphanumeric().many().collect::<String>();
    let  result  = format!("{}{}", initial, rest);
    pure(result)
));

let boolean = token(map!(
    BasicParser::symbol("true") => true,
    BasicParser::symbol("false") => false
));

let parser = (ident | boolean.map(|b| b.to_string()))
    .sep_till(char(','), eof());

// Parse comma-separated identifiers and booleans
let input = "foo, bar, true, false";
let result = parser.test(input);
```

## Architecture

- **`lib.rs`**: Main library with macros and integration tests
- **`errors.rs`**: Error handling traits and implementations
- **`lex.rs`**: Lexical analysis, tokenization, and character handling
- **`parsec/`**: Parser combinator implementations
- **`generic.rs`**: Generic utilities
- **`prelude.rs`**: Common imports

## Examples

See the test cases in `lib.rs` for practical usage examples, including identifier parsing, boolean parsing, and list parsing with separators.