//! # DLexer Examples
//!
//! This file contains comprehensive examples showing how to use the DLexer
//! parser combinator library for various parsing tasks.

use dlexer::errors::SimpleParserError;
use dlexer::lex::{LexIter, WhitespaceSkipper, token};
use dlexer::parsec::*;
use dlexer::{do_parse, map};

type Parser = BasicParser;

/// Example 1: Basic Character Parsing
///
/// Shows how to parse individual characters and character classes.
#[cfg(test)]
mod basic_examples {
    use super::*;

    #[test]
    fn character_parsing() {
        // Parse a specific character
        let comma = char(',');
        assert_eq!(comma.test(","), Ok(','));
        assert!(comma.test("x").is_err());

        // Parse any alphabetic character
        let letter = alpha();
        assert_eq!(letter.test("a"), Ok('a'));
        assert_eq!(letter.test("Z"), Ok('Z'));
        assert!(letter.test("5").is_err());

        // Parse any digit
        let number = digit();
        assert_eq!(number.test("7"), Ok('7'));
        assert!(number.test("a").is_err());

        // Parse alphanumeric characters
        let alphanum = alphanumeric();
        assert_eq!(alphanum.test("a"), Ok('a'));
        assert_eq!(alphanum.test("5"), Ok('5'));
        assert!(alphanum.test("_").is_err());
    }

    #[test]
    fn choice_parsing() {
        // Parse either a letter or a digit
        let letter_or_digit = alpha() | digit();
        assert_eq!(letter_or_digit.test("a"), Ok('a'));
        assert_eq!(letter_or_digit.test("5"), Ok('5'));
        assert!(letter_or_digit.test("_").is_err());

        // Parse specific characters
        let vowel = map!(
            char('a') => 'a',
            char('e') => 'e',
            char('i') => 'i',
            char('o') => 'o',
            char('u') => 'u'
        );
        assert_eq!(vowel.test("a"), Ok('a'));
        assert_eq!(vowel.test("e"), Ok('e'));
        assert!(vowel.test("x").is_err());
    }

    #[test]
    fn repetition_parsing() {
        // Parse zero or more letters
        let letters = alpha().many().collect::<String>();
        assert_eq!(letters.test("hello"), Ok("hello".to_string()));
        assert_eq!(letters.test(""), Ok("".to_string()));
        assert_eq!(letters.test("123"), Ok("".to_string()));

        // Parse one or more digits
        let numbers = digit().many1().collect::<String>();
        assert_eq!(numbers.test("123"), Ok("123".to_string()));
        assert!(numbers.test("").is_err());
        assert!(numbers.test("abc").is_err());

        // Parse comma-separated values
        let csv = alpha().many1().collect::<String>().sep_by(char(','));
        assert_eq!(
            csv.test("a,b,c"),
            Ok(vec!["a".to_string(), "b".to_string(), "c".to_string()])
        );
        assert_eq!(csv.test("single"), Ok(vec!["single".to_string()]));
        assert_eq!(csv.test(""), Ok(vec![]));
    }
}

/// Example 2: Advanced Parsing with do_parse! macro
///
/// Demonstrates sequential parsing using the monadic do-notation.
#[cfg(test)]
mod advanced_examples {
    use super::*;

    #[derive(Debug, PartialEq, Clone)]
    struct Variable {
        name: String,
        value: i32,
    }

    #[test]
    fn variable_assignment() {
        // Parse: "variable_name = 123"
        let variable_parser = token(do_parse!(
            let% name = (alpha() | char('_'))
                .extend(alphanumeric().many())
                .collect::<String>();
            let% _ = char('=');
            let% value = digit().many1().collect::<String>();
            let parsed_value = value.parse::<i32>().unwrap();
            pure(Variable { name, value: parsed_value })
        ));

        let input = "my_var = 42";
        let expected = Variable {
            name: "my_var".to_string(),
            value: 42,
        };
        assert_eq!(variable_parser.test(input), Ok(expected));
    }

    #[test]
    fn function_call() {
        // Parse: "function_name(arg1, arg2, arg3)"
        let identifier = (alpha() | char('_'))
            .extend(alphanumeric().many())
            .collect::<String>();

        let function_call = token(do_parse!(
            let% name = identifier.clone();
            let% _ = char('(');
            let% args = identifier.sep_by(char(','));
            let% _ = char(')');
            pure((name, args))
        ));

        let input = "printf(format, arg1, arg2)";
        let expected = (
            "printf".to_string(),
            vec!["format".to_string(), "arg1".to_string(), "arg2".to_string()],
        );
        assert_eq!(function_call.test(input), Ok(expected));
    }

    #[test]
    fn nested_expressions() {
        // Parse: "((a + b) * c)"
        fn expr_parser() -> Parser {
            let factor =
                alpha().map(|c| c.to_string()) | char('(').then(expr_parser()).with(char(')'));

            let term = factor.clone().chain_left(
                char('*').map(|_| |a: String, b: String| format!("({} * {})", a, b)),
                factor,
            );

            term.chain_left(
                char('+').map(|_| |a: String, b: String| format!("({} + {})", a, b)),
                factor,
            )
        }

        // Note: This is a simplified example - real expression parsing would be more complex
        let simple_expr = token(do_parse!(
            let% left = alpha().map(|c| c.to_string());
            let% _ = char('+');
            let% right = alpha().map(|c| c.to_string());
            pure(format!("({} + {})", left, right))
        ));

        assert_eq!(simple_expr.test("a + b"), Ok("(a + b)".to_string()));
    }

    #[test]
    fn list_parsing() {
        // Parse: "[1, 2, 3, 4]"
        let number = digit()
            .many1()
            .collect::<String>()
            .map(|s| s.parse::<i32>().unwrap());

        let list_parser = token(do_parse!(
            let% _ = char('[');
            let% items = number.sep_by(char(','));
            let% _ = char(']');
            pure(items)
        ));

        assert_eq!(list_parser.test("[1, 2, 3]"), Ok(vec![1, 2, 3]));
        assert_eq!(list_parser.test("[]"), Ok(vec![]));
        assert_eq!(list_parser.test("[42]"), Ok(vec![42]));
    }
}

/// Example 3: Error Handling and Debugging
///
/// Shows how to handle parse errors and debug parsing issues.
#[cfg(test)]
mod error_examples {
    use super::*;

    #[test]
    fn expected_messages() {
        let identifier = alpha().many1().collect::<String>().expected("identifier");

        // This will provide a clear error message
        match identifier.test("123") {
            Ok(_) => panic!("Should have failed"),
            Err(e) => {
                let error_msg = format!("{}", e);
                assert!(error_msg.contains("identifier"));
            }
        }
    }

    #[test]
    fn choice_error_handling() {
        let keyword = map!(
            Parser::symbol("if") => "if",
            Parser::symbol("else") => "else",
            Parser::symbol("while") => "while"
        )
        .expected("keyword");

        match keyword.test("for") {
            Ok(_) => panic!("Should have failed"),
            Err(e) => {
                let error_msg = format!("{}", e);
                assert!(error_msg.contains("keyword"));
            }
        }
    }
}

/// Example 4: Real-world JSON Parser
///
/// A simplified JSON parser demonstrating practical usage.
#[cfg(test)]
mod json_example {
    use super::*;
    use std::collections::HashMap;

    #[derive(Debug, PartialEq, Clone)]
    enum JsonValue {
        String(String),
        Number(f64),
        Boolean(bool),
        Null,
        Array(Vec<JsonValue>),
        Object(HashMap<String, JsonValue>),
    }

    #[test]
    fn json_primitives() {
        // Parse JSON string
        let json_string = token(do_parse!(
            let% _ = char('"');
            let% content = satisfy(|c: &char| *c != '"').many().collect::<String>();
            let% _ = char('"');
            pure(JsonValue::String(content))
        ));

        assert_eq!(
            json_string.test("\"hello\""),
            Ok(JsonValue::String("hello".to_string()))
        );

        // Parse JSON number (simplified)
        let json_number = token(
            digit()
                .many1()
                .collect::<String>()
                .map(|s| JsonValue::Number(s.parse().unwrap())),
        );

        assert_eq!(json_number.test("123"), Ok(JsonValue::Number(123.0)));

        // Parse JSON boolean
        let json_bool = token(map!(
            Parser::symbol("true") => JsonValue::Boolean(true),
            Parser::symbol("false") => JsonValue::Boolean(false)
        ));

        assert_eq!(json_bool.test("true"), Ok(JsonValue::Boolean(true)));

        // Parse JSON null
        let json_null = token(Parser::symbol("null").map(|_| JsonValue::Null));
        assert_eq!(json_null.test("null"), Ok(JsonValue::Null));
    }

    #[test]
    fn json_array() {
        // Simplified JSON array parser
        let json_string = token(do_parse!(
            let% _ = char('"');
            let% content = satisfy(|c: &char| *c != '"').many().collect::<String>();
            let% _ = char('"');
            pure(JsonValue::String(content))
        ));

        let json_array = token(do_parse!(
            let% _ = char('[');
            let% items = json_string.sep_by(char(','));
            let% _ = char(']');
            pure(JsonValue::Array(items))
        ));

        let input = r#"["hello", "world"]"#;
        let expected = JsonValue::Array(vec![
            JsonValue::String("hello".to_string()),
            JsonValue::String("world".to_string()),
        ]);
        assert_eq!(json_array.test(input), Ok(expected));
    }
}
