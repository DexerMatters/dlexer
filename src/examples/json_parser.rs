#![allow(dead_code)]

use crate::{
    lex::{number, symbol, token},
    map,
    parsec::{BasicParser, With, any, char, extra::build_ident, rec},
};

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

pub fn json_value() -> With<BasicParser, JsonValue> {
    let esc = char('\\')
        >> map!(
            char('"') => '"',
            char('n') => '\n',
            char('t') => '\t',
            char('\\') => '\\',
            char('\"') => '\"'
        );
    let string = token(
        (esc | any().not('"'))
            .many()
            .between(char('"'), char('"'))
            .collect()
            .expected("string"),
    );

    let number = number().expected("number") & JsonValue::Number;

    let boolean = map!(
        symbol("true") => true,
        symbol("false") => false
    )
    .expected("boolean")
        & JsonValue::Boolean;

    let null = symbol("null").expected("null") & |_| JsonValue::Null;

    let array = rec(json_value).sep(char(',')).between(char('['), char(']')) & JsonValue::Array;

    let key = string.clone() | build_ident(["json", "null", "true", "false"]);
    let object = (key.with(char(':')) + rec(json_value))
        .sep(char(','))
        .between(char('{'), char('}'))
        & JsonValue::Object;

    let string = string & JsonValue::String;

    object | array | string | number | boolean | null
}

#[cfg(test)]
mod tests {
    use crate::lex::{BlockSkipper, CharSkipper, LineSkipper};

    use super::*;
    #[test]
    fn test_json_parser() {
        let parser = json_value();
        let input = r#"{
            "name": "John", // This is a comment
            "age": /* This is a block comment */ 30,
            "is_student": false,
            "courses": ["Math", "Science"],
            "address": {
                "street": "123 Main St",
                "city": "Anytown"
            },
            "grades": [85.5, 90.0, 78.5],
            "graduated": null
        }"#;
        let result = parser.parse(
            input,
            [
                CharSkipper(['\n', '\t', ' ']).into(),
                LineSkipper("//").into(),
                BlockSkipper("/*", "*/").into(),
            ],
        );
        match result {
            Ok(value) => println!("Parsed JSON value: {:#?}", value),
            Err(e) => println!("Error parsing JSON: {}", e),
        }
    }
}
