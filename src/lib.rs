pub mod binary;
pub mod errors;
pub mod lex;
pub mod parsec;

//mod stream;

mod examples;

#[macro_export]
macro_rules! map {
    ($($parser:expr => $value:expr),*) => {
        $(
            $parser.map(|_| $value)
        )|*
    };
}

#[macro_export]
macro_rules! do_parse {
    // Single expression (not a binding)
    ($e:expr) => {
        $e
    };

    // One binding followed by more expressions
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
        lex::{WhitespaceSkipper, symbol, token},
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
}
