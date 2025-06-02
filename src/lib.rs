pub mod errors;
pub mod lex;
pub mod parsec;

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

    (let% $v:ident = $m:expr; $($rest:tt)*) => {
        {let $v = $m; do_parse!($($rest)*)}
    };

    (let $v:ident = $m:expr; $($rest:tt)*) => {
        {let $v = $m; do_parse!($($rest)*)}
    };

    ($m:expr; $($rest:tt)*) => {
        $m.then(do_parse!($($rest)*))
    };
}

#[cfg(test)]
mod tests {

    use crate::{
        lex::{CharSkipper, LineSkipper, token},
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
        let ident = token(do_parse!(
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
            P::symbol("true") => AST::Boolean(true),
            P::symbol("false") => AST::Boolean(false)
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
    fn block_test() {
        let p: With<P, _> = alpha()
            .many()
            .collect::<String>()
            .sep1_till(char(';'), eof());

        let path = "/home/dexer/Repos/rust/dlexer/tests/test.txt";
        match p.parse_file(
            path,
            [LineSkipper("//").into(), CharSkipper(['\n', ' ']).into()],
        ) {
            Ok(a) => println!("Parsed block: {:?}", a),
            Err(e) => println!("Error parsing block: {}", e),
        }
    }
}
