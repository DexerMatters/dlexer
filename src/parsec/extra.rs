use crate::parsec::{LexIterTrait, alpha, alphanumeric, char};
use crate::parsec::{Parsec, ParserError};

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
