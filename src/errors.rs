use std::fmt::Display;

use crate::lex::LexIterState;

pub trait ParserError: Display {
    type Context: ?Sized;
    fn eof(state: (LexIterState<Self::Context>, LexIterState<Self::Context>)) -> Self
    where
        Self: Sized,
    {
        Self::unexpected(state, EOFError)
    }
    fn unexpected<T: Display>(
        state: (LexIterState<Self::Context>, LexIterState<Self::Context>),
        item: T,
    ) -> Self;

    fn with_expected<T: Display>(self, expected: T) -> Self
    where
        Self: Sized;

    fn with_state(self, from: LexIterState<Self::Context>, to: LexIterState<Self::Context>)
    -> Self;

    fn from(&self) -> LexIterState<Self::Context>;

    fn to(&self) -> LexIterState<Self::Context> {
        self.from()
    }
}

#[derive(Debug, Clone)]
struct EOFError;

impl Display for EOFError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "end of input")
    }
}

#[derive(Debug)]
pub struct SimpleParserError<T: ?Sized> {
    expected: String,
    unexpected: String,
    from: LexIterState<T>,
    to: LexIterState<T>,
}

impl<T: ?Sized> Clone for SimpleParserError<T> {
    fn clone(&self) -> Self {
        Self {
            expected: self.expected.clone(),
            unexpected: self.unexpected.clone(),
            from: LexIterState::clone(&self.from),
            to: LexIterState::clone(&self.to),
        }
    }
}

impl<T: ?Sized> SimpleParserError<T> {
    pub fn new(
        expected: String,
        unexpected: String,
        state: (LexIterState<T>, LexIterState<T>),
    ) -> Self {
        Self {
            expected,
            unexpected,
            from: state.0,
            to: state.1,
        }
    }
}

impl<U: ?Sized> ParserError for SimpleParserError<U> {
    type Context = U;
    fn unexpected<T>(
        state: (LexIterState<Self::Context>, LexIterState<Self::Context>),
        item: T,
    ) -> Self
    where
        T: Display,
    {
        Self::new(String::new(), item.to_string(), state)
    }

    fn with_expected<T>(mut self, expected: T) -> Self
    where
        T: Display,
    {
        self.expected = expected.to_string();
        self
    }
    fn with_state(
        mut self,
        from: LexIterState<Self::Context>,
        to: LexIterState<Self::Context>,
    ) -> Self {
        self.from = from;
        self.to = to;
        self
    }
    fn from(&self) -> LexIterState<Self::Context> {
        self.from.clone()
    }
    fn to(&self) -> LexIterState<Self::Context> {
        self.to.clone()
    }
}

impl<T: ?Sized> Display for SimpleParserError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let from = format!("{}:{}", self.from.current_line, self.from.current_column);
        let to = format!("{}:{}", self.to.current_line, self.to.current_column);
        let expected = replace_escapes(&self.expected);
        let unexpected = replace_escapes(&self.unexpected);
        if self.expected.is_empty() {
            write!(f, "Unexpected '{}'", unexpected)?;
        } else {
            write!(f, "Expected '{}', found '{}'", expected, unexpected)?;
        }
        write!(f, " at {} to {}", from, to)
    }
}

fn replace_escapes(s: impl Display) -> String {
    s.to_string()
        .replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\t', "\\t")
        .replace('\r', "\\r")
        .replace('\0', "\\0")
}
