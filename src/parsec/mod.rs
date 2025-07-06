pub mod extra;

use std::{
    fmt::{Debug, Display},
    ops::{Add, BitAnd, BitOr, Range, RangeBounds, RangeInclusive, Shl, Shr},
    rc::Rc,
};

use crate::{
    errors::{ParserError, SimpleParserError},
    lex::{LexIter, LexIterState, LexIterTrait},
};

pub type BuildParser<S, E> = Parsec<S, E, <S as Iterator>::Item>;

pub type BasicParser = Parsec<LexIter, SimpleParserError<str>, <LexIter as LexIterTrait>::Item>;

pub trait Take<T> {
    type Output: RangeBounds<T>;
    fn as_range(self) -> Self::Output;
}

impl Take<usize> for Range<usize> {
    type Output = Range<usize>;
    fn as_range(self) -> Self::Output {
        self
    }
}

impl Take<usize> for RangeInclusive<usize> {
    type Output = RangeInclusive<usize>;
    fn as_range(self) -> Self::Output {
        self
    }
}

impl Take<usize> for usize {
    type Output = RangeInclusive<usize>;
    fn as_range(self) -> Self::Output {
        self - 1..=self - 1
    }
}

#[derive(Clone)]
pub struct Parsec<S: LexIterTrait, E: ParserError, A> {
    run: Rc<dyn Fn(S) -> Result<(S, A), E>>,
}

impl<S, E, A> Parsec<S, E, A>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
{
    pub fn new(run: impl Fn(S) -> Result<(S, A), E> + 'static) -> Self {
        Parsec { run: Rc::new(run) }
    }

    pub fn eval(&self, input: S) -> Result<(S, A), E> {
        (self.run)(input)
    }

    pub fn run(&self, input: S) -> Result<A, E> {
        (self.run)(input).map(|(_, value)| value)
    }

    pub fn map<B: 'static, F>(self, f: F) -> Parsec<S, E, B>
    where
        F: Fn(A) -> B + 'static,
    {
        Parsec::new(move |input| {
            let (next_input, value) = self.eval(input)?;
            Ok((next_input, f(value)))
        })
    }

    pub fn expected<T>(self, expected: T) -> Parsec<S, E, A>
    where
        T: Display + Clone + 'static,
        E: ParserError<Context = S::Context> + 'static,
    {
        Parsec::new(move |input: S| {
            let original_state = input.get_state();
            let result = self.eval(input);
            match result {
                Ok((next_input, value)) => Ok((next_input, value)),
                Err(error) => {
                    let to = error.to();
                    Err(error
                        .with_expected(expected.clone())
                        .with_state(original_state, to))
                }
            }
        })
    }

    pub fn bind<B: 'static, F>(self, f: F) -> Parsec<S, E, B>
    where
        F: Fn(A) -> Parsec<S, E, B> + 'static,
    {
        Parsec::new(move |input| {
            let (next_input, value) = self.eval(input)?;
            f(value).eval(next_input)
        })
    }

    pub fn apply<B: 'static, R: 'static>(self, arg: Parsec<S, E, B>) -> Parsec<S, E, R>
    where
        A: FnOnce(B) -> R + 'static,
    {
        Parsec::new(move |input| {
            let (next_input, func) = self.eval(input)?;
            let (final_input, arg_value) = arg.eval(next_input)?;
            Ok((final_input, func(arg_value)))
        })
    }

    pub fn then<B: 'static>(self, other: Parsec<S, E, B>) -> Parsec<S, E, B> {
        Parsec::new(move |input| {
            let (next_input, _) = self.eval(input)?;
            other.eval(next_input)
        })
    }

    pub fn with<B: 'static>(self, other: Parsec<S, E, B>) -> Parsec<S, E, A> {
        Parsec::new(move |input| {
            let (next_input, value) = self.eval(input)?;
            let (final_input, _) = other.eval(next_input)?;
            Ok((final_input, value))
        })
    }

    pub fn between<B: 'static, C: 'static>(
        self,
        left: Parsec<S, E, B>,
        right: Parsec<S, E, C>,
    ) -> Parsec<S, E, A> {
        Parsec::new(move |input: S| {
            let (next_input, _) = left.eval(input)?;
            let (final_input, value) = self.eval(next_input)?;
            let (end_input, _) = right.eval(final_input)?;
            Ok((end_input, value))
        })
    }

    pub fn or(self, other: Parsec<S, E, A>) -> Parsec<S, E, A>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| match self.eval(input.clone()) {
            Ok((next_input, value)) => Ok((next_input, value)),
            Err(err) => match other.eval(input) {
                Ok((next_input, value)) => Ok((next_input, value)),
                Err(_) => Err(err),
            },
        })
    }

    pub fn opt(self) -> Parsec<S, E, Option<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| match self.eval(input.clone()) {
            Ok((next_input, value)) => Ok((next_input, Some(value))),
            Err(_) => Ok((input, None)),
        })
    }

    pub fn one(self) -> Parsec<S, E, Vec<A>> {
        Parsec::new(move |input: S| {
            let (next_input, value) = self.eval(input)?;
            Ok((next_input, vec![value]))
        })
    }

    pub fn try_(self) -> Parsec<S, E, Result<A, E>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| match self.eval(input.clone()) {
            Ok((next_input, value)) => Ok((next_input, Ok(value))),
            Err(error) => Ok((input, Err(error))),
        })
    }

    pub fn many(self) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let mut results = Vec::new();
            let mut current_input = input;
            while let Ok((new_input, value)) = self.eval(current_input.clone()) {
                results.push(value);
                current_input = new_input;
            }
            Ok((current_input, results))
        })
    }

    pub fn many1(self) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
        A: Debug,
    {
        Parsec::new(move |input: S| {
            let (mut current_input, first_value) = self.eval(input)?;
            let mut results = vec![first_value];

            while let Ok((new_input, value)) = self.eval(current_input.clone()) {
                results.push(value);
                current_input = new_input;
            }

            Ok((current_input, results))
        })
    }

    pub fn take<R>(self, range: R) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
        R: Take<usize> + 'static,
    {
        let range = range.as_range();
        Parsec::new(move |input: S| {
            let mut results = Vec::new();
            let mut current_input = input;
            let mut count = 0;

            let start = match range.start_bound() {
                std::ops::Bound::Included(&n) => n,
                std::ops::Bound::Excluded(&n) => n + 1,
                std::ops::Bound::Unbounded => 0,
            };

            let end = match range.end_bound() {
                std::ops::Bound::Included(&n) => n + 1,
                std::ops::Bound::Excluded(&n) => n,
                std::ops::Bound::Unbounded => usize::MAX,
            };

            while count < end {
                match self.eval(current_input.clone()) {
                    Ok((new_input, value)) => {
                        results.push(value);
                        current_input = new_input;
                        count += 1;
                    }
                    Err(err) => {
                        if count < start {
                            return Err(err);
                        }
                        break;
                    }
                }
            }

            if count < start {
                return Err(E::eof((
                    current_input.get_state(),
                    current_input.get_state(),
                )));
            }

            Ok((current_input, results))
        })
    }

    pub fn many_till<B: 'static>(self, end: Parsec<S, E, B>) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let mut results = Vec::new();
            let mut current_input = input;

            // Already check if the end condition is met
            if end.eval(current_input.clone()).is_ok() {
                return Ok((current_input, results));
            }

            loop {
                match self.eval(current_input.clone()) {
                    Ok((new_input, value)) => {
                        results.push(value);
                        current_input = new_input;
                    }
                    Err(err) => return Err(err),
                }

                if end.eval(current_input.clone()).is_ok() {
                    return Ok((current_input, results));
                }
            }
        })
    }

    pub fn many1_till<B: 'static>(self, end: Parsec<S, E, B>) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let (mut current_input, first_value) = self.eval(input)?;
            let mut results = vec![first_value];

            loop {
                if end.eval(current_input.clone()).is_ok() {
                    return Ok((current_input, results));
                }

                match self.eval(current_input.clone()) {
                    Ok((new_input, value)) => {
                        results.push(value);
                        current_input = new_input;
                    }
                    Err(_) => return Ok((current_input, results)),
                }
            }
        })
    }

    pub fn sep<T: 'static>(self, sep: Parsec<S, E, T>) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let mut results = Vec::new();
            let mut current_input = input;

            if let Ok((new_input, first_value)) = self.eval(current_input.clone()) {
                results.push(first_value);
                current_input = new_input;
            } else {
                return Ok((current_input, results));
            }

            while let Ok((sep_input, _)) = sep.eval(current_input.clone()) {
                if let Ok((new_input, value)) = self.eval(sep_input) {
                    results.push(value);
                    current_input = new_input;
                } else {
                    break;
                }
            }
            Ok((current_input, results))
        })
    }
    pub fn sep1(self, sep: Parsec<S, E, S::Item>) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let (mut current_input, first_value) = self.eval(input)?;
            let mut results = vec![first_value];

            while let Ok((sep_input, _)) = sep.eval(current_input.clone()) {
                if let Ok((new_input, value)) = self.eval(sep_input) {
                    results.push(value);
                    current_input = new_input;
                } else {
                    break;
                }
            }
            Ok((current_input, results))
        })
    }

    pub fn sep_till<T: 'static, U: 'static>(
        self,
        sep: Parsec<S, E, T>,
        end: Parsec<S, E, U>,
    ) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let mut results = Vec::new();
            let mut current_input = input;

            // Check if the end condition is met immediately
            if end.eval(current_input.clone()).is_ok() {
                return Ok((current_input, results));
            }

            // Parse the first item if possible
            match self.eval(current_input.clone()) {
                Ok((new_input, value)) => {
                    results.push(value);
                    current_input = new_input;
                }
                Err(err) => return Err(err),
            }

            loop {
                // Check if the end condition is met
                if end.eval(current_input.clone()).is_ok() {
                    return Ok((current_input, results));
                }

                // Try to parse separator followed by an item
                match sep.eval(current_input.clone()) {
                    Ok((sep_input, _)) => match self.eval(sep_input) {
                        Ok((new_input, value)) => {
                            results.push(value);
                            current_input = new_input;
                        }
                        Err(err) => return Err(err),
                    },
                    Err(err) => {
                        // If we can't parse a separator, check if end condition is met
                        if end.eval(current_input.clone()).is_ok() {
                            return Ok((current_input, results));
                        } else {
                            // Neither separator nor end condition could be parsed
                            return Err(err);
                        }
                    }
                }
            }
        })
    }

    pub fn sep1_till<T: 'static, U: 'static>(
        self,
        sep: Parsec<S, E, T>,
        end: Parsec<S, E, U>,
    ) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            // Must parse at least one item
            let (mut current_input, first_value) = self.eval(input)?;
            let mut results = vec![first_value];

            loop {
                // Check if the end condition is met
                if end.eval(current_input.clone()).is_ok() {
                    return Ok((current_input, results));
                }

                // Try to parse separator followed by an item
                match sep.eval(current_input.clone()) {
                    Ok((sep_input, _)) => match self.eval(sep_input) {
                        Ok((new_input, value)) => {
                            results.push(value);
                            current_input = new_input;
                        }
                        Err(err) => return Err(err),
                    },
                    Err(err) => {
                        // If we can't parse a separator, check if end condition is met
                        if end.eval(current_input.clone()).is_ok() {
                            return Ok((current_input, results));
                        } else {
                            // Neither separator nor end condition could be parsed
                            return Err(err);
                        }
                    }
                }
            }
        })
    }

    pub fn sep_take<T: 'static, R>(
        self,
        sep: Parsec<S, E, T>,
        range: impl Take<usize> + 'static,
    ) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        let range = range.as_range();
        Parsec::new(move |input: S| {
            let mut results = Vec::new();
            let mut current_input = input;
            let mut count = 0;

            let start = match range.start_bound() {
                std::ops::Bound::Included(&n) => n,
                std::ops::Bound::Excluded(&n) => n + 1,
                std::ops::Bound::Unbounded => 0,
            };

            let end = match range.end_bound() {
                std::ops::Bound::Included(&n) => n + 1,
                std::ops::Bound::Excluded(&n) => n,
                std::ops::Bound::Unbounded => usize::MAX,
            };

            // Try to parse the first item
            if count < end {
                match self.eval(current_input.clone()) {
                    Ok((new_input, first_value)) => {
                        results.push(first_value);
                        current_input = new_input;
                        count += 1;
                    }
                    Err(err) => {
                        // If we can't parse the first item and minimum is 0, return empty
                        if start == 0 {
                            return Ok((current_input, results));
                        } else {
                            return Err(err);
                        }
                    }
                }
            }

            // Parse additional items with separators
            while count < end {
                match sep.eval(current_input.clone()) {
                    Ok((sep_input, _)) => {
                        match self.eval(sep_input) {
                            Ok((new_input, value)) => {
                                results.push(value);
                                current_input = new_input;
                                count += 1;
                            }
                            Err(err) => {
                                // If we can't parse the item after separator, check if we have enough
                                if count < start {
                                    return Err(err);
                                }
                                break;
                            }
                        }
                    }
                    Err(err) => {
                        // If we can't parse separator, check if we have enough items
                        if count < start {
                            return Err(err);
                        }
                        break;
                    }
                }
            }

            // Final check for minimum count
            if count < start {
                return Err(E::eof((
                    current_input.get_state(),
                    current_input.get_state(),
                )));
            }

            Ok((current_input, results))
        })
    }

    pub fn chain<F, B>(self, op: Parsec<S, E, F>, init: A) -> Parsec<S, E, A>
    where
        S: Clone,
        F: Fn(A, A) -> A + Clone + 'static,
        A: Clone,
    {
        Parsec::new(move |input: S| {
            let mut acc = init.clone();
            let mut inp = input.clone();
            loop {
                let op_res = op.eval(inp.clone());
                let val_res = self.eval(inp.clone());
                match (op_res, val_res) {
                    (Ok((_, f)), Ok((val_input, val))) => {
                        acc = f(acc, val);
                        inp = val_input;
                    }
                    _ => break,
                }
            }
            Ok((inp, acc))
        })
    }

    pub fn chain_right<F>(self, op: Parsec<S, E, F>, init: A) -> Parsec<S, E, A>
    where
        S: Clone,
        F: Fn(A, A) -> A + Clone + 'static,
        A: Clone,
    {
        fn parse_rec<S: LexIterTrait, E, A, F>(
            parser: &Parsec<S, E, A>,
            op: &Parsec<S, E, F>,
            input: S,
            init: &A,
        ) -> Result<(S, A), E>
        where
            S: Clone + 'static,
            E: ParserError<Context = S::Context> + 'static,
            F: Fn(A, A) -> A + Clone + 'static,
            A: Clone + 'static,
        {
            match parser.eval(input.clone()) {
                Ok((next_input, x)) => match op.eval(next_input.clone()) {
                    Ok((op_input, f)) => {
                        let (rest_input, y) = parse_rec(parser, op, op_input, init)?;
                        Ok((rest_input, f(x, y)))
                    }
                    Err(_) => Ok((next_input, x)),
                },
                Err(_) => Ok((input, init.clone())),
            }
        }

        let parser = self;
        let op = op;
        let init = init;
        Parsec::new(move |input: S| parse_rec(&parser, &op, input, &init))
    }

    pub fn pair<B: 'static>(self, other: Parsec<S, E, B>) -> Parsec<S, E, (A, B)> {
        Parsec::new(move |input: S| {
            let (next_input, value_a) = self.eval(input)?;
            let (final_input, value_b) = other.eval(next_input)?;
            Ok((final_input, (value_a, value_b)))
        })
    }

    pub fn extend(self, other: Parsec<S, E, Vec<A>>) -> Parsec<S, E, Vec<A>> {
        Parsec::new(move |input: S| {
            let (next_input, value) = self.eval(input)?;
            let (final_input, mut other_values) = other.eval(next_input)?;
            other_values.insert(0, value);
            Ok((final_input, other_values))
        })
    }

    pub fn hold<F>(self, f: F) -> Parsec<S, E, A>
    where
        F: Fn(&A) -> bool + 'static,
        A: Display,
        E: Clone + 'static,
    {
        Parsec::new(move |input: S| {
            let original_state = input.get_state();
            let (next_input, item) = self.eval(input)?;
            if f(&item) {
                Ok((next_input, item))
            } else {
                Err(E::unexpected(
                    (original_state, next_input.get_state()),
                    &item,
                ))
            }
        })
    }

    pub fn not(self, value: A) -> Parsec<S, E, A>
    where
        A: PartialEq + Display,
        E: Clone + 'static,
    {
        Parsec::new(move |input: S| {
            let original_state = input.get_state();
            let (next_input, item) = self.eval(input)?;
            if item == value {
                Err(
                    E::unexpected((original_state, next_input.get_state()), &item)
                        .with_expected(&value),
                )
            } else {
                Ok((next_input, item))
            }
        })
    }

    pub fn is(self, value: A) -> Parsec<S, E, A>
    where
        A: PartialEq + Display,
        E: Clone + 'static,
    {
        Parsec::new(move |input: S| {
            let original_state = input.get_state();
            let (next_input, item) = self.eval(input)?;
            if item == value {
                Ok((next_input, item))
            } else {
                Err(
                    E::unexpected((original_state, next_input.get_state()), &item)
                        .with_expected(&value),
                )
            }
        })
    }

    pub fn one_of(self, values: impl Iterator<Item = A> + Clone + 'static) -> Parsec<S, E, A>
    where
        A: PartialEq + Display,
        E: Clone + 'static,
    {
        Parsec::new(move |input: S| {
            let original_state = input.get_state();
            let (next_input, item) = self.eval(input)?;
            let mut values = values.clone();
            if values.any(|v| v == item) {
                Ok((next_input, item))
            } else {
                Err(
                    E::unexpected((original_state, next_input.get_state()), &item).with_expected(
                        &values.map(|v| v.to_string()).collect::<Vec<_>>().join(", "),
                    ),
                )
            }
        })
    }

    pub fn none_of(self, values: impl Iterator<Item = A> + Clone + 'static) -> Parsec<S, E, A>
    where
        A: PartialEq + Display,
        E: 'static,
    {
        Parsec::new(move |input: S| {
            let original_state = input.get_state();
            let (next_input, item) = self.eval(input)?;
            if values.clone().any(|v| v == item) {
                Err(E::unexpected(
                    (original_state, next_input.get_state()),
                    &item,
                ))
            } else {
                Ok((next_input, item))
            }
        })
    }

    pub fn into<B: 'static>(self) -> Parsec<S, E, B>
    where
        A: Into<B>,
    {
        self.map(|a| a.into())
    }

    pub fn states(self) -> Parsec<S, E, ((LexIterState<S::Context>, LexIterState<S::Context>), A)>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let original_state = input.get_state();
            let (next_input, value) = self.eval(input)?;
            let next_state = next_input.get_state();
            Ok((next_input, ((original_state, next_state), value)))
        })
    }
}

impl<S, E> Parsec<S, E, String>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    pub fn leak(self) -> Parsec<S, E, &'static str> {
        Parsec::new(move |input: S| {
            let (next_input, value) = self.eval(input)?;
            let leaked: &'static str = Box::leak(value.into_boxed_str());
            Ok((next_input, leaked))
        })
    }

    pub fn trim(self) -> Parsec<S, E, String> {
        Parsec::new(move |input: S| {
            let (next_input, value) = self.eval(input)?;
            let trimmed = value.trim().to_string();
            Ok((next_input, trimmed))
        })
    }
}

impl<S, E, A> Parsec<S, E, Parsec<S, E, A>>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
{
    pub fn join(self) -> Parsec<S, E, A> {
        Parsec::new(move |input: S| {
            let (next_input, parser) = self.eval(input)?;
            parser.eval(next_input)
        })
    }
}

impl<S, E, A> Parsec<S, E, Vec<Parsec<S, E, A>>>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
{
    pub fn sequence(self) -> Parsec<S, E, Vec<A>> {
        Parsec::new(move |input: S| {
            let (mut current_input, parsers) = self.eval(input)?;
            let mut results = Vec::with_capacity(parsers.len());

            for parser in parsers {
                let (next_input, value) = parser.eval(current_input)?;
                results.push(value);
                current_input = next_input;
            }

            Ok((current_input, results))
        })
    }
}

impl<const N: usize, S, E, A> Parsec<S, E, [Parsec<S, E, A>; N]>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: Debug + 'static,
{
    pub fn sequence(self) -> Parsec<S, E, [A; N]> {
        Parsec::new(move |input: S| {
            let (mut current_input, parsers) = self.eval(input)?;
            let mut results = Vec::with_capacity(N);

            for parser in parsers.into_iter() {
                let (next_input, value) = parser.eval(current_input)?;
                results.push(value);
                current_input = next_input;
            }

            // Convert Vec to array
            let array = results.try_into().unwrap();

            Ok((current_input, array))
        })
    }
}

impl<S, E, A> Parsec<S, E, Vec<A>>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
{
    pub fn collect<B: 'static>(self) -> Parsec<S, E, B>
    where
        A: Into<B>,
        B: FromIterator<A>,
    {
        self.map(|vec| vec.into_iter().collect())
    }

    pub fn append(self, other: Parsec<S, E, A>) -> Parsec<S, E, Vec<A>> {
        Parsec::new(move |input: S| {
            let (next_input, mut values) = self.eval(input)?;
            let (final_input, value) = other.eval(next_input)?;
            values.push(value);
            Ok((final_input, values))
        })
    }

    pub fn concat(self, other: Parsec<S, E, Vec<A>>) -> Parsec<S, E, Vec<A>>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let (next_input, mut values) = self.eval(input)?;
            let (final_input, other_values) = other.eval(next_input)?;
            values.extend(other_values);
            Ok((final_input, values))
        })
    }
}

impl<S, E> Parsec<S, E, Vec<String>>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    pub fn trim(self) -> Parsec<S, E, Vec<String>> {
        Parsec::new(move |input: S| {
            let (next_input, mut values) = self.eval(input)?;
            values.retain_mut(|s| !s.is_empty());
            Ok((next_input, values))
        })
    }
}

// === Utility Functions ===

pub fn pure<S, E, T>(value: T) -> Parsec<S, E, T>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    T: Clone + 'static,
{
    Parsec::new(move |input| Ok((input, value.clone())))
}

pub fn fail<S, E, A>(error: E) -> Parsec<S, E, A>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static + Clone,
    A: 'static,
{
    Parsec::new(move |_| Err(error.clone()))
}

pub fn any<S, E>() -> Parsec<S, E, S::Item>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    S::Item: 'static,
{
    Parsec::new(move |mut input: S| {
        let original_state = input.get_state();
        if let Some(item) = input.next() {
            Ok((input, item))
        } else {
            Err(E::eof((original_state, input.get_state())))
        }
    })
}

pub fn satisfy<S, E, F>(f: F) -> Parsec<S, E, S::Item>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    S::Item: Display + 'static,
    F: Fn(&S::Item) -> bool + 'static,
{
    Parsec::new(move |mut input: S| {
        let original_state = input.get_state();
        if let Some(item) = input.next() {
            if f(&item) {
                Ok((input, item))
            } else {
                Err(E::unexpected((original_state, input.get_state()), item))
            }
        } else {
            Err(E::eof((original_state, input.get_state())))
        }
    })
}

pub fn item<S, E>(expected: S::Item) -> Parsec<S, E, S::Item>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    S::Item: PartialEq + Display + Clone + 'static,
{
    let expected_ = expected.clone();
    satisfy::<S, E, _>(move |item| *item == expected_).expected(expected)
}

pub fn decimal_digit<S, E>() -> Parsec<S, E, char>
where
    S: LexIterTrait<Item = char> + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    satisfy::<S, E, _>(|c: &char| c.is_digit(10)).expected("digit")
}

pub fn hex_digit<S, E>() -> Parsec<S, E, char>
where
    S: LexIterTrait<Item = char> + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    satisfy::<S, E, _>(|c: &char| c.is_digit(16)).expected("hex digit")
}

pub fn octal_digit<S, E>() -> Parsec<S, E, char>
where
    S: LexIterTrait<Item = char> + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    satisfy::<S, E, _>(|c: &char| c.is_digit(8)).expected("octal digit")
}

pub fn digit<S, E>(radix: u32) -> Parsec<S, E, char>
where
    S: LexIterTrait<Item = char> + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    satisfy::<S, E, _>(move |c: &char| c.is_digit(radix)).expected("digit")
}

pub fn alpha<S, E>() -> Parsec<S, E, char>
where
    S: LexIterTrait<Item = char> + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    satisfy::<S, E, _>(|c: &char| c.is_alphabetic()).expected("alphabetic character")
}

pub fn alphanumeric<S, E>() -> Parsec<S, E, char>
where
    S: LexIterTrait<Item = char> + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    satisfy::<S, E, _>(|c: &char| c.is_alphanumeric()).expected("alphanumeric character")
}

pub fn whitespace<S, E>() -> Parsec<S, E, char>
where
    S: LexIterTrait<Item = char> + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    satisfy::<S, E, _>(|c: &char| c.is_whitespace()).expected("whitespace character")
}

pub fn newline<S, E>() -> Parsec<S, E, char>
where
    S: LexIterTrait<Item = char> + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    satisfy::<S, E, _>(|c: &char| *c == '\n').expected("newline character")
}

pub fn eof<S, E>() -> Parsec<S, E, ()>
where
    S: LexIterTrait<Item = char> + Clone + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    Parsec::new(move |mut input: S| {
        let original_state = input.get_state();
        if input.next().is_none() {
            Ok((input, ()))
        } else {
            Err(E::eof((original_state, input.get_state())))
        }
    })
}

pub fn char<S, E>(expected: char) -> Parsec<S, E, char>
where
    S: LexIterTrait<Item = char> + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    satisfy::<S, E, _>(move |c: &char| *c == expected).expected(expected)
}

pub fn state<S, E>() -> Parsec<S, E, LexIterState<S::Context>>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
{
    Parsec::new(move |input: S| {
        let state = input.get_state();
        Ok((input, state))
    })
}

impl<S, E, A> Parsec<S, E, Result<A, E>>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
{
    pub fn unwrap(self) -> Parsec<S, E, A>
    where
        S: Clone,
    {
        Parsec::new(move |input: S| {
            let (next_input, result) = self.eval(input)?;
            match result {
                Ok(value) => Ok((next_input, value)),
                Err(error) => Err(error),
            }
        })
    }

    pub fn unwrap_or(self, default: A) -> Parsec<S, E, A>
    where
        S: Clone,
        A: Clone,
    {
        Parsec::new(move |input: S| {
            let (next_input, result) = self.eval(input)?;
            match result {
                Ok(value) => Ok((next_input, value)),
                Err(_) => Ok((next_input, default.clone())),
            }
        })
    }

    pub fn unwrap_or_default(self) -> Parsec<S, E, A>
    where
        S: Clone,
        A: Default,
    {
        Parsec::new(move |input: S| {
            let (next_input, result) = self.eval(input)?;
            match result {
                Ok(value) => Ok((next_input, value)),
                Err(_) => Ok((next_input, A::default())),
            }
        })
    }
}

impl<S, E, A, B> Parsec<S, E, Result<A, B>>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
    B: 'static,
{
    pub fn internalize<F, D: Display>(self, f: F) -> Parsec<S, E, A>
    where
        F: Fn(B) -> D + 'static,
    {
        Parsec::new(move |input: S| {
            let original_state = input.get_state();
            let (next_input, result) = self.eval(input)?;
            match result {
                Ok(value) => Ok((next_input, value)),
                Err(error) => {
                    let msg = f(error);
                    Err(E::unexpected(
                        (original_state, next_input.get_state()),
                        &msg,
                    ))
                }
            }
        })
    }
}

impl<S, E, A> BitOr for Parsec<S, E, A>
where
    S: LexIterTrait + Clone + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
{
    type Output = Parsec<S, E, A>;

    fn bitor(self, other: Self) -> Self::Output {
        self.or(other)
    }
}

impl<S, E, A, B> Add<Parsec<S, E, B>> for Parsec<S, E, A>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
    B: 'static,
{
    type Output = Parsec<S, E, (A, B)>;

    fn add(self, other: Parsec<S, E, B>) -> Self::Output {
        self.pair(other)
    }
}

impl<S, E, A, B> Shr<Parsec<S, E, B>> for Parsec<S, E, A>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
    B: 'static,
{
    type Output = Parsec<S, E, B>;

    fn shr(self, other: Parsec<S, E, B>) -> Self::Output {
        self.then(other)
    }
}

impl<S, E, A, B> Shl<Parsec<S, E, B>> for Parsec<S, E, A>
where
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
    B: 'static,
{
    type Output = Parsec<S, E, A>;

    fn shl(self, other: Parsec<S, E, B>) -> Self::Output {
        self.with(other)
    }
}

impl<S, F, E, A, B> BitAnd<F> for Parsec<S, E, A>
where
    S: LexIterTrait + 'static,
    F: Fn(A) -> B + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
    B: 'static,
{
    type Output = Parsec<S, E, B>;

    fn bitand(self, other: F) -> Self::Output {
        self.map(other)
    }
}

pub fn rec<F, S, E, A>(f: F) -> Parsec<S, E, A>
where
    F: Fn() -> Parsec<S, E, A> + 'static,
    S: LexIterTrait + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
{
    Parsec::new(move |input: S| {
        let parser = f();
        parser.eval(input)
    })
}

pub fn branch<S, E, A>(parsers: Vec<Parsec<S, E, A>>) -> Parsec<S, E, A>
where
    S: LexIterTrait + Clone + 'static,
    E: ParserError<Context = S::Context> + 'static,
    A: 'static,
{
    Parsec::new(move |input: S| {
        if parsers.is_empty() {
            return Err(E::eof((input.get_state(), input.get_state())));
        }
        for parser in &parsers[..parsers.len() - 1] {
            if let Ok((next_input, value)) = parser.eval(input.clone()) {
                return Ok((next_input, value));
            }
        }
        let last_parser = &parsers[parsers.len() - 1];
        last_parser.eval(input)
    })
}

// Add this new trait to provide a clear path to the M associated type
pub trait ParserF {
    type MapOutput<U>;
    type MapError<E: ParserError>;
}

// Implement this trait for ParserBuilder
impl<S, E, U> ParserF for Parsec<S, E, U>
where
    S: LexIterTrait + 'static,
    E: ParserError + 'static,
    U: 'static,
{
    type MapOutput<T> = Parsec<S, E, T>;
    type MapError<T: ParserError> = Parsec<S, T, U>;
}
pub type With<P, T> = <P as ParserF>::MapOutput<T>;
pub type WithError<P, T> = <P as ParserF>::MapError<T>;
