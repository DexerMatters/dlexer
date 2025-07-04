#![allow(dead_code)]

use crate::{
    do_parse,
    lex::{number, symbol, token},
    map,
    parsec::{BasicParser, With, any, char, pure},
    prelude::build_ident,
};

#[derive(Debug, Clone)]
pub enum XmlNode {
    Element(XmlElement),
    Text(String),
    Comment(String),
}

#[derive(Debug, Clone)]
pub enum XmlValue {
    String(String),
    Number(f64),
    Boolean(bool),
}

#[derive(Debug, Clone)]
pub struct XmlElement {
    pub name: String,
    pub attributes: Vec<XmlAttribute>,
    pub children: Vec<XmlNode>,
}

#[derive(Debug, Clone)]
pub struct XmlAttribute {
    pub name: String,
    pub value: XmlValue,
}

type P = BasicParser;

pub fn ident() -> With<P, String> {
    token(build_ident(["xml", "xmlns", "xsi"]))
}

pub fn value() -> With<P, XmlValue> {
    let boolean: With<P, _> = map!(
        symbol("true") => XmlValue::Boolean(true),
        symbol("false") => XmlValue::Boolean(false)
    )
    .expected("boolean");

    let esc = char('\\')
        >> map!(
            char('"') => '"',
            char('n') => '\n',
            char('t') => '\t',
            char('\\') => '\\',
            char('\"') => '\"'
        );

    let string: With<P, _> = token(
        (esc | any().not('"'))
            .many()
            .between(char('"'), char('"'))
            .collect::<String>()
            & XmlValue::String,
    )
    .expected("string");

    let number: With<P, _> = number().expected("number") & XmlValue::Number;

    boolean | string | number
}

pub fn attribute() -> With<P, XmlAttribute> {
    ident() + (symbol("=") >> value()) & |(name, value)| XmlAttribute { name, value }
}

pub fn node() -> With<P, XmlNode> {
    let element = do_parse!(
        let% label = char('<') >> ident().leak();
        let% attrs = attribute().many() << char('>');
        let% children = node().many_till(symbol("</"));
        symbol(label).between(symbol("</"), char('>'));
        pure(XmlNode::Element(
            XmlElement {
                name: label.to_string(),
                attributes: attrs.clone(),
                children
            }
        ))
    );
    let comment = token(any().many_till(symbol("-->")))
        .between(symbol("<!--"), symbol("-->"))
        .collect::<String>()
        .trim()
        & XmlNode::Comment;

    let esc = char('&')
        >> map!(
            symbol("lt") => '<',
            symbol("gt") => '>',
            symbol("amp") => '&',
            symbol("quot") => '"',
            symbol("apos") => '\''
        )
        << char(';');
    let text = token((esc | any().none_of("\n<>".chars())).many1())
        .collect::<String>()
        .trim()
        & XmlNode::Text;
    element | comment | text
}

#[cfg(test)]
mod test {
    use crate::{examples::xml_parser::node, lex::CharSkipper};

    #[test]
    fn test_xml_parser() {
        let xml = r#"
        <note>
        <!-- This is a comment -->
            <to color="red" weight=80>
                Tove
            </to>
            <from>
                Jani
            </from>
            <heading>
                Reminder
            </heading>
            <body>
                Don't forget me this weekend! &lt;Escaped&gt;
            </body>
        </note>
    "#;

        let result = node().parse(xml, CharSkipper(['\n', '\t', ' ']));
        match result {
            Ok(node) => {
                println!("Parsed XML Node:\n {:#?}", node);
            }
            Err(e) => {
                println!("Error parsing XML: {}", e);
            }
        }
    }
}
