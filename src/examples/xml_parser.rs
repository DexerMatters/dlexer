#![allow(dead_code)]

use crate::{
    do_parse,
    lex::{symbol, token},
    map,
    parsec::{BasicParser, With, any, char, decimal_digit, pure},
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
    );

    let string: With<P, _> = token(
        any()
            .not('"')
            .many()
            .between(char('"'), char('"'))
            .collect::<String>()
            .map(XmlValue::String),
    )
    .expected("string");

    let integer = || decimal_digit().many1();
    let fractional = char('.').extend(integer()) | pure(Vec::new());
    let number: With<P, _> = token(
        integer()
            .concat(fractional)
            .collect::<String>()
            .map(|s| XmlValue::Number(s.parse().unwrap())),
    );

    boolean | string | number
}

pub fn attribute() -> With<P, XmlAttribute> {
    (ident() & symbol("=").then(value())).map(|(name, value)| XmlAttribute { name, value })
}

pub fn node() -> With<P, XmlNode> {
    let element = do_parse!(
        let% label = char('<').then(ident());
        let label_: &'static str = Box::leak(label.into_boxed_str());
        let% attrs = attribute().many().with(char('>'));
        let% children = node().many_till(symbol("</"));
        symbol(label_).between(symbol("</"), char('>'));
        pure(XmlNode::Element(XmlElement { name: label_.to_string(), attributes: attrs.clone(), children }))
    );
    let comment = any()
        .many()
        .between(symbol("<!--"), symbol("-->"))
        .collect::<String>()
        .map(|text| XmlNode::Comment(text));
    let text = token(any().many1_till(symbol("\n") | symbol("</")))
        .collect::<String>()
        .map(XmlNode::Text);
    element | comment | text
}

#[cfg(test)]
mod test {
    use crate::{examples::xml_parser::node, lex::CharSkipper};

    #[test]
    fn test_xml_parser() {
        let xml = r#"
        <note>
            <to color="red">Tove</to>
            <from>Jani</from>
            <heading>Reminder</heading>
            <body>Don't forget me this weekend!</body>
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
