#![allow(dead_code)]

use crate::parsec::{BasicParser, With, char, eof, hex_digit};

#[derive(Debug, Clone, Copy)]
pub struct Color {
    pub a: u8,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

pub fn hex_color() -> With<BasicParser, Color> {
    let u8_hex = hex_digit()
        .pair(hex_digit())
        .map(|(h1, h2)| u8::from_str_radix(&format!("{}{}", h1, h2), 16))
        .lift_err(|err| format!("Failed to parse u8 from hex digits: {:?}", err));

    u8_hex
        .many_till(eof())
        .between(char('#'), eof())
        .map(|v| match v.as_slice() {
            [a, r, g, b] => Ok(Color {
                a: *a,
                r: *r,
                g: *g,
                b: *b,
            }),
            [r, g, b] => Ok(Color {
                a: 255,
                r: *r,
                g: *g,
                b: *b,
            }),
            _ => Err("invalid hex color format"),
        })
        .lift_err(|e| e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_color() {
        let inputs = [
            "#FF5733",
            "#4CAF50",
            "#0000FF",
            "#FFFFFF",
            "#000000",
            // Alpha channel
            "#FF5733FF",
            "#4CAF50AA",
            "#0000FF80",
            // Invalid formats
            "#GGGGGG",     // Invalid hex digits
            "#123",        // Too short
            "#1234567",    // Too long
            "#123456788",  // Too long with alpha
            "#AAAAAAAAAA", // Invalid alpha channel
        ];

        for input in inputs {
            println!("==== Testing input: '{}' ====", input);
            let result = hex_color().test(input);
            match result {
                Ok(color) => {
                    println!("Parsed color: {:?}", color);
                }
                Err(err) => {
                    println!("Failed to parse color from '{}': {}", input, err);
                }
            }
        }
    }
}
