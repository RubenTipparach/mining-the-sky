//! Minimal 5x7 bitmap-font HUD. Glyphs are authored as readable row-art and
//! parsed into bitmasks at startup; text is emitted as clip-space triangle
//! quads (one per lit font pixel) drawn through the overlay shader. No external
//! UI dependency, and it works identically on native and wasm.

use std::collections::HashMap;

use crate::OverlayVertex;

/// One screen pixel maps to this many device pixels per font cell unit.
pub const GLYPH_PX: f32 = 2.0;
/// Cell advance (5 wide + 1 spacing) and line height (7 tall + 2 spacing).
pub const ADVANCE: f32 = 6.0 * GLYPH_PX;
pub const LINE_H: f32 = 9.0 * GLYPH_PX;

pub struct Hud {
    glyphs: HashMap<char, [u8; 7]>,
}

impl Hud {
    pub fn new() -> Hud {
        let mut glyphs = HashMap::new();
        for (c, rows) in FONT {
            let mut g = [0u8; 7];
            for (r, row) in rows.iter().enumerate() {
                let mut bits = 0u8;
                for (col, ch) in row.chars().enumerate() {
                    if ch == '#' {
                        bits |= 1 << col;
                    }
                }
                g[r] = bits;
            }
            glyphs.insert(*c, g);
        }
        Hud { glyphs }
    }

    fn glyph(&self, c: char) -> [u8; 7] {
        *self
            .glyphs
            .get(&c.to_ascii_uppercase())
            .unwrap_or(&[0u8; 7])
    }

    /// Append `s` starting at device-pixel `(x_px, y_px)` from the top-left.
    /// Returns the x cursor after the string (for inline coloured spans).
    pub fn text(
        &self,
        out: &mut Vec<OverlayVertex>,
        s: &str,
        x_px: f32,
        y_px: f32,
        color: [f32; 3],
        res: (f32, f32),
    ) -> f32 {
        let mut cx = x_px;
        for ch in s.chars() {
            let g = self.glyph(ch);
            for (row, bits) in g.iter().enumerate() {
                for col in 0..5u32 {
                    if bits & (1 << col) != 0 {
                        let px = cx + col as f32 * GLYPH_PX;
                        let py = y_px + row as f32 * GLYPH_PX;
                        push_quad(out, px, py, GLYPH_PX, color, res);
                    }
                }
            }
            cx += ADVANCE;
        }
        cx
    }
}

fn push_quad(out: &mut Vec<OverlayVertex>, px: f32, py: f32, s: f32, color: [f32; 3], res: (f32, f32)) {
    let (w, h) = res;
    let x0 = -1.0 + px * 2.0 / w;
    let x1 = -1.0 + (px + s) * 2.0 / w;
    let y0 = 1.0 - py * 2.0 / h;
    let y1 = 1.0 - (py + s) * 2.0 / h;
    let a = [x0, y0];
    let b = [x1, y0];
    let c = [x1, y1];
    let d = [x0, y1];
    for p in [a, b, c, a, c, d] {
        out.push(OverlayVertex { pos: p, color });
    }
}

/// 5x7 glyph set: digits, A-Z, and the punctuation the HUD uses.
const FONT: &[(char, [&str; 7])] = &[
    (' ', ["     ", "     ", "     ", "     ", "     ", "     ", "     "]),
    ('.', ["     ", "     ", "     ", "     ", "     ", " ##  ", " ##  "]),
    (':', ["     ", " ##  ", " ##  ", "     ", " ##  ", " ##  ", "     "]),
    ('/', ["    #", "    #", "   # ", "  #  ", " #   ", "#    ", "#    "]),
    ('+', ["     ", "  #  ", "  #  ", "#####", "  #  ", "  #  ", "     "]),
    ('-', ["     ", "     ", "     ", "#####", "     ", "     ", "     "]),
    ('0', [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "]),
    ('1', ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "]),
    ('2', [" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"]),
    ('3', [" ### ", "#   #", "    #", "  ## ", "    #", "#   #", " ### "]),
    ('4', ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "]),
    ('5', ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "]),
    ('6', [" ### ", "#    ", "#    ", "#### ", "#   #", "#   #", " ### "]),
    ('7', ["#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "]),
    ('8', [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "]),
    ('9', [" ### ", "#   #", "#   #", " ####", "    #", "    #", " ### "]),
    ('A', [" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"]),
    ('B', ["#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "]),
    ('C', [" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "]),
    ('D', ["#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "]),
    ('E', ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"]),
    ('F', ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "]),
    ('G', [" ### ", "#   #", "#    ", "# ###", "#   #", "#   #", " ### "]),
    ('H', ["#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"]),
    ('I', [" ### ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "]),
    ('J', ["  ###", "   # ", "   # ", "   # ", "#  # ", "#  # ", " ##  "]),
    ('K', ["#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"]),
    ('L', ["#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"]),
    ('M', ["#   #", "## ##", "# # #", "#   #", "#   #", "#   #", "#   #"]),
    ('N', ["#   #", "#   #", "##  #", "# # #", "#  ##", "#   #", "#   #"]),
    ('O', [" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "]),
    ('P', ["#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "]),
    ('Q', [" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"]),
    ('R', ["#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"]),
    ('S', [" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "]),
    ('T', ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "]),
    ('U', ["#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "]),
    ('V', ["#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "]),
    ('W', ["#   #", "#   #", "#   #", "#   #", "# # #", "## ##", "#   #"]),
    ('X', ["#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"]),
    ('Y', ["#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "]),
    ('Z', ["#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"]),
];
