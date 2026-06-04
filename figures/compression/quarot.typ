#import "@preview/cetz:0.3.2"

#set page(width: 30cm, height: 13cm, margin: 1cm, fill: white)
#set text(font: "Linux Libertine", size: 11pt)

#let text_fp16 = rgb("#C00000")
#let text_int4 = rgb("#0040A0")

#let fill_blue = rgb("#EDF3FD")
#let stroke_blue = rgb("#4A86E8")
#let fill_green = rgb("#EAF2EA")
#let stroke_green = rgb("#6AA84F")
#let fill_gray = rgb("#F3F3F3")
#let stroke_gray = rgb("#999999")

#let outer_green = rgb("#38761D")
#let outer_blue = rgb("#1155CC")
#let outer_legend = rgb("#A9C8E2")

#cetz.canvas({
  import cetz.draw: *

  // Common arrow style
  let arr = (end: ">", fill: black, length: 0.2, width: 0.15)

  // Node helper function
  let draw_node(name, pos, w, h, bg, border, body) = {
    rect(
      (pos.at(0) - w/2, pos.at(1) - h/2),
      (pos.at(0) + w/2, pos.at(1) + h/2),
      name: name,
      fill: bg,
      stroke: border + 1pt,
      radius: 0.2
    )
    content(pos, body)
  }

  // ==========================================
  // OUTER CONTAINERS
  // ==========================================

  // 1. Input Quantization (Green Box)
  rect((0.5, 3.0), (5.0, 9.0), stroke: outer_green + 1.2pt, radius: 0.4, fill: none)

  // 2. Core Computation (Blue Box)
  rect((6.6, 3.0), (22.4, 9.0), stroke: outer_blue + 1.2pt, radius: 0.4, fill: none)

  // 3. Output Projection (Green Box)
  rect((23.2, 3.0), (28.2, 9.0), stroke: outer_green + 1.2pt, radius: 0.4, fill: none)

  // ==========================================
  // HEADERS
  // ==========================================

  let header(x, num, title) = {
    circle((x, 9.8), radius: 0.35, fill: black, stroke: none)
    content((x, 9.8), text(white, weight: "bold", size: 12pt)[#num])
    content((x + 0.5, 9.8), text(size: 13pt, weight: "regular")[#title], anchor: "west")
  }

  header(1.5, 1, "Input Quantization")
  header(11.0, 2, "Core Computation (INT4)")
  header(23.5, 3, "Output Projection")

  // ==========================================
  // INNER NODES
  // ==========================================

  // Stage 1 (Moved significantly to the right)
  draw_node("norm", (1.4, 6.25), 1.6, 1.4, fill_gray, stroke_gray, text(size: 14pt)[$frac(x, |x|)$])
  draw_node("quant1", (3.8, 6.25), 1.8, 1.0, fill_green, stroke_green, [quantize])


  // Stage 2
  draw_node("w_gate", (9.5, 7.75), 3.2, 1.2, fill_blue, stroke_blue, text(size: 12pt)[$Q^T (alpha) W_"gate"$])
  draw_node("sigma", (12.3, 7.75), 1.4, 1.2, fill_gray, stroke_gray, text(size: 12pt)[$sigma$])
  draw_node("w_up", (9.5, 4.75), 3.2, 1.2, fill_blue, stroke_blue, text(size: 12pt)[$Q^T (alpha) W_"up"$])

  circle((13.8, 6.25), radius: 0.35, stroke: 1pt, fill: fill_gray, name: "mul")
  content("mul", text(size: 14pt)[$times$])

  // Widely spaced final three nodes
  draw_node("hadamard", (15.6, 6.25), 2.2, 1.0, fill_green, stroke_green, [hadamard])
  draw_node("quant2", (18.2, 6.25), 1.8, 1.0, fill_green, stroke_green, [quantize])
  draw_node("w_down", (21.0, 6.25), 2.4, 1.2, fill_blue, stroke_blue, text(size: 12pt)[$H W_"down" Q$])

  // Stage 3
  draw_node("inv_trans", (25.7, 7.5), 3.8, 1.5, fill_blue, stroke_blue, align(center)[Inverse Transform \ $U^T (...) V$])
  draw_node("final", (25.7, 4.5), 3.4, 1.5, fill_green, stroke_green, align(center)[Final Model \ $hat(W)$ (2-bit)])


  // ==========================================
  // ARROWS & LINES
  // ==========================================

  line("norm.east", "quant1.west", stroke: 1pt, mark: arr)
  line("quant1.east", (7.5, 6.25), stroke: 1pt, mark: arr) // Arrow into split

  // Split to branches
  line((7.5, 6.25), (7.5, 7.75), "w_gate.west", stroke: 1pt, mark: arr)
  line((7.5, 6.25), (7.5, 4.75), "w_up.west", stroke: 1pt, mark: arr)

  line("w_gate.east", "sigma.west", stroke: 1pt, mark: arr)

  // Routes into multiplier
  line("sigma.east", (13.8, 7.75), "mul.north", stroke: 1pt, mark: arr)
  line("w_up.east", (13.8, 4.75), "mul.south", stroke: 1pt, mark: arr)

  line("mul.east", "hadamard.west", stroke: 1pt, mark: arr)
  line("hadamard.east", "quant2.west", stroke: 1pt, mark: arr)
  line("quant2.east", "w_down.west", stroke: 1pt, mark: arr)

  // Final stage projection
  line("inv_trans.south", "final.north", stroke: 1pt, mark: arr)
  line("w_down.east", (25.7, 6.25), stroke: 1pt, mark: arr) // Injects into the column

  // ==========================================
  // TEXT LABELS (FP16 / INT4)
  // ==========================================

  let label(pos, c, t) = content(pos, text(fill: c, weight: "bold")[#t])

  label((1.4, 7.3), text_fp16, "FP16") // Centered over the shifted norm block
  label((5.6, 6.6), text_int4, "INT4")

  label((9.5, 8.6), text_int4, "INT4")
  label((9.5, 5.6), text_int4, "INT4")
  label((12.3, 8.6), text_fp16, "FP16")

  label((14.6, 5.5), text_fp16, "FP16") // Placed right next to the multiplication block

  label((18.2, 7.0), text_int4, "INT4")
  label((21.0, 7.0), text_int4, "INT4")

  label((22.8, 6.6), text_fp16, "FP16") // Entering Box 3


  // ==========================================
  // LEGEND (Bottom) - Extended for overflow
  // ==========================================

  // Box extended significantly to x=28.2 to match Output Projection box alignment
  rect((1.0, 0.5), (28.2, 2.0), stroke: outer_legend + 1pt, radius: 0.2, fill: rgb("#FCFCFC"))

  draw_node("leg1", (1.8, 1.25), 0.8, 0.5, fill_blue, stroke_blue, "")
  content((2.4, 1.25), anchor: "west", text(size: 10pt)[Matrix / Linear Ops \ (INT4)])

  draw_node("leg2", (7.0, 1.25), 0.8, 0.5, fill_green, stroke_green, "")
  content((7.6, 1.25), anchor: "west", text(size: 10pt)[Quantization / Element-wise Ops \ (INT4)])

  draw_node("leg3", (13.6, 1.25), 0.8, 0.5, fill_gray, stroke_gray, "")
  content((14.2, 1.25), anchor: "west", text(size: 10pt)[Activation / Non-linear Ops \ (FP16)])

  content((19.8, 1.25), anchor: "west", text(fill: text_int4, weight: "bold")[INT4])
  content((20.8, 1.25), anchor: "west", text(size: 10pt)[4-bit Integer])

  content((23.8, 1.25), anchor: "west", text(fill: text_fp16, weight: "bold")[FP16])
  // Now has massive amount of space to prevent any overflowing!
  content((24.8, 1.25), anchor: "west", text(size: 10pt)[16-bit Floating Point])
})
