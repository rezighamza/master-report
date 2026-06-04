#import "@preview/cetz:0.3.2"

#set page(width: auto, height: auto, margin: 20pt)
#set text(font: "Libertinus Serif", size: 10pt)

// Color Palette matching the image
#let c-blue      = rgb("#5b738b")
#let c-blue-fill = rgb("#eef2f5")
#let c-green     = rgb("#94b494")
#let c-teal      = rgb("#a3b8b6")
#let c-gray-axis = rgb("#d1d5db")
#let c-gray-text = rgb("#4b5563")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // ==========================================
  // PANEL A: Continuous Signal
  // ==========================================
  group(name: "panel-a", {
    // Titles
    content((0.0, 4.5), text(style: "italic", fill: c-gray-text, "(a) Continuous Signal"))

    // Axes
    line((-3.0, 0.0), (3.0, 0.0), stroke: c-gray-axis)
    line((-3.0, -0.5), (-3.0, 3.5), stroke: c-gray-axis)

    // Axis ticks
    for x in (-2.0, -1.0, 0.0, 1.0, 2.0) {
      line((x, -0.05), (x, 0.05), stroke: c-gray-axis)
    }

    // Axis Labels
    content((0.0, -1.0), text(size: 9pt, fill: c-gray-text, "x"))
    content((-3.6, 1.5), angle: 90deg, text(size: 9pt, fill: c-gray-text, $f(x)$))

    // Math Curve Data: f(x) = a * exp(-(x-mu)^2 / (2*sigma^2))
    let bell-curve(x) = 3.0 * calc.exp(-calc.pow(x, 2.0) / 1.5)
    let domain = range(-30, 31).map(x => x / 10.0)

    // Create path for filled area
    let fill-pts = domain.map(x => (x, bell-curve(x)))
    fill-pts.push((3.0, 0.0))
    fill-pts.push((-3.0, 0.0))

    // Draw filled area and curve
    line(..fill-pts, fill: c-blue-fill, stroke: none, close: true)
    line(..domain.map(x => (x, bell-curve(x))), stroke: (paint: c-blue, thickness: 1.5pt))

    // Draw discrete dots on the curve
    for x in range(-30, 31, step: 4) {
      let vx = x / 10.0
      circle((vx, bell-curve(vx)), radius: 0.06, fill: c-blue, stroke: none)
    }

    // Annotations
    // mu = 0
    content((-1.5, 3.7), text(fill: c-blue, $mu = 0$))
    line((-1.1, 3.5), (-0.1, 3.05), mark: (end: ">", fill: c-blue), stroke: (paint: c-blue, dash: "dashed"))

    // precision
    line((2.35, 0.1), (2.35, 2.5), stroke: (paint: c-blue, dash: "dotted"))
    line((2.35, 0.5), (2.35, 1.5), mark: (start: "<", end: ">", fill: c-blue), stroke: (paint: c-blue, dash: "dashed"))
    content((2.75, 2.05), anchor: "west", text(size: 8pt, fill: c-blue, "precision"))
  })

  // ==========================================
  // ARROW A -> B
  // ==========================================
  group({
    line((4.15, 0.55), (6.25, 0.55), mark: (end: ">", fill: c-blue), stroke: c-blue)
    content((5.2, 1.05), text(size: 8pt, style: "italic", fill: c-blue, [apply $Q(x)$]))
  })

  // ==========================================
  // PANEL B: Transfer Function Q(x)
  // ==========================================
  group(name: "panel-b", {
    // Correct translation vector coordinates
    translate((9.8, 0.0))

    // Title
    content((0.0, 4.5), text(style: "italic", fill: c-gray-text, "(b) Transfer Function Q(x)"))

    // Axes
    line((-3.0, 1.5), (3.0, 1.5), stroke: c-gray-axis) // X-axis shifted up visually
    line((-3.0, -0.5), (-3.0, 3.5), stroke: c-gray-axis)

    // Axis ticks
    for x in (-2.0, -1.0, 0.0, 1.0, 2.0) {
      line((x, 1.45), (x, 1.55), stroke: c-gray-axis)
    }

    // Axis Labels
    content((0.0, -1.0), text(size: 9pt, fill: c-gray-text, "Input"))
    content((-3.6, 1.5), angle: 90deg, text(size: 9pt, fill: c-gray-text, "Output"))

    // Diagonal reference line is drawn after the staircase so it stays visible.

    // Staircase Function & Y-axis dots
    let step-w = 0.4
    let step-h = 0.3
    let num-steps = 11
    let start-x = -2.4
    let start-y = 0.5

    let current-x = start-x
    let current-y = start-y

    let stair-pts = ((current-x, current-y),)

    for i in range(num-steps) {
      // Draw green dots on Y axis
      circle((-3.1, current-y), radius: 0.04, fill: c-green, stroke: none)

      let next-x = current-x + step-w
      stair-pts.push((next-x, current-y))
      current-y += step-h
      if i < num-steps - 1 { stair-pts.push((next-x, current-y)) }
      current-x = next-x
    }

    // Draw the stairs
    line(..stair-pts, stroke: (paint: c-green, thickness: 1.5pt))

    // Diagonal reference line
    line((-2.1, 1.0), (2.25, 3.9), stroke: (paint: c-gray-axis, dash: "dashed"))

    // Annotation
    content((3.25, 3.95), anchor: "east", text(size: 8pt, style: "italic", fill: c-green, [quantization levels]))
    line((2.35, 3.72), (1.85, 3.63), mark: (end: ">", fill: c-green), stroke: (paint: c-green, dash: "dashed"))
  })

  // ==========================================
  // ARROW B -> C
  // ==========================================
  group({
    line((13.15, 0.55), (16.55, 0.55), mark: (end: ">", fill: c-blue), stroke: c-blue)
    content((14.85, 1.05), text(size: 9pt, style: "italic", fill: c-blue, "discrete output"))
  })


  // ==========================================
  // PANEL C: Quantized Distribution
  // ==========================================
  group(name: "panel-c", {
    translate((20.2, -0.35))

    // Title
    content((0.0, 4.5), text(style: "italic", fill: c-gray-text, "(c) Quantized Distribution"))

    // Axes
    line((-3.0, 1.5), (3.0, 1.5), stroke: c-gray-axis)
    line((-3.0, 1.5), (-3.0, 3.5), stroke: c-gray-axis)

    // Axis Labels
    content((0.0, -1.0), text(size: 9pt, fill: c-gray-text, "x"))
    content((-3.6, 1.5), angle: 90deg, text(size: 9pt, fill: c-gray-text, $Q(f(x))$))

    // Background horizontal dashed lines (quantization levels)
    for i in range(7) {
      let y = 1.5 + (i * 0.3)
      line((-3.0, y), (3.0, y), stroke: (paint: rgb("#f3f4f6"), dash: "dotted"))
    }

    // Bar chart (Histogram)
    let bars = (0, 0, 0, 1, 1, 2, 4, 5, 6, 6, 5, 4, 2, 1, 1, 0, 0, 0)
    let bar-width = 0.3
    let start-x = -2.6

    for (i, steps) in bars.enumerate() {
      if steps > 0 {
        let x1 = start-x + (i * 0.32)
        let y1 = 1.5
        let x2 = x1 + bar-width
        let y2 = 1.5 + (steps * 0.3)
        rect((x1, y1), (x2, y2), fill: c-teal, stroke: c-blue)
      }
    }

    // Annotations
    // Information loss
    content((0.2, 0.75), text(size: 9pt, style: "italic", fill: c-teal, [information loss (#$epsilon$)]))
    line((0.8, 0.95), (0.25, 1.5), mark: (end: ">", fill: c-teal), stroke: (paint: c-teal, dash: "dashed"))

    // Delta Step
    let step-x = 2.35
    line((step-x, 1.5+0.3), (step-x + 0.6, 1.5+0.3), stroke: (paint: c-blue, dash: "dotted"))
    line((step-x, 1.5+0.6), (step-x + 0.6, 1.5+0.6), stroke: (paint: c-blue, dash: "dotted"))
    line((step-x + 0.45, 1.5+0.3), (step-x + 0.45, 1.5+0.6), mark: (start: "<", end: ">", fill: c-blue), stroke: c-blue)
    content((3.15, 1.95), anchor: "west", text(size: 8pt, fill: c-blue, $Delta " step"$))
  })

  // ==========================================
  // LEGEND
  // ==========================================
  group({
    translate((21.1, -2.35))

    // Legend Box
    rect((0.0, 0.0), (3.4, 1.8), fill: white, stroke: c-gray-axis)

    // Legend Title
    content((0.2, 1.4), anchor: "west", text(weight: "bold", fill: c-gray-text, "Legend"))

    // Item 1
    rect((0.2, 0.95), (0.55, 1.15), fill: c-blue, stroke: none)
    content((0.7, 1.05), anchor: "west", text(size: 9pt, fill: c-gray-text, "f(x): Continuous"))

    // Item 2
    rect((0.2, 0.55), (0.55, 0.75), fill: c-green, stroke: none)
    content((0.7, 0.65), anchor: "west", text(size: 9pt, fill: c-gray-text, "Q(x): Transfer Fn."))

    // Item 3
    rect((0.2, 0.15), (0.55, 0.35), fill: c-teal, stroke: none)
    content((0.7, 0.25), anchor: "west", text(size: 9pt, fill: c-gray-text, "Output: Quantized"))
  })

})
