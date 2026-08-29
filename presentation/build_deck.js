const pptxgen = require("pptxgenjs");
const path = require("path");
const FIG = (n) => path.join(__dirname, "figures", n);

const DEEP = "065A82", TEAL = "1C7293", MIDNIGHT = "21295C";
const AMBER = "A9541C";                       // findings
const INK = "233038", MUTED = "5A6B75", PAPER = "FFFFFF", TINT = "F1F5F8";
const GREEN = "12813F", RED = "C0392B";       // match the figures exactly

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                  // 13.333 x 7.5 in — before any slide
pres.author = "ChartQA Project";
pres.title = "Grounded Chart Question Answering — Week 1";

const BODY = { fontFace: "Calibri", fontSize: 15, color: INK };
const SMALL = { fontFace: "Calibri", fontSize: 12, color: MUTED };
const KIND = { BUILT: DEEP, MEASURED: TEAL, FOUND: AMBER };

/** Slide header with a BUILT / MEASURED / FOUND chip. */
function head(s, kind, title, sub) {
  if (kind) {
    s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 0.36, w: 1.32, h: 0.34,
      fill: { color: KIND[kind] }, rectRadius: 0.06,
      line: { color: KIND[kind], width: 0 } });
    s.addText(kind, { fontFace: "Calibri", fontSize: 10.5, bold: true, color: PAPER,
      x: 0.6, y: 0.36, w: 1.32, h: 0.34, align: "center", valign: "middle",
      charSpacing: 1.2, margin: 0, isTextBox: true });
  }
  s.addText(title, { fontFace: "Cambria", fontSize: 30, bold: true, color: INK,
    x: 0.6, y: 0.82, w: 12.15, h: 0.62, isTextBox: true });
  if (sub) {
    s.addText(sub, { ...SMALL, fontSize: 13, x: 0.62, y: 1.44, w: 12.15, h: 0.4,
      isTextBox: true });
  }
}

function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill || TINT },
    rectRadius: 0.08, line: { color: fill || TINT, width: 0 } });
}

function stat(s, x, y, w, value, label, colour, size) {
  s.addText(value, { fontFace: "Cambria", fontSize: size || 38, bold: true,
    color: colour || DEEP, x, y, w, h: 0.72, align: "center", margin: 0, isTextBox: true });
  s.addText(label, { ...SMALL, fontSize: 12, x, y: y + 0.72, w, h: 0.66,
    align: "center", valign: "top", margin: 0, isTextBox: true });
}

/** A coloured band with one conclusion in it. */
function band(s, y, text, colour, fill) {
  card(s, 0.6, y, 12.15, 1.0, fill);
  s.addText(text, { ...BODY, fontSize: 14.5, bold: true, color: colour,
    x: 0.92, y, w: 11.5, h: 1.0, valign: "middle", isTextBox: true });
}

const bullets = (items) => items.map((t, i) => ({
  text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 7 } }));

/* ═════════════════════════════════════════════════════════ 1 · title */
{
  const s = pres.addSlide();
  s.background = { color: MIDNIGHT };
  s.addText("Grounded Chart\nQuestion Answering", { fontFace: "Cambria", fontSize: 44,
    bold: true, color: PAPER, x: 0.9, y: 1.7, w: 7.6, h: 2.2, lineSpacing: 50,
    isTextBox: true });
  s.addText("Making a model show where it looked, and how it calculated",
    { fontFace: "Calibri", fontSize: 17, color: "9FC2D6", x: 0.95, y: 4.0, w: 8.2,
      h: 0.5, isTextBox: true });
  s.addShape(pres.ShapeType.roundRect, { x: 0.95, y: 5.0, w: 2.1, h: 0.6,
    fill: { color: TEAL }, rectRadius: 0.1, line: { color: TEAL, width: 0 } });
  s.addText("WEEK 1 OF 4", { fontFace: "Calibri", fontSize: 12.5, bold: true, color: PAPER,
    x: 0.95, y: 5.0, w: 2.1, h: 0.6, align: "center", valign: "middle", charSpacing: 1.4,
    margin: 0, isTextBox: true });
  s.addText("Team: [name]  ·  [name]  ·  [name]", { fontFace: "Calibri", fontSize: 13.5,
    color: "7FA3B8", x: 3.35, y: 5.0, w: 6, h: 0.6, valign: "middle", isTextBox: true });
  s.addImage({ path: FIG("fig2_grounded.png"), x: 9.15, y: 1.5, w: 3.5, h: 2.81,
    transparency: 14 });
}

/* ═════════════════════════════════════════════════════════ 2 · problem */
{
  const s = pres.addSlide();
  head(s, null, "The problem", "A right answer and a lucky guess look identical");
  s.addText([{ text: "A chart model reads a chart and gives an answer. That is all you get.",
      options: { breakLine: true, paraSpaceAfter: 12 } },
    { text: "You cannot tell whether it:", options: { breakLine: true, paraSpaceAfter: 8, bold: true } },
    ...bullets(["read the right two bars",
                "read the wrong bars and got lucky",
                "ignored the chart and guessed from the question"])],
    { ...BODY, x: 0.62, y: 1.95, w: 5.5, h: 3.1, isTextBox: true });
  band(s, 5.35, "So nobody can check its work — not us, not a user, not the model itself.",
       RED, "FBEAE8");
  s.addImage({ path: FIG("fig1_ungrounded.png"), x: 6.75, y: 1.85, w: 6.0, h: 4.71 });
}

/* ═════════════════════════════════════════════════════════ 3 · idea */
{
  const s = pres.addSlide();
  head(s, null, "Our idea", "Ask for three things instead of one, then check them");
  s.addImage({ path: FIG("fig2_grounded.png"), x: 0.6, y: 1.9, w: 6.4, h: 5.13 });
  [["1", "Where it looked", "boxes around the West and South bars"],
   ["2", "What it did", "difference(West, South)"],
   ["3", "The answer", "74"]].forEach(([n, h2, d], i) => {
    const y = 2.05 + i * 1.1;
    s.addShape(pres.ShapeType.ellipse, { x: 7.4, y: y + 0.05, w: 0.44, h: 0.44,
      fill: { color: DEEP }, line: { color: DEEP, width: 0 } });
    s.addText(n, { fontFace: "Calibri", fontSize: 13.5, bold: true, color: PAPER,
      x: 7.4, y: y + 0.05, w: 0.44, h: 0.44, align: "center", valign: "middle",
      margin: 0, isTextBox: true });
    s.addText(h2, { ...BODY, fontSize: 15.5, bold: true, x: 8.02, y, w: 4.75, h: 0.36,
      margin: 0, isTextBox: true });
    s.addText(d, { ...SMALL, fontSize: 12.5, x: 8.02, y: y + 0.38, w: 4.75, h: 0.4,
      margin: 0, isTextBox: true });
  });
  card(s, 7.4, 5.35, 5.35, 1.45, "E8F2EC");
  s.addText("A small ordinary program re-does the arithmetic from the boxes.\nIf it disagrees with the answer, the answer is unreliable — no human needed, no gold answer needed.",
    { ...BODY, fontSize: 13, color: GREEN, x: 7.62, y: 5.35, w: 4.9, h: 1.45,
      valign: "middle", isTextBox: true });
}

/* ═════════════════════════════════════════════════════════ 4 · plan */
{
  const s = pres.addSlide();
  head(s, null, "The four-week plan", "Week 1 is deliberately not about training");
  [["Week 1", "Data, measurement\nand baselines", true],
   ["Week 2", "Train the model", false],
   ["Week 3", "Evaluate it properly", false],
   ["Week 4", "Analyse and write up", false]].forEach(([lab, txt, on], i) => {
    const x = 0.62 + i * 3.13;
    card(s, x, 2.25, 2.85, 2.4, on ? DEEP : TINT);
    s.addText(lab, { fontFace: "Cambria", fontSize: 18, bold: true,
      color: on ? PAPER : INK, x, y: 2.5, w: 2.85, h: 0.45, align: "center",
      margin: 0, isTextBox: true });
    s.addText(txt, { fontFace: "Calibri", fontSize: 13, color: on ? "CFE4EF" : MUTED,
      x: x + 0.2, y: 3.08, w: 2.45, h: 1.2, align: "center", isTextBox: true });
  });
  band(s, 5.25, "Everything in weeks 2–4 depends on data we trust and scoring we trust. Week 1 built both, then measured where the untouched model starts.",
       INK, TINT);
}

/* ═════════════════════════════════════════════════════════ 5 · summary */
{
  const s = pres.addSlide();
  head(s, null, "What week 1 delivered", "Everything below is in the repository and reproducible");
  const cols = [
    ["BUILT", DEEP, ["a data pipeline over two datasets, hash-pinned",
                     "an evaluation harness — every metric twice",
                     "a plan miner over 28,299 questions",
                     "a chart generator, 8 types × 4 levels",
                     "three prompts and a strict output parser"]],
    ["MEASURED", TEAL, ["annotation coverage on 2,500 charts",
                        "our scoring vs official, 11,690 predictions",
                        "target size in visual tokens, 7,158 boxes",
                        "box correctness against rendered pixels",
                        "the untouched model on 200 questions"]],
    ["FOUND", AMBER, ["line charts have no box annotations",
                      "2/3 of targets are smaller than the model's grid",
                      "only 4 in 100 questions teach reasoning",
                      "the model's own arithmetic backs its answer 69% of the time"]],
  ];
  cols.forEach(([name, colour, items], i) => {
    const x = 0.62 + i * 4.13;
    card(s, x, 2.0, 3.85, 4.3, TINT);
    s.addShape(pres.ShapeType.roundRect, { x: x + 0.25, y: 2.25, w: 1.5, h: 0.36,
      fill: { color: colour }, rectRadius: 0.06, line: { color: colour, width: 0 } });
    s.addText(name, { fontFace: "Calibri", fontSize: 11, bold: true, color: PAPER,
      x: x + 0.25, y: 2.25, w: 1.5, h: 0.36, align: "center", valign: "middle",
      charSpacing: 1.2, margin: 0, isTextBox: true });
    s.addText(bullets(items), { ...BODY, fontSize: 12.5, x: x + 0.25, y: 2.78,
      w: 3.35, h: 3.35, isTextBox: true });
  });
  s.addText("Cost so far: $0 — free GPUs only.", { ...SMALL, fontSize: 13,
    x: 0.62, y: 6.5, w: 12.15, h: 0.4, isTextBox: true });
}

/* ═════════════════════════════════════════════════════════ 6 · BUILT data */
{
  const s = pres.addSlide();
  head(s, "BUILT", "A data pipeline over two public datasets",
       "Every archive checked against a fingerprint recorded before download");
  s.addTable([
    [{ text: "Dataset", options: { bold: true, color: PAPER, fill: { color: DEEP } } },
     { text: "Training", options: { bold: true, color: PAPER, fill: { color: DEEP }, align: "right" } },
     { text: "Validation", options: { bold: true, color: PAPER, fill: { color: DEEP }, align: "right" } },
     { text: "Test", options: { bold: true, color: PAPER, fill: { color: DEEP }, align: "right" } },
     { text: "Has boxes", options: { bold: true, color: PAPER, fill: { color: DEEP } } }],
    ["ChartQA", { text: "28,299", options: { align: "right" } }, { text: "1,920", options: { align: "right" } }, { text: "2,500", options: { align: "right" } }, "no"],
    ["RefChartQA", { text: "55,789", options: { align: "right" } }, { text: "6,223", options: { align: "right" } }, { text: "11,690", options: { align: "right" } }, "yes"],
  ], { x: 0.62, y: 2.0, w: 8.3, rowH: 0.4, fontFace: "Calibri", fontSize: 13.5,
       color: INK, border: { type: "solid", color: "D9E2E8", pt: 1 }, valign: "middle" });
  s.addText(bullets([
    "875 MB + 2.88 GB, every file SHA-256 verified",
    "we read ChartQA's zip, not the easier parquet — only the zip has the element boxes",
    "four sources normalised into one record type",
    "records store IDs and hashes, never dataset content (GPL-3.0 / AGPL-3.0)"]),
    { ...BODY, fontSize: 13.5, x: 0.62, y: 3.5, w: 8.3, h: 2.4, isTextBox: true });
  stat(s, 9.3, 2.15, 3.4, "18,317", "ChartQA training charts,\ncarrying 28,299 questions");
  stat(s, 9.3, 3.9, 3.4, "12.7", "element boxes on an\naverage annotated chart", TEAL);
  band(s, 6.1, "Reproducible from scratch: pinned versions, recorded hashes, no manual steps.",
       INK, TINT);
}

/* ═════════════════════════════════════════════════════════ 7 · FOUND coverage */
{
  const s = pres.addSlide();
  head(s, "FOUND", "What is actually inside the annotations",
       "We read 2,500 ChartQA annotations rather than trusting the description");
  s.addText("How many charts come with usable element boxes?", { ...BODY, fontSize: 15,
    bold: true, x: 0.62, y: 2.0, w: 7.6, h: 0.4, margin: 0, isTextBox: true });
  [["Vertical bar", "54.6% of ChartQA", 96.8, DEEP],
   ["Horizontal bar", "29.3%", 91.5, DEEP],
   ["Pie", "3.2%", 54.8, TEAL],
   ["Line", "12.9%", 0.0, RED]].forEach(([name, share, pct, colour], i) => {
    const y = 2.6 + i * 0.72;
    s.addText(name, { ...BODY, fontSize: 14, x: 0.62, y, w: 2.5, h: 0.46,
      valign: "middle", margin: 0, isTextBox: true });
    s.addText(share, { ...SMALL, fontSize: 11.5, x: 3.15, y, w: 1.6, h: 0.46,
      valign: "middle", margin: 0, isTextBox: true });
    card(s, 4.85, y + 0.09, 3.3, 0.28, "E3E9ED");
    if (pct > 0) card(s, 4.85, y + 0.09, 3.3 * pct / 100, 0.28, colour);
    s.addText(pct.toFixed(1) + "%", { fontFace: "Cambria", fontSize: 15, bold: true,
      color: colour, x: 8.3, y, w: 1.0, h: 0.46, valign: "middle", margin: 0,
      isTextBox: true });
  });
  stat(s, 9.6, 2.4, 3.15, "0.9999", "median r² of a bar's box height\nagainst its true value", GREEN, 32);
  s.addText("So where boxes exist, they are trustworthy.\nThe problem is coverage, not quality.",
    { ...SMALL, fontSize: 12, x: 9.6, y: 4.0, w: 3.15, h: 0.9, align: "center",
      isTextBox: true });
  band(s, 5.6, "Line charts have no box annotations at all — this data cannot teach a model to point at them.",
       RED, "FBEAE8");
}

/* ═════════════════════════════════════════════════════════ 8 · MEASURED audit */
{
  const s = pres.addSlide();
  head(s, "MEASURED", "Are the grounding annotations any good?",
       "A gate, not a curiosity: had this failed we would have had to build our own");
  card(s, 0.62, 2.0, 6.0, 3.3, TINT);
  s.addText("RefChartQA box audit", { ...BODY, fontSize: 15, bold: true,
    x: 0.9, y: 2.2, w: 5.4, h: 0.4, margin: 0, isTextBox: true });
  s.addText(bullets([
    "200 rows sampled, seed recorded",
    "each box must contain chart ink",
    "each box must mark a region, not the whole chart",
    "checked across all three question types"]),
    { ...BODY, fontSize: 13, x: 0.9, y: 2.7, w: 5.4, h: 2.4, isTextBox: true });
  stat(s, 7.0, 2.35, 2.7, "200 / 200", "acceptable", GREEN, 30);
  stat(s, 10.0, 2.35, 2.7, "100%", "on human, machine\nand PoT alike", GREEN);
  card(s, 7.0, 4.0, 5.75, 1.3, "E8F2EC");
  s.addText("Gate PASSED — RefChartQA's boxes go into training.",
    { ...BODY, fontSize: 14, bold: true, color: GREEN, x: 7.25, y: 4.0, w: 5.3, h: 1.3,
      valign: "middle", isTextBox: true });
  band(s, 5.65, "We set the pass mark before looking. A gate decided after seeing the result is not a gate.",
       INK, TINT);
}

/* ═════════════════════════════════════════════════════════ 9 · FOUND contamination */
{
  const s = pres.addSlide();
  head(s, "FOUND", "The same chart, hiding under two names",
       "RefChartQA is built from ChartQA's images — re-saved, so the files differ");
  card(s, 0.62, 2.1, 5.9, 2.5, "FBEAE8");
  stat(s, 0.62, 2.35, 5.9, "0 of 4,000", "matches found by comparing\nthe image FILES", RED);
  card(s, 6.85, 2.1, 5.9, 2.5, "E8F2EC");
  stat(s, 6.85, 2.35, 5.9, "99.9%", "matches found by comparing\nthe decoded PIXELS", GREEN);
  s.addText("Re-encoding a PNG changes every byte and no pixel. A file-based check finds nothing and reports it confidently.",
    { ...BODY, fontSize: 14, x: 0.62, y: 4.85, w: 12.15, h: 0.5, align: "center",
      isTextBox: true });
  band(s, 5.6, "Why it matters: without this we could train on a chart and then “test” on the same chart through the other dataset, and never know.",
       AMBER, "FDF1E6");
}

/* ═════════════════════════════════════════════════════════ 10 · BUILT eval */
{
  const s = pres.addSlide();
  head(s, "BUILT", "The measuring instrument, before the thing measured",
       "If we score ourselves with our own code, we cannot tell a gain from a bug");
  card(s, 0.62, 2.05, 5.9, 2.9, TINT);
  s.addText("The official scoring", { ...BODY, fontSize: 15, bold: true, color: DEEP,
    x: 0.9, y: 2.25, w: 5.3, h: 0.4, margin: 0, isTextBox: true });
  s.addText(bullets(["taken verbatim from each dataset's release",
                     "its SHA-256 recorded, so it cannot drift",
                     "the scorer of record for every reported number"]),
    { ...BODY, fontSize: 13, x: 0.9, y: 2.75, w: 5.3, h: 2.0, isTextBox: true });
  card(s, 6.85, 2.05, 5.9, 2.9, TINT);
  s.addText("Our own implementation", { ...BODY, fontSize: 15, bold: true, color: TEAL,
    x: 7.13, y: 2.25, w: 5.3, h: 0.4, margin: 0, isTextBox: true });
  s.addText(bullets(["confidence intervals — the official code gives none",
                     "breakdowns by box size, chart type, question kind",
                     "used for analysis, never for a headline"]),
    { ...BODY, fontSize: 13, x: 7.13, y: 2.75, w: 5.3, h: 2.0, isTextBox: true });
  band(s, 5.25, "Two implementations, one rule: where they disagree, the official one wins.",
       INK, TINT);
  s.addText("Building this first also meant evaluation was ready the moment a model was.",
    { ...SMALL, fontSize: 12.5, x: 0.92, y: 6.45, w: 11.5, h: 0.4, isTextBox: true });
}

/* ═════════════════════════════════════════════════════════ 11 · MEASURED agreement */
{
  const s = pres.addSlide();
  head(s, "MEASURED", "Does our scoring agree with the official scoring?",
       "The same 11,690 real predictions, scored twice");
  stat(s, 0.62, 2.3, 3.85, "11,690", "predictions scored\nby both programs");
  stat(s, 4.75, 2.3, 3.85, "0.07", "largest gap in the box score\n(percentage points)", TEAL);
  stat(s, 8.88, 2.3, 3.85, "0", "disagreements across\n423 borderline answers", GREEN);
  card(s, 0.62, 4.05, 12.15, 1.25, TINT);
  s.addText("Per subset — human 0.000, machine 0.068, PoT 0.036 percentage points. Perfect-box test cases agree to one part in a million.",
    { ...BODY, fontSize: 13.5, x: 0.92, y: 4.05, w: 11.5, h: 1.25, valign: "middle",
      isTextBox: true });
  band(s, 5.65, "So an improvement we report later is a real improvement, not a bug in our own scoring.",
       GREEN, "E8F2EC");
}

/* ═════════════════════════════════════════════════════════ 12 · FOUND subtoken */
{
  const s = pres.addSlide();
  head(s, "FOUND", "The targets are smaller than what the model can see",
       "The model does not see pixels — it sees the chart as a grid of 32×32 blocks");
  s.addImage({ path: FIG("fig3_subtoken.png"), x: 0.62, y: 1.9, w: 8.6, h: 3.94 });
  stat(s, 9.85, 2.1, 2.9, "67%", "of targets are thinner than\none block on some side", RED);
  stat(s, 9.85, 3.8, 2.9, "25%", "are smaller than one block\nin total area", AMBER);
  s.addText("7,158 boxes, RefChartQA training split, at 512 px.", { ...SMALL,
    fontSize: 11, x: 9.85, y: 5.3, w: 2.9, h: 0.5, isTextBox: true });
  band(s, 6.0, "More training cannot fix this. It tells us what to test in week 3: raise the resolution and see if the gain lands here.",
       AMBER, "FDF1E6");
}

/* ═════════════════════════════════════════════════════════ 13 · BUILT mining */
{
  const s = pres.addSlide();
  head(s, "BUILT", "A miner that recovers the calculation behind an answer",
       "ChartQA gives the chart's data table — so we can search for the operation");
  [["1", "Take the question and its answer", "“difference between West and South?” → 74"],
   ["2", "Try every operation over the gold table", "West−South, West+South, mean, ratio, …"],
   ["3", "Accept only if EXACTLY ONE explains it", "if two operations both give 74, we refuse"]]
   .forEach(([n, h2, d], i) => {
    const y = 2.1 + i * 1.15;
    s.addShape(pres.ShapeType.ellipse, { x: 0.62, y: y + 0.08, w: 0.46, h: 0.46,
      fill: { color: DEEP }, line: { color: DEEP, width: 0 } });
    s.addText(n, { fontFace: "Calibri", fontSize: 14, bold: true, color: PAPER,
      x: 0.62, y: y + 0.08, w: 0.46, h: 0.46, align: "center", valign: "middle",
      margin: 0, isTextBox: true });
    s.addText(h2, { ...BODY, fontSize: 15.5, bold: true, x: 1.3, y, w: 6.6, h: 0.4,
      margin: 0, isTextBox: true });
    s.addText(d, { ...SMALL, fontSize: 12.5, x: 1.3, y: y + 0.42, w: 6.6, h: 0.4,
      margin: 0, isTextBox: true });
  });
  card(s, 8.3, 2.1, 4.45, 3.3, "FDF1E6");
  s.addText("Why so strict?", { ...BODY, fontSize: 14.5, bold: true, color: AMBER,
    x: 8.58, y: 2.3, w: 3.9, h: 0.4, margin: 0, isTextBox: true });
  s.addText("ChartQA's own scoring allows 5% error. But 5% of the year “2014” is a window of ±100 years.\n\nUnder that rule, mining accepted 2096 as the explanation for an answer of 2019.\n\nThat is not a plan. It is a coincidence, and training on it teaches arithmetic that is wrong.",
    { ...BODY, fontSize: 12.5, x: 8.58, y: 2.75, w: 3.9, h: 2.5, isTextBox: true });
  band(s, 5.75, "Ambiguity is a rejection, never a guess. Every rejection records its reason.",
       INK, TINT);
}

/* ═════════════════════════════════════════════════════════ 14 · FOUND yield */
{
  const s = pres.addSlide();
  head(s, "FOUND", "There is very little in the data that teaches reasoning",
       "We ran the miner over all 28,299 ChartQA training questions");
  [["28,299", "questions searched", MUTED],
   ["14%", "have exactly one\nclear calculation", DEEP],
   ["74%", "of those are just\n“read one number”", TEAL],
   ["4 in 100", "teach real multi-step\nreasoning", RED]].forEach(([v, l, c], i) => {
    const x = 0.62 + i * 3.13;
    card(s, x, 2.2, 2.85, 2.2, TINT);
    stat(s, x, 2.45, 2.85, v, l, c, i === 3 ? 30 : 34);
    if (i < 3) s.addText("→", { fontFace: "Calibri", fontSize: 24, color: MUTED,
      x: x + 2.87, y: 3.05, w: 0.26, h: 0.45, align: "center", margin: 0, isTextBox: true });
  });
  card(s, 0.62, 4.7, 12.15, 1.1, TINT);
  s.addText("Why the other 86% are rejected: 53.9% ambiguous — several operations fit, so we cannot tell which was meant. 19.3% the answer is not a number. 7.2% nothing fits. 5.6% the answer is a category, like a year.",
    { ...BODY, fontSize: 13, x: 0.92, y: 4.7, w: 11.5, h: 1.1, valign: "middle",
      isTextBox: true });
  band(s, 6.05, "The real data can barely teach reasoning at all. So we generate our own.",
       AMBER, "FDF1E6");
}

/* ═════════════════════════════════════════════════════════ 15 · BUILT generator */
{
  const s = pres.addSlide();
  head(s, "BUILT", "A chart generator where we know the answer by construction",
       "8 chart types × 4 difficulty levels = 24,000 examples");
  s.addTable([
    [{ text: "Level", options: { bold: true, color: PAPER, fill: { color: DEEP } } },
     { text: "What it teaches", options: { bold: true, color: PAPER, fill: { color: DEEP } } },
     { text: "Example plan", options: { bold: true, color: PAPER, fill: { color: DEEP } } }],
    ["L1", "read one value", "lookup(West)"],
    ["L2", "compare two values", "difference(West, South)"],
    ["L3", "aggregate over everything", "mean()"],
    ["L4", "an operation inside another", "difference(West, mean())"],
  ], { x: 0.62, y: 2.0, w: 7.9, rowH: 0.42, fontFace: "Calibri", fontSize: 13.5,
       color: INK, border: { type: "solid", color: "D9E2E8", pt: 1 }, valign: "middle" });
  s.addText("bar · horizontal bar · grouped bar · line · multi-line · pie · scatter · area",
    { ...SMALL, fontSize: 12.5, x: 0.62, y: 4.5, w: 7.9, h: 0.4, isTextBox: true });
  stat(s, 8.85, 2.15, 3.9, "6,000", "level-4 examples — the kind\nreal data supplies 4 in 100 of", DEEP);
  card(s, 8.85, 3.85, 3.9, 1.5, "FDF1E6");
  s.addText("Randomised: colours, fonts, grids, dark and light themes, titles, label rotation — so the model cannot learn one look.",
    { ...BODY, fontSize: 12.5, x: 9.1, y: 3.85, w: 3.4, h: 1.5, valign: "middle",
      isTextBox: true });
  band(s, 5.6, "Held-out seeds and styles are sealed off now, for a robustness test in week 4.",
       INK, TINT);
}

/* ═════════════════════════════════════════════════════════ 16 · MEASURED boxes */
{
  const s = pres.addSlide();
  head(s, "MEASURED", "Proving the generated boxes are right",
       "Knowing where we drew a bar is not the same as knowing where its ink is");
  s.addImage({ path: FIG("fig4_verification.png"), x: 0.62, y: 1.95, w: 8.3, h: 3.55 });
  s.addText(bullets([
    "we measure each box against the rendered pixels",
    "kept only if the overlap is at least 0.70",
    "correct boxes score 0.84 – 0.99",
    "every example is checked, not a sample"]),
    { ...BODY, fontSize: 13.5, x: 9.2, y: 2.1, w: 3.55, h: 2.6, isTextBox: true });
  band(s, 5.75, "So “we know the answer because we drew it” is a guarantee, not a hope — no unverified box reaches training.",
       GREEN, "E8F2EC");
}

/* ═════════════════════════════════════════════════════════ 17 · BUILT prompts */
{
  const s = pres.addSlide();
  head(s, "BUILT", "What we ask for, and what we accept back",
       "A prompt is a setting — all three are fixed and hashed before any result");
  s.addTable([
    [{ text: "Prompt", options: { bold: true, color: PAPER, fill: { color: DEEP } } },
     { text: "Size", options: { bold: true, color: PAPER, fill: { color: DEEP }, align: "right" } },
     { text: "Used for", options: { bold: true, color: PAPER, fill: { color: DEEP } } }],
    ["plain", { text: "27 tokens", options: { align: "right" } }, "the published baseline, copied word for word"],
    ["structured", { text: "980 tokens", options: { align: "right" } }, "asking an untrained model for the full record"],
    ["training", { text: "117 tokens", options: { align: "right" } }, "what the fine-tuned model will see"],
  ], { x: 0.62, y: 2.0, w: 7.5, rowH: 0.42, fontFace: "Calibri", fontSize: 13,
       color: INK, border: { type: "solid", color: "D9E2E8", pt: 1 }, valign: "middle" });
  card(s, 0.62, 4.0, 7.5, 1.45, "FDF1E6");
  s.addText("Measured while designing them: pretty-printed JSON costs 80% more tokens than compact for identical content. Demanding compact cut the output 2.6× and raised valid output from 58% to 75%.",
    { ...BODY, fontSize: 12.5, x: 0.88, y: 4.0, w: 6.98, h: 1.45, valign: "middle",
      isTextBox: true });
  card(s, 8.45, 2.0, 4.3, 3.45, TINT);
  s.addText("The parser's one rule", { ...BODY, fontSize: 14.5, bold: true, color: DEEP,
    x: 8.72, y: 2.2, w: 3.8, h: 0.4, margin: 0, isTextBox: true });
  s.addText("We may DROP what the schema cannot hold.\nWe may UNWRAP a record buried in stray text.\nWe NEVER ADD a field the model did not produce.\n\nA lenient parser measures the parser, not the model.",
    { ...BODY, fontSize: 12.5, x: 8.72, y: 2.7, w: 3.8, h: 2.6, isTextBox: true });
  band(s, 5.7, "Invalid output counts as a failure — never quietly replaced with a plausible default.",
       INK, TINT);
}

/* ═════════════════════════════════════════════════════════ 18 · MEASURED baseline */
{
  const s = pres.addSlide();
  head(s, "MEASURED", "Where the untouched model starts",
       "200 frozen ChartQA validation questions, structured prompt, no training");
  s.addTable([
    [{ text: "", options: { fill: { color: DEEP } } },
     { text: "Result", options: { bold: true, color: PAPER, fill: { color: DEEP }, align: "right" } },
     { text: "Measured over", options: { bold: true, color: PAPER, fill: { color: DEEP } } }],
    ["Answers correct (relaxed)", { text: "50.0%", options: { align: "right", bold: true } }, "all 200 questions"],
    ["Output is valid JSON", { text: "66.5%", options: { align: "right" } }, "all 200 questions"],
    ["Output satisfies our schema", { text: "46.5%", options: { align: "right" } }, "all 200 questions, after repair"],
    ["Plan runs without error", { text: "94.4%", options: { align: "right" } }, "the 71 usable records"],
    ["Plan reproduces its own answer", { text: "69.0%", options: { align: "right", bold: true } }, "the 71 usable records"],
  ], { x: 0.62, y: 2.0, w: 12.15, rowH: 0.42, fontFace: "Calibri", fontSize: 13.5,
       color: INK, border: { type: "solid", color: "D9E2E8", pt: 1 }, valign: "middle" });
  card(s, 0.62, 4.75, 12.15, 1.15, "FDF1E6");
  s.addText("Read the last column carefully: the bottom two are conditional on a usable record existing. As a share of all 200 questions, the plan reproduces the answer 24.5% of the time.",
    { ...BODY, fontSize: 13, color: AMBER, x: 0.92, y: 4.75, w: 11.5, h: 1.15,
      valign: "middle", isTextBox: true });
  band(s, 6.15, "Two-thirds of the loss is output that never parses or never satisfies the schema — not wrong answers.",
       INK, TINT);
}

/* ═════════════════════════════════════════════════════════ 19 · FOUND the gap */
{
  const s = pres.addSlide();
  head(s, "FOUND", "The gap we are actually trying to close",
       "The model can already read charts. It cannot yet justify what it says");
  card(s, 0.62, 2.1, 5.9, 2.4, "E8F2EC");
  stat(s, 0.62, 2.4, 5.9, "50%", "of questions answered correctly\nwith no training at all", GREEN);
  card(s, 6.85, 2.1, 5.9, 2.4, "FBEAE8");
  stat(s, 6.85, 2.4, 5.9, "69%", "of its usable records: its own\narithmetic backs its own answer", RED);
  s.addText("Even when the answer is right, the stated reasoning often does not support it. That is what makes the answer uncheckable — and it is what training is for.",
    { ...BODY, fontSize: 14, x: 0.62, y: 4.75, w: 12.15, h: 0.6, align: "center",
      isTextBox: true });
  band(s, 5.65, "We are not mainly trying to make the model read charts better. We are trying to make its answers checkable.",
       INK, TINT);
}

/* ═════════════════════════════════════════════════════════ 20 · close */
{
  const s = pres.addSlide();
  s.background = { color: MIDNIGHT };
  s.addText("What is next", { fontFace: "Cambria", fontSize: 32, bold: true, color: PAPER,
    x: 0.62, y: 0.6, w: 12.15, h: 0.7, isTextBox: true });
  [["Week 2", "Train the model, in two stages:\nfirst pointing, then pointing + reasoning together"],
   ["Week 3", "Evaluate on the sealed test splits,\nand test whether higher resolution helps the small targets"],
   ["Week 4", "Separate visual error from reasoning error,\nand write it up"]]
   .forEach(([w, t], i) => {
    const y = 1.65 + i * 1.25;
    s.addShape(pres.ShapeType.roundRect, { x: 0.62, y, w: 1.7, h: 0.95,
      fill: { color: TEAL }, rectRadius: 0.08, line: { color: TEAL, width: 0 } });
    s.addText(w, { fontFace: "Calibri", fontSize: 15, bold: true, color: PAPER,
      x: 0.62, y, w: 1.7, h: 0.95, align: "center", valign: "middle", margin: 0,
      isTextBox: true });
    s.addText(t, { fontFace: "Calibri", fontSize: 14, color: "CFE4EF", x: 2.55, y,
      w: 10.2, h: 0.95, valign: "middle", isTextBox: true });
  });
  s.addShape(pres.ShapeType.roundRect, { x: 0.62, y: 5.6, w: 5.9, h: 1.1,
    fill: { color: "2E3A6E" }, rectRadius: 0.1, line: { color: "2E3A6E", width: 0 } });
  s.addText("Test splits stay sealed until\nevery decision is committed", { fontFace: "Calibri",
    fontSize: 14, bold: true, color: PAPER, x: 0.62, y: 5.6, w: 5.9, h: 1.1,
    align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addShape(pres.ShapeType.roundRect, { x: 6.85, y: 5.6, w: 5.9, h: 1.1,
    fill: { color: "2E3A6E" }, rectRadius: 0.1, line: { color: "2E3A6E", width: 0 } });
  s.addText("Cost so far: $0\nfree GPUs only", { fontFace: "Calibri", fontSize: 14,
    bold: true, color: PAPER, x: 6.85, y: 5.6, w: 5.9, h: 1.1, align: "center",
    valign: "middle", margin: 0, isTextBox: true });
}

pres.writeFile({ fileName: path.join(__dirname, "ChartQA-Week1.pptx") })
    .then((f) => console.log("wrote", f));
