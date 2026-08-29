const pptxgen = require("pptxgenjs");
const path = require("path");
const FIG = (n) => path.join(__dirname, "figures", n);

const DEEP = "065A82", TEAL = "1C7293", MIDNIGHT = "21295C";
const AMBER = "A9541C";
const INK = "233038", MUTED = "5A6B75", PAPER = "FFFFFF", TINT = "F1F5F8";
const GREEN = "12813F", RED = "C0392B";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "ChartQA Project";
pres.title = "Grounded Chart Question Answering — Week 1";

const BODY = { fontFace: "Calibri", fontSize: 16, color: INK };
const SMALL = { fontFace: "Calibri", fontSize: 12.5, color: MUTED };
const KIND = { BUILT: DEEP, MEASURED: TEAL, FOUND: AMBER };

function head(s, kind, title, sub) {
  if (kind) {
    s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 0.36, w: 1.32, h: 0.34,
      fill: { color: KIND[kind] }, rectRadius: 0.06, line: { color: KIND[kind], width: 0 } });
    s.addText(kind, { fontFace: "Calibri", fontSize: 10.5, bold: true, color: PAPER,
      x: 0.6, y: 0.36, w: 1.32, h: 0.34, align: "center", valign: "middle",
      charSpacing: 1.2, margin: 0, isTextBox: true });
  }
  s.addText(title, { fontFace: "Cambria", fontSize: 31, bold: true, color: INK,
    x: 0.6, y: 0.82, w: 12.15, h: 0.62, isTextBox: true });
  if (sub) s.addText(sub, { ...SMALL, fontSize: 13.5, x: 0.62, y: 1.46, w: 12.15, h: 0.4,
    isTextBox: true });
}

function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill || TINT },
    rectRadius: 0.08, line: { color: fill || TINT, width: 0 } });
}

function stat(s, x, y, w, value, label, colour, size) {
  s.addText(value, { fontFace: "Cambria", fontSize: size || 40, bold: true,
    color: colour || DEEP, x, y, w, h: 0.78, align: "center", margin: 0, isTextBox: true });
  s.addText(label, { ...SMALL, fontSize: 12.5, x, y: y + 0.78, w, h: 0.7,
    align: "center", valign: "top", margin: 0, isTextBox: true });
}

function band(s, y, text, colour, fill) {
  card(s, 0.6, y, 12.15, 1.0, fill);
  s.addText(text, { ...BODY, fontSize: 15, bold: true, color: colour,
    x: 0.92, y, w: 11.5, h: 1.0, valign: "middle", isTextBox: true });
}

const bullets = (items, size) => items.map((t, i) => ({
  text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 9 } }));

/* ───────────────────────────────────────────────────────── 1 · title */
{
  const s = pres.addSlide();
  s.background = { color: MIDNIGHT };
  s.addText("Grounded Chart\nQuestion Answering", { fontFace: "Cambria", fontSize: 44,
    bold: true, color: PAPER, x: 0.9, y: 1.7, w: 7.6, h: 2.2, lineSpacing: 50, isTextBox: true });
  s.addText("Making a model show where it looked, and how it calculated",
    { fontFace: "Calibri", fontSize: 17, color: "9FC2D6", x: 0.95, y: 4.0, w: 8.2, h: 0.5,
      isTextBox: true });
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

/* ───────────────────────────────────────────────────────── 2 · problem */
{
  const s = pres.addSlide();
  head(s, null, "The problem", "A right answer and a lucky guess look exactly the same");
  s.addText([{ text: "A chart model reads a chart and gives an answer.\nThat is all you get.",
      options: { breakLine: true, paraSpaceAfter: 14 } },
    { text: "You cannot tell whether it:", options: { breakLine: true, paraSpaceAfter: 10, bold: true } },
    ...bullets(["read the right things on the chart",
                "read the wrong things and got lucky",
                "ignored the chart and guessed from the question"])],
    { ...BODY, x: 0.62, y: 1.95, w: 5.5, h: 3.2, isTextBox: true });
  band(s, 5.4, "So nobody can check its work — not us, not a user, not the model itself.",
       RED, "FBEAE8");
  s.addImage({ path: FIG("fig1_ungrounded.png"), x: 6.75, y: 1.85, w: 6.0, h: 4.71 });
}

/* ───────────────────────────────────────────────────────── 3 · idea */
{
  const s = pres.addSlide();
  head(s, null, "Our idea", "Ask for three things instead of one — then check them against each other");
  s.addImage({ path: FIG("fig2_grounded.png"), x: 0.6, y: 1.95, w: 6.4, h: 5.13 });
  [["1", "Where it looked", "mark the parts of the chart it used"],
   ["2", "How it got there", "state the calculation, not a paragraph of prose"],
   ["3", "The answer", "the number itself"]].forEach(([n, h2, d], i) => {
    const y = 2.15 + i * 1.15;
    s.addShape(pres.ShapeType.ellipse, { x: 7.4, y: y + 0.05, w: 0.46, h: 0.46,
      fill: { color: DEEP }, line: { color: DEEP, width: 0 } });
    s.addText(n, { fontFace: "Calibri", fontSize: 14, bold: true, color: PAPER,
      x: 7.4, y: y + 0.05, w: 0.46, h: 0.46, align: "center", valign: "middle",
      margin: 0, isTextBox: true });
    s.addText(h2, { ...BODY, fontSize: 16.5, bold: true, x: 8.05, y, w: 4.7, h: 0.38,
      margin: 0, isTextBox: true });
    s.addText(d, { ...SMALL, fontSize: 13, x: 8.05, y: y + 0.42, w: 4.7, h: 0.44,
      margin: 0, isTextBox: true });
  });
  card(s, 7.4, 5.6, 5.35, 1.5, "E8F2EC");
  s.addText("Because the calculation is written down, a small program can redo it and check it matches the answer.\n\nNo human needed. No correct answer needed.",
    { ...BODY, fontSize: 13.5, color: GREEN, x: 7.65, y: 5.6, w: 4.85, h: 1.5,
      valign: "middle", isTextBox: true });
}

/* ───────────────────────────────────────────────────────── 4 · plan */
{
  const s = pres.addSlide();
  head(s, null, "The four-week plan", "Week 1 is deliberately not about training");
  [["Week 1", "Data, measurement\nand baselines", true],
   ["Week 2", "Train the model", false],
   ["Week 3", "Evaluate it properly", false],
   ["Week 4", "Analyse and write up", false]].forEach(([lab, txt, on], i) => {
    const x = 0.62 + i * 3.13;
    card(s, x, 2.3, 2.85, 2.4, on ? DEEP : TINT);
    s.addText(lab, { fontFace: "Cambria", fontSize: 19, bold: true, color: on ? PAPER : INK,
      x, y: 2.55, w: 2.85, h: 0.45, align: "center", margin: 0, isTextBox: true });
    s.addText(txt, { fontFace: "Calibri", fontSize: 13.5, color: on ? "CFE4EF" : MUTED,
      x: x + 0.2, y: 3.15, w: 2.45, h: 1.2, align: "center", isTextBox: true });
  });
  band(s, 5.4, "Weeks 2–4 all depend on data we trust and scoring we trust. Week 1 built both, then measured where the untouched model starts.",
       INK, TINT);
}

/* ───────────────────────────────────────────────────────── 5 · summary */
{
  const s = pres.addSlide();
  head(s, null, "What week 1 delivered", null);
  const cols = [
    ["BUILT", DEEP, ["a working data pipeline over two public datasets",
                     "an evaluation system, checked against the official one",
                     "a tool that recovers the calculation behind an answer",
                     "a chart generator that makes its own training data"]],
    ["MEASURED", TEAL, ["how well the datasets are actually annotated",
                        "whether our scoring matches the official scoring",
                        "how small the things the model must find are",
                        "how the untouched model performs today"]],
    ["FOUND", AMBER, ["one chart type has no usable annotations at all",
                      "the datasets overlap far more than filenames suggest",
                      "most targets are too small for the model to locate",
                      "almost nothing in the data teaches reasoning",
                      "the model answers well but cannot justify itself"]],
  ];
  cols.forEach(([name, colour, items], i) => {
    const x = 0.62 + i * 4.13;
    card(s, x, 1.9, 3.85, 4.5, TINT);
    s.addShape(pres.ShapeType.roundRect, { x: x + 0.25, y: 2.15, w: 1.5, h: 0.36,
      fill: { color: colour }, rectRadius: 0.06, line: { color: colour, width: 0 } });
    s.addText(name, { fontFace: "Calibri", fontSize: 11, bold: true, color: PAPER,
      x: x + 0.25, y: 2.15, w: 1.5, h: 0.36, align: "center", valign: "middle",
      charSpacing: 1.2, margin: 0, isTextBox: true });
    s.addText(bullets(items), { ...BODY, fontSize: 13, x: x + 0.25, y: 2.68, w: 3.35,
      h: 3.55, isTextBox: true });
  });
  s.addText("All of it is in the repository and reproducible.  ·  Cost so far: $0 — free GPUs only.",
    { ...SMALL, fontSize: 13, x: 0.62, y: 6.55, w: 12.15, h: 0.4, isTextBox: true });
}

/* ───────────────────────────────────────────────────────── 6 · BUILT data */
{
  const s = pres.addSlide();
  head(s, "BUILT", "The data foundation", "Two public datasets, version-locked so results stay reproducible");
  s.addTable([
    [{ text: "Dataset", options: { bold: true, color: PAPER, fill: { color: DEEP } } },
     { text: "Training", options: { bold: true, color: PAPER, fill: { color: DEEP }, align: "right" } },
     { text: "Validation", options: { bold: true, color: PAPER, fill: { color: DEEP }, align: "right" } },
     { text: "Test", options: { bold: true, color: PAPER, fill: { color: DEEP }, align: "right" } },
     { text: "Marks where to look?", options: { bold: true, color: PAPER, fill: { color: DEEP } } }],
    ["ChartQA", { text: "28,299", options: { align: "right" } }, { text: "1,920", options: { align: "right" } }, { text: "2,500", options: { align: "right" } }, "no"],
    ["RefChartQA", { text: "55,789", options: { align: "right" } }, { text: "6,223", options: { align: "right" } }, { text: "11,690", options: { align: "right" } }, "yes"],
  ], { x: 0.62, y: 2.05, w: 12.15, rowH: 0.46, fontFace: "Calibri", fontSize: 14,
       color: INK, border: { type: "solid", color: "D9E2E8", pt: 1 }, valign: "middle" });
  s.addText(bullets([
    "questions with answers, and for one dataset, the regions a good answer should point at",
    "every file locked to a specific version, so any number can be reproduced later",
    "four different sources brought into one common format",
    "test data set aside from day one and not looked at"]),
    { ...BODY, fontSize: 14.5, x: 0.62, y: 3.85, w: 12.15, h: 2.2, isTextBox: true });
  band(s, 6.1, "About 86,000 chart questions in total. The test portions stay sealed until every decision is locked.",
       INK, TINT);
}

/* ───────────────────────────────────────────────────────── 7 · FOUND coverage */
{
  const s = pres.addSlide();
  head(s, "FOUND", "The annotations are far patchier than advertised",
       "We inspected 2,500 charts instead of trusting the documentation");
  s.addText("How many charts actually come with usable annotations?", { ...BODY,
    fontSize: 16, bold: true, x: 0.62, y: 2.05, w: 8.4, h: 0.4, margin: 0, isTextBox: true });
  [["Bar charts (vertical)", 96.8, DEEP],
   ["Bar charts (horizontal)", 91.5, DEEP],
   ["Pie charts", 54.8, TEAL],
   ["Line charts", 0.0, RED]].forEach(([name, pct, colour], i) => {
    const y = 2.7 + i * 0.78;
    s.addText(name, { ...BODY, fontSize: 15, x: 0.62, y, w: 3.4, h: 0.5,
      valign: "middle", margin: 0, isTextBox: true });
    card(s, 4.2, y + 0.11, 4.2, 0.3, "E3E9ED");
    if (pct > 0) card(s, 4.2, y + 0.11, 4.2 * pct / 100, 0.3, colour);
    s.addText(pct.toFixed(1) + "%", { fontFace: "Cambria", fontSize: 17, bold: true,
      color: colour, x: 8.55, y, w: 1.1, h: 0.5, valign: "middle", margin: 0, isTextBox: true });
  });
  card(s, 10.0, 2.6, 2.75, 2.6, TINT);
  s.addText("Where annotations do exist, we confirmed they are accurate.\n\nThe problem is coverage, not quality.",
    { ...BODY, fontSize: 13.5, x: 10.25, y: 2.6, w: 2.25, h: 2.6, valign: "middle",
      isTextBox: true });
  band(s, 5.85, "Line charts cannot be used to teach the model where to look. We found this before building on top of them.",
       RED, "FBEAE8");
}

/* ───────────────────────────────────────────────────────── 8 · MEASURED audit */
{
  const s = pres.addSlide();
  head(s, "MEASURED", "Is the grounding data good enough to use?",
       "A go / no-go check, with the pass mark set before we looked");
  card(s, 0.62, 2.1, 6.1, 3.2, TINT);
  s.addText("What we checked, on a random sample", { ...BODY, fontSize: 15.5, bold: true,
    x: 0.9, y: 2.3, w: 5.5, h: 0.4, margin: 0, isTextBox: true });
  s.addText(bullets(["does the marked region actually contain part of the chart?",
                     "does it mark a specific element, not the whole picture?",
                     "does this hold across all question types?"]),
    { ...BODY, fontSize: 14, x: 0.9, y: 2.85, w: 5.5, h: 2.2, isTextBox: true });
  stat(s, 7.1, 2.45, 5.65, "200 / 200", "sampled annotations were acceptable", GREEN, 34);
  card(s, 7.1, 4.15, 5.65, 1.15, "E8F2EC");
  s.addText("Passed — so we can build on this data.", { ...BODY, fontSize: 15, bold: true,
    color: GREEN, x: 7.35, y: 4.15, w: 5.2, h: 1.15, valign: "middle", isTextBox: true });
  band(s, 5.65, "Had it failed, we would have had to build our own grounding data — a very different Week 2.",
       INK, TINT);
}

/* ───────────────────────────────────────────────────────── 9 · FOUND overlap */
{
  const s = pres.addSlide();
  head(s, "FOUND", "The two datasets overlap almost completely",
       "The second dataset is built from the first — but the files look different");
  card(s, 0.62, 2.15, 5.9, 2.5, "FBEAE8");
  stat(s, 0.62, 2.45, 5.9, "0 of 4,000", "overlaps found by comparing\nthe image files", RED);
  card(s, 6.85, 2.15, 5.9, 2.5, "E8F2EC");
  stat(s, 6.85, 2.45, 5.9, "99.9%", "overlaps found by comparing\nthe pictures themselves", GREEN);
  s.addText("Re-saving an image changes the file completely while the picture stays identical. The obvious check finds nothing — and says so confidently.",
    { ...BODY, fontSize: 15, x: 0.62, y: 4.95, w: 12.15, h: 0.55, align: "center",
      isTextBox: true });
  band(s, 5.75, "Without catching this, we could train on a chart and then “test” on the same chart — and never know our results were inflated.",
       AMBER, "FDF1E6");
}

/* ───────────────────────────────────────────────────────── 10 · scoring */
{
  const s = pres.addSlide();
  head(s, "BUILT", "Scoring we can actually trust",
       "Built before the model, because you cannot fix a ruler after measuring with it");
  s.addText(bullets([
    "we use each dataset's own official scoring program for every reported number",
    "we also wrote our own, so we can break results down and put error bars on them",
    "then we checked the two against each other on 11,690 real predictions"]),
    { ...BODY, fontSize: 15, x: 0.62, y: 2.0, w: 12.15, h: 1.6, isTextBox: true });
  stat(s, 0.62, 3.75, 3.85, "11,690", "predictions scored twice —\nonce by each program");
  stat(s, 4.75, 3.75, 3.85, "0.07", "largest gap between them,\nin percentage points", TEAL);
  stat(s, 8.88, 3.75, 3.85, "0", "disagreements on the\nhardest answer cases", GREEN);
  band(s, 5.75, "So any improvement we report later is a real improvement — not a bug in how we measure.",
       GREEN, "E8F2EC");
}

/* ───────────────────────────────────────────────────────── 11 · subtoken */
{
  const s = pres.addSlide();
  head(s, "FOUND", "Most targets are too small for the model to locate",
       "The model does not see fine detail — it sees the chart in coarse blocks");
  s.addImage({ path: FIG("fig3_subtoken.png"), x: 0.62, y: 2.0, w: 8.58, h: 3.94 });
  stat(s, 9.85, 2.2, 2.9, "67%", "of targets are smaller than one\nblock the model can see", RED);
  stat(s, 9.85, 3.95, 2.9, "25%", "are smaller than one block\nin every direction", AMBER);
  band(s, 6.15, "More training cannot fix this. It tells us exactly what to test in Week 3: give the model a sharper view.",
       AMBER, "FDF1E6");
}

/* ───────────────────────────────────────────────────────── 12 · reasoning scarcity */
{
  const s = pres.addSlide();
  head(s, "FOUND", "Almost nothing in the data teaches reasoning",
       "We searched all 28,299 training questions for one where the calculation is clear");
  [["28,299", "questions\nsearched", MUTED],
   ["14%", "have one clear\ncalculation behind them", DEEP],
   ["74%", "of those are just\n“read one number off the chart”", TEAL],
   ["4 in 100", "actually teach\nmulti-step reasoning", RED]].forEach(([v, l, c], i) => {
    const x = 0.62 + i * 3.13;
    card(s, x, 2.25, 2.85, 2.35, TINT);
    stat(s, x, 2.5, 2.85, v, l, c, i === 3 ? 30 : 34);
    if (i < 3) s.addText("→", { fontFace: "Calibri", fontSize: 24, color: MUTED,
      x: x + 2.87, y: 3.2, w: 0.26, h: 0.45, align: "center", margin: 0, isTextBox: true });
  });
  s.addText("Most questions are rejected because several different calculations would produce the same answer — so we cannot tell which one the question meant, and we refuse to guess.",
    { ...BODY, fontSize: 14.5, x: 0.62, y: 4.9, w: 12.15, h: 0.9, isTextBox: true });
  band(s, 6.0, "The real data can barely teach reasoning at all. That is why we make our own.",
       AMBER, "FDF1E6");
}

/* ───────────────────────────────────────────────────────── 13 · generator */
{
  const s = pres.addSlide();
  head(s, "BUILT", "So we generate our own charts",
       "When we draw the chart ourselves, we know the answer and the regions for certain");
  s.addImage({ path: FIG("fig4_verification.png"), x: 0.62, y: 2.05, w: 7.9, h: 3.386 });
  s.addText(bullets([
    "8 chart styles at 4 difficulty levels",
    "24,000 examples generated",
    "6,000 of them are the multi-step kind that real data almost never provides",
    "look and style randomised, so the model cannot memorise one appearance"]),
    { ...BODY, fontSize: 14, x: 8.75, y: 2.15, w: 4.0, h: 3.2, isTextBox: true });
  band(s, 5.75, "Every generated example is automatically checked against its own picture before it is allowed into training.",
       GREEN, "E8F2EC");
}

/* ───────────────────────────────────────────────────────── 14 · baseline */
{
  const s = pres.addSlide();
  head(s, "MEASURED", "Where the model stands today, before any training",
       "200 held-out questions, asked in exactly the format we will train towards");
  stat(s, 0.62, 2.3, 3.85, "50%", "of questions answered\ncorrectly", GREEN);
  stat(s, 4.75, 2.3, 3.85, "47%", "produce a properly\nstructured answer", AMBER);
  stat(s, 8.88, 2.3, 3.85, "69%", "of those: the stated calculation\nmatches the stated answer", RED);
  card(s, 0.62, 4.15, 12.15, 1.3, TINT);
  s.addText("Read together: the model can already read charts reasonably well. What it cannot yet do is produce an answer in a form anyone can check — and when it does, its own reasoning often does not support it.",
    { ...BODY, fontSize: 14.5, x: 0.92, y: 4.15, w: 11.5, h: 1.3, valign: "middle",
      isTextBox: true });
  band(s, 5.8, "This is the starting line. Every number we report in Week 3 is measured against it.",
       INK, TINT);
}

/* ───────────────────────────────────────────────────────── 15 · next */
{
  const s = pres.addSlide();
  s.background = { color: MIDNIGHT };
  s.addText("What is next", { fontFace: "Cambria", fontSize: 33, bold: true, color: PAPER,
    x: 0.62, y: 0.6, w: 12.15, h: 0.7, isTextBox: true });
  [["Week 2", "Train the model — first to point at the right things,\nthen to point and reason together"],
   ["Week 3", "Measure it properly on the sealed test data,\nand test whether a sharper view helps the small targets"],
   ["Week 4", "Separate seeing mistakes from reasoning mistakes,\nand write it up"]]
   .forEach(([w, t], i) => {
    const y = 1.7 + i * 1.3;
    s.addShape(pres.ShapeType.roundRect, { x: 0.62, y, w: 1.75, h: 1.0,
      fill: { color: TEAL }, rectRadius: 0.08, line: { color: TEAL, width: 0 } });
    s.addText(w, { fontFace: "Calibri", fontSize: 15.5, bold: true, color: PAPER,
      x: 0.62, y, w: 1.75, h: 1.0, align: "center", valign: "middle", margin: 0,
      isTextBox: true });
    s.addText(t, { fontFace: "Calibri", fontSize: 14.5, color: "CFE4EF", x: 2.6, y,
      w: 10.15, h: 1.0, valign: "middle", isTextBox: true });
  });
  s.addShape(pres.ShapeType.roundRect, { x: 0.62, y: 5.75, w: 5.9, h: 1.05,
    fill: { color: "2E3A6E" }, rectRadius: 0.1, line: { color: "2E3A6E", width: 0 } });
  s.addText("Test data stays sealed until every\ndecision is locked and written down",
    { fontFace: "Calibri", fontSize: 14, bold: true, color: PAPER, x: 0.62, y: 5.75,
      w: 5.9, h: 1.05, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addShape(pres.ShapeType.roundRect, { x: 6.85, y: 5.75, w: 5.9, h: 1.05,
    fill: { color: "2E3A6E" }, rectRadius: 0.1, line: { color: "2E3A6E", width: 0 } });
  s.addText("Cost so far: $0\nfree GPUs only", { fontFace: "Calibri", fontSize: 14,
    bold: true, color: PAPER, x: 6.85, y: 5.75, w: 5.9, h: 1.05, align: "center",
    valign: "middle", margin: 0, isTextBox: true });
}

pres.writeFile({ fileName: path.join(__dirname, "ChartQA-Week1.pptx") })
    .then((f) => console.log("wrote", f));
