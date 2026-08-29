const pptxgen = require("pptxgenjs");
const path = require("path");

const FIG = (n) => path.join(__dirname, "figures", n);

// Ocean palette: technical, measured, not the default corporate blue.
const DEEP = "065A82", TEAL = "1C7293", MIDNIGHT = "21295C";
const INK = "233038", MUTED = "5A6B75", PAPER = "FFFFFF", TINT = "F1F5F8";
const GREEN = "12813F", RED = "C0392B";     // match the figures exactly

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                 // 13.333 x 7.5 in — set BEFORE any slide
pres.author = "ChartQA Project";
pres.title = "Grounded Chart Question Answering — Week 1";

const TITLE = { fontFace: "Cambria", fontSize: 34, bold: true, color: INK };
const BODY = { fontFace: "Calibri", fontSize: 15, color: INK };
const SMALL = { fontFace: "Calibri", fontSize: 12, color: MUTED };

function heading(slide, text, sub) {
  slide.addText(text, { ...TITLE, x: 0.6, y: 0.42, w: 12.1, h: 0.7, isTextBox: true });
  if (sub) {
    slide.addText(sub, { ...SMALL, fontSize: 13.5, x: 0.62, y: 1.12, w: 12.1, h: 0.42,
                         isTextBox: true });
  }
}

/** A large number with a label under it — used instead of bullets for findings. */
function stat(slide, x, y, w, value, label, colour) {
  slide.addText(value, { fontFace: "Cambria", fontSize: 42, bold: true,
                         color: colour || DEEP, x, y, w, h: 0.78, align: "center",
                         margin: 0, isTextBox: true });
  slide.addText(label, { ...SMALL, fontSize: 12.5, x, y: y + 0.78, w, h: 0.68,
                         align: "center", valign: "top", margin: 0, isTextBox: true });
}

function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, fill: { color: fill || TINT }, rectRadius: 0.08,
    line: { color: fill || TINT, width: 0 },
  });
}

/* ------------------------------------------------------------------ 1 · title */
{
  const s = pres.addSlide();
  s.background = { color: MIDNIGHT };
  s.addText("Grounded Chart\nQuestion Answering", {
    fontFace: "Cambria", fontSize: 46, bold: true, color: PAPER,
    x: 0.9, y: 1.75, w: 7.6, h: 2.2, lineSpacing: 52, isTextBox: true });
  s.addText("Making a model show where it looked, and how it calculated", {
    fontFace: "Calibri", fontSize: 18, color: "9FC2D6",
    x: 0.95, y: 4.05, w: 8.2, h: 0.5, isTextBox: true });
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.95, y: 5.05, w: 2.1, h: 0.62, fill: { color: TEAL }, rectRadius: 0.1,
    line: { color: TEAL, width: 0 } });
  s.addText("WEEK 1 OF 4", { fontFace: "Calibri", fontSize: 13, bold: true, color: PAPER,
    x: 0.95, y: 5.05, w: 2.1, h: 0.62, align: "center", valign: "middle",
    charSpacing: 1.5, margin: 0, isTextBox: true });
  s.addText("Team: [name]  ·  [name]  ·  [name]", {
    fontFace: "Calibri", fontSize: 14, color: "7FA3B8",
    x: 3.4, y: 5.05, w: 6, h: 0.62, valign: "middle", isTextBox: true });
  s.addImage({ path: FIG("fig2_grounded.png"), x: 9.15, y: 1.55, w: 3.5, h: 2.81,
               transparency: 12 });
  s.addNotes("Week 1 of a four-week project. This week was data, measurement and baselines — everything before training.");
}

/* ---------------------------------------------------------------- 2 · problem */
{
  const s = pres.addSlide();
  heading(s, "The problem", "A right answer and a lucky guess look identical");
  s.addText([
    { text: "A chart model reads a chart and gives an answer.\nThat is all you get.", options: { breakLine: true, paraSpaceAfter: 12 } },
    { text: "You cannot tell whether it:", options: { breakLine: true, paraSpaceAfter: 8, bold: true } },
    { text: "read the right two bars", options: { bullet: true, breakLine: true } },
    { text: "read the wrong bars and got lucky", options: { bullet: true, breakLine: true } },
    { text: "ignored the chart and guessed from the question", options: { bullet: true, breakLine: true } },
  ], { ...BODY, x: 0.62, y: 1.85, w: 5.5, h: 3.2, isTextBox: true });
  card(s, 0.62, 5.25, 5.5, 1.15, "FBEAE8");
  s.addText("So we cannot check its work — and neither can it.", {
    ...BODY, fontSize: 16, bold: true, color: RED,
    x: 0.85, y: 5.25, w: 5.05, h: 1.15, valign: "middle", isTextBox: true });
  s.addImage({ path: FIG("fig1_ungrounded.png"), x: 6.75, y: 1.7, w: 6.0, h: 4.71 });
}

/* -------------------------------------------------------------------- 3 · idea */
{
  const s = pres.addSlide();
  heading(s, "Our idea", "Make the model produce three things instead of one");
  s.addImage({ path: FIG("fig2_grounded.png"), x: 0.6, y: 1.7, w: 6.5, h: 5.21 });
  const rows = [
    ["1", "Where it looked", "boxes around the West and South bars"],
    ["2", "What it did", "difference(West, South)"],
    ["3", "The answer", "74"],
  ];
  rows.forEach(([n, head, detail], i) => {
    const y = 1.85 + i * 1.15;
    s.addShape(pres.ShapeType.ellipse, { x: 7.45, y: y + 0.06, w: 0.46, h: 0.46,
      fill: { color: DEEP }, line: { color: DEEP, width: 0 } });
    s.addText(n, { fontFace: "Calibri", fontSize: 14, bold: true, color: PAPER,
      x: 7.45, y: y + 0.06, w: 0.46, h: 0.46, align: "center", valign: "middle",
      margin: 0, isTextBox: true });
    s.addText(head, { ...BODY, fontSize: 16, bold: true, x: 8.1, y, w: 4.7, h: 0.38,
      margin: 0, isTextBox: true });
    s.addText(detail, { ...SMALL, fontSize: 13, x: 8.1, y: y + 0.4, w: 4.7, h: 0.42,
      margin: 0, isTextBox: true });
  });
  card(s, 7.45, 5.4, 5.35, 1.35, "E8F2EC");
  s.addText("A small program re-does the arithmetic from the boxes.\nIf it disagrees with the answer, we know the answer is unreliable — with no human needed.",
    { ...BODY, fontSize: 13.5, color: GREEN, x: 7.68, y: 5.4, w: 4.9, h: 1.35,
      valign: "middle", isTextBox: true });
}

/* -------------------------------------------------------------------- 4 · plan */
{
  const s = pres.addSlide();
  heading(s, "The four-week plan", "Week 1 is deliberately not about training");
  const weeks = [
    ["Week 1", "Data, measurement,\nand baselines", true],
    ["Week 2", "Train the model", false],
    ["Week 3", "Evaluate it properly", false],
    ["Week 4", "Analyse and write up", false],
  ];
  weeks.forEach(([label, text, active], i) => {
    const x = 0.62 + i * 3.12;
    card(s, x, 2.05, 2.85, 2.5, active ? DEEP : TINT);
    s.addText(label, { fontFace: "Cambria", fontSize: 19, bold: true,
      color: active ? PAPER : INK, x, y: 2.32, w: 2.85, h: 0.5, align: "center",
      margin: 0, isTextBox: true });
    s.addText(text, { fontFace: "Calibri", fontSize: 13.5,
      color: active ? "CFE4EF" : MUTED, x: x + 0.2, y: 2.95, w: 2.45, h: 1.3,
      align: "center", isTextBox: true });
  });
  card(s, 0.62, 5.1, 12.1, 1.4, TINT);
  s.addText("Everything in weeks 2–4 depends on having data we trust and scoring we trust.\nSo week 1 built both, and measured where the untouched model starts.",
    { ...BODY, fontSize: 15, x: 0.95, y: 5.1, w: 11.5, h: 1.4, valign: "middle",
      isTextBox: true });
}

/* -------------------------------------------------------------------- 5 · data */
{
  const s = pres.addSlide();
  heading(s, "The data — and what is actually inside it",
          "We looked inside 2,500 annotations rather than trusting the description");
  s.addTable([
    [{ text: "Dataset", options: { bold: true, color: PAPER, fill: { color: DEEP } } },
     { text: "Training", options: { bold: true, color: PAPER, fill: { color: DEEP }, align: "right" } },
     { text: "Validation", options: { bold: true, color: PAPER, fill: { color: DEEP }, align: "right" } },
     { text: "Test", options: { bold: true, color: PAPER, fill: { color: DEEP }, align: "right" } }],
    ["ChartQA", { text: "28,299", options: { align: "right" } }, { text: "1,920", options: { align: "right" } }, { text: "2,500", options: { align: "right" } }],
    ["RefChartQA", { text: "55,789", options: { align: "right" } }, { text: "6,223", options: { align: "right" } }, { text: "11,690", options: { align: "right" } }],
  ], { x: 0.62, y: 1.9, w: 5.7, rowH: 0.42, fontFace: "Calibri", fontSize: 13.5,
       color: INK, border: { type: "solid", color: "D9E2E8", pt: 1 }, valign: "middle" });
  s.addText("Every file checked against a fingerprint, so we know exactly which version we have.",
    { ...SMALL, x: 0.62, y: 3.3, w: 5.7, h: 0.5, isTextBox: true });

  s.addText("How many charts come with box annotations?", {
    ...BODY, fontSize: 15, bold: true, x: 6.85, y: 1.9, w: 5.9, h: 0.4,
    margin: 0, isTextBox: true });
  const cov = [["Vertical bar", "97%", DEEP], ["Horizontal bar", "92%", DEEP],
               ["Pie", "55%", TEAL], ["Line", "0%", RED]];
  cov.forEach(([name, pct, colour], i) => {
    const y = 2.45 + i * 0.62;
    s.addText(name, { ...BODY, fontSize: 14, x: 6.85, y, w: 3.4, h: 0.5,
      valign: "middle", margin: 0, isTextBox: true });
    s.addText(pct, { fontFace: "Cambria", fontSize: 21, bold: true, color: colour,
      x: 10.3, y, w: 2.4, h: 0.5, align: "right", valign: "middle", margin: 0,
      isTextBox: true });
  });
  card(s, 0.62, 5.35, 12.1, 1.15, "FBEAE8");
  s.addText("Line charts have no box annotations at all — so this data cannot teach the model to point at them.",
    { ...BODY, fontSize: 15, bold: true, color: RED, x: 0.95, y: 5.35, w: 11.5,
      h: 1.15, valign: "middle", isTextBox: true });
}

/* ------------------------------------------------------------------ 6 · ruler */
{
  const s = pres.addSlide();
  heading(s, "We built the ruler before the thing we measure",
          "Before writing any model code, we wrote the scoring — then checked it");
  card(s, 0.62, 1.95, 12.1, 2.6, TINT);
  s.addText("We scored the same 11,690 real predictions twice: once with the official scoring program published with the dataset, once with ours.",
    { ...BODY, fontSize: 15, x: 1.0, y: 2.15, w: 11.3, h: 0.75, isTextBox: true });
  stat(s, 1.0, 2.95, 3.4, "11,690", "predictions scored\nby both programs");
  stat(s, 4.8, 2.95, 3.4, "0.07", "largest difference in the\nbox-accuracy score (points)", TEAL);
  stat(s, 8.6, 2.95, 3.4, "0", "disagreements on 423\ntricky answer cases", GREEN);
  card(s, 0.62, 4.9, 12.1, 1.15, "E8F2EC");
  s.addText("So any improvement we report later is a real improvement — not a bug in our own scoring.",
    { ...BODY, fontSize: 15, bold: true, color: GREEN, x: 0.95, y: 4.9, w: 11.5,
      h: 1.15, valign: "middle", isTextBox: true });
  s.addText("Doing this first also meant the evaluation was ready the moment the model was.",
    { ...SMALL, fontSize: 13, x: 0.95, y: 6.22, w: 11.5, h: 0.5, isTextBox: true });
}

/* --------------------------------------------------------------- 7 · subtoken */
{
  const s = pres.addSlide();
  heading(s, "What we learned #1 — the targets are tiny",
          "The model does not see pixels; it sees the chart as a grid of blocks");
  s.addImage({ path: FIG("fig3_subtoken.png"), x: 0.62, y: 1.72, w: 9.1, h: 4.17 });
  stat(s, 9.95, 2.0, 2.8, "67%", "of targets are narrower than\none block on at least one side", RED);
  stat(s, 9.95, 3.75, 2.8, "25%", "are smaller than one block\nin total area", TEAL);
  s.addText("Measured on RefChartQA at the model's normal input size.", {
    ...SMALL, fontSize: 11.5, x: 9.95, y: 5.3, w: 2.8, h: 0.6, isTextBox: true });
  card(s, 0.62, 6.05, 12.1, 0.95, "FBEAE8");
  s.addText("More training cannot fix this — it tells us to test higher resolutions in week 3.",
    { ...BODY, fontSize: 14.5, bold: true, color: RED, x: 0.95, y: 6.05, w: 11.5,
      h: 0.95, valign: "middle", isTextBox: true });
}

/* ---------------------------------------------------------------- 8 · scarcity */
{
  const s = pres.addSlide();
  heading(s, "What we learned #2 — there is little to teach with",
          "To teach the model how to calculate, we need questions where we know the calculation");
  s.addText("We searched all 28,299 training questions for one where exactly one arithmetic step reproduces the correct answer.",
    { ...BODY, fontSize: 15, x: 0.62, y: 1.9, w: 12.1, h: 0.6, isTextBox: true });
  const steps = [
    ["14%", "of questions gave a clear,\nsingle calculation", DEEP],
    ["74%", "of those were just\n“read one number”", TEAL],
    ["4 in 100", "questions teach real\nmulti-step reasoning", RED],
  ];
  steps.forEach(([value, label, colour], i) => {
    const x = 0.62 + i * 4.15;
    card(s, x, 2.7, 3.8, 2.15, TINT);
    stat(s, x, 2.95, 3.8, value, label, colour);
  });
  steps.slice(0, 2).forEach((_, i) => {
    s.addText("→", { fontFace: "Calibri", fontSize: 26, color: MUTED,
      x: 4.42 + i * 4.15, y: 3.5, w: 0.35, h: 0.5, align: "center", margin: 0,
      isTextBox: true });
  });
  card(s, 0.62, 5.35, 12.1, 1.15, "FBEAE8");
  s.addText("The real data can barely teach reasoning at all — so we generate our own.",
    { ...BODY, fontSize: 15, bold: true, color: RED, x: 0.95, y: 5.35, w: 11.5,
      h: 1.15, valign: "middle", isTextBox: true });
}

/* --------------------------------------------------------------- 9 · generator */
{
  const s = pres.addSlide();
  heading(s, "So we generate our own charts",
          "We know the right answer and the right boxes, because we drew them");
  s.addImage({ path: FIG("fig4_verification.png"), x: 0.62, y: 1.78, w: 8.5, h: 3.64 });
  const facts = [["8 × 4", "chart types ×\ndifficulty levels"],
                 ["24,000", "generated examples"],
                 ["≥ 0.70", "overlap required, or\nthe box is thrown away"]];
  facts.forEach(([value, label], i) => {
    stat(s, 9.55, 1.85 + i * 1.55, 3.2, value, label, i === 2 ? GREEN : DEEP);
  });
  card(s, 0.62, 5.62, 8.5, 1.25, TINT);
  s.addText("Each box is scored against where the chart's ink actually is. Across the 640 examples we checked by hand, correct boxes scored 0.84–0.99 — so no bad example reaches training.",
    { ...BODY, fontSize: 13.5, x: 0.88, y: 5.62, w: 8.0, h: 1.25, valign: "middle",
      isTextBox: true });
}

/* ------------------------------------------------------------- 10 · start/next */
{
  const s = pres.addSlide();
  s.background = { color: MIDNIGHT };
  s.addText("Where we start, and what is next", {
    fontFace: "Cambria", fontSize: 34, bold: true, color: PAPER,
    x: 0.62, y: 0.55, w: 12.1, h: 0.75, isTextBox: true });
  s.addText("The untouched model, before any training", {
    fontFace: "Calibri", fontSize: 14, color: "9FC2D6",
    x: 0.65, y: 1.28, w: 12.1, h: 0.4, isTextBox: true });

  const nums = [["50%", "answers correct"],
                ["47%", "produce a usable\nstructured output"],
                ["69%", "of usable outputs: the model’s own\ncalculation matches its own answer"]];
  nums.forEach(([value, label], i) => {
    const x = 0.62 + i * 4.15;
    s.addShape(pres.ShapeType.roundRect, { x, y: 2.0, w: 3.8, h: 2.0,
      fill: { color: "2E3A6E" }, rectRadius: 0.08,
      line: { color: "2E3A6E", width: 0 } });
    s.addText(value, { fontFace: "Cambria", fontSize: 40, bold: true, color: PAPER,
      x, y: 2.25, w: 3.8, h: 0.75, align: "center", margin: 0, isTextBox: true });
    s.addText(label, { fontFace: "Calibri", fontSize: 12.5, color: "9FC2D6",
      x: x + 0.2, y: 3.05, w: 3.4, h: 0.85, align: "center", isTextBox: true });
  });

  s.addText("Even when it answers correctly, its stated reasoning often does not support the answer.\nThat gap is what we are trying to close.",
    { fontFace: "Calibri", fontSize: 15, color: PAPER, x: 0.62, y: 4.3, w: 12.1,
      h: 0.9, isTextBox: true });

  s.addShape(pres.ShapeType.roundRect, { x: 0.62, y: 5.45, w: 6.0, h: 1.1,
    fill: { color: TEAL }, rectRadius: 0.1, line: { color: TEAL, width: 0 } });
  s.addText("Next: Week 2 — training", { fontFace: "Calibri", fontSize: 17, bold: true,
    color: PAPER, x: 0.62, y: 5.45, w: 6.0, h: 1.1, align: "center", valign: "middle",
    margin: 0, isTextBox: true });
  s.addShape(pres.ShapeType.roundRect, { x: 6.9, y: 5.45, w: 5.82, h: 1.1,
    fill: { color: "2E3A6E" }, rectRadius: 0.1, line: { color: "2E3A6E", width: 0 } });
  s.addText("Cost so far: $0  ·  free GPUs only", { fontFace: "Calibri", fontSize: 17,
    bold: true, color: PAPER, x: 6.9, y: 5.45, w: 5.82, h: 1.1, align: "center",
    valign: "middle", margin: 0, isTextBox: true });
}

pres.writeFile({ fileName: path.join(__dirname, "ChartQA-Week1.pptx") })
    .then((f) => console.log("wrote", f));
