import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "C:/Python/実験_修正";
const SKILL_DIR = "C:/Users/aoi7y/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations";
const TMP_DIR = "C:/Python/実験_修正/.codex-slides/fedpact_nexar_concrete_20260913";
const FINAL_PPTX = "C:/Python/実験_修正/output/presentations/FedPACT_NEXAR_具体例_研究会_20260914_v7.pptx";
const RUNTIME_PYTHON = "C:/Users/aoi7y/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
const FONT = "BIZ UDPGothic";

const LEGACY_RUN_DIR = path.join(
  workspaceDir,
  "artifacts/runs/fedpact_nexar_example/run_20260910_002139_presentation_case",
);
const RISK_RUN_DIR = path.join(
  workspaceDir,
  "artifacts/runs/fedpact_nexar_example/run_20260913_balanced_order101_risk_fullgrid",
);
const NORMAL_RUN_DIR = path.join(
  workspaceDir,
  "artifacts/runs/fedpact_nexar_example/run_20260913_balanced_order101_normal_fullgrid",
);
const ORDER_RUN_DIRS = [101, 202, 303].map((order) => path.join(
  workspaceDir,
  `artifacts/runs/fedpact_nexar_example/run_20260913_balanced_order${order}_${order === 101 ? "risk" : "diagnostic"}`,
));
ORDER_RUN_DIRS[0] = RISK_RUN_DIR;
const FULLBATCH_RUN_DIR = path.join(
  workspaceDir,
  "artifacts/runs/fedpact_nexar_example/run_20260913_balanced_fullbatch_diagnostic",
);
const FRAME_DIR = path.join(workspaceDir, "artifacts/datasets/frames/train");

const COLORS = {
  ink: "#102A43",
  muted: "#52667A",
  paper: "#F7F6F2",
  white: "#FFFFFF",
  navy: "#12304A",
  teal: "#168A7A",
  tealSoft: "#DDF1EC",
  blue: "#2F6FA5",
  blueSoft: "#E5EFF8",
  risk: "#D95D45",
  riskSoft: "#F7E3DE",
  amber: "#D28B23",
  amberSoft: "#F8ECCE",
  lavender: "#ECE6F5",
  line: "#C7D1D9",
  faint: "#E7ECEF",
  gray: "#7D8A97",
  dark: "#071724",
};

const presentation = Presentation.create({
  slideSize: { width: 1280, height: 720 },
});

function addText(slide, text, position, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: options.fontSize ?? 24,
    bold: options.bold ?? false,
    color: options.color ?? COLORS.ink,
    alignment: options.alignment ?? "left",
    verticalAlignment: options.verticalAlignment ?? "top",
    autoFit: options.autoFit ?? "none",
    wrap: options.wrap ?? "square",
    lineSpacing: options.lineSpacing ?? 1.0,
    insets: options.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  };
  return shape;
}

function addRichText(slide, runs, position, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text.set([runs]);
  shape.text.style = {
    typeface: FONT,
    fontSize: options.fontSize ?? 24,
    color: options.color ?? COLORS.ink,
    alignment: options.alignment ?? "left",
    verticalAlignment: options.verticalAlignment ?? "top",
    autoFit: options.autoFit ?? "none",
    wrap: "square",
    insets: options.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  };
  return shape;
}

function addBox(slide, position, options = {}) {
  return slide.shapes.add({
    geometry: options.geometry ?? "rect",
    position,
    fill: options.fill ?? COLORS.white,
    line: {
      style: options.lineStyle ?? "solid",
      fill: options.line ?? COLORS.line,
      width: options.lineWidth ?? 1,
    },
    borderRadius: options.borderRadius,
  });
}

function addLine(slide, x, y, width, color = COLORS.line, thickness = 2) {
  return slide.shapes.add({
    geometry: "line",
    position: { left: x, top: y, width, height: 0 },
    fill: "none",
    line: { style: "solid", fill: color, width: thickness },
  });
}

function addArrow(slide, source, target, options = {}) {
  return slide.shapes.connect(source, target, {
    kind: options.kind ?? "straight",
    fromSide: options.fromSide ?? "right",
    toSide: options.toSide ?? "left",
    line: { style: options.style ?? "solid", fill: options.color ?? COLORS.blue, width: options.width ?? 2.5 },
    // artifact-toolのconnectorはtail側がsource→targetの矢印先端になる。
    tail: { type: "triangle", width: "med", length: "med" },
  });
}

async function addImage(slide, imagePath, position, options = {}) {
  const bytes = await fs.readFile(imagePath);
  const ext = path.extname(imagePath).toLowerCase();
  const contentType = ext === ".png" ? "image/png" : "image/jpeg";
  if (options.border) {
    addBox(slide, {
      left: position.left - (options.borderWidth ?? 3),
      top: position.top - (options.borderWidth ?? 3),
      width: position.width + 2 * (options.borderWidth ?? 3),
      height: position.height + 2 * (options.borderWidth ?? 3),
    }, {
      fill: options.border,
      line: options.border,
      lineWidth: 0,
      geometry: options.geometry ?? "rect",
    });
  }
  return slide.images.add({
    blob: new Uint8Array(bytes),
    contentType,
    alt: options.alt ?? path.basename(imagePath),
    fit: options.fit ?? "cover",
    position,
    geometry: options.geometry ?? "rect",
    crop: options.crop,
  });
}

function addTitle(slide, title, section) {
  addText(slide, title, { left: 64, top: 42, width: 1080, height: 54 }, {
    fontSize: 36,
    bold: true,
    color: COLORS.ink,
  });
  if (section) {
    addText(slide, section, { left: 1158, top: 48, width: 62, height: 26 }, {
      fontSize: 16,
      bold: true,
      color: COLORS.blue,
      alignment: "right",
    });
  }
  addLine(slide, 64, 105, 1152, COLORS.line, 1.5);
}

function addFooter(slide, text, page) {
  addText(slide, text, { left: 64, top: 682, width: 1040, height: 20 }, {
    fontSize: 14,
    color: COLORS.gray,
  });
  addText(slide, String(page), { left: 1160, top: 680, width: 56, height: 22 }, {
    fontSize: 15,
    color: COLORS.gray,
    alignment: "right",
  });
}

function addProbabilityBar(slide, x, y, width, normal, risk, options = {}) {
  const h = options.height ?? 26;
  addBox(slide, { left: x, top: y, width, height: h }, { fill: COLORS.faint, line: COLORS.faint, lineWidth: 0 });
  addBox(slide, { left: x, top: y, width: width * normal, height: h }, { fill: COLORS.blue, line: COLORS.blue, lineWidth: 0 });
  addBox(slide, { left: x + width * normal, top: y, width: width * risk, height: h }, { fill: COLORS.risk, line: COLORS.risk, lineWidth: 0 });
  if (options.labels !== false) {
    addText(slide, `normal ${normal.toFixed(3)}`, { left: x, top: y - 25, width: width / 2, height: 22 }, {
      fontSize: options.fontSize ?? 17,
      color: COLORS.blue,
    });
    addText(slide, `risk ${risk.toFixed(3)}`, { left: x + width / 2, top: y - 25, width: width / 2, height: 22 }, {
      fontSize: options.fontSize ?? 17,
      color: COLORS.risk,
      alignment: "right",
    });
  }
}

function addMetricBand(slide, x, y, label, value, color, width = 320) {
  const labelWidth = width <= 270 ? 122 : 170;
  addText(slide, label, { left: x, top: y, width: labelWidth, height: 28 }, {
    fontSize: width <= 270 ? 18 : 19,
    color: COLORS.muted,
  });
  addText(slide, value, { left: x + labelWidth + 4, top: y - 1, width: width - labelWidth - 4, height: 28 }, {
    fontSize: width <= 270 ? 19 : 21,
    bold: true,
    color,
    alignment: "right",
  });
}

function addStageLabel(slide, text, x, y, color) {
  const box = addBox(slide, { left: x, top: y, width: 150, height: 34 }, {
    fill: color,
    line: color,
    lineWidth: 0,
    geometry: "roundRect",
  });
  addText(slide, text, { left: x + 10, top: y + 5, width: 130, height: 24 }, {
    fontSize: 17,
    bold: true,
    color: COLORS.white,
    alignment: "center",
    verticalAlignment: "middle",
  });
  return box;
}

function setNotes(slide, lines) {
  slide.speakerNotes.textFrame.setText(lines.join("\n"));
  slide.speakerNotes.setVisible(true);
}

const privateFrames = [4, 5, 6, 7].map((i) => path.join(FRAME_DIR, "d626422e71", `${String(i).padStart(3, "0")}.jpg`));
const privateOther = path.join(FRAME_DIR, "95c3bf1ba2", "007.jpg");
const seedImages = {
  A: path.join(FRAME_DIR, "e8fd46bae4", "007.jpg"),
  B: path.join(FRAME_DIR, "c393761248", "007.jpg"),
  C: path.join(FRAME_DIR, "8018ff931a", "007.jpg"),
  D: path.join(FRAME_DIR, "fe067d8a9f", "007.jpg"),
};
const transformImages = {
  identity: path.join(LEGACY_RUN_DIR, "server/transformation_candidates/rank01_00_identity_none/frame_03.png"),
  dark: path.join(LEGACY_RUN_DIR, "server/transformation_candidates/rank01_01_brightness_0p75/frame_03.png"),
  contrast: path.join(LEGACY_RUN_DIR, "server/transformation_candidates/rank01_06_contrast_1p2/frame_03.png"),
  crop: path.join(LEGACY_RUN_DIR, "server/transformation_candidates/rank01_07_center_crop_0p85/frame_03.png"),
};
const proxyFrames = [0, 1, 2, 3].map((i) => path.join(LEGACY_RUN_DIR, "server/proxy_frames", `proxy_${String(i).padStart(2, "0")}.png`));

const readJson = async (filePath) => JSON.parse(await fs.readFile(filePath, "utf8"));
const riskSummary = await readJson(path.join(RISK_RUN_DIR, "summary.json"));
const normalSummary = await readJson(path.join(NORMAL_RUN_DIR, "summary.json"));
const riskTrainTrace = await readJson(path.join(RISK_RUN_DIR, "client_local/private_trace.json"));
const riskEvalTrace = await readJson(path.join(RISK_RUN_DIR, "client_local/eval_trace.json"));
const riskLayer2 = await readJson(path.join(RISK_RUN_DIR, "server/layer2_trace.json"));
const normalLayer2 = await readJson(path.join(NORMAL_RUN_DIR, "server/layer2_trace.json"));
const orderTraces = await Promise.all(ORDER_RUN_DIRS.map(async (runDir, index) => ({
  order: [101, 202, 303][index],
  train: await readJson(path.join(runDir, "client_local/private_trace.json")),
  eval: await readJson(path.join(runDir, "client_local/eval_trace.json")),
})));
const fullBatchTrace = {
  train: await readJson(path.join(FULLBATCH_RUN_DIR, "client_local/private_trace.json")),
  eval: await readJson(path.join(FULLBATCH_RUN_DIR, "client_local/eval_trace.json")),
};

const frame = (sampleId, index = 7) => path.join(FRAME_DIR, sampleId, `${String(index).padStart(3, "0")}.jpg`);
const probability = (record, phase = "post") => Number(record[`${phase}_probability`][1]);
const deltaRisk = (record) => probability(record, "post") - probability(record, "pre");
const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
const recordsFor = (trace, label) => (trace.private_samples ?? trace.samples).filter((record) => Number(record.label) === label);
const meanRisk = (trace, label, phase) => mean(recordsFor(trace, label).map((record) => probability(record, phase)));

function addDot(slide, x, y, color, radius = 6) {
  return addBox(slide, { left: x - radius, top: y - radius, width: radius * 2, height: radius * 2 }, {
    geometry: "ellipse",
    fill: color,
    line: COLORS.white,
    lineWidth: 1.5,
  });
}

function addSlopePanel(slide, records, position, title) {
  const { left, top, width, height } = position;
  addText(slide, title, { left, top, width, height: 30 }, { fontSize: 21, bold: true });
  const plotTop = top + 46;
  const plotHeight = height - 86;
  const xPre = left + 110;
  const xPost = left + width - 72;
  [0, 0.5, 1].forEach((tick) => {
    const y = plotTop + (1 - tick) * plotHeight;
    addLine(slide, left + 60, y, width - 105, tick === 0.5 ? COLORS.line : COLORS.faint, 1);
    addText(slide, tick.toFixed(1), { left, top: y - 10, width: 50, height: 20 }, { fontSize: 14, color: COLORS.gray, alignment: "right" });
  });
  addText(slide, "q(risk)", { left: left + 2, top: plotTop - 30, width: 70, height: 22 }, { fontSize: 15, bold: true, color: COLORS.muted });
  addText(slide, "qᵖʳᵉ", { left: xPre - 28, top: plotTop + plotHeight + 14, width: 58, height: 24 }, { fontSize: 17, bold: true, alignment: "center" });
  addText(slide, "qᵖᵒˢᵗ", { left: xPost - 34, top: plotTop + plotHeight + 14, width: 68, height: 24 }, { fontSize: 17, bold: true, alignment: "center" });
  records.forEach((record) => {
    const color = Number(record.label) === 1 ? COLORS.risk : COLORS.blue;
    const yPre = plotTop + (1 - probability(record, "pre")) * plotHeight;
    const yPost = plotTop + (1 - probability(record, "post")) * plotHeight;
    slide.shapes.add({
      geometry: "line",
      position: { left: xPre, top: yPre, width: xPost - xPre, height: yPost - yPre },
      fill: "none",
      line: { style: "solid", fill: color, width: 2.2 },
    });
    addDot(slide, xPre, yPre, color, 5);
    addDot(slide, xPost, yPost, color, 5);
  });
}

function addMiniTable(slide, headers, rows, position, widths) {
  const rowHeight = position.height / (rows.length + 1);
  let x = position.left;
  headers.forEach((header, index) => {
    addBox(slide, { left: x, top: position.top, width: widths[index], height: rowHeight }, {
      fill: COLORS.navy,
      line: COLORS.white,
      lineWidth: 1,
    });
    addText(slide, header, { left: x + 6, top: position.top + 6, width: widths[index] - 12, height: rowHeight - 10 }, {
      fontSize: 16,
      bold: true,
      color: COLORS.white,
      alignment: index === 0 ? "left" : "center",
      verticalAlignment: "middle",
    });
    x += widths[index];
  });
  rows.forEach((row, rowIndex) => {
    x = position.left;
    row.forEach((cell, colIndex) => {
      addBox(slide, { left: x, top: position.top + rowHeight * (rowIndex + 1), width: widths[colIndex], height: rowHeight }, {
        fill: rowIndex % 2 === 0 ? COLORS.white : "#F0F3F5",
        line: COLORS.line,
        lineWidth: 1,
      });
      addText(slide, String(cell), { left: x + 6, top: position.top + rowHeight * (rowIndex + 1) + 6, width: widths[colIndex] - 12, height: rowHeight - 10 }, {
        fontSize: 15,
        bold: colIndex === 0,
        color: colIndex === 0 ? COLORS.ink : COLORS.muted,
        alignment: colIndex === 0 ? "left" : "center",
        verticalAlignment: "middle",
      });
      x += widths[colIndex];
    });
  });
}

if (false) {
// Legacy slides kept as a visual reference; the 2026-09-13 case deck is authored below.
// Slide 1
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.navy;
  await addImage(slide, privateFrames[3], { left: 650, top: 0, width: 630, height: 720 }, {
    alt: "NEXARの危険運転場面。前方車両へ接近している",
    fit: "cover",
  });
  addBox(slide, { left: 0, top: 0, width: 650, height: 720 }, { fill: COLORS.navy, line: COLORS.navy, lineWidth: 0 });
  addBox(slide, { left: 594, top: 0, width: 56, height: 720 }, { fill: COLORS.teal, line: COLORS.teal, lineWidth: 0 });
  addText(slide, "FedPACT Layer 2", { left: 66, top: 70, width: 480, height: 34 }, {
    fontSize: 20,
    bold: true,
    color: "#84D2C5",
  });
  addText(slide, "現場の適応知識を\n生画像なしで\nクラウドへ戻す", { left: 66, top: 135, width: 485, height: 220 }, {
    fontSize: 54,
    bold: true,
    color: COLORS.white,
    lineSpacing: 0.96,
  });
  addText(slide, "NEXARの具体例で追う\nseed探索・画像変換・proxy選択", { left: 70, top: 405, width: 450, height: 78 }, {
    fontSize: 25,
    color: "#DCE7EE",
    lineSpacing: 1.12,
  });
  addText(slide, "研究会説明用ケーススタディ", { left: 70, top: 586, width: 360, height: 26 }, {
    fontSize: 18,
    color: "#A9BCC9",
  });
  addText(slide, "YOSHIMURA Aoi   2026.09.10", { left: 70, top: 635, width: 420, height: 26 }, {
    fontSize: 18,
    color: "#DCE7EE",
  });
  setNotes(slide, [
    "冒頭では、モデルの性能値ではなく研究の問いを示す。",
    "フィジカルAIが現場経験を継続的に学ぶには、現場から学習へ知識を戻す循環が必要になる。一方で、現場の生画像を一か所へ集められない場合がある。",
    "今回の問いは、端末で獲得した適応知識を、生画像を送らずにクラウド側へ戻せるか、である。",
    "この発表ではFedPACTのLayer 2を、NEXARの具体例で説明する。",
    "Source: notes/advisor/CURRENT.md; notes/advisor/sources/raw/2026-09-08/professor_physical_ai_hook_context.md",
  ]);
}

// Slide 2
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.paper;
  addTitle(slide, "proxy generationが必要になる理由", "01");

  await addImage(slide, privateOther, { left: 64, top: 145, width: 420, height: 236 }, {
    alt: "端末だけが保持するNEXAR映像",
    fit: "cover",
  });
  addText(slide, "CLIENT", { left: 64, top: 400, width: 120, height: 30 }, {
    fontSize: 18,
    bold: true,
    color: COLORS.teal,
  });
  addText(slide, "現場固有の危険パターンを\nprivate映像から学習", { left: 64, top: 435, width: 420, height: 68 }, {
    fontSize: 26,
    bold: true,
    lineSpacing: 1.1,
  });
  addText(slide, "人物・位置・ナンバーを含み得る映像は端末内に残す", { left: 64, top: 520, width: 430, height: 56 }, {
    fontSize: 20,
    color: COLORS.muted,
  });

  addBox(slide, { left: 540, top: 130, width: 3, height: 472 }, { fill: COLORS.risk, line: COLORS.risk, lineWidth: 0 });
  addText(slide, "生画像を送れない壁", { left: 452, top: 610, width: 180, height: 24 }, {
    fontSize: 17,
    bold: true,
    color: COLORS.risk,
    alignment: "center",
  });

  const payload = addBox(slide, { left: 595, top: 160, width: 260, height: 110 }, {
    fill: COLORS.tealSoft,
    line: COLORS.teal,
    lineWidth: 2,
    geometry: "roundRect",
  });
  addText(slide, "判断の要約を送る", { left: 615, top: 177, width: 220, height: 28 }, {
    fontSize: 24,
    bold: true,
    color: COLORS.teal,
    alignment: "center",
  });
  addText(slide, "μ_post・Δμ_local\nfrequency・LoRA更新情報", { left: 604, top: 216, width: 242, height: 46 }, {
    fontSize: 15,
    color: COLORS.ink,
    alignment: "center",
  });
  const proxy = addBox(slide, { left: 920, top: 160, width: 275, height: 110 }, {
    fill: COLORS.blueSoft,
    line: COLORS.blue,
    lineWidth: 2,
    geometry: "roundRect",
  });
  addText(slide, "SERVER", { left: 940, top: 178, width: 235, height: 25 }, {
    fontSize: 19,
    bold: true,
    color: COLORS.blue,
    alignment: "center",
  });
  addText(slide, "seedから代理画像を作る", { left: 940, top: 217, width: 235, height: 30 }, {
    fontSize: 23,
    bold: true,
    alignment: "center",
  });
  addArrow(slide, payload, proxy, { color: COLORS.blue, width: 3 });

  const layer1 = addBox(slide, { left: 590, top: 345, width: 175, height: 70 }, {
    fill: COLORS.white,
    line: COLORS.gray,
    geometry: "roundRect",
  });
  addText(slide, "Layer 1\n端末で適応", { left: 605, top: 354, width: 145, height: 52 }, {
    fontSize: 20,
    bold: true,
    alignment: "center",
  });
  const layer2 = addBox(slide, { left: 805, top: 330, width: 190, height: 100 }, {
    fill: COLORS.riskSoft,
    line: COLORS.risk,
    lineWidth: 3,
    geometry: "roundRect",
  });
  addText(slide, "Layer 2\nproxy生成", { left: 825, top: 350, width: 150, height: 62 }, {
    fontSize: 25,
    bold: true,
    color: COLORS.risk,
    alignment: "center",
  });
  const layer3 = addBox(slide, { left: 1035, top: 345, width: 175, height: 70 }, {
    fill: COLORS.white,
    line: COLORS.gray,
    geometry: "roundRect",
  });
  addText(slide, "Layer 3\n基盤へ蒸留", { left: 1050, top: 354, width: 145, height: 52 }, {
    fontSize: 20,
    bold: true,
    alignment: "center",
  });
  addArrow(slide, layer1, layer2, { color: COLORS.risk, width: 3 });
  addArrow(slide, layer2, layer3, { color: COLORS.risk, width: 3 });
  addText(slide, "今回の説明範囲", { left: 820, top: 452, width: 160, height: 26 }, {
    fontSize: 18,
    bold: true,
    color: COLORS.risk,
    alignment: "center",
  });
  addText(slide, "端末の判断を再現するserver画像を探し、\nLayer 3へ渡せるpairを作る", { left: 595, top: 515, width: 610, height: 64 }, {
    fontSize: 23,
    bold: true,
    color: COLORS.ink,
    alignment: "center",
  });
  addFooter(slide, "研究方針：private raw dataは送らず、判断の要約を介してfoundation側へ接続", 2);
  setNotes(slide, [
    "左がclient、右がserverである。clientはprivate映像を保持したまま局所適応する。",
    "serverが画像を持たないと、異種foundation modelへ出力蒸留する入力がない。そこでLayer 2が、server所有seedから代理入力を作る。",
    "clientから送るbasic payloadはLoRA更新情報、μ_post、Δμ_local、frequency。μ_preはdelta計算にだけ使い、独立には送らない。",
    "この資料はLayer 2の必要性と処理を具体化する。",
    "Source: notes/advisor/CURRENT.md; SOURCE_INDEX.md D-041, D-055, D-059",
  ]);
}

// Slide 3
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.paper;
  addTitle(slide, "端末から届く目標は画像ではなく判断の変化", "02");
  addStageLabel(slide, "CLIENT 内", 64, 127, COLORS.teal);

  const frameW = 242;
  const frameH = 136;
  for (let i = 0; i < privateFrames.length; i += 1) {
    await addImage(slide, privateFrames[i], { left: 64 + i * (frameW + 8), top: 185, width: frameW, height: frameH }, {
      alt: `client private sequence frame ${i + 1}`,
      fit: "cover",
    });
  }
  addText(slide, "表示画像はrisk prototypeを作った3件のうち1件", { left: 64, top: 333, width: 550, height: 26 }, {
    fontSize: 17,
    color: COLORS.gray,
  });

  addText(slide, "局所適応前の平均出力", { left: 64, top: 397, width: 280, height: 30 }, {
    fontSize: 23,
    bold: true,
  });
  addProbabilityBar(slide, 64, 462, 420, 0.315, 0.685, { height: 30, fontSize: 19 });

  addText(slide, "局所適応後の平均出力  μ_post", { left: 550, top: 397, width: 390, height: 30 }, {
    fontSize: 23,
    bold: true,
  });
  addProbabilityBar(slide, 550, 462, 420, 0.067, 0.933, { height: 30, fontSize: 19 });

  addText(slide, "risk判断の変化", { left: 1014, top: 405, width: 200, height: 27 }, {
    fontSize: 20,
    color: COLORS.muted,
    alignment: "center",
  });
  addText(slide, "+0.248", { left: 1014, top: 445, width: 200, height: 58 }, {
    fontSize: 42,
    bold: true,
    color: COLORS.risk,
    alignment: "center",
  });
  addText(slide, "Δμ_local", { left: 1014, top: 505, width: 200, height: 28 }, {
    fontSize: 19,
    color: COLORS.risk,
    alignment: "center",
  });

  addBox(slide, { left: 64, top: 573, width: 1150, height: 72 }, {
    fill: COLORS.tealSoft,
    line: COLORS.teal,
    lineWidth: 1.5,
    geometry: "roundRect",
  });
  addRichText(slide, [
    { run: "SERVERへ送る   ", textStyle: { bold: true, color: COLORS.teal } },
    { run: "μ_post=[0.067, 0.933]   Δμ_local=[−0.248, +0.248]   frequency=3   LoRA更新情報", textStyle: { bold: true, color: COLORS.ink } },
  ], { left: 88, top: 596, width: 1100, height: 34 }, { fontSize: 21, alignment: "center" });
  addFooter(slide, "Diagnostic run: run_20260910_002139_presentation_case", 3);
  setNotes(slide, [
    "この例ではclient private側のrisk場面3件をclass-wise meanでまとめた。上の4フレームは3件のうち代表例である。",
    "局所適応前の平均risk確率は0.685、適応後は0.933。risk方向のlocal deltaは+0.248。",
    "clientは同一private sample上でpreとpostを計算する。serverへはμ_post、Δμ_local、frequencyを送るが、private画像、sample ID、μ_preは送らない。",
    "説明では、μ_postを『学習後にどんな判断をするようになったか』、Δμ_localを『学習で判断がどれだけ変わったか』と言い換える。",
    "Source: artifacts/runs/fedpact_nexar_example/run_20260910_002139_presentation_case/client_local/private_trace.json; summary.json",
  ]);
}

// Slide 4
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.paper;
  addTitle(slide, "探索距離はseed候補を順位付けするための尺度", "03");

  const target = addBox(slide, { left: 64, top: 165, width: 355, height: 190 }, {
    fill: COLORS.tealSoft,
    line: COLORS.teal,
    lineWidth: 2,
    geometry: "roundRect",
  });
  addText(slide, "端末から届いた目標", { left: 88, top: 187, width: 307, height: 32 }, {
    fontSize: 25,
    bold: true,
    color: COLORS.teal,
    alignment: "center",
  });
  addText(slide, "μ_post", { left: 88, top: 235, width: 307, height: 30 }, {
    fontSize: 22,
    bold: true,
    alignment: "center",
  });
  addProbabilityBar(slide, 95, 305, 294, 0.067, 0.933, { height: 28, fontSize: 17 });

  const model = addBox(slide, { left: 486, top: 205, width: 270, height: 110 }, {
    fill: COLORS.white,
    line: COLORS.blue,
    lineWidth: 2,
    geometry: "roundRect",
  });
  addText(slide, "updated global surrogate", { left: 505, top: 226, width: 232, height: 30 }, {
    fontSize: 21,
    bold: true,
    color: COLORS.blue,
    alignment: "center",
  });
  addText(slide, "seed画像を確率分布へ変換", { left: 505, top: 270, width: 232, height: 25 }, {
    fontSize: 17,
    color: COLORS.muted,
    alignment: "center",
  });

  const candidate = addBox(slide, { left: 824, top: 165, width: 390, height: 190 }, {
    fill: COLORS.blueSoft,
    line: COLORS.blue,
    lineWidth: 2,
    geometry: "roundRect",
  });
  addText(slide, "seed候補 k の出力", { left: 850, top: 187, width: 338, height: 32 }, {
    fontSize: 25,
    bold: true,
    color: COLORS.blue,
    alignment: "center",
  });
  addText(slide, "p_G^after(x_seed,k)", { left: 850, top: 235, width: 338, height: 30 }, {
    fontSize: 22,
    bold: true,
    alignment: "center",
  });
  addProbabilityBar(slide, 864, 305, 310, 0.089, 0.911, { height: 28, fontSize: 17 });

  addArrow(slide, target, model, { color: COLORS.teal, width: 3 });
  addArrow(slide, model, candidate, { color: COLORS.blue, width: 3 });

  addBox(slide, { left: 130, top: 410, width: 1020, height: 92 }, {
    fill: COLORS.white,
    line: COLORS.line,
    lineWidth: 1.5,
    geometry: "roundRect",
  });
  addText(slide, "d_k = D_KL( μ_post  ||  p_G^after(x_seed,k) )", { left: 170, top: 430, width: 940, height: 45 }, {
    fontSize: 31,
    bold: true,
    color: COLORS.ink,
    alignment: "center",
  });

  addText(slide, "なぜ距離が必要か", { left: 64, top: 550, width: 225, height: 30 }, {
    fontSize: 23,
    bold: true,
    color: COLORS.risk,
  });
  addText(slide, "多数のseedを並べるには、2つの確率分布のずれを1つの数値にする必要がある", { left: 300, top: 548, width: 900, height: 36 }, {
    fontSize: 22,
    bold: true,
  });
  addText(slide, "距離が小さいほど、updated global surrogateが端末の学習後判断に近い反応をする", { left: 300, top: 602, width: 900, height: 34 }, {
    fontSize: 21,
    color: COLORS.muted,
  });
  addFooter(slide, "retrievalではpost prototypeのみを用いる。画像の画素距離や見た目の類似度ではない", 4);
  setNotes(slide, [
    "探索距離は突然導入するのではなく、seed候補を順位付けする必要から導入する。",
    "updated global surrogateへseed画像を入れるとnormal/riskの確率分布が得られる。その分布と端末から届いたμ_postのずれをKL divergenceで1つの数値にする。",
    "式の向きはD_KL(μ_post || candidate)。μ_postを目標分布、candidateを再現側として評価する。",
    "距離が小さいほど、画像がprivate画像に似ているのではなく、モデルの反応が端末の学習後判断に近い。",
    "この例のBは[0.089, 0.911]で、目標[0.067, 0.933]に近く、post KLは0.00325。",
    "Source: notes/advisor/CURRENT.md; SOURCE_INDEX.md D-059; artifacts/.../server/layer2_trace.json",
  ]);
}

// Slide 5
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.paper;
  addTitle(slide, "8件のserver seedを評価し、最小距離のBを選択", "04");
  addText(slide, "目標  μ_post=[normal 0.067, risk 0.933]", { left: 64, top: 126, width: 600, height: 30 }, {
    fontSize: 22,
    bold: true,
    color: COLORS.teal,
  });
  addText(slide, "表示は8件中4件", { left: 1000, top: 130, width: 215, height: 24 }, {
    fontSize: 17,
    color: COLORS.gray,
    alignment: "right",
  });

  const candidates = [
    { key: "A", risk: 0.501, kl: 0.4463, rank: 8, note: "目標と離れる", color: COLORS.gray },
    { key: "B", risk: 0.911, kl: 0.0033, rank: 1, note: "最小距離", color: COLORS.teal },
    { key: "C", risk: 0.901, kl: 0.0065, rank: 3, note: "近い", color: COLORS.blue },
    { key: "D", risk: 0.856, kl: 0.0293, rank: 4, note: "やや離れる", color: COLORS.amber },
  ];
  const xs = [64, 363, 662, 961];
  for (let i = 0; i < candidates.length; i += 1) {
    const c = candidates[i];
    const x = xs[i];
    addText(slide, `候補 ${c.key}`, { left: x, top: 175, width: 260, height: 36 }, {
      fontSize: 27,
      bold: true,
      color: c.color,
    });
    if (c.key === "B") {
      addBox(slide, { left: x - 10, top: 164, width: 282, height: 450 }, {
        fill: "none",
        line: COLORS.teal,
        lineWidth: 4,
        geometry: "roundRect",
      });
    }
    await addImage(slide, seedImages[c.key], { left: x, top: 220, width: 260, height: 146 }, {
      alt: `server seed candidate ${c.key}`,
      fit: "cover",
    });
    addProbabilityBar(slide, x, 418, 260, 1 - c.risk, c.risk, { height: 24, fontSize: 16 });
    addMetricBand(slide, x, 472, "post KL", c.kl.toFixed(4), c.color, 260);
    addMetricBand(slide, x, 512, "順位", `${c.rank} / 8`, c.color, 260);
    addText(slide, c.note, { left: x, top: 563, width: 260, height: 30 }, {
      fontSize: 20,
      bold: c.key === "B",
      color: c.color,
      alignment: "center",
    });
  }
  addText(slide, "Bを次のtransformationへ渡す", { left: 335, top: 630, width: 340, height: 28 }, {
    fontSize: 20,
    bold: true,
    color: COLORS.teal,
    alignment: "center",
  });
  addFooter(slide, "選択基準：updated global surrogateの出力とμ_postのpost KLが最小", 5);
  setNotes(slide, [
    "server seedはclientへ配布しないcloud-only data。このrunでは8件を評価し、説明用に4件を表示した。",
    "各seedをupdated global surrogateへ入力し、μ_postとのpost KLを計算した。Bの0.0033が最小だったため、Bをretrieved seedとした。",
    "retrievalは画像の見た目やground-truth labelを直接使って選ぶ処理ではない。選択基準はモデル出力の一致度。",
    "候補Cも近いが、Bの方が目標分布へのKLが小さい。",
    "この段階ではdeltaをまだ使わない。deltaは次のproxy objectiveで使う。",
    "Source: artifacts/runs/fedpact_nexar_example/run_20260910_002139_presentation_case/server/layer2_trace.json",
  ]);
}

// Slide 6
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.paper;
  addTitle(slide, "Bを4通りに変換し、判断の再現誤差を比較", "05");
  addText(slide, "同じ4フレームへ各変換を独立に適用", { left: 64, top: 126, width: 500, height: 30 }, {
    fontSize: 22,
    bold: true,
    color: COLORS.blue,
  });
  addText(slide, "変換は累積しない", { left: 1000, top: 130, width: 215, height: 24 }, {
    fontSize: 17,
    color: COLORS.risk,
    alignment: "right",
  });

  const variants = [
    {
      key: "identity", title: "無変換", detail: "Bをそのまま使用",
      post: 0.003251, delta: 0.000035, total: 0.003287, color: COLORS.gray,
    },
    {
      key: "dark", title: "明るさ ×0.75", detail: "全フレームを25%暗く",
      post: 0.010856, delta: 0.001294, total: 0.012150, color: COLORS.amber,
    },
    {
      key: "contrast", title: "contrast ×1.20", detail: "明暗差を20%強調",
      post: 0.005854, delta: 0.000296, total: 0.006150, color: COLORS.blue,
    },
    {
      key: "crop", title: "center crop 0.85", detail: "中央85%を切り出して拡大",
      post: 0.002700, delta: 0.000554, total: 0.003254, color: COLORS.teal, selected: true,
    },
  ];
  const xs = [64, 363, 662, 961];
  for (let i = 0; i < variants.length; i += 1) {
    const v = variants[i];
    const x = xs[i];
    if (v.selected) {
      addBox(slide, { left: x - 10, top: 165, width: 282, height: 454 }, {
        fill: COLORS.tealSoft,
        line: COLORS.teal,
        lineWidth: 4,
        geometry: "roundRect",
      });
    }
    addText(slide, v.title, { left: x, top: 178, width: 260, height: 34 }, {
      fontSize: 24,
      bold: true,
      color: v.color,
      alignment: "center",
    });
    await addImage(slide, transformImages[v.key], { left: x, top: 226, width: 260, height: 146 }, {
      alt: `transformation candidate ${v.title}`,
      fit: "cover",
    });
    addText(slide, v.detail, { left: x, top: 386, width: 260, height: 28 }, {
      fontSize: 17,
      color: COLORS.muted,
      alignment: "center",
    });
    addMetricBand(slide, x, 435, "post KL", v.post.toFixed(5), v.color, 260);
    addMetricBand(slide, x, 475, "delta MSE", v.delta.toFixed(5), v.color, 260);
    addLine(slide, x, 516, 260, COLORS.line, 1);
    addMetricBand(slide, x, 536, "L_match", v.total.toFixed(5), v.color, 260);
    if (v.selected) {
      addText(slide, "選択", { left: x, top: 583, width: 260, height: 26 }, {
        fontSize: 21,
        bold: true,
        color: COLORS.teal,
        alignment: "center",
      });
    }
  }
  addText(slide, "L_match = post判断のずれ + 判断変化のずれ   小さいほど端末の適応挙動に近い", { left: 150, top: 640, width: 980, height: 28 }, {
    fontSize: 21,
    bold: true,
    color: COLORS.ink,
    alignment: "center",
  });
  addFooter(slide, "10変換を評価。スライドは説明しやすい4候補を表示", 6);
  setNotes(slide, [
    "retrieved seed Bから、無変換、明るさ、contrast、cropなど10候補を作った。スライドには4候補を示す。",
    "各変換はBへ独立に適用する。左から順に画像を加工したわけではない。",
    "post KLは学習後判断の一致、delta MSEはclientのΔμ_localとproxy上のglobal surrogate更新前後差の一致を測る。",
    "center crop 0.85はpost KLを0.003251から0.002700へ下げた。一方でdelta MSEは無変換より増えた。両者の和L_matchは0.003254で最小になった。",
    "無変換との差は小さいため、この1例からtransformationの有効性は主張しない。ここでは選択過程と評価量を説明する。",
    "今回のL_matchはλ_post=λ_delta=1、L_reg未導入。",
    "Source: artifacts/runs/fedpact_nexar_example/run_20260910_002139_presentation_case/server/layer2_trace.json",
  ]);
}

// Slide 7
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.paper;
  addTitle(slide, "最終proxyと端末目標をpairにしてLayer 3へ渡す", "06");
  addStageLabel(slide, "SERVER", 64, 126, COLORS.blue);
  const w = 218;
  const h = 123;
  for (let i = 0; i < proxyFrames.length; i += 1) {
    await addImage(slide, proxyFrames[i], { left: 64 + i * (w + 8), top: 185, width: w, height: h }, {
      alt: `selected proxy frame ${i + 1}`,
      fit: "cover",
    });
  }
  addText(slide, "x̃：center crop 0.85を適用した4フレーム", { left: 64, top: 323, width: 520, height: 28 }, {
    fontSize: 20,
    color: COLORS.teal,
    bold: true,
  });

  addText(slide, "端末の目標  μ_post", { left: 64, top: 398, width: 320, height: 32 }, {
    fontSize: 24,
    bold: true,
  });
  addProbabilityBar(slide, 64, 472, 410, 0.067, 0.933, { height: 30, fontSize: 18 });

  addText(slide, "proxy上のupdated surrogate出力", { left: 535, top: 398, width: 420, height: 32 }, {
    fontSize: 24,
    bold: true,
  });
  addProbabilityBar(slide, 535, 472, 410, 0.087, 0.913, { height: 30, fontSize: 18 });
  addText(slide, "proxy上のrisk変化 +0.224\n目標のrisk変化 +0.248", { left: 1010, top: 420, width: 205, height: 78 }, {
    fontSize: 20,
    color: COLORS.muted,
    alignment: "center",
    lineSpacing: 1.12,
  });

  const pair = addBox(slide, { left: 170, top: 555, width: 600, height: 98 }, {
    fill: COLORS.lavender,
    line: "#7A5AA6",
    lineWidth: 2,
    geometry: "roundRect",
  });
  addText(slide, "P = ( x̃ , μ_post )", { left: 205, top: 581, width: 530, height: 42 }, {
    fontSize: 34,
    bold: true,
    color: "#563B7B",
    alignment: "center",
  });
  const foundation = addBox(slide, { left: 900, top: 555, width: 285, height: 98 }, {
    fill: COLORS.blueSoft,
    line: COLORS.blue,
    lineWidth: 2,
    geometry: "roundRect",
  });
  addText(slide, "Layer 3へ\nfoundation LoRA蒸留用", { left: 920, top: 570, width: 245, height: 66 }, {
    fontSize: 20,
    bold: true,
    color: COLORS.blue,
    alignment: "center",
  });
  addArrow(slide, pair, foundation, { color: "#7A5AA6", width: 3 });
  addFooter(slide, "teacherはμ_post。proxy上のsurrogate出力をteacher labelにはしない", 7);
  setNotes(slide, [
    "最終出力はproxy画像x_tildeと端末のpost prototype μ_postのpair。",
    "この例でproxy上のupdated surrogateはrisk 0.913。端末目標risk 0.933に近い。",
    "proxy上で測ったglobal surrogateのrisk変化は+0.224、client local deltaの目標は+0.248。",
    "Layer 3のteacherは端末から届いたμ_post。proxy上のglobal surrogate出力をteacher labelへ置き換えない。",
    "このslideはLayer 2からLayer 3への受渡しを説明する。foundation性能の改善結果を示すslideではない。",
    "Source: notes/advisor/SOURCE_INDEX.md D-029, D-049, D-062; run summary.json; server/proxy_pair.json",
  ]);
}

// Slide 8 Appendix
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.paper;
  addTitle(slide, "補足：retrievalとproxy optimizationの評価量", "A1");

  addStageLabel(slide, "Retrieval", 64, 130, COLORS.blue);
  addText(slide, "seed候補の順位付け", { left: 64, top: 184, width: 500, height: 34 }, {
    fontSize: 27,
    bold: true,
  });
  addBox(slide, { left: 64, top: 240, width: 530, height: 100 }, {
    fill: COLORS.blueSoft,
    line: COLORS.blue,
    lineWidth: 2,
    geometry: "roundRect",
  });
  addText(slide, "D_KL( μ_post || p_G^after(x_seed) )", { left: 92, top: 272, width: 474, height: 40 }, {
    fontSize: 27,
    bold: true,
    alignment: "center",
  });
  addText(slide, "updated global surrogateのpost出力だけで検索", { left: 64, top: 365, width: 530, height: 32 }, {
    fontSize: 21,
    color: COLORS.muted,
  });

  addBox(slide, { left: 632, top: 130, width: 3, height: 430 }, { fill: COLORS.line, line: COLORS.line, lineWidth: 0 });

  addStageLabel(slide, "Optimization", 675, 130, COLORS.risk);
  addText(slide, "変換候補の再現誤差", { left: 675, top: 184, width: 540, height: 34 }, {
    fontSize: 27,
    bold: true,
  });
  addBox(slide, { left: 675, top: 240, width: 540, height: 148 }, {
    fill: COLORS.riskSoft,
    line: COLORS.risk,
    lineWidth: 2,
    geometry: "roundRect",
  });
  addText(slide, "L_match = λ_post D_KL(μ_post || p_after(x̃))\n+ λ_Δ MSE(Δμ_local, p_after(x̃) − p_before(x̃))", { left: 700, top: 258, width: 490, height: 110 }, {
    fontSize: 20,
    bold: true,
    alignment: "center",
    lineSpacing: 1.1,
  });
  addText(slide, "clientのdeltaとproxy上のglobal round差を比較", { left: 675, top: 405, width: 540, height: 32 }, {
    fontSize: 21,
    color: COLORS.muted,
  });

  addText(slide, "重要な区別", { left: 64, top: 470, width: 200, height: 32 }, {
    fontSize: 25,
    bold: true,
    color: COLORS.risk,
  });
  addText(slide, "Δμ_local", { left: 64, top: 525, width: 160, height: 30 }, {
    fontSize: 22,
    bold: true,
    color: COLORS.teal,
  });
  addText(slide, "client固有の局所適応前後差", { left: 220, top: 525, width: 360, height: 30 }, {
    fontSize: 21,
  });
  addText(slide, "p_after − p_before", { left: 675, top: 525, width: 220, height: 30 }, {
    fontSize: 22,
    bold: true,
    color: COLORS.blue,
  });
  addText(slide, "同一proxy上で測るglobal surrogate更新前後差", { left: 900, top: 525, width: 315, height: 54 }, {
    fontSize: 20,
  });
  addText(slide, "現在の診断run：λ_post=1、λ_Δ=1、L_reg未導入。Qinvの具体式とthresholdはscreening待ち", { left: 64, top: 616, width: 1150, height: 34 }, {
    fontSize: 20,
    bold: true,
    color: COLORS.gray,
  });
  addFooter(slide, "L_proxy = L_match + L_reg。seed逸脱・自然性・時間的一貫性は後続screeningで固定", 8);
  setNotes(slide, [
    "retrievalとoptimizationで評価量が違う。retrievalはupdated global surrogateのpost出力とμ_postのKLだけ。",
    "transformation後はpost KLに加えてdelta MSEも使う。",
    "Δμ_localはclient固有の局所適応差。p_after-p_beforeはproxy上で測るglobal surrogateのround前後差。同一の量ではない。",
    "現在の診断runではlambdaを1に置き、regularizationは未導入。Qinvの具体式とthresholdも未固定。",
    "教授から細部を聞かれた場合は、このslideで実装済み部分とscreening待ちを分けて説明する。",
    "Source: notes/advisor/CURRENT.md; SOURCE_INDEX.md D-043, D-049, D-050; FedPACT実験計画26060901.pdf 3-4ページ",
  ]);
}

// Slide 9 Appendix
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.paper;
  addTitle(slide, "補足：説明用runの条件", "A2");

  const leftX = 64;
  const rightX = 672;
  addText(slide, "実行条件", { left: leftX, top: 135, width: 500, height: 36 }, {
    fontSize: 28,
    bold: true,
    color: COLORS.blue,
  });
  const conditions = [
    ["surrogate", "Qwen2.5-VL-3B-Instruct"],
    ["client", "1 client  局所学習3 step"],
    ["private prototype", "risk 3件のclass-wise mean"],
    ["server seed", "8件  clientとsample非重複"],
    ["入力", "各サンプル4フレーム"],
    ["変換", "10候補からL_match最小を選択"],
  ];
  for (let i = 0; i < conditions.length; i += 1) {
    const y = 195 + i * 56;
    addText(slide, conditions[i][0], { left: leftX, top: y, width: 190, height: 28 }, {
      fontSize: 19,
      bold: true,
      color: COLORS.muted,
    });
    addText(slide, conditions[i][1], { left: leftX + 200, top: y, width: 370, height: 28 }, {
      fontSize: 20,
      color: COLORS.ink,
    });
    addLine(slide, leftX, y + 37, 560, COLORS.faint, 1);
  }

  addText(slide, "この例から言えること", { left: rightX, top: 135, width: 500, height: 36 }, {
    fontSize: 28,
    bold: true,
    color: COLORS.teal,
  });
  addText(slide, "Layer 2の入出力が実データで接続した", { left: rightX, top: 202, width: 520, height: 34 }, {
    fontSize: 23,
    bold: true,
  });
  addText(slide, "seed候補と変換候補を同じ基準で追跡できる", { left: rightX, top: 255, width: 520, height: 34 }, {
    fontSize: 23,
    bold: true,
  });
  addText(slide, "private ID・raw frameがserver payloadへ入っていない", { left: rightX, top: 308, width: 520, height: 34 }, {
    fontSize: 23,
    bold: true,
  });

  addText(slide, "まだ言えないこと", { left: rightX, top: 395, width: 500, height: 36 }, {
    fontSize: 28,
    bold: true,
    color: COLORS.risk,
  });
  addText(slide, "transformationが一貫してproxy fidelityを改善するか", { left: rightX, top: 462, width: 520, height: 34 }, {
    fontSize: 22,
    color: COLORS.ink,
  });
  addText(slide, "proxyがfoundationのF1・AUCを改善するか", { left: rightX, top: 515, width: 520, height: 34 }, {
    fontSize: 22,
    color: COLORS.ink,
  });
  addText(slide, "複数clientのFedAvg後でも同じ再現度を得られるか", { left: rightX, top: 556, width: 520, height: 50 }, {
    fontSize: 20,
    color: COLORS.ink,
  });
  addBox(slide, { left: 64, top: 617, width: 1150, height: 40 }, {
    fill: COLORS.amberSoft,
    line: COLORS.amber,
    lineWidth: 1,
    geometry: "roundRect",
  });
  addText(slide, "位置づけ：処理説明と実装診断のためのcase study。性能の有効性を示す評価実験ではない", { left: 82, top: 626, width: 1115, height: 24 }, {
    fontSize: 19,
    bold: true,
    color: "#7A4A00",
    alignment: "center",
  });
  addFooter(slide, "run_20260910_002139_presentation_case  seed=42", 9);
  setNotes(slide, [
    "このslideは質疑用。今回の数値を性能結果として過大解釈しないために条件と範囲を明示する。",
    "単一clientなので、このrunではFedAvg後global surrogateがそのclientのlocal after stateと一致する。",
    "seed数8、変換数10は説明用診断の規模。本評価ではscreeningで候補集合とlambdaを固定し、複数runで確認する。",
    "client private、server seed、evaluation-onlyのID重複は0。server payload内のprivate sample IDも0件。",
    "Source: artifacts/runs/fedpact_nexar_example/run_20260910_002139_presentation_case/summary.json; client_local/data_splits.json; client_upload/server_payload.json",
  ]);
}
}

const fmt = (value, digits = 3) => Number(value).toFixed(digits);
const fmtSigned = (value, digits = 3) => `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(digits)}`;

// Slide 1 — title
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.navy;
  await addImage(slide, frame("a8adfca63e", 7), { left: 650, top: 0, width: 630, height: 720 }, {
    alt: "NEXAR評価用risk動画の最終フレーム。夜間交差点で左右から車両が接近している",
    fit: "cover",
  });
  addBox(slide, { left: 0, top: 0, width: 650, height: 720 }, { fill: COLORS.navy, line: COLORS.navy, lineWidth: 0 });
  addBox(slide, { left: 594, top: 0, width: 56, height: 720 }, { fill: COLORS.teal, line: COLORS.teal, lineWidth: 0 });
  addText(slide, "FedPACT / NEXAR concrete case", { left: 66, top: 62, width: 490, height: 34 }, {
    fontSize: 20, bold: true, color: "#84D2C5",
  });
  addText(slide, "Layer 2 動作確認と\n局所更新の\n順序依存性", { left: 66, top: 128, width: 500, height: 245 }, {
    fontSize: 47, bold: true, color: COLORS.white, lineSpacing: 0.98,
  });
  addText(slide, "実動画の反応変化から、\n代理データ候補へ至る過程を追う", { left: 70, top: 410, width: 470, height: 78 }, {
    fontSize: 25, color: "#DCE7EE", lineSpacing: 1.12,
  });
  addText(slide, "研究会説明用・診断実験", { left: 70, top: 584, width: 360, height: 26 }, {
    fontSize: 18, color: "#A9BCC9",
  });
  addText(slide, "YOSHIMURA Aoi   2026.09.13", { left: 70, top: 635, width: 420, height: 26 }, {
    fontSize: 18, color: "#DCE7EE",
  });
  setNotes(slide, [
    "つかみ：現場の生映像をcloudへ集めず、局所適応で得た挙動だけを代理データへ変換したい。",
    "ただし、代理化の入力となる局所更新自体が不安定なら、Layer 2が忠実でも有用な知識転移にはならない。",
    "今回はNEXARの具体的な動画と実測値を使い、Layer 2の経路と、その前段で見つかった問題を説明する。",
    "Source: notes/advisor/CURRENT.md; docs/fedpact_nexar_concrete_case_20260913.md",
  ]);
}

// Slide 2 — concrete hook
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.paper;
  addTitle(slide, "同じ局所学習で、risk動画もnormal動画もrisk確率が下がった", "01");
  addText(slide, "評価用動画6本のうちrisk／normal各1本を例示。映像内容の違いより、共通方向の出力移動が目立つ。", {
    left: 64, top: 120, width: 1140, height: 32,
  }, { fontSize: 21, color: COLORS.muted });

  const examples = [
    { id: "a8adfca63e", label: "risk動画 / 評価用", record: riskEvalTrace.samples.find((r) => r.sample_id === "a8adfca63e"), x: 64, color: COLORS.risk },
    { id: "1db89ecde7", label: "normal動画 / 評価用", record: riskEvalTrace.samples.find((r) => r.sample_id === "1db89ecde7"), x: 650, color: COLORS.blue },
  ];
  for (const example of examples) {
    addText(slide, example.label, { left: example.x, top: 170, width: 310, height: 28 }, {
      fontSize: 20, bold: true, color: example.color,
    });
    for (let i = 0; i < 4; i += 1) {
      await addImage(slide, frame(example.id, i + 4), { left: example.x + i * 135, top: 208, width: 128, height: 112 }, {
        alt: `${example.label} ${i + 1}/4`, fit: "cover",
      });
    }
    const pre = probability(example.record, "pre");
    const post = probability(example.record, "post");
    addProbabilityBar(slide, example.x, 378, 540, 1 - pre, pre, { height: 23 });
    addText(slide, "局所学習前", { left: example.x, top: 334, width: 130, height: 24 }, { fontSize: 16, color: COLORS.gray });
    addProbabilityBar(slide, example.x, 457, 540, 1 - post, post, { height: 23 });
    addText(slide, "局所学習後", { left: example.x, top: 413, width: 130, height: 24 }, { fontSize: 16, color: COLORS.gray });
    addText(slide, `Δq(risk) = ${fmtSigned(post - pre)}`, { left: example.x, top: 495, width: 540, height: 40 }, {
      fontSize: 28, bold: true, color: post - pre < 0 ? COLORS.blue : COLORS.risk, alignment: "center",
    });
  }
  addBox(slide, { left: 64, top: 566, width: 1126, height: 72 }, {
    fill: COLORS.amberSoft, line: COLORS.amber, lineWidth: 1.2, geometry: "roundRect",
  });
  addText(slide, "この時点で「risk知識を学べた」とは言えない。まず局所更新の健全性を診断する必要がある。", {
    left: 86, top: 586, width: 1082, height: 34,
  }, { fontSize: 23, bold: true, color: "#7A4A00", alignment: "center" });
  addFooter(slide, "評価用動画の例 / order seed 101", 2);
  setNotes(slide, [
    "左はpositiveラベルの評価用動画a8adfca63e、右はnegativeラベルの評価用動画1db89ecde7。どちらも局所学習、prototype作成、server検索には未使用。",
    `左はq(risk) ${fmt(probability(examples[0].record, "pre"))}→${fmt(probability(examples[0].record, "post"))}、右は${fmt(probability(examples[1].record, "pre"))}→${fmt(probability(examples[1].record, "post"))}。`,
    "単一フレームの見た目から因果特徴を断定しない。ここで言えるのは出力確率の変化だけ。",
    "Source: artifacts/runs/fedpact_nexar_example/run_20260913_balanced_order101_risk_fullgrid/client_local/eval_trace.json; artifacts/datasets/frames/train/<sample_id>",
  ]);
}

// Slide 3 — design and scope
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.paper;
  addTitle(slide, "固定したデータ分割と、今回動かした範囲", "02");
  const columns = [
    { x: 64, w: 480, title: "CLIENT PRIVATE TRAIN", count: "6本", color: COLORS.teal, fill: COLORS.tealSoft },
    { x: 570, w: 270, title: "CLIENT 評価用", count: "6本", color: COLORS.blue, fill: COLORS.blueSoft },
    { x: 866, w: 350, title: "SERVER SEED", count: "8本", color: COLORS.amber, fill: COLORS.amberSoft },
  ];
  for (const col of columns) {
    addBox(slide, { left: col.x, top: 136, width: col.w, height: 382 }, { fill: col.fill, line: col.color, lineWidth: 1.2, geometry: "roundRect" });
    addText(slide, col.title, { left: col.x + 18, top: 154, width: col.w - 118, height: 30 }, { fontSize: col.w < 300 ? 15 : 18, bold: true, color: col.color });
    addText(slide, col.count, { left: col.x + col.w - 100, top: 150, width: 80, height: 36 }, { fontSize: 28, bold: true, color: col.color, alignment: "right" });
  }
  const trainRiskIds = ["d626422e71", "95c3bf1ba2", "06e20a0fd0"];
  const trainNormalIds = ["f00eb9a8eb", "7cc5b04204", "b9f2f42190"];
  for (let i = 0; i < 3; i += 1) {
    await addImage(slide, frame(trainRiskIds[i], 7), { left: 82 + i * 149, top: 210, width: 138, height: 100 }, { alt: `private risk ${i + 1}`, fit: "cover" });
    await addImage(slide, frame(trainNormalIds[i], 7), { left: 82 + i * 149, top: 356, width: 138, height: 100 }, { alt: `private normal ${i + 1}`, fit: "cover" });
  }
  addText(slide, "risk ×3", { left: 82, top: 188, width: 120, height: 22 }, { fontSize: 16, bold: true, color: COLORS.risk });
  addText(slide, "normal ×3", { left: 82, top: 334, width: 120, height: 22 }, { fontSize: 16, bold: true, color: COLORS.blue });
  await addImage(slide, frame("a8adfca63e", 7), { left: 590, top: 220, width: 110, height: 90 }, { alt: "評価用risk動画の例", fit: "cover" });
  await addImage(slide, frame("1db89ecde7", 7), { left: 712, top: 220, width: 110, height: 90 }, { alt: "評価用normal動画の例", fit: "cover" });
  addText(slide, "risk ×3 / normal ×3\n学習・prototype・\nserver検索に未使用", { left: 590, top: 338, width: 232, height: 100 }, {
    fontSize: 18, bold: true, color: COLORS.ink, alignment: "center", lineSpacing: 1.12,
  });
  const seedIds = ["56b6d7fa8c", "a34d7d0109", "c393761248", "4b8432de81"];
  for (let i = 0; i < 4; i += 1) {
    await addImage(slide, frame(seedIds[i], 7), { left: 885 + (i % 2) * 154, top: 214 + Math.floor(i / 2) * 128, width: 142, height: 104 }, { alt: `server seed ${i + 1}`, fit: "cover" });
  }
  addText(slide, "risk ×4 / normal ×4\nclientへ配布しないcloud-only pool", { left: 886, top: 456, width: 310, height: 56 }, {
    fontSize: 16, bold: true, color: COLORS.ink, alignment: "center",
  });
  addBox(slide, { left: 64, top: 548, width: 1152, height: 90 }, { fill: COLORS.white, line: COLORS.line, lineWidth: 1, geometry: "roundRect" });
  addText(slide, "Qwen2.5-VL-3B / fixed checkpoint  |  split seed 42  |  LoRA + classification head更新", { left: 84, top: 563, width: 1110, height: 28 }, {
    fontSize: 20, bold: true, color: COLORS.ink, alignment: "center",
  });
  addText(slide, "Step I-A → I-B  |  q・μᵖᵒˢᵗ = softmax確率、Δq・Δμ = その差  |  λ_post = λ_Δ = 1", { left: 84, top: 602, width: 1110, height: 26 }, {
    fontSize: 18, color: COLORS.muted, alignment: "center",
  });
  addFooter(slide, "固定split・ID重複なし・private raw data非送信", 3);
  setNotes(slide, [
    "今回の該当範囲はFedPACT実験計画の第I段階、Step I-AとStep I-Bの縦切り確認。三層3〜5 round接続やLayer 3更新は未実施。",
    "client学習用、client評価用、server seedは同一split seed 42で固定し、ID重複なし。評価用動画は局所学習、prototype作成、server検索に使わない。",
    "今回のserver payloadにはtrainable update（client LoRA＋classification head）とclass-wise μ_post、Δμ、π、class_countを含める。private raw frameとsample IDは含めない。",
    "PDFの基本protocolで用いる重みはπ。class_countは今回の実装schemaに併記した監査用fieldで、現runではπと同値。",
    "Source: notes/advisor/sources/raw/2026-09-01/FedPACT実験計画260901.pdf pp.3-4; artifacts/runs/fedpact_nexar_example/run_20260913_balanced_order101_risk_fullgrid/client_local/data_splits.json; client_upload/server_payload.json",
  ]);
}

// Slide 4 — per-sample slopes
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.paper;
  addTitle(slide, "order 101では、12本すべてがnormal側へ移動した", "03");
  addSlopePanel(slide, riskTrainTrace.private_samples, { left: 64, top: 135, width: 540, height: 430 }, "private train 6本");
  addSlopePanel(slide, riskEvalTrace.samples, { left: 668, top: 135, width: 540, height: 430 }, "評価用動画 6本");
  addBox(slide, { left: 64, top: 574, width: 1144, height: 64 }, { fill: COLORS.white, line: COLORS.line, lineWidth: 1, geometry: "roundRect" });
  addText(slide, "平均Δq(risk)", { left: 90, top: 592, width: 160, height: 26 }, { fontSize: 18, bold: true, color: COLORS.muted });
  addText(slide, "train normal −0.139\ntrain risk −0.090", { left: 270, top: 584, width: 350, height: 48 }, { fontSize: 18, bold: true, color: COLORS.blue, lineSpacing: 1.05 });
  addText(slide, "評価用 normal −0.101\n評価用 risk −0.130", { left: 690, top: 584, width: 420, height: 48 }, { fontSize: 18, bold: true, color: COLORS.blue, lineSpacing: 1.05 });
  addText(slide, "risk label", { left: 510, top: 118, width: 90, height: 20 }, { fontSize: 14, bold: true, color: COLORS.risk, alignment: "right" });
  addText(slide, "normal label", { left: 1096, top: 118, width: 112, height: 20 }, { fontSize: 14, bold: true, color: COLORS.blue, alignment: "right" });
  addFooter(slide, "赤=risk label　青=normal label　縦軸=q(risk)", 4);
  setNotes(slide, [
    "個別sampleをすべて表示し、平均だけの印象操作を避ける。order 101では学習用6本・評価用6本がすべてnormal方向へ移動した。",
    "ただし、この1順序だけでクラス非選択性を断定しない。次slideで学習順を変えた結果を示す。",
    "Source: artifacts/runs/fedpact_nexar_example/run_20260913_balanced_order101_risk_fullgrid/client_local/private_trace.json; client_local/eval_trace.json",
  ]);
}

// Slide 5 — order sensitivity and full-batch control
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.paper;
  addTitle(slide, "学習順だけでΔqの符号が反転した", "04");
  const statRows = [
    ["train normal", 0], ["train risk", 1], ["評価用 normal", 0], ["評価用 risk", 1],
  ];
  const avgRows = statRows.map(([name, label], rowIndex) => {
    const splitKey = rowIndex < 2 ? "train" : "eval";
    const pre = meanRisk(orderTraces[0][splitKey], label, "pre");
    const cells = [name, fmt(pre)];
    for (const item of orderTraces) {
      const post = meanRisk(item[splitKey], label, "post");
      cells.push(`${fmt(post)} (${fmtSigned(post - pre)})`);
    }
    const fullPost = meanRisk(fullBatchTrace[splitKey], label, "post");
    cells.push(`${fmt(fullPost)} (${fmtSigned(fullPost - pre)})`);
    return cells;
  });
  addMiniTable(slide,
    ["平均q(risk)", "pre", "order 101", "order 202", "order 303", "full-batch"],
    avgRows,
    { left: 64, top: 145, width: 1152, height: 275 },
    [190, 110, 210, 210, 210, 222],
  );
  const gapRows = ["train", "評価用"].map((name, index) => {
    const splitKey = index === 0 ? "train" : "eval";
    const preGap = meanRisk(orderTraces[0][splitKey], 1, "pre") - meanRisk(orderTraces[0][splitKey], 0, "pre");
    const posts = orderTraces.map((item) => meanRisk(item[splitKey], 1, "post") - meanRisk(item[splitKey], 0, "post"));
    const fullGap = meanRisk(fullBatchTrace[splitKey], 1, "post") - meanRisk(fullBatchTrace[splitKey], 0, "post");
    return [name, fmt(preGap), ...posts.map((v) => fmt(v)), fmt(fullGap)];
  });
  addText(slide, "class間差 = mean q(risk | risk) − mean q(risk | normal)", { left: 64, top: 445, width: 650, height: 28 }, {
    fontSize: 18, bold: true, color: COLORS.muted,
  });
  addMiniTable(slide,
    ["class間差", "pre", "101", "202", "303", "full-batch"],
    gapRows,
    { left: 64, top: 480, width: 820, height: 126 },
    [180, 110, 130, 130, 130, 140],
  );
  addBox(slide, { left: 914, top: 466, width: 302, height: 156 }, { fill: COLORS.amberSoft, line: COLORS.amber, lineWidth: 1, geometry: "roundRect" });
  addText(slide, "診断", { left: 936, top: 483, width: 80, height: 25 }, { fontSize: 18, bold: true, color: "#7A4A00" });
  addText(slide, "batch 1：順序で符号反転\n6本同時：shift縮小\n原因は未分離", { left: 936, top: 514, width: 258, height: 92 }, {
    fontSize: 17, bold: true, color: COLORS.ink, lineSpacing: 1.08,
  });
  addFooter(slide, "同じsplit・checkpoint・sample exposure=6 / class間差は非丸め値から算出", 5);
  setNotes(slide, [
    "order 101は最初に詳細追跡用として固定し、202と303はその後に学習順感度を確認するため追加した。",
    "batch size 1ではq_postとΔqの符号が学習順で反転。評価用動画のclass間差も改善・悪化が一貫しない。",
    "batch size 1は6 optimizer steps、full-batch相当は6動画を同時に1 optimizer stepで更新した。総sample exposureは6に揃えたが、batch構成とoptimizer step数が交絡している。",
    "共通方向shiftが小さくなった事実は示せるが、batch化が原因とはまだ言えない。次実験でoptimizer step数を記録・統制して切り分ける。",
    "Source: run_20260913_balanced_order101_risk_fullgrid; run_20260913_balanced_order202_diagnostic; run_20260913_balanced_order303_diagnostic; run_20260913_balanced_fullbatch_diagnostic",
  ]);
}

// Slide 6 — prototype formation
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.paper;
  addTitle(slide, "risk 3本の反応を平均し、送信prototypeを作る", "05");
  const riskRecords = recordsFor(riskTrainTrace, 1);
  for (let i = 0; i < riskRecords.length; i += 1) {
    const record = riskRecords[i];
    const x = 64 + i * 270;
    await addImage(slide, frame(record.sample_id, 7), { left: x, top: 150, width: 244, height: 150 }, { alt: `private risk sample ${record.sample_id}`, fit: "cover" });
    addText(slide, record.sample_id, { left: x, top: 310, width: 244, height: 22 }, { fontSize: 15, color: COLORS.gray, alignment: "center" });
    addText(slide, `${fmt(probability(record, "pre"))} → ${fmt(probability(record, "post"))}`, { left: x, top: 342, width: 244, height: 32 }, {
      fontSize: 24, bold: true, color: COLORS.risk, alignment: "center",
    });
    addText(slide, `Δq(risk) ${fmtSigned(deltaRisk(record))}`, { left: x, top: 380, width: 244, height: 26 }, {
      fontSize: 18, bold: true, color: deltaRisk(record) < 0 ? COLORS.blue : COLORS.risk, alignment: "center",
    });
  }
  addBox(slide, { left: 892, top: 144, width: 324, height: 288 }, { fill: COLORS.tealSoft, line: COLORS.teal, lineWidth: 1.2, geometry: "roundRect" });
  addText(slide, "risk prototype / order 101", { left: 914, top: 166, width: 280, height: 28 }, { fontSize: 20, bold: true, color: COLORS.teal });
  addText(slide, "μᵖᵒˢᵗₖ,ᵣ = [0.404, 0.596]", { left: 914, top: 222, width: 280, height: 36 }, { fontSize: 25, bold: true, color: COLORS.ink });
  addText(slide, "Δμₖ,ᵣ = [+0.090, −0.090]", { left: 914, top: 278, width: 280, height: 36 }, { fontSize: 25, bold: true, color: COLORS.ink });
  addText(slide, "πₖ,ᵣ = 3", { left: 914, top: 336, width: 280, height: 34 }, { fontSize: 25, bold: true, color: COLORS.ink });
  addText(slide, "πは平均に寄与したrisk動画数", { left: 914, top: 384, width: 280, height: 24 }, { fontSize: 16, color: COLORS.muted });
  addBox(slide, { left: 64, top: 462, width: 1128, height: 78 }, { fill: COLORS.riskSoft, line: COLORS.risk, lineWidth: 1, geometry: "roundRect" });
  addText(slide, "注意：risk prototypeでもrisk確率は平均0.090低下。Layer 2は、この局所挙動の有用性までは保証しない。", {
    left: 86, top: 486, width: 1084, height: 34,
  }, { fontSize: 22, bold: true, color: "#8C2F22", alignment: "center" });
  const normalProto = normalSummary.prototype;
  addText(slide, `normal側も送信： μᵖᵒˢᵗ=[${normalProto.mu_post.map((v) => fmt(v)).join(", ")}]　Δμ=[${normalProto.delta_mu_local.map((v) => fmtSigned(v)).join(", ")}]　π=3`, {
    left: 76, top: 574, width: 1110, height: 32,
  }, { fontSize: 20, bold: true, color: COLORS.blue, alignment: "center" });
  addText(slide, "今回の送信：trainable update（client LoRA + classification head）+ μᵖᵒˢᵗ + Δμ + π + class count\n非送信：raw video・frame・sample ID・独立したμᵖʳᵉ", {
    left: 76, top: 607, width: 1110, height: 46,
  }, { fontSize: 15.5, color: COLORS.muted, alignment: "center", lineSpacing: 1.05 });
  addFooter(slide, "成分順：[normal, risk] / q・μᵖᵒˢᵗはsoftmax確率、Δq・Δμはその差", 6);
  setNotes(slide, [
    "3本それぞれのq_pre、q_post、Δqを先に示し、その平均がprototypeになることを明示する。",
    "左からd626..., 95c3..., 06e2...。数値はriskクラス確率。μ_postとΔμは2クラス確率ベクトル。",
    "π=3はrisk prototypeの平均に寄与したprivate sample数。−0.090だけを送るのではなく、確率和0を保つ2成分vector[+0.090,−0.090]として扱う。",
    "実装payloadはtrainable updateにclient LoRAとclassification headを含み、prototypeにはπ（frequency）とclass_countを併記する。class_countはPDFの基本protocolにはない実装上の監査field。",
    "Source: artifacts/runs/fedpact_nexar_example/run_20260913_balanced_order101_risk_fullgrid/client_local/private_trace.json; client_upload/server_payload.json; summary.json",
  ]);
}

// Slide 7 — retrieval versus full-grid selection
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.paper;
  addTitle(slide, "μᵖᵒˢᵗの検索1位は、L_matchの全体最小ではなかった", "06");
  addText(slide, "検索距離：Dₖₗ( μᵖᵒˢᵗₖ,ᵣ  ||  p_Gᵗ⁺¹(s) )　— seed画像そのものではなく、連合集約後モデルの反応を比較", {
    left: 64, top: 122, width: 1140, height: 34,
  }, { fontSize: 21, bold: true, color: COLORS.muted });

  const retrieval = riskLayer2.retrieval.candidates;
  const variants = riskLayer2.transformation_and_optimization.candidates;
  const bestForSeed = (seedId) => variants.filter((v) => v.seed_id === seedId).sort((a, b) => a.metrics.proxy_objective - b.metrics.proxy_objective)[0];
  const showRanks = [1, 2, 3, 6];
  for (let i = 0; i < showRanks.length; i += 1) {
    const item = retrieval.find((v) => v.rank === showRanks[i]);
    const best = bestForSeed(item.seed_id);
    const x = 64 + i * 286;
    const selected = item.seed_id === riskSummary.selected_proxy.seed_id;
    addBox(slide, { left: x, top: 176, width: 260, height: 344 }, {
      fill: COLORS.white, line: selected ? COLORS.teal : COLORS.line, lineWidth: selected ? 4 : 1, geometry: "roundRect",
    });
    await addImage(slide, frame(item.seed_id, 7), { left: x + 12, top: 190, width: 236, height: 150 }, { alt: `server seed rank ${item.rank}`, fit: "cover" });
    addText(slide, `retrieval rank ${item.rank}`, { left: x + 16, top: 352, width: 228, height: 26 }, {
      fontSize: 19, bold: true, color: selected ? COLORS.teal : COLORS.blue, alignment: "center",
    });
    addText(slide, `Dₖₗ = ${fmt(item.retrieval_post_kl, 6)}`, { left: x + 16, top: 389, width: 228, height: 24 }, { fontSize: 17, color: COLORS.muted, alignment: "center" });
    const t = best.transformation.value == null ? best.transformation.name : `${best.transformation.name} ${best.transformation.value}`;
    addText(slide, `best transform\n${t}`, { left: x + 16, top: 426, width: 228, height: 46 }, { fontSize: 17, bold: true, color: COLORS.ink, alignment: "center" });
    addText(slide, `L_match(min)\n${fmt(best.metrics.proxy_objective, 6)}`, { left: x + 16, top: 468, width: 228, height: 48 }, {
      fontSize: 16, bold: true, color: selected ? COLORS.teal : COLORS.muted, alignment: "center", lineSpacing: 1.0,
    });
  }
  const rank1 = retrieval.find((v) => v.rank === 1);
  const rank1Best = bestForSeed(rank1.seed_id);
  addBox(slide, { left: 64, top: 548, width: 1144, height: 88 }, { fill: COLORS.amberSoft, line: COLORS.amber, lineWidth: 1, geometry: "roundRect" });
  addText(slide, `top-1内の最小 ${fmt(rank1Best.metrics.proxy_objective, 6)}　→　8×10全探索の最小 ${fmt(riskSummary.selected_proxy.metrics.proxy_objective, 6)}（rank 6 seed）`, {
    left: 88, top: 566, width: 1096, height: 32,
  }, { fontSize: 24, bold: true, color: "#7A4A00", alignment: "center" });
  addText(slide, "小規模な8 seedだからこそ、top-N比較より先に全探索で見落としを確認できる", { left: 88, top: 604, width: 1096, height: 24 }, {
    fontSize: 18, color: COLORS.muted, alignment: "center",
  });
  addFooter(slide, "risk prototype / 8 seed × 10 transformations = 80 candidates", 7);
  setNotes(slide, [
    "retrieval distanceはμ_postと連合集約後global surrogate M_G^{t+1}上のseed出力p_G^{t+1}(s)とのKL。画像空間の近さではなく、post behaviorの近さで順位付けする。",
    "M_G^tは連合集約前、M_G^{t+1}は連合集約後のglobal surrogate。今回は1 clientなのでM_G^{t+1}は局所学習後surrogateと一致する。",
    "全80候補を評価した結果、retrieval rank 6の4b8432de81 + center_crop(0.85)がL_match最小。",
    "この結果はtop-1 retrievalを採用すべきでない可能性を示すが、seed数8の診断に限定される。大規模候補集合での効率評価は未実施。",
    "Source: artifacts/runs/fedpact_nexar_example/run_20260913_balanced_order101_risk_fullgrid/server/layer2_trace.json; summary.json",
  ]);
}

// Slide 8 — transformations
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.paper;
  addTitle(slide, "rank 6 seedを変換し、postとdeltaの再現誤差を比較", "07");
  const variants = riskLayer2.transformation_and_optimization.candidates.filter((v) => v.seed_id === riskSummary.selected_proxy.seed_id);
  const picks = [
    ["identity", null], ["brightness", 0.75], ["contrast", 1.2], ["center_crop", 0.85], ["gaussian_blur", 0.75],
  ].map(([name, value]) => variants.find((v) => v.transformation.name === name && v.transformation.value === value));
  for (let i = 0; i < picks.length; i += 1) {
    const candidate = picks[i];
    const x = 48 + i * 245;
    const selected = candidate.transformation.name === "center_crop" && candidate.transformation.value === 0.85;
    const label = candidate.transformation.value == null ? candidate.transformation.name : `${candidate.transformation.name}\n${candidate.transformation.value}`;
    addBox(slide, { left: x, top: 148, width: 226, height: 390 }, {
      fill: selected ? COLORS.tealSoft : COLORS.white, line: selected ? COLORS.teal : COLORS.line, lineWidth: selected ? 4 : 1, geometry: "roundRect",
    });
    await addImage(slide, path.join(RISK_RUN_DIR, "server", candidate.frame_files[3]), { left: x + 10, top: 160, width: 206, height: 150 }, {
      alt: `seed変換候補 ${label}`, fit: "cover",
    });
    addText(slide, label, { left: x + 12, top: 322, width: 202, height: 48 }, { fontSize: 19, bold: true, color: selected ? COLORS.teal : COLORS.ink, alignment: "center" });
    addText(slide, `post KL  ${fmt(candidate.metrics.post_kl, 6)}\ndelta MSE  ${fmt(candidate.metrics.delta_mse, 6)}\nL_match  ${fmt(candidate.metrics.proxy_objective, 6)}`, {
      left: x + 14, top: 386, width: 198, height: 96,
    }, { fontSize: 15, color: COLORS.muted, lineSpacing: 1.18, alignment: "center" });
    if (selected) addText(slide, "数値上の最小", { left: x + 22, top: 497, width: 182, height: 24 }, { fontSize: 17, bold: true, color: COLORS.teal, alignment: "center" });
  }
  addText(slide, "L_match = KL( μᵖᵒˢᵗₖ,ᵣ || p_Gᵗ⁺¹(x̃ₖ,ᵣ) ) + MSE( Δμₖ,ᵣ , p_Gᵗ⁺¹(x̃ₖ,ᵣ) − p_Gᵗ(x̃ₖ,ᵣ) )", {
    left: 64, top: 548, width: 1152, height: 32,
  }, { fontSize: 18, bold: true, color: COLORS.ink, alignment: "center" });
  addText(slide, "M_Gᵗ = 連合集約前、M_Gᵗ⁺¹ = 連合集約後（今回は1 clientのため局所学習後surrogateと一致）", {
    left: 64, top: 584, width: 1152, height: 24,
  }, { fontSize: 15.5, color: COLORS.muted, alignment: "center" });
  addBox(slide, { left: 126, top: 616, width: 1028, height: 42 }, { fill: COLORS.riskSoft, line: COLORS.risk, lineWidth: 1, geometry: "roundRect" });
  addText(slide, "λ_post=λ_Δ=1、L_reg・Qinv未導入。数値最小でも意味的に妥当とはまだ言えない。", { left: 148, top: 626, width: 984, height: 24 }, {
    fontSize: 17, bold: true, color: "#8C2F22", alignment: "center",
  });
  addFooter(slide, `source seed ${riskSummary.selected_proxy.seed_id} / retrieval rank ${riskSummary.selected_proxy.retrieval_rank}`, 8);
  setNotes(slide, [
    "各候補は同じseed動画の全8フレームへ同一変換を適用している。ここでは説明用に最終フレームを表示。",
    "選択基準はλ_post=λ_delta=1のL_matchのみ。proxy側のdeltaはp_G^{t+1}(x_tilde)-p_G^t(x_tilde)であり、client側のq^{post}-q^{pre}とは記号を分ける。",
    "seed逸脱、自然性、時間一貫性、多様性のL_regとQinvは未導入。",
    "したがってcenter crop 0.85は、定義した数値目的に最も近い候補であって、意味的妥当性が保証された代理画像ではない。",
    "Source: artifacts/runs/fedpact_nexar_example/run_20260913_balanced_order101_risk_fullgrid/server/transformation_candidates; server/layer2_trace.json",
  ]);
}

// Slide 9 — both proxy pairs
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.paper;
  addTitle(slide, "risk / normal両prototypeでproxy pairまで生成した", "08");
  const cases = [
    { title: "risk prototype", x: 64, runDir: RISK_RUN_DIR, summary: riskSummary, color: COLORS.risk },
    { title: "normal prototype", x: 654, runDir: NORMAL_RUN_DIR, summary: normalSummary, color: COLORS.blue },
  ];
  for (const item of cases) {
    addText(slide, item.title, { left: item.x, top: 132, width: 520, height: 30 }, { fontSize: 24, bold: true, color: item.color });
    for (let i = 0; i < 3; i += 1) {
      await addImage(slide, path.join(item.runDir, "server/proxy_frames", `proxy_${String(i + 1).padStart(2, "0")}.png`), {
        left: item.x + i * 174, top: 178, width: 164, height: 126,
      }, { alt: `${item.title} proxy frame ${i + 2}`, fit: "cover" });
    }
    const p = item.summary.prototype;
    const s = item.summary.selected_proxy;
    addText(slide, `μᵖᵒˢᵗ = [${p.mu_post.map((v) => fmt(v)).join(", ")}]\nΔμ = [${p.delta_mu_local.map((v) => fmtSigned(v)).join(", ")}]　π=${p.frequency}`, {
      left: item.x, top: 326, width: 520, height: 68,
    }, { fontSize: 20, bold: true, color: COLORS.ink, alignment: "center", lineSpacing: 1.15 });
    addText(slide, `seed ${s.seed_id} / seed label=${s.seed_label}\n${s.transformation.name}(${s.transformation.value})\npost KL ${fmt(s.metrics.post_kl, 6)}　delta MSE ${fmt(s.metrics.delta_mse, 6)}\nL_match ${fmt(s.metrics.proxy_objective, 6)}`, {
      left: item.x, top: 414, width: 520, height: 104,
    }, { fontSize: 18, color: COLORS.muted, alignment: "center", lineSpacing: 1.14 });
  }
  addBox(slide, { left: 64, top: 542, width: 1120, height: 84 }, { fill: COLORS.amberSoft, line: COLORS.amber, lineWidth: 1, geometry: "roundRect" });
  addText(slide, "Pₖ,ᵣ = ( x̃ₖ,ᵣ , μᵖᵒˢᵗₖ,ᵣ ) をLayer 3へ渡せる", { left: 90, top: 557, width: 1070, height: 32 }, {
    fontSize: 26, bold: true, color: "#7A4A00", alignment: "center",
  });
  addText(slide, "ただしrisk側の数値最小候補はnormalラベルseed。L_matchだけでは意味的妥当性を保証できない。", { left: 90, top: 596, width: 1070, height: 24 }, {
    fontSize: 18, bold: true, color: "#8C2F22", alignment: "center",
  });
  addFooter(slide, "数値上のproxy candidate。foundation transferは未実施", 9);
  setNotes(slide, [
    "riskとnormalの双方でclass-wise prototypeと数値上の最小proxy候補を生成し、P=(x_tilde, μ_post)として保存した。",
    "risk側ではnormalラベルseed 4b8432de81が全80候補中のL_match最小。これはL_matchが意味的ラベル整合性を含まないことを示す。",
    "ここから、same-class retrieval制約やL_reg / quality screeningを今後検討する根拠が得られた。ただし具体方式は未確定。",
    "Source: run_20260913_balanced_order101_risk_fullgrid/server/proxy_pair.json; run_20260913_balanced_order101_normal_fullgrid/server/proxy_pair.json; summaries",
  ]);
}

// Slide 10 — conclusion and next experiment
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.paper;
  addTitle(slide, "実装上は経路が通った。研究上はStep I-Aの安定化が先", "09");
  addBox(slide, { left: 64, top: 142, width: 536, height: 290 }, { fill: COLORS.tealSoft, line: COLORS.teal, lineWidth: 1.2, geometry: "roundRect" });
  addText(slide, "確認できたこと", { left: 88, top: 164, width: 240, height: 32 }, { fontSize: 24, bold: true, color: COLORS.teal });
  addText(slide, "1　balanced private 6本でqᵖʳᵉ / qᵖᵒˢᵗ / Δqを保存\n2　risk / normalの両prototypeをprivate raw dataなしで送信\n3　8 seed × 10変換を全探索しproxy pairを生成\n4　top-1固定の見落としを実測", {
    left: 90, top: 215, width: 480, height: 184,
  }, { fontSize: 21, bold: true, color: COLORS.ink, lineSpacing: 1.22 });
  addBox(slide, { left: 630, top: 142, width: 586, height: 290 }, { fill: COLORS.riskSoft, line: COLORS.risk, lineWidth: 1.2, geometry: "roundRect" });
  addText(slide, "まだ解けていないこと", { left: 654, top: 164, width: 280, height: 32 }, { fontSize: 24, bold: true, color: COLORS.risk });
  addText(slide, "• batch size 1の局所更新は順序依存\n• 未見動画のclass分離改善は一貫しない\n• 数値最小proxyの意味的妥当性は未保証\n• L_reg / Qinv / 2-client / Layer 3 / 3〜5 roundは未実施", {
    left: 658, top: 215, width: 520, height: 176,
  }, { fontSize: 21, bold: true, color: COLORS.ink, lineSpacing: 1.25 });
  addText(slide, "次にやる実験", { left: 64, top: 466, width: 220, height: 32 }, { fontSize: 24, bold: true, color: COLORS.blue });
  const next = [
    ["1", "更新parameter・head状態・各更新量・全runのqᵖʳᵉ一致を監査"],
    ["2", "optimizer step数とsample exposureを記録し、層化mini-batch / gradient accumulationを比較"],
    ["3", "order 101 / 202 / 303で安定性確認後、2-client → Layer 3 → 3〜5 round"],
  ];
  for (let i = 0; i < next.length; i += 1) {
    addBox(slide, { left: 66, top: 512 + i * 50, width: 34, height: 34 }, { geometry: "ellipse", fill: COLORS.blue, line: COLORS.blue, lineWidth: 0 });
    addText(slide, next[i][0], { left: 66, top: 518 + i * 50, width: 34, height: 22 }, { fontSize: 17, bold: true, color: COLORS.white, alignment: "center" });
    addText(slide, next[i][1], { left: 118, top: 512 + i * 50, width: 1068, height: 36 }, { fontSize: 21, bold: true, color: COLORS.ink });
  }
  addFooter(slide, "Layer 2は入力された局所挙動を追う。入力信号の質は別途保証が必要", 10);
  setNotes(slide, [
    "結論を実装上と研究上に分ける。Layer 2の処理経路は通ったが、有用な知識転移を示してはいない。",
    "最初に更新parameter、classification headの状態、各parameterの更新量、全runのq_pre一致を監査する。",
    "その後、optimizer step数と総sample exposureを記録しながら層化mini-batchとgradient accumulationをscreeningする。multiple epochを先に増やさない。",
    "複数学習順で安定性を確認し、安定してから2-clientとLayer 3へ進む。",
    "Source: docs/fedpact_nexar_concrete_case_20260913.md; notes/advisor/sources/raw/2026-09-01/FedPACT実験計画260901.pdf",
  ]);
}

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const previewDir = path.join(TMP_DIR, "preview");
await fs.mkdir(previewDir, { recursive: true });
for (let i = 0; i < presentation.slides.items.length; i += 1) {
  const slide = presentation.slides.items[i];
  const png = await presentation.export({ slide, format: "png", scale: 1.5 });
  await fs.writeFile(path.join(previewDir, `slide-${String(i + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(previewDir, `slide-${String(i + 1).padStart(2, "0")}.layout.json`), await layout.text());
}
const montage = await presentation.export({ format: "png", montage: true, scale: 0.6 });
await fs.writeFile(path.join(previewDir, "montage.png"), new Uint8Array(await montage.arrayBuffer()));

const { finalizePresentation } = await import(pathToFileURL(
  path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs"),
).href);
const stagingDir = path.join(TMP_DIR, "finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const requirements = {
  explicitTotalSlideCount: 10,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
};
const fontPolicy = {
  basis: "design",
  families: [FONT],
};
const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  requiredNativeTableOwnerSlides: [],
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "FedPACT_NEXAR_具体例_研究会_20260914_v7.validation.json"),
});
console.log(JSON.stringify({ final: FINAL_PPTX, result }, null, 2));
