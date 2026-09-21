import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const workspace = "C:\\Python\\実験_修正\\tmp\\presentations\\progress_0904";
const starterPptx = `${workspace}\\template-starter.pptx`;
const finalPptx = "C:\\Users\\aoi7y\\OneDrive - 国立大学法人東海国立大学機構\\folder\\卒研_明光\\進捗会0904_吉村有生.pptx";
const previewDir = `${workspace}\\final-preview`;
const layoutDir = `${workspace}\\final-layout`;

const C = {
  navy: "#203864",
  text: "#202020",
  cloudFill: "#D9EAF7",
  cloudLine: "#4472C4",
  clientFill: "#FCE4D6",
  clientLine: "#ED7D31",
  dataFill: "#E2F0D9",
  dataLine: "#70AD47",
  keyFill: "#FFF2CC",
  keyLine: "#BF9000",
  futureFill: "#E7E6E6",
  futureLine: "#7F7F7F",
  white: "#FFFFFF",
  softBlue: "#F3F7FB",
  softOrange: "#FFF8F2",
};

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

function parseInspect(snapshot) {
  return snapshot.ndjson
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line));
}

function findAid(records, slide, kind, predicate, label) {
  const matches = records.filter(
    (record) => record.slide === slide && record.kind === kind && predicate(record),
  );
  if (matches.length !== 1) {
    throw new Error(`Expected one ${label} on slide ${slide}; found ${matches.length}`);
  }
  return matches[0].id;
}

function setText(presentation, id, value) {
  const shape = presentation.resolve(id);
  shape.text = value;
  return shape;
}

function formatContentTitle(presentation, id) {
  const shape = presentation.resolve(id);
  shape.position = { left: 97, top: 10, width: 1070, height: 62 };
  shape.text.style = {
    fontSize: 46,
    typeface: "Arial",
    color: C.navy,
    alignment: "left",
    verticalAlignment: "middle",
    autoFit: "shrinkText",
    insets: { top: 0, right: 0, bottom: 0, left: 0 },
  };
}

function formatContentBody(presentation, id) {
  const shape = presentation.resolve(id);
  shape.position = { left: 32, top: 105, width: 1120, height: 500 };
  shape.text.style = {
    fontSize: 36,
    typeface: "Arial",
    color: C.navy,
    alignment: "left",
    verticalAlignment: "top",
    autoFit: "shrinkText",
    insets: { top: 0, right: 0, bottom: 0, left: 0 },
  };
}

function setStructuredText(presentation, id, paragraphs) {
  const shape = presentation.resolve(id);
  shape.text.set(paragraphs);
  return shape;
}

function addSources(slide, lines) {
  slide.speakerNotes.textFrame.setText([
    "[Sources]",
    ...lines.map((line) => `- ${line}`),
    "[/Sources]",
  ]);
  slide.speakerNotes.setVisible(true);
}

function addBox(slide, name, position, text, options = {}) {
  const geometry = options.geometry || "roundRect";
  const config = {
    geometry,
    name,
    position,
    fill: options.fill || C.white,
    line: {
      style: options.lineStyle || "solid",
      fill: options.line || C.navy,
      width: options.lineWidth || 1.5,
    },
  };
  if (["rect", "textbox", "roundRect"].includes(geometry)) {
    config.borderRadius = geometry === "roundRect" ? "rounded-md" : 0;
  }
  const shape = slide.shapes.add(config);
  shape.text = text;
  shape.text.style = {
    fontSize: options.fontSize || 21,
    typeface: "Arial",
    color: options.color || C.text,
    bold: Boolean(options.bold),
    alignment: options.align || "center",
    verticalAlignment: options.vAlign || "middle",
    autoFit: "shrinkText",
    insets: options.insets || { top: 5, right: 7, bottom: 5, left: 7 },
  };
  return shape;
}

function addText(slide, name, position, text, options = {}) {
  return addBox(slide, name, position, text, {
    geometry: "textbox",
    fill: options.fill || "none",
    line: options.line || "none",
    lineWidth: 0,
    fontSize: options.fontSize || 21,
    bold: options.bold,
    color: options.color || C.navy,
    align: options.align || "left",
    vAlign: options.vAlign || "middle",
    insets: options.insets || { top: 1, right: 2, bottom: 1, left: 2 },
  });
}

function connect(slide, from, to, options = {}) {
  return slide.shapes.connect(from, to, {
    kind: options.kind || "elbow",
    fromSide: options.fromSide,
    toSide: options.toSide,
    line: {
      style: options.dashed ? "dashed" : "solid",
      fill: options.color || C.navy,
      width: options.width || 2,
    },
    tail: { type: "triangle", width: "sm", length: "sm" },
  });
}

function addLane(slide, name, top, height, label, fill, line) {
  const lane = slide.shapes.add({
    geometry: "roundRect",
    name,
    position: { left: 58, top, width: 1150, height },
    fill: "none",
    line: { style: "solid", fill: line, width: 1 },
    borderRadius: "rounded-lg",
  });
  addText(slide, `${name}-label`, { left: 68, top: top + 4, width: 115, height: 27 }, label, {
    fontSize: 20,
    bold: true,
    color: line,
  });
  return lane;
}

function buildOverallFlow(slide) {
  const cloud1 = addLane(slide, "s3-cloud-lane-1", 96, 122, "クラウド", C.softBlue, C.cloudLine);
  const client = addLane(slide, "s3-client-lane", 224, 122, "クライアント", C.softOrange, C.clientLine);
  const cloud2 = addLane(slide, "s3-cloud-lane-2", 352, 326, "クラウド", C.softBlue, C.cloudLine);

  const start = addBox(slide, "s3-start", { left: 505, top: 101, width: 270, height: 34 }, "ラウンド t 開始", {
    fill: C.keyFill,
    line: C.keyLine,
    fontSize: 20,
    bold: true,
  });
  const distribute = addBox(slide, "s3-distribute", { left: 392, top: 151, width: 496, height: 54 }, "第1層：global surrogate LoRAを配布", {
    fill: C.cloudFill,
    line: C.cloudLine,
    fontSize: 22,
    bold: true,
    geometry: "rect",
  });
  const privateData = addBox(slide, "s3-private", { left: 94, top: 260, width: 210, height: 58 }, "private data\n端末外へ送信しない", {
    fill: C.dataFill,
    line: C.dataLine,
    fontSize: 17,
    geometry: "parallelogram",
  });
  const local = addBox(slide, "s3-local", { left: 340, top: 244, width: 600, height: 84 }, "局所LoRA適応＋prototype化\nμpost・Δμ・frequencyを生成", {
    fill: C.clientFill,
    line: C.clientLine,
    fontSize: 22,
    bold: true,
    geometry: "rect",
  });
  const aggregate = addBox(slide, "s3-aggregate", { left: 340, top: 374, width: 600, height: 58 }, "FedAvgでglobal surrogate LoRAを更新", {
    fill: C.cloudFill,
    line: C.cloudLine,
    fontSize: 22,
    bold: true,
    geometry: "rect",
  });
  const seed = addBox(slide, "s3-seed", { left: 94, top: 475, width: 210, height: 62 }, "server seed\nクラウドのみ保有", {
    fill: C.dataFill,
    line: C.dataLine,
    fontSize: 19,
    geometry: "parallelogram",
  });
  const proxy = addBox(slide, "s3-proxy", { left: 340, top: 464, width: 600, height: 84 }, "第2層：proxy入力を探索・調整・検証\n更新前後surrogateでpost／deltaを評価", {
    fill: C.keyFill,
    line: C.keyLine,
    fontSize: 21,
    bold: true,
    geometry: "rect",
  });
  const distill = addBox(slide, "s3-distill", { left: 340, top: 580, width: 600, height: 66 }, "第3層：選択した P=(x̃, μpost) で\nfoundation LoRAを知識蒸留", {
    fill: C.cloudFill,
    line: C.cloudLine,
    fontSize: 21,
    bold: true,
    geometry: "rect",
  });
  const foundation = addBox(slide, "s3-foundation", { left: 986, top: 583, width: 190, height: 62 }, "更新済み\nfoundation LoRA", {
    fill: C.keyFill,
    line: C.keyLine,
    fontSize: 19,
    geometry: "parallelogram",
  });

  connect(slide, start, distribute, { fromSide: "bottom", toSide: "top", kind: "straight" });
  connect(slide, distribute, local, { fromSide: "bottom", toSide: "top", kind: "straight" });
  connect(slide, privateData, local, { fromSide: "right", toSide: "left", kind: "straight", color: C.dataLine });
  connect(slide, local, aggregate, { fromSide: "bottom", toSide: "top", kind: "straight" });
  connect(slide, aggregate, proxy, { fromSide: "bottom", toSide: "top", kind: "straight" });
  connect(slide, seed, proxy, { fromSide: "right", toSide: "left", kind: "straight", color: C.dataLine });
  connect(slide, proxy, distill, { fromSide: "bottom", toSide: "top", kind: "straight" });
  connect(slide, distill, foundation, { fromSide: "right", toSide: "left", kind: "straight" });
  connect(slide, aggregate, distribute, { fromSide: "right", toSide: "right", kind: "elbow", color: C.cloudLine });

  addText(slide, "s3-label-distribute", { left: 735, top: 203, width: 235, height: 28 }, "A_G^t・分類head", { fontSize: 18, align: "center" });
  addText(slide, "s3-label-upload", { left: 764, top: 328, width: 380, height: 31 }, "LoRA更新・μpost・Δμ・frequency", { fontSize: 18, align: "center" });
  addText(slide, "s3-label-global", { left: 752, top: 432, width: 360, height: 28 }, "A_G^t / A_G^(t+1)", { fontSize: 18, align: "center" });
  addText(slide, "s3-label-pair", { left: 700, top: 548, width: 330, height: 29 }, "選択済みproxy pair", { fontSize: 18, align: "center" });
  addText(slide, "s3-label-next", { left: 948, top: 191, width: 230, height: 54 }, "A_G^(t+1)\n次ラウンドへ再配布", { fontSize: 18, bold: true, align: "center", fill: C.white });

  cloud1.sendToBack();
  client.sendToBack();
  cloud2.sendToBack();
}

function buildLayerOne(slide) {
  const cloud = addLane(slide, "s4-cloud-lane", 98, 205, "クラウド", C.softBlue, C.cloudLine);
  const client = addLane(slide, "s4-client-lane", 312, 366, "クライアント", C.softOrange, C.clientLine);

  const agt = addBox(slide, "s4-agt", { left: 108, top: 146, width: 230, height: 66 }, "global surrogate\nLoRA  A_G^t", {
    fill: C.cloudFill,
    line: C.cloudLine,
    fontSize: 21,
    bold: true,
    geometry: "parallelogram",
  });
  const distribute = addBox(slide, "s4-distribute", { left: 405, top: 146, width: 180, height: 66 }, "クライアントへ\n配布", {
    fill: C.cloudFill,
    line: C.cloudLine,
    fontSize: 21,
    bold: true,
    geometry: "rect",
  });
  const fedavg = addBox(slide, "s4-fedavg", { left: 750, top: 125, width: 220, height: 66 }, "FedAvg集約", {
    fill: C.cloudFill,
    line: C.cloudLine,
    fontSize: 22,
    bold: true,
    geometry: "rect",
  });
  const agnext = addBox(slide, "s4-agnext", { left: 1010, top: 125, width: 170, height: 66 }, "A_G^(t+1)", {
    fill: C.keyFill,
    line: C.keyLine,
    fontSize: 22,
    bold: true,
    geometry: "parallelogram",
  });
  const layer2 = addBox(slide, "s4-layer2", { left: 750, top: 215, width: 430, height: 60 }, "第2層へ：A_G^t / A_G^(t+1) とprototype", {
    fill: C.keyFill,
    line: C.keyLine,
    fontSize: 19,
    geometry: "parallelogram",
  });

  const privateData = addBox(slide, "s4-private", { left: 95, top: 430, width: 205, height: 75 }, "private data\nD_k^t", {
    fill: C.dataFill,
    line: C.dataLine,
    fontSize: 22,
    bold: true,
    geometry: "parallelogram",
  });
  const pre = addBox(slide, "s4-pre", { left: 350, top: 345, width: 205, height: 62 }, "学習前推論\nq_pre", {
    fill: C.clientFill,
    line: C.clientLine,
    fontSize: 21,
    geometry: "rect",
  });
  const adapt = addBox(slide, "s4-adapt", { left: 350, top: 445, width: 205, height: 62 }, "LoRA局所適応", {
    fill: C.clientFill,
    line: C.clientLine,
    fontSize: 21,
    bold: true,
    geometry: "rect",
  });
  const post = addBox(slide, "s4-post", { left: 350, top: 545, width: 205, height: 62 }, "学習後推論\nq_post", {
    fill: C.clientFill,
    line: C.clientLine,
    fontSize: 21,
    geometry: "rect",
  });
  const proto = addBox(slide, "s4-proto", { left: 625, top: 425, width: 250, height: 105 }, "logit prototype化\nμpost\nΔμ=μpost−μpre\nfrequency", {
    fill: C.keyFill,
    line: C.keyLine,
    fontSize: 20,
    bold: true,
    geometry: "rect",
  });
  const uploadLora = addBox(slide, "s4-upload-lora", { left: 940, top: 343, width: 235, height: 70 }, "送信①\nLoRA更新・head", {
    fill: C.dataFill,
    line: C.dataLine,
    fontSize: 20,
    geometry: "parallelogram",
  });
  const uploadProto = addBox(slide, "s4-upload-proto", { left: 925, top: 455, width: 260, height: 88 }, "送信②\nμpost・Δμ・frequency", {
    fill: C.dataFill,
    line: C.dataLine,
    fontSize: 17,
    geometry: "parallelogram",
  });
  const noPre = addBox(slide, "s4-no-pre", { left: 625, top: 565, width: 250, height: 70 }, "μpreはdelta計算だけに使用\nサーバへ独立には送信しない", {
    fill: C.futureFill,
    line: C.futureLine,
    fontSize: 18,
    lineStyle: "dashed",
    geometry: "rect",
  });

  connect(slide, agt, distribute, { fromSide: "right", toSide: "left", kind: "straight" });
  connect(slide, distribute, pre, { fromSide: "bottom", toSide: "top", kind: "elbow" });
  connect(slide, distribute, adapt, { fromSide: "bottom", toSide: "top", kind: "elbow" });
  connect(slide, privateData, pre, { fromSide: "right", toSide: "left", color: C.dataLine });
  connect(slide, privateData, adapt, { fromSide: "right", toSide: "left", color: C.dataLine });
  connect(slide, privateData, post, { fromSide: "right", toSide: "left", color: C.dataLine });
  connect(slide, pre, adapt, { fromSide: "bottom", toSide: "top", kind: "straight" });
  connect(slide, adapt, post, { fromSide: "bottom", toSide: "top", kind: "straight" });
  connect(slide, pre, proto, { fromSide: "right", toSide: "left", kind: "elbow" });
  connect(slide, post, proto, { fromSide: "right", toSide: "left", kind: "elbow" });
  connect(slide, adapt, uploadLora, { fromSide: "right", toSide: "left", kind: "elbow" });
  connect(slide, proto, uploadProto, { fromSide: "right", toSide: "left", kind: "straight" });
  connect(slide, uploadLora, fedavg, { fromSide: "top", toSide: "bottom", kind: "elbow" });
  connect(slide, fedavg, agnext, { fromSide: "right", toSide: "left", kind: "straight" });
  connect(slide, agt, layer2, { fromSide: "bottom", toSide: "left", kind: "elbow", color: C.cloudLine });
  connect(slide, agnext, layer2, { fromSide: "bottom", toSide: "right", kind: "elbow", color: C.cloudLine });
  connect(slide, uploadProto, layer2, { fromSide: "top", toSide: "bottom", kind: "elbow", color: C.dataLine });
  connect(slide, agnext, distribute, { fromSide: "top", toSide: "top", kind: "elbow", color: C.cloudLine });

  addText(slide, "s4-next-label", { left: 1000, top: 88, width: 190, height: 32 }, "次ラウンドへ再配布", { fontSize: 18, bold: true, align: "center" });
  addText(slide, "s4-private-label", { left: 92, top: 514, width: 220, height: 56 }, "raw sampleは\n端末外へ出ない", { fontSize: 18, bold: true, color: C.dataLine, align: "center" });

  cloud.sendToBack();
  client.sendToBack();
}

function buildLayerTwo(slide) {
  const agt = addBox(slide, "s5-mgt", { left: 70, top: 112, width: 220, height: 62 }, "更新前\nM_G^t", {
    fill: C.cloudFill,
    line: C.cloudLine,
    fontSize: 21,
    bold: true,
    geometry: "parallelogram",
  });
  const agnext = addBox(slide, "s5-mgnext", { left: 330, top: 112, width: 220, height: 62 }, "更新後\nM_G^(t+1)", {
    fill: C.cloudFill,
    line: C.cloudLine,
    fontSize: 21,
    bold: true,
    geometry: "parallelogram",
  });
  const proto = addBox(slide, "s5-prototypes", { left: 590, top: 112, width: 270, height: 62 }, "μpost・Δμ・frequency", {
    fill: C.dataFill,
    line: C.dataLine,
    fontSize: 18,
    bold: true,
    geometry: "parallelogram",
  });
  const seed = addBox(slide, "s5-seed", { left: 910, top: 112, width: 250, height: 62 }, "server seed候補", {
    fill: C.dataFill,
    line: C.dataLine,
    fontSize: 21,
    bold: true,
    geometry: "parallelogram",
  });

  const retrieve = addBox(slide, "s5-retrieve", { left: 100, top: 255, width: 225, height: 74 }, "seed検索\npost prototypeに近い候補", {
    fill: C.cloudFill,
    line: C.cloudLine,
    fontSize: 20,
    geometry: "rect",
  });
  const transform = addBox(slide, "s5-transform", { left: 390, top: 255, width: 225, height: 74 }, "変換 Tφ\n時刻・crop・明度・blur等", {
    fill: C.cloudFill,
    line: C.cloudLine,
    fontSize: 19,
    geometry: "rect",
  });
  const evaluate = addBox(slide, "s5-evaluate", { left: 680, top: 225, width: 330, height: 136 }, "global surrogateで再評価\npost一致：KL(μpost || p_G^(t+1)(x̃))\ndelta一致：D(Δμ, p_G^(t+1)(x̃)−p_G^t(x̃))", {
    fill: C.keyFill,
    line: C.keyLine,
    fontSize: 19,
    bold: true,
    geometry: "rect",
  });
  const decision = addBox(slide, "s5-decision", { left: 1070, top: 240, width: 120, height: 110 }, "品質条件を\n満たす？", {
    fill: C.keyFill,
    line: C.keyLine,
    fontSize: 19,
    bold: true,
    geometry: "diamond",
  });
  const regularization = addBox(slide, "s5-reg", { left: 390, top: 445, width: 330, height: 100 }, "妥当性の制約 Lreg\nseedからの逸脱・自然性\n時間的一貫性・多様性", {
    fill: C.futureFill,
    line: C.futureLine,
    fontSize: 19,
    geometry: "rect",
  });
  const proxy = addBox(slide, "s5-proxy-output", { left: 930, top: 475, width: 250, height: 80 }, "代理入力 x̃\n次：proxy pairを構成", {
    fill: C.dataFill,
    line: C.dataLine,
    fontSize: 21,
    bold: true,
    geometry: "parallelogram",
  });
  const note = addBox(slide, "s5-note", { left: 100, top: 458, width: 225, height: 82 }, "一度だけforwardする処理ではなく\n選択・調整・再評価を反復", {
    fill: C.softOrange,
    line: C.clientLine,
    fontSize: 18,
    bold: true,
    geometry: "rect",
  });

  connect(slide, seed, retrieve, { fromSide: "bottom", toSide: "top", color: C.dataLine });
  connect(slide, retrieve, transform, { fromSide: "right", toSide: "left", kind: "straight" });
  connect(slide, transform, evaluate, { fromSide: "right", toSide: "left", kind: "straight" });
  connect(slide, agt, evaluate, { fromSide: "bottom", toSide: "top", kind: "elbow", color: C.cloudLine });
  connect(slide, agnext, evaluate, { fromSide: "bottom", toSide: "top", kind: "elbow", color: C.cloudLine });
  connect(slide, proto, evaluate, { fromSide: "bottom", toSide: "top", kind: "elbow", color: C.dataLine });
  connect(slide, regularization, evaluate, { fromSide: "top", toSide: "bottom", kind: "straight", color: C.futureLine });
  connect(slide, evaluate, decision, { fromSide: "right", toSide: "left", kind: "straight" });
  connect(slide, decision, proxy, { fromSide: "bottom", toSide: "top", kind: "elbow", color: C.dataLine });
  connect(slide, decision, transform, { fromSide: "left", toSide: "bottom", kind: "elbow", color: C.clientLine });

  addText(slide, "s5-yes", { left: 1080, top: 390, width: 100, height: 30 }, "Yes", { fontSize: 19, bold: true, color: C.dataLine, align: "center" });
  addText(slide, "s5-no", { left: 735, top: 377, width: 270, height: 45 }, "No：候補を再選択・再調整", { fontSize: 19, bold: true, color: C.clientLine, align: "center", fill: C.white });
}

function buildLayerThree(slide) {
  const proxy = addBox(slide, "s6-proxy", { left: 70, top: 155, width: 210, height: 70 }, "代理入力 x̃", {
    fill: C.dataFill,
    line: C.dataLine,
    fontSize: 22,
    bold: true,
    geometry: "parallelogram",
  });
  const teacher = addBox(slide, "s6-teacher", { left: 70, top: 265, width: 210, height: 70 }, "teacher μpost", {
    fill: C.dataFill,
    line: C.dataLine,
    fontSize: 22,
    bold: true,
    geometry: "parallelogram",
  });
  const pair = addBox(slide, "s6-pair", { left: 340, top: 190, width: 250, height: 105 }, "proxy pair\nP=(x̃, μpost)", {
    fill: C.keyFill,
    line: C.keyLine,
    fontSize: 25,
    bold: true,
    geometry: "rect",
  });
  const selection = addBox(slide, "s6-selection", { left: 650, top: 135, width: 290, height: 205 }, "検証・選択・重み付け\n\n基本：ρ=frequency π\n第II段階：ρ=Qinv×π\n後段：Qcons・Qnovel・Qretain", {
    fill: C.keyFill,
    line: C.keyLine,
    fontSize: 18,
    bold: true,
    geometry: "rect",
  });
  const distill = addBox(slide, "s6-distill", { left: 995, top: 180, width: 220, height: 115 }, "知識蒸留\nLcurrent=Σρ・KL\n更新：foundation LoRAのみ", {
    fill: C.cloudFill,
    line: C.cloudLine,
    fontSize: 19,
    bold: true,
    geometry: "rect",
  });
  const foundation = addBox(slide, "s6-foundation", { left: 975, top: 390, width: 240, height: 78 }, "更新済み\nfoundation LoRA", {
    fill: C.dataFill,
    line: C.dataLine,
    fontSize: 22,
    bold: true,
    geometry: "parallelogram",
  });
  const future = addBox(slide, "s6-future", { left: 330, top: 430, width: 555, height: 105 }, "第III段階の拡張\nreplay memory・固定anchor・Qretain\n必要な場合のみclient固有LoRAの残差経路", {
    fill: C.futureFill,
    line: C.futureLine,
    fontSize: 20,
    lineStyle: "dashed",
    geometry: "rect",
  });
  const note = addBox(slide, "s6-note", { left: 70, top: 455, width: 210, height: 75 }, "global surrogateの出力を\nteacherにはしない", {
    fill: C.softOrange,
    line: C.clientLine,
    fontSize: 18,
    bold: true,
    geometry: "rect",
  });

  connect(slide, proxy, pair, { fromSide: "right", toSide: "left", kind: "elbow", color: C.dataLine });
  connect(slide, teacher, pair, { fromSide: "right", toSide: "left", kind: "elbow", color: C.dataLine });
  connect(slide, pair, selection, { fromSide: "right", toSide: "left", kind: "straight" });
  connect(slide, selection, distill, { fromSide: "right", toSide: "left", kind: "straight" });
  connect(slide, distill, foundation, { fromSide: "bottom", toSide: "top", kind: "straight", color: C.dataLine });
  connect(slide, future, distill, { fromSide: "right", toSide: "bottom", kind: "elbow", color: C.futureLine, dashed: true });

  addText(slide, "s6-pair-label", { left: 420, top: 145, width: 205, height: 32 }, "第2層の出力", { fontSize: 18, bold: true, align: "center" });
  addText(slide, "s6-selected-label", { left: 875, top: 105, width: 300, height: 32 }, "選択済みpair＋重みρ", { fontSize: 18, bold: true, align: "center" });
}

async function main() {
  await fs.mkdir(previewDir, { recursive: true });
  await fs.mkdir(layoutDir, { recursive: true });

  const presentation = await PresentationFile.importPptx(await FileBlob.load(starterPptx));
  const records = parseInspect(
    await presentation.inspect({
      kind: "slide,textbox,shape,image,table",
      include: "id,slide,name,text,textPreview,placeholder,rows,cols",
      maxChars: 100000,
    }),
  );

  const ids = {
    s1Title: findAid(records, 1, "textbox", (r) => r.placeholder === "title", "title"),
    s1Date: findAid(records, 1, "textbox", (r) => r.text === "7月31日報告会", "meeting date"),
    s1Affiliation: findAid(records, 1, "textbox", (r) => r.placeholder === "subtitle", "affiliation"),
    s2Title: findAid(records, 2, "textbox", (r) => r.placeholder === "title", "title"),
    s2Page: findAid(records, 2, "textbox", (r) => r.placeholder === "slideNumber", "page marker"),
    s2Body: findAid(records, 2, "textbox", (r) => r.name === "コンテンツ プレースホルダー 3", "body"),
    s3Title: findAid(records, 3, "textbox", (r) => r.placeholder === "title", "title"),
    s3Page: findAid(records, 3, "textbox", (r) => r.placeholder === "slideNumber", "page marker"),
    s3Image: findAid(records, 3, "image", (r) => r.name === "コンテンツ プレースホルダー 22", "workflow image"),
    s4Title: findAid(records, 4, "textbox", (r) => r.placeholder === "title", "title"),
    s4Page: findAid(records, 4, "textbox", (r) => r.placeholder === "slideNumber", "page marker"),
    s4Image: findAid(records, 4, "image", (r) => r.name === "コンテンツ プレースホルダー 22", "workflow image"),
    s5Title: findAid(records, 5, "textbox", (r) => r.placeholder === "title", "title"),
    s5Page: findAid(records, 5, "textbox", (r) => r.placeholder === "slideNumber", "page marker"),
    s5Image: findAid(records, 5, "image", (r) => r.name === "コンテンツ プレースホルダー 22", "workflow image"),
    s6Title: findAid(records, 6, "textbox", (r) => r.placeholder === "title", "title"),
    s6Page: findAid(records, 6, "textbox", (r) => r.placeholder === "slideNumber", "page marker"),
    s6Image: findAid(records, 6, "image", (r) => r.name === "コンテンツ プレースホルダー 22", "workflow image"),
    s7Title: findAid(records, 7, "textbox", (r) => r.placeholder === "title", "title"),
    s7Page: findAid(records, 7, "textbox", (r) => r.placeholder === "slideNumber", "page marker"),
    s7Body: findAid(records, 7, "textbox", (r) => r.name === "コンテンツ プレースホルダー 3", "body"),
    s8Title: findAid(records, 8, "textbox", (r) => r.placeholder === "title", "title"),
    s8Page: findAid(records, 8, "textbox", (r) => r.placeholder === "slideNumber", "page marker"),
    s8Body: findAid(records, 8, "textbox", (r) => r.name === "コンテンツ プレースホルダー 3", "body"),
  };

  setText(presentation, ids.s1Title, "FedPACTの全体処理フロー\n代理データを介した異種モデル間知識転移");
  setText(presentation, ids.s1Date, "9月4日研究会");
  setText(presentation, ids.s1Affiliation, "名古屋大学\n情報学研究科\n知能システム学専攻　1年");
  addSources(presentation.slides.items[0], [
    "進捗会0731_吉村有生.pptx（視覚テンプレート）",
    "FedPACT手法提案260901.pdf",
    "FedPACT実験計画260901.pdf",
  ]);

  setText(presentation, ids.s2Title, "private適応知識を異種基盤モデルへ転移する");
  setText(presentation, ids.s2Page, "2");
  setStructuredText(presentation, ids.s2Body, [
    { bulletCharacter: "■", marginLeft: 58, indent: -28, runs: ["目的：生データを送らず，局所適応知識を基盤モデルへ転移する"] },
    { bulletCharacter: "■", marginLeft: 58, indent: -28, runs: ["3B surrogateのLoRAは，7B基盤へ直接統合できない"] },
    { bulletCharacter: "■", marginLeft: 58, indent: -28, runs: ["private入力が異なるため，client logitsを直接対応付けられない"] },
    { bulletCharacter: "■", marginLeft: 58, indent: -28, runs: ["server seed上のproxy pairで二つの壁を接続する"] },
  ]);
  addSources(presentation.slides.items[1], [
    "FedPACT手法提案260901.pdf p.3",
    "FedPACT実験計画260901.pdf p.1-p.2",
  ]);

  for (const imageId of [ids.s3Image, ids.s4Image, ids.s5Image, ids.s6Image]) {
    presentation.resolve(imageId).delete();
  }

  setText(presentation, ids.s3Title, "三層でprivate適応を基盤モデルへ接続する");
  setText(presentation, ids.s3Page, "3");
  buildOverallFlow(presentation.slides.items[2]);
  addSources(presentation.slides.items[2], [
    "FedPACT手法提案260901.pdf p.4-p.7",
    "FedPACT実験計画260901.pdf p.3-p.4",
  ]);

  setText(presentation, ids.s4Title, "第1層：privateデータは端末内に保持");
  setText(presentation, ids.s4Page, "4");
  buildLayerOne(presentation.slides.items[3]);
  addSources(presentation.slides.items[3], [
    "FedPACT手法提案260901.pdf p.4-p.5",
    "FedPACT実験計画260901.pdf p.3-p.4",
  ]);

  setText(presentation, ids.s5Title, "第2層：proxyを反復的に探索・調整する");
  setText(presentation, ids.s5Page, "5");
  buildLayerTwo(presentation.slides.items[4]);
  addSources(presentation.slides.items[4], [
    "FedPACT手法提案260901.pdf p.5-p.6",
    "FedPACT実験計画260901.pdf p.3-p.4",
  ]);

  setText(presentation, ids.s6Title, "第3層：proxy pairで基盤LoRAを更新");
  setText(presentation, ids.s6Page, "6");
  buildLayerThree(presentation.slides.items[5]);
  addSources(presentation.slides.items[5], [
    "FedPACT手法提案260901.pdf p.6-p.7",
    "FedPACT実験計画260901.pdf p.4, p.6, p.8",
  ]);

  setText(presentation, ids.s7Title, "今後の予定");
  setText(presentation, ids.s7Page, "7");
  setStructuredText(presentation, ids.s7Body, [
    { bulletCharacter: "┃", marginLeft: 24, indent: -16, runs: ["第I段階　ローカル実装・3〜5 round確認（現在地）"] },
    { bulletCharacter: "┃", marginLeft: 24, indent: -16, runs: ["第II段階　3B→7B以上の基本FedPACT予備実験"] },
    { bulletCharacter: "┃", marginLeft: 24, indent: -16, runs: ["第III段階　継続学習・replay・anchor"] },
    { bulletCharacter: "┃", marginLeft: 24, indent: -16, runs: ["第IV段階　本実験・ablation"] },
  ]);
  addSources(presentation.slides.items[6], [
    "FedPACT実験計画260901.pdf p.1-p.10",
  ]);

  setText(presentation, ids.s8Title, "要点：private適応をproxy経由で転移する");
  setText(presentation, ids.s8Page, "8");
  setStructuredText(presentation, ids.s8Body, [
    { bulletCharacter: "■", marginLeft: 58, indent: -28, runs: ["μpost・Δμ・frequencyでprivate適応を表現する"] },
    { bulletCharacter: "■", marginLeft: 58, indent: -28, runs: ["global surrogateとserver seedからproxy入力を得る"] },
    { bulletCharacter: "■", marginLeft: 58, indent: -28, runs: ["選択したpairを異種foundation LoRAへ蒸留する"] },
  ]);
  addSources(presentation.slides.items[7], [
    "FedPACT手法提案260901.pdf p.9-p.10",
    "FedPACT実験計画260901.pdf p.10",
  ]);

  for (const titleId of [ids.s2Title, ids.s3Title, ids.s4Title, ids.s5Title, ids.s6Title, ids.s7Title, ids.s8Title]) {
    formatContentTitle(presentation, titleId);
  }
  for (const bodyId of [ids.s2Body, ids.s7Body, ids.s8Body]) {
    formatContentBody(presentation, bodyId);
  }

  for (let index = 0; index < presentation.slides.items.length; index += 1) {
    const slide = presentation.slides.items[index];
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(`${previewDir}\\${stem}.png`, await presentation.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(`${layoutDir}\\${stem}.layout.json`, await layout.text());
  }

  await writeBlob(`${workspace}\\final-montage.webp`, await presentation.export({ format: "webp", montage: true, scale: 1 }));

  const inspect = await presentation.inspect({
    kind: "slide,textbox,shape,image,table,notes",
    include: "id,slide,name,bbox,textPreview,isPlaceholder",
    maxChars: 200000,
  });
  await fs.writeFile(`${workspace}\\final-inspect.ndjson`, inspect.ndjson);

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(finalPptx);
  console.log(finalPptx);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
