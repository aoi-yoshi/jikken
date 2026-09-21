import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "C:\\Python\\実験_修正";
const skillDir = "C:\\Users\\aoi7y\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.909.12148\\skills\\presentations";
const runtimePython = "C:\\Users\\aoi7y\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";
const buildDir = path.join(workspaceDir, "tmp", "presentations", "fedpact_0918_10min");
const outputDir = path.join(workspaceDir, "artifacts", "presentations");
const finalPath = path.join(outputDir, "FedPACT_研究会_10分_20260918_v11.pptx");
const fontFamily = "Yu Gothic";

const C = {
  bg: "#F7F5EF",
  paper: "#FFFFFF",
  navy: "#173B4B",
  teal: "#147B78",
  aqua: "#69C3BE",
  bluePale: "#E6F1F3",
  orange: "#E7833D",
  orangePale: "#FAE9D8",
  red: "#B94A3A",
  redPale: "#F5DFDA",
  ink: "#26373D",
  muted: "#617078",
  line: "#C9D2D3",
  lineDark: "#87979C",
};

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });

const presentation = Presentation.create({
  slideSize: { width: 1280, height: 720 },
});

function rect(slide, left, top, width, height, fill, lineFill = "none", lineWidth = 0) {
  return slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill,
    line: { fill: lineFill, width: lineWidth },
  });
}

function textBox(slide, text, left, top, width, height, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: opts.fill ?? "none",
    line: { fill: opts.lineFill ?? "none", width: opts.lineWidth ?? 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: opts.typeface ?? fontFamily,
    fontSize: opts.fontSize ?? 24,
    bold: opts.bold ?? false,
    color: opts.color ?? C.ink,
    alignment: opts.alignment ?? "left",
    verticalAlignment: opts.verticalAlignment ?? "top",
    autoFit: opts.autoFit ?? "shrinkText",
    wrap: "square",
    insets: opts.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
    lineSpacing: opts.lineSpacing,
  };
  return shape;
}

function title(slide, heading, number) {
  rect(slide, 0, 0, 1280, 720, C.bg);
  rect(slide, 40, 36, 7, 54, C.teal);
  textBox(slide, heading, 62, 34, 1125, 62, {
    fontSize: 40,
    bold: true,
    color: C.navy,
    verticalAlignment: "middle",
  });
  textBox(slide, String(number).padStart(2, "0"), 1190, 46, 42, 26, {
    fontSize: 16,
    bold: true,
    color: C.teal,
    alignment: "right",
  });
  rect(slide, 48, 680, 1184, 1.5, C.line);
  textBox(slide, "FedPACT / NEXAR　研究会　2026.09.18", 52, 687, 780, 18, {
    fontSize: 12,
    color: C.muted,
  });
  textBox(slide, `${number} / 10`, 1100, 687, 126, 18, {
    fontSize: 12,
    color: C.muted,
    alignment: "right",
  });
}

async function addImage(slide, filePath, left, top, width, height, alt, fit = "cover") {
  const blob = await fs.readFile(filePath);
  const ext = path.extname(filePath).toLowerCase();
  const contentType = ext === ".png" ? "image/png" : "image/jpeg";
  return slide.images.add({
    blob,
    contentType,
    alt,
    fit,
    position: { left, top, width, height },
  });
}

function caption(slide, label, left, top, width, color = C.muted) {
  return textBox(slide, label, left, top, width, 24, {
    fontSize: 15,
    bold: true,
    color,
  });
}

function note(slide, timing, script, sources) {
  slide.speakerNotes.textFrame.setText(
    `【目安 ${timing}】\n${script}\n\n【根拠】\n${sources.join("\n")}`,
  );
}

function styleTable(table, { headerFill = C.paper, headerColor = C.navy, bodyFont = 18 } = {}) {
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  for (let r = 0; r < table.rows.length; r += 1) {
    for (let c = 0; c < table.columns.length; c += 1) {
      const cell = table.getCell(r, c);
      cell.fill = r === 0 ? headerFill : (r % 2 === 0 ? "#F0F4F3" : C.paper);
      cell.text.style = {
        typeface: fontFamily,
        fontSize: r === 0 ? 17 : bodyFont,
        bold: r === 0,
        color: r === 0 ? headerColor : C.ink,
        alignment: c === 0 ? "left" : "center",
        verticalAlignment: "middle",
        autoFit: "shrinkText",
        insets: { left: 7, right: 7, top: 4, bottom: 4 },
      };
    }
  }
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: table.columns.length }).assign({
    fill: headerFill,
    textStyle: {
      typeface: fontFamily,
      bold: true,
      color: headerColor,
    },
  });
}

function tableValues(rawValues, headerPt = 13, bodyPt = 15) {
  return rawValues.map((row, r) => row.map((value) => ([{
    run: String(value),
    textStyle: {
      typeface: fontFamily,
      fontSize: `${r === 0 ? headerPt : bodyPt}pt`,
      bold: r === 0,
      color: r === 0 ? C.navy : C.ink,
    },
  }])));
}

// Slide 1: use case, conventional gap, and the role of FedPACT.
{
  const s = presentation.slides.add();
  title(s, "異なる現場で得た適応を、cloud側のfoundationへ還元する", 1);

  textBox(s, "同じ危険検知task・共通のnormal／riskでも、場所や時間が変われば、各clientが経験する状況は異なる", 70, 103, 1115, 38, {
    fontSize: 23,
    bold: true,
    color: C.teal,
    alignment: "center",
  });

  // 1. Use setting and value.
  textBox(s, "1　想定する利用場面と価値", 58, 151, 366, 32, {
    fontSize: 21,
    bold: true,
    color: C.navy,
  });
  await addImage(
    s,
    path.join(workspaceDir, "artifacts", "datasets", "frames", "train", "003d6c3909", "003.jpg"),
    58, 193, 176, 112,
    "昼間の市街地を走るNEXAR映像",
    "cover",
  );
  await addImage(
    s,
    path.join(workspaceDir, "artifacts", "datasets", "frames", "train", "b82e955a9fdf13e2", "003.jpg"),
    246, 193, 176, 112,
    "夜間の交差点で横から車が接近するNEXAR映像",
    "cover",
  );
  textBox(s, "client A\n昼間・市街地", 59, 311, 174, 45, {
    fontSize: 16,
    bold: true,
    color: C.teal,
    alignment: "center",
  });
  textBox(s, "client B\n夜間・交差点", 247, 311, 174, 45, {
    fontSize: 16,
    bold: true,
    color: C.teal,
    alignment: "center",
  });
  textBox(s, "各clientは、自分のprivate dataから\n異なる局所適応挙動を獲得する", 66, 375, 349, 58, {
    fontSize: 21,
    bold: true,
    color: C.ink,
    alignment: "center",
  });
  rect(s, 58, 456, 364, 92, C.bluePale);
  textBox(s, "目指す価値", 78, 469, 324, 24, {
    fontSize: 18,
    bold: true,
    color: C.teal,
    alignment: "center",
  });
  textBox(s, "各現場の経験を、共通taskの\nprediction quality改善に生かす", 76, 499, 328, 42, {
    fontSize: 20,
    bold: true,
    color: C.navy,
    alignment: "center",
  });

  // 2. The gap left by FedAvg and shared-data distillation.
  rect(s, 448, 151, 2, 413, C.line);
  textBox(s, "2　従来手法で残る接続問題", 474, 151, 334, 32, {
    fontSize: 21,
    bold: true,
    color: C.navy,
  });
  textBox(s, "client surrogate", 488, 213, 135, 46, {
    fontSize: 18,
    bold: true,
    color: C.navy,
    fill: C.bluePale,
    alignment: "center",
    verticalAlignment: "middle",
  });
  textBox(s, "FedAvg", 640, 204, 72, 22, {
    fontSize: 16,
    bold: true,
    color: C.teal,
    alignment: "center",
  });
  rect(s, 620, 235, 92, 3, C.aqua);
  textBox(s, "global\nsurrogate", 723, 207, 88, 58, {
    fontSize: 18,
    bold: true,
    color: C.navy,
    fill: C.bluePale,
    alignment: "center",
    verticalAlignment: "middle",
  });
  textBox(s, "×", 615, 292, 58, 58, {
    fontSize: 38,
    bold: true,
    color: C.red,
    alignment: "center",
    verticalAlignment: "middle",
  });
  textBox(s, "異種のcloud foundationへ\nparameterをそのまま渡せない", 489, 354, 310, 63, {
    fontSize: 21,
    bold: true,
    color: C.ink,
    alignment: "center",
  });
  rect(s, 485, 446, 318, 1.5, C.line);
  textBox(s, "既存蒸留は共通・公開dataを使う例が多い\n今回は整合用dataを\nclient／cloudで共有しない", 472, 466, 344, 71, {
    fontSize: 17,
    color: C.muted,
    alignment: "center",
  });

  // 3. FedPACT's role.
  rect(s, 832, 151, 2, 413, C.line);
  textBox(s, "3　FedPACTが担う接続", 856, 151, 354, 32, {
    fontSize: 21,
    bold: true,
    color: C.navy,
  });
  textBox(s, "局所適応後の予測と\n変化方向の要約", 875, 205, 315, 64, {
    fontSize: 21,
    bold: true,
    color: C.navy,
    fill: C.bluePale,
    alignment: "center",
    verticalAlignment: "middle",
  });
  rect(s, 1029, 269, 4, 25, C.aqua);
  textBox(s, "server seed上で\nserver-side proxyとして顕在化", 875, 294, 315, 68, {
    fontSize: 21,
    bold: true,
    color: C.teal,
    fill: "#DDF0ED",
    alignment: "center",
    verticalAlignment: "middle",
  });
  rect(s, 1029, 362, 4, 25, C.aqua);
  textBox(s, "異種foundationへ\n反映できるかを検証", 875, 387, 315, 68, {
    fontSize: 21,
    bold: true,
    color: C.navy,
    fill: C.orangePale,
    alignment: "center",
    verticalAlignment: "middle",
  });
  textBox(s, "proxy-mediated transferが\n2つのモデルをつなぐ", 882, 480, 301, 58, {
    fontSize: 21,
    bold: true,
    color: C.orange,
    alignment: "center",
  });

  rect(s, 58, 584, 1164, 69, C.navy);
  textBox(s, "対象：共通task上のtask-relevantな予測挙動　（private sampleの意味内容全体は対象外）", 82, 603, 1116, 32, {
    fontSize: 21,
    bold: true,
    color: "#FFFFFF",
    alignment: "center",
    verticalAlignment: "middle",
  });
  note(
    s,
    "0:00–1:00",
    "まず、想定する利用場面です。すべてのclientとcloudは同じ危険検知taskと共通のnormal／riskを持ちます。一方、昼の市街地や夜間の道路ではprivate dataが異なり、各clientは異なる局所適応挙動を得ます。この経験をcloudへ戻し、共通taskの改善に生かすことが目的です。ただしFedAvgは同じ構造のsurrogate間を統合できますが、異種foundationへparameterを直接渡せません。また今回は整合用dataも共有しません。そこでFedPACTは、局所適応後の予測と変化方向を要約し、server seed上のproxyとして表して、この間を接続します。対象は場面の意味内容全体ではなく、共通task上の予測挙動です。server seedだけで十分な場合は追加効果が小さくなるため、効果は今後比較します。今回は性能の証明ではなく、この情報の流れを追跡できるかを確認しました。",
    [
      "notes/advisor/sources/raw/2026-09-16/slack_zettsu_reply_to_chen_ja_date_unknown_received_2026-09-16.md（11:00、target scenario、既存手法との違い、server seedとproxyの位置付け）",
      "notes/advisor/sources/raw/2026-09-01/FedPACT手法提案260901.pdf（研究目的とproxy-mediated transfer）",
      "画像: artifacts/datasets/frames/train/003d6c3909/003.jpg, artifacts/datasets/frames/train/b82e955a9fdf13e2/003.jpg",
    ],
  );
}

// Slide 2: problem and meaning
{
  const s = presentation.slides.add();
  title(s, "同じriskでも、現場の状況は大きく異なる", 2);
  await addImage(
    s,
    path.join(workspaceDir, "artifacts", "datasets", "frames", "train", "4b013e9c446fbc83", "003.jpg"),
    60, 128, 500, 281,
    "昼間の市街地で歩行者と車両が見えるNEXAR映像",
  );
  await addImage(
    s,
    path.join(workspaceDir, "artifacts", "datasets", "frames", "train", "665192ef7039e5f7", "003.jpg"),
    720, 128, 500, 281,
    "夜間の高速道路で車両が接近するNEXAR映像",
  );
  caption(s, "client A：昼間・市街地", 60, 104, 500, C.teal);
  caption(s, "client B：夜間・高速道路", 720, 104, 500, C.teal);
  textBox(s, "FedPACTが再現する対象", 70, 450, 330, 34, {
    fontSize: 22,
    bold: true,
    color: C.navy,
  });
  textBox(s, "共通タスク上で、private dataによる局所学習後の予測と、その変化", 70, 492, 1130, 48, {
    fontSize: 29,
    bold: true,
    color: C.teal,
    alignment: "center",
  });
  rect(s, 70, 562, 1140, 72, C.bluePale);
  textBox(s, "研究の意味：server seedだけでは事前に網羅しにくい現場差を、映像そのものではなく予測挙動の要約としてfoundationへ還元する", 94, 579, 1092, 42, {
    fontSize: 21,
    color: C.ink,
    alignment: "center",
    verticalAlignment: "middle",
  });
  note(
    s,
    "1:00–1:30",
    "具体的には、昼の市街地と夜の高速道路では映像の意味内容が違います。FedPACTはfull scene semanticsを同じだとは仮定しません。現在のnormal／riskという共通出力空間で、各clientのprivate dataによって予測がどう変わったかをserver側で再構成し、異なるfoundation modelへ渡すことが対象です。",
    [
      "notes/advisor/sources/raw/2026-09-16/slack_zettsu_reply_to_chen_ja_date_unknown_received_2026-09-16.md（target scenario、Similar logits、server seedの説明）",
      "notes/advisor/sources/raw/2026-09-01/FedPACT手法提案260901.pdf（基準表記）",
    ],
  );
}

// Slide 3: what changed since last week
{
  const s = presentation.slides.add();
  title(s, "9月15日以降、研究の焦点をLayer 2へ絞った", 3);
  const cols = [70, 432, 794];
  const dates = ["9 / 15", "9 / 16", "9 / 17"];
  const heads = [
    "複雑さを減らす",
    "第I段階の目的を固定",
    "3 roundの接続を確認",
  ];
  const bodies = [
    "機能を一度に入れると失敗原因を切り分けられない。中心課題と最小構成を先に確認する。",
    "三層構造は維持し、研究の中核を第2層に置く。第I段階では性能より、処理とstateの追跡を確認する。",
    "入力条件を修正した。2 clientからfoundation更新、次roundのstate継承までをログ付きで実行した。",
  ];
  for (let i = 0; i < 3; i += 1) {
    textBox(s, dates[i], cols[i], 130, 315, 44, {
      fontSize: 24,
      bold: true,
      color: i === 2 ? C.orange : C.teal,
    });
    rect(s, cols[i], 181, 300, 4, i === 2 ? C.orange : C.aqua);
    textBox(s, heads[i], cols[i], 208, 315, 42, {
      fontSize: 26,
      bold: true,
      color: C.navy,
    });
    textBox(s, bodies[i], cols[i], 270, 315, 205, {
      fontSize: 19,
      color: C.ink,
      lineSpacing: 1.12,
    });
  }
  rect(s, 70, 526, 1025, 1.5, C.lineDark);
  rect(s, 1095, 518, 18, 18, C.orange);
  textBox(s, "今週の報告範囲", 70, 552, 260, 32, {
    fontSize: 20,
    bold: true,
    color: C.muted,
  });
  textBox(s, "Layer 2の有効性を主張する前に、どの処理で崩れるかを追える実装を完成させる", 330, 546, 820, 50, {
    fontSize: 24,
    bold: true,
    color: C.navy,
  });
  note(
    s,
    "1:30–2:20",
    "先週以降の大きな更新は三つです。Chen先生から、現在の構成では問題が起きた箇所を切り分けにくいという指摘がありました。教授は三層構造を維持しつつ、Layer 2を研究の中核に置き、第I段階では性能よりも接続と追跡可能性を確認する方針を明確にしました。その方針に沿って、9月17日に3 roundを実行しました。",
    [
      "notes/advisor/sources/raw/2026-09-15/slack_2026-09-15.md",
      "notes/advisor/sources/raw/2026-09-15/Comments and suggestions_260915.pdf, pp.1–5",
      "notes/advisor/sources/raw/2026-09-16/slack_zettsu_experiment_feedback_date_unknown_received_2026-09-16.md",
      "notes/advisor/sources/raw/2026-09-16/slack_zettsu_reply_to_chen_ja_date_unknown_received_2026-09-16.md",
    ],
  );
}

// Slide 4: flowchart placeholder
{
  const s = presentation.slides.add();
  title(s, "FedPACT全体フロー", 4);
  rect(s, 82, 136, 1116, 456, "#ECEFEB", C.lineDark, 2);
  textBox(s, "ここに9月16日Slack共有版のフローチャートを挿入", 220, 300, 840, 60, {
    fontSize: 29,
    bold: true,
    color: C.muted,
    alignment: "center",
    verticalAlignment: "middle",
  });
  textBox(s, "今回の実験では、この図のLayer 1からLayer 3と次roundへのstate継承までを確認", 145, 616, 990, 32, {
    fontSize: 21,
    bold: true,
    color: C.teal,
    alignment: "center",
  });
  note(
    s,
    "2:20–3:10",
    "ここは自作の9月16日共有版フローチャートへ差し替えてください。説明では、Layer 1は複数clientのsurrogate更新をFedAvgする部分、Layer 2はclientの予測挙動をserver-side seed上でproxyとして顕在化する中核部分、Layer 3はP_{k,r}=(x̃_{k,r}, μ^{post}_{k,r})でfoundation LoRAを更新する部分、とだけ整理します。今回の実験は、この三層をまたいで次roundまでstateが正しく渡るかを確認しています。",
    [
      "notes/advisor/sources/raw/2026-09-16/slack_zettsu_flow_feedback_date_unknown_received_2026-09-16.md",
      "notes/advisor/sources/raw/2026-09-15/進捗会0911_吉村有生_3.pptx（実日付9月15日、最終承認前）",
    ],
  );
}

// Slide 5: experiment scope and corrected inputs
{
  const s = presentation.slides.add();
  title(s, "今回の実験は三層の接続確認", 5);
  textBox(s, "確認した問い", 70, 118, 220, 32, {
    fontSize: 20,
    bold: true,
    color: C.teal,
  });
  textBox(s, "各roundで処理とstateが正しくつながり、保存したログから入力と出力を追えるか", 70, 154, 570, 86, {
    fontSize: 28,
    bold: true,
    color: C.navy,
  });
  textBox(s, "2 client　各clientは normal 3本＋risk 3本", 72, 270, 545, 32, {
    fontSize: 21,
    color: C.ink,
  });
  textBox(s, "server-side seed　normal 30本＋risk 30本", 72, 316, 545, 32, {
    fontSize: 21,
    color: C.ink,
  });
  textBox(s, "各round　4 prototype、60 proxy候補、4 pair", 72, 362, 545, 32, {
    fontSize: 21,
    color: C.ink,
  });
  textBox(s, "実行　1 roundを確認後、3 round連続", 72, 408, 545, 32, {
    fontSize: 21,
    color: C.ink,
  });
  textBox(s, "第I段階：surrogate / foundation は同じ3B基盤の独立LoRA、1 run", 72, 454, 545, 30, {
    fontSize: 17,
    color: C.muted,
  });

  caption(s, "risk：time_of_alert と time_of_event の範囲", 690, 112, 490, C.orange);
  await addImage(s, path.join(workspaceDir, "artifacts", "datasets", "frames", "train", "4873a3b2179d335e", "000.jpg"), 690, 144, 236, 133, "risk映像のalert時点");
  await addImage(s, path.join(workspaceDir, "artifacts", "datasets", "frames", "train", "4873a3b2179d335e", "003.jpg"), 944, 144, 236, 133, "risk映像のevent時点");
  caption(s, "normal：対応するriskと同じ絶対時刻", 690, 302, 490, C.teal);
  await addImage(s, path.join(workspaceDir, "artifacts", "datasets", "frames", "train", "a20548830ab4cd89", "000.jpg"), 690, 334, 236, 133, "normal映像の対応時刻1");
  await addImage(s, path.join(workspaceDir, "artifacts", "datasets", "frames", "train", "a20548830ab4cd89", "003.jpg"), 944, 334, 236, 133, "normal映像の対応時刻4");
  rect(s, 70, 510, 1110, 102, C.orangePale);
  textBox(s, "旧runではriskの注釈区間内に選択frameが0枚だったため不合格とした。入力規則を修正し、全画像を確認してから再実行した。", 98, 531, 1054, 60, {
    fontSize: 23,
    bold: true,
    color: C.ink,
    alignment: "center",
    verticalAlignment: "middle",
  });
  note(
    s,
    "3:10–4:10",
    "今回の合格条件は、2 clientの局所更新からFedAvg、Layer 2、foundation更新、次roundまでをログで追えることです。性能比較は行いません。重要なのは、最初のrunでrisk区間を選べていない問題を発見し、その結果を研究上の証拠から外したことです。修正版ではriskのalertからeventまでの4時刻を使い、normalは対応するriskと同じ絶対時刻にそろえました。",
    [
      "notes/advisor/CURRENT.md（2026-09-17 01:12時点の入力問題）",
      "artifacts/datasets/fedpact_stage1/manifest_paired_alert_quartiles.jsonl",
      "artifacts/runs/fedpact_stage1/p1d_full_20260917_02/summary.json",
    ],
  );
}

// Slide 6: concrete client sample and prototype
{
  const s = presentation.slides.add();
  title(s, "具体例：局所学習でrisk方向へ0.295変化", 6);
  const frames = ["000.jpg", "001.jpg", "002.jpg", "003.jpg"];
  for (let i = 0; i < frames.length; i += 1) {
    await addImage(
      s,
      path.join(workspaceDir, "artifacts", "datasets", "frames", "train", "15a4f810cc4c2aef", frames[i]),
      70 + i * 280, 128, 250, 141,
      `risk映像の注釈区間フレーム${i + 1}`,
    );
  }
  caption(s, "time_of_alert", 70, 281, 250, C.muted);
  caption(s, "約33%", 350, 281, 250, C.muted);
  caption(s, "約67%", 630, 281, 250, C.muted);
  caption(s, "time_of_event", 910, 281, 250, C.muted);
  textBox(s, "この動画のrisk成分", 70, 330, 300, 32, {
    fontSize: 20,
    bold: true,
    color: C.teal,
  });
  const nums = [
    ["q^{pre}", "0.349"],
    ["q^{post}", "0.644"],
    ["Δq", "+0.295"],
  ];
  for (let i = 0; i < nums.length; i += 1) {
    textBox(s, nums[i][0], 72 + i * 190, 383, 150, 28, {
      fontSize: 20,
      bold: true,
      color: C.muted,
      alignment: "center",
    });
    textBox(s, nums[i][1], 72 + i * 190, 418, 150, 58, {
      fontSize: 38,
      bold: true,
      color: i === 2 ? C.orange : C.navy,
      alignment: "center",
    });
  }
  rect(s, 680, 336, 500, 190, C.bluePale);
  textBox(s, "3本のrisk動画から作ったprototype", 708, 360, 444, 32, {
    fontSize: 23,
    bold: true,
    color: C.navy,
    alignment: "center",
  });
  textBox(s, "μ^{post}_{k,r} のrisk成分　0.724\nΔμ_{k,r} のrisk成分　+0.238\nπ_{k,r}　3", 735, 410, 390, 100, {
    fontSize: 24,
    color: C.ink,
    alignment: "center",
    lineSpacing: 1.15,
  });
  rect(s, 70, 560, 1110, 60, C.paper, C.line, 1);
  textBox(s, "serverへ送るのは Q_{k,r}=(μ^{post}_{k,r}, Δμ_{k,r}, π_{k,r})。private動画、raw frame、sample IDは送らない。", 95, 575, 1060, 34, {
    fontSize: 21,
    bold: true,
    color: C.teal,
    alignment: "center",
  });
  note(
    s,
    "4:10–5:20",
    "一つの動画を具体的に追います。白い車が横から接近するrisk動画で、局所学習前のrisk確率は0.349、学習後は0.644、変化はプラス0.295でした。同じclassの3本を平均すると、μ^{post}_{k,r}のrisk成分は0.724、Δμ_{k,r}はプラス0.238、frequencyは3です。serverへ送るのはこのprototypeとモデル更新であり、private映像やsample IDではありません。",
    [
      "artifacts/runs/fedpact_stage1/p1d_layer1_20260917_01/client_local/client-0/private_trace.json（sample 15a4f810cc4c2aef）",
      "artifacts/runs/fedpact_stage1/p1d_layer1_20260917_01/communication/client-0/payload.json（r1-client-0-class1）",
      "artifacts/runs/fedpact_stage1/p1d_layer1_20260917_01/client_local/client-0/communication_audit.json",
    ],
  );
}

// Slide 7: server proxy result
{
  const s = presentation.slides.add();
  title(s, "最終出力は一致したが、変化量は再現できなかった", 7);
  caption(s, "serverが選んだ s^*_{k,r}：normal label、変換なし", 70, 110, 1140, C.teal);
  for (let i = 0; i < 4; i += 1) {
    await addImage(
      s,
      path.join(workspaceDir, "artifacts", "datasets", "frames", "train", "4d02e2a2eeb2b840", `${String(i).padStart(3, "0")}.jpg`),
      70 + i * 280, 145, 250, 141,
      `選択されたserver-side seedのフレーム${i + 1}`,
    );
  }
  textBox(s, "post behavior", 110, 338, 250, 28, {
    fontSize: 19,
    bold: true,
    color: C.muted,
    alignment: "center",
  });
  textBox(s, "target 0.724", 105, 377, 260, 48, {
    fontSize: 32,
    bold: true,
    color: C.navy,
    alignment: "center",
  });
  textBox(s, "proxy 0.724", 105, 426, 260, 42, {
    fontSize: 27,
    bold: true,
    color: C.teal,
    alignment: "center",
  });
  textBox(s, "post KL = 8.35 × 10^-10", 95, 478, 280, 28, {
    fontSize: 18,
    color: C.muted,
    alignment: "center",
  });

  rect(s, 476, 326, 2, 205, C.line);
  textBox(s, "local delta", 545, 338, 250, 28, {
    fontSize: 19,
    bold: true,
    color: C.muted,
    alignment: "center",
  });
  textBox(s, "target +0.238", 535, 377, 270, 48, {
    fontSize: 32,
    bold: true,
    color: C.navy,
    alignment: "center",
  });
  textBox(s, "proxy +0.051", 535, 426, 270, 42, {
    fontSize: 27,
    bold: true,
    color: C.red,
    alignment: "center",
  });
  textBox(s, "delta MSE = 0.0349", 530, 478, 280, 28, {
    fontSize: 18,
    color: C.muted,
    alignment: "center",
  });

  rect(s, 895, 328, 285, 205, C.orangePale);
  textBox(s, "重要な診断", 922, 348, 230, 30, {
    fontSize: 21,
    bold: true,
    color: C.orange,
    alignment: "center",
  });
  textBox(s, "最終のrisk確率だけなら一致する。\n局所学習で動いた方向と量は、現在の候補では十分に再現できていない。", 920, 397, 235, 105, {
    fontSize: 21,
    bold: true,
    color: C.ink,
    alignment: "center",
    verticalAlignment: "middle",
  });
  textBox(s, "Layer 2で評価すべき対象が、post一致だけでは足りないことを実例で確認", 100, 570, 1080, 42, {
    fontSize: 24,
    bold: true,
    color: C.teal,
    alignment: "center",
  });
  note(
    s,
    "5:20–6:30",
    "前のprototypeに対して、serverはnormal labelのseedを選びました。更新後global surrogateのrisk出力は0.724でtargetとほぼ一致しました。一方、局所学習による変化量はtargetがプラス0.238、proxy上ではプラス0.051でした。つまり、最終出力だけを近づけても、clientが学習した変化を十分に再現したとは言えません。ここがLayer 2の中心的な評価対象です。",
    [
      "artifacts/runs/fedpact_stage1/p1d_full_20260917_02/server/layer2_trace.json（r1-client-0-class1 selection）",
      "artifacts/runs/fedpact_stage1/p1d_full_20260917_02/server/proxy_pairs.json",
    ],
  );
}

// Slide 8: four prototype diagnostic table
{
  const s = presentation.slides.add();
  title(s, "round 1では、4件中1件でdeltaの方向が逆転", 8);
  const values = [
    ["prototype", "target Δrisk", "proxy Δrisk", "方向", "seed label"],
    ["client-0 / normal", "+0.094", "+0.072", "一致", "risk"],
    ["client-0 / risk", "+0.238", "+0.051", "一致", "normal"],
    ["client-1 / normal", "−0.103", "+0.056", "逆向き", "risk"],
    ["client-1 / risk", "+0.005", "+0.010", "一致", "normal"],
  ];
  const table = s.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: 90,
    top: 134,
    width: 1100,
    height: 300,
    columnWidths: [290, 210, 210, 180, 210],
    values: tableValues(values, 13, 15),
  });
  styleTable(table, { bodyFont: 20 });
  table.getCell(3, 2).fill = C.redPale;
  table.getCell(3, 3).fill = C.redPale;
  table.getCell(3, 2).text.style = { typeface: fontFamily, fontSize: 20, bold: true, color: C.red, alignment: "center", verticalAlignment: "middle", autoFit: "shrinkText" };
  table.getCell(3, 3).text.style = { typeface: fontFamily, fontSize: 20, bold: true, color: C.red, alignment: "center", verticalAlignment: "middle", autoFit: "shrinkText" };

  rect(s, 90, 470, 1100, 72, C.bluePale);
  textBox(s, "prototype class と seed label はround 1で4件すべて不一致。round 2・3では各3件が一致したため、1 roundの結果を一般化しない。", 112, 487, 1055, 42, {
    fontSize: 21,
    bold: true,
    color: C.ink,
    alignment: "center",
    verticalAlignment: "middle",
  });
  textBox(s, "原因はまだ一つに決められない。60 seedのlabelとmodel argmaxの一致は43件で、model誤分類と検索挙動が混在している。", 105, 570, 1070, 50, {
    fontSize: 22,
    bold: true,
    color: C.orange,
    alignment: "center",
  });
  note(
    s,
    "6:30–7:25",
    "round 1の4 prototypeを並べると、deltaの符号は3件で一致し、client-1のnormalだけ逆向きでした。また、prototype classと選択seed labelは4件すべて不一致でした。ただしround 2と3では各3件が一致しています。さらに、60本のseedのうちlabelと更新後global surrogateのargmaxが一致したのは43本です。原因を検索だけに帰属せず、model誤分類、検索挙動、postでtop 3へ絞ってからdeltaを比較する候補生成を分けて確認します。",
    [
      "artifacts/runs/fedpact_stage1/p1d_full_20260917_02/server/layer2_trace.json",
      "artifacts/reports/fedpact_stage1_20260917/slack_report.md（診断上の観測事実）",
    ],
  );
}

// Slide 9: three-round state and remaining problem
{
  const s = presentation.slides.add();
  title(s, "3 roundの状態継承は確認、Layer 3に診断点が残った", 9);
  const values = [
    ["round", "入力 M_G", "出力 M_G", "client LoRA update norm", "FedAvg再計算差", "各roundの4 pair上の加重KL"],
    ["1", "M_G^0", "M_G^1", "0.346 / 0.292", "5.82 × 10^-11", "0.0540 から 0.0534"],
    ["2", "M_G^1", "M_G^2", "0.323 / 0.280", "5.82 × 10^-11", "18.478 から 18.339"],
    ["3", "M_G^2", "M_G^3", "0.320 / 0.280", "1.16 × 10^-10", "16.131 から 16.178"],
  ];
  const table = s.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: 55,
    top: 134,
    width: 1170,
    height: 255,
    columnWidths: [90, 135, 135, 255, 220, 335],
    values: tableValues(values, 12, 13.5),
  });
  styleTable(table, { bodyFont: 18 });
  table.getCell(3, 5).fill = C.redPale;
  table.getCell(3, 5).text.style = { typeface: fontFamily, fontSize: 18, bold: true, color: C.red, alignment: "center", verticalAlignment: "middle", autoFit: "shrinkText" };

  textBox(s, "各roundでpairが異なるため、KLをround間の性能trendとして比較しない", 120, 398, 1040, 24, {
    fontSize: 16,
    color: C.muted,
    alignment: "center",
  });

  rect(s, 70, 430, 525, 142, C.bluePale);
  textBox(s, "確認できたこと", 96, 450, 470, 28, {
    fontSize: 22,
    bold: true,
    color: C.teal,
  });
  textBox(s, "M_G^0 から M_G^3まで更新し、round間のcheckpointとtensor digestが一致。各client更新とFedAvgを再計算できた。", 96, 492, 465, 62, {
    fontSize: 21,
    color: C.ink,
  });
  rect(s, 625, 430, 585, 142, C.orangePale);
  textBox(s, "残った問題", 652, 450, 530, 28, {
    fontSize: 22,
    bold: true,
    color: C.orange,
  });
  textBox(s, "加重KLを減らす条件をround 3で満たせず、実行確認として更新した。処理は動いたが、安定したfoundation更新とは言えない。", 652, 492, 525, 62, {
    fontSize: 21,
    color: C.ink,
  });
  textBox(s, "この結果から性能向上、proxy品質、未知データへの転移、形式的privacy保証は主張しない", 100, 614, 1080, 34, {
    fontSize: 22,
    bold: true,
    color: C.red,
    alignment: "center",
  });
  note(
    s,
    "7:25–8:45",
    "3 roundを通して、global surrogateとfoundationのcheckpointおよびtensor digestがround間で一致し、state継承を確認しました。client更新は毎round非ゼロで、FedAvgの独立再計算差も10のマイナス10程度でした。一方、round 3では学習率10のマイナス6から10のマイナス11まで、同じ4 pair上の加重KLを減らす条件を満たせませんでした。第I段階の実行確認として更新は完了しましたが、安定した知識転移の証拠にはしません。",
    [
      "artifacts/reports/fedpact_stage1_20260917/three_round_state_lineage.md",
      "artifacts/reports/fedpact_stage1_20260917/three_round_state_lineage.json",
      "artifacts/runs/fedpact_stage1/p1d_round3_full_20260917_02/summary.json",
    ],
  );
}

// Slide 10: next steps
{
  const s = presentation.slides.add();
  title(s, "今後：5 roundの前に、失敗箇所を切り分ける", 10);
  const items = [
    {
      n: "1",
      h: "Layer 3の更新を再確認",
      b: "同じ4 pairと同じ損失で、gradient、重み、FP32計算、finite differenceを照合する。十分小さい1回更新で損失が増えないことを合格条件にする。",
      color: C.red,
    },
    {
      n: "2",
      h: "問題が解消したら5 roundへ",
      b: "M_Gとfoundationのstate継承、client送信内容、実行時間を同じ形式で保存する。",
      color: C.orange,
    },
    {
      n: "3",
      h: "第II段階でLayer 2を比較",
      b: "Server Only、Post Only、Post + Deltaを同じ条件で比較し、seed内容、model予測、delta再現を分けて評価する。",
      color: C.teal,
    },
  ];
  for (let i = 0; i < items.length; i += 1) {
    const y = 126 + i * 155;
    textBox(s, items[i].n, 76, y, 60, 60, {
      fontSize: 34,
      bold: true,
      color: "#FFFFFF",
      fill: items[i].color,
      alignment: "center",
      verticalAlignment: "middle",
    });
    textBox(s, items[i].h, 166, y - 2, 940, 40, {
      fontSize: 27,
      bold: true,
      color: C.navy,
    });
    textBox(s, items[i].b, 166, y + 47, 980, 78, {
      fontSize: 21,
      color: C.ink,
      lineSpacing: 1.1,
    });
    if (i < items.length - 1) rect(s, 166, y + 134, 1000, 1.5, C.line);
  }
  rect(s, 70, 597, 1110, 58, C.navy);
  textBox(s, "次の問い：clientのadaptation signalは、server-side seedだけの場合を超えてfoundation更新へ寄与するか", 96, 611, 1058, 32, {
    fontSize: 22,
    bold: true,
    color: "#FFFFFF",
    alignment: "center",
    verticalAlignment: "middle",
  });
  note(
    s,
    "8:45–10:00",
    "次は5 roundへ直ちに進まず、round 3のfoundation更新を最小条件で切り分けます。これが解消した後に5 roundのstate継承を確認します。その後、第II段階でServer Only、Post Only、Post + Deltaを同じ条件で比較します。最終評価には、局所学習、prototype作成、server検索、条件選択に使っていない評価用データを用います。今週の到達点は、3 roundを通し、次に見るべき場所をLayer 2のdelta再現とLayer 3の更新へ絞れたことです。",
    [
      "notes/advisor/sources/raw/2026-09-16/slack_zettsu_experiment_feedback_date_unknown_received_2026-09-16.md",
      "notes/advisor/sources/raw/2026-09-15/Comments and suggestions_260915.pdf, p.4",
      "artifacts/reports/fedpact_stage1_20260917/three_round_state_lineage.md",
    ],
  );
}

const draftPath = path.join(buildDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(draftPath);

const { finalizePresentation } = await import(
  pathToFileURL(path.join(skillDir, "container_tools", "artifact_tool_utils.mjs")).href,
);
const result = await finalizePresentation({
  workspaceDir,
  candidatePath: draftPath,
  finalPath,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-heading-fit",
    "--require-native-table-slide", "8",
    "--require-native-table-slide", "9",
  ],
  explicitTotalSlideCount: 10,
  requiredNativeTableOwnerSlides: [8, 9],
  requiredNativeChartOwnerSlides: [],
  fontPolicy: {
    basis: "design",
    families: [fontFamily],
  },
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, "FedPACT_研究会_10分_20260918_v11.validation.json"),
});

console.log(JSON.stringify({ finalPath, draftPath, result }, null, 2));
