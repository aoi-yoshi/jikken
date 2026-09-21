import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "C:\\Python\\実験_修正";
const skillDir = "C:\\Users\\aoi7y\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.909.12148\\skills\\presentations";
const runtimePython = "C:\\Users\\aoi7y\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";
const buildDir = path.join(workspaceDir, "tmp", "presentations", "fedpact_0918_8slides");
const outputDir = path.join(workspaceDir, "artifacts", "presentations");
const finalPath = path.join(outputDir, "FedPACT_研究会_10分_9枚_付録3枚_20260918_v20.pptx");
const fontFamily = "Yu Gothic";
const mainSlideTotal = 9;
const appendixSlideTotal = 3;

const C = {
  bg: "#F7F5EF",
  paper: "#FFFFFF",
  navy: "#173B4B",
  teal: "#147B78",
  aqua: "#69C3BE",
  bluePale: "#E6F1F3",
  greenPale: "#DDF0ED",
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

const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

function rect(slide, left, top, width, height, fill, lineFill = "none", lineWidth = 0) {
  return slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill,
    line: { fill: lineFill, width: lineWidth },
  });
}

function circle(slide, left, top, size, fill, lineFill = "none", lineWidth = 0) {
  return slide.shapes.add({
    geometry: "ellipse",
    position: { left, top, width: size, height: size },
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

function title(slide, heading, number, opts = {}) {
  const numberLabel = opts.appendix ? `A${number}` : String(number).padStart(2, "0");
  const footerLabel = opts.appendix ? `付録 ${number} / ${appendixSlideTotal}` : `${number} / ${mainSlideTotal}`;
  rect(slide, 0, 0, 1280, 720, C.bg);
  rect(slide, 40, 36, 7, 54, C.teal);
  textBox(slide, heading, 62, 34, 1110, 62, {
    fontSize: 38,
    bold: true,
    color: C.navy,
    verticalAlignment: "middle",
  });
  textBox(slide, numberLabel, 1187, 45, 45, 28, {
    fontSize: 16,
    bold: true,
    color: C.teal,
    alignment: "right",
  });
  rect(slide, 48, 680, 1184, 1.5, C.line);
  textBox(slide, "FedPACT / NEXAR　研究会　2026.09.18", 52, 687, 760, 18, {
    fontSize: 12,
    color: C.muted,
  });
  textBox(slide, footerLabel, 1100, 687, 126, 18, {
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

function caption(slide, label, left, top, width, color = C.muted, align = "left") {
  return textBox(slide, label, left, top, width, 24, {
    fontSize: 15,
    bold: true,
    color,
    alignment: align,
  });
}

function note(slide, timing, script, sources) {
  slide.speakerNotes.textFrame.setText(
    `【目安 ${timing}】\n${script}\n\n【根拠】\n${sources.join("\n")}`,
  );
}

function tableValues(rawValues, headerPt = 12, bodyPt = 15) {
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

function styleTable(table, bodyFont = 18) {
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  for (let r = 0; r < table.rows.length; r += 1) {
    for (let c = 0; c < table.columns.length; c += 1) {
      const cell = table.getCell(r, c);
      cell.fill = r === 0 ? C.bluePale : (r % 2 === 0 ? "#F0F4F3" : C.paper);
      cell.text.style = {
        typeface: fontFamily,
        fontSize: r === 0 ? 16 : bodyFont,
        bold: r === 0,
        color: r === 0 ? C.navy : C.ink,
        alignment: c === 0 ? "left" : "center",
        verticalAlignment: "middle",
        autoFit: "shrinkText",
        insets: { left: 7, right: 7, top: 4, bottom: 4 },
      };
    }
  }
}

// 1. Use case, value, gap, and the role of FedPACT.
{
  const s = presentation.slides.add();
  title(s, "複数の現場で得た経験を、中央側AIへ生かす", 1);

  await addImage(
    s,
    path.join(workspaceDir, "artifacts", "datasets", "frames", "train", "003d6c3909", "003.jpg"),
    62, 122, 260, 146,
    "昼間の市街地を走るNEXAR映像",
  );
  await addImage(
    s,
    path.join(workspaceDir, "artifacts", "datasets", "frames", "train", "b82e955a9fdf13e2", "003.jpg"),
    338, 122, 260, 146,
    "夜間の道路を走るNEXAR映像",
  );
  caption(s, "現場A　昼間・市街地", 62, 278, 260, C.teal, "center");
  caption(s, "現場B　夜間・道路", 338, 278, 260, C.teal, "center");
  textBox(s, "同じ危険判定でも、場所・時間・天候により経験する動画が異なる", 70, 318, 520, 70, {
    fontSize: 24,
    bold: true,
    color: C.navy,
    alignment: "center",
    verticalAlignment: "middle",
  });
  textBox(s, "価値", 74, 410, 72, 28, { fontSize: 19, bold: true, color: C.teal });
  textBox(s, "各現場が学んだ判定の変化を、サービス全体の更新に使う", 74, 445, 510, 66, {
    fontSize: 25,
    bold: true,
    color: C.ink,
  });

  rect(s, 636, 122, 2, 408, C.line);
  textBox(s, "現在の接続問題", 674, 122, 240, 34, {
    fontSize: 22,
    bold: true,
    color: C.orange,
  });
  textBox(s, "連合平均", 680, 188, 140, 48, {
    fontSize: 21,
    bold: true,
    color: C.navy,
    fill: C.bluePale,
    alignment: "center",
    verticalAlignment: "middle",
  });
  textBox(s, "同じ構造の小型AI同士はまとめられる", 842, 188, 330, 52, {
    fontSize: 18,
    color: C.ink,
    verticalAlignment: "middle",
  });
  textBox(s, "しかし", 681, 264, 94, 30, { fontSize: 18, bold: true, color: C.red });
  textBox(s, "構造や規模が異なる中央側AIへ、更新値をそのまま渡せない", 681, 302, 480, 78, {
    fontSize: 25,
    bold: true,
    color: C.navy,
  });
  textBox(s, "現場動画を中央へ集めない条件では、共通の入力を使った橋渡しも難しい", 681, 406, 480, 64, {
    fontSize: 20,
    color: C.muted,
  });
  rect(s, 675, 495, 500, 72, C.orangePale);
  textBox(s, "FedPACT：現場で生じた判定の変化を、中央側の動画上に表して学習へ使う", 700, 511, 450, 43, {
    fontSize: 22,
    bold: true,
    color: C.ink,
    alignment: "center",
    verticalAlignment: "middle",
  });

  textBox(s, "発表の順番", 66, 589, 150, 25, { fontSize: 17, bold: true, color: C.muted });
  const stages = ["変化を測る", "複数動画を要約", "中央側の動画上に表す", "中央側AIを更新"];
  for (let i = 0; i < stages.length; i += 1) {
    const x = 224 + i * 248;
    circle(s, x, 579, 42, i === 2 ? C.orange : C.teal);
    textBox(s, String(i + 1), x, 584, 42, 30, {
      fontSize: 19,
      bold: true,
      color: "#FFFFFF",
      alignment: "center",
      verticalAlignment: "middle",
    });
    textBox(s, stages[i], x + 54, 583, 175, 36, {
      fontSize: 18,
      bold: true,
      color: C.navy,
      verticalAlignment: "middle",
    });
    if (i < stages.length - 1) rect(s, x + 213, 599, 28, 2, C.lineDark);
  }
  textBox(s, "対象は共通タスク上の予測挙動。映像の意味内容全体や危険原因の説明は対象外", 90, 638, 1100, 28, {
    fontSize: 18,
    bold: true,
    color: C.teal,
    alignment: "center",
  });
  note(
    s,
    "0:00–1:10",
    "今回は、複数の自動運転車やロボットが、同じ危険判定を行う場面を想定しています。判定内容は共通でも、場所や時間、天候が違うため、それぞれが経験する動画は異なります。ある現場で得た経験をサービス全体へ反映できれば、中央側のAIも継続的に更新できます。しかし、現場の動画をそのまま中央へ集めることは難しく、連合平均でまとめた小型AIの更新を、構造や規模が異なる中央側AIへ直接渡すこともできません。そこでFedPACTでは、現場でAIの判定がどのように変わったかを数値で受け取り、その変化を中央側が持つ動画上に表して学習へ使います。この後は、変化を測る、複数動画をまとめる、中央側の動画上に表す、中央側AIを更新する、という順番で説明します。",
    [
      "notes/advisor/sources/raw/2026-09-16/slack_zettsu_reply_to_chen_ja_date_unknown_received_2026-09-16.md（利用場面、接続問題、適用範囲）",
      "notes/advisor/sources/raw/2026-09-01/FedPACT手法提案260901.pdf（研究目的、三層構造）",
      "画像: artifacts/datasets/frames/train/003d6c3909/003.jpg, artifacts/datasets/frames/train/b82e955a9fdf13e2/003.jpg",
    ],
  );
}

// 2. One real video: qpre, qpost and delta.
{
  const s = presentation.slides.add();
  title(s, "1本の動画で、学習前後の判定変化を測る", 2);
  const sampleDir = path.join(workspaceDir, "artifacts", "datasets", "frames", "train", "15a4f810cc4c2aef");
  const labels = ["危険区間の開始", "約33%", "約67%", "危険事象の時点"];
  for (let i = 0; i < 4; i += 1) {
    await addImage(s, path.join(sampleDir, `${String(i).padStart(3, "0")}.jpg`), 58 + i * 296, 122, 270, 152, `白い車が横から接近するrisk動画のフレーム${i + 1}`);
    caption(s, labels[i], 58 + i * 296, 285, 270, i === 3 ? C.orange : C.muted, "center");
  }
  textBox(s, "q = [通常の確率, 危険の確率]　この例では危険の確率だけを表示", 76, 325, 1128, 30, {
    fontSize: 19,
    bold: true,
    color: C.teal,
    alignment: "center",
  });
  const vals = [
    { label: "学習前  qᵖʳᵉ", value: "0.349", sub: "局所学習前の危険確率", color: C.navy },
    { label: "学習後  qᵖᵒˢᵗ", value: "0.644", sub: "同じ動画を学習後に再判定", color: C.navy },
    { label: "変化量  Δq", value: "+0.295", sub: "危険側へ動いた量", color: C.orange },
  ];
  for (let i = 0; i < vals.length; i += 1) {
    const x = 98 + i * 390;
    textBox(s, vals[i].label, x, 385, 300, 32, {
      fontSize: 20,
      bold: true,
      color: C.muted,
      alignment: "center",
    });
    textBox(s, vals[i].value, x, 424, 300, 68, {
      fontSize: 42,
      bold: true,
      color: vals[i].color,
      alignment: "center",
    });
    textBox(s, vals[i].sub, x, 493, 300, 36, {
      fontSize: 18,
      color: C.ink,
      alignment: "center",
    });
    if (i < vals.length - 1) rect(s, x + 340, 393, 2, 135, C.line);
  }
  rect(s, 74, 569, 1132, 70, C.bluePale);
  textBox(s, "ここで分かるのは、同じ入力への判定が危険側へ動いたことまで。危険理由や性能向上はまだ分からない。", 100, 582, 1080, 48, {
    fontSize: 20,
    bold: true,
    color: C.ink,
    alignment: "center",
    verticalAlignment: "middle",
  });
  note(
    s,
    "1:10–2:10",
    "ここでは、白い車が横から接近する危険動画を例にします。この実験では、AIが出した通常と危険の確率を記録しています。q^{pre}は局所学習前、q^{post}は局所学習後に同じ動画を判定した結果です。この動画では、危険確率が0.349から0.644へ変わり、Δqはプラス0.295でした。つまり、この入力に対するAIの出力が危険側へ動いたことが分かります。ただし、この値だけから危険を正しく学習した、あるいは危険原因を理解したとは言えません。次に、このような1本ごとの反応を複数動画からまとめます。",
    [
      "artifacts/runs/fedpact_stage1/p1d_layer1_20260917_01/client_local/client-0/private_trace.json（sample 15a4f810cc4c2aef）",
      "artifacts/datasets/fedpact_stage1/manifest_paired_alert_quartiles.jsonl（危険区間、4フレーム）",
    ],
  );
  rect(s, 40, 36, 7, 54, C.teal);
}

// 3. Three videos form Q.
{
  const s = presentation.slides.add();
  title(s, "複数動画の反応を、3つの値に要約する", 3);
  const samples = [
    { id: "4873a3b2179d335e", post: "0.600", delta: "+0.336" },
    { id: "15a4f810cc4c2aef", post: "0.644", delta: "+0.295" },
    { id: "4b013e9c446fbc83", post: "0.929", delta: "+0.082" },
  ];
  textBox(s, "同じ現場側AIの危険動画3本", 64, 112, 630, 30, { fontSize: 21, bold: true, color: C.teal });
  for (let i = 0; i < samples.length; i += 1) {
    const x = 64 + i * 210;
    await addImage(s, path.join(workspaceDir, "artifacts", "datasets", "frames", "train", samples[i].id, "003.jpg"), x, 154, 190, 107, `client-0のrisk動画${i + 1}`);
    textBox(s, `学習後 危険 ${samples[i].post}\n変化量 ${samples[i].delta}`, x, 272, 190, 53, {
      fontSize: 17,
      bold: true,
      color: C.ink,
      alignment: "center",
    });
  }
  rect(s, 271, 151, 196, 113, "none", C.orange, 3);
  textBox(s, "2枚目の動画", 278, 158, 92, 22, {
    fontSize: 12,
    bold: true,
    color: "#FFFFFF",
    fill: C.orange,
    alignment: "center",
    verticalAlignment: "middle",
  });
  rect(s, 710, 118, 2, 260, C.line);
  textBox(s, "3本を平均して作る Qₖ,ᵣ", 750, 118, 450, 34, {
    fontSize: 25,
    bold: true,
    color: C.navy,
    alignment: "center",
  });
  textBox(s, "μᵖᵒˢᵗₖ,ᵣ", 770, 181, 180, 30, { fontSize: 21, bold: true, color: C.muted, alignment: "center" });
  textBox(s, "0.724", 770, 218, 180, 56, { fontSize: 38, bold: true, color: C.navy, alignment: "center" });
  textBox(s, "学習後の危険確率", 770, 278, 180, 26, { fontSize: 16, color: C.muted, alignment: "center" });
  textBox(s, "Δμₖ,ᵣ", 990, 181, 180, 30, { fontSize: 21, bold: true, color: C.muted, alignment: "center" });
  textBox(s, "+0.238", 990, 218, 180, 56, { fontSize: 38, bold: true, color: C.orange, alignment: "center" });
  textBox(s, "平均した変化量", 990, 278, 180, 26, { fontSize: 16, color: C.muted, alignment: "center" });
  textBox(s, "πₖ,ᵣ = 3　まとめた動画の本数", 770, 324, 400, 38, {
    fontSize: 22,
    bold: true,
    color: C.teal,
    alignment: "center",
  });
  textBox(s, "k：現場側AI　　r：通常・危険の動画群", 765, 369, 410, 24, {
    fontSize: 15,
    color: C.muted,
    alignment: "center",
  });

  rect(s, 64, 400, 1140, 1.5, C.lineDark);
  textBox(s, "中央側へ送る", 90, 435, 240, 30, { fontSize: 22, bold: true, color: C.teal });
  textBox(s, "小型AIの更新情報", 92, 485, 300, 48, {
    fontSize: 22,
    bold: true,
    color: C.navy,
    fill: C.bluePale,
    alignment: "center",
    verticalAlignment: "middle",
  });
  textBox(s, "Qₖ,ᵣ = ( μᵖᵒˢᵗₖ,ᵣ,  Δμₖ,ᵣ,  πₖ,ᵣ )", 420, 485, 490, 48, {
    fontSize: 22,
    bold: true,
    color: C.navy,
    fill: C.greenPale,
    alignment: "center",
    verticalAlignment: "middle",
  });
  textBox(s, "中央側へ送らない", 930, 435, 240, 30, { fontSize: 22, bold: true, color: C.red });
  textBox(s, "元動画・画像・動画ごとの識別情報", 930, 485, 250, 52, {
    fontSize: 20,
    bold: true,
    color: C.red,
    alignment: "center",
    verticalAlignment: "middle",
  });
  textBox(s, "小型AIの更新は連合平均へ、Qₖ,ᵣは代理データ生成へ。Qₖ,ᵣ自体は平均しない。", 100, 587, 1080, 40, {
    fontSize: 22,
    bold: true,
    color: C.ink,
    alignment: "center",
  });
  note(
    s,
    "2:10–3:10",
    "1本の動画の結果をそのまま中央側へ送るわけではありません。先ほどの動画を含む危険動画3本について、学習後の危険確率を平均すると0.724になります。これがμ^{post}_{k,r}です。学習前後の変化量の平均はプラス0.238で、これがΔμ_{k,r}です。まとめた本数π_{k,r}は3です。この三つをQ_{k,r}として送ります。中央側へ送るのは、この要約と小型AIの更新情報です。元の動画、画像、動画ごとの識別情報は送りません。また、小型AIの更新は連合平均へ渡しますが、Q_{k,r}は平均せず、次の代理データ生成へ渡します。",
    [
      "artifacts/runs/fedpact_stage1/p1d_layer1_20260917_01/client_local/client-0/private_trace.json",
      "artifacts/runs/fedpact_stage1/p1d_layer1_20260917_01/communication/client-0/payload.json（r1-client-0-class1）",
      "notes/advisor/sources/raw/2026-09-16/slack_zettsu_flow_feedback_date_unknown_received_2026-09-16.md（Q生成と別経路）",
    ],
  );
}

// 4. Server-side seed search, transformations, x-tilde and P.
{
  const s = presentation.slides.add();
  title(s, "中央側の動画から、代理データを選ぶ", 4);
  textBox(s, "入力", 64, 112, 120, 28, { fontSize: 19, bold: true, color: C.muted });
  textBox(s, "Qₖ,ᵣ", 66, 153, 180, 54, {
    fontSize: 27,
    bold: true,
    color: C.navy,
    fill: C.greenPale,
    alignment: "center",
    verticalAlignment: "middle",
  });
  textBox(s, "危険 0.724\n変化 +0.238", 66, 220, 180, 66, {
    fontSize: 20,
    color: C.ink,
    alignment: "center",
  });
  rect(s, 269, 181, 52, 3, C.aqua);
  textBox(s, "中央側の候補動画 60本", 338, 123, 260, 35, { fontSize: 23, bold: true, color: C.navy, alignment: "center" });
  textBox(s, "通常30本　危険30本", 338, 162, 260, 28, { fontSize: 18, color: C.muted, alignment: "center" });
  await addImage(s, path.join(workspaceDir, "artifacts", "datasets", "frames", "train", "4d02e2a2eeb2b840", "003.jpg"), 346, 208, 245, 138, "検索で1位になった中央側動画");
  caption(s, "検索1位：正解ラベルは通常", 346, 355, 245, C.orange, "center");
  rect(s, 614, 181, 52, 3, C.aqua);
  textBox(s, "上位3本を5通りに変換", 684, 123, 505, 35, { fontSize: 23, bold: true, color: C.navy, alignment: "center" });
  textBox(s, "1つのQにつき15候補　4つのQで合計60候補", 684, 162, 505, 28, { fontSize: 18, color: C.muted, alignment: "center" });

  const transBase = path.join(workspaceDir, "artifacts", "runs", "fedpact_stage1", "p1d_full_20260917_02", "server", "transformation_candidates");
  const transforms = [
    ["r1-client-0-class1-rank01-00-identity", "変更なし"],
    ["r1-client-0-class1-rank01-01-brightness", "暗くする"],
    ["r1-client-0-class1-rank01-02-brightness", "明るくする"],
    ["r1-client-0-class1-rank01-03-contrast", "コントラスト"],
    ["r1-client-0-class1-rank01-04-center_crop", "中央を切り出す"],
  ];
  for (let i = 0; i < transforms.length; i += 1) {
    const x = 667 + i * 111;
    await addImage(s, path.join(transBase, transforms[i][0], "frame_03.png"), x, 208, 102, 72, `中央側動画の変換候補: ${transforms[i][1]}`);
    if (i === 0) rect(s, x - 3, 205, 108, 78, "none", C.orange, 3);
    textBox(s, transforms[i][1], x - 4, 291, 110, 38, {
      fontSize: 14,
      bold: i === 0,
      color: i === 0 ? C.orange : C.muted,
      alignment: "center",
    });
  }

  rect(s, 64, 403, 1140, 1.5, C.lineDark);
  textBox(s, "選択基準", 70, 430, 160, 28, { fontSize: 21, bold: true, color: C.teal });
  textBox(s, "① 学習後の判定が 0.724 に近い", 80, 478, 350, 48, {
    fontSize: 22,
    bold: true,
    color: C.navy,
    fill: C.bluePale,
    alignment: "center",
    verticalAlignment: "middle",
  });
  textBox(s, "② 連合平均前後の変化が +0.238 に近い", 462, 478, 420, 48, {
    fontSize: 22,
    bold: true,
    color: C.navy,
    fill: C.orangePale,
    alignment: "center",
    verticalAlignment: "middle",
  });
  rect(s, 902, 501, 45, 3, C.aqua);
  textBox(s, "Pₖ,ᵣ = ( x̃ₖ,ᵣ,\nμᵖᵒˢᵗₖ,ᵣ )", 966, 456, 238, 92, {
    fontSize: 22,
    bold: true,
    color: C.navy,
    fill: C.greenPale,
    alignment: "center",
    verticalAlignment: "middle",
  });
  textBox(s, "x̃ₖ,ᵣ：選ばれた変換後動画", 966, 555, 238, 24, {
    fontSize: 15,
    color: C.muted,
    alignment: "center",
  });
  textBox(s, "変化量は候補選択に使い、πₖ,ᵣは学習時の重みに使う。教師値は学習後の判定 0.724", 110, 600, 1060, 42, {
    fontSize: 21,
    bold: true,
    color: C.ink,
    alignment: "center",
  });
  note(
    s,
    "3:10–4:35",
    "ここでも、2枚目と3枚目で追った同じr1-client-0-class1を使います。中央側には通常動画30本と危険動画30本、合計60本を用意しています。これは現場の知識そのものではなく、現場側の判定変化を表すための材料です。まず、連合平均後の小型AIを使い、μ^{post}_{k,r}に近い判定を出す動画を探します。上位3本について、変更なし、暗くする、明るくする、コントラストを上げる、中央を切り出す、という5種類を試します。一つのQにつき15候補で、四つのQでは合計60候補です。候補を選ぶときは、学習後の判定だけでなく、連合平均前後の変化がΔμ_{k,r}に近いかも確認します。選んだ入力x̃_{k,r}と学習後の目標値μ^{post}_{k,r}からP_{k,r}を作ります。",
    [
      "artifacts/runs/fedpact_stage1/p1d_full_20260917_02/server/layer2_trace.json（60 seed、top 3、5変換、選択）",
      "artifacts/runs/fedpact_stage1/p1d_full_20260917_02/server/proxy_pairs.json（Pの構成）",
      "画像: artifacts/runs/fedpact_stage1/p1d_full_20260917_02/server/transformation_candidates/r1-client-0-class1-rank01-*",
    ],
  );
  rect(s, 40, 36, 7, 54, C.teal);
}

// 5. Experimental settings used for the formal three-round run.
{
  const s = presentation.slides.add();
  title(s, "今回の実験設定", 5);
  textBox(s, "第I段階の目的", 72, 112, 190, 30, { fontSize: 21, bold: true, color: C.teal });
  textBox(s, "3層の実行・接続と、次のラウンドへの状態継承を確認", 278, 107, 910, 42, {
    fontSize: 25,
    bold: true,
    color: C.navy,
    verticalAlignment: "middle",
  });

  const values = [
    ["区分", "設定", "区分", "設定"],
    ["予測課題", "NEXAR動画の通常／危険の二値判定", "ラウンド", "3回"],
    ["現場側", "2台、各6本（通常3・危険3）", "中央側候補", "60本（通常30・危険30）"],
    ["データ分割", "現場側12本・中央側60本・確認用2本・最終評価用2本は動画単位で非重複", "反復条件", "乱数の初期値42の1実行。3ラウンドで同じ現場動画を使用し、最適化状態は毎回初期化"],
    ["入力画像", "1動画4枚。危険区間の4時刻と、対応する通常動画の同じ4時刻", "要約", "softmax確率（T=1）を各台・各判定区分で平均し、4つのQₖ,ᵣ"],
    ["使用モデル", "現場側・中央側とも Qwen2.5-VL-3B-Instruct", "更新対象", "現場側：軽量更新部分（LoRA）＋分類層\n中央側：軽量更新部分（LoRA）のみ"],
    ["現場側学習", "AdamW、学習率 1×10⁻⁴、1回に1動画、6回更新", "軽量更新", "r=8、α=16、dropout=0"],
    ["連合平均", "各台6本のため重みは0.5ずつ", "代理データ", "正解ラベルで制限せず60本を検索し、上位3本×5変換から4つのPₖ,ᵣを選択"],
  ];
  const table = s.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: 66,
    top: 168,
    width: 1150,
    height: 374,
    columnWidths: [132, 452, 142, 424],
    values: tableValues(values, 11.5, 13.5),
  });
  styleTable(table, 16);
  for (let r = 1; r < table.rows.length; r += 1) {
    table.getCell(r, 0).text.style = { typeface: fontFamily, fontSize: 16, bold: true, color: C.teal, alignment: "left", verticalAlignment: "middle", autoFit: "shrinkText", insets: { left: 8, right: 6, top: 4, bottom: 4 } };
    table.getCell(r, 2).text.style = { typeface: fontFamily, fontSize: 16, bold: true, color: C.teal, alignment: "left", verticalAlignment: "middle", autoFit: "shrinkText", insets: { left: 8, right: 6, top: 4, bottom: 4 } };
  }

  rect(s, 72, 603, 1136, 50, C.orangePale);
  textBox(s, "今回の合格条件は実行・接続・状態継承。性能向上と異なる規模のAI間の転移は未確認", 96, 612, 1088, 30, {
    fontSize: 21,
    bold: true,
    color: C.red,
    alignment: "center",
    verticalAlignment: "middle",
  });
  note(
    s,
    "4:35–5:15",
    "ここで、後半の結果を読むための実験条件を整理します。今回はNEXAR動画の通常と危険の二値判定を扱い、現場側AIを二台、三ラウンドで動かしました。各台は通常三本、危険三本を使い、一動画から四枚を入力します。危険動画は警告から事象までの区間を四分割し、通常動画も対応する同じ時刻を使いました。中央側には通常三十本と危険三十本、合計六十本を用意しました。各ラウンドでは四つのQ_{k,r}を作り、六十本を検索した後、上位三本を五通りに変換して四つのP_{k,r}を選びます。第I段階なので現場側と中央側は同じ3Bモデルです。今回の結果は接続確認であり、性能向上や異なる規模のAI間の転移を示すものではありません。",
    [
      "config/fedpact_stage1.yaml",
      "artifacts/runs/fedpact_stage1/p1d_full_20260917_02/summary.json",
      "artifacts/reports/fedpact_stage1_20260917/three_round_state_lineage.json",
    ],
  );
  rect(s, 40, 36, 7, 54, C.teal);
}

// 6. Flowchart placeholder for the user's 9/16 diagram.
{
  const s = presentation.slides.add();
  title(s, "FedPACT全体フローと今回の実験位置", 6);
  rect(s, 74, 120, 1132, 440, "#ECEFEB", C.lineDark, 2);
  textBox(s, "ここに9月16日Slack共有版の\nフローチャートを挿入", 235, 284, 810, 96, {
    fontSize: 31,
    bold: true,
    color: C.muted,
    alignment: "center",
    verticalAlignment: "middle",
  });
  textBox(s, "差し替え時の確認", 94, 138, 200, 28, { fontSize: 18, bold: true, color: C.teal });
  textBox(s, "小型AIの更新と Qₖ,ᵣ を別経路で表示　／　Layer 2を今回の中心として表示", 306, 140, 850, 24, {
    fontSize: 17,
    color: C.muted,
    alignment: "right",
  });
  textBox(s, "今回の確認範囲", 82, 592, 190, 28, { fontSize: 20, bold: true, color: C.teal });
  const flow = ["Layer 1\n2台の小型AIを連合平均", "Layer 2\n代理データを生成", "Layer 3\n中央側AIを更新", "次のラウンド\n状態を引き継ぐ"];
  for (let i = 0; i < flow.length; i += 1) {
    const x = 282 + i * 235;
    textBox(s, flow[i], x, 578, 195, 64, {
      fontSize: 18,
      bold: true,
      color: i === 1 ? C.orange : C.navy,
      alignment: "center",
      verticalAlignment: "middle",
    });
    if (i < flow.length - 1) rect(s, x + 202, 609, 25, 2, C.aqua);
  }
  textBox(s, "合格条件：3層と次のラウンドが正しくつながり、入力・出力・状態をログから追えること", 100, 646, 1080, 26, {
    fontSize: 18,
    bold: true,
    color: C.ink,
    alignment: "center",
  });
  note(
    s,
    "5:15–5:50",
    "ここまでの処理を全体図の中で整理します。第1層では二つの現場側AIが局所学習を行い、小型AIの更新情報を連合平均します。一方、判定の変化をまとめたQ_{k,r}は別経路で第2層へ渡します。第2層が本研究の中心で、Q_{k,r}を中央側が持つ動画上の代理データとして表します。第3層ではP_{k,r}を使って中央側AIを更新し、その状態を次のラウンドへ引き継ぎます。今回の合格は処理の接続に対するものであり、性能の良さを示すものではありません。",
    [
      "notes/advisor/sources/raw/2026-09-16/slack_zettsu_flow_feedback_date_unknown_received_2026-09-16.md",
      "notes/advisor/sources/raw/2026-09-16/slack_zettsu_experiment_feedback_date_unknown_received_2026-09-16.md",
    ],
  );
}

// 7. Concrete proxy result.
{
  const s = presentation.slides.add();
  title(s, "学習後の判定は一致したが、変化量には差が残った", 7);
  const proxyDir = path.join(workspaceDir, "artifacts", "runs", "fedpact_stage1", "p1d_full_20260917_02", "server", "transformation_candidates", "r1-client-0-class1-rank01-00-identity");
  for (let i = 0; i < 4; i += 1) {
    await addImage(s, path.join(proxyDir, `frame_0${i}.png`), 58 + i * 296, 122, 270, 152, `選ばれた代理データのフレーム${i + 1}`);
  }
  caption(s, "4枚目で選んだ中央側動画：正解ラベルは通常、変換なし", 58, 284, 1158, C.orange, "center");

  textBox(s, "学習後の危険確率", 90, 350, 470, 34, { fontSize: 23, bold: true, color: C.teal, alignment: "center" });
  textBox(s, "現場側の目標", 110, 409, 180, 28, { fontSize: 18, bold: true, color: C.muted, alignment: "center" });
  textBox(s, "0.724", 110, 444, 180, 56, { fontSize: 40, bold: true, color: C.navy, alignment: "center" });
  textBox(s, "代理データ", 355, 409, 180, 28, { fontSize: 18, bold: true, color: C.muted, alignment: "center" });
  textBox(s, "0.724", 355, 444, 180, 56, { fontSize: 40, bold: true, color: C.teal, alignment: "center" });
  textBox(s, "ほぼ一致", 220, 513, 220, 32, { fontSize: 22, bold: true, color: C.teal, alignment: "center" });

  rect(s, 620, 345, 2, 210, C.line);
  textBox(s, "学習による危険方向の変化", 660, 350, 520, 34, { fontSize: 23, bold: true, color: C.teal, alignment: "center" });
  textBox(s, "現場側の目標", 690, 409, 190, 28, { fontSize: 18, bold: true, color: C.muted, alignment: "center" });
  textBox(s, "+0.238", 690, 444, 190, 56, { fontSize: 40, bold: true, color: C.navy, alignment: "center" });
  textBox(s, "代理データ", 950, 409, 190, 28, { fontSize: 18, bold: true, color: C.muted, alignment: "center" });
  textBox(s, "+0.051", 950, 444, 190, 56, { fontSize: 40, bold: true, color: C.red, alignment: "center" });
  textBox(s, "0.673 から 0.724", 948, 505, 195, 25, { fontSize: 16, color: C.muted, alignment: "center" });
  textBox(s, "差が残る", 825, 532, 190, 30, { fontSize: 22, bold: true, color: C.red, alignment: "center" });

  rect(s, 78, 589, 1124, 62, C.orangePale);
  textBox(s, "最終的な判定の一致だけでは、現場で生じた学習変化を再現できたとは判断できない", 105, 604, 1070, 35, {
    fontSize: 23,
    bold: true,
    color: C.ink,
    alignment: "center",
    verticalAlignment: "middle",
  });
  note(
    s,
    "5:50–7:00",
    "2枚目から追ってきた危険動画3本の要約を、そのまま追跡します。現場側で求めた学習後の危険確率は0.724でした。中央側で選んだ代理データについて、連合平均後の小型AIが出した危険確率も0.724で、学習後の判定はほぼ一致しています。一方、現場側の変化量はプラス0.238でしたが、代理データでは危険確率が0.673から0.724へ変化しており、その差はプラス0.051でした。方向は同じですが、変化量は十分に再現できていません。また、元動画の正解ラベルは通常でした。ただし、中央側の60本では正解ラベルと小型AIの判定が一致したものが43本だけなので、通常動画が選ばれた原因を検索処理だけに求めることはできません。",
    [
      "artifacts/runs/fedpact_stage1/p1d_full_20260917_02/server/layer2_trace.json（r1-client-0-class1）",
      "artifacts/runs/fedpact_stage1/p1d_full_20260917_02/server/proxy_pairs.json",
      "artifacts/reports/fedpact_stage1_20260917/slack_report.md（labelと予測の一致43/60）",
    ],
  );
  rect(s, 40, 36, 7, 54, C.teal);
}

// 8. Three-round execution and state lineage.
{
  const s = presentation.slides.add();
  title(s, "3ラウンドの状態継承を確認、性能改善は未確認", 8);
  const values = [
    ["ラウンド", "2台の局所更新", "連合平均の再計算差", "次回入力との照合", "中央側AI更新（各ラウンドで選んだ4組）"],
    ["1", "2台とも非ゼロ", "5.82 × 10⁻¹¹", "一致", "加重KL 0.0540 から 0.0534"],
    ["2", "2台とも非ゼロ", "5.82 × 10⁻¹¹", "一致", "加重KL 18.478 から 18.339"],
    ["3", "2台とも非ゼロ", "1.16 × 10⁻¹⁰", "一致", "加重KL 16.131 から 16.178"],
  ];
  const table = s.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: 65,
    top: 132,
    width: 1150,
    height: 244,
    columnWidths: [115, 225, 250, 220, 340],
    values: tableValues(values, 11.5, 14),
  });
  styleTable(table, 18);
  table.getCell(3, 4).fill = C.redPale;
  table.getCell(3, 4).text.style = { typeface: fontFamily, fontSize: 18, bold: true, color: C.red, alignment: "center", verticalAlignment: "middle", autoFit: "shrinkText" };
  textBox(s, "各ラウンドで代理データが異なるため、ずれの値をラウンド間の性能推移として比較しない", 90, 390, 1100, 28, {
    fontSize: 17,
    color: C.muted,
    alignment: "center",
  });

  rect(s, 72, 440, 520, 132, C.bluePale);
  textBox(s, "確認できたこと", 98, 459, 460, 28, { fontSize: 22, bold: true, color: C.teal });
  textBox(s, "2台の更新は毎回ゼロではない\n連合平均を別計算で再現\n小型AIと中央側AIの状態が次回へ一致", 98, 499, 460, 60, {
    fontSize: 20,
    color: C.ink,
  });
  rect(s, 624, 440, 584, 132, C.orangePale);
  textBox(s, "残った診断点", 650, 459, 520, 28, { fontSize: 22, bold: true, color: C.orange });
  textBox(s, "1ラウンド目は変化の方向が4件中3件一致\n3ラウンド目は中央側AIのずれが増加", 650, 499, 520, 60, {
    fontSize: 21,
    bold: true,
    color: C.ink,
  });
  textBox(s, "今回の合格は『実行・接続・状態継承』。代理データの品質や学習効果の合格ではない。", 96, 613, 1088, 36, {
    fontSize: 22,
    bold: true,
    color: C.red,
    alignment: "center",
  });
  note(
    s,
    "7:00–8:25",
    "3ラウンド全体の結果です。各ラウンドで二つの現場側AIが同じ状態から開始し、両方の小型AIが実際に更新されたことを確認しました。連合平均を別に再計算した差は、最大でも約1.16掛ける10のマイナス10でした。また、前のラウンドの出力と次のラウンドの入力が一致していることを、モデルの照合値から確認しました。一方、代理データの再現結果には問題が残っています。1ラウンド目では四つのうち三つで変化の方向が一致しましたが、一つは逆方向でした。さらに3ラウンド目では、中央側AIを更新した後のずれが16.1307から16.1776へ増加しました。今回確認できたのは3ラウンドの処理と状態継承であり、性能が改善した、安定して知識を移せた、という結論ではありません。",
    [
      "artifacts/reports/fedpact_stage1_20260917/three_round_state_lineage.md",
      "artifacts/reports/fedpact_stage1_20260917/three_round_state_lineage.json",
      "artifacts/reports/fedpact_stage1_20260917/slack_report.md",
    ],
  );
}

// 9. Conclusion and next steps.
{
  const s = presentation.slides.add();
  title(s, "結論と今後", 9);
  textBox(s, "今回確認できたこと", 76, 118, 500, 34, { fontSize: 24, bold: true, color: C.teal });
  textBox(s, "・代表動画の反応を要約し、代理データへ対応付け\n・3層を接続して中央側AIを更新\n・3ラウンドの状態をログから追跡", 82, 172, 500, 150, {
    fontSize: 21,
    color: C.ink,
    lineSpacing: 1.14,
  });
  rect(s, 622, 118, 2, 220, C.line);
  textBox(s, "まだ言えないこと", 664, 118, 500, 34, { fontSize: 24, bold: true, color: C.red });
  textBox(s, "・性能向上と未知動画への転移\n・代理データが現場の学習を十分に再現したこと\n・構造や規模が異なるAI間の転移\n・情報漏えいへの形式的保証", 670, 172, 500, 165, {
    fontSize: 22,
    color: C.ink,
    lineSpacing: 1.12,
  });

  rect(s, 72, 366, 1136, 1.5, C.lineDark);
  textBox(s, "今後の順序", 78, 391, 200, 30, { fontSize: 22, bold: true, color: C.navy });
  const next = [
    ["1", "中央側AIの更新方向を確認", "同じ4組を固定し、ごく小さい1回の更新でずれが減るかを確認"],
    ["2", "問題解消後に5ラウンド", "同じ記録形式で、三層の接続と次回への状態継承を確認"],
    ["3", "第II段階で比較・独立評価", "中央側動画のみ、学習後、学習後＋変化を未使用動画で比較"],
  ];
  for (let i = 0; i < next.length; i += 1) {
    const y = 438 + i * 66;
    circle(s, 84, y, 42, i === 0 ? C.red : (i === 1 ? C.orange : C.teal));
    textBox(s, next[i][0], 84, y + 5, 42, 30, { fontSize: 19, bold: true, color: "#FFFFFF", alignment: "center", verticalAlignment: "middle" });
    textBox(s, next[i][1], 150, y - 2, 330, 34, { fontSize: 22, bold: true, color: C.navy });
    textBox(s, next[i][2], 492, y - 2, 680, 42, { fontSize: 19, color: C.ink, verticalAlignment: "middle" });
  }
  rect(s, 72, 633, 1136, 38, C.navy);
  textBox(s, "到達点：効果を正しく検証できる土台と、次に確認すべき失敗箇所を特定した", 92, 640, 1096, 24, {
    fontSize: 20,
    bold: true,
    color: "#FFFFFF",
    alignment: "center",
    verticalAlignment: "middle",
  });
  note(
    s,
    "8:25–10:00",
    "今回の第I段階では、代表動画の学習前後の判定からQ_{k,r}を作り、中央側で代理データを生成し、中央側AIを更新して次のラウンドへ状態を渡すところまでを対応付けました。三層の処理と3ラウンドの状態継承を、ログから追跡できるようになったことは言えます。一方で、性能向上、代理データの品質、未知動画への転移はまだ言えません。今回の第I段階は同じ3B基盤を使用しているため、構造や規模の異なるAI間で移せたとも言えません。今後は、まず同じ四つの教師データを固定し、ごく小さい1回の更新で中央側AIのずれが減ることを確認します。問題を解消した後に5ラウンドの接続と状態継承を確認します。その後、第II段階で中央側動画だけの場合、学習後の判定を使う場合、学習前後の変化も使う場合を同じ条件で比較し、選択に使っていない動画で評価します。最後に構造や規模が異なるAIへ進みます。",
    [
      "notes/advisor/sources/raw/2026-09-16/slack_zettsu_experiment_feedback_date_unknown_received_2026-09-16.md",
      "notes/advisor/sources/raw/2026-09-16/slack_zettsu_reply_to_chen_ja_date_unknown_received_2026-09-16.md",
      "artifacts/reports/fedpact_stage1_20260917/three_round_state_lineage.md",
    ],
  );
  rect(s, 40, 36, 7, 54, C.teal);
}

// Appendix 1. Detailed proxy reproduction results across all rounds.
{
  const s = presentation.slides.add();
  title(s, "付録1　選択した12組の代理データ結果", 1, { appendix: true });
  textBox(s, "変化の方向は7／12組で一致。prototypeと選択動画の正解ラベルは6／12組で一致", 70, 108, 1140, 34, {
    fontSize: 21,
    bold: true,
    color: C.navy,
    alignment: "center",
  });
  const values = [
    ["R", "対象", "選択動画", "目標Δ危険", "代理Δ危険", "方向", "学習後の誤差\n(post KL)", "変化量の誤差\n(delta error)", "合計誤差\n(L_match)"],
    ["1", "現場0／通常", "危険", "+0.0938", "+0.0716", "一致", "2.04×10⁻⁴", "4.94×10⁻⁴", "6.99×10⁻⁴"],
    ["1", "現場0／危険", "通常", "+0.2377", "+0.0509", "一致", "8.35×10⁻¹⁰", "3.489×10⁻²", "3.489×10⁻²"],
    ["1", "現場1／通常", "危険", "−0.1027", "+0.0559", "不一致", "3.51×10⁻⁴", "2.514×10⁻²", "2.549×10⁻²"],
    ["1", "現場1／危険", "通常", "+0.00546", "+0.00968", "一致", "9.45×10⁻⁵", "1.78×10⁻⁵", "1.12×10⁻⁴"],
    ["2", "現場0／通常", "危険", "+0.03448", "+0.01343", "一致", "2.27×10⁻⁷", "4.43×10⁻⁴", "4.43×10⁻⁴"],
    ["2", "現場0／危険", "危険", "+0.17458", "+0.00516", "一致", "2.67×10⁻⁵", "2.870×10⁻²", "2.873×10⁻²"],
    ["2", "現場1／通常", "通常", "−0.11137", "+0.00511", "不一致", "9.15×10⁻⁵", "1.357×10⁻²", "1.366×10⁻²"],
    ["2", "現場1／危険", "危険", "−0.00697", "+0.00882", "不一致", "1.16×10⁻⁴", "2.49×10⁻⁴", "3.66×10⁻⁴"],
    ["3", "現場0／通常", "危険", "+0.00723", "−0.00603", "不一致", "2.55×10⁻⁵", "1.76×10⁻⁴", "2.01×10⁻⁴"],
    ["3", "現場0／危険", "危険", "+0.14805", "+0.00013", "一致", "8.30×10⁻⁵", "2.188×10⁻²", "2.196×10⁻²"],
    ["3", "現場1／通常", "通常", "−0.11538", "−0.02566", "一致", "4.62×10⁻⁴", "8.05×10⁻³", "8.51×10⁻³"],
    ["3", "現場1／危険", "危険", "−0.01229", "+0.00081", "不一致", "4.74×10⁻⁴", "1.71×10⁻⁴", "6.46×10⁻⁴"],
  ];
  const table = s.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: 64,
    top: 151,
    width: 1152,
    height: 448,
    columnWidths: [55, 145, 105, 125, 125, 115, 145, 165, 172],
    values: tableValues(values, 10.5, 10.5),
  });
  styleTable(table, 13.5);
  for (let c = 0; c < table.columns.length; c += 1) {
    table.getCell(0, c).text.style = { typeface: fontFamily, fontSize: 11.5, bold: true, color: C.navy, alignment: "center", verticalAlignment: "middle", autoFit: "shrinkText", insets: { left: 3, right: 3, top: 2, bottom: 2 } };
  }
  for (const row of [3, 7, 8, 9, 12]) {
    table.getCell(row, 5).fill = C.redPale;
    table.getCell(row, 5).text.style = { typeface: fontFamily, fontSize: 13.5, bold: true, color: C.red, alignment: "center", verticalAlignment: "middle", autoFit: "shrinkText" };
  }
  textBox(s, "Δ危険は危険確率の変化。値が小さいだけでは、動画内容の意味が一致したとは判断しない", 76, 623, 1128, 34, {
    fontSize: 18,
    bold: true,
    color: C.red,
    alignment: "center",
  });
  note(
    s,
    "質疑用",
    "各ラウンドで選択した四組、合計十二組の代理データについて、現場側が示した危険確率の変化と、代理データ上で連合集約前後の小型AIが示した危険確率の変化を並べています。方向一致は十二組中七組でした。post KLが小さくてもdelta errorが大きい組があるため、学習後の判定だけの一致を代理データ再現の成功とは扱いません。",
    ["artifacts/reports/fedpact_stage1_20260917/three_round_state_lineage.json"],
  );
}

// Appendix 2. Update magnitudes, recomputation and foundation diagnostics.
{
  const s = presentation.slides.add();
  title(s, "付録2　更新量と状態継承の数値", 2, { appendix: true });
  textBox(s, "現場側の更新と連合平均", 72, 108, 520, 30, { fontSize: 22, bold: true, color: C.teal });
  const localValues = [
    ["R", "軽量更新量\n現場0", "軽量更新量\n現場1", "分類層更新量\n現場0", "分類層更新量\n現場1", "連合平均の\n再計算差", "次回入力"],
    ["1", "0.346093", "0.291624", "0.012345", "0.009560", "5.82×10⁻¹¹", "一致"],
    ["2", "0.322764", "0.280291", "0.011066", "0.009495", "5.82×10⁻¹¹", "一致"],
    ["3", "0.320213", "0.280363", "0.010754", "0.009461", "1.16×10⁻¹⁰", "一致"],
  ];
  const localTable = s.tables.add({
    rows: localValues.length,
    columns: localValues[0].length,
    left: 68,
    top: 148,
    width: 1144,
    height: 176,
    columnWidths: [58, 165, 165, 180, 180, 210, 186],
    values: tableValues(localValues, 10.5, 12),
  });
  styleTable(localTable, 15);

  textBox(s, "中央側AIの更新", 72, 354, 520, 30, { fontSize: 22, bold: true, color: C.teal });
  const foundationValues = [
    ["R", "勾配の大きさ", "更新量", "加重KL 更新前", "加重KL 更新後", "受理条件", "更新対象外"],
    ["1", "4.37146", "0.00135671", "0.054042", "0.053431", "非増加", "不変"],
    ["2", "65.9988", "0.00135747", "18.4783", "18.3390", "非増加", "不変"],
    ["3", "66.4649", "0.00135746", "16.1307", "16.1776", "実行確認", "不変"],
  ];
  const foundationTable = s.tables.add({
    rows: foundationValues.length,
    columns: foundationValues[0].length,
    left: 68,
    top: 394,
    width: 1144,
    height: 176,
    columnWidths: [58, 170, 160, 190, 190, 190, 186],
    values: tableValues(foundationValues, 10.5, 12),
  });
  styleTable(foundationTable, 15);
  foundationTable.getCell(3, 4).fill = C.redPale;
  foundationTable.getCell(3, 4).text.style = { typeface: fontFamily, fontSize: 15, bold: true, color: C.red, alignment: "center", verticalAlignment: "middle", autoFit: "shrinkText" };
  foundationTable.getCell(3, 5).fill = C.orangePale;
  textBox(s, "各ラウンドで選択した4組が異なるため、加重KLをラウンド間の性能推移として比較しない", 76, 601, 1128, 38, {
    fontSize: 18,
    bold: true,
    color: C.red,
    alignment: "center",
  });
  note(
    s,
    "質疑用",
    "上段は二台の現場側AIの軽量更新量と分類層更新量、連合平均を別計算したときの最大差です。下段は中央側AIの軽量更新部分に対する勾配、更新量、同じ四組上の加重KLです。三ラウンド目は非増加条件を満たさなかったため、性能確認ではなく有限でゼロではない更新の実行確認として保存しました。",
    [
      "artifacts/reports/fedpact_stage1_20260917/three_round_state_lineage.json",
      "artifacts/runs/fedpact_stage1/p1d_round3_full_20260917_02/summary.json",
    ],
  );
}

// Appendix 3. Timing and implementation-level communication audit.
{
  const s = presentation.slides.add();
  title(s, "付録3", 3, { appendix: true });
  textBox(s, "処理時間と通信監査", 206, 34, 760, 62, {
    fontSize: 38,
    bold: true,
    color: C.navy,
    verticalAlignment: "middle",
  });
  textBox(s, "処理時間（秒）", 72, 108, 420, 30, { fontSize: 22, bold: true, color: C.teal });
  const timingValues = [
    ["R", "現場側学習", "60本検索", "変換保存", "変化量・目的関数", "中央側AI読込・更新", "第2・3層合計"],
    ["1", "96.5", "28.6", "4.0", "52.9", "171.8", "279.1"],
    ["2", "97.1", "27.0", "4.1", "52.7", "173.6", "272.5"],
    ["3", "131.0", "27.1", "4.2", "52.7", "178.2", "277.3"],
  ];
  const timingTable = s.tables.add({
    rows: timingValues.length,
    columns: timingValues[0].length,
    left: 68,
    top: 148,
    width: 1144,
    height: 184,
    columnWidths: [58, 165, 145, 145, 205, 235, 191],
    values: tableValues(timingValues, 10.5, 12),
  });
  styleTable(timingTable, 15);

  textBox(s, "中央側候補60本の正解ラベルと、連合集約後の小型AIの判定", 74, 366, 650, 30, {
    fontSize: 20,
    bold: true,
    color: C.navy,
  });
  const agreeValues = [
    ["ラウンド", "一致本数", "割合"],
    ["1", "43／60", "71.7%"],
    ["2", "44／60", "73.3%"],
    ["3", "43／60", "71.7%"],
  ];
  const agreeTable = s.tables.add({
    rows: agreeValues.length,
    columns: agreeValues[0].length,
    left: 76,
    top: 414,
    width: 500,
    height: 176,
    columnWidths: [150, 180, 170],
    values: tableValues(agreeValues, 11, 12),
  });
  styleTable(agreeTable, 16);

  textBox(s, "通信・更新対象の監査", 664, 366, 500, 30, { fontSize: 20, bold: true, color: C.navy });
  textBox(s, "3／3ラウンド　生の動画・画像・現場内の識別番号を中央側へ保存していない\n6／6送信　許可した項目だけを送信\n3／3ラウンド　中央側AIの分類層と更新対象外部分は不変", 664, 418, 520, 154, {
    fontSize: 20,
    color: C.ink,
    lineSpacing: 1.16,
  });
  rect(s, 658, 590, 534, 54, C.orangePale);
  textBox(s, "通信監査は実装確認であり、形式的なプライバシー保証ではない", 676, 603, 500, 28, {
    fontSize: 18,
    bold: true,
    color: C.red,
    alignment: "center",
  });
  note(
    s,
    "質疑用",
    "各処理時間は実行確認と異常検出のために記録した値で、他方式との速度比較ではありません。正解ラベルと小型AIの判定が一致した中央側動画は、各ラウンドで六十本中四十三本、四十四本、四十三本でした。選択動画のラベル不一致には検索処理だけでなく、小型AI自体の誤分類も混在します。通信監査は生の動画や画像を送っていないことの実装確認であり、形式的なプライバシー保証ではありません。",
    [
      "artifacts/reports/fedpact_stage1_20260917/three_round_state_lineage.json",
      "artifacts/reports/fedpact_stage1_20260917/slack_report.md",
    ],
  );
}

const draftPath = path.join(buildDir, "candidate_v20.pptx");
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
    "--require-native-table-slide", "5",
    "--require-native-table-slide", "8",
    "--require-native-table-slide", "10",
    "--require-native-table-slide", "11",
    "--require-native-table-slide", "12",
  ],
  explicitTotalSlideCount: 12,
  requiredNativeTableOwnerSlides: [5, 8, 10, 11, 12],
  requiredNativeChartOwnerSlides: [],
  fontPolicy: { basis: "design", families: [fontFamily] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, "FedPACT_研究会_10分_9枚_付録3枚_20260918_v20.validation.json"),
});

console.log(JSON.stringify({ finalPath, draftPath, result }, null, 2));
