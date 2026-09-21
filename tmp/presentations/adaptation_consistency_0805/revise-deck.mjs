import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const workspace = "C:\\Python\\実験_修正\\tmp\\presentations\\adaptation_consistency_0805";
const input = `${workspace}\\template-starter.pptx`;
const output = "C:\\Python\\実験_修正\\output\\presentation\\Adaptation_Consistency_三層処理フロー_0805.pptx";
const previewDir = `${workspace}\\final-preview`;
const layoutDir = `${workspace}\\final-layout`;

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function anchorByText(records, oldText) {
  const record = records.find(
    (item) => item.kind === "textbox" && item.text === oldText,
  );
  if (!record) throw new Error(`Could not find textbox with text: ${oldText}`);
  return record.id;
}

function anchorBySlide(records, kind, slideNumber) {
  const record = records.find(
    (item) => item.kind === kind && item.slide === slideNumber,
  );
  if (!record) throw new Error(`Could not find ${kind} on slide ${slideNumber}`);
  return record.id;
}

function replaceAllText(presentation, records, oldText, newText) {
  const shape = presentation.resolve(anchorByText(records, oldText));
  shape.text = newText;
}

function setTableValues(presentation, records, slideNumber, values) {
  const table = presentation.resolve(anchorBySlide(records, "table", slideNumber));
  for (let row = 0; row < values.length; row += 1) {
    for (let col = 0; col < values[row].length; col += 1) {
      table.getCell(row, col).value = values[row][col];
    }
  }
}

function setNotes(presentation, records, slideNumber, lines) {
  const slide = presentation.resolve(anchorBySlide(records, "slide", slideNumber));
  slide.speakerNotes.textFrame.setText(lines.join("\n"));
  slide.speakerNotes.setVisible(true);
}

async function main() {
  await fs.mkdir(previewDir, { recursive: true });
  await fs.mkdir(layoutDir, { recursive: true });

  const presentation = await PresentationFile.importPptx(await FileBlob.load(input));
  const before = await presentation.inspect({
    kind: "slide,textbox,table,notes",
    maxChars: 30000,
  });
  const records = before.ndjson
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));

  replaceAllText(
    presentation,
    records,
    "surrogateモデルを用いた連合知識蒸留\n中心主張と評価設計",
    "Adaptation Consistencyの再設計\n共有データなしの三層構造",
  );
  replaceAllText(presentation, records, "7月31日報告会", "8月5日検討版");

  replaceAllText(presentation, records, "今回の結論", "方向修正の結論");
  replaceAllText(
    presentation,
    records,
    "細部の実装より先に，研究の「出口」を固定する\nFedMD・FedDFに対して，何を改善する研究かを明示する\n何ができれば成功かを，評価指標と比較条件で定義する\nLpred・Lgrad・Lpostは，目的に応じて後から選択する",
    "Lpred・Lgrad・Lpostの並列構成をやめ，役割を三層に分離する\n更新方向と忘却は，同一構造のsurrogate間のFL／FCLで扱う\n異種モデル間は，共通出力空間のlogits蒸留に限定する\n研究の中心は，何を代理ペアとして選び，何を再現するかである",
  );

  replaceAllText(
    presentation,
    records,
    "関連研究との位置づけ",
    "近い先行研究と，本研究の焦点",
  );
  setTableValues(presentation, records, 3, [
    ["手法", "共有入力", "サーバ側の知識化", "本研究との差"],
    ["FedMD", "public dataを共有", "同一sampleのlogitsを統合", "共有整合入力が必要"],
    ["FedDF", "server unlabeled data", "local model ensembleを蒸留", "server dataに依存"],
    ["FedDGM", "raw data非共有", "軌跡に合う合成dataを生成", "trajectory再現が中心"],
    ["本研究", "整合data非共有", "global surrogate→代理pair", "出力挙動を異種基盤へ再現"],
    ["中心課題", "server seed／生成器", "pairの生成・選択・調整", "何を蒸留するかを設計"],
  ]);

  replaceAllText(
    presentation,
    records,
    "研究の中心主張（暫定）",
    "Adaptation Consistencyの再定義",
  );
  replaceAllText(
    presentation,
    records,
    "エッジ上の小型surrogateから，端末固有の新しい知識を大型基盤へ反映する\n対象：同一モデル系列のサイズ差（client 3B／server 7B）\n制約：生データを送らず，固定公開D_alignへの依存を抑える\n価値：端末資源を抑え，server・client性能を維持／改善する",
    "privateデータで局所適応したsurrogate LoRAの出力挙動を，\nデータを送らずglobal surrogateへ連合統合する\nクラウドで（代理入力，teacher logits）の対応ペアへ変換する\n共通出力空間から異種の大型基盤LoRA上に再現する",
  );

  replaceAllText(
    presentation,
    records,
    "成功条件：同等性能で資源・共有データを減らす",
    "成功条件は，各層とend-to-endで分けて測る",
  );
  setTableValues(presentation, records, 5, [
    ["評価対象", "主指標", "比較基準", "成功判定"],
    ["第1層 性能", "global／client F1", "FedAvg", "改善／非劣化"],
    ["client drift", "update cosine・分散", "local only", "ずれを低減"],
    ["忘却", "past-task F1・BWT", "保持策なし", "忘却を低減"],
    ["第2層 fidelity", "surrogate出力KL", "random seed", "忠実度を向上"],
    ["被覆・多様性", "class／domain coverage", "confidence選択", "偏りを低減"],
    ["privacy", "再構成・近傍漏洩", "保護なし", "漏洩を抑制"],
    ["第3層 再現", "client-only test F1", "server data上界", "gapを縮小"],
    ["異種成立", "3B→7B transfer gain", "3B→3B", "異サイズでも成立"],
    ["end-to-end", "精度・bytes・time", "FedMD／DF／DGM", "Pareto優位"],
  ]);

  replaceAllText(presentation, records, "比較する5条件", "三層処理フロー：場所と受け渡し");
  setTableValues(presentation, records, 6, [
    ["段階", "場所", "入力 → 処理", "出力 → 次"],
    ["第1層① 局所適応", "client", "global LoRA＋private data", "ΔLoRA＋state"],
    ["第1層② 連合統合", "cloud", "ΔLoRA → FL／FCL", "global S LoRA"],
    ["第2層 ★ 代理pair", "cloud", "seed＋global S → 選択／生成", "S再forward →（x̃，z̃）"],
    ["第3層 異種蒸留", "cloud", "pair → KD", "foundation LoRA"],
    ["次round", "cloud→client", "global S LoRA配布", "第1層①へ"],
  ]);

  replaceAllText(presentation, records, "実験の順序", "検証は三層を切り分ける");
  replaceAllText(
    presentation,
    records,
    "一度に複数の論点を入れず，外側から検証する\n段階1：Lpredのみで7B/3Bフローと評価軸を検証する\n段階2：D_alignの量・domain・共有範囲を変えて依存度を測る\n段階3：不足する指標にだけLgrad・Lpost・optimizerを追加する",
    "段階1：FedAvg基準で第1層の精度・drift・forgettingを確認する\n段階2：server seedの選択＋global surrogate再forwardでpairを作る\n段階3：固定server data上界と比較し，3B→7Bの蒸留損失を測る\n段階4：diversity・replay・出力統計・生成器を一つずつ追加する",
  );

  replaceAllText(presentation, records, "直近の実施項目", "次に固定する4点");
  replaceAllText(
    presentation,
    records,
    "まず研究の外側を確定し，実装箇所を後から絞る\n中心主張を1文に固定する\n関連研究との差分表と成功条件を確定する\n比較条件を実装し，資源・精度・収束・共有量を計測する",
    "共通出力空間：分類logitsか，生成出力の意味表現か\n局所logits：送らない／集約統計だけ送る／別方式か\n代理入力：server seedから選択するか，生成器で作るか\n下り経路：大型基盤の知識をclientへ還元するか",
  );

  const userSource = [
    "[Sources]",
    "- User-provided research direction, 2026-08-05.",
    "[/Sources]",
  ];
  setNotes(presentation, records, 1, userSource);
  setNotes(presentation, records, 2, userSource);
  setNotes(presentation, records, 3, [
    "[Sources]",
    "- FedMD: https://arxiv.org/abs/1910.03581",
    "- FedDF: https://proceedings.neurips.cc/paper/2020/hash/18df51b97ccd68128e994804f3eccc87-Abstract.html",
    "- FedDGM: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09992.pdf",
    "- User-provided research direction, 2026-08-05.",
    "[/Sources]",
  ]);
  setNotes(presentation, records, 4, userSource);
  setNotes(presentation, records, 5, userSource);
  setNotes(presentation, records, 6, userSource);
  setNotes(presentation, records, 7, userSource);
  setNotes(presentation, records, 8, userSource);

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(
      `${previewDir}\\${stem}.png`,
      await presentation.export({ slide, format: "png", scale: 2 }),
    );
    await fs.writeFile(
      `${layoutDir}\\${stem}.layout.json`,
      await (await slide.export({ format: "layout" })).text(),
      "utf8",
    );
  }

  await writeBlob(
    `${workspace}\\final-montage.webp`,
    await presentation.export({ format: "webp", montage: true, scale: 1 }),
  );

  const inspection = await presentation.inspect({
    kind: "slide,textbox,table,notes,layout",
    maxChars: 30000,
  });
  await fs.writeFile(`${workspace}\\final-inspect.ndjson`, inspection.ndjson, "utf8");

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(output);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
