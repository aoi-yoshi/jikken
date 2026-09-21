import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const workspace = "C:\\Python\\実験_修正\\tmp\\pptx\\0731_revision";
const input = `${workspace}\\template-starter.pptx`;
const output =
  "C:\\Users\\aoi7y\\OneDrive - 国立大学法人東海国立大学機構\\folder\\卒研_明光\\進捗会0731_吉村有生_改訂版.pptx";
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
    "同一系列・異容量VLMの双方向蒸留\n新規性と成功条件",
  );

  replaceAllText(
    presentation,
    records,
    "細部の実装より先に，研究の「出口」を固定する\nFedMD・FedDFに対して，何を改善する研究かを明示する\n何ができれば成功かを，評価指標と比較条件で定義する\nLpred・Lgrad・Lpostは，目的に応じて後から選択する",
    "Deep Researchの結論：双方向性そのものは新規ではない\nFedMKT・FedProxy・Fed-ETが双方向／知識還元を既に扱う\n主張を「7B/3B動画VLM×少量共有×実用コスト」に限定する\nまずlogit蒸留で両側改善を検証し，Lgrad/Lpostは必要時だけ追加する",
  );

  replaceAllText(
    presentation,
    records,
    "関連研究との位置づけ",
    "双方向性だけでは新規性にならない",
  );
  setTableValues(presentation, records, 3, [
    ["研究", "既存の到達点", "新規性への反論", "残る差分"],
    ["FedMKT", "LLM↔SLM蒸留", "双方向・知識還元済", "VLM・動画・少量共有"],
    ["FedProxy", "proxy→LLM統合", "小型側知識の還元済", "循環・VLM・動画"],
    ["Fed-ET", "大型↔小型", "feedback loop済", "分類中心・public依存"],
    ["FedMD/DF", "logit共有／蒸留", "異種KDの基礎", "双方改善・実用評価"],
    ["本研究", "7B↔3B VLM", "組合せだけでは弱い", "少量共有＋両側改善＋コスト"],
  ]);

  replaceAllText(
    presentation,
    records,
    "研究の中心主張（暫定）",
    "採用する中心主張",
  );
  replaceAllText(
    presentation,
    records,
    "エッジ上の小型surrogateから，端末固有の新しい知識を大型基盤へ反映する\n対象：同一モデル系列のサイズ差（client 3B／server 7B）\n制約：生データを送らず，固定公開D_alignへの依存を抑える\n価値：端末資源を抑え，server・client性能を維持／改善する",
    "問い：少量共有で，双方向循環は一方向法より両側を改善できるか\n対象：Qwen2.5-VL 7B/3Bによる車載動画分類\n制約：生動画は共有せず，alignment bufferを小規模にする\n成功：両側改善＋通信・時間・memoryのPareto優位",
  );

  replaceAllText(
    presentation,
    records,
    "成功条件：同等性能で資源・共有データを減らす",
    "成功条件：両側改善とPareto優位を同時に示す",
  );
  setTableValues(presentation, records, 5, [
    ["評価軸", "測定", "比較対象", "成功判定"],
    ["client性能", "AUROC/AUPRC/F1", "独立3B・S→C", "提案法が改善"],
    ["server全体", "AUROC/F1", "独立7B・C→S", "悪化させない"],
    ["新条件還元", "client条件別F1", "C→Sのみ", "serverが改善"],
    ["双方向効果", "server/client両指標", "各one-way", "同時に改善"],
    ["収束", "目標到達round", "FedDF-like", "少ない"],
    ["通信", "送受信bytes", "全条件", "Pareto改善"],
    ["計算", "総wall-clock", "同一計算予算", "実用範囲"],
    ["memory", "peak GPU memory", "7B直接適応", "client負荷低減"],
    ["共有data", "buffer量", "tiny/small/medium", "少量でも優位"],
  ]);

  setTableValues(presentation, records, 6, [
    ["条件", "server", "client", "検証目的"],
    ["独立学習", "7B LoRA", "3B LoRA", "蒸留なし基準"],
    ["S→Cのみ", "teacher", "KD＋LoRA", "client効果"],
    ["C→Sのみ", "KD更新", "local LoRA", "server還元"],
    ["FedMD/DF-like", "logit統合", "shared buffer", "既存型との比較"],
    ["提案法", "循環KD", "KD＋local適応", "両側同時改善"],
  ]);

  replaceAllText(
    presentation,
    records,
    "一度に複数の論点を入れず，外側から検証する\n段階1：Lpredのみで7B/3Bフローと評価軸を検証する\n段階2：D_alignの量・domain・共有範囲を変えて依存度を測る\n段階3：不足する指標にだけLgrad・Lpost・optimizerを追加する",
    "主張を決めるのは，手法の細部ではなく比較実験である\n段階1：昼/夜・晴/雨・camera差で3〜4 clientsとhold-outを作る\n段階2：Lpredのみの5条件を，同一学習量・同一計算予算で比較する\n段階3：buffer量を変え，不足時のみLgrad/Lpostを追加する",
  );

  replaceAllText(
    presentation,
    records,
    "まず研究の外側を確定し，実装箇所を後から絞る\n中心主張を1文に固定する\n関連研究との差分表と成功条件を確定する\n比較条件を実装し，資源・精度・収束・共有量を計測する",
    "次に行うのは，ベースラインと評価系の確定である\nclient domain分割とserver/client test sliceを固定する\n5条件を実装し，精度・bytes・time・memoryを同時記録する\nFedMKT・FedProxy・Fed-ETを読み，差分の表現を確定する",
  );

  setNotes(presentation, records, 1, [
    "[Sources]",
    "- 進捗会0724_吉村有生.pptx（視覚テンプレート）",
    "[/Sources]",
  ]);
  setNotes(presentation, records, 2, [
    "[Sources]",
    "- 異種VLM連合学習における双方向蒸留研究の新規性と成功条件.pdf pp.1, 5, 9",
    "- FedMKT: https://arxiv.org/abs/2406.02224",
    "- FedProxy: https://arxiv.org/abs/2604.19015",
    "- Fed-ET: https://www.ijcai.org/proceedings/2022/0399.pdf",
    "[/Sources]",
  ]);
  setNotes(presentation, records, 3, [
    "[Sources]",
    "- FedMD: https://arxiv.org/abs/1910.03581",
    "- FedDF: https://proceedings.neurips.cc/paper/2020/hash/18df51b97ccd68128e994804f3eccc87-Abstract.html",
    "- Fed-ET: https://www.ijcai.org/proceedings/2022/0399.pdf",
    "- FedMKT: https://arxiv.org/abs/2406.02224",
    "- FedProxy: https://arxiv.org/abs/2604.19015",
    "[/Sources]",
  ]);
  setNotes(presentation, records, 4, [
    "[Sources]",
    "- 異種VLM連合学習における双方向蒸留研究の新規性と成功条件.pdf pp.6-8",
    "- Qwen2.5-VL technical report: https://arxiv.org/abs/2502.13923",
    "[/Sources]",
  ]);
  setNotes(presentation, records, 5, [
    "[Sources]",
    "- 異種VLM連合学習における双方向蒸留研究の新規性と成功条件.pdf pp.7-8",
    "- FedDF: https://proceedings.neurips.cc/paper/2020/hash/18df51b97ccd68128e994804f3eccc87-Abstract.html",
    "[/Sources]",
  ]);
  setNotes(presentation, records, 6, [
    "[Sources]",
    "- 異種VLM連合学習における双方向蒸留研究の新規性と成功条件.pdf pp.7-8",
    "- FedMD: https://arxiv.org/abs/1910.03581",
    "- FedDF: https://proceedings.neurips.cc/paper/2020/hash/18df51b97ccd68128e994804f3eccc87-Abstract.html",
    "[/Sources]",
  ]);
  setNotes(presentation, records, 7, [
    "[Sources]",
    "- 異種VLM連合学習における双方向蒸留研究の新規性と成功条件.pdf pp.8-9",
    "[/Sources]",
  ]);
  setNotes(presentation, records, 8, [
    "[Sources]",
    "- 異種VLM連合学習における双方向蒸留研究の新規性と成功条件.pdf pp.9-10",
    "[/Sources]",
  ]);

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
