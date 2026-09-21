import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const workspace = "C:\\Python\\実験_修正\\tmp\\presentations\\progress_0731";
const starterPptx = `${workspace}\\template-starter.pptx`;
const finalPptx = "C:\\Python\\実験_修正\\output\\presentation\\進捗会0731_吉村有生.pptx";
const previewDir = `${workspace}\\final-preview`;
const layoutDir = `${workspace}\\final-layout`;

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

function setText(presentation, id, value) {
  const shape = presentation.resolve(id);
  shape.text = value;
  return shape;
}

function setStructuredText(presentation, id, paragraphs) {
  const shape = presentation.resolve(id);
  shape.text.set(paragraphs);
  return shape;
}

function fillTable(presentation, id, values) {
  const table = presentation.resolve(id);
  for (let r = 0; r < values.length; r += 1) {
    for (let c = 0; c < values[r].length; c += 1) {
      table.cells.set(r, c, values[r][c]);
    }
  }
  return table;
}

function addSources(slide, lines) {
  slide.speakerNotes.textFrame.setText([
    "[Sources]",
    ...lines.map((line) => `- ${line}`),
    "[/Sources]",
  ]);
  slide.speakerNotes.setVisible(true);
}

function parseInspect(snapshot) {
  return snapshot.ndjson
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line));
}

function findAid(records, slide, kind, predicate, label) {
  const matches = records.filter(
    (record) =>
      record.slide === slide &&
      record.kind === kind &&
      predicate(record),
  );
  if (matches.length !== 1) {
    throw new Error(
      `Expected one ${label} on slide ${slide}; found ${matches.length}`,
    );
  }
  return matches[0].id;
}

async function main() {
  await fs.mkdir(previewDir, { recursive: true });
  await fs.mkdir(layoutDir, { recursive: true });

  const presentation = await PresentationFile.importPptx(
    await FileBlob.load(starterPptx),
  );
  const records = parseInspect(
    await presentation.inspect({
      kind: "slide,textbox,shape,table",
      include: "id,slide,name,text,textPreview,placeholder,rows,cols",
      maxChars: 100000,
    }),
  );

  const ids = {
    s1Title: findAid(
      records,
      1,
      "textbox",
      (r) => r.placeholder === "title",
      "title",
    ),
    s1Date: findAid(
      records,
      1,
      "textbox",
      (r) => r.text === "7月24日報告会",
      "date",
    ),
    s2Title: findAid(records, 2, "textbox", (r) => r.placeholder === "title", "title"),
    s2Page: findAid(
      records,
      2,
      "textbox",
      (r) => r.placeholder === "slideNumber",
      "page marker",
    ),
    s2Body: findAid(
      records,
      2,
      "textbox",
      (r) => r.name === "コンテンツ プレースホルダー 3",
      "body",
    ),
    s3Title: findAid(records, 3, "textbox", (r) => r.placeholder === "title", "title"),
    s3Page: findAid(
      records,
      3,
      "textbox",
      (r) => r.placeholder === "slideNumber",
      "page marker",
    ),
    s3Placeholder: findAid(
      records,
      3,
      "shape",
      (r) => r.name === "コンテンツ プレースホルダー 3",
      "empty placeholder",
    ),
    s3Table: findAid(records, 3, "table", (r) => r.rows === 6 && r.cols === 4, "table"),
    s4Title: findAid(records, 4, "textbox", (r) => r.placeholder === "title", "title"),
    s4Page: findAid(
      records,
      4,
      "textbox",
      (r) => r.placeholder === "slideNumber",
      "page marker",
    ),
    s4Body: findAid(
      records,
      4,
      "textbox",
      (r) => r.name === "コンテンツ プレースホルダー 3",
      "body",
    ),
    s5Title: findAid(records, 5, "textbox", (r) => r.placeholder === "title", "title"),
    s5Page: findAid(
      records,
      5,
      "textbox",
      (r) => r.placeholder === "slideNumber",
      "page marker",
    ),
    s5Table: findAid(records, 5, "table", (r) => r.rows === 10 && r.cols === 4, "table"),
    s6Title: findAid(records, 6, "textbox", (r) => r.placeholder === "title", "title"),
    s6Page: findAid(
      records,
      6,
      "textbox",
      (r) => r.placeholder === "slideNumber",
      "page marker",
    ),
    s6Placeholder: findAid(
      records,
      6,
      "shape",
      (r) => r.name === "コンテンツ プレースホルダー 3",
      "empty placeholder",
    ),
    s6Table: findAid(records, 6, "table", (r) => r.rows === 6 && r.cols === 4, "table"),
    s7Title: findAid(records, 7, "textbox", (r) => r.placeholder === "title", "title"),
    s7Page: findAid(
      records,
      7,
      "textbox",
      (r) => r.placeholder === "slideNumber",
      "page marker",
    ),
    s7Body: findAid(
      records,
      7,
      "textbox",
      (r) => r.name === "コンテンツ プレースホルダー 3",
      "body",
    ),
    s8Title: findAid(records, 8, "textbox", (r) => r.placeholder === "title", "title"),
    s8Page: findAid(
      records,
      8,
      "textbox",
      (r) => r.placeholder === "slideNumber",
      "page marker",
    ),
    s8Body: findAid(
      records,
      8,
      "textbox",
      (r) => r.name === "コンテンツ プレースホルダー 3",
      "body",
    ),
  };

  // Slide 1 — title
  setText(
    presentation,
    ids.s1Title,
    "surrogateモデルを用いた連合知識蒸留\n中心主張と評価設計",
  );
  setText(presentation, ids.s1Date, "7月31日報告会");
  addSources(presentation.slides.items[0], [
    "進捗会0724_吉村有生.pptx（視覚テンプレート）",
  ]);

  // Slide 2 — conclusion
  setText(presentation, ids.s2Title, "今回の結論");
  setText(presentation, ids.s2Page, "2");
  setStructuredText(presentation, ids.s2Body, [
    { runs: ["細部の実装より先に，研究の「出口」を固定する"] },
    {
      bulletCharacter: "□",
      marginLeft: 52,
      indent: -28,
      runs: ["FedMD・FedDFに対して，何を改善する研究かを明示する"],
    },
    {
      bulletCharacter: "□",
      marginLeft: 52,
      indent: -28,
      runs: ["何ができれば成功かを，評価指標と比較条件で定義する"],
    },
    {
      bulletCharacter: "□",
      marginLeft: 52,
      indent: -28,
      runs: ["Lpred・Lgrad・Lpostは，目的に応じて後から選択する"],
    },
  ]);
  addSources(presentation.slides.items[1], [
    "研究会_0724.pdf p.1, p.5",
  ]);

  // Slide 3 — prior work comparison
  setText(presentation, ids.s3Title, "関連研究との位置づけ");
  setText(presentation, ids.s3Page, "3");
  presentation.resolve(ids.s3Placeholder).delete();
  fillTable(presentation, ids.s3Table, [
    ["手法", "モデル", "情報", "残る論点"],
    ["FedMD", "異種client", "公開data上のlogits", "proxy dataへ依存"],
    ["FedDF", "異種client→server", "非label data＋ensemble", "server蒸留は既存"],
    ["Fed-ET", "小型client→大型server", "weighted consensus logits", "小型→大型は既存"],
    ["FedMKT", "SLM↔LLM", "公開data＋相互転送", "双方向循環も既存"],
    ["本研究", "VLM 3B↔7B", "少量共有情報＋循環", "資源・新知識・共有量を評価"],
  ]);
  addSources(presentation.slides.items[2], [
    "FedMD: https://arxiv.org/abs/1910.03581",
    "FedDF: https://arxiv.org/abs/2006.07242",
    "Fed-ET: https://www.ijcai.org/proceedings/2022/399",
    "FedMKT: https://aclanthology.org/2025.coling-main.17/",
  ]);

  // Slide 4 — provisional central claim
  setText(presentation, ids.s4Title, "研究の中心主張（暫定）");
  setText(presentation, ids.s4Page, "4");
  setStructuredText(presentation, ids.s4Body, [
    {
      runs: [
        "エッジ上の小型surrogateから，端末固有の新しい知識を大型基盤へ反映する",
      ],
    },
    {
      bulletCharacter: "□",
      marginLeft: 52,
      indent: -28,
      runs: ["対象：同一モデル系列のサイズ差（client 3B／server 7B）"],
    },
    {
      bulletCharacter: "□",
      marginLeft: 52,
      indent: -28,
      runs: ["制約：生データを送らず，固定公開D_alignへの依存を抑える"],
    },
    {
      bulletCharacter: "□",
      marginLeft: 52,
      indent: -28,
      runs: ["価値：端末資源を抑え，server・client性能を維持／改善する"],
    },
  ]);
  addSources(presentation.slides.items[3], [
    "研究会_0724.pdf p.1, p.4, p.8",
  ]);

  // Slide 5 — success criteria
  setText(
    presentation,
    ids.s5Title,
    "成功条件：同等性能で資源・共有データを減らす",
  );
  setText(presentation, ids.s5Page, "5");
  fillTable(presentation, ids.s5Table, [
    ["評価軸", "測定", "比較基準", "成功判定"],
    ["server性能", "Accuracy／F1", "server単独", "改善"],
    ["client性能", "次round後F1", "client単独", "非劣化／改善"],
    ["収束", "目標F1到達round", "logits蒸留", "少ないround"],
    ["端末memory", "peak GPU memory", "7B端末", "3Bで削減"],
    ["端末時間", "学習秒／round", "7B端末", "短縮"],
    ["通信", "bytes／round", "LoRA／logits", "同等性能で削減"],
    ["共有data", "|D_align|・属性", "固定公開data", "小規模化"],
    ["モデル差", "3B/3B vs 7B/3B", "同サイズ", "異サイズでも成立"],
    ["新知識", "client-only test F1", "server単独", "client知識を反映"],
  ]);
  addSources(presentation.slides.items[4], [
    "研究会_0724.pdf p.2-p.5, p.8",
    "FedDF: https://arxiv.org/abs/2006.07242",
    "Fed-ET: https://www.ijcai.org/proceedings/2022/399",
  ]);

  // Slide 6 — comparison conditions
  setText(presentation, ids.s6Title, "比較する5条件");
  setText(presentation, ids.s6Page, "6");
  presentation.resolve(ids.s6Placeholder).delete();
  fillTable(presentation, ids.s6Table, [
    ["条件", "server", "client", "役割"],
    ["中央集約上限", "7B＋全data", "—", "性能上限"],
    ["server単独", "7B", "—", "client知識なし"],
    ["client単独", "—", "3B", "局所性能の基準"],
    ["logits蒸留", "7B", "3B", "FedDF／ET型比較"],
    ["提案循環", "7B", "3B", "server・client双方を更新"],
  ]);
  addSources(presentation.slides.items[5], [
    "研究会_0724.pdf p.2-p.5",
    "FedDF: https://arxiv.org/abs/2006.07242",
    "Fed-ET: https://www.ijcai.org/proceedings/2022/399",
  ]);

  // Slide 7 — experiment order
  setText(presentation, ids.s7Title, "実験の順序");
  setText(presentation, ids.s7Page, "7");
  setStructuredText(presentation, ids.s7Body, [
    { runs: ["一度に複数の論点を入れず，外側から検証する"] },
    {
      bulletCharacter: "□",
      marginLeft: 52,
      indent: -28,
      runs: ["段階1：Lpredのみで7B/3Bフローと評価軸を検証する"],
    },
    {
      bulletCharacter: "□",
      marginLeft: 52,
      indent: -28,
      runs: ["段階2：D_alignの量・domain・共有範囲を変えて依存度を測る"],
    },
    {
      bulletCharacter: "□",
      marginLeft: 52,
      indent: -28,
      runs: ["段階3：不足する指標にだけLgrad・Lpost・optimizerを追加する"],
    },
  ]);
  addSources(presentation.slides.items[6], [
    "研究会_0724.pdf p.5-p.6",
  ]);

  // Slide 8 — closing action
  setText(presentation, ids.s8Title, "直近の実施項目");
  setText(presentation, ids.s8Page, "8");
  setStructuredText(presentation, ids.s8Body, [
    { runs: ["まず研究の外側を確定し，実装箇所を後から絞る"] },
    {
      bulletCharacter: "□",
      marginLeft: 52,
      indent: -28,
      runs: ["中心主張を1文に固定する"],
    },
    {
      bulletCharacter: "□",
      marginLeft: 52,
      indent: -28,
      runs: ["関連研究との差分表と成功条件を確定する"],
    },
    {
      bulletCharacter: "□",
      marginLeft: 52,
      indent: -28,
      runs: ["比較条件を実装し，資源・精度・収束・共有量を計測する"],
    },
  ]);
  addSources(presentation.slides.items[7], [
    "研究会_0724.pdf p.5-p.6, p.8",
  ]);

  for (let index = 0; index < presentation.slides.items.length; index += 1) {
    const slide = presentation.slides.items[index];
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(
      `${previewDir}\\${stem}.png`,
      await presentation.export({ slide, format: "png", scale: 1 }),
    );
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(`${layoutDir}\\${stem}.layout.json`, await layout.text());
  }

  await writeBlob(
    `${workspace}\\final-montage.webp`,
    await presentation.export({ format: "webp", montage: true, scale: 1 }),
  );

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(finalPptx);
  console.log(finalPptx);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
