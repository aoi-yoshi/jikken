from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(r"C:\Python\実験_修正")
SOURCE_DIR = ROOT / "notes" / "advisor" / "sources" / "raw" / "2026-09-01"
OUTPUT_DIR = ROOT / "output" / "pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\NotoSansJP-VF.ttf")
FONT_NAME = "NotoSansJP"

GREEN = (0.08, 0.58, 0.43)
GREEN_FILL = (0.66, 0.90, 0.80)
AMBER = (0.83, 0.53, 0.10)
AMBER_FILL = (1.00, 0.88, 0.58)
RED = (0.82, 0.24, 0.20)
RED_FILL = (0.96, 0.72, 0.69)
INK = (0.07, 0.16, 0.25)
WHITE = (1.0, 1.0, 1.0)


def y_from_top(page_height: float, top: float, height: float) -> float:
    return page_height - top - height


def draw_region(
    c: canvas.Canvas,
    page_height: float,
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    stroke: tuple[float, float, float],
    fill: tuple[float, float, float],
    fill_alpha: float = 0.10,
    line_width: float = 1.6,
) -> None:
    y = y_from_top(page_height, top, height)
    c.saveState()
    c.setFillColorRGB(*fill)
    c.setStrokeColorRGB(*stroke)
    c.setFillAlpha(fill_alpha)
    c.setStrokeAlpha(0.90)
    c.setLineWidth(line_width)
    c.roundRect(left, y, width, height, 4, stroke=1, fill=1)
    c.restoreState()


def draw_tag(
    c: canvas.Canvas,
    page_height: float,
    *,
    left: float,
    top: float,
    width: float,
    text: str,
    color: tuple[float, float, float],
    font_size: float = 8.0,
) -> None:
    height = 18
    y = y_from_top(page_height, top, height)
    c.saveState()
    c.setFillColorRGB(*WHITE)
    c.setFillAlpha(0.94)
    c.setStrokeColorRGB(*color)
    c.setStrokeAlpha(1.0)
    c.setLineWidth(1.0)
    c.roundRect(left, y, width, height, 4, stroke=1, fill=1)
    c.setFillColorRGB(*color)
    c.setFillAlpha(1.0)
    c.setFont(FONT_NAME, font_size)
    c.drawCentredString(left + width / 2, y + 5.1, text)
    c.restoreState()


def draw_legend(c: canvas.Canvas, page_height: float, note: str) -> None:
    top = 12
    left = 35
    height = 21
    width = 390
    y = y_from_top(page_height, top, height)
    c.saveState()
    c.setFillColorRGB(1, 1, 1)
    c.setFillAlpha(0.95)
    c.setStrokeColorRGB(0.75, 0.79, 0.83)
    c.setLineWidth(0.7)
    c.roundRect(left, y, width, height, 4, stroke=1, fill=1)
    c.setFont(FONT_NAME, 7.4)
    c.setFillColorRGB(*INK)
    c.drawString(left + 8, y + 6.1, note)
    x = left + 222
    for label, color in (("実施", GREEN), ("一部", AMBER), ("未実施", RED)):
        c.setFillColorRGB(*color)
        c.rect(x, y + 6.5, 7, 7, stroke=0, fill=1)
        c.setFillColorRGB(*INK)
        c.drawString(x + 10, y + 6.0, label)
        x += 52
    c.restoreState()


def overlay_page(width: float, height: float, draw_fn) -> PdfReader:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(width, height))
    draw_fn(c, width, height)
    c.showPage()
    c.save()
    buf.seek(0)
    return PdfReader(buf)


def annotate_pdf(source: Path, output: Path, page_drawers: dict[int, object]) -> None:
    reader = PdfReader(str(source))
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    for page_number, draw_fn in page_drawers.items():
        page = writer.pages[page_number - 1]
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        overlay = overlay_page(width, height, draw_fn)
        page.merge_page(overlay.pages[0], over=True)

    metadata = dict(reader.metadata or {})
    metadata["/Title"] = f"{metadata.get('/Title', source.stem)} - 今回実施範囲マーク"
    metadata["/Subject"] = "2026-09-10の単一client Layer 2 case studyに対応する実施範囲"
    writer.add_metadata(metadata)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        writer.write(stream)


def plan_page_3(c: canvas.Canvas, _w: float, h: float) -> None:
    draw_legend(c, h, "今回の縦切りrun  第I段階 Step I-B")
    draw_region(c, h, left=43, top=512, width=510, height=319, stroke=GREEN, fill=GREEN_FILL, fill_alpha=0.075)
    draw_tag(c, h, left=373, top=514, width=174, text="今回実施：I-B 1〜3", color=GREEN)


def plan_page_4(c: canvas.Canvas, _w: float, h: float) -> None:
    draw_legend(c, h, "今回の縦切りrun  第I段階 Step I-B")
    draw_region(c, h, left=43, top=41, width=510, height=190, stroke=AMBER, fill=AMBER_FILL, fill_alpha=0.075)
    draw_tag(c, h, left=365, top=43, width=182, text="一部：離散変換＋L_match", color=AMBER, font_size=7.4)
    draw_region(c, h, left=43, top=237, width=510, height=81, stroke=AMBER, fill=AMBER_FILL, fill_alpha=0.075)
    draw_tag(c, h, left=387, top=239, width=160, text="pair構成のみ実施", color=AMBER)
    draw_region(c, h, left=43, top=321, width=510, height=277, stroke=RED, fill=RED_FILL, fill_alpha=0.025, line_width=1.8)
    draw_tag(c, h, left=349, top=323, width=198, text="未完：I-C統合・蒸留", color=RED)


def plan_page_10(c: canvas.Canvas, _w: float, h: float) -> None:
    draw_legend(c, h, "Phase対応の一覧  2026-09-10時点")
    draw_tag(c, h, left=378, top=96, width=169, text="今回：Phase 2〜4", color=AMBER)
    draw_region(c, h, left=53, top=159, width=490, height=27, stroke=GREEN, fill=GREEN_FILL, fill_alpha=0.16, line_width=1.0)
    draw_region(c, h, left=53, top=186, width=490, height=25, stroke=GREEN, fill=GREEN_FILL, fill_alpha=0.16, line_width=1.0)
    draw_region(c, h, left=53, top=211, width=490, height=25, stroke=AMBER, fill=AMBER_FILL, fill_alpha=0.18, line_width=1.0)
    draw_region(c, h, left=53, top=236, width=490, height=23, stroke=RED, fill=RED_FILL, fill_alpha=0.13, line_width=1.0)


def method_page_5(c: canvas.Canvas, _w: float, h: float) -> None:
    draw_legend(c, h, "Layer 2の式と今回runの対応")
    draw_region(c, h, left=45, top=82, width=505, height=343, stroke=GREEN, fill=GREEN_FILL, fill_alpha=0.055)
    draw_tag(c, h, left=407, top=84, width=137, text="実施：prototype", color=GREEN)
    draw_region(c, h, left=45, top=428, width=505, height=218, stroke=GREEN, fill=GREEN_FILL, fill_alpha=0.055)
    draw_tag(c, h, left=407, top=430, width=137, text="実施：seed検索", color=GREEN)
    draw_region(c, h, left=45, top=648, width=505, height=188, stroke=AMBER, fill=AMBER_FILL, fill_alpha=0.06)
    draw_tag(c, h, left=361, top=650, width=183, text="一部：変換＋L_match", color=AMBER)


def method_page_6(c: canvas.Canvas, _w: float, h: float) -> None:
    draw_legend(c, h, "Layer 2評価量とLayer 3受渡し")
    draw_region(c, h, left=45, top=37, width=505, height=128, stroke=RED, fill=RED_FILL, fill_alpha=0.05)
    draw_tag(c, h, left=411, top=39, width=133, text="未実施：L_reg", color=RED)
    draw_region(c, h, left=45, top=168, width=505, height=205, stroke=GREEN, fill=GREEN_FILL, fill_alpha=0.045)
    draw_tag(c, h, left=395, top=170, width=149, text="実施：delta MSE", color=GREEN)
    draw_region(c, h, left=45, top=382, width=505, height=232, stroke=AMBER, fill=AMBER_FILL, fill_alpha=0.045)
    draw_tag(c, h, left=382, top=384, width=162, text="pairのみ／蒸留未実施", color=AMBER, font_size=7.2)
    draw_region(c, h, left=45, top=617, width=505, height=217, stroke=RED, fill=RED_FILL, fill_alpha=0.045)
    draw_tag(c, h, left=406, top=619, width=138, text="未実施：Qinv", color=RED)


def method_page_9(c: canvas.Canvas, _w: float, h: float) -> None:
    draw_legend(c, h, "段階的実装計画との対応  2026-09-10時点")
    draw_tag(c, h, left=374, top=417, width=170, text="Phase 2・3実施／4一部", color=AMBER, font_size=7.2)
    draw_region(c, h, left=61, top=501, width=474, height=39, stroke=GREEN, fill=GREEN_FILL, fill_alpha=0.16, line_width=1.0)
    draw_region(c, h, left=61, top=540, width=474, height=34, stroke=GREEN, fill=GREEN_FILL, fill_alpha=0.16, line_width=1.0)
    draw_region(c, h, left=61, top=574, width=474, height=35, stroke=AMBER, fill=AMBER_FILL, fill_alpha=0.18, line_width=1.0)
    draw_region(c, h, left=61, top=598, width=474, height=34, stroke=RED, fill=RED_FILL, fill_alpha=0.13, line_width=1.0)


def main() -> None:
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))

    annotate_pdf(
        SOURCE_DIR / "FedPACT実験計画260901.pdf",
        OUTPUT_DIR / "FedPACT実験計画260901_今回実施範囲マーク.pdf",
        {3: plan_page_3, 4: plan_page_4, 10: plan_page_10},
    )
    annotate_pdf(
        SOURCE_DIR / "FedPACT手法提案260901.pdf",
        OUTPUT_DIR / "FedPACT手法提案260901_今回実施範囲マーク.pdf",
        {5: method_page_5, 6: method_page_6, 9: method_page_9},
    )


if __name__ == "__main__":
    main()
