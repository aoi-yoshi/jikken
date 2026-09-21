import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from pypdf import PdfReader


PML_NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
WML_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def natural_key(name: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def pptx_text(path: Path):
    result = {"type": "pptx", "path": str(path), "slides": [], "notes": []}
    with zipfile.ZipFile(path) as archive:
        slide_names = sorted(
            (
                name
                for name in archive.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            ),
            key=natural_key,
        )
        for slide_name in slide_names:
            root = ET.fromstring(archive.read(slide_name))
            paragraphs = []
            for paragraph in root.findall(".//a:p", PML_NS):
                text = "".join(node.text or "" for node in paragraph.findall(".//a:t", PML_NS))
                if text.strip():
                    paragraphs.append(text.strip())
            result["slides"].append(
                {
                    "slide_number": len(result["slides"]) + 1,
                    "text": paragraphs,
                }
            )

        note_names = sorted(
            (
                name
                for name in archive.namelist()
                if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", name)
            ),
            key=natural_key,
        )
        for note_name in note_names:
            root = ET.fromstring(archive.read(note_name))
            paragraphs = []
            for paragraph in root.findall(".//a:p", PML_NS):
                text = "".join(node.text or "" for node in paragraph.findall(".//a:t", PML_NS))
                if text.strip():
                    paragraphs.append(text.strip())
            result["notes"].append(
                {
                    "notes_number": len(result["notes"]) + 1,
                    "text": paragraphs,
                }
            )
    return result


def docx_text(path: Path):
    result = {"type": "docx", "path": str(path), "blocks": []}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
        body = root.find("w:body", WML_NS)
        if body is None:
            return result
        for child in body:
            if child.tag == f"{{{WML_NS['w']}}}p":
                text = "".join(node.text or "" for node in child.findall(".//w:t", WML_NS))
                if text.strip():
                    result["blocks"].append({"kind": "paragraph", "text": text.strip()})
            elif child.tag == f"{{{WML_NS['w']}}}tbl":
                rows = []
                for row in child.findall(".//w:tr", WML_NS):
                    cells = []
                    for cell in row.findall("./w:tc", WML_NS):
                        pieces = []
                        for paragraph in cell.findall(".//w:p", WML_NS):
                            text = "".join(
                                node.text or "" for node in paragraph.findall(".//w:t", WML_NS)
                            )
                            if text.strip():
                                pieces.append(text.strip())
                        cells.append("\n".join(pieces))
                    rows.append(cells)
                result["blocks"].append({"kind": "table", "rows": rows})
    return result


def pdf_text(path: Path):
    result = {"type": "pdf", "path": str(path), "pages": []}
    reader = PdfReader(path)
    for number, page in enumerate(reader.pages, start=1):
        result["pages"].append(
            {
                "page_number": number,
                "text": (page.extract_text() or "").strip(),
            }
        )
    return result


def main():
    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    for raw_path in sys.argv[2:]:
        path = Path(raw_path)
        suffix = path.suffix.lower()
        if suffix == ".pptx":
            data = pptx_text(path)
        elif suffix == ".docx":
            data = docx_text(path)
        elif suffix == ".pdf":
            data = pdf_text(path)
        else:
            raise ValueError(f"Unsupported extension: {suffix}")
        output_path = output_dir / f"{path.stem}.json"
        output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{path.name}\t{output_path}\t{output_path.stat().st_size}")


if __name__ == "__main__":
    main()
