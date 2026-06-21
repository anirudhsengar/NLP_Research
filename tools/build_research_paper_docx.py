#!/usr/bin/env python3
from __future__ import annotations

import html
import re
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any


W_NS = "http://purl.oclc.org/ooxml/wordprocessingml/main"
R_NS = "http://purl.oclc.org/ooxml/officeDocument/relationships"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
W14_NS = "http://schemas.microsoft.com/office/word/2010/wordml"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
IMAGE_REL_TYPE = "http://purl.oclc.org/ooxml/officeDocument/relationships/image"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EMU_PER_INCH = 914400


DOCUMENT_PREFIX = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:a="{A_NS}" xmlns:mc="{MC_NS}" xmlns:pic="{PIC_NS}" xmlns:r="{R_NS}" xmlns:w="{W_NS}" xmlns:w14="{W14_NS}" xmlns:wp="{WP_NS}" mc:Ignorable="w14">
  <w:body>
'''

DOCUMENT_SUFFIX = """  </w:body>
</w:document>
"""


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def text_runs(text: str, *, bold: bool = False, italic: bool = False, size: int | None = None) -> str:
    parts = text.split("\n")
    chunks = []
    for idx, part in enumerate(parts):
        rpr = ""
        if bold or italic or size:
            props = []
            if bold:
                props.append("<w:b/>")
            if italic:
                props.append("<w:i/>")
            if size:
                props.append(f'<w:sz w:val="{size}"/>')
                props.append(f'<w:szCs w:val="{size}"/>')
            rpr = "<w:rPr>" + "".join(props) + "</w:rPr>"
        space = ' xml:space="preserve"' if part.startswith(" ") or part.endswith(" ") else ""
        chunks.append(f"<w:r>{rpr}<w:t{space}>{esc(part)}</w:t></w:r>")
        if idx != len(parts) - 1:
            chunks.append("<w:r><w:br/></w:r>")
    return "".join(chunks)


def paragraph(
    text: str = "",
    *,
    style: str = "BodyText",
    align: str | None = None,
    bold: bool = False,
    italic: bool = False,
    size: int | None = None,
    extra_ppr: str = "",
) -> str:
    ppr = [f'<w:pStyle w:val="{style}"/>']
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    ppr.append(extra_ppr)
    return "    <w:p><w:pPr>" + "".join(ppr) + "</w:pPr>" + text_runs(
        text, bold=bold, italic=italic, size=size
    ) + "</w:p>\n"


def manual_heading1(text: str, number: int) -> str:
    return paragraph(
        f"{roman(number)}. {text.upper()}",
        style="BodyText",
        align="center",
        bold=True,
        size=20,
        extra_ppr='<w:spacing w:before="8pt" w:after="4pt"/><w:ind w:firstLine="0pt"/>',
    )


def manual_heading2(text: str, number: int) -> str:
    letter = chr(ord("A") + number)
    return paragraph(
        f"{letter}. {text}",
        style="BodyText",
        italic=True,
        size=18,
        extra_ppr='<w:spacing w:before="6pt" w:after="3pt"/><w:ind w:firstLine="0pt"/>',
    )


def manual_component_heading(text: str) -> str:
    return paragraph(
        text,
        style="BodyText",
        align="center",
        bold=True,
        size=18,
        extra_ppr='<w:spacing w:before="8pt" w:after="4pt"/><w:ind w:firstLine="0pt"/>',
    )


def roman(number: int) -> str:
    values = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    result = []
    remaining = number
    for value, symbol in values:
        while remaining >= value:
            result.append(symbol)
            remaining -= value
    return "".join(result)


def section_break_to_two_columns() -> str:
    sect = """
<w:sectPr>
  <w:type w:val="continuous"/>
  <w:pgSz w:w="612pt" w:h="792pt" w:code="1"/>
  <w:pgMar w:top="54pt" w:right="44.65pt" w:bottom="72pt" w:left="44.65pt" w:header="36pt" w:footer="36pt" w:gutter="0pt"/>
  <w:cols w:space="36pt"/>
  <w:docGrid w:linePitch="360"/>
</w:sectPr>
"""
    return paragraph("", style="BodyText", extra_ppr=sect)


def final_two_column_section() -> str:
    return """
    <w:sectPr>
      <w:type w:val="continuous"/>
      <w:pgSz w:w="612pt" w:h="792pt" w:code="1"/>
      <w:pgMar w:top="54pt" w:right="44.65pt" w:bottom="72pt" w:left="44.65pt" w:header="36pt" w:footer="36pt" w:gutter="0pt"/>
      <w:cols w:num="2" w:space="18pt"/>
      <w:docGrid w:linePitch="360"/>
    </w:sectPr>
"""


def table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    col_count = max(len(row) for row in rows)
    width_pt = max(34.0, 250.0 / col_count)
    xml = [
        '    <w:tbl><w:tblPr><w:tblW w:w="250pt" w:type="dxa"/>',
        '<w:tblLayout w:type="fixed"/>',
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>',
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>',
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>',
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/>',
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/>',
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/></w:tblBorders>',
        "</w:tblPr><w:tblGrid>",
    ]
    xml.extend([f'<w:gridCol w:w="{width_pt:.2f}pt"/>' for _ in range(col_count)])
    xml.append("</w:tblGrid>")
    for row_index, row in enumerate(rows):
        xml.append("<w:tr><w:trPr><w:cantSplit/></w:trPr>")
        padded = row + [""] * (col_count - len(row))
        for cell in padded:
            xml.append(f'<w:tc><w:tcPr><w:tcW w:w="{width_pt:.2f}pt" w:type="dxa"/></w:tcPr>')
            xml.append(
                paragraph(
                    cell,
                    style="tablecolhead" if row_index == 0 else "tablecopy",
                    bold=row_index == 0,
                    size=13,
                ).strip()
            )
            xml.append("</w:tc>")
        xml.append("</w:tr>")
    xml.append("</w:tbl>\n")
    return "".join(xml)


def image_block(image: dict[str, Any]) -> str:
    caption = str(image["caption"])
    width_px = int(image["width_px"])
    height_px = int(image["height_px"])
    figure_width_inches = 3.35
    cx = int(figure_width_inches * EMU_PER_INCH)
    cy = int(cx * height_px / max(width_px, 1))
    name = esc(str(image["media_name"]))
    caption_xml = esc(caption)
    r_id = esc(str(image["r_id"]))
    doc_pr_id = int(image["doc_pr_id"])
    drawing = f'''
    <w:p>
      <w:pPr><w:keepNext/><w:keepLines/><w:jc w:val="center"/><w:spacing w:before="4pt" w:after="2pt"/><w:ind w:firstLine="0pt"/></w:pPr>
      <w:r>
        <w:drawing>
          <wp:inline distT="0" distB="0" distL="0" distR="0">
            <wp:extent cx="{cx}" cy="{cy}"/>
            <wp:effectExtent l="0" t="0" r="0" b="0"/>
            <wp:docPr id="{doc_pr_id}" name="{name}" descr="{caption_xml}"/>
            <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
            <a:graphic>
              <a:graphicData uri="{PIC_NS}">
                <pic:pic>
                  <pic:nvPicPr>
                    <pic:cNvPr id="0" name="{name}" descr="{caption_xml}"/>
                    <pic:cNvPicPr/>
                  </pic:nvPicPr>
                  <pic:blipFill>
                    <a:blip r:embed="{r_id}"/>
                    <a:stretch><a:fillRect/></a:stretch>
                  </pic:blipFill>
                  <pic:spPr>
                    <a:xfrm>
                      <a:off x="0" y="0"/>
                      <a:ext cx="{cx}" cy="{cy}"/>
                    </a:xfrm>
                    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
                  </pic:spPr>
                </pic:pic>
              </a:graphicData>
            </a:graphic>
          </wp:inline>
        </w:drawing>
      </w:r>
    </w:p>
'''
    caption_paragraph = paragraph(
        caption,
        style="BodyText",
        align="center",
        italic=True,
        size=14,
        extra_ppr='<w:keepLines/><w:spacing w:before="0pt" w:after="6pt"/><w:ind w:firstLine="0pt"/>',
    )
    return drawing + caption_paragraph


def parse_markdown(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    title = lines[0].removeprefix("# ").strip()
    authors = ""
    affiliation = ""
    sections: list[tuple[str, list[str]]] = []
    current_heading = None
    current_lines: list[str] = []

    for line in lines[1:]:
        if line.startswith("Authors:"):
            authors = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Affiliation:"):
            affiliation = line.split(":", 1)[1].strip()
            continue
        if line.startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, current_lines))
            current_heading = line[3:].strip()
            current_lines = []
        else:
            if current_heading is not None:
                current_lines.append(line)
    if current_heading is not None:
        sections.append((current_heading, current_lines))
    return {
        "title": title,
        "authors": authors,
        "affiliation": affiliation,
        "sections": sections,
    }


def consume_blocks(lines: list[str]) -> list[tuple[str, object]]:
    blocks: list[tuple[str, object]] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue
        if line.startswith("### "):
            blocks.append(("h2", line[4:].strip()))
            idx += 1
            continue
        image_match = image_match_for(line)
        if image_match is not None:
            blocks.append(("image", image_match))
            idx += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while idx < len(lines) and lines[idx].startswith("|"):
                if not re.fullmatch(r"\|[\s:\-|]+\|", lines[idx].strip()):
                    table_lines.append(lines[idx])
                idx += 1
            rows = [
                [cell.strip() for cell in row.strip().strip("|").split("|")]
                for row in table_lines
            ]
            blocks.append(("table", rows))
            continue
        if re.match(r"\[\d+\] ", line):
            refs = []
            while idx < len(lines):
                ref_line = lines[idx].strip()
                if ref_line:
                    refs.append(ref_line)
                idx += 1
            blocks.append(("refs", refs))
            continue
        paragraph_lines = [line.strip()]
        idx += 1
        while idx < len(lines) and lines[idx].strip() and not lines[idx].startswith(("### ", "|")):
            if image_match_for(lines[idx]) is not None:
                break
            if re.match(r"\[\d+\] ", lines[idx]):
                break
            paragraph_lines.append(lines[idx].strip())
            idx += 1
        blocks.append(("p", " ".join(paragraph_lines)))
    return blocks


def image_match_for(line: str) -> dict[str, str] | None:
    match = re.fullmatch(r"!\[(?P<caption>.*?)\]\((?P<path>.*?)\)", line.strip())
    if match is None:
        return None
    return {"caption": match.group("caption").strip(), "path": match.group("path").strip()}


def section_to_xml(
    heading: str,
    lines: list[str],
    *,
    context: dict[str, Any],
    section_number: int | None = None,
) -> str:
    if heading == "Abstract":
        text = " ".join(line.strip() for line in lines if line.strip())
        return paragraph("Abstract- " + text, style="Abstract")
    if heading == "Keywords":
        text = " ".join(line.strip() for line in lines if line.strip())
        return paragraph("Keywords- " + text, style="Keywords")

    if heading in {"Acknowledgment", "References"}:
        out = [manual_component_heading(heading)]
    else:
        if section_number is None:
            raise ValueError(f"Missing section number for heading {heading!r}")
        out = [manual_heading1(heading, section_number)]
    subsection_number = 0
    for kind, payload in consume_blocks(lines):
        if kind == "h2":
            out.append(manual_heading2(str(payload), subsection_number))
            subsection_number += 1
        elif kind == "table":
            out.append(table(payload))  # type: ignore[arg-type]
        elif kind == "image":
            out.append(image_block(register_image(context, payload)))  # type: ignore[arg-type]
        elif kind == "refs":
            for ref in payload:  # type: ignore[union-attr]
                out.append(paragraph(ref, style="BodyText"))
        else:
            text = str(payload)
            if text.startswith("TABLE "):
                out.append(paragraph(text, style="BodyText", extra_ppr="<w:keepNext/>"))
            elif re.match(r"\d+\. ", text):
                out.append(paragraph(text, style="BodyText"))
            else:
                out.append(paragraph(text, style="BodyText"))
    return "".join(out)


def register_image(context: dict[str, Any], payload: dict[str, str]) -> dict[str, Any]:
    path = Path(payload["path"])
    if not path.is_absolute():
        path = context["markdown_dir"] / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Figure image not found: {path}")
    if path.suffix.lower() != ".png":
        raise ValueError(f"Only PNG figures are supported by this builder: {path}")
    width_px, height_px = png_dimensions(path)
    image_index = len(context["images"]) + 1
    media_name = f"paper_figure_{image_index}.png"
    image = {
        "caption": payload["caption"],
        "path": path,
        "r_id": f"rIdFigure{image_index}",
        "media_name": media_name,
        "doc_pr_id": 1000 + image_index,
        "width_px": width_px,
        "height_px": height_px,
    }
    context["images"].append(image)
    return image


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or not header.startswith(PNG_SIGNATURE):
        raise ValueError(f"Not a valid PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def build_document_xml(data: dict, context: dict[str, Any]) -> str:
    pieces = [DOCUMENT_PREFIX]
    pieces.append(paragraph(data["title"], style="papertitle", align="center"))
    pieces.append(paragraph(data["authors"], style="Author", align="center", size=18))
    pieces.append(paragraph(data["affiliation"], style="Author", align="center", italic=True, size=16))

    body_started = False
    section_number = 1
    for heading, lines in data["sections"]:
        if heading not in {"Abstract", "Keywords"} and not body_started:
            pieces.append(section_break_to_two_columns())
            body_started = True
        if heading in {"Abstract", "Keywords", "Acknowledgment", "References"}:
            pieces.append(section_to_xml(heading, lines, context=context))
        else:
            pieces.append(section_to_xml(heading, lines, context=context, section_number=section_number))
            section_number += 1
    pieces.append(final_two_column_section())
    pieces.append(DOCUMENT_SUFFIX)
    return "".join(pieces)


def build_docx(template: Path, markdown: Path, output: Path) -> None:
    data = parse_markdown(markdown)
    context: dict[str, Any] = {"markdown_dir": markdown.resolve().parent, "images": []}
    document_xml = build_document_xml(data, context).encode("utf-8")
    title = esc(data["title"])
    core_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{title}</dc:title>
  <dc:creator>CSAI411 project team</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
</cp:coreProperties>
'''.encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(template, "r") as zin:
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                content = zin.read(item.filename)
                if item.filename == "word/document.xml":
                    content = document_xml
                elif item.filename == "docProps/core.xml":
                    content = core_xml
                elif item.filename == "word/_rels/document.xml.rels":
                    content = add_image_relationships(content.decode("utf-8"), context["images"]).encode("utf-8")
                elif item.filename == "[Content_Types].xml":
                    content = add_png_content_type(content.decode("utf-8")).encode("utf-8")
                zout.writestr(item, content)
            for image in context["images"]:
                zout.writestr(f"word/media/{image['media_name']}", Path(image["path"]).read_bytes())


def add_image_relationships(content: str, images: list[dict[str, Any]]) -> str:
    if not images:
        return content
    additions = "".join(
        f'<Relationship Id="{esc(str(image["r_id"]))}" Type="{IMAGE_REL_TYPE}" '
        f'Target="media/{esc(str(image["media_name"]))}"/>'
        for image in images
    )
    return content.replace("</Relationships>", additions + "</Relationships>")


def add_png_content_type(content: str) -> str:
    if 'Extension="png"' in content:
        return content
    return content.replace("</Types>", '<Default Extension="png" ContentType="image/png"/></Types>')


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: build_research_paper_docx.py TEMPLATE.docx DRAFT.md OUTPUT.docx", file=sys.stderr)
        return 2
    build_docx(Path(argv[1]), Path(argv[2]), Path(argv[3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
