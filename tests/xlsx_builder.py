"""构造带内嵌图片的 xlsx，供测试使用。

## 为什么需要它

真实语料 `data/processed/*.xlsx` 已被 `.gitignore` 忽略（`data/` 整目录不入库），
CI 里根本不存在，于是 `ingestion/xlsx_images.py` 这段手写的 OOXML 遍历逻辑——
本阶段最容易出错的自研代码——在 CI 里完全没有覆盖。

这里的做法是**在测试运行时现造一个结构等价的 xlsx**：不往仓库里塞二进制文件，
又能让 CI 走到完整链路。

## 为什么不用 openpyxl 加图片

`ws.add_image()` 需要 **Pillow** 才能把图片物化成对象，而本项目没有（也不需要）
安装 Pillow。所以先用 openpyxl 生成纯文本工作表，再往 zip 里**注入**
drawing / rels / media 三件套，并 patch 两处 XML。注入的结构严格遵循 OOXML：

    [Content_Types].xml                     ← 补 png 的 Default、drawing 的 Override
    xl/worksheets/sheet1.xml                ← 补 <drawing r:id="rId1"/>
    xl/worksheets/_rels/sheet1.xml.rels     ← rId1 → ../drawings/drawing1.xml
    xl/drawings/drawing1.xml                ← 锚点坐标 + r:embed
    xl/drawings/_rels/drawing1.xml.rels     ← rId → ../media/imageN.png
    xl/media/imageN.png                     ← 图片本体
"""

from __future__ import annotations

import re
import shutil
import struct
import zipfile
import zlib
from pathlib import Path

import openpyxl

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_A_TAG = "http://schemas.openxmlformats.org/drawingml/2006/main"
_R_TAG = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_XDR_TAG = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"

_DRAWING_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing"
_IMAGE_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
_DRAWING_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.drawing+xml"


def make_png(width: int = 8, height: int = 8, rgb: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    """生成一张纯色 PNG。

    用 stdlib 的 `struct` + `zlib` 手写，避免为了测试引入 Pillow。
    8 位真彩色（color type 2），每条扫描线前置一个 filter 字节 0。
    """

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (
        PNG_SIGNATURE
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _anchor_xml(row: int, col: int, rid: str) -> str:
    """一个 twoCellAnchor，锚定在 0 基的 (row, col)。"""
    return f"""  <xdr:twoCellAnchor>
    <xdr:from>
      <xdr:col>{col}</xdr:col><xdr:colOff>0</xdr:colOff>
      <xdr:row>{row}</xdr:row><xdr:rowOff>0</xdr:rowOff>
    </xdr:from>
    <xdr:to>
      <xdr:col>{col + 4}</xdr:col><xdr:colOff>0</xdr:colOff>
      <xdr:row>{row + 8}</xdr:row><xdr:rowOff>0</xdr:rowOff>
    </xdr:to>
    <xdr:pic>
      <xdr:nvPicPr>
        <xdr:cNvPr id="{row + 1}" name="Picture {row + 1}"/>
        <xdr:cNvPicPr/>
      </xdr:nvPicPr>
      <xdr:blipFill>
        <a:blip xmlns:r="{_R_TAG}" r:embed="{rid}"/>
        <a:stretch><a:fillRect/></a:stretch>
      </xdr:blipFill>
      <xdr:spPr>
        <a:xfrm><a:off x="0" y="0"/><a:ext cx="1000000" cy="1000000"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
      </xdr:spPr>
    </xdr:pic>
    <xdr:clientData/>
  </xdr:twoCellAnchor>
"""


def _drawing_xml(anchors: list[tuple[int, int, str]]) -> bytes:
    body = "".join(_anchor_xml(row, col, rid) for row, col, rid in anchors)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<xdr:wsDr xmlns:xdr="{_XDR_TAG}" xmlns:a="{_A_TAG}">\n'
        f"{body}</xdr:wsDr>"
    ).encode()


def _rels_xml(entries: list[tuple[str, str, str]]) -> bytes:
    body = "".join(
        f'<Relationship Id="{rid}" Type="{rtype}" Target="{target}"/>'
        for rid, rtype, target in entries
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{body}</Relationships>"
    ).encode()


def build_xlsx(
    path: Path,
    rows: list[tuple[str, str]],
    images_at: list[int] | None = None,
    header: tuple[str, str] = ("问", "答"),
    sheet_name: str = "Sheet1",
) -> Path:
    """生成一个 xlsx，可选地在指定数据行旁嵌入图片。

    `images_at` 是**数据行的 0 基序号**列表；第 n 行对应 Excel 第 `n + 2` 行
    （表头占第 1 行），因此锚点用的 0 基行号是 `n + 1`。
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append(list(header))
    for question, answer in rows:
        ws.append([question, answer])
    wb.save(path)

    if not images_at:
        return path

    anchors: list[tuple[int, int, str]] = []
    rel_entries: list[tuple[str, str, str]] = []
    media: dict[str, bytes] = {}

    for i, data_index in enumerate(images_at, start=1):
        rid = f"rId{i}"
        # 表头 + 前面若干数据行 → 该图片所在行的 0 基锚点行号
        anchor_row = data_index + 1
        anchors.append((anchor_row, 1, rid))
        rel_entries.append((rid, _IMAGE_REL_TYPE, f"../media/image{i}.png"))
        media[f"xl/media/image{i}.png"] = make_png(rgb=(30 * i % 256, 90, 200))

    _inject(path, anchors, rel_entries, media)
    return path


def _inject(
    path: Path,
    anchors: list[tuple[int, int, str]],
    rel_entries: list[tuple[str, str, str]],
    media: dict[str, bytes],
) -> None:
    """把 drawing / rels / media 注入已生成的 xlsx（原地重写 zip）。"""
    tmp = path.with_suffix(".tmp.xlsx")

    with zipfile.ZipFile(path) as src:
        names = set(src.namelist())
        existing_sheet_rels = (
            src.read("xl/worksheets/_rels/sheet1.xml.rels")
            if "xl/worksheets/_rels/sheet1.xml.rels" in names
            else None
        )
        parts = {name: src.read(name) for name in src.namelist()}

    # 1) 工作表引用 drawing 部件
    sheet_path = "xl/worksheets/sheet1.xml"
    sheet = parts[sheet_path].decode("utf-8")
    if "xmlns:r=" not in sheet:
        sheet = sheet.replace(
            "<worksheet ",
            f'<worksheet xmlns:r="{_R_TAG}" ',
            1,
        )
    sheet = re.sub(
        r"</worksheet>",
        '<drawing r:id="rId1"/></worksheet>',
        sheet,
        count=1,
    )
    parts[sheet_path] = sheet.encode("utf-8")

    # 2) 工作表的 rels 指向 drawing
    sheet_rel_entries: list[tuple[str, str, str]] = []
    if existing_sheet_rels:
        sheet_rel_entries.extend(
            (m[0], m[1], m[2])
            for m in re.findall(
                rb'Id="([^"]+)" Type="([^"]+)" Target="([^"]+)"', existing_sheet_rels
            )
        )
    sheet_rel_entries.append(("rId1", _DRAWING_REL_TYPE, "../drawings/drawing1.xml"))
    parts["xl/worksheets/_rels/sheet1.xml.rels"] = _rels_xml(sheet_rel_entries)

    # 3) drawing 本体与它自己的 rels
    parts["xl/drawings/drawing1.xml"] = _drawing_xml(anchors)
    parts["xl/drawings/_rels/drawing1.xml.rels"] = _rels_xml(rel_entries)

    # 4) 图片本体
    parts.update(media)

    # 5) [Content_Types].xml 补两个声明，否则 Excel/openpyxl 打不开
    ct = parts["[Content_Types].xml"].decode("utf-8")
    if 'Extension="png"' not in ct:
        ct = ct.replace(
            "</Types>",
            '<Default Extension="png" ContentType="image/png"/>'
            f'<Override PartName="/xl/drawings/drawing1.xml" ContentType="{_DRAWING_CONTENT_TYPE}"/>'
            "</Types>",
            1,
        )
    parts["[Content_Types].xml"] = ct.encode("utf-8")

    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for name, data in parts.items():
            out.writestr(name, data)

    shutil.move(str(tmp), str(path))
