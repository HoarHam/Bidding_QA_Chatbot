"""从 xlsx 中提取内嵌图片，并定位每张图锚定在哪一行。

为什么不用 pandas：`pd.read_excel` 只读单元格值，**内嵌图片会被静默丢弃**。
本语料里有 3 张 PNG（锚定在同一行的 B 列），其中那一行的答案正文本身就是
「2023年上半年手术机器人招采情况如下：」——真正的答案全在图片里，
不提取图片这一行就是无法回答的。

为什么不用 openpyxl 的 `ws._images`：那条路依赖 Pillow 才能物化 Image 对象，
且不同版本填充行为不一致；而我们要的恰恰是 **锚点坐标**，
直接遍历 OOXML 的 drawing 部件最稳定，也不引入额外依赖。

对应关系（xlsx 是 zip 包）：
    xl/workbook.xml + xl/_rels/workbook.xml.rels      → 定位目标 sheet 部件
    xl/worksheets/_rels/sheetN.xml.rels               → 找到 drawing 部件
    xl/drawings/drawingN.xml                          → 锚点(from/row) + r:embed(rId)
    xl/drawings/_rels/drawingN.xml.rels               → rId → xl/media/imageN.png
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {
    "xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
}

#: 三种锚点形式。实际语料用的是 twoCellAnchor，但另两种在别处很常见，一并处理。
_ANCHOR_TAGS = ("twoCellAnchor", "oneCellAnchor", "absoluteAnchor")

_REL_NS = NS["pr"]
_PACKAGE_REL_DRAWING = "/drawing"


@dataclass(frozen=True, slots=True)
class EmbeddedImage:
    """一张内嵌图片及其锚点。

    `anchor_row` 是 0 基行号（OOXML 约定），`anchor_col` 是 0 基列号。
    转成用户可见的 Excel 行号需要 +1，见 `to_excel_row()`。
    """

    name: str
    data: bytes
    anchor_row: int
    anchor_col: int

    @property
    def sha1(self) -> str:
        return hashlib.sha1(self.data).hexdigest()

    @property
    def suffix(self) -> str:
        return Path(self.name).suffix.lower() or ".png"

    def to_excel_row(self) -> int:
        """转成 1 基的 Excel 行号（与 `reader.QARow.excel_row` 同一坐标系）。"""
        return self.anchor_row + 1

    def to_data_index(self) -> int:
        """锚点行 → 问答对序号。

        表头占第 1 行，所以 0 基锚点行 `r` 对应第 `r+1` 行；
        该行的问答对序号 = `(r+1) - 2` = `r - 1`。
        例：锚点 row=51 → Excel 第 52 行 → 序号 50。
        """
        return self.anchor_row - 1


def _rels_for(zf: zipfile.ZipFile, part: str) -> dict[str, str]:
    """读取某个部件的 .rels，返回 {rId: 目标路径（已相对包根归一化）}。"""
    part_dir = part.rsplit("/", 1)[0] if "/" in part else ""
    rels_path = f"{part_dir}/_rels/{part.rsplit('/', 1)[-1]}.rels"
    if rels_path not in zf.namelist():
        return {}

    root = ET.fromstring(zf.read(rels_path))
    out: dict[str, str] = {}
    for rel in root.findall(f"{{{_REL_NS}}}Relationship"):
        target = rel.get("Target", "")
        rid = rel.get("Id", "")
        if not rid or not target:
            continue
        if target.startswith("/"):
            resolved = target.lstrip("/")
        else:
            resolved = f"{part_dir}/{target}" if part_dir else target
        # 归一化 ../ 片段
        parts: list[str] = []
        for seg in resolved.split("/"):
            if seg == "..":
                if parts:
                    parts.pop()
            elif seg not in ("", "."):
                parts.append(seg)
        out[rid] = "/".join(parts)
    return out


def _find_sheet_part(zf: zipfile.ZipFile, sheet_name: str) -> str | None:
    """按 sheet 名找到对应的 xl/worksheets/sheetN.xml。"""
    try:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
    except KeyError:
        return None
    rels = _rels_for(zf, "xl/workbook.xml")
    for sheet in wb.findall(f"{{{NS['main']}}}sheets/{{{NS['main']}}}sheet"):
        if sheet.get("name") != sheet_name:
            continue
        rid = sheet.get(f"{{{NS['r']}}}id")
        if rid and rid in rels:
            return rels[rid]
    return None


def _parse_anchors(drawing_xml: bytes, drawing_rels: dict[str, str]) -> list[tuple[int, int, str]]:
    """返回 [(row, col, rId)]，跳过拿不到坐标或图片引用的锚点。"""
    root = ET.fromstring(drawing_xml)
    found: list[tuple[int, int, str]] = []

    for anchor_tag in _ANCHOR_TAGS:
        for anchor in root.findall(f"{{{NS['xdr']}}}{anchor_tag}"):
            frm = anchor.find(f"{{{NS['xdr']}}}from")
            if frm is None:
                # absoluteAnchor 用绝对坐标，无法映射到行，跳过（语料中不存在）
                continue
            col_el = frm.find(f"{{{NS['xdr']}}}col")
            row_el = frm.find(f"{{{NS['xdr']}}}row")
            if col_el is None or row_el is None or col_el.text is None or row_el.text is None:
                continue

            blip = anchor.find(f".//{{{NS['a']}}}blip")
            if blip is None:
                continue
            rid = blip.get(f"{{{NS['r']}}}embed")
            if not rid or rid not in drawing_rels:
                continue

            found.append((int(row_el.text), int(col_el.text), rid))

    return found


def extract_images(xlsx_path: Path, sheet_name: str = "Sheet1") -> list[EmbeddedImage]:
    """提取指定 sheet 上的全部内嵌图片（按锚点顺序）。

    找不到 drawing 部件时返回空列表而非抛错——并非每个 xlsx 都有图，
    入库流程不应该因为「没有图」而失败。
    """
    images: list[EmbeddedImage] = []

    with zipfile.ZipFile(xlsx_path) as zf:
        names = set(zf.namelist())
        sheet_part = _find_sheet_part(zf, sheet_name)
        if sheet_part is None or sheet_part not in names:
            return []

        sheet_rels = _rels_for(zf, sheet_part)
        drawing_parts = [
            target
            for target in sheet_rels.values()
            if target.endswith(".xml") and "/drawings/" in target
        ]
        if not drawing_parts:
            # 退一步：某些写出器不经过 rels，直接扫 drawings 目录
            drawing_parts = sorted(n for n in names if n.startswith("xl/drawings/drawing"))

        for drawing_part in drawing_parts:
            if drawing_part not in names:
                continue
            drawing_rels = _rels_for(zf, drawing_part)
            for row, col, rid in _parse_anchors(zf.read(drawing_part), drawing_rels):
                media_part = drawing_rels[rid]
                if media_part not in names:
                    continue
                images.append(
                    EmbeddedImage(
                        name=media_part.rsplit("/", 1)[-1],
                        data=zf.read(media_part),
                        anchor_row=row,
                        anchor_col=col,
                    )
                )

    return images


def save_images(images: list[EmbeddedImage], dest_dir: Path) -> list[Path]:
    """把图片落盘到 dest_dir，返回写入的路径列表。同名文件直接覆盖。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for img in images:
        out = dest_dir / f"{img.sha1[:8]}-{img.name}"
        out.write_bytes(img.data)
        written.append(out)
    return written
