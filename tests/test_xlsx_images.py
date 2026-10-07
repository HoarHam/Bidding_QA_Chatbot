"""图片锚点提取测试。

这是本阶段最容易出错的自研逻辑（手写 OOXML 遍历），所以拆成两层：

1. **XML 层面的单元测试**——直接喂手写 XML，验证命名空间处理、
   相对/绝对路径解析、`../` 归一化、锚点坐标提取。这些在 CI 里也能跑，
   因为不需要真实语料。
2. **真实语料的端到端断言**——挂在 `requires_corpus` 上，本地跑。
"""

from __future__ import annotations

import zipfile

from bidding_qa_chatbot.ingestion.xlsx_images import (
    EmbeddedImage,
    _parse_anchors,
    _rels_for,
    extract_images,
)
from tests.conftest import CORPUS_XLSX, requires_corpus

DRAWING_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
          xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <xdr:twoCellAnchor>
    <xdr:from><xdr:col>1</xdr:col><xdr:colOff>0</xdr:colOff>
              <xdr:row>51</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:from>
    <xdr:to><xdr:col>8</xdr:col><xdr:row>60</xdr:row></xdr:to>
    <xdr:pic>
      <xdr:nvPicPr><xdr:cNvPr id="1" name="Picture 1"/></xdr:nvPicPr>
      <xdr:blipFill><a:blip r:embed="rId1"/></xdr:blipFill>
    </xdr:pic>
  </xdr:twoCellAnchor>
  <xdr:twoCellAnchor>
    <xdr:from><xdr:col>1</xdr:col><xdr:row>51</xdr:row></xdr:from>
    <xdr:to><xdr:col>8</xdr:col><xdr:row>60</xdr:row></xdr:to>
    <xdr:pic><xdr:blipFill><a:blip r:embed="rId2"/></xdr:blipFill></xdr:pic>
  </xdr:twoCellAnchor>
  <xdr:absoluteAnchor>
    <xdr:pic><xdr:blipFill><a:blip r:embed="rId3"/></xdr:blipFill></xdr:pic>
  </xdr:absoluteAnchor>
</xdr:wsDr>
"""

DRAWING_RELS = {
    "rId1": "xl/media/image1.png",
    "rId2": "xl/media/image2.png",
    "rId3": "xl/media/image3.png",
}


def test_parse_anchors_returns_row_col_and_rid():
    anchors = _parse_anchors(DRAWING_XML, DRAWING_RELS)
    # absoluteAnchor 没有 from 坐标，无法映射到行，应被跳过
    assert anchors == [(51, 1, "rId1"), (51, 1, "rId2")]


def test_parse_anchors_skips_unresolvable_rid():
    """rId 在 rels 里找不到时必须跳过，而不是抛错——坏文件不该让入库整体失败。"""
    anchors = _parse_anchors(DRAWING_XML, {"rId1": "xl/media/image1.png"})
    assert anchors == [(51, 1, "rId1")]


def test_anchor_row_maps_to_data_index():
    """锚点 0 基行 51 → Excel 第 52 行 → 问答对序号 50。"""
    img = EmbeddedImage(name="image1.png", data=b"x", anchor_row=51, anchor_col=1)
    assert img.to_excel_row() == 52
    assert img.to_data_index() == 50


def test_rels_for_resolves_relative_and_parent_segments(tmp_path):
    xlsx = tmp_path / "b.xlsx"
    with zipfile.ZipFile(xlsx, "w") as zf:
        zf.writestr(
            "xl/worksheets/_rels/sheet1.xml.rels",
            """<?xml version="1.0"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type=".../drawing" Target="../drawings/drawing1.xml"/>
  <Relationship Id="rId2" Type="..." Target="/xl/media/image1.png"/>
</Relationships>""",
        )
    with zipfile.ZipFile(xlsx) as zf:
        rels = _rels_for(zf, "xl/worksheets/sheet1.xml")

    # ../ 被正确归一化，绝对路径去掉前导 /
    assert rels["rId1"] == "xl/drawings/drawing1.xml"
    assert rels["rId2"] == "xl/media/image1.png"


def test_rels_for_missing_returns_empty(tmp_path):
    xlsx = tmp_path / "b.xlsx"
    with zipfile.ZipFile(xlsx, "w") as zf:
        zf.writestr("dummy.txt", "x")
    with zipfile.ZipFile(xlsx) as zf:
        assert _rels_for(zf, "xl/worksheets/sheet1.xml") == {}


def test_extract_images_on_file_without_drawings_returns_empty(tmp_path):
    """没有图不是错误——入库不该因为没有图片而失败。"""
    xlsx = tmp_path / "b.xlsx"
    with zipfile.ZipFile(xlsx, "w") as zf:
        zf.writestr("xl/workbook.xml", "<workbook/>")
    assert extract_images(xlsx) == []


@requires_corpus
def test_real_corpus_has_three_anchored_images():
    images = extract_images(CORPUS_XLSX)
    assert len(images) == 3
    for img in images:
        assert img.anchor_row == 51
        assert img.anchor_col == 1
        assert img.to_data_index() == 50
        assert img.data[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic
    # 三张图内容不同，sha1 应互不相同（否则视觉缓存会串味）
    assert len({img.sha1 for img in images}) == 3
