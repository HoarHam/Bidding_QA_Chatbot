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


# ---------------------------------------------------------------------------
# 以下用例使用**运行时现造**的 xlsx（tests/xlsx_builder.py）。
# 真实语料被 .gitignore 忽略、CI 里不存在，没有这组用例，
# 上面那段手写的 OOXML 遍历逻辑在 CI 中就完全没有覆盖。
# ---------------------------------------------------------------------------


def _synthetic(tmp_path, rows, images_at):
    from tests.xlsx_builder import build_xlsx

    return build_xlsx(tmp_path / "synthetic.xlsx", rows, images_at=images_at)


def test_synthetic_xlsx_roundtrips_through_reader(tmp_path):
    """注入图片之后，文件必须仍是合法 xlsx——否则测试夹具本身就是坏的。"""
    from bidding_qa_chatbot.ingestion.reader import read_qa_rows

    xlsx = _synthetic(
        tmp_path,
        [("问题甲", "答案甲"), ("问题乙", "答案乙"), ("问题丙", "答案丙")],
        images_at=[1],
    )
    rows = read_qa_rows(xlsx)
    assert [r.question for r in rows] == ["问题甲", "问题乙", "问题丙"]


def test_synthetic_image_anchor_maps_to_correct_data_index(tmp_path):
    """图片锚在第 2 个数据行旁，就应当映射到 data_index=1。"""
    xlsx = _synthetic(tmp_path, [("甲", "a"), ("乙", "b"), ("丙", "c")], images_at=[1])

    images = extract_images(xlsx)
    assert len(images) == 1
    img = images[0]
    assert img.anchor_col == 1  # B 列，与真实语料一致
    assert img.to_data_index() == 1
    assert img.to_excel_row() == 3  # 表头 1 行 + 2 行数据
    assert img.data.startswith(b"\x89PNG\r\n\x1a\n")


def test_synthetic_multiple_images_same_anchor(tmp_path):
    """真实语料有 3 张图锚在同一格，必须全部提取且互不覆盖。"""
    xlsx = _synthetic(tmp_path, [("甲", "a"), ("乙", "b")], images_at=[0, 0, 0])

    images = extract_images(xlsx)
    assert len(images) == 3
    assert {img.to_data_index() for img in images} == {0}
    # 内容不同 → sha1 不同 → 视觉缓存不会串味
    assert len({img.sha1 for img in images}) == 3


def test_synthetic_no_images_returns_empty(tmp_path):
    xlsx = _synthetic(tmp_path, [("甲", "a")], images_at=None)
    assert extract_images(xlsx) == []


def test_synthetic_png_is_valid_image(tmp_path):
    """自制的 PNG 必须真的能被解码，否则视觉模型那一步会拿到垃圾。"""
    import struct
    import zlib

    from tests.xlsx_builder import make_png

    data = make_png(width=4, height=3, rgb=(10, 20, 30))
    assert data.startswith(b"\x89PNG\r\n\x1a\n")

    # 解析 IHDR：宽高应与请求一致
    assert data[12:16] == b"IHDR"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (4, 3)

    # 首个 IDAT 能被 zlib 解开，说明数据段没有写坏
    idx = data.index(b"IDAT")
    length = struct.unpack(">I", data[idx - 4 : idx])[0]
    assert zlib.decompress(data[idx + 4 : idx + 4 + length])
