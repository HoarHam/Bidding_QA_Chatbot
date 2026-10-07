"""稳定的分块标识。

重跑入库必须收敛到同一批 point，而不是不断堆积新点，所以：

- **不用 `uuid4`**——每次入库都会产生新点，集合会无限膨胀。
- **不用问题文本当 ID**——改一个标点就会让旧点变成孤儿。
- 用 `uuid5(NAMESPACE_URL, chunk_id)`：`chunk_id` 由数据集名 + 序号构成，
  完全确定性，于是「先删后插」变成纯粹的幂等覆盖。
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

#: uuid5 的命名空间。固定常量，改动会使所有既有 point 失效。
_NAMESPACE = uuid.NAMESPACE_URL


def dataset_name_from_path(xlsx_path: Path) -> str:
    """从文件名推导数据集名，如 `问题答案样例数据0113.xlsx` → `qa0113`。

    取不到数字时退化为文件名的哈希前缀，保证仍然稳定且唯一。
    """
    stem = xlsx_path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return f"qa{digits}"
    return f"ds{hashlib.sha1(stem.encode('utf-8')).hexdigest()[:6]}"


def make_chunk_id(dataset: str, data_index: int) -> str:
    return f"{dataset}-{data_index:04d}"


def make_point_id(chunk_id: str) -> str:
    """Qdrant 只接受 uint64 或 UUID，故用 uuid5 把字符串映射成确定性 UUID。"""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def content_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def make_source_label(excel_row: int, dataset_cn: str = "招投标问答样例数据0113") -> str:
    """给人看的引用标签，会直接显示在「参考来源」卡片上。"""
    return f"{dataset_cn} · 第{excel_row}行"


def make_image_url(chunk_id: str, index: int) -> str:
    """图片的对外访问路径，供前端 `<img src>` 使用。"""
    return f"/api/sources/{chunk_id}/images/{index}"
