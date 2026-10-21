"""附件上传与解析：存储、格式提取、OCR 上下文组装。

- `store.py`    上传校验/落盘/sidecar 元数据/所有权校验/TTL 清理
- `extract.py`  图片直接 OCR；PDF 渲染逐页 OCR；DOCX 正文 + 内嵌图片 OCR
- `service.py`  本轮解析、历史附件注入、落库形态

OCR 由 `llm/ocr.py` 的 `OllamaOCR`（本地 GLM-OCR，原生 /api/generate）提供。
"""

from bidding_qa_chatbot.attachments.service import (
    AttachmentContext,
    PreparedAttachment,
    build_attachment_context,
    format_attachment_text,
    prepare_attachments,
)
from bidding_qa_chatbot.attachments.store import (
    AttachmentError,
    AttachmentKind,
    AttachmentRecord,
    AttachmentStore,
    classify_upload,
    sniff_image_type,
)

__all__ = [
    "AttachmentContext",
    "AttachmentError",
    "AttachmentKind",
    "AttachmentRecord",
    "AttachmentStore",
    "PreparedAttachment",
    "build_attachment_context",
    "classify_upload",
    "format_attachment_text",
    "prepare_attachments",
    "sniff_image_type",
]
