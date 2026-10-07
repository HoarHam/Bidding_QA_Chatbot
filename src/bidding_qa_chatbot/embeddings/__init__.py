"""本地向量与精排模型（需求 6）。

**本包内任何模块都不得在模块顶层导入 `sentence_transformers` / `transformers` / `torch`。**

原因：`huggingface_hub` 在 import 时读取 `HF_ENDPOINT`，而本机无法直连 huggingface.co。
`config.py` 在导入时执行 `load_dotenv()` 把 `HF_ENDPOINT` 写进 `os.environ`；
如果某个模块在 `config` 之前被导入且顶层 import 了这些库，环境变量就还没设上，
模型下载会先卡住数分钟再报一个和配置无关的网络错误。

因此模型一律在工厂函数内部延迟导入，并用 `lru_cache` 保证只加载一次。

## 为什么加载时要先试 `local_files_only=True`

`huggingface_hub` 默认在每次加载模型时向远端发起一批 HEAD 请求，检查
`processor_config.json`、`video_preprocessor_config.json` 等**可选**文件是否有更新。
国内网络访问镜像站并不稳定，实测这些请求会反复超时重试（每个文件 5 次退避重试），
把服务启动拖慢到分钟级——而模型其实早已在本地缓存里了。

所以加载策略是「先本地、后远端」：先用 `local_files_only=True` 读缓存（无网络、瞬时完成），
只有缓存里没有时才回退到联网下载。首次运行能正常下载，后续启动不再被网络拖住。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def load_with_cache_fallback[T](model_name: str, loader: Callable[[bool], T]) -> T:
    """先尝试只用本地缓存加载，失败再联网下载。

    `loader` 接收一个 `local_files_only` 布尔值并返回模型实例。
    """
    try:
        model = loader(True)
        logger.info("从本地缓存加载 %s（已跳过联网校验）", model_name)
        return model
    except Exception as exc:  # noqa: BLE001
        logger.info("本地缓存未命中 %s（%s），转为联网加载", model_name, type(exc).__name__)
    model = loader(False)
    logger.info("联网加载完成 %s", model_name)
    return model


def is_cached(model_name: str) -> Any:
    """模型权重是否已在本地缓存中。用于启动前的快速判断。"""
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(model_name, "model.safetensors"), str)
    except Exception:  # noqa: BLE001
        return False
