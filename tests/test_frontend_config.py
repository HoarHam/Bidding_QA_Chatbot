"""前端配置守卫。

这些断言针对的是「静态就能查、但一旦破坏后果很严重」的配置项——
它们坏掉时页面**看起来完全正常**（服务端渲染一切正常），
只有真正去点按钮才会发现整个前端是死的，非常难排查。

放在 Python 测试里是因为 CI 只跑 pytest，不为几个静态检查再引入一套 JS 测试运行时。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
NEXT_CONFIG = FRONTEND / "next.config.ts"
COMPOSER = FRONTEND / "components" / "Composer.tsx"

pytestmark = pytest.mark.skipif(not FRONTEND.exists(), reason="前端尚未创建")


def test_allowed_dev_origins_covers_loopback_aliases():
    """必须放行 127.0.0.1。

    Next 16 默认拦截陌生来源对开发资源（/_next/hmr 等）的请求。
    `localhost` 与 `127.0.0.1` 在它看来是**不同来源**——用 127.0.0.1 打开页面时，
    客户端 JS 被拦下，于是按钮、Enter、切换器**全部失效**，
    而服务端渲染的 HTML 完全正常，页面「能打开」但是死的。

    这个坑实测踩过：curl 不带 Origin 头测不出问题，只有真浏览器会暴露。
    """
    assert NEXT_CONFIG.exists(), "next.config.ts 不存在"
    source = NEXT_CONFIG.read_text(encoding="utf-8")

    match = re.search(r"allowedDevOrigins\s*:\s*\[([^\]]*)\]", source)
    assert match, "next.config.ts 缺少 allowedDevOrigins —— 用 127.0.0.1 访问时前端会完全失去交互"

    origins = {o.strip().strip("\"'") for o in match.group(1).split(",") if o.strip()}
    assert "127.0.0.1" in origins, f"allowedDevOrigins 未包含 127.0.0.1，实际为 {origins}"


def test_api_rewrite_points_at_backend():
    """开发代理：前端所有请求走 /api 相对路径，靠这条 rewrite 转发到后端。"""
    source = NEXT_CONFIG.read_text(encoding="utf-8")
    assert "/api/:path*" in source, "缺少 /api 代理 rewrite"


def test_example_questions_are_real_text_not_placeholder():
    """示例问题必须以真实文本渲染。

    `placeholder` 按浏览器规范**不可选中**，把示例放进去用户就没法复制。
    断言示例以字符串字面量出现在组件里，并且没有出现在 placeholder 中。
    """
    source = COMPOSER.read_text(encoding="utf-8")
    assert "EXAMPLES" in source, "Composer 应把示例问题单独定义为常量"

    placeholder = re.search(r'placeholder="([^"]*)"', source)
    assert placeholder, "Composer 缺少 placeholder"
    text = placeholder.group(1)
    assert "异议" not in text and "保证金" not in text, (
        "示例问题不应写进 placeholder（placeholder 不可选中，用户无法复制）"
    )


def test_enter_guard_uses_native_is_composing():
    """Enter 的组字判断必须用原生 `isComposing`。

    曾经用 onCompositionStart/End 自己维护一个 ref，一旦 compositionend
    没有触发（中文输入法下并不罕见），该 ref 会永远停在 true，
    Enter 就再也不能发送、只能换行。
    """
    source = COMPOSER.read_text(encoding="utf-8")
    assert "isComposing" in source, "Enter 处理应使用原生 isComposing 判断组字状态"
    assert "composingRef" not in source, (
        "不要用自维护的 ref 判断组字——它可能卡在 true，导致 Enter 永久失效"
    )


def test_setturns_does_not_spread_history():
    """`setTurns` 不能把 history 展开进去。

    `history` 取自当前 `turns`，是**发给后端的对话上下文**，不是要新增的 UI 项。
    曾经写成 `setTurns(prev => [...prev, ...history, 新用户消息, 新助手消息])`，
    结果每发一条消息整个会话就被复制一遍，React 立刻报
    "Encountered two children with the same key"。

    这个 bug 由真实浏览器测试发现——静态检查和 curl 都看不出来。
    """
    shell = FRONTEND / "components" / "ChatShell.tsx"
    source = shell.read_text(encoding="utf-8")

    for block in re.findall(r"setTurns\(\(prev\)\s*=>\s*\[(.*?)\]\)", source, re.S):
        assert "...history" not in block, (
            "setTurns 里出现了 ...history：history 是发给后端的上下文，"
            "展开进 UI 列表会导致会话重复、React key 冲突"
        )


def test_only_one_send_button_label():
    """页面上只应有一个「发送」按钮。

    之前有一个纯提示文字「⏎ 发送」，外观像按钮却点不动，与真按钮并排极易混淆。
    这里限定「发送」只作为发送按钮的文案出现（提示语改写为不含该词的措辞）。
    """
    source = COMPOSER.read_text(encoding="utf-8")
    literal_count = source.count(">发送<") + len(re.findall(r"^\s*发送\s*$", source, re.M))
    assert literal_count <= 1, (
        f"Composer 里出现了 {literal_count} 处独立的「发送」文案，应只有发送按钮一处"
    )
