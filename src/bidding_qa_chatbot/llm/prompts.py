"""提示词与上下文组装。

系统提示词的硬性要求：**只能依据给定上下文作答**，并在上下文不足时明确说
「根据知识库无法确定」，而不是用模型自身记忆编造。招投标领域涉及法规与质疑投诉
时效，编造是比「答不出来」严重得多的失败。

上下文按**整块**丢弃而非截断：被腰斩的法规条文比没有更危险。
"""

from __future__ import annotations

SYSTEM_PROMPT = """你是一名招投标与政府采购领域的专业问答助手。

回答要求：
1. **只依据下面提供的「参考资料」作答，不要使用你自己的记忆补充事实。**
2. 如果参考资料不足以回答问题，直接回复「根据知识库无法确定」，并说明缺少哪方面的信息。
   不要猜测，不要编造法规条款、时限或数字。
3. 引用具体规则时，在句末用方括号标注来源，格式为 `[来源：source_label]`，
   source_label 必须与参考资料中给出的完全一致。
4. 使用简体中文，条理清晰。适合用表格呈现的内容（如对比、时限、金额档位）请使用 Markdown 表格。
5. 不要复述这些要求，不要使用「根据您提供的资料」之类的套话，直接回答问题本身。
"""

NO_CONTEXT_REPLY = "根据知识库无法确定：本次检索没有找到与问题相关的资料。"


def build_context(sources, max_chars: int) -> str:
    """把检索结果拼成参考资料块。

    超出预算时**整块丢弃**（从最后一名开始），不做字符级截断——
    半截的法规条文可能产生误导性结论。
    """
    blocks: list[str] = []
    used = 0

    for i, src in enumerate(sources, start=1):
        block = f"【资料{i}】source_label: {src.source_label}\n问：{src.question}\n答：{src.answer}"
        if src.images:
            described = [img for img in src.images if img.get("description")]
            if described:
                block += "\n图片内容：\n" + "\n\n".join(
                    str(img["description"]) for img in described
                )

        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)

    return "\n\n".join(blocks)


def build_messages(query: str, context: str, history=None) -> list:
    """组装最终发给 LLM 的消息列表。"""
    from bidding_qa_chatbot.llm.base import ChatMessage

    messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]

    for turn in history or []:
        messages.append(ChatMessage(role=turn.role, content=turn.content))

    if context.strip():
        user_content = f"参考资料：\n\n{context}\n\n用户问题：{query}"
    else:
        user_content = f"参考资料：（无）\n\n用户问题：{query}"

    messages.append(ChatMessage(role="user", content=user_content))
    return messages
