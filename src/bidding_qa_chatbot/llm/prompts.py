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


_AGENT_SYSTEM_HEAD = """你是一名招投标与政府采购领域的专业问答助手，可以使用工具检索内部知识库。

工作方式：
1. 先判断问题是否属于招投标/政府采购领域，以及是否需要查阅法规或事实性依据。
2. **涉及法规条款、各类期限（质疑期、投诉期、公告期、保证金退还期等）、
   金额门槛、比例、资格条件、评审办法时，必须先调用 search_bidding_knowledge 检索，
   不要凭自身记忆作答**——这类信息时效性强，记忆容易过时或出错。
3. 检索后**只依据检索到的资料**作答。资料不足时如实说「根据知识库无法确定」，
   并说明缺哪方面的信息；不要编造法规条款、时限或数字。
4. 引用具体规则时，在句末用 `[来源：source_label]` 标注出处，source_label
   必须与资料中给出的完全一致。
5. 与招投标无关的闲聊可以直接回答，不需要调用工具。
"""

#: 图谱工具启用时（ENABLE_NEO4J=true 且工具已注册）追加的说明。
#: 工具未注册时**不能**提它，否则模型会试图调用一个不存在的工具。
AGENT_GRAPH_PROMPT = """
另外，涉及**具体交易事实与统计**时，必须先调用 query_bidding_graph 查询招投标知识图谱，
不要凭印象估算或编造，例如：
- 某标的物（空调、打印/复印纸、网上超市采购等）的交易频次、交易笔数、中标金额；
- 某供应商/采购人/代理机构/地区与哪些标的物有关联；
- 某标的物的中标供应商、采购人、代理机构、所在地。
一个问题同时涉及法规和具体交易数据时，知识库与图谱两个工具都要调用，再综合作答；
工具返回空结果时如实说「没有查到」，不要用记忆补充。
图谱工具优先用结构化参数（subject_matter / supplier / purchaser / agency / location
搭配 aspect）；若它返回「匹配到多个候选」，先向用户确认是哪一个，不要自行猜测作答。
"""

#: 用户本轮开启联网搜索、且联网工具已注册时追加的说明。
AGENT_WEB_SEARCH_PROMPT = """
另外，用户已开启**联网搜索**。涉及时效性强的外部公开信息（最新政策/新闻/公告、
本地知识库与知识图谱覆盖不到的内容）时，可以调用联网搜索工具：
- `search_web_keyword`：关键词搜索，适合具体名词、数字与公告的核实（查询用关键词组合）；
- `search_web_semantic`：语义搜索，适合概念性、开放式、多角度的问题（查询用完整问句）。
优先使用本地知识库与知识图谱；联网结果为空或不可用时如实说明，不要编造。
引用联网结果时在句末标注 `[来源：URL]`，URL 必须来自工具返回的结果。
"""

_AGENT_SYSTEM_TAIL = """
输出要求：简体中文，条理清晰；适合用表格呈现的内容（对比、时限、金额档位）请用 Markdown 表格。
不要复述这些要求，不要写「根据您提供的资料」之类的套话，直接回答。

**调用工具前不要输出任何文字。** 不要写「让我查一下」「I'll look this up」之类的过渡语——
那会混进最终回答里。直接发起工具调用，等拿到资料后再一次性作答。
"""

#: 无图谱时的完整系统提示词（图谱说明由 build_agent_messages 按需插在中间）
AGENT_SYSTEM_PROMPT = _AGENT_SYSTEM_HEAD + _AGENT_SYSTEM_TAIL


def build_agent_messages(
    query: str,
    history: list | None = None,
    *,
    graph_enabled: bool = False,
    web_search_enabled: bool = False,
) -> list:
    """Agent 路径的消息：不预置资料，由模型自行决定是否检索。

    `graph_enabled` / `web_search_enabled` 由「该工具在本请求中是否真正可用」决定
    （routes_chat 传入）——提示词必须与本次提供的工具集合一致，
    不能提一个模型调不到的工具。
    """
    from bidding_qa_chatbot.llm.base import ChatMessage

    system = (
        _AGENT_SYSTEM_HEAD
        + (AGENT_GRAPH_PROMPT if graph_enabled else "")
        + (AGENT_WEB_SEARCH_PROMPT if web_search_enabled else "")
        + _AGENT_SYSTEM_TAIL
    )
    messages = [ChatMessage(role="system", content=system)]
    for turn in history or []:
        messages.append(ChatMessage(role=turn.role, content=turn.content))
    messages.append(ChatMessage(role="user", content=query))
    return messages


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
