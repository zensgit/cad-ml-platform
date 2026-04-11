"""
Function Calling Engine for the CAD Copilot.

Orchestrates LLM tool-use loops: the engine sends messages to the LLM,
intercepts tool_use blocks, executes the corresponding tool, feeds the
result back to the LLM, and repeats until the model produces a final
text response.  Supports Claude (Anthropic), OpenAI, and an offline
fallback mode.
"""

import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from .tools import TOOL_REGISTRY, BaseTool

logger = logging.getLogger(__name__)

# Maximum tool-use iterations to prevent infinite loops
_MAX_TOOL_ROUNDS = 10


def _get_system_prompt() -> str:
    """Return the CAD Copilot system prompt with chain-of-thought reasoning."""
    tool_lines = "\n".join(
        f"  - **{t.name}**: {t.description}" for t in TOOL_REGISTRY.values()
    )
    return (
        "你是 CAD-ML 智能助手（Copilot），帮助工程师分析 CAD 图纸并提供制造决策支持。\n\n"
        "## 可用工具\n"
        f"{tool_lines}\n\n"
        "## 思维链推理框架\n"
        "面对每个问题，你必须按以下步骤思考：\n\n"
        "**第一步：理解意图**\n"
        "- 用户想知道什么？是分类、成本、工艺、还是质量问题？\n"
        "- 是否涉及多个分析维度？（如既问成本又问工艺）\n"
        "- 用户的专业水平如何？（根据用词判断：新手用通俗语言，专家用技术术语）\n\n"
        "**第二步：制定分析计划**\n"
        "- 需要调用哪些工具？按什么顺序？\n"
        "- 是否需要先分类，再基于分类结果查询工艺/成本？\n"
        "- 是否需要交叉验证？（如分类结果与工艺推荐是否一致）\n\n"
        "**第三步：执行工具调用**\n"
        "- 依次调用工具，获取数据\n"
        "- 如果某个工具结果异常（置信度低、数据缺失），主动说明并建议补充信息\n\n"
        "**第四步：综合推理**\n"
        "- 整合多个工具的结果，寻找一致性或矛盾\n"
        "- 如果工具结果互相矛盾，分析可能原因并给出判断\n"
        "- 考虑上下文：用户之前的问题可能提供额外线索\n\n"
        "**第五步：生成回答**\n"
        "- 先给结论，再给支撑数据\n"
        "- 对不确定的部分明确标注置信度\n"
        "- 如果信息不足，主动询问（材料？批量？精度要求？）\n\n"
        "## 跨域推理能力\n"
        "你可以组合不同工具的结果进行深层分析：\n"
        "- **分类+工艺+成本**：识别零件类型 → 推荐工艺 → 估算成本 → 给出优化建议\n"
        "- **特征+相似度**：提取特征 → 搜索相似件 → 对比差异 → 推荐复用\n"
        "- **质量+知识库**：评估质量问题 → 查询标准规范 → 给出改进方案\n"
        "- **成本对比**：不同材料/工艺的成本对比 → 推荐性价比最优方案\n\n"
        "## 不确定性表达\n"
        "- 高置信度 (>0.8)：直接陈述结论\n"
        "- 中置信度 (0.5-0.8)：\"根据分析，这很可能是...，但建议确认...\"\n"
        "- 低置信度 (<0.5)：\"当前数据不足以确定，建议补充以下信息：...\"\n\n"
        "## 使用规则\n"
        "1. 优先使用工具获取数据，不要猜测技术参数。\n"
        "2. 如果用户提供了文件ID，优先使用该 ID 调用相关分析工具。\n"
        "3. 综合多个工具的结果，给出完整、准确的回答。\n"
        "4. 回答时使用中文，保持简洁专业。技术术语保留英文原文。\n"
        "5. 成本数据单位为 CNY（人民币），精确到小数点后两位。\n"
        "6. 如果工具返回结果中含有 note 字段，不需要向用户暴露内部错误细节。\n"
    )


class FunctionCallingEngine:
    """LLM function-calling engine with multi-round tool execution.

    Parameters
    ----------
    llm_provider : str
        One of ``"claude"``, ``"openai"``, or ``"offline"``.
    model : str or None
        Model name override.  Defaults are provider-specific.
    """

    def __init__(
        self,
        llm_provider: str = "claude",
        model: Optional[str] = None,
    ):
        self._provider_name = llm_provider
        self._tools: Dict[str, BaseTool] = dict(TOOL_REGISTRY)
        self._system_prompt = _get_system_prompt()

        # Resolve LLM client --------------------------------------------------
        self._anthropic_client: Any = None
        self._openai_client: Any = None

        if llm_provider == "claude":
            try:
                import anthropic  # noqa: F811

                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self._anthropic_client = anthropic.Anthropic(api_key=api_key)
                    self._model = model or "claude-sonnet-4-20250514"
                else:
                    logger.warning("ANTHROPIC_API_KEY not set -- falling back to offline mode")
                    self._provider_name = "offline"
            except ImportError:
                logger.warning("anthropic package not installed -- falling back to offline mode")
                self._provider_name = "offline"

        elif llm_provider == "openai":
            try:
                import openai  # noqa: F811

                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self._openai_client = openai.OpenAI(api_key=api_key)
                    self._model = model or "gpt-4-turbo-preview"
                else:
                    logger.warning("OPENAI_API_KEY not set -- falling back to offline mode")
                    self._provider_name = "offline"
            except ImportError:
                logger.warning("openai package not installed -- falling back to offline mode")
                self._provider_name = "offline"

        if self._provider_name == "offline":
            self._model = model or "offline"

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str:
        """Return the system prompt (useful for testing)."""
        return self._system_prompt

    def _build_tool_definitions_anthropic(self) -> List[Dict[str, Any]]:
        """Build tool schemas in Anthropic format."""
        return [tool.to_schema() for tool in self._tools.values()]

    def _build_tool_definitions_openai(self) -> List[Dict[str, Any]]:
        """Build tool schemas in OpenAI function-calling format."""
        defs = []
        for tool in self._tools.values():
            defs.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            })
        return defs

    # ------------------------------------------------------------------
    # Chat entry point
    # ------------------------------------------------------------------

    async def chat(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        file_ids: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Run a full chat turn, yielding text chunks as they arrive.

        If the LLM decides to call tools, the engine executes them, feeds
        results back, and continues until the model returns a final text
        response (or ``_MAX_TOOL_ROUNDS`` is reached).
        """
        if self._provider_name == "claude" and self._anthropic_client:
            async for chunk in self._chat_claude(user_message, conversation_history, file_ids):
                yield chunk
        elif self._provider_name == "openai" and self._openai_client:
            async for chunk in self._chat_openai(user_message, conversation_history, file_ids):
                yield chunk
        else:
            async for chunk in self._chat_offline(user_message, conversation_history, file_ids):
                yield chunk

    # ------------------------------------------------------------------
    # Claude (Anthropic) backend
    # ------------------------------------------------------------------

    async def _chat_claude(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]],
        file_ids: Optional[List[str]],
    ) -> AsyncGenerator[str, None]:
        messages = self._build_messages(user_message, conversation_history, file_ids)
        tools = self._build_tool_definitions_anthropic()

        for _ in range(_MAX_TOOL_ROUNDS):
            response = self._anthropic_client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=self._system_prompt,
                tools=tools,
                messages=messages,
            )

            # Collect text blocks and tool_use blocks
            text_parts: List[str] = []
            tool_uses: List[Dict[str, Any]] = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            if not tool_uses:
                # Final answer -- yield all text
                for part in text_parts:
                    yield part
                return

            # Execute each tool use and build tool_result messages
            # First, add the assistant message containing tool uses
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tu in tool_uses:
                result = await self._execute_tool(tu["name"], tu["input"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })

            messages.append({"role": "user", "content": tool_results})

            # Yield intermediate text if any
            for part in text_parts:
                yield part

        # Safety: exceeded max rounds
        yield "\n[已达到最大工具调用轮次，返回当前结果]"

    # ------------------------------------------------------------------
    # OpenAI backend
    # ------------------------------------------------------------------

    async def _chat_openai(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]],
        file_ids: Optional[List[str]],
    ) -> AsyncGenerator[str, None]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
        ]
        if conversation_history:
            messages.extend(conversation_history)

        content = user_message
        if file_ids:
            content = f"[关联文件: {', '.join(file_ids)}]\n{user_message}"
        messages.append({"role": "user", "content": content})

        tools = self._build_tool_definitions_openai()

        for _ in range(_MAX_TOOL_ROUNDS):
            response = self._openai_client.chat.completions.create(
                model=self._model,
                max_tokens=4096,
                messages=messages,
                tools=tools,
            )
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                messages.append(choice.message)
                for tc in choice.message.tool_calls:
                    func = tc.function
                    args = json.loads(func.arguments) if func.arguments else {}
                    result = await self._execute_tool(func.name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
            else:
                text = choice.message.content or ""
                yield text
                return

        yield "\n[已达到最大工具调用轮次，返回当前结果]"

    # ------------------------------------------------------------------
    # Offline / template backend
    # ------------------------------------------------------------------

    async def _chat_offline(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]],
        file_ids: Optional[List[str]],
    ) -> AsyncGenerator[str, None]:
        """Generate a template response without calling any LLM.

        If file IDs are provided, run all applicable tools and summarise.
        """
        parts: List[str] = []

        if file_ids:
            parts.append(f"收到您的请求，正在分析文件: {', '.join(file_ids)}\n")
            for fid in file_ids:
                # Run a subset of tools
                classify_result = await self._execute_tool("classify_part", {"file_id": fid})
                feature_result = await self._execute_tool("extract_features", {"file_id": fid})
                cost_result = await self._execute_tool("estimate_cost", {"file_id": fid})

                parts.append(f"## 文件 {fid} 分析结果\n")
                parts.append(f"- 分类: {classify_result.get('label', 'N/A')} "
                             f"(置信度 {classify_result.get('confidence', 0):.0%})")
                parts.append(f"- 特征维度: {feature_result.get('dimension', 'N/A')} "
                             f"(版本 {feature_result.get('version', 'N/A')})")
                parts.append(f"- 预估成本: {cost_result.get('total', 'N/A')} "
                             f"{cost_result.get('currency', 'CNY')}\n")
        else:
            # Knowledge query
            kb_result = await self._execute_tool("query_knowledge", {"query": user_message})
            if kb_result.get("results"):
                parts.append("根据知识库查询结果：\n")
                for item in kb_result["results"][:3]:
                    parts.append(f"- {item.get('summary', '')}")
            else:
                parts.append("抱歉，当前为离线模式，无法调用LLM生成回答。")
                parts.append(f"您的问题: {user_message}")
                parts.append("请配置 ANTHROPIC_API_KEY 或 OPENAI_API_KEY 以启用完整功能。")

        yield "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]],
        file_ids: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """Build the message list for the Anthropic API."""
        messages: List[Dict[str, Any]] = []
        if conversation_history:
            messages.extend(conversation_history)

        content = user_message
        if file_ids:
            content = f"[关联文件: {', '.join(file_ids)}]\n{user_message}"
        messages.append({"role": "user", "content": content})
        return messages

    async def _execute_tool(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Look up and execute a tool by name."""
        tool = self._tools.get(name)
        if tool is None:
            logger.error("Unknown tool requested: %s", name)
            return {"error": f"未知工具: {name}"}

        logger.info("Executing tool %s with params %s", name, params)
        try:
            return await tool.execute(params)
        except Exception as exc:
            logger.exception("Tool %s raised an exception", name)
            return {"error": f"工具执行失败: {exc}"}
