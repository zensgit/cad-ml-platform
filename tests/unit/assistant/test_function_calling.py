"""
Unit tests for the Function Calling engine, tool registry, and report generator.
"""

import asyncio
import pytest

from src.core.assistant.tools import (
    TOOL_REGISTRY,
    BaseTool,
    ClassifyTool,
    SimilarityTool,
    CostTool,
    FeatureTool,
    ProcessTool,
    QualityTool,
    KnowledgeTool,
)
from src.core.assistant.function_calling import FunctionCallingEngine
from src.core.assistant.report_generator import AnalysisReportGenerator


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

EXPECTED_TOOL_NAMES = {
    "classify_part",
    "search_similar",
    "estimate_cost",
    "extract_features",
    "recommend_process",
    "assess_quality",
    "query_knowledge",
}


class TestToolRegistry:
    """Tests for the TOOL_REGISTRY completeness and schema validity."""

    def test_tool_registry_complete(self):
        """All 7 tools are registered."""
        assert set(TOOL_REGISTRY.keys()) == EXPECTED_TOOL_NAMES

    def test_tool_registry_count(self):
        """Registry contains exactly 7 tools."""
        assert len(TOOL_REGISTRY) == 7

    def test_all_tools_are_base_tool_instances(self):
        """Every registered tool inherits from BaseTool."""
        for tool in TOOL_REGISTRY.values():
            assert isinstance(tool, BaseTool)

    def test_tool_schemas_valid(self):
        """Each tool has name, description, and a well-formed input_schema."""
        for name, tool in TOOL_REGISTRY.items():
            assert isinstance(tool.name, str) and tool.name == name
            assert isinstance(tool.description, str) and len(tool.description) > 0
            schema = tool.input_schema
            assert isinstance(schema, dict)
            assert schema.get("type") == "object"
            assert "properties" in schema
            assert "required" in schema
            # required fields must exist in properties
            for req in schema["required"]:
                assert req in schema["properties"], f"{name}: required field {req!r} missing from properties"

    def test_tool_to_schema(self):
        """to_schema() returns a valid provider-agnostic definition."""
        for tool in TOOL_REGISTRY.values():
            s = tool.to_schema()
            assert s["name"] == tool.name
            assert s["description"] == tool.description
            assert s["input_schema"] is tool.input_schema


# ---------------------------------------------------------------------------
# Individual tool execution (offline / fallback paths)
# ---------------------------------------------------------------------------

class TestToolExecution:
    """Execute each tool and verify fallback output structure."""

    @pytest.mark.asyncio
    async def test_classify_tool_fallback(self):
        result = await ClassifyTool().execute({"file_id": "test-001"})
        assert "label" in result
        assert "confidence" in result

    @pytest.mark.asyncio
    async def test_similarity_tool_fallback(self):
        result = await SimilarityTool().execute({"file_id": "test-001"})
        assert "results" in result
        assert "count" in result

    @pytest.mark.asyncio
    async def test_cost_tool_fallback(self):
        result = await CostTool().execute({"file_id": "test-001"})
        assert "total" in result
        assert "currency" in result
        assert result["currency"] == "CNY"
        assert isinstance(result["total"], (int, float))
        assert result["total"] > 0

    @pytest.mark.asyncio
    async def test_feature_tool_fallback(self):
        result = await FeatureTool().execute({"file_id": "test-001"})
        assert "dimension" in result
        assert "version" in result
        assert result["version"] == "v3"

    @pytest.mark.asyncio
    async def test_feature_tool_v4(self):
        result = await FeatureTool().execute({"file_id": "test-001", "version": "v4"})
        assert result["version"] == "v4"

    @pytest.mark.asyncio
    async def test_process_tool_fallback(self):
        result = await ProcessTool().execute({"file_id": "test-001"})
        assert "primary_process" in result
        assert "alternatives" in result
        assert isinstance(result["alternatives"], list)

    @pytest.mark.asyncio
    async def test_quality_tool_fallback(self):
        result = await QualityTool().execute({"file_id": "test-001"})
        assert "overall_score" in result
        assert "issues" in result
        assert "suggestions" in result

    @pytest.mark.asyncio
    async def test_knowledge_tool_fallback(self):
        result = await KnowledgeTool().execute({"query": "304不锈钢强度"})
        assert "results" in result
        assert "count" in result


# ---------------------------------------------------------------------------
# FunctionCallingEngine
# ---------------------------------------------------------------------------

class TestFunctionCallingEngine:
    """Tests for the FunctionCallingEngine class."""

    def test_offline_mode_init(self):
        """Engine falls back to offline mode gracefully."""
        engine = FunctionCallingEngine(llm_provider="offline")
        assert engine._provider_name == "offline"

    def test_system_prompt_contains_tools(self):
        """System prompt mentions every registered tool name."""
        engine = FunctionCallingEngine(llm_provider="offline")
        prompt = engine.get_system_prompt()
        for tool_name in EXPECTED_TOOL_NAMES:
            assert tool_name in prompt, f"Tool {tool_name!r} not found in system prompt"

    def test_tool_definitions_anthropic(self):
        """Anthropic-format definitions are well-formed."""
        engine = FunctionCallingEngine(llm_provider="offline")
        defs = engine._build_tool_definitions_anthropic()
        assert len(defs) == 7
        for d in defs:
            assert "name" in d
            assert "description" in d
            assert "input_schema" in d

    def test_tool_definitions_openai(self):
        """OpenAI-format definitions are well-formed."""
        engine = FunctionCallingEngine(llm_provider="offline")
        defs = engine._build_tool_definitions_openai()
        assert len(defs) == 7
        for d in defs:
            assert d["type"] == "function"
            assert "function" in d
            fn = d["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn

    @pytest.mark.asyncio
    async def test_offline_mode_works(self):
        """Offline chat returns a non-empty response."""
        engine = FunctionCallingEngine(llm_provider="offline")
        chunks = []
        async for chunk in engine.chat("304不锈钢的强度是多少?"):
            chunks.append(chunk)
        response = "".join(chunks)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_offline_mode_with_file_ids(self):
        """Offline chat with file_ids runs analysis tools."""
        engine = FunctionCallingEngine(llm_provider="offline")
        chunks = []
        async for chunk in engine.chat("分析这个零件", file_ids=["test-file-001"]):
            chunks.append(chunk)
        response = "".join(chunks)
        assert "test-file-001" in response

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Requesting an unknown tool returns an error dict."""
        engine = FunctionCallingEngine(llm_provider="offline")
        result = await engine._execute_tool("nonexistent_tool", {})
        assert "error" in result


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

class TestReportGenerator:
    """Tests for the AnalysisReportGenerator."""

    @pytest.mark.asyncio
    async def test_report_generator_format(self):
        """Report contains all expected sections."""
        gen = AnalysisReportGenerator()
        report = await gen.generate_full_report("test-file-001")

        expected_sections = [
            "概要",
            "分类结果",
            "几何特征",
            "推荐工艺",
            "成本估算",
            "质量评估",
            "改进建议",
        ]
        for section in expected_sections:
            assert section in report, f"Section {section!r} not found in report"

    @pytest.mark.asyncio
    async def test_report_contains_file_id(self):
        """Report references the input file ID."""
        gen = AnalysisReportGenerator()
        report = await gen.generate_full_report("drawing-xyz-42")
        assert "drawing-xyz-42" in report

    @pytest.mark.asyncio
    async def test_report_is_markdown(self):
        """Report is valid Markdown (contains headings and tables)."""
        gen = AnalysisReportGenerator()
        report = await gen.generate_full_report("test-001")
        assert report.startswith("# ")
        assert "|" in report  # tables
        assert "---" in report  # horizontal rules
