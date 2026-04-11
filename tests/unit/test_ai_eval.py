"""Unit tests for the AI quality evaluation suite."""

from __future__ import annotations

import asyncio

import pytest

from src.ml.evaluation.ai_eval import AIEvaluationSuite


@pytest.fixture()
def suite() -> AIEvaluationSuite:
    return AIEvaluationSuite()


# ------------------------------------------------------------------
# Test cases
# ------------------------------------------------------------------


class TestBuildTestCases:
    def test_build_test_cases_not_empty(self, suite: AIEvaluationSuite) -> None:
        """At least 10 golden test cases exist."""
        assert len(suite.test_cases) >= 10

    def test_each_case_has_required_keys(self, suite: AIEvaluationSuite) -> None:
        for tc in suite.test_cases:
            assert "question" in tc
            assert "expected_contains" in tc
            assert "category" in tc


# ------------------------------------------------------------------
# Cost evaluation
# ------------------------------------------------------------------


class TestCostEvaluation:
    def test_evaluate_cost_all_pass(self, suite: AIEvaluationSuite) -> None:
        result = asyncio.run(suite.evaluate_cost_estimation())
        assert result["category"] == "cost"
        assert result["cases"] > 0
        for detail in result["details"]:
            assert detail["pass"], f"Cost check failed: {detail['test']}"

    def test_cost_avg_score_is_one(self, suite: AIEvaluationSuite) -> None:
        result = asyncio.run(suite.evaluate_cost_estimation())
        assert result["avg_score"] == 1.0


# ------------------------------------------------------------------
# Hybrid intelligence evaluation
# ------------------------------------------------------------------


class TestIntelligenceEvaluation:
    def test_evaluate_intelligence_all_pass(self, suite: AIEvaluationSuite) -> None:
        result = asyncio.run(suite.evaluate_hybrid_intelligence())
        assert result["category"] == "intelligence"
        assert result["cases"] > 0
        for detail in result["details"]:
            assert detail["pass"], f"Intelligence check failed: {detail['test']}"

    def test_intelligence_avg_score_is_one(self, suite: AIEvaluationSuite) -> None:
        result = asyncio.run(suite.evaluate_hybrid_intelligence())
        assert result["avg_score"] == 1.0


# ------------------------------------------------------------------
# Knowledge graph evaluation
# ------------------------------------------------------------------


class TestKnowledgeGraphEvaluation:
    def test_evaluate_knowledge_graph_scores(self, suite: AIEvaluationSuite) -> None:
        result = asyncio.run(suite.evaluate_knowledge_graph())
        assert result["category"] == "knowledge_graph"
        assert result["cases"] > 0
        assert result["avg_score"] > 0, "Expected knowledge graph avg_score > 0"

    def test_knowledge_graph_details_have_scores(
        self, suite: AIEvaluationSuite
    ) -> None:
        result = asyncio.run(suite.evaluate_knowledge_graph())
        for detail in result["details"]:
            assert "score" in detail
            assert "confidence" in detail


# ------------------------------------------------------------------
# Full evaluation
# ------------------------------------------------------------------


class TestFullEvaluation:
    def test_full_evaluation_has_verdict(self, suite: AIEvaluationSuite) -> None:
        result = asyncio.run(suite.run_full_evaluation())
        assert result["verdict"] in ("PASS", "NEEDS_IMPROVEMENT")
        assert "overall_score" in result
        assert "total_cases" in result
        assert result["total_cases"] > 0

    def test_full_evaluation_categories_count(
        self, suite: AIEvaluationSuite
    ) -> None:
        result = asyncio.run(suite.run_full_evaluation())
        assert len(result["categories"]) == 3


# ------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------


class TestReportGeneration:
    def test_report_is_markdown(self, suite: AIEvaluationSuite) -> None:
        result = asyncio.run(suite.run_full_evaluation())
        report = suite.generate_report(result)
        assert report.startswith("# ")
        assert "## " in report
        assert "Overall Score" in report

    def test_report_contains_verdict(self, suite: AIEvaluationSuite) -> None:
        result = asyncio.run(suite.run_full_evaluation())
        report = suite.generate_report(result)
        assert result["verdict"] in report
