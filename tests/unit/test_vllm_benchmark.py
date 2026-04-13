"""
Tests for the vLLM benchmark suite.

Covers:
  - Test prompt suite loading and filtering
  - Quality evaluation / scoring logic
  - Dry-run simulation
  - Report generation and formatting
  - Result data structure serialization
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on sys.path so the script can be imported
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Import the benchmark module — register it in sys.modules first so dataclasses
# can resolve the module __dict__ on Python 3.9.
import importlib.util

_mod_name = "benchmark_vllm_quantization"
_spec = importlib.util.spec_from_file_location(
    _mod_name,
    str(ROOT / "scripts" / "benchmark_vllm_quantization.py"),
)
bench = importlib.util.module_from_spec(_spec)
sys.modules[_mod_name] = bench
_spec.loader.exec_module(bench)


# =========================================================================
# Test Prompt Suite
# =========================================================================

class TestPromptSuite:
    """Tests for the CAD-domain test prompt suite."""

    def test_prompt_count_minimum(self):
        """Suite must have at least 20 prompts."""
        assert len(bench.CAD_TEST_PROMPTS) >= 20

    def test_all_prompts_have_required_fields(self):
        """Every prompt must have id, category, prompt text, and expected_keywords."""
        for p in bench.CAD_TEST_PROMPTS:
            assert p.id, f"Prompt missing id"
            assert p.category, f"Prompt {p.id} missing category"
            assert p.prompt, f"Prompt {p.id} has empty prompt text"
            assert len(p.expected_keywords) > 0, f"Prompt {p.id} has no expected keywords"

    def test_prompt_ids_unique(self):
        """All prompt IDs must be unique."""
        ids = [p.id for p in bench.CAD_TEST_PROMPTS]
        assert len(ids) == len(set(ids)), f"Duplicate prompt IDs found"

    def test_categories_cover_expected_domains(self):
        """Suite should cover all major CAD domain categories."""
        categories = set(p.category for p in bench.CAD_TEST_PROMPTS)
        required = {"part_classification", "material_analysis", "drawing_interpretation", "technical_qa"}
        assert required.issubset(categories), f"Missing categories: {required - categories}"

    def test_bilingual_coverage(self):
        """Suite should include both Chinese and English prompts."""
        languages = set(p.language for p in bench.CAD_TEST_PROMPTS)
        assert "zh" in languages
        assert "en" in languages

    def test_filter_by_category(self):
        """get_test_prompts should filter by category."""
        ma_prompts = bench.get_test_prompts(category="material_analysis")
        assert len(ma_prompts) >= 2
        assert all(p.category == "material_analysis" for p in ma_prompts)

    def test_filter_no_category_returns_all(self):
        """get_test_prompts with no filter returns all prompts."""
        all_prompts = bench.get_test_prompts()
        assert len(all_prompts) == len(bench.CAD_TEST_PROMPTS)

    def test_negative_keywords_exist_on_some_prompts(self):
        """At least one prompt should have negative keywords for hallucination detection."""
        has_negative = any(len(p.negative_keywords) > 0 for p in bench.CAD_TEST_PROMPTS)
        assert has_negative, "No prompts have negative_keywords for hallucination detection"


# =========================================================================
# Quality Evaluation
# =========================================================================

class TestQualityEvaluation:
    """Tests for the response quality evaluation logic."""

    def _make_prompt(self, keywords=None, negative=None):
        return bench.TestPrompt(
            id="test-01",
            category="test",
            prompt="test prompt",
            expected_keywords=keywords or ["steel", "strength"],
            negative_keywords=negative or [],
        )

    def test_perfect_response(self):
        """Response containing all keywords should score 1.0."""
        prompt = self._make_prompt(keywords=["steel", "strength"])
        response = "This steel part has high tensile strength."
        score = bench.evaluate_response_quality(prompt, response)
        assert score.keyword_score == 1.0
        assert score.keyword_hits == 2
        assert score.is_relevant is True

    def test_partial_response(self):
        """Response with some keywords should score proportionally."""
        prompt = self._make_prompt(keywords=["steel", "strength", "hardness"])
        response = "Steel is commonly used in manufacturing."
        score = bench.evaluate_response_quality(prompt, response)
        assert score.keyword_hits == 1
        assert abs(score.keyword_score - 1 / 3) < 0.01

    def test_empty_response(self):
        """Empty response should score 0 and not be relevant."""
        prompt = self._make_prompt()
        score = bench.evaluate_response_quality(prompt, "")
        assert score.keyword_score == 0
        assert score.is_relevant is False

    def test_hallucination_detection(self):
        """Response with negative keywords should flag hallucination."""
        prompt = self._make_prompt(
            keywords=["carbon_steel"],
            negative=["stainless_steel", "aluminum"],
        )
        response = "This is aluminum alloy, commonly used for stainless_steel parts."
        score = bench.evaluate_response_quality(prompt, response)
        assert score.has_hallucination is True
        assert score.negative_hits == 2

    def test_no_hallucination(self):
        """Clean response should not flag hallucination."""
        prompt = self._make_prompt(keywords=["bolt"], negative=["nut"])
        response = "This is a bolt with hexagonal head."
        score = bench.evaluate_response_quality(prompt, response)
        assert score.has_hallucination is False
        assert score.negative_hits == 0

    def test_case_insensitive_matching(self):
        """Keyword matching should be case-insensitive."""
        prompt = self._make_prompt(keywords=["CNC", "Steel"])
        response = "cnc machining of steel parts"
        score = bench.evaluate_response_quality(prompt, response)
        assert score.keyword_hits == 2

    def test_relevance_threshold(self):
        """Response needs >= 40% keywords and >= 20 chars to be relevant."""
        prompt = self._make_prompt(keywords=["steel", "bolt", "thread", "torque", "flange"])
        # 2/5 = 40%, meets threshold; response >= 20 chars
        response = "This steel bolt is used in heavy machinery assemblies."
        score = bench.evaluate_response_quality(prompt, response)
        assert score.is_relevant is True

        # 1/5 = 20%, below threshold
        response2 = "This steel component is used in general applications."
        score2 = bench.evaluate_response_quality(prompt, response2)
        assert score2.is_relevant is False


# =========================================================================
# Model Config Loading
# =========================================================================

class TestModelConfigLoading:
    """Tests for model configuration loading."""

    def test_load_defaults(self):
        """Should return 4 default model configs when YAML is missing."""
        configs = bench.load_model_configs()
        assert len(configs) >= 4
        names = [c.name for c in configs]
        assert "deepseek-coder-6.7b" in names
        assert "qwen2-7b" in names

    def test_config_has_required_fields(self):
        """Each model config must have all required fields."""
        configs = bench.load_model_configs()
        for c in configs:
            assert c.name
            assert c.hf_id
            assert c.vram_fp16
            assert c.vram_awq
            assert len(c.strengths) > 0
            assert c.recommended_quantization in ("fp16", "awq", "gptq", "int8")


# =========================================================================
# Dry-Run Simulation
# =========================================================================

class TestDryRun:
    """Tests for the dry-run simulation mode."""

    def test_dry_run_returns_result(self):
        """Dry-run should return a valid ModelBenchmarkResult."""
        result = bench.run_dry_run_benchmark(
            model_name="test-model",
            quantization="awq",
            concurrency_levels=[1, 10],
        )
        assert result.model_name == "test-model"
        assert result.quantization == "awq"
        assert len(result.concurrency_results) == 2
        assert len(result.quality_scores) > 0
        assert 0 <= result.avg_quality_score <= 1.0
        assert 0 <= result.relevance_rate <= 1.0

    def test_dry_run_with_stress(self):
        """Dry-run with stress test should set optimal_concurrency."""
        result = bench.run_dry_run_benchmark(
            model_name="test-model",
            quantization="fp16",
            stress=True,
        )
        assert result.optimal_concurrency is not None
        assert result.throughput_plateau_rps is not None
        assert result.throughput_plateau_rps > 0

    def test_dry_run_concurrency_results_valid(self):
        """Simulated concurrency results should have plausible values."""
        result = bench.run_dry_run_benchmark(concurrency_levels=[1, 10, 50])
        for cr in result.concurrency_results:
            assert cr.concurrency > 0
            assert cr.successful == cr.concurrency  # no failures in dry-run
            assert cr.failed == 0
            assert cr.p50_latency > 0
            assert cr.p95_latency >= cr.p50_latency
            assert cr.p99_latency >= cr.p95_latency
            assert cr.req_per_sec > 0


# =========================================================================
# Report Generation
# =========================================================================

class TestReportGeneration:
    """Tests for report serialization and formatting."""

    def _make_result(self):
        return bench.run_dry_run_benchmark(
            model_name="test-model",
            quantization="awq",
            concurrency_levels=[1, 10],
        )

    def test_result_to_dict_serializable(self):
        """result_to_dict output must be JSON-serializable."""
        result = self._make_result()
        d = bench.result_to_dict(result)
        json_str = json.dumps(d, ensure_ascii=False)
        assert len(json_str) > 0

    def test_result_to_dict_has_summary(self):
        """result_to_dict must include summary section."""
        result = self._make_result()
        d = bench.result_to_dict(result)
        assert "summary" in d
        assert "avg_quality_score" in d["summary"]
        assert "hallucination_count" in d["summary"]
        assert "relevance_rate" in d["summary"]

    def test_comparison_table_formatting(self):
        """format_comparison_table should produce readable output."""
        r1 = bench.run_dry_run_benchmark("model-a", "awq", [1, 10])
        r2 = bench.run_dry_run_benchmark("model-b", "fp16", [1, 10])
        table = bench.format_comparison_table([r1, r2])
        assert "model-a" in table
        assert "model-b" in table
        assert "P95" in table or "P50" in table
        assert "Performance Targets" in table

    def test_save_report_creates_json(self):
        """save_report should create a valid JSON file."""
        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = bench.save_report([result], output_dir=tmpdir)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert "models" in data
            assert len(data["models"]) == 1
            assert data["models"][0]["model_name"] == "test-model"

    def test_save_report_multiple_models(self):
        """save_report should handle multiple model results."""
        r1 = bench.run_dry_run_benchmark("model-a", "awq")
        r2 = bench.run_dry_run_benchmark("model-b", "fp16")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = bench.save_report([r1, r2], output_dir=tmpdir)
            with open(path) as f:
                data = json.load(f)
            assert len(data["models"]) == 2


# =========================================================================
# Data Structure Tests
# =========================================================================

class TestDataStructures:
    """Tests for benchmark data structures."""

    def test_single_request_result_fields(self):
        r = bench.SingleRequestResult(error=False, latency=0.05, tokens_out=50, tps=1000, response="hello")
        assert r.latency == 0.05
        assert r.response == "hello"

    def test_concurrency_result_fields(self):
        cr = bench.ConcurrencyResult(
            concurrency=10, total_requests=10, successful=9, failed=1,
            p50_latency=0.03, p95_latency=0.08, p99_latency=0.12,
            avg_tps=40, total_duration=1.0, req_per_sec=9.0,
        )
        assert cr.failed == 1
        assert cr.req_per_sec == 9.0

    def test_quality_score_fields(self):
        qs = bench.QualityScore(
            prompt_id="pc-01", keyword_hits=3, keyword_total=5,
            keyword_score=0.6, negative_hits=0, has_hallucination=False,
            response_length=200, is_relevant=True,
        )
        assert qs.keyword_score == 0.6
        assert qs.is_relevant is True


# =========================================================================
# Integration-style dry-run test
# =========================================================================

class TestDryRunIntegration:
    """End-to-end dry-run benchmark producing a full report."""

    def test_compare_all_dry_run(self):
        """Simulate --compare-all --dry-run and verify report."""
        configs = bench.load_model_configs()
        results = []
        for mc in configs:
            r = bench.run_dry_run_benchmark(
                model_name=mc.name,
                quantization=mc.recommended_quantization,
                concurrency_levels=[1, 10],
            )
            results.append(r)

        assert len(results) >= 4

        with tempfile.TemporaryDirectory() as tmpdir:
            path = bench.save_report(results, output_dir=tmpdir)
            with open(path) as f:
                data = json.load(f)
            assert data["prompt_count"] >= 20
            assert len(data["models"]) >= 4

    def test_quantization_sweep_dry_run(self):
        """Simulate --quantization-sweep --dry-run for one model."""
        results = []
        for quant in ["fp16", "awq", "gptq", "int8"]:
            r = bench.run_dry_run_benchmark(
                model_name="deepseek-coder-6.7b",
                quantization=quant,
                concurrency_levels=[1, 10],
            )
            results.append(r)

        assert len(results) == 4
        quants = [r.quantization for r in results]
        assert set(quants) == {"fp16", "awq", "gptq", "int8"}
