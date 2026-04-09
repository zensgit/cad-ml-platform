# Benchmark Knowledge Domain Readiness Validation

## Goal

Move benchmark knowledge output from a single foundation status into explicit
domain-level evidence for:

- `tolerance`
- `standards`
- `gdt`

The intent is to let companion, bundle, release decision, and runbook surfaces
show which engineering knowledge domain is weak, not only that knowledge is
"partial".

## Delivered

### 1. Domain-level readiness in the reusable benchmark helper

Updated [src/core/benchmark/knowledge_readiness.py](../src/core/benchmark/knowledge_readiness.py)
to emit:

- `knowledge_readiness.priority_domains`
- `knowledge_readiness.domains`
- `knowledge_readiness.domain_focus_areas`

Domain grouping now rolls up component-level readiness into benchmark-facing
domains:

- `tolerance` -> tolerance grades / fits
- `standards` -> standards + design-standards tables
- `gdt` -> GD&T symbols / datums / recommendations

### 2. Release surfaces now expose domain detail

Updated these exporters:

- [scripts/export_benchmark_companion_summary.py](../scripts/export_benchmark_companion_summary.py)
- [scripts/export_benchmark_artifact_bundle.py](../scripts/export_benchmark_artifact_bundle.py)
- [scripts/export_benchmark_release_decision.py](../scripts/export_benchmark_release_decision.py)
- [scripts/export_benchmark_release_runbook.py](../scripts/export_benchmark_release_runbook.py)

New payload fields now flow through those surfaces:

- `knowledge_domains`
- `knowledge_domain_focus_areas`
- `knowledge_priority_domains`

Rendered Markdown now includes dedicated sections for:

- `Knowledge Domains`
- `Knowledge Domain Focus Areas`

### 3. Contract coverage

Updated unit coverage to verify:

- ready vs partial domain rollups
- domain-level focus area propagation
- companion / bundle / release / runbook payload presence
- Markdown rendering of domain sections

## Validation

Commands:

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_readiness.py \
  scripts/export_benchmark_knowledge_readiness.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_readiness.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

flake8 \
  src/core/benchmark/knowledge_readiness.py \
  scripts/export_benchmark_knowledge_readiness.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_readiness.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_knowledge_readiness.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

Result:

- `py_compile` passed
- `flake8` passed
- `19 passed`

## Outcome

The benchmark stack can now answer:

- which domain is weak
- which components inside that domain are weak
- which missing metrics caused that status
- what action should be taken first

That closes an important gap in the "competitive surpass" plan: standards,
tolerance, and GD&T are now visible as first-class benchmark governance
surfaces rather than hidden behind one coarse readiness flag.
