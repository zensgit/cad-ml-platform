# Benchmark Knowledge Domain Release Gate Validation

## Scope

This change adds a new benchmark control-plane component:

- `knowledge_domain_release_gate`

It converts knowledge benchmark readiness into a release-facing gate by combining:

- capability matrix
- capability drift
- action plan
- control-plane status
- control-plane drift
- release-surface alignment

The main-layer delivery in this branch also wires the new component into:

- benchmark artifact bundle
- benchmark companion summary

## Changed Files

- `src/core/benchmark/knowledge_domain_release_gate.py`
- `src/core/benchmark/__init__.py`
- `scripts/export_benchmark_knowledge_domain_release_gate.py`
- `scripts/export_benchmark_artifact_bundle.py`
- `scripts/export_benchmark_companion_summary.py`
- `tests/unit/test_benchmark_knowledge_domain_release_gate.py`

## Design

### Output contract

`knowledge_domain_release_gate` emits:

- `status`
- `summary`
- `gate_open`
- `blocking_reasons`
- `warning_reasons`
- `releasable_domains`
- `blocked_domains`
- `partial_domains`
- `priority_domains`
- `recommended_first_action`

### Status policy

- `knowledge_domain_release_gate_ready`
  - control-plane is ready
  - no blocking drift
  - release-surface alignment is aligned
  - no blocked domains remain
- `knowledge_domain_release_gate_blocked`
  - any release blocker exists
  - any regressed drift exists
  - release-surface alignment diverges
  - action plan is still blocked
- `knowledge_domain_release_gate_partial`
  - no hard blockers remain, but warnings or partial domains still exist
- `knowledge_domain_release_gate_unavailable`
  - inputs are missing or unknown

### Downstream surfaces

The main-layer integration adds:

- bundle component row: `knowledge_domain_release_gate`
- companion component row: `knowledge_domain_release_gate`
- bundle payload fields:
  - `knowledge_domain_release_gate_status`
  - `knowledge_domain_release_gate_summary`
  - `knowledge_domain_release_gate_gate_open`
  - `knowledge_domain_release_gate_blocking_reasons`
  - `knowledge_domain_release_gate_releasable_domains`
- companion payload fields with the same release-gate summary
- markdown sections:
  - `## Knowledge Domain Release Gate`

## Validation

### Static checks

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_domain_release_gate.py \
  scripts/export_benchmark_knowledge_domain_release_gate.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  tests/unit/test_benchmark_knowledge_domain_release_gate.py
```

Passed.

```bash
flake8 \
  src/core/benchmark/knowledge_domain_release_gate.py \
  scripts/export_benchmark_knowledge_domain_release_gate.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  tests/unit/test_benchmark_knowledge_domain_release_gate.py \
  --max-line-length=100
```

Passed.

### Unit tests

```bash
pytest -q tests/unit/test_benchmark_knowledge_domain_release_gate.py
```

Result:

- `5 passed`

```bash
pytest -q \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py
```

Result:

- `20 passed`

### Diff hygiene

```bash
git diff --check
```

Passed.

## Notes

- This branch only implements the main layer plus downstream bundle/companion surfaces.
- CI wiring and PR comment layers should be stacked separately after this branch lands.
