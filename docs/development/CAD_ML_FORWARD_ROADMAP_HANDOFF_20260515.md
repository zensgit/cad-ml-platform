# CAD ML Forward Roadmap — Hand-off

Date: 2026-05-15
Branch: `phase3-vectors-batch-similarity-router-20260429`
Rollback safety tag: `pre-split-backup-20260515` (pre-split HEAD)

## 1. Verified checkpoint (what is done)

Codex's whole forward roadmap (Phases 1–6) was split into reviewable
commits, regressions fixed, and CI-verified on Python 3.10+3.11 via
`workflow_dispatch` (this is a stacked PR — heavy gates do not auto-run).

| Hash | Content | CI |
|---|---|---|
| `17a28676` | Phase 1 vectors facade closeout | green |
| `be48a1e7` | Phase 2 readiness registry | green |
| `ea95776c` | Phase 3 forward scorecard + gate | green |
| `0deb3a9e` | Phase 4 strict B-Rep golden | green |
| `c67dcbec` | Phase 5+6 DecisionService + knowledge | green |
| `58a08657` | docs — split verification | — |
| `4852ddb1` | Stage 0 fix: restore vectors `get_api_key`/`get_admin_token` | CI/Tiered/Gov ✅ |
| `9cf0500b` | Stage 1: readiness load-error contract (graph2d/uvnet/pointnet) + dual fixtures | CI(3.10+3.11)/Tiered/Gov ✅, 0 failing tests |
| `fbf751a5` | Stage 2a: dedicated OCC B-Rep eval workflow | YAML+pin-guard ✅ (runtime deferred — see §2) |

Every autonomous, low-risk stage is complete and verified. Remaining
roadmap value is **human-in-the-loop** or **correctly gated** — not more
code.

## 2. Standing constraints (do not re-discover these)

- **Local Python is 3.9.6 and cannot run the codebase** (PEP-604 at
  FastAPI import; CI target = 3.11). No local pytest is authoritative;
  `py_compile` is the only local gate.
- **Stacked-PR CI topology:** PR #472 base is
  `phase3-vectors-list-router-…`, not `main`. `ci.yml`,
  `ci-tiered-tests.yml`, `governance-gates.yml` only auto-run on
  PR→main; on this branch they must be `gh workflow run … --ref <branch>`.
- **`code-quality.yml` has no `workflow_dispatch`** → never runs until a
  PR→main. Accepted gap.
- **A NEW workflow file cannot be `workflow_dispatch`-ed from a stacked
  branch** — GitHub registers dispatch from the default branch's copy
  only. `brep-golden-eval.yml` (Stage 2a) therefore cannot be
  runtime-smoke-tested until it lands on `main`; it is statically
  verified only.

## 3. Human steps to unblock the remaining stages

### A. Land the stack toward `main` (unblocks 2a runtime + code-quality)
Retarget/merge the #472 chain so the new workflow registers and the
heavy gates + `code-quality.yml` run on the real integration path. Until
then 2a runtime and `code-quality.yml` stay structurally deferred (a
deliberate, accepted trade of the "keep stacking" decision).

### B. Curate the B-Rep golden set (unblocks 2a data)
1. Collect 50–100 real, license-clear STEP/IGES files.
2. Build a real manifest mirroring `config/brep_golden_manifest.example.json`.
   Each case (verified schema): `id`, `path`, `format` (`step`|`iges`),
   `source_type` (must NOT be `fixture`/`synthetic_demo`/`generated_mock`
   for release-eligible), `release_eligible: true`, `part_family`,
   `license`, `expected_behavior: "parse_success"`, `expected_topology`
   `{faces_min, edges_min, solids_min, graph_nodes_min, surface_types[]}`,
   `tags[]`. Top-level: `schema_version`, `name`, `description`, `root`,
   `cases[]`.
3. Validate (exact):
   `python3 scripts/validate_brep_golden_manifest.py --manifest <path> --min-release-samples 50 --fail-on-not-release-ready`
   — must exit 0.
4. Run the eval: once §A is done, dispatch the `B-Rep Golden Eval (OCC)`
   workflow with input `brep_golden_manifest_json=<path>` and
   `brep_golden_manifest_fail_on_not_release_ready=true`. Before §A, the
   pipeline can only be exercised by running
   `scripts/ci/build_brep_golden_manifest_optional.sh` on an
   OCC-provisioned machine with `BREP_GOLDEN_MANIFEST_ENABLE=true
   BREP_GOLDEN_EVAL_ENABLE=true`.

### C. Reviewed manufacturing labels (unblocks 2b)
Tooling already exists — `scripts/build_manufacturing_review_manifest.py`
(`--help` is the source of truth for the full flag set; do not assume).
Verified-relevant flags: `--from-results-csv`, `--reviewer-template-csv`
(emit the fill template), `--validate-manifest`, `--min-reviewed-samples`
(default 30), `--output-csv`. The apply/merge-approved path and its exact
flags are documented in the Codex feature docs
`docs/development/CAD_ML_REVIEWER_TEMPLATE_APPLY_*` and
`CAD_ML_REVIEW_MANIFEST_MERGE_*` — follow those + `--help`, do not
hand-type unverified flags.
**Filled labels must live in a tracked path under `data/manifests/`**,
NOT the gitignored generated `reports/benchmark/`.

### D. Wire real evidence into the scorecard (2c — ONLY after B+C)
Pass real `--brep-summary` (from §B's `summary.json`) and
`--manufacturing-evidence-summary` (from §C) into
`scripts/export_forward_scorecard.py` in the CI/release job. **Do not do
this before B+C exist** — the scorecard would read synthetic defaults and
go misleadingly green (plan risk).

## 4. Phase 7 (parametric/generative) — gated

Remains design-only until Stages 1+2 are scorecard-credible with real
data. Do not start.

## 5. Rollback

Each commit above is independently revertable by hash. Full reset:
`git reset --hard pre-split-backup-20260515`. `reports/benchmark/` is
gitignored (generated artifacts only).
