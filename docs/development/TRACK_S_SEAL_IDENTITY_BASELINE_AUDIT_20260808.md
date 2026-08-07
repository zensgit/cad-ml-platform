# Track S — SEAL + Production Identity Baseline Audit

**Date**: 2026-08-08  
**Authority**: 90-day plan §7 / §11; #537 SEAL; #538 identity; #545 cost_cap revert  
**Status**: Engineering residual audit (docs-only)

---

## 1. Scope

Preserve (do not rebuild):

| Baseline | PR | Invariant |
|---|---|---|
| Assistant SEAL | #537 | Hosted provider opt-in; no silent hosted egress default |
| Production identity fail-closed | #538 | Production refuses insecure defaults |
| No cost_cap revive | #545 | `src/core/assistant/cost_cap.py` must remain absent unless owner re-ratifies |

---

## 2. Checklist

| Check | Method | Expected |
|---|---|---|
| SEAL opt-in env documented | `.env.example` `ASSISTANT_HOSTED_PROVIDER_OPT_IN` | default off / commented false |
| Identity fail-closed docs | `.env.example` production identity block | present |
| cost_cap absent | `test ! -f src/core/assistant/cost_cap.py` | ABSENT |
| Workbench does not open hosted LLM for samples | isolated-sample runbook I2 | present |
| Decision flag separate from SEAL | `REVIEW_REUSE_DECISIONS_ENABLED` | default off |

### Commands

```bash
rg -n "ASSISTANT_HOSTED_PROVIDER_OPT_IN|ENVIRONMENT|API_KEY" .env.example | head
test ! -f src/core/assistant/cost_cap.py && echo "cost_cap ABSENT OK"
rg -n "REVIEW_REUSE_DECISIONS_ENABLED" .env.example src/core/review_reuse
```

---

## 3. Findings (workbench tranche)

- Isolated-sample runbook forbids hosted LLM egress for sample processing.
- Decision enablement is a separate env flag; enabling decisions does **not** enable hosted assistant providers.
- #545 residual board / cost_cap was not reintroduced by workbench files.

---

## 4. Acceptance

| Criterion | Status |
|---|---|
| SEAL baseline not dismantled by workbench | Pass |
| Identity baseline not dismantled by workbench | Pass |
| cost_cap not revived | Pass |
| Exact-head SEAL unit re-run | Residual (run assistant SEAL unit suite at merge head) |

---

## 5. Assistant boundary (plan §11)

90-day runtime does **not** include assistant-as-explainer redesign. Preserve seal only.
