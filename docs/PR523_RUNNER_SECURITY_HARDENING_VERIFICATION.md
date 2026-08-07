# PR #523 Runner Security Hardening — Verification

## Scope

Workflow / CI tests / verification docs only. No product runtime changes,
required-check weakening, repository settings, commit, push, or merge.

Preserved from earlier PR #523 / #526 work already on `main`:

- Makefile honors `actions/setup-python` `$pythonLocation`
- Docker build job remains pinned to GitHub-hosted (registry egress)

## Threat model (why this exists)

This is a **public** repository. Repo variable `CADML_LINUX_RUNNER=cad-ml`
selects persistent self-hosted runners (`cadml-linux-1/2`). Any
`pull_request` job that resolves `runs-on` to those labels can execute
untrusted PR code (same-repo branches and forks) on long-lived machines.

Self-hosted runners may register **arbitrary custom labels**, including names
that look like GitHub-hosted prefixes (`ubuntu-private`). Prefix/regex
classification of labels is therefore **not** proof of GitHub-hosted compute.

Skip-proof gates cannot be defended with a growing denylist of shell tokens
(`|| true`, `set +e`, …). Custom `shell:`, job/step `if:`, and
`defaults.run.shell` sit **outside** the `run` string and bypass substring
checks. The durable approach is **exact standalone commands** for a fixed set
of critical steps.

## Invariants

| # | Invariant | Enforcement |
|---|-----------|-------------|
| 1 | No PR path selects self-hosted | Canonical CADML is fail-closed; PR → `ubuntu-latest` |
| 2 | Self-hosted only for trusted refs | `push@main` / `schedule` / `workflow_dispatch@main` |
| 3 | Literal hosted labels are an explicit set | Currently only `ubuntu-latest` |
| 4 | Concurrency is event-scoped (exact form) | Canonical cancel group only |
| 5 | `uvnet-inspector-gate` is skip-proof | Three exact critical steps + structural constraints |

## Trusted `runs-on` expression

```yaml
runs-on: ${{ ((github.event_name == 'push' && github.ref == 'refs/heads/main') ||
  (github.event_name == 'schedule') ||
  (github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main')) &&
  (vars.CADML_LINUX_RUNNER || 'ubuntu-latest') || 'ubuntu-latest' }}
```

## Approved GitHub-hosted labels (literal `runs-on`)

Policy set in tests (deliberate extension required for any addition):

- `ubuntu-latest` **only**

Rejected: `ubuntu-private`, `ubuntu-evil`, `ubuntu-22.04`, `windows-latest`,
`macos-latest`, multi-label lists with custom tags.

## Concurrency expression

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.event_name }}-${{ github.event.pull_request.number || github.run_id }}
  cancel-in-progress: true
```

Token presence without the exact form is insufficient.

### `cancel-in-progress` exact contract (fail-closed)

When `concurrency.cancel-in-progress` is **present**, only BaseLoader literal
strings are accepted:

| Value | Meaning |
|-------|---------|
| `"true"` | Cancellation **on** → group must equal the exact canonical form above |
| `"false"` | Cancellation **off** → cannot cross-cancel; group form free |

**Rejected** (fail-closed): `${{ true }}`, `yes`, `1`, `True`, unknown spellings.
Expression forms used to skip group validation while GitHub still cancels.

## `uvnet-inspector-gate` (structural skip-proof)

### Workflow layout

1. Ordinary **Install dependencies** (`requirements.txt` / dev) — **no** torch pin.
2. **Install pinned torch (skip-proof)** — exact:
   `python -m pip install "torch==2.1.0"`
3. **Assert torch importable…** — exact:
   `python -c "import torch; print('torch', torch.__version__)"`
4. **Run uvnet checkpoint inspector tests…** — exact:
   `python -m pytest tests/unit/test_uvnet_checkpoint_inspect.py -v -rs`

### Structural requirements (tests)

- `runs-on: ubuntu-latest` (approved set).
- **No** job-level `if:`.
- **No** job-level `needs:` (or only an empty list) — a skipped upstream must
  never skip this security proof (required-context skip/success class).
- **No** `continue-on-error` **key** on the job or on any critical step —
  key must be **absent**, not merely parse as false. Expression forms like
  `${{ true }}` bypass spelling-based truthy checks.
- **No** workflow- or job-level `defaults.run.shell`.
- Workflow `on.pull_request` **positive trigger contract** (not only a paths denylist):
  - value is a **mapping** or unfiltered **null/empty** form;
  - reject sequence/scalar activity shorthand (e.g. `[closed]`);
  - reject `types` and `branches-ignore` entirely;
  - if `branches` is set, it must be a list **containing `main`**
    (`[main, master]` is valid);
  - continue rejecting `paths` and `paths-ignore`.
- Each of the three critical steps:
  - **no** `if:` (any event filter is a skip surface),
  - **no** `shell:` override,
  - **no** `continue-on-error` key,
  - `run` **exactly equals** the canonical standalone command after whitespace
    normalization (no trailing `|| …`, `; exit 0`, extra statements).
- Dependency install must remain separate and must not contain `torch==`.

This closes soft-fail / skip classes outside denylisted shell tokens: job
`if`, event-gated step `if`, `shell: bash {0}`, defaults shells,
`continue-on-error: ${{ true }}`, and `needs: [skipped-upstream]`.

## Regression tests

```bash
# Requires Python >= 3.10 (repo standard; CI uses 3.11).
python3.11 -m pytest tests/unit/test_workflow_runner_security.py -q
```

Dynamic discovery of every file under `.github/workflows/` for **both**
`.yml` and `.yaml` (GitHub accepts either; dropping one extension is fail-open).
No workflow name allowlist. Observed tmp_path test locks both globs.

Proves:

1. `runs-on` is only approved literal `ubuntu-latest` or exact canonical CADML.
2. Custom / dynamic / unresolved labels and expressions are rejected.
3. PR and untrusted non-PR contexts never select self-hosted under CADML.
4. Cancel concurrency: literal `"true"` requires exact canonical group; literal
   `"false"` is no-cancel; dynamic/expression booleans are rejected.
5. `uvnet-inspector-gate` matches the structural exact-command contract above.
6. Mutations for: `ubuntu-private`, matrix/vars/inputs dynamics, job `if: false`,
   pytest `if: push`, step/job/workflow `shell` defaults, `|| echo swallowed`,
   `; exit 0`, folded torch into deps, missing critical step,
   `continue-on-error: ${{ true }}` (job + step), `needs: [upstream]`,
   missing `pull_request`, PR `paths` / `paths-ignore`, `types: [closed]`,
   `branches: [feature/**]`, `branches-ignore: [main]`, sequence `[closed]`.

## Pre-fix RED / post-fix GREEN

| Case | Hazard | Post-fix |
|------|--------|----------|
| Unconditional CADML on PR | Self-hosted on PR | Banned |
| `ubuntu-private` prefix match | Custom self-hosted tag | Rejected (not in set) |
| `\|\| true` / token denylist | Infinite shell dialects | Exact command equality |
| Job `if: false` | Gate never runs | Rejected |
| Step `if: push` | Skips on PR | Rejected |
| `shell: bash {0}` / defaults | Exit-code ignore | Rejected |
| `… \|\| echo` / `; exit 0` | Always-green run | Rejected (≠ canonical) |
| `continue-on-error: ${{ true }}` | Soft-fail via expression | Key must be absent |
| `needs: [skipped-upstream]` | Required check skip/success | No needs (independent) |
| PR `paths` / missing PR trigger | Gate never scheduled | Unfiltered / main-targeting PR |
| `types: [closed]` | Only closed activity | Reject `types` |
| `branches: [feature/**]` | Never runs for main PRs | `branches` must include `main` |
| `branches-ignore: [main]` | Excludes default branch | Reject `branches-ignore` |
| `pull_request: [closed]` | Activity sequence shorthand | Mapping or null/empty only |
| Drop `*.yaml` discovery | evil.yaml invisible to guards | Both extensions required |
| `cancel-in-progress: ${{ true }}` | Skips group check, GH cancels | Only literal `true`/`false` |

## Residual risks

1. **Approved-label set is narrow by design.** New real GH-hosted labels need a
   deliberate add to `APPROVED_GITHUB_HOSTED_LABELS`.
2. **New dynamic `runs-on` dialects** need an explicit canonical form in policy.
3. **Repository settings** (`CADML_LINUX_RUNNER`, runner registration) are outside
   this PR; unset var fails closed to hosted.
4. **Pre-existing** non-cancel concurrency groups (`evaluation-soft-mode-smoke`,
   `hybrid-blind-strict-real-e2e`) left unchanged.
5. **Exact-command contract** means intentional flags/args changes to the three
   critical steps require a paired test/policy update (by design).
6. Hosted torch install costs PR minutes for this gate only (intentional).
7. **`continue-on-error` / `needs` key absence** is deliberate: any spelling or
   expression form of soft-fail/skip coupling is rejected without enumerating
   GHA expression dialects.
