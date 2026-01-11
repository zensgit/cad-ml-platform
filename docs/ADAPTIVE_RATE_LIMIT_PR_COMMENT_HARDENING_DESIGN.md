# Adaptive Rate Limit PR Comment Hardening Design

## Scope
- Make the PR comment job resilient to invalid JSON artifacts.
- Grant minimal permissions needed to post PR comments.

## Problem Statement
- The PR comment job failed when artifact JSON contained invalid data.
- The job used a read-only token, which can block comment creation.

## Design
- Add job-level permissions for `actions: read`, `contents: read`, `issues: write`, and
  `pull-requests: write` so comments can be posted.
- Guard `jq` parsing with a validity check and fallback values when artifacts are
  missing or invalid.

## Impact
- No API changes; improves CI signal by preventing false failures in the PR comment job.

## Validation
- Parsed the updated workflow with `yaml.safe_load`.
