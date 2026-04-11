# Eval Reporting Closeout Decision And Residual Risk — Design

日期：2026-04-05

## Scope

Batch 23B 基于 Batch 23A 的真实 GitHub-hosted `push/main` run 证据，给出 `eval reporting` rationalization / consolidate 主线的最终 closeout 结论。

## Closeout Decision

结论：`closeout-ready`

依据：

1. 真实 qualifying run `24066289833` 已完成 `success`
2. evaluate job `success`
3. deploy-pages job `success`
4. Pages 成功部署到 `https://zensgit.github.io/cad-ml-platform/`
5. 收口后的 retained post-deploy surfaces 全部 materialize
6. consolidated deploy-pages summary 实际执行成功

因此，Batch 8 到 Batch 22 的 eval reporting rationalization / consolidate 主线，已经在 GitHub-hosted `push/main` 真实运行中完成端到端验证。

## Residual Risk Classification

Residual risk 不是代码结构性 blocker，而是两个 operational / process-level 项：

### 1. Hybrid Superpass Strict Gate Operational Mode

- 当前仓库 Actions variable：
  - `HYBRID_SUPERPASS_FAIL_ON_FAILED = false`
- 真实成功 run 依赖该 soft-mode 配置，strict-gate step 因此没有阻断 evaluate job

含义：

- 如果未来该变量被改回 `true`
- 且 underlying hybrid superpass gate 条件仍未满足
- `push/main` 可能再次在 evaluate job 中被 strict gate 阻断，导致 deploy-pages 不执行

这属于 operational policy 风险，而不是 eval reporting pipeline 本身的结构问题。

### 2. Eval Reporting Commit Status Permission Gap

- `Post Eval Reporting status check` step 在 run `24066289833` 中执行成功
- 但日志明确记录：
  - `Status check skipped (fail-soft): Resource not accessible by integration`
- 同一 `head sha` 的 commit status API 返回 `total_count = 0`

含义：

- 独立 `Eval Reporting` commit status surface 没有真正写入 merge commit
- 但该 surface 在设计上本来就是 fail-soft，不阻断主 workflow

这属于 permission / integration posture 风险，而不是 Pages / artifact / summary 主线失败。

## Why This Is Still Closeout-Ready

即使存在上述 residual risks，本轮仍判定为 `closeout-ready`，因为：

1. 本轮主目标是验证收口后的 eval reporting workflow 是否在 GitHub-hosted `push/main` run 中端到端跑通
2. 这个目标已经被真实 run 证明满足
3. 剩余问题不再要求继续开新的 rationalization / refactor batch
4. 若未来需要修复，也应是独立、最小、运营向的 follow-up，而不是继续扩展 eval reporting 主线

## Recommended Stop Condition

建议现在停止继续开新的 eval reporting batch。

后续只在以下条件成立时再开最小修复：

1. 必须让独立 `Eval Reporting` commit status 真正 materialize
2. 必须把 `HYBRID_SUPERPASS_FAIL_ON_FAILED` 恢复为 strict 模式并保持 `push/main` 绿灯
3. 真实 GitHub-hosted run 再次暴露 Pages / artifact / summary regression

否则，本主线到此收口。

