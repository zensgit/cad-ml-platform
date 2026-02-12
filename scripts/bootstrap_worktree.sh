#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <branch> [target_dir] [base_ref]"
  echo "Example: $0 feat/graph2d-tuning ../cad-ml-platform-graph2d main"
  exit 1
fi

branch="$1"
target_dir="${2:-../$(basename "$PWD")-${branch//\//-}}"
base_ref="${3:-main}"

if [[ -e "$target_dir" ]]; then
  echo "Target exists: $target_dir"
  exit 1
fi

if git show-ref --verify --quiet "refs/heads/$branch"; then
  git worktree add "$target_dir" "$branch"
else
  git worktree add -b "$branch" "$target_dir" "$base_ref"
fi

echo "Worktree ready: $target_dir"
echo "Next:"
echo "  cd $target_dir"
echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
