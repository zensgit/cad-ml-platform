#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import httpx


def _headers(api_key: str) -> Dict[str, str]:
    return {"X-API-Key": api_key}


def _request(method: str, url: str, api_key: str, **kwargs: Any) -> httpx.Response:
    return httpx.request(method, url, headers=_headers(api_key), timeout=20.0, **kwargs)


def _fetch_vector_ids(base_url: str, api_key: str, limit: int) -> List[str]:
    resp = _request(
        "GET",
        f"{base_url}/api/v1/vectors/",
        api_key,
        params={"source": "memory", "limit": limit},
    )
    resp.raise_for_status()
    payload = resp.json()
    return [item.get("id") for item in payload.get("vectors", []) if item.get("id")]


def _write_report(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Vector migration batch runner")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--api-key", default="test", help="API key")
    parser.add_argument("--to-version", default="v4", help="Target feature version")
    parser.add_argument("--limit", type=int, default=200, help="Max vector ids to migrate")
    parser.add_argument("--preview-limit", type=int, default=10, help="Preview sample size")
    parser.add_argument("--apply", action="store_true", help="Apply migration (default: dry-run)")
    parser.add_argument("--report-path", default="", help="Optional report path")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    to_version = args.to_version
    dry_run = not args.apply

    ids = _fetch_vector_ids(base_url, args.api_key, args.limit)

    report_lines = [
        f"# Vector Migration Batch Report ({datetime.utcnow().date().isoformat()})",
        "",
        f"- base_url: `{base_url}`",
        f"- to_version: `{to_version}`",
        f"- dry_run: `{dry_run}`",
        f"- id_count: {len(ids)}",
        "",
    ]

    if not ids:
        report_lines.append("## Result")
        report_lines.append("- No vectors found in memory store; nothing to migrate.")
        report_path = Path(
            args.report_path
            or f"reports/DEV_VECTOR_MIGRATION_BATCH_{datetime.utcnow().strftime('%Y%m%d')}.md"
        )
        _write_report(report_path, report_lines)
        return 0

    preview_resp = _request(
        "GET",
        f"{base_url}/api/v1/vectors/migrate/preview",
        args.api_key,
        params={"to_version": to_version, "limit": args.preview_limit},
    )
    preview_resp.raise_for_status()
    preview = preview_resp.json()

    migrate_resp = _request(
        "POST",
        f"{base_url}/api/v1/vectors/migrate",
        args.api_key,
        json={"ids": ids, "to_version": to_version, "dry_run": dry_run},
    )
    migrate_resp.raise_for_status()
    migrate = migrate_resp.json()

    report_lines.extend(
        [
            "## Preview",
            f"- total_vectors: {preview.get('total_vectors')}",
            f"- by_version: {json.dumps(preview.get('by_version', {}))}",
            f"- estimated_dimension_changes: {json.dumps(preview.get('estimated_dimension_changes', {}))}",
            f"- migration_feasible: {preview.get('migration_feasible')}",
            f"- warnings: {json.dumps(preview.get('warnings', []))}",
            "",
            "## Migration",
            f"- total: {migrate.get('total')}",
            f"- migrated: {migrate.get('migrated')}",
            f"- skipped: {migrate.get('skipped')}",
            f"- dry_run_total: {migrate.get('dry_run_total')}",
        ]
    )

    report_path = Path(
        args.report_path
        or f"reports/DEV_VECTOR_MIGRATION_BATCH_{datetime.utcnow().strftime('%Y%m%d')}.md"
    )
    _write_report(report_path, report_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
