from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_backfill_prefers_cache_manifest_then_hash_fallback(tmp_path: Path) -> None:
    from scripts.backfill_manifest_cache_paths import backfill_manifest_cache_paths

    cache_dir = tmp_path / "data" / "graph_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "data" / "manifests" / "manifest.csv"
    cache_manifest = cache_dir / "cache_manifest.csv"

    mapped_file = "/fixtures/from-cache-manifest.dxf"
    mapped_cache = cache_dir / "mapped.pt"
    mapped_cache.write_text("mapped", encoding="utf-8")

    hashed_file = "/fixtures/from-hash-fallback.dxf"
    hashed_cache = cache_dir / f"{hashlib.md5(hashed_file.encode()).hexdigest()}.pt"
    hashed_cache.write_text("hashed", encoding="utf-8")

    missing_file = "/fixtures/unfillable.dxf"

    _write_csv(
        cache_manifest,
        ["file_path", "cache_path", "taxonomy_v2_class"],
        [[mapped_file, str(mapped_cache), "A"]],
    )
    _write_csv(
        manifest,
        ["file_path", "cache_path", "taxonomy_v2_class"],
        [
            [mapped_file, "", "A"],
            [hashed_file, "", "B"],
            [missing_file, "", "C"],
        ],
    )

    result = backfill_manifest_cache_paths(manifest_path=manifest, cache_dir=cache_dir)
    rows = _read_rows(manifest)

    assert result["filled"] == 2
    assert result["remaining"] == 1
    assert rows[0]["cache_path"] == str(mapped_cache)
    assert rows[1]["cache_path"] == str(hashed_cache)
    assert rows[2]["cache_path"] == ""


def test_backfill_main_fails_closed_when_rows_remain_missing_cache_path(
    tmp_path: Path,
    capsys,
) -> None:
    from scripts.backfill_manifest_cache_paths import main

    cache_dir = tmp_path / "data" / "graph_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "data" / "manifests" / "manifest.csv"

    _write_csv(
        manifest,
        ["file_path", "cache_path", "taxonomy_v2_class"],
        [["/fixtures/unfillable.dxf", "", "X"]],
    )

    rc = main(["--manifest", str(manifest), "--cache-dir", str(cache_dir)])
    captured = capsys.readouterr()
    rows = _read_rows(manifest)

    assert rc == 1
    assert "FATAL:" in captured.out
    assert rows[0]["cache_path"] == ""
