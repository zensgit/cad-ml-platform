#!/usr/bin/env python3
"""Auto-label unlabeled DWG entries using DXF text hints and knowledge rules."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import ezdxf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.knowledge.dynamic.manager import get_knowledge_manager

logger = logging.getLogger(__name__)


def _load_synonyms(path: Path | None) -> Dict[str, List[str]]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    output: Dict[str, List[str]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            output[str(key)] = [str(item) for item in value if str(item).strip()]
    return output


_MPLUS_PATTERN = re.compile(r"\\M\+([0-9A-Fa-f]{4,6})")
_MTEXT_FMT_PATTERN = re.compile(r"\{[^;]*;")
_CN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,12}")
_CANDIDATE_SUFFIXES = (
    "装配图",
    "视图",
    "机构",
    "系统",
    "箱体",
    "壳体",
    "工作台",
    "组件",
    "部件",
    "蜗杆",
    "蜗轮",
    "刀架",
    "滑板",
    "立柱",
    "床身",
    "轴",
    "座",
    "盖",
    "板",
    "泵",
    "阀",
)
_STOPWORDS = {
    "设计员",
    "主任设计师",
    "标准化审查",
    "产品工艺师",
    "技术部部长",
    "总工程师",
    "技术要求",
    "不装",
    "并用件",
    "代替",
    "右部的件",
    "毫米",
}
_STOP_SUBSTRINGS = (
    "不得",
    "更改",
    "公司",
    "文件号",
    "焊缝",
    "调整",
    "装配",
    "精调",
    "齿轮",
    "工作面",
    "备注",
    "比例",
    "单位",
)


def _decode_mplus(text: str) -> str:
    def _replace(match: re.Match) -> str:
        hex_str = match.group(1)[-4:]
        try:
            return bytes.fromhex(hex_str).decode("gbk")
        except Exception:
            return ""

    return _MPLUS_PATTERN.sub(_replace, text)


def _strip_mtext_formatting(text: str) -> str:
    text = _MTEXT_FMT_PATTERN.sub("", text)
    text = text.replace("}", "")
    text = text.replace("\\P", " ")
    text = re.sub(r"\\\\[A-Za-z]+[0-9\\.]*", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def _normalize_text(text: str) -> str:
    text = _decode_mplus(text)
    return _strip_mtext_formatting(text)


def _iter_text_entities(doc: ezdxf.EzdxfDocument) -> List[str]:
    texts: List[str] = []

    def _collect(entity) -> None:
        dtype = entity.dxftype()
        if dtype == "TEXT":
            texts.append(str(getattr(entity.dxf, "text", "")))
        elif dtype == "MTEXT":
            texts.append(str(getattr(entity, "text", "")))
        elif dtype in {"ATTRIB", "ATTDEF"}:
            texts.append(str(getattr(entity.dxf, "text", "")))

    for layout in doc.layouts:
        for entity in layout:
            _collect(entity)
    for block in doc.blocks:
        for entity in block:
            _collect(entity)
    return texts


def _extract_text(doc: ezdxf.EzdxfDocument) -> str:
    texts = []
    for raw in _iter_text_entities(doc):
        cleaned = _normalize_text(raw)
        if cleaned:
            texts.append(cleaned)
    return " ".join(texts)


def _pick_label(text: str) -> Tuple[str, float]:
    if not text:
        return "", 0.0
    km = get_knowledge_manager()
    hints = km.get_part_hints(text=text, geometric_features=None, entity_counts=None)
    if not hints:
        return "", 0.0
    label, score = max(hints.items(), key=lambda item: item[1])
    return str(label), float(score)


def _heuristic_label(text: str) -> str:
    if not text:
        return ""
    candidates = []
    for match in _CN_PATTERN.finditer(text):
        phrase = match.group(0)
        if phrase in _STOPWORDS:
            continue
        if any(stop in phrase for stop in _STOP_SUBSTRINGS):
            continue
        score = 0
        for suffix in _CANDIDATE_SUFFIXES:
            if phrase.endswith(suffix):
                score += 3
                break
        if len(phrase) >= 4:
            score += 1
        if score >= 3:
            candidates.append((score, len(phrase), phrase))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    best = candidates[0][2]
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-label unlabeled DWG entries.")
    parser.add_argument(
        "--input-csv",
        default="reports/MECH_4000_DWG_UNLABELED_LABELS_TEMPLATE_20260119.csv",
        help="Unlabeled mapping CSV",
    )
    parser.add_argument(
        "--dxf-dir",
        required=True,
        help="DXF directory containing converted files",
    )
    parser.add_argument(
        "--synonyms-json",
        default="data/knowledge/label_synonyms_template.json",
        help="Synonyms JSON (for English labels)",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Output CSV (defaults to overwrite input)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    dxf_dir = Path(args.dxf_dir)
    if not dxf_dir.exists():
        raise FileNotFoundError(str(dxf_dir))

    synonyms = _load_synonyms(Path(args.synonyms_json))

    rows: List[Dict[str, str]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))

    for row in rows:
        file_name = row.get("file_name", "")
        stem = Path(file_name).stem
        dxf_path = dxf_dir / f"{stem}.dxf"
        label = ""
        label_en = ""
        score = 0.0
        note = "auto_label:missing_dxf"
        if dxf_path.exists():
            try:
                doc = ezdxf.readfile(dxf_path)
                text = _extract_text(doc)
                label, score = _pick_label(text)
                if label:
                    candidates = synonyms.get(label, [])
                    label_en = candidates[0] if candidates else ""
                    note = f"auto_label:rule_match score={score:.2f}"
                else:
                    heuristic = _heuristic_label(text)
                    if heuristic:
                        label = heuristic
                        candidates = synonyms.get(label, [])
                        label_en = candidates[0] if candidates else ""
                        note = "auto_label:heuristic_match"
                    else:
                        note = "auto_label:no_text_match"
            except Exception as exc:
                logger.warning("DXF parse failed for %s: %s", dxf_path, exc)
                note = f"auto_label:parse_error {exc}"
        row["label_cn"] = label
        row["label_en"] = label_en
        row["notes"] = note

    output_path = Path(args.output_csv) if args.output_csv else input_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_name", "relative_path", "source_dir", "label_cn", "label_en", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote auto-labeled rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
