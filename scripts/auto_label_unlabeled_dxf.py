#!/usr/bin/env python3
"""Auto-label unlabeled DWG entries using DXF text hints and knowledge rules."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import ezdxf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.knowledge.dynamic.manager import get_knowledge_manager
from src.core.dedupcad_precision.cad_pipeline import DxfRenderConfig, render_dxf_to_png

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
    "腿",
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
    "精调",
    "工作面",
    "备注",
    "比例",
    "单位",
)
_NAME_HINTS = (
    "装配图",
    "示意图",
    "原理图",
    "系统图",
    "零件图",
    "视图",
)
_FALLBACK_CONFIDENCE = {
    "auto_label:filename_numeric": 0.7,
    "auto_label:filename_pinyin": 0.7,
}


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
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _collapse_cjk_spaces(text: str) -> str:
    return re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)


def _normalize_text(text: str) -> str:
    text = _decode_mplus(text)
    text = _strip_mtext_formatting(text)
    return _collapse_cjk_spaces(text)


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
        elif dtype == "INSERT":
            attribs = getattr(entity, "attribs", [])
            for attrib in attribs:
                texts.append(str(getattr(attrib.dxf, "text", "")))

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


def _clamp_score(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _reason_from_note(note: str) -> str:
    if note.startswith("auto_label:"):
        return note.split(":", 1)[1].split()[0]
    return note


def _paddle_available() -> bool:
    try:
        import paddle  # noqa: F401
        from paddleocr import PaddleOCR  # noqa: F401

        return True
    except Exception:
        return False


def _build_ocr_manager(provider: str):
    if provider != "paddle":
        logger.warning("OCR provider %s not supported in this script.", provider)
        return None
    if not _paddle_available():
        logger.warning("PaddleOCR is not available; OCR disabled.")
        return None
    try:
        from src.core.ocr.manager import OcrManager
        from src.core.ocr.providers.paddle import PaddleOcrProvider

        return OcrManager(providers={"paddle": PaddleOcrProvider(enable_preprocess=True)})
    except Exception as exc:
        logger.warning("Failed to initialize OCR manager: %s", exc)
        return None


def _iter_ocr_layouts(layout_mode: str) -> List[str]:
    if not layout_mode:
        return ["model"]
    normalized = layout_mode.strip().lower()
    if normalized in {"both", "all"}:
        return ["paperspace", "model"]
    if normalized in {"paper", "paperspace", "psp"}:
        return ["paperspace"]
    if normalized in {"model", "modelspace", "msp"}:
        return ["model"]
    if "," in layout_mode:
        return [item.strip() for item in layout_mode.split(",") if item.strip()]
    return [layout_mode]


def _render_dxf_bytes(
    dxf_path: Path, render_cfg: DxfRenderConfig, layout_name: str | None
) -> bytes | None:
    if not dxf_path.exists():
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="ocr_render_") as tmp_dir:
            png_path = Path(tmp_dir) / "render.png"
            render_dxf_to_png(dxf_path, png_path, config=render_cfg, layout_name=layout_name)
            return png_path.read_bytes()
    except Exception as exc:
        logger.warning("DXF render failed for OCR %s: %s", dxf_path, exc)
        return None


def _crop_image_bytes(image_bytes: bytes, crop_ratio: float) -> bytes | None:
    if crop_ratio <= 0:
        return image_bytes
    try:
        from PIL import Image
        from io import BytesIO

        image = Image.open(BytesIO(image_bytes))
        width, height = image.size
        crop_w = int(width * crop_ratio)
        crop_h = int(height * crop_ratio)
        if crop_w <= 0 or crop_h <= 0:
            return image_bytes
        left = max(0, width - crop_w)
        upper = max(0, height - crop_h)
        right = width
        lower = height
        cropped = image.crop((left, upper, right, lower))
        out = BytesIO()
        cropped.save(out, format="PNG")
        return out.getvalue()
    except Exception as exc:
        logger.warning("OCR crop failed: %s", exc)
        return image_bytes


def _extract_ocr_text(
    dxf_path: Path,
    manager,
    provider: str,
    render_cfg: DxfRenderConfig,
    layout_mode: str,
    crop_ratio: float,
) -> Tuple[str, float, str]:
    for layout_name in _iter_ocr_layouts(layout_mode):
        image_bytes = _render_dxf_bytes(dxf_path, render_cfg, layout_name)
        if not image_bytes:
            continue
        image_bytes = _crop_image_bytes(image_bytes, crop_ratio)
        try:
            result = asyncio.run(manager.extract(image_bytes, strategy=provider))
        except Exception as exc:
            logger.warning("OCR failed for %s (%s): %s", dxf_path, layout_name, exc)
            continue
        parts = []
        if result.text:
            parts.append(result.text)
        if result.title_block and result.title_block.part_name:
            parts.append(result.title_block.part_name)
        if parts:
            confidence = float(result.calibrated_confidence or result.confidence or 0.0)
            return " ".join(parts), confidence, layout_name
    return "", 0.0, ""


def _heuristic_label(text: str) -> str:
    if not text:
        return ""
    candidates = []
    for match in _CN_PATTERN.finditer(text):
        phrase = match.group(0)
        if phrase in _STOPWORDS:
            continue
        stop_hit = any(stop in phrase for stop in _STOP_SUBSTRINGS)
        score = 0
        for suffix in _CANDIDATE_SUFFIXES:
            if phrase.endswith(suffix):
                score += 3
                break
        if stop_hit and score < 3:
            continue
        if len(phrase) >= 4:
            score += 1
        if score >= 3:
            candidates.append((score, len(phrase), phrase))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    best = candidates[0][2]
    return best


def _filename_fallback(file_name: str, text: str) -> Tuple[str, str]:
    upper = file_name.upper()
    if "拨叉" in text:
        return "拨叉", "auto_label:text_keyword"
    if upper in {f"{i}.DWG" for i in range(1, 11)}:
        return "练习零件图", "auto_label:filename_numeric"
    if upper.startswith("GB") and "铰制螺栓" in text:
        return "铰制螺栓", "auto_label:filename_standard"
    if upper.startswith("FU200-02-01") and "铸件" in text:
        return "铸件图", "auto_label:filename_series_casting"
    if upper.startswith("FU200-02-13") and "角钢" in text:
        return "角钢", "auto_label:filename_series_angle"
    if upper.startswith("JDA00000"):
        if "齿轮罩" in text:
            return "齿轮罩", "auto_label:filename_series_title"
        if "上壳" in text:
            return "上壳", "auto_label:filename_series_title"
    if upper.startswith("JDC00000"):
        if "油管布置示意图" in text:
            return "油管布置示意图", "auto_label:filename_series_title"
        if "油管布置" in text:
            return "油管布置示意图", "auto_label:filename_series_title"
    if upper.startswith("ZHITUI"):
        return "支腿", "auto_label:filename_pinyin"
    for hint in _NAME_HINTS:
        if hint in text:
            return hint, "auto_label:name_hint"
    return "", ""


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
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence required to keep auto labels",
    )
    parser.add_argument(
        "--enable-ocr",
        action="store_true",
        help="Enable OCR on rendered DXF images to extract additional text",
    )
    parser.add_argument(
        "--ocr-provider",
        default="paddle",
        help="OCR provider to use (default: paddle)",
    )
    parser.add_argument("--ocr-render-size-px", type=int, default=1600)
    parser.add_argument("--ocr-render-dpi", type=int, default=250)
    parser.add_argument("--ocr-render-margin", type=float, default=0.08)
    parser.add_argument(
        "--ocr-layout",
        default="both",
        help="OCR layout to render: model|paper|both|<layout-name>",
    )
    parser.add_argument(
        "--ocr-crop-ratio",
        type=float,
        default=0.0,
        help="Crop bottom-right corner ratio before OCR (0 disables).",
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

    ocr_manager = None
    render_cfg = None
    if args.enable_ocr:
        os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "1")
        os.environ.setdefault("DEDUPCAD2_RENDER_TEXT", "1")
        logging.getLogger("ezdxf").setLevel(logging.WARNING)
        logging.getLogger("ezdxf.addons.drawing").setLevel(logging.WARNING)
        ocr_manager = _build_ocr_manager(args.ocr_provider)
        render_cfg = DxfRenderConfig(
            size_px=args.ocr_render_size_px,
            dpi=args.ocr_render_dpi,
            margin_ratio=args.ocr_render_margin,
        )

    for row in rows:
        file_name = row.get("file_name", "")
        stem = Path(file_name).stem
        dxf_path = dxf_dir / f"{stem}.dxf"
        label = ""
        label_en = ""
        confidence = 0.0
        ocr_used = False
        ocr_confidence = 0.0
        ocr_layout = ""
        auto_label_reason = "missing_dxf"
        note = "auto_label:missing_dxf"
        if dxf_path.exists():
            try:
                doc = ezdxf.readfile(dxf_path)
                text = _extract_text(doc)
                if ocr_manager and render_cfg:
                    ocr_text, ocr_confidence, ocr_layout = _extract_ocr_text(
                        dxf_path,
                        ocr_manager,
                        args.ocr_provider,
                        render_cfg,
                        args.ocr_layout,
                        args.ocr_crop_ratio,
                    )
                    if ocr_text:
                        text = f"{text} {ocr_text}".strip()
                        ocr_used = True
                label, score = _pick_label(text)
                if label:
                    confidence = _clamp_score(score)
                    candidates = synonyms.get(label, [])
                    label_en = candidates[0] if candidates else ""
                    auto_label_reason = "rule_match_ocr" if ocr_used else "rule_match"
                    note = f"auto_label:rule_match score={confidence:.2f}"
                else:
                    heuristic = _heuristic_label(text)
                    if heuristic:
                        label = heuristic
                        confidence = 0.7
                        candidates = synonyms.get(label, [])
                        label_en = candidates[0] if candidates else ""
                        auto_label_reason = "heuristic_match_ocr" if ocr_used else "heuristic_match"
                        note = "auto_label:heuristic_match"
                    else:
                        fallback, fallback_note = _filename_fallback(file_name, text)
                        if fallback:
                            label = fallback
                            confidence = _FALLBACK_CONFIDENCE.get(fallback_note, 0.4)
                            candidates = synonyms.get(label, [])
                            label_en = candidates[0] if candidates else ""
                            auto_label_reason = _reason_from_note(fallback_note)
                            note = fallback_note
                        else:
                            auto_label_reason = "no_text_match"
                            note = "auto_label:no_text_match"
            except Exception as exc:
                logger.warning("DXF parse failed for %s: %s", dxf_path, exc)
                auto_label_reason = "parse_error"
                note = f"auto_label:parse_error {exc}"
        if label and confidence < args.min_confidence:
            note = (
                f"{note} candidate={label} (below threshold {args.min_confidence:.2f})"
            )
            label = ""
            label_en = ""
        row["label_cn"] = label
        row["label_en"] = label_en
        row["label_confidence"] = f"{confidence:.2f}"
        row["auto_label_reason"] = auto_label_reason
        row["ocr_used"] = "1" if ocr_used else "0"
        row["ocr_confidence"] = f"{ocr_confidence:.2f}"
        row["ocr_layout"] = ocr_layout
        row["notes"] = note

    output_path = Path(args.output_csv) if args.output_csv else input_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_name",
                "relative_path",
                "source_dir",
                "label_cn",
                "label_en",
                "label_confidence",
                "auto_label_reason",
                "ocr_used",
                "ocr_confidence",
                "ocr_layout",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote auto-labeled rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
