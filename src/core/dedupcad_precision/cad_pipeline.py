from __future__ import annotations

import logging
import math
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DxfRenderConfig:
    size_px: int = 1024
    dpi: int = 200
    margin_ratio: float = 0.05


def extract_geom_json_from_dxf(dxf_path: Path) -> Dict[str, Any]:
    if not dxf_path.exists():
        raise FileNotFoundError(str(dxf_path))
    if dxf_path.suffix.lower() != ".dxf":
        raise ValueError("Expected .dxf file")
    from src.core.dedupcad_precision.vendor.dxf_extract import extract_dxf

    base = extract_dxf(str(dxf_path))

    # Best-effort enrichment: dimensions/hatches are optional but help precision scoring.
    try:
        import ezdxf  # type: ignore

        from src.core.dedupcad_precision.vendor.parsers import (
            parse_dimensions,
            parse_hatches,
            parse_text_content,
        )

        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()
        entities = list(msp)

        dims = parse_dimensions(entities)
        if dims:
            base["dimensions"] = dims

        hatches = parse_hatches(entities)
        if hatches:
            base["hatches"] = hatches

        text_content = parse_text_content(entities)
        if text_content:
            base["text_content"] = text_content
    except Exception as e:
        logger.debug("dxf_enrichment_failed", extra={"error": str(e), "path": str(dxf_path)})

    return base


def _resolve_layout(doc, layout_name: Optional[str]):
    if not layout_name:
        return doc.modelspace()
    normalized = layout_name.strip()
    if not normalized:
        return doc.modelspace()
    lower = normalized.lower()
    if lower in {"model", "modelspace", "msp"}:
        return doc.modelspace()
    if lower in {"paper", "paperspace", "psp", "layout"}:
        for layout in doc.layouts:
            if layout.name.lower() != "model":
                return layout
        return doc.modelspace()
    try:
        layout = doc.layouts.get(normalized)
        if layout is not None:
            return layout
    except Exception:
        pass
    try:
        return doc.layout(normalized)
    except Exception:
        return doc.modelspace()


def render_dxf_to_png(
    dxf_path: Path,
    out_png_path: Path,
    *,
    config: Optional[DxfRenderConfig] = None,
    layout_name: Optional[str] = None,
) -> None:
    """Render DXF to PNG using ezdxf + matplotlib (headless)."""
    cfg = config or DxfRenderConfig()
    if cfg.size_px <= 0:
        raise ValueError("size_px must be > 0")
    if cfg.dpi <= 0:
        raise ValueError("dpi must be > 0")
    if cfg.margin_ratio < 0:
        raise ValueError("margin_ratio must be >= 0")

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import ezdxf  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        from ezdxf import bbox  # type: ignore
        from ezdxf.addons.drawing import Frontend  # type: ignore
        from ezdxf.addons.drawing.config import (  # type: ignore
            BackgroundPolicy,
            ColorPolicy,
            Configuration,
            HatchPolicy,
            ProxyGraphicPolicy,
            TextPolicy,
        )
        from ezdxf.addons.drawing.matplotlib import MatplotlibBackend  # type: ignore
        from ezdxf.addons.drawing.properties import RenderContext  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"DXF render dependencies missing: {e}") from e

    doc = ezdxf.readfile(str(dxf_path))
    layout = _resolve_layout(doc, layout_name)
    render_text = os.getenv("DEDUPCAD2_RENDER_TEXT", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    render_hatch = os.getenv("DEDUPCAD2_RENDER_HATCH", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    # Extents computation may fail on some DXFs (e.g., broken proxy-graphics).
    # Best-effort: compute extents while skipping known-problematic entities.
    text_types = {"TEXT", "MTEXT", "ATTRIB", "ATTDEF"}
    excluded_types = {"ACAD_PROXY_ENTITY", "PROXY_ENTITY"}
    if not render_text:
        excluded_types = excluded_types | text_types
    if not render_hatch:
        excluded_types.add("HATCH")
    entities_for_bbox = [e for e in layout if e.dxftype() not in excluded_types]
    try:
        ext = bbox.extents(entities_for_bbox or layout)
        xmin, ymin, _ = ext.extmin
        xmax, ymax, _ = ext.extmax
    except Exception as e:
        logger.warning("dxf_render_extents_failed", extra={"error": str(e), "path": str(dxf_path)})
        xmin = ymin = 0.0
        xmax = ymax = 1.0

    # Guard against NaN/Inf extents.
    if not all(math.isfinite(float(v)) for v in (xmin, ymin, xmax, ymax)):
        logger.warning("dxf_render_extents_invalid", extra={"path": str(dxf_path)})
        xmin = ymin = 0.0
        xmax = ymax = 1.0
    w = float(xmax) - float(xmin)
    h = float(ymax) - float(ymin)
    if w <= 1e-9:
        w = 1.0
    if h <= 1e-9:
        h = 1.0
    margin = max(w, h) * float(cfg.margin_ratio)

    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig_size = cfg.size_px / float(cfg.dpi)
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=cfg.dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_aspect("equal")
    ax.set_xlim(float(xmin) - margin, float(xmax) + margin)
    ax.set_ylim(float(ymin) - margin, float(ymax) + margin)
    ax.axis("off")

    text_policy_default = getattr(TextPolicy, "ACCURATE", TextPolicy.FILLING)
    base_render_cfg = Configuration(
        color_policy=ColorPolicy.MONOCHROME,
        background_policy=BackgroundPolicy.WHITE,
    ).with_changes(
        # Avoid hard failures on invalid proxy-graphics (common in DWG->DXF flows).
        proxy_graphic_policy=ProxyGraphicPolicy.IGNORE,
        hatch_policy=HatchPolicy.IGNORE if not render_hatch else HatchPolicy.NORMAL,
        # Optionally skip text rendering to avoid font issues in minimal containers.
        text_policy=TextPolicy.IGNORE if not render_text else text_policy_default,
    )
    ctx = RenderContext(doc, export_mode=True)
    ctx.set_current_layout(layout)
    backend = MatplotlibBackend(ax)

    def filter_entities(entity: Any) -> bool:
        dxftype = entity.dxftype()
        if dxftype in {"ACAD_PROXY_ENTITY", "PROXY_ENTITY"}:
            return False
        if not render_text and dxftype in text_types:
            return False
        if not render_hatch and dxftype == "HATCH":
            return False
        return True

    try:
        Frontend(ctx, backend, config=base_render_cfg).draw_layout(
            layout, filter_func=filter_entities
        )
    except Exception as e:
        # Fallback: some MTEXT variants trigger layout errors in ezdxf's renderer.
        # For dedup rendering, a text-less thumbnail is preferable to a hard failure.
        try:
            fallback_cfg = base_render_cfg.with_changes(text_policy=TextPolicy.IGNORE)
            Frontend(ctx, backend, config=fallback_cfg).draw_layout(
                layout,
                filter_func=filter_entities,
            )
        except Exception:
            raise RuntimeError(f"DXF render failed: {e}") from e
    fig.savefig(str(out_png_path), dpi=cfg.dpi, facecolor="white")
    plt.close(fig)


@dataclass(frozen=True)
class OdaConverterConfig:
    exe_path: Path
    output_version: str = "ACAD2018"
    recurse: int = 0
    audit: int = 1


def convert_dwg_to_dxf_oda(dwg_path: Path, out_dxf_path: Path, *, cfg: OdaConverterConfig) -> None:
    """Convert a single DWG to DXF by invoking ODAFileConverter.exe.

    Notes:
    - ODAFileConverter works on directories, so we stage in a temp folder.
    """
    if not cfg.exe_path.exists():
        raise FileNotFoundError(str(cfg.exe_path))
    if not dwg_path.exists():
        raise FileNotFoundError(str(dwg_path))
    if dwg_path.suffix.lower() != ".dwg":
        raise ValueError("Expected .dwg file")

    out_dxf_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dwg2dxf_") as td:
        tmp_in = Path(td) / "in"
        tmp_out = Path(td) / "out"
        tmp_in.mkdir(parents=True, exist_ok=True)
        tmp_out.mkdir(parents=True, exist_ok=True)
        staged = tmp_in / dwg_path.name
        staged.write_bytes(dwg_path.read_bytes())

        cmd = [
            str(cfg.exe_path),
            str(tmp_in),
            str(tmp_out),
            str(cfg.output_version),
            "DXF",
            str(int(cfg.recurse)),
            str(int(cfg.audit)),
        ]
        try:
            subprocess.run(
                cmd,
                # On macOS, ODAFileConverter shipped as a .app bundle may rely on
                # its working directory to locate Qt plugins/resources.
                cwd=str(cfg.exe_path.parent),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ODA conversion failed: rc={e.returncode} stdout={e.stdout!r} stderr={e.stderr!r}"
            ) from e

        produced = tmp_out / (dwg_path.stem + ".dxf")
        if not produced.exists():
            candidates = list(tmp_out.rglob("*.dxf"))
            raise RuntimeError(f"ODA conversion produced no DXF: candidates={len(candidates)}")
        out_dxf_path.write_bytes(produced.read_bytes())


def convert_dwg_to_dxf_cmd(dwg_path: Path, out_dxf_path: Path, *, cmd_template: str) -> None:
    """Convert DWG to DXF by running a user-provided command template.

    Template supports placeholders: {input}, {output}.
    """
    if not dwg_path.exists():
        raise FileNotFoundError(str(dwg_path))
    if dwg_path.suffix.lower() != ".dwg":
        raise ValueError("Expected .dwg file")
    out_dxf_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = cmd_template.format(input=str(dwg_path), output=str(out_dxf_path))
    cmd_parts = shlex.split(cmd, posix=os.name != "nt")
    if not cmd_parts:
        raise ValueError("DWG->DXF command template resolved to empty command")
    subprocess.run(cmd_parts, check=True)
    if not out_dxf_path.exists():
        raise RuntimeError("DWG->DXF command finished but output missing")


def _default_oda_candidates() -> list[Path]:
    candidates = [
        Path("/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter"),
        Path("/opt/ODAFileConverter/ODAFileConverter"),
    ]
    program_files = os.getenv("ProgramFiles")
    if program_files:
        candidates.append(Path(program_files) / "ODA" / "ODAFileConverter" / "ODAFileConverter.exe")
    program_files_x86 = os.getenv("ProgramFiles(x86)")
    if program_files_x86:
        candidates.append(
            Path(program_files_x86) / "ODA" / "ODAFileConverter" / "ODAFileConverter.exe"
        )
    return candidates


def resolve_oda_exe_from_env() -> Optional[Path]:
    v = os.getenv("ODA_FILE_CONVERTER_EXE", "").strip()
    if v:
        p = Path(v)
        return p if p.exists() else None
    for candidate in _default_oda_candidates():
        if candidate.exists():
            return candidate
    return None


def resolve_dwg_converter() -> str:
    return (os.getenv("DWG_CONVERTER", "auto") or "auto").strip().lower()


def resolve_dwg_cmd_template(converter: str) -> str:
    env_map = {
        "cmd": "DWG_TO_DXF_CMD",
        "autocad": "DWG_AUTOCAD_CMD",
        "bricscad": "DWG_BRICSCAD_CMD",
        "draftsight": "DWG_DRAFTSIGHT_CMD",
    }
    if converter == "auto":
        for key in env_map.values():
            value = os.getenv(key, "").strip()
            if value:
                return value
        return ""
    env_key = env_map.get(converter, "DWG_TO_DXF_CMD")
    return os.getenv(env_key, "").strip()


def convert_dwg_to_dxf(dwg_path: Path, out_dxf_path: Path) -> None:
    converter = resolve_dwg_converter()
    if converter in {"auto", "oda"}:
        oda_exe = resolve_oda_exe_from_env()
        if oda_exe is not None:
            convert_dwg_to_dxf_oda(
                dwg_path,
                out_dxf_path,
                cfg=OdaConverterConfig(
                    exe_path=oda_exe,
                    output_version=os.getenv("ODA_OUTPUT_VERSION", "ACAD2018"),
                ),
            )
            return
        if converter == "oda":
            raise RuntimeError("ODA converter not found: set ODA_FILE_CONVERTER_EXE")

    cmd_template = resolve_dwg_cmd_template(converter)
    if cmd_template:
        convert_dwg_to_dxf_cmd(dwg_path, out_dxf_path, cmd_template=cmd_template)
        return

    if converter == "auto":
        raise RuntimeError(
            "DWG conversion unavailable: set ODA_FILE_CONVERTER_EXE, DWG_TO_DXF_CMD, "
            "DWG_AUTOCAD_CMD, DWG_BRICSCAD_CMD, or DWG_DRAFTSIGHT_CMD"
        )
    raise RuntimeError(f"DWG conversion unavailable for converter '{converter}'")
