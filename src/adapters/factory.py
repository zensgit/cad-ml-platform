"""Adapter factory dispatching format-specific parsers to CadDocument.

Initial implementation ships DXF & STL lightweight parsers; others fallback to stub.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from src.models.cad_document import BoundingBox, CadDocument, CadEntity


class _BaseAdapter:
    format: str

    async def convert(self, data: bytes, *, file_name: str = "unknown") -> Dict[str, Any]:  # legacy
        doc = await self.parse(data, file_name=file_name)
        return doc.to_unified_dict()

    async def parse(self, data: bytes, *, file_name: str) -> CadDocument:
        raise NotImplementedError


class DxfAdapter(_BaseAdapter):
    format = "dxf"

    async def parse(self, data: bytes, *, file_name: str) -> CadDocument:
        # Attempt lightweight DXF parsing; fallback to stub if library missing.
        layers: Dict[str, int] = {}
        entities: list[CadEntity] = []
        bbox = BoundingBox()
        entity_counts: Dict[str, int] = {}
        try:
            from src.utils.dxf_io import read_dxf_document_from_bytes

            doc = read_dxf_document_from_bytes(data)
            msp = doc.modelspace()
            msp_entities = list(msp)
            for e in msp_entities:
                kind = e.dxftype()
                layer = e.dxf.layer
                layers[layer] = layers.get(layer, 0) + 1
                entity_counts[kind] = entity_counts.get(kind, 0) + 1
                entities.append(CadEntity(kind=kind, layer=layer))
                # crude bbox accumulation (extents if present)
                try:
                    box = e.bbox()
                    if box:
                        (min_x, min_y, _), (max_x, max_y, _) = box.extents
                        bbox.min_x = min(bbox.min_x, min_x)
                        bbox.min_y = min(bbox.min_y, min_y)
                        bbox.max_x = max(bbox.max_x, max_x)
                        bbox.max_y = max(bbox.max_y, max_y)
                except Exception:
                    pass
            metadata = {"parser": "ezdxf"}
            try:
                from src.core.dedupcad_precision.vendor.parsers import (
                    parse_dimensions,
                    parse_text_content,
                )

                text_content = parse_text_content(msp_entities)
                if text_content:
                    metadata["text_content"] = text_content
                    metadata["text"] = " ".join(text_content)
                    metadata["text_count"] = len(text_content)
                dims = parse_dimensions(msp_entities)
                if dims:
                    metadata["dimension_count"] = len(dims)
            except Exception:
                pass
            return CadDocument(
                file_name=file_name,
                format=self.format,
                entities=entities,
                layers=layers,
                bounding_box=bbox,
                metadata=metadata,
                raw_stats={"entity_counts": entity_counts} if entity_counts else {},
            )
        except Exception:  # library absent or parse error
            return CadDocument(
                file_name=file_name,
                format=self.format,
                entities=entities,
                layers=layers,
                bounding_box=bbox,
                metadata={"parser": "stub"},
            )


class JsonAdapter(_BaseAdapter):
    format = "json"

    async def parse(self, data: bytes, *, file_name: str) -> CadDocument:
        entities: list[CadEntity] = []
        layers: Dict[str, int] = {}
        bbox = BoundingBox()
        entity_counts: Dict[str, int] = {}
        metadata: Dict[str, Any] = {"parser": "json"}
        bbox_initialized = False

        def _update_bbox_xy(x: float, y: float) -> None:
            nonlocal bbox_initialized
            if not bbox_initialized:
                bbox.min_x = bbox.max_x = float(x)
                bbox.min_y = bbox.max_y = float(y)
                bbox_initialized = True
            else:
                bbox.min_x = min(bbox.min_x, float(x))
                bbox.min_y = min(bbox.min_y, float(y))
                bbox.max_x = max(bbox.max_x, float(x))
                bbox.max_y = max(bbox.max_y, float(y))

        def _update_bbox_entity(ent: Dict[str, Any]) -> None:
            etype = str(ent.get("type", ""))
            if etype == "LINE":
                start = ent.get("start") or []
                end = ent.get("end") or []
                if len(start) >= 2:
                    _update_bbox_xy(start[0], start[1])
                if len(end) >= 2:
                    _update_bbox_xy(end[0], end[1])
            elif etype in {"CIRCLE", "ARC"}:
                center = ent.get("center") or []
                radius = float(ent.get("radius") or 0.0)
                if len(center) >= 2:
                    _update_bbox_xy(center[0] - radius, center[1] - radius)
                    _update_bbox_xy(center[0] + radius, center[1] + radius)
            elif etype in {"LWPOLYLINE", "POLYLINE"}:
                points = ent.get("points") or []
                for pt in points:
                    if pt is not None and len(pt) >= 2:
                        _update_bbox_xy(pt[0], pt[1])
            elif etype == "ELLIPSE":
                center = ent.get("center") or []
                major = ent.get("major") or [0.0, 0.0]
                ratio = float(ent.get("ratio") or 0.0)
                if len(center) >= 2:
                    major_len = (float(major[0]) ** 2 + float(major[1]) ** 2) ** 0.5
                    radius = major_len * max(1.0, ratio)
                    _update_bbox_xy(center[0] - radius, center[1] - radius)
                    _update_bbox_xy(center[0] + radius, center[1] + radius)
            elif etype in {"SPLINE", "LEADER"}:
                points = ent.get("control_points") or ent.get("vertices") or []
                for pt in points:
                    if pt is not None and len(pt) >= 2:
                        _update_bbox_xy(pt[0], pt[1])

        try:
            payload = json.loads(data.decode("utf-8", errors="ignore") or "{}")
            if isinstance(payload, list):
                payload = {"entities": payload}
            if isinstance(payload, dict):
                meta = payload.get("meta") or payload.get("metadata")
                if isinstance(meta, dict):
                    metadata["meta"] = meta
            try:
                from src.core.dedupcad_precision.vendor.v2_normalize import normalize_v2

                if isinstance(payload, dict):
                    payload = normalize_v2(payload)
            except Exception:
                pass

            text_content = payload.get("text_content") if isinstance(payload, dict) else None
            if isinstance(text_content, list):
                cleaned = [str(t) for t in text_content if str(t).strip()]
                if cleaned:
                    metadata["text_content"] = cleaned
                    metadata["text"] = " ".join(cleaned)
                    metadata["text_count"] = len(cleaned)

            dimensions = payload.get("dimensions") if isinstance(payload, dict) else None
            if isinstance(dimensions, list):
                metadata["dimension_count"] = len(dimensions)
            elif isinstance(dimensions, dict):
                metadata["dimension_count"] = len(dimensions)

            raw_entities = payload.get("entities", []) if isinstance(payload, dict) else []
            if isinstance(raw_entities, list):
                for ent in raw_entities:
                    if not isinstance(ent, dict):
                        continue
                    kind = str(ent.get("type", "UNKNOWN"))
                    layer = ent.get("layer")
                    entities.append(CadEntity(kind=kind, layer=layer))
                    entity_counts[kind] = entity_counts.get(kind, 0) + 1
                    if layer:
                        layers[layer] = layers.get(layer, 0) + 1
                    _update_bbox_entity(ent)
        except Exception:
            return CadDocument(
                file_name=file_name,
                format=self.format,
                entities=entities,
                layers=layers,
                bounding_box=bbox,
                metadata={"parser": "stub"},
            )

        return CadDocument(
            file_name=file_name,
            format=self.format,
            entities=entities,
            layers=layers,
            bounding_box=bbox,
            metadata=metadata,
            raw_stats={"entity_counts": entity_counts} if entity_counts else {},
        )


class StlAdapter(_BaseAdapter):
    format = "stl"

    async def parse(self, data: bytes, *, file_name: str) -> CadDocument:
        entities: list[CadEntity] = []
        bbox = BoundingBox()
        try:
            from io import BytesIO

            import trimesh  # type: ignore

            mesh = trimesh.load(BytesIO(data), file_type="stl")
            # Each facet treated as entity for initial stats
            for _ in mesh.faces:
                entities.append(CadEntity(kind="FACET"))
            min_x, min_y, min_z = mesh.bounds[0]
            max_x, max_y, max_z = mesh.bounds[1]
            bbox.min_x, bbox.min_y, bbox.min_z = float(min_x), float(min_y), float(min_z)
            bbox.max_x, bbox.max_y, bbox.max_z = float(max_x), float(max_y), float(max_z)
            layers: Dict[str, int] = {}
            return CadDocument(
                file_name=file_name,
                format=self.format,
                entities=entities,
                layers=layers,
                bounding_box=bbox,
                metadata={"parser": "trimesh", "facets": len(entities)},
                raw_stats={"facet_count": len(entities)},
            )
        except Exception:
            return CadDocument(
                file_name=file_name,
                format=self.format,
                entities=entities,
                layers={},
                bounding_box=bbox,
                metadata={"parser": "stub"},
            )


class StubAdapter(_BaseAdapter):
    format = "stub"

    async def parse(self, data: bytes, *, file_name: str) -> CadDocument:  # noqa: ARG002
        return CadDocument(
            file_name=file_name,
            format=self.format,
            metadata={"parser": "stub"},
        )


class AdapterFactory:
    _mapping = {
        "dxf": DxfAdapter,
        "dwg": DxfAdapter,  # dwg may be pre-converted externally; treat as dxf for now
        "stl": StlAdapter,
        "json": JsonAdapter,
        "step": None,  # placeholder will be replaced after class definition
        "stp": None,
        "iges": None,
        "igs": None,
    }

    @staticmethod
    def get_adapter(fmt: str) -> _BaseAdapter:
        fmt = fmt.lower()
        cls = AdapterFactory._mapping.get(fmt, StubAdapter)
        return cls()


class StepIgesAdapter(_BaseAdapter):
    format = "step"

    async def parse(self, data: bytes, *, file_name: str) -> CadDocument:  # noqa: ARG002
        # Attempt lightweight bounding box extraction using pythonocc if available.
        entities: list[CadEntity] = []
        bbox = BoundingBox()
        metadata: Dict[str, Any] = {"parser": "stub"}
        try:
            from OCC.Core.Bnd import Bnd_Box  # type: ignore
            from OCC.Core.BRepBndLib import brepbndlib  # type: ignore
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox  # type: ignore
            from OCC.Core.IFSelect import IFSelect_RetDone  # type: ignore
            from OCC.Core.STEPControl import STEPControl_Reader  # type: ignore
            from OCC.Core.TopAbs import TopAbs_SOLID  # type: ignore
            from OCC.Core.TopExp import TopExp_Explorer  # type: ignore

            # Reading actual STEP from bytes is non-trivial; many libs expect file path.
            # Fallback: create dummy box to simulate geometry if reader fails.
            reader = STEPControl_Reader()
            # Write bytes to temp file to allow reader usage if possible.
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            status = reader.ReadFile(tmp_path)
            if status == IFSelect_RetDone:
                reader.TransferRoots()
                shape = reader.OneShape()
                box = Bnd_Box()
                brepbndlib.Add(shape, box)
                # Approx bbox extraction
                xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
                bbox.min_x, bbox.min_y, bbox.min_z = xmin, ymin, zmin
                bbox.max_x, bbox.max_y, bbox.max_z = xmax, ymax, zmax
                exp = TopExp_Explorer(shape, TopAbs_SOLID)
                count = 0
                while exp.More():
                    entities.append(CadEntity(kind="SOLID"))
                    count += 1
                    exp.Next()
                metadata = {"parser": "pythonocc", "solids": count}
            else:
                # Synthetic geometry fallback
                dummy = BRepPrimAPI_MakeBox(10, 10, 10).Shape()
                box = Bnd_Box()
                brepbndlib.Add(dummy, box)
                xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
                bbox.min_x, bbox.min_y, bbox.min_z = xmin, ymin, zmin
                bbox.max_x, bbox.max_y, bbox.max_z = xmax, ymax, zmax
                metadata = {"parser": "pythonocc_fallback"}
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        except Exception:
            # Keep stub document
            pass
        return CadDocument(
            file_name=file_name,
            format=self.format,
            entities=entities,
            bounding_box=bbox,
            metadata=metadata,
        )


# Assign mapping after class definition
AdapterFactory._mapping["step"] = StepIgesAdapter
AdapterFactory._mapping["stp"] = StepIgesAdapter
AdapterFactory._mapping["iges"] = StepIgesAdapter
AdapterFactory._mapping["igs"] = StepIgesAdapter
