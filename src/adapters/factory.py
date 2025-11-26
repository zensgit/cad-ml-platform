"""Adapter factory dispatching format-specific parsers to CadDocument.

Initial implementation ships DXF & STL lightweight parsers; others fallback to stub.
"""

from __future__ import annotations

from typing import Any, Dict

from src.models.cad_document import CadDocument, CadEntity, BoundingBox


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
        try:
            import ezdxf  # type: ignore
            from io import BytesIO

            stream = BytesIO(data)
            doc = ezdxf.read(stream)
            msp = doc.modelspace()
            for e in msp:
                kind = e.dxftype()
                layer = e.dxf.layer
                layers[layer] = layers.get(layer, 0) + 1
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
        except Exception:  # library absent or parse error
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
            metadata={"parser": "ezdxf"},
        )


class StlAdapter(_BaseAdapter):
    format = "stl"

    async def parse(self, data: bytes, *, file_name: str) -> CadDocument:
        entities: list[CadEntity] = []
        bbox = BoundingBox()
        try:
            import trimesh  # type: ignore
            from io import BytesIO

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
            from OCC.Core.STEPControl import STEPControl_Reader  # type: ignore
            from OCC.Core.IFSelect import IFSelect_RetDone  # type: ignore
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox  # type: ignore
            from OCC.Core.Bnd import Bnd_Box  # type: ignore
            from OCC.Core.BRepBndLib import brepbndlib_Add  # type: ignore
            from OCC.Core.TopExp import TopExp_Explorer  # type: ignore
            from OCC.Core.TopAbs import TopAbs_SOLID  # type: ignore

            # Reading actual STEP from bytes is non-trivial; many libs expect file path.
            # Fallback: create dummy box to simulate geometry if reader fails.
            reader = STEPControl_Reader()
            # Write bytes to temp file to allow reader usage if possible.
            import tempfile, os

            with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            status = reader.ReadFile(tmp_path)
            if status == IFSelect_RetDone:
                reader.TransferRoots()
                shape = reader.OneShape()
                box = Bnd_Box()
                brepbndlib_Add(shape, box)
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
                brepbndlib_Add(dummy, box)
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
