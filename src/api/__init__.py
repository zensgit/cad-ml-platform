"""API路由聚合"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from fastapi import APIRouter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouterImportRecord:
    """Single router import/registration diagnostic record."""

    name: str
    module: str
    required: bool
    imported: bool
    registered: bool = False
    prefix: Optional[str] = None
    tags: Tuple[str, ...] = ()
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    registration_error_type: Optional[str] = None
    registration_error_message: Optional[str] = None


@dataclass
class RouterImportDiagnostics:
    """Mutable router import diagnostics container."""

    records: List[RouterImportRecord] = field(default_factory=list)

    def add_import(
        self,
        *,
        name: str,
        module: str,
        required: bool,
        imported: bool,
        error: Optional[BaseException] = None,
    ) -> None:
        self.records.append(
            RouterImportRecord(
                name=name,
                module=module,
                required=required,
                imported=imported,
                error_type=type(error).__name__ if error is not None else None,
                error_message=str(error) if error is not None else None,
            )
        )

    def mark_registered(
        self,
        *,
        name: str,
        prefix: Optional[str],
        tags: Sequence[str],
    ) -> None:
        for idx in range(len(self.records) - 1, -1, -1):
            record = self.records[idx]
            if record.name != name:
                continue
            self.records[idx] = RouterImportRecord(
                name=record.name,
                module=record.module,
                required=record.required,
                imported=record.imported,
                registered=True,
                prefix=prefix,
                tags=tuple(tags),
                error_type=record.error_type,
                error_message=record.error_message,
            )
            return

    def mark_registration_error(
        self,
        *,
        name: str,
        prefix: Optional[str],
        tags: Sequence[str],
        error: BaseException,
    ) -> None:
        for idx in range(len(self.records) - 1, -1, -1):
            record = self.records[idx]
            if record.name != name:
                continue
            self.records[idx] = RouterImportRecord(
                name=record.name,
                module=record.module,
                required=record.required,
                imported=record.imported,
                registered=False,
                prefix=prefix,
                tags=tuple(tags),
                error_type=record.error_type,
                error_message=record.error_message,
                registration_error_type=type(error).__name__,
                registration_error_message=str(error),
            )
            return

    def snapshot(self) -> Dict[str, Any]:
        imported = sum(1 for record in self.records if record.imported)
        registered = sum(1 for record in self.records if record.registered)
        required_failures = sum(
            1 for record in self.records if record.required and not record.imported
        )
        optional_failures = sum(
            1 for record in self.records if not record.required and not record.imported
        )
        status = "ok"
        if required_failures or optional_failures:
            status = "degraded"

        def _record_to_dict(record: RouterImportRecord) -> Dict[str, Any]:
            return {
                "name": record.name,
                "module": record.module,
                "required": record.required,
                "imported": record.imported,
                "registered": record.registered,
                "prefix": record.prefix,
                "tags": list(record.tags),
                "error_type": record.error_type,
                "error_message": record.error_message,
                "registration_error_type": record.registration_error_type,
                "registration_error_message": record.registration_error_message,
            }

        return {
            "status": status,
            "imported_count": imported,
            "registered_count": registered,
            "required_failures": required_failures,
            "optional_failures": optional_failures,
            "records": [_record_to_dict(record) for record in self.records],
        }


_ROUTER_IMPORT_DIAGNOSTICS = RouterImportDiagnostics()


def _load_router_module(
    *,
    diagnostics: RouterImportDiagnostics,
    name: str,
    module: str,
    required: bool = False,
    importer: Callable[[str], Any] = importlib.import_module,
) -> Any:
    try:
        loaded = importer(module)
        diagnostics.add_import(
            name=name,
            module=module,
            required=required,
            imported=True,
        )
        return loaded
    except Exception as exc:
        logger.warning(
            "router_import_failed",
            extra={
                "router": name,
                "module_path": module,
                "required": required,
                "error_type": type(exc).__name__,
            },
        )
        diagnostics.add_import(
            name=name,
            module=module,
            required=required,
            imported=False,
            error=exc,
        )
        if required:
            raise
        return None


def _include_router(
    v1_router: APIRouter,
    *,
    diagnostics: RouterImportDiagnostics,
    name: str,
    module: Any,
    prefix: Optional[str] = None,
    tags: Sequence[str] = (),
) -> None:
    if module is None:
        return
    try:
        if prefix is None:
            v1_router.include_router(module.router, tags=list(tags))  # type: ignore[attr-defined]
        else:
            v1_router.include_router(
                module.router, prefix=prefix, tags=list(tags)
            )  # type: ignore[attr-defined]
        diagnostics.mark_registered(name=name, prefix=prefix, tags=tags)
    except Exception as exc:
        diagnostics.mark_registration_error(
            name=name,
            prefix=prefix,
            tags=tags,
            error=exc,
        )
        logger.warning(
            "router_registration_failed",
            extra={
                "router": name,
                "prefix": prefix,
                "error_type": type(exc).__name__,
            },
        )
        raise


def get_router_import_diagnostics() -> Dict[str, Any]:
    """Return a stable snapshot of router import/registration status."""
    return _ROUTER_IMPORT_DIAGNOSTICS.snapshot()


def _import_router(name: str, module: str, required: bool = False) -> Any:
    return _load_router_module(
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name=name,
        module=module,
        required=required,
    )


# Import critical routers first (health). Other routers are imported best-effort.
health = _import_router("health", "src.api.v1.health", required=True)
capabilities = _import_router("capabilities", "src.api.v1.capabilities")
analyze = _import_router("analyze", "src.api.v1.analyze")
ocr = _import_router("ocr", "src.api.v1.ocr")
vision = _import_router("vision", "src.api.v1.vision")
drawing = _import_router("drawing", "src.api.v1.drawing")
drift = _import_router("drift", "src.api.v1.drift")
vectors = _import_router("vectors", "src.api.v1.vectors")
process = _import_router("process", "src.api.v1.process")
vectors_stats = _import_router("vectors_stats", "src.api.v1.vectors_stats")
features = _import_router("features", "src.api.v1.features")
model = _import_router("model", "src.api.v1.model")
maintenance = _import_router("maintenance", "src.api.v1.maintenance")
dedup = _import_router("dedup", "src.api.v1.dedup")
feedback = _import_router("feedback", "src.api.v1.feedback")
render = _import_router("render", "src.api.v1.render")
active_learning = _import_router("active_learning", "src.api.v1.active_learning")
compare = _import_router("compare", "src.api.v1.compare")
twin = _import_router("twin", "src.api.v1.twin")
materials = _import_router("materials", "src.api.v1.materials")
tolerance = _import_router("tolerance", "src.api.v1.tolerance")
standards = _import_router("standards", "src.api.v1.standards")
design_standards = _import_router(
    "design_standards", "src.api.v1.design_standards"
)
assistant = _import_router("assistant", "src.api.v1.assistant")
cost = _import_router("cost", "src.api.v1.cost")
diff = _import_router("diff", "src.api.v1.diff")
pointcloud = _import_router("pointcloud", "src.api.v1.pointcloud")

api_router = APIRouter()

# 注册v1版本API
v1_router = APIRouter(prefix="/v1")

# 核心分析模块
# IMPORTANT: drift router must be registered BEFORE analyze router
# because analyze has a catch-all /{analysis_id} route that would match /drift
if drift is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="drift",
        module=drift,
        prefix="/analyze",
        tags=["漂移"],
    )
if analyze is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="analyze",
        module=analyze,
        prefix="/analyze",
        tags=["分析"],
    )
if compare is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="compare",
        module=compare,
        prefix="/compare",
        tags=["对比"],
    )

# 向量相关模块
if vectors is not None:
    # Vector routes under /vectors (resource semantics)
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="vectors",
        module=vectors,
        prefix="/vectors",
        tags=["向量"],
    )
if vectors_stats is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="vectors_stats",
        module=vectors_stats,
        prefix="/vectors_stats",
        tags=["向量统计"],
    )

# 工艺和特征模块
if process is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="process",
        module=process,
        prefix="/analyze",
        tags=["工艺规则"],
    )
if features is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="features",
        module=features,
        prefix="/features",
        tags=["特征"],
    )

# 模型和维护模块
if model is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="model",
        module=model,
        prefix="/model",
        tags=["模型"],
    )
if maintenance is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="maintenance",
        module=maintenance,
        prefix="/maintenance",
        tags=["维护"],
    )

# 健康检查
_include_router(
    v1_router,
    diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
    name="health",
    module=health,
    tags=["健康"],
)
if capabilities is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="capabilities",
        module=capabilities,
        tags=["能力协商"],
    )

# 子路由内部已移除前缀，这里统一加资源前缀
if vision is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="vision",
        module=vision,
        prefix="/vision",
        tags=["视觉"],
    )
if ocr is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="ocr",
        module=ocr,
        prefix="/ocr",
        tags=["OCR"],
    )
if drawing is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="drawing",
        module=drawing,
        prefix="/drawing",
        tags=["Drawing"],
    )
if dedup is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="dedup",
        module=dedup,
        prefix="/dedup",
        tags=["查重"],
    )
if feedback is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="feedback",
        module=feedback,
        prefix="/feedback",
        tags=["反馈"],
    )
if render is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="render",
        module=render,
        prefix="/render",
        tags=["渲染"],
    )
if active_learning is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="active_learning",
        module=active_learning,
        prefix="/active-learning",
        tags=["主动学习"],
    )
if twin is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="twin",
        module=twin,
        prefix="/twin",
        tags=["数字孪生"],
    )
if materials is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="materials",
        module=materials,
        prefix="/materials",
        tags=["材料"],
    )
if tolerance is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="tolerance",
        module=tolerance,
        prefix="/tolerance",
        tags=["公差"],
    )
if standards is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="standards",
        module=standards,
        prefix="/standards",
        tags=["标准库"],
    )
if design_standards is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="design_standards",
        module=design_standards,
        prefix="/design-standards",
        tags=["设计标准"],
    )
if assistant is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="assistant",
        module=assistant,
        prefix="/assistant",
        tags=["智能助手"],
    )

if cost is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="cost",
        module=cost,
        prefix="/cost",
        tags=["成本估算"],
    )
if diff is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="diff",
        module=diff,
        prefix="/diff",
        tags=["图纸对比"],
    )
if pointcloud is not None:
    _include_router(
        v1_router,
        diagnostics=_ROUTER_IMPORT_DIAGNOSTICS,
        name="pointcloud",
        module=pointcloud,
        prefix="/pointcloud",
        tags=["3D点云"],
    )

api_router.include_router(v1_router)
if compare is not None:
    api_router.include_router(
        compare.router, prefix="/compare", tags=["对比"], include_in_schema=False
    )  # type: ignore[attr-defined]

__all__ = ["api_router", "get_router_import_diagnostics"]
