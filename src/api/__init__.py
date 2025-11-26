"""API路由聚合"""
from fastapi import APIRouter

# Import critical routers first (health). Other routers are imported best-effort
from src.api.v1 import health
try:
    from src.api.v1 import analyze  # type: ignore
except Exception:
    analyze = None  # type: ignore
try:
    from src.api.v1 import ocr, vision  # type: ignore
except Exception:
    ocr = None  # type: ignore
    vision = None  # type: ignore
try:
    from src.api.v1 import drift  # type: ignore
except Exception:
    drift = None  # type: ignore
try:
    from src.api.v1 import vectors  # type: ignore
except Exception:
    vectors = None  # type: ignore
try:
    from src.api.v1 import process  # type: ignore
except Exception:
    process = None  # type: ignore
try:
    from src.api.v1 import vectors_stats  # type: ignore
except Exception:
    vectors_stats = None  # type: ignore
try:
    from src.api.v1 import features  # type: ignore
except Exception:
    features = None  # type: ignore
try:
    from src.api.v1 import model  # type: ignore
except Exception:
    model = None  # type: ignore
try:
    from src.api.v1 import maintenance  # type: ignore
except Exception:
    maintenance = None  # type: ignore

api_router = APIRouter()

# 注册v1版本API
v1_router = APIRouter(prefix="/v1")

# 核心分析模块
if analyze is not None:
    v1_router.include_router(analyze.router, prefix="/analyze", tags=["分析"])  # type: ignore
if drift is not None:
    v1_router.include_router(drift.router, prefix="/analyze", tags=["漂移"])  # type: ignore

# 向量相关模块
if vectors is not None:
    # Vector routes under /vectors (resource semantics)
    v1_router.include_router(vectors.router, prefix="/vectors", tags=["向量"])  # type: ignore
if vectors_stats is not None:
    v1_router.include_router(vectors_stats.router, prefix="/analyze", tags=["向量统计"])  # type: ignore

# 工艺和特征模块
if process is not None:
    v1_router.include_router(process.router, prefix="/analyze", tags=["工艺规则"])  # type: ignore
if features is not None:
    v1_router.include_router(features.router, prefix="/features", tags=["特征"])  # type: ignore

# 模型和维护模块
if model is not None:
    v1_router.include_router(model.router, prefix="/model", tags=["模型"])  # type: ignore
if maintenance is not None:
    v1_router.include_router(maintenance.router, prefix="/maintenance", tags=["维护"])  # type: ignore

# 健康检查
v1_router.include_router(health.router, tags=["健康"])

# 子路由内部已移除前缀，这里统一加资源前缀
if vision is not None:
    v1_router.include_router(vision.router, prefix="/vision", tags=["视觉"])  # type: ignore
if ocr is not None:
    v1_router.include_router(ocr.router, prefix="/ocr", tags=["OCR"])  # type: ignore

api_router.include_router(v1_router)

__all__ = ["api_router"]
