"""Drawing version diff package.

Provides geometry and annotation comparison between DXF drawing revisions,
along with structured reporting (Markdown, ECN).
"""

from src.core.diff.annotation_diff import AnnotationDiff
from src.core.diff.geometry_diff import GeometryDiff
from src.core.diff.models import DiffReport, DiffResult
from src.core.diff.report import DiffReportGenerator

__all__ = [
    "GeometryDiff",
    "AnnotationDiff",
    "DiffResult",
    "DiffReport",
    "DiffReportGenerator",
]
