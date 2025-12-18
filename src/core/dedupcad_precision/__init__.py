"""DedupCAD precision (v2 JSON) verification.

This package vendors the core geometric/semantic JSON similarity logic from the
`dedupcad` repository so `cad-ml-platform` can run L4 precision checks after
`dedupcad-vision` returns visual candidates.
"""

from .store import GeomJsonStore, GeomJsonStoreConfig
from .verifier import PrecisionVerifier

__all__ = [
    "GeomJsonStore",
    "GeomJsonStoreConfig",
    "PrecisionVerifier",
]

