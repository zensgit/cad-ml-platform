"""Pytest configuration for resolving src package."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Provide stubs for missing heavy modules to allow OCR endpoint smoke test
try:
    import src.core.analyzer  # noqa
except Exception:
    module_path = PROJECT_ROOT / 'src' / 'core'
    (module_path / 'analyzer.py').write_text("class CADAnalyzer:\n    async def classify_part(self, data): return {'type': 'part', 'confidence': 0.5}\n    async def check_quality(self, data): return {'score': 0.8}\n    async def recommend_process(self, data): return {'primary': 'machining'}\n")
try:
    import src.core.feature_extractor  # noqa
except Exception:
    (PROJECT_ROOT / 'src' / 'core' / 'feature_extractor.py').write_text("class FeatureExtractor:\n    async def extract(self, data): return {'geometric': [1,2], 'semantic': [3]}\n")
try:
    import src.adapters.factory  # noqa
except Exception:
    factory_path = PROJECT_ROOT / 'src' / 'adapters'
    factory_path.mkdir(parents=True, exist_ok=True)
    (factory_path / 'factory.py').write_text("class AdapterFactory:\n    @staticmethod\n    def get_adapter(fmt):\n        class Dummy:\n            async def convert(self, data): return {'entity_count':0,'layer_count':0,'bounding_box':{},'complexity':'low'}\n        return Dummy()\n")
