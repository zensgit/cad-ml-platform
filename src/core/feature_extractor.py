from typing import Any, Dict, List


class FeatureExtractor:
    async def extract(self, data: bytes) -> Dict[str, List[Any]]:
        return {'geometric': [1, 2], 'semantic': [3]}
