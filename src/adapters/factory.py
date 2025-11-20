from typing import Any, Dict


class AdapterFactory:
    @staticmethod
    def get_adapter(fmt: str):
        """Return a lightweight adapter stub for given format.

        Args:
            fmt: CAD format identifier (e.g., 'dxf', 'step').
        Returns:
            Adapter-like object with async convert(data) -> Dict.
        """

        class Dummy:
            async def convert(self, data: bytes) -> Dict[str, Any]:
                return {
                    "entity_count": 0,
                    "layer_count": 0,
                    "bounding_box": {},
                    "complexity": "low",
                }

        return Dummy()
