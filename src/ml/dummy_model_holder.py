from typing import List


class DummyModel:
    def predict(self, xs: List[List[float]]) -> List[str]:
        return ["x" for _ in xs]


__all__ = ["DummyModel"]
