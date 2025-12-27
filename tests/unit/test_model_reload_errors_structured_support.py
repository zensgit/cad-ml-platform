"""Support objects for model reload structured error tests.

Top-level classes/functions so they are pickleable in tests.
"""


class DummyModel:
    def predict(self, xs):  # pragma: no cover - trivial
        return ["A"] * len(xs)


def dummy_function():  # pragma: no cover - used to trigger GLOBAL opcode
    return 1


class GoodModel:
    def predict(self, xs):  # pragma: no cover - trivial
        return [0] * len(xs)


class VersionedModel:
    """A model with version metadata for version mismatch tests."""

    __version__ = "v2.0"

    def predict(self, xs):  # pragma: no cover - trivial
        return ["test"] * len(xs)
