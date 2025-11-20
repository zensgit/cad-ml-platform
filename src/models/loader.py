"""Model loader stubs to satisfy imports during OCR scaffold phase."""
_loaded = True  # Assume loaded for now


async def load_models() -> None:
    global _loaded
    _loaded = True


def models_loaded() -> bool:
    return _loaded
