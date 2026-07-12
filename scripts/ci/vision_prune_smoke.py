#!/usr/bin/env python3
"""Runtime import smoke for the Phase 0 A2b vision prune.

py_compile / AST checks are NOT sufficient here — a prior slice shipped a
mis-delete that only surfaced as a runtime ModuleNotFoundError. This script
actually imports and calls the live entry points into src/core/vision so a
wrong deletion fails CI, not just a static scan.

Covers exactly the transitive-import closure the A2b prune is defended by:
  - src/api/v1/vision.py's named imports from src.core.vision
  - src/core/providers/vision.py (the core-provider bridge)
  - create_vision_provider("stub") actually instantiating a provider
"""

from __future__ import annotations

import sys


def main() -> int:
    import src.core.vision  # noqa: F401

    from src.core.vision import (  # noqa: F401
        ResilientVisionProvider,
        VisionAnalyzeRequest,
        VisionAnalyzeResponse,
        VisionInputError,
        VisionManager,
        VisionProviderError,
        create_vision_provider,
        get_available_providers,
    )

    import src.core.providers.vision  # noqa: F401

    provider = create_vision_provider("stub")
    if provider.provider_name not in ("stub", "deepseek_stub"):
        print(f"::error::unexpected stub provider_name: {provider.provider_name!r}")
        return 1

    providers = get_available_providers()
    if "stub" not in providers:
        print("::error::get_available_providers() missing 'stub' entry")
        return 1

    print(f"vision-prune-smoke: OK (provider={provider.provider_name!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
