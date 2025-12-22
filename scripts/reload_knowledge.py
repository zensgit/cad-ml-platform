#!/usr/bin/env python3
"""
Reload dynamic knowledge rules from disk.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.knowledge.dynamic.manager import get_knowledge_manager


def main() -> None:
    km = get_knowledge_manager()
    prev = km.get_version()
    km.reload()
    curr = km.get_version()
    changed = prev != curr
    print(f"knowledge_version_prev={prev}")
    print(f"knowledge_version_curr={curr}")
    print(f"knowledge_changed={changed}")


if __name__ == "__main__":
    main()
