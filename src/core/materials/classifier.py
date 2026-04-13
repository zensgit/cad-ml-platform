"""
材料分类与属性系统 -- 向后兼容包装模块

此文件保留以确保所有通过 ``from src.core.materials.classifier import ...``
进行导入的代码继续正常工作。实际实现已拆分到以下子模块：

- data_models.py  -- 枚举、数据类与数据库
- classify.py     -- 分类与搜索函数
- properties.py   -- 属性查询函数
- processing.py   -- 工艺推荐函数
- equivalence.py  -- 材料等价表函数
- cost.py         -- 成本查询函数
- compatibility.py -- 兼容性检查函数
- export.py       -- 数据导出函数
"""

# Re-export everything so ``from src.core.materials.classifier import X``
# keeps working.

from src.core.materials.data_models import (  # noqa: F401
    MATERIAL_DATABASE,
    MATERIAL_EQUIVALENCE,
    MATERIAL_MATCH_PATTERNS,
    MaterialCategory,
    MaterialGroup,
    MaterialInfo,
    MaterialProperties,
    MaterialSubCategory,
    ProcessRecommendation,
)

from src.core.materials.classify import (  # noqa: F401
    PINYIN_MAP,
    classify_material_detailed,
    classify_material_simple,
    search_materials,
    _calculate_similarity,
)

from src.core.materials.properties import (  # noqa: F401
    get_material_info,
    search_by_properties,
)

from src.core.materials.processing import (  # noqa: F401
    APPLICATION_MAP,
    MATERIAL_ALTERNATIVES,
    get_alternative_materials,
    get_material_recommendations,
    get_process_recommendations,
    list_applications,
)

from src.core.materials.equivalence import (  # noqa: F401
    find_equivalent_material,
    get_material_equivalence,
    list_material_standards,
)

from src.core.materials.cost import (  # noqa: F401
    COST_TIER_DESCRIPTIONS,
    MATERIAL_COST_DATA,
    compare_material_costs,
    get_cost_tier_info,
    get_material_cost,
    search_by_cost,
)

from src.core.materials.compatibility import (  # noqa: F401
    GALVANIC_RISK_THRESHOLDS,
    GALVANIC_SERIES,
    WELD_COMPATIBILITY,
    check_full_compatibility,
    check_galvanic_corrosion,
    check_heat_treatment_compatibility,
    check_weld_compatibility,
)

from src.core.materials.export import (  # noqa: F401
    export_equivalence_csv,
    export_materials_csv,
)
