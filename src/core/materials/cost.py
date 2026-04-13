"""
材料成本数据与查询

提供材料成本信息、比较和筛选功能。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.core.materials.classify import classify_material_detailed
from src.core.materials.data_models import (
    MATERIAL_DATABASE,
    MaterialGroup,
)

logger = logging.getLogger(__name__)



# 成本等级定义
COST_TIER_DESCRIPTIONS: Dict[int, Dict[str, str]] = {
    1: {"name": "经济型", "description": "成本最低，常规应用", "range": "基准价"},
    2: {"name": "标准型", "description": "性价比高，广泛应用", "range": "1.5-3x 基准"},
    3: {"name": "中高端", "description": "特殊性能，专业应用", "range": "3-8x 基准"},
    4: {"name": "高端型", "description": "高性能材料，精密应用", "range": "8-20x 基准"},
    5: {"name": "特种型", "description": "极端性能，特殊环境", "range": "20x+ 基准"},
}


# 材料成本数据 (grade -> (cost_tier, cost_index))
# cost_tier: 1-5 等级
# cost_index: 相对成本指数，以 Q235B=1.0 为基准
MATERIAL_COST_DATA: Dict[str, Tuple[int, float]] = {
    # 碳素钢 (Tier 1-2)
    "Q235B": (1, 1.0),      # 基准材料
    "Q345R": (1, 1.2),
    "10": (1, 1.0),
    "15": (1, 1.0),
    "20": (1, 1.1),
    "35": (1, 1.2),
    "45": (1, 1.3),
    "50": (1, 1.4),

    # 合金钢 (Tier 2)
    "40Cr": (2, 1.8),
    "42CrMo": (2, 2.2),
    "GCr15": (2, 2.5),
    "GCr15SiMn": (2, 3.0),  # 高淬透性轴承钢
    "GCr18Mo": (3, 5.0),    # 航空级轴承钢
    "20Cr13": (2, 2.0),
    "2Cr13": (2, 2.0),
    "20CrMnTi": (2, 1.8),
    "20Cr": (2, 1.6),
    "38CrMoAl": (2, 2.5),
    "30CrMnSi": (2, 2.2),
    # 弹簧钢 (Tier 2)
    "65Mn": (2, 1.6),
    "60Si2Mn": (2, 1.8),
    "50CrVA": (2, 2.2),
    # 工具钢 (Tier 3)
    "Cr12MoV": (3, 6.0),
    "H13": (3, 5.5),
    "W18Cr4V": (3, 8.0),
    "W6Mo5Cr4V2": (3, 7.0),
    # 特殊钢材 (Tier 2-3)
    "9Cr18": (3, 5.0),      # 高碳马氏体不锈钢/刀具钢
    "12Cr1MoV": (2, 2.8),   # 耐热钢
    "Mn13": (2, 2.5),       # 高锰耐磨钢
    # 精密合金 (Tier 4)
    "4J36": (4, 15.0),      # 因瓦合金
    "4J29": (4, 18.0),      # 可伐合金
    "4J42": (4, 16.0),      # 恒弹性合金
    "1J79": (4, 25.0),      # 坡莫合金
    # 电工钢 (Tier 2-3)
    "50W470": (2, 2.5),     # 无取向硅钢
    "30Q130": (3, 5.0),     # 取向硅钢
    # 焊接材料 (Tier 2)
    "ER308L": (2, 4.0),     # 不锈钢焊丝
    "ER316L": (2, 5.0),     # 含钼不锈钢焊丝
    "ER70S-6": (2, 1.5),    # 碳钢焊丝
    "E7018": (1, 1.2),      # 低氢焊条
    # 复合材料 (Tier 4-5)
    "CFRP": (5, 80.0),      # 碳纤维复合材料
    "GFRP": (3, 8.0),       # 玻璃钢
    # 粉末冶金 (Tier 2)
    "FC-0208": (2, 2.0),    # 烧结铁
    "FN-0205": (2, 3.0),    # 烧结铁镍

    # 耐热钢/高温合金 (Tier 3-5)
    "310S": (3, 5.0),
    "GH3030": (4, 40.0),
    "GH4169": (5, 70.0),
    "GH4099": (5, 85.0),
    "A-286": (4, 30.0),       # 铁镍基高温合金
    "Waspaloy": (5, 85.0),    # 镍基高温合金
    "Rene41": (5, 100.0),     # 高温合金，极贵

    # 高强度结构钢 (Tier 1-2)
    "Q460": (1, 1.5),         # 高强度结构钢
    "Q550": (2, 2.0),         # 高强度结构钢
    "Q690": (2, 2.5),         # 超高强度结构钢

    # 锅炉/压力容器钢 (Tier 1-3)
    "20G": (1, 1.3),          # 锅炉用碳素钢
    "15CrMoG": (2, 2.5),      # 锅炉用合金钢
    "12Cr2Mo1R": (3, 4.0),    # 压力容器用钢

    # 管线钢 (Tier 1-2)
    "X52": (1, 1.4),          # 管线钢
    "X65": (2, 1.8),          # 高强度管线钢
    "X80": (2, 2.2),          # 超高强度管线钢

    # 模具钢补充 (Tier 2-3)
    "DC53": (3, 8.0),         # 冷作模具钢
    "S136": (3, 6.0),         # 塑料模具钢
    "NAK80": (3, 7.0),        # 预硬塑料模具钢

    # 易切削钢 (Tier 1-2)
    "12L14": (2, 1.8),        # 含铅易切削钢
    "Y15": (1, 1.5),          # 硫系易切削钢
    "Y40Mn": (2, 2.0),        # 易切削调质钢

    # 耐磨钢板 (Tier 2-3)
    "NM400": (2, 4.0),        # 耐磨钢板
    "NM500": (3, 5.5),        # 高硬度耐磨钢板
    "Hardox450": (3, 6.5),    # 悍达耐磨钢板

    # 弹簧钢补充 (Tier 2)
    "55CrSi": (2, 2.5),       # 硅铬弹簧钢

    # 低温钢 (Tier 2-4)
    "09MnNiD": (3, 4.5),      # 3.5%镍低温钢
    "16MnDR": (2, 2.8),       # 低温压力容器钢
    "9Ni钢": (4, 15.0),       # 9%镍钢，较贵

    # 电接触材料 (Tier 3-5)
    "AgCdO": (4, 25.0),       # 银氧化镉触点
    "AgSnO2": (4, 28.0),      # 银氧化锡触点，环保
    "CuW70": (4, 35.0),       # 铜钨合金

    # 轴承钢 (Tier 2-3)
    "GCr15": (2, 2.5),        # 高碳铬轴承钢
    "GCr15SiMn": (2, 3.0),    # 高碳铬硅锰轴承钢
    "GCr4": (2, 2.8),         # 渗碳轴承钢

    # 弹簧钢 (Tier 2)
    "60Si2Mn": (2, 2.0),      # 硅锰弹簧钢
    "60Si2CrA": (2, 2.5),     # 硅铬弹簧钢
    "50CrVA": (2, 3.0),       # 铬钒弹簧钢

    # 耐热不锈钢补充 (Tier 2-3)
    "2Cr13": (2, 2.8),        # 马氏体不锈钢
    "1Cr17": (2, 2.5),        # 铁素体不锈钢
    "0Cr25Ni20": (3, 5.0),    # 耐热不锈钢

    # 齿轮钢 (Tier 2)
    "20CrMnTi": (2, 2.2),     # 渗碳齿轮钢
    "20CrMo": (2, 2.5),       # 铬钼渗碳钢
    "20CrNiMo": (2, 3.5),     # 铬镍钼渗碳钢

    # 航空铝合金补充 (Tier 2-3)
    "5A06": (2, 2.2),         # 防锈铝合金
    "2A14": (3, 4.0),         # 高强铝合金
    "7A04": (3, 5.0),         # 超高强铝合金

    # 气门钢 (Tier 3-4)
    "4Cr10Si2Mo": (3, 8.0),   # 马氏体耐热钢
    "5Cr21Mn9Ni4N": (4, 15.0),  # 奥氏体耐热钢
    "4Cr14Ni14W2Mo": (4, 25.0),  # 高温气门钢

    # 链条钢/渗碳硼钢 (Tier 2)
    "20MnVB": (2, 2.0),       # 渗碳硼钢
    "15MnVB": (2, 1.8),       # 低碳硼钢
    "22MnB5": (2, 2.5),       # 热成形钢

    # 电工硅钢补充 (Tier 2-3)
    "B50A600": (2, 2.5),      # 无取向硅钢
    "B35A230": (3, 4.5),      # 高效无取向硅钢
    "B27R090": (3, 6.0),      # 取向硅钢

    # 高温合金补充 (Tier 5 - 极高成本)
    "GH2132": (4, 45.0),      # 铁基高温合金
    "K403": (5, 120.0),       # 镍基铸造高温合金
    "K418": (5, 150.0),       # 高性能铸造高温合金

    # 不锈钢 (Tier 2-3)
    "S30408": (2, 3.5),     # 304
    "S30403": (2, 3.8),     # 304L
    "S31603": (3, 4.5),     # 316L
    "S31608": (3, 4.2),     # 316
    "321": (2, 4.0),        # 321
    "347": (3, 4.5),        # 347
    "430": (2, 2.5),        # 430 铁素体
    "410": (2, 2.8),        # 410 马氏体
    "17-4PH": (3, 8.0),     # 17-4PH 沉淀硬化
    "2205": (3, 6.0),       # 双相
    "2507": (3, 8.0),       # 超级双相
    # 超级奥氏体不锈钢 (Tier 4)
    "904L": (4, 15.0),      # 904L 超级奥氏体
    "254SMO": (4, 18.0),    # 254SMO 6Mo超级奥氏体
    "316Ti": (3, 5.5),      # 316Ti 钛稳定

    # 铸铁 (Tier 1)
    "HT200": (1, 0.8),
    "HT250": (1, 0.9),
    "HT300": (1, 1.0),
    "QT400": (1, 1.0),
    "QT500-7": (1, 1.2),
    "QT600-3": (1, 1.4),
    "QT700-2": (1, 1.6),

    # 铝合金 (Tier 2)
    "6061": (2, 2.8),
    "7075": (2, 4.0),
    "2024": (2, 3.8),
    "5052": (2, 2.5),
    "5083": (2, 3.2),
    "2A12": (2, 3.5),
    "6063": (2, 2.2),
    # 铸造铝合金 (Tier 2)
    "A356": (2, 3.0),       # 铸造铝硅镁合金
    "ZL102": (2, 2.5),      # 铸造铝硅合金
    "ADC12": (2, 2.8),      # 压铸铝合金
    # 铸造铝合金补充 (Tier 2-3)
    "ZL101": (2, 3.2),      # 高纯铸造铝合金
    "ZL104": (2, 2.8),      # 铸造铝硅铜合金
    "ZL201": (3, 5.0),      # 高强度铸造铝铜合金

    # 锌合金 (Tier 1-2)
    "Zamak3": (1, 1.5),     # 通用锌合金
    "Zamak5": (1, 1.8),     # 高强度锌合金
    "ZA-8": (2, 2.2),       # 高铝锌合金

    # 铜合金 (Tier 2-3)
    "H62": (2, 5.0),
    "H68": (2, 5.5),
    "HPb59-1": (2, 5.2),
    "Cu65": (3, 7.0),       # 紫铜
    "QBe2": (4, 25.0),      # 铍铜，贵
    "QAl9-4": (3, 8.0),
    "QAl10-3-1.5": (3, 9.0),
    "QSn4-3": (3, 7.5),
    "QSn6.5-0.1": (3, 8.5),
    "CuNi10Fe1Mn": (3, 10.0),  # 白铜
    # 铜合金补充 (Tier 2-3)
    "QSn7-0.2": (3, 9.0),      # 高锡磷青铜
    "QSn4-0.3": (2, 7.0),      # 低锡磷青铜
    "ZCuSn10P1": (3, 8.0),     # 铸造锡磷青铜
    "HSi80-3": (2, 6.0),       # 硅黄铜
    "HAl77-2": (2, 6.5),       # 铝黄铜

    # 耐磨铸铁 (Tier 2-3)
    "NiHard1": (2, 3.5),       # 镍硬铸铁1型
    "NiHard4": (3, 4.5),       # 镍硬铸铁4型
    "Cr26": (3, 5.0),          # 高铬铸铁

    # 蠕墨铸铁 (Tier 1-2)
    "RuT300": (1, 1.2),        # 蠕墨铸铁
    "RuT350": (2, 1.5),        # 蠕墨铸铁
    "RuT400": (2, 1.8),        # 高强度蠕墨铸铁

    # 可锻铸铁 (Tier 1-2)
    "KTH300-06": (1, 1.3),     # 黑心可锻铸铁
    "KTZ450-06": (2, 1.6),     # 珠光体可锻铸铁
    "KTZ550-04": (2, 2.0),     # 高强度可锻铸铁

    # 铸造镁合金 (Tier 2-3)
    "ZM5": (2, 4.0),           # 压铸镁合金
    "AM60B": (2, 4.5),         # 高韧镁合金
    "AZ63": (3, 5.5),          # 航空镁合金

    # 粉末冶金材料 (Tier 2-3)
    "Fe-Cu-C": (2, 2.5),       # 铁铜碳粉末冶金
    "Fe-Ni-Cu": (2, 3.5),      # 铁镍铜粉末冶金
    "316L-PM": (3, 8.0),       # 不锈钢粉末冶金

    # 硬质合金 (Tier 4-5)
    "YW1": (4, 58.0),          # 通用型硬质合金

    # 结构陶瓷 (Tier 3-4)
    "Al2O3-99": (3, 15.0),     # 99氧化铝陶瓷
    "Si3N4": (4, 80.0),        # 氮化硅陶瓷
    "ZrO2-3Y": (4, 60.0),      # 氧化锆陶瓷

    # 难熔金属 (Tier 4-5)
    "Mo-1": (4, 40.0),         # 纯钼
    "TZM": (5, 80.0),          # 钛锆钼合金
    "W-1": (5, 60.0),          # 纯钨
    "Ta-1": (5, 200.0),        # 纯钽

    # 铝青铜 (Tier 2-3)
    "QAl9-4": (2, 5.0),        # 铝青铜
    "QAl10-4-4": (3, 8.0),     # 镍铝青铜

    # 铍铜 (Tier 3-4)
    "QBe2": (4, 25.0),         # 铍青铜
    "QBe1.9": (4, 22.0),       # 低铍铜
    "CuNi2Si": (3, 12.0),      # 镍硅铜

    # 无铅焊锡 (Tier 2)
    "SAC305": (2, 3.5),        # 无铅焊锡
    "SAC387": (2, 4.5),        # 高银无铅焊锡
    "Sn99.3Cu0.7": (2, 2.5),   # 无银无铅焊锡

    # 钎焊合金 (Tier 3-4)
    "BAg-5": (4, 35.0),        # 银钎料
    "BCu-1": (2, 6.0),         # 纯铜钎料
    "BNi-2": (3, 20.0),        # 镍基钎料

    # 形状记忆合金 (Tier 4-5)
    "NiTi": (5, 150.0),        # 镍钛记忆合金
    "CuZnAl": (3, 15.0),       # 铜锌铝记忆合金
    "CuAlNi": (3, 18.0),       # 铜铝镍记忆合金

    # 电触头材料 (Tier 3-4)
    "AgCdO": (4, 40.0),        # 银氧化镉触点
    "AgSnO2": (4, 45.0),       # 银氧化锡触点
    "CuW": (4, 50.0),          # 钨铜合金

    # 轴承合金 (Tier 2-3)
    "ZChSnSb11-6": (3, 12.0),  # 锡基巴氏合金
    "ZChPbSb16-16-2": (2, 5.0), # 铅基巴氏合金
    "CuPb24Sn4": (2, 8.0),     # 铜铅合金轴承

    # 热电偶合金 (Tier 3-4)
    "Chromel": (3, 20.0),      # 镍铬合金
    "Alumel": (3, 18.0),       # 镍铝合金
    "Constantan": (3, 15.0),   # 康铜

    # 永磁材料 (Tier 4-5)
    "NdFeB": (4, 80.0),        # 钕铁硼永磁
    "SmCo": (5, 200.0),        # 钐钴永磁
    "Alnico": (3, 30.0),       # 铝镍钴永磁

    # 电阻合金 (Tier 2-3)
    "Cr20Ni80": (3, 15.0),     # 镍铬电热合金
    "Manganin": (3, 25.0),     # 锰铜合金
    "Karma": (4, 40.0),        # 卡玛合金

    # 低膨胀合金 (Tier 3-4)
    "Invar": (3, 25.0),        # 因瓦合金
    "Kovar": (4, 35.0),        # 可伐合金
    "4J32": (4, 45.0),         # 超因瓦合金

    # 超导材料 (Tier 5)
    "NbTi": (5, 150.0),        # 铌钛超导合金
    "Nb3Sn": (5, 250.0),       # 铌三锡超导体
    "YBCO": (5, 500.0),        # 钇钡铜氧高温超导

    # 核工业材料 (Tier 4-5)
    "Zircaloy-4": (4, 80.0),   # 锆合金-4
    "Hafnium": (5, 800.0),     # 铪（稀有金属）
    "B4C": (4, 60.0),          # 碳化硼

    # 医用合金 (Tier 4-5)
    "CoCrMo": (4, 55.0),       # 钴铬钼医用合金
    "Ti6Al4V-ELI": (5, 85.0),  # 医用钛合金
    "316L-Medical": (4, 25.0), # 医用不锈钢

    # 光学材料 (Tier 4-5)
    "Fused-Silica": (4, 50.0),      # 熔融石英
    "Sapphire": (5, 200.0),         # 蓝宝石
    "Germanium": (5, 350.0),        # 锗

    # 电池材料 (Tier 3-4)
    "LiFePO4": (3, 15.0),           # 磷酸铁锂
    "NMC": (4, 25.0),               # 三元正极材料
    "Graphite-Battery": (3, 8.0),  # 电池负极石墨

    # 半导体材料 (Tier 4-5)
    "Silicon-Wafer": (4, 50.0),       # 硅晶圆
    "GaAs": (5, 300.0),               # 砷化镓
    "SiC-Semiconductor": (5, 500.0),  # 碳化硅半导体

    # 热界面材料 (Tier 2-4)
    "Thermal-Paste": (2, 5.0),        # 导热硅脂
    "Thermal-Pad": (3, 15.0),         # 导热垫片
    "Graphene-TIM": (4, 80.0),        # 石墨烯导热材料

    # 增材制造材料 (Tier 4-5)
    "AlSi10Mg-AM": (4, 60.0),         # 3D打印铝合金
    "IN718-AM": (5, 150.0),           # 3D打印镍基高温合金
    "Ti64-AM": (5, 200.0),            # 3D打印钛合金

    # 硬质合金 (Tier 4-5)
    "WC-Co": (4, 80.0),               # 钨钴硬质合金
    "Stellite": (4, 65.0),            # 司太立合金
    "CBN": (5, 500.0),                # 立方氮化硼

    # 热障涂层材料 (Tier 4-5)
    "YSZ": (4, 45.0),                 # 氧化钇稳定氧化锆
    "Al2O3-TBC": (3, 25.0),           # 氧化铝热障涂层
    "MCrAlY": (4, 70.0),              # 金属粘结层

    # 电磁屏蔽材料 (Tier 3-4)
    "Mu-Metal": (4, 50.0),            # 坡莫合金
    "Permalloy": (3, 35.0),           # 铁镍软磁合金
    "Copper-Mesh": (3, 20.0),         # 铜网屏蔽材料

    # 钛合金 (Tier 4)
    "TA1": (4, 12.0),
    "TA2": (4, 15.0),
    "TC4": (4, 25.0),
    "TC11": (4, 35.0),
    "TB6": (4, 40.0),
    "TC21": (4, 38.0),

    # 镁合金 (Tier 2-3)
    "AZ31B": (2, 3.0),
    "AZ91D": (2, 2.5),
    "ZK60": (3, 5.0),

    # 硬质合金 (Tier 4)
    "YG8": (4, 50.0),
    "YT15": (4, 55.0),
    "YG6": (4, 52.0),

    # 耐蚀合金 (Tier 4-5)
    "C276": (5, 80.0),
    "C22": (5, 75.0),
    "Inconel625": (5, 60.0),
    "Inconel718": (5, 70.0),
    "Monel400": (4, 35.0),
    "MonelK500": (4, 45.0),
    "HastelloyB3": (5, 90.0),
    "Stellite6": (5, 100.0),
    "Incoloy825": (4, 40.0),

    # 工程塑料 (Tier 1-3)
    "PTFE": (2, 3.0),
    "RPTFE": (2, 3.5),
    "PEEK": (4, 50.0),      # PEEK 很贵
    "POM": (2, 2.5),
    "PA66": (2, 2.0),
    "PC": (2, 2.2),
    "UHMWPE": (2, 3.0),
    # 高性能工程塑料 (Tier 3-5)
    "PPS": (3, 8.0),
    "PI": (5, 80.0),        # 聚酰亚胺很贵
    "PSU": (3, 12.0),
    "PEI": (4, 25.0),

    # 橡胶 (Tier 1-2)
    "EPDM": (1, 1.5),
    "VMQ": (2, 4.0),

    # 聚氨酯 (Tier 2)
    "聚氨酯": (2, 3.0),

    # 玻璃 (Tier 2)
    "硼硅玻璃": (2, 2.5),
    "钢化玻璃": (2, 1.8),

    # 陶瓷/纤维 (Tier 2)
    "硅酸铝": (2, 2.0),

    # 新增材料成本
    # 不锈钢
    "631": (3, 8.0),           # 17-7PH沉淀硬化
    "15-5PH": (3, 7.0),        # 15-5PH沉淀硬化
    "Nitronic50": (3, 10.0),   # 高氮奥氏体
    # 铝合金
    "3003": (1, 1.2),          # 防锈铝
    "1060": (1, 1.0),          # 工业纯铝
    "LY12": (2, 2.0),          # 硬铝
    "6005": (2, 1.8),          # 轨道交通铝合金
    "6101": (2, 1.6),          # 导电铝合金
    # 工具钢
    "D2": (3, 5.5),            # SKD11
    "O1": (2, 3.5),            # 油淬工具钢
    "A2": (3, 5.0),            # 空淬工具钢
    # 高温合金
    "Haynes230": (5, 120.0),   # 镍基高温合金
    "HastelloyX": (5, 100.0),  # 哈氏合金X
    "MP35N": (5, 200.0),       # 超高强度医用合金
    "L605": (5, 150.0),        # 钴基高温合金
    "Waspaloy": (5, 130.0),    # 镍基时效高温合金
    "Rene41": (5, 140.0),      # 镍基高温合金
    "Elgiloy": (5, 180.0),     # 钴铬镍钼合金
    "Phynox": (5, 190.0),      # 医用钴基合金
    # 钛合金
    "Ti-6242": (4, 60.0),      # 近α型钛合金
    # 铜合金
    "C11000": (2, 3.0),        # 电解铜
    "C26000": (2, 2.5),        # 黄铜
    "C52100": (3, 8.0),        # 磷青铜
    # 镁合金
    "WE43": (4, 40.0),         # 稀土镁合金
    "AM60": (2, 3.5),          # 压铸镁合金
    # 不锈钢
    "13-8Mo": (4, 20.0),       # 沉淀硬化不锈钢
    "22Cr双相钢": (3, 8.0),    # 双相不锈钢
    # 工程塑料
    "PVDF": (3, 15.0),         # 聚偏氟乙烯
    "LCP": (4, 25.0),          # 液晶聚合物
    "PPSU": (4, 22.0),         # 聚苯砜
    # 橡胶
    "NBR": (1, 1.5),           # 丁腈橡胶
    "FKM": (3, 12.0),          # 氟橡胶

    # 组合件 - 无固定成本
    # "组焊件" 和 "组合件" 成本取决于组成部件，不列入固定成本表
}


def get_material_cost(grade: str) -> Optional[Dict[str, Any]]:
    """
    获取材料成本信息

    Args:
        grade: 材料牌号

    Returns:
        成本信息字典，包含:
        - tier: 成本等级 (1-5)
        - tier_name: 等级名称
        - tier_description: 等级描述
        - cost_index: 相对成本指数
        - price_range: 价格区间描述
    """
    # 先尝试获取材料信息
    info = classify_material_detailed(grade)
    if not info:
        return None

    # 使用规范牌号
    grade = info.grade

    # 查找成本数据
    cost_data = MATERIAL_COST_DATA.get(grade)
    if not cost_data:
        # 尝试根据材料组估算
        group_defaults = {
            MaterialGroup.CARBON_STEEL: (1, 1.2),
            MaterialGroup.ALLOY_STEEL: (2, 2.0),
            MaterialGroup.STAINLESS_STEEL: (2, 4.0),
            MaterialGroup.CORROSION_RESISTANT: (4, 50.0),
            MaterialGroup.CAST_IRON: (1, 0.9),
            MaterialGroup.ALUMINUM: (2, 3.0),
            MaterialGroup.COPPER: (2, 6.0),
            MaterialGroup.TITANIUM: (4, 20.0),
            MaterialGroup.NICKEL: (4, 40.0),
            MaterialGroup.ENGINEERING_PLASTIC: (2, 3.0),
            MaterialGroup.FLUOROPOLYMER: (2, 3.5),
            MaterialGroup.RUBBER: (1, 2.0),
        }
        cost_data = group_defaults.get(info.group, (2, 5.0))

    tier, cost_index = cost_data
    tier_info = COST_TIER_DESCRIPTIONS.get(tier, {})

    return {
        "grade": grade,
        "name": info.name,
        "tier": tier,
        "tier_name": tier_info.get("name", "未知"),
        "tier_description": tier_info.get("description", ""),
        "cost_index": cost_index,
        "price_range": tier_info.get("range", ""),
        "group": info.group.value,
    }


def compare_material_costs(
    grades: List[str],
    include_missing: bool = False,
) -> Any:
    """
    比较多个材料的成本

    Args:
        grades: 材料牌号列表
        include_missing: 是否返回未命中的材料列表

    Returns:
        成本比较结果，按成本指数排序；
        include_missing=True 时返回 (results, missing)
    """
    results = []
    missing = []

    for grade in grades:
        cost_info = get_material_cost(grade)
        if cost_info:
            results.append(cost_info)
        else:
            missing.append(grade)

    # 按成本指数排序
    results.sort(key=lambda x: float(x.get("cost_index") or 0.0))

    # 添加相对比较
    if results:
        min_cost = results[0]["cost_index"]
        for r in results:
            r["relative_to_cheapest"] = round(r["cost_index"] / min_cost, 2)

    if include_missing:
        return results, missing
    return results


def search_by_cost(
    max_tier: Optional[int] = None,
    max_cost_index: Optional[float] = None,
    category: Optional[str] = None,
    group: Optional[str] = None,
    limit: int = 20,
    include_estimated: bool = False,
) -> List[Dict[str, Any]]:
    """
    按成本筛选材料

    Args:
        max_tier: 最大成本等级 (1-5)
        max_cost_index: 最大成本指数
        category: 限定材料类别
        group: 限定材料组
        limit: 返回数量限制

    Returns:
        符合条件的材料列表
    """
    results = []

    for grade, info in MATERIAL_DATABASE.items():
        # 类别过滤
        if category and info.category.value != category:
            continue
        if group and info.group.value != group:
            continue

        tier = None
        cost_index = None
        tier_name = "未知"

        if include_estimated:
            cost_info = get_material_cost(grade)
            if not cost_info:
                continue
            tier = cost_info["tier"]
            cost_index = cost_info["cost_index"]
            tier_name = cost_info["tier_name"]
        else:
            cost_data = MATERIAL_COST_DATA.get(grade)
            if not cost_data:
                continue
            tier, cost_index = cost_data
            tier_info = COST_TIER_DESCRIPTIONS.get(tier, {})
            tier_name = tier_info.get("name", "未知")

        # 成本等级过滤
        if max_tier and tier and tier > max_tier:
            continue

        # 成本指数过滤
        if max_cost_index and cost_index and cost_index > max_cost_index:
            continue

        results.append({
            "grade": grade,
            "name": info.name,
            "category": info.category.value,
            "group": info.group.value,
            "tier": tier,
            "tier_name": tier_name,
            "cost_index": cost_index,
        })

    # 按成本指数排序
    results.sort(key=lambda x: float(x.get("cost_index") or 0.0))

    return results[:limit]


def get_cost_tier_info() -> List[Dict[str, Any]]:
    """
    获取成本等级定义

    Returns:
        成本等级列表
    """
    return [
        {
            "tier": tier,
            "name": info["name"],
            "description": info["description"],
            "price_range": info["range"],
        }
        for tier, info in COST_TIER_DESCRIPTIONS.items()
    ]
