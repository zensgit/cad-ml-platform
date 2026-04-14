"""DXF text-content keyword classifier for B5.1 text-fusion.

Uses a curated keyword dictionary for each of the 24 taxonomy classes.
Returns a soft probability vector (softmax-normalised keyword hit scores).
Returns an empty dict when no keywords match (no signal contributed to fusion).

Design principles:
  - Class-specific keywords take precedence over generic engineering terms
  - No keyword match → empty result (classifier abstains, no noise)
  - All matching is case-insensitive substring search
  - Hit score = hits / keyword_count (normalised per class, then softmax)
"""

from __future__ import annotations

import math
from typing import Optional


class TextContentClassifier:
    """Classify DXF content by keyword matching against a 24-class dictionary.

    Usage::

        clf = TextContentClassifier()
        probs = clf.predict_probs("技术要求 轴承座孔 铸件不得有气孔")
        # {'轴承座': 0.62, '箱体': 0.18, ...}  or  {} if no match
    """

    # 24-class keyword dictionary.
    # Keywords are ordered by discriminative power (most specific first).
    # Chinese engineering terms sourced from actual DXF samples.
    KEYWORDS: dict[str, list[str]] = {
        # High-precision keywords: specific to each class, rarely cross-contaminate
        "法兰": [
            # 原有（高区分度）
            "对焊法兰", "平焊法兰", "螺纹法兰", "法兰盘",
            "喷涂Halar", "NB/T47010", "flange", "法兰密封面",
            # B5.4c 新增：标准号变体（DXF 实测常见）
            "NB/T 47010", "NB/T 47023", "NB/T 47044",
            "GB/T 9119", "GB/T 9115", "HG/T 20592", "HG/T 20615",
            # 密封面类型
            "RF面", "FF面", "突面法兰", "全平面法兰", "凹凸面",
            # 规格特征（DN/PN 常见于法兰技术要求）
            "PN16", "PN25", "PN40", "法兰厚度", "螺栓孔圆",
        ],
        "轴类": [
            # 原有
            "花键轴", "阶梯轴", "齿轮轴", "心轴", "传动轴",
            "shaft", "spindle", "键槽", "退刀槽",
            # B5.4c 新增：轴类特有加工特征
            "中心孔", "外圆磨", "轴颈", "轴肩", "轴端",
            "平键", "半圆键", "矩形花键", "渐开线花键",
            "轴承配合", "跑合面", "调质处理",
        ],
        "箱体": [
            # 原有
            "箱体", "减速箱", "齿轮箱", "housing",
            "电加热箱", "保温棉", "机壳", "机座",
            # B5.4c 新增：箱体特有装配特征（不含轴承座共享词）
            "箱盖", "箱座", "端盖螺栓孔",
            "窥视孔", "放油孔", "通气孔", "迷宫密封",
            "轴端密封", "铸造圆角R", "非加工面涂漆",
        ],
        "轴承座": [
            "轴承座", "轴承支座", "轴承支架", "轴承孔",
            "铸件不得有砂眼", "未注铸造圆角", "轴承室", "轴承盖",
        ],
        "传动件": [
            "齿轮", "皮带轮", "链轮", "联轴器", "蜗轮",
            "蜗杆", "transmission", "gear", "sprocket",
        ],
        "阀门": [
            "阀门", "球阀", "蝶阀", "闸阀", "截止阀", "阀体",
            "valve", "与铰链配做", "阀杆", "阀盖",
        ],
        "罐体": [
            "罐体", "储罐", "贮罐", "卧式罐", "立式罐",
            "TSG 21", "GB/T 150", "压力容器安全技术", "管口表",
        ],
        "过滤器": [
            "过滤器", "滤芯", "过滤精度", "filter", "滤网",
            "过滤元件", "袋式过滤", "篮式过滤",
        ],
        "换热器": [
            "换热器", "管板", "折流板", "换热管", "heat exchanger",
            "壳程", "管程", "列管", "浮头", "GB/T 13296",
        ],
        "泵": [
            "泵体", "叶轮", "泵壳", "pump", "离心泵", "螺杆泵",
            "齿轮泵", "泵盖", "泵座",
        ],
        "搅拌器": [
            "搅拌桨", "agitator", "mixer", "搅拌轴",
            "桨叶", "推进式", "搅拌器", "叶片承压板",
        ],
        "弹簧": [
            "弹簧", "碟簧", "spring", "压缩弹簧",
            "拉伸弹簧", "扭转弹簧", "弹性模量",
        ],
        "分离器": [
            "分离器", "旋风分离", "沉降分离", "separator",
            "气液分离", "油水分离", "除雾器", "旋风器",
        ],
        "筒体": [
            "筒体", "筒节", "纵焊缝", "环焊缝",
            "卷板", "圆筒体", "筒壳",
        ],
        "封头": [
            "封头", "椭圆封头", "半球封头", "锥形封头",
            "平封头", "蝶形封头", "封头厚度",
        ],
        "支架": [
            "支架", "鞍座", "地脚", "托架",
            "bracket", "支撑架", "底脚螺栓",
        ],
        "盖罩": [
            "端盖", "防护罩", "压盖", "密封盖",
            "轴承盖板", "观察盖", "检查盖",
        ],
        "液压组件": [
            "液压缸", "活塞", "hydraulic", "油缸",
            "液压站", "油压缸", "液压阀", "柱塞",
        ],
        "板类": [
            "隔板", "挡板", "衬板", "折流板垫片",
            "δ=", "钢板厚", "平板δ",
        ],
        "旋转组件": [
            "转子", "rotor", "转鼓", "旋转体",
            "飞轮", "转盘组件", "回转台",
        ],
        "锥体": [
            "锥体", "锥管", "大小头", "锥台",
            "变径管", "渐扩管", "锥形筒",
        ],
        "紧固件": [
            "螺栓组件", "双头螺柱", "六角螺母",
            "GB145", "中心孔A", "表面镀铬", "镀锌",
        ],
        "进出料装置": [
            "进料口", "出料口", "加料口", "下料口",
            "进料管", "排料管", "inlet", "outlet",
        ],
        "人孔": [
            "人孔", "手孔", "manhole", "人孔盖",
            "人孔法兰", "清扫孔", "检查孔盖",
        ],
    }

    # B5.7b: Co-occurrence keyword groups (conditional matching).
    # These require at least `min_hits` out of the group keywords to fire.
    # This targets classes where single-keyword matching is unreliable
    # (e.g. 法兰 rarely contains "法兰" but often has standard-number + RF面 combo).
    COOCCURRENCE: dict[str, dict] = {
        "法兰": {
            "min_hits": 2,
            "keywords": [
                "密封面粗糙度", "RF面", "FF面", "突面", "凹凸面",
                "螺栓孔圆", "法兰厚度", "PN16", "PN25", "PN40",
                "NB/T", "HG/T", "GB/T 9119", "GB/T 9115",
            ],
            "bonus": 0.15,  # Added to raw_scores when co-occurrence fires
        },
        "箱体": {
            "min_hits": 2,
            "keywords": [
                "箱盖", "箱座", "窥视孔", "放油孔", "通气孔",
                "非加工面涂漆", "迷宫密封",
            ],
            "bonus": 0.12,
        },
    }

    # Minimum text length to attempt classification
    MIN_TEXT_LEN: int = 4

    # B5.6b: Minimum margin between top-1 and top-2 to return a prediction.
    # When margin < threshold, the classifier abstains — avoids ambiguous
    # text signals dragging down Graph2D accuracy in the fusion.
    MIN_MARGIN: float = 0.30

    def predict_probs(self, text: str) -> dict[str, float]:
        """Return softmax-normalised class scores from keyword matching.

        Args:
            text: Plain text extracted from a DXF file.

        Returns:
            Dict mapping class name → probability. Empty dict if no keywords
            matched or if the margin between top-1 and top-2 is too low
            (classifier abstains — no contribution to fusion score).
        """
        if not text or len(text.strip()) < self.MIN_TEXT_LEN:
            return {}

        text_lower = text.lower()
        raw_scores: dict[str, float] = {}

        for cls, keywords in self.KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw.lower() in text_lower)
            if hits > 0:
                # Weight by hit density (hits per keyword count)
                raw_scores[cls] = hits / len(keywords)

        # B5.7b: Co-occurrence bonus — when multiple condition keywords
        # co-appear, add a score bonus to the target class.
        for cls, cfg in self.COOCCURRENCE.items():
            co_hits = sum(1 for kw in cfg["keywords"] if kw.lower() in text_lower)
            if co_hits >= cfg["min_hits"]:
                bonus = cfg["bonus"] * (co_hits / len(cfg["keywords"]))
                raw_scores[cls] = raw_scores.get(cls, 0.0) + bonus

        if not raw_scores:
            return {}

        # Softmax normalisation
        max_score = max(raw_scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in raw_scores.items()}
        total = sum(exp_scores.values())
        probs = {k: v / total for k, v in exp_scores.items()}

        # B5.6b: Abstain when top-1 vs top-2 margin is too low.
        # This prevents ambiguous keyword matches (e.g. 箱体 vs 轴承座
        # sharing generic terms) from injecting noise into the fusion.
        if len(probs) >= 2:
            sorted_probs = sorted(probs.values(), reverse=True)
            if sorted_probs[0] - sorted_probs[1] < self.MIN_MARGIN:
                return {}

        return probs

    def top_class(self, text: str) -> Optional[str]:
        """Return the top predicted class, or None if no keywords matched."""
        probs = self.predict_probs(text)
        if not probs:
            return None
        return max(probs, key=probs.get)

    def top_classes(self, text: str, n: int = 3) -> list[tuple[str, float]]:
        """Return top-n (class, prob) pairs, sorted by probability descending."""
        probs = self.predict_probs(text)
        return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:n]
