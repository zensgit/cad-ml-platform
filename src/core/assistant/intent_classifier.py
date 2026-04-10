"""
Trained intent classifier for the CAD-ML Assistant.

Replaces simple regex-based query analysis with a TF-IDF + LogisticRegression
multi-class classifier.  Falls back to a keyword-based heuristic when
scikit-learn is not available.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Trained intent classifier replacing regex-based query_analyzer.

    Uses TF-IDF + LogisticRegression for multi-class intent classification.
    Falls back to regex patterns if sklearn is unavailable.
    """

    # ------------------------------------------------------------------
    # Training data -- at least 60 real Chinese manufacturing queries
    # ------------------------------------------------------------------

    TRAINING_DATA: List[Tuple[str, str]] = [
        # ---- Material property (10) ----------------------------------
        ("SUS304的密度是多少", "material_property"),
        ("铝合金6061的强度", "material_property"),
        ("Q235的硬度", "material_property"),
        ("钛合金的价格", "material_property"),
        ("不锈钢的耐腐蚀性", "material_property"),
        ("45号钢的抗拉强度是多少", "material_property"),
        ("黄铜H62的导电性", "material_property"),
        ("PA66尼龙的密度", "material_property"),
        ("7075铝合金的屈服强度", "material_property"),
        ("40Cr的热处理硬度", "material_property"),
        # ---- Material comparison (5) ---------------------------------
        ("碳钢和不锈钢哪个好", "material_comparison"),
        ("6061和7075铝合金对比", "material_comparison"),
        ("Q235和45钢的区别", "material_comparison"),
        ("304和316不锈钢哪个耐腐蚀", "material_comparison"),
        ("钛合金和铝合金哪个轻", "material_comparison"),
        # ---- Material selection (5) ----------------------------------
        ("选什么材料做法兰盘", "material_selection"),
        ("轴承用什么材料好", "material_selection"),
        ("耐高温用什么材料", "material_selection"),
        ("海洋环境用什么钢", "material_selection"),
        ("齿轮推荐什么材质", "material_selection"),
        # ---- Process route (8) ---------------------------------------
        ("法兰盘怎么加工", "process_route"),
        ("这个零件用什么工艺", "process_route"),
        ("钛合金能线切割吗", "process_route"),
        ("壳体零件加工流程", "process_route"),
        ("铝合金适合CNC加工吗", "process_route"),
        ("齿轮加工工艺路线", "process_route"),
        ("复杂曲面用什么加工", "process_route"),
        ("轴类零件的加工顺序", "process_route"),
        # ---- Cutting parameters (5) ----------------------------------
        ("CNC车削参数", "cutting_parameters"),
        ("铣削速度多少合适", "cutting_parameters"),
        ("不锈钢的切削速度", "cutting_parameters"),
        ("钻孔进给量多少", "cutting_parameters"),
        ("铝合金主轴转速设置", "cutting_parameters"),
        # ---- Tolerance lookup (8) ------------------------------------
        ("IT7的公差是多少", "tolerance_lookup"),
        ("IT6公差值", "tolerance_lookup"),
        ("25mm轴的IT8公差", "tolerance_lookup"),
        ("公差等级IT9对应多少", "tolerance_lookup"),
        ("查一下IT10公差", "tolerance_lookup"),
        ("IT7公差等级的数值", "tolerance_lookup"),
        ("50mm直径IT6的公差带", "tolerance_lookup"),
        ("IT8公差范围查询", "tolerance_lookup"),
        # ---- Fit selection (4) ---------------------------------------
        ("H7/g6是什么配合", "fit_selection"),
        ("过渡配合选什么", "fit_selection"),
        ("轴承配合推荐公差", "fit_selection"),
        ("间隙配合和过盈配合区别", "fit_selection"),
        # ---- Fit calculation (2) -------------------------------------
        ("轴和孔的配合间隙", "fit_calculation"),
        ("H7/g6的最大间隙", "fit_calculation"),
        # ---- GD&T interpretation (8) ---------------------------------
        ("什么是平面度", "gdt_interpretation"),
        ("位置度公差含义", "gdt_interpretation"),
        ("同轴度怎么理解", "gdt_interpretation"),
        ("跳动公差是什么意思", "gdt_interpretation"),
        ("几何公差符号含义", "gdt_interpretation"),
        ("平面度和平行度的区别", "gdt_interpretation"),
        ("圆柱度是什么", "gdt_interpretation"),
        ("形位公差怎么看", "gdt_interpretation"),
        # ---- GD&T application (5) ------------------------------------
        ("圆柱度怎么标注", "gdt_application"),
        ("怎么标注平行度", "gdt_application"),
        ("垂直度公差标注方法", "gdt_application"),
        ("位置度标注步骤", "gdt_application"),
        ("形位公差标注规范", "gdt_application"),
        # ---- Cost estimation (8) -------------------------------------
        ("这个零件多少钱", "cost_estimation"),
        ("加工费用估算", "cost_estimation"),
        ("批量100件的成本", "cost_estimation"),
        ("钛合金零件成本高吗", "cost_estimation"),
        ("单件和批量加工价格差", "cost_estimation"),
        ("估算一下加工费", "cost_estimation"),
        ("零件成本怎么算", "cost_estimation"),
        ("这个件报价多少", "cost_estimation"),
        # ---- Welding parameters (8) ----------------------------------
        ("304不锈钢焊接参数", "welding_parameters"),
        ("铝合金焊接用什么焊丝", "welding_parameters"),
        ("碳钢焊接电流多大", "welding_parameters"),
        ("不锈钢TIG焊接要求", "welding_parameters"),
        ("钛合金焊接保护气体", "welding_parameters"),
        ("MIG焊接参数设置", "welding_parameters"),
        ("焊丝型号选择", "welding_parameters"),
        ("焊接电压电流推荐", "welding_parameters"),
        # ---- General question (5) ------------------------------------
        ("你好", "general_question"),
        ("帮我分析一下", "general_question"),
        ("谢谢", "general_question"),
        ("请问你能做什么", "general_question"),
        ("这个系统怎么用", "general_question"),
    ]

    # ------------------------------------------------------------------
    # Keyword-based fallback patterns
    # ------------------------------------------------------------------

    _FALLBACK_PATTERNS: List[Tuple[str, str]] = [
        (r"密度|强度|硬度|弹性|热膨胀|熔点|导电|耐腐蚀", "material_property"),
        (r"对比|区别|哪个好|比较|哪个", "material_comparison"),
        (r"选什么材|用什么材|推荐.*材", "material_selection"),
        (r"加工|工艺|流程|怎么做|线切割|铣削|车削|磨削|CNC", "process_route"),
        (r"切削.*速度|进给|转速|切削参数", "cutting_parameters"),
        (r"IT\d|公差.*等级|公差值", "tolerance_lookup"),
        (r"配合.*选|H\d.*[a-z]\d|间隙配合|过盈配合|过渡配合", "fit_selection"),
        (r"配合.*间隙|最大间隙|最小过盈", "fit_calculation"),
        (r"什么是.*度|公差.*含义|几何公差|GD&T|跳动", "gdt_interpretation"),
        (r"怎么标注|标注方法|标注", "gdt_application"),
        (r"多少钱|成本|费用|价格.*加工|加工.*价格", "cost_estimation"),
        (r"焊接|焊丝|焊缝|焊接参数|TIG|MIG", "welding_parameters"),
    ]

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._model: Any = None
        self._vectorizer: Any = None
        self._label_encoder: Any = None
        self._trained: bool = False
        self._train()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(self) -> None:
        """Train the classifier on built-in examples."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            logger.info(
                "sklearn not available; IntentClassifier will use fallback patterns."
            )
            self._trained = False
            return

        texts = [t[0] for t in self.TRAINING_DATA]
        labels = [t[1] for t in self.TRAINING_DATA]

        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4)
        )
        X = self._vectorizer.fit_transform(texts)

        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(labels)

        self._model = LogisticRegression(max_iter=1000, C=1.0)
        self._model.fit(X, y)
        self._trained = True

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, query: str) -> Dict[str, Any]:
        """Classify a query into an intent.

        Returns
        -------
        dict
            ``{"intent": str, "confidence": float, "top_3": list, "method": str}``
        """
        if not self._trained:
            return self._classify_fallback(query)

        X = self._vectorizer.transform([query])
        proba = self._model.predict_proba(X)[0]
        top_indices = proba.argsort()[::-1][:3]
        top_3: List[Tuple[str, float]] = [
            (
                self._label_encoder.inverse_transform([i])[0],
                float(proba[i]),
            )
            for i in top_indices
        ]

        return {
            "intent": top_3[0][0],
            "confidence": top_3[0][1],
            "top_3": top_3,
            "method": "trained_classifier",
        }

    def _classify_fallback(self, query: str) -> Dict[str, Any]:
        """Keyword / regex-based fallback classification."""
        for pattern, intent in self._FALLBACK_PATTERNS:
            if re.search(pattern, query):
                return {
                    "intent": intent,
                    "confidence": 0.0,
                    "top_3": [(intent, 0.0)],
                    "method": "fallback",
                }
        return {
            "intent": "general_question",
            "confidence": 0.0,
            "top_3": [("general_question", 0.0)],
            "method": "fallback",
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_supported_intents(self) -> List[str]:
        """Return all supported intent labels."""
        if self._label_encoder is not None:
            return list(self._label_encoder.classes_)
        return sorted(set(t[1] for t in self.TRAINING_DATA))

    @property
    def is_trained(self) -> bool:
        """Whether the sklearn model was successfully trained."""
        return self._trained
