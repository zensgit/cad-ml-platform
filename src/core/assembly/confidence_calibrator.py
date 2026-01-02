"""
置信度校准与融合系统
实现Platt Scaling和Isotonic Regression校准
支持DS证据理论融合
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Optional sklearn imports - gracefully degrade if not available
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Stub classes for when sklearn is not available

    class IsotonicRegression:
        def __init__(self, *args, **kwargs):
            raise ImportError("sklearn not installed. Install with: pip install scikit-learn")

    class LogisticRegression:
        def __init__(self, *args, **kwargs):
            raise ImportError("sklearn not installed. Install with: pip install scikit-learn")


@dataclass
class CalibratedConfidence:
    """校准后的置信度"""

    raw_confidence: float
    calibrated_confidence: float
    per_source_weights: Dict[str, float]
    calibration_method: str
    uncertainty: float  # 不确定性度量


class PlattScaling:
    """Platt校准（Sigmoid校准）"""

    def __init__(self):
        self.calibrator = LogisticRegression()
        self.fitted = False

    def fit(self, confidence_scores: np.ndarray, true_labels: np.ndarray) -> None:
        """训练校准器"""
        # 将置信度转换为logit空间
        epsilon = 1e-10
        confidence_scores = np.clip(confidence_scores, epsilon, 1 - epsilon)
        logit_scores = np.log(confidence_scores / (1 - confidence_scores))

        self.calibrator.fit(logit_scores.reshape(-1, 1), true_labels)
        self.fitted = True

    def calibrate(self, confidence: float) -> float:
        """校准单个置信度"""
        if not self.fitted:
            return confidence

        epsilon = 1e-10
        confidence = np.clip(confidence, epsilon, 1 - epsilon)
        logit = np.log(confidence / (1 - confidence))

        calibrated = self.calibrator.predict_proba([[logit]])[0, 1]
        return float(calibrated)


class IsotonicCalibration:
    """保序回归校准"""

    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.fitted = False

    def fit(self, confidence_scores: np.ndarray, true_labels: np.ndarray) -> None:
        """训练校准器"""
        self.calibrator.fit(confidence_scores, true_labels)
        self.fitted = True

    def calibrate(self, confidence: float) -> float:
        """校准单个置信度"""
        if not self.fitted:
            return confidence

        calibrated = self.calibrator.predict([confidence])[0]
        return float(calibrated)


class DSEvidenceFusion:
    """Dempster-Shafer证据理论融合"""

    @staticmethod
    def combine_evidence(evidence_list: List[Dict]) -> Dict:
        """
        融合多源证据

        Args:
            evidence_list: 证据列表，每个证据包含:
                - source: 证据来源
                - confidence: 置信度
                - uncertainty: 不确定性

        Returns:
            融合后的结果
        """
        if not evidence_list:
            return {
                "confidence": 0.0,
                "uncertainty": 1.0,
                "conflict": 0.0,
                "per_source_weights": {},
            }

        # 初始化质量函数
        masses = []
        for evidence in evidence_list:
            conf = evidence["confidence"]
            unc = evidence.get("uncertainty", 1 - conf)

            # 基本概率分配
            mass = {"positive": conf, "negative": 0.0, "uncertain": unc}
            masses.append(mass)

        # DS组合规则
        combined = masses[0].copy()
        conflicts = []

        for mass in masses[1:]:
            new_combined = {}
            K = 0  # 冲突系数

            # 计算组合质量
            for h1, m1 in combined.items():
                for h2, m2 in mass.items():
                    if h1 == "uncertain" or h2 == "uncertain":
                        # 不确定性传播
                        key = h1 if h2 == "uncertain" else h2
                    elif h1 == h2:
                        key = h1
                    else:
                        # 冲突
                        K += m1 * m2
                        continue

                    if key not in new_combined:
                        new_combined[key] = 0
                    new_combined[key] += m1 * m2

            # 归一化（处理冲突）
            if K < 0.999:  # 避免除零
                for key in new_combined:
                    new_combined[key] /= 1 - K
            else:
                # 冲突太大，保持原状
                new_combined = combined

            conflicts.append(K)
            combined = new_combined

        # 计算权重
        per_source_weights = {}
        total_conf = sum(e["confidence"] for e in evidence_list)
        if total_conf > 0:
            for i, evidence in enumerate(evidence_list):
                weight = evidence["confidence"] / total_conf
                per_source_weights[evidence["source"]] = weight

        return {
            "confidence": combined.get("positive", 0.0),
            "uncertainty": combined.get("uncertain", 0.0),
            "conflict": np.mean(conflicts) if conflicts else 0.0,
            "per_source_weights": per_source_weights,
        }


class LogOddsWeighting:
    """对数几率加权融合"""

    @staticmethod
    def combine_confidence(confidence_list: List[Tuple[float, float]]) -> float:
        """
        使用对数几率加权融合置信度

        Args:
            confidence_list: [(confidence, weight), ...]

        Returns:
            融合后的置信度
        """
        if not confidence_list:
            return 0.5

        epsilon = 1e-10
        log_odds_sum = 0.0
        weight_sum = 0.0

        for conf, weight in confidence_list:
            # 转换到对数几率空间
            conf = np.clip(conf, epsilon, 1 - epsilon)
            log_odds = np.log(conf / (1 - conf))

            log_odds_sum += log_odds * weight
            weight_sum += weight

        if weight_sum == 0:
            return 0.5

        # 平均对数几率
        avg_log_odds = log_odds_sum / weight_sum

        # 转换回概率空间
        probability = 1 / (1 + np.exp(-avg_log_odds))
        return float(probability)


class ConfidenceCalibrationSystem:
    """置信度校准与融合系统"""

    def __init__(self, method: str = "isotonic"):
        """
        Args:
            method: 校准方法 'platt' 或 'isotonic'
        """
        self.method = method
        self.calibrator = None  # Lazy initialization

        # Only create calibrator if sklearn is available
        if SKLEARN_AVAILABLE:
            if method == "platt":
                self.calibrator = PlattScaling()
            else:
                self.calibrator = IsotonicCalibration()
        else:
            # sklearn not available - calibrator will return raw confidence
            pass

        self.ds_fusion = DSEvidenceFusion()
        self.log_odds = LogOddsWeighting()

        # 缓存校准模型
        self.model_path = Path("models/calibration")
        self.model_path.mkdir(parents=True, exist_ok=True)

    def train_calibrator(self, evaluation_data: List[Dict]) -> Dict:
        """
        在评测数据上训练校准器

        Args:
            evaluation_data: 评测数据，包含预测置信度和真实标签
        """
        if self.calibrator is None:
            raise ImportError("sklearn not available. Cannot train calibrator without sklearn.")

        confidence_scores = []
        true_labels = []

        for data in evaluation_data:
            confidence_scores.append(data["predicted_confidence"])
            true_labels.append(data["is_correct"])

        confidence_scores = np.array(confidence_scores)
        true_labels = np.array(true_labels)

        # 训练校准器
        self.calibrator.fit(confidence_scores, true_labels)

        # 保存模型
        self.save_calibrator()

        # 计算校准指标
        return self._calculate_calibration_metrics(confidence_scores, true_labels)

    def calibrate_and_fuse(
        self, evidence_list: List[Dict], fusion_method: str = "ds"
    ) -> CalibratedConfidence:
        """
        校准并融合证据

        Args:
            evidence_list: 证据列表
            fusion_method: 融合方法 'ds' 或 'log_odds'

        Returns:
            校准融合后的置信度
        """
        if not evidence_list:
            return CalibratedConfidence(
                raw_confidence=0.0,
                calibrated_confidence=0.0,
                per_source_weights={},
                calibration_method=self.method,
                uncertainty=1.0,
            )

        # 校准各个证据的置信度
        calibrated_evidence = []
        for evidence in evidence_list:
            raw_conf = evidence["confidence"]
            # Use calibrator if available, otherwise use raw confidence
            if self.calibrator is not None:
                cal_conf = self.calibrator.calibrate(raw_conf)
            else:
                cal_conf = raw_conf  # No calibration when sklearn unavailable

            calibrated_evidence.append(
                {
                    "source": evidence.get("source", "unknown"),
                    "confidence": cal_conf,
                    "uncertainty": 1 - cal_conf,
                    "raw_confidence": raw_conf,
                }
            )

        # 融合证据
        if fusion_method == "ds":
            fusion_result = self.ds_fusion.combine_evidence(calibrated_evidence)
            final_confidence = fusion_result["confidence"]
            uncertainty = fusion_result["uncertainty"]
            per_source_weights = fusion_result["per_source_weights"]
        else:
            # 对数几率加权
            conf_weight_pairs = [(e["confidence"], 1.0) for e in calibrated_evidence]
            final_confidence = self.log_odds.combine_confidence(conf_weight_pairs)
            uncertainty = 1 - final_confidence

            # 计算权重
            per_source_weights = {}
            for e in calibrated_evidence:
                per_source_weights[e["source"]] = e["confidence"] / len(calibrated_evidence)

        # 计算原始置信度（未校准）
        raw_confidence = np.mean([e["raw_confidence"] for e in calibrated_evidence])

        return CalibratedConfidence(
            raw_confidence=raw_confidence,
            calibrated_confidence=final_confidence,
            per_source_weights=per_source_weights,
            calibration_method=f"{self.method}_{fusion_method}",
            uncertainty=uncertainty,
        )

    def _calculate_calibration_metrics(
        self, confidence_scores: np.ndarray, true_labels: np.ndarray
    ) -> Dict:
        """计算校准指标"""

        # Brier Score
        brier_score = np.mean((confidence_scores - true_labels) ** 2)

        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_mask = (confidence_scores >= bin_boundaries[i]) & (
                confidence_scores < bin_boundaries[i + 1]
            )

            if np.sum(bin_mask) > 0:
                bin_confidence = np.mean(confidence_scores[bin_mask])
                bin_accuracy = np.mean(true_labels[bin_mask])
                bin_weight = np.sum(bin_mask) / len(confidence_scores)

                ece += bin_weight * np.abs(bin_confidence - bin_accuracy)

        # 校准后的指标
        if self.calibrator is not None:
            calibrated_scores = np.array([self.calibrator.calibrate(c) for c in confidence_scores])
            calibrated_brier = np.mean((calibrated_scores - true_labels) ** 2)
        else:
            # No calibration available
            calibrated_brier = brier_score

        return {
            "brier_score_before": float(brier_score),
            "brier_score_after": float(calibrated_brier),
            "expected_calibration_error": float(ece),
            "improvement": float(brier_score - calibrated_brier),
        }

    def save_calibrator(self) -> None:
        """保存校准模型"""
        model_file = self.model_path / f"{self.method}_calibrator.pkl"
        with open(model_file, "wb") as f:
            # Trusted local artifact only; not for untrusted input.
            pickle.dump(self.calibrator, f)  # nosec B301

    def load_calibrator(self) -> bool:
        """加载校准模型"""
        model_file = self.model_path / f"{self.method}_calibrator.pkl"
        if model_file.exists():
            with open(model_file, "rb") as f:
                # Trusted local artifact only; not for untrusted input.
                self.calibrator = pickle.load(f)  # nosec B301
                return True
        return False


# 使用示例
if __name__ == "__main__":
    # 创建校准系统
    calibration_system = ConfidenceCalibrationSystem(method="isotonic")

    # 模拟评测数据
    evaluation_data = [
        {"predicted_confidence": 0.9, "is_correct": 1},
        {"predicted_confidence": 0.8, "is_correct": 1},
        {"predicted_confidence": 0.7, "is_correct": 0},
        {"predicted_confidence": 0.6, "is_correct": 1},
        {"predicted_confidence": 0.5, "is_correct": 0},
    ]

    # 训练校准器
    metrics = calibration_system.train_calibrator(evaluation_data)
    print(f"校准指标: {metrics}")

    # 融合证据
    evidence_list = [
        {"source": "geometric", "confidence": 0.9},
        {"source": "textual", "confidence": 0.7},
        {"source": "rule_based", "confidence": 0.8},
    ]

    result = calibration_system.calibrate_and_fuse(evidence_list, fusion_method="ds")
    print(f"校准融合结果: {result}")
