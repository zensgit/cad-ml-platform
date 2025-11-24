#!/usr/bin/env python3
"""
自动参数校准器 (Auto Calibrator)
自动优化限流参数，运行A/B测试，评估性能影响
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import hashlib
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 导入兼容性处理
try:
    from .adaptive_rate_limiter import (
        RateLimitConfig,
        SystemMetrics,
        Decision,
        AdaptiveRateLimiter,
    )
    from .rate_limit_analyzer import (
        TrafficAnalysis,
        TrafficMetrics,
        PatternType,
        RateLimitAnalyzer,
    )
except ImportError:
    from adaptive_rate_limiter import (
        RateLimitConfig,
        SystemMetrics,
        Decision,
        AdaptiveRateLimiter,
    )
    from rate_limit_analyzer import (
        TrafficAnalysis,
        TrafficMetrics,
        PatternType,
        RateLimitAnalyzer,
    )

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizationGoal(Enum):
    """优化目标类型"""
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_LATENCY = "minimize_latency"
    BALANCE_PERFORMANCE = "balance_performance"
    MINIMIZE_ERROR_RATE = "minimize_error_rate"
    MAXIMIZE_AVAILABILITY = "maximize_availability"
    MINIMIZE_COST = "minimize_cost"


class TestStatus(Enum):
    """A/B测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class Parameters:
    """限流参数配置"""
    rate_limit: float  # 请求速率限制
    burst_size: int  # 突发容量
    window_size: float  # 时间窗口大小（秒）
    max_retries: int  # 最大重试次数
    backoff_factor: float  # 退避因子
    cpu_threshold: float  # CPU阈值
    memory_threshold: float  # 内存阈值
    latency_threshold: float  # 延迟阈值（毫秒）
    error_threshold: float  # 错误率阈值

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Parameters':
        """从字典创建"""
        return cls(**data)

    def mutate(self, mutation_rate: float = 0.1) -> 'Parameters':
        """生成变异参数"""
        mutated = {}
        for key, value in self.to_dict().items():
            if isinstance(value, (int, float)):
                # 随机变异参数
                if random.random() < mutation_rate:
                    change = random.uniform(-0.2, 0.2)
                    if isinstance(value, int):
                        mutated[key] = max(1, int(value * (1 + change)))
                    else:
                        mutated[key] = max(0.01, value * (1 + change))
                else:
                    mutated[key] = value
            else:
                mutated[key] = value
        return Parameters.from_dict(mutated)


@dataclass
class PerformanceScore:
    """性能评分"""
    throughput: float  # 吞吐量 (req/s)
    latency_p50: float  # P50延迟 (ms)
    latency_p95: float  # P95延迟 (ms)
    latency_p99: float  # P99延迟 (ms)
    error_rate: float  # 错误率 (0-1)
    availability: float  # 可用性 (0-1)
    cost: float  # 成本评分
    user_satisfaction: float  # 用户满意度 (0-1)
    overall_score: float = 0.0  # 综合评分

    def calculate_overall(self, weights: Dict[str, float]) -> float:
        """计算综合评分"""
        score = 0.0
        score += weights.get('throughput', 0.2) * min(1.0, self.throughput / 10000)
        score += weights.get('latency', 0.3) * max(0, 1.0 - self.latency_p95 / 1000)
        score += weights.get('error_rate', 0.2) * (1.0 - self.error_rate)
        score += weights.get('availability', 0.2) * self.availability
        score += weights.get('satisfaction', 0.1) * self.user_satisfaction
        self.overall_score = score
        return score


@dataclass
class TestResult:
    """A/B测试结果"""
    test_id: str
    variant_a: Parameters
    variant_b: Parameters
    score_a: PerformanceScore
    score_b: PerformanceScore
    sample_size: int
    duration: float  # 测试时长（秒）
    confidence_level: float  # 置信水平
    p_value: float  # 统计显著性
    winner: str  # "A", "B", or "none"
    improvement: float  # 改进百分比
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class CalibrationHistory:
    """校准历史记录"""
    timestamp: datetime
    original_params: Parameters
    optimized_params: Parameters
    performance_before: PerformanceScore
    performance_after: PerformanceScore
    optimization_goal: OptimizationGoal
    success: bool
    notes: str = ""


class AutoCalibrator:
    """自动参数校准器"""

    def __init__(
        self,
        optimization_goal: OptimizationGoal = OptimizationGoal.BALANCE_PERFORMANCE,
        config_file: Optional[str] = None
    ):
        """
        初始化校准器

        Args:
            optimization_goal: 优化目标
            config_file: 配置文件路径
        """
        self.goal = optimization_goal
        self.config_file = config_file or "calibration_config.json"

        # 优化器配置
        self.max_iterations = 100
        self.tolerance = 1e-6
        self.learning_rate = 0.01
        self.momentum = 0.9

        # A/B测试配置
        self.min_sample_size = 1000
        self.max_test_duration = 3600  # 最长1小时
        self.confidence_threshold = 0.95
        self.significance_level = 0.05

        # 历史记录
        self.calibration_history: List[CalibrationHistory] = []
        self.test_results: List[TestResult] = []

        # 当前最优参数
        self.best_params: Optional[Parameters] = None
        self.best_score: Optional[PerformanceScore] = None

        # 性能评分权重
        self.score_weights = self._get_score_weights()

        # 参数边界
        self.param_bounds = {
            'rate_limit': (10, 10000),
            'burst_size': (10, 1000),
            'window_size': (1, 300),
            'max_retries': (1, 10),
            'backoff_factor': (1.0, 5.0),
            'cpu_threshold': (0.5, 0.95),
            'memory_threshold': (0.5, 0.95),
            'latency_threshold': (10, 5000),
            'error_threshold': (0.001, 0.1),
        }

        # 异步执行器
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 加载配置
        self._load_config()

    def _get_score_weights(self) -> Dict[str, float]:
        """获取评分权重"""
        weights = {
            OptimizationGoal.MAXIMIZE_THROUGHPUT: {
                'throughput': 0.5,
                'latency': 0.2,
                'error_rate': 0.15,
                'availability': 0.1,
                'satisfaction': 0.05,
            },
            OptimizationGoal.MINIMIZE_LATENCY: {
                'throughput': 0.15,
                'latency': 0.5,
                'error_rate': 0.15,
                'availability': 0.1,
                'satisfaction': 0.1,
            },
            OptimizationGoal.BALANCE_PERFORMANCE: {
                'throughput': 0.25,
                'latency': 0.25,
                'error_rate': 0.2,
                'availability': 0.2,
                'satisfaction': 0.1,
            },
            OptimizationGoal.MINIMIZE_ERROR_RATE: {
                'throughput': 0.1,
                'latency': 0.15,
                'error_rate': 0.5,
                'availability': 0.2,
                'satisfaction': 0.05,
            },
            OptimizationGoal.MAXIMIZE_AVAILABILITY: {
                'throughput': 0.15,
                'latency': 0.15,
                'error_rate': 0.2,
                'availability': 0.4,
                'satisfaction': 0.1,
            },
        }
        return weights.get(self.goal, weights[OptimizationGoal.BALANCE_PERFORMANCE])

    def calibrate_parameters(
        self,
        current_params: Parameters,
        metrics: SystemMetrics,
        traffic_analysis: TrafficAnalysis
    ) -> Parameters:
        """
        校准限流参数

        Args:
            current_params: 当前参数
            metrics: 系统指标
            traffic_analysis: 流量分析

        Returns:
            优化后的参数
        """
        logger.info(f"开始参数校准，优化目标: {self.goal.value}")

        # 评估当前性能
        current_score = self._evaluate_performance(current_params, metrics)
        logger.info(f"当前性能评分: {current_score.overall_score:.3f}")

        # 根据流量模式选择优化策略
        if traffic_analysis.pattern == PatternType.SPIKE:
            # 流量激增，增加限流阈值
            optimized_params = self._optimize_for_spike(current_params, metrics)
        elif traffic_analysis.pattern == PatternType.DDOS:
            # DDoS攻击，降低限流阈值
            optimized_params = self._optimize_for_attack(current_params, metrics)
        elif traffic_analysis.pattern == PatternType.CRAWLER:
            # 爬虫流量，区分对待
            optimized_params = self._optimize_for_crawler(current_params, metrics)
        else:
            # 正常流量，渐进优化
            optimized_params = self._gradient_optimize(current_params, metrics)

        # 验证新参数
        if self._validate_parameters(optimized_params, metrics):
            # 评估新参数性能
            new_score = self._evaluate_performance(optimized_params, metrics)
            logger.info(f"新参数性能评分: {new_score.overall_score:.3f}")

            # 记录校准历史
            history = CalibrationHistory(
                timestamp=datetime.now(),
                original_params=current_params,
                optimized_params=optimized_params,
                performance_before=current_score,
                performance_after=new_score,
                optimization_goal=self.goal,
                success=new_score.overall_score > current_score.overall_score,
                notes=f"流量模式: {traffic_analysis.pattern.value}"
            )
            self.calibration_history.append(history)

            # 更新最优参数
            if self.best_score is None or new_score.overall_score > self.best_score.overall_score:
                self.best_params = optimized_params
                self.best_score = new_score
                logger.info("更新最优参数")

            return optimized_params
        else:
            logger.warning("新参数验证失败，保持当前参数")
            return current_params

    def _gradient_optimize(
        self,
        params: Parameters,
        metrics: SystemMetrics
    ) -> Parameters:
        """
        梯度优化算法

        Args:
            params: 初始参数
            metrics: 系统指标

        Returns:
            优化后的参数
        """
        # 定义目标函数
        def objective(x: np.ndarray) -> float:
            # 将数组转换为参数
            test_params = self._array_to_params(x)
            # 评估性能
            score = self._evaluate_performance(test_params, metrics)
            # 返回负分数（最小化问题）
            return -score.overall_score

        # 初始参数数组
        x0 = self._params_to_array(params)

        # 获取参数边界
        bounds = self._get_bounds()

        # 运行优化
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.tolerance,
            }
        )

        # 返回优化后的参数
        if result.success:
            return self._array_to_params(result.x)
        else:
            logger.warning(f"优化失败: {result.message}")
            return params

    def _optimize_for_spike(
        self,
        params: Parameters,
        metrics: SystemMetrics
    ) -> Parameters:
        """
        针对流量激增优化

        Args:
            params: 当前参数
            metrics: 系统指标

        Returns:
            优化后的参数
        """
        optimized = Parameters(
            rate_limit=params.rate_limit * 1.5,  # 提高50%限流阈值
            burst_size=params.burst_size * 2,  # 增加突发容量
            window_size=params.window_size * 0.8,  # 缩短时间窗口
            max_retries=max(1, params.max_retries - 1),  # 减少重试
            backoff_factor=params.backoff_factor * 1.2,  # 增加退避
            cpu_threshold=min(0.9, params.cpu_threshold + 0.05),  # 提高CPU阈值
            memory_threshold=min(0.9, params.memory_threshold + 0.05),  # 提高内存阈值
            latency_threshold=params.latency_threshold * 1.2,  # 容忍更高延迟
            error_threshold=min(0.05, params.error_threshold * 1.5),  # 容忍更多错误
        )
        return self._apply_bounds(optimized)

    def _optimize_for_attack(
        self,
        params: Parameters,
        metrics: SystemMetrics
    ) -> Parameters:
        """
        针对攻击流量优化

        Args:
            params: 当前参数
            metrics: 系统指标

        Returns:
            优化后的参数
        """
        optimized = Parameters(
            rate_limit=params.rate_limit * 0.3,  # 大幅降低限流阈值
            burst_size=max(10, params.burst_size // 3),  # 减少突发容量
            window_size=params.window_size * 1.5,  # 延长时间窗口
            max_retries=1,  # 最小重试
            backoff_factor=params.backoff_factor * 2,  # 增加退避
            cpu_threshold=max(0.6, params.cpu_threshold - 0.1),  # 降低CPU阈值
            memory_threshold=max(0.6, params.memory_threshold - 0.1),  # 降低内存阈值
            latency_threshold=params.latency_threshold * 0.5,  # 严格延迟限制
            error_threshold=max(0.001, params.error_threshold * 0.3),  # 严格错误限制
        )
        return self._apply_bounds(optimized)

    def _optimize_for_crawler(
        self,
        params: Parameters,
        metrics: SystemMetrics
    ) -> Parameters:
        """
        针对爬虫流量优化

        Args:
            params: 当前参数
            metrics: 系统指标

        Returns:
            优化后的参数
        """
        optimized = Parameters(
            rate_limit=params.rate_limit * 0.5,  # 降低限流阈值
            burst_size=params.burst_size // 2,  # 减少突发容量
            window_size=params.window_size * 2,  # 延长时间窗口
            max_retries=2,  # 少量重试
            backoff_factor=params.backoff_factor * 1.5,  # 增加退避
            cpu_threshold=params.cpu_threshold,  # 保持CPU阈值
            memory_threshold=params.memory_threshold,  # 保持内存阈值
            latency_threshold=params.latency_threshold * 2,  # 放宽延迟限制
            error_threshold=params.error_threshold,  # 保持错误限制
        )
        return self._apply_bounds(optimized)

    def run_ab_test(
        self,
        variant_a: Parameters,
        variant_b: Parameters,
        duration: float = 3600,
        traffic_split: float = 0.5
    ) -> TestResult:
        """
        运行A/B测试

        Args:
            variant_a: A组参数
            variant_b: B组参数
            duration: 测试时长（秒）
            traffic_split: 流量分割比例

        Returns:
            测试结果
        """
        test_id = self._generate_test_id()
        logger.info(f"开始A/B测试 {test_id}")

        result = TestResult(
            test_id=test_id,
            variant_a=variant_a,
            variant_b=variant_b,
            score_a=PerformanceScore(0, 0, 0, 0, 0, 0, 0, 0),
            score_b=PerformanceScore(0, 0, 0, 0, 0, 0, 0, 0),
            sample_size=0,
            duration=duration,
            confidence_level=0,
            p_value=1.0,
            winner="none",
            improvement=0,
            status=TestStatus.RUNNING,
            start_time=datetime.now()
        )

        # 模拟测试运行
        try:
            # 收集性能数据
            samples_a = []
            samples_b = []

            start_time = time.time()
            while time.time() - start_time < duration:
                # 模拟性能采样
                if random.random() < traffic_split:
                    # A组
                    score = self._simulate_performance(variant_a)
                    samples_a.append(score.overall_score)
                else:
                    # B组
                    score = self._simulate_performance(variant_b)
                    samples_b.append(score.overall_score)

                # 检查样本量
                if len(samples_a) >= self.min_sample_size and \
                   len(samples_b) >= self.min_sample_size:
                    break

                time.sleep(0.1)  # 模拟采样间隔

            # 计算统计结果
            if samples_a and samples_b:
                # T检验
                t_stat, p_value = stats.ttest_ind(samples_a, samples_b)

                # 计算平均性能
                mean_a = np.mean(samples_a)
                mean_b = np.mean(samples_b)

                # 更新结果
                result.score_a.overall_score = mean_a
                result.score_b.overall_score = mean_b
                result.sample_size = len(samples_a) + len(samples_b)
                result.p_value = p_value
                result.confidence_level = 1 - p_value

                # 判断获胜者
                if p_value < self.significance_level:
                    if mean_a > mean_b:
                        result.winner = "A"
                        result.improvement = (mean_a - mean_b) / mean_b * 100
                    else:
                        result.winner = "B"
                        result.improvement = (mean_b - mean_a) / mean_a * 100

                result.status = TestStatus.COMPLETED
                logger.info(f"A/B测试完成，获胜者: {result.winner}, 改进: {result.improvement:.2f}%")
            else:
                raise ValueError("无法收集足够的样本")

        except Exception as e:
            result.status = TestStatus.FAILED
            result.error_message = str(e)
            logger.error(f"A/B测试失败: {e}")

        result.end_time = datetime.now()
        result.duration = (result.end_time - result.start_time).total_seconds()

        # 记录测试结果
        self.test_results.append(result)

        return result

    def evaluate_performance(
        self,
        params: Parameters,
        metrics: Optional[SystemMetrics] = None
    ) -> PerformanceScore:
        """
        评估参数性能

        Args:
            params: 参数配置
            metrics: 系统指标（可选）

        Returns:
            性能评分
        """
        return self._evaluate_performance(params, metrics)

    def _evaluate_performance(
        self,
        params: Parameters,
        metrics: Optional[SystemMetrics] = None
    ) -> PerformanceScore:
        """
        内部性能评估方法

        Args:
            params: 参数配置
            metrics: 系统指标

        Returns:
            性能评分
        """
        # 如果没有提供真实指标，使用模拟
        if metrics is None:
            return self._simulate_performance(params)

        # 基于真实指标计算性能
        score = PerformanceScore(
            throughput=self._estimate_throughput(params, metrics),
            latency_p50=metrics.latency_p50,
            latency_p95=metrics.latency_p95,
            latency_p99=metrics.latency_p99,
            error_rate=metrics.error_rate,
            availability=1.0 - metrics.error_rate,
            cost=self._estimate_cost(params, metrics),
            user_satisfaction=self._estimate_satisfaction(params, metrics)
        )

        # 计算综合评分
        score.calculate_overall(self.score_weights)

        return score

    def _simulate_performance(self, params: Parameters) -> PerformanceScore:
        """
        模拟性能评分

        Args:
            params: 参数配置

        Returns:
            模拟的性能评分
        """
        # 基于参数生成模拟性能
        base_throughput = params.rate_limit * 0.9
        base_latency = 100 / params.rate_limit * 1000

        score = PerformanceScore(
            throughput=base_throughput * random.uniform(0.8, 1.2),
            latency_p50=base_latency * random.uniform(0.7, 0.9),
            latency_p95=base_latency * random.uniform(1.5, 2.0),
            latency_p99=base_latency * random.uniform(2.5, 3.5),
            error_rate=max(0, 1 - params.error_threshold) * random.uniform(0.5, 1.5),
            availability=min(1.0, 0.99 * random.uniform(0.98, 1.02)),
            cost=100 / params.rate_limit * random.uniform(0.9, 1.1),
            user_satisfaction=min(1.0, 1 - base_latency / 1000) * random.uniform(0.9, 1.1)
        )

        score.calculate_overall(self.score_weights)
        return score

    def _estimate_throughput(
        self,
        params: Parameters,
        metrics: SystemMetrics
    ) -> float:
        """估算吞吐量"""
        # 基于CPU和内存使用率估算
        cpu_factor = min(1.0, params.cpu_threshold / max(0.01, metrics.cpu_usage))
        mem_factor = min(1.0, params.memory_threshold / max(0.01, metrics.memory_usage))
        return params.rate_limit * cpu_factor * mem_factor

    def _estimate_cost(
        self,
        params: Parameters,
        metrics: SystemMetrics
    ) -> float:
        """估算成本"""
        # 简化的成本模型
        cpu_cost = metrics.cpu_usage * 100
        mem_cost = metrics.memory_usage * 50
        bandwidth_cost = params.rate_limit * 0.01
        return cpu_cost + mem_cost + bandwidth_cost

    def _estimate_satisfaction(
        self,
        params: Parameters,
        metrics: SystemMetrics
    ) -> float:
        """估算用户满意度"""
        # 基于延迟和错误率
        latency_factor = max(0, 1 - metrics.latency_p95 / params.latency_threshold)
        error_factor = max(0, 1 - metrics.error_rate / params.error_threshold)
        return latency_factor * 0.7 + error_factor * 0.3

    def _validate_parameters(
        self,
        params: Parameters,
        metrics: SystemMetrics
    ) -> bool:
        """
        验证参数合法性

        Args:
            params: 参数配置
            metrics: 系统指标

        Returns:
            是否有效
        """
        # 检查参数边界
        for key, (min_val, max_val) in self.param_bounds.items():
            value = getattr(params, key)
            if value < min_val or value > max_val:
                logger.warning(f"参数 {key}={value} 超出边界 [{min_val}, {max_val}]")
                return False

        # 检查系统限制
        if metrics.cpu_usage > 0.95:
            logger.warning("CPU使用率过高，不适合调整参数")
            return False

        if metrics.memory_usage > 0.95:
            logger.warning("内存使用率过高，不适合调整参数")
            return False

        return True

    def _apply_bounds(self, params: Parameters) -> Parameters:
        """应用参数边界"""
        bounded = params.to_dict()
        for key, (min_val, max_val) in self.param_bounds.items():
            if key in bounded:
                bounded[key] = max(min_val, min(max_val, bounded[key]))
        return Parameters.from_dict(bounded)

    def _params_to_array(self, params: Parameters) -> np.ndarray:
        """参数转数组"""
        return np.array([
            params.rate_limit,
            params.burst_size,
            params.window_size,
            params.max_retries,
            params.backoff_factor,
            params.cpu_threshold,
            params.memory_threshold,
            params.latency_threshold,
            params.error_threshold,
        ])

    def _array_to_params(self, x: np.ndarray) -> Parameters:
        """数组转参数"""
        return Parameters(
            rate_limit=x[0],
            burst_size=int(x[1]),
            window_size=x[2],
            max_retries=int(x[3]),
            backoff_factor=x[4],
            cpu_threshold=x[5],
            memory_threshold=x[6],
            latency_threshold=x[7],
            error_threshold=x[8],
        )

    def _get_bounds(self) -> List[Tuple[float, float]]:
        """获取优化边界"""
        return [
            self.param_bounds['rate_limit'],
            self.param_bounds['burst_size'],
            self.param_bounds['window_size'],
            self.param_bounds['max_retries'],
            self.param_bounds['backoff_factor'],
            self.param_bounds['cpu_threshold'],
            self.param_bounds['memory_threshold'],
            self.param_bounds['latency_threshold'],
            self.param_bounds['error_threshold'],
        ]

    def _generate_test_id(self) -> str:
        """生成测试ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
        return f"test_{timestamp}_{random_suffix}"

    def get_recommendation(self, traffic_analysis: TrafficAnalysis) -> Dict[str, Any]:
        """
        获取参数调整建议

        Args:
            traffic_analysis: 流量分析

        Returns:
            调整建议
        """
        recommendations = {
            'pattern': traffic_analysis.pattern.value,
            'confidence': traffic_analysis.confidence,
            'suggestions': []
        }

        if traffic_analysis.pattern == PatternType.SPIKE:
            recommendations['suggestions'] = [
                "增加burst_size以处理突发流量",
                "提高rate_limit阈值",
                "考虑启用自动扩容"
            ]
        elif traffic_analysis.pattern == PatternType.DDOS:
            recommendations['suggestions'] = [
                "立即降低rate_limit",
                "启用更严格的IP限制",
                "考虑启用验证码或挑战机制"
            ]
        elif traffic_analysis.pattern == PatternType.CRAWLER:
            recommendations['suggestions'] = [
                "为爬虫设置专门的限流规则",
                "增加window_size以平滑请求",
                "考虑提供API接口"
            ]
        elif traffic_analysis.pattern == PatternType.BRUTE_FORCE:
            recommendations['suggestions'] = [
                "减少max_retries",
                "增加backoff_factor",
                "考虑临时封禁可疑IP"
            ]
        else:
            recommendations['suggestions'] = [
                "当前参数适合正常流量",
                "继续监控系统指标",
                "考虑运行A/B测试优化"
            ]

        # 添加基于历史的建议
        if self.best_params:
            recommendations['best_params'] = self.best_params.to_dict()
            recommendations['best_score'] = self.best_score.overall_score if self.best_score else 0

        return recommendations

    def rollback_parameters(self) -> Optional[Parameters]:
        """
        回滚到上一个成功的参数配置

        Returns:
            回滚的参数，如果没有历史则返回None
        """
        # 查找最近的成功配置
        for history in reversed(self.calibration_history):
            if history.success:
                logger.info(f"回滚到 {history.timestamp} 的配置")
                return history.optimized_params

        logger.warning("没有找到可回滚的成功配置")
        return None

    def _save_config(self):
        """保存配置到文件"""
        config = {
            'optimization_goal': self.goal.value,
            'best_params': self.best_params.to_dict() if self.best_params else None,
            'best_score': self.best_score.overall_score if self.best_score else 0,
            'calibration_history': [
                {
                    'timestamp': h.timestamp.isoformat(),
                    'original_params': h.original_params.to_dict(),
                    'optimized_params': h.optimized_params.to_dict(),
                    'performance_before': h.performance_before.overall_score,
                    'performance_after': h.performance_after.overall_score,
                    'success': h.success,
                    'notes': h.notes
                }
                for h in self.calibration_history[-10:]  # 保留最近10条
            ]
        }

        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"配置已保存到 {self.config_file}")

    def _load_config(self):
        """从文件加载配置"""
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)

                # 加载最优参数
                if config.get('best_params'):
                    self.best_params = Parameters.from_dict(config['best_params'])
                    logger.info("加载最优参数配置")

                # 加载历史记录
                for h in config.get('calibration_history', []):
                    history = CalibrationHistory(
                        timestamp=datetime.fromisoformat(h['timestamp']),
                        original_params=Parameters.from_dict(h['original_params']),
                        optimized_params=Parameters.from_dict(h['optimized_params']),
                        performance_before=PerformanceScore(0, 0, 0, 0, 0, 0, 0, 0, h['performance_before']),
                        performance_after=PerformanceScore(0, 0, 0, 0, 0, 0, 0, 0, h['performance_after']),
                        optimization_goal=OptimizationGoal(self.goal.value),
                        success=h['success'],
                        notes=h.get('notes', '')
                    )
                    self.calibration_history.append(history)

                logger.info(f"加载了 {len(self.calibration_history)} 条历史记录")

            except Exception as e:
                logger.error(f"加载配置失败: {e}")

    def export_report(self) -> Dict[str, Any]:
        """
        导出校准报告

        Returns:
            报告数据
        """
        report = {
            'summary': {
                'total_calibrations': len(self.calibration_history),
                'successful_calibrations': sum(1 for h in self.calibration_history if h.success),
                'total_ab_tests': len(self.test_results),
                'optimization_goal': self.goal.value,
                'best_score': self.best_score.overall_score if self.best_score else 0,
            },
            'current_best': self.best_params.to_dict() if self.best_params else None,
            'recent_calibrations': [
                {
                    'timestamp': h.timestamp.isoformat(),
                    'success': h.success,
                    'improvement': (h.performance_after.overall_score - h.performance_before.overall_score) /
                                 h.performance_before.overall_score * 100 if h.performance_before.overall_score > 0 else 0,
                    'notes': h.notes
                }
                for h in self.calibration_history[-5:]
            ],
            'recent_tests': [
                {
                    'test_id': t.test_id,
                    'winner': t.winner,
                    'improvement': t.improvement,
                    'confidence': t.confidence_level,
                    'status': t.status.value
                }
                for t in self.test_results[-5:]
            ]
        }

        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="自动参数校准器 - 优化限流参数"
    )
    parser.add_argument(
        '--goal',
        type=str,
        choices=['throughput', 'latency', 'balance', 'error_rate', 'availability'],
        default='balance',
        help='优化目标'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='运行A/B测试'
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='导出报告'
    )
    parser.add_argument(
        '--rollback',
        action='store_true',
        help='回滚到上一个成功配置'
    )

    args = parser.parse_args()

    # 目标映射
    goal_map = {
        'throughput': OptimizationGoal.MAXIMIZE_THROUGHPUT,
        'latency': OptimizationGoal.MINIMIZE_LATENCY,
        'balance': OptimizationGoal.BALANCE_PERFORMANCE,
        'error_rate': OptimizationGoal.MINIMIZE_ERROR_RATE,
        'availability': OptimizationGoal.MAXIMIZE_AVAILABILITY,
    }

    # 创建校准器
    calibrator = AutoCalibrator(optimization_goal=goal_map[args.goal])

    if args.rollback:
        # 回滚配置
        params = calibrator.rollback_parameters()
        if params:
            print(f"已回滚到参数配置:")
            print(json.dumps(params.to_dict(), indent=2))
        else:
            print("没有可回滚的配置")

    elif args.test:
        # 运行A/B测试
        print("运行A/B测试...")

        # 创建测试参数
        params_a = Parameters(
            rate_limit=1000,
            burst_size=100,
            window_size=60,
            max_retries=3,
            backoff_factor=2.0,
            cpu_threshold=0.8,
            memory_threshold=0.85,
            latency_threshold=100,
            error_threshold=0.01
        )

        params_b = params_a.mutate(0.2)  # 20%变异

        # 运行测试
        result = calibrator.run_ab_test(params_a, params_b, duration=60)

        # 输出结果
        print(f"\n测试ID: {result.test_id}")
        print(f"状态: {result.status.value}")
        print(f"样本量: {result.sample_size}")
        print(f"A组评分: {result.score_a.overall_score:.3f}")
        print(f"B组评分: {result.score_b.overall_score:.3f}")
        print(f"P值: {result.p_value:.4f}")
        print(f"获胜者: {result.winner}")
        if result.winner != "none":
            print(f"改进: {result.improvement:.2f}%")

    elif args.export:
        # 导出报告
        report = calibrator.export_report()
        print(json.dumps(report, indent=2))

    else:
        # 运行校准
        print(f"开始参数校准，优化目标: {args.goal}")

        # 模拟当前参数和指标
        current_params = Parameters(
            rate_limit=1000,
            burst_size=100,
            window_size=60,
            max_retries=3,
            backoff_factor=2.0,
            cpu_threshold=0.8,
            memory_threshold=0.85,
            latency_threshold=100,
            error_threshold=0.01
        )

        metrics = SystemMetrics(
            cpu_usage=0.65,
            memory_usage=0.70,
            latency_p50=50,
            latency_p95=95,
            latency_p99=150,
            error_rate=0.005,
            request_rate=800,
            active_connections=5000
        )

        # 创建流量分析
        analyzer = RateLimitAnalyzer()
        traffic_analysis = TrafficAnalysis(
            pattern=PatternType.NORMAL,
            confidence=0.85,
            anomaly_score=0.1,
            metrics=TrafficMetrics(
                request_count=10000,
                unique_ips=1000,
                avg_request_rate=800,
                peak_request_rate=1200,
                error_rate=0.005,
                avg_response_time=60,
                bandwidth_usage=100 * 1024 * 1024,
                time_window=60
            ),
            recommendations=[]
        )

        # 运行校准
        optimized = calibrator.calibrate_parameters(
            current_params,
            metrics,
            traffic_analysis
        )

        # 输出结果
        print("\n优化后的参数:")
        print(json.dumps(optimized.to_dict(), indent=2))

        # 获取建议
        recommendations = calibrator.get_recommendation(traffic_analysis)
        print("\n参数调整建议:")
        print(json.dumps(recommendations, indent=2))

        # 保存配置
        calibrator._save_config()


if __name__ == "__main__":
    main()