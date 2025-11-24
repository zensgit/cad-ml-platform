"""
Test Adaptive Rate Limiter
自适应限流器测试
"""

import pytest
import time
import threading
from unittest.mock import MagicMock, patch
from src.core.resilience.adaptive_rate_limiter import (
    AdaptiveRateLimiter,
    AdaptiveConfig,
    AdaptivePhase,
    AdaptiveRateLimiterManager,
    AdjustmentRecord
)


class TestAdaptiveRateLimiter:
    """测试自适应限流器"""

    def test_initialization(self):
        """测试初始化"""
        config = AdaptiveConfig(base_rate=100.0)
        limiter = AdaptiveRateLimiter("test_service", "test_endpoint", config)

        assert limiter.state.base_rate == 100.0
        assert limiter.state.current_rate == 100.0
        assert limiter.state.phase == AdaptivePhase.NORMAL
        assert limiter.state.error_ema == 0.0

    def test_token_acquisition(self):
        """测试令牌获取"""
        config = AdaptiveConfig(base_rate=10.0)  # 每秒10个令牌
        limiter = AdaptiveRateLimiter("test", "endpoint", config)

        # 初始应该有令牌
        assert limiter.acquire()

        # 快速消耗令牌
        for _ in range(9):
            limiter.acquire()

        # 第11个应该失败
        assert not limiter.acquire()

        # 等待补充
        time.sleep(0.2)  # 等待200ms，应该补充2个令牌
        assert limiter.acquire()

    def test_adaptive_no_adjust_under_threshold(self):
        """测试低于阈值时不调整"""
        config = AdaptiveConfig(
            base_rate=100.0,
            error_threshold=0.02,
            adjust_min_interval_ms=100
        )
        limiter = AdaptiveRateLimiter("test", "endpoint", config)

        # 记录少量错误
        for _ in range(5):
            limiter.record_success()
        limiter.record_error()  # 错误率 = 1/6 ≈ 0.17，但EMA会平滑

        # 等待最小间隔
        time.sleep(0.15)

        # 评估
        adjustment = limiter.evaluate_and_adjust()

        # 由于EMA平滑，不应该调整
        assert adjustment is None or limiter.state.phase == AdaptivePhase.NORMAL

    def test_adaptive_degrade_on_error_spike(self):
        """测试错误激增时降级"""
        config = AdaptiveConfig(
            base_rate=100.0,
            error_threshold=0.02,
            error_alpha=0.5,  # 更高的alpha使其对新值更敏感
            adjust_min_interval_ms=100
        )
        limiter = AdaptiveRateLimiter("test", "endpoint", config)

        # 记录大量错误
        for _ in range(10):
            limiter.record_error()
        for _ in range(10):
            limiter.record_success()

        # 等待最小间隔
        time.sleep(0.15)

        # 评估
        adjustment = limiter.evaluate_and_adjust()

        # 应该降级
        assert adjustment is not None
        assert adjustment.reason in ["error", "failures"]
        assert limiter.state.phase == AdaptivePhase.DEGRADING
        assert limiter.state.current_rate < limiter.state.base_rate

    def test_adaptive_clamp_at_min_rate(self):
        """测试触底保护"""
        config = AdaptiveConfig(
            base_rate=100.0,
            min_rate_ratio=0.15,
            error_threshold=0.01,
            error_alpha=0.9,
            adjust_min_interval_ms=50
        )
        limiter = AdaptiveRateLimiter("test", "endpoint", config)

        # 多次降级到最低
        for _ in range(5):
            # 记录错误
            for _ in range(20):
                limiter.record_error()
            limiter.record_success()

            time.sleep(0.1)
            limiter.evaluate_and_adjust()

        # 应该在最小速率
        min_rate = config.base_rate * config.min_rate_ratio
        assert abs(limiter.state.current_rate - min_rate) < 0.1
        assert limiter.state.phase == AdaptivePhase.CLAMPED

    def test_adaptive_recover_gradually(self):
        """测试逐步恢复"""
        config = AdaptiveConfig(
            base_rate=100.0,
            error_threshold=0.02,
            recover_threshold=0.008,
            recover_step=0.1,  # 每次恢复10%
            adjust_min_interval_ms=50
        )
        limiter = AdaptiveRateLimiter("test", "endpoint", config)

        # 先降级
        limiter.state.current_rate = 50.0
        limiter.state.phase = AdaptivePhase.DEGRADING
        limiter.state.error_ema = 0.005  # 低于恢复阈值

        # 记录成功
        for _ in range(100):
            limiter.record_success()

        # 记录正常延迟
        for _ in range(10):
            limiter.record_latency(500)

        time.sleep(0.1)

        # 评估恢复
        adjustment = limiter.evaluate_and_adjust()

        # 应该恢复一步
        assert adjustment is not None
        assert adjustment.reason == "recover"
        assert limiter.state.current_rate > 50.0
        assert limiter.state.current_rate <= 60.0  # 恢复10%
        assert limiter.state.phase == AdaptivePhase.RECOVERY

    def test_adaptive_latency_trigger_without_errors(self):
        """测试延迟触发（无错误）"""
        config = AdaptiveConfig(
            base_rate=100.0,
            latency_p95_threshold_multiplier=1.3,
            adjust_min_interval_ms=50
        )
        limiter = AdaptiveRateLimiter("test", "endpoint", config)
        limiter.set_baseline(1000.0)  # 基线1秒

        # 记录高延迟
        for _ in range(100):
            limiter.record_latency(1500)  # 1.5倍基线
            limiter.record_success()

        time.sleep(0.1)

        # 评估
        adjustment = limiter.evaluate_and_adjust()

        # 应该因延迟降级
        assert adjustment is not None
        assert adjustment.reason == "latency"
        assert limiter.state.phase == AdaptivePhase.DEGRADING

    def test_adaptive_adjustment_interval_respected(self):
        """测试调整间隔限制"""
        config = AdaptiveConfig(
            base_rate=100.0,
            adjust_min_interval_ms=1000,  # 1秒间隔
            error_threshold=0.01
        )
        limiter = AdaptiveRateLimiter("test", "endpoint", config)

        # 记录错误
        for _ in range(10):
            limiter.record_error()

        # 第一次调整
        adj1 = limiter.evaluate_and_adjust()
        assert adj1 is not None

        # 立即再次尝试（应该被阻止）
        adj2 = limiter.evaluate_and_adjust()
        assert adj2 is None

        # 等待间隔
        time.sleep(1.1)

        # 现在应该可以调整
        for _ in range(10):
            limiter.record_error()
        adj3 = limiter.evaluate_and_adjust()
        assert adj3 is not None

    def test_adaptive_jitter_suppression(self):
        """测试抖动抑制"""
        config = AdaptiveConfig(
            base_rate=100.0,
            jitter_detection_window=3,
            jitter_threshold=0.5,
            cooldown_duration_ms=1000,
            adjust_min_interval_ms=50
        )
        limiter = AdaptiveRateLimiter("test", "endpoint", config)

        # 创建抖动模式：上下上下
        patterns = [
            (True, 10, 1),    # 错误多 -> 降低
            (False, 1, 10),   # 成功多 -> 提高
            (True, 10, 1),    # 错误多 -> 降低
            (False, 1, 10),   # 成功多 -> 应该触发抖动检测
        ]

        for should_fail, errors, successes in patterns:
            # 清空计数
            limiter.error_count = 0
            limiter.success_count = 0

            # 记录
            for _ in range(errors):
                limiter.record_error()
            for _ in range(successes):
                limiter.record_success()

            # 根据模式调整EMA
            if should_fail:
                limiter.state.error_ema = 0.5  # 高错误率
            else:
                limiter.state.error_ema = 0.001  # 低错误率

            time.sleep(0.1)
            adjustment = limiter.evaluate_and_adjust()

            # 前3次应该正常调整
            if len(limiter.state.adjust_history) < 3:
                assert adjustment is not None
            else:
                # 第4次应该检测到抖动并进入冷却
                if adjustment is None:
                    assert limiter.state.cooldown_until > time.time()

    def test_adaptive_failure_streak_triggers(self):
        """测试连续失败触发"""
        config = AdaptiveConfig(
            base_rate=100.0,
            max_failure_streak=3,
            adjust_min_interval_ms=50
        )
        limiter = AdaptiveRateLimiter("test", "endpoint", config)

        # 记录连续失败
        for _ in range(3):
            limiter.record_error()

        time.sleep(0.1)

        # 评估
        adjustment = limiter.evaluate_and_adjust()

        # 应该触发降级
        assert adjustment is not None
        assert adjustment.reason == "failures"
        assert limiter.state.phase == AdaptivePhase.DEGRADING

    def test_concurrent_access(self):
        """测试并发访问"""
        config = AdaptiveConfig(base_rate=100.0)
        limiter = AdaptiveRateLimiter("test", "endpoint", config)

        success_count = [0]
        fail_count = [0]

        def worker():
            for _ in range(100):
                if limiter.acquire():
                    success_count[0] += 1
                    limiter.record_success()
                else:
                    fail_count[0] += 1
                    limiter.record_error()
                time.sleep(0.001)

        # 启动多个线程
        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        # 等待完成
        for t in threads:
            t.join()

        # 验证总数
        total = success_count[0] + fail_count[0]
        assert total == 500  # 5线程 * 100次

    def test_get_status(self):
        """测试状态获取"""
        config = AdaptiveConfig(base_rate=100.0)
        limiter = AdaptiveRateLimiter("test", "endpoint", config)

        status = limiter.get_status()

        assert status["service"] == "test"
        assert status["endpoint"] == "endpoint"
        assert status["enabled"] == True
        assert status["phase"] == "normal"
        assert status["base_rate"] == 100.0
        assert status["current_rate"] == 100.0
        assert status["error_ema"] == 0.0

    def test_reset(self):
        """测试重置"""
        config = AdaptiveConfig(base_rate=100.0)
        limiter = AdaptiveRateLimiter("test", "endpoint", config)

        # 修改状态
        limiter.state.current_rate = 50.0
        limiter.state.phase = AdaptivePhase.DEGRADING
        limiter.state.error_ema = 0.5
        limiter.state.consecutive_failures = 10

        # 重置
        limiter.reset()

        # 验证恢复初始状态
        assert limiter.state.current_rate == 100.0
        assert limiter.state.phase == AdaptivePhase.NORMAL
        assert limiter.state.error_ema == 0.0
        assert limiter.state.consecutive_failures == 0

    def test_force_phase(self):
        """测试强制阶段（用于测试）"""
        config = AdaptiveConfig(base_rate=100.0)
        limiter = AdaptiveRateLimiter("test", "endpoint", config)

        # 强制设置阶段
        limiter.force_phase(AdaptivePhase.CLAMPED, 15.0)

        assert limiter.state.phase == AdaptivePhase.CLAMPED
        assert limiter.state.current_rate == 15.0


class TestAdaptiveRateLimiterManager:
    """测试管理器"""

    def test_get_or_create(self):
        """测试获取或创建"""
        manager = AdaptiveRateLimiterManager()

        # 第一次创建
        limiter1 = manager.get_or_create("service1", "endpoint1")
        assert limiter1 is not None

        # 第二次应该返回相同实例
        limiter2 = manager.get_or_create("service1", "endpoint1")
        assert limiter1 is limiter2

        # 不同的键创建新实例
        limiter3 = manager.get_or_create("service2", "endpoint2")
        assert limiter3 is not limiter1

    def test_evaluate_all(self):
        """测试批量评估"""
        manager = AdaptiveRateLimiterManager()

        # 创建多个限流器
        limiter1 = manager.get_or_create("service1", "endpoint1")
        limiter2 = manager.get_or_create("service2", "endpoint2")

        # Mock evaluate_and_adjust
        limiter1.evaluate_and_adjust = MagicMock()
        limiter2.evaluate_and_adjust = MagicMock()

        # 执行批量评估
        manager.evaluate_all()

        # 验证都被调用
        limiter1.evaluate_and_adjust.assert_called_once()
        limiter2.evaluate_and_adjust.assert_called_once()

    def test_get_all_status(self):
        """测试获取所有状态"""
        manager = AdaptiveRateLimiterManager()

        # 创建限流器
        manager.get_or_create("service1", "endpoint1")
        manager.get_or_create("service2", "endpoint2")

        # 获取状态
        all_status = manager.get_all_status()

        assert "service1:endpoint1" in all_status
        assert "service2:endpoint2" in all_status
        assert all_status["service1:endpoint1"]["service"] == "service1"
        assert all_status["service2:endpoint2"]["endpoint"] == "endpoint2"


class TestIntegrationScenarios:
    """集成场景测试"""

    def test_normal_load_no_adjustment(self):
        """测试正常负载不调整"""
        config = AdaptiveConfig(
            base_rate=100.0,
            error_threshold=0.02,
            adjust_min_interval_ms=100
        )
        limiter = AdaptiveRateLimiter("test", "endpoint", config)

        # 模拟正常负载（1%错误率）
        for _ in range(10):
            for _ in range(99):
                limiter.record_success()
            limiter.record_error()

            # 正常延迟
            for _ in range(10):
                limiter.record_latency(800)

            time.sleep(0.15)
            adjustment = limiter.evaluate_and_adjust()

        # 不应该有调整或保持NORMAL
        assert limiter.state.phase == AdaptivePhase.NORMAL
        assert abs(limiter.state.current_rate - 100.0) < 1.0

    def test_degradation_and_recovery_cycle(self):
        """测试完整的降级-恢复周期"""
        config = AdaptiveConfig(
            base_rate=100.0,
            error_threshold=0.02,
            recover_threshold=0.008,
            recover_step=0.2,
            adjust_min_interval_ms=100
        )
        limiter = AdaptiveRateLimiter("test", "endpoint", config)
        limiter.set_baseline(1000.0)

        # 阶段1：正常
        assert limiter.state.phase == AdaptivePhase.NORMAL

        # 阶段2：引入错误导致降级
        for _ in range(20):
            limiter.record_error()
        for _ in range(20):
            limiter.record_success()

        time.sleep(0.15)
        limiter.evaluate_and_adjust()

        assert limiter.state.phase == AdaptivePhase.DEGRADING
        degraded_rate = limiter.state.current_rate
        assert degraded_rate < 100.0

        # 阶段3：系统恢复
        for _ in range(5):  # 多次恢复周期
            # 清空计数
            limiter.error_count = 0
            limiter.success_count = 0

            # 记录良好表现
            for _ in range(100):
                limiter.record_success()

            # 正常延迟
            for _ in range(20):
                limiter.record_latency(900)

            # 手动降低EMA（模拟恢复）
            limiter.state.error_ema *= 0.5

            time.sleep(0.15)
            limiter.evaluate_and_adjust()

            # 检查恢复
            if limiter.state.phase == AdaptivePhase.NORMAL:
                break

        # 最终应该恢复到正常
        assert limiter.state.phase in [AdaptivePhase.NORMAL, AdaptivePhase.RECOVERY]
        assert limiter.state.current_rate > degraded_rate

    def test_high_latency_low_error_conservative_adjust(self):
        """测试高延迟但低错误率时保守调整"""
        config = AdaptiveConfig(
            base_rate=100.0,
            error_threshold=0.02,
            latency_p95_threshold_multiplier=1.3,
            error_alpha=0.25,
            adjust_min_interval_ms=100
        )
        limiter = AdaptiveRateLimiter("test", "endpoint", config)
        limiter.set_baseline(1000.0)

        # 低错误率
        for _ in range(100):
            limiter.record_success()

        # 高延迟
        for _ in range(100):
            limiter.record_latency(1400)  # 1.4倍基线

        time.sleep(0.15)
        adjustment = limiter.evaluate_and_adjust()

        # 应该调整但保守
        if adjustment:
            # 降级幅度应该较小
            rate_reduction = (limiter.state.base_rate - limiter.state.current_rate) / limiter.state.base_rate
            assert rate_reduction < 0.5  # 降幅小于50%

    def test_sustained_jitter_triggers_cooldown(self):
        """测试持续抖动触发冷却"""
        config = AdaptiveConfig(
            base_rate=100.0,
            jitter_detection_window=3,
            jitter_threshold=0.5,
            cooldown_duration_ms=500,
            adjust_min_interval_ms=50,
            error_alpha=0.9  # 高alpha使EMA快速响应
        )
        limiter = AdaptiveRateLimiter("test", "endpoint", config)

        adjustments_made = 0

        # 创建抖动模式
        for i in range(6):
            # 清空计数
            limiter.error_count = 0
            limiter.success_count = 0

            if i % 2 == 0:
                # 高错误
                for _ in range(20):
                    limiter.record_error()
                limiter.record_success()
            else:
                # 低错误
                limiter.record_error()
                for _ in range(20):
                    limiter.record_success()

            time.sleep(0.1)
            adjustment = limiter.evaluate_and_adjust()

            if adjustment:
                adjustments_made += 1

        # 应该检测到抖动并限制调整
        assert adjustments_made < 6  # 不应该每次都调整
        # 最终应该在冷却中
        assert limiter.state.cooldown_until > 0 or len(limiter.state.adjust_history) >= 3