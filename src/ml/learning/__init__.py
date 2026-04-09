"""
Learning module -- closes the feedback loop between user corrections and model improvement.

Exports:
    FeedbackLearningPipeline: Ingests corrections, adapts fusion weights, triggers updates.
    SmartSampler: Selects the most informative samples for human labeling.
"""

from src.ml.learning.feedback_loop import FeedbackLearningPipeline
from src.ml.learning.smart_sampler import SmartSampler

__all__ = ["FeedbackLearningPipeline", "SmartSampler"]
