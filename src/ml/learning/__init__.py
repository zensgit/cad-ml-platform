"""
Learning module -- selects the most informative samples for human labeling.

Exports:
    SmartSampler: Selects the most informative samples for human labeling.

Note: this package previously exported ``FeedbackLearningPipeline`` and its docstring
claimed the module "closes the feedback loop between user corrections and model
improvement". It did not. The pipeline computed EMA fusion-branch weights and persisted
them to ``weight_history.jsonl``, but **no inference path ever read them back** -- the
live classifier resolves its branch weights from env/config at construction. It was
write-only plumbing with a single test as its only consumer, so Phase 0 slice B1 deleted
it rather than "wiring" it.

The real flywheel already has a working spine (classifier -> ``low_conf.csv`` ->
``auto_retrain.sh``, guarded by a hard governance gate). What it lacks is a feedback
SOURCE and a human-review action -- see the positioning/roadmap design (merged in #499),
track B. That is a build, not a reconnection.
"""

from src.ml.learning.smart_sampler import SmartSampler

__all__ = ["SmartSampler"]
