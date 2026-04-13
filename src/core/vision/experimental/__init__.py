"""Experimental / prototype vision modules.

These modules were identified during Phase A2 audit as having no external
consumers outside ``src/core/vision/``.  They are preserved here for future
evaluation but are **not** part of the production API surface.

Moved modules (2026-04-13):
  - audit_logger        (666 LOC)  duplicate of audit_logging
  - automl_engine       (893 LOC)  ML experiment framework stub
  - compliance_checker  (828 LOC)  duplicate of compliance
  - data_lifecycle     (1280 LOC)  advanced data management stub
  - encryption_manager  (636 LOC)  duplicate of encryption
  - experiment_tracker  (945 LOC)  ML experiment tracking stub
  - feature_store       (873 LOC)  feature store stub (used only by data_lifecycle)
  - intelligent_automation (1534 LOC)  automation framework stub
  - model_registry      (943 LOC)  model registry stub
  - pipeline_orchestrator (886 LOC) pipeline orchestrator stub
  - security_scanner    (731 LOC)  duplicate of security_audit
"""
