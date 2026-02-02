"""ETL Pipeline Module.

Provides complete ETL (Extract, Transform, Load) capabilities:
- Data sources (files, APIs, databases)
- Data transformations (mapping, filtering, validation)
- Data sinks (files, APIs, databases)
- Pipeline orchestration
"""

from src.core.etl.sources import (
    SourceType,
    SourceConfig,
    FileSourceConfig,
    DatabaseSourceConfig,
    APISourceConfig,
    Record,
    DataSource,
    MemorySource,
    CSVFileSource,
    JSONFileSource,
    APISource,
    create_source,
)
from src.core.etl.transforms import (
    TransformType,
    TransformResult,
    Transform,
    MapTransform,
    FilterTransform,
    ValidateTransform,
    TypeConvertTransform,
    EnrichTransform,
    SelectFieldsTransform,
    DropFieldsTransform,
    FunctionTransform,
    ChainTransform,
)
from src.core.etl.sinks import (
    SinkType,
    SinkConfig,
    FileSinkConfig,
    DatabaseSinkConfig,
    APISinkConfig,
    WriteResult,
    DataSink,
    MemorySink,
    CSVFileSink,
    JSONFileSink,
    APISink,
    create_sink,
)
from src.core.etl.pipeline import (
    PipelineStatus,
    PipelineMetrics,
    PipelineResult,
    PipelineConfig,
    Pipeline,
    PipelineBuilder,
    register_pipeline,
    get_pipeline,
    list_pipelines,
)

__all__ = [
    # Sources
    "SourceType",
    "SourceConfig",
    "FileSourceConfig",
    "DatabaseSourceConfig",
    "APISourceConfig",
    "Record",
    "DataSource",
    "MemorySource",
    "CSVFileSource",
    "JSONFileSource",
    "APISource",
    "create_source",
    # Transforms
    "TransformType",
    "TransformResult",
    "Transform",
    "MapTransform",
    "FilterTransform",
    "ValidateTransform",
    "TypeConvertTransform",
    "EnrichTransform",
    "SelectFieldsTransform",
    "DropFieldsTransform",
    "FunctionTransform",
    "ChainTransform",
    # Sinks
    "SinkType",
    "SinkConfig",
    "FileSinkConfig",
    "DatabaseSinkConfig",
    "APISinkConfig",
    "WriteResult",
    "DataSink",
    "MemorySink",
    "CSVFileSink",
    "JSONFileSink",
    "APISink",
    "create_sink",
    # Pipeline
    "PipelineStatus",
    "PipelineMetrics",
    "PipelineResult",
    "PipelineConfig",
    "Pipeline",
    "PipelineBuilder",
    "register_pipeline",
    "get_pipeline",
    "list_pipelines",
]
