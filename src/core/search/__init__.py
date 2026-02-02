"""Full-text Search module.

Provides:
- Elasticsearch integration
- Index management
- Query DSL builder
- Search aggregations
"""

from src.core.search.client import (
    SearchClient,
    ElasticsearchClient,
    InMemorySearchClient,
    get_search_client,
)
from src.core.search.index import (
    IndexManager,
    IndexMapping,
    FieldType,
    get_index_manager,
)
from src.core.search.query import (
    QueryBuilder,
    BoolQuery,
    MatchQuery,
    TermQuery,
    RangeQuery,
    WildcardQuery,
    FuzzyQuery,
)
from src.core.search.aggregations import (
    Aggregation,
    TermsAggregation,
    DateHistogramAggregation,
    RangeAggregation,
    StatsAggregation,
)

__all__ = [
    # Client
    "SearchClient",
    "ElasticsearchClient",
    "InMemorySearchClient",
    "get_search_client",
    # Index
    "IndexManager",
    "IndexMapping",
    "FieldType",
    "get_index_manager",
    # Query
    "QueryBuilder",
    "BoolQuery",
    "MatchQuery",
    "TermQuery",
    "RangeQuery",
    "WildcardQuery",
    "FuzzyQuery",
    # Aggregations
    "Aggregation",
    "TermsAggregation",
    "DateHistogramAggregation",
    "RangeAggregation",
    "StatsAggregation",
]
