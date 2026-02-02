"""Search Query DSL Builder.

Provides a fluent interface for building Elasticsearch queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class Query:
    """Base query class."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Elasticsearch query dict."""
        raise NotImplementedError


@dataclass
class MatchAllQuery(Query):
    """Match all documents."""

    boost: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        if self.boost != 1.0:
            return {"match_all": {"boost": self.boost}}
        return {"match_all": {}}


@dataclass
class MatchQuery(Query):
    """Full-text match query."""

    field: str
    query: str
    operator: str = "or"  # or, and
    fuzziness: Optional[str] = None  # AUTO, 0, 1, 2
    minimum_should_match: Optional[str] = None
    analyzer: Optional[str] = None
    boost: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        query_body: Dict[str, Any] = {"query": self.query}

        if self.operator != "or":
            query_body["operator"] = self.operator

        if self.fuzziness:
            query_body["fuzziness"] = self.fuzziness

        if self.minimum_should_match:
            query_body["minimum_should_match"] = self.minimum_should_match

        if self.analyzer:
            query_body["analyzer"] = self.analyzer

        if self.boost != 1.0:
            query_body["boost"] = self.boost

        return {"match": {self.field: query_body}}


@dataclass
class MatchPhraseQuery(Query):
    """Match phrase query."""

    field: str
    query: str
    slop: int = 0
    analyzer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        query_body: Dict[str, Any] = {"query": self.query}

        if self.slop:
            query_body["slop"] = self.slop

        if self.analyzer:
            query_body["analyzer"] = self.analyzer

        return {"match_phrase": {self.field: query_body}}


@dataclass
class MultiMatchQuery(Query):
    """Multi-field match query."""

    query: str
    fields: List[str]
    type: str = "best_fields"  # best_fields, most_fields, cross_fields, phrase, phrase_prefix
    operator: str = "or"
    fuzziness: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        query_body: Dict[str, Any] = {
            "query": self.query,
            "fields": self.fields,
            "type": self.type,
        }

        if self.operator != "or":
            query_body["operator"] = self.operator

        if self.fuzziness:
            query_body["fuzziness"] = self.fuzziness

        return {"multi_match": query_body}


@dataclass
class TermQuery(Query):
    """Exact term match query."""

    field: str
    value: Any
    boost: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        if self.boost != 1.0:
            return {"term": {self.field: {"value": self.value, "boost": self.boost}}}
        return {"term": {self.field: self.value}}


@dataclass
class TermsQuery(Query):
    """Multiple exact term match query."""

    field: str
    values: List[Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"terms": {self.field: self.values}}


@dataclass
class RangeQuery(Query):
    """Range query."""

    field: str
    gte: Optional[Any] = None
    gt: Optional[Any] = None
    lte: Optional[Any] = None
    lt: Optional[Any] = None
    format: Optional[str] = None
    boost: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        range_body: Dict[str, Any] = {}

        if self.gte is not None:
            range_body["gte"] = self.gte
        if self.gt is not None:
            range_body["gt"] = self.gt
        if self.lte is not None:
            range_body["lte"] = self.lte
        if self.lt is not None:
            range_body["lt"] = self.lt
        if self.format:
            range_body["format"] = self.format
        if self.boost != 1.0:
            range_body["boost"] = self.boost

        return {"range": {self.field: range_body}}


@dataclass
class WildcardQuery(Query):
    """Wildcard pattern query."""

    field: str
    value: str
    boost: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        if self.boost != 1.0:
            return {"wildcard": {self.field: {"value": self.value, "boost": self.boost}}}
        return {"wildcard": {self.field: self.value}}


@dataclass
class PrefixQuery(Query):
    """Prefix query."""

    field: str
    value: str
    boost: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        if self.boost != 1.0:
            return {"prefix": {self.field: {"value": self.value, "boost": self.boost}}}
        return {"prefix": {self.field: self.value}}


@dataclass
class FuzzyQuery(Query):
    """Fuzzy match query."""

    field: str
    value: str
    fuzziness: str = "AUTO"
    prefix_length: int = 0
    max_expansions: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fuzzy": {
                self.field: {
                    "value": self.value,
                    "fuzziness": self.fuzziness,
                    "prefix_length": self.prefix_length,
                    "max_expansions": self.max_expansions,
                }
            }
        }


@dataclass
class ExistsQuery(Query):
    """Field exists query."""

    field: str

    def to_dict(self) -> Dict[str, Any]:
        return {"exists": {"field": self.field}}


@dataclass
class RegexpQuery(Query):
    """Regular expression query."""

    field: str
    value: str
    flags: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        query_body: Dict[str, Any] = {"value": self.value}
        if self.flags:
            query_body["flags"] = self.flags
        return {"regexp": {self.field: query_body}}


@dataclass
class BoolQuery(Query):
    """Boolean compound query."""

    must: List[Query] = field(default_factory=list)
    must_not: List[Query] = field(default_factory=list)
    should: List[Query] = field(default_factory=list)
    filter: List[Query] = field(default_factory=list)
    minimum_should_match: Optional[Union[int, str]] = None
    boost: float = 1.0

    def add_must(self, query: Query) -> "BoolQuery":
        self.must.append(query)
        return self

    def add_must_not(self, query: Query) -> "BoolQuery":
        self.must_not.append(query)
        return self

    def add_should(self, query: Query) -> "BoolQuery":
        self.should.append(query)
        return self

    def add_filter(self, query: Query) -> "BoolQuery":
        self.filter.append(query)
        return self

    def to_dict(self) -> Dict[str, Any]:
        bool_body: Dict[str, Any] = {}

        if self.must:
            bool_body["must"] = [q.to_dict() for q in self.must]

        if self.must_not:
            bool_body["must_not"] = [q.to_dict() for q in self.must_not]

        if self.should:
            bool_body["should"] = [q.to_dict() for q in self.should]

        if self.filter:
            bool_body["filter"] = [q.to_dict() for q in self.filter]

        if self.minimum_should_match is not None:
            bool_body["minimum_should_match"] = self.minimum_should_match

        if self.boost != 1.0:
            bool_body["boost"] = self.boost

        return {"bool": bool_body}


@dataclass
class NestedQuery(Query):
    """Nested object query."""

    path: str
    query: Query
    score_mode: str = "avg"  # avg, max, min, sum, none

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nested": {
                "path": self.path,
                "query": self.query.to_dict(),
                "score_mode": self.score_mode,
            }
        }


class QueryBuilder:
    """Fluent query builder.

    Example:
        query = (QueryBuilder()
            .must(MatchQuery("name", "test"))
            .filter(TermQuery("status", "active"))
            .should(MatchQuery("tags", "important"))
            .build())
    """

    def __init__(self):
        self._bool_query = BoolQuery()

    def must(self, query: Query) -> "QueryBuilder":
        """Add must clause."""
        self._bool_query.add_must(query)
        return self

    def must_not(self, query: Query) -> "QueryBuilder":
        """Add must_not clause."""
        self._bool_query.add_must_not(query)
        return self

    def should(self, query: Query) -> "QueryBuilder":
        """Add should clause."""
        self._bool_query.add_should(query)
        return self

    def filter(self, query: Query) -> "QueryBuilder":
        """Add filter clause."""
        self._bool_query.add_filter(query)
        return self

    def minimum_should_match(self, value: Union[int, str]) -> "QueryBuilder":
        """Set minimum_should_match."""
        self._bool_query.minimum_should_match = value
        return self

    def boost(self, value: float) -> "QueryBuilder":
        """Set boost."""
        self._bool_query.boost = value
        return self

    def match(self, field: str, query: str, **kwargs: Any) -> "QueryBuilder":
        """Add match query to must."""
        return self.must(MatchQuery(field=field, query=query, **kwargs))

    def term(self, field: str, value: Any, **kwargs: Any) -> "QueryBuilder":
        """Add term query to filter."""
        return self.filter(TermQuery(field=field, value=value, **kwargs))

    def terms(self, field: str, values: List[Any]) -> "QueryBuilder":
        """Add terms query to filter."""
        return self.filter(TermsQuery(field=field, values=values))

    def range(self, field: str, **kwargs: Any) -> "QueryBuilder":
        """Add range query to filter."""
        return self.filter(RangeQuery(field=field, **kwargs))

    def exists(self, field: str) -> "QueryBuilder":
        """Add exists query to filter."""
        return self.filter(ExistsQuery(field=field))

    def wildcard(self, field: str, value: str, **kwargs: Any) -> "QueryBuilder":
        """Add wildcard query to must."""
        return self.must(WildcardQuery(field=field, value=value, **kwargs))

    def fuzzy(self, field: str, value: str, **kwargs: Any) -> "QueryBuilder":
        """Add fuzzy query to must."""
        return self.must(FuzzyQuery(field=field, value=value, **kwargs))

    def build(self) -> Dict[str, Any]:
        """Build the query."""
        # If only one clause with one query, simplify
        if (
            len(self._bool_query.must) == 1
            and not self._bool_query.must_not
            and not self._bool_query.should
            and not self._bool_query.filter
        ):
            return self._bool_query.must[0].to_dict()

        if (
            not self._bool_query.must
            and not self._bool_query.must_not
            and not self._bool_query.should
            and not self._bool_query.filter
        ):
            return {"match_all": {}}

        return self._bool_query.to_dict()


# ============================================================================
# Convenience Functions
# ============================================================================

def match_all(boost: float = 1.0) -> Dict[str, Any]:
    """Create match_all query."""
    return MatchAllQuery(boost=boost).to_dict()


def match(field: str, query: str, **kwargs: Any) -> Dict[str, Any]:
    """Create match query."""
    return MatchQuery(field=field, query=query, **kwargs).to_dict()


def term(field: str, value: Any, **kwargs: Any) -> Dict[str, Any]:
    """Create term query."""
    return TermQuery(field=field, value=value, **kwargs).to_dict()


def terms(field: str, values: List[Any]) -> Dict[str, Any]:
    """Create terms query."""
    return TermsQuery(field=field, values=values).to_dict()


def range_query(field: str, **kwargs: Any) -> Dict[str, Any]:
    """Create range query."""
    return RangeQuery(field=field, **kwargs).to_dict()


def bool_query(
    must: Optional[List[Dict[str, Any]]] = None,
    must_not: Optional[List[Dict[str, Any]]] = None,
    should: Optional[List[Dict[str, Any]]] = None,
    filter: Optional[List[Dict[str, Any]]] = None,
    minimum_should_match: Optional[Union[int, str]] = None,
) -> Dict[str, Any]:
    """Create bool query from raw dicts."""
    bool_body: Dict[str, Any] = {}

    if must:
        bool_body["must"] = must

    if must_not:
        bool_body["must_not"] = must_not

    if should:
        bool_body["should"] = should

    if filter:
        bool_body["filter"] = filter

    if minimum_should_match is not None:
        bool_body["minimum_should_match"] = minimum_should_match

    return {"bool": bool_body}
