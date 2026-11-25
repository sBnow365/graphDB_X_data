"""Project exception hierarchy. Catch these instead of bare ``Exception``."""


class GELMError(Exception):
    """Base class for all project errors."""


class ConfigError(GELMError):
    """Missing or invalid configuration."""


class DatasetError(GELMError):
    """Dataset missing, unreadable or failing validation."""


class GraphConnectionError(GELMError):
    """Neo4j unreachable or authentication failed."""


class GDSUnavailableError(GELMError):
    """Graph Data Science plugin or a required procedure is missing."""


class UnsafeQueryError(GELMError):
    """A Cypher query was rejected by the safety validator."""


class LLMError(GELMError):
    """The LLM backend failed or returned unusable output."""
