"""Config parsing for HINDSIGHT_API_ENTITY_TRGM_SIMILARITY_THRESHOLD.

Static (server-level) float applied as ``SET pg_trgm.similarity_threshold`` on
every pool connection. pg_trgm only accepts a threshold in (0, 1], so an
out-of-range value must fail fast at config load rather than break every
connection's setup. These tests pin the default, the parse, and the bounds.
"""

import pytest

from hindsight_api.config import (
    DEFAULT_ENTITY_MERGE_MIN_SIMILARITY,
    DEFAULT_ENTITY_TRGM_SIMILARITY_THRESHOLD,
    ENV_ENTITY_MERGE_MIN_SIMILARITY,
    ENV_ENTITY_TRGM_SIMILARITY_THRESHOLD,
    HindsightConfig,
)


class TestEntityTrgmThresholdConfig:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv(ENV_ENTITY_TRGM_SIMILARITY_THRESHOLD, raising=False)
        config = HindsightConfig.from_env()
        assert config.entity_trgm_similarity_threshold == DEFAULT_ENTITY_TRGM_SIMILARITY_THRESHOLD

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(ENV_ENTITY_TRGM_SIMILARITY_THRESHOLD, "0.3")
        config = HindsightConfig.from_env()
        assert config.entity_trgm_similarity_threshold == 0.3

    def test_upper_bound_one_is_valid(self, monkeypatch):
        monkeypatch.setenv(ENV_ENTITY_TRGM_SIMILARITY_THRESHOLD, "1.0")
        config = HindsightConfig.from_env()
        assert config.entity_trgm_similarity_threshold == 1.0

    @pytest.mark.parametrize("value", ["0", "0.0", "-0.1", "1.5"])
    def test_out_of_range_fails_fast(self, monkeypatch, value):
        monkeypatch.setenv(ENV_ENTITY_TRGM_SIMILARITY_THRESHOLD, value)
        with pytest.raises(ValueError, match="entity_trgm_similarity_threshold"):
            HindsightConfig.from_env()


class TestEntityTrgmProbeThreshold:
    """The probe threshold the pool actually sets is never below the merge floor.

    The resolver drops every candidate whose trigram similarity is under
    ``entity_merge_min_similarity``, so a probe admitted below it fetches rows only to
    discard them -- 73% of the rows on a live 8.5k-entity bank, at 3.5x the query time.
    """

    def test_defaults_probe_at_the_merge_floor(self, monkeypatch):
        monkeypatch.delenv(ENV_ENTITY_TRGM_SIMILARITY_THRESHOLD, raising=False)
        monkeypatch.delenv(ENV_ENTITY_MERGE_MIN_SIMILARITY, raising=False)
        config = HindsightConfig.from_env()
        assert config.entity_trgm_probe_threshold == DEFAULT_ENTITY_MERGE_MIN_SIMILARITY

    def test_lowered_merge_floor_lowers_the_probe(self, monkeypatch):
        monkeypatch.delenv(ENV_ENTITY_TRGM_SIMILARITY_THRESHOLD, raising=False)
        monkeypatch.setenv(ENV_ENTITY_MERGE_MIN_SIMILARITY, "0.2")
        config = HindsightConfig.from_env()
        assert config.entity_trgm_probe_threshold == 0.2

    def test_stricter_probe_is_kept(self, monkeypatch):
        monkeypatch.setenv(ENV_ENTITY_TRGM_SIMILARITY_THRESHOLD, "0.5")
        monkeypatch.delenv(ENV_ENTITY_MERGE_MIN_SIMILARITY, raising=False)
        config = HindsightConfig.from_env()
        assert config.entity_trgm_probe_threshold == 0.5
