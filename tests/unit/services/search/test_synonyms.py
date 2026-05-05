"""
Unit tests for synonym expansion in synonyms.py.

expand_terms() is a pure function — no fixtures or mocking required.
Key behaviours tested:
- Original terms always get weight 1.0
- Synonyms get SYNONYM_WEIGHT (0.5)
- Single-word synonyms are expanded for known terms
- Multi-word query phrases trigger multi-word synonym expansion
- Tokens covered by a multi-word match are skipped in the single-word loop
"""

import pytest
from app.services.search.synonyms import expand_terms, SYNONYM_WEIGHT


@pytest.mark.unit
class TestExpandTermsSingleWord:
    def test_original_term_has_weight_one(self):
        weights = expand_terms(["diabetes"])
        assert weights["diabetes"] == 1.0

    def test_known_synonym_added_with_synonym_weight(self):
        # "diabetes" group includes "sukkersyke" (single-word)
        weights = expand_terms(["diabetes"])
        assert "sukkersyke" in weights
        assert weights["sukkersyke"] == SYNONYM_WEIGHT

    def test_multi_word_synonym_not_added_as_expansion(self):
        # "diabetes mellitus" is a 2-token phrase — must not be split and injected
        weights = expand_terms(["diabetes"])
        assert "mellitus" not in weights

    def test_unknown_term_passes_through_unchanged(self):
        weights = expand_terms(["xyzzyfoo123"])
        assert weights == {"xyzzyfoo123": 1.0}

    def test_empty_input_returns_empty_dict(self):
        assert expand_terms([]) == {}

    def test_multiple_terms_all_present(self):
        weights = expand_terms(["diabetes", "slag"])
        assert weights["diabetes"] == 1.0
        assert weights["slag"] == 1.0

    def test_single_word_synonym_of_slag(self):
        # "slag" group: "hjerneslag", "apopleksi", "cerebralt insult", "stroke"
        weights = expand_terms(["slag"])
        assert "apopleksi" in weights
        assert "stroke" in weights
        assert weights["apopleksi"] == SYNONYM_WEIGHT


@pytest.mark.unit
class TestExpandTermsMultiWord:
    def test_multi_word_match_adds_single_word_synonym(self):
        # "høyt blodtrykk" is a multi-word synonym group member
        # Its single-word synonym is "hypertensjon"
        weights = expand_terms(["høyt", "blodtrykk"])
        assert "hypertensjon" in weights
        assert weights["hypertensjon"] == SYNONYM_WEIGHT

    def test_original_tokens_keep_weight_one_after_multi_word_match(self):
        weights = expand_terms(["høyt", "blodtrykk"])
        assert weights["høyt"] == 1.0
        assert weights["blodtrykk"] == 1.0

    def test_partial_multi_word_does_not_expand(self):
        # "høyt" alone cannot complete the phrase "høyt blodtrykk"
        weights = expand_terms(["høyt"])
        assert "hypertensjon" not in weights

    def test_multi_word_match_at_end_of_list(self):
        # Multi-word phrase as the last two tokens
        weights = expand_terms(["annen", "term", "høyt", "blodtrykk"])
        assert "hypertensjon" in weights

    def test_covered_tokens_skipped_in_single_word_loop(self):
        # "høyt" and "blodtrykk" are covered by the multi-word match;
        # "slag" at index 2 is NOT covered and should still expand.
        weights = expand_terms(["høyt", "blodtrykk", "slag"])
        # Multi-word expansion
        assert "hypertensjon" in weights
        # Single-word expansion for uncovered "slag"
        assert "stroke" in weights
        # Weight check
        assert weights["hypertensjon"] == SYNONYM_WEIGHT
        assert weights["stroke"] == SYNONYM_WEIGHT

    def test_synonym_weight_not_overwritten_by_later_expansion(self):
        # "slag" and "hjerneslag" are in the same synonym group.
        # "slag" expands to include "hjerneslag" at weight 0.5,
        # but "hjerneslag" is already weight 1.0 (original term) so it stays 1.0.
        weights = expand_terms(["slag", "hjerneslag"])
        assert weights["slag"] == 1.0
        assert weights["hjerneslag"] == 1.0
