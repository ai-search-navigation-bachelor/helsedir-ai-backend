"""
Unit tests for MLService.

XGBoost and file system are mocked throughout — no model files needed.
Each test instantiates a fresh MLService() to avoid state leakage.
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from app.services.search.ml_service import MLService


# ---------------------------------------------------------------------------
# is_ranking_available
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIsRankingAvailable:
    def test_false_on_fresh_instance(self):
        assert MLService().is_ranking_available() is False

    def test_true_after_flag_set(self):
        svc = MLService()
        svc._ranking_loaded = True
        assert svc.is_ranking_available() is True


# ---------------------------------------------------------------------------
# load_ranking_model
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadRankingModel:
    def test_returns_false_when_disabled_and_not_forced(self):
        svc = MLService()
        with patch("app.services.search.ml_service.settings") as s:
            s.ml_ranking_enabled = False
            assert svc.load_ranking_model(force=False) is False

    def test_returns_false_when_model_file_missing(self, tmp_path):
        svc = MLService()
        with patch("app.services.search.ml_service.settings") as s:
            s.ml_ranking_enabled = True
            s.ml_models_dir = str(tmp_path)  # ranking/reranker.json does not exist
            assert svc.load_ranking_model() is False

    def test_returns_true_and_sets_flag_when_file_exists(self, tmp_path):
        ranking_dir = tmp_path / "ranking"
        ranking_dir.mkdir()
        (ranking_dir / "reranker.json").write_text("{}")

        mock_ranker = MagicMock()
        svc = MLService()

        with patch("app.services.search.ml_service.settings") as s, \
             patch("app.services.search.ml_service.HealthContentRanker.load", return_value=mock_ranker), \
             patch.object(svc, "_refresh_ctr_cache"):
            s.ml_ranking_enabled = True
            s.ml_models_dir = str(tmp_path)
            result = svc.load_ranking_model()

        assert result is True
        assert svc._ranking_loaded is True
        assert svc.ranking_model is mock_ranker

    def test_returns_false_on_load_exception(self, tmp_path):
        ranking_dir = tmp_path / "ranking"
        ranking_dir.mkdir()
        (ranking_dir / "reranker.json").write_text("{}")

        svc = MLService()
        with patch("app.services.search.ml_service.settings") as s, \
             patch("app.services.search.ml_service.HealthContentRanker.load",
                   side_effect=RuntimeError("corrupt model")):
            s.ml_ranking_enabled = True
            s.ml_models_dir = str(tmp_path)
            result = svc.load_ranking_model()

        assert result is False
        assert svc._ranking_loaded is False


# ---------------------------------------------------------------------------
# get_ranking_scores
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetRankingScores:
    def test_returns_zeros_when_no_model(self):
        svc = MLService()
        scores = svc.get_ranking_scores([{"semantic_score": 0.5}])
        assert scores == [0.0]

    def test_returns_zeros_for_multiple_features_when_no_model(self):
        svc = MLService()
        scores = svc.get_ranking_scores([{"x": 1.0}, {"x": 2.0}, {"x": 3.0}])
        assert scores == [0.0, 0.0, 0.0]

    def test_delegates_to_ranking_model_predict(self):
        svc = MLService()
        mock_ranker = MagicMock()
        mock_ranker.predict.return_value = [0.9, 0.6]
        svc.ranking_model = mock_ranker

        features = [{"semantic_score": 0.9}, {"semantic_score": 0.4}]
        scores = svc.get_ranking_scores(features)

        assert scores == [0.9, 0.6]
        mock_ranker.predict.assert_called_once_with(features)

    def test_returns_zeros_on_predict_exception(self):
        svc = MLService()
        mock_ranker = MagicMock()
        mock_ranker.predict.side_effect = RuntimeError("broken")
        svc.ranking_model = mock_ranker

        scores = svc.get_ranking_scores([{"x": 1.0}])
        assert scores == [0.0]


# ---------------------------------------------------------------------------
# get_ranking_scores_with_contributions
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetRankingScoresWithContributions:
    def test_returns_zero_tuples_when_no_model(self):
        svc = MLService()
        result = svc.get_ranking_scores_with_contributions([{"x": 0.5}])
        assert result == [(0.0, {})]

    def test_delegates_to_predict_with_contributions(self):
        svc = MLService()
        mock_ranker = MagicMock()
        mock_ranker.predict_with_contributions.return_value = [
            (0.9, {"bm25_score": 0.4, "semantic_score": 0.5})
        ]
        svc.ranking_model = mock_ranker

        result = svc.get_ranking_scores_with_contributions([{"bm25_score": 0.4}])

        assert result[0][0] == pytest.approx(0.9)
        assert "bm25_score" in result[0][1]

    def test_returns_zero_tuples_on_exception(self):
        svc = MLService()
        mock_ranker = MagicMock()
        mock_ranker.predict_with_contributions.side_effect = RuntimeError("shap error")
        svc.ranking_model = mock_ranker

        result = svc.get_ranking_scores_with_contributions([{"x": 1.0}])
        assert result == [(0.0, {})]


# ---------------------------------------------------------------------------
# get_ctr_map (caching behaviour)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetCTRMap:
    def test_returns_dict(self):
        svc = MLService()
        # Force cache to be "fresh" so no DB call is made
        svc._ctr_last_refresh = time.monotonic()
        svc._ctr_map = {}
        result = svc.get_ctr_map()
        assert isinstance(result, dict)

    def test_returns_cached_map_without_refresh_when_fresh(self):
        svc = MLService()
        svc._ctr_map = {"001": 0.15, "002": 0.08}
        svc._ctr_last_refresh = time.monotonic()  # just refreshed — TTL not expired

        result = svc.get_ctr_map()
        assert result == {"001": 0.15, "002": 0.08}

    def test_refreshes_from_db_when_ttl_expired(self):
        svc = MLService()
        svc._ctr_last_refresh = float("-inf")  # always expired regardless of system uptime

        with patch(
            "app.services.data.database_service.database_service.get_content_ctr_map",
            return_value={"003": 0.25},
        ):
            result = svc.get_ctr_map()

        assert result.get("003") == pytest.approx(0.25)

    def test_ctr_map_is_empty_dict_on_db_exception(self):
        svc = MLService()
        svc._ctr_last_refresh = float("-inf")  # always expired regardless of system uptime

        with patch(
            "app.services.data.database_service.database_service.get_content_ctr_map",
            side_effect=RuntimeError("db down"),
        ):
            result = svc.get_ctr_map()

        assert isinstance(result, dict)
