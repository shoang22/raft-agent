"""Unit tests for LinearRegressionTotalPredictor persistence (save/load)."""
import pytest

from raft_agent.adapters.ml_model import LinearRegressionTotalPredictor
from raft_agent.domain.models import Order

_TRAINING_ORDERS = [
    Order(orderId="1", buyer="Alice", state="OH", total=500.0),
    Order(orderId="2", buyer="Bob", state="TX", total=200.0),
    Order(orderId="3", buyer="Carol", state="CA", total=800.0),
]

_PREDICT_ORDER = Order(orderId="99", buyer="Dave", state="OH", total=None)


class TestLinearRegressionPersistence:
    def test_untrained_predictor_returns_none(self, tmp_path):
        p = LinearRegressionTotalPredictor(model_path=str(tmp_path / "model.joblib"))
        assert p.predict(_PREDICT_ORDER) is None

    def test_retrain_saves_model_to_disk(self, tmp_path):
        path = tmp_path / "model.joblib"
        p = LinearRegressionTotalPredictor(model_path=str(path))
        p.retrain(_TRAINING_ORDERS)
        assert path.exists()

    def test_loaded_model_can_predict(self, tmp_path):
        path = tmp_path / "model.joblib"
        p1 = LinearRegressionTotalPredictor(model_path=str(path))
        p1.retrain(_TRAINING_ORDERS)

        p2 = LinearRegressionTotalPredictor(model_path=str(path))
        result = p2.predict(_PREDICT_ORDER)

        assert result is not None
        assert isinstance(result, float)
        assert result >= 0.0

    def test_loaded_model_matches_trained_model_predictions(self, tmp_path):
        path = tmp_path / "model.joblib"
        p1 = LinearRegressionTotalPredictor(model_path=str(path))
        p1.retrain(_TRAINING_ORDERS)
        prediction_before = p1.predict(_PREDICT_ORDER)

        p2 = LinearRegressionTotalPredictor(model_path=str(path))
        prediction_after = p2.predict(_PREDICT_ORDER)

        assert prediction_before == prediction_after

    def test_retrain_below_minimum_does_not_save(self, tmp_path):
        path = tmp_path / "model.joblib"
        p = LinearRegressionTotalPredictor(model_path=str(path))
        p.retrain([_TRAINING_ORDERS[0]])  # only 1 order — below minimum of 2
        assert not path.exists()

    def test_corrupt_model_file_starts_untrained(self, tmp_path):
        path = tmp_path / "model.joblib"
        path.write_bytes(b"not a valid joblib file")

        p = LinearRegressionTotalPredictor(model_path=str(path))
        assert p.predict(_PREDICT_ORDER) is None
