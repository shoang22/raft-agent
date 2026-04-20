"""Linear regression adapter for predicting missing order totals."""
import abc
import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

from raft_agent.domain.models import Order, USState

logger = logging.getLogger(__name__)

_STATE_INDEX: dict[str, int] = {s.value: i for i, s in enumerate(USState)}
_N_STATES = len(USState)

DEFAULT_MODEL_PATH = "total_predictor.joblib"


def _featurize(order: Order) -> np.ndarray:
    vec = np.zeros(_N_STATES)
    idx = _STATE_INDEX.get(order.state.value, -1)
    if idx >= 0:
        vec[idx] = 1.0
    return vec


class AbstractTotalPredictor(abc.ABC):
    @abc.abstractmethod
    def predict(self, order: Order) -> Optional[float]:
        """Return a predicted total, or None if the model is not yet trained."""
        raise NotImplementedError

    @abc.abstractmethod
    def retrain(self, orders: list[Order]) -> None:
        """Retrain the model synchronously on the given orders."""
        raise NotImplementedError

    @abc.abstractmethod
    async def retrain_async(self, orders: list[Order]) -> None:
        """Kick off retraining without blocking the event loop."""
        raise NotImplementedError


class LinearRegressionTotalPredictor(AbstractTotalPredictor):
    """One-hot encodes US state and fits a linear regression to predict total.

    The fitted model is persisted to `model_path` after every retrain and loaded
    from that path on construction so predictions survive process restarts.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self._model_path = Path(model_path)
        self._lock = threading.Lock()
        self._model: Optional[LinearRegression] = self._load()

    def _load(self) -> Optional[LinearRegression]:
        if self._model_path.exists():
            try:
                model = joblib.load(self._model_path)
                logger.info("Loaded model from %s", self._model_path)
                return model
            except Exception:
                logger.exception("Failed to load model from %s; starting untrained", self._model_path)
        return None

    def _save(self, model: LinearRegression) -> None:
        try:
            joblib.dump(model, self._model_path)
            logger.info("Saved model to %s", self._model_path)
        except Exception:
            logger.exception("Failed to save model to %s", self._model_path)

    def predict(self, order: Order) -> Optional[float]:
        with self._lock:
            if self._model is None:
                return None
            features = _featurize(order).reshape(1, -1)
            raw = float(self._model.predict(features)[0])
            return round(max(raw, 0.0), 2)

    def retrain(self, orders: list[Order]) -> None:
        training = [o for o in orders if o.total is not None]
        if len(training) < 2:
            logger.warning("Skipping retrain: need ≥2 orders with known totals, got %d", len(training))
            return
        X = np.array([_featurize(o) for o in training])
        y = np.array([o.total for o in training])
        model = LinearRegression()
        model.fit(X, y)
        self._save(model)
        with self._lock:
            self._model = model
        logger.info("Model retrained on %d orders", len(training))

    async def retrain_async(self, orders: list[Order]) -> None:
        """Run retrain in a thread pool without blocking the event loop (fire-and-forget)."""
        asyncio.create_task(asyncio.to_thread(self.retrain, orders))
