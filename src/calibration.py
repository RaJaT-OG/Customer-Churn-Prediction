"""
calibration.py
===============
Standalone module for the CalibratedModel wrapper class.

IMPORTANT — why this lives in its own file:
Pickle records a class's import path as whatever module it was defined in
at *save* time. If CalibratedModel were defined inside
unified_churn_pipeline.py and that file were ever run directly
(`python -m src.unified_churn_pipeline` or `python unified_churn_pipeline.py`),
Python would execute it as `__main__`, and joblib/pickle would record the
class as belonging to `__main__` — not `src.unified_churn_pipeline`. Any
other script (like app.py, which has its own separate `__main__`) would
then fail to unpickle it with:

    AttributeError: Can't get attribute 'CalibratedModel' on <module '__main__'>

Keeping the class in its own module that is never executed as a script
guarantees a stable import path (`src.calibration.CalibratedModel`) no
matter how training is invoked, and no matter what file loads the model
later.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


class CalibratedModel:
    """
    Wraps a fitted classifier + a fitted IsotonicRegression calibrator.
    predict_proba() returns isotonic-calibrated probabilities instead of
    the raw (SMOTE-biased) ones.

    We use this instead of sklearn's CalibratedClassifierCV(cv="prefit")
    or the newer FrozenEstimator wrapper because both embed sklearn-version-
    specific internals into the pickle, which breaks if the model is
    unpickled in a different Python environment (different sklearn version)
    than it was trained in. This wrapper only depends on plain
    IsotonicRegression, which has been stable across sklearn versions for
    years.
    """

    def __init__(self, base_model, calibrator: IsotonicRegression):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw_pos = self.base_model.predict_proba(X)[:, 1]
        calib_pos = np.clip(self.calibrator.predict(raw_pos), 0.0, 1.0)
        return np.column_stack([1.0 - calib_pos, calib_pos])

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
