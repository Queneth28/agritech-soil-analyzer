"""
Helpers shared by train_model.py and app.py for the crop suitability model.

The model is a multi-output regressor: one suitability score (0-1) per crop.
Two shapes are supported:
  - RandomForestRegressor (native multi-output, one set of trees)
  - sklearn MultiOutputRegressor wrapping one LightGBM model per crop
"""

import numpy as np


def feature_importance(model):
    """Normalised feature importance averaged over all crop outputs."""
    if hasattr(model, 'estimators_') and hasattr(model, 'estimator'):  # MultiOutputRegressor
        per_crop = []
        for est in model.estimators_:
            imp = np.asarray(est.feature_importances_, dtype=float)
            per_crop.append(imp / imp.sum() if imp.sum() > 0 else imp)
        imp = np.mean(per_crop, axis=0)
    else:
        imp = np.asarray(model.feature_importances_, dtype=float)
    return imp / imp.sum() if imp.sum() > 0 else imp


class CropShapExplainer:
    """Lazily builds TreeExplainers and returns SHAP values for one crop output."""

    def __init__(self, model):
        import shap  # optional dependency, imported only when explanations are used
        self._shap = shap
        self.model = model
        self._explainers = {}

    def _explainer(self, key, target):
        if key not in self._explainers:
            self._explainers[key] = self._shap.TreeExplainer(target)
        return self._explainers[key]

    def explain(self, X, output_idx):
        """SHAP values (n_features,) for the first row of X and one crop output."""
        if hasattr(self.model, 'estimators_') and hasattr(self.model, 'estimator'):
            est = self.model.estimators_[output_idx]
            return np.asarray(self._explainer(output_idx, est).shap_values(X))[0]
        sv = self._explainer('all', self.model).shap_values(X)
        if isinstance(sv, list):                 # list of (n, features) per output
            return np.asarray(sv[output_idx])[0]
        sv = np.asarray(sv)                      # (n, features, outputs)
        return sv[0, :, output_idx]
