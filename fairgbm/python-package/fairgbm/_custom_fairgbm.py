import pandas as pd
import numpy as np
from fairgbm import FairGBMClassifier


class FairGBMCustom:
    def __init__(self, *, fairness_vars, **gbm_kwargs):
        self.fairness_vars = fairness_vars
        # só passa kwargs que o FairGBM conhece
        self._inner = FairGBMClassifier(**gbm_kwargs)

    def fit(self, X, y, **fit_kwargs):
        sen = self.fairness_vars["sen_var"]
        unpriv = {d[sen] for d in self.fairness_vars["unprivileged_groups"]}
        S = pd.Series(np.where(X[sen].isin(unpriv), 1, 0), index=X.index)
        return self._inner.fit(X, y, constraint_group=S, **fit_kwargs)

    def __getattr__(self, name):
        # delega tudo o mais para o modelo interno
        return getattr(self._inner, name)
