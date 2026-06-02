"""
Scikit-learn style wrapper for the original "Wasserstein Fair Classification" codebase.

Design goals:
- Keep the author's files (basic_costs.py / combined_costs.py / optimal_transport.py) unchanged.
- Implement ONLY what is missing for sklearn-like usage: fit / predict_proba / predict,
  plus barycenter computation exactly as described in the paper:
    - POT (Flamary & Courty, 2017)
    - fixed support bins on [0,1]
    - iterative KL-projection method (Benamou et al., 2015)
    - sample barycenter atoms {s̄_i} from the computed barycenter distribution.

USAGE with run_experiment_baseline:
    
    estimator_kwargs = {
        "fairness_vars": fairness_vars,
        "awareness": False,  # False = demographically blind (não usa sen_var como feature)
        "beta": 30.0,
        "alpha": 0.5,
        "max_iter": 3000,
        "lr": 0.01,
    }
    
    results = run_experiment_baseline(
        df=df,
        target_name=TARGET_NAME,
        sen_var=sen_var,
        awareness=True,  # Utilização do atributo sensível pelo classificador. Verdadeiro para a referencia.
        fold_index=fold_index,
        exp_id="wasserstein_fair",
        estimator_class=WessersteinFairClassifier,
        estimator_kwargs=estimator_kwargs,
        fairness_vars=fairness_vars,
    )
"""

from __future__ import annotations

import sys
import types
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


def _import_author_modules():
    """
    Import author's modules robustly under both:
      1) a package layout `wasserstein_fairness.*`
      2) flat files in the working directory
    and ensure `combined_costs.py` can import `basic_costs` and `optimal_transport`.
    """
    try:
        # Preferred: author/package layout
        from wasserstein_fairness import basic_costs, optimal_transport, combined_costs  # type: ignore
        return basic_costs, optimal_transport, combined_costs
    except Exception:
        # Flat-file layout: create a lightweight package alias so that
        # `from wasserstein_fairness import basic_costs` inside combined_costs works.
        import importlib

        basic_costs = importlib.import_module("basic_costs")
        optimal_transport = importlib.import_module("optimal_transport")

        pkg = sys.modules.get("wasserstein_fairness")
        if pkg is None:
            pkg = types.ModuleType("wasserstein_fairness")
            sys.modules["wasserstein_fairness"] = pkg
        pkg.basic_costs = basic_costs
        pkg.optimal_transport = optimal_transport

        combined_costs = importlib.import_module("combined_costs")
        return basic_costs, optimal_transport, combined_costs


basic_costs, optimal_transport, combined_costs = _import_author_modules()


def _ensure_basic_costs_api():
    """
    combined_costs expects these exact symbols:
      - basic_costs.wasserstein_one_loss_gradient
      - basic_costs.wass_barycenter_loss_gradient

    Some versions of the author's file expose:
      - wasserstein_one_loss_gradient_method_one / method_two
      - wass_one_barycenter_loss_gradient

    We add aliases in-memory, without editing author files.
    """
    if not hasattr(basic_costs, "wasserstein_one_loss_gradient"):
        if hasattr(basic_costs, "wasserstein_one_loss_gradient_method_one"):
            basic_costs.wasserstein_one_loss_gradient = basic_costs.wasserstein_one_loss_gradient_method_one
        elif hasattr(basic_costs, "wasserstein_one_loss_gradient_method_two"):
            basic_costs.wasserstein_one_loss_gradient = basic_costs.wasserstein_one_loss_gradient_method_two
        else:
            raise AttributeError(
                "basic_costs is missing Wasserstein-1 gradient function. "
                "Expected 'wasserstein_one_loss_gradient' or '*_method_one/two'."
            )

    if not hasattr(basic_costs, "wass_barycenter_loss_gradient"):
        if hasattr(basic_costs, "wass_one_barycenter_loss_gradient"):
            basic_costs.wass_barycenter_loss_gradient = basic_costs.wass_one_barycenter_loss_gradient
        else:
            raise AttributeError(
                "basic_costs is missing barycenter gradient function. "
                "Expected 'wass_barycenter_loss_gradient' or 'wass_one_barycenter_loss_gradient'."
            )


def _as_list(x: Union[str, Sequence[str], None]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Use author's sigmoid if available (keeps behavior consistent)
    if hasattr(basic_costs, "sigmoid"):
        return basic_costs.sigmoid(z)
    # fallback
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class _BarycenterResult:
    bins: np.ndarray          # edges
    centers: np.ndarray       # bin centers
    pbar: np.ndarray          # barycenter distribution over centers
    samples: np.ndarray       # sampled atoms (baryscore)


def _compute_baryscore_pot(
    group_scores: Sequence[np.ndarray],
    group_weights: np.ndarray,
    *,
    n_bins: int,
    reg: float,
    n_samples: int,
    bary_max_iter: int = 1000,  # Paper: "number of iterations M = 1000"
    random_state: Optional[int],
    eps: float = 1e-12,
) -> _BarycenterResult:
    """
    Compute the empirical barycenter distribution and sample atoms, as described
    in Wasserstein Fair Classification (Algorithm 1 implementation details):
      - Use POT (Flamary & Courty, 2017)
      - Fix barycenter support to equal-width bins on [0,1]
      - Use iterative KL-projection method (Benamou et al., 2015)
      - Sample atoms from the computed barycenter distribution

    This requires the POT library (`pip install POT`).
    """
    try:
        import ot  # type: ignore
    except Exception as e:
        raise ImportError(
            "POT library not found. To match the paper's barycenter procedure, install POT:\n"
            "  pip install POT\n"
        ) from e

    # Support on [0,1] as in the paper
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0

    # Build per-group histograms on the shared support
    cols = []
    for s in group_scores:
        s = np.asarray(s, dtype=float).reshape(-1)
        s = np.clip(s, 0.0, 1.0)
        h, _ = np.histogram(s, bins=bins)
        p = h.astype(float) + eps
        p /= p.sum()
        cols.append(p)

    A = np.stack(cols, axis=1)  # shape (n_bins, n_groups)
    w = np.asarray(group_weights, dtype=float).reshape(-1)
    w = w / w.sum()

    # Cost matrix for Wass-1 on the fixed support
    M = np.abs(centers[:, None] - centers[None, :])

    # POT barycenter via iterative Bregman projections (KL-projection)
    # Paper: "used number of iterations M = 1000"
    pbar = ot.bregman.barycenter(A, M, reg, w, numItermax=bary_max_iter)
    pbar = np.asarray(pbar, dtype=float).reshape(-1)
    pbar = np.maximum(pbar, 0.0)
    pbar = pbar / (pbar.sum() + eps)

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(centers), size=int(n_samples), replace=True, p=pbar)
    samples = centers[idx].astype(float)

    return _BarycenterResult(bins=bins, centers=centers, pbar=pbar, samples=samples)


class WessersteinFairClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn style classifier implementing Wasserstein Fair Classification
    (Jiang et al., UAI 2019):

      minimize  alpha * LogisticLoss(theta)
             + (1-alpha) * beta * sum_a p_a * W1( p(S|A=a), p(S̄) )

    where S are model beliefs (sigmoid scores) and S̄ is the Wasserstein-1 barycenter
    of group score distributions.

    Parameters
    ----------
    fairness_vars : dict
        Dictionary with keys:
        - "sen_var": str - name of sensitive variable column
        - "privileged_groups": list of dicts - e.g., [{"gender": 1}]
        - "unprivileged_groups": list of dicts - e.g., [{"gender": 0}]
    
    awareness : bool, default=True
        If True, includes sensitive columns as features (standard Wass-1 Penalty).
        If False, excludes sensitive columns (Demographically-Blind variant).
    
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    
    beta : float, default=30.0
        Penalization weight for the Wasserstein fairness loss.
    
    alpha : float, default=0.5
        Tradeoff between logistic loss and Wasserstein loss:
        loss = alpha * logistic_loss + (1-alpha) * beta * wasserstein_loss
    
    lr : float, default=0.01
        Learning rate for gradient descent.
    
    max_iter : int, default=3000
        Maximum number of gradient descent iterations.
    
    init_from_lr : bool, default=True
        If True, initialize theta from a pre-trained LogisticRegression
        (as described in paper Section 5.1).
    
    barycenter : bool, default=True
        Whether to use barycenter-based Wasserstein-1 loss.
    
    bary_every : int, default=20
        Compute barycenter every K steps (K in Algorithm 1).
    
    bary_bins : int, default=100
        Number of bins for barycenter support on [0,1].
    
    bary_reg : float, default=1e-2
        Entropic regularization for POT barycenter computation.
    
    verbose : bool, default=False
        Print training progress.

    References
    ----------
    Ray Jiang, Aldo Pacchiano, Tom Stepleton, Heinrich Jiang, Silvia Chiappa.
    "Wasserstein Fair Classification." UAI 2019.
    """

    def __init__(
        self,
        *,
        fairness_vars: Dict[str, Any],
        awareness: bool = True,
        fit_intercept: bool = True,
        scaler: Any = None,
        beta: float = 30.0,
        alpha: float = 0.5,
        distance: str = "wasserstein-1",
        delta: float = 0.005,
        solver: str = "line",         # for Wass-1, this should be "line" (sparse OT on R)
        lambda_: float = 0.001,        # only used for smoothed/sinkhorn solver
        lr: float = 0.01,
        max_iter: int = 3000,
        verbose: bool = False,
        # barycenter config
        barycenter: bool = True,
        bary_every: int = 20,          # K in Algorithm 1
        bary_bins: int = 100,          # fixed support bins on [0,1]
        bary_reg: float = 1e-2,        # entropic regularization for POT barycenter
        bary_n_samples: int = 200,     # N̄ samples from p̂(S̄)
        bary_max_iter: int = 1000,     # Paper: "number of iterations M = 1000" for POT
        random_state: Optional[int] = None,
        # initialization
        init_from_lr: bool = True,     # initialize from LogisticRegression (paper sec 5.1)
        # strictness
        require_all_groups_present: bool = False,
    ):
        self.fairness_vars = fairness_vars
        self.awareness = awareness
        self.fit_intercept = fit_intercept
        self.scaler = scaler

        self.beta = beta
        self.alpha = alpha
        self.distance = distance
        self.delta = delta
        self.solver = solver
        self.lambda_ = lambda_
        self.lr = lr
        self.max_iter = max_iter
        self.verbose = verbose

        self.barycenter = barycenter
        self.bary_every = bary_every
        self.bary_bins = bary_bins
        self.bary_reg = bary_reg
        self.bary_n_samples = bary_n_samples
        self.bary_max_iter = bary_max_iter
        self.random_state = random_state
        
        self.init_from_lr = init_from_lr
        self.require_all_groups_present = require_all_groups_present

    # ---------------------- internal helpers ----------------------

    def _get_sensitive_cols(self) -> List[str]:
        return _as_list(self.fairness_vars.get("sen_var"))

    def _get_group_defs(self) -> List[Dict[str, Any]]:
        return list(self.fairness_vars.get("privileged_groups", [])) + list(
            self.fairness_vars.get("unprivileged_groups", [])
        )

    def _compute_group_masks(self, X_df: pd.DataFrame) -> List[np.ndarray]:
        """Compute boolean masks for each protected group."""
        groups = self._get_group_defs()
        if not groups:
            raise ValueError(
                "fairness_vars must include privileged_groups/unprivileged_groups "
                "with at least one group dict."
            )

        masks: List[np.ndarray] = []
        for g in groups:
            m = np.ones(len(X_df), dtype=bool)
            for col, val in g.items():
                if col not in X_df.columns:
                    raise KeyError(
                        f"Sensitive/group column '{col}' not found in X_df. "
                        f"Available columns: {list(X_df.columns)}. "
                        f"Make sure to pass awareness=True to run_experiment_baseline "
                        f"so the sensitive column is preserved in X."
                    )
                m &= (X_df[col].to_numpy() == val)
            masks.append(m)
        return masks

    def _get_feature_cols(self, X_df: pd.DataFrame) -> List[str]:
        """Get feature columns (optionally excluding sensitive columns)."""
        sens = set(self._get_sensitive_cols())
        if self.awareness:
            return list(X_df.columns)
        return [c for c in X_df.columns if c not in sens]

    def _prepare_X(self, X_df: pd.DataFrame, fit: bool) -> np.ndarray:
        """Prepare feature matrix for model training/prediction."""
        X = X_df[self.feature_cols_].to_numpy(copy=False)
        # force numeric
        try:
            X = X.astype(float)
        except Exception as e:
            raise TypeError(
                "All model feature columns must be numeric (castable to float). "
                f"Non-numeric columns in feature_cols_: {self.feature_cols_}"
            ) from e

        if self.scaler is None:
            return X

        if fit:
            self.scaler_ = self.scaler
            return self.scaler_.fit_transform(X)
        return self.scaler_.transform(X)

    def _prepare_y(self, y: Any) -> np.ndarray:
        """Prepare target vector (ensure binary 0/1)."""
        y_arr = np.asarray(y).reshape(-1)
        uniq = np.unique(y_arr)

        if set(uniq.tolist()) <= {0, 1}:
            return y_arr.astype(float)
        if set(uniq.tolist()) <= {-1, 1}:
            return ((y_arr + 1.0) / 2.0).astype(float)
        if y_arr.dtype == bool:
            return y_arr.astype(float)

        raise ValueError(f"y must be binary (0/1 or -1/1). Found classes: {uniq.tolist()}")

    def _initialize_theta(
        self, X_model: np.ndarray, y_arr: np.ndarray, n_features: int
    ) -> np.ndarray:
        """
        Initialize theta, optionally from a pre-trained LogisticRegression.
        Paper Section 5.1: "as initial model parameters θ0 we used the ones 
        given by the trained logistic regression."
        """
        theta_len = n_features + 1 if self.fit_intercept else n_features
        
        if self.init_from_lr:
            try:
                lr_model = LogisticRegression(
                    fit_intercept=self.fit_intercept,
                    max_iter=1000,
                    random_state=self.random_state,
                    solver='lbfgs'
                )
                lr_model.fit(X_model, y_arr)
                
                if self.fit_intercept:
                    theta = np.zeros(theta_len, dtype=float)
                    theta[:-1] = lr_model.coef_.ravel()
                    theta[-1] = lr_model.intercept_[0]
                else:
                    theta = lr_model.coef_.ravel().copy()
                
                if self.verbose:
                    print(f"Initialized theta from LogisticRegression")
                return theta
            except Exception as e:
                if self.verbose:
                    print(f"Failed to initialize from LR: {e}. Using zeros.")
                return np.zeros(theta_len, dtype=float)
        
        return np.zeros(theta_len, dtype=float)

    # ---------------------- public sklearn API ----------------------

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Any):
        """
        Fit the Wasserstein Fair Classifier.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. MUST be a DataFrame containing the sensitive column(s)
            for computing group masks. Use awareness=True in run_experiment_baseline.
        
        y : array-like
            Binary target (0/1 or -1/1).

        Returns
        -------
        self
        """
        _ensure_basic_costs_api()

        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            warnings.warn(
                "X is a numpy array. WessersteinFairClassifier works best with "
                "pandas DataFrame containing the sensitive column. Attempting to "
                "create DataFrame with generic column names."
            )
            X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        if self.distance != "wasserstein-1":
            raise ValueError(
                "This wrapper targets Wasserstein Fair Classification; "
                "set distance='wasserstein-1'."
            )

        if self.solver != "line":
            warnings.warn(
                "For Wasserstein-1 in the paper, solver='line' is recommended "
                "(hard/sparse OT on the real line). Current solver: '{}'.".format(self.solver)
            )

        if self.barycenter and self.beta <= 0:
            warnings.warn(
                "barycenter=True but beta<=0; fairness penalty is disabled."
            )

        # Store column information
        self.sensitive_cols_ = self._get_sensitive_cols()
        self.group_defs_ = self._get_group_defs()
        self.feature_cols_ = self._get_feature_cols(X_df)
        self.all_columns_ = list(X_df.columns)

        y_arr = self._prepare_y(y)

        # Compute group masks using the full DataFrame (including sensitive cols)
        masks = self._compute_group_masks(X_df)

        # Prepare model features (may exclude sensitive cols if awareness=False)
        X_model = self._prepare_X(X_df, fit=True)

        # Build protected datasets in model feature space
        x_protected: List[np.ndarray] = []
        kept_groups: List[Dict[str, Any]] = []
        missing_groups: List[Dict[str, Any]] = []
        
        for g, m in zip(self.group_defs_, masks):
            if np.any(m):
                x_protected.append(X_model[m])
                kept_groups.append(g)
            else:
                missing_groups.append(g)

        self.missing_groups_ = missing_groups
        self.group_defs_ = kept_groups
        
        if self.require_all_groups_present and missing_groups:
            raise ValueError(f"Some groups have 0 samples in this fold: {missing_groups}")

        if len(x_protected) < 1:
            raise ValueError("No protected groups with samples; cannot compute fairness loss.")

        # Initialize theta
        n_features = X_model.shape[1]
        self.theta_ = self._initialize_theta(X_model, y_arr, n_features)

        self.history_ = []
        self.barycenter_result_ = None
        self.baryscore_ = None
        
        # Persistent baryscore that survives between recomputations
        # Paper: "compute the barycenter distribution once every K steps"
        # The barycenter is REUSED until the next recomputation
        current_baryscore = None

        # Training loop (Algorithm 1 from the paper)
        for i in range(int(self.max_iter)):
            
            # Algorithm 1: compute barycenter once every K steps
            # The computed barycenter is used until the next recomputation
            if self.barycenter and self.beta > 0 and (i % int(self.bary_every) == 0):
                group_scores = [
                    basic_costs.predict_prob(xp, self.theta_) 
                    for xp in x_protected
                ]
                group_sizes = np.asarray([len(s) for s in group_scores], dtype=float)
                
                if group_sizes.sum() > 0:
                    res = _compute_baryscore_pot(
                        group_scores,
                        group_sizes,
                        n_bins=int(self.bary_bins),
                        reg=float(self.bary_reg),
                        n_samples=int(self.bary_n_samples),
                        bary_max_iter=int(self.bary_max_iter),
                        random_state=self.random_state,
                    )
                    self.barycenter_result_ = res
                    current_baryscore = res.samples  # Update persistent baryscore
                    self.baryscore_ = current_baryscore

            data_all = (X_model, y_arr)

            # Use current_baryscore (persists between recomputations)
            grad, loss_log, loss_wass = combined_costs.gradient_line_logistic(
                data_all,
                x_protected,
                self.theta_,
                self.beta,
                self.alpha,
                distance=self.distance,
                delta=self.delta,
                baryscore=current_baryscore,  # Uses last computed barycenter
            )

            self.theta_ -= self.lr * grad

            if self.verbose and (i % 100 == 0 or i == self.max_iter - 1):
                total = self.alpha * loss_log + (1.0 - self.alpha) * self.beta * loss_wass
                grad_norm = np.linalg.norm(grad)
                theta_norm = np.linalg.norm(self.theta_)
                step_size = self.lr * grad_norm
                self.history_.append({
                    "iter": int(i),
                    "loss_total": float(total),
                    "loss_log": float(loss_log),
                    "loss_wass": float(loss_wass),
                    "grad_norm": float(grad_norm),
                })
                print(
                    f"Iter {i:05d} | total={total:.6f} | "
                    f"log={loss_log:.6f} | wass={loss_wass:.6f} | "
                    f"grad_norm={grad_norm:.6f} | step={step_size:.6f} | theta_norm={theta_norm:.4f}"
                )

        # sklearn convenience attributes
        if self.fit_intercept:
            self.coef_ = self.theta_[:-1].reshape(1, -1)
            self.intercept_ = np.array([self.theta_[-1]])
        else:
            self.coef_ = self.theta_.reshape(1, -1)
            self.intercept_ = np.array([0.0])

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = len(self.feature_cols_)

        return self

    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Compute raw logits for input samples."""
        check_is_fitted(self, ["theta_", "feature_cols_"])

        if isinstance(X, pd.DataFrame):
            X_arr = X[self.feature_cols_].to_numpy(copy=False)
        else:
            X_arr = np.asarray(X, dtype=float)

        if hasattr(self, "scaler_") and self.scaler_ is not None:
            X_arr = self.scaler_.transform(X_arr)

        if self.fit_intercept:
            w = self.theta_[:-1]
            b = self.theta_[-1]
            return X_arr.dot(w) + b
        return X_arr.dot(self.theta_)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Class probabilities [P(y=0), P(y=1)]
        """
        check_is_fitted(self, ["theta_", "feature_cols_"])

        if isinstance(X, pd.DataFrame):
            X_model = X[self.feature_cols_].to_numpy(copy=False)
        else:
            X_model = np.asarray(X)

        X_model = np.asarray(X_model, dtype=float)

        if hasattr(self, "scaler_") and self.scaler_ is not None:
            X_model = self.scaler_.transform(X_model)

        # Use author's predict_prob to keep intercept handling identical
        p1 = basic_costs.predict_prob(X_model, self.theta_)
        p1 = np.asarray(p1, dtype=float).reshape(-1)
        p1 = np.clip(p1, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
        
        threshold : float, default=0.5
            Decision threshold.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= float(threshold)).astype(int)