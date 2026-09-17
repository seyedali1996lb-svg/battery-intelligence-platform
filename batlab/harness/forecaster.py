"""
Model-agnostic forecaster adapters — the plug-in seam of the validation harness.

Why this module exists
----------------------
Every accuracy number batlab reports was, until this module, produced by one
model: the platform's own GradientBoostingRegressor, constructed inside
run_lco(), run_lco_quantiles(), and run_prospective(). That couples the
validation *methodology* to a *model*, which is backwards. Leave-cell-out
folds, label-provenance rules, conformal interval calibration, the metric
gate and sealed replication bundles are the transferable part — they should
be able to grade any forecaster, not just this platform's.

So the seam is a two-method Protocol plus a factory contract:

    class Forecaster(Protocol):
        def fit(self, X, y) -> Any: ...
        def predict(self, X) -> np.ndarray: ...

    class IntervalForecaster(Forecaster, Protocol):
        def predict_interval(self, X) -> tuple[np.ndarray, np.ndarray]: ...

The FACTORY is where the honesty lives. The harness calls the factory once
per fold and once per target, so a fold's fit can never carry state from
another fold's fit — the most common way a hand-rolled cross-validation
silently leaks. For the same reason a *fitted* estimator passed as a template
is rejected here rather than deep-copied (see as_factory): a model that has
already seen the held-out cell produces a beautiful, meaningless R², and
silently accepting that is exactly the failure mode this whole package is
built to make impossible.

Adapters
--------
    SklearnForecaster(estimator, scale=False)     any sklearn-compatible estimator
    SklearnIntervalForecaster(point, q10, q90)    an interval from three estimators
    CallableForecaster(fit_fn, predict_fn)        anything else, including torch
    torch_forecaster(module_factory, ...)         a standard torch training loop
    as_factory(model)                             normalizes all of the above

Preprocessing is deliberately NOT added for a model you supply: if your model
needs scaling, pass a sklearn Pipeline, or wrap it in SklearnForecaster(est,
scale=True). The platform's own default factory (default_forecaster) uses
scale=True because that is the configuration its published numbers were
measured under — reproducing them exactly is the point of keeping a default.

What this module does not do
----------------------------
It does not check that the model's features are causal, that its labels are
honest, or that its interval is calibrated. Those are the harness's job
(batlab.harness.validate_forecaster). A model plugged in here gets graded,
not trusted.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Protocol, cast, runtime_checkable

import numpy as np
import pandas as pd

__all__ = [
    "CallableForecaster",
    "Forecaster",
    "ForecasterFactory",
    "ForecasterLike",
    "IntervalForecaster",
    "SklearnForecaster",
    "SklearnIntervalForecaster",
    "as_factory",
    "default_forecaster",
    "default_interval_forecaster",
    "fit_forecaster",
    "forecaster_identity",
    "has_predict_interval",
    "torch_forecaster",
]


# ---------------------------------------------------------------------------
# Protocols and type aliases
# ---------------------------------------------------------------------------

@runtime_checkable
class Forecaster(Protocol):
    """Anything that can be fitted on a feature matrix and predict on it."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Fit on X (n_features columns) against target y. May return self."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict a point estimate per row of X."""
        ...


@runtime_checkable
class IntervalForecaster(Protocol):
    """A forecaster that can also produce a prediction interval."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_interval(self, X: pd.DataFrame) -> "tuple[np.ndarray, np.ndarray]":
        """Return (lower, upper) — nominally the Q10/Q90 pair."""
        ...


# A factory returns a FRESH, UNFITTED forecaster every time it is called.
ForecasterFactory = Callable[[], Forecaster]

# Everything as_factory() accepts as a "model".
ForecasterLike = Any


def has_predict_interval(model: Any) -> bool:
    """True when `model` exposes a callable predict_interval().

    Duck-typed on purpose: runtime_checkable Protocols with non-method
    members are unreliable, and a user's model is usually not our class.
    """
    return callable(getattr(model, "predict_interval", None))


def fit_forecaster(
    model: Any,
    X: pd.DataFrame,
    y: "pd.Series | pd.DataFrame",
) -> Any:
    """Fit `model`, returning the fitted object.

    sklearn estimators return self from fit(); other implementations return
    None. Both are honoured, so a user's model is never silently unfitted
    because its fit() followed the mutating convention.

    `y` is a Series for a caller that has one target per row and a DataFrame
    for a caller that pooled per-cell labels with `pd.concat` (which
    `run_lco` and its calibration pass do). Both are handed to the model's own
    fit() unchanged — this helper forwards, it does not reshape.
    """
    out = model.fit(X, y)
    return model if out is None else out


# ---------------------------------------------------------------------------
# sklearn-compatible estimators
# ---------------------------------------------------------------------------

def _clone_estimator(estimator: Any) -> Any:
    """A fresh, unfitted copy of `estimator`.

    sklearn's clone() is preferred (it resets fitted state and validates the
    parameter set); a non-sklearn estimator falls back to a deep copy.
    """
    try:
        from sklearn.base import clone

        return clone(estimator)
    except Exception:
        return copy.deepcopy(estimator)


def _feature_matrix(X: Any) -> np.ndarray:
    return np.asarray(X, dtype=float)


class SklearnForecaster:
    """Wrap any scikit-learn-compatible estimator as a Forecaster.

    The template is cloned on every fit(), so one SklearnForecaster instance
    can be fitted once per fold without carrying state between folds.

    Parameters
    ----------
    estimator : an unfitted (or cloneable) sklearn-style estimator exposing
        fit(X, y) and predict(X).
    scale : add the platform's StandardScaler before the estimator. Off by
        default — a model you supply owns its own preprocessing (pass a
        Pipeline for that). The platform's default factory turns it on
        because its published numbers were measured with it.
    clip_min : clamp predictions from below (the platform clips RUL point
        predictions at 0 cycles). None leaves predictions untouched.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        scale: bool = False,
        clip_min: "float | None" = None,
    ) -> None:
        self._template = estimator
        self._scale = scale
        self._clip_min = clip_min
        self._estimator: Any = None
        self._scaler: Any = None

    # -- factory contract ---------------------------------------------------
    def factory(self) -> ForecasterFactory:
        """A factory rebuilding this adapter unfitted (one fresh model per fold)."""
        template, scale, clip = self._template, self._scale, self._clip_min
        return lambda: SklearnForecaster(template, scale=scale, clip_min=clip)

    # -- Forecaster ---------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnForecaster":
        estimator = _clone_estimator(self._template)
        X_used = _feature_matrix(X)
        if self._scale:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_used = scaler.fit_transform(X_used)
            self._scaler = scaler
        else:
            self._scaler = None
        estimator.fit(X_used, np.asarray(y, dtype=float))
        self._estimator = estimator
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("SklearnForecaster.predict() called before fit()")
        X_used = _feature_matrix(X)
        if self._scaler is not None:
            X_used = self._scaler.transform(X_used)
        pred = np.asarray(self._estimator.predict(X_used), dtype=float)
        if self._clip_min is not None:
            pred = np.clip(pred, self._clip_min, None)
        return pred

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"SklearnForecaster({type(self._template).__name__}, "
            f"scale={self._scale}, clip_min={self._clip_min})"
        )


class SklearnIntervalForecaster(SklearnForecaster):
    """Point + Q10/Q90 quantile estimators sharing ONE feature scaler.

    This is the shape the platform's RUL interval has always had: a point
    regressor plus two quantile regressors fitted on the same training
    matrix. Sharing one scaler (rather than one per estimator) is not an
    optimization — it is what makes the three predictions comparable, and
    it is required for a published interval to be reproducible.

    Note on inverted intervals: quantile regressors can cross, producing
    q10 > q90 on a row. That is reported as-is (the harness's calibration
    section measures the resulting coverage); silently swapping the bounds
    would change served numbers and hide the crossing.
    """

    def __init__(
        self,
        point_estimator: Any,
        q10_estimator: Any,
        q90_estimator: Any,
        *,
        scale: bool = True,
        clip_min: "float | None" = 0.0,
    ) -> None:
        super().__init__(point_estimator, scale=scale, clip_min=None)
        self._templates = (point_estimator, q10_estimator, q90_estimator)
        self._interval_clip_min = clip_min
        self._q10: Any = None
        self._q90: Any = None

    def factory(self) -> ForecasterFactory:
        point, q10, q90 = self._templates
        scale, clip = self._scale, self._interval_clip_min
        return lambda: SklearnIntervalForecaster(
            point, q10, q90, scale=scale, clip_min=clip
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnIntervalForecaster":
        X_used = _feature_matrix(X)
        if self._scale:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_used = scaler.fit_transform(X_used)
            self._scaler = scaler
        else:
            self._scaler = None

        y_arr = np.asarray(y, dtype=float)
        models = []
        for template in self._templates:
            estimator = _clone_estimator(template)
            estimator.fit(X_used, y_arr)
            models.append(estimator)
        self._estimator, self._q10, self._q90 = models
        return self

    def predict_interval(self, X: pd.DataFrame) -> "tuple[np.ndarray, np.ndarray]":
        if self._q10 is None or self._q90 is None:
            raise RuntimeError(
                "SklearnIntervalForecaster.predict_interval() called before fit()"
            )
        X_used = _feature_matrix(X)
        if self._scaler is not None:
            X_used = self._scaler.transform(X_used)
        q10 = np.asarray(self._q10.predict(X_used), dtype=float)
        q90 = np.asarray(self._q90.predict(X_used), dtype=float)
        if self._interval_clip_min is not None:
            q10 = np.clip(q10, self._interval_clip_min, None)
            q90 = np.clip(q90, self._interval_clip_min, None)
        return q10, q90

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"SklearnIntervalForecaster({type(self._templates[0]).__name__}, "
            f"scale={self._scale})"
        )


# ---------------------------------------------------------------------------
# Everything else
# ---------------------------------------------------------------------------

class CallableForecaster:
    """Wrap arbitrary fit/predict callables as a Forecaster.

    The escape hatch for models that are not sklearn-shaped: PyTorch
    modules, Keras models, a hand-written ODE fit, a subprocess call to
    MATLAB. `fit_fn(X, y)` returns whatever read-only state `predict_fn`
    needs (typically a fitted object); passing `interval_fn(state, X)`
    additionally makes the model interval-capable.

    Example (PyTorch)
    -----------------
    >>> def fit_fn(X, y):                       # doctest: +SKIP
    ...     net = torch.nn.Sequential(          # your architecture
    ...         torch.nn.Linear(X.shape[1], 32), torch.nn.ReLU(),
    ...         torch.nn.Linear(32, 1))
    ...     ...                                  # your training loop
    ...     return net
    >>> forecaster = CallableForecaster(        # doctest: +SKIP
    ...     fit_fn,
    ...     lambda net, X: net(torch.as_tensor(X, dtype=torch.float32)).detach().numpy().ravel(),
    ... )

    `torch_forecaster()` below builds exactly this for the standard
    single-target regression loop.
    """

    def __init__(
        self,
        fit_fn: Callable[[pd.DataFrame, pd.Series], Any],
        predict_fn: Callable[[Any, pd.DataFrame], Any],
        *,
        interval_fn: "Callable[[Any, pd.DataFrame], tuple[np.ndarray, np.ndarray]] | None" = None,
        label: str = "CallableForecaster",
    ) -> None:
        self._fit_fn = fit_fn
        self._predict_fn = predict_fn
        self._interval_fn = interval_fn
        self._label = label
        self._state: Any = None
        self._fitted = False
        if interval_fn is not None:
            # Instance-level method: the model is interval-capable only when
            # an interval_fn was actually supplied, which is what
            # has_predict_interval() must report.
            self.predict_interval = lambda X: _as_interval(interval_fn(self._state, X))  # type: ignore[method-assign]

    def factory(self) -> ForecasterFactory:
        return lambda: CallableForecaster(
            self._fit_fn,
            self._predict_fn,
            interval_fn=self._interval_fn,
            label=self._label,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CallableForecaster":
        self._state = self._fit_fn(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # A fit_fn may legitimately return None (training in place), so the
        # never-fitted case is tracked with its own flag rather than by
        # testing the state for None.
        if not self._fitted:
            raise RuntimeError("CallableForecaster.predict() called before fit()")
        return np.asarray(self._predict_fn(self._state, X), dtype=float).reshape(-1)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return self._label


def _as_interval(pair: Any) -> "tuple[np.ndarray, np.ndarray]":
    lo, hi = pair
    return (
        np.asarray(lo, dtype=float).reshape(-1),
        np.asarray(hi, dtype=float).reshape(-1),
    )


def torch_forecaster(
    module_factory: Callable[[int], Any],
    *,
    epochs: int = 200,
    lr: float = 1e-2,
    batch_size: int = 256,
    device: str = "cpu",
    seed: int = 42,
    scale: bool = True,
) -> ForecasterFactory:
    """A factory producing a Forecaster that trains a PyTorch module per fold.

    `module_factory(in_features)` must return an UNTRAINED torch.nn.Module
    mapping (batch, in_features) -> (batch, 1). The standard loop (Adam +
    MSE, full-batch or minibatch) is here so a deep model can be graded by
    the harness in one line; anything more exotic belongs in
    CallableForecaster, where you own the loop.

    PyTorch is imported lazily, inside fit(): batlab itself does not depend
    on torch, and `pip install batlab` must keep working without it.

    This returns a POINT forecaster. Intervals for a point model come from
    the harness's conformal-residual calibration (batlab.harness), which is
    distribution-free and needs no retraining — or pass your own
    interval_fn via CallableForecaster if you have a real uncertainty head.
    """

    def _make() -> Forecaster:
        def fit_fn(X: pd.DataFrame, y: pd.Series) -> Any:
            torch = _import_torch()
            from sklearn.preprocessing import StandardScaler

            torch.manual_seed(seed)
            net = module_factory(int(X.shape[1]))
            net.to(device)
            net.train()

            X_arr = np.asarray(X, dtype=np.float32)
            y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            scaler = None
            if scale:
                scaler = StandardScaler()
                X_arr = scaler.fit_transform(X_arr).astype(np.float32)

            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            loss_fn = torch.nn.MSELoss()
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(X_arr), torch.from_numpy(y_arr)
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=min(batch_size, len(dataset)), shuffle=True
            )
            for _ in range(max(1, epochs)):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss_fn(net(xb), yb).backward()
                    optimizer.step()
            net.eval()
            return {"net": net, "scaler": scaler}

        def predict_fn(state: Any, X: pd.DataFrame) -> Any:
            torch = _import_torch()

            X_arr = np.asarray(X, dtype=np.float32)
            scaler = state.get("scaler")
            if scaler is not None:
                X_arr = scaler.transform(X_arr).astype(np.float32)
            with torch.no_grad():
                out = state["net"](torch.from_numpy(X_arr).to(device))
            return out.detach().cpu().numpy().reshape(-1)

        return CallableForecaster(
            fit_fn, predict_fn, label=f"torch_forecaster({getattr(module_factory, '__name__', 'module')})"
        )

    return _make


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "torch_forecaster() needs PyTorch, which batlab does not depend on "
            "— install it with `pip install torch`, or use CallableForecaster "
            "with your own training loop."
        ) from exc
    return torch


# ---------------------------------------------------------------------------
# Normalization: anything -> a factory
# ---------------------------------------------------------------------------

def _reject_prefitted(estimator: Any) -> None:
    """Refuse a template that already carries a fit.

    Falling back to a deep copy here would LOOK helpful and be catastrophic:
    the copied model already saw every cell in the fleet, including the ones
    the harness is about to hold out, and its scores would be meaningless in
    exactly the way this package exists to prevent. Only sklearn estimators
    are checked (sklearn exposes the check); for your own class, the
    docstring contract is on you.
    """
    try:
        from sklearn.base import BaseEstimator
        from sklearn.utils.validation import check_is_fitted
    except Exception:  # pragma: no cover - sklearn is a hard dependency
        return
    if not isinstance(estimator, BaseEstimator):
        return
    try:
        check_is_fitted(estimator)
    except Exception:
        return  # not fitted — the expected case
    raise ValueError(
        "The estimator passed to the harness is already fitted. A harness fold "
        "must train from scratch on the training cells only, so accepting a "
        "pre-fitted model would leak the held-out cells into its own score. "
        "Pass an unfitted estimator, a factory (a zero-argument callable "
        "returning a fresh model), or sklearn.pipeline.make_pipeline(...)."
    )


def as_factory(model: ForecasterLike, *, default: "ForecasterFactory | None" = None) -> "ForecasterFactory | None":
    """Normalize anything the harness accepts as a model into a factory.

    Accepted inputs:

    * ``None``                -> `default` (the platform's GBRT factory).
    * our adapters            -> their own ``factory()`` (fresh instance per fold).
    * an unfitted estimator   -> cloned per fold, no extra preprocessing.
    * a zero-argument callable-> used as the factory as-is (it must return a
      fresh, unfitted model on every call; the harness calls it per fold).
    * anything else with fit/predict -> deep-copied per fold.

    Raises ValueError for a pre-fitted sklearn estimator (see
    _reject_prefitted) and TypeError for an object that is neither a model
    nor a callable, because silently grading the wrong object is worse than
    failing.
    """
    if model is None:
        return default

    # Our own adapters know how to rebuild themselves unfitted.
    for adapter_type in (SklearnForecaster, SklearnIntervalForecaster, CallableForecaster):
        if isinstance(model, adapter_type):
            return model.factory()

    has_model_api = callable(getattr(model, "fit", None)) and callable(
        getattr(model, "predict", None)
    )
    if has_model_api:
        _reject_prefitted(model)
        return lambda: copy.deepcopy(model)

    if callable(model):
        # A zero-argument callable IS the factory contract; the return type is
        # the caller's promise about what it returns, which the harness
        # re-checks on the first call (a factory returning a fitted model or a
        # non-model raises there rather than producing a number).
        return cast(ForecasterFactory, model)

    raise TypeError(
        "model must be None (platform default), an estimator exposing "
        "fit/predict, a zero-argument factory returning one, or a "
        f"batlab.harness adapter — got {type(model).__name__}."
    )


# ---------------------------------------------------------------------------
# The platform's own default: the model its published numbers were measured on
# ---------------------------------------------------------------------------

def default_forecaster(seed: int = 42) -> ForecasterFactory:
    """Factory for the platform's default point model (GBRT, scaled).

    Exactly the configuration batlab.validation.lco.run_lco() has always
    used — so `validate_forecaster(cells)` with no model passed reproduces
    the platform's published SOH/RUL numbers rather than approximating them.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    from batlab.models.gbrt import GBRT_PARAMS

    params = {**GBRT_PARAMS, "random_state": seed}
    return lambda: SklearnForecaster(
        GradientBoostingRegressor(**params), scale=True, clip_min=None
    )


def default_interval_forecaster(seed: int = 42) -> ForecasterFactory:
    """Factory for the platform's default RUL interval model (point + Q10/Q90).

    Byte-for-byte the configuration run_lco_quantiles() has always used: one
    shared scaler, GBRT_PARAMS for the point estimate, GBRT_QUANTILE_PARAMS
    with quantile loss for the two bounds, clipped at 0 cycles.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    from batlab.models.gbrt import GBRT_PARAMS, GBRT_QUANTILE_PARAMS

    point_params = {**GBRT_PARAMS, "random_state": seed}
    quantile_params = {**GBRT_QUANTILE_PARAMS, "random_state": seed}
    return lambda: SklearnIntervalForecaster(
        GradientBoostingRegressor(**point_params),
        GradientBoostingRegressor(loss="quantile", alpha=0.10, **quantile_params),
        GradientBoostingRegressor(loss="quantile", alpha=0.90, **quantile_params),
        scale=True,
        clip_min=0.0,
    )


def forecaster_identity(model_or_factory: Any) -> dict:
    """A small, JSON-safe description of what was graded — for the report.

    Records the class name, the sklearn hyperparameters when the object
    exposes them, and the factory's own name when one was supplied, so a
    harness report says which model produced its numbers instead of "the
    default".

    A sandboxed model is described by the child that holds it (see
    batlab.harness.sandbox), with `sandboxed: true` and the limits it ran
    under — so the report, and the bundle sealed from it, say where the model's
    code actually ran.
    """
    identity: dict = {}
    target = model_or_factory
    if target is None:
        return {"source": "platform default (GBRT, scaled)", "class": "GradientBoostingRegressor"}

    # A model that runs in another process (batlab.harness.sandbox) reports what
    # that process actually holds. Reading the proxy's own class name here would
    # put "<type '_RemoteModel'>" in every report and every sealed bundle: true
    # about the transport, useless as a description of the model that was
    # graded — and the model identity is what a reviewer checks the number
    # against.
    proxy_identity = getattr(target, "sandboxed_identity", None)
    if isinstance(proxy_identity, dict):
        return dict(proxy_identity)

    if callable(target) and not hasattr(target, "fit"):
        identity["factory"] = getattr(target, "__name__", None) or repr(target)
        try:
            target = target()  # probe one fresh instance for its class/params
        except Exception:
            return identity

    identity["class"] = type(target).__name__
    get_params = getattr(target, "get_params", None)
    if callable(get_params):
        try:
            identity["params"] = {
                k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v))
                for k, v in get_params().items()
            }
        except Exception:  # pragma: no cover - defensive
            pass
    return identity
