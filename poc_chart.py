"""
Predictor Operating Characteristic (POC) Chart
===============================================

A modified ROC-style chart for evaluating classification predictors.
Instead of TPR vs FPR, the POC plots **Precision** (x-axis) vs **Accuracy**
(y-axis) for each predictor, with a vertical baseline line showing where
random guessing would fall.

When ``positive_classes`` is supplied, the baseline is computed from the
actual class distribution in the data (frequency of the positive classes
for positive mode, frequency of the negative classes for negative mode).
Otherwise the baseline defaults to ``1 / num_classes``.

Points to the right of the baseline are beating random chance on precision.
The horizontal distance from the baseline ("alpha") quantifies how much
better (or worse) the predictor is compared to guessing.

Supports two modes:

- **"positive"** (default): X = Positive Predictive Value (precision),
  Y = accuracy of positives, baseline at positive-class frequency.
- **"negative"**: X = Negative Predictive Value,
  Y = accuracy of negatives, baseline at negative-class frequency.

Usage
-----
Single predictor (positive mode)::

    from poc_chart import poc_chart

    predicted = ['stay', 'churn', 'stay', 'stay', 'churn']
    actual    = ['stay', 'churn', 'churn', 'stay', 'churn']
    data, ax = poc_chart(predicted, actual, num_classes=2, names="My Model")

With class-distribution-aware baseline (recommended for imbalanced data)::

    data, ax = poc_chart(predicted, actual, num_classes=2, names="My Model",
                         positive_classes={"stay"})

Negative predictive value mode::

    data, ax = poc_chart(predicted, actual, num_classes=2, names="My Model",
                         mode="negative", positive_classes={"stay"})

Multiple predictors::

    data, ax = poc_chart(
        predicted_labels={"hive": preds_hive, "node_1": preds_n1},
        true_labels={"hive": actuals_hive, "node_1": actuals_n1},
        num_classes=3,
    )

Pre-computed metrics::

    from poc_chart import poc_data_from_metrics, plot_poc

    data = poc_data_from_metrics(
        metrics=[
            {"name": "Node A", "accuracy": 0.85, "precision": 0.90},
            {"name": "Node B", "accuracy": 0.72, "precision": 0.65},
        ],
        num_classes=5,
    )
    ax = plot_poc(data)

Side-by-side comparison::

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    poc_chart(preds, actuals, num_classes=3, names="Model", mode="positive", ax=ax1)
    poc_chart(preds, actuals, num_classes=3, names="Model", mode="negative", ax=ax2)
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.axes

__all__ = [
    "poc_chart",
    "plot_poc",
    "compute_poc_data",
    "poc_data_from_metrics",
    "compute_predictor_metrics",
    "PredictorMetrics",
    "POCChartData",
]

# Default color palette (matches IA.Dev frontend node colors, then falls back to tab10)
_DEFAULT_COLORS = [
    "#09D59B",  # hive green
    "#9A8BFF",  # purple
    "#FFD22F",  # gold
    "#1E90FF",  # dodger blue
    "#FF6B6B",  # coral
    "#00CED1",  # dark turquoise
    "#FF8C00",  # dark orange
    "#DA70D6",  # orchid
]


@dataclass
class PredictorMetrics:
    """Metrics for a single predictor on the POC chart."""

    name: str
    accuracy: float  # y-axis value (mode-dependent)
    precision: float  # x-axis value (mode-dependent)
    alpha: float  # precision - baseline (distance from random chance)
    num_correct: int = 0
    num_incorrect: int = 0
    num_unknown: int = 0
    response_rate: float = 1.0  # (correct + incorrect) / total
    # Negative mode field (populated when mode="negative")
    npv: Optional[float] = None  # Negative Predictive Value


_LABELS = {
    "positive": {
        "x": "Precision of Positives (Positive Predictive Value)",
        "y": "Accuracy of Positives",
        "title": "Predictor Operating Characteristic",
    },
    "negative": {
        "x": "Precision of Negatives (Negative Predictive Value)",
        "y": "Accuracy of Negatives",
        "title": "Predictor Operating Characteristic (Negative)",
    },
}


@dataclass
class POCChartData:
    """Container for all data needed to render a POC chart."""

    predictors: List[PredictorMetrics]
    baseline: float  # random chance line position
    num_classes: int
    mode: str = "positive"
    x_label: str = "Precision of Positives (Positive Predictive Value)"
    y_label: str = "Accuracy of Positives"


def _baseline_from_labels(
    true_labels: List[Any],
    mode: str,
    positive_classes: set,
) -> float:
    """Compute the baseline from the frequency of positive/negative classes.

    - **positive** mode: frequency of ``positive_classes`` in the data.
    - **negative** mode: frequency of non-positive classes (1 - positive freq).
    """
    total = len(true_labels)
    if total == 0:
        return 0.5
    pos_count = sum(1 for label in true_labels if label in positive_classes)
    pos_freq = pos_count / total
    return pos_freq if mode == "positive" else 1.0 - pos_freq


def compute_predictor_metrics(
    name: str,
    predicted_labels: List[Any],
    true_labels: List[Any],
    num_classes: int,
    unknown_value: Any = None,
    baseline: Optional[float] = None,
    mode: str = "positive",
    positive_classes: Optional[set] = None,
) -> PredictorMetrics:
    """Compute POC metrics for a single predictor.

    When ``positive_classes`` is provided, metrics are computed using a
    polarity-based confusion matrix (matching the approach in
    ``test_harness.processUtilityPolarityResults``):

    - **TP**: predicted positive AND actual positive
    - **FP**: predicted positive AND actual negative
    - **TN**: predicted negative AND actual negative
    - **FN**: predicted negative AND actual positive
    - **PPV** = TP / (TP + FP),  **NPV** = TN / (TN + FN)
    - **Accuracy of positives** = TP / actual positives  (positive recall)
    - **Accuracy of negatives** = TN / actual negatives  (negative recall)

    Without ``positive_classes``, falls back to overall precision / accuracy.

    Parameters
    ----------
    name : str
        Display name for this predictor.
    predicted_labels : list
        Predicted class labels (same length as true_labels).
    true_labels : list
        Ground-truth class labels.
    num_classes : int
        Number of distinct classes in the dataset.
    unknown_value : any, optional
        Value representing an abstention / no-response. These are excluded
        from precision but count as incorrect for accuracy.
    baseline : float, optional
        Custom baseline position. Overrides ``positive_classes`` when set.
        Defaults to ``1 / num_classes`` for positive mode and
        ``(num_classes - 1) / num_classes`` for negative mode when
        ``positive_classes`` is also None.
    mode : str, optional
        ``"positive"`` (default) for PPV or ``"negative"`` for NPV.
    positive_classes : set, optional
        Class labels considered "positive" (e.g. ``{"stay"}``).
        When provided, metrics and baseline are computed from the
        polarity-based confusion matrix and class frequencies.

    Returns
    -------
    PredictorMetrics
    """
    if mode not in ("positive", "negative"):
        raise ValueError(f"mode must be 'positive' or 'negative', got '{mode}'")

    if len(predicted_labels) != len(true_labels):
        raise ValueError(
            f"predicted_labels ({len(predicted_labels)}) and true_labels "
            f"({len(true_labels)}) must have the same length"
        )

    if baseline is None:
        if positive_classes is not None:
            baseline = _baseline_from_labels(true_labels, mode, positive_classes)
        else:
            baseline = (
                1.0 / num_classes if mode == "positive"
                else (num_classes - 1) / num_classes
            )

    total = len(true_labels)

    # --- Polarity-based confusion matrix (when positive_classes is known) ---
    if positive_classes is not None:
        tp = fp = tn = fn = num_unknown = 0
        for pred, actual in zip(predicted_labels, true_labels):
            if pred == unknown_value and unknown_value is not None:
                num_unknown += 1
            elif pred in positive_classes:          # positive prediction
                if actual in positive_classes:
                    tp += 1                         # correct positive
                else:
                    fp += 1                         # incorrect positive
            else:                                   # negative prediction
                if actual not in positive_classes:
                    tn += 1                         # correct negative
                else:
                    fn += 1                         # incorrect negative

        num_correct = tp + tn
        num_incorrect = fp + fn
        responded = num_correct + num_incorrect
        response_rate = responded / total if total > 0 else 0.0
        total_actual_pos = sum(1 for a in true_labels if a in positive_classes)
        total_actual_neg = total - total_actual_pos

        if mode == "positive":
            pos_preds = tp + fp
            prec = tp / pos_preds if pos_preds > 0 else 0.0       # PPV
            acc = tp / total_actual_pos if total_actual_pos > 0 else 0.0  # positive recall
            alpha = prec - baseline
            return PredictorMetrics(
                name=name, accuracy=acc, precision=prec, alpha=alpha,
                num_correct=num_correct, num_incorrect=num_incorrect,
                num_unknown=num_unknown, response_rate=response_rate,
            )
        else:
            neg_preds = tn + fn
            npv = tn / neg_preds if neg_preds > 0 else 0.0        # NPV
            acc = tn / total_actual_neg if total_actual_neg > 0 else 0.0  # negative recall
            alpha = npv - baseline
            return PredictorMetrics(
                name=name, accuracy=acc, precision=npv, alpha=alpha,
                num_correct=num_correct, num_incorrect=num_incorrect,
                num_unknown=num_unknown, response_rate=response_rate,
                npv=npv,
            )

    # --- Fallback: overall metrics (no positive_classes) ---
    num_correct = 0
    num_incorrect = 0
    num_unknown = 0

    for pred, actual in zip(predicted_labels, true_labels):
        if pred == unknown_value and unknown_value is not None:
            num_unknown += 1
        elif pred == actual:
            num_correct += 1
        else:
            num_incorrect += 1

    responded = num_correct + num_incorrect
    response_rate = responded / total if total > 0 else 0.0

    # Overall accuracy and precision (same for both modes without polarity info)
    acc = num_correct / total if total > 0 else 0.0
    prec = num_correct / responded if responded > 0 else 0.0
    alpha = prec - baseline

    return PredictorMetrics(
        name=name, accuracy=acc, precision=prec, alpha=alpha,
        num_correct=num_correct, num_incorrect=num_incorrect,
        num_unknown=num_unknown, response_rate=response_rate,
        npv=prec if mode == "negative" else None,
    )


def compute_poc_data(
    predictors: Dict[str, Tuple[List[Any], List[Any]]],
    num_classes: int,
    unknown_value: Any = None,
    baseline: Optional[float] = None,
    mode: str = "positive",
    positive_classes: Optional[set] = None,
) -> POCChartData:
    """Compute POC data for multiple predictors from raw labels.

    Parameters
    ----------
    predictors : dict
        ``{name: (predicted_labels, true_labels)}`` for each predictor.
    num_classes : int
        Number of distinct classes in the dataset.
    unknown_value : any, optional
        Value representing an abstention / no-response.
    baseline : float, optional
        Custom baseline position. Overrides ``positive_classes`` when set.
        Defaults to ``1 / num_classes`` for positive mode and
        ``(num_classes - 1) / num_classes`` for negative mode when
        ``positive_classes`` is also None.
    mode : str, optional
        ``"positive"`` (default) for PPV or ``"negative"`` for NPV.
    positive_classes : set, optional
        Class labels considered "positive" (e.g. ``{"stay"}``).
        When provided, the baseline is computed from the frequency of
        these classes in the true labels (positive mode) or the frequency
        of the remaining classes (negative mode).

    Returns
    -------
    POCChartData
    """
    if baseline is None:
        if positive_classes is not None:
            all_true = []
            for _name, (_preds, actuals) in predictors.items():
                all_true.extend(actuals)
            baseline = _baseline_from_labels(all_true, mode, positive_classes)
        else:
            baseline = (
                1.0 / num_classes if mode == "positive"
                else (num_classes - 1) / num_classes
            )

    labels = _LABELS[mode]
    metrics_list = []
    for name, (preds, actuals) in predictors.items():
        m = compute_predictor_metrics(
            name, preds, actuals, num_classes,
            unknown_value=unknown_value, baseline=baseline, mode=mode,
            positive_classes=positive_classes,
        )
        metrics_list.append(m)

    return POCChartData(
        predictors=metrics_list,
        baseline=baseline,
        num_classes=num_classes,
        mode=mode,
        x_label=labels["x"],
        y_label=labels["y"],
    )


def poc_data_from_metrics(
    metrics: List[Dict[str, Any]],
    num_classes: int,
    baseline: Optional[float] = None,
    mode: str = "positive",
) -> POCChartData:
    """Build POCChartData from pre-computed metric dicts.

    Parameters
    ----------
    metrics : list of dict
        For **positive** mode: each dict must have ``name``, ``accuracy``,
        ``precision``.
        For **negative** mode: each dict must have ``name``,
        ``accuracy_of_negatives``, ``npv``.
        Optional keys: ``num_correct``, ``num_incorrect``, ``num_unknown``,
        ``response_rate``.
    num_classes : int
        Number of distinct classes in the dataset.
    baseline : float, optional
        Custom baseline position. Defaults to ``1 / num_classes`` for
        positive mode, ``(num_classes - 1) / num_classes`` for negative mode.
    mode : str, optional
        ``"positive"`` (default) for PPV or ``"negative"`` for NPV.

    Returns
    -------
    POCChartData
    """
    if baseline is None:
        baseline = (
            1.0 / num_classes if mode == "positive"
            else (num_classes - 1) / num_classes
        )

    labels = _LABELS[mode]
    predictors = []
    for m in metrics:
        if mode == "positive":
            prec = m["precision"]
            acc = m["accuracy"]
            npv_val = m.get("npv")
        else:
            prec = m["npv"]
            acc = m["accuracy_of_negatives"]
            npv_val = prec

        predictors.append(PredictorMetrics(
            name=m["name"],
            accuracy=acc,
            precision=prec,
            alpha=prec - baseline,
            num_correct=m.get("num_correct", 0),
            num_incorrect=m.get("num_incorrect", 0),
            num_unknown=m.get("num_unknown", 0),
            response_rate=m.get("response_rate", 1.0),
            npv=npv_val,
        ))

    return POCChartData(
        predictors=predictors,
        baseline=baseline,
        num_classes=num_classes,
        mode=mode,
        x_label=labels["x"],
        y_label=labels["y"],
    )


def plot_poc(
    data: POCChartData,
    *,
    figsize: Tuple[float, float] = (10, 6),
    colors: Optional[List[str]] = None,
    annotate: bool = True,
    title: Optional[str] = None,
    marker_size: int = 120,
    baseline_color: str = "#00ffff",
    baseline_label: str = "Random Chance",
    show_alpha: bool = True,
    save_path: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """Render a POC chart using matplotlib.

    Parameters
    ----------
    data : POCChartData
        Computed POC data (from compute_poc_data or poc_data_from_metrics).
    figsize : tuple, optional
        Figure size in inches. Ignored if ``ax`` is provided.
    colors : list of str, optional
        Colors for each predictor point. Defaults to the built-in palette.
    annotate : bool, optional
        If True, label each point with predictor name and alpha.
    title : str, optional
        Chart title.
    marker_size : int, optional
        Size of scatter markers.
    baseline_color : str, optional
        Color of the vertical baseline line. Default cyan (#00ffff).
    baseline_label : str, optional
        Legend label for the baseline line.
    show_alpha : bool, optional
        If True, draw dashed lines from baseline to each point.
    save_path : str, optional
        If provided, save the figure to this path (PNG, 300 dpi).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if colors is None:
        colors = _DEFAULT_COLORS

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Axis setup
    if title is None:
        title = _LABELS.get(data.mode, _LABELS["positive"])["title"]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(data.x_label, fontsize=12)
    ax.set_ylabel(data.y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Vertical baseline (random chance)
    ax.axvline(
        x=data.baseline,
        color=baseline_color,
        linewidth=2,
        label=f"{baseline_label} ({data.baseline:.2f})",
        zorder=2,
    )

    # Plot each predictor
    for i, p in enumerate(data.predictors):
        color = colors[i % len(colors)]

        ax.scatter(
            p.precision, p.accuracy,
            s=marker_size, c=color, edgecolors="black", linewidths=0.5,
            zorder=4, label=f"{p.name} (\u03b1={p.alpha:+.2f})",
        )

        # Dashed line showing alpha distance from baseline
        if show_alpha:
            ax.plot(
                [data.baseline, p.precision], [p.accuracy, p.accuracy],
                color=color, linestyle="--", linewidth=1, alpha=0.6, zorder=3,
            )

        # Annotation with name
        if annotate:
            ax.annotate(
                p.name,
                (p.precision, p.accuracy),
                textcoords="offset points",
                xytext=(10, 8),
                fontsize=9,
                color=color,
                fontweight="bold",
            )

    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    plt.tight_layout()

    if save_path is not None:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def poc_chart(
    predicted_labels: Union[List[Any], Dict[str, List[Any]]],
    true_labels: Union[List[Any], Dict[str, List[Any]]],
    num_classes: int,
    *,
    names: Optional[Union[str, List[str]]] = None,
    unknown_value: Any = None,
    baseline: Optional[float] = None,
    mode: str = "positive",
    positive_classes: Optional[set] = None,
    **plot_kwargs,
) -> Tuple[POCChartData, matplotlib.axes.Axes]:
    """Compute metrics and plot a POC chart in one call.

    Parameters
    ----------
    predicted_labels : list or dict
        Single predictor: a list of predicted labels.
        Multiple predictors: ``{name: [predicted_labels]}``.
    true_labels : list or dict
        Single predictor: a list of true labels.
        Multiple predictors: ``{name: [true_labels]}``.
    num_classes : int
        Number of distinct classes in the dataset.
    names : str or list of str, optional
        Predictor name(s). Required when passing flat lists.
        Ignored when passing dicts (names come from dict keys).
    unknown_value : any, optional
        Value representing an abstention / no-response.
    baseline : float, optional
        Custom baseline position. Overrides ``positive_classes`` when set.
        Defaults to ``1 / num_classes`` for positive mode and
        ``(num_classes - 1) / num_classes`` for negative mode when
        ``positive_classes`` is also None.
    mode : str, optional
        ``"positive"`` (default) for PPV or ``"negative"`` for NPV.
    positive_classes : set, optional
        Class labels considered "positive" (e.g. ``{"stay"}``).
        When provided, the baseline is computed from the frequency of
        these classes in ``true_labels`` (positive mode) or the frequency
        of the remaining classes (negative mode).
    **plot_kwargs
        Additional keyword arguments passed to ``plot_poc()``.

    Returns
    -------
    (POCChartData, matplotlib.axes.Axes)
    """
    # Normalize inputs to dict form
    if isinstance(predicted_labels, dict):
        predictors_dict = {
            name: (predicted_labels[name], true_labels[name])
            for name in predicted_labels
        }
    else:
        if names is None:
            names = "Predictor"
        if isinstance(names, str):
            names = [names]
        if isinstance(names, list) and len(names) == 1:
            predictors_dict = {names[0]: (predicted_labels, true_labels)}
        else:
            raise ValueError(
                "When passing flat lists, 'names' must be a single string. "
                "For multiple predictors, pass dicts instead."
            )

    data = compute_poc_data(
        predictors_dict, num_classes,
        unknown_value=unknown_value, baseline=baseline, mode=mode,
        positive_classes=positive_classes,
    )
    ax = plot_poc(data, **plot_kwargs)
    return data, ax
