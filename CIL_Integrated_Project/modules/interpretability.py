"""
Task 8 Module: Model Interpretability & Explainability
Original source: CIL78 (1).ipynb

Implements:
- CNN Intermediate Layer Activation / Feature Map Visualizations (Conv1D layer outputs).
- Inspection of 4 representative operational cases:
  * True Positive (TP): Correctly identified machine failure.
  * True Negative (TN): Correctly identified normal operating condition.
  * False Positive (FP): False alarm.
  * False Negative (FN): Missed failure condition.
- Conceptual comparison: Linear Model Coefficients vs Deep Convolutional Feature Maps.
- LIME (Local Interpretable Model-agnostic Explanations) integration with graceful fallback.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config

HAS_LIME = False
try:
    import lime
    from lime import lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False


def find_diagnostic_sample_indices(
    y_test: Union_Series = None,
    y_pred: np.ndarray = None
) -> Dict[str, Optional[int]]:
    """
    Identifies representative sample indices from test set:
    TN, TP, FP, FN.
    Direct reproduction of CIL78 (1).ipynb Cell 15.
    """
    if y_test is None or y_pred is None:
        return {"TN": 1, "TP": 21, "FP": 0, "FN": 17}

    y_true_arr = np.array(y_test).ravel()
    y_pred_arr = np.array(y_pred).ravel()

    tn_idx = np.where((y_true_arr == 0) & (y_pred_arr == 0))[0]
    tp_idx = np.where((y_true_arr == 1) & (y_pred_arr == 1))[0]
    fp_idx = np.where((y_true_arr == 0) & (y_pred_arr == 1))[0]
    fn_idx = np.where((y_true_arr == 1) & (y_pred_arr == 0))[0]

    return {
        "TN": int(tn_idx[0]) if len(tn_idx) > 0 else None,
        "TP": int(tp_idx[0]) if len(tp_idx) > 0 else None,
        "FP": int(fp_idx[0]) if len(fp_idx) > 0 else None,
        "FN": int(fn_idx[0]) if len(fn_idx) > 0 else None,
    }


def plot_input_feature_bar(
    feature_names: List[str],
    scaled_values: np.ndarray,
    sample_idx: int,
    sample_type: str,
    actual_label: Optional[int] = None,
    predicted_label: Optional[int] = None,
    probability: Optional[float] = None
) -> plt.Figure:
    """
    Plots horizontal or vertical bar chart of scaled feature inputs for a given diagnostic sample.
    Direct reproduction of CIL78 (1).ipynb Cell 16.
    """
    fig, ax = plt.subplots(figsize=(8, 3.5))
    values = scaled_values.flatten()
    colors = ["#E63946" if v > 0 else "#457B9D" for v in values]

    bars = ax.bar(feature_names, values, color=colors, edgecolor="black", alpha=0.85)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Scaled Value (Z-Score)", fontsize=10)
    ax.set_title(
        f"Input Feature Profile — Sample {sample_idx} ({sample_type})\n"
        f"Actual: {actual_label} | Predicted: {predicted_label} | P(Failure): {probability:.4f}"
        if probability is not None else f"Input Features for Sample {sample_idx} ({sample_type})",
        fontsize=11,
        fontweight="bold"
    )
    ax.set_xticklabels(feature_names, rotation=35, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_activation_heatmap(
    activations: np.ndarray,
    layer_name: str,
    max_filters: int = 32
) -> plt.Figure:
    """
    Plots intermediate activation matrix across filters for a convolutional layer.
    activations shape: (timesteps/features, filters)
    """
    # activations may be (1, length, filters) or (length, filters)
    if activations.ndim == 3:
        activations = activations[0]

    num_filters = min(activations.shape[-1], max_filters)
    matrix = activations[:, :num_filters]

    fig, ax = plt.subplots(figsize=(10, max(2.5, matrix.shape[0] * 0.6)))
    sns.heatmap(
        matrix.T,
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": "Activation Magnitude", "shrink": 0.8}
    )
    ax.set_title(f"Activation Heatmap for Layer: {layer_name} (First {num_filters} Filters)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Feature / Sequence Position", fontsize=10)
    ax.set_ylabel("Filter Index", fontsize=10)
    fig.tight_layout()
    return fig


def get_interpretability_comparison_text() -> Dict[str, Any]:
    """
    Returns structured comparative analysis between Baseline ML and 1D CNN interpretability.
    Direct reproduction of CIL78 (1).ipynb Markdown Cells 13-14.
    """
    return {
        "baseline_ml": {
            "title": "Baseline ML Model (Logistic Regression)",
            "method": "Direct Coefficient Analysis (Odds Ratios)",
            "output": "Explicit per-feature coefficients indicating linear directional impact on failure probability.",
            "strengths": "Maximum transparency, straightforward engineering interpretation, direct translation to operating thresholds.",
            "limitations": "Constrained by linearity; unable to capture multi-sensor compound interactions (e.g., torque spikes coupled with elevated tool wear)."
        },
        "cnn": {
            "title": "1D Convolutional Neural Network (CNN)",
            "method": "Intermediate Feature Map Activations & Local Surrogate Explanations (LIME)",
            "output": "Filter-level spatial activation heatmaps across temporal/tabular features and localized importance weights.",
            "strengths": "Captures non-linear coupled patterns and localized sensor interactions across physical metrics.",
            "limitations": "Higher abstraction level; individual filter activations require surrogate methods (such as LIME) for localized attribution."
        },
        "synthesis": (
            "Directness vs. Capacity: Linear regression provides immediate intuitive coefficients, "
            "whereas 1D CNN provides far superior failure classification precision (0.500 vs. 0.142) on imbalanced data. "
            "Pairing CNN with activation maps and LIME bridges the transparency gap, providing both predictive power and explainability."
        )
    }


def explain_sample_lime(
    training_data: np.ndarray,
    feature_names: List[str],
    sample_to_explain: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    case_name: str = "Diagnostic Sample"
) -> Tuple[Optional[Any], Optional[plt.Figure], str]:
    """
    Generates a LIME explanation for an individual prediction using LimeTabularExplainer.
    Direct reproduction of CIL78 (1).ipynb Cell 17.
    Returns (explanation_obj, pyplot_figure, summary_message).
    """
    if not HAS_LIME:
        msg = (
            "LIME package is not installed in the current environment.\n"
            "To enable live LIME generation, install via: pip install lime.\n"
            "Pre-computed sample explanation profiles are displayed below."
        )
        return None, None, msg

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=training_data,
        feature_names=feature_names,
        class_names=["No Failure", "Failure"],
        mode="classification"
    )

    exp = explainer.explain_instance(
        data_row=sample_to_explain,
        predict_fn=predict_fn,
        num_features=len(feature_names)
    )

    fig = exp.as_pyplot_figure()
    fig.set_size_inches(8, 4.5)
    plt.title(f"LIME Feature Contributions — {case_name}", fontsize=12, pad=12)
    plt.tight_layout()

    return exp, fig, "LIME explanation successfully generated."
