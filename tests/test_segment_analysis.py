import numpy as np
import pandas as pd

from src.explainability.segment_analysis import (
    calculate_segment_shap_summary,
    calculate_segment_feature_importance,
)


def test_calculate_segment_shap_summary():
    shap_values = np.array([
        [0.2, -0.1],
        [0.4, 0.3],
        [-0.1, 0.2],
    ])

    segments = pd.Series(
        ["A", "A", "B"],
        name="EffectiveSegment",
    )

    result = calculate_segment_shap_summary(
        shap_values=shap_values,
        segments=segments,
        feature_names=["feature_a", "feature_b"],
    )

    assert set(result["EffectiveSegment"]) == {"A", "B"}
    assert "Mean Absolute SHAP" in result.columns
    assert "Mean SHAP" in result.columns

    group_a = result[result["EffectiveSegment"] == "A"].iloc[0]

    assert group_a["Mean Absolute SHAP"] > 0
    assert group_a["Mean SHAP"] != 0


def test_calculate_segment_feature_importance():
    shap_values = np.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0],
    ])

    segments = pd.Series(
        ["A", "A", "B", "B"],
        name="EffectiveSegment",
    )

    result = calculate_segment_feature_importance(
        shap_values=shap_values,
        segments=segments,
        feature_names=["feature_a", "feature_b"],
    )

    assert set(result["EffectiveSegment"]) == {"A", "B"}
    assert set(result["Feature"]) == {"feature_a", "feature_b"}

    group_a_feature_a = result[
        (result["EffectiveSegment"] == "A")
        & (result["Feature"] == "feature_a")
    ].iloc[0]

    assert group_a_feature_a["Mean Absolute SHAP"] == 1.5