import numpy as np

from src.explainability.shap_analysis import (
    get_transformed_feature_names,
    aggregate_shap_importance
)


class DummyPreprocessor:

    def get_feature_names_out(self):
        return np.array(
            [
                "num__Active_subscribers",
                "num__Total_SUBs",
                "cat__CRM_PID_Value_Segment_Gold",
            ]
        )


def test_get_transformed_feature_names():
    preprocessor = DummyPreprocessor()

    result = get_transformed_feature_names(preprocessor)

    assert list(result) == [
        "num__Active_subscribers",
        "num__Total_SUBs",
        "cat__CRM_PID_Value_Segment_Gold",
    ]

def test_aggregate_shap_importance_groups_transformed_features():
    shap_values = np.array(
        [
            [1.0, -2.0, 0.5],
            [2.0, 1.0, -0.5],
        ]
    )

    feature_names = [
        "numeric__ARPU",
        "categorical__CRM_PID_Value_Segment_Gold",
        "categorical__CRM_PID_Value_Segment_Silver",
    ]

    result = aggregate_shap_importance(
        shap_values=shap_values,
        feature_names=feature_names,
    )

    assert set(result["Feature"]) == {
        "ARPU",
        "CRM_PID_Value_Segment",
    }

    segment_importance = result.loc[
        result["Feature"] == "CRM_PID_Value_Segment",
        "Mean Absolute SHAP",
    ].iloc[0]

    assert segment_importance == 2.0