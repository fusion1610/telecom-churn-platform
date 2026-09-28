import numpy as np
import pandas as pd


def calculate_segment_shap_summary(
    shap_values,
    segments,
    feature_names,
):
    """Calculate aggregate SHAP statistics by customer segment."""
    shap_array = np.asarray(shap_values)

    if shap_array.ndim != 2:
        raise ValueError("shap_values must be a 2D array.")

    if len(segments) != shap_array.shape[0]:
        raise ValueError(
            "segments length must match the number of SHAP observations."
        )

    if len(feature_names) != shap_array.shape[1]:
        raise ValueError(
            "feature_names length must match the number of SHAP features."
        )

    frame = pd.DataFrame(
        shap_array,
        columns=feature_names,
        index=segments.index,
    )

    frame["_segment"] = segments.to_numpy()

    rows = []

    for segment, group in frame.groupby("_segment", dropna=False):
        group_shap = group.drop(columns="_segment")

        rows.append(
            {
                "EffectiveSegment": segment,
                "Mean SHAP": group_shap.to_numpy().mean(),
                "Mean Absolute SHAP": np.abs(
                    group_shap.to_numpy()
                ).mean(),
                "Customers": len(group),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("Mean Absolute SHAP", ascending=False)
        .reset_index(drop=True)
    )


def calculate_segment_feature_importance(
    shap_values,
    segments,
    feature_names,
):
    """Calculate feature-level SHAP importance within each segment."""
    shap_array = np.asarray(shap_values)

    if shap_array.ndim != 2:
        raise ValueError("shap_values must be a 2D array.")

    if len(segments) != shap_array.shape[0]:
        raise ValueError(
            "segments length must match the number of SHAP observations."
        )

    if len(feature_names) != shap_array.shape[1]:
        raise ValueError(
            "feature_names length must match the number of SHAP features."
        )

    rows = []

    segments_array = segments.to_numpy()

    for segment in pd.unique(segments_array):
        mask = segments_array == segment

        segment_shap = shap_array[mask]

        for feature_index, feature_name in enumerate(feature_names):
            feature_values = segment_shap[:, feature_index]

            rows.append(
                {
                    "EffectiveSegment": segment,
                    "Feature": feature_name,
                    "Mean SHAP": feature_values.mean(),
                    "Mean Absolute SHAP": np.abs(feature_values).mean(),
                    "Customers": int(mask.sum()),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["EffectiveSegment", "Mean Absolute SHAP"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )