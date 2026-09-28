import numpy as np
import pandas as pd
import shap


def get_transformed_feature_names(preprocessor):
    """Return feature names produced by the fitted preprocessor."""
    return preprocessor.get_feature_names_out()


def transform_features(model, X):
    """Transform raw features using a fitted pipeline preprocessor."""
    preprocessor = model.named_steps["preprocessor"]

    return preprocessor.transform(X)


def build_tree_explainer(model):
    """Build a SHAP TreeExplainer for the fitted tree model."""
    classifier = model.named_steps["classifier"]

    return shap.TreeExplainer(classifier)


def calculate_shap_values(model, X):
    """Calculate SHAP values for a fitted tree-model pipeline."""

    X_transformed = transform_features(model, X)

    explainer = build_tree_explainer(model)

    shap_values = explainer.shap_values(X_transformed)

    return shap_values

def aggregate_shap_importance(
    shap_values,
    feature_names,
):
    """Aggregate one-hot encoded SHAP values to original features."""

    shap_array = np.asarray(shap_values)

    aggregated = {}

    for index, feature_name in enumerate(feature_names):
        if "__" in feature_name:
            _, feature_name = feature_name.split(
                "__",
                maxsplit=1,
            )

        # Identify the original categorical feature.
        if feature_name.startswith("CRM_PID_Value_Segment_"):
            original_feature = "CRM_PID_Value_Segment"
        elif feature_name.startswith("EffectiveSegment_"):
            original_feature = "EffectiveSegment"
        else:
            original_feature = feature_name

        aggregated.setdefault(
            original_feature,
            0.0,
        )

        aggregated[original_feature] += np.abs(
            shap_array[:, index]
        ).mean()

    result = pd.DataFrame(
        {
            "Feature": list(aggregated.keys()),
            "Mean Absolute SHAP": list(aggregated.values()),
        }
    )

    result["Relative Importance"] = (
        result["Mean Absolute SHAP"]
        / result["Mean Absolute SHAP"].sum()
    )

    return result.sort_values(
        "Mean Absolute SHAP",
        ascending=False,
    ).reset_index(drop=True)