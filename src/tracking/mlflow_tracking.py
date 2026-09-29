from pathlib import Path

import mlflow


DEFAULT_EXPERIMENT_NAME = "telecom-churn"


def configure_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
) -> str:
    """Configure MLflow tracking and return the experiment ID."""
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    return experiment_id


def log_model_parameters(
    *,
    model_type: str,
    calibration: str,
    calibration_cv: int,
    threshold: float,
    random_state: int,
    hyperparameters: dict,
) -> None:
    """Log model configuration to the active MLflow run."""
    mlflow.set_tag("model_type", model_type)
    mlflow.set_tag("calibration", calibration)

    mlflow.log_param("calibration_cv", calibration_cv)
    mlflow.log_param("threshold", threshold)
    mlflow.log_param("random_state", random_state)

    for name, value in hyperparameters.items():
        mlflow.log_param(name, value)


def log_model_metrics(metrics: dict) -> None:
    """Log evaluation metrics to the active MLflow run."""
    for name, value in metrics.items():
        mlflow.log_metric(name, float(value))


def log_artifact_file(path: str | Path) -> None:
    """Log a single file as an MLflow artifact."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Artifact does not exist: {path}")

    mlflow.log_artifact(str(path))