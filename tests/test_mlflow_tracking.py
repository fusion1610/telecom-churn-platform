import mlflow
import pytest

from src.tracking.mlflow_tracking import (
    configure_mlflow,
    log_artifact_file,
    log_model_metrics,
    log_model_parameters,
)


@pytest.fixture
def mlflow_test_environment(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "test-telecom-churn"

    mlflow.delete_experiment(
        mlflow.get_experiment_by_name(experiment_name).experiment_id
    ) if mlflow.get_experiment_by_name(experiment_name) else None

    configure_mlflow(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )

    return experiment_name


def test_configure_mlflow_creates_experiment(mlflow_test_environment):
    experiment = mlflow.get_experiment_by_name(mlflow_test_environment)

    assert experiment is not None
    assert experiment.name == mlflow_test_environment


def test_log_model_parameters(mlflow_test_environment):
    with mlflow.start_run():
        log_model_parameters(
            model_type="HistGradientBoostingClassifier",
            calibration="sigmoid",
            calibration_cv=5,
            threshold=0.07,
            random_state=42,
            hyperparameters={
                "max_iter": 300,
                "learning_rate": 0.05,
            },
        )

        run = mlflow.get_run(mlflow.active_run().info.run_id)

    assert run.data.tags["model_type"] == "HistGradientBoostingClassifier"
    assert run.data.tags["calibration"] == "sigmoid"
    assert run.data.params["threshold"] == "0.07"
    assert run.data.params["random_state"] == "42"
    assert run.data.params["max_iter"] == "300"


def test_log_model_metrics(mlflow_test_environment):
    with mlflow.start_run():
        log_model_metrics(
            {
                "test_pr_auc": 0.086978,
                "test_roc_auc": 0.578495,
                "test_f1": 0.136784,
            }
        )

        run = mlflow.get_run(mlflow.active_run().info.run_id)

    assert run.data.metrics["test_pr_auc"] == pytest.approx(0.086978)
    assert run.data.metrics["test_roc_auc"] == pytest.approx(0.578495)
    assert run.data.metrics["test_f1"] == pytest.approx(0.136784)


def test_log_artifact_file(mlflow_test_environment, tmp_path):
    artifact = tmp_path / "evaluation.txt"
    artifact.write_text("MLflow artifact test")

    with mlflow.start_run():
        log_artifact_file(artifact)
        run_id = mlflow.active_run().info.run_id

    client = mlflow.MlflowClient()
    artifacts = client.list_artifacts(run_id)

    assert any(item.path == "evaluation.txt" for item in artifacts)