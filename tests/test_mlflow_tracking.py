import mlflow

from src.tracking.mlflow_tracking import configure_mlflow


def test_configure_mlflow(tmp_path, monkeypatch):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "test-telecom-churn"

    experiment_id = configure_mlflow(experiment_name)

    experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    assert experiment is not None
    assert experiment.experiment_id == experiment_id