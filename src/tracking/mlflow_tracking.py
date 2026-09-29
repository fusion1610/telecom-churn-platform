import mlflow


DEFAULT_EXPERIMENT_NAME = "telecom-churn"


def configure_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> str:
    """
    Configure the MLflow experiment and return its experiment ID.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    return experiment_id