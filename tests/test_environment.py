import importlib


def test_runtime_dependencies_are_importable():
    required_modules = [
        "fastapi",
        "uvicorn",
        "mlflow",
        "pandas",
        "numpy",
        "sklearn",
        "pydantic",
    ]

    for module_name in required_modules:
        assert importlib.import_module(module_name) is not None