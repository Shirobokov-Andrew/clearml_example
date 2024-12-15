import pandas as pd
from catboost import CatBoostClassifier
from clearml import Task, Dataset, TaskTypes
from sklearn.metrics import roc_auc_score, log_loss
from pathlib import Path


def train_catboost(
    train_dataset_prefix: str = "train",
    test_dataset_name: str = "test",
    project_name: str = "Titanic Pipeline",
    model_num: int = 2,
    output_model_path_dir: str = "../models/",
) -> None:
    """Train a CatBoost model.

    Args:
        :param train_dataset_prefix: Name of the ClearML train dataset.
        :param test_dataset_name:N ame of the ClearML val dataset.
        :param project_name: Project where the dataset is stored.
        :param model_num: Model number to train.
        :param output_model_path_dir: Path to save the trained model.
    """
    task = Task.init(
        project_name="Titanic Pipeline",
        task_name=f"Train {model_num}",
        tags=[f"model_{model_num}"],
        task_type=TaskTypes.training,
    )
    task_params = {
        "train_dataset_prefix": train_dataset_prefix,
        "test_dataset_name": test_dataset_name,
        "project_name": project_name,
        "output_model_path_dir": output_model_path_dir,
        "model_num": model_num,
    }
    task_params = task.connect(task_params)

    # Load datasets
    train_dataset = Dataset.get(dataset_project=project_name, dataset_name=f"{train_dataset_prefix}{model_num}")
    test_dataset = Dataset.get(dataset_project=project_name, dataset_name=f"{test_dataset_name}")
    train_local_path = Path(train_dataset.get_local_copy()) / f"data/{train_dataset_prefix}{model_num}.csv"
    test_local_path = Path(test_dataset.get_local_copy()) / f"data/{test_dataset_name}.csv"
    train_df = pd.read_csv(train_local_path)
    test_df = pd.read_csv(test_local_path)
    x_train, y_train, x_test, y_test = (
        train_df.iloc[:, :-1],
        train_df.iloc[:, -1],
        test_df.iloc[:, :-1],
        test_df.iloc[:, -1],
    )

    # Train model
    boosting_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": 200,
        "random_seed": 42,
        "iterations": 50,
        "max_depth": 3,
    }
    model = CatBoostClassifier(**boosting_params)
    model.fit(
        x_train,
        y_train,
        eval_set=(x_test, y_test),
        use_best_model=True,
        plot=True,
    )

    # Log metrics
    y_pred = model.predict_proba(x_test)[:, 1]
    task.get_logger().report_single_value(name="Test LogLoss", value=log_loss(y_test, y_pred))
    task.get_logger().report_single_value("Test ROC AUC", value=roc_auc_score(y_test, y_pred))

    # Save model
    output_model_path = Path(output_model_path_dir) / f"model{model_num}.cbm"
    model.save_model(str(output_model_path))
    task.upload_artifact(name=f"catboost model", artifact_object=output_model_path)

    task.close()


# Here we can optionally add click or argparse to set specific arguments for train_catboost function
if __name__ == "__main__":
    train_catboost()
