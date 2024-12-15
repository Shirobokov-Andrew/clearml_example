import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from clearml import Task, Dataset, TaskTypes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split


def train_blender(
    project_name: str = "Titanic Pipeline",
    test_dataset_name: str = "test",
    base_model_artifact_name: str = "catboost model",
    output_model_path: str = "../models/blender_model.pkl",
) -> None:
    """Train a blending model using Logistic Regression.

    Args:
        :param project_name: ClearML project name.
        :param test_dataset_name: Name of the test dataset containing true labels.
        :param base_model_artifact_name: Artifact model name from training base model.
        :param output_model_path: Path to save the blending model.
    """
    task = Task.init(project_name="Titanic Pipeline", task_name="Train Blender", task_type=TaskTypes.training)

    # Load datasets and models
    test_dataset = Dataset.get(dataset_project=project_name, dataset_name=f"{test_dataset_name}")
    test_local_path = Path(test_dataset.get_local_copy()) / f"data/{test_dataset_name}.csv"
    model1_path = (
        Task.get_task(project_name=project_name, task_name="Train 1")
        .artifacts[base_model_artifact_name]
        .get_local_copy()
    )
    model2_path = (
        Task.get_task(project_name=project_name, task_name="Train 2")
        .artifacts[base_model_artifact_name]
        .get_local_copy()
    )

    model1 = CatBoostClassifier()
    model1.load_model(model1_path)
    model2 = CatBoostClassifier()
    model2.load_model(model2_path)

    test_df = pd.read_csv(test_local_path)
    x, y = test_df.drop(columns="Survived"), test_df["Survived"]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

    scores_train_1 = model1.predict_proba(train_x)[:, 1]
    scores_train_2 = model2.predict_proba(train_x)[:, 1]
    scores_test_1 = model1.predict_proba(test_x)[:, 1]
    scores_test_2 = model2.predict_proba(test_x)[:, 1]

    # Train blending model
    scores_train = np.column_stack((scores_train_1, scores_train_2))
    scores_test = np.column_stack((scores_test_1, scores_test_2))
    model = LogisticRegression()
    model.fit(scores_train, train_y)

    # Log metrics
    y_pred = model.predict_proba(scores_test)[:, 1]
    task.get_logger().report_single_value(name="Test LogLoss", value=log_loss(test_y, y_pred))
    task.get_logger().report_single_value(name="Test ROC AUC", value=roc_auc_score(test_y, y_pred))

    # Save blending model
    output_model_path = Path(output_model_path)
    joblib.dump(model, str(output_model_path))
    task.upload_artifact(name=f"blending model", artifact_object=output_model_path)

    task.close()


# Here we can optionally add click or argparse to set specific arguments for train_blender function
if __name__ == "__main__":
    train_blender()
