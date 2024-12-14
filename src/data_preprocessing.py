import pandas as pd
from clearml import Task, Dataset, TaskTypes
from pathlib import Path
from sklearn.model_selection import train_test_split


def preprocess_data(
    input_path: str = "../datasets/titanic.csv",
    project_name: str = "Titanic Pipeline",
) -> None:
    """Preprocess data and save splits as ClearML datasets.

    Args:
        :param input_path: Path to the input CSV file.
        :param project_name: ClearML project name.
    """
    task = Task.init(
        project_name=project_name,
        task_name="Titanic data preprocessing",
        task_type=TaskTypes.data_processing,
        tags=["ver1"],
    )
    # Preparing two train parts for two models and one test part
    df = pd.read_csv(input_path)
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=42, shuffle=True)
    train_df_1 = train_df.iloc[: len(train_df) // 2, :]
    train_df_2 = train_df.iloc[len(train_df) // 2 :, :]

    x_train_1 = train_df_1.drop(columns=["Name", "Ticket", "Cabin", "Survived"])
    x_train_2 = train_df_2.drop(columns=["Name", "Ticket", "Cabin", "Survived"])
    x_test = test_df.drop(columns=["Name", "Ticket", "Cabin", "Survived"])

    x_train_1[x_train_1["Sex"] == "female"] = 1
    x_train_1[x_train_1["Sex"] == "male"] = 0
    x_train_2[x_train_2["Sex"] == "female"] = 1
    x_train_2[x_train_2["Sex"] == "male"] = 0
    x_test[x_test["Sex"] == "female"] = 1
    x_test[x_test["Sex"] == "male"] = 0

    y_train_1 = train_df_1["Survived"]
    y_train_2 = train_df_2["Survived"]
    y_test = test_df["Survived"]

    # Uploading some artifact
    task.upload_artifact(name="x_train_1_describe", artifact_object=x_train_1.describe())
    task.upload_artifact(name="x_train_2_describe", artifact_object=x_train_2.describe())
    task.upload_artifact(name="x_test_describe", artifact_object=x_test.describe())

    # Save datasets
    for name, data in [
        ("train1", (x_train_1, y_train_1)),
        ("train2", (x_train_2, y_train_2)),
        ("test", (x_test, y_test)),
    ]:
        data_path = Path(f"../datasets/{name}.csv")
        pd.concat(data, axis=1).to_csv(str(data_path), index=False)
        dataset = Dataset.create(dataset_project=project_name, dataset_name=f"{name}")
        dataset.add_files(path=str(data_path), dataset_path="data")
        dataset.upload()
        dataset.finalize()

    task.close()


# Here we can optionally add click or argparse to set specific arguments for preprocess_data function
if __name__ == "__main__":
    preprocess_data()
