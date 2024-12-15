from clearml import PipelineController

from data_preprocessing import preprocess_data
from train_base_model import train_catboost
from train_blending_model import train_blender


def run_pipeline() -> None:
    """Run the entire pipeline."""
    pipeline = PipelineController(project="Titanic Pipeline", name="Titanic ML Pipeline")

    pipeline.add_function_step(
        name="preprocess_data",
        function=preprocess_data,
        function_kwargs={
            "input_path": "../datasets/titanic.csv",
            "project_name": "Titanic Pipeline",
        },
    )

    pipeline.add_function_step(
        name="train_model_1",
        function=train_catboost,
        function_kwargs={
            "train_dataset_prefix": "train",
            "test_dataset_name": "test",
            "project_name": "Titanic Pipeline",
            "model_num": 1,
            "output_model_path_dir": "../models/",
        },
        parents=["preprocess_data"],
    )

    pipeline.add_function_step(
        name="train_model_2",
        function=train_catboost,
        function_kwargs={
            "train_dataset_prefix": "train",
            "test_dataset_name": "test",
            "project_name": "Titanic Pipeline",
            "model_num": 2,
            "output_model_path_dir": "../models/",
        },
        parents=["preprocess_data"],
    )

    pipeline.add_function_step(
        name="train_blender",
        function=train_blender,
        function_kwargs={
            "project_name": "Titanic Pipeline",
            "test_dataset_name": "test",
            "base_model_artifact_name": "catboost model",
            "output_model_path": "../models/blender_model.pkl",
        },
        parents=["train_model_1", "train_model_2"],
    )

    pipeline.start_locally(run_pipeline_steps_locally=True)
    pipeline.wait()


if __name__ == "__main__":
    run_pipeline()
