from clearml import PipelineController, Task


def run_pipeline() -> None:
    """Run the entire pipeline with ClearML tasks for each step."""
    pipeline = PipelineController(
        project="Titanic Pipeline",
        name="Titanic ML Pipeline",
        version="0.1",
    )

    # Step 1: Preprocess Data
    preprocess_task = Task.create(
        project_name="Titanic Pipeline",
        task_name="Preprocess Data",
        task_type=Task.TaskTypes.data_processing,
        script="./data_preprocessing.py",
        requirements_file="../requirements.txt",
        docker="python:3.12-slim",
        docker_args="--network=host",
    )
    pipeline.add_step(
        name="preprocess_data",
        base_task_id=preprocess_task.id,
        parameter_override={
            "General/input_path": "../datasets/titanic.csv",
        },
        execution_queue="local",  # This makes sure it runs locally
    )

    # Step 2: Train Model 1
    train_model_1_task = Task.create(
        project_name="Titanic Pipeline",
        task_name="Train CatBoost Model 1",
        task_type=Task.TaskTypes.training,
        script="./train_base_model.py",
        requirements_file="../requirements.txt",
        docker="python:3.12-slim",
        docker_args="--network=host",
    )
    pipeline.add_step(
        name="train_model_1",
        base_task_id=train_model_1_task.id,
        parents=["preprocess_data"],
        parameter_override={
            "General/model_num": 1,
            "General/output_model_path_dir": "../models/",
        },
        execution_queue="local",  # Run locally
    )

    # Step 3: Train Model 2
    train_model_2_task = Task.create(
        project_name="Titanic Pipeline",
        task_name="Train CatBoost Model 2",
        task_type=Task.TaskTypes.training,
        script="./train_base_model.py",
        requirements_file="../requirements.txt",
        docker="python:3.12-slim",
        docker_args="--network=host",
    )
    pipeline.add_step(
        name="train_model_2",
        base_task_id=train_model_2_task.id,
        parents=["preprocess_data"],
        parameter_override={
            "General/model_num": 2,
            "General/output_model_path_dir": "../models/",
        },
        execution_queue="local",  # Run locally
    )

    # Step 4: Train Blender
    train_blender_task = Task.create(
        project_name="Titanic Pipeline",
        task_name="Train Blender Model",
        task_type=Task.TaskTypes.training,
        script="./train_blending_model.py",
        requirements_file="../requirements.txt",
        docker="python:3.12-slim",
        docker_args="--network=host",
    )
    pipeline.add_step(
        name="train_blender",
        base_task_id=train_blender_task.id,
        parents=["train_model_1", "train_model_2"],
        parameter_override={
            "General/output_model_path": "../models/blender_model.pkl",
        },
        execution_queue="local",  # Run locally
    )

    # Start the pipeline
    pipeline.start(queue="local")  # Specify 'local' here to enforce local execution
    pipeline.wait()


if __name__ == "__main__":
    run_pipeline()
