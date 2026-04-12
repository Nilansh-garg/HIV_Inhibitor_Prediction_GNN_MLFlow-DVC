import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen= True)
class DataIngestionConfig:
    root_dir:Path
    source_URL: str
    local_data_file: Path
    unzipped_data_dir: Path 
    
# 1. Configuration Entity
@dataclass(frozen= True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    train_file: Path
    test_file: Path
    
@dataclass(frozen = True)
class DataPreparationConfig:
    root_dir: Path
    train_csv_path: Path
    test_csv_path: Path
    train_graph_dir: Path
    test_graph_dir: Path
    

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    params_features_size: int
    params_embedding_size: int
    params_num_classes: int


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    base_model_path: Path
    training_data_path: Path
    test_data_path: Path
    params_epochs: int
    params_batch_size: int
    params_learning_rate: float

@dataclass(frozen=True)
class EvaluationConfig:
    base_model_path: Path
    path_of_model: Path
    training_data: Path  # Path to your processed dataset
    score_util_path: Path
    all_params: dict
    mlflow_uri: str
    params_batch_size: int
    params_device: str

