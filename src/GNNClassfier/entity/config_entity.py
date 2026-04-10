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
    