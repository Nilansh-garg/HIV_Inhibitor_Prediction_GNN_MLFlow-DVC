from src.GNNClassfier.constants import *
from src.GNNClassfier.utils.common import read_yaml, create_directories
from src.GNNClassfier.entity.config_entity import DataIngestionConfig, DataTransformationConfig, DataPreparationConfig

class ConfigurationManager:
    def __init__(self, config_file_path: Path = CONFIG_FILE_PATH, params_file_path: Path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzipped_data_dir=Path(config.unzipped_data_dir)
        )
        
        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            train_file=Path(config.train_file),
            test_file=Path(config.test_file)
        )

    def get_data_preparation_config(self) -> DataPreparationConfig:
        config = self.config.data_preparation
        trans_config = self.config.data_transformation
        
        create_directories([config.root_dir, config.train_graph_dir, config.test_graph_dir])

        return DataPreparationConfig(
            root_dir=Path(config.root_dir),
            train_csv_path=Path(trans_config.train_file),
            test_csv_path=Path(trans_config.test_file),
            train_graph_dir=Path(config.train_graph_dir),
            test_graph_dir=Path(config.test_graph_dir)
        )