from src.GNNClassfier.config.configuration import ConfigurationManager
from src.GNNClassfier.components.data_preperation import MoleculeGraphGenerator
from src.GNNClassfier import logger

STAGE_NAME = "Dataset Preparation stage"

class DatasetPreparationPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        prep_config = config_manager.get_data_preparation_config()
        generator = MoleculeGraphGenerator(config=prep_config)
        
        # Process Training Set Graphs
        generator.generate_graphs(prep_config.train_csv_path, prep_config.train_graph_dir, "Training")
        
        # Process Test Set Graphs
        generator.generate_graphs(prep_config.test_csv_path, prep_config.test_graph_dir, "Test")

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DatasetPreparationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e