from src.GNNClassfier.config.configuration import ConfigurationManager
from src.GNNClassfier.components.Training_model import ModelTrainer
from src.GNNClassfier import logger

STAGE_NAME = "TRAINING_MODEL_STAGE"

class DatasetPreparationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            training_config = config_manager.get_training_config()
            trainer = ModelTrainer(training_config)
            trainer.train_model()
        except Exception as e:
            print(f"Error during training: {e}")
            

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DatasetPreparationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e