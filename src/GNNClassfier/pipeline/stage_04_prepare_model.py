from src.GNNClassfier.config.configuration import ConfigurationManager
from src.GNNClassfier.components.prepare_model import PrepareBaseModel
from src.GNNClassfier import logger

STAGE_NAME = "PREPARE_MODEL_STAGE"

class DatasetPreparationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            prepare_config = config_manager.get_prepare_base_model_config()
            prepare = PrepareBaseModel(prepare_config)
            prepare.init_and_save_model()
        except Exception as e:
            print(f"Error during preparation: {e}")

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DatasetPreparationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e