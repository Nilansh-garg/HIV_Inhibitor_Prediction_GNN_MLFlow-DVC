from src.GNNClassfier import logger
from src.GNNClassfier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.GNNClassfier.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from src.GNNClassfier.pipeline.stage_03_dataset_preparation import DatasetPreparationPipeline

# Stage 1: Data Ingestion
STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

# Stage 2: Data Transformation
STAGE_NAME = "Data Transformation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_transformation = DataTransformationTrainingPipeline()
   data_transformation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

# Stage 3: Dataset Preparation (Graph Creation)
STAGE_NAME = "Dataset Preparation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   dataset_preparation = DatasetPreparationPipeline()
   dataset_preparation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e