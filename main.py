from src.GNNClassfier import logger
from src.GNNClassfier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.GNNClassfier.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from src.GNNClassfier.pipeline.stage_03_dataset_preparation import DatasetPreparationPipeline

# Executing sequentially to ensure all artifacts are generated correctly
stages = [
    ("Data Ingestion", DataIngestionTrainingPipeline()),
    ("Data Transformation", DataTransformationTrainingPipeline()),
    ("Dataset Preparation", DatasetPreparationPipeline())
]

for stage_name, pipeline in stages:
    try:
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")
        pipeline.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"Error in {stage_name}: {str(e)}")
        raise e