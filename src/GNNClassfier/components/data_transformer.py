import os
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.GNNClassfier.constants import *
from src.GNNClassfier.utils.common import read_yaml, create_directories
from src.GNNClassfier import logger
from src.GNNClassfier.entity.config_entity import DataTransformationConfig

# 3. Component Class
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def oversample_and_split(self):
        try:
            # Load dataset
            df = pd.read_csv(self.config.data_path)
            logger.info(f"Loaded dataset from {self.config.data_path}. Shape: {df.shape}")

            # Train-Test Split (before oversampling to prevent data leakage)
            train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["HIV_active"])
            logger.info("Split data into train and test sets with stratification.")

            # Apply Oversampling to the TRAIN set only
            neg_class_count = train["HIV_active"].value_counts()[0]
            pos_class_count = train["HIV_active"].value_counts()[1]
            multiplier = int(neg_class_count / pos_class_count) - 1

            replicated_pos = [train[train["HIV_active"] == 1]] * multiplier
            train_oversampled = pd.concat([train] + replicated_pos, ignore_index=True)
            
            # Shuffle and reset index
            train_oversampled = train_oversampled.sample(frac=1, random_state=42).reset_index(drop=True)
            train_oversampled["index"] = train_oversampled.index
            
            logger.info(f"Oversampling complete. New train shape: {train_oversampled.shape}")

            # Save files
            train_oversampled.to_csv(self.config.train_file, index=False)
            test.to_csv(self.config.test_file, index=False)
            
            logger.info(f"Saved train and test files to {self.config.root_dir}")

        except Exception as e:
            raise e
        