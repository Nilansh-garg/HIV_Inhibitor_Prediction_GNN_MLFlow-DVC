import os
import torch
from dataclasses import dataclass
from pathlib import Path
from torch_geometric.loader import DataLoader
import mlflow
from src.GNNClassfier.config.configuration import EvaluationConfig
from src.GNNClassfier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(self.config.params_device)
        self.score = None

    def _load_test_data(self):
        # Loading the .pt file containing the Test Dataset (PyG format)
        self.test_dataset = torch.load(self.config.training_data)
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.config.params_batch_size, 
            shuffle=False
        )

    def evaluation(self):
        # Load the PyTorch GNN model
        self.model = torch.load(self.config.path_of_model)
        self.model.to(self.device)
        self.model.eval()
        
        self._load_test_data()
        
        total_loss = 0
        correct = 0
        criterion = torch.nn.BCELoss()

        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y.float().view(-1, 1))
                
                total_loss += loss.item()
                pred = (out > 0.5).float()
                correct += (pred == data.y.view(-1, 1)).sum().item()

        self.score = [total_loss / len(self.test_loader), correct / len(self.test_dataset)]

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("HIV-Inhibitor-GNN-Evaluation")

        with mlflow.start_run(run_name="evaluation_run"):
            # Log all parameters from params.yaml
            mlflow.log_params(self.config.all_params)

            # Log Metrics
            mlflow.log_metrics({
                "test_loss": self.score[0],
                "test_accuracy": self.score[1]
            })

            # Log Model using the PyTorch flavor
            if self.model:
                mlflow.pytorch.log_model(
                    self.model, 
                    artifact_path="model",
                    registered_model_name="HIV_GAT_Model"
                )