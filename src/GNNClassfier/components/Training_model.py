import torch
import os 
from torch_geometric.loader import DataLoader
from dataclasses import dataclass
from pathlib import Path
# from src.GNNClassfier.components.Training_model import ModelTrainer
from src.GNNClassfier.config.configuration import TrainingConfig

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_model(self):
        # 1. Load Data
        train_data = torch.load(self.config.training_data_path)
        test_data = torch.load(self.config.test_data_path)
        
        train_loader = DataLoader(train_data, batch_size=self.config.params_batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.config.params_batch_size, shuffle=False)

        # 2. Load Model
        model = torch.load(self.config.base_model_path).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.params_learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        # 3. Training Loop
        print(f"Starting training on {self.device} for {self.config.params_epochs} epochs...")
        for epoch in range(1, self.config.params_epochs + 1):
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                out = model(data.x.float(), data.edge_index, data.batch)
                loss = criterion(out, data.y.long().squeeze())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            model.eval()
            correct = 0
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(self.device)
                    out = model(data.x.float(), data.edge_index, data.batch)
                    pred = out.argmax(dim=1)
                    correct += (pred == data.y.long().squeeze()).sum().item()
            
            acc = correct / len(test_data)
            print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc:.4f}")

        # 4. Save Trained Model
        torch.save(model.state_dict(), self.config.trained_model_path)
        print(f"Trained weights saved to: {self.config.trained_model_path}")