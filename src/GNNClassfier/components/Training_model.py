import torch
import os 
from torch_geometric.loader import DataLoader
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from src.GNNClassfier.config.configuration import TrainingConfig

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_model(self):
        # 1. Helper function to load all .pt files from the processed directory
        def load_graphs_from_dir(directory_path):
            processed_dir = Path(directory_path) / "processed"
            
            if not processed_dir.exists():
                raise FileNotFoundError(f"The directory {processed_dir} does not exist. Check your Stage 03 output.")
            
            graph_files = sorted([str(f) for f in processed_dir.glob("*.pt")])
            
            if len(graph_files) == 0:
                raise ValueError(f"No .pt files found in {processed_dir}")
                
            print(f"Loading {len(graph_files)} graphs from {processed_dir}...")
            # Using tqdm for data loading as it takes time for 62k graphs
            return [torch.load(f, weights_only=False) for f in tqdm(graph_files, desc="Loading Data")]

        # 2. Load Data
        train_data = load_graphs_from_dir(self.config.training_data_path)
        test_data = load_graphs_from_dir(self.config.test_data_path)
        
        train_loader = DataLoader(train_data, batch_size=self.config.params_batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.config.params_batch_size, shuffle=False)

        # 3. Load Model
        model = torch.load(self.config.base_model_path, weights_only=False).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.params_learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        # --- Local Early Stopping Parameters ---
        best_acc = 0.0
        patience = 5  # Stop if accuracy doesn't improve for 5 epochs
        counter = 0
        
        # 4. Training Loop
        print(f"Starting training on {self.device} for {self.config.params_epochs} epochs...")
        
        for epoch in range(1, self.config.params_epochs + 1):
            model.train()
            total_loss = 0
            
            # Adding tqdm for training batches
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.params_epochs} [Train]")
            for data in train_pbar:
                data = data.to(self.device)
                optimizer.zero_grad()
                
                out = model(data.x.float(), data.edge_index, data.batch)
                loss = criterion(out, data.y.long().squeeze())
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                train_pbar.set_postfix(loss=loss.item())
            
            # Validation
            model.eval()
            correct = 0
            with torch.no_grad():
                for data in tqdm(test_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                    data = data.to(self.device)
                    out = model(data.x.float(), data.edge_index, data.batch)
                    pred = out.argmax(dim=1)
                    target = data.y.long().squeeze()
                    correct += (pred == target).sum().item()
            
            avg_loss = total_loss / len(train_loader)
            acc = correct / len(test_data)
            
            print(f"Summary -> Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Test Acc: {acc:.4f}")

            # --- Early Stopping Logic ---
            if acc > best_acc:
                best_acc = acc
                counter = 0
                # Save the best model weights immediately when accuracy improves
                os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
                torch.save(model.state_dict(), self.config.trained_model_path)
                print(f"--> Accuracy improved! Saving best weights to {self.config.trained_model_path}")
            else:
                counter += 1
                print(f"--> No improvement in accuracy. EarlyStopping counter: {counter}/{patience}")

            if counter >= patience:
                print(f"Stopping early at epoch {epoch}. Best accuracy achieved: {best_acc:.4f}")
                break

        print("Training process completed.")