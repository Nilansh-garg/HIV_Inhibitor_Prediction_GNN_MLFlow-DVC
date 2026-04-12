import os
import torch
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdmolops
from torch_geometric.data import Data

class PredictionPipeline:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load the entire model object as per your Training_model.ipynb logic
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.eval()

    def _get_node_features(self, mol):
        all_node_features = []
        for atom in mol.GetAtoms():
            node_feats = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs()
            ]
            all_node_features.append(node_feats)
        return torch.tensor(np.array(all_node_features), dtype=torch.float)

    def _get_adjacency_info(self, mol):
        adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
        row, col = np.where(adj_matrix != 0)
        edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
        return edge_index

    def predict(self, smiles: str):
        """
        Takes a SMILES string and returns the prediction
        """
        mol_obj = Chem.MolFromSmiles(smiles)
        if mol_obj is None:
            return {"error": "Invalid SMILES string"}

        # Prepare graph data
        node_features = self._get_node_features(mol_obj).to(self.device)
        edge_index = self._get_adjacency_info(mol_obj).to(self.device)
        
        # Create a batch of 1
        batch_index = torch.zeros(node_features.shape[0], dtype=torch.long).to(self.device)

        with torch.no_grad():
            # Forward pass
            # Note: Using the signature from your GNN class: forward(x, edge_index, batch_index)
            output = self.model(node_features, edge_index, batch_index)
            prediction = torch.argmax(output, dim=1).item()
            probability = torch.softmax(output, dim=1).tolist()[0]

        status = "Active" if prediction == 1 else "Inactive"
        return {
            "smiles": smiles,
            "prediction": status,
            "confidence": max(probability)
        }