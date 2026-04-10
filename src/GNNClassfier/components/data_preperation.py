import zipfile
import gdown
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdmolops
from torch_geometric.data import Data, Dataset
from src.GNNClassfier import logger
from src.GNNClassfier.utils.common import get_size
from src.GNNClassfier.entity.config_entity import DataPreparationConfig

# 3. Graph Construction Logic (The Component)
class MoleculeGraphGenerator:
    def __init__(self, config: DataPreparationConfig):
        self.config = config

    def generate_graphs(self, csv_path: Path, output_dir: Path, set_name: str):
        df = pd.read_csv(csv_path)
        logger.info(f"Generating graphs for {set_name} set ({len(df)} molecules)...")
        
        # Ensure processed sub-directory exists (PyTorch Geometric requirement)
        processed_dir = output_dir / "processed"
        os.makedirs(processed_dir, exist_ok=True)

        for i, row in tqdm(enumerate(df.itertuples()), total=len(df), desc=f"Processing {set_name}"):
            mol_obj = Chem.MolFromSmiles(row.smiles)
            if mol_obj is None: continue

            node_features = self._get_node_features(mol_obj)
            edge_features = self._get_edge_features(mol_obj)
            edge_index = self._get_adjacency_info(mol_obj)
            label = torch.tensor([row.HIV_active], dtype=torch.int64)

            data = Data(x=node_features, 
                        edge_index=edge_index, 
                        edge_attr=edge_features, 
                        y=label, 
                        smiles=row.smiles)

            torch.save(data, os.path.join(processed_dir, f"data_{i}.pt"))

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

    def _get_edge_features(self, mol):
        all_edge_features = []
        for bond in mol.GetBonds():
            edge_feats = [bond.GetBondTypeAsDouble(), int(bond.IsInRing())]
            all_edge_features.append(edge_feats)
            # Add reverse edge features for undirected graphs
            all_edge_features.append(edge_feats)
        return torch.tensor(np.array(all_edge_features), dtype=torch.float)

    def _get_adjacency_info(self, mol):
        adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
        row, col = np.where(adj_matrix != 0)
        edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
        return edge_index