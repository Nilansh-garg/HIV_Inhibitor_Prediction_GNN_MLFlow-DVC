# GNN Architecture Definition
import os
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from src.GNNClassfier.config.configuration import PrepareBaseModelConfig

class GNN(torch.nn.Module):
    def __init__(self, features_size, embedding_size, num_classes):
        super(GNN, self).__init__()
        # 3 GAT Blocks with TopKPooling
        self.conv1 = GATConv(features_size, embedding_size, heads=3, dropout=0.3)
        self.head_transform1 = Linear(embedding_size*3, embedding_size)
        self.pool1 = TopKPooling(embedding_size, ratio=0.8)

        self.conv2 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.3)
        self.head_transform2 = Linear(embedding_size*3, embedding_size)
        self.pool2 = TopKPooling(embedding_size, ratio=0.8)

        self.conv3 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.3)
        self.head_transform3 = Linear(embedding_size*3, embedding_size)
        self.pool3 = TopKPooling(embedding_size, ratio=0.8)

        self.linear1 = Linear(embedding_size*2, embedding_size)
        self.linear2 = Linear(embedding_size, num_classes)

    def forward(self, x, edge_index, batch_index):
        # Block 1
        x = self.conv1(x, edge_index)
        x = self.head_transform1(x)
        x, edge_index, _, batch_index, _, _ = self.pool1(x, edge_index, None, batch_index)
        x1 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        # Block 2
        x = self.conv2(x, edge_index)
        x = self.head_transform2(x)
        x, edge_index, _, batch_index, _, _ = self.pool2(x, edge_index, None, batch_index)
        x2 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        # Block 3
        x = self.conv3(x, edge_index)
        x = self.head_transform3(x)
        x, edge_index, _, batch_index, _, _ = self.pool3(x, edge_index, None, batch_index)
        x3 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        x = x1 + x2 + x3
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.linear2(x)

# Component Class
class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def init_and_save_model(self):
        model = GNN(
            features_size=self.config.params_features_size,
            embedding_size=self.config.params_embedding_size,
            num_classes=self.config.params_num_classes
        )
        torch.save(model, self.config.base_model_path)
        print(f"Base model saved to: {self.config.base_model_path}")
