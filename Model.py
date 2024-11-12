from torch_geometric.nn import global_max_pool
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import gvp
from torch_geometric.data import Dataset, Data
import torch_geometric
import os
import torch
from Bio.PDB import PDBParser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from gvp import GVP, GVPConvLayer, LayerNorm
from torch_geometric.utils import dense_to_sparse
import wandb
import torch.optim as optim
import argparse

class GVP_GNN(nn.Module):
    def __init__(
        self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim,
        num_layers=5, drop_rate=0.0, embedding=(1,0)
    ):
        super(GVP_GNN, self).__init__()
        #activations=(F.relu, F.relu) # Set activation functions, first is SiLU and second is None

        self.W_v = nn.Sequential(
            #LayerNorm(node_in_dim),  # Apply LayerNorm normalization to node inputs
            GVP(
                node_in_dim, node_h_dim, activations=(None, None), vector_gate=False
            )  # Process nodes using the GVP layer
        )
        self.W_e = nn.Sequential(
            #LayerNorm(edge_in_dim),  # Apply LayerNorm normalization to edge inputs
            GVP(
                edge_in_dim, edge_h_dim, activations=(None, None), vector_gate= False
            )  # Process edges using the GVP layer
        )
        
    
        self.layers = nn.ModuleList(
            GVPConvLayer(
                node_h_dim,
                edge_h_dim,
                #activations = activations,
                vector_gate=True,
                #drop_rate=drop_rate,
            )
            for _ in range(num_layers)
        )  # Create multiple GVP convolution layers    
        
        ns, _ = (
            node_h_dim  # Extract the scalar feature count from node hidden dimensions
        )
        
        self.W_out = nn.Sequential(
            #LayerNorm(
            #    node_h_dim
            #),  # Apply LayerNorm normalization to node hidden features
            GVP(node_h_dim, (ns, 0), vector_gate=True), # Put activations=activations back
        )  # Output node features using the GVP layer

        self.dense = nn.Sequential(
            nn.Linear(ns + embedding[1], ns),
            nn.SiLU(inplace=True),  # Define a linear layer and SiLU activation
            nn.Linear(ns, 1),                 # Second linear layer (additional hidden layer)
            #nn.SiLU(inplace=True),             # Activation function
            #nn.Linear(ns, ns // 2),            # Third linear layer (optional)
            #nn.SiLU(inplace=True),             # Activation function
            #nn.Dropout(p=drop_rate),           # Optional dropout layer, uncomment if needed
            #nn.Linear(ns // 2, 1)              # Final linear layer to perform regression
            #nn.Dropout(p=drop_rate),  # Add a Dropout layer
        )        


    def forward(self, h_V, h_E,edge_index,batch, pembedding):
        h_V = self.W_v(h_V)  # Process node features using the GVP layer
        h_E = self.W_e(h_E)  # Process edge features using the GVP layer

        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)

        out = self.W_out(h_V)
        out = global_mean_pool(out, batch)
        #out = torch.cat([out, pembedding.squeeze(0)], dim=-1) # Going to add this once I need to add another embedding
        out = self.dense(out)
        return out
    

        

