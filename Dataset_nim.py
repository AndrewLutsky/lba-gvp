import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Dataset, Data
import torch_geometric
import os
import torch
from Bio.PDB import PDBParser,  NeighborSearch
import numpy as np
from torch_geometric.utils import dense_to_sparse
import wandb
import argparse
from torch_geometric.nn import knn_graph
from gvp.data import _rbf

class PDBBindDataset(Dataset):
    def __init__(self, df, element_list=["S", "P", "O", "N", "C", "H"], embeddings_on=True):
        """
        Initialize the dataset with a DataFrame containing file paths and target values.

        Parameters:
        - df (pd.DataFrame): DataFrame with 'filepath' and 'target' columns
        - element_list (list of str): List of elements for one-hot encoding
        """
        super(PDBBindDataset, self).__init__()
        self.df = df  # DataFrame with file paths and target values
        self.embeddings = pd.read_csv("out_25.csv")
        self.embeddings.columns = ['filepath'] + [f'Feature_{i}' for i in range(self.embeddings.shape[1] - 1)]
        self.embeddings_on = embeddings_on
        self.element_list = element_list  # Elements for one-hot encoding
        self.parser = PDBParser(QUIET=True)
        self.encoder = OneHotEncoder(sparse=False)
        unique_elements = np.array(element_list).reshape(-1, 1)
        self.encoder.fit(unique_elements)
                            
            
    def len(self):
        return len(self.df)  # Number of entries based on the DataFrame
    def one_hot_encode_element(self, element):
        encoded_element = self.encoder.transform(np.array([[element]]))
        return encoded_element
 
    def get(self, idx):
        # Get file path and target value from the DataFrame
        row = self.df.iloc[idx]
        pocket_path = row['filepath']
        if self.embeddings_on:
            embedding = self.embeddings[self.embeddings['filepath'] == row['filepath']]
            feature_columns = embedding.filter(regex='^Feature_')
            feature_array = feature_columns.values.flatten()
        
        
        target = row['dG']  # The target value for prediction

        # Print the file being loaded (optional for debugging)
        print(f"Loading file: {os.path.basename(pocket_path)}")

        # Parse the structure
        structure = self.parser.get_structure(f"pocket_{idx}", pocket_path)
        
        # Lists for scalar (one-hot) and vector (position) features
        node_scalar_features = []
        node_vector_features = []
        atom_positions = []
        atom_map = {}
        edge_idx = []
        edge_attr = []
        atom_idx = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    atoms = list(residue.get_atoms())
                    for atom in atoms:
                        try:
                            element = atom.element
                            # One-hot encoding for scalar features
                            scalar_feature = self.one_hot_encode_element(element)
                            node_scalar_features.append(scalar_feature)

                            # Vector feature (e.g., atomic positions)
                            position = atom.coord  # Numpy array (x, y, z)
                            node_vector_features.append(position)
                            atom_positions.append(position)
                            atom_map[atom] = atom_idx
                            atom_idx += 1   
                        except Exception as e:
                            print(e) 
        
         
        #node_vector_features = torch.tensor(node_vector_features)       
        #node_scalar_features = torch.tensor(node_scalar_features)       
        # Convert lists to torch tensors
        n_s = torch.tensor(node_scalar_features,dtype=torch.float32).squeeze(1) # Should be of shape (num_atoms, 6)
        n_v = torch.tensor(node_vector_features,dtype=torch.float32).unsqueeze(1)  # Shape: (num_atoms, 1, 3)
        edge_index = knn_graph(torch.tensor(atom_positions), k=5, loop=False).T
        node_vector_features = torch.tensor(node_vector_features)       
        node_scalar_features = torch.tensor(node_scalar_features)       
        for pair in edge_index:
            first_atom = node_vector_features[pair[0]]
            second_atom = node_vector_features[pair[1]]
            displacement = second_atom - first_atom
            
            edge_attr.append(displacement)
            del first_atom
            del second_atom
            del displacement
        
        lengths = [np.sqrt(sum(i * i for i in vector)) for vector in edge_attr]

        norm_edge_attr =[vector / length for vector, length in zip(edge_attr, lengths)]             
        norm_edge_attr = torch.stack(norm_edge_attr)

        e_s = _rbf(torch.tensor(lengths), D_max=3.5, D_count=16)
        e_v = norm_edge_attr.unsqueeze(1)
        print(n_s.shape, n_v.shape)
        print(e_s.shape, e_v.shape)
        print(edge_index.T.shape)
        pembedding=torch.tensor(feature_array)
        print("Pembedding")
        print(pembedding.shape)
        # Create Data object for PyTorch Geometric with target value
        data = Data(
            x=n_s, # Should be of shape(num_nodes, 6)
            pos=n_v, # Should be of shape (num_nodes, 3)
            edge_index=edge_index.T, # Should be of shape (2, num_edges)
            edge_attr=e_v,  # Should be of shape (num_edges, 3)
            edge_scalars=e_s,
            pembedding=pembedding,
            y=torch.tensor([target], dtype=torch.float32) # Should be of shape(1, 1)
        )

        return data

