import pandas as pd
import argparse
import Model
import Dataset
from sklearn.preprocessing import OneHotEncoder
import os
import torch
import torch.nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import wandb  # Make sure to install wandb with `pip install wandb`
from sklearn.model_selection import train_test_split


def main(epochs, lr, wd):
    train_df = pd.read_csv("testing_fit.csv")
    train_data, test_data = train_test_split(train_df, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    train_dataset = Dataset.PDBBindDataset(train_data)
    val_dataset = Dataset.PDBBindDataset(val_data)

    # Create directory for saving models if it doesn't exist
    checkpoint_dir = "./models/gvp_2007_fitting_test"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize WandB
    wandb.init(project="GNN_fitting")  # Replace with your WandB username or team 
    # Assume dataset is already loaded and split into train and val sets
    train_loader = DataLoader(train_dataset,batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8,shuffle=True, num_workers=2)
    
    # Define model, optimizer, and loss function
    model = Model.GVP_GNN(
        node_in_dim = (6, 1),
        node_h_dim = (256, 128),
        edge_in_dim = (16, 1),
        edge_h_dim = (256, 128),
        num_layers=5
                            )  # Replace with your model class definition
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    wandb.watch(model, log='all', log_freq = 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Initialize variables for checkpointing
    best_val_loss = float('inf')
    
    model.train()
    # Training loop with checkpointing
    for epoch in range(epochs):
        train_loss = 0.0
    
        model.train()
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            h_V = (batch.x, batch.pos)
            h_E = (batch.edge_scalars, batch.edge_attr) 
        

            out = model(h_V, h_E, batch.edge_index, batch.batch, batch.pembedding)
            loss = torch.nn.MSELoss()
            loss_value = loss(out, batch.y.unsqueeze(1))  # Adjust for regression tasks if needed
            
            # Backward pass
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            # Accumulate batch loss
            train_loss += loss_value.item()
    
        # Average training loss for logging
        avg_train_loss = train_loss / len(train_loader)
    
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                # Forward pass
                h_V = (batch.x, batch.pos)
                h_E = (batch.edge_scalars, batch.edge_attr)
                
                out = model(h_V, h_E, batch.edge_index, batch.batch, batch.pembedding)
                loss_val = torch.nn.MSELoss()
                
                val_loss += loss_val(out, batch.y.unsqueeze(1)).item()
                
        # Calculate average validation loss and accuracy
        avg_val_loss = val_loss / len(val_loader)
    
        # Log metrics to WandB
        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": avg_train_loss,
            "Validation Loss": avg_val_loss,
        })
    
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    
        # Checkpoint: Save the model if it has the best validation loss so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}_val_loss_{avg_val_loss:.4f}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Finish WandB run
    wandb.finish()
    return    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified epochs and learning rate.")

    # Add arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--wd', type=float, default=1e-1, help='weight decay')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of convolution layers')
    #parser.add_argument()
    # Parse arguments
    args = parser.parse_args()
    main(args.epochs, args.lr, args.wd) 
