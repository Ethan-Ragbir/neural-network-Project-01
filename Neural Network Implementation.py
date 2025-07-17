import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuralNetwork(nn.Module):
    """A customizable feedforward neural network implemented in PyTorch.
    
    Args:
        layer_sizes (List[int]): List of integers representing the size of each layer.
        activation (str): Activation function to use ('relu', 'tanh', 'sigmoid').
        dropout_rate (float): Dropout probability for regularization (default: 0.0).
    """
    def __init__(self, layer_sizes: List[int], activation: str = 'relu', dropout_rate: float = 0.0):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Build the network layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        
        logger.info(f"Initialized neural network with {len(layer_sizes)-1} layers: {layer_sizes}")
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Returns the specified activation function."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        if activation not in activations:
            logger.error(f"Unsupported activation function: {activation}")
            raise ValueError(f"Activation {activation} not supported. Choose from {list(activations.keys())}")
        return activations[activation]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation and dropout to all but last layer
                x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x

class Trainer:
    """Utility class to train and evaluate the neural network.
    
    Args:
        model (NeuralNetwork): The neural network model to train.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Device to run the model on ('cpu' or 'cuda').
    """
    def __init__(self, model: NeuralNetwork, learning_rate: float = 0.001, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_history = []
        
        logger.info(f"Trainer initialized with learning rate {learning_rate} on {device}")
    
    def train(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 100, batch_size: int = 32) -> List[float]:
        """Train the model on the provided data.
        
        Args:
            x (torch.Tensor): Input data tensor.
            y (torch.Tensor): Target data tensor.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
        
        Returns:
            List[float]: Training loss history.
        """
        self.model.train()
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            self.loss_history.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return self.loss_history
    
    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Evaluate the model on the provided data.
        
        Args:
            x (torch.Tensor): Input data tensor.
            y (torch.Tensor): Target data tensor.
        
        Returns:
            float: Mean loss on the evaluation data.
        """
        self.model.eval()
        with torch.no_grad():
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
        return loss.item()

def generate_synthetic_data(n_samples: int = 1000, n_features: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data for testing the neural network.
    
    Args:
        n_samples (int): Number of samples to generate.
        n_features (int): Number of features per sample.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Input and target tensors.
    """
    x = torch.randn(n_samples, n_features)
    y = torch.sum(x, dim=1, keepdim=True) + torch.randn(n_samples, 1) * 0.1
    logger.info(f"Generated synthetic dataset with {n_samples} samples and {n_features} features")
    return x, y

if __name__ == "__main__":
    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Initialize model
    layer_sizes = [10, 64, 32, 1]  # Example architecture
    model = NeuralNetwork(layer_sizes, activation='relu', dropout_rate=0.3)
    
    # Generate synthetic data
    x, y = generate_synthetic_data(n_samples=1000, n_features=10)
    
    # Train model
    trainer = Trainer(model, learning_rate=0.001, device=device)
    trainer.train(x, y, epochs=100, batch_size=32)
    
    # Evaluate model
    test_loss = trainer.evaluate(x, y)
    logger.info(f"Final test loss: {test_loss:.4f}")