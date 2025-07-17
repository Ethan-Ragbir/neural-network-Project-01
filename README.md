Neural Network Implementation
A modular and flexible feedforward neural network implemented in PyTorch, designed for ease of use and customization.
Features

Configurable layer sizes and activation functions (ReLU, Tanh, Sigmoid)
Dropout regularization
Training utilities with loss tracking and batch processing
Synthetic data generation for testing
Comprehensive logging
Support for CPU and GPU training

Installation

Ensure Python 3.8+ is installed.
Install required dependencies:

pip install torch numpy

Usage
import torch
from neural_network import NeuralNetwork, Trainer, generate_synthetic_data

# Generate synthetic data
x, y = generate_synthetic_data(n_samples=1000, n_features=10)

# Initialize model
model = NeuralNetwork(layer_sizes=[10, 64, 32, 1], activation='relu', dropout_rate=0.3)

# Train model
trainer = Trainer(model, learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu')
trainer.train(x, y, epochs=100, batch_size=32)

# Evaluate model
test_loss = trainer.evaluate(x, y)
print(f"Test Loss: {test_loss:.4f}")

Project Structure

neural_network.py: Core neural network implementation and training utilities
requirements.txt: Project dependencies
tests/: Unit tests for the neural network
examples/: Example scripts demonstrating usage

Running Tests
pip install pytest
pytest tests/

Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.
License
MIT License
