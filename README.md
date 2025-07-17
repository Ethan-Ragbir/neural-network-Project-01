#  Neural Network Project 01 

A modular feedforward neural network implementation built with PyTorch, designed for both regression and classification tasks. This implementation provides a flexible architecture with configurable layers, activation functions, and training parameters, along with comprehensive testing and documentation.

## Overview

This project implements a fully connected feedforward neural network with customizable architecture and training procedures. The network supports variable layer configurations, multiple activation functions, dropout regularization, and both CPU and GPU execution. The implementation includes a dedicated trainer class for streamlined model training, synthetic data generation for testing, and comprehensive documentation including mathematical foundations.

### Key Features

**Flexible Architecture**: Configurable layer sizes, activation functions (ReLU, Tanh, Sigmoid), and dropout rates
**Optimized Training**: Mini-batch training with Adam optimizer and configurable learning parameters  
**Hardware Acceleration**: Automatic CPU/GPU device detection and utilization
**Comprehensive Logging**: Detailed training progress tracking and loss monitoring
**Robust Testing**: Unit test suite with continuous integration
**Technical Documentation**: LaTeX documentation with mathematical derivations and system flowcharts
**Open Source**: MIT licensed for unrestricted use and modification

## Technical Specifications

### Dependencies

- Python 3.8 or higher
- PyTorch 1.9 or higher
- NumPy for numerical operations
- pytest for testing framework

Install dependencies using:
```bash
pip install -r requirements.txt
```

### Installation

Clone the repository:
```bash
git clone https://github.com/<your-username>/awesome-neural-network.git
cd awesome-neural-network
```

Install required packages:
```bash
pip install torch numpy pytest
```

For documentation compilation (optional):
Requires TeX Live distribution for LaTeX compilation:
```bash
pdflatex docs/neural_network_documentation.tex
```

## Usage

### Basic Implementation

The following example demonstrates network initialization, training, and evaluation with synthetic data:

```python
import torch
from neural_network import NeuralNetwork, Trainer, generate_synthetic_data

# Generate synthetic dataset
x, y = generate_synthetic_data(n_samples=1000, n_features=10)

# Initialize network architecture
model = NeuralNetwork(
    layer_sizes=[10, 64, 32, 1], 
    activation='relu', 
    dropout_rate=0.3
)

# Configure trainer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = Trainer(model, learning_rate=0.001, device=device)

# Execute training
trainer.train(x, y, epochs=100, batch_size=32)

# Evaluate performance
test_loss = trainer.evaluate(x, y)
print(f"Final test loss: {test_loss:.4f}")
```

### Default Example Execution

Run the included example with default parameters:
```bash
python neural_network.py
```

This executes training on a synthetic dataset with 1000 samples and 10 features, using a [10, 64, 32, 1] network architecture for 100 epochs. Expected output includes device information, network initialization details, and training progress:

```
2025-07-16 19:57:00,000 - INFO - Using device: cpu
2025-07-16 19:57:00,001 - INFO - Initialized neural network with 3 layers: [10, 64, 32, 1]
2025-07-16 19:57:00,002 - INFO - Generated synthetic dataset with 1000 samples and 10 features
2025-07-16 19:57:00,003 - INFO - Trainer initialized with learning rate 0.001 on cpu
2025-07-16 19:57:00,100 - INFO - Epoch 10/100, Loss: 0.1234
...
2025-07-16 19:57:01,200 - INFO - Epoch 100/100, Loss: 0.0123
2025-07-16 19:57:01,201 - INFO - Final test loss: 0.0115
```

The low final test loss (0.0115) indicates successful convergence and accurate prediction capabilities.

## Configuration Options

### Network Architecture

**Layer Configuration**: Specify layer sizes as a list where the first element represents input features and the last represents output dimensions. Example: [20, 128, 64, 2] for 20 input features, two hidden layers with 128 and 64 neurons, and 2 output neurons.

**Activation Functions**:
- `relu`: Rectified Linear Unit - recommended for general use and faster convergence
- `tanh`: Hyperbolic tangent - suitable for normalized input data
- `sigmoid`: Sigmoid function - appropriate for binary classification outputs

**Dropout Regularization**: Configure dropout rate (0.0 to 1.0) to prevent overfitting. Higher values (0.4-0.5) provide stronger regularization for complex datasets.

### Training Parameters

**Learning Rate**: Controls optimization step size
- 0.001: Default balanced setting
- 0.01: Faster convergence for stable datasets
- 0.0001: Conservative approach for sensitive optimization

**Batch Size**: Number of samples processed per training iteration
- 16: Suitable for noisy data or limited memory
- 32: Default balanced setting
- 64: Faster training for large datasets

**Epochs**: Number of complete dataset iterations. Adjust based on convergence monitoring and validation performance.

### Custom Dataset Integration

For external datasets, ensure proper tensor formatting:

```python
import pandas as pd
import torch

# Load external data
data = pd.read_csv('dataset.csv')
x = torch.tensor(data[['feature1', 'feature2', 'feature3']].values, dtype=torch.float32)
y = torch.tensor(data['target'].values, dtype=torch.float32).reshape(-1, 1)

# Verify dimensions match network configuration
print(f"Input shape: {x.shape}, Output shape: {y.shape}")
```

Ensure network input size matches feature count and output size corresponds to target dimensions.

## Classification Implementation

For classification tasks, modify the network output layer to match the number of classes and adjust the loss function:

```python
# Multi-class setup (3 classes example)
model = NeuralNetwork(layer_sizes=[10, 64, 32, 3], activation='relu', dropout_rate=0.3)

# Update trainer loss function for classification
# Modify in Trainer class initialization:
# self.criterion = nn.CrossEntropyLoss()
```

Target labels should be class indices (0, 1, 2, etc.) rather than one-hot encoded vectors.

## Testing

Execute the test suite to verify implementation integrity:
```bash
pytest tests/
```

The test suite validates:
- Network initialization and parameter configuration
- Activation function implementations
- Training loop execution without errors
- Device allocation and tensor operations

## Project Structure

```
awesome-neural-network/
├── neural_network.py           # Core network implementation
├── README.md                   # Project documentation
├── requirements.txt            # Dependency specifications
├── tests/
│   └── test_neural_network.py  # Unit testing suite
├── docs/
│   └── neural_network_documentation.tex  # Technical documentation with mathematical derivations
└── .github/
    └── workflows/
        └── ci.yml              # Continuous integration configuration
```

## Documentation

The `docs/neural_network_documentation.tex` file provides comprehensive technical documentation including:

**Mathematical Foundations**: Complete derivations of forward propagation, backpropagation, and optimization algorithms
**System Architecture**: Detailed flowcharts illustrating data flow and training procedures
**Implementation Analysis**: Explanation of design decisions and algorithmic choices
**Performance Evaluation**: Analysis of synthetic dataset results and convergence characteristics
**Extension Guidelines**: Recommendations for advanced features and real-world applications

Generate PDF documentation:
```bash
pdflatex docs/neural_network_documentation.tex
```

## Performance Optimization

### GPU Acceleration

The implementation automatically detects and utilizes CUDA-compatible GPUs when available:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = Trainer(model, learning_rate=0.001, device=device)
```

GPU acceleration significantly improves training performance for large datasets and complex architectures.

### Hyperparameter Optimization

**Learning Rate Scheduling**: Consider implementing learning rate decay for improved convergence
**Batch Size Scaling**: Larger batch sizes improve GPU utilization but may require learning rate adjustments
**Regularization Tuning**: Balance dropout rate with network capacity to optimize generalization
**Architecture Search**: Experiment with layer widths and depths based on dataset complexity

## Contributing

Development contributions are welcome through the standard GitHub workflow:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement-name`)
3. Implement changes with appropriate testing
4. Commit modifications (`git commit -m "Description of changes"`)
5. Push to branch (`git push origin feature/enhancement-name`)
6. Submit pull request with detailed description

Bug reports and feature requests should be submitted through the GitHub Issues interface. Follow established coding standards and include comprehensive test coverage for new functionality.

## License

This project is released under the MIT License, permitting unrestricted use, modification, and distribution. See the LICENSE file for complete terms and conditions.

## Acknowledgments

This implementation builds upon the PyTorch framework and incorporates established neural network principles from the machine learning research community. Initial development by Ethan Ragbir with ongoing contributions from the open-source community.

## Support

For technical questions, implementation issues, or collaboration opportunities, please use the GitHub Issues system or contribute to project discussions. Community engagement and knowledge sharing are encouraged to advance the project's development and applications.
