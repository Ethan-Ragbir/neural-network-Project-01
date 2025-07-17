import pytest
import torch
from neural_network import NeuralNetwork, Trainer, generate_synthetic_data

def test_neural_network_initialization():
    model = NeuralNetwork([10, 20, 1], activation='relu', dropout_rate=0.2)
    assert len(model.layers) == 2
    assert isinstance(model.layers[0], torch.nn.Linear)
    assert model.layers[0].in_features == 10
    assert model.layers[0].out_features == 20

def test_invalid_activation():
    with pytest.raises(ValueError):
        NeuralNetwork([10, 20, 1], activation='invalid')

def test_trainer():
    model = NeuralNetwork([10, 20, 1])
    trainer = Trainer(model, learning_rate=0.001)
    x, y = generate_synthetic_data(n_samples=100, n_features=10)
    losses = trainer.train(x, y, epochs=5, batch_size=10)
    assert len(losses) == 5
    assert all(isinstance(loss, float) for loss in losses)