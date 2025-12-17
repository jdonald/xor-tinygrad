#!/usr/bin/env python3
"""
XOR Neural Network - Train a network to recognize XOR function using tinygrad.
"""

import argparse
import json
import time
import random
from tinygrad import Tensor, Device
from tinygrad.nn import Linear
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict


class XORNet:
    """Simple 2-layer neural network for XOR function."""

    def __init__(self, hidden_size=4):
        self.hidden = Linear(2, hidden_size)
        self.output = Linear(hidden_size, 1)

    def __call__(self, x):
        x = self.hidden(x).sigmoid()
        x = self.output(x).sigmoid()
        return x

    def parameters(self):
        """Get all trainable parameters."""
        return [self.hidden.weight, self.hidden.bias, self.output.weight, self.output.bias]


def xor_label(a: float, b: float) -> float:
    """Compute XOR label for two float inputs using 0.5 threshold."""
    a_bool = a >= 0.5
    b_bool = b >= 0.5
    return 1.0 if a_bool != b_bool else 0.0


def generate_data(num_samples: int, seed: int) -> list[dict]:
    """Generate random training/test data for XOR function."""
    random.seed(seed)
    data = []
    for _ in range(num_samples):
        a = random.random()
        b = random.random()
        label = xor_label(a, b)
        data.append({"inputs": [a, b], "label": label})
    return data


def save_data(data: list[dict], filepath: str):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} samples to {filepath}")


def load_data(filepath: str) -> list[dict]:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def data_to_tensors(data: list[dict], device: str) -> tuple[Tensor, Tensor]:
    """Convert data list to tinygrad tensors."""
    inputs = Tensor([[d["inputs"][0], d["inputs"][1]] for d in data], device=device)
    labels = Tensor([[d["label"]] for d in data], device=device)
    return inputs, labels


def train(model: XORNet, data_path: str, weights_path: str, epochs: int = 1000, lr: float = 1.0, device: str = None):
    """Train the model and save weights."""
    if device is None:
        device = "GPU" if Device.DEFAULT in ["GPU", "CUDA", "METAL"] else "CPU"

    data = load_data(data_path)
    inputs, labels = data_to_tensors(data, device)

    optimizer = SGD(model.parameters(), lr=lr)

    # Enable training mode
    Tensor.training = True

    print(f"Training on {device} with {len(data)} samples for {epochs} epochs...")

    for epoch in range(epochs):
        outputs = model(inputs)

        # Binary Cross Entropy Loss
        loss = -(labels * outputs.log() + (1 - labels) * (1 - outputs).log()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy():.6f}")

    state_dict = get_state_dict(model)
    safe_save(state_dict, weights_path)
    print(f"Saved weights to {weights_path}")


def test(model: XORNet, data_path: str, weights_path: str):
    """Test the model and report error rate with GPU/CPU benchmarks."""
    # Disable training mode
    Tensor.training = False

    data = load_data(data_path)

    # Test on GPU first (if available)
    gpu_available = Device.DEFAULT in ["GPU", "CUDA", "METAL"]
    if gpu_available:
        device = Device.DEFAULT
        model_gpu = XORNet()
        state_dict = safe_load(weights_path)

        inputs, labels = data_to_tensors(data, device)

        # Warmup
        for _ in range(10):
            _ = model_gpu(inputs).realize()

        Device[device].synchronize()
        start_time = time.perf_counter()
        outputs = model_gpu(inputs).realize()
        Device[device].synchronize()
        gpu_time = time.perf_counter() - start_time

        predictions = (outputs.numpy() >= 0.5).astype(float)
        correct = (predictions == labels.numpy()).sum()
        error_rate = 1.0 - (correct / len(data))

        print(f"\n=== GPU Benchmark ===")
        print(f"Device: {Device.DEFAULT}")
        print(f"Inference time: {gpu_time * 1000:.4f} ms")
        print(f"Samples: {len(data)}")
        print(f"Correct: {int(correct)}/{len(data)}")
        print(f"Error rate: {error_rate * 100:.2f}%")
    else:
        print("\nGPU not available, skipping GPU benchmark.")

    # Test on CPU
    device = "CPU"
    model_cpu = XORNet()
    state_dict = safe_load(weights_path)
    # Manually assign weights to ensure correct device placement
    model_cpu.hidden.weight = state_dict['hidden.weight'].to(device).realize()
    model_cpu.hidden.bias = state_dict['hidden.bias'].to(device).realize()
    model_cpu.output.weight = state_dict['output.weight'].to(device).realize()
    model_cpu.output.bias = state_dict['output.bias'].to(device).realize()

    inputs, labels = data_to_tensors(data, device)

    # Warmup
    for _ in range(10):
        _ = model_cpu(inputs).realize()

    start_time = time.perf_counter()
    outputs = model_cpu(inputs).realize()
    cpu_time = time.perf_counter() - start_time

    predictions = (outputs.numpy() >= 0.5).astype(float)
    correct = (predictions == labels.numpy()).sum()
    error_rate = 1.0 - (correct / len(data))

    print(f"\n=== CPU Benchmark ===")
    print(f"Inference time: {cpu_time * 1000:.4f} ms")
    print(f"Samples: {len(data)}")
    print(f"Correct: {int(correct)}/{len(data)}")
    print(f"Error rate: {error_rate * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="XOR Neural Network - Train and test a network to recognize XOR function")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate data command
    gen_parser = subparsers.add_parser("generate", help="Generate random training/test data")
    gen_parser.add_argument("--seed", type=int, required=True, help="Random seed for data generation")
    gen_parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate (default: 1000)")
    gen_parser.add_argument("--output", type=str, default="data.json", help="Output file path (default: data.json)")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the network and save weights")
    train_parser.add_argument("--data", type=str, default="data.json", help="Training data file (default: data.json)")
    train_parser.add_argument("--weights", type=str, default="weights.safetensors", help="Output weights file (default: weights.safetensors)")
    train_parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs (default: 1000)")
    train_parser.add_argument("--lr", type=float, default=1.0, help="Learning rate (default: 1.0)")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test the network and report error rate")
    test_parser.add_argument("--data", type=str, default="data.json", help="Test data file (default: data.json)")
    test_parser.add_argument("--weights", type=str, default="weights.safetensors", help="Weights file to load (default: weights.safetensors)")

    args = parser.parse_args()

    if args.command == "generate":
        data = generate_data(args.samples, args.seed)
        save_data(data, args.output)
    elif args.command == "train":
        model = XORNet()
        train(model, args.data, args.weights, args.epochs, args.lr)
    elif args.command == "test":
        model = XORNet()
        test(model, args.data, args.weights)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
