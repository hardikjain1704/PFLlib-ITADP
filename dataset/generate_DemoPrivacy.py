"""
Generate a small demo dataset with features: image, label, phone_number.

This dataset is used to test the Purpose Limitation Validator.
The purpose validator should reject 'phone_number' for image_classification.
"""

import numpy as np
import os
import json

def generate_demo_dataset(output_dir=None):
    """Create a small 5-client MNIST-like dataset with an extra 'phone_number' field."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'DemoPrivacy')
    
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    np.random.seed(42)
    num_clients = 5
    num_classes = 10
    samples_per_client_train = 100
    samples_per_client_test = 20

    statistic = []

    for client_id in range(num_clients):
        # Generate random image-like data (1x28x28 flattened)
        x_train = np.random.randn(samples_per_client_train, 1, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, num_classes, samples_per_client_train).astype(np.int64)

        x_test = np.random.randn(samples_per_client_test, 1, 28, 28).astype(np.float32)
        y_test = np.random.randint(0, num_classes, samples_per_client_test).astype(np.int64)

        # Add a fake 'phone_number' feature to demonstrate purpose violation
        phone_numbers_train = [f"+1-555-{np.random.randint(1000,9999)}" for _ in range(samples_per_client_train)]
        phone_numbers_test = [f"+1-555-{np.random.randint(1000,9999)}" for _ in range(samples_per_client_test)]

        # Save in PFLlib format
        train_data = {'x': x_train, 'y': y_train}
        test_data = {'x': x_test, 'y': y_test}

        np.savez(os.path.join(train_dir, f'{client_id}.npz'), data=train_data)
        np.savez(os.path.join(test_dir, f'{client_id}.npz'), data=test_data)

        # Also save phone numbers separately (to demonstrate the violation)
        with open(os.path.join(train_dir, f'{client_id}_phone_numbers.json'), 'w') as f:
            json.dump(phone_numbers_train, f)

        # Compute statistics
        client_stat = []
        for label in np.unique(y_train):
            client_stat.append((int(label), int(np.sum(y_train == label))))
        statistic.append(client_stat)

        print(f"Client {client_id}: {samples_per_client_train} train, {samples_per_client_test} test samples")

    # Save config
    config = {
        "num_clients": num_clients,
        "num_classes": num_classes,
        "dataset_features": ["image", "label", "phone_number"],
        "note": "This dataset contains a 'phone_number' feature that violates purpose limitation for image_classification.",
        "Size of samples for labels in clients": statistic,
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nDemo dataset generated at: {output_dir}")
    print(f"Features: {config['dataset_features']}")
    print("NOTE: 'phone_number' should trigger a purpose violation for image_classification.")
    return output_dir


if __name__ == "__main__":
    generate_demo_dataset()
