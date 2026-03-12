"""
Federated Client – runs on each laptop.

Workflow per round:
  1. Fetch global model from aggregation server
  2. Train locally on the assigned PneumoniaMNIST shard
  3. Upload updated weights back to server
  4. Wait for the next round

Usage:
    python client.py --client-id 0 --server-url http://<server-ip>:9000
"""

import argparse
import io
import os
import sys
import time

import requests
import torch
import torch.nn as nn
import torch.optim as optim

# ── Resolve project paths ─────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SYSTEM_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_SYSTEM_DIR)
sys.path.insert(0, _SYSTEM_DIR)

from models.pneumonia_cnn import PneumoniaCNN
from datasets.pneumonia_mnist_loader import load_pneumonia_shard


def fetch_global_model(server_url: str) -> dict:
    """Download the current global model weights from the server."""
    resp = requests.get(f"{server_url}/global-model", timeout=30)
    resp.raise_for_status()
    buf = io.BytesIO(resp.content)
    state_dict = torch.load(buf, map_location="cpu", weights_only=False)
    return state_dict


def fetch_round_config(server_url: str) -> dict:
    """Get current round configuration from the server."""
    resp = requests.get(f"{server_url}/round-config", timeout=10)
    resp.raise_for_status()
    return resp.json()


def train_local(
    model: PneumoniaCNN,
    train_loader,
    epochs: int = 2,
    lr: float = 0.001,
    device: str = "cpu",
) -> dict:
    """Train the model locally and return metrics.

    Returns:
        dict with keys: num_samples, avg_loss, accuracy
    """
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            epoch_correct += predicted.eq(labels).sum().item()
            epoch_samples += images.size(0)

        avg = epoch_loss / max(epoch_samples, 1)
        acc = epoch_correct / max(epoch_samples, 1)
        print(f"    Epoch {epoch + 1}/{epochs}: loss={avg:.4f}, acc={acc:.4f}")

        total_loss += epoch_loss
        total_correct += epoch_correct
        total_samples += epoch_samples

    # Overall metrics across all epochs
    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    # num_samples is per-epoch count (what matters for FedAvg weighting)
    per_epoch_samples = total_samples // max(epochs, 1)

    return {
        "num_samples": per_epoch_samples,
        "avg_loss": round(avg_loss, 6),
        "accuracy": round(accuracy, 6),
    }


def evaluate(model: PneumoniaCNN, test_loader, device: str = "cpu") -> dict:
    """Evaluate the model on the test set."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += images.size(0)

    return {
        "test_loss": round(total_loss / max(total_samples, 1), 6),
        "test_accuracy": round(total_correct / max(total_samples, 1), 6),
        "test_samples": total_samples,
    }


def upload_weights(
    server_url: str,
    client_id: int,
    model: PneumoniaCNN,
    num_samples: int,
    local_loss: float,
    local_accuracy: float,
):
    """Upload local model weights to the aggregation server."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)

    resp = requests.post(
        f"{server_url}/upload-weights",
        data={
            "client_id": client_id,
            "num_samples": num_samples,
            "local_loss": local_loss,
            "local_accuracy": local_accuracy,
        },
        files={"weights_file": ("weights.pt", buf, "application/octet-stream")},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def run_client(
    client_id: int,
    server_url: str,
    device: str = "cpu",
    poll_interval: float = 3.0,
):
    """Main client loop: poll server, train, upload, repeat."""
    print(f"═══════════════════════════════════════════════════")
    print(f"  Federated Client {client_id}")
    print(f"  Server: {server_url}")
    print(f"  Device: {device}")
    print(f"═══════════════════════════════════════════════════")

    # Load local dataset shard
    print(f"\n[Client {client_id}] Loading PneumoniaMNIST shard {client_id}...")
    train_loader, test_loader, info = load_pneumonia_shard(
        shard_id=client_id, num_shards=2, batch_size=32
    )
    print(f"[Client {client_id}] Shard info: {info}")

    last_round = 0

    while True:
        # Poll for round config
        try:
            config = fetch_round_config(server_url)
        except requests.exceptions.ConnectionError:
            print(f"[Client {client_id}] Server not reachable. Retrying in {poll_interval}s...")
            time.sleep(poll_interval)
            continue
        except Exception as e:
            print(f"[Client {client_id}] Error fetching config: {e}. Retrying...")
            time.sleep(poll_interval)
            continue

        status = config.get("status", "idle")
        current_round = config.get("current_round", 0)

        if status == "completed":
            print(f"\n[Client {client_id}] ✅ Training completed! Fetching final model...")
            # Fetch and evaluate final model
            try:
                global_weights = fetch_global_model(server_url)
                model = PneumoniaCNN(num_classes=2)
                model.load_state_dict(global_weights)
                test_metrics = evaluate(model, test_loader, device)
                print(f"[Client {client_id}] Final test metrics: {test_metrics}")
            except Exception as e:
                print(f"[Client {client_id}] Could not evaluate final model: {e}")
            break

        if status == "idle":
            time.sleep(poll_interval)
            continue

        if status != "waiting" or current_round <= last_round:
            time.sleep(poll_interval)
            continue

        # ── New round: train and upload ────────────────────────
        local_epochs = config.get("local_epochs", 2)
        lr = config.get("learning_rate", 0.001)

        print(f"\n{'─' * 50}")
        print(f"[Client {client_id}] Round {current_round}: fetching global model...")
        try:
            global_weights = fetch_global_model(server_url)
        except Exception as e:
            print(f"[Client {client_id}] Failed to fetch global model: {e}")
            time.sleep(poll_interval)
            continue

        model = PneumoniaCNN(num_classes=2)
        model.load_state_dict(global_weights)

        print(f"[Client {client_id}] Round {current_round}: training locally "
              f"(epochs={local_epochs}, lr={lr})...")

        metrics = train_local(
            model, train_loader, epochs=local_epochs, lr=lr, device=device
        )
        print(f"[Client {client_id}] Round {current_round}: "
              f"loss={metrics['avg_loss']:.4f}, acc={metrics['accuracy']:.4f}")

        # Test evaluation
        test_metrics = evaluate(model, test_loader, device)
        print(f"[Client {client_id}] Round {current_round}: "
              f"test_loss={test_metrics['test_loss']:.4f}, test_acc={test_metrics['test_accuracy']:.4f}")

        # Upload weights
        print(f"[Client {client_id}] Round {current_round}: uploading weights...")
        try:
            result = upload_weights(
                server_url,
                client_id,
                model,
                metrics["num_samples"],
                metrics["avg_loss"],
                metrics["accuracy"],
            )
            print(f"[Client {client_id}] Round {current_round}: upload {result.get('status', 'done')}")
        except requests.exceptions.HTTPError as e:
            print(f"[Client {client_id}] Round {current_round}: upload REJECTED – {e.response.text}")
        except Exception as e:
            print(f"[Client {client_id}] Round {current_round}: upload failed – {e}")

        last_round = current_round
        time.sleep(1)  # brief pause before polling again


def main():
    parser = argparse.ArgumentParser(description="PFLlib Federated Client")
    parser.add_argument("--client-id", type=int, required=True,
                        help="Client ID (0 for Laptop A, 1 for Laptop B)")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:9000",
                        help="URL of the aggregation server")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (cpu or cuda)")
    parser.add_argument("--poll-interval", type=float, default=3.0,
                        help="Seconds between server polls")
    args = parser.parse_args()

    run_client(
        client_id=args.client_id,
        server_url=args.server_url,
        device=args.device,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
