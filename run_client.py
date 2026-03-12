#!/usr/bin/env python3
"""
run_client.py – Launch script for a federated client.

Laptop A:  python run_client.py --client-id 0
Laptop B:  python run_client.py --client-id 1 --server-url http://<server-ip>:9000

Usage:
    python run_client.py --client-id <0|1> [--server-url http://127.0.0.1:9000] [--device cpu]
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "system"))

from system.federated.client import run_client


def main():
    parser = argparse.ArgumentParser(description="PFLlib Federated Client")
    parser.add_argument("--client-id", type=int, required=True,
                        help="Client ID: 0 for Laptop A, 1 for Laptop B")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:9000",
                        help="URL of the aggregation server")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device: cpu or cuda")
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
