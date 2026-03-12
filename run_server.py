#!/usr/bin/env python3
"""
run_server.py – Launch script for Laptop A.

Starts the federated aggregation server on port 9001.
Laptop A also acts as client 0 (run run_client.py --client-id 0 in a second terminal).

Usage:
    python run_server.py [--host 0.0.0.0] [--port 9001]
"""

import argparse
import os
import sys

# Ensure project root is on the path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "system"))


def main():
    parser = argparse.ArgumentParser(description="PFLlib Federated Aggregation Server")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind address (use 0.0.0.0 for network access)")
    parser.add_argument("--port", type=int, default=9001,
                        help="Server port (default: 9001)")
    args = parser.parse_args()

    print("═══════════════════════════════════════════════════")
    print("  PFLlib Federated Aggregation Server")
    print(f"  Listening on http://{args.host}:{args.port}")
    print("═══════════════════════════════════════════════════")
    print()
    print("INSTRUCTIONS:")
    print("  1. Start this server:       python run_server.py")
    print("  2. Start client 0 (Laptop A): python run_client.py --client-id 0")
    print("  3. Start client 1 (Laptop B): python run_client.py --client-id 1 --server-url http://<server-ip>:9001")
    print("  4. Open dashboard:          cd frontend_dashboard && npm run dev")
    print("  5. Start training from the dashboard or via:")
    print(f"     curl -X POST http://127.0.0.1:{args.port}/start-round")
    print()

    import uvicorn
    uvicorn.run(
        "system.federated.server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
