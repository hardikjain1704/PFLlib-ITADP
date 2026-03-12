# PFLlib Federated Learning Demo – Two-Laptop Setup

## Architecture

```
┌─────────────────────────────────────────────┐
│  Laptop A (Server + Client 0)               │
│                                             │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ run_server   │  │ run_client --id 0    │  │
│  │ (port 9000)  │  │ trains on shard 0    │  │
│  │ FedAvg agg.  │  │ uploads weights      │  │
│  └──────┬───────┘  └──────────────────────┘  │
│         │                                    │
│  ┌──────┴───────┐  ┌──────────────────────┐  │
│  │ backend_api   │  │ frontend_dashboard   │  │
│  │ (port 8000)   │  │ (port 5173)          │  │
│  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────┘
         ▲
         │  HTTP (model weights only)
         ▼
┌─────────────────────────────────────────────┐
│  Laptop B (Client 1 only)                   │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │ run_client --client-id 1             │   │
│  │ --server-url http://<LaptopA-IP>:9000│   │
│  │ trains on shard 1, uploads weights   │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

## Prerequisites

```bash
pip install -r requirements.txt
cd frontend_dashboard && npm install
```

## Quick Start (Single Machine)

**Terminal 1 – Aggregation Server:**
```bash
python run_server.py
```

**Terminal 2 – Client 0 (Laptop A):**
```bash
python run_client.py --client-id 0
```

**Terminal 3 – Client 1 (Laptop B, or same machine):**
```bash
python run_client.py --client-id 1
```

**Terminal 4 – Backend API:**
```bash
cd backend_api && uvicorn main:app --reload --port 8000
```

**Terminal 5 – Frontend Dashboard:**
```bash
cd frontend_dashboard && npm run dev
```

Then open http://localhost:5173 in your browser.

## Two-Laptop Setup

### Laptop A (find your IP, e.g., `192.168.1.100`)

```bash
# Terminal 1: Server
python run_server.py --host 0.0.0.0 --port 9000

# Terminal 2: Client 0
python run_client.py --client-id 0 --server-url http://127.0.0.1:9000

# Terminal 3: Backend API
cd backend_api && uvicorn main:app --reload --port 8000

# Terminal 4: Dashboard
cd frontend_dashboard && npm run dev
```

### Laptop B

```bash
python run_client.py --client-id 1 --server-url http://192.168.1.100:9000
```

## Workflow

1. **Submit Consent** via the dashboard Consent Manager page
   - Both clients must consent with purpose `image_classification`
2. **Start Training** via the Training Control page
   - Click "Start Federated Training"
   - Both clients will automatically begin training
3. **Monitor Progress** in real-time
   - Training Control shows round history and server logs
   - Data Transparency shows per-client contribution charts

## Consent System

Each client must register consent with:
```json
{
  "client_id": 0,
  "consent": true,
  "purpose": "image_classification"
}
```

If a client consents with a mismatched purpose (e.g., `text_classification`), their weight uploads are blocked.

## Dataset

- **PneumoniaMNIST** (binary classification: Normal vs Pneumonia)
- 28×28 grayscale chest X-ray images
- Auto-downloaded via the `medmnist` package
- Split into 2 deterministic shards for federated training

## Model

- **PneumoniaCNN**: 2-layer CNN (Conv→Pool→Conv→Pool→FC→FC)
- ~420K parameters
- Lightweight enough for CPU training

## Logs

All logs are in `logs/`:
- `consent_records.json` – consent with purpose
- `transparency_log.json` – per-round participation
- `training_log.json` – round metrics (loss, accuracy, contributions)
- `purpose_violations.json` – blocked purpose attempts
