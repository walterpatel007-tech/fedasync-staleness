# 🧠 Federated Asynchronous Learning (FedAsync & FedBuff)

This repository implements **FedAsync** (Asynchronous Federated Learning) and **FedBuff** (Buffered Asynchronous Federated Learning) using **PyTorch Lightning**.  
Both frameworks simulate heterogeneous client behavior and perform asynchronous updates to a central server.

---

## 📦 Project Structure

```
FEDASYNC-STALENESS/
│
├── FedAsync/
│   ├── client.py
│   ├── server.py
│   ├── run.py
│   └── config.yaml
│
├── FedBuff/
│   ├── client.py
│   ├── server.py
│   ├── run.py
│   └── config.yml
│
├── solution/
│   ├── __init__.py
│   ├── client.py
│   ├── pipeline.py
│   ├── server.py
│   ├── run.py
│   └── config.yaml
│
├── utils/
│   ├── helper.py
│   ├── model.py
│   └── partitioning.py
│
├── checkpoints/
├── logs/
├── results/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ 1. Setup Environment

### Create a Python virtual environment

```bash
python -m venv .venv
```

### Activate the environment

**Windows**
```bash
.venv\Scripts\activate.bat
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ 2. Running the Frameworks

### Run FedAsync
```bash
python -m FedAsync.run
```

### Run FedBuff
```bash
python -m FedBuff.run
```

### Run the Flower based solution
```bash
python -m solution.run
```

Both scripts automatically initialize a server and multiple clients according to your configuration.

The new ``solution`` package is fully object-oriented and consists of three
main modules:

| Module | Responsibility |
|--------|----------------|
| ``solution.client`` | Implements :class:`SolutionClient`, a Flower ``NumPyClient`` that encapsulates the PyTorch training and evaluation loops. |
| ``solution.server`` | Contains the :class:`SolutionStrategy` FedAvg subclass plus helper builders for the staleness-aware aggregation logic. |
| ``solution.pipeline`` | Offers the fully object-oriented :class:`SolutionPipeline` wrapper that prepares data, clients, and strategies programmatically. |
| ``solution.run`` | Provides the orchestration entry point that instantiates :class:`SolutionPipeline` for ``python -m solution.run``. |

All knobs live in ``solution/config.yaml`` and can be overridden via the
``FEDASYNC_SOLUTION_CONFIG`` environment variable.

---

## 📊 3. Outputs and Logs

| File | Description |
|------|--------------|
| `logs/FedAsync.csv` | Global model metrics (aggregations, losses, accuracies, time) |
| `logs/FedAsyncClientParticipation.csv` | Per-client participation details (ID, local metrics) |
| `checkpoints/` | Intermediate global model checkpoints |
| `results/FedAsyncModel.pt` | Final global model weights |

Only concise `[LOG] ...` lines are printed to console when evaluations are logged.

---

## 🧪 4. Updating `requirements.txt`

If you install or update dependencies during development, regenerate:

```bash
pip freeze > requirements.txt
```

---

## 🧠 5. Key Features

- **Asynchronous aggregation** — Clients update server immediately after local training.
- **Client heterogeneity simulation** — Random per-client delays to mimic real-world latency.
- **PyTorch Lightning** — Ensures reproducibility, checkpointing, and clean training.
- **Automatic logging** — Global and client-level logs stored in CSV format.
- **Config-driven** — All behavior customizable via `config.yaml`.

---

## ✅ Example Workflow

```bash
# Create and activate environment
python -m venv .venv
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Run FedAsync
python -m FedAsync.run
```

Check `logs/` for training progress and `results/FedAsyncModel.pt` for the saved model.
