# KATO Tutorials

Interactive Jupyter notebook tutorials for [KATO](https://github.com/sevakavakians/kato) — a deterministic machine learning engine offering complete transparency, observability, traceability and real-time editability.
KATO adheres to Excite AI principles for Verification & Validation.
https://medium.com/@sevakavakians/what-is-excite-ai-712afd372af4

## Contents

- **`kato_tutorial.ipynb`** — Comprehensive tutorial covering KATO's core capabilities: sequence learning, pattern recognition, prediction, classification, and regression across multiple real-world datasets (telecom churn, handwritten character recognition, California housing, auto MPG, etc.).
- **`churn_analysis.ipynb`** — Binary churn classification using KATO's prediction ensemble with a 441-combination parameter sweep across ranking algorithms, recall thresholds, ensemble sizes, and weighting metrics. Evaluates using the Predictor Operating Characteristic (POC) chart.
- **`churn_analysis_emotives.ipynb`** — Extends the churn analysis by enabling KATO's affinity-weighted pattern matching with symbol emotive bias. Tests whether frequency-normalized affinity weights improve prediction accuracy and make ensemble predictions viable.
- **`poc_chart.py`** — Utility module for Predictor Operating Characteristic (POC) chart generation and predictor metrics computation, used by the churn analysis notebooks.
- **`data/`** — Datasets used by the tutorials (auto-mpg.csv, churn.csv).

## Prerequisites

- A running KATO server (see the [KATO repo](https://github.com/sevakavakians/kato) for setup instructions)
- Python 3.10+
- Jupyter Notebook or JupyterLab

## Getting Started

1. Clone this repo and start the KATO server per the [KATO README](https://github.com/sevakavakians/kato#readme).
2. Open any notebook in Jupyter and run cells sequentially from the top.
3. The churn analysis notebooks require the KATO server to be accessible at `http://kato:8000` (Docker) or `http://localhost:8000` (host).
