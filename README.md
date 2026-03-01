# sql-logreg

Logistic Regression optimizer benchmarking with PostgreSQL

## Overview

Compares logistic regression implementations:
- GradientDescent (fixed, production-ready)
- L-BFGS (scipy optimization)
- MADlib (PostgreSQL ML extension)
- SQL (educational reference)

Dataset: UCI Wine Quality (binary classification)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

### Local

```bash
sqllogreg train --optimizer=lbfgs
sqllogreg benchmark --optimizers=gd --optimizers=lbfgs
sqllogreg evaluate results.json
```

### Docker

```bash
docker-compose up
docker-compose run app train --optimizer=gd
docker-compose run app benchmark --optimizers=gd --optimizers=lbfgs
```

## Architecture

```
src/sqllogreg/
├── optimizers/    # Strategy pattern implementations
├── data/          # Composable preprocessing pipeline
├── metrics/       # Evaluation and results
├── benchmark/     # Orchestration
└── cli.py         # Click-based interface
```

## Requirements

- Python 3.11+
- PostgreSQL 14+ (for MADlib/SQL optimizers)
- Docker (optional)