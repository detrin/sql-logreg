# SQL LogReg Refactoring Design

**Date:** 2026-03-01
**Status:** Approved
**Goal:** Refactor notebook-based prototypes into production-grade package with benchmarking capabilities

## Overview

Transform the current notebook-based implementations into a minimal, sleek OOP package that enables fair comparison of logistic regression optimizers: GradientDescent (fixed), L-BFGS, MADlib, and raw SQL.

## Architecture

**Pattern:** Strategy pattern with dependency injection

**Core Flow:**
1. CLI parses command and options
2. Data pipeline loads and preprocesses wine dataset
3. Optimizers instantiated (DB engine injected for SQL-based ones)
4. Benchmark runner executes optimizers sequentially
5. Results collected as structured objects
6. Output generated (JSON, summary table)

**Design Principles:**
- No global state
- Explicit dependencies
- Strategy pattern for optimizers (each owns training loop)
- Composable data preprocessors
- Structured result objects (not dicts)

## Package Structure

```
src/sqllogreg/
├── optimizers/
│   ├── base.py          # BaseOptimizer ABC
│   ├── gradient.py      # GradientDescent (fixed)
│   ├── lbfgs.py         # L-BFGS wrapper
│   ├── madlib.py        # MADlib wrapper
│   └── sql.py           # Raw SQL (reference)
├── data/
│   ├── loader.py        # CSV loader with target creation
│   ├── scaler.py        # StandardScaler wrapper
│   ├── sampler.py       # SMOTE oversampling
│   └── splitter.py      # Train/test splitter
├── metrics/
│   ├── evaluator.py     # AUC, F1, accuracy calculator
│   └── result.py        # TrainResult, BenchmarkResult dataclasses
├── benchmark/
│   └── runner.py        # Benchmark orchestrator
└── cli.py               # Click-based CLI
```

## Component Designs

### BaseOptimizer

```python
class BaseOptimizer(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainResult:
        pass

    def predict(self, X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
        # Shared sigmoid prediction
        pass
```

All optimizers implement this interface. Each owns its training loop and convergence logic.

### Optimizers

**GradientDescent:**
- Fixes bugs: removes 2x gradient factor, implements L2 regularization
- Adds learning rate decay: `lr = lr_init / (1 + decay * epoch)`
- Convergence check: gradient norm threshold
- Returns TrainResult with loss history, timing, iterations

**LBFGS:**
- Wraps `scipy.optimize.minimize(method='L-BFGS-B')`
- Converts to optimization problem with objective and gradient
- Returns TrainResult with scipy convergence info

**MADlib:**
- Constructor takes SQLAlchemy engine (dependency injection)
- Executes `madlib.logregr_train()` on temp postgres table
- Extracts coefficients and metadata
- Returns TrainResult

**SQLOptimizer:**
- Constructor takes SQLAlchemy engine
- Migrates notebook SQL logic into class
- Kept as reference baseline
- Returns TrainResult

### Data Pipeline

**DataLoader:**
- Loads CSV, creates binary target (quality >= 6.5)
- Returns X (features), y (target)

**Scaler:**
- Wraps StandardScaler
- Stores fit params for test transform

**Sampler:**
- SMOTE oversampling via imbalanced-learn's RandomOverSampler
- Balances minority class

**Splitter:**
- Returns (X_train, X_test, y_train, y_test)
- Default 80/20 split

**Usage:**
```python
X, y = DataLoader().load("data/winequality-red.csv")
X = Scaler().fit_transform(X)
X, y = Sampler().resample(X, y)
X_train, X_test, y_train, y_test = Splitter().split(X, y)
```

### Result Objects

**TrainResult (dataclass):**
```python
@dataclass
class TrainResult:
    optimizer_name: str
    weights: np.ndarray
    bias: float
    train_time: float
    iterations: int
    final_loss: float
    train_auc: float
    train_f1: float
    test_auc: float
    test_f1: float
    convergence_info: dict
```

**BenchmarkResult (dataclass):**
```python
@dataclass
class BenchmarkResult:
    results: list[TrainResult]

    def to_dataframe(self) -> pd.DataFrame:
        pass

    def to_json(self, path: str):
        pass

    def summary(self) -> str:
        pass
```

**Evaluator:**
```python
class Evaluator:
    def calculate_metrics(self, y_true, y_pred, y_prob) -> dict:
        # Returns {"auc": ..., "f1": ..., "accuracy": ...}
        pass
```

### Benchmark Runner

```python
class BenchmarkRunner:
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator

    def run(self, optimizers: list[BaseOptimizer],
            X_train, y_train, X_test, y_test) -> BenchmarkResult:
        # For each optimizer:
        #   1. Time .fit() call
        #   2. Get predictions
        #   3. Calculate metrics
        #   4. Build TrainResult
        # Return BenchmarkResult
        pass
```

**Behaviors:**
- Sequential execution (fair timing)
- Exception handling per optimizer
- Progress logging
- Consistent evaluation via Evaluator

### CLI Interface

**Commands:**
```bash
sqllogreg train --optimizer=lbfgs --data=data/winequality-red.csv
sqllogreg benchmark --optimizers=gd --optimizers=lbfgs --optimizers=madlib
sqllogreg evaluate benchmark_results.json
```

**Implementation:**
- Click-based with group and subcommands
- DATABASE_URL from environment variable
- Data/results mounted as volumes in Docker
- Output: JSON files + stdout summaries

## Docker Setup

**docker-compose.yml:**
```yaml
services:
  postgres:
    image: postgres:14-alpine
    environment:
      POSTGRES_DB: sqllogreg
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_INITDB_ARGS: "-c shared_buffers=256MB -c max_connections=200"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-madlib.sh:/docker-entrypoint-initdb.d/init-madlib.sh

  app:
    build: .
    depends_on:
      - postgres
    environment:
      DATABASE_URL: postgresql://postgres:postgres@postgres:5432/sqllogreg
    volumes:
      - ./data:/app/data
      - ./results:/app/results
    command: benchmark --optimizers=gd --optimizers=lbfgs --optimizers=madlib
```

**PostgreSQL Config:**
- Connection pooling: max_connections=200
- Memory: shared_buffers=256MB, work_mem=16MB
- MADlib extension via init script

**Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install -e .
ENTRYPOINT ["sqllogreg"]
CMD ["--help"]
```

## Benchmark Metrics

Each optimizer compared on:
- Training time (seconds)
- Convergence iterations
- Final loss
- Test AUC
- Test F1
- Memory usage
- Lines of code

## Implementation Notes

**ML Engineering Style:**
- Zero comments/docstrings
- DRY principles
- Minimal sleek code
- Virtual environment activation before Python execution

**Baseline Comparison:**
- Python GradientDescent (current buggy) as baseline
- Fixed GradientDescent shows impact of corrections
- L-BFGS demonstrates second-order optimization gains
- MADlib shows production PostgreSQL ML performance
- SQL shows educational reference (inefficient)

## Success Criteria

1. Package structure replaces notebooks entirely
2. All optimizers produce valid TrainResult
3. Benchmark runs compare 4+ optimizers fairly
4. Docker command override works for all CLI modes
5. PostgreSQL runs production config with MADlib
6. Results serializable to JSON for analysis
7. Code adheres to ML engineering style (no comments, DRY)
