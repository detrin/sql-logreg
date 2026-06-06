# SQL Logistic Regression Benchmark

Comprehensive benchmark comparing Logistic Regression optimizers: Python, SQL, and MADlib implementations.

## 🎯 Overview

This project implements and benchmarks Logistic Regression using **5 different optimization methods**:

1. **GradientDescent (Python)** - First-order gradient descent with L2 regularization
2. **LBFGS (Python)** - Second-order quasi-Newton using scipy.optimize
3. **GradientSQL** - Pure SQL gradient descent using PostgreSQL
4. **LBFGS_SQL** - SQL-based quasi-Newton using PostgreSQL arrays
5. **MADlib** - Production PostgreSQL ML (requires separate installation)

**Dataset**: UCI Wine Quality (binary classification: good wine ≥ 6 vs bad wine < 6)

## 🏆 Benchmark Results

| Optimizer | Time (s) | Iterations | Test AUC | Test F1 | Speedup | Status |
|-----------|----------|------------|----------|---------|---------|--------|
| **Python LBFGS** | **0.002** ⚡ | 6 | 0.876 | 0.797 | **6,017x** | ✅ Tested |
| **MADlib (est.)** | 0.010-0.050 🔥 | 10-20 | ~0.87 | ~0.80 | 200-1000x | 📋 [See docs](MADLIB_COMPARISON.md) |
| **Python GD** | **0.057** ✅ | 1000 | 0.875 | 0.813 | **181x** | ✅ Tested |
| **LBFGS_SQL** | **1.434** 📊 | 100 | 0.868 | 0.813 | **7.2x** | ✅ Tested |
| **GradientSQL** | **10.348** 📚 | 1000 | 0.875 | 0.812 | 1.0x | ✅ Tested |

### 💡 Key Insights

- **Python L-BFGS is 6,017x faster** than SQL gradient descent
- **SQL L-BFGS is 7x faster** than SQL gradient descent (second-order wins even in SQL!)
- **All methods achieve similar quality** (AUC ~0.87-0.88)
- **SQL implementations prove ML algorithms CAN run in databases** (but slower)

**Full analysis**: See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for detailed comparison
**MADlib comparison**: See [MADLIB_COMPARISON.md](MADLIB_COMPARISON.md) for production SQL ML

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Run full benchmark with all 4 working optimizers
docker-compose up -d
docker-compose run app benchmark \
  --optimizers=gd \
  --optimizers=lbfgs \
  --optimizers=gradient_sql \
  --optimizers=lbfgs_sql \
  --output=/app/results/full_benchmark.json

# View results
cat results/full_benchmark.json
```

### Local Installation

```bash
# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Run individual optimizer
sqllogreg train --optimizer=lbfgs

# Run benchmark
sqllogreg benchmark --optimizers=gd --optimizers=lbfgs --output=results.json
```

## 📖 Usage Examples

### Train Single Optimizer

```bash
# Python optimizers (no database required)
docker-compose run app train --optimizer=gd
docker-compose run app train --optimizer=lbfgs

# SQL optimizers (requires PostgreSQL)
docker-compose run app train --optimizer=gradient_sql
docker-compose run app train --optimizer=lbfgs_sql
```

### Run Custom Benchmark

```bash
# Compare Python optimizers
docker-compose run app benchmark \
  --optimizers=gd \
  --optimizers=lbfgs \
  --output=/app/results/python_benchmark.json

# Compare SQL optimizers
docker-compose run app benchmark \
  --optimizers=gradient_sql \
  --optimizers=lbfgs_sql \
  --output=/app/results/sql_benchmark.json

# All optimizers
docker-compose run app benchmark \
  --optimizers=gd \
  --optimizers=lbfgs \
  --optimizers=gradient_sql \
  --optimizers=lbfgs_sql \
  --output=/app/results/full_benchmark.json
```

### Debugging PostgreSQL

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U postgres -d sqllogreg

# List tables created by SQL optimizers
docker-compose exec postgres psql -U postgres -d sqllogreg -c "\dt"

# Inspect SQL L-BFGS state
docker-compose exec postgres psql -U postgres -d sqllogreg -c \
  "SELECT id, bias, array_length(weights, 1) FROM lbfgs_sql_logreg_state LIMIT 5;"

# Check convergence history
docker-compose exec postgres psql -U postgres -d sqllogreg -c \
  "SELECT iteration, loss FROM gradient_sql_logreg_convergence ORDER BY iteration LIMIT 10;"
```

## 🏗️ Architecture

```
src/sqllogreg/
├── optimizers/
│   ├── base.py              # Abstract base optimizer
│   ├── gradient.py          # Python gradient descent
│   ├── lbfgs.py             # Python L-BFGS
│   ├── gradient_sql.py      # SQL gradient descent
│   ├── lbfgs_sql.py         # SQL L-BFGS
│   └── madlib.py            # MADlib wrapper
├── data/
│   ├── loader.py            # CSV data loading
│   ├── scaler.py            # Feature scaling
│   ├── sampler.py           # SMOTE resampling
│   └── splitter.py          # Train/test split
├── metrics/
│   ├── evaluator.py         # AUC, F1 metrics
│   └── result.py            # Result dataclasses
├── benchmark/
│   └── runner.py            # Benchmark orchestration
└── cli.py                   # Click CLI interface
```

**Design Patterns**:
- **Strategy Pattern**: Optimizer implementations
- **Dependency Injection**: Database engine for SQL optimizers
- **Structured Results**: Type-safe result objects

## 🧪 Running Tests

```bash
# Local
pytest tests/ -v

# Docker
docker-compose run app pytest tests/ -v
```

## 📊 Technical Details

### Python Gradient Descent

- **Algorithm**: Batch gradient descent with L2 regularization
- **Learning rate**: 0.01 with decay
- **Convergence**: Gradient norm < 1e-5
- **Iterations**: Max 1000

### Python L-BFGS

- **Algorithm**: Limited-memory BFGS (scipy.optimize.minimize)
- **Memory**: 10 vector pairs (default)
- **Convergence**: Extremely fast (6 iterations typical)
- **Method**: Second-order quasi-Newton

### SQL Gradient Descent

- **Algorithm**: Pure SQL gradient descent using nested subqueries
- **Storage**: Relational tables (one column per weight)
- **Update**: Individual weight updates via SQL aggregations
- **Iterations**: Max 1000

**Key SQL snippet**:
```sql
UPDATE coefficients
SET w_i = w_i - learning_rate * (
    SELECT AVG((sigmoid(logit) - y) * x_i)
    FROM data, coefficients
)
```

### SQL L-BFGS

- **Algorithm**: Simplified quasi-Newton using PostgreSQL arrays
- **Storage**: PostgreSQL FLOAT[] arrays for vectors
- **Update**: Array operations for vector math
- **Iterations**: Max 100 (converges faster)

**Key SQL snippet**:
```sql
UPDATE state
SET weights = ARRAY[
    weights[1] - alpha * gradient[1],
    weights[2] - alpha * gradient[2],
    ...
]
```

## 🎓 When to Use Each Optimizer

| Use Case | Recommended Optimizer | Reason |
|----------|----------------------|---------|
| **Production ML** | Python LBFGS | Fastest, proven, industry standard |
| **Data in PostgreSQL** | MADlib | No data movement, production SQL ML |
| **Learning algorithms** | GradientSQL or LBFGS_SQL | Transparent, educational |
| **Simple baseline** | Python GD | Easy to understand, good reference |
| **Research** | Any | Compare convergence properties |

## 📚 Documentation

- **[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)** - Full benchmark analysis and technical details
- **[MADLIB_COMPARISON.md](MADLIB_COMPARISON.md)** - MADlib installation, comparison, and when to use it
- **[docs/plans/](docs/plans/)** - Implementation plans and design documents

## 🛠️ Requirements

- **Python**: 3.11+
- **PostgreSQL**: 14+ (for SQL optimizers)
- **Docker**: Optional but recommended
- **MADlib**: Optional (see [MADLIB_COMPARISON.md](MADLIB_COMPARISON.md))

## 📦 Dependencies

```
scikit-learn, numpy, pandas, scipy
matplotlib, sqlalchemy, psycopg2-binary
imbalanced-learn, python-dotenv, click
```

## 🤝 Contributing

This is an educational/research project demonstrating:
- ML algorithm implementation in SQL
- Performance comparison: Python vs SQL
- Production ML libraries (MADlib)
- Software engineering best practices (Strategy pattern, DI, type safety)

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- UCI Machine Learning Repository (Wine Quality dataset)
- Apache MADlib project
- PostgreSQL community
