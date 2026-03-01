# SQL LogReg Refactoring & Benchmarking

## Phase 1: Package Structure Refactoring

- [ ] Design composite pattern architecture for optimizers
- [ ] Create package structure (src/sqllogreg/)
- [ ] Implement base optimizer interface
- [ ] Implement GradientDescent optimizer (fixed version)
- [ ] Implement LBFGS optimizer
- [ ] Implement MADlib optimizer wrapper
- [ ] Create data loader abstraction
- [ ] Create model evaluator with metrics (AUC, F1, accuracy)
- [ ] Create benchmark runner with timing
- [ ] Remove notebooks, migrate logic to modules

## Phase 2: Docker & PostgreSQL Setup

- [ ] Create Dockerfile for application
- [ ] Create docker-compose.yml with postgres service
- [ ] Add production-grade postgres config (connection pooling, memory, workers)
- [ ] Add MADlib extension to postgres container
- [ ] Add mode parameter to docker-compose (train/benchmark/evaluate)
- [ ] Create entrypoint script handling mode routing
- [ ] Add environment variables for config
- [ ] Test postgres connection and MADlib installation

## Phase 3: Core Implementation

- [ ] Fix gradient computation bug (remove 2x factor)
- [ ] Implement actual regularization in GradientDescent
- [ ] Add learning rate decay
- [ ] Implement mini-batch support
- [ ] Add convergence checking (gradient norm)
- [ ] Implement L-BFGS using scipy.optimize
- [ ] Implement MADlib training wrapper
- [ ] Add result serialization (JSON/CSV)

## Phase 4: Benchmarking Infrastructure

- [ ] Create benchmark config (iterations, datasets, metrics)
- [ ] Implement timing decorator for each optimizer
- [ ] Create results aggregator
- [ ] Add memory profiling
- [ ] Generate comparison reports
- [ ] Add CLI for running specific benchmarks
- [ ] Test all optimizers produce valid results

## Phase 5: Documentation & Usage

- [ ] Create README with setup instructions
- [ ] Document docker-compose modes
- [ ] Add benchmark running instructions
- [ ] Include interpretation of results
- [ ] Add troubleshooting section

## Benchmark Comparisons

Target metrics for each optimizer:
- Training time (seconds)
- Convergence iterations
- Final loss
- Test AUC/F1
- Memory usage
- Lines of code

Optimizers to compare:
1. Baseline: Python GradientDescent (sklearn-style)
2. Fixed: Python GradientDescent (corrected implementation)
3. L-BFGS: scipy.optimize
4. MADlib: PostgreSQL extension
5. SQL: Raw SQL implementation (reference only)
