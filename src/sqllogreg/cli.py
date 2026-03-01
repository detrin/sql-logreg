import click
from sqlalchemy import create_engine
from sqllogreg.data.loader import DataLoader
from sqllogreg.data.scaler import Scaler
from sqllogreg.data.sampler import Sampler
from sqllogreg.data.splitter import Splitter
from sqllogreg.optimizers.gradient import GradientDescent
from sqllogreg.optimizers.lbfgs import LBFGS
from sqllogreg.optimizers.madlib import MADlib
from sqllogreg.optimizers.gradient_sql import GradientSQLOptimizer
from sqllogreg.metrics.evaluator import Evaluator
from sqllogreg.benchmark.runner import BenchmarkRunner

@click.group()
def cli():
    pass

@cli.command()
@click.option('--optimizer', type=click.Choice(['gd', 'lbfgs', 'madlib', 'gradient_sql', 'lbfgs_sql']), required=True)
@click.option('--data', default='data/winequality-red.csv')
@click.option('--db-url', envvar='DATABASE_URL')
def train(optimizer, data, db_url):
    X, y = DataLoader().load(data)
    X = Scaler().fit_transform(X)
    X, y = Sampler().resample(X, y)
    X_train, X_test, y_train, y_test = Splitter().split(X, y)

    if optimizer == 'gd':
        opt = GradientDescent()
    elif optimizer == 'lbfgs':
        opt = LBFGS()
    elif optimizer == 'madlib':
        engine = create_engine(db_url)
        opt = MADlib(engine)
    elif optimizer == 'gradient_sql':
        engine = create_engine(db_url)
        opt = GradientSQLOptimizer(engine)
    elif optimizer == 'lbfgs_sql':
        from sqllogreg.optimizers.lbfgs_sql import LBFGSSQLOptimizer
        engine = create_engine(db_url)
        opt = LBFGSSQLOptimizer(engine)

    runner = BenchmarkRunner(Evaluator())
    result = runner.run([opt], X_train, y_train, X_test, y_test)

    print(result.summary())

@cli.command()
@click.option('--optimizers', multiple=True, default=['gd', 'lbfgs'])
@click.option('--data', default='data/winequality-red.csv')
@click.option('--db-url', envvar='DATABASE_URL')
@click.option('--output', default='benchmark_results.json')
def benchmark(optimizers, data, db_url, output):
    X, y = DataLoader().load(data)
    X = Scaler().fit_transform(X)
    X, y = Sampler().resample(X, y)
    X_train, X_test, y_train, y_test = Splitter().split(X, y)

    opts = []
    for opt_name in optimizers:
        if opt_name == 'gd':
            opts.append(GradientDescent())
        elif opt_name == 'lbfgs':
            opts.append(LBFGS())
        elif opt_name == 'madlib':
            engine = create_engine(db_url)
            opts.append(MADlib(engine))
        elif opt_name == 'gradient_sql':
            engine = create_engine(db_url)
            opts.append(GradientSQLOptimizer(engine))
        elif opt_name == 'lbfgs_sql':
            from sqllogreg.optimizers.lbfgs_sql import LBFGSSQLOptimizer
            engine = create_engine(db_url)
            opts.append(LBFGSSQLOptimizer(engine))

    runner = BenchmarkRunner(Evaluator())
    result = runner.run(opts, X_train, y_train, X_test, y_test)

    result.to_json(output)
    print(result.summary())

@cli.command()
@click.argument('results_file')
def evaluate(results_file):
    import json
    with open(results_file) as f:
        data = json.load(f)

    print("\nBenchmark Results:")
    for r in data:
        print(f"\n{r['optimizer_name']}:")
        print(f"  Train Time: {r['train_time']:.2f}s")
        print(f"  Iterations: {r['iterations']}")
        print(f"  Test AUC: {r['test_auc']:.3f}")
        print(f"  Test F1: {r['test_f1']:.3f}")

if __name__ == '__main__':
    cli()
