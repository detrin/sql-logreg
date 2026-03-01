from setuptools import setup, find_packages

setup(
    name="sqllogreg",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "scikit-learn",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "sqlalchemy",
        "psycopg2",
        "imbalanced-learn",
        "python-dotenv",
        "click",
    ],
    entry_points={
        "console_scripts": [
            "sqllogreg=sqllogreg.cli:cli",
        ],
    },
)
