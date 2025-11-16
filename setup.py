"""
QuantScale: Distributed Training Framework for Financial Time Series

Setup configuration for package installation.

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

from setuptools import find_packages, setup

# Read README for long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = (
        "Production-grade distributed training framework for financial time series"
    )

# Core dependencies
requirements = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "mlflow>=2.8.0",
    "ray[tune]>=2.8.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "tensorboard>=2.14.0",
]

# Development dependencies
requirements_dev = [
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.5.0",
    "isort>=5.12.0",
    "pre-commit>=3.4.0",
    "bandit>=1.7.5",
]

# Test dependencies
requirements_test = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-xdist>=3.3.0",
    "pytest-mock>=3.11.0",
]

setup(
    name="quantscale",
    version="0.1.0",
    author="Vithushan Jeyapahan",
    author_email="finance@vijeth.com",
    description="Production-grade distributed training framework for financial time series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tradervijeth/quantscale-distributed-training",
    packages=find_packages(exclude=["tests", "scripts", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": requirements_dev,
        "test": requirements_test,
        "all": requirements_dev + requirements_test,
        "onnx": [
            "onnx>=1.15.0",
            "onnxruntime>=1.16.0",
        ],
    },
    license="Apache License 2.0",
    keywords="distributed-training deep-learning financial-ml pytorch time-series",
)
