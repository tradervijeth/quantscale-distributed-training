"""
QuantScale: Distributed Training Framework for Financial Time Series

Setup configuration for package installation.

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def read_requirements(filename):
    """Read requirements from file."""
    try:
        with open(filename, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

requirements = read_requirements("requirements.txt")
requirements_dev = read_requirements("requirements-dev.txt")
requirements_test = read_requirements("requirements-test.txt")

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
