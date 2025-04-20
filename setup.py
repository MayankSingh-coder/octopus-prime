from setuptools import setup, find_packages

setup(
    name="single-layer-perceptron",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.23.0",
    ],
)