from setuptools import setup, find_packages

setup(
    name="bayesian-ai-control",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
    ],)