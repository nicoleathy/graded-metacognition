from setuptools import find_packages, setup

setup(
    name="esma",
    version="0.0.1",
    description="This repository is template for my python project.",
    python_requires=">=3.12",
    install_requires=[
        "torch",
        "matplotlib",
        "numpy",
        "tqdm",
        "transformers",
        "datasets",
        "wandb",
        "accelerate",
        "scipy",
        "scikit-learn",
    ],
    url="https://github.com/cosmoquester/ESMA",
    author="Sangjun Park",
    packages=find_packages(exclude=["tests"]),
)
