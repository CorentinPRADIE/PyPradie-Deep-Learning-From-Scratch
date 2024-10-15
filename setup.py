from setuptools import setup, find_packages

setup(
    name="pypradie",
    version="0.1.0",
    author="Corentin Pradie",
    description="A PyTorch mimic deep learning framework.",
    packages=find_packages(),  # Automatically includes all packages with __init__.py
    install_requires=[
        "numpy",
        "torch",
        "pandas"
    ],
)
