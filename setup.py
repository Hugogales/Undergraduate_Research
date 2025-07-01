#!/usr/bin/env python3
"""
Setup script for the Soccer Environment package.
"""

from setuptools import setup, find_packages

# Read the requirements file
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read the README file for long description
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A multi-agent soccer environment for reinforcement learning research."

setup(
    name="soccer-env",
    version="0.1.0",
    author="Soccer Environment Team",
    author_email="",
    description="A multi-agent soccer environment following PettingZoo standards",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/soccer-env",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Simulation",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "pandas>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "soccer-env-test=examples.test_basic_env:main",
            "soccer-env-train=examples.rllib_training:main",
            "soccer-play=scripts.play:main",
            "soccer-train=scripts.train:main",
            "soccer-train-parallel=scripts.train_parallel:main",
            "soccer-replay=scripts.replay:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 