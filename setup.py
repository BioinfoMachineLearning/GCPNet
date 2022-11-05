#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="GCPNet",
    version="0.0.1",
    description="A PyTorch implementation of Geometry-Complete SE(3)-Equivariant Perceptron Networks (GCPNets)",
    author="",
    author_email="",
    url="https://github.com/BioinfoMachineLearning/GCPNet",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
