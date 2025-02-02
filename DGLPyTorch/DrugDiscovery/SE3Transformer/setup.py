from setuptools import find_packages, setup

setup(
    name="se3-transformer",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    version="1.2.0",
    description="PyTorch + DGL implementation of SE(3)-Transformers",
    author="Alexandre Milesi",
    author_email="alexandrem@nvidia.com",
)
