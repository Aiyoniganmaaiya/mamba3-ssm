from setuptools import setup, find_packages

setup(
    name="mamba3-ssm",
    version="0.1.0",
    description="Mamba-3: Improved Sequence Modeling using State Space Principles",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "einops>=0.7",
    ],
)
