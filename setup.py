from setuptools import setup, find_packages

setup(
    name="mamba3-ssm",
    version="0.1.1",
    description="Mamba-3: Improved Sequence Modeling using State Space Principles",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Aiyoniganmaaiya",
    author_email="gerintacc004@gmail.com",
    url="https://github.com/Aiyoniganmaaiya/mamba3-ssm",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "einops>=0.7",
    ],
    extras_require={
        "dev": ["pytest"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    keywords=[
        "mamba", "ssm", "state-space-model", "transformer",
        "pytorch", "language-model", "deep-learning",
    ],
)
