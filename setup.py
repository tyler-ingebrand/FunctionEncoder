from setuptools import setup, find_packages

setup(
    name="FunctionEncoder",
    version="0.0.4",
    author="Tyler Ingebrand",
    author_email="tyleringebrand@utexas.edu",
    description="A package for learning basis functions over arbitrary function sets. This allows even high-dimensional problems to be solved via a minimal number of basis functions. This allows for zero-shot transfer within these spaces, and also a mechanism for fully informative function representation via the coefficients of the basis functions. Hilbert spaces are nifty.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tyler-ingebrand/FunctionEncoder",
    packages=find_packages(exclude=["Examples", "imgs"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "tqdm",
        "tensorboard",
        "numpy<=1.26.4"
    ],
)
