from setuptools import setup, find_packages

setup(
    name="protorl",
    version="0.1",
    author="Phil Tabor",
    author_email="phil@neuralnet.ai",
    url="https://www.github.com/philtabor/protorl",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy==1.21.*",
        "torch==1.11.*",
        "gym==0.26.*",
        "gym[box2d]",
        "atari-py==0.2.6",
        "mpi4py",
        "opencv-python",
        "matplotlib",
        "ale-py"
    ],
    description="Torch based deep RL framework for rapid prototyping",
    python_requires=">=3.8",
)
