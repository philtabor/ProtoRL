from setuptools import setup, find_packages

setup(
    name="protorl",
    version="0.3.0",
    author="Phil Tabor",
    author_email="phil@neuralnet.ai",
    url="https://www.github.com/philtabor/protorl",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23, <2.0",
        "torch==2.2",
        "gymnasium",
        "gymnasium[accept-rom-license]",
        "gymnasium[mujoco]",
        "gymnasium[box2d]",
        "gymnasium[atari]",
        "opencv-python",
        "matplotlib",
    ],
    description="Torch based deep RL framework for rapid prototyping",
    python_requires=">=3.8, <3.12",
)
