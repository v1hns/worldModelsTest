"""
Optional editable install:  pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="vggt-slam",
    version="0.4.0",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy>=1.24",
        "Pillow>=10.0",
        "huggingface_hub>=0.20",
        "opencv-python>=4.8",
        "matplotlib>=3.7",
        "scipy>=1.10",
        "tqdm>=4.65",
    ],
    extras_require={
        "accel": ["faiss-cpu>=1.7.4"],
        "viz": ["open3d>=0.18.0"],
    },
    description="VGGT-SLAM 2.0 implementation (arXiv:2601.19887)",
)
