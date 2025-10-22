#!/usr/bin/env python3
"""
AI Navigation Assistant - Setup Configuration
Python project for real-time AI-powered navigation assistance
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("backend/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-navigation-assistant",
    version="1.0.0",
    author="AI Navigation Team",
    description="Real-time AI-powered navigation assistance with object detection and path guidance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-navigation-assistant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video :: Capture",
        "Topic :: Internet :: WWW/HTTP :: WSGI :: Application",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-navigation-backend=backend.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "backend": ["*.pt", "models/*"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ai-navigation-assistant/issues",
        "Source": "https://github.com/yourusername/ai-navigation-assistant",
        "Documentation": "https://github.com/yourusername/ai-navigation-assistant#readme",
    },
)