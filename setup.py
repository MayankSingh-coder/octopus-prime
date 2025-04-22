from setuptools import setup, find_packages

setup(
    name="neural_network_lm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.23.0",
        "tk>=8.6.0",
    ],
    extras_require={
        "web": ["Flask>=2.0.0", "Flask-Cors>=3.0.0"],
        "advanced": ["gensim>=4.0.0", "transformers>=4.0.0", "torch>=1.0.0", "tokenizers>=0.10.0"],
    },
    entry_points={
        "console_scripts": [
            "neural-network-lm-ui=neural_network_lm.ui.complete_mlp_ui:main",
        ],
    },
    author="Neural Network LM Team",
    author_email="example@example.com",
    description="A neural network-based language model inspired by the human brain",
    long_description=open("README_NEW.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neural-network-lm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)