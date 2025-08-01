from setuptools import setup, find_packages

setup(
    name="chatterbox",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.26.0",
        "librosa==0.11.0",
        "s3tokenizer",
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "transformers==4.46.3",
        "diffusers==0.29.0",
        "resemble-perth==1.0.1",
        "conformer==0.3.2",
        "safetensors==0.5.3",
        "gradio"
    ],
)
