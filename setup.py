from setuptools import setup, find_packages

setup(
    name="GPT_from_scratch",
    description="Scalable GPT implementation using Distributed Data Parallel for efficient, multi-GPU training of transformer models.",
    version="1.0",
    packages=find_packages(),
    license="MIT",
    url='https://github.com/MatinKhajavi/GPT-from-scratch',
    install_requires=[
        'datasets>=2.20.0',
        'numpy>=1.26.4',
        'Requests>=2.31.0',
        'tiktoken>=0.7.0',
        'torch>=2.4.0',
        'tqdm>=4.66.4',
        'matplotlib>=3.9.1',
        'seaborn>=0.12.2'
      ],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)