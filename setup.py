from setuptools import setup, find_packages

setup(
    name="gpt_from_scratch",
    description="Scalable GPT implementation using Distributed Data Parallel for efficient, multi-GPU training of transformer models.",
    version="1.1",
    packages=find_packages(),
    license="MIT",
    author="Matin Khajavi",
    url='https://github.com/MatinKhajavi/GPT-from-scratch',
    install_requires=[
        'datasets',
        'numpy',
        'Requests',
        'tiktoken>=0.7.0',
        'torch',
        'tqdm',
        'matplotlib',
        'seaborn'
      ],
    classifiers=[
          "Programming Language :: Python :: 3.12",
          'Environment :: Console',
          'Framework :: Jupyter',
          "License :: OSI Approved :: MIT License",
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
      ],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
