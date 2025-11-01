from setuptools import setup, find_packages

setup(
    name="biosig",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'PyWavelets',
        'ipython',
        'jupyter',
        'pandas',
        'matplotlib',
        'ipywidgets',
        'scikit-learn'
    ]
)