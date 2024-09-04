from setuptools import setup, find_packages

setup(
    name='LPDynPy',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'rasterio',
        'scikit-learn',
        'matplotlib'
    ],
    author='Converted by ChatGPT',
    description='Python package converted from LPDynR R package',
)