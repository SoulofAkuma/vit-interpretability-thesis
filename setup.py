from setuptools import setup, find_packages

setup(
    name='vit-interpretability-thesis',
    version='0.0.1',
    packages=find_packages(),
    package_data={
        'src': ['data/*']
    }
)