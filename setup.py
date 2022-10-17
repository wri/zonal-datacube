from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="zonal-datacube",
    version="0.0.3",
    description="Lambda function to run serverless on the fly raster analysis",
    packages=["zonal_datacube"],
    author="Justin Terry",
    license="MIT",
    install_requires=required,
)
