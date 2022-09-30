from setuptools import setup

setup(
    name="zonal-datacube",
    version="0.0.1",
    description="Lambda function to run serverless on the fly raster analysis",
    packages=["zonal_datacube"],
    author="Justin Terry",
    license="MIT",
    install_requires=[
        "dask",
        "dask[distributed]",
        "geopandas",
        "shapely",
        "odc-stac",
        "rasterio",
        "xarray",
        "rtree",
        "rio-stac",
        "pystac_client",
        "s3fs",
    ],
)
