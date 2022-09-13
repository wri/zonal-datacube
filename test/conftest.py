import geopandas as gpd
import numpy as np
import pystac
import pytest
import rasterio
from odc import stac
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from zonal_datacube.fishnet import fishnet


def create_diamond(center_x, center_y, length):
    return Polygon(
        [
            (center_x - length, center_y),
            (center_x, center_y + length),
            (center_x + length, center_y),
            (center_x, center_y - length),
            (center_x - length, center_y),
        ]
    )


def create_raster(name, arr):
    tr = from_origin(0, 10, 0.1, 0.1)
    dataset = rasterio.open(
        name,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype=np.uint8,
        crs="epsg:4326",
        transform=tr,
    )

    dataset.write(arr, 1)
    dataset.close()


@pytest.fixture
def small_diamond_features():
    diamond1 = create_diamond(1, 1, 0.5)
    diamond2 = create_diamond(6, 6, 2)

    features = gpd.GeoDataFrame([diamond1, diamond2], columns=["geometry"]).set_crs(
        "EPSG:4326"
    )
    features["id"] = [1, 2]
    return features


@pytest.fixture
def small_diamond_features_fishnetted(small_diamond_features):
    fishnet_features = fishnet(
        small_diamond_features, min_x=0, min_y=0, max_x=10, max_y=10, cell_size=1
    )
    return fishnet_features


@pytest.fixture
def stac_items():
    return [
        pystac.Item.from_file("fixtures/stac_items/checkerboard.geojson"),
        pystac.Item.from_file("fixtures/stac_items/half_and_half_hamburger.geojson"),
        pystac.Item.from_file("fixtures/stac_items/half_and_half_hotdog.geojson"),
    ]


@pytest.fixture
def small_datacube(stac_items):
    return stac.load(stac_items, bbox=(0, 0, 1, 1))


# checkerboard = np.indices((100, 100)).sum(axis=0) % 2
