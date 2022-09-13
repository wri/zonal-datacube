import numpy as np

from zonal_datacube.fishnet import create_fishnet_grid, fishnet


def test_create_fishnet():
    fishnet_grid = create_fishnet_grid(
        min_x=0, min_y=0, max_x=10, max_y=10, cell_size=1
    )
    assert len(fishnet_grid.index) == 100
    assert fishnet_grid.geometry.total_bounds.astype(np.uint8).tolist() == [
        0,
        0,
        10,
        10,
    ]
    assert set(fishnet_grid.columns) == {"fishnet_wkt", "geometry"}


def test_fishnet_features(small_diamond_features):
    fishnet_features = fishnet(
        small_diamond_features, min_x=0, min_y=0, max_x=10, max_y=10, cell_size=1
    )

    assert len(fishnet_features.index) == 28
    assert set(fishnet_features.columns) == {"geometry", "fishnet_wkt", "id"}
