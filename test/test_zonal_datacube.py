from shapely.geometry import box

from zonal_datacube.zonal_datacube import ZonalDataCube


def test_mask_datacube(small_diamond_features, small_datacube):
    geom = small_diamond_features["geometry"][0]
    tile_geom = geom.intersection(box(0, 0, 1, 1))
    masked_datacube = ZonalDataCube._mask_datacube(tile_geom, small_datacube)

    # check shape is maintained
    assert masked_datacube.longitude.shape[0] == 10
    assert masked_datacube.latitude.shape[0] == 10

    # check mask correctly applyed to data variables
    assert masked_datacube.checkerboard.sum() == 25
    assert masked_datacube.half_and_half_hotdog.sum() == 45


def test_analyze_partition(small_diamond_features_fishnetted, stac_items):
    def analysis_func(x):
        return x.sum()

    result = ZonalDataCube._analyze_partition(
        small_diamond_features_fishnetted, stac_items, [analysis_func]
    )
    assert result


def test_basic_counter():
    pass
