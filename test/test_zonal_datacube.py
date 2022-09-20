import geopandas as gpd
import pandas as pd

from zonal_datacube.analysis_functions import AnalysisFunction
from zonal_datacube.zonal_datacube import ZonalDataCube


def test_mask_datacube(small_diamond_features, small_datacube):
    geom = small_diamond_features["geometry"][0]
    masked_datacube = ZonalDataCube._mask_datacube(geom, small_datacube, (0, 0, 1, 1))

    # check shape is maintained
    assert masked_datacube.longitude.shape[0] == 10
    assert masked_datacube.latitude.shape[0] == 10

    # check mask correctly applied to data variables
    assert masked_datacube.checkerboard.sum() == 4
    assert masked_datacube.half_and_half_hotdog.sum() == 10


def test_analyze_partition(small_diamond_features_fishnetted, stac_items):
    def func(feature, datacube):
        return 1

    analysis_funcs = {"results": AnalysisFunction(func=func, agg="sum")}
    empty_result = pd.DataFrame(columns=["id", "results"])

    small_diamond_features_fishnetted["zone_wkt"] = small_diamond_features_fishnetted[
        "geometry"
    ].apply(lambda x: x.wkt)
    del small_diamond_features_fishnetted["geometry"]

    result = ZonalDataCube._analyze_partition(
        small_diamond_features_fishnetted,
        stac_items,
        analysis_funcs,
        empty_result=empty_result,
    )

    assert result.groupby(["id"]).sum().results.to_list() == [4, 24]


def test_get_dask_zones(small_diamond_features):
    dask_zones = ZonalDataCube._get_dask_zones(
        small_diamond_features, (0, 0, 10, 10), 1, 10
    )
    assert set(dask_zones.columns) == {"zone_wkt", "id"}
    assert dask_zones.index.name == "fishnet_wkt"
    assert dask_zones.npartitions == 10
    assert dask_zones.compute().shape == (28, 2)


def test_get_meta(small_diamond_features):
    def func(feature, datacube):
        return 1

    analysis_funcs = {
        "first": AnalysisFunction(func=func, agg="sum"),
        "second": AnalysisFunction(func=func, agg="sum", meta={"second_col": "uint8"}),
        "third": AnalysisFunction(
            func=func, agg="sum", meta={"third_col1": "int32", "third_col2": "float32"}
        ),
    }
    meta = ZonalDataCube._get_meta(small_diamond_features, analysis_funcs)

    actual_dtypes = meta.dtypes.to_dict()
    expected_dtypes = {
        "id": "int64",
        "first": "float64",
        "second_col": "uint8",
        "third_col1": "int32",
        "third_col2": "float32",
    }

    for col in actual_dtypes.keys():
        if col == "geometry":
            assert isinstance(actual_dtypes[col], gpd.array.GeometryDtype)
        else:
            assert actual_dtypes[col] == expected_dtypes[col]


def test_sum(small_zonal_datacube, small_datacube_expected_stats):
    actual_results = small_zonal_datacube.analyze(analysis_funcs=["sum"])

    assert (
        small_datacube_expected_stats["sum_checkerboard"]
        == actual_results["sum_checkerboard"]
    ).all()
    assert (
        small_datacube_expected_stats["sum_half_and_half_hotdog"]
        == actual_results["sum_half_and_half_hotdog"]
    ).all()
    assert (
        small_datacube_expected_stats["sum_half_and_half_hamburger"]
        == actual_results["sum_half_and_half_hamburger"]
    ).all()
