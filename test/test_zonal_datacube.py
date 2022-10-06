import geopandas as gpd
from dask.distributed import Client, LocalCluster
from pandas.testing import assert_frame_equal

from zonal_datacube.analysis_functions import AnalysisFunction, sum, min, max, mean, count
from zonal_datacube.zonal_datacube import ZonalDataCube


def test_mask_datacube(small_diamond_features, small_datacube):
    geom = small_diamond_features["geometry"][0]
    masked_datacube = ZonalDataCube._mask_datacube_by_geom(
        geom, small_datacube, (0, 0, 1, 1)
    )

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

    small_diamond_features_fishnetted["zone_wkt"] = small_diamond_features_fishnetted[
        "geometry"
    ].apply(lambda x: x.wkt)
    del small_diamond_features_fishnetted["geometry"]

    result = ZonalDataCube._analyze_partition(
        small_diamond_features_fishnetted, stac_items, analysis_funcs
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


def test_get_meta(small_diamond_features_fishnetted):
    def func(feature, datacube):
        return 1

    analysis_funcs = {
        "first": AnalysisFunction(func=func, agg="sum"),
        "second": AnalysisFunction(func=func, agg="sum", meta={"second_col": "uint8"}),
        "third": AnalysisFunction(
            func=func, agg="sum", meta={"third_col1": "int32", "third_col2": "float32"}
        ),
    }
    meta = ZonalDataCube._get_meta(small_diamond_features_fishnetted, analysis_funcs)

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
    actual_results = small_zonal_datacube.analyze(funcs=[sum])
    assert (
            small_datacube_expected_stats["sum"]
            == actual_results["sum"]
    ).all()


def test_multiple_stats(small_zonal_datacube, small_datacube_expected_stats):
    stats = [min, max, mean, count]
    actual_results = small_zonal_datacube.analyze(funcs=stats)

    assert_frame_equal(
        actual_results[["min", "max", "mean", "count"]],
        small_datacube_expected_stats[["min", "max", "mean", "count"]],
        check_dtype=False
    )


def test_with_client(small_zonal_datacube, small_datacube_expected_stats):
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit="1GB")
    client = Client(cluster)  # noqa F841

    # just testing no exceptions raised from trying to run via a cluster,
    # which can happen if you try to pass unserializable types between
    # workers
    small_zonal_datacube.analyze(funcs=["sum"])
