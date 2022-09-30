import pandas as pd

from zonal_datacube.analysis_functions import (
    AnalysisFunction,
    combine_agg_dicts,
    get_default_analysis_function,
    hectare_area,
    sum,
)


def test_get_default_analysis_function(stac_items):
    expected_func = get_default_analysis_function("sum", stac_items)
    actual_func = AnalysisFunction(
        func=sum,
        agg={
            "sum_checkerboard": "sum",
            "sum_half_and_half_hotdog": "sum",
            "sum_half_and_half_hamburger": "sum",
        },
        meta={
            "sum_checkerboard": "float64",
            "sum_half_and_half_hotdog": "float64",
            "sum_half_and_half_hamburger": "float64",
        },
    )
    assert expected_func == actual_func


def test_combine_agg_dicts():
    def dummy(zone, datacube):
        return zone

    actual = combine_agg_dicts(
        {
            "func1": AnalysisFunction(func=dummy, agg="sum", meta={"func1": "float64"}),
            "func2": AnalysisFunction(
                func=dummy,
                agg={"result1": "sum", "result2": "avg"},
                meta={"result1": "float64", "result2": "float64"},
            ),
            "func3": AnalysisFunction(
                func=dummy,
                agg={"result3": "min", "result4": "max"},
                meta={"result3": "float64", "result4": "float64"},
            ),
        }
    )

    expected = {
        "func1": "sum",
        "result1": "sum",
        "result2": "avg",
        "result3": "min",
        "result4": "max",
    }

    assert actual == expected


def test_hectare_area(small_datacube, small_diamond_features):
    attributes = pd.Series({"id": 0})
    area = hectare_area(attributes, small_datacube)

    # TODO compare to equal area projection
    assert area.all()
