import pandas as pd

from zonal_datacube.analysis_functions import (
    AnalysisFunction,
    combine_agg_dicts,
    hectare_area,
)


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
