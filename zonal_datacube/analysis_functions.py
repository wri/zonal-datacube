from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Union

from geopandas import GeoDataFrame
from numpy import dtype
from pandas import DataFrame, Series
from xarray import Dataset

from .geo_utils import get_hectare_area

agg_functions_type = Literal["sum", "mean", "max", "min"]


@dataclass
class AnalysisFunction:
    """Analysis function."""

    func: Callable[[GeoDataFrame, Dataset], Union[dtype, DataFrame]]
    agg: Union[Dict[str, agg_functions_type], agg_functions_type]
    meta: Optional[Dict[str, dtype]] = None


def _sum_func(datacube, zone):
    return Series({"sum": datacube.sum().to_pandas().sum()})


def _count_func(datacube, zone):
    # TODO how to propagate NoData value?
    return Series({"count": datacube.count().to_pandas().sum()})


def _mean_func(datacube, zone):
    return Series({"mean": datacube.mean().to_pandas().mean()})


def _min_func(datacube, zone):
    return Series({"min": datacube.min().to_pandas().min()})


def _max_func(datacube, zone):
    return Series({"max": datacube.max().to_pandas().max()})


def _hectare_area_func(datacube, zone):
    pixel_width = datacube.latitude[0] - datacube.latitude[1]
    hectare_area_per_latitude = get_hectare_area(datacube.latitude, pixel_width)
    count_per_latitude = datacube.count(dim=["longitude"])
    total_areas = (hectare_area_per_latitude * count_per_latitude).sum()
    return Series({"hectare_area": total_areas.to_pandas().sum()})


sum = AnalysisFunction(
    func=_sum_func,
    agg={"sum": "sum"},
    meta={"sum": "float64"},
)

count = AnalysisFunction(
    func=_count_func,
    agg={"count": "sum"},
    meta={"count": "float64"},
)

mean = AnalysisFunction(
    func=_mean_func,
    agg={"mean": "mean"},
    meta={"mean": "float64"},
)

min = AnalysisFunction(
    func=_min_func,
    agg={"min": "min"},
    meta={"min": "float64"},
)

max = AnalysisFunction(
    func=_max_func,
    agg={"max": "max"},
    meta={"max": "float64"},
)

hectare_area = AnalysisFunction(
    func=_hectare_area_func,
    agg={"hectare_area": "sum"},
    meta={"hectare_area": "float64"},
)


def combine_agg_dicts(
    funcs: Dict[str, AnalysisFunction]
) -> Dict[str, agg_functions_type]:
    combined_agg = {}
    for func in funcs:
        combined_agg.update(func.agg)

    return combined_agg
