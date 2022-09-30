from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Union

from geopandas import GeoDataFrame
from numpy import dtype
from pandas import DataFrame
from xarray import Dataset

from .geo_utils import get_hectare_area

agg_functions_type = Literal["sum", "mean", "max", "min"]


@dataclass
class AnalysisFunction:
    """Analysis function."""

    func: Callable[[GeoDataFrame, Dataset], Union[dtype, DataFrame]]
    agg: Union[Dict[str, agg_functions_type], agg_functions_type]
    meta: Optional[Dict[str, dtype]] = None


def sum(zone, datacube):
    return datacube.sum().to_pandas().add_prefix("sum_")


def count(zone, datacube):
    # TODO how to propagate NoData value?
    return datacube.count().to_pandas().add_prefix("count_")


def mean(zone, datacube):
    return datacube.mean().to_pandas().add_prefix("mean_")


def min(zone, datacube):
    return datacube.min().to_pandas().add_prefix("min_")


def max(zone, datacube):
    return datacube.max().to_pandas().add_prefix("max_")


def hectare_area(zone, datacube):
    pixel_width = datacube.latitude[0] - datacube.latitude[1]
    hectare_area_per_latitude = get_hectare_area(datacube.latitude, pixel_width)
    count_per_latitude = datacube.count(dim=["longitude"])
    total_areas = (hectare_area_per_latitude * count_per_latitude).sum()
    return total_areas.to_pandas().add_prefix("hectare_area_")


def get_default_analysis_function(func, stac_items):
    """Get the AnalysisFunction for the default analysis function name. Default
    analysis functions return a column for all STAC items with the the function
    name prefixed to the STAC item ID (e.g. sum on stac_id_1 would become
    sum_stac_id_1). Default function columns are always dtype float64.

    :param func: Default function name, one of (sum|mean|min|max)
    :param stac_items: STAC items to be processed by analysis function
    :return:
        The AnalysisFunction object representing the default function
    """
    result_columns = [f"{func}_" + item.id for item in stac_items]
    meta = {col: "float64" for col in result_columns}

    return {
        "sum": AnalysisFunction(
            func=sum,
            agg={col: "sum" for col in result_columns},
            meta=meta,
        ),
        "count": AnalysisFunction(
            func=count,
            agg={col: "sum" for col in result_columns},
            meta=meta,
        ),
        "min": AnalysisFunction(
            func=min,
            agg={col: "min" for col in result_columns},
            meta=meta,
        ),
        "max": AnalysisFunction(
            func=max,
            agg={col: "max" for col in result_columns},
            meta=meta,
        ),
        "mean": AnalysisFunction(
            func=mean,
            agg={col: "mean" for col in result_columns},
            meta=meta,
        ),
        "hectare_area": AnalysisFunction(
            func=hectare_area,
            agg={col: "sum" for col in result_columns},
            meta=meta,
        ),
    }[func]


def combine_agg_dicts(
    analysis_functions: Dict[str, AnalysisFunction]
) -> Dict[str, agg_functions_type]:
    combined_agg = {}
    for name, func in analysis_functions.items():
        if isinstance(func.agg, dict):
            combined_agg.update(func.agg)
        else:
            combined_agg[name] = func.agg

    return combined_agg
