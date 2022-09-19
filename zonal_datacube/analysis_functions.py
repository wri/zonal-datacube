from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Union

from geopandas import GeoDataFrame
from numpy import dtype
from pandas import DataFrame
from xarray import Dataset

agg_functions_type = Literal["sum", "mean", "max", "min"]


@dataclass
class AnalysisFunction:
    """Analysis function."""

    func: Callable[[GeoDataFrame, Dataset], Union[dtype, DataFrame]]
    agg: Union[Dict[str, agg_functions_type], agg_functions_type]
    meta: Optional[Dict[str, dtype]] = None


def sum(zone, datacube):
    return datacube.sum().to_pandas().add_prefix("sum_")


def area(zone, datacube):
    raise NotImplementedError


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
    agg = {col: "sum" for col in result_columns}

    return {
        "sum": AnalysisFunction(
            func=sum,
            agg=agg,
            meta=meta,
        )
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
