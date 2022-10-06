from datetime import datetime
from typing import Dict, List, Union, cast

import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
import pystac
import rasterio
from odc import stac
from rasterio import features
from shapely import wkt

from .analysis_functions import (
    AnalysisFunction,
    combine_agg_dicts,
)
from .fishnet import fishnet


class ZonalDataCube:
    """A Dask-backed, lazy-loaded raster datacube that will only be read and
    analyzed in specified zones."""

    # Random time to set all STAC items to when spatial_only
    # The time dimension will be removed after reading into xarray
    DEFAULT_TIME = datetime.fromisoformat("2000-01-01T00:00:00")

    def __init__(
        self,
        zones: gpd.GeoDataFrame,
        stac_items: List[pystac.Item],
    ):
        """Creates a new zonal datacube. The datacube is lazy-loaded using the
        STAC item specifications, and only portions of the datacube that fall
        in the zones will be loaded at analysis time.

        :param zones:
            A GeoPandas DataFrame with a geometry column containing polygonal zones,
            as well as any other attributes related to the zones.
        :param stac_items:
            A list of STAC Items
            These will be read into a XArray dataset where each item is a data variable.
        """

        # allow this to be passed as a param once we support
        # time series
        self.spatial_only = True

        self.zones = zones
        self.stac_items = self._get_stac_items(stac_items, self.spatial_only)
        self.attribute_columns = self._get_attribute_columns()

        # TODO calculate optimal cell size and partitions based on
        #  STAC items and features
        self.cell_size = 1
        self.npartitions = 200

        self.bounds = self._get_rounded_bounding_box(self.zones.total_bounds, self.cell_size)
        self.dask_zones = None  # get on first analysis


    def analyze(self, funcs: List[AnalysisFunction]):
        """Run analyses on the datacube. This can default zonal statistics
        methods (e.g. sum, mean, max) or custom analysis functions that process
        the datacube.

        :param funcs:
            A list of the analysis functions to run across the datacube.
            The list can either be the name of a default statistics function
            (count, sum, mean, min, max, area) or a custom AnalysisFunction.
        :return:
            A GeoPandas DataFrame containing the original geometry and attributes,
            as well additional attributes added from the analyses.
        """

        # get on first analysis and then save as attribute
        if self.dask_zones is None:
            self.dask_zones = self._get_dask_zones(
                self.zones, self.bounds, self.cell_size, self.npartitions
            )

        meta = self._get_meta(self.dask_zones, funcs)
        mapped_partitions = self.dask_zones.map_partitions(
            self._analyze_partition,
            self.stac_items,
            funcs,
            meta=meta,
        )

        agg_spec = combine_agg_dicts(funcs)
        grouped_results = (
            mapped_partitions.groupby(self.attribute_columns)
            .agg(agg_spec)
            .reset_index()
        )

        return grouped_results.compute()

    def _get_attribute_columns(self):
        return list(set(self.zones.columns.to_list()) - {"geometry"})

    @staticmethod
    def _analyze_partition(partition, stac_items, funcs):
        partition_results = []

        for fishnet_wkt, zones_per_tile in partition.groupby("fishnet_wkt"):
            tile = wkt.loads(fishnet_wkt)
            datacube = stac.load(stac_items, bbox=tile.bounds)
            masked_datacube = ZonalDataCube._set_no_data_mask(datacube)

            for _, zone in zones_per_tile.iterrows():
                zone_geometry = wkt.loads(zone.zone_wkt)
                geom_masked_datacube = ZonalDataCube._mask_datacube_by_geom(
                    zone_geometry, masked_datacube, tile.bounds
                )

                zone_attributes = zone.drop("zone_wkt")
                result = zone_attributes.copy()
                for func in funcs:
                    func_result = func.func(zone_attributes, geom_masked_datacube)
                    result = pd.concat([result, func_result])

                partition_results.append(result)

        return pd.DataFrame(partition_results)

    @staticmethod
    def _get_dask_zones(zones, bounds, cell_size, npartitions):
        # fishnet features to make them partition more efficiently in Dask
        fishnetted_zones = fishnet(zones, *bounds, cell_size)

        # Dask can't understand the geometry type,
        # so just serialize to WKT and drop geometry
        fishnetted_zones["zone_wkt"] = fishnetted_zones["geometry"].apply(
            lambda x: x.wkt
        )
        del fishnetted_zones["geometry"]

        dask_zones = dd.from_pandas(fishnetted_zones, npartitions=npartitions)

        # Index based on the fishnet geometry so features
        # that overlap the same fishnet grid cell
        # are processed in the same partition.
        # This ensures we only need to read the raster data
        # only once per grid cell across the cluster
        indexed_zones = dask_zones.set_index("fishnet_wkt", npartitions=npartitions)

        return indexed_zones

    @staticmethod
    def _get_meta(zones, funcs):
        meta = dd.utils.make_meta(zones)

        # drop geom columns for meta
        if "zone_wkt" in meta:
            meta = meta.drop("zone_wkt", axis=1)
        elif "fishnet_wkt" in meta:
            meta = meta.drop("fishnet_wkt", axis=1)

        for func in funcs:
            if func.meta:
                for col, type in func.meta.items():
                    meta[col] = pd.Series(dtype=type)
            else:
                # TODO how do we determine dtype if they didn't specify meta?
                meta[col] = pd.Series(dtype="float64")

        return meta


    def _get_stac_items(self, stac_items, spatial_only):
        """Create copies of the STAC items and mutate any properties for our
        processing."""
        new_items = []

        for original_item in stac_items:
            new_item = original_item.full_copy()

            # if spatial only, set all items to the same datetime so STAC clients
            # will see them as aligned on the same time dimension
            if spatial_only:
                new_item.datetime = self.DEFAULT_TIME

            new_items.append(new_item)

        return new_items

    @staticmethod
    def _mask_datacube_by_geom(geom, datacube, bounds, all_touched=False):
        if not geom.is_valid:
            geom = geom.buffer(0)

        height = datacube.longitude.shape[0]
        width = datacube.latitude.shape[0]

        geom_mask = features.geometry_mask(
            [geom],
            out_shape=(height, width),
            transform=rasterio.transform.from_bounds(*bounds, width, height),
            invert=True,
            all_touched=all_touched,
        )

        return datacube.where(geom_mask)

    @staticmethod
    def _set_no_data_mask(datacube):
        """ODC STAC attaches the nodata value from STAC as an attribute to each
        data variable, but doesn't actually apply it. This function will
        iterate through each data variable and apply each nodata value as a
        mask.

        :param datacube: Datacube to mask
        :return:
        """

        for name, da in datacube.data_vars.items():
            datacube[name] = da.where(da != da.nodata)

        return datacube

    @staticmethod
    def _get_rounded_bounding_box(
            bounds, cell_size
    ):
        """Round bounding box to divide evenly into cell_size x cell_size tiles from
        plane origin."""
        return (
            bounds[0] - (bounds[0] % cell_size),
            bounds[1] - (bounds[1] % cell_size),
            bounds[2] + (-bounds[2] % cell_size),
            bounds[3] + (-bounds[3] % cell_size),
        )


# items = []
# datacube = ZonalDataCube(features, items)
# results = datacube.analyze(analysis_funcs=["sum", "mean", "min", "max"])
#
# client = EcsCluster(
#     params="sfdsf"
# )
#
# def func(feature, datacube):
#     # blah blah blah
#     return datacube.sum() + 1
#
# {
#     "sum_plus_one": {
#         "func": func,
#         "agg": "sum",
#         "meta": {
#             "col1": "uint8",
#             "col2": "bool",
#         }
#     }
# }
