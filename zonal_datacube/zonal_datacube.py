from datetime import datetime

import dask.dataframe as dd
import geopandas as gpd
import pandas as pd
import rasterio
from odc import stac
from rasterio import features
from shapely import wkt


class ZonalDataCube:

    # For datacubes where time doesn't matter,
    # we'll set all STAC items to the same datetime
    DEFAULT_TIME = datetime.fromisoformat("2000-01-01T00:00:00")

    def __init__(
        self,
        features,
        stac_items,
        bounds=(-180, -90, 180, 90),
        cell_size=1,
        npartitions=200,
        spatial_only=True,
    ):
        self.features = features
        self.npartitions = npartitions
        self.fishnet_features = self._apply_fishnet_to_features(bounds, cell_size)
        self.bounds = bounds
        self.stac_items = self._get_stac_items(stac_items, spatial_only)

    def analyze(self, analysis_funcs):
        fishnet_features = self._fishnet_features(self.features)

        dd_features = dd.from_pandas(fishnet_features, npartitions=self.npartitions)
        indexed_features = dd_features.set_index(
            "fishnet_wkt", npartitions=self.npartitions
        )

        feature_meta = dd.utils.make_meta(self.features)
        results = indexed_features.map_partitions(
            self._analyze_partition,
            args=(self.stac_items, analysis_funcs),
            meta=feature_meta,
        )

        # results.groupby[self.features.columns]

        return results.compute()

    @staticmethod
    def _analyze_partition(partition, stac_items, analysis_funcs):
        results = []
        for fishnet_wkt, features_per_tile in partition.groupby("fishnet_wkt"):
            tile = wkt.loads(fishnet_wkt)
            datacube = stac.load(stac_items, bbox=tile.bounds)
            results_per_tile = []

            for _, feature in features_per_tile.iterrows():
                masked_datacube = ZonalDataCube._mask_datacube(
                    feature.geometry, datacube
                )

                results_per_feature = [func(masked_datacube) for func in analysis_funcs]

                results_per_tile.append(results_per_feature)

            results.append(results_per_tile)
        if results:
            return pd.concat(results)
        else:
            return pd.DataFrame().reindex(columns=features_per_tile.columns)

    def _apply_fishnet_to_features(self, bounds=(-180, -90, 180, 90), cell_size=1):
        fishnet = self._create_fishnet(bounds, cell_size)
        return gpd.sjoin(self.features, fishnet)

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
    def _mask_datacube(geom, datacube):
        if not geom.is_valid:
            geom = geom.buffer(0)

        height = datacube.longitude.shape[0]
        width = datacube.latitude.shape[0]

        geom_mask = features.geometry_mask(
            [geom],
            out_shape=(height, width),
            transform=rasterio.transform.from_bounds(*geom.bounds, width, height),
            invert=True,
        )

        return geom_mask * datacube


# {"sum": {"func": lambda x: x, "meta": feature + "sum", "agg": "sum"}}
#
# datacube = ZonalDataCube(features, items)
# results = datacube.analyze(analysis_funcs=["sum", "mean", "min", "max"])
