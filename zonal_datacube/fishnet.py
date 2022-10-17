import geopandas as gpd
import dask_geopandas
import shapely.geometry as geometry
import shapely.wkt as wkt


def create_fishnet_grid(min_x, min_y, max_x, max_y, cell_size) -> gpd.GeoDataFrame:
    x, y = (min_x, min_y)
    geom_array = []

    # Polygon Size
    while y < max_y:
        while x < max_x:
            geom = geometry.Polygon(
                [
                    (x, y),
                    (x, y + cell_size),
                    (x + cell_size, y + cell_size),
                    (x + cell_size, y),
                    (x, y),
                ]
            )
            geom_array.append(geom)
            x += cell_size
        x = min_x
        y += cell_size

    fishnet = gpd.GeoDataFrame(geom_array, columns=["geometry"]).set_crs("EPSG:4326")
    return fishnet


def fishnet(gdf, min_x, min_y, max_x, max_y, cell_size):
    fishnet_grid = dask_geopandas.from_geopandas(
        create_fishnet_grid(min_x, min_y, max_x, max_y, cell_size),
        npartitions=gdf.npartitions
    ).spatial_shuffle()

    # preserve geometry in second column for join
    fishnet_grid["fishnet_wkt"] = fishnet_grid.geometry.apply(lambda x: wkt.dumps(x))

    # TODO default predicate intersects might occasionally grab extra tiles that only
    # touch, but doesn't seem like geopandas has a predicate for geometries
    # that intersect only interiors
    join_result = gdf.sjoin(fishnet_grid)

    # intersect the fishnet geometry to apply it to each geometry
    # join_result.geometry = join_result.geometry.intersection(join_result.fishnet_geometry)
    # join_result["fishnet_wkt"] = join_result.fishnet_geometry.apply(lambda x: wkt.dumps(x))

    # remove intermediate columns
    del join_result["index_right"]

    return join_result
