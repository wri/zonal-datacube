import geopandas as gpd
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
    fishnet["fishnet_wkt"] = fishnet["geometry"].apply(wkt.dumps)
    return fishnet


def fishnet(gdf, min_x, min_y, max_x, max_y, cell_size):
    fishnet_grid = create_fishnet_grid(min_x, min_y, max_x, max_y, cell_size)

    # TODO default predicate intersects might occasionally grab extra tiles that only
    # touch, but doesn't seem like geopandas has a predicate for geometries
    # that intersect only interiors
    join_result = gpd.sjoin(gdf, fishnet_grid)

    # remove fishnet index column
    del join_result["index_right"]
    return join_result
