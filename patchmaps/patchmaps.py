#!/usr/bin/env python
# coding: utf-8



import geopandas as gpd
import numpy as np
from itertools import product

import pandas as pd
import shapely.affinity
from shapely.geometry.polygon import Polygon
from shapely.ops import transform

import pyproj
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from sklearn.decomposition import PCA

import math

def get_structure(poly: Polygon, crs='epsg:4326', working_width=36, factor=2, tramline=None) -> gpd.GeoDataFrame:

    edge_length = working_width * factor
    edge_length = 10
    if factor % 2 == 0:
        parallel_shift = 4
    else:
        parallel_shift = 2
   
    input_crs = CRS.from_user_input(crs).name
    # find correct EPSG for calculation in meter
    utm_crs_list = query_utm_crs_info(datum_name=input_crs, area_of_interest=AreaOfInterest(
        west_lon_degree=poly.bounds[0],
        south_lat_degree=poly.bounds[1],
        east_lon_degree=poly.bounds[2],
        north_lat_degree=poly.bounds[3]))
    utm = CRS.from_epsg(utm_crs_list[0].code)
    # to utm (meters) TODO verify this.
    project = pyproj.Transformer.from_crs(crs, utm, always_xy=True).transform
    poly = transform(project, poly)



    if tramline is None:
        coords = np.array(poly.geoms[0].exterior.coords)
        pca = PCA(n_components=2)
        pca.fit(coords)
        # print("JO")
        # print(type(poly))
        # print(poly.geoms[0].exterior.centroid)
        # print(f'''jo {poly.geoms[0].exterior.centroid.x} {poly.geoms[0].exterior.centroid.y}''')
        p0 = poly.geoms[0].exterior.centroid.x, poly.geoms[0].exterior.centroid.y
        p1 = p0[0] + 10 * pca.components_[0][0], p0[1] + 10 * pca.components_[0][1]
    else:
        tramline = tramline.to_crs('{}'.format(utm))
        p0 = tramline["geometry"][0].coords[0]  # First coordinate of permanent traffic lane
        p1 = tramline["geometry"][0].coords[1]
    def unit_vector(vector):
        return vector / np.linalg.norm(vector)
    def angle_between(v1, v2):
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    x_diff = poly.bounds[2] - poly.bounds[0]
    y_diff = poly.bounds[3] - poly.bounds[1]
    angle = angle_between(np.array([x_diff, y_diff]), pca.components_[0])

    rotated = shapely.affinity.rotate(poly, angle, origin=(p0[0], p0[1]), use_radians=True)
    x_diff = rotated.bounds[2] - rotated.bounds[0]
    y_diff = rotated.bounds[3] - rotated.bounds[1]

    ##get the right dimension for layout
    dimension = x_diff / edge_length, y_diff / edge_length
    dimension_a = math.ceil(dimension[0])
    dimension_b = math.ceil(dimension[1])

    # Second coordinate of permanent traffic lane
    q1 = np.array([edge_length, 0])
    q2 = np.array([0, edge_length])

    # top left of grid
    so = np.array([rotated.bounds[0], rotated.bounds[1]])
    def compute_poly(i, j):
        s = so + i * q1 + j * q2
        patch = Polygon([s, s + q1, s + (q1 + q2), s + q2])
        return shapely.affinity.rotate(patch, -angle, origin=(p0[0], p0[1]), use_radians=True)

    polies = [compute_poly(i, j) for i, j in product(range(0, dimension_a), range(0, dimension_b))]
    # polies = [compute_poly(0, 0)]
    data = gpd.GeoDataFrame({'geometry': polies})
    data.crs = '{}'.format(utm)
    # Alternatively use clip and clip to polygon
    patches_within = data.clip(poly, keep_geom_type=True)
    # patches_within = data[data.intersects(poly)]

    # transform back to original crs
    patches_within = patches_within.to_crs(crs)

    return patches_within
