import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import pickle
import ast

def load_geo_file(geo_file_path):
    df = pd.read_csv(geo_file_path)
    geometry = [LineString(ast.literal_eval(line)) for line in df['coordinates']]
    return gpd.GeoDataFrame(df, geometry=geometry)