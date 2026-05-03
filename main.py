import os
import geopandas as gpd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from graph.vector import Vector
from src.ams545.visualize import visualize_trapmap_construction

load_dotenv()
postgres_user = os.getenv("POSTGRES_USER", "postgres")
postgres_password = os.getenv("POSTGRES_PASSWORD")
postgres_db = os.getenv("POSTGRES_DB", "gis")

engine = create_engine(
    f"postgresql://{postgres_user}:{postgres_password}@localhost:5432/{postgres_db}"
)

sql = "SELECT * FROM world_biomes"
gdf = gpd.read_postgis(sql, con=engine, geom_col='geometry') 
print("finished loading data from PostGIS")

exploded = gdf.explode(ignore_index=True)
print("finished exploding geometries")

coord_dict = {}
for idx, row in exploded.iterrows():
    coords = [Vector(lat, long) for lat, long in row['geometry'].exterior.coords]
    if row['BIOME_NAME'] in coord_dict:
        coord_dict[row['BIOME_NAME']].append(coords)
    else:
        coord_dict[row['BIOME_NAME']] = [coords]
print("finished converting geometries to coordinate lists")

if __name__ == "__main__":
    visualize_trapmap_construction(coord_dict, title="Trapezoidal Map — Biomes", shuffle=True)