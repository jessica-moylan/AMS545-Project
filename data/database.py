import os
import geopandas as gpd
import urllib.request   
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

postgres_user = os.getenv("POSTGRES_USER")
postgres_password = os.getenv("POSTGRES_PASSWORD")
postgres_db = os.getenv("POSTGRES_DB")

# ST_CoverageSimplify uses the layer CRS units. This dataset is in EPSG:4326,
# so convert the target tolerance from kilometers to approximate degrees.
SIMPLIFY_TOLERANCE_KM = 30
KM_PER_DEGREE = 111.32

engine = create_engine(
    f"postgresql://{postgres_user}:{postgres_password}@localhost:5432/{postgres_db}"
)

gdf = gpd.read_file("data/Ecoregions2017.shp")

url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(url)
sa_boundary = world[world['ADMIN'] == 'New Zealand']
boundry = gpd.clip(gdf, sa_boundary)

with engine.begin() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))

boundry.to_postgis("world_biomes", engine, if_exists="replace", index=False)

with engine.begin() as conn:
    # Fix any individually invalid geometries first
    conn.execute(text("""
        UPDATE world_biomes
        SET geometry = ST_MakeValid(geometry)
        WHERE NOT ST_IsValid(geometry)
    """))

    # Check coverage validity before cleaning
    result = conn.execute(text("""
        SELECT COUNT(*) FROM (
            SELECT ST_CoverageInvalidEdges(geometry) OVER () AS invalid_edges
            FROM world_biomes
        ) sub
        WHERE invalid_edges IS NOT NULL
    """))
    bad_count = result.scalar()

    # Clean the coverage by fixing gaps, overlaps, and mismatched edges
    conn.execute(text("DROP TABLE IF EXISTS world_biomes_clean"))
    conn.execute(text("""
        CREATE TABLE world_biomes_clean AS
        SELECT
            "OBJECTID",
            "ECO_NAME",
            "BIOME_NUM",
            "BIOME_NAME",
            "REALM",
            "COLOR",
            "LICENSE",
            ST_CoverageClean(geometry, -1, 0.09) OVER () AS geometry
        FROM world_biomes
    """))

    # Verify the clean table passes coverage validation
    result = conn.execute(text("""
        SELECT COUNT(*) FROM (
            SELECT ST_CoverageInvalidEdges(geometry) OVER () AS invalid_edges
            FROM world_biomes_clean
        ) sub
        WHERE invalid_edges IS NOT NULL
    """))
    still_bad = result.scalar()

    # Simplify the clean coverage.
    # Tolerance is in degrees here because the geometries are stored in EPSG:4326.
    conn.execute(text("DROP TABLE IF EXISTS world_biomes_simplified"))
    conn.execute(text("""
        CREATE TABLE world_biomes_simplified AS
        SELECT
            "OBJECTID",
            "ECO_NAME",
            "BIOME_NUM",
            "BIOME_NAME",
            "REALM",
            "COLOR",
            "LICENSE",
            ST_CoverageSimplify(geometry, :simplify_tolerance) OVER () AS geometry
        FROM world_biomes_clean
    """), {"simplify_tolerance": SIMPLIFY_TOLERANCE_KM / KM_PER_DEGREE})
    conn.execute(text("DROP TABLE IF EXISTS world_biomes_final"))
    conn.execute(text("""
        CREATE TABLE world_biomes_final AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY "BIOME_NAME") AS cartodb_id,
            "BIOME_NAME" AS biome_name,
            ST_Multi(ST_Union(geometry)) AS the_geom,
            ST_Transform(ST_SetSRID(ST_Multi(ST_Union(geometry)), 4326), 3857) AS the_geom_webmercator
        FROM world_biomes_simplified
        GROUP BY "BIOME_NAME"
    """))
    
    print("finish post-processing")