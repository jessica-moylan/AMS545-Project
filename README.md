# AMS545 Project

This is an adaptation of https://github.com/micycle1/TrapMap into Python with some crucial changes with the implementation. 

## What This Project Does

- Builds trapezoidal maps from polygon boundaries.
- Loads and processes biome data from [ecoregions](https://ecoregions.appspot.com/) into PostGIS using Docker 
- Visualizes map construction step-by-step.
- Supports point lookup to find a containing biome/region.

## Quick Start

1. Install dependencies with Pixi.
    ```bash
   pixi shell 
   ```
2. Start PostGIS:
   ```bash
   docker compose up -d
   ```
3. Create a `.env` file in the project root:
   ```env
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=your_password
   POSTGRES_DB=gis
   ```
4. Build/load database tables:
   ```bash
   pixi run database
   ```
5. Run the visualizer:
   ```bash
   pixi run demo
   ```
6. Run the program with ecological data
   ```bash
   pixi run biomes
   ```

## Project Layout

- `src/ams545/`: trap map core logic, utilities, visualization.
- `graph/`: geometry primitives (`Vector`, `Segment`, `Trapezoid`, etc.).
- `data/`: PostGIS ingestion and preprocessing the unzipped data from ecoregions should go here
- `main.py`: entry point for loading data and launching visualization.
